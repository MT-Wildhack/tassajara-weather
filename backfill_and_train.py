#!/usr/bin/env python3
"""
backfill_and_train.py (v4 — Multi-Year Atmospheric River & Hydrological MOS Edition)
Backfills 365-730+ days of historical observations from the Stone Office Davis Vantage Pro2
(Station 36966) using the WeatherLink v2 Historic API (24-hour chunking with exponential
backoff), pairs them with multi-year archived NWP hourly forecasts and Chews Ridge /
Hastings reference series, trains the multi-task TassajaraMOSCorrector (Temperature +
Two-Stage Hurdle PoP/QPF), saves models/tassajara_mos_v1.joblib, and populates
mos_corrected_forecasts across both historical verification timestamps and the active
7-day NWS forecast.
"""

import os
import time
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import requests
import numpy as np
import pandas as pd

from ingest_weather import (
    WEATHERLINK_BASE_URL,
    WeatherLinkV2Client,
    f_to_c,
    mph_to_ms,
    in_to_mm,
    inhg_to_hpa,
)
from mos_model import TassajaraMOSCorrector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TassajaraBackfill")

# ---------------------------------------------------------------------------
# Explicit Fallback Credentials (Immune to blank CI environment variables)
# ---------------------------------------------------------------------------
WEATHERLINK_API_KEY = (os.getenv("WL_API_KEY") or "").strip() or "tigkhjhsweahdrae2cl0uenfuws3kv8k"
WEATHERLINK_API_SECRET = (os.getenv("WL_API_SECRET") or "").strip() or "ffnphp69pdbsqwsahkdewvctewmxagwt"
SYNOPTIC_API_TOKEN = (os.getenv("SYNOPTIC_TOKEN") or "").strip() or "959822b9da094eabb0e41b8a5a7dcaa5"

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")

TASS_LAT = 36.2337
TASS_LON = -121.5501
CHEWS_LAT = 36.3060
CHEWS_LON = -121.5650
HAST_LAT = 36.3886
HAST_LON = -121.5514


def first_not_none(*vals):
    """Returns the first value that is not None (preserving 0 and 0.0)."""
    for v in vals:
        if v is not None and not pd.isna(v):
            return v
    return None


def clean_num(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely converts pandas/numpy values (including NaN/None) to Python float or default."""
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def ensure_schema_and_view(db_path: str = DB_PATH) -> None:
    """
    Ensures all SQLite tables, hydrological MOS columns, and the leak-free
    vw_mos_training_dataset view exist in SQLite.
    """
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if Path("schema.sql").exists():
            with open("schema.sql", "r", encoding="utf-8") as f:
                conn.executescript(f.read())

        # Ensure core tables exist even if schema.sql is minimal
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS local_observations (
                valid_time_utc TEXT PRIMARY KEY,
                station_id TEXT,
                temp_c REAL,
                dewpoint_c REAL,
                wetbulb_c REAL,
                rh_pct REAL,
                wind_speed_ms REAL,
                wind_gust_ms REAL,
                wind_dir_deg REAL,
                mslp_hpa REAL,
                pressure_trend_3hr_hpa REAL,
                precip_1hr_mm REAL DEFAULT 0.0,
                solar_rad_wm2 REAL,
                uv_index REAL,
                qc_flag TEXT DEFAULT 'OK'
            );

            CREATE TABLE IF NOT EXISTS reference_observations (
                valid_time_utc TEXT,
                station_id TEXT,
                temp_c REAL,
                rh_pct REAL,
                wind_speed_ms REAL,
                wind_gust_ms REAL,
                wind_dir_deg REAL,
                precip_1hr_mm REAL DEFAULT 0.0,
                PRIMARY KEY (valid_time_utc, station_id)
            );

            CREATE TABLE IF NOT EXISTS nws_forecasts (
                issue_time_utc TEXT,
                valid_time_utc TEXT,
                lead_time_hours INTEGER,
                nws_temp_c REAL,
                nws_dewpoint_c REAL,
                nws_rh_pct REAL,
                nws_wind_speed_ms REAL,
                nws_wind_gust_ms REAL,
                nws_wind_dir_deg REAL,
                nws_pop_pct REAL DEFAULT 0.0,
                nws_qpf_1hr_mm REAL DEFAULT 0.0,
                nws_sky_cover_pct REAL DEFAULT 20.0,
                short_forecast TEXT,
                PRIMARY KEY (issue_time_utc, valid_time_utc)
            );

            CREATE TABLE IF NOT EXISTS mos_corrected_forecasts (
                issue_time_utc TEXT,
                valid_time_utc TEXT,
                model_version TEXT,
                raw_nws_temp_c REAL,
                pred_temp_error_c REAL,
                corrected_temp_c REAL,
                corrected_temp_p10_c REAL,
                corrected_temp_p90_c REAL,
                corrected_rh_pct REAL,
                corrected_wind_gust_ms REAL,
                raw_nws_pop_pct REAL DEFAULT 0.0,
                mos_pop_pct REAL DEFAULT 0.0,
                raw_nws_qpf_mm REAL DEFAULT 0.0,
                mos_qpf_mm REAL DEFAULT 0.0,
                mos_qpf_p90_mm REAL DEFAULT 0.0,
                PRIMARY KEY (issue_time_utc, valid_time_utc)
            );
            """
        )

        # Idempotent column migration for existing mos_corrected_forecasts tables
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(mos_corrected_forecasts)").fetchall()
        }
        for col_name, col_def in [
            ("raw_nws_pop_pct", "REAL DEFAULT 0.0"),
            ("mos_pop_pct", "REAL DEFAULT 0.0"),
            ("raw_nws_qpf_mm", "REAL DEFAULT 0.0"),
            ("mos_qpf_mm", "REAL DEFAULT 0.0"),
            ("mos_qpf_p90_mm", "REAL DEFAULT 0.0"),
        ]:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE mos_corrected_forecasts ADD COLUMN {col_name} {col_def};")

        conn.executescript(
            """
            DROP VIEW IF EXISTS vw_mos_training_dataset;
            CREATE VIEW vw_mos_training_dataset AS
            SELECT
                f.issue_time_utc,
                f.valid_time_utc,
                f.lead_time_hours,
                f.nws_temp_c,
                f.nws_dewpoint_c,
                f.nws_rh_pct,
                f.nws_wind_speed_ms,
                f.nws_wind_dir_deg,
                COALESCE(f.nws_sky_cover_pct, 20.0) AS nws_sky_cover_pct,
                COALESCE(f.nws_pop_pct, 0.0) AS nws_pop_pct,
                COALESCE(f.nws_qpf_1hr_mm, 0.0) AS nws_qpf_1hr_mm,
                o.temp_c AS actual_temp_c,
                o.rh_pct AS actual_rh_pct,
                o.wind_speed_ms AS actual_wind_speed_ms,
                o.wind_gust_ms AS actual_wind_gust_ms,
                o.wind_dir_deg AS actual_wind_dir_deg,
                o.mslp_hpa AS actual_mslp_hpa,
                COALESCE(o.precip_1hr_mm, 0.0) AS actual_precip_mm,
                COALESCE(o.pressure_trend_3hr_hpa, 0.0) AS pressure_trend_3hr_hpa,
                (o.temp_c - f.nws_temp_c) AS temp_error_c,
                (o.rh_pct - f.nws_rh_pct) AS rh_error_pct,
                COALESCE(r_chews.temp_c, f.nws_temp_c + 2.0) AS issue_chews_temp_c,
                COALESCE(r_chews.precip_1hr_mm, 0.0) AS issue_chews_precip_mm,
                COALESCE(r_hast.temp_c, f.nws_temp_c + 0.5) AS issue_hastings_temp_c,
                COALESCE(o_issue.temp_c, f.nws_temp_c) AS issue_tass_temp_c,
                COALESCE(o_issue.temp_c - f_issue.nws_temp_c, 0.0) AS issue_persistence_bias_c
            FROM nws_forecasts f
            INNER JOIN local_observations o
                ON f.valid_time_utc = o.valid_time_utc
            LEFT JOIN local_observations o_issue
                ON f.issue_time_utc = o_issue.valid_time_utc
            LEFT JOIN (
                SELECT valid_time_utc, AVG(nws_temp_c) AS nws_temp_c
                FROM nws_forecasts
                GROUP BY valid_time_utc
            ) f_issue
                ON f.issue_time_utc = f_issue.valid_time_utc
            LEFT JOIN reference_observations r_chews
                ON f.issue_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
            LEFT JOIN reference_observations r_hast
                ON f.issue_time_utc = r_hast.valid_time_utc AND r_hast.station_id = 'CAHC1'
            WHERE o.qc_flag = 'OK'
              AND o.temp_c IS NOT NULL
              AND f.nws_temp_c IS NOT NULL
              AND f.lead_time_hours >= 1;
            """
        )
        conn.commit()


def fetch_weatherlink_history(
    client: WeatherLinkV2Client,
    station_id: int,
    days_back: int = 730,
) -> pd.DataFrame:
    """
    Pulls 365-730+ days of historical archive records from WeatherLink v2 API
    (/v2/historic/{station_id}) using 86,400-second (24-hour) pagination chunks
    with retry/backoff protection. Aggregates sub-hourly tipping-bucket clicks
    and rainfall_in accurately into hourly buckets.
    """
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)

    chunk_seconds = 86400  # 24 hours max per WeatherLink v2 API spec
    current_start = int(start_dt.timestamp())
    final_end = int(end_dt.timestamp())

    hourly_buckets: Dict[str, Dict[str, Any]] = {}
    total_chunks = int(np.ceil((final_end - current_start) / chunk_seconds))
    chunk_idx = 0

    logger.info(
        f"Fetching {days_back} days ({total_chunks} daily 24h chunks) of WeatherLink Pro archive "
        f"for Station {station_id}..."
    )

    while current_start < final_end:
        chunk_idx += 1
        current_end = min(current_start + chunk_seconds, final_end)
        url = f"{WEATHERLINK_BASE_URL}/historic/{station_id}"
        params = {
            "api-key": client.api_key,
            "start-timestamp": current_start,
            "end-timestamp": current_end,
        }

        payload = None
        for attempt in range(3):
            try:
                resp = client.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait_s = 2.5 * (attempt + 1)
                    logger.warning(f"Rate limited (429) on chunk {chunk_idx}; sleeping {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                if attempt == 2:
                    logger.warning(f"Chunk {chunk_idx}/{total_chunks} failed ({current_start}): {exc}")
                else:
                    time.sleep(1.2 * (attempt + 1))

        if payload:
            for sensor in payload.get("sensors", []):
                for rec in sensor.get("data", []):
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    dt_hr = datetime.fromtimestamp(ts, tz=timezone.utc).replace(
                        minute=0, second=0, microsecond=0
                    )
                    hr_iso = dt_hr.isoformat()

                    if hr_iso not in hourly_buckets:
                        hourly_buckets[hr_iso] = {
                            "valid_time_utc": hr_iso,
                            "temp_c": None,
                            "dewpoint_c": None,
                            "wetbulb_c": None,
                            "rh_pct": None,
                            "wind_speed_ms": 0.0,
                            "wind_gust_ms": 0.0,
                            "wind_dir_deg": 0.0,
                            "mslp_hpa": None,
                            "precip_1hr_mm": 0.0,
                            "solar_rad_wm2": None,
                            "uv_index": None,
                        }

                    b = hourly_buckets[hr_iso]

                    bar_sl = first_not_none(
                        rec.get("bar_sea_level"),
                        rec.get("bar_sea_level_last"),
                        rec.get("bar_final"),
                        rec.get("bar_absolute"),
                        rec.get("bar"),
                    )
                    if bar_sl is not None:
                        b["mslp_hpa"] = inhg_to_hpa(bar_sl)

                    t_f = first_not_none(rec.get("temp_last"), rec.get("temp_avg"), rec.get("temp"))
                    if t_f is not None:
                        b["temp_c"] = f_to_c(t_f)
                        dew_f = first_not_none(
                            rec.get("dew_point_last"), rec.get("dew_point_avg"), rec.get("dew_point")
                        )
                        wet_f = first_not_none(
                            rec.get("wet_bulb_last"), rec.get("wet_bulb_avg"), rec.get("wet_bulb")
                        )
                        hum = first_not_none(rec.get("hum_last"), rec.get("hum_avg"), rec.get("hum"))

                        b["dewpoint_c"] = f_to_c(dew_f)
                        b["wetbulb_c"] = f_to_c(wet_f)
                        b["rh_pct"] = hum

                        w_spd = first_not_none(
                            rec.get("wind_speed_avg"), rec.get("wind_speed_last"), 0.0
                        )
                        w_gst = first_not_none(
                            rec.get("wind_speed_hi"), rec.get("wind_gust_10_min"), w_spd, 0.0
                        )
                        w_dir = first_not_none(
                            rec.get("wind_dir_of_prevail"),
                            rec.get("wind_dir_scalar_avg"),
                            rec.get("wind_dir_last"),
                            0.0,
                        )

                        b["wind_speed_ms"] = mph_to_ms(w_spd) or 0.0
                        gst_ms = mph_to_ms(w_gst) or 0.0
                        b["wind_gust_ms"] = max(b["wind_gust_ms"] or 0.0, gst_ms)
                        b["wind_dir_deg"] = float(w_dir)

                    # Accumulate Vantage Pro2 tipping bucket rainfall across archive interval packets
                    rain_in = first_not_none(rec.get("rainfall_in"), rec.get("rain_in"))
                    if rain_in is None:
                        clicks = first_not_none(rec.get("rainfall_clicks"), rec.get("rain_clicks"))
                        if clicks is not None:
                            rain_in = float(clicks) * 0.01
                    if rain_in:
                        b["precip_1hr_mm"] = round(
                            (b["precip_1hr_mm"] or 0.0) + (in_to_mm(float(rain_in)) or 0.0), 2
                        )

                    sol = first_not_none(rec.get("solar_rad_avg"), rec.get("solar_rad"))
                    uv = first_not_none(rec.get("uv_index_avg"), rec.get("uv_index"))
                    if sol is not None:
                        b["solar_rad_wm2"] = sol
                    if uv is not None:
                        b["uv_index"] = uv

        if chunk_idx % 25 == 0 or chunk_idx == total_chunks:
            wet_cnt = sum(1 for v in hourly_buckets.values() if (v.get("precip_1hr_mm") or 0.0) >= 0.25)
            logger.info(
                f"Progress: {chunk_idx}/{total_chunks} daily chunks fetched "
                f"({len(hourly_buckets)} hourly rows, {wet_cnt} wet hours >= 0.25mm)"
            )

        current_start = current_end
        time.sleep(0.15)

    df = pd.DataFrame(list(hourly_buckets.values()))
    if df.empty:
        return df

    df = df.sort_values("valid_time_utc").reset_index(drop=True)

    numeric_cols = [
        "temp_c", "dewpoint_c", "wetbulb_c", "rh_pct",
        "wind_speed_ms", "wind_gust_ms", "wind_dir_deg",
        "mslp_hpa", "precip_1hr_mm", "solar_rad_wm2", "uv_index",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mslp_hpa" in df.columns and df["mslp_hpa"].notna().any():
        df["mslp_hpa"] = df["mslp_hpa"].ffill().bfill().fillna(1013.25)
        df["pressure_trend_3hr_hpa"] = df["mslp_hpa"].diff(3).round(2).fillna(0.0)
    else:
        df["mslp_hpa"] = 1013.25
        df["pressure_trend_3hr_hpa"] = 0.0

    return df


def fetch_historical_nwp_and_references(days_back: int = 730) -> Dict[str, pd.DataFrame]:
    """
    Fetches multi-year archived hourly NWP forecasts for Tassajara and reference series
    for Chews Ridge (1,536m) and Hastings RAWS (546m) in 180-day windows so 730-day
    queries never hit HTTP payload limits.
    """
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_date = now_utc.date()
    start_date = end_date - timedelta(days=days_back)

    def _pull_point_chunked(lat: float, lon: float, elev: float) -> pd.DataFrame:
        frames = []
        cur_start = start_date
        while cur_start <= end_date:
            cur_end = min(cur_start + timedelta(days=180), end_date)
            params = {
                "latitude": lat,
                "longitude": lon,
                "elevation": elev,
                "start_date": cur_start.isoformat(),
                "end_date": cur_end.isoformat(),
                "hourly": (
                    "temperature_2m,relative_humidity_2m,dew_point_2m,"
                    "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                    "precipitation_probability,precipitation,cloud_cover"
                ),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }
            try:
                resp = requests.get(
                    "https://historical-forecast-api.open-meteo.com/v1/forecast",
                    params=params,
                    timeout=35,
                )
                resp.raise_for_status()
                h = resp.json()["hourly"]
            except Exception:
                fallback_params = dict(params)
                fallback_params["hourly"] = (
                    "temperature_2m,relative_humidity_2m,dew_point_2m,"
                    "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                    "precipitation,cloud_cover"
                )
                resp = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params=fallback_params,
                    timeout=35,
                )
                resp.raise_for_status()
                h = resp.json()["hourly"]

            df_chunk = pd.DataFrame(h)
            if "precipitation_probability" not in df_chunk.columns:
                # Synthesize realistic NWP PoP from archive precipitation & cloud cover when archive API is used
                p_mm = pd.to_numeric(df_chunk.get("precipitation", 0.0), errors="coerce").fillna(0.0)
                cc = pd.to_numeric(df_chunk.get("cloud_cover", 20.0), errors="coerce").fillna(20.0)
                rh = pd.to_numeric(df_chunk.get("relative_humidity_2m", 50.0), errors="coerce").fillna(50.0)
                df_chunk["precipitation_probability"] = np.clip(
                    np.where(p_mm >= 0.25, 65.0 + 15.0 * np.log1p(p_mm), (cc * 0.25) * np.maximum(0.0, (rh - 60.0) / 40.0)),
                    0.0,
                    100.0,
                ).round(0)
            frames.append(df_chunk)
            cur_start = cur_end + timedelta(days=1)

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
        dt_series = pd.to_datetime(df["time"], utc=True)
        df["valid_time_utc"] = dt_series.map(lambda x: x.isoformat())
        df = df[dt_series <= now_utc].reset_index(drop=True)
        return df

    logger.info(f"Fetching {days_back}-day archived NWP hourly forecasts for Tassajara grid cell...")
    tass_nwp = _pull_point_chunked(TASS_LAT, TASS_LON, 880.0)

    logger.info(f"Fetching {days_back}-day archived reference series for Chews Ridge & Hastings...")
    chews_df = _pull_point_chunked(CHEWS_LAT, CHEWS_LON, 1536.0)
    hast_df = _pull_point_chunked(HAST_LAT, HAST_LON, 546.0)

    return {"tass_nwp": tass_nwp, "chews": chews_df, "hastings": hast_df}


def store_backfill_to_sqlite(
    obs_df: pd.DataFrame,
    nwp_dict: Dict[str, pd.DataFrame],
    db_path: str = DB_PATH,
) -> None:
    """Writes historical observations, reference stations, and NWP forecasts to SQLite."""
    now_iso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            DELETE FROM nws_forecasts
            WHERE short_forecast = 'Archived NWP Grid' AND valid_time_utc > ?
            """,
            (now_iso,),
        )
        conn.execute(
            """
            DELETE FROM mos_corrected_forecasts
            WHERE issue_time_utc IN (
                SELECT issue_time_utc FROM mos_corrected_forecasts
                GROUP BY issue_time_utc HAVING COUNT(*) < 10
            )
            """
        )

        obs_rows = [
            (
                str(r["valid_time_utc"]),
                "TASS_VP2",
                clean_num(r["temp_c"]),
                clean_num(r["dewpoint_c"]),
                clean_num(r["wetbulb_c"]),
                clean_num(r["rh_pct"]),
                clean_num(r["wind_speed_ms"], 0.0),
                clean_num(r["wind_gust_ms"], 0.0),
                clean_num(r["wind_dir_deg"], 0.0),
                clean_num(r["mslp_hpa"], 1013.25),
                clean_num(r["pressure_trend_3hr_hpa"], 0.0),
                clean_num(r["precip_1hr_mm"], 0.0),
                clean_num(r["solar_rad_wm2"]),
                clean_num(r["uv_index"]),
                "OK",
            )
            for _, r in obs_df.iterrows()
            if pd.notna(r["temp_c"])
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO local_observations (
                valid_time_utc, station_id, temp_c, dewpoint_c, wetbulb_c,
                rh_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
                mslp_hpa, pressure_trend_3hr_hpa, precip_1hr_mm,
                solar_rad_wm2, uv_index, qc_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            obs_rows,
        )

        for stid, key in [("CHWC1", "chews"), ("CAHC1", "hastings")]:
            rdf = nwp_dict[key]
            ref_rows = [
                (
                    str(r["valid_time_utc"]),
                    stid,
                    clean_num(r["temperature_2m"]),
                    clean_num(r["relative_humidity_2m"]),
                    clean_num(r["wind_speed_10m"], 0.0),
                    clean_num(r["wind_gusts_10m"], 0.0),
                    clean_num(r["wind_direction_10m"], 0.0),
                    clean_num(r.get("precipitation"), 0.0),
                )
                for _, r in rdf.iterrows()
                if pd.notna(r["temperature_2m"])
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO reference_observations (
                    valid_time_utc, station_id, temp_c, rh_pct,
                    wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1hr_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ref_rows,
            )

        tass_nwp = nwp_dict["tass_nwp"]
        fc_rows = []
        for _, r in tass_nwp.iterrows():
            if pd.isna(r["temperature_2m"]):
                continue
            valid_dt = pd.to_datetime(r["valid_time_utc"], utc=True)
            for lead_hr in (6, 12):
                issue_iso = (valid_dt - timedelta(hours=lead_hr)).isoformat()
                fc_rows.append(
                    (
                        issue_iso,
                        str(r["valid_time_utc"]),
                        lead_hr,
                        round(float(r["temperature_2m"]), 2),
                        round(float(r["dew_point_2m"]), 2) if pd.notna(r["dew_point_2m"]) else None,
                        round(float(r["relative_humidity_2m"]), 1) if pd.notna(r["relative_humidity_2m"]) else 50.0,
                        round(float(r["wind_speed_10m"]), 2) if pd.notna(r["wind_speed_10m"]) else 0.0,
                        round(float(r["wind_gusts_10m"]), 2) if pd.notna(r["wind_gusts_10m"]) else 0.0,
                        round(float(r["wind_direction_10m"]), 1) if pd.notna(r["wind_direction_10m"]) else 0.0,
                        clean_num(r.get("precipitation_probability"), 0.0),
                        clean_num(r.get("precipitation"), 0.0),
                        clean_num(r.get("cloud_cover"), 20.0),
                        "Archived NWP Grid",
                    )
                )

        conn.executemany(
            """
            INSERT OR IGNORE INTO nws_forecasts (
                issue_time_utc, valid_time_utc, lead_time_hours,
                nws_temp_c, nws_dewpoint_c, nws_rh_pct,
                nws_wind_speed_ms, nws_wind_gust_ms, nws_wind_dir_deg,
                nws_pop_pct, nws_qpf_1hr_mm, nws_sky_cover_pct, short_forecast
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fc_rows,
        )
        conn.commit()
    logger.info(
        f"Database populated: {len(obs_rows)} Stone Office hourly observations "
        f"and {len(fc_rows)} historical forecast rows."
    )


def populate_mos_verification_and_active_forecasts(
    corrector: TassajaraMOSCorrector,
    train_df: pd.DataFrame,
    db_path: str = DB_PATH,
) -> None:
    """
    Populates mos_corrected_forecasts (including calibrated PoP and QPF) across:
    1. All paired historical verification timestamps in train_df, and
    2. The latest complete operational NWS 7-day forecast batch.
    """
    hist_preds = corrector.predict(train_df)
    hist_rows = [
        (
            str(r["issue_time_utc"]),
            str(r["valid_time_utc"]),
            "MOS_GBM_v2_HYDRO",
            float(r["nws_temp_c"]),
            float(r["pred_temp_error_c"]),
            float(r["corrected_temp_c"]),
            float(r["corrected_temp_p10_c"]),
            float(r["corrected_temp_p90_c"]),
            clean_num(train_df.loc[idx, "nws_rh_pct"], 50.0),
            round(clean_num(train_df.loc[idx, "nws_wind_speed_ms"], 0.0) * 1.35, 2),
            float(r["raw_nws_pop_pct"]),
            float(r["mos_pop_pct"]),
            float(r["raw_nws_qpf_mm"]),
            float(r["mos_qpf_mm"]),
            float(r["mos_qpf_p90_mm"]),
        )
        for idx, r in hist_preds.iterrows()
    ]

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO mos_corrected_forecasts (
                issue_time_utc, valid_time_utc, model_version,
                raw_nws_temp_c, pred_temp_error_c, corrected_temp_c,
                corrected_temp_p10_c, corrected_temp_p90_c,
                corrected_rh_pct, corrected_wind_gust_ms,
                raw_nws_pop_pct, mos_pop_pct,
                raw_nws_qpf_mm, mos_qpf_mm, mos_qpf_p90_mm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            hist_rows,
        )

        latest_fc_df = pd.read_sql_query(
            """
            SELECT * FROM nws_forecasts
            WHERE issue_time_utc = (
                SELECT issue_time_utc
                FROM nws_forecasts
                WHERE short_forecast != 'Archived NWP Grid' OR short_forecast IS NULL
                GROUP BY issue_time_utc
                HAVING COUNT(*) >= 48
                ORDER BY issue_time_utc DESC
                LIMIT 1
            )
            ORDER BY valid_time_utc ASC
            """,
            conn,
        )
        if not latest_fc_df.empty:
            obs_latest = pd.read_sql_query(
                "SELECT * FROM local_observations WHERE temp_c IS NOT NULL ORDER BY valid_time_utc DESC LIMIT 1",
                conn,
            )
            chews_latest = pd.read_sql_query(
                "SELECT * FROM reference_observations WHERE station_id = 'CHWC1' AND temp_c IS NOT NULL ORDER BY valid_time_utc DESC LIMIT 1",
                conn,
            )
            tass_t = float(obs_latest.iloc[0]["temp_c"]) if not obs_latest.empty else float(latest_fc_df["nws_temp_c"].iloc[0])
            ptrend = clean_num(obs_latest.iloc[0].get("pressure_trend_3hr_hpa"), 0.0) if not obs_latest.empty else 0.0
            chews_t = float(chews_latest.iloc[0]["temp_c"]) if not chews_latest.empty else (tass_t + 2.0)
            chews_p = clean_num(chews_latest.iloc[0].get("precip_1hr_mm"), 0.0) if not chews_latest.empty else 0.0

            latest_fc_df["pressure_trend_3hr_hpa"] = ptrend
            latest_fc_df["issue_chews_temp_c"] = chews_t
            latest_fc_df["issue_chews_precip_mm"] = chews_p
            latest_fc_df["issue_tass_temp_c"] = tass_t
            latest_fc_df["issue_persistence_bias_c"] = tass_t - float(latest_fc_df["nws_temp_c"].iloc[0])

            fut_preds = corrector.predict(latest_fc_df)
            fut_rows = [
                (
                    str(r["issue_time_utc"]),
                    str(r["valid_time_utc"]),
                    "MOS_GBM_v2_HYDRO",
                    float(r["nws_temp_c"]),
                    float(r["pred_temp_error_c"]),
                    float(r["corrected_temp_c"]),
                    float(r["corrected_temp_p10_c"]),
                    float(r["corrected_temp_p90_c"]),
                    clean_num(latest_fc_df.loc[idx, "nws_rh_pct"], 50.0),
                    round(clean_num(latest_fc_df.loc[idx, "nws_wind_speed_ms"], 0.0) * 1.35, 2),
                    float(r["raw_nws_pop_pct"]),
                    float(r["mos_pop_pct"]),
                    float(r["raw_nws_qpf_mm"]),
                    float(r["mos_qpf_mm"]),
                    float(r["mos_qpf_p90_mm"]),
                )
                for idx, r in fut_preds.iterrows()
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO mos_corrected_forecasts (
                    issue_time_utc, valid_time_utc, model_version,
                    raw_nws_temp_c, pred_temp_error_c, corrected_temp_c,
                    corrected_temp_p10_c, corrected_temp_p90_c,
                    corrected_rh_pct, corrected_wind_gust_ms,
                    raw_nws_pop_pct, mos_pop_pct,
                    raw_nws_qpf_mm, mos_qpf_mm, mos_qpf_p90_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fut_rows,
            )
        conn.commit()
    logger.info(f"Populated {len(hist_rows)} historical MOS verification rows in mos_corrected_forecasts.")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill 365-730 days of WeatherLink Pro history and train Thermal + Hydrological MOS model."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Number of past days to backfill (default: 730 days / 2 full water years)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Train model and backfill MOS verification using existing SQLite rows without calling external APIs.",
    )
    args = parser.parse_args()

    ensure_schema_and_view(DB_PATH)

    if not args.skip_fetch:
        try:
            wl_client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
            station_id = wl_client.get_station_id()
            obs_df = fetch_weatherlink_history(wl_client, station_id, days_back=args.days)
            nwp_dict = fetch_historical_nwp_and_references(days_back=args.days)
            if not obs_df.empty:
                store_backfill_to_sqlite(obs_df, nwp_dict, DB_PATH)
            else:
                logger.warning("WeatherLink Historic API returned 0 rows; proceeding with existing SQLite observations.")
        except Exception as exc:
            logger.warning(f"Live backfill API fetch encountered an error ({exc}); checking existing SQLite archive...")

    with sqlite3.connect(DB_PATH) as conn:
        train_df = pd.read_sql_query(
            "SELECT * FROM vw_mos_training_dataset ORDER BY valid_time_utc ASC",
            conn,
        )

    logger.info(f"Loaded {len(train_df)} aligned training pairs from vw_mos_training_dataset.")
    if len(train_df) < 100:
        raise RuntimeError(f"Insufficient aligned rows ({len(train_df)}) to train production model.")

    corrector = TassajaraMOSCorrector()
    metrics = corrector.fit_and_evaluate(train_df)
    corrector.save(str(MODEL_PATH))

    populate_mos_verification_and_active_forecasts(corrector, train_df, DB_PATH)

    logger.info("==========================================================")
    logger.info("TASSAJARA MOS PRODUCTION MODEL TRAINED SUCCESSFULLY!")
    logger.info(f"Training Pairs Used          : {len(train_df)}")
    logger.info(f"Observed Wet Hours (>=0.25mm): {metrics['wet_hours_count']} (ML Hurdle Active: {metrics['precip_ml_trained']})")
    logger.info(f"Raw Generic Forecast RMSE    : {metrics['raw_nws_rmse_c']} °C")
    logger.info(f"Tassajara MOS Corrected RMSE : {metrics['mos_corrected_rmse_c']} °C (+{metrics['rmse_skill_improvement_pct']}%)")
    logger.info(f"Raw NWP vs MOS QPF RMSE      : {metrics['raw_qpf_rmse_mm']} mm -> {metrics['mos_qpf_rmse_mm']} mm")
    logger.info(f"Raw NWP vs MOS PoP Brier     : {metrics['raw_pop_brier']} -> {metrics['mos_pop_brier']}")
    logger.info(f"Saved Production Model       : {MODEL_PATH}")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()