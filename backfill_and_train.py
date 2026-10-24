#!/usr/bin/env python3
"""
backfill_and_train.py (v2)
Backfills historical observations from the Stone Office Davis Vantage Pro2
using the WeatherLink v2 Historic API (Pro Tier, 24-hour chunking), pairs them
with archived NWP hourly forecasts and Chews Ridge / Hastings reference series,
trains the TassajaraMOSCorrector Gradient Boosted Tree ensemble, and saves
models/tassajara_mos_v1.joblib.
"""

import os
import time
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import requests
import numpy as np
import pandas as pd

from ingest_weather import (
    WEATHERLINK_API_KEY,
    WEATHERLINK_API_SECRET,
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
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TassajaraBackfill")

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")

TASS_LAT = 36.2337
TASS_LON = -121.5501
CHEWS_LAT = 36.3060
CHEWS_LON = -121.5650
HAST_LAT = 36.3886
HAST_LON = -121.5514


def clean_num(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely converts pandas/numpy values (including NaN/None) to Python float or default."""
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def ensure_schema_and_view(db_path: str = DB_PATH) -> None:
    """Ensures all tables and the updated vw_mos_training_dataset view exist in SQLite."""
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if Path("schema.sql").exists():
            with open("schema.sql", "r", encoding="utf-8") as f:
                conn.executescript(f.read())

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
                o.temp_c AS actual_temp_c,
                o.rh_pct AS actual_rh_pct,
                o.wind_speed_ms AS actual_wind_speed_ms,
                o.wind_gust_ms AS actual_wind_gust_ms,
                o.wind_dir_deg AS actual_wind_dir_deg,
                o.mslp_hpa AS actual_mslp_hpa,
                COALESCE(o.pressure_trend_3hr_hpa, 0.0) AS pressure_trend_3hr_hpa,
                (o.temp_c - f.nws_temp_c) AS temp_error_c,
                (o.rh_pct - f.nws_rh_pct) AS rh_error_pct,
                COALESCE(r_chews.temp_c, f.nws_temp_c + 2.0) AS issue_chews_temp_c,
                COALESCE(r_hast.temp_c, f.nws_temp_c + 0.5) AS issue_hastings_temp_c,
                COALESCE(o_issue.temp_c, o.temp_c) AS issue_tass_temp_c,
                COALESCE(o_issue.temp_c - f.nws_temp_c, 0.0) AS issue_persistence_bias_c
            FROM nws_forecasts f
            INNER JOIN local_observations o
                ON f.valid_time_utc = o.valid_time_utc
            LEFT JOIN local_observations o_issue
                ON f.issue_time_utc = o_issue.valid_time_utc
            LEFT JOIN reference_observations r_chews
                ON f.valid_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
            LEFT JOIN reference_observations r_hast
                ON f.valid_time_utc = r_hast.valid_time_utc AND r_hast.station_id = 'CAHC1'
            WHERE o.qc_flag = 'OK' AND o.temp_c IS NOT NULL AND f.nws_temp_c IS NOT NULL;
            """
        )
        conn.commit()


def fetch_weatherlink_history(
    client: WeatherLinkV2Client,
    station_id: int,
    days_back: int = 90
) -> pd.DataFrame:
    """
    Pulls historical archive records from WeatherLink v2 API (/v2/historic/{station_id}).
    Iterates in 24-hour chunks and coerces all numeric columns to float64 to prevent
    NoneType arithmetic errors on missing barometer or solar records.
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
        f"Fetching {days_back} days of WeatherLink Pro archive for Station {station_id} "
        f"in {total_chunks} daily chunks..."
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

        try:
            resp = client.session.get(url, params=params, timeout=25)
            if resp.status_code == 429:
                logger.warning("Rate limited by WeatherLink API; sleeping 3 seconds...")
                time.sleep(3.0)
                continue
            resp.raise_for_status()
            payload = resp.json()

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
                            "wind_speed_ms": None,
                            "wind_gust_ms": None,
                            "wind_dir_deg": None,
                            "mslp_hpa": None,
                            "precip_1hr_mm": 0.0,
                            "solar_rad_wm2": None,
                            "uv_index": None,
                        }

                    b = hourly_buckets[hr_iso]

                    # Barometric sensor fields (supports all Davis logger / WLL archive keys)
                    bar_sl = (
                        rec.get("bar_sea_level")
                        or rec.get("bar_sea_level_last")
                        or rec.get("bar_final")
                        or rec.get("bar_absolute")
                        or rec.get("bar")
                    )
                    if bar_sl is not None:
                        b["mslp_hpa"] = inhg_to_hpa(bar_sl)

                    # ISS temperature / humidity / wind / rain archive fields
                    t_f = rec.get("temp_last") or rec.get("temp_avg") or rec.get("temp")
                    if t_f is not None:
                        b["temp_c"] = f_to_c(t_f)
                        dew_f = rec.get("dew_point_last") or rec.get("dew_point_avg") or rec.get("dew_point")
                        wet_f = rec.get("wet_bulb_last") or rec.get("wet_bulb")
                        hum = rec.get("hum_last") or rec.get("hum_avg") or rec.get("hum")

                        b["dewpoint_c"] = f_to_c(dew_f)
                        b["wetbulb_c"] = f_to_c(wet_f)
                        b["rh_pct"] = hum

                        w_spd = rec.get("wind_speed_avg") or rec.get("wind_speed_last")
                        w_gst = rec.get("wind_speed_hi") or rec.get("wind_gust_10_min")
                        w_dir = (
                            rec.get("wind_dir_of_prevail")
                            or rec.get("wind_dir_scalar_avg")
                            or rec.get("wind_dir_last")
                        )

                        b["wind_speed_ms"] = mph_to_ms(w_spd)
                        if w_gst is not None:
                            gst_ms = mph_to_ms(w_gst)
                            if gst_ms is not None:
                                b["wind_gust_ms"] = max(b["wind_gust_ms"] or 0.0, gst_ms)
                        b["wind_dir_deg"] = w_dir

                        rain_in = rec.get("rainfall_in")
                        if rain_in is None and rec.get("rainfall_clicks") is not None:
                            rain_in = rec["rainfall_clicks"] * 0.01
                        if rain_in:
                            b["precip_1hr_mm"] = round((b["precip_1hr_mm"] or 0.0) + (in_to_mm(rain_in) or 0.0), 2)

                        sol = rec.get("solar_rad_avg") or rec.get("solar_rad")
                        uv = rec.get("uv_index_avg") or rec.get("uv_index")
                        if sol is not None:
                            b["solar_rad_wm2"] = sol
                        if uv is not None:
                            b["uv_index"] = uv

        except Exception as exc:
            logger.warning(f"Chunk {chunk_idx}/{total_chunks} failed ({current_start}): {exc}")

        if chunk_idx % 10 == 0 or chunk_idx == total_chunks:
            logger.info(
                f"Progress: {chunk_idx}/{total_chunks} days fetched "
                f"({len(hourly_buckets)} hourly observation rows assembled)"
            )

        current_start = current_end
        time.sleep(0.2)

    df = pd.DataFrame(list(hourly_buckets.values()))
    if df.empty:
        return df

    df = df.sort_values("valid_time_utc").reset_index(drop=True)

    # Coerce all telemetry columns to float64 so None values become np.nan
    numeric_cols = [
        "temp_c", "dewpoint_c", "wetbulb_c", "rh_pct",
        "wind_speed_ms", "wind_gust_ms", "wind_dir_deg",
        "mslp_hpa", "precip_1hr_mm", "solar_rad_wm2", "uv_index"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute 3-hour barometric pressure trend safely on float64 series
    if "mslp_hpa" in df.columns and df["mslp_hpa"].notna().any():
        df["pressure_trend_3hr_hpa"] = df["mslp_hpa"].ffill().diff(3).round(2).fillna(0.0)
    else:
        df["pressure_trend_3hr_hpa"] = 0.0

    return df


def fetch_historical_nwp_and_references(days_back: int = 90) -> Dict[str, pd.DataFrame]:
    """
    Fetches archived hourly NWP forecasts for Tassajara and reference series
    for Chews Ridge (1,536m) and Hastings RAWS (546m).
    """
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days_back)

    def _pull_point(lat: float, lon: float, elev: float) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "elevation": elev,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
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
                params=params, timeout=30
            )
            resp.raise_for_status()
        except Exception:
            fallback_params = dict(params)
            fallback_params["hourly"] = (
                "temperature_2m,relative_humidity_2m,dew_point_2m,"
                "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                "precipitation,cloud_cover"
            )
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=fallback_params, timeout=30
            )
            resp.raise_for_status()

        h = resp.json()["hourly"]
        df = pd.DataFrame(h)
        if "precipitation_probability" not in df.columns:
            df["precipitation_probability"] = 0.0
        df["valid_time_utc"] = pd.to_datetime(df["time"], utc=True).map(lambda x: x.isoformat())
        return df

    logger.info("Fetching archived NWP hourly forecasts for Tassajara grid cell...")
    tass_nwp = _pull_point(TASS_LAT, TASS_LON, 880.0)

    logger.info("Fetching archived reference boundary conditions for Chews Ridge & Hastings...")
    chews_df = _pull_point(CHEWS_LAT, CHEWS_LON, 1536.0)
    hast_df = _pull_point(HAST_LAT, HAST_LON, 546.0)

    return {"tass_nwp": tass_nwp, "chews": chews_df, "hastings": hast_df}


def store_backfill_to_sqlite(
    obs_df: pd.DataFrame,
    nwp_dict: Dict[str, pd.DataFrame],
    db_path: str = DB_PATH
) -> None:
    """Writes historical observations, reference stations, and NWP forecasts to SQLite."""
    with sqlite3.connect(db_path) as conn:
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
            valid_dt = pd.to_datetime(r["valid_time_utc"])
            for lead_hr in (6, 12):
                issue_iso = (valid_dt - timedelta(hours=lead_hr)).isoformat()
                fc_rows.append((
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
                ))

        conn.executemany(
            """
            INSERT OR REPLACE INTO nws_forecasts (
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


def main():
    parser = argparse.ArgumentParser(description="Backfill WeatherLink Pro history and train MOS model.")
    parser.add_argument("--days", type=int, default=90, help="Number of past days to backfill (default: 90)")
    args = parser.parse_args()

    ensure_schema_and_view(DB_PATH)

    wl_client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
    station_id = wl_client.get_station_id()

    obs_df = fetch_weatherlink_history(wl_client, station_id, days_back=args.days)
    if obs_df.empty:
        raise RuntimeError(
            "No historical observations returned. Verify that your WeatherLink Pro subscription "
            "is linked to station 36966."
        )

    nwp_dict = fetch_historical_nwp_and_references(days_back=args.days)
    store_backfill_to_sqlite(obs_df, nwp_dict, DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        train_df = pd.read_sql_query("SELECT * FROM vw_mos_training_dataset ORDER BY valid_time_utc ASC", conn)

    logger.info(f"Loaded {len(train_df)} aligned training pairs from vw_mos_training_dataset.")
    if len(train_df) < 100:
        raise RuntimeError(f"Insufficient aligned rows ({len(train_df)}) to train production model.")

    corrector = TassajaraMOSCorrector()
    metrics = corrector.fit_and_evaluate(train_df)
    corrector.save(str(MODEL_PATH))

    logger.info("==========================================================")
    logger.info("TASSAJARA MOS PRODUCTION MODEL TRAINED SUCCESSFULLY!")
    logger.info(f"Training Pairs Used          : {len(train_df)}")
    logger.info(f"Raw Generic Forecast RMSE    : {metrics['raw_nws_rmse_c']} °C")
    logger.info(f"Tassajara MOS Corrected RMSE : {metrics['mos_corrected_rmse_c']} °C")
    logger.info(f"Forecast Skill Improvement   : +{metrics['rmse_skill_improvement_pct']}%")
    logger.info(f"Saved Production Model       : {MODEL_PATH}")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()
