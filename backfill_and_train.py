#!/usr/bin/env python3
"""
backfill_and_train.py (v5.3 — Provenance-Tagged, Leak-Free 5-Station Santa Lucia MOS Backfill)
Backfills up to 730 days of historical observations from the Stone Office Davis
Vantage Pro2 using the WeatherLink v2 Historic API (Pro Tier, 24-hour chunking),
merges observations with source provenance into SQLite (allowing authentic WL_HISTORIC
records to supersede smeared live estimates while protecting real RAWS archives),
builds a deduplicated leak-free vw_mos_training_dataset view, trains TassajaraMOSCorrector,
and populates mos_corrected_forecasts.
"""

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
    WEATHERLINK_API_KEY,
    WEATHERLINK_API_SECRET,
    WEATHERLINK_BASE_URL,
    REFERENCE_STATIONS,
    INDOOR_OR_HEALTH_DST,
    WeatherLinkV2Client,
    extract_archive_rain_mm,
    extract_record_wind,
    f_to_c,
    inhg_to_hpa,
)
from mos_model import (
    TassajaraMOSCorrector,
    engineer_mos_features,
    analytical_hydro_hurdle_fallback,
    compute_diurnal_ewma_bias,
    predict_dual_baseline_with_ewma,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TassajaraBackfill")

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")

TASS_LAT = 36.2337
TASS_LON = -121.5501


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
    Ensures all tables, provenance columns, hydrological columns, and the
    deduplicated, leak-free 5-station vw_mos_training_dataset view exist in SQLite.
    """
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect(db_path, timeout=30) as conn:
        if Path("schema.sql").exists():
            with open("schema.sql", "r", encoding="utf-8") as f:
                conn.executescript(f.read())

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS local_observations (
                valid_time_utc TEXT PRIMARY KEY,
                station_id TEXT DEFAULT 'TASS_VP2',
                temp_c REAL,
                dewpoint_c REAL,
                wetbulb_c REAL,
                rh_pct REAL,
                wind_speed_ms REAL DEFAULT 0.0,
                wind_gust_ms REAL DEFAULT 0.0,
                wind_dir_deg REAL,
                mslp_hpa REAL,
                pressure_trend_3hr_hpa REAL DEFAULT 0.0,
                precip_1hr_mm REAL DEFAULT 0.0,
                solar_rad_wm2 REAL,
                uv_index REAL,
                qc_flag TEXT DEFAULT 'OK',
                source TEXT DEFAULT 'WL_LIVE'
            );

            CREATE TABLE IF NOT EXISTS reference_observations (
                valid_time_utc TEXT,
                station_id TEXT,
                temp_c REAL,
                rh_pct REAL,
                wind_speed_ms REAL DEFAULT 0.0,
                wind_gust_ms REAL DEFAULT 0.0,
                wind_dir_deg REAL,
                precip_1hr_mm REAL DEFAULT 0.0,
                source TEXT DEFAULT 'RAWS_SYNOPTIC',
                PRIMARY KEY (valid_time_utc, station_id)
            );

            CREATE TABLE IF NOT EXISTS nws_forecasts (
                issue_time_utc TEXT,
                valid_time_utc TEXT,
                lead_time_hours INTEGER,
                nws_temp_c REAL,
                nws_dewpoint_c REAL,
                nws_rh_pct REAL,
                nws_wind_speed_ms REAL DEFAULT 0.0,
                nws_wind_gust_ms REAL DEFAULT 0.0,
                nws_wind_dir_deg REAL DEFAULT 0.0,
                nws_pop_pct REAL DEFAULT 0.0,
                nws_qpf_1hr_mm REAL DEFAULT 0.0,
                nws_sky_cover_pct REAL DEFAULT 20.0,
                short_forecast TEXT,
                om_temp_c REAL,
                om_dewpoint_c REAL,
                om_rh_pct REAL,
                om_wind_speed_ms REAL,
                om_wind_gust_ms REAL,
                om_wind_dir_deg REAL,
                om_sky_cover_pct REAL,
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
                raw_nws_pop_pct REAL,
                mos_pop_pct REAL,
                raw_nws_qpf_mm REAL,
                mos_qpf_mm REAL,
                mos_qpf_p90_mm REAL,
                PRIMARY KEY (issue_time_utc, valid_time_utc, model_version)
            );
            """
        )

        mos_cols = {
            r[1]
            for r in conn.execute("PRAGMA table_info(mos_corrected_forecasts)").fetchall()
        }
        for col_name in (
            "raw_nws_pop_pct",
            "mos_pop_pct",
            "raw_nws_qpf_mm",
            "mos_qpf_mm",
            "mos_qpf_p90_mm",
            "ewma_bias_applied_c",
        ):
            if col_name not in mos_cols:
                dflt_sql = " DEFAULT 0.0" if col_name == "ewma_bias_applied_c" else ""
                conn.execute(
                    f"ALTER TABLE mos_corrected_forecasts ADD COLUMN {col_name} REAL{dflt_sql};"
                )

        nws_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(nws_forecasts)").fetchall()
        }
        for col_name, dflt in (
            ("nws_wind_gust_ms", "0.0"),
            ("nws_qpf_1hr_mm", "0.0"),
            ("nws_sky_cover_pct", "20.0"),
        ):
            if col_name not in nws_cols:
                conn.execute(
                    f"ALTER TABLE nws_forecasts ADD COLUMN {col_name} REAL DEFAULT {dflt};"
                )
        for om_col in (
            "om_temp_c",
            "om_dewpoint_c",
            "om_rh_pct",
            "om_wind_speed_ms",
            "om_wind_gust_ms",
            "om_wind_dir_deg",
            "om_sky_cover_pct",
        ):
            if om_col not in nws_cols:
                conn.execute(f"ALTER TABLE nws_forecasts ADD COLUMN {om_col} REAL;")

        # Idempotent provenance column migrations for existing databases
        for tbl, default_src in (
            ("local_observations", "'WL_LIVE'"),
            ("reference_observations", "'RAWS_SYNOPTIC'"),
        ):
            t_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()}
            if "source" not in t_cols:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN source TEXT DEFAULT {default_src};")

        conn.executescript(
            """
            DROP INDEX IF EXISTS idx_nws_valid_time;
            DROP VIEW IF EXISTS vw_mos_training_dataset;
            CREATE VIEW vw_mos_training_dataset AS
            WITH canonical_fc AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY valid_time_utc
                        ORDER BY
                            CASE
                                WHEN om_temp_c IS NOT NULL THEN 0
                                WHEN short_forecast = 'Archived NWP Grid' THEN 1
                                ELSE 2
                            END ASC,
                            ABS(lead_time_hours - 6) ASC,
                            issue_time_utc DESC
                    ) AS rn
                FROM nws_forecasts
                WHERE nws_temp_c IS NOT NULL
                  AND lead_time_hours >= 1
                  AND issue_time_utc < valid_time_utc
            )
            SELECT
                f.issue_time_utc,
                f.valid_time_utc,
                f.lead_time_hours,
                COALESCE(f.om_temp_c, f.nws_temp_c) AS nws_temp_c,
                COALESCE(f.om_dewpoint_c, f.nws_dewpoint_c) AS nws_dewpoint_c,
                COALESCE(f.om_rh_pct, f.nws_rh_pct) AS nws_rh_pct,
                COALESCE(f.om_wind_speed_ms, f.nws_wind_speed_ms) AS nws_wind_speed_ms,
                COALESCE(f.nws_wind_gust_ms, COALESCE(f.om_wind_speed_ms, f.nws_wind_speed_ms) * 1.35) AS nws_wind_gust_ms,
                COALESCE(f.om_wind_dir_deg, f.nws_wind_dir_deg) AS nws_wind_dir_deg,
                COALESCE(f.om_sky_cover_pct, f.nws_sky_cover_pct, 20.0) AS nws_sky_cover_pct,
                COALESCE(f.nws_pop_pct, 0.0) AS nws_pop_pct,
                COALESCE(f.nws_qpf_1hr_mm, 0.0) AS nws_qpf_1hr_mm,
                o.temp_c AS actual_temp_c,
                o.rh_pct AS actual_rh_pct,
                o.wind_speed_ms AS actual_wind_speed_ms,
                o.wind_gust_ms AS actual_wind_gust_ms,
                o.wind_dir_deg AS actual_wind_dir_deg,
                o.mslp_hpa AS actual_mslp_hpa,
                -- LEAK-FREE: barometric tendency at forecast ISSUE time, never target valid_time_utc
                COALESCE(o_issue.pressure_trend_3hr_hpa, 0.0) AS pressure_trend_3hr_hpa,
                COALESCE(o.precip_1hr_mm, 0.0) AS actual_precip_mm,
                (o.temp_c - COALESCE(f.om_temp_c, f.nws_temp_c)) AS temp_error_c,
                (o.rh_pct - COALESCE(f.om_rh_pct, f.nws_rh_pct)) AS rh_error_pct,
                -- CHWC1: Chews Ridge (1,536m Interior Crest)
                COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.83) AS issue_chews_temp_c,
                COALESCE(r_chews.rh_pct, f.nws_rh_pct) AS issue_chews_rh_pct,
                COALESCE(r_chews.precip_1hr_mm, 0.0) AS issue_chews_precip_mm,
                -- CAHC1: Hastings Reservation (546m Foothill Valley)
                COALESCE(r_hast.temp_c, r_arroyo.temp_c - 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) - 0.60) AS issue_hastings_temp_c,
                -- ASRC1: Arroyo Seco RAWS (268m Gorge Outlet)
                COALESCE(r_arroyo.temp_c, r_hast.temp_c + 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) + 1.41) AS issue_arroyo_temp_c,
                COALESCE(r_arroyo.rh_pct, r_hast.rh_pct, f.nws_rh_pct) AS issue_arroyo_rh_pct,
                -- PPSC1: Big Sur RAWS (108m Coastal Slope)
                COALESCE(r_bigsur.temp_c, COALESCE(o_issue.temp_c, f.nws_temp_c) + 2.45) AS issue_bigsur_temp_c,
                COALESCE(r_bigsur.rh_pct, f.nws_rh_pct) AS issue_bigsur_rh_pct,
                COALESCE(r_bigsur.precip_1hr_mm, 0.0) AS issue_bigsur_precip_mm,
                -- VTCC1: Ventana Cone (1,453m Windward Crest)
                COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.29) AS issue_ventana_temp_c,
                COALESCE(r_vent.precip_1hr_mm, r_chews.precip_1hr_mm, 0.0) AS issue_ventana_precip_mm,
                -- Local Stone Office Initialization State (strictly causal forecast at issue_time_utc)
                COALESCE(o_issue.temp_c, f.nws_temp_c) AS issue_tass_temp_c,
                COALESCE(
                    o_issue.temp_c - (
                        SELECT COALESCE(fi.om_temp_c, fi.nws_temp_c) FROM nws_forecasts fi
                        WHERE fi.valid_time_utc = f.issue_time_utc
                          AND fi.issue_time_utc <= f.issue_time_utc
                          AND (fi.short_forecast = 'Archived NWP Grid' OR fi.om_temp_c IS NOT NULL)
                        ORDER BY fi.issue_time_utc DESC LIMIT 1
                    ),
                    0.0
                ) AS issue_persistence_bias_c
            FROM canonical_fc f
            INNER JOIN local_observations o
                ON f.valid_time_utc = o.valid_time_utc
            LEFT JOIN local_observations o_issue
                ON f.issue_time_utc = o_issue.valid_time_utc
            LEFT JOIN reference_observations r_chews
                ON f.issue_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
            LEFT JOIN reference_observations r_hast
                ON f.issue_time_utc = r_hast.valid_time_utc AND r_hast.station_id = 'CAHC1'
            LEFT JOIN reference_observations r_arroyo
                ON f.issue_time_utc = r_arroyo.valid_time_utc AND r_arroyo.station_id = 'ASRC1'
            LEFT JOIN reference_observations r_bigsur
                ON f.issue_time_utc = r_bigsur.valid_time_utc AND r_bigsur.station_id = 'PPSC1'
            LEFT JOIN reference_observations r_vent
                ON f.issue_time_utc = r_vent.valid_time_utc AND r_vent.station_id = 'VTCC1'
            WHERE f.rn = 1
              AND o.qc_flag = 'OK'
              AND COALESCE(o.source, 'WL_LIVE') != 'OPENMETEO_SYNTH'
              AND o.temp_c IS NOT NULL;
            """
        )
        conn.commit()


def fetch_weatherlink_history(
    client: WeatherLinkV2Client,
    station_id: int,
    days_back: float = 730.0,
) -> pd.DataFrame:
    """
    Pulls historical archive records from WeatherLink v2 API (/v2/historic/{station_id}).
    Strictly filters out indoor/health sensors, deduplicates rain tips at station-timestamp
    scope, averages all wind speeds (including calm 0.0 m/s packets) and hourly temperatures,
    and preserves True North (0.0 deg) wind directions.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=float(days_back))

    chunk_seconds = 86400  # 24 hours max per WeatherLink v2 API spec
    current_start = int(start_dt.timestamp())
    final_end = int(end_dt.timestamp())

    hourly_buckets: Dict[str, Dict[str, Any]] = {}
    total_chunks = max(1, int(np.ceil((final_end - current_start) / chunk_seconds)))
    chunk_idx = 0

    logger.info(
        f"Fetching {days_back:.1f} days of WeatherLink Pro archive for Station {station_id} "
        f"in {total_chunks} chunk(s)..."
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
            resp = client._get_with_retry(url, params=params, timeout=25)
            payload = resp.json()

            # Station-level rain timestamp dedup prevents double-counting across multi-sensor packets
            seen_rain_ts = set()

            for sensor in payload.get("sensors", []):
                dst = int(sensor.get("data_structure_type") or 0)
                stype = int(sensor.get("sensor_type") or 0)

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
                            "_temps_c": [],
                            "temp_c": None,
                            "dewpoint_c": None,
                            "wetbulb_c": None,
                            "rh_pct": None,
                            "_wind_speeds_ms": [],
                            "wind_speed_ms": 0.0,
                            "wind_gust_ms": 0.0,
                            "wind_dir_deg": None,
                            "mslp_hpa": None,
                            "precip_1hr_mm": 0.0,
                            "solar_rad_wm2": None,
                            "uv_index": None,
                        }

                    b = hourly_buckets[hr_iso]

                    # 1. Barometric pressure (from Barometer sensor or ISS/Console packet)
                    bar_sl = first_not_none(
                        rec.get("bar_sea_level"),
                        rec.get("bar_sea_level_last"),
                        rec.get("bar_final"),
                        rec.get("bar"),
                        rec.get("pressure_last"),
                        rec.get("bar_absolute"),
                    )
                    if bar_sl is not None:
                        b["mslp_hpa"] = inhg_to_hpa(bar_sl)

                    # Skip indoor/health/AirLink sensors for outdoor temperature, wind, and rain
                    if (dst in INDOOR_OR_HEALTH_DST or stype in {323, 326, 504, 506}) and "temp_out" not in rec:
                        continue

                    # 2. Accumulate tipping-bucket rain across sub-hourly archive intervals
                    if ts not in seen_rain_ts:
                        rain_mm_val = extract_archive_rain_mm(rec)
                        if rain_mm_val > 0.0:
                            b["precip_1hr_mm"] = round(
                                (b["precip_1hr_mm"] or 0.0) + rain_mm_val, 3
                            )
                            seen_rain_ts.add(ts)

                    # 3. Outdoor temperature (hourly mean) & humidity
                    t_f = first_not_none(
                        rec.get("temp_out"),
                        rec.get("temp_out_last"),
                        rec.get("temp_out_avg"),
                        rec.get("temp_last"),
                        rec.get("temp_avg"),
                        rec.get("temp"),
                    )
                    if t_f is not None:
                        t_c_val = f_to_c(t_f)
                        if t_c_val is not None:
                            b["_temps_c"].append(t_c_val)
                            b["temp_c"] = round(float(np.mean(b["_temps_c"])), 2)
                        dew_f = first_not_none(
                            rec.get("dew_point_out"),
                            rec.get("dew_point_last"),
                            rec.get("dew_point_avg"),
                            rec.get("dew_point"),
                        )
                        wet_f = first_not_none(
                            rec.get("wet_bulb_out"),
                            rec.get("wet_bulb_last"),
                            rec.get("wet_bulb_avg"),
                            rec.get("wet_bulb"),
                        )
                        hum = first_not_none(
                            rec.get("hum_out"),
                            rec.get("hum_out_last"),
                            rec.get("hum_last"),
                            rec.get("hum_avg"),
                            rec.get("hum"),
                        )
                        if dew_f is not None:
                            b["dewpoint_c"] = f_to_c(dew_f)
                        if wet_f is not None:
                            b["wetbulb_c"] = f_to_c(wet_f)
                        if hum is not None:
                            b["rh_pct"] = float(hum)

                    # 4. Wind speed (including calm 0.0 m/s packets), gust, and True North 0.0 deg
                    spd_ms, gst_ms, dir_deg = extract_record_wind(rec)
                    if spd_ms is not None and spd_ms >= 0.0:
                        b["_wind_speeds_ms"].append(spd_ms)
                        b["wind_speed_ms"] = round(float(np.mean(b["_wind_speeds_ms"])), 2)
                    if gst_ms is not None and gst_ms >= 0.0:
                        b["wind_gust_ms"] = max(b["wind_gust_ms"] or 0.0, gst_ms, b["wind_speed_ms"])
                    if dir_deg is not None and 0.0 <= dir_deg <= 360.0:
                        b["wind_dir_deg"] = dir_deg

                    sol = first_not_none(rec.get("solar_rad_avg"), rec.get("solar_rad"))
                    uv = first_not_none(rec.get("uv_index_avg"), rec.get("uv_index"), rec.get("uv"))
                    if sol is not None:
                        b["solar_rad_wm2"] = float(sol)
                    if uv is not None:
                        b["uv_index"] = float(uv)

        except Exception as exc:
            logger.warning(f"Chunk {chunk_idx}/{total_chunks} failed ({current_start}): {exc}")

        if chunk_idx % 25 == 0 or chunk_idx == total_chunks:
            wet_cnt = sum(
                1
                for v in hourly_buckets.values()
                if (v.get("precip_1hr_mm") or 0.0) >= 0.254
            )
            logger.info(
                f"Progress: {chunk_idx}/{total_chunks} daily chunks fetched "
                f"({len(hourly_buckets)} hourly rows, {wet_cnt} wet hours >= 0.25mm)"
            )

        current_start = current_end
        if total_chunks > 2:
            time.sleep(0.18)

    for b in hourly_buckets.values():
        b.pop("_wind_speeds_ms", None)
        b.pop("_temps_c", None)

    df = pd.DataFrame(list(hourly_buckets.values()))
    if df.empty:
        return df

    df = df.sort_values("valid_time_utc").reset_index(drop=True)

    numeric_cols = [
        "temp_c",
        "dewpoint_c",
        "wetbulb_c",
        "rh_pct",
        "wind_speed_ms",
        "wind_gust_ms",
        "wind_dir_deg",
        "mslp_hpa",
        "precip_1hr_mm",
        "solar_rad_wm2",
        "uv_index",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mslp_hpa" in df.columns and df["mslp_hpa"].notna().any():
        df["mslp_hpa"] = df["mslp_hpa"].ffill().bfill().fillna(1013.25)
        _dt_idx = pd.to_datetime(df["valid_time_utc"], utc=True)
        _mslp_hr = pd.Series(df["mslp_hpa"].values, index=_dt_idx).sort_index()
        _mslp_hr = _mslp_hr[~_mslp_hr.index.duplicated(keep="last")]
        _full_idx = pd.date_range(_mslp_hr.index.min(), _mslp_hr.index.max(), freq="1h", tz="UTC")
        _trend_3h = _mslp_hr.reindex(_full_idx).diff(3).round(2)
        df["pressure_trend_3hr_hpa"] = _dt_idx.map(_trend_3h).fillna(0.0).values
    else:
        df["mslp_hpa"] = 1013.25
        df["pressure_trend_3hr_hpa"] = 0.0

    return df


def fetch_historical_nwp_and_references(days_back: int = 730) -> Dict[str, pd.DataFrame]:
    """
    Fetches archived hourly NWP forecasts for Tassajara and boundary series for all
    5 Santa Lucia reference stations (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) in 180-day
    chunks with rate-limit backoff. Clamps all returned rows to valid_time_utc <= now_utc.
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
            h = None
            for attempt in range(3):
                try:
                    resp = requests.get(
                        "https://historical-forecast-api.open-meteo.com/v1/forecast",
                        params=params,
                        timeout=35,
                    )
                    if resp.status_code == 429:
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    h = resp.json().get("hourly")
                    break
                except Exception:
                    try:
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
                        if resp.status_code == 429:
                            time.sleep(2.5 * (attempt + 1))
                            continue
                        resp.raise_for_status()
                        h = resp.json().get("hourly")
                        break
                    except Exception as exc:
                        if attempt == 2:
                            logger.warning(
                                f"Open-Meteo chunk {cur_start}..{cur_end} failed: {exc}"
                            )
                        time.sleep(1.5 * (attempt + 1))

            if h:
                df_chunk = pd.DataFrame(h)
                if "precipitation_probability" not in df_chunk.columns:
                    p_mm = pd.to_numeric(
                        df_chunk.get("precipitation", 0.0), errors="coerce"
                    ).fillna(0.0)
                    cc = pd.to_numeric(
                        df_chunk.get("cloud_cover", 20.0), errors="coerce"
                    ).fillna(20.0)
                    rh = pd.to_numeric(
                        df_chunk.get("relative_humidity_2m", 50.0), errors="coerce"
                    ).fillna(50.0)
                    df_chunk["precipitation_probability"] = np.clip(
                        np.where(
                            p_mm >= 0.25,
                            65.0 + 15.0 * np.log1p(p_mm),
                            (cc * 0.25) * np.maximum(0.0, (rh - 60.0) / 40.0),
                        ),
                        0.0,
                        100.0,
                    ).round(0)
                frames.append(df_chunk)
            cur_start = cur_end + timedelta(days=1)
            time.sleep(0.35)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
        dt_series = pd.to_datetime(df["time"], utc=True)
        df["valid_time_utc"] = dt_series.map(lambda x: x.isoformat())
        df = df[dt_series <= now_utc].reset_index(drop=True)
        return df

    logger.info("Fetching archived NWP hourly forecasts for Tassajara grid cell...")
    out_dict: Dict[str, pd.DataFrame] = {
        "tass_nwp": _pull_point_chunked(TASS_LAT, TASS_LON, 880.0)
    }

    for stid, meta in REFERENCE_STATIONS.items():
        logger.info(
            f"Fetching archived reference boundary conditions for {stid} "
            f"({meta['name']}, {meta['elev_m']:.0f}m)..."
        )
        out_dict[stid] = _pull_point_chunked(
            float(meta["lat"]), float(meta["lon"]), float(meta["elev_m"])
        )

    return out_dict


def store_observations_to_sqlite(
    obs_df: pd.DataFrame,
    db_path: str = DB_PATH,
    source: str = "WL_HISTORIC",
) -> int:
    """
    Upserts WeatherLink local observations to SQLite with provenance tracking:
    authentic WL_HISTORIC archive records can overwrite previously smeared live rain,
    while preserving non-null wind directions (including True North 0.0 deg).
    """
    if obs_df is None or obs_df.empty:
        return 0
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
            clean_num(r["wind_dir_deg"], None),
            clean_num(r["mslp_hpa"], 1013.25),
            clean_num(r["pressure_trend_3hr_hpa"], 0.0),
            clean_num(r["precip_1hr_mm"], 0.0),
            clean_num(r["solar_rad_wm2"]),
            clean_num(r["uv_index"]),
            "OK",
            str(source),
        )
        for _, r in obs_df.iterrows()
        if pd.notna(r["temp_c"])
    ]
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.executemany(
            """
            INSERT INTO local_observations (
                valid_time_utc, station_id, temp_c, dewpoint_c, wetbulb_c,
                rh_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
                mslp_hpa, pressure_trend_3hr_hpa, precip_1hr_mm,
                solar_rad_wm2, uv_index, qc_flag, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(valid_time_utc) DO UPDATE SET
                temp_c = COALESCE(excluded.temp_c, local_observations.temp_c),
                dewpoint_c = COALESCE(excluded.dewpoint_c, local_observations.dewpoint_c),
                wetbulb_c = COALESCE(excluded.wetbulb_c, local_observations.wetbulb_c),
                rh_pct = COALESCE(excluded.rh_pct, local_observations.rh_pct),
                wind_speed_ms = CASE
                    WHEN excluded.source = 'WL_HISTORIC' THEN COALESCE(excluded.wind_speed_ms, local_observations.wind_speed_ms)
                    ELSE MAX(COALESCE(local_observations.wind_speed_ms, 0.0), COALESCE(excluded.wind_speed_ms, 0.0))
                END,
                wind_gust_ms = MAX(COALESCE(local_observations.wind_gust_ms, 0.0), COALESCE(excluded.wind_gust_ms, 0.0)),
                wind_dir_deg = COALESCE(excluded.wind_dir_deg, local_observations.wind_dir_deg),
                mslp_hpa = COALESCE(excluded.mslp_hpa, local_observations.mslp_hpa),
                pressure_trend_3hr_hpa = COALESCE(excluded.pressure_trend_3hr_hpa, local_observations.pressure_trend_3hr_hpa),
                precip_1hr_mm = CASE
                    WHEN excluded.source = 'WL_HISTORIC' THEN COALESCE(excluded.precip_1hr_mm, local_observations.precip_1hr_mm)
                    ELSE MAX(COALESCE(local_observations.precip_1hr_mm, 0.0), COALESCE(excluded.precip_1hr_mm, 0.0))
                END,
                solar_rad_wm2 = COALESCE(excluded.solar_rad_wm2, local_observations.solar_rad_wm2),
                uv_index = COALESCE(excluded.uv_index, local_observations.uv_index),
                source = CASE
                    WHEN local_observations.source = 'WL_HISTORIC' AND excluded.source != 'WL_HISTORIC' THEN local_observations.source
                    ELSE excluded.source
                END
            """,
            obs_rows,
        )
        conn.commit()
    logger.info(
        f"Upserted {len(obs_rows)} Stone Office hourly observation rows ({source}) to SQLite."
    )
    return len(obs_rows)


def store_backfill_to_sqlite(
    obs_df: Optional[pd.DataFrame],
    nwp_dict: Dict[str, pd.DataFrame],
    db_path: str = DB_PATH,
) -> None:
    """
    Writes historical observations, 5-station reference series (INSERT OR IGNORE so real
    RAWS/NWS archives are never overwritten), and single-lead (6h) archived NWP forecasts.
    """
    now_iso = (
        datetime.now(timezone.utc)
        .replace(minute=0, second=0, microsecond=0)
        .isoformat()
    )

    if obs_df is not None and not obs_df.empty:
        store_observations_to_sqlite(obs_df, db_path, source="WL_HISTORIC")

    with sqlite3.connect(db_path, timeout=30) as conn:
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

        for stid in REFERENCE_STATIONS.keys():
            rdf = nwp_dict.get(stid, pd.DataFrame())
            if rdf is None or rdf.empty:
                continue
            ref_rows = [
                (
                    str(r["valid_time_utc"]),
                    stid,
                    clean_num(r["temperature_2m"]),
                    clean_num(r["relative_humidity_2m"]),
                    clean_num(r["wind_speed_10m"], 0.0),
                    clean_num(r["wind_gusts_10m"], 0.0),
                    clean_num(r["wind_direction_10m"], None),
                    clean_num(r.get("precipitation"), 0.0),
                    "OPENMETEO_SYNTH",
                )
                for _, r in rdf.iterrows()
                if pd.notna(r["temperature_2m"])
            ]
            # INSERT OR IGNORE preserves genuine Synoptic/NWS station observations
            conn.executemany(
                """
                INSERT OR IGNORE INTO reference_observations (
                    valid_time_utc, station_id, temp_c, rh_pct,
                    wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1hr_mm, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ref_rows,
            )

        tass_nwp = nwp_dict.get("tass_nwp", pd.DataFrame())
        fc_rows = []
        if tass_nwp is not None and not tass_nwp.empty:
            for _, r in tass_nwp.iterrows():
                if pd.isna(r["temperature_2m"]):
                    continue
                valid_dt = pd.to_datetime(r["valid_time_utc"], utc=True)
                # Single canonical 6-hour lead per archived timestamp (prevents 2x pseudoreplication)
                for lead_hr in (6,):
                    issue_iso = (valid_dt - timedelta(hours=lead_hr)).isoformat()
                    fc_rows.append(
                        (
                            issue_iso,
                            str(r["valid_time_utc"]),
                            lead_hr,
                            round(float(r["temperature_2m"]), 2),
                            round(float(r["dew_point_2m"]), 2)
                            if pd.notna(r["dew_point_2m"])
                            else None,
                            round(float(r["relative_humidity_2m"]), 1)
                            if pd.notna(r["relative_humidity_2m"])
                            else 50.0,
                            round(float(r["wind_speed_10m"]), 2)
                            if pd.notna(r["wind_speed_10m"])
                            else 0.0,
                            round(float(r["wind_gusts_10m"]), 2)
                            if pd.notna(r["wind_gusts_10m"])
                            else 0.0,
                            round(float(r["wind_direction_10m"]), 1)
                            if pd.notna(r["wind_direction_10m"])
                            else 0.0,
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
        f"NWP & 5-Station Reference backfill stored: {len(fc_rows)} historical forecast rows."
    )


def _ensure_hydro_pred_cols(preds: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
    if "mos_qpf_mm" in preds.columns and "mos_pop_pct" in preds.columns:
        return preds
    out = preds.copy()
    X_f = engineer_mos_features(src_df)
    h_fb = analytical_hydro_hurdle_fallback(X_f)
    out["raw_nws_pop_pct"] = X_f["nws_pop_pct"].round(1).values
    out["mos_pop_pct"] = h_fb["mos_pop_pct"].values
    out["raw_nws_qpf_mm"] = X_f["nws_qpf_1hr_mm"].round(3).values
    out["mos_qpf_mm"] = h_fb["mos_qpf_mm"].values
    out["mos_qpf_p90_mm"] = h_fb["mos_qpf_p90_mm"].values
    return out


def populate_mos_verification_and_active_forecasts(
    corrector: TassajaraMOSCorrector,
    train_df: pd.DataFrame,
    db_path: str = DB_PATH,
    oof_preds_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Populates all 15 thermal + hydrological columns in mos_corrected_forecasts across:
    1. Historical verification timestamps (strictly INSERT OR IGNORE; never overwrites
       genuine live operational rows; prefers out-of-fold CV predictions when available), and
    2. Future hours (>= now_utc) of the latest operational NWS 7-day forecast batch.
    """
    now_iso = (
        datetime.now(timezone.utc)
        .replace(minute=0, second=0, microsecond=0)
        .isoformat()
    )
    use_oof = isinstance(oof_preds_df, pd.DataFrame) and not oof_preds_df.empty
    if use_oof:
        hist_preds = oof_preds_df.reset_index(drop=True)
        hist_src = train_df.loc[oof_preds_df.index].reset_index(drop=True)
        hist_model_ver = "MOS_GBM_v3_OOF"
    else:
        hist_preds = _ensure_hydro_pred_cols(corrector.predict(train_df), train_df)
        hist_src = train_df.reset_index(drop=True)
        hist_model_ver = "MOS_GBM_v3_HYDRO"

    with sqlite3.connect(db_path, timeout=30) as conn:
        existing_op_pairs = set(
            conn.execute(
                """
                SELECT issue_time_utc, valid_time_utc
                FROM mos_corrected_forecasts
                WHERE model_version != 'MOS_GBM_v3_OOF'
                """
            ).fetchall()
        )
    hist_rows = [
        (
            str(r["issue_time_utc"]),
            str(r["valid_time_utc"]),
            hist_model_ver,
            float(r["nws_temp_c"]),
            float(r["pred_temp_error_c"]),
            float(r["corrected_temp_c"]),
            float(r["corrected_temp_p10_c"]),
            float(r["corrected_temp_p90_c"]),
            clean_num(hist_src.loc[idx, "nws_rh_pct"], 50.0),
            round(
                max(
                    clean_num(hist_src.loc[idx, "nws_wind_gust_ms"], 0.0) or 0.0,
                    (clean_num(hist_src.loc[idx, "nws_wind_speed_ms"], 0.0) or 0.0) * 1.35,
                ),
                2,
            ),
            clean_num(r.get("raw_nws_pop_pct"), 0.0),
            clean_num(r.get("mos_pop_pct"), 0.0),
            clean_num(r.get("raw_nws_qpf_mm"), 0.0),
            clean_num(r.get("mos_qpf_mm"), 0.0),
            clean_num(r.get("mos_qpf_p90_mm"), 0.0),
        )
        for idx, r in hist_preds.iterrows()
        if (str(r["issue_time_utc"]), str(r["valid_time_utc"])) not in existing_op_pairs
    ]

    with sqlite3.connect(db_path, timeout=30) as conn:
        if use_oof:
            conn.execute("DELETE FROM mos_corrected_forecasts WHERE model_version = 'MOS_GBM_v3_OOF'")
        if hist_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO mos_corrected_forecasts (
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
              AND valid_time_utc >= ?
            ORDER BY valid_time_utc ASC
            """,
            conn,
            params=(now_iso,),
        )
        if not latest_fc_df.empty:
            obs_latest = pd.read_sql_query(
                "SELECT * FROM local_observations WHERE temp_c IS NOT NULL ORDER BY valid_time_utc DESC LIMIT 1",
                conn,
            )
            anchor_ts = (
                str(obs_latest.iloc[0]["valid_time_utc"])
                if not obs_latest.empty
                else datetime.now(timezone.utc).isoformat()
            )
            refs_latest_df = pd.read_sql_query(
                """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY valid_time_utc DESC) AS rn
                    FROM reference_observations
                    WHERE (temp_c IS NOT NULL OR precip_1hr_mm IS NOT NULL)
                      AND valid_time_utc >= strftime('%Y-%m-%dT%H:%M:%S+00:00', ?, '-6 hours')
                )
                SELECT * FROM ranked WHERE rn = 1
                """,
                conn,
                params=(anchor_ts,),
            )
            ref_map = (
                {str(r["station_id"]): r for _, r in refs_latest_df.iterrows()}
                if not refs_latest_df.empty
                else {}
            )

            _om_t0 = clean_num(latest_fc_df["om_temp_c"].iloc[0], None) if "om_temp_c" in latest_fc_df.columns else None
            _om_rh0 = clean_num(latest_fc_df["om_rh_pct"].iloc[0], None) if "om_rh_pct" in latest_fc_df.columns else None
            base_nws_t = _om_t0 if _om_t0 is not None else float(latest_fc_df["nws_temp_c"].iloc[0])
            base_nws_rh = _om_rh0 if _om_rh0 is not None else (clean_num(latest_fc_df["nws_rh_pct"].iloc[0], 50.0) or 50.0)
            tass_t = (
                float(obs_latest.iloc[0]["temp_c"])
                if not obs_latest.empty
                else base_nws_t
            )
            ptrend = (
                clean_num(obs_latest.iloc[0].get("pressure_trend_3hr_hpa"), 0.0)
                if not obs_latest.empty
                else 0.0
            )

            def _ref_val(stid: str, col: str, default_val: float) -> float:
                row_s = ref_map.get(stid)
                if row_s is not None and pd.notna(row_s.get(col)):
                    return float(row_s[col])
                return default_val

            vent_raw_t = _ref_val("VTCC1", "temp_c", np.nan)
            chews_default_t = (vent_raw_t - 0.54) if not np.isnan(vent_raw_t) else (tass_t - 6.83)
            chews_t = _ref_val("CHWC1", "temp_c", chews_default_t)
            chews_rh = _ref_val("CHWC1", "rh_pct", base_nws_rh)
            chews_p = _ref_val("CHWC1", "precip_1hr_mm", 0.0)

            arroyo_raw_t = _ref_val("ASRC1", "temp_c", np.nan)
            hast_default_t = (arroyo_raw_t - 2.01) if not np.isnan(arroyo_raw_t) else (tass_t - 0.60)
            hast_t = _ref_val("CAHC1", "temp_c", hast_default_t)
            arroyo_t = _ref_val("ASRC1", "temp_c", hast_t + 2.01)
            arroyo_rh = _ref_val("ASRC1", "rh_pct", base_nws_rh)

            bigsur_t = _ref_val("PPSC1", "temp_c", tass_t + 2.45)
            bigsur_rh = _ref_val("PPSC1", "rh_pct", base_nws_rh)
            bigsur_p = _ref_val("PPSC1", "precip_1hr_mm", 0.0)

            vent_t = _ref_val("VTCC1", "temp_c", chews_t + 0.54)
            vent_p = _ref_val("VTCC1", "precip_1hr_mm", chews_p)

            latest_fc_df["pressure_trend_3hr_hpa"] = ptrend
            latest_fc_df["issue_chews_temp_c"] = chews_t
            latest_fc_df["issue_chews_rh_pct"] = chews_rh
            latest_fc_df["issue_chews_precip_mm"] = chews_p
            latest_fc_df["issue_hastings_temp_c"] = hast_t
            latest_fc_df["issue_arroyo_temp_c"] = arroyo_t
            latest_fc_df["issue_arroyo_rh_pct"] = arroyo_rh
            latest_fc_df["issue_bigsur_temp_c"] = bigsur_t
            latest_fc_df["issue_bigsur_rh_pct"] = bigsur_rh
            latest_fc_df["issue_bigsur_precip_mm"] = bigsur_p
            latest_fc_df["issue_ventana_temp_c"] = vent_t
            latest_fc_df["issue_ventana_precip_mm"] = vent_p
            obs_gap_hrs = 999.0
            if not obs_latest.empty:
                obs_gap_hrs = abs(
                    (
                        pd.to_datetime(latest_fc_df["valid_time_utc"].iloc[0], utc=True)
                        - pd.to_datetime(anchor_ts, utc=True)
                    ).total_seconds()
                ) / 3600.0
            base_om_t = clean_num(latest_fc_df["om_temp_c"].iloc[0], base_nws_t) if "om_temp_c" in latest_fc_df.columns else base_nws_t
            latest_fc_df["issue_tass_temp_c"] = tass_t
            latest_fc_df["issue_persistence_bias_c"] = (tass_t - base_nws_t) if obs_gap_hrs <= 2.0 else 0.0
            latest_fc_df["issue_om_persistence_bias_c"] = (tass_t - base_om_t) if obs_gap_hrs <= 2.0 else 0.0

            ewma_map = compute_diurnal_ewma_bias(conn, anchor_ts)
            fut_preds = _ensure_hydro_pred_cols(
                predict_dual_baseline_with_ewma(corrector, latest_fc_df, ewma_bias_by_block=ewma_map, om_weight=0.50),
                latest_fc_df,
            )
            fut_rows = [
                (
                    str(r["issue_time_utc"]),
                    str(r["valid_time_utc"]),
                    "MOS_GBM_v4_DUAL",
                    float(r["nws_temp_c"]),
                    float(r["pred_temp_error_c"]),
                    float(r["corrected_temp_c"]),
                    float(r["corrected_temp_p10_c"]),
                    float(r["corrected_temp_p90_c"]),
                    clean_num(latest_fc_df.loc[idx, "nws_rh_pct"], 50.0),
                    round(
                        max(
                            clean_num(latest_fc_df.loc[idx, "nws_wind_gust_ms"], 0.0) or 0.0,
                            (clean_num(latest_fc_df.loc[idx, "nws_wind_speed_ms"], 0.0) or 0.0) * 1.35,
                        ),
                        2,
                    ),
                    clean_num(r.get("raw_nws_pop_pct"), 0.0),
                    clean_num(r.get("mos_pop_pct"), 0.0),
                    clean_num(r.get("raw_nws_qpf_mm"), 0.0),
                    clean_num(r.get("mos_qpf_mm"), 0.0),
                    clean_num(r.get("mos_qpf_p90_mm"), 0.0),
                    clean_num(r.get("ewma_bias_applied_c"), 0.0),
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
                    raw_nws_qpf_mm, mos_qpf_mm, mos_qpf_p90_mm,
                    ewma_bias_applied_c
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fut_rows,
            )
        conn.commit()
    logger.info(
        f"Populated {len(hist_rows)} historical MOS verification rows in mos_corrected_forecasts."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill WeatherLink Pro history and train 5-station Thermal + Hydrological MOS model."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Number of past days to backfill (default: 730)",
    )
    parser.add_argument(
        "--skip-weatherlink",
        action="store_true",
        help="Preserve existing local_observations in SQLite and only backfill the 5-station Open-Meteo series before training.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Train model and backfill MOS verification using existing SQLite rows without calling external APIs.",
    )
    args = parser.parse_args()

    ensure_schema_and_view(DB_PATH)

    if not args.skip_fetch:
        if not args.skip_weatherlink:
            try:
                wl_client = WeatherLinkV2Client(
                    WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET
                )
                station_id = wl_client.get_station_id()
                obs_df = fetch_weatherlink_history(
                    wl_client, station_id, days_back=float(args.days)
                )
                if not obs_df.empty:
                    store_observations_to_sqlite(obs_df, DB_PATH, source="WL_HISTORIC")
                else:
                    logger.warning(
                        "WeatherLink Historic API returned 0 rows; proceeding with existing SQLite observations."
                    )
            except Exception as exc:
                logger.warning(
                    f"Live WeatherLink backfill encountered an error ({exc}); preserving existing SQLite observations..."
                )

        try:
            nwp_dict = fetch_historical_nwp_and_references(days_back=args.days)
            store_backfill_to_sqlite(None, nwp_dict, DB_PATH)
        except Exception as exc:
            logger.warning(
                f"Historical NWP/reference backfill encountered an error ({exc}); proceeding with existing SQLite archive..."
            )

    with sqlite3.connect(DB_PATH) as conn:
        train_df = pd.read_sql_query(
            "SELECT * FROM vw_mos_training_dataset ORDER BY valid_time_utc ASC",
            conn,
        )

    logger.info(
        f"Loaded {len(train_df)} aligned training pairs from vw_mos_training_dataset."
    )
    if len(train_df) < 48:
        raise RuntimeError(
            f"Insufficient aligned rows ({len(train_df)}) to train production model."
        )

    corrector = TassajaraMOSCorrector()
    metrics = corrector.fit_and_evaluate(train_df)
    corrector.save(str(MODEL_PATH))

    populate_mos_verification_and_active_forecasts(
        corrector, train_df, DB_PATH, oof_preds_df=getattr(corrector, "oof_predictions_df", None)
    )

    logger.info("==========================================================")
    logger.info("TASSAJARA MOS PRODUCTION MODEL TRAINED SUCCESSFULLY!")
    logger.info(f"Training Pairs Used          : {len(train_df)}")
    logger.info(f"Observed Wet Hours (>=0.01in): {metrics.get('wet_hours_count', 0)}")
    logger.info(f"Raw Generic Forecast RMSE    : {metrics['raw_nws_rmse_c']} °C")
    logger.info(f"Tassajara MOS Corrected RMSE : {metrics['mos_corrected_rmse_c']} °C")
    logger.info(
        f"Thermal Skill Improvement    : +{metrics['rmse_skill_improvement_pct']}%"
    )
    logger.info(f"Saved Production Model       : {MODEL_PATH}")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()