#!/usr/bin/env python3
"""
run_hourly_mos_pipeline.py
Master hourly runner script for the Tassajara Zen Mountain Center (TZMC)
Microclimate Forecasting Pipeline. Initializes SQLite schema/views, logs
Stone Office Davis Vantage Pro2 and RAWS reference observations, ingests
operational NWS hourly grids, generates MOS quantile forecasts, and evaluates
facility alert thresholds.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import joblib

from ingest_weather import (
    WeatherLinkV2Client,
    WEATHERLINK_API_KEY,
    WEATHERLINK_API_SECRET,
    fetch_regional_reference_stations,
    compute_microclimate_diagnostics,
)
from ingest_nws import fetch_and_store_nws_hourly
from mos_model import compute_solar_and_canyon_shadow
from alert_engine import FacilityAlertManager
from backfill_and_train import ensure_schema_and_view

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")


def init_db_if_needed():
    """Ensures all SQLite tables and the leak-free vw_mos_training_dataset view exist."""
    ensure_schema_and_view(DB_PATH)


def log_observations():
    """Fetches Vantage Pro2 and reference station data and saves to SQLite."""
    client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
    tass = client.fetch_current_observations()
    refs = fetch_regional_reference_stations(["CHWC1", "CAHC1"])
    diag = compute_microclimate_diagnostics(tass, refs)

    # Snap timestamp to top-of-the-hour so it matches NWS hourly forecast timestamps
    dt_hour = pd.to_datetime(tass["valid_time_utc"], utc=True).floor("h").isoformat()

    # Coalesce calm wind readings (0 or None -> 0.0) so nocturnal calms are never stored as NULL
    wind_speed_ms = tass.get("wind_speed_ms") if tass.get("wind_speed_ms") is not None else 0.0
    wind_gust_ms = tass.get("wind_gust_ms") if tass.get("wind_gust_ms") is not None else wind_speed_ms
    wind_dir_deg = tass.get("wind_dir_deg") if tass.get("wind_dir_deg") is not None else 0.0
    precip_1hr_mm = tass.get("precip_1hr_mm") if tass.get("precip_1hr_mm") is not None else 0.0

    with sqlite3.connect(DB_PATH) as conn:
        # Fallback for mslp_hpa and 3-hour barometric trend from recent SQLite observations if omitted
        mslp_hpa = tass.get("mslp_hpa")
        if mslp_hpa is None:
            row_p = conn.execute(
                "SELECT mslp_hpa FROM local_observations WHERE mslp_hpa IS NOT NULL ORDER BY valid_time_utc DESC LIMIT 1"
            ).fetchone()
            mslp_hpa = float(row_p[0]) if row_p else 1013.25

        ptrend_3hr = tass.get("pressure_trend_hpa")
        if ptrend_3hr is None:
            row_p3 = conn.execute(
                "SELECT mslp_hpa FROM local_observations WHERE mslp_hpa IS NOT NULL ORDER BY valid_time_utc DESC LIMIT 1 OFFSET 2"
            ).fetchone()
            ptrend_3hr = round(float(mslp_hpa) - float(row_p3[0]), 2) if row_p3 else 0.0

        tass["mslp_hpa"] = mslp_hpa
        tass["pressure_trend_hpa"] = ptrend_3hr
        tass["wind_speed_ms"] = wind_speed_ms
        tass["wind_gust_ms"] = wind_gust_ms
        tass["wind_dir_deg"] = wind_dir_deg

        conn.execute(
            """
            INSERT OR REPLACE INTO local_observations (
                valid_time_utc, station_id, temp_c, dewpoint_c, wetbulb_c,
                rh_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
                mslp_hpa, pressure_trend_3hr_hpa, precip_1hr_mm,
                solar_rad_wm2, uv_index, qc_flag
            ) VALUES (?, 'TASS_VP2', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dt_hour,
                tass.get("temp_c"),
                tass.get("dewpoint_c"),
                tass.get("wetbulb_c"),
                tass.get("rh_pct"),
                wind_speed_ms,
                wind_gust_ms,
                wind_dir_deg,
                mslp_hpa,
                ptrend_3hr,
                precip_1hr_mm,
                tass.get("solar_rad_wm2"),
                tass.get("uv_index"),
                diag["qc_flag"],
            ),
        )
        for stid, rdata in refs.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO reference_observations (
                    valid_time_utc, station_id, temp_c, rh_pct,
                    wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1hr_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dt_hour,
                    stid,
                    rdata.get("temp_c"),
                    rdata.get("rh_pct"),
                    rdata.get("wind_speed_ms") if rdata.get("wind_speed_ms") is not None else 0.0,
                    rdata.get("wind_gust_ms") if rdata.get("wind_gust_ms") is not None else 0.0,
                    rdata.get("wind_dir_deg") if rdata.get("wind_dir_deg") is not None else 0.0,
                    rdata.get("precip_1hr_mm") if rdata.get("precip_1hr_mm") is not None else 0.0,
                ),
            )
        conn.commit()
    return tass, refs, diag


def generate_mos_forecast(tass, refs):
    """Applies the trained MOS ensemble (or topographic physics fallback) to the latest full NWS forecast."""
    with sqlite3.connect(DB_PATH) as conn:
        fc_df = pd.read_sql_query(
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
        if fc_df.empty:
            return pd.DataFrame()

        # Fallback to latest stored Chews Ridge temperature if live Synoptic fetch was empty
        chews_temp_c = refs.get("CHWC1", {}).get("temp_c")
        if chews_temp_c is None:
            row_ch = conn.execute(
                "SELECT temp_c FROM reference_observations WHERE station_id = 'CHWC1' AND temp_c IS NOT NULL ORDER BY valid_time_utc DESC LIMIT 1"
            ).fetchone()
            chews_temp_c = float(row_ch[0]) if row_ch else None

    fc_df["pressure_trend_3hr_hpa"] = (
        tass.get("pressure_trend_hpa") if tass.get("pressure_trend_hpa") is not None else 0.0
    )
    fc_df["issue_chews_temp_c"] = (
        chews_temp_c if chews_temp_c is not None else (fc_df["nws_temp_c"] + 2.0)
    )
    fc_df["issue_tass_temp_c"] = (
        tass.get("temp_c") if tass.get("temp_c") is not None else fc_df["nws_temp_c"]
    )
    fc_df["issue_persistence_bias_c"] = (
        fc_df["issue_tass_temp_c"] - float(fc_df["nws_temp_c"].iloc[0])
    )

    if MODEL_PATH.exists():
        mos = joblib.load(MODEL_PATH)
        preds = mos.predict(fc_df)
        model_ver = "MOS_GBM_v1"
    else:
        solar = compute_solar_and_canyon_shadow(fc_df["valid_time_utc"])
        bias = (
            -3.8 * (solar["solar_el_deg"] <= 0).astype(float)
            - 2.5 * solar["in_canyon_shadow"]
            + 1.5 * (solar["effective_insolation"] > 0.65).astype(float)
        )
        preds = fc_df[["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
        preds["pred_temp_error_c"] = bias.round(2)
        preds["corrected_temp_c"] = (fc_df["nws_temp_c"] + bias).round(2)
        preds["corrected_temp_p10_c"] = (preds["corrected_temp_c"] - 2.2).round(2)
        preds["corrected_temp_p90_c"] = (preds["corrected_temp_c"] + 2.2).round(2)
        model_ver = "TOPO_PHYSICS_COLDSTART"

    rows = [
        (
            str(r["issue_time_utc"]),
            str(r["valid_time_utc"]),
            model_ver,
            float(r["nws_temp_c"]),
            float(r["pred_temp_error_c"]),
            float(r["corrected_temp_c"]),
            float(r["corrected_temp_p10_c"]),
            float(r["corrected_temp_p90_c"]),
            float(fc_df.loc[idx, "nws_rh_pct"]) if pd.notna(fc_df.loc[idx, "nws_rh_pct"]) else 50.0,
            round(float(fc_df.loc[idx, "nws_wind_speed_ms"] or 0.0) * 1.35, 2),
        )
        for idx, r in preds.iterrows()
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO mos_corrected_forecasts (
                issue_time_utc, valid_time_utc, model_version,
                raw_nws_temp_c, pred_temp_error_c, corrected_temp_c,
                corrected_temp_p10_c, corrected_temp_p90_c,
                corrected_rh_pct, corrected_wind_gust_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

        return pd.read_sql_query(
            "SELECT * FROM mos_corrected_forecasts WHERE issue_time_utc = ? ORDER BY valid_time_utc ASC",
            conn,
            params=(str(fc_df["issue_time_utc"].iloc[0]),),
        )


if __name__ == "__main__":
    print("1. Checking SQLite database schema & views...")
    init_db_if_needed()
    print("2. Fetching Stone Office Vantage Pro2 & Reference Station data...")
    tass_obs, ref_obs, diag = log_observations()
    print("3. Fetching standard NWS hourly forecast...")
    fetch_and_store_nws_hourly(DB_PATH)
    print("4. Computing Tassajara canyon-corrected forecast...")
    mos_df = generate_mos_forecast(tass_obs, ref_obs)
    print("5. Evaluating freeze, wind, and fire-weather alert thresholds...")
    AlertMgr = FacilityAlertManager(DB_PATH)
    AlertMgr.evaluate_forecasts_and_hardware(mos_df, diag, ref_obs.get("CHWC1", {}))
    print("Pipeline run complete! Database updated: tassajara_weather.db")
