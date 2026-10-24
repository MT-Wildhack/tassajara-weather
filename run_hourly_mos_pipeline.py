#!/usr/bin/env python3
"""
run_hourly_mos_pipeline.py
Master runner script for non-programmers. Automatically initializes the SQLite
database file if missing, logs Stone Office observations, fetches NWS forecasts,
calculates MOS canyon corrections, and checks facility alerts.
"""
import os
import sqlite3
from pathlib import Path
import pandas as pd
import joblib

from ingest_weather import (
    WeatherLinkV2Client, WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET,
    fetch_regional_reference_stations, compute_microclimate_diagnostics
)
from ingest_nws import fetch_and_store_nws_hourly
from mos_model import TassajaraMOSCorrector, compute_solar_and_canyon_shadow
from alert_engine import FacilityAlertManager

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")

def init_db_if_needed():
    """Creates the SQLite database file from schema.sql on first run."""
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        with open("schema.sql", "r", encoding="utf-8") as f:
            conn.executescript(f.read())

def log_observations():
    """Fetches Vantage Pro2 and reference station data and saves to SQLite."""
    client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
    tass = client.fetch_current_observations()
    refs = fetch_regional_reference_stations(["CHWC1", "CAHC1"])
    diag = compute_microclimate_diagnostics(tass, refs)

    # Snap timestamp to top-of-the-hour so it matches NWS hourly forecast timestamps
    dt_hour = pd.to_datetime(tass["valid_time_utc"]).floor("h").isoformat()

    with sqlite3.connect(DB_PATH) as conn:
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
                dt_hour, tass["temp_c"], tass["dewpoint_c"], tass["wetbulb_c"],
                tass["rh_pct"], tass["wind_speed_ms"], tass["wind_gust_ms"],
                tass["wind_dir_deg"], tass["mslp_hpa"], tass["pressure_trend_hpa"],
                tass["precip_1hr_mm"], tass["solar_rad_wm2"], tass["uv_index"],
                diag["qc_flag"]
            )
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
                    dt_hour, stid, rdata.get("temp_c"), rdata.get("rh_pct"),
                    rdata.get("wind_speed_ms"), rdata.get("wind_gust_ms"),
                    rdata.get("wind_dir_deg"), rdata.get("precip_1hr_mm", 0.0)
                )
            )
        conn.commit()
    return tass, refs, diag

def generate_mos_forecast(tass, refs):
    """Applies the MOS model (or physics baseline during initial data collection)."""
    with sqlite3.connect(DB_PATH) as conn:
        fc_df = pd.read_sql_query(
            """
            SELECT * FROM nws_forecasts
            WHERE issue_time_utc = (SELECT MAX(issue_time_utc) FROM nws_forecasts)
            ORDER BY valid_time_utc ASC
            """,
            conn
        )
    if fc_df.empty:
        return pd.DataFrame()

    fc_df["pressure_trend_3hr_hpa"] = tass.get("pressure_trend_hpa") or 0.0
    fc_df["issue_chews_temp_c"] = refs.get("CHWC1", {}).get("temp_c") or (fc_df["nws_temp_c"] + 2.0)
    fc_df["issue_tass_temp_c"] = tass.get("temp_c") or fc_df["nws_temp_c"]
    fc_df["issue_persistence_bias_c"] = fc_df["issue_tass_temp_c"] - fc_df["nws_temp_c"].iloc[0]

    if MODEL_PATH.exists():
        mos = joblib.load(MODEL_PATH)
        preds = mos.predict(fc_df)
        model_ver = "MOS_GBM_v1"
    else:
        # Cold-start topographic physics approximation until 14+ days of logs accumulate
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
            r["issue_time_utc"], r["valid_time_utc"], model_ver,
            r["nws_temp_c"], r["pred_temp_error_c"], r["corrected_temp_c"],
            r["corrected_temp_p10_c"], r["corrected_temp_p90_c"],
            fc_df.loc[idx, "nws_rh_pct"], fc_df.loc[idx, "nws_wind_speed_ms"] * 1.35
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
            rows
        )
        conn.commit()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM mos_corrected_forecasts WHERE issue_time_utc = ? ORDER BY valid_time_utc ASC",
            conn, params=(fc_df["issue_time_utc"].iloc[0],)
        )

if __name__ == "__main__":
    print("1. Checking SQLite database...")
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