#!/usr/bin/env python3
"""
run_hourly_mos_pipeline.py (v4 — Thermal + Two-Stage Hydrological Hurdle Edition)
Master hourly runner script for the Tassajara Zen Mountain Center (TZMC)
Microclimate Forecasting Pipeline. Initializes SQLite schema/views, logs
Stone Office Vantage Pro2 and RAWS reference observations (with API fallback),
ingests operational NWS hourly grids, generates MOS quantile thermal and
hydrological (PoP/QPF) forecasts, and evaluates facility alert thresholds.
"""

import sqlite3
import logging
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
from mos_model import (
    compute_solar_and_canyon_shadow,
    engineer_mos_features,
    analytical_hydro_hurdle_fallback,
)
from alert_engine import FacilityAlertManager
from backfill_and_train import ensure_schema_and_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TassajaraPipeline")

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")


def init_db_if_needed():
    """Ensures all SQLite tables, hydrological columns, and vw_mos_training_dataset exist."""
    ensure_schema_and_view(DB_PATH)


def log_observations():
    """
    Fetches Vantage Pro2 and reference station data with defensive fallback
    to the latest SQLite archive if external APIs are unreachable or unconfigured.
    """
    tass = {}
    refs = {}
    diag = {"qc_flag": "OK", "qc_notes": []}

    # 1. Try Live WeatherLink Ingest
    try:
        client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
        tass = client.fetch_current_observations()
    except Exception as exc:
        logger.warning(
            f"Live WeatherLink API fetch failed ({exc}); falling back to latest SQLite observation..."
        )

    # 2. Try Live Synoptic Reference Ingest
    try:
        refs = fetch_regional_reference_stations(["CHWC1", "CAHC1"])
    except Exception as exc:
        logger.warning(f"Live reference RAWS fetch failed ({exc}); using SQLite archive...")

    with sqlite3.connect(DB_PATH) as conn:
        if not tass or tass.get("temp_c") is None:
            row_t = conn.execute(
                """
                SELECT valid_time_utc, temp_c, dewpoint_c, wetbulb_c, rh_pct,
                       wind_speed_ms, wind_gust_ms, wind_dir_deg, mslp_hpa,
                       pressure_trend_3hr_hpa, precip_1hr_mm, solar_rad_wm2,
                       uv_index, qc_flag
                FROM local_observations
                ORDER BY valid_time_utc DESC LIMIT 1
                """
            ).fetchone()
            if row_t:
                tass = {
                    "valid_time_utc": row_t[0],
                    "temp_c": row_t[1],
                    "dewpoint_c": row_t[2],
                    "wetbulb_c": row_t[3],
                    "rh_pct": row_t[4],
                    "wind_speed_ms": row_t[5],
                    "wind_gust_ms": row_t[6],
                    "wind_dir_deg": row_t[7],
                    "mslp_hpa": row_t[8],
                    "pressure_trend_3hr_hpa": row_t[9],
                    "precip_1hr_mm": row_t[10],
                    "solar_rad_wm2": row_t[11],
                    "uv_index": row_t[12],
                }
                diag["qc_flag"] = row_t[13] or "OK"

        if not tass or tass.get("temp_c") is None:
            tass = {
                "valid_time_utc": pd.Timestamp.now(tz="UTC")
                .replace(minute=0, second=0, microsecond=0)
                .isoformat(),
                "temp_c": 15.0,
                "dewpoint_c": 5.0,
                "rh_pct": 50.0,
                "wind_speed_ms": 1.0,
                "wind_gust_ms": 1.5,
                "wind_dir_deg": 295.0,
                "mslp_hpa": 1013.25,
                "pressure_trend_3hr_hpa": 0.0,
                "precip_1hr_mm": 0.0,
            }

        for stid in ["CHWC1", "CAHC1"]:
            if stid not in refs or not refs[stid].get("temp_c"):
                r_row = conn.execute(
                    """
                    SELECT temp_c, rh_pct, wind_speed_ms, wind_gust_ms,
                           wind_dir_deg, precip_1hr_mm
                    FROM reference_observations
                    WHERE station_id = ? AND temp_c IS NOT NULL
                    ORDER BY valid_time_utc DESC LIMIT 1
                    """,
                    (stid,),
                ).fetchone()
                if r_row:
                    refs[stid] = {
                        "station_id": stid,
                        "temp_c": r_row[0],
                        "rh_pct": r_row[1],
                        "wind_speed_ms": r_row[2],
                        "wind_gust_ms": r_row[3],
                        "wind_dir_deg": r_row[4],
                        "precip_1hr_mm": r_row[5],
                    }

        diag = compute_microclimate_diagnostics(tass, refs)

        dt_hour = pd.to_datetime(tass["valid_time_utc"], utc=True).floor("h").isoformat()
        wind_speed_ms = tass.get("wind_speed_ms") if tass.get("wind_speed_ms") is not None else 0.0
        wind_gust_ms = (
            tass.get("wind_gust_ms") if tass.get("wind_gust_ms") is not None else wind_speed_ms
        )
        wind_dir_deg = tass.get("wind_dir_deg") if tass.get("wind_dir_deg") is not None else 0.0
        precip_1hr_mm = tass.get("precip_1hr_mm") if tass.get("precip_1hr_mm") is not None else 0.0

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
            if rdata and rdata.get("temp_c") is not None:
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
                        rdata.get("wind_speed_ms", 0.0),
                        rdata.get("wind_gust_ms", 0.0),
                        rdata.get("wind_dir_deg", 0.0),
                        rdata.get("precip_1hr_mm", 0.0),
                    ),
                )
        conn.commit()

    return tass, refs, diag


def generate_mos_forecast(tass, refs):
    """Applies the trained MOS ensemble (thermal + hydrological hurdle) to the latest full NWS forecast."""
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

        chews_temp_c = refs.get("CHWC1", {}).get("temp_c")
        chews_precip_mm = refs.get("CHWC1", {}).get("precip_1hr_mm")
        if chews_temp_c is None:
            row_ch = conn.execute(
                """
                SELECT temp_c, precip_1hr_mm FROM reference_observations
                WHERE station_id = 'CHWC1' AND temp_c IS NOT NULL
                ORDER BY valid_time_utc DESC LIMIT 1
                """
            ).fetchone()
            if row_ch:
                chews_temp_c = float(row_ch[0])
                chews_precip_mm = float(row_ch[1] or 0.0)

    fc_df["pressure_trend_3hr_hpa"] = (
        tass.get("pressure_trend_hpa") if tass.get("pressure_trend_hpa") is not None else 0.0
    )
    fc_df["issue_chews_temp_c"] = (
        chews_temp_c if chews_temp_c is not None else (fc_df["nws_temp_c"] + 2.0)
    )
    fc_df["issue_chews_precip_mm"] = (
        chews_precip_mm if chews_precip_mm is not None else 0.0
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
        model_ver = "MOS_GBM_v2_HYDRO"
    else:
        solar = compute_solar_and_canyon_shadow(fc_df["valid_time_utc"])
        bias = (
            -3.8 * (solar["solar_el_deg"] <= 0).astype(float)
            - 2.5 * solar["in_canyon_shadow"]
            + 1.5 * (solar["effective_insolation"] > 0.65).astype(float)
        )
        X_feats = engineer_mos_features(fc_df)
        hydro_fb = analytical_hydro_hurdle_fallback(X_feats)

        preds = fc_df[["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
        preds["pred_temp_error_c"] = bias.round(2)
        preds["corrected_temp_c"] = (fc_df["nws_temp_c"] + bias).round(2)
        preds["corrected_temp_p10_c"] = (preds["corrected_temp_c"] - 2.2).round(2)
        preds["corrected_temp_p90_c"] = (preds["corrected_temp_c"] + 2.2).round(2)
        preds["raw_nws_pop_pct"] = X_feats["nws_pop_pct"].round(1).values
        preds["mos_pop_pct"] = hydro_fb["mos_pop_pct"].values
        preds["raw_nws_qpf_mm"] = X_feats["nws_qpf_1hr_mm"].round(2).values
        preds["mos_qpf_mm"] = hydro_fb["mos_qpf_mm"].values
        preds["mos_qpf_p90_mm"] = hydro_fb["mos_qpf_p90_mm"].values
        model_ver = "TOPO_PHYSICS_COLDSTART"

    # Ensure hydrological columns are present even if an older .joblib artifact was loaded
    if "mos_qpf_mm" not in preds.columns:
        X_feats = engineer_mos_features(fc_df)
        hydro_fb = analytical_hydro_hurdle_fallback(X_feats)
        preds["raw_nws_pop_pct"] = X_feats["nws_pop_pct"].round(1).values
        preds["mos_pop_pct"] = hydro_fb["mos_pop_pct"].values
        preds["raw_nws_qpf_mm"] = X_feats["nws_qpf_1hr_mm"].round(2).values
        preds["mos_qpf_mm"] = hydro_fb["mos_qpf_mm"].values
        preds["mos_qpf_p90_mm"] = hydro_fb["mos_qpf_p90_mm"].values

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
            float(r["raw_nws_pop_pct"]),
            float(r["mos_pop_pct"]),
            float(r["raw_nws_qpf_mm"]),
            float(r["mos_qpf_mm"]),
            float(r["mos_qpf_p90_mm"]),
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
                corrected_rh_pct, corrected_wind_gust_ms,
                raw_nws_pop_pct, mos_pop_pct,
                raw_nws_qpf_mm, mos_qpf_mm, mos_qpf_p90_mm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    print("1. Checking SQLite schema & views...")
    init_db_if_needed()

    print("2. Ingesting Stone Office Vantage Pro2 & Reference observations...")
    tass_obs, ref_obs, diag = log_observations()

    print("3. Ingesting standard NWS hourly forecast...")
    try:
        fetch_and_store_nws_hourly(DB_PATH)
    except Exception as exc:
        logger.warning(
            f"NWS forecast ingest API unreachable ({exc}); proceeding with existing SQLite forecast archive..."
        )

    print("4. Computing Tassajara canyon-corrected forecast...")
    mos_df = generate_mos_forecast(tass_obs, ref_obs)

    print("5. Evaluating facility alert thresholds...")
    AlertMgr = FacilityAlertManager(DB_PATH)
    AlertMgr.evaluate_forecasts_and_hardware(mos_df, diag, ref_obs.get("CHWC1", {}))

    print("Pipeline run complete! Database updated: tassajara_weather.db")