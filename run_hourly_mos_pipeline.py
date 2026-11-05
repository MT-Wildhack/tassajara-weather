#!/usr/bin/env python3
"""
run_hourly_mos_pipeline.py
Operational hourly orchestrator for the Tassajara Zen Mountain Center (TZMC) MOS Pipeline:
  1. Backfills the past 48 hours of Stone Office Vantage Pro2 archive records via /v2/historic,
     then overlays live /v2/current telemetry and reconciles 24-hour rainfall totals using
     ISO-8601-safe SQLite timestamp comparisons.
  2. Ingests the 5-station Santa Lucia boundary array (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) via Synoptic API.
  3. Pulls operational 7-day hourly NWS / NWP forecasts (via ingest_nws.py).
  4. Runs 25-feature Thermal + Hydrological Hurdle MOS inference using models/tassajara_mos_v1.joblib.
  5. Computes Antecedent Precipitation Index (API_t), effective runoff coefficient (C_t),
     6h/24h/72h routed creek discharge projections, and 4-Tier Flood & Tassajara Road Washout
     warnings via hydro_risk_engine.py.
  6. Evaluates operational canyon alerts (via alert_engine.py if present).
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from ingest_weather import (
    WEATHERLINK_API_KEY,
    WEATHERLINK_API_SECRET,
    REFERENCE_STATIONS,
    WeatherLinkV2Client,
    fetch_regional_reference_stations,
    compute_microclimate_diagnostics,
)
from mos_model import (
    TassajaraMOSCorrector,
    engineer_mos_features,
    analytical_hydro_hurdle_fallback,
)
from hydro_risk_engine import (
    ensure_hydro_schema,
    evaluate_and_store_hydro_risk,
)
from backfill_and_train import (
    ensure_schema_and_view,
    fetch_weatherlink_history,
    store_observations_to_sqlite,
    clean_num,
)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/hourly_mos_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("TassajaraHourlyPipeline")

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")


def snap_to_hour_iso(ts_str: Optional[str]) -> str:
    """Normalizes any UTC timestamp string to top-of-the-hour ISO-8601 format (+00:00)."""
    if not ts_str:
        dt = datetime.now(timezone.utc)
    else:
        dt = pd.to_datetime(ts_str, utc=True).to_pydatetime()
    return dt.replace(minute=0, second=0, microsecond=0).isoformat()


def ingest_live_observations_and_references(db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    1. Backfills the past 48 hours of Stone Office /v2/historic records non-destructively.
    2. Overlays the live /v2/current observation and reconciles 24-hour rainfall totals.
    3. Pulls all 5 Santa Lucia reference stations from Synoptic Data API.
    """
    wl_client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
    station_id = wl_client.get_station_id()

    try:
        recent_hist_df = fetch_weatherlink_history(wl_client, station_id, days_back=2.0)
        if not recent_hist_df.empty:
            store_observations_to_sqlite(recent_hist_df, db_path)
    except Exception as exc:
        logger.warning(f"48-hour historic catch-up skipped ({exc}); proceeding with live snapshot.")

    tass_obs = wl_client.fetch_current_observations(station_id)
    ref_obs = fetch_regional_reference_stations(list(REFERENCE_STATIONS.keys()))
    diag = compute_microclimate_diagnostics(tass_obs, ref_obs)

    obs_hour_iso = snap_to_hour_iso(tass_obs.get("valid_time_utc"))

    with sqlite3.connect(db_path) as conn:
        ptrend_3h = tass_obs.get("pressure_trend_hpa")
        if ptrend_3h is None and tass_obs.get("mslp_hpa") is not None:
            prev_row = conn.execute(
                """
                SELECT mslp_hpa FROM local_observations
                WHERE mslp_hpa IS NOT NULL
                  AND valid_time_utc <= strftime('%Y-%m-%dT%H:%M:%S+00:00', ?, '-2 hours')
                ORDER BY valid_time_utc DESC LIMIT 1
                """,
                (obs_hour_iso,),
            ).fetchone()
            if prev_row and prev_row[0] is not None:
                ptrend_3h = round(float(tass_obs["mslp_hpa"]) - float(prev_row[0]), 2)
        ptrend_3h = clean_num(ptrend_3h, 0.0)

        existing_hr = conn.execute(
            """
            SELECT temp_c, dewpoint_c, wetbulb_c, rh_pct, mslp_hpa,
                   precip_1hr_mm, wind_speed_ms, wind_gust_ms, wind_dir_deg
            FROM local_observations
            WHERE valid_time_utc = ?
            """,
            (obs_hour_iso,),
        ).fetchone()

        hist_temp = existing_hr[0] if existing_hr else None
        hist_dew = existing_hr[1] if existing_hr else None
        hist_wet = existing_hr[2] if existing_hr else None
        hist_rh = existing_hr[3] if existing_hr else None
        hist_mslp = existing_hr[4] if existing_hr else None
        hist_hr_rain = float(existing_hr[5] or 0.0) if existing_hr else 0.0
        hist_hr_wspd = float(existing_hr[6] or 0.0) if existing_hr else 0.0
        hist_hr_wgst = float(existing_hr[7] or 0.0) if existing_hr else 0.0
        hist_hr_wdir = float(existing_hr[8] or 0.0) if existing_hr else 0.0

        cur_temp = clean_num(tass_obs.get("temp_c"), hist_temp)
        cur_dew = clean_num(tass_obs.get("dewpoint_c"), hist_dew)
        cur_wet = clean_num(tass_obs.get("wetbulb_c"), hist_wet)
        cur_rh = clean_num(tass_obs.get("rh_pct"), hist_rh)
        cur_mslp = clean_num(tass_obs.get("mslp_hpa"), hist_mslp or 1013.25)

        cur_hr_rain = max(hist_hr_rain, clean_num(tass_obs.get("precip_1hr_mm"), 0.0) or 0.0)
        cur_wspd = max(hist_hr_wspd, clean_num(tass_obs.get("wind_speed_ms"), 0.0) or 0.0)
        cur_wgst = max(hist_hr_wgst, cur_wspd, clean_num(tass_obs.get("wind_gust_ms"), 0.0) or 0.0)
        raw_wdir = clean_num(tass_obs.get("wind_dir_deg"), 0.0) or 0.0
        cur_wdir = raw_wdir if raw_wdir > 0.0 else hist_hr_wdir

        if cur_temp is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO local_observations (
                    valid_time_utc, station_id, temp_c, dewpoint_c, wetbulb_c,
                    rh_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
                    mslp_hpa, pressure_trend_3hr_hpa, precip_1hr_mm,
                    solar_rad_wm2, uv_index, qc_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obs_hour_iso,
                    "TASS_VP2",
                    cur_temp,
                    cur_dew,
                    cur_wet,
                    cur_rh,
                    cur_wspd,
                    cur_wgst,
                    cur_wdir,
                    cur_mslp,
                    ptrend_3h,
                    cur_hr_rain,
                    clean_num(tass_obs.get("solar_rad_wm2")),
                    clean_num(tass_obs.get("uv_index")),
                    diag.get("qc_flag", "OK"),
                ),
            )

        # Reconcile 24-hour / storm rain if /v2/historic did not report sub-hourly clicks
        reported_24h_mm = max(
            clean_num(tass_obs.get("precip_24hr_mm"), 0.0) or 0.0,
            clean_num(tass_obs.get("precip_storm_mm"), 0.0) or 0.0,
        )
        if reported_24h_mm > 0.0:
            window_start_iso = conn.execute(
                "SELECT strftime('%Y-%m-%dT%H:%M:%S+00:00', ?, '-23 hours')",
                (obs_hour_iso,),
            ).fetchone()[0]

            db_24h_sum = float(
                conn.execute(
                    """
                    SELECT COALESCE(SUM(precip_1hr_mm), 0.0)
                    FROM local_observations
                    WHERE valid_time_utc >= ? AND valid_time_utc <= ?
                    """,
                    (window_start_iso, obs_hour_iso),
                ).fetchone()[0]
                or 0.0
            )

            shortfall_mm = round(reported_24h_mm - db_24h_sum, 3)
            if shortfall_mm >= 0.20:
                target_row = conn.execute(
                    """
                    SELECT o.valid_time_utc
                    FROM local_observations o
                    LEFT JOIN (
                        SELECT valid_time_utc, MAX(COALESCE(mos_qpf_mm, 0.0)) AS mqpf
                        FROM mos_corrected_forecasts
                        GROUP BY valid_time_utc
                    ) m ON o.valid_time_utc = m.valid_time_utc
                    WHERE o.valid_time_utc >= ? AND o.valid_time_utc <= ?
                    ORDER BY COALESCE(m.mqpf, 0.0) DESC, COALESCE(o.rh_pct, 0.0) DESC, o.valid_time_utc DESC
                    LIMIT 1
                    """,
                    (window_start_iso, obs_hour_iso),
                ).fetchone()
                target_hr = target_row[0] if target_row else obs_hour_iso
                conn.execute(
                    """
                    UPDATE local_observations
                    SET precip_1hr_mm = ROUND(COALESCE(precip_1hr_mm, 0.0) + ?, 3)
                    WHERE valid_time_utc = ?
                    """,
                    (shortfall_mm, target_hr),
                )
                logger.info(
                    f"Reconciled {shortfall_mm:.2f} mm ({shortfall_mm/25.4:.2f} in) of 24h console rain "
                    f"into local_observations at {target_hr}."
                )

        ref_rows = []
        for stid, rdata in ref_obs.items():
            if not rdata.get("online"):
                continue
            ref_rows.append(
                (
                    obs_hour_iso,
                    stid,
                    clean_num(rdata.get("temp_c")),
                    clean_num(rdata.get("rh_pct")),
                    clean_num(rdata.get("wind_speed_ms"), 0.0),
                    clean_num(rdata.get("wind_gust_ms"), 0.0),
                    clean_num(rdata.get("wind_dir_deg"), 0.0),
                    clean_num(rdata.get("precip_1hr_mm"), 0.0),
                )
            )

        if ref_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO reference_observations (
                    valid_time_utc, station_id, temp_c, rh_pct,
                    wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1hr_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ref_rows,
            )
        conn.commit()

    logger.info(
        f"Synced Stone Office ({obs_hour_iso}: {cur_temp} C, "
        f"wind={cur_wspd * 2.23694:.1f} mph, gust={cur_wgst * 2.23694:.1f} mph, "
        f"24h_rain={reported_24h_mm / 25.4:.2f} in) & {len(ref_rows)}/5 Santa Lucia refs."
    )
    return {"tass_obs": tass_obs, "ref_obs": ref_obs, "diag": diag, "ptrend_3h": ptrend_3h}


def run_nws_forecast_ingest(db_path: str = DB_PATH) -> None:
    """Invokes ingest_nws.py if present to refresh the operational 7-day NWS forecast."""
    try:
        import ingest_nws

        for fn_name in (
            "ingest_nws_forecasts",
            "fetch_and_store_nws_forecasts",
            "run_nws_ingest",
            "main",
        ):
            fn = getattr(ingest_nws, fn_name, None)
            if callable(fn):
                try:
                    fn(db_path)
                except TypeError:
                    fn()
                logger.info(f"Executed ingest_nws.{fn_name}() successfully.")
                return
    except Exception as exc:
        logger.warning(f"NWS forecast ingest hook skipped or encountered error: {exc}")


def generate_operational_mos_forecasts(db_path: str = DB_PATH) -> int:
    """Runs 5-station MOS inference on the latest NWS forecast batch and commits to SQLite."""
    with sqlite3.connect(db_path) as conn:
        latest_fc_df = pd.read_sql_query(
            """
            SELECT * FROM nws_forecasts
            WHERE issue_time_utc = (
                SELECT issue_time_utc
                FROM nws_forecasts
                WHERE short_forecast != 'Archived NWP Grid' OR short_forecast IS NULL
                GROUP BY issue_time_utc
                HAVING COUNT(*) >= 24
                ORDER BY issue_time_utc DESC
                LIMIT 1
            )
            ORDER BY valid_time_utc ASC
            """,
            conn,
        )
        if latest_fc_df.empty:
            logger.warning("No operational NWS forecast batch found in nws_forecasts.")
            return 0

        obs_latest = pd.read_sql_query(
            """
            SELECT * FROM local_observations
            WHERE temp_c IS NOT NULL
            ORDER BY valid_time_utc DESC LIMIT 1
            """,
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

        base_nws_t = float(latest_fc_df["nws_temp_c"].iloc[0])
        base_nws_rh = clean_num(latest_fc_df["nws_rh_pct"].iloc[0], 50.0) or 50.0
        tass_t = float(obs_latest.iloc[0]["temp_c"]) if not obs_latest.empty else base_nws_t
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

        bigsur_t = _ref_val("PPSC1", "temp_c", arroyo_t + 1.04)
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
        latest_fc_df["issue_tass_temp_c"] = tass_t
        latest_fc_df["issue_persistence_bias_c"] = tass_t - base_nws_t

        if MODEL_PATH.exists():
            corrector: TassajaraMOSCorrector = joblib.load(MODEL_PATH)
            preds = corrector.predict(latest_fc_df)
            model_ver = "MOS_GBM_v2_HYDRO"
        else:
            X_f = engineer_mos_features(latest_fc_df)
            h_fb = analytical_hydro_hurdle_fallback(X_f)
            preds = latest_fc_df[["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
            preds["pred_temp_error_c"] = -1.2
            preds["corrected_temp_c"] = (preds["nws_temp_c"] - 1.2).round(2)
            preds["corrected_temp_p10_c"] = (preds["corrected_temp_c"] - 2.1).round(2)
            preds["corrected_temp_p90_c"] = (preds["corrected_temp_c"] + 2.1).round(2)
            preds["raw_nws_pop_pct"] = X_f["nws_pop_pct"].round(1).values
            preds["mos_pop_pct"] = h_fb["mos_pop_pct"].values
            preds["raw_nws_qpf_mm"] = X_f["nws_qpf_1hr_mm"].round(3).values
            preds["mos_qpf_mm"] = h_fb["mos_qpf_mm"].values
            preds["mos_qpf_p90_mm"] = h_fb["mos_qpf_p90_mm"].values
            model_ver = "TOPO_HYDRO_PHYSICS"

        fut_rows = [
            (
                str(r["issue_time_utc"]),
                str(r["valid_time_utc"]),
                model_ver,
                float(r["nws_temp_c"]),
                float(r["pred_temp_error_c"]),
                float(r["corrected_temp_c"]),
                float(r["corrected_temp_p10_c"]),
                float(r["corrected_temp_p90_c"]),
                clean_num(latest_fc_df.loc[idx, "nws_rh_pct"], 50.0),
                round(clean_num(latest_fc_df.loc[idx, "nws_wind_speed_ms"], 0.0) * 1.35, 2),
                clean_num(r.get("raw_nws_pop_pct"), 0.0),
                clean_num(r.get("mos_pop_pct"), 0.0),
                clean_num(r.get("raw_nws_qpf_mm"), 0.0),
                clean_num(r.get("mos_qpf_mm"), 0.0),
                clean_num(r.get("mos_qpf_p90_mm"), 0.0),
            )
            for idx, r in preds.iterrows()
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

    logger.info(f"Stored {len(fut_rows)} operational MOS forecast hours ({model_ver}).")
    return len(fut_rows)


def run_hydrological_risk_evaluation(db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Computes current Antecedent Precipitation Index (API), effective runoff coefficient (C),
    6h/24h/72h forward creek runoff & discharge projections, and records the 4-tier
    Flood & Tassajara Road Washout alert status in SQLite.
    """
    try:
        res = evaluate_and_store_hydro_risk(db_path=db_path)
        s = res.get("summary", {})
        logger.info(
            f"Hydrological Warning Engine Updated: [{s.get('alert_code')}] "
            f"API={s.get('api_in', 0.0):.2f} in | C={s.get('runoff_coeff_c', 0.08):.2f} | "
            f"Peak 24h Discharge={s.get('peak_discharge_24hr_cfs', 6.5):.1f} CFS"
        )
        return res
    except Exception as exc:
        logger.error(f"Hydrological risk engine evaluation failed: {exc}")
        return None


def run_alert_evaluation(db_path: str = DB_PATH) -> None:
    """Invokes alert_engine.py if present."""
    try:
        import alert_engine

        for fn_name in ("evaluate_alerts", "run_alerts", "check_alerts", "main"):
            fn = getattr(alert_engine, fn_name, None)
            if callable(fn):
                try:
                    fn(db_path)
                except TypeError:
                    fn()
                logger.info(f"Executed alert_engine.{fn_name}() successfully.")
                return
    except Exception as exc:
        logger.debug(f"Alert engine hook skipped: {exc}")


def main():
    ensure_schema_and_view(DB_PATH)
    ensure_hydro_schema(DB_PATH)
    try:
        ingest_live_observations_and_references(DB_PATH)
    except Exception as exc:
        logger.error(f"Live observation ingest error: {exc}")

    run_nws_forecast_ingest(DB_PATH)
    generate_operational_mos_forecasts(DB_PATH)
    run_hydrological_risk_evaluation(DB_PATH)
    run_alert_evaluation(DB_PATH)


if __name__ == "__main__":
    main()
