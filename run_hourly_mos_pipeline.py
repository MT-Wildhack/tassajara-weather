#!/usr/bin/env python3
"""
run_hourly_mos_pipeline.py (v2.0 — Exact Incremental Rain Delta Tracking, Provenance Tagging,
Continuous 3-Hr MSLP Tendency, Forecast Pruning & Loud Failure Exit Codes)
"""

import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

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
    get_local_day_boundaries_utc,
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
    dt = datetime.now(timezone.utc) if not ts_str else pd.to_datetime(ts_str, utc=True).to_pydatetime()
    return dt.replace(minute=0, second=0, microsecond=0).isoformat()


def _ensure_ingest_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_state (
            key TEXT PRIMARY KEY,
            val_text TEXT,
            val_num REAL,
            updated_utc TEXT
        )
        """
    )


def _get_state_num(conn: sqlite3.Connection, key: str) -> Tuple[Optional[str], Optional[float]]:
    row = conn.execute("SELECT val_text, val_num FROM ingest_state WHERE key = ?", (key,)).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _set_state(conn: sqlite3.Connection, key: str, val_text: Optional[str], val_num: Optional[float], now_iso: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ingest_state (key, val_text, val_num, updated_utc) VALUES (?, ?, ?, ?)",
        (key, val_text, val_num, now_iso),
    )


def ingest_live_observations_and_references(db_path: str = DB_PATH) -> Dict[str, Any]:
    wl_client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
    station_id = wl_client.get_station_id()

    try:
        recent_hist_df = fetch_weatherlink_history(wl_client, station_id, days_back=2.0)
        if not recent_hist_df.empty:
            store_observations_to_sqlite(recent_hist_df, db_path, source="WL_HISTORIC")
    except Exception as exc:
        logger.warning(f"48-hour historic catch-up skipped ({exc}); proceeding with live snapshot.")

    tass_obs = wl_client.fetch_current_observations(station_id)
    ref_obs = fetch_regional_reference_stations(list(REFERENCE_STATIONS.keys()))
    diag = compute_microclimate_diagnostics(tass_obs, ref_obs)

    obs_hour_iso = snap_to_hour_iso(tass_obs.get("valid_time_utc"))

    with sqlite3.connect(db_path) as conn:
        _ensure_ingest_state_table(conn)

        # Continuous 3-hour barometric tendency from SQLite MSLP (never relies on discrete dst=2 codes)
        ptrend_3h = tass_obs.get("pressure_trend_hpa")
        if ptrend_3h is None and tass_obs.get("mslp_hpa") is not None:
            prev_row = conn.execute(
                """
                SELECT mslp_hpa FROM local_observations
                WHERE mslp_hpa IS NOT NULL
                  AND valid_time_utc <= strftime('%Y-%m-%dT%H:%M:%S+00:00', ?, '-3 hours')
                  AND valid_time_utc >= strftime('%Y-%m-%dT%H:%M:%S+00:00', ?, '-6 hours')
                ORDER BY valid_time_utc DESC LIMIT 1
                """,
                (obs_hour_iso, obs_hour_iso),
            ).fetchone()
            if prev_row and prev_row[0] is not None:
                ptrend_3h = round(float(tass_obs["mslp_hpa"]) - float(prev_row[0]), 2)
        ptrend_3h = clean_num(ptrend_3h, 0.0)

        existing_hr = conn.execute(
            """
            SELECT temp_c, dewpoint_c, wetbulb_c, rh_pct, mslp_hpa,
                   precip_1hr_mm, wind_speed_ms, wind_gust_ms, wind_dir_deg, source
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
        hist_hr_wdir = existing_hr[8] if existing_hr else None

        cur_temp = clean_num(tass_obs.get("temp_c"), hist_temp)
        cur_dew = clean_num(tass_obs.get("dewpoint_c"), hist_dew)
        cur_wet = clean_num(tass_obs.get("wetbulb_c"), hist_wet)
        cur_rh = clean_num(tass_obs.get("rh_pct"), hist_rh)
        cur_mslp = clean_num(tass_obs.get("mslp_hpa"), hist_mslp or 1013.25)

        # Conservation-of-mass clock-hour rain (WL_HISTORIC base + post-archive tips only;
        # never uses rolling rainfall_last_60_min_mm across hour boundaries)
        _, today_mid_iso, _ = get_local_day_boundaries_utc(
            pd.to_datetime(obs_hour_iso, utc=True).to_pydatetime()
        )
        console_day_mm = clean_num(tass_obs.get("precip_day_mm"), 0.0) or 0.0
        raw_console_year_mm = clean_num(tass_obs.get("precip_year_mm"), None)
        prev_day_mid_iso, prev_console_day_mm = _get_state_num(conn, "last_console_rain_day_mm")
        _, prev_console_year_mm = _get_state_num(conn, "last_console_rain_year_mm")

        prior_today_row = conn.execute(
            """
            SELECT COALESCE(SUM(precip_1hr_mm), 0.0)
            FROM local_observations
            WHERE valid_time_utc >= ?
              AND valid_time_utc < ?
            """,
            (today_mid_iso, obs_hour_iso),
        ).fetchone()
        db_prior_today_mm = float(prior_today_row[0] or 0.0) if prior_today_row else 0.0

        delta_year_mm = (
            round(raw_console_year_mm - prev_console_year_mm, 3)
            if (raw_console_year_mm is not None and prev_console_year_mm is not None and raw_console_year_mm >= prev_console_year_mm)
            else None
        )
        # Guard against console clock lagging past local midnight before rain_day resets
        console_not_reset_at_midnight = (
            obs_hour_iso == today_mid_iso
            and prev_day_mid_iso is not None
            and prev_day_mid_iso != today_mid_iso
            and prev_console_day_mm is not None
            and prev_console_day_mm > 0.0
            and console_day_mm >= prev_console_day_mm
            and (delta_year_mm is None or delta_year_mm < (console_day_mm - 0.01))
        )
        if console_not_reset_at_midnight:
            console_this_hr_mm = round(max(0.0, console_day_mm - prev_console_day_mm), 3)
        else:
            console_this_hr_mm = round(max(0.0, console_day_mm - db_prior_today_mm), 3)

        cur_hr_rain = round(max(hist_hr_rain, console_this_hr_mm), 3)

        _set_state(conn, "last_console_rain_day_mm", today_mid_iso, console_day_mm, obs_hour_iso)
        if raw_console_year_mm is not None:
            _set_state(conn, "last_console_rain_year_mm", obs_hour_iso, raw_console_year_mm, obs_hour_iso)
        _set_state(conn, "last_obs_hour_iso", obs_hour_iso, None, obs_hour_iso)

        cur_wspd = max(hist_hr_wspd, clean_num(tass_obs.get("wind_speed_ms"), 0.0) or 0.0)
        cur_wgst = max(hist_hr_wgst, cur_wspd, clean_num(tass_obs.get("wind_gust_ms"), 0.0) or 0.0)
        raw_wdir = clean_num(tass_obs.get("wind_dir_deg"), None)
        cur_wdir = raw_wdir if raw_wdir is not None else hist_hr_wdir

        if cur_temp is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO local_observations (
                    valid_time_utc, station_id, temp_c, dewpoint_c, wetbulb_c,
                    rh_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
                    mslp_hpa, pressure_trend_3hr_hpa, precip_1hr_mm,
                    solar_rad_wm2, uv_index, qc_flag, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    "WL_LIVE",
                ),
            )

        ref_rows = []
        for stid, rdata in ref_obs.items():
            src_r = str(rdata.get("source") or "")
            if src_r == "OFFLINE" or (rdata.get("temp_c") is None and rdata.get("precip_day_mm") is None):
                continue
            # Compute hourly rain increment from monotonic precip_today[in] only on same-day runs <= 2h
            if rdata.get("precip_day_mm") is not None and (rdata.get("precip_1hr_mm") or 0.0) == 0.0:
                ref_day_mm = float(rdata["precip_day_mm"])
                st_key = f"last_ref_rain_day_mm_{stid}"
                prev_ref_ts, prev_ref_day = _get_state_num(conn, st_key)
                if prev_ref_day is not None and prev_ref_ts:
                    try:
                        gap_sec = abs((pd.to_datetime(obs_hour_iso, utc=True) - pd.to_datetime(prev_ref_ts, utc=True)).total_seconds())
                    except Exception:
                        gap_sec = 999999.0
                    same_local_day = prev_ref_ts >= today_mid_iso
                    if gap_sec <= 7200.0 and same_local_day and ref_day_mm >= prev_ref_day:
                        ref_delta = round(ref_day_mm - prev_ref_day, 2)
                        if ref_delta > 0.0:
                            rdata["precip_1hr_mm"] = ref_delta
                _set_state(conn, st_key, obs_hour_iso, ref_day_mm, obs_hour_iso)
            ref_rows.append((
                obs_hour_iso,
                stid,
                clean_num(rdata.get("temp_c")),
                clean_num(rdata.get("rh_pct")),
                clean_num(rdata.get("wind_speed_ms"), 0.0),
                clean_num(rdata.get("wind_gust_ms"), 0.0),
                clean_num(rdata.get("wind_dir_deg"), None),
                clean_num(rdata.get("precip_1hr_mm"), 0.0),
                str(rdata.get("source") or "RAWS_SYNOPTIC"),
            ))

        if ref_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO reference_observations (
                    valid_time_utc, station_id, temp_c, rh_pct,
                    wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1hr_mm, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ref_rows,
            )
        conn.commit()

    logger.info(
        f"Synced Stone Office ({obs_hour_iso}: {cur_temp} C, ptrend={ptrend_3h:+.2f} hPa, "
        f"1h_rain={cur_hr_rain:.3f} mm, day_rain={console_day_mm:.2f} mm) & {len(ref_rows)}/5 refs."
    )
    return {"tass_obs": tass_obs, "ref_obs": ref_obs, "diag": diag, "ptrend_3h": ptrend_3h}


def run_nws_forecast_ingest(db_path: str = DB_PATH) -> int:
    import ingest_nws
    n_rows = ingest_nws.fetch_and_store_nws_hourly(db_path)
    if hasattr(ingest_nws, "prune_old_forecasts"):
        ingest_nws.prune_old_forecasts(db_path, keep_days=14)
    return int(n_rows or 0)


def generate_operational_mos_forecasts(db_path: str = DB_PATH) -> int:
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
            if not refs_latest_df.empty else {}
        )

        base_nws_t = float(latest_fc_df["nws_temp_c"].iloc[0])
        base_nws_rh = clean_num(latest_fc_df["nws_rh_pct"].iloc[0], 50.0) or 50.0
        tass_t = float(obs_latest.iloc[0]["temp_c"]) if not obs_latest.empty else base_nws_t
        ptrend = (
            clean_num(obs_latest.iloc[0].get("pressure_trend_3hr_hpa"), 0.0)
            if not obs_latest.empty else 0.0
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
        latest_fc_df["issue_tass_temp_c"] = tass_t
        latest_fc_df["issue_persistence_bias_c"] = tass_t - base_nws_t

        if MODEL_PATH.exists():
            corrector: TassajaraMOSCorrector = joblib.load(MODEL_PATH)
            preds = corrector.predict(latest_fc_df)
            model_ver = "MOS_GBM_v3_HYDRO"
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
                round(max(clean_num(latest_fc_df.loc[idx, "nws_wind_gust_ms"], 0.0) or 0.0, (clean_num(latest_fc_df.loc[idx, "nws_wind_speed_ms"], 0.0) or 0.0) * 1.35), 2),
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


def run_alert_evaluation(db_path: str = DB_PATH) -> None:
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
    critical_failures = []

    try:
        ingest_live_observations_and_references(DB_PATH)
    except Exception as exc:
        logger.error(f"Live observation ingest error: {exc}")
        critical_failures.append(f"live_ingest: {exc}")

    try:
        n_nws = run_nws_forecast_ingest(DB_PATH)
        if n_nws == 0:
            critical_failures.append("nws_ingest returned 0 rows")
    except Exception as exc:
        logger.error(f"NWS forecast ingest error: {exc}")
        critical_failures.append(f"nws_ingest: {exc}")

    try:
        generate_operational_mos_forecasts(DB_PATH)
    except Exception as exc:
        logger.error(f"MOS inference error: {exc}")
        critical_failures.append(f"mos_inference: {exc}")

    try:
        evaluate_and_store_hydro_risk(db_path=DB_PATH)
    except Exception as exc:
        logger.error(f"Hydrological risk engine failure: {exc}")
        critical_failures.append(f"hydro_risk: {exc}")

    run_alert_evaluation(DB_PATH)

    if critical_failures:
        logger.error(f"Pipeline completed with critical failure(s): {'; '.join(critical_failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()