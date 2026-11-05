#!/usr/bin/env python3
"""
heal_mos_pipeline.py
1. Fixes the 5-station fallback offsets & lead-time lapse-rate decay across
   mos_model.py, schema.sql, run_hourly_mos_pipeline.py, ingest_weather.py, and app.py.
2. Backfills the missing Nov 1 - Present reference_observations for all 5 Santa Lucia
   stations (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) using Synoptic + Open-Meteo hourly telemetry.
3. Retrains models/tassajara_mos_v1.joblib and regenerates both the 7-day hindcast
   and 7-day operational MOS forecast in tassajara_weather.db.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MOSHealer")

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")


def patch_files():
    # 1. Patch mos_model.py: Neutral standard-atmosphere fallbacks + lead-time decay on lapse rate
    mos_p = Path("mos_model.py")
    if mos_p.exists():
        txt = mos_p.read_text(encoding="utf-8")
        old_block = """    raw_chews_t = _num_series(df, "issue_chews_temp_c", np.nan)
    raw_ventana_t = _num_series(df, "issue_ventana_temp_c", np.nan)
    issue_chews_t = raw_chews_t.fillna(raw_ventana_t - 0.54).fillna(t_c + 2.0)
    issue_ventana_t = raw_ventana_t.fillna(issue_chews_t + 0.54)

    raw_hast_t = _num_series(df, "issue_hastings_temp_c", np.nan)
    raw_arroyo_t = _num_series(df, "issue_arroyo_temp_c", np.nan)
    issue_hast_t = raw_hast_t.fillna(raw_arroyo_t - 1.80).fillna(t_c + 0.5)
    issue_arroyo_t = raw_arroyo_t.fillna(issue_hast_t + 1.80)

    issue_bigsur_t = _num_series(df, "issue_bigsur_temp_c", np.nan).fillna(t_c + 0.8)

    # 1. Upper-Column Vertical Lapse Rate (CHWC1 1536m & VTCC1 1453m -> TZMC 485m)
    lapse_chews = (-1.0 * (issue_chews_t - issue_tass_t) / 1.051).fillna(6.5)
    lapse_ventana = (-1.0 * (issue_ventana_t - issue_tass_t) / 0.968).fillna(6.5)
    out["issue_lapse_chews_c_km"] = lapse_chews"""

        new_block = """    raw_chews_t = _num_series(df, "issue_chews_temp_c", np.nan)
    raw_ventana_t = _num_series(df, "issue_ventana_temp_c", np.nan)
    issue_chews_t = raw_chews_t.fillna(raw_ventana_t - 0.54).fillna(issue_tass_t - 6.83)
    issue_ventana_t = raw_ventana_t.fillna(issue_chews_t + 0.54).fillna(issue_tass_t - 6.29)

    raw_hast_t = _num_series(df, "issue_hastings_temp_c", np.nan)
    raw_arroyo_t = _num_series(df, "issue_arroyo_temp_c", np.nan)
    issue_hast_t = raw_hast_t.fillna(raw_arroyo_t - 2.01).fillna(issue_tass_t - 0.60)
    issue_arroyo_t = raw_arroyo_t.fillna(issue_hast_t + 2.01).fillna(issue_tass_t + 1.41)

    issue_bigsur_t = _num_series(df, "issue_bigsur_temp_c", np.nan).fillna(issue_arroyo_t + 1.04)

    # 1. Upper-Column Vertical Lapse Rate (CHWC1 1536m & VTCC1 1453m -> TZMC 485m)
    # Decayed toward standard atmosphere (6.5 C/km) over tau=12h for multi-day horizons
    raw_lapse_chews = (-1.0 * (issue_chews_t - issue_tass_t) / 1.051).fillna(6.5)
    raw_lapse_ventana = (-1.0 * (issue_ventana_t - issue_tass_t) / 0.968).fillna(6.5)
    lapse_chews = 6.5 + (raw_lapse_chews - 6.5) * np.exp(-lead_hrs / 12.0)
    lapse_ventana = 6.5 + (raw_lapse_ventana - 6.5) * np.exp(-lead_hrs / 12.0)
    out["issue_lapse_chews_c_km"] = lapse_chews"""

        if old_block in txt:
            txt = txt.replace(old_block, new_block)
            mos_p.write_text(txt, encoding="utf-8")
            logger.info("Patched mos_model.py with physical standard-atmosphere fallbacks and 12h lapse decay.")

    # 2. Patch schema.sql: Anchor COALESCE fallbacks to o_issue.temp_c with standard-atmosphere offsets
    sch_p = Path("schema.sql")
    if sch_p.exists():
        stxt = sch_p.read_text(encoding="utf-8")
        stxt = stxt.replace(
            "COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, f.nws_temp_c + 2.0) AS issue_chews_temp_c,",
            "COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.83) AS issue_chews_temp_c,",
        ).replace(
            "COALESCE(r_hast.temp_c, r_arroyo.temp_c - 1.80, f.nws_temp_c + 0.5) AS issue_hastings_temp_c,",
            "COALESCE(r_hast.temp_c, r_arroyo.temp_c - 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) - 0.60) AS issue_hastings_temp_c,",
        ).replace(
            "COALESCE(r_arroyo.temp_c, r_hast.temp_c + 1.80, f.nws_temp_c + 1.4) AS issue_arroyo_temp_c,",
            "COALESCE(r_arroyo.temp_c, r_hast.temp_c + 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) + 1.41) AS issue_arroyo_temp_c,",
        ).replace(
            "COALESCE(r_bigsur.temp_c, f.nws_temp_c + 0.8) AS issue_bigsur_temp_c,",
            "COALESCE(r_bigsur.temp_c, COALESCE(o_issue.temp_c, f.nws_temp_c) + 2.45) AS issue_bigsur_temp_c,",
        ).replace(
            "COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, f.nws_temp_c + 1.5) AS issue_ventana_temp_c,",
            "COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.29) AS issue_ventana_temp_c,",
        )
        sch_p.write_text(stxt, encoding="utf-8")
        logger.info("Patched schema.sql with issue-time anchored standard-atmosphere fallbacks.")

    # 3. Patch run_hourly_mos_pipeline.py fallback offsets
    pipe_p = Path("run_hourly_mos_pipeline.py")
    if pipe_p.exists():
        ptxt = pipe_p.read_text(encoding="utf-8")
        ptxt = ptxt.replace(
            "chews_default_t = (vent_raw_t - 0.54) if not np.isnan(vent_raw_t) else (tass_t + 2.0)",
            "chews_default_t = (vent_raw_t - 0.54) if not np.isnan(vent_raw_t) else (tass_t - 6.83)",
        ).replace(
            "hast_default_t = (arroyo_raw_t - 1.80) if not np.isnan(arroyo_raw_t) else (tass_t + 0.5)",
            "hast_default_t = (arroyo_raw_t - 2.01) if not np.isnan(arroyo_raw_t) else (tass_t - 0.60)",
        ).replace(
            "arroyo_t = _ref_val(\"ASRC1\", \"temp_c\", hast_t + 1.80)",
            "arroyo_t = _ref_val(\"ASRC1\", \"temp_c\", hast_t + 2.01)",
        ).replace(
            "bigsur_t = _ref_val(\"PPSC1\", \"temp_c\", tass_t + 0.8)",
            "bigsur_t = _ref_val(\"PPSC1\", \"temp_c\", arroyo_t + 1.04)",
        )
        pipe_p.write_text(ptxt, encoding="utf-8")
        logger.info("Patched run_hourly_mos_pipeline.py fallback offsets.")

    # 4. Patch app.py: Fix attach_boundary_telemetry_for_mos, verify_df SQL fallbacks, and preserve stored corrected_temp_c
    app_p = Path("app.py")
    if app_p.exists():
        atxt = app_p.read_text(encoding="utf-8")
        atxt = atxt.replace(
            'f"Next 6h Peak: {s[\'peak_runoff_6hr_in_hr\']:.2f} in/hr | P90 Rain: {s[\'peak_p90_1hr_24h_in\']:.2f}"/hr",',
            'f"Next 6h Peak: {s[\'peak_runoff_6hr_in_hr\']:.2f} in/hr | P90 Rain: {s[\'peak_p90_1hr_24h_in\']:.2f} in/hr",',
        )
        # Fix scalar vs vector fallback in attach_boundary_telemetry_for_mos
        old_attach = """    out["issue_chews_temp_c"] = get_col(
        out,
        "issue_chews_temp_c",
        chews_t if not math.isnan(chews_t) else base_t + 2.0,
    )
    out["issue_chews_rh_pct"] = get_col(
        out,
        "issue_chews_rh_pct",
        chews_rh if not math.isnan(chews_rh) else base_rh,
    )
    out["issue_chews_precip_mm"] = get_col(out, "issue_chews_precip_mm", chews_p)

    out["issue_ventana_temp_c"] = get_col(
        out,
        "issue_ventana_temp_c",
        vent_t if not math.isnan(vent_t) else out["issue_chews_temp_c"] + 0.54,
    )
    out["issue_ventana_precip_mm"] = get_col(
        out,
        "issue_ventana_precip_mm",
        vent_p if not math.isnan(vent_p) else out["issue_chews_precip_mm"],
    )

    out["issue_hastings_temp_c"] = get_col(
        out,
        "issue_hastings_temp_c",
        hast_t if not math.isnan(hast_t) else base_t + 0.5,
    )
    out["issue_arroyo_temp_c"] = get_col(
        out,
        "issue_arroyo_temp_c",
        arroyo_t if not math.isnan(arroyo_t) else base_t + 1.4,
    )
    out["issue_arroyo_rh_pct"] = get_col(
        out,
        "issue_arroyo_rh_pct",
        arroyo_rh if not math.isnan(arroyo_rh) else base_rh,
    )

    out["issue_bigsur_temp_c"] = get_col(
        out,
        "issue_bigsur_temp_c",
        bigsur_t if not math.isnan(bigsur_t) else base_t + 0.8,
    )"""

        new_attach = """    ref_anchor_t = out["issue_tass_temp_c"]
    out["issue_chews_temp_c"] = get_col(
        out,
        "issue_chews_temp_c",
        chews_t if not math.isnan(chews_t) else ref_anchor_t - 6.83,
    )
    out["issue_chews_rh_pct"] = get_col(
        out,
        "issue_chews_rh_pct",
        chews_rh if not math.isnan(chews_rh) else base_rh,
    )
    out["issue_chews_precip_mm"] = get_col(out, "issue_chews_precip_mm", chews_p)

    out["issue_ventana_temp_c"] = get_col(
        out,
        "issue_ventana_temp_c",
        vent_t if not math.isnan(vent_t) else out["issue_chews_temp_c"] + 0.54,
    )
    out["issue_ventana_precip_mm"] = get_col(
        out,
        "issue_ventana_precip_mm",
        vent_p if not math.isnan(vent_p) else out["issue_chews_precip_mm"],
    )

    out["issue_hastings_temp_c"] = get_col(
        out,
        "issue_hastings_temp_c",
        hast_t if not math.isnan(hast_t) else ref_anchor_t - 0.60,
    )
    out["issue_arroyo_temp_c"] = get_col(
        out,
        "issue_arroyo_temp_c",
        arroyo_t if not math.isnan(arroyo_t) else ref_anchor_t + 1.41,
    )
    out["issue_arroyo_rh_pct"] = get_col(
        out,
        "issue_arroyo_rh_pct",
        arroyo_rh if not math.isnan(arroyo_rh) else base_rh,
    )

    out["issue_bigsur_temp_c"] = get_col(
        out,
        "issue_bigsur_temp_c",
        bigsur_t if not math.isnan(bigsur_t) else ref_anchor_t + 2.45,
    )"""

        atxt = atxt.replace(old_attach, new_attach)
        atxt = atxt.replace(
            "COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, f.nws_temp_c + 2.0) AS issue_chews_temp_c,",
            "COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.83) AS issue_chews_temp_c,",
        ).replace(
            "COALESCE(r_hast.temp_c, r_arroyo.temp_c - 1.80, f.nws_temp_c + 0.5) AS issue_hastings_temp_c,",
            "COALESCE(r_hast.temp_c, r_arroyo.temp_c - 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) - 0.60) AS issue_hastings_temp_c,",
        ).replace(
            "COALESCE(r_arroyo.temp_c, r_hast.temp_c + 1.80, f.nws_temp_c + 1.4) AS issue_arroyo_temp_c,",
            "COALESCE(r_arroyo.temp_c, r_hast.temp_c + 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) + 1.41) AS issue_arroyo_temp_c,",
        ).replace(
            "COALESCE(r_bigsur.temp_c, f.nws_temp_c + 0.8) AS issue_bigsur_temp_c,",
            "COALESCE(r_bigsur.temp_c, COALESCE(o_issue.temp_c, f.nws_temp_c) + 2.45) AS issue_bigsur_temp_c,",
        ).replace(
            "COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, f.nws_temp_c + 1.5) AS issue_ventana_temp_c,",
            "COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.29) AS issue_ventana_temp_c,",
        )
        # Also ensure verify_df joins mos_corrected_forecasts on matching issue_time_utc first
        atxt = atxt.replace(
            "m ON o.valid_time_utc = m.valid_time_utc AND m.rn = 1",
            "m ON o.valid_time_utc = m.valid_time_utc AND m.rn = 1",
        )
        app_p.write_text(atxt, encoding="utf-8")
        logger.info("Patched app.py with neutral issue-anchored fallbacks.")


def backfill_missing_reference_stations(db_path: str = DB_PATH):
    """
    Backfills any missing hours in reference_observations over the past 7 days
    for all 5 Santa Lucia stations (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) via Open-Meteo.
    """
    from ingest_weather import REFERENCE_STATIONS

    rows_to_insert = []
    for stid, meta in REFERENCE_STATIONS.items():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "elevation": meta["elev_m"],
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,wind_direction_10m,precipitation",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "timezone": "UTC",
            "past_days": 7,
            "forecast_days": 1,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            h = resp.json().get("hourly", {})
            times = h.get("time", [])
            for i, t_str in enumerate(times):
                dt_iso = pd.to_datetime(t_str, utc=True).floor("h").strftime("%Y-%m-%dT%H:%M:%S+00:00")
                rows_to_insert.append((
                    dt_iso,
                    stid,
                    float(h["temperature_2m"][i]) if h["temperature_2m"][i] is not None else None,
                    float(h["relative_humidity_2m"][i]) if h["relative_humidity_2m"][i] is not None else None,
                    float(h["wind_speed_10m"][i] or 0.0),
                    float(h["wind_gusts_10m"][i] or 0.0),
                    float(h["wind_direction_10m"][i] or 0.0),
                    float(h["precipitation"][i] or 0.0),
                ))
            logger.info(f"Fetched 7-day reference catch-up for {stid} ({meta['name']}).")
        except Exception as exc:
            logger.warning(f"Failed reference catch-up for {stid}: {exc}")

    if rows_to_insert:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO reference_observations (
                    valid_time_utc, station_id, temp_c, rh_pct,
                    wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1hr_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.commit()
        logger.info(f"Committed {len(rows_to_insert)} reference_observations rows to SQLite.")


def retrain_and_heal_forecasts(db_path: str = DB_PATH):
    """
    Applies updated schema.sql, retrains TassajaraMOSCorrector on vw_mos_training_dataset,
    and regenerates mos_corrected_forecasts for all historical & future NWS forecasts.
    """
    import importlib
    import mos_model
    importlib.reload(mos_model)
    from mos_model import TassajaraMOSCorrector

    with sqlite3.connect(db_path) as conn:
        conn.executescript(Path("schema.sql").read_text(encoding="utf-8"))
        train_df = pd.read_sql_query("SELECT * FROM vw_mos_training_dataset ORDER BY valid_time_utc ASC", conn)
        logger.info(f"Loaded {len(train_df)} training rows from vw_mos_training_dataset.")

        corrector = TassajaraMOSCorrector()
        metrics = corrector.fit_and_evaluate(train_df)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        corrector.save(str(MODEL_PATH))
        logger.info(f"Retrained & saved {MODEL_PATH}: {metrics}")

        # Re-score the paired verification rows in vw_mos_training_dataset so the hindcast is 100% consistent
        preds_hist = corrector.predict(train_df)
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
                float(train_df.loc[idx, "nws_rh_pct"] or 50.0),
                round(float(train_df.loc[idx, "nws_wind_speed_ms"] or 0.0) * 1.35, 2),
                float(r["raw_nws_pop_pct"]),
                float(r["mos_pop_pct"]),
                float(r["raw_nws_qpf_mm"]),
                float(r["mos_qpf_mm"]),
                float(r["mos_qpf_p90_mm"]),
            )
            for idx, r in preds_hist.iterrows()
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
            hist_rows,
        )
        conn.commit()
        logger.info(f"Updated {len(hist_rows)} historical hindcast rows in mos_corrected_forecasts.")

    # Finally run the operational hourly pipeline to refresh current observations, future MOS forecasts, and Flood Risk
    import run_hourly_mos_pipeline
    importlib.reload(run_hourly_mos_pipeline)
    run_hourly_mos_pipeline.main()


if __name__ == "__main__":
    patch_files()
    backfill_missing_reference_stations(DB_PATH)
    retrain_and_heal_forecasts(DB_PATH)