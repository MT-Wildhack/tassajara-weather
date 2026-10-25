#!/usr/bin/env python3
"""
app.py (v7) — Tassajara Zen Mountain Center Operational Microclimate & MOS Dashboard
Line-Wrapped & Schema-Resilient Edition:
- Every line strictly < 86 characters to prevent browser/editor word-wrap SyntaxErrors
- Uses SELECT * + Pandas merges so SQLite never throws OperationalError
"""

import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Tassajara Shop Weather & MOS Forecast",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")
LOCAL_TZ = "America/Los_Angeles"


def c_to_f(c):
    if c is None:
        return np.nan
    return c * 9.0 / 5.0 + 32.0


def ms_to_mph(ms):
    if ms is None:
        return 0.0
    return ms * 2.23694


def mm_to_in(mm):
    if mm is None:
        return 0.0
    return mm / 25.4


def hpa_to_inhg(hpa):
    if hpa is None or pd.isna(hpa):
        return 29.92
    return hpa * 0.02953


def deg_to_cardinal(deg):
    if deg is None or pd.isna(deg):
        return "Calm"
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    try:
        ix = int(round(float(deg) / 22.5)) % 16
        return dirs[ix]
    except Exception:
        return "Calm"


def safe_float(val, default=0.0):
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def get_col(df: pd.DataFrame, col_name: str, default_val=np.nan) -> pd.Series:
    if col_name in df.columns:
        s = pd.to_numeric(df[col_name], errors="coerce")
        return s.fillna(default_val)
    return pd.Series(default_val, index=df.index, dtype="float64")


def compute_canyon_solar_state(dt_utc: datetime):
    lat_rad = math.radians(36.2337)
    lon_deg = -121.5501
    doy = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0

    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    tst = (hour_utc * 60.0 + eqtime + 4.0 * lon_deg) % 1440.0
    ha = math.radians((tst / 4.0) - 180.0)

    sin_el = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    )
    el_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))

    if el_deg <= 0.0:
        return "Nighttime Cold-Air Drainage", el_deg
    if el_deg < 24.0:
        return "In Canyon Ridge Shadow", el_deg
    return "Direct Canyon Sun", el_deg


def fallback_physics_mos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt_utc = pd.to_datetime(out["valid_time_utc"], utc=True)
    dt_local = dt_utc.dt.tz_convert(LOCAL_TZ)
    hr_local = dt_local.dt.hour

    is_night = (hr_local <= 7) | (hr_local >= 19)
    is_shade = (hr_local >= 15) & (hr_local < 19)
    wind_ms = get_col(out, "nws_wind_speed_ms", 1.5)
    calm_factor = np.clip(1.0 - (wind_ms / 6.0), 0.2, 1.0)

    offset_c = np.where(
        is_night,
        -2.2 * calm_factor,
        np.where(is_shade, -1.6 * calm_factor, 0.6 * calm_factor),
    )
    raw_c = get_col(out, "raw_nws_temp_c", 15.0)
    out["corrected_temp_c"] = (raw_c + offset_c).round(2)
    out["corrected_temp_p10_c"] = (out["corrected_temp_c"] - 2.1).round(2)
    out["corrected_temp_p90_c"] = (out["corrected_temp_c"] + 2.1).round(2)
    return out


@st.cache_data(ttl=180)
def load_all_dashboard_data():
    if not Path(DB_PATH).exists():
        return (
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    obs_df = pd.DataFrame()
    ref_df = pd.DataFrame()
    mos_df = pd.DataFrame()
    nws_df = pd.DataFrame()
    alerts_df = pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        try:
            q_obs = (
                "SELECT * FROM local_observations "
                "ORDER BY valid_time_utc DESC LIMIT 200"
            )
            obs_df = pd.read_sql_query(q_obs, conn)
            if "temp_c" in obs_df.columns:
                obs_df = obs_df[obs_df["temp_c"].notna()]
                obs_df = obs_df.reset_index(drop=True)
        except Exception:
            obs_df = pd.DataFrame()

        try:
            q_ref = (
                "SELECT * FROM reference_observations "
                "ORDER BY valid_time_utc DESC LIMIT 50"
            )
            ref_df = pd.read_sql_query(q_ref, conn)
        except Exception:
            ref_df = pd.DataFrame()

        try:
            q_mos = (
                "SELECT * FROM mos_corrected_forecasts "
                "ORDER BY valid_time_utc DESC LIMIT 600"
            )
            mos_df = pd.read_sql_query(q_mos, conn)
        except Exception:
            mos_df = pd.DataFrame()

        try:
            q_nws = (
                "SELECT * FROM nws_forecasts "
                "ORDER BY valid_time_utc DESC LIMIT 1000"
            )
            nws_df = pd.read_sql_query(q_nws, conn)
        except Exception:
            nws_df = pd.DataFrame()

        try:
            q_alt = (
                "SELECT * FROM facility_alert_log "
                "ORDER BY triggered_at_utc DESC LIMIT 15"
            )
            alerts_df = pd.read_sql_query(q_alt, conn)
        except Exception:
            alerts_df = pd.DataFrame()

    ref_dict = {}
    if not ref_df.empty and "station_id" in ref_df.columns:
        for stid, grp in ref_df.groupby("station_id"):
            ref_dict[str(stid)] = grp.iloc[0].to_dict()

    future_fc_df = pd.DataFrame()
    if not mos_df.empty and "issue_time_utc" in mos_df.columns:
        latest_issue = mos_df["issue_time_utc"].max()
        mask_latest = mos_df["issue_time_utc"] == latest_issue
        future_fc_df = mos_df[mask_latest].copy()
        future_fc_df = future_fc_df.sort_values("valid_time_utc")
        future_fc_df = future_fc_df.reset_index(drop=True)

        if not nws_df.empty and "valid_time_utc" in nws_df.columns:
            sort_col = (
                "issue_time_utc"
                if "issue_time_utc" in nws_df.columns
                else "valid_time_utc"
            )
            nws_latest = (
                nws_df.sort_values(sort_col)
                .groupby("valid_time_utc", as_index=False)
                .last()
            )
            extra_cols = [
                c
                for c in nws_latest.columns
                if c not in future_fc_df.columns or c == "valid_time_utc"
            ]
            future_fc_df = pd.merge(
                future_fc_df,
                nws_latest[extra_cols],
                on="valid_time_utc",
                how="left",
            )

    verify_df = pd.DataFrame()
    if not obs_df.empty and "valid_time_utc" in obs_df.columns:
        obs_clean = obs_df.copy()
        rename_map = {
            "temp_c": "actual_temp_c",
            "rh_pct": "actual_rh_pct",
            "wind_speed_ms": "actual_wind_speed_ms",
            "wind_gust_ms": "actual_wind_gust_ms",
            "precip_1hr_mm": "actual_precip_mm",
        }
        active_renames = {
            k: v for k, v in rename_map.items() if k in obs_clean.columns
        }
        obs_clean = obs_clean.rename(columns=active_renames)

        mos_agg = pd.DataFrame()
        if not mos_df.empty and "valid_time_utc" in mos_df.columns:
            mos_agg = mos_df.groupby("valid_time_utc", as_index=False).last()

        nws_agg = pd.DataFrame()
        if not nws_df.empty and "valid_time_utc" in nws_df.columns:
            nws_agg = nws_df.groupby("valid_time_utc", as_index=False).last()
            has_nws_t = "nws_temp_c" in nws_agg.columns
            no_raw_t = "raw_nws_temp_c" not in nws_agg.columns
            if has_nws_t and no_raw_t:
                nws_agg["raw_nws_temp_c"] = nws_agg["nws_temp_c"]

        verify_df = obs_clean.copy()
        if not mos_agg.empty:
            m_cols = [
                c
                for c in mos_agg.columns
                if c not in verify_df.columns or c == "valid_time_utc"
            ]
            verify_df = pd.merge(
                verify_df,
                mos_agg[m_cols],
                on="valid_time_utc",
                how="left",
            )

        if not nws_agg.empty:
            n_cols = [
                c
                for c in nws_agg.columns
                if c not in verify_df.columns or c == "valid_time_utc"
            ]
            verify_df = pd.merge(
                verify_df,
                nws_agg[n_cols],
                on="valid_time_utc",
                how="left",
            )
            has_raw_v = "raw_nws_temp_c" in verify_df.columns
            has_nws_a = "nws_temp_c" in nws_agg.columns
            if has_raw_v and has_nws_a:
                nws_map = nws_agg.set_index("valid_time_utc")["nws_temp_c"]
                mapped_t = verify_df["valid_time_utc"].map(nws_map)
                verify_df["raw_nws_temp_c"] = (
                    verify_df["raw_nws_temp_c"].combine_first(mapped_t)
                )

        if "raw_nws_temp_c" in verify_df.columns:
            mask_valid = (
                verify_df["raw_nws_temp_c"].notna()
                & verify_df["actual_temp_c"].notna()
            )
            verify_df = verify_df[mask_valid].copy()
            verify_df = verify_df.sort_values("valid_time_utc").tail(72)
            verify_df = verify_df.reset_index(drop=True)

            if "corrected_temp_c" not in verify_df.columns:
                verify_df["corrected_temp_c"] = np.nan
                verify_df["corrected_temp_p10_c"] = np.nan
                verify_df["corrected_temp_p90_c"] = np.nan

            missing_mos = verify_df["corrected_temp_c"].isna()
            if missing_mos.any():
                fb = fallback_physics_mos(verify_df.loc[missing_mos])
                verify_df.loc[missing_mos, "corrected_temp_c"] = (
                    fb["corrected_temp_c"]
                )
                verify_df.loc[missing_mos, "corrected_temp_p10_c"] = (
                    fb["corrected_temp_p10_c"]
                )
                verify_df.loc[missing_mos, "corrected_temp_p90_c"] = (
                    fb["corrected_temp_p90_c"]
                )
        else:
            verify_df = pd.DataFrame()

    return ref_dict, obs_df, future_fc_df, verify_df, alerts_df


# ---------------------------------------------------------------------------
# Header & Top Operational Status Bar
# ---------------------------------------------------------------------------
(
    ref_dict,
    obs_df,
    future_fc_df,
    verify_df,
    alerts_df,
) = load_all_dashboard_data()

now_utc = datetime.now(timezone.utc)
solar_state, solar_el = compute_canyon_solar_state(now_utc)

if MODEL_PATH.exists():
    active_model_name = "MOS_GBM_v1 (Trained ML Ensemble)"
else:
    active_model_name = "TOPO_PHYSICS (Canyon Shadow & Drainage Baseline)"

if not future_fc_df.empty and "model_version" in future_fc_df.columns:
    mv = future_fc_df["model_version"].dropna()
    if not mv.empty:
        active_model_name = str(mv.iloc[0])

st.markdown(
    "## 🌲 Tassajara Zen Mountain Center — "
    "Canyon Weather & Shop Operations Monitor"
)

if not obs_df.empty and "valid_time_utc" in obs_df.columns:
    last_obs_dt = pd.to_datetime(
        obs_df.iloc[0]["valid_time_utc"], utc=True
    ).tz_convert(LOCAL_TZ)
    last_obs_str = last_obs_dt.strftime("%a %b %d, %I:%M %p %Z")
else:
    last_obs_str = "Awaiting telemetry"

st.caption(
    f"**Station:** Stone Office Davis Vantage Pro2 (`1,590 ft` / `485 m`) | "
    f"**Ridge Baselines:** Chews Ridge `CHWC1` (`5,040 ft`) & "
    f"Hastings `CAHC1` (`1,792 ft`) | "
    f"**Last Updated:** {last_obs_str} | "
    f"**Active Engine:** `{active_model_name}`"
)

# ---------------------------------------------------------------------------
# Section 1: Actionable Shop Operational Advisories (Next 48 Hours)
# ---------------------------------------------------------------------------
if not future_fc_df.empty:
    fc_48 = future_fc_df.head(48).copy()
    fc_48["valid_local"] = pd.to_datetime(
        fc_48["valid_time_utc"], utc=True
    ).dt.tz_convert(LOCAL_TZ)

    corr_t = get_col(fc_48, "corrected_temp_c", 15.0)
    p10_t = get_col(fc_48, "corrected_temp_p10_c", corr_t - 2.0)

    min_temp_f = c_to_f(float(corr_t.min()))
    min_p10_f = c_to_f(float(p10_t.min()))
    min_row_idx = corr_t.idxmin()
    min_time_str = fc_48.loc[min_row_idx, "valid_local"].strftime(
        "%a %I:%M %p"
    )
    max_temp_f = c_to_f(float(corr_t.max()))

    rh_col_48 = get_col(fc_48, "corrected_rh_pct", np.nan).combine_first(
        get_col(fc_48, "nws_rh_pct", 45.0)
    )
    min_rh_48 = safe_float(
