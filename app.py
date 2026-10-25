#!/usr/bin/env python3
"""
app.py (v6) — Tassajara Zen Mountain Center Operational Microclimate & MOS Dashboard
Schema-Resilient Edition: Uses SELECT * + Pandas merges and defensive column helpers
so SQLite never throws OperationalError (no such column / table) regardless of DB version.
"""

import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

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


# ---------------------------------------------------------------------------
# Unit & Physics Helpers
# ---------------------------------------------------------------------------
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
    """Safely returns a numeric/series column from a DataFrame or a default Series if missing."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(default_val)
    return pd.Series(default_val, index=df.index, dtype="float64")


def compute_canyon_solar_state(dt_utc: datetime):
    """
    Calculates solar elevation and whether the Stone Office (1,590 ft)
    is currently in direct canyon sun, behind the Black Cone / southern ridge shadow,
    or in nocturnal katabatic drainage.
    """
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

    sin_el = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    el_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))

    if el_deg <= 0.0:
        return "Nighttime Cold-Air Drainage", el_deg
    elif el_deg < 24.0:
        return "In Canyon Ridge Shadow", el_deg
    else:
        return "Direct Canyon Sun", el_deg


def fallback_physics_mos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Tassajara topographic solar-shadow & cold-air pooling correction
    when historical rows do not yet have pre-saved mos_corrected_forecasts entries.
    """
    out = df.copy()
    dt_utc = pd.to_datetime(out["valid_time_utc"], utc=True)
    dt_local = dt_utc.dt.tz_convert(LOCAL_TZ)
    hr_local = dt_local.dt.hour

    is_night = (hr_local <= 7) | (hr_local >= 19)
    is_afternoon_shade = (hr_local >= 15) & (hr_local < 19)
    wind_ms = get_col(out, "nws_wind_speed_ms", 1.5)
    calm_factor = np.clip(1.0 - (wind_ms / 6.0), 0.2, 1.0)

    offset_c = np.where(
        is_night,
        -2.2 * calm_factor,
        np.where(is_afternoon_shade, -1.6 * calm_factor, 0.6 * calm_factor),
    )
    raw_c = get_col(out, "raw_nws_temp_c", 15.0)
    out["corrected_temp_c"] = (raw_c + offset_c).round(2)
    out["corrected_temp_p10_c"] = (out["corrected_temp_c"] - 2.1).round(2)
    out["corrected_temp_p90_c"] = (out["corrected_temp_c"] + 2.1).round(2)
    return out


# ---------------------------------------------------------------------------
# Schema-Safe Database Loader (Zero hardcoded column assumptions in SQL)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=180)
def load_all_dashboard_data():
    if not Path(DB_PATH).exists():
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    obs_df = pd.DataFrame()
    ref_df = pd.DataFrame()
    mos_df = pd.DataFrame()
    nws_df = pd.DataFrame()
    alerts_df = pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        # 1. Local observations (SELECT * is immune to missing-column errors)
        try:
            obs_df = pd.read_sql_query(
                "SELECT * FROM local_observations ORDER BY valid_time_utc DESC LIMIT 200",
                conn,
            )
            if "temp_c" in obs_df.columns:
                obs_df = obs_df[obs_df["temp_c"].notna()].reset_index(drop=True)
        except Exception:
            obs_df = pd.DataFrame()

        # 2. Reference observations (Chews Ridge CHWC1 & Hastings CAHC1)
        try:
            ref_df = pd.read_sql_query(
                "SELECT * FROM reference_observations ORDER BY valid_time_utc DESC LIMIT 50",
                conn,
            )
        except Exception:
            ref_df = pd.DataFrame()

        # 3. MOS Corrected Forecasts (all recent rows)
        try:
            mos_df = pd.read_sql_query(
                "SELECT * FROM mos_corrected_forecasts ORDER BY valid_time_utc DESC LIMIT 600",
                conn,
            )
        except Exception:
            mos_df = pd.DataFrame()

        # 4. Raw NWS Forecasts (all recent rows)
        try:
            nws_df = pd.read_sql_query(
                "SELECT * FROM nws_forecasts ORDER BY valid_time_utc DESC LIMIT 1000",
                conn,
            )
        except Exception:
            nws_df = pd.DataFrame()

        # 5. Facility Alerts
        try:
            alerts_df = pd.read_sql_query(
                "SELECT * FROM facility_alert_log ORDER BY triggered_at_utc DESC LIMIT 15",
                conn,
            )
        except Exception:
            alerts_df = pd.DataFrame()

    # Build latest reference dictionary
    ref_dict = {}
    if not ref_df.empty and "station_id" in ref_df.columns:
        for stid, grp in ref_df.groupby("station_id"):
            ref_dict[str(stid)] = grp.iloc[0].to_dict()

    # Build Future Forecast DataFrame (Latest issue_time_utc from mos_df, merged with nws_df)
    future_fc_df = pd.DataFrame()
    if not mos_df.empty and "issue_time_utc" in mos_df.columns:
        latest_issue = mos_df["issue_time_utc"].max()
        future_fc_df = mos_df[mos_df["issue_time_utc"] == latest_issue].copy()
        future_fc_df = future_fc_df.sort_values("valid_time_utc").reset_index(drop=True)

        if not nws_df.empty and "valid_time_utc" in nws_df.columns:
            nws_latest = (
                nws_df.sort_values("issue_time_utc" if "issue_time_utc" in nws_df.columns else "valid_time_utc")
                .groupby("valid_time_utc", as_index=False)
                .last()
            )
            extra_nws_cols = [c for c in nws_latest.columns if c not in future_fc_df.columns or c == "valid_time_utc"]
            future_fc_df = pd.merge(
                future_fc_df,
                nws_latest[extra_nws_cols],
                on="valid_time_utc",
                how="left",
            )

    # Build Historical Verification DataFrame (Past 72 hours where BOTH actuals and forecasts exist)
    verify_df = pd.DataFrame()
    if not obs_df.empty and "valid_time_utc" in obs_df.columns:
        obs_clean = obs_df.copy()
        rename_obs = {}
        if "temp_c" in obs_clean.columns:
            rename_obs["temp_c"] = "actual_temp_c"
        if "rh_pct" in obs_clean.columns:
            rename_obs["rh_pct"] = "actual_rh_pct"
        if "wind_speed_ms" in obs_clean.columns:
            rename_obs["wind_speed_ms"] = "actual_wind_speed_ms"
        if "wind_gust_ms" in obs_clean.columns:
            rename_obs["wind_gust_ms"] = "actual_wind_gust_ms"
        if "precip_1hr_mm" in obs_clean.columns:
            rename_obs["precip_1hr_mm"] = "actual_precip_mm"
        obs_clean = obs_clean.rename(columns=rename_obs)

        mos_agg = pd.DataFrame()
        if not mos_df.empty and "valid_time_utc" in mos_df.columns:
            mos_agg = mos_df.groupby("valid_time_utc", as_index=False).last()

        nws_agg = pd.DataFrame()
        if not nws_df.empty and "valid_time_utc" in nws_df.columns:
            nws_agg = nws_df.groupby("valid_time_utc", as_index=False).last()
            if "nws_temp_c" in nws_agg.columns and "raw_nws_temp_c" not in nws_agg.columns:
                nws_agg["raw_nws_temp_c"] = nws_agg["nws_temp_c"]

        verify_df = obs_clean.copy()
        if not mos_agg.empty:
            m_cols = [c for c in mos_agg.columns if c not in verify_df.columns or c == "valid_time_utc"]
            verify_df = pd.merge(verify_df, mos_agg[m_cols], on="valid_time_utc", how="left")
        if not nws_agg.empty:
            n_cols = [c for c in nws_agg.columns if c not in verify_df.columns or c == "valid_time_utc"]
            verify_df = pd.merge(verify_df, nws_agg[n_cols], on="valid_time_utc", how="left")
            if "raw_nws_temp_c" in verify_df.columns and
