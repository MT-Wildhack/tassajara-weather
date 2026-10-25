#!/usr/bin/env python3
"""
app.py — Tassajara Zen Mountain Center Meteorological & MOS Forecast Monitor
Publication-grade scientific monitoring suite for Station 36966 (Stone Office Vantage Pro2),
Chews Ridge (CHWC1) & Hastings (CAHC1) boundary references, and MOS ensemble verification.
"""

import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="TZMC Atmospheric & MOS Forecast Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Publication-grade scientific CSS styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 6px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 0.76rem !important;
        font-variant-numeric: tabular-nums;
    }
    h1, h2, h3 {
        letter-spacing: -0.01em;
        font-weight: 600;
    }
    .met-header-bar {
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        padding-bottom: 0.6rem;
        margin-bottom: 1.0rem;
    }
    .met-subbanner {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.80rem;
        color: #94a3b8;
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38bdf8;
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")
LOCAL_TZ = "America/Los_Angeles"

ELEV_TASS_M = 485.0
ELEV_CHEWS_M = 1536.0
ELEV_HAST_M = 546.0


# ---------------------------------------------------------------------------
# Meteorological Unit & Thermodynamic Conversions
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
    if col_name in df.columns:
        s = pd.to_numeric(df[col_name], errors="coerce")
        return s.fillna(default_val)
    return pd.Series(default_val, index=df.index, dtype="float64")


def compute_vpd_kpa(temp_c: float, rh_pct: float) -> float:
    """Computes Vapor Pressure Deficit (kPa) via Tetens equation."""
    if temp_c is None or rh_pct is None or pd.isna(temp_c) or pd.isna(rh_pct):
        return 0.0
    es = 0.61078 * math.exp((17.27 * float(temp_c)) / (float(temp_c) + 237.3))
    ea = es * (np.clip(float(rh_pct), 0.0, 100.0) / 100.0)
    return max(0.0, round(es - ea, 2))


def compute_wetbulb_stull_c(temp_c: float, rh_pct: float) -> float:
    """Stull (2011) empirical psychrometric wet-bulb approximation (°C)."""
    t = float(temp_c)
    r = np.clip(float(rh_pct), 1.0, 100.0)
    wb = (
        t * math.atan(0.151977 * math.sqrt(r + 8.313659))
        + math.atan(t + r)
        - math.atan(r - 1.676331)
        + 0.00391838 * (r ** 1.5) * math.atan(0.023101 * r)
        - 4.686035
    )
    return round(wb, 2)


def compute_canyon_solar_state(dt_utc: datetime):
    """Computes solar elevation, azimuth, and Tassajara Canyon horizon shadow regime."""
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

    cos_az = (math.sin(decl) - math.sin(lat_rad) * sin_el) / (
        math.cos(lat_rad) * math.cos(math.radians(el_deg)) + 1e-6
    )
    az_base = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    az_deg = (360.0 - az_base) % 360.0 if ha > 0 else az_base

    horizon_deg = 18.0
    if 145.0 <= az_deg <= 255.0:
        horizon_deg = 32.0  # Black Cone Trail / Pine Ridge SW wall
    elif 45.0 <= az_deg <= 115.0:
        horizon_deg = 23.0  # Eastern canyon wall

    if el_deg <= 0.0:
        return "Nocturnal Katabatic Drainage", el_deg, az_deg, horizon_deg
    if el_deg < horizon_deg:
        return "Canyon Wall Topographic Shadow", el_deg, az_deg, horizon_deg
    return "Direct Canyon Insolation", el_deg, az_deg, horizon_deg


def fallback_physics_mos(df: pd.DataFrame) -> pd.DataFrame:
    """Topographic radiation-drainage physical MOS fallback when model weights are absent."""
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
        np.where(is_shade, -1.6 * calm_factor, 0.8 * calm_factor),
    )
    raw_c = get_col(out, "raw_nws_temp_c", get_col(out, "nws_temp_c", 15.0))
    out["pred_temp_error_c"] = offset_c.round(2)
    out["corrected_temp_c"] = (raw_c + offset_c).round(2)
    out["corrected_temp_p10_c"] = (out["corrected_temp_c"] - 2.1).round(2)
    out["corrected_temp_p90_c"] = (out["corrected_temp_c"] + 2.1).round(2)
    return out


def apply_mos_inference_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every row in `df` has MOS corrected_temp_c, p10, and p90.
    Uses trained `models/tassajara_mos_v1.joblib` if available, else physical baseline.
    """
    if df.empty:
        return df
    out = df.copy()
    if "corrected_temp_c" not in out.columns:
        out["corrected_temp_c"] = np.nan
        out["corrected_temp_p10_c"] = np.nan
        out["corrected_temp_p90_c"] = np.nan

    missing_mask = out["corrected_temp_c"].isna()
    if not missing_mask.any():
        return out

    sub = out.loc[missing_mask].copy()
    if "nws_temp_c" not in sub.columns and "raw_nws_temp_c" in sub.columns:
        sub["nws_temp_c"] = sub["raw_nws_temp_c"]
    if "raw_nws_temp_c" not in sub.columns and "nws_temp_c" in sub.columns:
        sub["raw_nws_temp_c"] = sub["nws_temp_c"]

    if MODEL_PATH.exists():
        try:
            mos_model = joblib.load(MODEL_PATH)
            if "lead_time_hours" not in sub.columns:
                sub["lead_time_hours"] = 6
            sub["lead_time_hours"] = pd.to_numeric(sub["lead_time_hours"], errors="coerce").fillna(6)
            sub["nws_rh_pct"] = get_col(sub, "nws_rh_pct", 50.0)
            sub["nws_wind_speed_ms"] = get_col(sub, "nws_wind_speed_ms", 1.5)
            sub["nws_wind_dir_deg"] = get_col(sub, "nws_wind_dir_deg", 295.0)
            sub["nws_pop_pct"] = get_col(sub, "nws_pop_pct", 0.0)
            sub["nws_sky_cover_pct"] = get_col(sub, "nws_sky_cover_pct", 20.0)
            sub["pressure_trend_3hr_hpa"] = get_col(sub, "pressure_trend_3hr_hpa", 0.0)
            sub["issue_chews_temp_c"] = get_col(sub, "issue_chews_temp_c", sub["nws_temp_c"] + 2.0)
            sub["issue_tass_temp_c"] = get_col(sub, "issue_tass_temp_c", sub["nws_temp_c"])
            sub["issue_persistence_bias_c"] = get_col(sub, "issue_persistence_bias_c", 0.0)
            if "issue_time_utc" not in sub.columns:
                sub["issue_time_utc"] = sub["valid_time_utc"]

            preds = mos_model.predict(sub)
            out.loc[missing_mask, "pred_temp_error_c"] = preds["pred_temp_error_c"].values
            out.loc[missing_mask, "corrected_temp_c"] = preds["corrected_temp_c"].values
            out.loc[missing_mask, "corrected_temp_p10_c"] = preds["corrected_temp_p10_c"].values
            out.loc[missing_mask, "corrected_temp_p90_c"] = preds["corrected_temp_p90_c"].values
            if "model_version" not in out.columns:
                out["model_version"] = "MOS_GBM_v1"
            else:
                out.loc[missing_mask, "model_version"] = "MOS_GBM_v1"
            return out
        except Exception:
            pass

    fb = fallback_physics_mos(sub)
    out.loc[missing_mask, "corrected_temp_c"] = fb["corrected_temp_c"].values
    out.loc[missing_mask, "corrected_temp_p10_c"] = fb["corrected_temp_p10_c"].values
    out.loc[missing_mask, "corrected_temp_p90_c"] = fb["corrected_temp_p90_c"].values
    return out


# ---------------------------------------------------------------------------
# Data Loading & SQL Alignment
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120)
def load_all_dashboard_data():
    if not Path(DB_PATH).exists():
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        # 1. Recent Local Observations (Stone Office Vantage Pro2)
        try:
            obs_df = pd.read_sql_query(
                """
                SELECT * FROM local_observations
                WHERE temp_c IS NOT NULL
                ORDER BY valid_time_utc DESC
                LIMIT 240
                """,
                conn,
            )
        except Exception:
            obs_df = pd.DataFrame()

        # 2. Latest Reference Station Observations (CHWC1 & CAHC1)
        ref_dict = {}
        try:
            if not obs_df.empty:
                latest_obs_ts = str(obs_df.iloc[0]["valid_time_utc"])
                ref_df = pd.read_sql_query(
                    """
                    SELECT * FROM reference_observations
                    WHERE valid_time_utc <= ?
                    ORDER BY valid_time_utc DESC
                    LIMIT 20
                    """,
                    conn,
                    params=(latest_obs_ts,),
                )
            else:
                ref_df = pd.read_sql_query(
                    "SELECT * FROM reference_observations ORDER BY valid_time_utc DESC LIMIT 20",
                    conn,
                )
            if not ref_df.empty and "station_id" in ref_df.columns:
                for stid, grp in ref_df.groupby("station_id"):
                    ref_dict[str(stid)] = grp.iloc[0].to_dict()
        except Exception:
            ref_dict = {}

        # 3. Future 7-Day Forecast Series (Select latest complete issue batch >= 48 hours)
        future_fc_df = pd.DataFrame()
        try:
            nws_issue_row = pd.read_sql_query(
                """
                SELECT issue_time_utc, COUNT(*) AS cnt
                FROM nws_forecasts
                WHERE short_forecast != 'Archived NWP Grid' OR short_forecast IS NULL
                GROUP BY issue_time_utc
                HAVING COUNT(*) >= 48
                ORDER BY issue_time_utc DESC
                LIMIT 1
                """,
                conn,
            )
            if nws_issue_row.empty:
                nws_issue_row = pd.read_sql_query(
                    """
                    SELECT issue_time_utc, COUNT(*) AS cnt
                    FROM nws_forecasts
                    GROUP BY issue_time_utc
                    HAVING COUNT(*) >= 24
                    ORDER BY issue_time_utc DESC
                    LIMIT 1
                    """,
                    conn,
                )

            if not nws_issue_row.empty:
                latest_nws_issue = str(nws_issue_row.iloc[0]["issue_time_utc"])
                future_fc_df = pd.read_sql_query(
                    """
                    SELECT
                        f.issue_time_utc,
                        f.valid_time_utc,
                        f.lead_time_hours,
                        f.nws_temp_c,
                        f.nws_temp_c AS raw_nws_temp_c,
                        f.nws_dewpoint_c,
                        f.nws_rh_pct,
                        f.nws_wind_speed_ms,
                        f.nws_wind_gust_ms,
                        f.nws_wind_dir_deg,
                        f.nws_pop_pct,
                        f.nws_qpf_1hr_mm,
                        f.nws_sky_cover_pct,
                        f.short_forecast,
                        m.model_version,
                        m.pred_temp_error_c,
                        m.corrected_temp_c,
                        m.corrected_temp_p10_c,
                        m.corrected_temp_p90_c,
                        m.corrected_rh_pct,
                        m.corrected_wind_gust_ms
                    FROM nws_forecasts f
                    LEFT JOIN (
                        SELECT
                            valid_time_utc,
                            model_version,
                            pred_temp_error_c,
                            corrected_temp_c,
                            corrected_temp_p10_c,
                            corrected_temp_p90_c,
                            corrected_rh_pct,
                            corrected_wind_gust_ms
                        FROM mos_corrected_forecasts
                        WHERE issue_time_utc = (
                            SELECT issue_time_utc
                            FROM mos_corrected_forecasts
                            GROUP BY issue_time_utc
                            HAVING COUNT(*) >= 24
                            ORDER BY issue_time_utc DESC
                            LIMIT 1
                        )
                    ) m ON f.valid_time_utc = m.valid_time_utc
                    WHERE f.issue_time_utc = ?
                    ORDER BY f.valid_time_utc ASC
                    """,
                    conn,
                    params=(latest_nws_issue,),
                )
                future_fc_df = apply_mos_inference_if_needed(future_fc_df)
                if "corrected_rh_pct" in future_fc_df.columns:
                    future_fc_df["corrected_rh_pct"] = future_fc_df["corrected_rh_pct"].combine_first(
                        future_fc_df["nws_rh_pct"]
                    )
                if "corrected_wind_gust_ms" in future_fc_df.columns:
                    future_fc_df["corrected_wind_gust_ms"] = future_fc_df["corrected_wind_gust_ms"].combine_first(
                        future_fc_df["nws_wind_gust_ms"]
                    ).combine_first(future_fc_df["nws_wind_speed_ms"] * 1.35)
        except Exception:
            future_fc_df = pd.DataFrame()

        # 4. Paired Historical Verification Dataset (Direct SQL INNER JOIN)
        verify_df = pd.DataFrame()
        try:
            verify_df = pd.read_sql_query(
                """
                SELECT
                    o.valid_time_utc,
                    o.temp_c AS actual_temp_c,
                    o.dewpoint_c AS actual_dewpoint_c,
                    o.wetbulb_c AS actual_wetbulb_c,
                    o.rh_pct AS actual_rh_pct,
                    o.wind_speed_ms AS actual_wind_speed_ms,
                    o.wind_gust_ms AS actual_wind_gust_ms,
                    o.wind_dir_deg AS actual_wind_dir_deg,
                    o.mslp_hpa AS actual_mslp_hpa,
                    COALESCE(o.pressure_trend_3hr_hpa, 0.0) AS pressure_trend_3hr_hpa,
                    COALESCE(o.precip_1hr_mm, 0.0) AS actual_precip_mm,
                    f.issue_time_utc,
                    f.lead_time_hours,
                    f.nws_temp_c,
                    f.nws_temp_c AS raw_nws_temp_c,
                    f.nws_dewpoint_c,
                    f.nws_rh_pct,
                    f.nws_wind_speed_ms,
                    f.nws_wind_gust_ms,
                    f.nws_wind_dir_deg,
                    COALESCE(f.nws_pop_pct, 0.0) AS nws_pop_pct,
                    COALESCE(f.nws_qpf_1hr_mm, 0.0) AS nws_qpf_1hr_mm,
                    COALESCE(f.nws_sky_cover_pct, 20.0) AS nws_sky_cover_pct,
                    f.short_forecast,
                    COALESCE(r_chews.temp_c, f.nws_temp_c + 2.0) AS issue_chews_temp_c,
                    COALESCE(o_issue.temp_c, f.nws_temp_c) AS issue_tass_temp_c,
                    COALESCE(o_issue.temp_c - f_issue.nws_temp_c, 0.0) AS issue_persistence_bias_c,
                    m.corrected_temp_c,
                    m.corrected_temp_p10_c,
                    m.corrected_temp_p90_c,
                    m.corrected_rh_pct,
                    m.corrected_wind_gust_ms,
                    m.model_version
                FROM local_observations o
                INNER JOIN (
                    SELECT
                        valid_time_utc,
                        issue_time_utc,
                        lead_time_hours,
                        nws_temp_c,
                        nws_dewpoint_c,
                        nws_rh_pct,
                        nws_wind_speed_ms,
                        nws_wind_gust_ms,
                        nws_wind_dir_deg,
                        nws_pop_pct,
                        nws_qpf_1hr_mm,
                        nws_sky_cover_pct,
                        short_forecast,
                        ROW_NUMBER() OVER (
                            PARTITION BY valid_time_utc
                            ORDER BY
                                CASE WHEN short_forecast != 'Archived NWP Grid' THEN 0 ELSE 1 END ASC,
                                CASE WHEN lead_time_hours >= 1 THEN 0 ELSE 1 END ASC,
                                ABS(lead_time_hours - 6) ASC,
                                issue_time_utc DESC
                        ) AS rn
                    FROM nws_forecasts
                    WHERE nws_temp_c IS NOT NULL
                ) f ON o.valid_time_utc = f.valid_time_utc AND f.rn = 1
                LEFT JOIN local_observations o_issue
                    ON f.issue_time_utc = o_issue.valid_time_utc
                LEFT JOIN (
                    SELECT valid_time_utc, AVG(nws_temp_c) AS nws_temp_c
                    FROM nws_forecasts
                    GROUP BY valid_time_utc
                ) f_issue ON f.issue_time_utc = f_issue.valid_time_utc
                LEFT JOIN reference_observations r_chews
                    ON f.issue_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
                LEFT JOIN (
                    SELECT
                        valid_time_utc,
                        corrected_temp_c,
                        corrected_temp_p10_c,
                        corrected_temp_p90_c,
                        corrected_rh_pct,
                        corrected_wind_gust_ms,
                        model_version,
                        ROW_NUMBER() OVER (
                            PARTITION BY valid_time_utc
                            ORDER BY
                                CASE WHEN model_version = 'MOS_GBM_v1' THEN 0 ELSE 1 END ASC,
                                issue_time_utc DESC
                        ) AS rn
                    FROM mos_corrected_forecasts
                ) m ON o.valid_time_utc = m.valid_time_utc AND m.rn = 1
                WHERE o.temp_c IS NOT NULL AND o.qc_flag = 'OK'
                ORDER BY o.valid_time_utc ASC
                """,
                conn,
            )
            if not verify_df.empty:
                verify_df = apply_mos_inference_if_needed(verify_df)
        except Exception:
            verify_df = pd.DataFrame()

    return ref_dict, obs_df, future_fc_df, verify_df


# ---------------------------------------------------------------------------
# Header & Station Metadata Banner
# ---------------------------------------------------------------------------
ref_dict, obs_df, future_fc_df, verify_df = load_all_dashboard_data()

now_utc = datetime.now(timezone.utc)
solar_state, solar_el, solar_az, horizon_elev = compute_canyon_solar_state(now_utc)

active_model_name = "MOS_GBM_v1 (HistGradientBoosting Quantile Ensemble)" if MODEL_PATH.exists() else "TOPO_PHYSICS (Analytical Baseline)"

st.markdown(
    '<div class="met-header-bar">'
    '<h2 style="margin-bottom:0.2rem;">Tassajara Zen Mountain Center — Microclimate & MOS Forecast Suite</h2>'
    '</div>',
    unsafe_allow_html=True,
)

if not obs_df.empty and "valid_time_utc" in obs_df.columns:
    last_obs_dt = pd.to_datetime(obs_df.iloc[0]["valid_time_utc"], utc=True).tz_convert(LOCAL_TZ)
    last_obs_str = last_obs_dt.strftime("%Y-%m-%d %H:%M %Z")
else:
    last_obs_str = "Awaiting Telemetry"

if not future_fc_df.empty and "issue_time_utc" in future_fc_df.columns:
    fc_issue_dt = pd.to_datetime(future_fc_df.iloc[0]["issue_time_utc"], utc=True).tz_convert(LOCAL_TZ)
    fc_issue_str = fc_issue_dt.strftime("%Y-%m-%d %H:%M %Z")
else:
    fc_issue_str = "N/A"

st.markdown(
    f'<div class="met-subbanner">'
    f'<b>PRIMARY ISS:</b> Stone Office Davis Vantage Pro2 (ID 36966 | 36.2337°N, 121.5501°W | 485 m / 1,590 ft MSL) &nbsp;&bull;&nbsp; '
    f'<b>BOUNDARY REFS:</b> Chews Ridge CHWC1 (1,536 m) &amp; Hastings CAHC1 (546 m)<br>'
    f'<b>LAST OBSERVATION:</b> {last_obs_str} &nbsp;&bull;&nbsp; '
    f'<b>NWS/MOS INIT:</b> {fc_issue_str} &nbsp;&bull;&nbsp; '
    f'<b>SOLAR GEOMETRY:</b> {solar_state} (El: {solar_el:.1f}°, Az: {solar_az:.0f}°, Ridge Horizon: {horizon_elev:.0f}°) &nbsp;&bull;&nbsp; '
    f'<b>ENGINE:</b> {active_model_name}'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1: Live Meteorological & Boundary-Layer Telemetry (8-Card Grid)
# ---------------------------------------------------------------------------
if not obs_df.empty:
    latest = obs_df.iloc[0]
    last_24 = obs_df.head(24)

    temp_c = safe_float(latest.get("temp_c"), 15.0)
    temp_f = c_to_f(temp_c)
    t24_series = get_col(last_24, "temp_c", temp_c)
    hi_24_f = c_to_f(safe_float(t24_series.max(), temp_c))
    lo_24_f = c_to_f(safe_float(t24_series.min(), temp_c))

    rh_pct = safe_float(latest.get("rh_pct"), 50.0)
    dew_c = safe_float(latest.get("dewpoint_c"), temp_c - 4.0)
    dew_f = c_to_f(dew_c)

    wetbulb_c = latest.get("wetbulb_c")
    if wetbulb_c is None or pd.isna(wetbulb_c):
        wetbulb_c = compute_wetbulb_stull_c(temp_c, rh_pct)
    else:
        wetbulb_c = float(wetbulb_c)
    wetbulb_f = c_to_f(wetbulb_c)
    vpd_kpa = compute_vpd_kpa(temp_c, rh_pct)

    # Backfill wind/barometer from most recent valid reading if current sensor packet omitted them
    valid_wind_rows = last_24[last_24["wind_speed_ms"].notna()] if "wind_speed_ms" in last_24.columns else pd.DataFrame()
    wind_spd_ms = safe_float(latest.get("wind_speed_ms"), safe_float(valid_wind_rows.iloc[0]["wind_speed_ms"], 0.0) if not valid_wind_rows.empty else 0.0)
    wind_gst_ms = safe_float(latest.get("wind_gust_ms"), safe_float(valid_wind_rows.iloc[0]["wind_gust_ms"], 0.0) if not valid_wind_rows.empty else 0.0)
    wind_dir_val = latest.get("wind_dir_deg")
    if (wind_dir_val is None or pd.isna(wind_dir_val)) and not valid_wind_rows.empty:
        wind_dir_val = valid_wind_rows.iloc[0].get("wind_dir_deg")

    wind_spd_mph = ms_to_mph(wind_spd_ms)
    wind_gst_mph = ms_to_mph(wind_gst_ms)
    wind_card = deg_to_cardinal(wind_dir_val) if wind_spd_mph > 0.2 else "Calm"

    valid_bar_rows = last_24[last_24["mslp_hpa"].notna()] if "mslp_hpa" in last_24.columns else pd.DataFrame()
    mslp_hpa = safe_float(latest.get("mslp_hpa"), safe_float(valid_bar_rows.iloc[0]["mslp_hpa"], 1013.25) if not valid_bar_rows.empty else 1013.25)
    mslp_inhg = hpa_to_inhg(mslp_hpa)
    ptrend_hpa = safe_float(latest.get("pressure_trend_3hr_hpa"), 0.0)
    if ptrend_hpa > 0.5:
        ptrend_label = f"+{ptrend_hpa:.1f} hPa / 3h (Rising)"
    elif ptrend_hpa < -0.5:
        ptrend_label = f"{ptrend_hpa:.1f} hPa / 3h (Falling)"
    else:
        ptrend_label = f"{ptrend_hpa:+.1f} hPa / 3h (Steady)"

    rain_24_mm = float(get_col(last_24, "precip_1hr_mm", 0.0).sum())
    rain_24_in = mm_to_in(rain_24_mm)
    last_hr_rain = mm_to_in(safe_float(latest.get("precip_1hr_mm"), 0.0))

    # Boundary-layer vertical lapse rate & canyon confinement
    chews_temp_c = safe_float(ref_dict.get("CHWC1", {}).get("temp_c"), np.nan)
    hast_temp_c = safe_float(ref_dict.get("CAHC1", {}).get("temp_c"), np.nan)

    if not math.isnan(chews_temp_c):
        dz_km = (ELEV_CHEWS_M - ELEV_TASS_M) / 1000.0  # 1.051 km
        lapse_c_km = -1.0 * (chews_temp_c - temp_c) / dz_km
        lapse_str = f"{lapse_c_km:+.1f} °C/km"
        stability_state = (
            "Thermal Inversion (Cold Pool)"
            if lapse_c_km < 0.0
            else ("Isothermal / Stable" if lapse_c_km < 4.0 else "Well-Mixed Boundary Layer")
        )
    else:
        lapse_str = "N/A"
        stability_state = "Awaiting CHWC1"

    if not math.isnan(hast_temp_c):
        hast_f = c_to_f(hast_temp_c)
        canyon_delta_f = temp_f - hast_f
        confinement_str = f"{canyon_delta_f:+.1f} °F vs CAHC1"
        ref_sub_str = f"CHWC1: {c_to_f(chews_temp_c):.1f}°F | CAHC1: {hast_f:.1f}°F" if not math.isnan(chews_temp_c) else f"CAHC1: {hast_f:.1f}°F"
    else:
        confinement_str = "N/A"
        ref_sub_str = "Reference RAWS offline"

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric(
        "Dry-Bulb Air Temperature",
        f"{temp_f:.1f} °F ({temp_c:.1f} °C)",
        f"24h Extremes: {lo_24_f:.1f}°F / {hi_24_f:.1f}°F",
        delta_color="off",
    )
    r1c2.metric(
        "Relative Humidity & Dewpoint",
        f"{rh_pct:.0f} % RH",
        f"Dewpoint: {dew_f:.1f} °F ({dew_c:.1f} °C)",
        delta_color="off",
    )
    r1c3.metric(
        "Wet-Bulb & Vapor Pressure Deficit",
        f"{wetbulb_f:.1f} °F WB",
        f"VPD: {vpd_kpa:.2f} kPa ({'Arid' if vpd_kpa >= 1.6 else 'Moist/Moderate'})",
        delta_color="off",
    )
    r1c4.metric(
        "Anemometer Wind Vector",
        f"{wind_spd_mph:.1f} mph ({wind_card})",
        f"10m Peak Gust: {wind_gst_mph:.1f} mph",
        delta_color="off",
    )

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric(
        "Sea-Level Barometric Pressure",
        f"{mslp_inhg:.2f} inHg ({mslp_hpa:.1f} hPa)",
        ptrend_label,
        delta_color="off",
    )
    r2c2.metric(
        "Precipitation Accumulation (24h)",
        f"{rain_24_in:.2f} in ({rain_24_mm:.1f} mm)",
        f"1-Hr Rate: {last_hr_rain:.2f} in/hr",
        delta_color="off",
    )
    r2c3.metric(
        "Vertical Lapse Rate (CHWC1 → TZMC)",
        lapse_str,
        stability_state,
        delta_color="off",
    )
    r2c4.metric(
        "Canyon Thermal Confinement",
        confinement_str,
        ref_sub_str,
        delta_color="off",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Forecast Accuracy Verification Scorecard (MOS vs. NWS)
# ---------------------------------------------------------------------------
ver_hdr_col1, ver_hdr_col2 = st.columns([3, 1])
with ver_hdr_col1:
    st.subheader("Forecast Accuracy Verification Scorecard: Tassajara MOS vs. Raw NWP/NWS")
with ver_hdr_col2:
    eval_window = st.selectbox(
        "Verification Evaluation Window",
        options=["Past 24 Hours", "Past 72 Hours", "Past 7 Days (168h)", "Full Historical Archive"],
        index=1,
    )

if not verify_df.empty and len(verify_df) >= 3:
    window_map = {
        "Past 24 Hours": 24,
        "Past 72 Hours": 72,
        "Past 7 Days (168h)": 168,
        "Full Historical Archive": len(verify_df),
    }
    n_eval = window_map.get(eval_window, 72)
    v_sub = verify_df.tail(n_eval).copy()
    v_sub["valid_local"] = pd.to_datetime(v_sub["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    v_sub["actual_f"] = c_to_f(get_col(v_sub, "actual_temp_c", 15.0))
    v_sub["nws_f"] = c_to_f(get_col(v_sub, "raw_nws_temp_c", 15.0))
    v_sub["mos_f"] = c_to_f(get_col(v_sub, "corrected_temp_c", 15.0))
    v_sub["p10_f"] = c_to_f(get_col(v_sub, "corrected_temp_p10_c", get_col(v_sub, "corrected_temp_c", 15.0) - 2.1))
    v_sub["p90_f"] = c_to_f(get_col(v_sub, "corrected_temp_p90_c", get_col(v_sub, "corrected_temp_c", 15.0) + 2.1))

    v_sub["nws_err_f"] = v_sub["nws_f"] - v_sub["actual_f"]
    v_sub["mos_err_f"] = v_sub["mos_f"] - v_sub["actual_f"]
    v_sub["nws_abs_err_f"] = v_sub["nws_err_f"].abs()
    v_sub["mos_abs_err_f"] = v_sub["mos_err_f"].abs()

    nws_mae_f = float(v_sub["nws_abs_err_f"].mean())
    mos_mae_f = float(v_sub["mos_abs_err_f"].mean())
    nws_rmse_f = float(np.sqrt(np.mean(v_sub["nws_err_f"] ** 2)))
    mos_rmse_f = float(np.sqrt(np.mean(v_sub["mos_err_f"] ** 2)))
    nws_bias_f = float(v_sub["nws_err_f"].mean())
    mos_bias_f = float(v_sub["mos_err_f"].mean())

    skill_pct = (1.0 - (mos_rmse_f / max(nws_rmse_f, 0.01))) * 100.0
    hours_mos_won = int((v_sub["mos_abs_err_f"] <= v_sub["nws_abs_err_f"]).sum())
    total_eval_hrs = len(v_sub)
    win_rate_pct = (hours_mos_won / max(total_eval_hrs, 1)) * 100.0

    in_bounds = ((v_sub["actual_f"] >= v_sub["p10_f"]) & (v_sub["actual_f"] <= v_sub["p90_f"])).mean() * 100.0

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric(
        "Raw NWP/NWS Error (MAE)",
        f"± {nws_mae_f:.2f} °F",
        f"RMSE: ± {nws_rmse_f:.2f} °F | Bias: {nws_bias_f:+.2f} °F",
        delta_color="off",
    )
    s2.metric(
        "Tassajara MOS Error (MAE)",
        f"± {mos_mae_f:.2f} °F",
        f"{mos_mae_f - nws_mae_f:+.2f} °F MAE vs Raw NWP",
        delta_color="inverse",
    )
    s3.metric(
        "MOS RMSE Skill Score",
        f"{skill_pct:+.1f} %",
        f"MOS RMSE: ± {mos_rmse_f:.2f} °F (Bias: {mos_bias_f:+.2f} °F)",
        delta_color="normal" if skill_pct >= 0 else "inverse",
    )
    s4.metric(
        "Hourly Win Frequency",
        f"{win_rate_pct:.1f} % ({hours_mos_won}/{total_eval_hrs}h)",
        f"Evaluated over {total_eval_hrs} paired hours",
        delta_color="off",
    )
    s5.metric(
        "P10–P90 Interval Coverage",
        f"{in_bounds:.1f} %",
        f"Nominal 80% Quantile Envelope",
        delta_color="off",
    )

    with st.expander(f"Inspect Paired Hourly Verification Table & Residual Diagnostics ({total_eval_hrs} Hours)", expanded=False):
        ver_table = pd.DataFrame({
            "Valid Time (Local)": v_sub["valid_local"].dt.strftime("%Y-%m-%d %H:%M"),
            "Lead (hr)": pd.to_numeric(v_sub["lead_time_hours"], errors="coerce").fillna(6).astype(int),
            "Actual (°F)": v_sub["actual_f"].round(1),
            "MOS Forecast (°F)": v_sub["mos_f"].round(1),
            "MOS [P10, P90] (°F)": (
                "[" + v_sub["p10_f"].round(1).astype(str) + ", " + v_sub["p90_f"].round(1).astype(str) + "]"
            ),
            "Raw NWS/NWP (°F)": v_sub["nws_f"].round(1),
            "MOS Residual (°F)": v_sub["mos_err_f"].round(2),
            "NWS Residual (°F)": v_sub["nws_err_f"].round(2),
            "|MOS| Gain (°F)": (v_sub["nws_abs_err_f"] - v_sub["mos_abs_err_f"]).round(2),
            "Superior Model": np.where(v_sub["mos_abs_err_f"] <= v_sub["nws_abs_err_f"], "Tassajara MOS", "Raw NWS"),
        }).iloc[::-1].reset_index(drop=True)
        st.dataframe(ver_table, use_container_width=True, hide_index=True)
else:
    st.warning(
        "No paired observation-forecast records found in tassajara_weather.db. "
        "Trigger the GitHub Actions workflow (`hourly_weather.yml`) with backfill enabled."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: 3-Panel Interactive Plotly Microclimate & Forecast Suite
# ---------------------------------------------------------------------------
ctrl_col1, ctrl_col2 = st.columns([3, 1])
with ctrl_col1:
    st.subheader("Microclimate Time-Series: Verification Hindcast & Operational Forecast")
with ctrl_col2:
    horizon_choice = st.selectbox(
        "Timeline Horizon",
        options=["72h Forecast + Past 48h Hindcast", "7-Day Forecast + Past 72h Hindcast"],
        index=0,
    )

if not future_fc_df.empty or not verify_df.empty:
    is_short_horizon = "72h" in horizon_choice
    max_future_hrs = 72 if is_short_horizon else 168
    past_hrs = 48 if is_short_horizon else 72

    fut_sub = future_fc_df.head(max_future_hrs).copy() if not future_fc_df.empty else pd.DataFrame()
    past_sub = verify_df.tail(past_hrs).copy() if not verify_df.empty else pd.DataFrame()

    if not past_sub.empty and not fut_sub.empty:
        past_sub = past_sub[~past_sub["valid_time_utc"].isin(fut_sub["valid_time_utc"])]
        combined_df = pd.concat([past_sub, fut_sub], ignore_index=True, sort=False)
    elif not fut_sub.empty:
        combined_df = fut_sub.copy()
    else:
        combined_df = past_sub.copy()

    combined_df = combined_df.sort_values("valid_time_utc").reset_index(drop=True)
    combined_df["valid_local"] = pd.to_datetime(combined_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    # Overlay any latest actuals from obs_df onto combined_df
    if not obs_df.empty and "valid_time_utc" in obs_df.columns:
        obs_dedup = obs_df.drop_duplicates(subset=["valid_time_utc"]).set_index("valid_time_utc")
        for col_dest, col_src in [
            ("actual_temp_c", "temp_c"),
            ("actual_dewpoint_c", "dewpoint_c"),
            ("actual_precip_mm", "precip_1hr_mm"),
            ("actual_rh_pct", "rh_pct"),
            ("actual_wind_gust_ms", "wind_gust_ms"),
        ]:
            if col_dest not in combined_df.columns:
                combined_df[col_dest] = np.nan
            if col_src in obs_dedup.columns:
                mapped = combined_df["valid_time_utc"].map(obs_dedup[col_src])
                combined_df[col_dest] = combined_df[col_dest].combine_first(mapped)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        row_heights=[0.50, 0.25, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Temperature & Dewpoint Profile (°F): Stone Office Actuals vs. MOS Quantile Fan (P10–P90) vs. Raw NWS",
            "Boundary-Layer Kinematics & Moisture: 10m Peak Wind Gusts (mph) & Relative Humidity (%)",
            "Hydrological Outlook: Hourly Precipitation Rate (in/hr) & Probability of Precipitation (%)",
        ),
    )

    # Panel 1: Temperature Fan Chart
    corr_c_series = get_col(combined_df, "corrected_temp_c", 15.0)
    p90_f = c_to_f(get_col(combined_df, "corrected_temp_p90_c", corr_c_series + 2.1))
    p10_f = c_to_f(get_col(combined_df, "corrected_temp_p10_c", corr_c_series - 2.1))
    raw_nws_f = c_to_f(get_col(combined_df, "raw_nws_temp_c", corr_c_series))
    dew_f_series = c_to_f(
        get_col(combined_df, "actual_dewpoint_c", np.nan).combine_first(
            get_col(combined_df, "nws_dewpoint_c", corr_c_series - 5.0)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pd.concat([combined_df["valid_local"], combined_df["valid_local"][::-1]]),
            y=pd.concat([p90_f, p10_f[::-1]]),
            fill="toself",
            fillcolor="rgba(249, 115, 22, 0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name="MOS P10–P90 Confidence Band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=dew_f_series.round(1),
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.45)", width=1.2, dash="dot"),
            name="Dewpoint Temperature (°F)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=raw_nws_f.round(1),
            mode="lines",
            line=dict(color="#60a5fa", width=2.0, dash="dash"),
            name="Raw NWS Grid Forecast (°F)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=c_to_f(corr_c_series).round(1),
            mode="lines",
            line=dict(color="#f97316", width=2.8),
            name="Tassajara MOS Corrected (°F)",
        ),
        row=1,
        col=1,
    )

    if "actual_temp_c" in combined_df.columns:
        actual_pts = combined_df[combined_df["actual_temp_c"].notna()]
        if not actual_pts.empty:
            act_y = c_to_f(pd.to_numeric(actual_pts["actual_temp_c"], errors="coerce")).round(1)
            fig.add_trace(
                go.Scatter(
                    x=actual_pts["valid_local"],
                    y=act_y,
                    mode="lines+markers",
                    line=dict(color="#10b981", width=2.2),
                    marker=dict(color="#10b981", size=5.5, symbol="circle"),
                    name="Stone Office Observed (°F)",
                ),
                row=1,
                col=1,
            )

    fig.add_hline(
        y=32.0,
        line_dash="dot",
        line_color="#38bdf8",
        annotation_text="Freeze Threshold (32.0 °F)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Panel 2: Wind Gusts & Relative Humidity
    gust_series_ms = (
        get_col(combined_df, "actual_wind_gust_ms", np.nan)
        .combine_first(get_col(combined_df, "corrected_wind_gust_ms", np.nan))
        .combine_first(get_col(combined_df, "nws_wind_gust_ms", np.nan))
        .combine_first(get_col(combined_df, "nws_wind_speed_ms", 0.0) * 1.35)
        .fillna(0.0)
    )
    gust_mph_series = gust_series_ms * 2.23694

    rh_series = (
        get_col(combined_df, "actual_rh_pct", np.nan)
        .combine_first(get_col(combined_df, "corrected_rh_pct", np.nan))
        .combine_first(get_col(combined_df, "nws_rh_pct", 50.0))
        .fillna(50.0)
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=gust_mph_series.round(1),
            mode="lines",
            line=dict(color="#eab308", width=2.0),
            name="Peak Wind Gust (mph)",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=rh_series.round(0),
            mode="lines",
            line=dict(color="#a855f7", width=1.8, dash="dot"),
            name="Relative Humidity (%)",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Panel 3: Precipitation & PoP%
    nws_rain_in = (get_col(combined_df, "nws_qpf_1hr_mm", 0.0) / 25.4).round(3)
    actual_rain_in = (get_col(combined_df, "actual_precip_mm", 0.0) / 25.4).round(3)
    pop_pct = get_col(combined_df, "nws_pop_pct", 0.0).clip(0.0, 100.0)

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=pop_pct.round(0),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="rgba(56, 189, 248, 0.70)", width=1.5),
            name="Precipitation Probability (%)",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=combined_df["valid_local"],
            y=nws_rain_in,
            name="Forecast QPF (in/hr)",
            marker_color="#38bdf8",
            opacity=0.8,
        ),
        row=3,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=combined_df["valid_local"],
            y=actual_rain_in,
            name="Observed Tipping Bucket (in/hr)",
            marker_color="#10b981",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )

    # Vertical initialization boundary marker
    if not obs_df.empty:
        divider_ts = pd.to_datetime(obs_df.iloc[0]["valid_time_utc"], utc=True).tz_convert(LOCAL_TZ)
        for r_idx in (1, 2, 3):
            fig.add_vline(
                x=divider_ts,
                line_width=1.4,
                line_dash="dash",
                line_color="rgba(226, 232, 240, 0.55)",
                row=r_idx,
                col=1,
            )

    max_rain_val = float(max(nws_rain_in.max(), actual_rain_in.max(), 0.0))
    rain_y_top = max(0.20, round(max_rain_val * 1.25, 2))
    max_gust_val = float(max(gust_mph_series.max(), 15.0))

    fig.update_yaxes(title_text="Temperature (°F)", gridcolor="rgba(148,163,184,0.12)", row=1, col=1)
    fig.update_yaxes(
        title_text="Wind Gust (mph)",
        range=[0, max_gust_val * 1.25],
        gridcolor="rgba(148,163,184,0.12)",
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(title_text="RH (%)", range=[0, 105], showgrid=False, row=2, col=1, secondary_y=True)
    fig.update_yaxes(
        title_text="Rain Rate (in/hr)",
        range=[0, rain_y_top],
        gridcolor="rgba(148,163,184,0.12)",
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(title_text="PoP (%)", range=[0, 105], showgrid=False, row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local Time (America/Los_Angeles)", gridcolor="rgba(148,163,184,0.12)", row=3, col=1)

    fig.update_layout(
        height=840,
        margin=dict(t=60, b=40, l=55, r=55),
        hovermode="x unified",
        barmode="group",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: 7-Day Meteorological Planning Summary Table
# ---------------------------------------------------------------------------
if not future_fc_df.empty:
    st.subheader("7-Day Synoptic & Microclimate Planning Outlook")
    daily_df = future_fc_df.copy()
    daily_df["valid_local"] = pd.to_datetime(daily_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    daily_df["Date_Key"] = daily_df["valid_local"].dt.strftime("%a, %b %d")
    daily_df["Sort_Date"] = daily_df["valid_local"].dt.date

    daily_df["_corr_t"] = get_col(daily_df, "corrected_temp_c", 15.0)
    daily_df["_p10_t"] = get_col(daily_df, "corrected_temp_p10_c", daily_df["_corr_t"] - 2.1)
    daily_df["_p90_t"] = get_col(daily_df, "corrected_temp_p90_c", daily_df["_corr_t"] + 2.1)
    daily_df["_nws_t"] = get_col(daily_df, "raw_nws_temp_c", daily_df["_corr_t"])
    daily_df["_pop"] = get_col(daily_df, "nws_pop_pct", 0.0)
    daily_df["_qpf"] = get_col(daily_df, "nws_qpf_1hr_mm", 0.0)
    daily_df["_rh"] = get_col(daily_df, "corrected_rh_pct", np.nan).combine_first(
        get_col(daily_df, "nws_rh_pct", 45.0)
    )
    daily_df["_gust"] = (
        get_col(daily_df, "corrected_wind_gust_ms", np.nan)
        .combine_first(get_col(daily_df, "nws_wind_gust_ms", np.nan))
        .combine_first(get_col(daily_df, "nws_wind_speed_ms", 0.0) * 1.35)
    )

    daily_summary = (
        daily_df.groupby(["Sort_Date", "Date_Key"], as_index=False)
        .agg(
            MOS_High_F=("_corr_t", lambda s: round(c_to_f(float(s.max())), 1)),
            MOS_Low_F=("_corr_t", lambda s: round(c_to_f(float(s.min())), 1)),
            Cold_Pool_Floor_F=("_p10_t", lambda s: round(c_to_f(float(s.min())), 1)),
            Thermal_Peak_P90_F=("_p90_t", lambda s: round(c_to_f(float(s.max())), 1)),
            NWS_High_F=("_nws_t", lambda s: round(c_to_f(float(s.max())), 1)),
            NWS_Low_F=("_nws_t", lambda s: round(c_to_f(float(s.min())), 1)),
            Max_PoP_Pct=("_pop", lambda s: int(round(safe_float(s.max(), 0.0), 0))),
            Total_Precip_In=("_qpf", lambda s: round(mm_to_in(float(s.sum())), 2)),
            Min_RH_Pct=("_rh", lambda s: int(round(safe_float(s.min(), 40.0), 0))),
            Peak_Gust_MPH=("_gust", lambda s: round(ms_to_mph(safe_float(s.max(), 0.0)), 1)),
            Hours_Count=("_corr_t", "count"),
        )
        .sort_values("Sort_Date")
    )

    def _met_regime(row):
        notes = []
        if row["Cold_Pool_Floor_F"] <= 32.0:
            notes.append("Hard Freeze Risk (P10 ≤ 32°F)")
        elif row["Cold_Pool_Floor_F"] <= 36.0:
            notes.append("Nocturnal Cold-Pool Frost Risk")
        if row["Total_Precip_In"] >= 0.25 or row["Max_PoP_Pct"] >= 50:
            notes.append("Frontal Precipitation Likely")
        elif row["Max_PoP_Pct"] >= 25:
            notes.append("Slight Chance Light Rain")
        if row["Min_RH_Pct"] <= 20 and row["Peak_Gust_MPH"] >= 20:
            notes.append("Elevated Fire-Weather Subsidence")
        if not notes:
            notes.append("Diurnal Katabatic / Anabatic Valley Regime")
        return " | ".join(notes)

    daily_summary["Canyon Meteorological Regime"] = daily_summary.apply(_met_regime, axis=1)

    display_daily = daily_summary[
        [
            "Date_Key",
            "MOS_High_F",
            "MOS_Low_F",
            "Cold_Pool_Floor_F",
            "Thermal_Peak_P90_F",
            "NWS_High_F",
            "NWS_Low_F",
            "Max_PoP_Pct",
            "Total_Precip_In",
            "Min_RH_Pct",
            "Peak_Gust_MPH",
            "Canyon Meteorological Regime",
        ]
    ].rename(
        columns={
            "Date_Key": "Date",
            "MOS_High_F": "MOS High (°F)",
            "MOS_Low_F": "MOS Low (°F)",
            "Cold_Pool_Floor_F": "P10 Floor (°F)",
            "Thermal_Peak_P90_F": "P90 Ceiling (°F)",
            "NWS_High_F": "Raw NWS High (°F)",
            "NWS_Low_F": "Raw NWS Low (°F)",
            "Max_PoP_Pct": "Max PoP (%)",
            "Total_Precip_In": "QPF Total (in)",
            "Min_RH_Pct": "Min RH (%)",
            "Peak_Gust_MPH": "Peak Gust (mph)",
        }
    )

    st.dataframe(display_daily, use_container_width=True, hide_index=True)
