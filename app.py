#!/usr/bin/env python3
"""
app.py — Tassajara Zen Mountain Center Meteorological & Hydrological MOS Forecast Monitor
Publication-grade scientific monitoring suite for Station 36966 (Stone Office Vantage Pro2),
Chews Ridge (CHWC1) & Hastings (CAHC1) boundary references, and Thermal + Hydrological MOS verification.
Includes comprehensive scientific tooltips (help="...") across all metrics, controls, and tables.
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

from mos_model import engineer_mos_features, analytical_hydro_hurdle_fallback

st.set_page_config(
    page_title="TZMC Atmospheric & Hydrological MOS Monitor",
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
    """Topographic radiation-drainage + Orographic-Virga physical MOS fallback."""
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

    X_feats = engineer_mos_features(out)
    hydro_fb = analytical_hydro_hurdle_fallback(X_feats)
    out["raw_nws_pop_pct"] = get_col(out, "nws_pop_pct", 0.0).round(1)
    out["mos_pop_pct"] = hydro_fb["mos_pop_pct"].values
    out["raw_nws_qpf_mm"] = get_col(out, "nws_qpf_1hr_mm", 0.0).round(2)
    out["mos_qpf_mm"] = hydro_fb["mos_qpf_mm"].values
    out["mos_qpf_p90_mm"] = hydro_fb["mos_qpf_p90_mm"].values
    return out


def apply_mos_inference_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every row in `df` has MOS corrected_temp_c, p10, p90, mos_pop_pct,
    mos_qpf_mm, and mos_qpf_p90_mm.
    """
    if df.empty:
        return df
    out = df.copy()
    for col in [
        "corrected_temp_c",
        "corrected_temp_p10_c",
        "corrected_temp_p90_c",
        "mos_pop_pct",
        "mos_qpf_mm",
        "mos_qpf_p90_mm",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    missing_mask = out["corrected_temp_c"].isna() | out["mos_qpf_mm"].isna() | out["mos_pop_pct"].isna()
    if not missing_mask.any():
        return out

    sub = out.loc[missing_mask].copy()
    if "nws_temp_c" not in sub.columns and "raw_nws_temp_c" in sub.columns:
        sub["nws_temp_c"] = sub["raw_nws_temp_c"]
    if "raw_nws_temp_c" not in sub.columns and "nws_temp_c" in sub.columns:
        sub["raw_nws_temp_c"] = sub["nws_temp_c"]

    sub["lead_time_hours"] = get_col(sub, "lead_time_hours", 6.0)
    sub["nws_rh_pct"] = get_col(sub, "nws_rh_pct", 50.0)
    sub["nws_wind_speed_ms"] = get_col(sub, "nws_wind_speed_ms", 1.5)
    sub["nws_wind_dir_deg"] = get_col(sub, "nws_wind_dir_deg", 295.0)
    sub["nws_pop_pct"] = get_col(sub, "nws_pop_pct", 0.0)
    sub["nws_qpf_1hr_mm"] = get_col(sub, "nws_qpf_1hr_mm", 0.0)
    sub["nws_sky_cover_pct"] = get_col(sub, "nws_sky_cover_pct", 20.0)
    sub["pressure_trend_3hr_hpa"] = get_col(sub, "pressure_trend_3hr_hpa", 0.0)
    sub["issue_chews_temp_c"] = get_col(sub, "issue_chews_temp_c", sub["nws_temp_c"] + 2.0)
    sub["issue_chews_precip_mm"] = get_col(sub, "issue_chews_precip_mm", 0.0)
    sub["issue_tass_temp_c"] = get_col(sub, "issue_tass_temp_c", sub["nws_temp_c"])
    sub["issue_persistence_bias_c"] = get_col(sub, "issue_persistence_bias_c", 0.0)
    if "issue_time_utc" not in sub.columns:
        sub["issue_time_utc"] = sub["valid_time_utc"]

    if MODEL_PATH.exists():
        try:
            mos_model = joblib.load(MODEL_PATH)
            preds = mos_model.predict(sub)
            for c in [
                "pred_temp_error_c",
                "corrected_temp_c",
                "corrected_temp_p10_c",
                "corrected_temp_p90_c",
                "raw_nws_pop_pct",
                "mos_pop_pct",
                "raw_nws_qpf_mm",
                "mos_qpf_mm",
                "mos_qpf_p90_mm",
            ]:
                if c in preds.columns:
                    out.loc[missing_mask, c] = preds[c].values
                elif c not in out.columns or out.loc[missing_mask, c].isna().any():
                    fb = fallback_physics_mos(sub)
                    out.loc[missing_mask, c] = fb[c].values
            return out
        except Exception:
            pass

    fb = fallback_physics_mos(sub)
    for c in [
        "pred_temp_error_c",
        "corrected_temp_c",
        "corrected_temp_p10_c",
        "corrected_temp_p90_c",
        "raw_nws_pop_pct",
        "mos_pop_pct",
        "raw_nws_qpf_mm",
        "mos_qpf_mm",
        "mos_qpf_p90_mm",
    ]:
        out.loc[missing_mask, c] = fb[c].values
    return out


# ---------------------------------------------------------------------------
# Data Loading & SQL Alignment
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120)
def load_all_dashboard_data():
    if not Path(DB_PATH).exists():
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

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

        # Archive-wide Precipitation Stats (Water Year / Total Archive)
        archive_stats = {"total_obs_hours": 0, "wet_hours": 0, "total_rain_mm": 0.0, "max_1hr_mm": 0.0}
        try:
            row_st = conn.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN COALESCE(precip_1hr_mm, 0.0) >= 0.254 THEN 1 ELSE 0 END),
                    COALESCE(SUM(precip_1hr_mm), 0.0),
                    COALESCE(MAX(precip_1hr_mm), 0.0)
                FROM local_observations
                WHERE temp_c IS NOT NULL
                """
            ).fetchone()
            if row_st:
                archive_stats = {
                    "total_obs_hours": int(row_st[0] or 0),
                    "wet_hours": int(row_st[1] or 0),
                    "total_rain_mm": float(row_st[2] or 0.0),
                    "max_1hr_mm": float(row_st[3] or 0.0),
                }
        except Exception:
            pass

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

        # Inspect mos_corrected_forecasts columns for backward-compatible SQL queries
        mos_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(mos_corrected_forecasts)").fetchall()
        }
        has_hydro_cols = "mos_qpf_mm" in mos_cols and "mos_pop_pct" in mos_cols
        hydro_select_sql = (
            "m.mos_pop_pct, m.mos_qpf_mm, m.mos_qpf_p90_mm"
            if has_hydro_cols
            else "NULL AS mos_pop_pct, NULL AS mos_qpf_mm, NULL AS mos_qpf_p90_mm"
        )
        hydro_inner_sql = (
            "mos_pop_pct, mos_qpf_mm, mos_qpf_p90_mm"
            if has_hydro_cols
            else "NULL AS mos_pop_pct, NULL AS mos_qpf_mm, NULL AS mos_qpf_p90_mm"
        )

        # 3. Future 7-Day Forecast Series
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
                    f"""
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
                        m.corrected_wind_gust_ms,
                        {hydro_select_sql}
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
                            corrected_wind_gust_ms,
                            {hydro_inner_sql}
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

        # 4. Paired Historical Verification Dataset
        verify_df = pd.DataFrame()
        try:
            verify_df = pd.read_sql_query(
                f"""
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
                    COALESCE(r_chews.precip_1hr_mm, 0.0) AS issue_chews_precip_mm,
                    COALESCE(o_issue.temp_c, f.nws_temp_c) AS issue_tass_temp_c,
                    COALESCE(o_issue.temp_c - f_issue.nws_temp_c, 0.0) AS issue_persistence_bias_c,
                    m.corrected_temp_c,
                    m.corrected_temp_p10_c,
                    m.corrected_temp_p90_c,
                    m.corrected_rh_pct,
                    m.corrected_wind_gust_ms,
                    {hydro_select_sql},
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
                        {hydro_inner_sql},
                        model_version,
                        ROW_NUMBER() OVER (
                            PARTITION BY valid_time_utc
                            ORDER BY
                                CASE WHEN model_version LIKE 'MOS_GBM%' THEN 0 ELSE 1 END ASC,
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

    return ref_dict, obs_df, future_fc_df, verify_df, archive_stats


# ---------------------------------------------------------------------------
# Header & Station Metadata Banner
# ---------------------------------------------------------------------------
ref_dict, obs_df, future_fc_df, verify_df, archive_stats = load_all_dashboard_data()

now_utc = datetime.now(timezone.utc)
solar_state, solar_el, solar_az, horizon_elev = compute_canyon_solar_state(now_utc)

active_model_name = (
    "MOS_GBM_v2_HYDRO (Thermal Quantile + Two-Stage Hurdle PoP/QPF)"
    if MODEL_PATH.exists()
    else "TOPO_HYDRO_PHYSICS (Analytical Baseline)"
)

st.markdown(
    '<div class="met-header-bar">'
    '<h2 style="margin-bottom:0.2rem;">Tassajara Zen Mountain Center — Microclimate & Hydrological MOS Forecast Suite</h2>'
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

    valid_wind_rows = last_24[last_24["wind_speed_ms"].notna()] if "wind_speed_ms" in last_24.columns else pd.DataFrame()
    wind_spd_ms = safe_float(
        latest.get("wind_speed_ms"),
        safe_float(valid_wind_rows.iloc[0]["wind_speed_ms"], 0.0) if not valid_wind_rows.empty else 0.0,
    )
    wind_gst_ms = safe_float(
        latest.get("wind_gust_ms"),
        safe_float(valid_wind_rows.iloc[0]["wind_gust_ms"], 0.0) if not valid_wind_rows.empty else 0.0,
    )
    wind_dir_val = latest.get("wind_dir_deg")
    if (wind_dir_val is None or pd.isna(wind_dir_val)) and not valid_wind_rows.empty:
        wind_dir_val = valid_wind_rows.iloc[0].get("wind_dir_deg")

    wind_spd_mph = ms_to_mph(wind_spd_ms)
    wind_gst_mph = ms_to_mph(wind_gst_ms)
    wind_card = deg_to_cardinal(wind_dir_val) if wind_spd_mph > 0.2 else "Calm"

    valid_bar_rows = last_24[last_24["mslp_hpa"].notna()] if "mslp_hpa" in last_24.columns else pd.DataFrame()
    mslp_hpa = safe_float(
        latest.get("mslp_hpa"),
        safe_float(valid_bar_rows.iloc[0]["mslp_hpa"], 1013.25) if not valid_bar_rows.empty else 1013.25,
    )
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

    # Next 72h MOS Calibrated QPF & PoP Summary
    if not future_fc_df.empty:
        next_72 = future_fc_df.head(72)
        mos_qpf_72_mm = float(get_col(next_72, "mos_qpf_mm", get_col(next_72, "nws_qpf_1hr_mm", 0.0)).sum())
        mos_qpf_72_in = mm_to_in(mos_qpf_72_mm)
        max_mos_pop_72 = float(get_col(next_72, "mos_pop_pct", get_col(next_72, "nws_pop_pct", 0.0)).max())
        rain_sub_str = f"1h: {last_hr_rain:.2f}\" | 72h MOS QPF: {mos_qpf_72_in:.2f}\" (PoP {max_mos_pop_72:.0f}%)"
    else:
        rain_sub_str = f"1-Hr Rate: {last_hr_rain:.2f} in/hr | Archive: {mm_to_in(archive_stats['total_rain_mm']):.2f} in"

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
        ref_sub_str = (
            f"CHWC1: {c_to_f(chews_temp_c):.1f}°F | CAHC1: {hast_f:.1f}°F"
            if not math.isnan(chews_temp_c)
            else f"CAHC1: {hast_f:.1f}°F"
        )
    else:
        confinement_str = "N/A"
        ref_sub_str = "Reference RAWS offline"

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric(
        "Dry-Bulb Air Temperature",
        f"{temp_f:.1f} °F ({temp_c:.1f} °C)",
        f"24h Extremes: {lo_24_f:.1f}°F / {hi_24_f:.1f}°F",
        delta_color="off",
        help=(
            "Ambient dry-bulb air temperature recorded by the radiation-shielded thermistor on the "
            "Stone Office Davis Vantage Pro2 ISS (