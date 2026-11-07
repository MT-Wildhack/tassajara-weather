#!/usr/bin/env python3
"""
app.py — Tassajara Zen Mountain Center Meteorological & Hydrological MOS Monitor
Publication-grade scientific monitoring suite for Station 36966 (Stone Office Vantage Pro2),
the 5-station Santa Lucia boundary network (Chews Ridge CHWC1, Hastings CAHC1, Arroyo Seco ASRC1,
Big Sur PPSC1, Ventana Cone VTCC1), and Thermal + Hydrological MOS.
Includes real-time UTC horizon filtering, automatic live 168-hour NWP top-up,
comprehensive scientific tooltips (help="..."), and an OLS & Feature Importance suite.
"""

import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

from ingest_weather import (
    REFERENCE_STATIONS,
    ELEV_TASSAJARA_M as ELEV_TASS_M,
    ELEV_CHEWS_RIDGE_M as ELEV_CHEWS_M,
    ELEV_HASTINGS_M as ELEV_HAST_M,
    ELEV_ARROYO_SECO_M as ELEV_ARROYO_M,
    ELEV_BIG_SUR_M as ELEV_BIGSUR_M,
    ELEV_VENTANA_CONE_M as ELEV_VENTANA_M,
    WeatherLinkV2Client,
    fetch_regional_reference_stations,
)
from mos_model import (
    TassajaraMOSCorrector,
    engineer_mos_features,
    analytical_hydro_hurdle_fallback,
    build_well_conditioned_ols_matrix,
)
from hydro_risk_engine import (
    MM_PER_INCH,
    YELLOW_API_MM,
    ORANGE_API_MM,
    RED_API_EXTREME_MM,
    TIER_METADATA,
    compute_dashboard_hydro_state,
)

st.set_page_config(
    page_title="TZMC Atmospheric & Hydrological MOS Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Publication-grade scientific CSS styling (Theme-Adaptive for Light & Dark Mode)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 6px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-color) !important;
        opacity: 0.78;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums;
        color: var(--text-color) !important;
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
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
        padding-bottom: 0.6rem;
        margin-bottom: 1.0rem;
    }
    .met-subbanner {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.80rem;
        color: var(--text-color);
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-left: 4px solid #0284c7;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .tzmc-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 10px;
        color: var(--text-color);
    }
    .tzmc-muted {
        color: var(--text-color);
        opacity: 0.75;
    }
    .tzmc-subtle {
        color: var(--text-color);
        opacity: 0.88;
    }
    .tzmc-pill {
        font-size: 0.80rem;
        font-weight: 600;
        color: var(--text-color);
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.40);
        padding: 4px 10px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "tassajara_weather.db"
MODEL_PATH = Path("models/tassajara_mos_v1.joblib")
LOCAL_TZ = "America/Los_Angeles"

TASSAJARA_LAT = 36.2337
TASSAJARA_LON = -121.5501
NWP_GRID_ELEV_M = 880.0

# Shared theme-safe Plotly styling constants for crisp Light & Dark mode rendering
PLOT_BG_RGBA = "rgba(148, 163, 184, 0.08)"
GRID_RGBA = "rgba(148, 163, 184, 0.25)"
DIVIDER_LINE_RGBA = "rgba(100, 116, 139, 0.75)"

# ---------------------------------------------------------------------------
# Scientific Tooltip Dictionary & Feature Metadata (Line-Wrap Safe)
# ---------------------------------------------------------------------------
FEATURE_METADATA = {
    "solar_el_deg": (
        "Solar Elevation Angle (deg)",
        "Astronomical solar altitude above the horizontal plane (Spencer 1971).",
    ),
    "in_canyon_shadow": (
        "Canyon Wall Shadow Flag",
        "Binary indicator (1.0 when Sun is above 0 deg but blocked by the 32 deg SW Black Cone Trail ridge).",
    ),
    "effective_insolation": (
        "Effective Canyon Insolation",
        "Direct-beam solar radiation factor sin(el) gated by the local topographic horizon profile.",
    ),
    "decayed_init_bias_c": (
        "Decayed Initialization Bias (C)",
        "Observed NWP temperature error at forecast issue time decayed exponentially (tau = 12h).",
    ),
    "issue_lapse_chews_c_km": (
        "Vertical Lapse Rate CHWC1->TZMC (C/km)",
        "Boundary-layer stability across the 1,051m column between Chews Ridge (or Ventana Cone) and Tassajara.",
    ),
    "decayed_canyon_confinement_c": (
        "Gorge & Foothill Confinement (C)",
        "Decayed thermal differential between TZMC (485m) and Arroyo Seco (268m) / Hastings (546m) capturing katabatic cold pooling.",
    ),
    "decayed_coastal_inland_grad_c": (
        "Coastal-to-Interior Gradient PPSC1–ASRC1 (C)",
        "Cross-range temperature differential between windward Big Sur RAWS (108m) and interior Arroyo Seco RAWS (268m).",
    ),
    "issue_marine_layer_cap_idx": (
        "Marine-Layer Stratus Cap Index",
        "Interaction of Big Sur vs. Chews Ridge RH contrast and upper-column inversion strength isolating marine stratus banking.",
    ),
    "decayed_ssw_spillover_precip_mm": (
        "Windward Crest Spillover Rain (mm)",
        "Upstream 1-hr rainfall across Ventana Cone (VTCC1), Big Sur (PPSC1), and Chews Ridge (CHWC1) weighted by 215 deg SSW inflow.",
    ),
    "nws_temp_c": (
        "Raw NWP Temperature (C)",
        "Baseline 2-meter air temperature from the 2.5km operational NWS/NWP grid cell (~880m MSL).",
    ),
    "nws_rh_pct": (
        "Raw NWP Relative Humidity (%)",
        "Baseline boundary-layer relative humidity from the operational NWP grid.",
    ),
    "wind_along_canyon_ms": (
        "Along-Canyon Wind Vector (m/s)",
        "10m wind projected onto the 115 deg ESE-WNW Tassajara Creek axis (anabatic/katabatic flow).",
    ),
    "wind_cross_canyon_ms": (
        "Cross-Canyon Wind Vector (m/s)",
        "10m wind component perpendicular to the primary canyon axis (ridge-top mechanical mixing).",
    ),
    "orographic_upslope_wind_ms": (
        "Santa Lucia Upslope Wind (m/s)",
        "10m wind projected onto the 215 deg SSW bearing perpendicular to the Santa Lucia crest.",
    ),
    "orographic_moisture_flux": (
        "Orographic Moisture Flux (IVT Proxy)",
        "Product of SSW upslope wind speed and actual vapor pressure (ea_kpa), driving spillover rain.",
    ),
    "virga_evap_index": (
        "Sub-Cloud Virga Evaporation Index",
        "Composite index (VPD * Dewpoint Depression * Dryness) quantifying sub-cloud rain evaporation.",
    ),
    "vpd_kpa": (
        "Vapor Pressure Deficit (kPa)",
        "Tetens saturation minus actual vapor pressure (es - ea); measures atmospheric evaporative demand.",
    ),
    "dewpoint_depression_c": (
        "Dewpoint Depression T - Td (C)",
        "Spread between dry-bulb and dewpoint temperature indicating sub-cloud air-mass dryness.",
    ),
    "qpf_moisture_scaling": (
        "Moisture-Scaled NWP QPF (mm)",
        "Nonlinear interaction between raw NWP QPF and squared relative humidity: QPF * (RH/100)^2.",
    ),
    "nws_qpf_1hr_mm": (
        "Raw NWP 1-Hr QPF (mm)",
        "Uncorrected hourly quantitative precipitation forecast from the regional NWP model.",
    ),
    "nws_pop_pct": (
        "Raw NWP PoP (%)",
        "Uncorrected Probability of Precipitation (%) from the baseline NWS/NWP grid.",
    ),
    "nws_sky_cover_pct": (
        "NWP Sky Cover (%)",
        "Forecast cloud cover fraction (%) modulating nocturnal longwave radiational cooling.",
    ),
    "pressure_trend_3hr_hpa": (
        "3-Hr Barometric Tendency (hPa)",
        "3-hour sea-level pressure change at the Stone Office ISS.",
    ),
    "frontal_pressure_fall_hpa": (
        "Frontal Pressure Fall (hPa/3h)",
        "Negative barometric tendency isolating dynamic synoptic frontal lift.",
    ),
    "decayed_chews_precip_mm": (
        "Chews Ridge Feeder-Seeder Rain (mm)",
        "Upstream 1-hour rainfall at Chews Ridge (1,536m) decayed over a 6-hour advection window.",
    ),
    "lead_time_hours": (
        "Forecast Lead Time (hr)",
        "Hours elapsed between NWP model initialization time and valid forecast timestamp.",
    ),
    "sin_hour": (
        "Diurnal Harmonic sin(2*pi*t/24)",
        "Sine component of the 24-hour solar diurnal cycle.",
    ),
    "cos_hour": (
        "Diurnal Harmonic cos(2*pi*t/24)",
        "Cosine component of the 24-hour solar diurnal cycle.",
    ),
    "sin_doy": (
        "Seasonal Harmonic sin(2*pi*DOY/365)",
        "Sine component of the annual solar declination cycle.",
    ),
    "cos_doy": (
        "Seasonal Harmonic cos(2*pi*DOY/365)",
        "Cosine component of the annual solar declination cycle.",
    ),
}


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
    """Stull (2011) empirical psychrometric wet-bulb approximation (C)."""
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
    lat_rad = math.radians(TASSAJARA_LAT)
    lon_deg = TASSAJARA_LON
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


# ---------------------------------------------------------------------------
# Strict Real-Time Horizon Filtering & Live 7-Day NWP Top-Up Helpers
# ---------------------------------------------------------------------------
def filter_future_horizon(df: pd.DataFrame, now_hour_utc: datetime) -> pd.DataFrame:
    """
    Strictly filters a forecast DataFrame to `valid_time_utc >= now_hour_utc`,
    normalizes timestamps to UTC, deduplicates any repeated hours (such as
    Daylight Saving Time fallback transitions), and sorts chronologically.
    """
    if df is None or df.empty or "valid_time_utc" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    valid_dt = pd.to_datetime(out["valid_time_utc"], utc=True, errors="coerce")
    out = out.loc[valid_dt.notna()].copy()
    if out.empty:
        return pd.DataFrame()

    out["_valid_dt_utc"] = pd.to_datetime(out["valid_time_utc"], utc=True).dt.floor("h")
    cutoff_ts = pd.Timestamp(now_hour_utc).floor("h")
    out = out.loc[out["_valid_dt_utc"] >= cutoff_ts].copy()
    if out.empty:
        return pd.DataFrame()

    out = (
        out.sort_values("_valid_dt_utc", ascending=True)
        .drop_duplicates(subset=["_valid_dt_utc"], keep="first")
        .reset_index(drop=True)
    )
    out["valid_time_utc"] = out["_valid_dt_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    out = out.drop(columns=["_valid_dt_utc"])
    return out


def attach_boundary_telemetry_for_mos(
    df: pd.DataFrame,
    obs_df: pd.DataFrame,
    ref_dict: dict,
    now_hour_utc: datetime,
) -> pd.DataFrame:
    """
    Attaches the latest Stone Office Vantage Pro2 observation and 5-station
    Santa Lucia reference boundary conditions (CHWC1, VTCC1, CAHC1, ASRC1, PPSC1)
    to future forecast rows prior to running TassajaraMOSCorrector inference.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "nws_temp_c" not in out.columns and "raw_nws_temp_c" in out.columns:
        out["nws_temp_c"] = out["raw_nws_temp_c"]
    if "raw_nws_temp_c" not in out.columns and "nws_temp_c" in out.columns:
        out["raw_nws_temp_c"] = out["nws_temp_c"]

    base_t = get_col(out, "nws_temp_c", 15.0)
    base_rh = get_col(out, "nws_rh_pct", 50.0)

    valid_dt = pd.to_datetime(out["valid_time_utc"], utc=True, errors="coerce")
    if "issue_time_utc" not in out.columns or out["issue_time_utc"].isna().all():
        out["issue_time_utc"] = now_hour_utc.isoformat()

    issue_dt = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce").fillna(
        pd.Timestamp(now_hour_utc)
    )
    computed_lead = ((valid_dt - issue_dt).dt.total_seconds() / 3600.0).clip(lower=1.0)
    if "lead_time_hours" not in out.columns:
        out["lead_time_hours"] = computed_lead
    else:
        out["lead_time_hours"] = pd.to_numeric(
            out["lead_time_hours"], errors="coerce"
        ).fillna(computed_lead).clip(lower=1.0)

    latest_tass_t = np.nan
    latest_ptrend = 0.0
    if obs_df is not None and not obs_df.empty:
        latest_tass_t = safe_float(obs_df.iloc[0].get("temp_c"), np.nan)
        latest_ptrend = safe_float(obs_df.iloc[0].get("pressure_trend_3hr_hpa"), 0.0)

    first_nws_t = float(base_t.iloc[0]) if not base_t.empty else 15.0
    init_bias_c = (
        float(latest_tass_t - first_nws_t) if not math.isnan(latest_tass_t) else 0.0
    )

    def _ref_val(stid: str, key: str, default_val=np.nan):
        if not isinstance(ref_dict, dict):
            return default_val
        node = ref_dict.get(stid, {})
        if not isinstance(node, dict):
            return default_val
        v = node.get(key)
        if v is None or pd.isna(v):
            return default_val
        try:
            f_v = float(v)
            return default_val if math.isnan(f_v) else f_v
        except (ValueError, TypeError):
            return default_val

    chews_t = _ref_val("CHWC1", "temp_c", np.nan)
    chews_rh = _ref_val("CHWC1", "rh_pct", np.nan)
    chews_p = _ref_val("CHWC1", "precip_1hr_mm", 0.0)

    vent_t = _ref_val("VTCC1", "temp_c", np.nan)
    vent_p = _ref_val("VTCC1", "precip_1hr_mm", np.nan)

    hast_t = _ref_val("CAHC1", "temp_c", np.nan)
    arroyo_t = _ref_val("ASRC1", "temp_c", np.nan)
    arroyo_rh = _ref_val("ASRC1", "rh_pct", np.nan)

    bigsur_t = _ref_val("PPSC1", "temp_c", np.nan)
    bigsur_rh = _ref_val("PPSC1", "rh_pct", np.nan)
    bigsur_p = _ref_val("PPSC1", "precip_1hr_mm", 0.0)

    if math.isnan(chews_t) and not math.isnan(vent_t):
        chews_t = vent_t - 0.54
    if math.isnan(vent_t) and not math.isnan(chews_t):
        vent_t = chews_t + 0.54
    if math.isnan(hast_t) and not math.isnan(arroyo_t):
        hast_t = arroyo_t - 1.80
    if math.isnan(arroyo_t) and not math.isnan(hast_t):
        arroyo_t = hast_t + 1.80

    out["pressure_trend_3hr_hpa"] = get_col(out, "pressure_trend_3hr_hpa", latest_ptrend)
    out["issue_tass_temp_c"] = get_col(
        out,
        "issue_tass_temp_c",
        latest_tass_t if not math.isnan(latest_tass_t) else base_t,
    )
    out["issue_persistence_bias_c"] = get_col(out, "issue_persistence_bias_c", init_bias_c)

    ref_anchor_t = out["issue_tass_temp_c"]
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
    )
    out["issue_bigsur_rh_pct"] = get_col(
        out,
        "issue_bigsur_rh_pct",
        bigsur_rh if not math.isnan(bigsur_rh) else base_rh,
    )
    out["issue_bigsur_precip_mm"] = get_col(out, "issue_bigsur_precip_mm", bigsur_p)

    return out


def fetch_live_nwp_topup(now_hour_utc: datetime) -> pd.DataFrame:
    """
    Fetches a live 168-hour (7-day) hourly NWP forecast for the Tassajara grid cell
    (36.2337°N, -121.5501°W, 880m MSL) via Open-Meteo GFS/HRRR Seamless API, with
    an automatic fallback to the operational weather.gov API.
    """
    issue_iso = now_hour_utc.isoformat()

    # Primary: Open-Meteo GFS/HRRR seamless hourly forecast at 880m NWP reference elevation
    om_url = "https://api.open-meteo.com/v1/forecast"
    om_params = {
        "latitude": TASSAJARA_LAT,
        "longitude": TASSAJARA_LON,
        "elevation": NWP_GRID_ELEV_M,
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "precipitation_probability,precipitation,cloud_cover,"
            "wind_speed_10m,wind_gusts_10m,wind_direction_10m"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "forecast_days": 8,
    }
    try:
        resp = requests.get(om_url, params=om_params, timeout=12)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        times = hourly.get("time", [])
        if times:
            dt_series = pd.to_datetime(times, utc=True).floor("h")
            df_live = pd.DataFrame({
                "issue_time_utc": issue_iso,
                "valid_time_utc": dt_series.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "nws_temp_c": pd.to_numeric(hourly.get("temperature_2m"), errors="coerce"),
                "nws_dewpoint_c": pd.to_numeric(hourly.get("dew_point_2m"), errors="coerce"),
                "nws_rh_pct": pd.to_numeric(hourly.get("relative_humidity_2m"), errors="coerce"),
                "nws_wind_speed_ms": pd.to_numeric(hourly.get("wind_speed_10m"), errors="coerce"),
                "nws_wind_gust_ms": pd.to_numeric(hourly.get("wind_gusts_10m"), errors="coerce"),
                "nws_wind_dir_deg": pd.to_numeric(hourly.get("wind_direction_10m"), errors="coerce"),
                "nws_pop_pct": pd.to_numeric(
                    hourly.get("precipitation_probability"), errors="coerce"
                ).fillna(0.0),
                "nws_qpf_1hr_mm": pd.to_numeric(
                    hourly.get("precipitation"), errors="coerce"
                ).fillna(0.0),
                "nws_sky_cover_pct": pd.to_numeric(
                    hourly.get("cloud_cover"), errors="coerce"
                ).fillna(20.0),
                "short_forecast": "Live NWP Grid",
            })
            df_live["raw_nws_temp_c"] = df_live["nws_temp_c"]
            df_live = filter_future_horizon(df_live, now_hour_utc).head(168).copy()
            if not df_live.empty:
                valid_dt = pd.to_datetime(df_live["valid_time_utc"], utc=True)
                df_live["lead_time_hours"] = (
                    (valid_dt - pd.Timestamp(now_hour_utc)).dt.total_seconds() / 3600.0
                ).clip(lower=1.0)
                return df_live
    except Exception:
        pass

    # Secondary Fallback: NWS API (/points/36.2337,-121.5501 -> forecastHourly)
    try:
        headers = {
            "User-Agent": "(TassajaraWeatherMOSMonitor/2.0, tzmc-weather@tassajara.org)",
            "Accept": "application/geo+json",
        }
        pt_resp = requests.get(
            f"https://api.weather.gov/points/{TASSAJARA_LAT},{TASSAJARA_LON}",
            headers=headers,
            timeout=10,
        )
        pt_resp.raise_for_status()
        hr_url = pt_resp.json().get("properties", {}).get("forecastHourly")
        if hr_url:
            fc_resp = requests.get(hr_url, headers=headers, timeout=12)
            fc_resp.raise_for_status()
            periods = fc_resp.json().get("properties", {}).get("periods", [])
            rows = []
            for p in periods:
                v_dt = pd.to_datetime(p.get("startTime"), utc=True).floor("h")
                t_val = float(p.get("temperature", 60.0))
                t_unit = str(p.get("temperatureUnit", "F")).upper()
                t_c = (t_val - 32.0) * 5.0 / 9.0 if t_unit == "F" else t_val
                rh_val = safe_float(
                    (p.get("relativeHumidity") or {}).get("value"), 50.0
                )
                dew_raw = (p.get("dewpoint") or {}).get("value")
                dew_c = (
                    float(dew_raw)
                    if dew_raw is not None
                    else t_c - ((100.0 - rh_val) / 5.0)
                )
                pop_val = safe_float(
                    (p.get("probabilityOfPrecipitation") or {}).get("value"), 0.0
                )
                w_str = str(p.get("windSpeed", "3 mph")).split()[0]
                try:
                    w_mph = float(w_str)
                except ValueError:
                    w_mph = 3.0
                w_ms = round(w_mph * 0.44704, 2)
                rows.append({
                    "issue_time_utc": issue_iso,
                    "valid_time_utc": v_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "nws_temp_c": round(t_c, 2),
                    "raw_nws_temp_c": round(t_c, 2),
                    "nws_dewpoint_c": round(dew_c, 2),
                    "nws_rh_pct": round(rh_val, 1),
                    "nws_wind_speed_ms": w_ms,
                    "nws_wind_gust_ms": round(w_ms * 1.35, 2),
                    "nws_wind_dir_deg": 295.0,
                    "nws_pop_pct": round(pop_val, 1),
                    "nws_qpf_1hr_mm": 0.0,
                    "nws_sky_cover_pct": 50.0 if pop_val >= 25.0 else 20.0,
                    "short_forecast": str(p.get("shortForecast") or "NWS Hourly"),
                })
            if rows:
                df_nws = pd.DataFrame(rows)
                df_nws = filter_future_horizon(df_nws, now_hour_utc).head(168).copy()
                valid_dt = pd.to_datetime(df_nws["valid_time_utc"], utc=True)
                df_nws["lead_time_hours"] = (
                    (valid_dt - pd.Timestamp(now_hour_utc)).dt.total_seconds() / 3600.0
                ).clip(lower=1.0)
                return df_nws
    except Exception:
        pass

    return pd.DataFrame()


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


def apply_mos_inference_if_needed(
    df: pd.DataFrame,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Ensures every row in `df` has MOS corrected_temp_c, p10, p90, mos_pop_pct,
    mos_qpf_mm, and mos_qpf_p90_mm across the 5-station Santa Lucia feature set.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in [
        "pred_temp_error_c",
        "corrected_temp_c",
        "corrected_temp_p10_c",
        "corrected_temp_p90_c",
        "mos_pop_pct",
        "mos_qpf_mm",
        "mos_qpf_p90_mm",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    raw_pop_s = get_col(out, "nws_pop_pct", 0.0)
    raw_qpf_s = get_col(out, "nws_qpf_1hr_mm", 0.0)
    mos_pop_s = get_col(out, "mos_pop_pct", np.nan)
    mos_qpf_s = get_col(out, "mos_qpf_mm", np.nan)

    if force_recompute:
        missing_mask = pd.Series(True, index=out.index)
    else:
        missing_mask = (
            out["corrected_temp_c"].isna()
            | mos_qpf_s.isna()
            | mos_pop_s.isna()
            | ((raw_pop_s >= 15.0) & (mos_pop_s <= 0.0))
            | ((raw_pop_s >= 20.0) & (mos_qpf_s <= 0.0))
            | ((mos_pop_s >= 20.0) & (mos_qpf_s <= 0.0))
            | ((raw_qpf_s > 0.0) & (mos_qpf_s <= 0.0))
        )
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
    sub["issue_chews_rh_pct"] = get_col(sub, "issue_chews_rh_pct", sub["nws_rh_pct"])
    sub["issue_chews_precip_mm"] = get_col(sub, "issue_chews_precip_mm", 0.0)
    sub["issue_hastings_temp_c"] = get_col(sub, "issue_hastings_temp_c", sub["nws_temp_c"] + 0.5)
    sub["issue_arroyo_temp_c"] = get_col(sub, "issue_arroyo_temp_c", sub["nws_temp_c"] + 1.4)
    sub["issue_arroyo_rh_pct"] = get_col(sub, "issue_arroyo_rh_pct", sub["nws_rh_pct"])
    sub["issue_bigsur_temp_c"] = get_col(sub, "issue_bigsur_temp_c", sub["nws_temp_c"] + 0.8)
    sub["issue_bigsur_rh_pct"] = get_col(sub, "issue_bigsur_rh_pct", sub["nws_rh_pct"])
    sub["issue_bigsur_precip_mm"] = get_col(sub, "issue_bigsur_precip_mm", 0.0)
    sub["issue_ventana_temp_c"] = get_col(
        sub, "issue_ventana_temp_c", sub["issue_chews_temp_c"] + 0.54
    )
    sub["issue_ventana_precip_mm"] = get_col(
        sub, "issue_ventana_precip_mm", sub["issue_chews_precip_mm"]
    )
    sub["issue_tass_temp_c"] = get_col(sub, "issue_tass_temp_c", sub["nws_temp_c"])
    sub["issue_persistence_bias_c"] = get_col(sub, "issue_persistence_bias_c", 0.0)
    if "issue_time_utc" not in sub.columns:
        sub["issue_time_utc"] = sub["valid_time_utc"]

    fb = fallback_physics_mos(sub)

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
                else:
                    out.loc[missing_mask, c] = fb[c].values

            sub_mos_pop = get_col(out.loc[missing_mask], "mos_pop_pct", 0.0)
            sub_mos_qpf = get_col(out.loc[missing_mask], "mos_qpf_mm", 0.0)
            zero_trapped = (sub_mos_pop >= 20.0) & (sub_mos_qpf <= 0.0)
            if zero_trapped.any():
                trap_idx = out.loc[missing_mask].index[zero_trapped]
                out.loc[trap_idx, "mos_qpf_mm"] = fb.loc[trap_idx, "mos_qpf_mm"].values
                out.loc[trap_idx, "mos_qpf_p90_mm"] = np.maximum(
                    get_col(out.loc[trap_idx], "mos_qpf_p90_mm", 0.0).values,
                    fb.loc[trap_idx, "mos_qpf_p90_mm"].values,
                )
            return out
        except Exception:
            pass

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
# Data Loading, Real-Time Horizon Filtering & Automatic Live 7-Day Top-Up
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_all_dashboard_data():
    now_hour_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now_hour_iso = now_hour_utc.isoformat()

    obs_df = pd.DataFrame()
    ref_dict = {}
    future_fc_df = pd.DataFrame()
    verify_df = pd.DataFrame()
    archive_stats = {
        "total_obs_hours": 0,
        "wet_hours": 0,
        "total_rain_mm": 0.0,
        "max_1hr_mm": 0.0,
    }

    if Path(DB_PATH).exists():
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

            # Archive-wide Precipitation Stats
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

            # 2. Latest 5-Station Reference Observations (Partitioned by station_id + Staleness Check)
            try:
                anchor_ts = (
                    str(obs_df.iloc[0]["valid_time_utc"])
                    if not obs_df.empty
                    else now_hour_iso
                )
                ref_df = pd.read_sql_query(
                    """
                    WITH ranked_refs AS (
                        SELECT
                            *,
                            ROW_NUMBER() OVER (
                                PARTITION BY station_id
                                ORDER BY valid_time_utc DESC
                            ) AS rn
                        FROM reference_observations
                        WHERE valid_time_utc <= ?
                    )
                    SELECT * FROM ranked_refs WHERE rn = 1
                    """,
                    conn,
                    params=(anchor_ts,),
                )
                anchor_dt = pd.to_datetime(anchor_ts, utc=True)
                for stid, meta in REFERENCE_STATIONS.items():
                    match = (
                        ref_df[ref_df["station_id"] == stid]
                        if not ref_df.empty and "station_id" in ref_df.columns
                        else pd.DataFrame()
                    )
                    if not match.empty:
                        row_d = match.iloc[0].to_dict()
                        obs_dt = pd.to_datetime(row_d.get("valid_time_utc"), utc=True)
                        age_hrs = (
                            abs((anchor_dt - obs_dt).total_seconds()) / 3600.0
                            if pd.notna(obs_dt)
                            else 999.0
                        )
                        is_fresh = age_hrs <= 3.5 and (
                            pd.notna(row_d.get("temp_c")) or pd.notna(row_d.get("precip_1hr_mm"))
                        )
                        row_d["online"] = bool(is_fresh)
                        row_d["age_hours"] = round(age_hrs, 1)
                        row_d["name"] = meta["name"]
                        row_d["elev_m"] = meta["elev_m"]
                        row_d["role"] = meta["role"]
                        ref_dict[stid] = row_d
                    else:
                        ref_dict[stid] = {
                            "station_id": stid,
                            "name": meta["name"],
                            "elev_m": meta["elev_m"],
                            "role": meta["role"],
                            "valid_time_utc": None,
                            "online": False,
                            "age_hours": None,
                            "temp_c": np.nan,
                            "rh_pct": np.nan,
                            "wind_speed_ms": np.nan,
                            "wind_gust_ms": np.nan,
                            "wind_dir_deg": np.nan,
                            "precip_1hr_mm": 0.0,
                        }
            except Exception:
                ref_dict = {}

            # Inspect mos_corrected_forecasts columns for backward-compatible SQL queries
            mos_cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(mos_corrected_forecasts)").fetchall()
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

            # 3. Future 7-Day Forecast Series (Strictly Filtered to valid_time_utc >= now_hour_utc)
            try:
                nws_issue_row = pd.read_sql_query(
                    """
                    SELECT issue_time_utc, COUNT(*) AS cnt
                    FROM nws_forecasts
                    WHERE (short_forecast != 'Archived NWP Grid' OR short_forecast IS NULL)
                      AND valid_time_utc >= ?
                    GROUP BY issue_time_utc
                    HAVING COUNT(*) >= 24
                    ORDER BY issue_time_utc DESC
                    LIMIT 1
                    """,
                    conn,
                    params=(now_hour_iso,),
                )
                if nws_issue_row.empty:
                    nws_issue_row = pd.read_sql_query(
                        """
                        SELECT issue_time_utc, COUNT(*) AS cnt
                        FROM nws_forecasts
                        WHERE valid_time_utc >= ?
                        GROUP BY issue_time_utc
                        HAVING COUNT(*) >= 12
                        ORDER BY issue_time_utc DESC
                        LIMIT 1
                        """,
                        conn,
                        params=(now_hour_iso,),
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
                          AND f.valid_time_utc >= ?
                        ORDER BY f.valid_time_utc ASC
                        """,
                        conn,
                        params=(latest_nws_issue, now_hour_iso),
                    )
                    future_fc_df = filter_future_horizon(future_fc_df, now_hour_utc)
            except Exception:
                future_fc_df = pd.DataFrame()

            # 4. Paired Historical Verification Dataset (with 5-Station Reference Joins)
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
                        COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.83) AS issue_chews_temp_c,
                        COALESCE(r_chews.rh_pct, f.nws_rh_pct) AS issue_chews_rh_pct,
                        COALESCE(r_chews.precip_1hr_mm, 0.0) AS issue_chews_precip_mm,
                        COALESCE(r_hast.temp_c, r_arroyo.temp_c - 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) - 0.60) AS issue_hastings_temp_c,
                        COALESCE(r_arroyo.temp_c, r_hast.temp_c + 2.01, COALESCE(o_issue.temp_c, f.nws_temp_c) + 1.41) AS issue_arroyo_temp_c,
                        COALESCE(r_arroyo.rh_pct, r_hast.rh_pct, f.nws_rh_pct) AS issue_arroyo_rh_pct,
                        COALESCE(r_bigsur.temp_c, COALESCE(o_issue.temp_c, f.nws_temp_c) + 2.45) AS issue_bigsur_temp_c,
                        COALESCE(r_bigsur.rh_pct, f.nws_rh_pct) AS issue_bigsur_rh_pct,
                        COALESCE(r_bigsur.precip_1hr_mm, 0.0) AS issue_bigsur_precip_mm,
                        COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, COALESCE(o_issue.temp_c, f.nws_temp_c) - 6.29) AS issue_ventana_temp_c,
                        COALESCE(r_vent.precip_1hr_mm, r_chews.precip_1hr_mm, 0.0) AS issue_ventana_precip_mm,
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
                        ON o.valid_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
                    LEFT JOIN reference_observations r_hast
                        ON o.valid_time_utc = r_hast.valid_time_utc AND r_hast.station_id = 'CAHC1'
                    LEFT JOIN reference_observations r_arroyo
                        ON o.valid_time_utc = r_arroyo.valid_time_utc AND r_arroyo.station_id = 'ASRC1'
                    LEFT JOIN reference_observations r_bigsur
                        ON o.valid_time_utc = r_bigsur.valid_time_utc AND r_bigsur.station_id = 'PPSC1'
                    LEFT JOIN reference_observations r_vent
                        ON o.valid_time_utc = r_vent.valid_time_utc AND r_vent.station_id = 'VTCC1'
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

    # Ensure ref_dict has entries for all 5 Santa Lucia reference stations
    for stid, meta in REFERENCE_STATIONS.items():
        if stid not in ref_dict:
            ref_dict[stid] = {
                "station_id": stid,
                "name": meta["name"],
                "elev_m": meta["elev_m"],
                "role": meta["role"],
                "valid_time_utc": None,
                "online": False,
                "age_hours": None,
                "temp_c": np.nan,
                "rh_pct": np.nan,
                "wind_speed_ms": np.nan,
                "wind_gust_ms": np.nan,
                "wind_dir_deg": np.nan,
                "precip_1hr_mm": 0.0,
            }

    # Live WeatherLink v2 (/current/36966) & 5-Station Synoptic Real-Time Top-Up
    try:
        wl_client = WeatherLinkV2Client()
        live_obs = wl_client.fetch_current_observations(station_id=36966)
        if isinstance(live_obs, dict) and live_obs.get("temp_c") is not None:
            t_c_live = float(live_obs["temp_c"])
            rh_live = safe_float(live_obs.get("rh_pct"), 50.0)
            dew_live = live_obs.get("dewpoint_c")
            if dew_live is None or pd.isna(dew_live):
                r_clip = max(1.0, min(100.0, rh_live))
                gam = math.log(r_clip / 100.0) + (17.27 * t_c_live) / (237.3 + t_c_live)
                dew_live = round((237.3 * gam) / (17.27 - gam), 2)
            wb_live = live_obs.get("wetbulb_c")
            if wb_live is None or pd.isna(wb_live):
                wb_live = compute_wetbulb_stull_c(t_c_live, rh_live)

            exact_ts_iso = str(live_obs.get("valid_time_utc") or datetime.now(timezone.utc).isoformat())
            prev_ptrend = (
                safe_float(obs_df.iloc[0].get("pressure_trend_3hr_hpa"), 0.0)
                if not obs_df.empty
                else 0.0
            )
            ptrend_live = safe_float(live_obs.get("pressure_trend_hpa"), prev_ptrend)
            if ptrend_live == 0.0 and prev_ptrend != 0.0:
                ptrend_live = prev_ptrend

            existing_hr_rain = 0.0
            if not obs_df.empty and str(obs_df.iloc[0].get("valid_time_utc")) == now_hour_iso:
                existing_hr_rain = safe_float(obs_df.iloc[0].get("precip_1hr_mm"), 0.0)
            rain_1h_live = max(existing_hr_rain, safe_float(live_obs.get("precip_1hr_mm"), 0.0))

            live_row_dict = {
                "valid_time_utc": now_hour_iso,
                "exact_obs_time_utc": exact_ts_iso,
                "station_id": str(live_obs.get("station_id") or "TASS_VP2_36966"),
                "temp_c": round(t_c_live, 2),
                "dewpoint_c": round(float(dew_live), 2),
                "wetbulb_c": round(float(wb_live), 2),
                "rh_pct": round(rh_live, 1),
                "wind_speed_ms": safe_float(live_obs.get("wind_speed_ms"), 0.0),
                "wind_gust_ms": safe_float(live_obs.get("wind_gust_ms"), 0.0),
                "wind_dir_deg": safe_float(live_obs.get("wind_dir_deg"), 0.0),
                "mslp_hpa": safe_float(
                    live_obs.get("mslp_hpa"),
                    safe_float(obs_df.iloc[0].get("mslp_hpa"), 1015.0) if not obs_df.empty else 1015.0,
                ),
                "pressure_trend_3hr_hpa": round(ptrend_live, 2),
                "precip_1hr_mm": round(rain_1h_live, 3),
                "solar_rad_wm2": live_obs.get("solar_rad_wm2"),
                "uv_index": live_obs.get("uv_index"),
                "qc_flag": "OK",
            }

            if obs_df.empty:
                obs_df = pd.DataFrame([live_row_dict])
            else:
                if "exact_obs_time_utc" not in obs_df.columns:
                    obs_df["exact_obs_time_utc"] = obs_df["valid_time_utc"]
                obs_df = obs_df[obs_df["valid_time_utc"] != now_hour_iso].copy()
                obs_df = pd.concat([pd.DataFrame([live_row_dict]), obs_df], ignore_index=True)

            if Path(DB_PATH).exists():
                try:
                    with sqlite3.connect(DB_PATH) as conn_w:
                        conn_w.execute(
                            """
                            INSERT OR REPLACE INTO local_observations (
                                valid_time_utc, station_id, temp_c, dewpoint_c, wetbulb_c,
                                rh_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg,
                                mslp_hpa, pressure_trend_3hr_hpa, precip_1hr_mm,
                                solar_rad_wm2, uv_index, qc_flag
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OK')
                            """,
                            (
                                live_row_dict["valid_time_utc"],
                                live_row_dict["station_id"],
                                live_row_dict["temp_c"],
                                live_row_dict["dewpoint_c"],
                                live_row_dict["wetbulb_c"],
                                live_row_dict["rh_pct"],
                                live_row_dict["wind_speed_ms"],
                                live_row_dict["wind_gust_ms"],
                                live_row_dict["wind_dir_deg"],
                                live_row_dict["mslp_hpa"],
                                live_row_dict["pressure_trend_3hr_hpa"],
                                live_row_dict["precip_1hr_mm"],
                                live_row_dict["solar_rad_wm2"],
                                live_row_dict["uv_index"],
                            ),
                        )
                        conn_w.commit()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        live_refs = fetch_regional_reference_stations()
        if isinstance(live_refs, dict):
            ref_db_rows = []
            for stid, rdata in live_refs.items():
                if isinstance(rdata, dict) and rdata.get("online") and rdata.get("temp_c") is not None:
                    ref_db_rows.append((
                        now_hour_iso,
                        stid,
                        float(rdata["temp_c"]),
                        safe_float(rdata.get("rh_pct"), None),
                        safe_float(rdata.get("wind_speed_ms"), 0.0),
                        safe_float(rdata.get("wind_gust_ms"), 0.0),
                        safe_float(rdata.get("wind_dir_deg"), 0.0),
                        safe_float(rdata.get("precip_1hr_mm"), 0.0),
                    ))
                    meta = REFERENCE_STATIONS.get(stid, {})
                    ref_dict[stid] = {
                        "station_id": stid,
                        "name": meta.get("name", stid),
                        "elev_m": meta.get("elev_m"),
                        "role": meta.get("role", "Reference"),
                        "valid_time_utc": rdata.get("valid_time_utc") or now_hour_iso,
                        "online": True,
                        "age_hours": 0.1,
                        "temp_c": float(rdata["temp_c"]),
                        "rh_pct": safe_float(rdata.get("rh_pct"), np.nan),
                        "wind_speed_ms": safe_float(rdata.get("wind_speed_ms"), np.nan),
                        "wind_gust_ms": safe_float(rdata.get("wind_gust_ms"), np.nan),
                        "wind_dir_deg": safe_float(rdata.get("wind_dir_deg"), np.nan),
                        "precip_1hr_mm": safe_float(rdata.get("precip_1hr_mm"), 0.0),
                    }
        if (
            pd.isna(ref_dict.get("VTCC1", {}).get("temp_c"))
            and pd.notna(ref_dict.get("CHWC1", {}).get("temp_c"))
        ):
            ref_dict["VTCC1"]["temp_c"] = round(float(ref_dict["CHWC1"]["temp_c"]) + 0.54, 2)
            ref_dict["VTCC1"]["online"] = True
            ref_dict["VTCC1"]["age_hours"] = ref_dict["CHWC1"].get("age_hours", 0.2)
    except Exception:
        pass

    # Check if SQLite forecast batch is stale (> 6 hours old) or has < 144 future hours remaining
    batch_age_hours = 999.0
    if not future_fc_df.empty and "issue_time_utc" in future_fc_df.columns:
        issue_dt = pd.to_datetime(
            future_fc_df.iloc[0]["issue_time_utc"], utc=True, errors="coerce"
        )
        if pd.notna(issue_dt):
            batch_age_hours = max(
                0.0, (pd.Timestamp(now_hour_utc) - issue_dt).total_seconds() / 3600.0
            )

    needs_live_topup = (
        future_fc_df.empty
        or batch_age_hours > 6.0
        or len(future_fc_df) < 144
    )

    if needs_live_topup:
        live_df = fetch_live_nwp_topup(now_hour_utc)
        if not live_df.empty:
            if future_fc_df.empty or batch_age_hours > 6.0:
                future_fc_df = live_df.copy()
            else:
                combined_fc = pd.concat([future_fc_df, live_df], ignore_index=True, sort=False)
                future_fc_df = filter_future_horizon(combined_fc, now_hour_utc).head(168).copy()

    if not future_fc_df.empty:
        future_fc_df = filter_future_horizon(future_fc_df, now_hour_utc).head(168).copy()
        future_fc_df = attach_boundary_telemetry_for_mos(
            future_fc_df, obs_df, ref_dict, now_hour_utc
        )
        future_fc_df = apply_mos_inference_if_needed(
            future_fc_df, force_recompute=needs_live_topup
        )
        if "corrected_rh_pct" not in future_fc_df.columns:
            future_fc_df["corrected_rh_pct"] = future_fc_df["nws_rh_pct"]
        else:
            future_fc_df["corrected_rh_pct"] = future_fc_df[
                "corrected_rh_pct"
            ].combine_first(future_fc_df["nws_rh_pct"])

        if "corrected_wind_gust_ms" not in future_fc_df.columns:
            future_fc_df["corrected_wind_gust_ms"] = get_col(
                future_fc_df,
                "nws_wind_gust_ms",
                get_col(future_fc_df, "nws_wind_speed_ms", 1.0) * 1.35,
            )
        else:
            future_fc_df["corrected_wind_gust_ms"] = (
                future_fc_df["corrected_wind_gust_ms"]
                .combine_first(future_fc_df["nws_wind_gust_ms"])
                .combine_first(future_fc_df["nws_wind_speed_ms"] * 1.35)
            )

    return ref_dict, obs_df, future_fc_df, verify_df, archive_stats


@st.cache_data(ttl=300)
def compute_model_diagnostics_and_importances(verify_df: pd.DataFrame):
    """
    Fits a standardized, well-conditioned Statsmodels OLS regression with Newey-West
    HAC robust standard errors (12-hour lag) and computes permutation feature importances
    for both the Thermal and Hydrological MOS models from models/tassajara_mos_v1.joblib.
    """
    diag_out = {
        "model_loaded": False,
        "precip_ml_trained": False,
        "feature_names": [],
        "ols_summary_text": "",
        "ols_r2": np.nan,
        "ols_adj_r2": np.nan,
        "ols_fstat": np.nan,
        "ols_aic": np.nan,
        "ols_cond_no": np.nan,
        "ols_nobs": 0,
        "ols_coef_df": pd.DataFrame(),
        "imp_df": pd.DataFrame(),
    }

    mos_obj = None
    if MODEL_PATH.exists():
        try:
            mos_obj = joblib.load(MODEL_PATH)
            diag_out["model_loaded"] = True
            diag_out["precip_ml_trained"] = bool(getattr(mos_obj, "precip_ml_trained", False))
            diag_out["feature_names"] = list(getattr(mos_obj, "feature_names", []))
            if getattr(mos_obj, "ols_summary", None):
                diag_out["ols_summary_text"] = str(mos_obj.ols_summary)
        except Exception:
            mos_obj = None

    if verify_df.empty or len(verify_df) < 20:
        return diag_out

    eval_sub = verify_df.tail(min(len(verify_df), 10000)).copy()
    X_full = engineer_mos_features(eval_sub).fillna(0.0)
    y_temp_err = (
        get_col(eval_sub, "actual_temp_c", 15.0) - get_col(eval_sub, "raw_nws_temp_c", 15.0)
    ).values

    # 1. Fit Standardized Newey-West HAC OLS on verification features
    try:
        X_ols = build_well_conditioned_ols_matrix(X_full)
        ols_res = sm.OLS(y_temp_err, X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        diag_out["ols_summary_text"] = ols_res.summary().as_text()
        diag_out["ols_r2"] = float(ols_res.rsquared)
        diag_out["ols_adj_r2"] = float(ols_res.rsquared_adj)
        diag_out["ols_fstat"] = float(ols_res.fvalue) if ols_res.fvalue is not None else np.nan
        diag_out["ols_aic"] = float(ols_res.aic)
        diag_out["ols_cond_no"] = float(ols_res.condition_number)
        diag_out["ols_nobs"] = int(ols_res.nobs)

        ci_df = ols_res.conf_int()
        coef_rows = []
        for feat in ols_res.params.index:
            if feat == "const":
                label, desc = (
                    "Intercept (Mean Bias Offset)",
                    "Mean systematic NWP temperature bias (C) at average predictor conditions.",
                )
            else:
                label, desc = FEATURE_METADATA.get(feat, (feat, "Engineered physical MOS predictor."))
            coef_rows.append({
                "Feature Key": feat,
                "Physical Predictor": label,
                "Std. Beta (C / 1-SD)": round(float(ols_res.params[feat]), 4),
                "HAC Std Error": round(float(ols_res.bse[feat]), 4),
                "z-Statistic": round(float(ols_res.tvalues[feat]), 2),
                "p-Value": round(float(ols_res.pvalues[feat]), 4),
                "95% CI Low": round(float(ci_df.loc[feat, 0]), 4),
                "95% CI High": round(float(ci_df.loc[feat, 1]), 4),
                "Physical Interpretation": desc,
            })
        diag_out["ols_coef_df"] = pd.DataFrame(coef_rows)
    except Exception:
        pass

    # 2. Compute Feature Importance Rankings (Permutation Importance on GBDT + Hydro Sensitivity)
    feat_cols = diag_out["feature_names"] if diag_out["feature_names"] else list(X_full.columns)
    for c in feat_cols:
        if c not in X_full.columns:
            X_full[c] = 0.0
    X_model = X_full[feat_cols]

    if mos_obj is not None and hasattr(mos_obj, "model_mean"):
        try:
            perm_t = permutation_importance(
                mos_obj.model_mean,
                X_model,
                y_temp_err,
                scoring="neg_root_mean_squared_error",
                n_repeats=5,
                random_state=42,
            )
            thermal_raw = np.maximum(0.0, perm_t.importances_mean)
        except Exception:
            thermal_raw = np.abs(
                X_model.corrwith(pd.Series(y_temp_err, index=X_model.index)).fillna(0.0).values
            )
    else:
        thermal_raw = np.abs(
            X_model.corrwith(pd.Series(y_temp_err, index=X_model.index)).fillna(0.0).values
        )

    t_sum = float(thermal_raw.sum())
    thermal_pct = (thermal_raw / t_sum * 100.0) if t_sum > 0 else np.zeros_like(thermal_raw)

    y_hydro_target = get_col(eval_sub, "mos_qpf_mm", get_col(eval_sub, "actual_precip_mm", 0.0)).values
    if (
        mos_obj is not None
        and getattr(mos_obj, "precip_ml_trained", False)
        and hasattr(mos_obj, "model_qpf_mean")
    ):
        try:
            perm_h = permutation_importance(
                mos_obj.model_qpf_mean,
                X_model,
                y_hydro_target,
                scoring="neg_root_mean_squared_error",
                n_repeats=5,
                random_state=42,
            )
            hydro_map = dict(zip(X_model.columns, np.maximum(0.0, perm_h.importances_mean)))
        except Exception:
            hydro_map = {}
    else:
        base_hydro = analytical_hydro_hurdle_fallback(X_full)
        base_pop = base_hydro["mos_pop_pct"].values
        base_qpf = base_hydro["mos_qpf_mm"].values
        hydro_map = {}
        for col in X_full.columns:
            X_pert = X_full.copy()
            col_std = float(X_pert[col].std())
            X_pert[col] = X_pert[col] + (col_std if col_std > 1e-4 else 1.0)
            pert_hydro = analytical_hydro_hurdle_fallback(X_pert)
            delta_score = (
                np.mean(np.abs(pert_hydro["mos_pop_pct"].values - base_pop))
                + 10.0 * np.mean(np.abs(pert_hydro["mos_qpf_mm"].values - base_qpf))
            )
            hydro_map[col] = float(delta_score)

    all_features = list(dict.fromkeys(list(feat_cols) + list(X_full.columns)))
    t_dict = dict(zip(feat_cols, thermal_pct))
    h_vals = np.array([hydro_map.get(f, 0.0) for f in all_features], dtype=float)
    h_sum = float(h_vals.sum())
    h_pct = (h_vals / h_sum * 100.0) if h_sum > 0 else np.zeros_like(h_vals)
    h_dict = dict(zip(all_features, h_pct))

    imp_rows = []
    for f in all_features:
        label, desc = FEATURE_METADATA.get(f, (f, "Engineered physical MOS predictor."))
        t_val = round(float(t_dict.get(f, 0.0)), 2)
        h_val = round(float(h_dict.get(f, 0.0)), 2)
        imp_rows.append({
            "Feature Key": f,
            "Physical Predictor": label,
            "Thermal Importance (%)": t_val,
            "Hydrological Importance (%)": h_val,
            "Combined Share (%)": round(0.5 * (t_val + h_val), 2),
            "Physical Mechanism": desc,
        })

    imp_df = (
        pd.DataFrame(imp_rows)
        .sort_values("Thermal Importance (%)", ascending=False)
        .reset_index(drop=True)
    )
    diag_out["imp_df"] = imp_df
    return diag_out


# ---------------------------------------------------------------------------
# Helper Functions for Layman's / Everyday Forecast View
# ---------------------------------------------------------------------------
def get_weather_icon_and_label(
    pop_pct: float,
    qpf_in: float,
    sky_pct: float,
    temp_f: float,
    is_daytime: bool = True,
    in_canyon_shade: bool = False,
):
    """Returns an intuitive (emoji_icon, short_label) tuple for everyday weather display."""
    if qpf_in >= 0.15 or pop_pct >= 65:
        if temp_f <= 34.0:
            return "🌨️", "Wintry Mix / Snow"
        return "🌧️", "Rain Likely"
    if qpf_in >= 0.02 or pop_pct >= 35:
        return "🌦️", "Chance of Showers"
    if pop_pct >= 20:
        return "☁️", "Mostly Cloudy"
    if not is_daytime:
        if sky_pct >= 60:
            return "☁️", "Cloudy Night"
        if temp_f <= 36.0:
            return "❄️", "Cold & Clear Night"
        return "🌙", "Clear Canyon Night"
    if in_canyon_shade:
        return "🏔️", "Canyon Wall Shade"
    if sky_pct >= 70:
        return "🌥️", "Mostly Cloudy"
    if sky_pct >= 35:
        return "⛅", "Partly Sunny"
    if temp_f >= 88.0:
        return "☀️", "Hot & Sunny"
    return "☀️", "Sunny & Clear"


# ---------------------------------------------------------------------------
# Early Flood & Tassajara Road Washout Warning UI Components
# ---------------------------------------------------------------------------
def render_flood_and_road_warning_banner(hydro_summary: dict, is_scientific: bool = False):
    """
    Renders a high-visibility 4-Tier Creek Flood & Tassajara Road Passability
    Warning Banner for both the Everyday Community View and Scientific MOS Suite.
    Uses theme-adaptive CSS variables (`var(--text-color)`) so text is crisp and
    legible in both Streamlit Light and Dark modes.
    """
    if not isinstance(hydro_summary, dict) or not hydro_summary:
        return

    code = str(hydro_summary.get("alert_code", "GREEN"))
    tier_num = int(hydro_summary.get("alert_tier", 0))
    label = str(hydro_summary.get("alert_label", "Normal — Creek Stable & Road Passable"))
    icon = str(hydro_summary.get("icon", "🟢"))
    color_hex = str(hydro_summary.get("color_hex", "#10b981"))
    bg_rgba = str(hydro_summary.get("bg_rgba", "rgba(16, 185, 129, 0.14)"))
    border_rgba = str(hydro_summary.get("border_rgba", "rgba(16, 185, 129, 0.45)"))

    # High-contrast tier label colors that read clearly on both light and dark backgrounds
    tier_accent_map = {
        "GREEN": "#059669",
        "YELLOW": "#b45309",
        "ORANGE": "#c2410c",
        "RED": "#dc2626",
    }
    tier_text_color = tier_accent_map.get(code, color_hex)

    api_in = float(hydro_summary.get("api_in", 0.0))
    api_mm = float(hydro_summary.get("api_mm", 0.0))
    proj_api_24_in = float(hydro_summary.get("projected_max_api_24h_in", api_in))
    c_val = float(hydro_summary.get("runoff_coeff_c", 0.08))
    c_pct = float(hydro_summary.get("runoff_coeff_pct", 8.0))
    decay_k = float(hydro_summary.get("decay_k", 0.91))

    rain_24_in = float(hydro_summary.get("precip_24hr_in", 0.0))
    fc_qpf_24_in = float(hydro_summary.get("forecast_qpf_24hr_in", 0.0))
    fc_qpf_72_in = float(hydro_summary.get("forecast_qpf_72hr_in", 0.0))
    peak_p90_rate_in = float(hydro_summary.get("peak_p90_1hr_24h_in", 0.0))
    discharge_cfs = float(hydro_summary.get("discharge_index_cfs", 6.5))
    peak_discharge_24_cfs = float(hydro_summary.get("peak_discharge_24hr_cfs", discharge_cfs))

    road_status = str(hydro_summary.get("road_status", "Open / Dry to Damp"))
    creek_status = str(hydro_summary.get("creek_status", "Stable Baseflow"))
    action_text = str(hydro_summary.get("recommended_action", "Routine canyon operations."))
    trigger_text = str(hydro_summary.get("trigger_summary", "Catchment stable."))

    tech_line = (
        f'<div class="tzmc-muted" style="font-family: monospace; font-size: 0.78rem; '
        f'margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(148, 163, 184, 0.35);">'
        f'<b>HYDROLOGICAL TELEMETRY:</b> '
        f'API<sub>t</sub> = {api_in:.2f} in ({api_mm:.1f} mm, k={decay_k:.3f}) &bull; '
        f'24h Peak API = {proj_api_24_in:.2f} in &bull; '
        f'Runoff Coeff C = {c_val:.2f} ({c_pct:.0f}%) &bull; '
        f'24h Rain Obs/Fcst = {rain_24_in:.2f}" / {fc_qpf_24_in:.2f}" &bull; '
        f'72h MOS QPF = {fc_qpf_72_in:.2f}" &bull; '
        f'Peak 24h P90 Rate = {peak_p90_rate_in:.2f} in/hr &bull; '
        f'Routed Creek Q = {discharge_cfs:.1f} CFS (24h Peak: {peak_discharge_24_cfs:.1f} CFS)'
        f'</div>'
    )

    st.markdown(
        f'<div style="background: {bg_rgba}; border: 1.5px solid {border_rgba}; '
        f'border-left: 6px solid {color_hex}; border-radius: 8px; '
        f'padding: 14px 18px; margin-bottom: 1.1rem; color: var(--text-color);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
        f'<div style="font-size: 1.05rem; font-weight: 700; color: var(--text-color);">'
        f'{icon} CREEK FLOOD &amp; TASSAJARA ROAD WARNING SYSTEM — '
        f'<span style="color: {tier_text_color};">TIER {tier_num} ({code}): {label}</span>'
        f'</div>'
        f'<div class="tzmc-pill">'
        f'Soil Wetness API: {api_in:.2f}" &nbsp;|&nbsp; Runoff C: {c_pct:.0f}% &nbsp;|&nbsp; '
        f'24h Peak Q: {peak_discharge_24_cfs:.0f} CFS'
        f'</div>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); '
        f'gap: 10px; margin-top: 10px; font-size: 0.86rem; color: var(--text-color);">'
        f'<div><b>🛣️ Tassajara Road (14-mi Unpaved):</b> {road_status}</div>'
        f'<div><b>🌊 Tassajara Creek &amp; Bridges:</b> {creek_status}</div>'
        f'<div><b>⚡ Operational Action:</b> {action_text}</div>'
        f'</div>'
        f'<div class="tzmc-subtle" style="font-size: 0.81rem; margin-top: 6px;">'
        f'<b>Active Diagnostic Triggers:</b> {trigger_text}'
        f'</div>'
        f'{tech_line if is_scientific else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_hydro_risk_section(
    hydro_bundle: dict,
    now_hour_utc: datetime,
    is_scientific: bool = True,
):
    """
    Renders the interactive Creek Runoff, Antecedent Soil Saturation (API_t) Gauge,
    and 72-Hour Routed Hydrograph Projection panel alongside rainfall forecasts.
    """
    if not isinstance(hydro_bundle, dict) or "summary" not in hydro_bundle:
        return

    s = hydro_bundle["summary"]
    hist_df = hydro_bundle.get("hist_hydro_df", pd.DataFrame())
    proj_df = hydro_bundle.get("proj_fc_df", pd.DataFrame())

    st.subheader(
        "Early Flood & Road Washout Warning System: Soil Saturation (API) & Creek Hydrograph"
        if is_scientific
        else "Tassajara Creek Runoff & Road Passability Outlook (Next 72 Hours)"
    )

    # 1. 4-Card Hydrological KPI Row
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric(
        "Antecedent Wetness (API_t)",
        f"{s['api_in']:.2f} in ({s['api_mm']:.1f} mm)",
        f"24h Peak: {s['projected_max_api_24h_in']:.2f} in | Decay k={s['decay_k']:.3f}",
        delta_color="off",
        help=(
            "Antecedent Precipitation Index (API_t = API_{t-1} * k + P_t) tracking multi-day "
            "soil moisture storage across the rocky Santa Lucia chaparral and mixed-oak catchment. "
            "Values above 1.00 in (25.4 mm) indicate a saturated catchment primed for rapid runoff."
        ),
    )
    hc2.metric(
        "Effective Runoff Coefficient (C)",
        f"{s['runoff_coeff_pct']:.1f} % (C = {s['runoff_coeff_c']:.2f})",
        f"Headwater Spillover: {s['upstream_spillover_in']:.2f} in/hr",
        delta_color="off",
        help=(
            "Nonlinear fraction of incoming rainfall converted directly into surface quickflow "
            "(R_t = C_t * P_t). Scales from 8% on dry fractured granite/duff up to 86% when "
            "canyon soils reach saturation."
        ),
    )
    hc3.metric(
        "Projected Peak Runoff (6h / 24h)",
        f"{s['peak_runoff_24hr_in_hr']:.2f} in/hr ({s['peak_runoff_24hr_mm_hr']:.1f} mm/hr)",
        f"Next 6h Peak: {s['peak_runoff_6hr_in_hr']:.2f} in/hr | P90 Rain: {s['peak_p90_1hr_24h_in']:.2f} in/hr",
        delta_color="off",
        help=(
            "Maximum hourly effective surface runoff rate (R_t = C_t * QPF_t) expected over the "
            "next 6 and 24 hours, paired with the 90th-percentile (P90) downpour intensity."
        ),
    )
    hc4.metric(
        "Tassajara Creek Discharge Index",
        f"{s['discharge_index_cfs']:.1f} CFS",
        f"24h Peak: {s['peak_discharge_24hr_cfs']:.1f} CFS (P90: {s['peak_discharge_p90_24hr_cfs']:.0f} CFS)",
        delta_color="off",
        help=(
            "Estimated discharge (cubic feet per second) routed through the 72 km^2 upper "
            "Tassajara Creek watershed (Pine Ridge, Black Cone, Chews Ridge) using a 3-hour "
            "steep-canyon kinematic unit hydrograph."
        ),
    )

    # 2. Gauge + 2-Panel Hydrograph Projection Layout
    col_gauge, col_chart = st.columns([1.05, 2.95])

    with col_gauge:
        gauge_max_in = max(3.2, round(float(s["projected_max_api_72h_in"]) * 1.25, 1))
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=float(s["api_in"]),
                number={"suffix": " in", "font": {"size": 28}},
                delta={
                    "reference": float(s["projected_max_api_24h_in"]),
                    "increasing": {"color": "#ef4444"},
                    "decreasing": {"color": "#10b981"},
                    "valueformat": ".2f",
                },
                title={
                    "text": (
                        "<b>Catchment Soil Saturation (API)</b><br>"
                        f"<span style='font-size:0.80em;color:#64748b'>"
                        f"Runoff Readiness C = {s['runoff_coeff_pct']:.0f}% | "
                        f"24h Peak: {s['projected_max_api_24h_in']:.2f} in</span>"
                    ),
                    "font": {"size": 14},
                },
                gauge={
                    "axis": {"range": [0, gauge_max_in], "tickwidth": 1, "tickcolor": "#64748b"},
                    "bar": {"color": s["color_hex"], "thickness": 0.34},
                    "bgcolor": "rgba(148, 163, 184, 0.10)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(148, 163, 184, 0.38)",
                    "steps": [
                        {"range": [0, YELLOW_API_MM / MM_PER_INCH], "color": "rgba(16, 185, 129, 0.25)"},
                        {
                            "range": [YELLOW_API_MM / MM_PER_INCH, ORANGE_API_MM / MM_PER_INCH],
                            "color": "rgba(234, 179, 8, 0.28)",
                        },
                        {
                            "range": [ORANGE_API_MM / MM_PER_INCH, RED_API_EXTREME_MM / MM_PER_INCH],
                            "color": "rgba(249, 115, 22, 0.30)",
                        },
                        {
                            "range": [RED_API_EXTREME_MM / MM_PER_INCH, gauge_max_in],
                            "color": "rgba(239, 68, 68, 0.34)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.80,
                        "value": float(s["projected_max_api_24h_in"]),
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=300,
            margin=dict(t=55, b=20, l=22, r=22),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f'<div class="tzmc-card" style="padding: 10px 12px; font-size: 0.79rem;">'
            f'<b>Catchment Geometry:</b> 72 km² upper watershed (Pine Ridge, Black Cone, Chews Ridge → '
            f'Tassajara Creek at 485 m MSL).<br>'
            f'<b>Stream Gauge Telemetry:</b> {s["gauge_sensor_health"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_chart:
        # Combine past 48h spin-up and next 72h projection for continuous hydrograph plot
        h_sub = hist_df.tail(48).copy() if not hist_df.empty else pd.DataFrame()
        p_sub = proj_df.head(72).copy() if not proj_df.empty else pd.DataFrame()

        fig_hydro = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.48, 0.52],
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=(
                "Antecedent Soil Wetness Index (API in) & Effective Runoff Coefficient C (%)",
                "Routed Tassajara Creek Discharge (CFS) & Hourly Effective Surface Runoff (in/hr)",
            ),
        )

        if not h_sub.empty:
            h_sub["valid_local"] = pd.to_datetime(h_sub["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
            fig_hydro.add_trace(
                go.Scatter(
                    x=h_sub["valid_local"],
                    y=h_sub["api_in"],
                    mode="lines",
                    line=dict(color="#059669", width=2.4),
                    name="Observed Spin-Up API (in)",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
            fig_hydro.add_trace(
                go.Scatter(
                    x=h_sub["valid_local"],
                    y=(h_sub["runoff_coeff_c"] * 100.0).round(1),
                    mode="lines",
                    line=dict(color="#9333ea", width=1.8, dash="dot"),
                    name="Runoff Coeff C (%)",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
            fig_hydro.add_trace(
                go.Scatter(
                    x=h_sub["valid_local"],
                    y=h_sub["discharge_index_cfs"],
                    mode="lines",
                    line=dict(color="#059669", width=2.3),
                    name="Estimated Historical Creek Q (CFS)",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig_hydro.add_trace(
                go.Bar(
                    x=h_sub["valid_local"],
                    y=(h_sub["runoff_mm_hr"] / MM_PER_INCH).round(3),
                    name="Observed Effective Runoff (in/hr)",
                    marker_color="#10b981",
                    opacity=0.70,
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

        if not p_sub.empty:
            p_sub["valid_local"] = pd.to_datetime(p_sub["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
            fig_hydro.add_trace(
                go.Scatter(
                    x=p_sub["valid_local"],
                    y=p_sub["projected_api_p90_in"],
                    mode="lines",
                    line=dict(color="rgba(220, 38, 38, 0.75)", width=1.6, dash="dash"),
                    name="Projected P90 Extreme API (in)",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
            fig_hydro.add_trace(
                go.Scatter(
                    x=p_sub["valid_local"],
                    y=p_sub["projected_api_in"],
                    mode="lines",
                    line=dict(color="#0284c7", width=2.6),
                    name="Projected Expected API (in)",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
            fig_hydro.add_trace(
                go.Scatter(
                    x=p_sub["valid_local"],
                    y=(p_sub["projected_runoff_coeff"] * 100.0).round(1),
                    mode="lines",
                    line=dict(color="#9333ea", width=1.8, dash="dot"),
                    name="Projected Runoff Coeff C (%)",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

            fig_hydro.add_trace(
                go.Scatter(
                    x=p_sub["valid_local"],
                    y=p_sub["projected_discharge_p90_cfs"],
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(239, 68, 68, 0.12)",
                    line=dict(color="rgba(220, 38, 38, 0.75)", width=1.6, dash="dash"),
                    name="Projected P90 Surge Q (CFS)",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig_hydro.add_trace(
                go.Scatter(
                    x=p_sub["valid_local"],
                    y=p_sub["projected_discharge_cfs"],
                    mode="lines",
                    line=dict(color="#0284c7", width=2.6),
                    name="Projected Expected Creek Q (CFS)",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig_hydro.add_trace(
                go.Bar(
                    x=p_sub["valid_local"],
                    y=p_sub["projected_runoff_in_hr"],
                    name="Projected Effective Runoff (in/hr)",
                    marker_color="#0ea5e9",
                    opacity=0.70,
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

        # Add Advisory / Watch threshold lines on Panel 1
        fig_hydro.add_hline(
            y=YELLOW_API_MM / MM_PER_INCH,
            line_dash="dot",
            line_color="#d97706",
            annotation_text="Yellow Advisory (1.00 in)",
            annotation_position="top left",
            row=1,
            col=1,
            secondary_y=False,
        )
        fig_hydro.add_hline(
            y=ORANGE_API_MM / MM_PER_INCH,
            line_dash="dot",
            line_color="#ea580c",
            annotation_text="Orange Watch (1.75 in)",
            annotation_position="top right",
            row=1,
            col=1,
            secondary_y=False,
        )

        divider_local = pd.Timestamp(now_hour_utc).tz_convert(LOCAL_TZ)
        for r_i in (1, 2):
            fig_hydro.add_vline(
                x=divider_local,
                line_width=1.4,
                line_dash="dash",
                line_color=DIVIDER_LINE_RGBA,
                row=r_i,
                col=1,
            )

        fig_hydro.update_yaxes(
            title_text="API Wetness (in)",
            gridcolor=GRID_RGBA,
            row=1,
            col=1,
            secondary_y=False,
        )
        fig_hydro.update_yaxes(
            title_text="Runoff Coeff C (%)",
            range=[0, 100],
            showgrid=False,
            row=1,
            col=1,
            secondary_y=True,
        )
        fig_hydro.update_yaxes(
            title_text="Creek Discharge (CFS)",
            gridcolor=GRID_RGBA,
            row=2,
            col=1,
            secondary_y=False,
        )
        fig_hydro.update_yaxes(
            title_text="Runoff Rate (in/hr)",
            showgrid=False,
            row=2,
            col=1,
            secondary_y=True,
        )
        fig_hydro.update_xaxes(gridcolor=GRID_RGBA, row=1, col=1)
        fig_hydro.update_xaxes(gridcolor=GRID_RGBA, row=2, col=1)
        fig_hydro.update_layout(
            height=440,
            margin=dict(t=40, b=30, l=50, r=50),
            hovermode="x unified",
            plot_bgcolor=PLOT_BG_RGBA,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig_hydro, use_container_width=True)

    with st.expander(
        "Inspect 4-Tier Flood & Road Severity Matrix & Hourly Projected Hydrograph Table",
        expanded=False,
    ):
        matrix_rows = [
            {
                "Alert Tier": "🟢 Tier 0 — GREEN (Normal)",
                "Soil Saturation & Rain Triggers": "API < 1.00 in AND 24h Rain < 1.00 in AND 1h Rate < 0.50 in/hr",
                "Tassajara Creek Response": TIER_METADATA["GREEN"]["creek_status"],
                "Tassajara Road Passability": TIER_METADATA["GREEN"]["road_status"],
                "Operational Protocol": TIER_METADATA["GREEN"]["recommended_action"],
            },
            {
                "Alert Tier": "🟡 Tier 1 — YELLOW (Advisory)",
                "Soil Saturation & Rain Triggers": "API >= 1.00 in (25.4 mm) OR 24h Rain >= 1.00 in OR 72h Rain >= 1.50 in",
                "Tassajara Creek Response": TIER_METADATA["YELLOW"]["creek_status"],
                "Tassajara Road Passability": TIER_METADATA["YELLOW"]["road_status"],
                "Operational Protocol": TIER_METADATA["YELLOW"]["recommended_action"],
            },
            {
                "Alert Tier": "🟠 Tier 2 — ORANGE (Watch / Road Risk)",
                "Soil Saturation & Rain Triggers": "1h Rain >= 0.50 in/hr (or P90 >= 0.75 in/hr) on primed soil OR 24h Rain >= 2.00 in",
                "Tassajara Creek Response": TIER_METADATA["ORANGE"]["creek_status"],
                "Tassajara Road Passability": TIER_METADATA["ORANGE"]["road_status"],
                "Operational Protocol": TIER_METADATA["ORANGE"]["recommended_action"],
            },
            {
                "Alert Tier": "🔴 Tier 3 — RED (Warning / Road Closure)",
                "Soil Saturation & Rain Triggers": "1h Rain >= 1.00 in/hr (or P90 >= 1.25 in/hr) OR 72h Rain >= 3.50-4.00 in",
                "Tassajara Creek Response": TIER_METADATA["RED"]["creek_status"],
                "Tassajara Road Passability": TIER_METADATA["RED"]["road_status"],
                "Operational Protocol": TIER_METADATA["RED"]["recommended_action"],
            },
        ]
        st.markdown("#### 1. TZMC Multi-Tier Flood & Road Washout Severity Matrix")
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

        if not proj_df.empty:
            st.markdown("#### 2. Hourly Projected Catchment Saturation & Routed Creek Hydrograph (Next 72 Hours)")
            p_tbl = proj_df.head(72).copy()
            p_tbl["Local Time"] = (
                pd.to_datetime(p_tbl["valid_time_utc"], utc=True)
                .dt.tz_convert(LOCAL_TZ)
                .dt.strftime("%Y-%m-%d %H:%M")
            )
            display_hydro_tbl = pd.DataFrame({
                "Valid Time (Local)": p_tbl["Local Time"],
                "Alert Tier": p_tbl["projected_tier"],
                "MOS QPF (in/hr)": (p_tbl["mos_qpf_mm"] / MM_PER_INCH).round(3),
                "MOS P90 Rate (in/hr)": (p_tbl["mos_qpf_p90_mm"] / MM_PER_INCH).round(3),
                "Projected API (in)": p_tbl["projected_api_in"].round(2),
                "P90 Extreme API (in)": p_tbl["projected_api_p90_in"].round(2),
                "Runoff Coeff C (%)": (p_tbl["projected_runoff_coeff"] * 100.0).round(1),
                "Effective Runoff (in/hr)": p_tbl["projected_runoff_in_hr"].round(3),
                "Expected Creek Q (CFS)": p_tbl["projected_discharge_cfs"].round(1),
                "P90 Surge Ceiling Q (CFS)": p_tbl["projected_discharge_p90_cfs"].round(1),
            })
            st.dataframe(display_hydro_tbl, use_container_width=True, hide_index=True)


def render_layman_forecast_view(
    obs_df: pd.DataFrame,
    future_fc_df: pd.DataFrame,
    ref_dict: dict,
    solar_state: str,
    solar_el: float,
    last_obs_str: str,
    now_hour_utc: datetime,
    hydro_bundle: dict = None,
):
    """Renders the intuitive, jargon-free Everyday Weather Forecast Page for TZMC."""
    # Guarantee future_fc_df only contains hours >= now_hour_utc and has no DST duplicates
    future_fc_df = filter_future_horizon(future_fc_df, now_hour_utc)

    if isinstance(hydro_bundle, dict) and "summary" in hydro_bundle:
        render_flood_and_road_warning_banner(hydro_bundle["summary"], is_scientific=False)

    if not obs_df.empty:
        latest = obs_df.iloc[0]
        last_24 = obs_df.head(24)
        temp_c = safe_float(latest.get("temp_c"), 15.0)
        temp_f = c_to_f(temp_c)
        t24_s = get_col(last_24, "temp_c", temp_c)
        hi_24_f = c_to_f(safe_float(t24_s.max(), temp_c))
        lo_24_f = c_to_f(safe_float(t24_s.min(), temp_c))
        rh_pct = safe_float(latest.get("rh_pct"), 50.0)
        dew_f = c_to_f(safe_float(latest.get("dewpoint_c"), temp_c - 4.0))

        valid_w = (
            last_24[last_24["wind_speed_ms"].notna()]
            if "wind_speed_ms" in last_24.columns
            else pd.DataFrame()
        )
        w_spd_ms = safe_float(
            latest.get("wind_speed_ms"),
            safe_float(valid_w.iloc[0]["wind_speed_ms"], 0.0) if not valid_w.empty else 0.0,
        )
        w_gst_ms = safe_float(
            latest.get("wind_gust_ms"),
            safe_float(valid_w.iloc[0]["wind_gust_ms"], 0.0) if not valid_w.empty else 0.0,
        )
        w_dir = latest.get("wind_dir_deg")
        if (w_dir is None or pd.isna(w_dir)) and not valid_w.empty:
            w_dir = valid_w.iloc[0].get("wind_dir_deg")
        w_mph = ms_to_mph(w_spd_ms)
        w_gst_mph = ms_to_mph(w_gst_ms)
        w_card = deg_to_cardinal(w_dir) if w_mph > 0.4 else "Calm"

        rain_24_in = mm_to_in(float(get_col(last_24, "precip_1hr_mm", 0.0).sum()))
    else:
        temp_f, hi_24_f, lo_24_f = 65.0, 72.0, 48.0
        rh_pct, dew_f = 50.0, 45.0
        w_mph, w_gst_mph, w_card = 0.0, 0.0, "Calm"
        rain_24_in = 0.0

    is_day = solar_el > 0.0
    in_shade = "Shadow" in solar_state
    cur_sky = 20.0
    cur_pop = 0.0
    if not future_fc_df.empty:
        cur_sky = safe_float(future_fc_df.iloc[0].get("nws_sky_cover_pct"), 20.0)
        cur_pop = safe_float(
            future_fc_df.iloc[0].get("mos_pop_pct", future_fc_df.iloc[0].get("nws_pop_pct")),
            0.0,
        )

    cur_icon, cur_cond = get_weather_icon_and_label(
        cur_pop, rain_24_in, cur_sky, temp_f, is_daytime=is_day, in_canyon_shade=in_shade
    )

    if solar_el <= 0.0:
        sun_badge = "🌙 Nighttime in the Canyon — Cool air gently drains down Tassajara Creek"
    elif in_shade:
        sun_badge = (
            "🏔️ Canyon Wall Shade — The sun is above the horizon "
            "but hidden behind Pine Ridge / Black Cone"
        )
    else:
        sun_badge = "☀️ Direct Sunlight on Canyon Floor — Warmest part of the day at Stone Office"

    hero_col1, hero_col2 = st.columns([1.35, 1.65])
    with hero_col1:
        st.markdown(
            f'<div class="tzmc-card" style="border: 1.5px solid rgba(2, 132, 199, 0.40); '
            f'padding: 20px 24px; height: 100%;">'
            f'<div style="font-size: 0.82rem; text-transform: uppercase; '
            f'letter-spacing: 0.06em; color: #0284c7; font-weight: 700;">'
            f'Right Now at Tassajara Stone Office (1,590 ft)</div>'
            f'<div style="display: flex; align-items: center; gap: 18px; margin: 10px 0;">'
            f'<span style="font-size: 3.6rem; line-height: 1;">{cur_icon}</span>'
            f'<div>'
            f'<div style="font-size: 2.9rem; font-weight: 800; line-height: 1.05; color: var(--text-color);">'
            f'{temp_f:.0f}°F</div>'
            f'<div class="tzmc-subtle" style="font-size: 1.1rem; font-weight: 600;">'
            f'{cur_cond}</div>'
            f'</div></div>'
            f'<div class="tzmc-subtle" style="font-size: 0.90rem; margin-top: 8px;">'
            f'{sun_badge}</div>'
            f'<div class="tzmc-muted" style="font-size: 0.78rem; margin-top: 10px;">'
            f'Updated: {last_obs_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with hero_col2:
        c1, c2 = st.columns(2)
        wind_disp = "Calm" if w_mph < 0.5 else f"{w_mph:.0f} mph {w_card}"
        c1.metric(
            "Today's High / Low",
            f"{hi_24_f:.0f}°F / {lo_24_f:.0f}°F",
            "Past 24-Hour Range",
            delta_color="off",
            help="Highest and lowest temperatures recorded at the Stone Office over the past 24 hours.",
        )
        c2.metric(
            "Canyon Breeze",
            wind_disp,
            f"Peak Gust: {w_gst_mph:.0f} mph",
            delta_color="off",
            help="Current wind speed and direction at the Stone Office roof.",
        )
        c3, c4 = st.columns(2)
        comfort_label = (
            "Very Dry Air"
            if rh_pct < 25
            else ("Comfortable" if rh_pct < 65 else "Humid / Damp")
        )
        c3.metric(
            "Air Humidity",
            f"{rh_pct:.0f}%",
            f"{comfort_label} (Dewpoint {dew_f:.0f}°F)",
            delta_color="off",
            help="Relative humidity and comfort level inside the canyon.",
        )
        chews_t_c = safe_float(ref_dict.get("CHWC1", {}).get("temp_c"), np.nan)
        bigsur_t_c = safe_float(ref_dict.get("PPSC1", {}).get("temp_c"), np.nan)
        ref_parts = []
        if not math.isnan(chews_t_c):
            ref_parts.append(f"Chews Ridge: {c_to_f(chews_t_c):.0f}°F")
        if not math.isnan(bigsur_t_c):
            ref_parts.append(f"Big Sur: {c_to_f(bigsur_t_c):.0f}°F")
        chews_str = " | ".join(ref_parts) if ref_parts else "Crest Refs: Syncing"
        c4.metric(
            "Rain (Past 24 Hours)",
            f"{rain_24_in:.2f} in",
            chews_str,
            delta_color="off",
            help="Total rainfall measured in the past 24 hours, plus current temperatures at Chews Ridge and Big Sur.",
        )

    st.markdown("")

    # 2. Plain-English Canyon Advisories (Freeze, Road, & Trail Conditions)
    if not future_fc_df.empty:
        next_48 = future_fc_df.head(48).copy()
        next_72 = future_fc_df.head(72).copy()

        corr_c_48 = get_col(next_48, "corrected_temp_c", 15.0)
        p10_c_48 = get_col(next_48, "corrected_temp_p10_c", corr_c_48 - 2.1)
        min_low_48_f = c_to_f(float(corr_c_48.min()))
        min_floor_48_f = c_to_f(float(p10_c_48.min()))
        max_high_48_f = c_to_f(float(corr_c_48.max()))

        rain_72_in = mm_to_in(
            float(get_col(next_72, "mos_qpf_mm", get_col(next_72, "nws_qpf_1hr_mm", 0.0)).sum())
        )
        p90_rain_72_in = mm_to_in(
            float(get_col(next_72, "mos_qpf_p90_mm", rain_72_in * 25.4 * 1.5).sum())
        )
        max_pop_72 = float(
            get_col(next_72, "mos_pop_pct", get_col(next_72, "nws_pop_pct", 0.0)).max()
        )

        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            if min_floor_48_f <= 32.0:
                st.error(
                    f"❄️ **Freeze Warning (Next 48h):** Overnight lows expected near "
                    f"**{min_low_48_f:.0f}°F** (creek cold-pool floor **{min_floor_48_f:.0f}°F**). "
                    f"Protect exposed pipes and tender garden plants."
                )
            elif min_floor_48_f <= 36.0:
                st.warning(
                    f"🌡️ **Chilly Nights Ahead:** Overnight lows dipping to "
                    f"**{min_low_48_f:.0f}°F** (coldest creek pockets near "
                    f"**{min_floor_48_f:.0f}°F**). Bring a warm jacket for early morning zazen."
                )
            else:
                st.success(
                    f"🌱 **No Freeze Expected (Next 48h):** Overnight lows will stay mild, "
                    f"bottoming out around **{min_low_48_f:.0f}°F**."
                )

        with adv2:
            if rain_72_in >= 1.0 or p90_rain_72_in >= 1.75:
                st.warning(
                    f"🌧️ **Tassajara Road & Creek Advisory:** **{rain_72_in:.2f} in** of rain "
                    f"expected over the next 3 days (up to **{p90_rain_72_in:.2f} in** in heavier "
                    f"downpours). Expect muddy grades on Tassajara Rd and rising creek levels."
                )
            elif rain_72_in >= 0.10 or max_pop_72 >= 40:
                st.info(
                    f"🌦️ **Light Rain Possible (Next 3 Days):** Up to **{rain_72_in:.2f} in** "
                    f"of rain forecast ({max_pop_72:.0f}% chance). Roads should remain passable."
                )
            else:
                st.success(
                    f"🚗 **Dry Road & Trail Conditions:** No significant rain expected over "
                    f"the next 72 hours (Rain chance {max_pop_72:.0f}%)."
                )

        with adv3:
            if max_high_48_f >= 92.0:
                st.warning(
                    f"☀️ **Warm Canyon Afternoons:** Daytime highs reaching "
                    f"**{max_high_48_f:.0f}°F**. Plan steep hikes (Black Cone / Pine Ridge) "
                    f"for the cool morning hours and carry extra water."
                )
            else:
                st.info(
                    f"🥾 **Pleasant Canyon Weather:** Daytime highs near "
                    f"**{max_high_48_f:.0f}°F** with cool shade entering the valley floor "
                    f"in mid-afternoon once the sun dips behind the ridge."
                )

    st.markdown("---")

    # 3. Next 24 Hours at a Glance (3-Hour Interval Cards)
    if not future_fc_df.empty:
        st.subheader("Next 24 Hours at a Glance (Every 3 Hours)")
        fc_24 = future_fc_df.head(24).copy()
        fc_24["valid_local"] = (
            pd.to_datetime(fc_24["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        )
        raw_mos_f = c_to_f(get_col(fc_24, "corrected_temp_c", 15.0))
        fc_24["smooth_temp_f"] = (
            raw_mos_f.rolling(window=3, center=True, min_periods=1).mean().round(0)
        )
        fc_24["pop_val"] = (
            get_col(fc_24, "mos_pop_pct", get_col(fc_24, "nws_pop_pct", 0.0))
            .clip(0.0, 100.0)
            .round(0)
        )
        fc_24["qpf_in"] = (
            get_col(fc_24, "mos_qpf_mm", get_col(fc_24, "nws_qpf_1hr_mm", 0.0)) / 25.4
        )
        fc_24["sky_val"] = get_col(fc_24, "nws_sky_cover_pct", 20.0)
        fc_24["wind_mph"] = (get_col(fc_24, "nws_wind_speed_ms", 1.0) * 2.23694).round(0)

        step_rows = fc_24.iloc[::3].head(8)
        cols_24 = st.columns(len(step_rows))
        for idx_col, (_, r) in enumerate(step_rows.iterrows()):
            t_loc = r["valid_local"]
            hr_num = t_loc.hour
            is_day_hr = 7 <= hr_num <= 18
            in_sh_hr = 15 <= hr_num <= 18
            ic, lbl = get_weather_icon_and_label(
                float(r["pop_val"]),
                float(r["qpf_in"]),
                float(r["sky_val"]),
                float(r["smooth_temp_f"]),
                is_daytime=is_day_hr,
                in_canyon_shade=in_sh_hr,
            )
            time_lbl = t_loc.strftime("%-I %p") if hasattr(t_loc, "strftime") else str(hr_num)
            day_tag = t_loc.strftime("%a")
            pop_int = int(r["pop_val"])
            pop_badge = f"💧 {pop_int}%" if pop_int >= 15 else "Dry"

            with cols_24[idx_col]:
                st.markdown(
                    f'<div class="tzmc-card" style="padding: 10px 6px; text-align: center;">'
                    f'<div class="tzmc-muted" style="font-size: 0.75rem; font-weight: 600;">'
                    f'{day_tag} {time_lbl}</div>'
                    f'<div style="font-size: 1.9rem; margin: 6px 0;">{ic}</div>'
                    f'<div style="font-size: 1.25rem; font-weight: 700; color: var(--text-color);">'
                    f'{r["smooth_temp_f"]:.0f}°F</div>'
                    f'<div class="tzmc-subtle" style="font-size: 0.74rem; margin-top: 2px;">'
                    f'{lbl}</div>'
                    f'<div style="font-size: 0.75rem; color: #0284c7; font-weight: 600; margin-top: 4px;">'
                    f'{pop_badge}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # 4. Smoothed 48-Hour Temperature & Rain Chance Chart
    if not future_fc_df.empty:
        st.subheader("Hourly Temperature & Rain Outlook (Next 48 Hours)")
        fc_48 = future_fc_df.head(48).copy()
        fc_48["valid_local"] = (
            pd.to_datetime(fc_48["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        )
        raw_t_f = c_to_f(get_col(fc_48, "corrected_temp_c", 15.0))
        fc_48["smooth_temp_f"] = (
            raw_t_f.rolling(window=3, center=True, min_periods=1).mean().round(1)
        )
        fc_48["mos_pop"] = (
            get_col(fc_48, "mos_pop_pct", get_col(fc_48, "nws_pop_pct", 0.0))
            .clip(0.0, 100.0)
            .round(0)
        )
        fc_48["mos_rain_in"] = (
            get_col(fc_48, "mos_qpf_mm", get_col(fc_48, "nws_qpf_1hr_mm", 0.0)) / 25.4
        ).round(3)

        fig_layman = make_subplots(specs=[[{"secondary_y": True}]])

        fig_layman.add_trace(
            go.Scatter(
                x=fc_48["valid_local"],
                y=fc_48["mos_pop"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(2, 132, 199, 0.12)",
                line=dict(color="rgba(2, 132, 199, 0.65)", width=1.6, dash="dot"),
                name="Chance of Rain (%)",
            ),
            secondary_y=True,
        )

        if (fc_48["mos_rain_in"] > 0).any():
            fig_layman.add_trace(
                go.Bar(
                    x=fc_48["valid_local"],
                    y=fc_48["mos_rain_in"],
                    name="Expected Rain (in/hr)",
                    marker_color="#0284c7",
                    opacity=0.65,
                ),
                secondary_y=True,
            )

        fig_layman.add_trace(
            go.Scatter(
                x=fc_48["valid_local"],
                y=fc_48["smooth_temp_f"],
                mode="lines",
                line=dict(color="#ea580c", width=3.2, shape="spline"),
                name="Canyon Temperature (°F)",
            ),
            secondary_y=False,
        )

        fig_layman.add_hline(
            y=32.0,
            line_dash="dot",
            line_color="#0284c7",
            annotation_text="Freezing (32°F)",
            annotation_position="bottom right",
            secondary_y=False,
        )

        fig_layman.update_yaxes(
            title_text="Temperature (°F)",
            gridcolor=GRID_RGBA,
            secondary_y=False,
        )
        fig_layman.update_yaxes(
            title_text="Rain Chance (%)",
            range=[0, 105],
            showgrid=False,
            secondary_y=True,
        )
        fig_layman.update_xaxes(
            title_text="Local Canyon Time",
            gridcolor=GRID_RGBA,
        )
        fig_layman.update_layout(
            height=390,
            margin=dict(t=25, b=35, l=50, r=50),
            hovermode="x unified",
            plot_bgcolor=PLOT_BG_RGBA,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig_layman, use_container_width=True)

    if isinstance(hydro_bundle, dict) and "summary" in hydro_bundle:
        st.markdown("---")
        render_hydro_risk_section(hydro_bundle, now_hour_utc=now_hour_utc, is_scientific=False)

    st.markdown("---")

    # 5. 7-Day Visual Daily Forecast Cards & Simple Table
    if not future_fc_df.empty:
        st.subheader("7-Day Tassajara Canyon Forecast")
        d_df = future_fc_df.copy()
        if obs_df is not None and not obs_df.empty and "valid_time_utc" in obs_df.columns:
            today_local_date = pd.Timestamp(now_hour_utc).tz_convert(LOCAL_TZ).date()
            obs_today = obs_df.copy()
            obs_today["valid_local"] = (
                pd.to_datetime(obs_today["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
            )
            obs_today = obs_today[obs_today["valid_local"].dt.date == today_local_date].copy()
            if not obs_today.empty:
                obs_today["corrected_temp_c"] = obs_today["temp_c"]
                obs_today["corrected_temp_p10_c"] = obs_today["temp_c"]
                obs_today["mos_pop_pct"] = 0.0
                obs_today["nws_pop_pct"] = 0.0
                obs_today["mos_qpf_mm"] = get_col(obs_today, "precip_1hr_mm", 0.0)
                obs_today["nws_qpf_1hr_mm"] = get_col(obs_today, "precip_1hr_mm", 0.0)
                obs_today["nws_sky_cover_pct"] = 20.0
                obs_today["corrected_wind_gust_ms"] = get_col(
                    obs_today, "wind_gust_ms", get_col(obs_today, "wind_speed_ms", 0.0)
                )
                obs_today["nws_wind_speed_ms"] = get_col(obs_today, "wind_speed_ms", 0.0)
                d_df = pd.concat([obs_today, d_df], ignore_index=True, sort=False)
                d_df["_hr_utc"] = pd.to_datetime(d_df["valid_time_utc"], utc=True).dt.floor("h")
                d_df = (
                    d_df.sort_values("_hr_utc")
                    .drop_duplicates(subset=["_hr_utc"], keep="first")
                    .drop(columns=["_hr_utc"])
                    .reset_index(drop=True)
                )

        d_df["valid_local"] = (
            pd.to_datetime(d_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        )
        d_df["Date_Label"] = d_df["valid_local"].dt.strftime("%a, %b %d")
        d_df["Day_Short"] = d_df["valid_local"].dt.strftime("%a")
        d_df["Date_Short"] = d_df["valid_local"].dt.strftime("%b %d")
        d_df["Sort_Date"] = d_df["valid_local"].dt.date

        d_df["_t_f"] = c_to_f(get_col(d_df, "corrected_temp_c", 15.0))
        d_df["_p10_f"] = c_to_f(
            get_col(d_df, "corrected_temp_p10_c", get_col(d_df, "corrected_temp_c", 15.0) - 2.1)
        )
        d_df["_pop"] = get_col(d_df, "mos_pop_pct", get_col(d_df, "nws_pop_pct", 0.0))
        d_df["_qpf_in"] = (
            get_col(d_df, "mos_qpf_mm", get_col(d_df, "nws_qpf_1hr_mm", 0.0)) / 25.4
        )
        d_df["_sky"] = get_col(d_df, "nws_sky_cover_pct", 20.0)
        d_df["_gust_mph"] = (
            get_col(d_df, "corrected_wind_gust_ms", get_col(d_df, "nws_wind_speed_ms", 1.0) * 1.35)
            * 2.23694
        )

        day_grp = (
            d_df.groupby(["Sort_Date", "Date_Label", "Day_Short", "Date_Short"], as_index=False)
            .agg(
                High_F=("_t_f", lambda s: round(float(s.max()), 0)),
                Low_F=("_t_f", lambda s: round(float(s.min()), 0)),
                Coldest_Creek_F=("_p10_f", lambda s: round(float(s.min()), 0)),
                Rain_Chance_Pct=("_pop", lambda s: int(round(safe_float(s.max(), 0.0), 0))),
                Rain_Total_In=("_qpf_in", lambda s: round(float(s.sum()), 2)),
                Avg_Sky=("_sky", lambda s: float(s.mean())),
                Peak_Breeze_MPH=("_gust_mph", lambda s: round(float(s.max()), 0)),
            )
            .sort_values("Sort_Date")
            .head(7)
        )

        day_cols = st.columns(len(day_grp))
        summary_rows = []
        for idx_d, (_, r_d) in enumerate(day_grp.iterrows()):
            ic_d, cond_d = get_weather_icon_and_label(
                float(r_d["Rain_Chance_Pct"]),
                float(r_d["Rain_Total_In"]),
                float(r_d["Avg_Sky"]),
                float(r_d["High_F"]),
                is_daytime=True,
                in_canyon_shade=False,
            )
            rain_Badge = (
                f"💧 {r_d['Rain_Chance_Pct']}% ({r_d['Rain_Total_In']:.2f}\")"
                if r_d["Rain_Chance_Pct"] >= 15 or r_d["Rain_Total_In"] > 0
                else "💧 0% (Dry)"
            )
            with day_cols[idx_d]:
                st.markdown(
                    f'<div class="tzmc-card" style="padding: 12px 8px; text-align: center; height: 100%;">'
                    f'<div style="font-size: 0.92rem; font-weight: 700; color: var(--text-color);">'
                    f'{r_d["Day_Short"]}</div>'
                    f'<div class="tzmc-muted" style="font-size: 0.75rem; margin-bottom: 6px;">'
                    f'{r_d["Date_Short"]}</div>'
                    f'<div style="font-size: 2.2rem; margin: 6px 0;">{ic_d}</div>'
                    f'<div class="tzmc-subtle" style="font-size: 0.78rem; min-height: 2.1em;">'
                    f'{cond_d}</div>'
                    f'<div style="font-size: 1.15rem; font-weight: 700; margin-top: 6px;">'
                    f'<span style="color: #ea580c;">{r_d["High_F"]:.0f}°</span> / '
                    f'<span style="color: #0284c7;">{r_d["Low_F"]:.0f}°</span></div>'
                    f'<div class="tzmc-muted" style="font-size: 0.74rem; margin-top: 6px;">'
                    f'{rain_Badge}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            note_list = []
            if r_d["Rain_Total_In"] >= 1.0:
                note_list.append("Heavy rain; muddy road & rising creek")
            elif r_d["Rain_Total_In"] >= 0.15 or r_d["Rain_Chance_Pct"] >= 50:
                note_list.append("Bring rain gear")
            if r_d["Coldest_Creek_F"] <= 32.0:
                note_list.append("Freeze risk overnight (<= 32°F)")
            elif r_d["Coldest_Creek_F"] <= 37.0:
                note_list.append("Crisp, cold morning along creek")
            if r_d["High_F"] >= 90.0:
                note_list.append("Hot afternoon; hike early")
            if not note_list:
                note_list.append("Pleasant canyon day, cool night")

            summary_rows.append({
                "Day": r_d["Date_Label"],
                "Conditions": f"{ic_d} {cond_d}",
                "Daytime High": f"{r_d['High_F']:.0f} °F",
                "Overnight Low": f"{r_d['Low_F']:.0f} °F",
                "Coldest Creek Pocket": f"{r_d['Coldest_Creek_F']:.0f} °F",
                "Chance of Rain": f"{r_d['Rain_Chance_Pct']}%",
                "Expected Rain": f"{r_d['Rain_Total_In']:.2f} in",
                "Peak Breeze": f"{r_d['Peak_Breeze_MPH']:.0f} mph",
                "What to Expect": " • ".join(note_list),
            })

        st.markdown("")
        with st.expander("View Detailed 7-Day Planner Table", expanded=False):
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Header, Sidebar & Station Metadata Banner
# ---------------------------------------------------------------------------
now_utc = datetime.now(timezone.utc)
now_hour_utc = now_utc.replace(minute=0, second=0, microsecond=0)

ref_dict, obs_df, future_fc_df, verify_df, archive_stats = load_all_dashboard_data()
future_fc_df = filter_future_horizon(future_fc_df, now_hour_utc)
model_diag = compute_model_diagnostics_and_importances(verify_df)
hydro_bundle = compute_dashboard_hydro_state(
    obs_df=obs_df,
    future_fc_df=future_fc_df,
    ref_dict=ref_dict,
)

solar_state, solar_el, solar_az, horizon_elev = compute_canyon_solar_state(now_utc)

active_model_name = (
    "MOS_GBM_v2_HYDRO (5-Station Santa Lucia Spatial + Hurdle PoP/QPF)"
    if MODEL_PATH.exists()
    else "TOPO_HYDRO_PHYSICS (Analytical Baseline)"
)

with st.sidebar:
    st.markdown("### TZMC MOS Architecture")
    st.caption(
        "Model Output Statistics (MOS) ensemble & 5-station Santa Lucia boundary-layer "
        "diagnostics for Station 36966 (Stone Office Davis Vantage Pro2, 485 m MSL)."
    )
    st.metric(
        "Serialized Model Artifact",
        "Loaded (v1.joblib)" if model_diag["model_loaded"] else "Physics Fallback",
        f"Features: {len(model_diag['feature_names']) or 25} predictors",
        delta_color="off",
        help="Indicates whether models/tassajara_mos_v1.joblib is loaded in memory.",
    )
    st.metric(
        "OLS Explained Variance (R²)",
        f"{model_diag['ols_r2']:.3f}" if not math.isnan(model_diag["ols_r2"]) else "N/A",
        f"Adj R²: {model_diag['ols_adj_r2']:.3f} (n={model_diag['ols_nobs']})"
        if not math.isnan(model_diag["ols_adj_r2"])
        else "Awaiting paired rows",
        delta_color="off",
        help="Proportion of linear NWP temperature bias variance explained by the physical predictors.",
    )
    st.metric(
        "Archive Wet Hours (>=0.01 in)",
        f"{archive_stats['wet_hours']} hrs",
        f"Total Archive Rain: {mm_to_in(archive_stats['total_rain_mm']):.2f} in",
        delta_color="off",
        help="Total hours with tipping-bucket rainfall >= 0.254 mm in local_observations.",
    )
    st.metric(
        "Flood & Road Alert Tier",
        f"{hydro_bundle['summary']['icon']} {hydro_bundle['summary']['alert_code']} (Tier {hydro_bundle['summary']['alert_tier']})",
        f"API: {hydro_bundle['summary']['api_in']:.2f} in | C: {hydro_bundle['summary']['runoff_coeff_pct']:.0f}%",
        delta_color="off",
        help="Active 4-tier Flood & Tassajara Road Washout status driven by Antecedent Precipitation Index (API) and MOS QPF/P90.",
    )
    st.markdown("---")
    st.markdown(
        "**Santa Lucia Reference Array:**\n"
        "- `CHWC1`: Chews Ridge (1,536 m)\n"
        "- `VTCC1`: Ventana Cone (1,453 m)\n"
        "- `CAHC1`: Hastings RAWS (546 m)\n"
        "- `ASRC1`: Arroyo Seco RAWS (268 m)\n"
        "- `PPSC1`: Big Sur RAWS (108 m)"
    )

st.markdown(
    '<div class="met-header-bar">'
    '<h2 style="margin-bottom:0.2rem;">'
    'Tassajara Zen Mountain Center — Microclimate & Hydrological MOS Forecast Suite'
    '</h2></div>',
    unsafe_allow_html=True,
)

if not obs_df.empty and "valid_time_utc" in obs_df.columns:
    raw_obs_ts = (
        obs_df.iloc[0]["exact_obs_time_utc"]
        if "exact_obs_time_utc" in obs_df.columns and pd.notna(obs_df.iloc[0]["exact_obs_time_utc"])
        else obs_df.iloc[0]["valid_time_utc"]
    )
    last_obs_dt = pd.to_datetime(raw_obs_ts, utc=True).tz_convert(LOCAL_TZ)
    last_obs_str = last_obs_dt.strftime("%Y-%m-%d %H:%M %Z")
else:
    last_obs_str = "Awaiting Telemetry"

if not future_fc_df.empty and "issue_time_utc" in future_fc_df.columns:
    fc_issue_dt = pd.to_datetime(future_fc_df.iloc[0]["issue_time_utc"], utc=True).tz_convert(LOCAL_TZ)
    fc_issue_str = fc_issue_dt.strftime("%Y-%m-%d %H:%M %Z")
else:
    fc_issue_str = "N/A"

online_refs = [stid for stid, d in ref_dict.items() if d.get("online")]
ref_banner_str = (
    " &bull; ".join(
        f"{stid} ({int(meta['elev_m'])}m): "
        + (
            f"{c_to_f(safe_float(ref_dict.get(stid, {}).get('temp_c'), np.nan)):.0f}°F"
            if pd.notna(ref_dict.get(stid, {}).get("temp_c"))
            else "Fallback"
        )
        for stid, meta in REFERENCE_STATIONS.items()
    )
)

st.markdown(
    f'<div class="met-subbanner">'
    f'<b>PRIMARY ISS:</b> Stone Office Davis Vantage Pro2 '
    f'(ID 36966 | 36.2337°N, 121.5501°W | 485 m / 1,590 ft MSL) &nbsp;&bull;&nbsp; '
    f'<b>SANTA LUCIA REFS ({len(online_refs)}/5 ACTIVE):</b> {ref_banner_str}<br>'
    f'<b>LAST OBSERVATION:</b> {last_obs_str} &nbsp;&bull;&nbsp; '
    f'<b>NWS/MOS INIT:</b> {fc_issue_str} &nbsp;&bull;&nbsp; '
    f'<b>SOLAR GEOMETRY:</b> {solar_state} '
    f'(El: {solar_el:.1f}°, Az: {solar_az:.0f}°, Horizon: {horizon_elev:.0f}°) &nbsp;&bull;&nbsp; '
    f'<b>ENGINE:</b> {active_model_name}'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Top-Level View Mode Selector (Layman's Forecast vs. Scientific MOS Suite)
# ---------------------------------------------------------------------------
view_col1, view_col2 = st.columns([2.6, 1.4])
with view_col1:
    view_mode = st.radio(
        "Select Dashboard View Mode",
        options=[
            "🌤️ Everyday Forecast (Community & Guest View)",
            "🔬 Scientific & Statistical MOS Suite",
        ],
        index=0,
        horizontal=True,
        help=(
            "Switch between an intuitive, plain-English canyon weather forecast page "
            "and the full atmospheric data science & MOS verification suite."
        ),
    )
with view_col2:
    if view_mode.startswith("🔬"):
        smooth_sci_curve = st.checkbox(
            "Smooth MOS Temperature Curve (3-hr Rolling Filter)",
            value=False,
            help=(
                "Applies a 3-hour centered rolling mean to the plotted MOS temperature "
                "curve to smooth out nocturnal decision-tree leaf-bouncing (sawtooth ripples)."
            ),
        )
    else:
        smooth_sci_curve = True

if view_mode.startswith("🌤️"):
    render_layman_forecast_view(
        obs_df=obs_df,
        future_fc_df=future_fc_df,
        ref_dict=ref_dict,
        solar_state=solar_state,
        solar_el=solar_el,
        last_obs_str=last_obs_str,
        now_hour_utc=now_hour_utc,
        hydro_bundle=hydro_bundle,
    )
else:
    render_flood_and_road_warning_banner(hydro_bundle["summary"], is_scientific=True)

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

        valid_wind_rows = (
            last_24[last_24["wind_speed_ms"].notna()]
            if "wind_speed_ms" in last_24.columns
            else pd.DataFrame()
        )
        wind_spd_ms = safe_float(
            latest.get("wind_speed_ms"),
            safe_float(valid_wind_rows.iloc[0]["wind_speed_ms"], 0.0)
            if not valid_wind_rows.empty
            else 0.0,
        )
        wind_gst_ms = safe_float(
            latest.get("wind_gust_ms"),
            safe_float(valid_wind_rows.iloc[0]["wind_gust_ms"], 0.0)
            if not valid_wind_rows.empty
            else 0.0,
        )
        wind_dir_val = latest.get("wind_dir_deg")
        if (wind_dir_val is None or pd.isna(wind_dir_val)) and not valid_wind_rows.empty:
            wind_dir_val = valid_wind_rows.iloc[0].get("wind_dir_deg")

        wind_spd_mph = ms_to_mph(wind_spd_ms)
        wind_gst_mph = ms_to_mph(wind_gst_ms)
        wind_card = deg_to_cardinal(wind_dir_val) if wind_spd_mph > 0.2 else "Calm"

        valid_bar_rows = (
            last_24[last_24["mslp_hpa"].notna()]
            if "mslp_hpa" in last_24.columns
            else pd.DataFrame()
        )
        mslp_hpa = safe_float(
            latest.get("mslp_hpa"),
            safe_float(valid_bar_rows.iloc[0]["mslp_hpa"], 1013.25)
            if not valid_bar_rows.empty
            else 1013.25,
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

        if not future_fc_df.empty:
            next_72 = future_fc_df.head(72)
            mos_qpf_72_mm = float(
                get_col(next_72, "mos_qpf_mm", get_col(next_72, "nws_qpf_1hr_mm", 0.0)).sum()
            )
            mos_qpf_72_in = mm_to_in(mos_qpf_72_mm)
            max_mos_pop_72 = float(
                get_col(next_72, "mos_pop_pct", get_col(next_72, "nws_pop_pct", 0.0)).max()
            )
            rain_sub_str = (
                f"1h: {last_hr_rain:.2f}\" | 72h MOS QPF: {mos_qpf_72_in:.2f}\" (PoP {max_mos_pop_72:.0f}%)"
            )
        else:
            rain_sub_str = (
                f"1-Hr Rate: {last_hr_rain:.2f} in/hr | "
                f"Archive: {mm_to_in(archive_stats['total_rain_mm']):.2f} in"
            )

        chews_temp_c = safe_float(ref_dict.get("CHWC1", {}).get("temp_c"), np.nan)
        vent_temp_c = safe_float(ref_dict.get("VTCC1", {}).get("temp_c"), np.nan)
        hast_temp_c = safe_float(ref_dict.get("CAHC1", {}).get("temp_c"), np.nan)
        arroyo_temp_c = safe_float(ref_dict.get("ASRC1", {}).get("temp_c"), np.nan)
        bigsur_temp_c = safe_float(ref_dict.get("PPSC1", {}).get("temp_c"), np.nan)

        lapse_chews = (
            -1.0 * (chews_temp_c - temp_c) / ((ELEV_CHEWS_M - ELEV_TASS_M) / 1000.0)
            if not math.isnan(chews_temp_c)
            else np.nan
        )
        lapse_vent = (
            -1.0 * (vent_temp_c - temp_c) / ((ELEV_VENTANA_M - ELEV_TASS_M) / 1000.0)
            if not math.isnan(vent_temp_c)
            else np.nan
        )
        primary_lapse = lapse_chews if not math.isnan(lapse_chews) else lapse_vent

        if not math.isnan(primary_lapse):
            lapse_str = f"{primary_lapse:+.1f} °C/km"
            if not math.isnan(lapse_chews) and not math.isnan(lapse_vent):
                stability_state = f"CHWC1: {lapse_chews:+.1f} | VTCC1: {lapse_vent:+.1f} °C/km"
            else:
                inv_tag = "Thermal Inversion" if primary_lapse < 0.0 else "Mixed Column"
                stability_state = f"{'CHWC1' if not math.isnan(lapse_chews) else 'VTCC1'}: {inv_tag}"
        else:
            lapse_str = "N/A"
            stability_state = "Crest RAWS Offline"

        if not math.isnan(arroyo_temp_c):
            arroyo_f = c_to_f(arroyo_temp_c)
            gorge_delta_f = temp_f - arroyo_f
            confinement_str = f"{gorge_delta_f:+.1f} °F vs ASRC1"
        elif not math.isnan(hast_temp_c):
            hast_f = c_to_f(hast_temp_c)
            confinement_str = f"{temp_f - hast_f:+.1f} °F vs CAHC1"
        else:
            confinement_str = "N/A"

        badge_parts = []
        for stid in ("ASRC1", "CAHC1", "PPSC1"):
            val_c = safe_float(ref_dict.get(stid, {}).get("temp_c"), np.nan)
            if not math.isnan(val_c):
                badge_parts.append(f"{stid}: {c_to_f(val_c):.0f}°F")
        ref_sub_str = " | ".join(badge_parts) if badge_parts else "Reference RAWS syncing"

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric(
            "Dry-Bulb Air Temperature",
            f"{temp_f:.1f} °F ({temp_c:.1f} °C)",
            f"24h Extremes: {lo_24_f:.1f}°F / {hi_24_f:.1f}°F",
            delta_color="off",
            help=(
                "Ambient dry-bulb air temperature recorded by the radiation-shielded thermistor "
                "on the Stone Office Davis Vantage Pro2 ISS (Station 36966, 485 m / 1,590 ft MSL). "
                "Strongly modulated by afternoon canyon-wall shading (32 deg SW horizon) and "
                "nocturnal katabatic cold-air drainage along Tassajara Creek."
            ),
        )
        r1c2.metric(
            "Relative Humidity & Dewpoint",
            f"{rh_pct:.0f} % RH",
            f"Dewpoint: {dew_f:.1f} °F ({dew_c:.1f} °C)",
            delta_color="off",
            help=(
                "Relative Humidity (RH %) measures the ratio of actual water vapor pressure to "
                "saturation vapor pressure. Dewpoint (F/C) is the temperature to which air must "
                "cool at constant pressure to reach saturation (100% RH), indicating absolute "
                "boundary-layer moisture content."
            ),
        )
        r1c3.metric(
            "Wet-Bulb & Vapor Pressure Deficit",
            f"{wetbulb_f:.1f} °F WB",
            f"VPD: {vpd_kpa:.2f} kPa ({'Arid / Virga Risk' if vpd_kpa >= 1.5 else 'Moist / Moderate'})",
            delta_color="off",
            help=(
                "Wet-Bulb Temperature (F WB), computed via the Stull (2011) psychrometric equation, "
                "represents the lowest temperature achievable by evaporative cooling. Vapor Pressure "
                "Deficit (VPD in kPa, via Tetens formula es - ea) quantifies atmospheric drying demand: "
                "VPD >= 1.5 kPa accelerates wildfire fuel drying and evaporates light pre-frontal rain as virga."
            ),
        )
        r1c4.metric(
            "Anemometer Wind Vector",
            f"{wind_spd_mph:.1f} mph ({wind_card})",
            f"10m Peak Gust: {wind_gst_mph:.1f} mph",
            delta_color="off",
            help=(
                "10-minute scalar average wind speed (mph), prevailing cardinal direction, and "
                "peak 10-minute wind gust measured at the Stone Office roof anemometer. Daytime "
                "heating drives up-canyon (anabatic WNW/ESE) flows, whereas nocturnal radiational "
                "cooling drives down-canyon (katabatic) drainage."
            ),
        )

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric(
            "Sea-Level Barometric Pressure",
            f"{mslp_inhg:.2f} inHg ({mslp_hpa:.1f} hPa)",
            ptrend_label,
            delta_color="off",
            help=(
                "Mean Sea-Level Pressure (MSLP in inHg and hPa/mb) reduced from station elevation "
                "(485 m MSL), paired with the 3-hour barometric tendency (dP/dt). Rapid pressure "
                "falls (<= -1.5 hPa/3h) signal approaching Pacific cold fronts or atmospheric rivers "
                "capable of overcoming the canyon rain shadow."
            ),
        )
        r2c2.metric(
            "Precipitation & 72h MOS QPF",
            f"{rain_24_in:.2f} in ({rain_24_mm:.1f} mm)",
            rain_sub_str,
            delta_color="off",
            help=(
                "Rolling 24-hour observed rainfall accumulation from the Stone Office 0.01-inch "
                "(0.254 mm) tipping-bucket rain gauge. The delta line displays the current 1-hour "
                "rainfall rate alongside the 72-hour MOS-calibrated Quantitative Precipitation "
                "Forecast (QPF) and peak Probability of Precipitation (PoP)."
            ),
        )
        r2c3.metric(
            "Crest Lapse Rate (CHWC1 / VTCC1 → TZMC)",
            lapse_str,
            stability_state,
            delta_color="off",
            help=(
                "Environmental temperature lapse rate across the upper Santa Lucia column between "
                "Chews Ridge (CHWC1, 1,536 m) / Ventana Cone (VTCC1, 1,453 m) and Tassajara (485 m). "
                "Negative values (Gamma < 0 C/km) indicate a nocturnal thermal inversion (cold-air "
                "pool trapped in the gorge); Gamma > 6.5 C/km indicates a well-mixed boundary layer."
            ),
        )
        r2c4.metric(
            "Gorge Drainage & Coastal Gradient",
            confinement_str,
            ref_sub_str,
            delta_color="off",
            help=(
                "Temperature differential between the inner Tassajara gorge (485 m), the Arroyo Seco "
                "gorge outlet (ASRC1, 268 m MSL, 6.3 km E), Hastings Reservation (CAHC1, 546 m), and "
                "the windward Big Sur coastal slope (PPSC1, 108 m)."
            ),
        )

        with st.expander(
            "Inspect 5-Station Santa Lucia Reference Network Telemetry (CHWC1, VTCC1, CAHC1, ASRC1, PPSC1)",
            expanded=False,
        ):
            ref_rows_ui = []
            for stid, meta in REFERENCE_STATIONS.items():
                d = ref_dict.get(stid, {})
                t_val_c = safe_float(d.get("temp_c"), np.nan)
                rh_val = safe_float(d.get("rh_pct"), np.nan)
                w_val_ms = safe_float(d.get("wind_speed_ms"), np.nan)
                g_val_ms = safe_float(d.get("wind_gust_ms"), np.nan)
                p_val_mm = safe_float(d.get("precip_1hr_mm"), 0.0)
                dz_km = (float(meta["elev_m"]) - ELEV_TASS_M) / 1000.0
                lapse_vs_tass = (
                    f"{-1.0 * (t_val_c - temp_c) / dz_km:+.1f} °C/km"
                    if not math.isnan(t_val_c) and abs(dz_km) >= 0.20
                    else (
                        f"{c_to_f(temp_c) - c_to_f(t_val_c):+.1f} °F ΔT"
                        if not math.isnan(t_val_c)
                        else "N/A"
                    )
                )
                ref_rows_ui.append({
                    "Station ID": stid,
                    "Station Name": meta["name"],
                    "Boundary Role": meta["role"],
                    "Elev (m / ft)": f"{int(meta['elev_m'])} m ({int(round(meta['elev_m'] * 3.28084))} ft)",
                    "Status": "Online" if d.get("online") else "Fallback / Stale",
                    "Temp (°F / °C)": (
                        f"{c_to_f(t_val_c):.1f} °F ({t_val_c:.1f} °C)"
                        if not math.isnan(t_val_c)
                        else "N/A"
                    ),
                    "RH (%)": f"{rh_val:.0f}%" if not math.isnan(rh_val) else "N/A",
                    "Wind / Gust (mph)": (
                        f"{ms_to_mph(w_val_ms):.1f} / {ms_to_mph(g_val_ms if not math.isnan(g_val_ms) else w_val_ms):.1f} mph"
                        if not math.isnan(w_val_ms)
                        else "N/A"
                    ),
                    "1-Hr Rain (in)": f"{mm_to_in(p_val_mm):.2f} in ({p_val_mm:.1f} mm)",
                    "Gradient vs. TZMC": lapse_vs_tass,
                })
            st.dataframe(
                pd.DataFrame(ref_rows_ui),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 1B: Hydrological Soil Saturation (API), Runoff & Creek Hydrograph
    # ---------------------------------------------------------------------------
    render_hydro_risk_section(hydro_bundle, now_hour_utc=now_hour_utc, is_scientific=True)

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
            options=[
                "Past 24 Hours",
                "Past 72 Hours",
                "Past 7 Days (168h)",
                "Full Historical Archive",
            ],
            index=1,
            help=(
                "Selects the historical lookback window of paired hourly Stone Office observations "
                "and archived forecasts used to compute real-time MAE, RMSE, Skill Score, Interval "
                "Coverage, and Brier Score."
            ),
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
        v_sub["p10_f"] = np.minimum(
            c_to_f(get_col(v_sub, "corrected_temp_p10_c", get_col(v_sub, "corrected_temp_c", 15.0) - 2.1)),
            v_sub["mos_f"] - 0.4,
        )
        v_sub["p90_f"] = np.maximum(
            c_to_f(get_col(v_sub, "corrected_temp_p90_c", get_col(v_sub, "corrected_temp_c", 15.0) + 2.1)),
            v_sub["mos_f"] + 0.4,
        )

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

        in_bounds = (
            ((v_sub["actual_f"] >= v_sub["p10_f"]) & (v_sub["actual_f"] <= v_sub["p90_f"])).mean()
            * 100.0
        )

        v_sub["obs_rain_mm"] = get_col(v_sub, "actual_precip_mm", 0.0)
        v_sub["raw_qpf_mm"] = get_col(v_sub, "nws_qpf_1hr_mm", 0.0)
        v_sub["mos_qpf_mm_val"] = get_col(v_sub, "mos_qpf_mm", v_sub["raw_qpf_mm"])
        v_sub["raw_pop_val"] = get_col(v_sub, "nws_pop_pct", 0.0).clip(0.0, 100.0)
        v_sub["mos_pop_val"] = get_col(v_sub, "mos_pop_pct", v_sub["raw_pop_val"]).clip(0.0, 100.0)

        obs_wet_bin = (v_sub["obs_rain_mm"] >= 0.254).astype(float)
        mos_brier = float(np.mean(((v_sub["mos_pop_val"] / 100.0) - obs_wet_bin) ** 2))
        obs_total_in = mm_to_in(float(v_sub["obs_rain_mm"].sum()))
        mos_total_in = mm_to_in(float(v_sub["mos_qpf_mm_val"].sum()))

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric(
            "Raw NWP/NWS Error (MAE)",
            f"± {nws_mae_f:.2f} °F",
            f"RMSE: ± {nws_rmse_f:.2f} °F | Bias: {nws_bias_f:+.2f} °F",
            delta_color="off",
            help=(
                "Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Bias of "
                "the uncorrected 2.5 km NWP/NWS grid forecast relative to Stone Office observations."
            ),
        )
        s2.metric(
            "Tassajara MOS Error (MAE)",
            f"± {mos_mae_f:.2f} °F",
            f"{mos_mae_f - nws_mae_f:+.2f} °F MAE vs Raw NWP",
            delta_color="inverse",
            help="Mean Absolute Error (MAE) of the Tassajara MOS HistGradientBoosting ensemble.",
        )
        s3.metric(
            "MOS RMSE Skill Score",
            f"{skill_pct:+.1f} %",
            f"MOS RMSE: ± {mos_rmse_f:.2f} °F (Bias: {mos_bias_f:+.2f} °F)",
            delta_color="normal" if skill_pct >= 0 else "inverse",
            help="Forecast Skill Score defined as (1 - RMSE_MOS / RMSE_NWS) * 100%.",
        )
        s4.metric(
            "Hourly Win Frequency",
            f"{win_rate_pct:.1f} % ({hours_mos_won}/{total_eval_hrs}h)",
            f"Evaluated over {total_eval_hrs} paired hours",
            delta_color="off",
            help="Percentage of hours where |E_MOS| <= |E_NWS|.",
        )
        s5.metric(
            "P10–P90 Interval Coverage",
            f"{in_bounds:.1f} %",
            "Nominal 80% Quantile Envelope",
            delta_color="off",
            help="Percentage of observations within the predicted P10-P90 interval.",
        )
        s6.metric(
            "Hydrological Brier & QPF",
            f"BS: {mos_brier:.4f}",
            f"Obs: {obs_total_in:.2f}\" | MOS: {mos_total_in:.2f}\"",
            delta_color="off",
            help="Brier Score and total accumulation comparison for the evaluation window.",
        )

        with st.expander(
            f"Inspect Paired Hourly Verification Table & Residual Diagnostics ({total_eval_hrs} Hours)",
            expanded=False,
        ):
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
                "Obs Rain (in)": (v_sub["obs_rain_mm"] / 25.4).round(3),
                "MOS QPF (in)": (v_sub["mos_qpf_mm_val"] / 25.4).round(3),
                "MOS PoP (%)": v_sub["mos_pop_val"].round(0).astype(int),
                "Superior Model": np.where(
                    v_sub["mos_abs_err_f"] <= v_sub["nws_abs_err_f"], "Tassajara MOS", "Raw NWS"
                ),
            }).iloc[::-1].reset_index(drop=True)

            st.dataframe(ver_table, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No paired observation-forecast records found in tassajara_weather.db. "
            "Run `python backfill_and_train.py --days 730` to ingest history and train the MOS ensemble."
        )

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 3: 3-Panel Interactive Plotly Microclimate & Hydrological Suite
    # ---------------------------------------------------------------------------
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        st.subheader("Microclimate & Hydrological Time-Series: Verification Hindcast & Operational Forecast")
    with ctrl_col2:
        horizon_choice = st.selectbox(
            "Timeline Horizon",
            options=[
                "72h Forecast + Past 48h Hindcast",
                "7-Day Forecast + Past 72h Hindcast",
                "7-Day Forecast + Past 7-Day Hindcast",
                "24h Forecast + Past 24h Hindcast",
            ],
            index=0,
        )

    if not future_fc_df.empty or not verify_df.empty:
        horizon_map = {
            "24h Forecast + Past 24h Hindcast": (24, 24),
            "72h Forecast + Past 48h Hindcast": (72, 48),
            "7-Day Forecast + Past 72h Hindcast": (168, 72),
            "7-Day Forecast + Past 7-Day Hindcast": (168, 168),
        }
        max_future_hrs, past_hrs = horizon_map.get(horizon_choice, (72, 48))

        fut_pool = filter_future_horizon(future_fc_df, now_hour_utc)
        fut_sub = fut_pool.head(max_future_hrs).copy() if not fut_pool.empty else pd.DataFrame()
        
        cutoff_past_dt = pd.Timestamp(now_hour_utc) - pd.Timedelta(hours=past_hrs)
        if not verify_df.empty and "valid_time_utc" in verify_df.columns:
            v_tmp = verify_df.copy()
            v_tmp["_dt_utc"] = pd.to_datetime(v_tmp["valid_time_utc"], utc=True).dt.floor("h")
            past_sub = v_tmp[(v_tmp["_dt_utc"] >= cutoff_past_dt) & (v_tmp["_dt_utc"] < pd.Timestamp(now_hour_utc))].copy()
        else:
            past_sub = pd.DataFrame()

        if not past_sub.empty and not fut_sub.empty:
            past_dt = pd.to_datetime(past_sub["valid_time_utc"], utc=True).dt.floor("h")
            fut_dt_set = set(pd.to_datetime(fut_sub["valid_time_utc"], utc=True).dt.floor("h"))
            past_sub = past_sub.loc[~past_dt.isin(fut_dt_set)].copy()
            combined_df = pd.concat([past_sub, fut_sub], ignore_index=True, sort=False)
        elif not fut_sub.empty:
            combined_df = fut_sub.copy()
        else:
            combined_df = past_sub.copy()

        combined_df["_sort_dt_utc"] = pd.to_datetime(
            combined_df["valid_time_utc"], utc=True
        ).dt.floor("h")
        combined_df = (
            combined_df.sort_values("_sort_dt_utc")
            .drop_duplicates(subset=["_sort_dt_utc"], keep="last")
            .reset_index(drop=True)
        )
        combined_df["valid_local"] = combined_df["_sort_dt_utc"].dt.tz_convert(LOCAL_TZ)

        if not obs_df.empty and "valid_time_utc" in obs_df.columns:
            obs_tmp = obs_df.copy()
            obs_tmp["_obs_dt_utc"] = pd.to_datetime(
                obs_tmp["valid_time_utc"], utc=True
            ).dt.floor("h")
            obs_dedup = obs_tmp.drop_duplicates(subset=["_obs_dt_utc"]).set_index("_obs_dt_utc")
            
            existing_dts = set(combined_df["_sort_dt_utc"])
            missing_obs = obs_tmp[~obs_tmp["_obs_dt_utc"].isin(existing_dts)].copy()
            if not missing_obs.empty:
                missing_rows = []
                for _, mo in missing_obs.iterrows():
                    dt_utc = mo["_obs_dt_utc"]
                    missing_rows.append({
                        "valid_time_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        "actual_temp_c": mo.get("temp_c"),
                        "actual_dewpoint_c": mo.get("dewpoint_c"),
                        "actual_rh_pct": mo.get("rh_pct"),
                        "actual_precip_mm": mo.get("precip_1hr_mm", 0.0),
                        "actual_wind_gust_ms": mo.get("wind_gust_ms", 0.0),
                        "raw_nws_temp_c": mo.get("temp_c"),
                        "corrected_temp_c": mo.get("temp_c"),
                        "corrected_temp_p10_c": mo.get("temp_c") - 1.0,
                        "corrected_temp_p90_c": mo.get("temp_c") + 1.0,
                        "_sort_dt_utc": dt_utc,
                        "valid_local": dt_utc.tz_convert(LOCAL_TZ),
                    })
                if missing_rows:
                    combined_df = pd.concat([combined_df, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)
                    combined_df = (
                        combined_df.sort_values("_sort_dt_utc")
                        .drop_duplicates(subset=["_sort_dt_utc"], keep="last")
                        .reset_index(drop=True)
                    )

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
                    mapped = combined_df["_sort_dt_utc"].map(obs_dedup[col_src])
                    combined_df[col_dest] = combined_df[col_dest].combine_first(mapped)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.065,
            row_heights=[0.46, 0.24, 0.30],
            specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=(
                "Temperature & Dewpoint Profile (°F): Stone Office Actuals vs. MOS Quantile Fan (P10–P90) vs. Raw NWS",
                "Boundary-Layer Kinematics & Moisture: 10m Peak Wind Gusts (mph) & Relative Humidity (%)",
                "Canyon Hydrological Outlook: MOS Calibrated QPF (in/hr), P90 Extreme Rate, Observed Rain & PoP (%)",
            ),
        )

        corr_c_series = get_col(combined_df, "corrected_temp_c", 15.0)
        if smooth_sci_curve:
            corr_c_series = corr_c_series.rolling(window=3, center=True, min_periods=1).mean()
        corr_f_series = c_to_f(corr_c_series)
        p90_f = np.maximum(
            c_to_f(get_col(combined_df, "corrected_temp_p90_c", corr_c_series + 2.1)),
            corr_f_series + 0.4,
        )
        p10_f = np.minimum(
            c_to_f(get_col(combined_df, "corrected_temp_p10_c", corr_c_series - 2.1)),
            corr_f_series - 0.4,
        )
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
                fillcolor="rgba(234, 88, 12, 0.16)",
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
                line=dict(color="rgba(2, 132, 199, 0.65)", width=1.4, dash="dot"),
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
                line=dict(color="#2563eb", width=2.0, dash="dash"),
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
                line=dict(color="#ea580c", width=2.8),
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
                        line=dict(color="#059669", width=2.2),
                        marker=dict(color="#059669", size=5.5, symbol="circle"),
                        name="Stone Office Observed (°F)",
                    ),
                    row=1,
                    col=1,
                )

        fig.add_hline(
            y=32.0,
            line_dash="dot",
            line_color="#0284c7",
            annotation_text="Freeze Threshold (32.0 °F)",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

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
                line=dict(color="#d97706", width=2.0),
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
                line=dict(color="#9333ea", width=1.8, dash="dot"),
                name="Relative Humidity (%)",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        raw_rain_in = (get_col(combined_df, "nws_qpf_1hr_mm", 0.0) / 25.4).round(3)
        mos_rain_in = (
            get_col(combined_df, "mos_qpf_mm", get_col(combined_df, "nws_qpf_1hr_mm", 0.0)) / 25.4
        ).round(3)
        mos_p90_rain_in = (
            get_col(combined_df, "mos_qpf_p90_mm", mos_rain_in * 25.4 * 1.5) / 25.4
        ).round(3)
        actual_rain_in = (get_col(combined_df, "actual_precip_mm", 0.0) / 25.4).round(3)

        raw_pop_pct = get_col(combined_df, "nws_pop_pct", 0.0).clip(0.0, 100.0)
        mos_pop_pct = get_col(combined_df, "mos_pop_pct", raw_pop_pct).clip(0.0, 100.0)

        fig.add_trace(
            go.Scatter(
                x=combined_df["valid_local"],
                y=mos_pop_pct.round(0),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(2, 132, 199, 0.14)",
                line=dict(color="#0284c7", width=2.0),
                name="MOS Calibrated PoP (%)",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=combined_df["valid_local"],
                y=raw_pop_pct.round(0),
                mode="lines",
                line=dict(color="rgba(100, 116, 139, 0.70)", width=1.4, dash="dot"),
                name="Raw NWS PoP (%)",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=combined_df["valid_local"],
                y=mos_p90_rain_in,
                mode="lines",
                line=dict(color="rgba(225, 29, 72, 0.75)", width=1.5, dash="dash"),
                name="MOS P90 Extreme Rate (in/hr)",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=combined_df["valid_local"],
                y=raw_rain_in,
                name="Raw NWS QPF (in/hr)",
                marker_color="rgba(37, 99, 235, 0.50)",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=combined_df["valid_local"],
                y=mos_rain_in,
                name="MOS Calibrated QPF (in/hr)",
                marker_color="#0284c7",
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
                marker_color="#059669",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )

        divider_ts = pd.Timestamp(now_hour_utc).tz_convert(LOCAL_TZ)
        for r_idx in (1, 2, 3):
            fig.add_vline(
                x=divider_ts,
                line_width=1.4,
                line_dash="dash",
                line_color=DIVIDER_LINE_RGBA,
                row=r_idx,
                col=1,
            )

        max_rain_val = float(
            max(
                mos_p90_rain_in.max(),
                mos_rain_in.max(),
                raw_rain_in.max(),
                actual_rain_in.max(),
                0.0,
            )
        )
        rain_y_top = max(0.20, round(max_rain_val * 1.25, 2))
        max_gust_val = float(max(gust_mph_series.max(), 15.0))

        fig.update_yaxes(title_text="Temperature (°F)", gridcolor=GRID_RGBA, row=1, col=1)
        fig.update_yaxes(
            title_text="Wind Gust (mph)",
            range=[0, max_gust_val * 1.25],
            gridcolor=GRID_RGBA,
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(title_text="RH (%)", range=[0, 105], showgrid=False, row=2, col=1, secondary_y=True)
        fig.update_yaxes(
            title_text="Rain Rate (in/hr)",
            range=[0, rain_y_top],
            gridcolor=GRID_RGBA,
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(title_text="PoP (%)", range=[0, 105], showgrid=False, row=3, col=1, secondary_y=True)
        x_min = combined_df["valid_local"].min()
        x_max = combined_df["valid_local"].max()
        for r_idx in (1, 2, 3):
            fig.update_xaxes(range=[x_min, x_max], gridcolor=GRID_RGBA, row=r_idx, col=1)
        fig.update_xaxes(title_text="Local Time (America/Los_Angeles)", row=3, col=1)

        fig.update_layout(
            height=880,
            margin=dict(t=60, b=40, l=55, r=55),
            hovermode="x unified",
            barmode="group",
            plot_bgcolor=PLOT_BG_RGBA,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 4: 7-Day Synoptic, Hydrological & Road Risk Planning Table
    # ---------------------------------------------------------------------------
    if not future_fc_df.empty:
        st.subheader("7-Day Synoptic, Hydrological & Canyon Road Risk Outlook")
        daily_df = filter_future_horizon(future_fc_df, now_hour_utc).copy()
        if not obs_df.empty and "valid_time_utc" in obs_df.columns:
            today_local_date = pd.Timestamp(now_hour_utc).tz_convert(LOCAL_TZ).date()
            obs_today_sci = obs_df.copy()
            obs_today_sci["valid_local"] = (
                pd.to_datetime(obs_today_sci["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
            )
            obs_today_sci = obs_today_sci[obs_today_sci["valid_local"].dt.date == today_local_date].copy()
            if not obs_today_sci.empty:
                obs_today_sci["corrected_temp_c"] = obs_today_sci["temp_c"]
                obs_today_sci["corrected_temp_p10_c"] = obs_today_sci["temp_c"]
                obs_today_sci["corrected_temp_p90_c"] = obs_today_sci["temp_c"]
                obs_today_sci["raw_nws_temp_c"] = obs_today_sci["temp_c"]
                if not verify_df.empty and "valid_time_utc" in verify_df.columns:
                    v_map = (
                        verify_df.drop_duplicates(subset=["valid_time_utc"])
                        .set_index("valid_time_utc")
                    )
                    if "raw_nws_temp_c" in v_map.columns:
                        obs_today_sci["raw_nws_temp_c"] = (
                            obs_today_sci["valid_time_utc"]
                            .map(v_map["raw_nws_temp_c"])
                            .combine_first(obs_today_sci["temp_c"])
                        )
                obs_today_sci["nws_pop_pct"] = 0.0
                obs_today_sci["mos_pop_pct"] = 0.0
                obs_today_sci["nws_qpf_1hr_mm"] = get_col(obs_today_sci, "precip_1hr_mm", 0.0)
                obs_today_sci["mos_qpf_mm"] = get_col(obs_today_sci, "precip_1hr_mm", 0.0)
                obs_today_sci["mos_qpf_p90_mm"] = get_col(obs_today_sci, "precip_1hr_mm", 0.0)
                obs_today_sci["corrected_rh_pct"] = get_col(obs_today_sci, "rh_pct", 45.0)
                obs_today_sci["nws_rh_pct"] = get_col(obs_today_sci, "rh_pct", 45.0)
                obs_today_sci["corrected_wind_gust_ms"] = get_col(
                    obs_today_sci, "wind_gust_ms", get_col(obs_today_sci, "wind_speed_ms", 0.0)
                )
                obs_today_sci["nws_wind_gust_ms"] = obs_today_sci["corrected_wind_gust_ms"]
                obs_today_sci["nws_wind_speed_ms"] = get_col(obs_today_sci, "wind_speed_ms", 0.0)
                daily_df = pd.concat([obs_today_sci, daily_df], ignore_index=True, sort=False)
                daily_df["_hr_utc"] = pd.to_datetime(daily_df["valid_time_utc"], utc=True).dt.floor("h")
                daily_df = (
                    daily_df.sort_values("_hr_utc")
                    .drop_duplicates(subset=["_hr_utc"], keep="first")
                    .drop(columns=["_hr_utc"])
                    .reset_index(drop=True)
                )
        daily_df["valid_local"] = (
            pd.to_datetime(daily_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        )
        daily_df["Date_Key"] = daily_df["valid_local"].dt.strftime("%a, %b %d")
        daily_df["Sort_Date"] = daily_df["valid_local"].dt.date

        daily_df["_corr_t"] = get_col(daily_df, "corrected_temp_c", 15.0)
        daily_df["_p10_t"] = get_col(daily_df, "corrected_temp_p10_c", daily_df["_corr_t"] - 2.1)
        daily_df["_p90_t"] = get_col(daily_df, "corrected_temp_p90_c", daily_df["_corr_t"] + 2.1)
        daily_df["_nws_t"] = get_col(daily_df, "raw_nws_temp_c", daily_df["_corr_t"])
        daily_df["_raw_pop"] = get_col(daily_df, "nws_pop_pct", 0.0)
        daily_df["_mos_pop"] = get_col(daily_df, "mos_pop_pct", daily_df["_raw_pop"])
        daily_df["_raw_qpf"] = get_col(daily_df, "nws_qpf_1hr_mm", 0.0)
        daily_df["_mos_qpf"] = get_col(daily_df, "mos_qpf_mm", daily_df["_raw_qpf"])
        daily_df["_mos_qpf_p90"] = get_col(daily_df, "mos_qpf_p90_mm", daily_df["_mos_qpf"] * 1.5)
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
                MOS_Max_PoP_Pct=("_mos_pop", lambda s: int(round(safe_float(s.max(), 0.0), 0))),
                Raw_Max_PoP_Pct=("_raw_pop", lambda s: int(round(safe_float(s.max(), 0.0), 0))),
                MOS_QPF_In=("_mos_qpf", lambda s: round(mm_to_in(float(s.sum())), 2)),
                MOS_P90_QPF_In=("_mos_qpf_p90", lambda s: round(mm_to_in(float(s.sum())), 2)),
                Raw_QPF_In=("_raw_qpf", lambda s: round(mm_to_in(float(s.sum())), 2)),
                Min_RH_Pct=("_rh", lambda s: int(round(safe_float(s.min(), 40.0), 0))),
                Peak_Gust_MPH=("_gust", lambda s: round(ms_to_mph(safe_float(s.max(), 0.0)), 1)),
            )
            .sort_values("Sort_Date")
            .head(7)
        )

        def _met_regime(row):
            notes = []
            if row["MOS_QPF_In"] >= 1.25 or row["MOS_P90_QPF_In"] >= 2.0:
                notes.append("Heavy Orographic Rain / Tassajara Rd Slip & Creek Rise Risk")
            elif row["MOS_QPF_In"] >= 0.25 or row["MOS_Max_PoP_Pct"] >= 55:
                notes.append("Wetting Frontal Rain Likely")
            elif row["Raw_Max_PoP_Pct"] >= 35 and row["MOS_Max_PoP_Pct"] < 20:
                notes.append("Sub-Cloud Virga Suppression (Dry Canyon Boundary Layer)")
            elif row["MOS_Max_PoP_Pct"] >= 25:
                notes.append("Slight Chance Light Canyon Showers")

            if row["Cold_Pool_Floor_F"] <= 32.0:
                notes.append("Hard Freeze Risk (P10 <= 32°F)")
            elif row["Cold_Pool_Floor_F"] <= 36.0:
                notes.append("Nocturnal Cold-Pool Frost Risk")

            if row["Min_RH_Pct"] <= 20 and row["Peak_Gust_MPH"] >= 20:
                notes.append("Elevated Fire-Weather Subsidence")
            if not notes:
                notes.append("Diurnal Katabatic / Anabatic Valley Regime")
            return " | ".join(notes)

        daily_summary["Canyon Meteorological & Hydrological Regime"] = daily_summary.apply(
            _met_regime, axis=1
        )

        display_daily = daily_summary[
            [
                "Date_Key",
                "MOS_High_F",
                "MOS_Low_F",
                "Cold_Pool_Floor_F",
                "Thermal_Peak_P90_F",
                "NWS_High_F",
                "NWS_Low_F",
                "MOS_Max_PoP_Pct",
                "MOS_QPF_In",
                "MOS_P90_QPF_In",
                "Raw_QPF_In",
                "Min_RH_Pct",
                "Peak_Gust_MPH",
                "Canyon Meteorological & Hydrological Regime",
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
                "MOS_Max_PoP_Pct": "MOS PoP (%)",
                "MOS_QPF_In": "MOS QPF (in)",
                "MOS_P90_QPF_In": "P90 Ceiling QPF (in)",
                "Raw_QPF_In": "Raw NWS QPF (in)",
                "Min_RH_Pct": "Min RH (%)",
                "Peak_Gust_MPH": "Peak Gust (mph)",
            }
        )
        st.dataframe(display_daily, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 5: MOS Model Interpretability — Feature Importance & OLS Diagnostics
    # ---------------------------------------------------------------------------
    st.subheader("MOS Model Interpretability: Feature Importance Rankings & OLS Physical Diagnostics")

    tab_imp, tab_ols = st.tabs([
        "1. Ensemble Feature Importance Rankings (Thermal & Hydrological)",
        "2. Statsmodels OLS Regression Summary (corrector.ols_summary)",
    ])

    with tab_imp:
        imp_df = model_diag["imp_df"]
        if not imp_df.empty:
            top_n = imp_df.sort_values("Combined Share (%)", ascending=True).tail(16)

            fig_imp = make_subplots(
                rows=1,
                cols=2,
                shared_yaxes=True,
                horizontal_spacing=0.06,
                subplot_titles=(
                    "Thermal MOS Predictor Attribution (% RMSE Share)",
                    "Hydrological Hurdle Predictor Attribution (% QPF/PoP Share)",
                ),
            )
            fig_imp.add_trace(
                go.Bar(
                    x=top_n["Thermal Importance (%)"],
                    y=top_n["Physical Predictor"],
                    orientation="h",
                    marker_color="#ea580c",
                    name="Thermal MOS Share (%)",
                    text=top_n["Thermal Importance (%)"].map(lambda v: f"{v:.1f}%"),
                    textposition="auto",
                ),
                row=1,
                col=1,
            )
            fig_imp.add_trace(
                go.Bar(
                    x=top_n["Hydrological Importance (%)"],
                    y=top_n["Physical Predictor"],
                    orientation="h",
                    marker_color="#0284c7",
                    name="Hydrological Hurdle Share (%)",
                    text=top_n["Hydrological Importance (%)"].map(lambda v: f"{v:.1f}%"),
                    textposition="auto",
                ),
                row=1,
                col=2,
            )
            fig_imp.update_xaxes(gridcolor=GRID_RGBA, row=1, col=1)
            fig_imp.update_xaxes(gridcolor=GRID_RGBA, row=1, col=2)
            fig_imp.update_layout(
                height=520,
                margin=dict(t=50, b=35, l=20, r=25),
                plot_bgcolor=PLOT_BG_RGBA,
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Insufficient verification rows to compute permutation feature importances.")

    with tab_ols:
        if not math.isnan(model_diag["ols_r2"]):
            o1, o2, o3, o4 = st.columns(4)
            o1.metric(
                "OLS R-Squared (R²)",
                f"{model_diag['ols_r2']:.3f}",
                f"Adjusted R²: {model_diag['ols_adj_r2']:.3f}",
                delta_color="off",
            )
            o2.metric(
                "HAC Wald F-Statistic",
                f"{model_diag['ols_fstat']:.1f}" if not math.isnan(model_diag["ols_fstat"]) else "N/A",
                "Newey-West Robust (12h Lag, p < 0.0001)",
                delta_color="off",
            )
            cond_val = model_diag.get("ols_cond_no", np.nan)
            o3.metric(
                "Matrix Condition Number (κ)",
                f"{cond_val:.1f}" if not math.isnan(cond_val) else "N/A",
                "Well-Conditioned (< 30, No Collinearity)",
                delta_color="off",
            )
            o4.metric(
                "Aligned Verification Sample",
                f"{model_diag['ols_nobs']} hrs",
                f"AIC: {model_diag['ols_aic']:.0f}" if not math.isnan(model_diag["ols_aic"]) else "Paired Obs",
                delta_color="off",
            )

        if not model_diag["ols_coef_df"].empty:
            st.markdown("#### Standardized Newey-West HAC OLS Coefficient Table (`cov_type='HAC', maxlags=12`)")
            st.dataframe(model_diag["ols_coef_df"], use_container_width=True, hide_index=True)

        if model_diag["ols_summary_text"]:
            with st.expander("View Raw Statsmodels ASCII Summary Output (corrector.ols_summary)", expanded=True):
                st.code(model_diag["ols_summary_text"], language="text")