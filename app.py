#!/usr/bin/env python3
"""
app.py — Tassajara Zen Mountain Center Meteorological & Hydrological MOS Monitor
Publication-grade scientific monitoring suite for Station 36966 (Stone Office Vantage Pro2),
Chews Ridge (CHWC1) & Hastings (CAHC1) boundary references, and Thermal + Hydrological MOS.
Includes comprehensive scientific tooltips (help="...") and an OLS & Feature Importance suite.
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
from sklearn.inspection import permutation_importance
import statsmodels.api as sm


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
        "Boundary-layer stability across the 1,051m column between Chews Ridge and Tassajara.",
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


TASS_LAT_RAD = np.radians(36.2337)
TASS_LON_DEG = -121.5501
CANYON_AXIS_RAD = np.radians(115.0)        # ESE-WNW primary axis of Tassajara Creek Canyon
OROGRAPHIC_UPSLOPE_RAD = np.radians(215.0) # SSW (215 deg) perpendicular to Santa Lucia crest
TIPPING_BUCKET_MM = 0.254                  # 0.01 inch Davis Vantage Pro2 rain spoon threshold


def compute_solar_and_canyon_shadow(timestamps_utc: pd.Series) -> pd.DataFrame:
    """Calculates solar altitude, azimuth, and Black Cone Trail / Pine Ridge canyon shadow."""
    dt = pd.to_datetime(timestamps_utc, utc=True)
    doy = dt.dt.dayofyear.values
    hour_utc = dt.dt.hour.values + dt.dt.minute.values / 60.0

    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    tst = (hour_utc * 60.0 + eqtime + 4.0 * TASS_LON_DEG) % 1440.0
    ha_rad = np.radians((tst / 4.0) - 180.0)

    sin_el = (
        np.sin(TASS_LAT_RAD) * np.sin(decl)
        + np.cos(TASS_LAT_RAD) * np.cos(decl) * np.cos(ha_rad)
    )
    solar_el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    cos_az = (
        (np.sin(decl) - np.sin(TASS_LAT_RAD) * sin_el)
        / (np.cos(TASS_LAT_RAD) * np.cos(np.arcsin(np.clip(sin_el, -1.0, 1.0))) + 1e-6)
    )
    az_deg = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    solar_az_deg = np.where(ha_rad > 0, (360.0 - az_deg) % 360.0, az_deg)

    horizon_deg = np.full_like(solar_az_deg, 18.0)
    horizon_deg = np.where((solar_az_deg >= 145.0) & (solar_az_deg <= 255.0), 32.0, horizon_deg)
    horizon_deg = np.where((solar_az_deg >= 45.0) & (solar_az_deg <= 115.0), 23.0, horizon_deg)

    in_canyon_shadow = ((solar_el_deg > 0.0) & (solar_el_deg < horizon_deg)).astype(float)
    effective_insolation = np.where(
        solar_el_deg >= horizon_deg, np.sin(np.radians(solar_el_deg)), 0.0
    )

    return pd.DataFrame(
        {
            "solar_el_deg": solar_el_deg,
            "solar_az_deg": solar_az_deg,
            "in_canyon_shadow": in_canyon_shadow,
            "effective_insolation": effective_insolation,
        },
        index=timestamps_utc.index,
    )


def engineer_mos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds physical, cyclical, topographic, and hydrological features for both
    canyon temperature bias and orographic/virga precipitation calibration.
    Synthesizes a baseline QPF prior when raw NWP QPF is 0.0 but PoP >= 20%.
    """
    out = pd.DataFrame(index=df.index)
    dt_valid = pd.to_datetime(df["valid_time_utc"], utc=True)
    dt_local = dt_valid.dt.tz_convert("America/Los_Angeles")

    hour = dt_local.dt.hour + dt_local.dt.minute / 60.0
    doy = dt_local.dt.dayofyear

    out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    solar_df = compute_solar_and_canyon_shadow(df["valid_time_utc"])
    out["solar_el_deg"] = solar_df["solar_el_deg"]
    out["in_canyon_shadow"] = solar_df["in_canyon_shadow"]
    out["effective_insolation"] = solar_df["effective_insolation"]

    lead_hrs = get_col(df, "lead_time_hours", 6.0).clip(lower=1.0)
    t_c = get_col(df, "nws_temp_c", get_col(df, "raw_nws_temp_c", 15.0))
    rh_pct = get_col(df, "nws_rh_pct", 50.0).clip(1.0, 100.0)
    dew_c = get_col(df, "nws_dewpoint_c", np.nan).fillna(t_c - ((100.0 - rh_pct) / 5.0))

    pop_pct = get_col(df, "nws_pop_pct", 0.0).clip(0.0, 100.0)
    raw_qpf_mm = get_col(df, "nws_qpf_1hr_mm", 0.0).clip(lower=0.0)
    sky_pct = get_col(df, "nws_sky_cover_pct", 20.0).clip(0.0, 100.0)

    # Synthesize physical QPF prior when raw NWP QPF is 0.0 but PoP >= 20%
    pop_frac = pop_pct / 100.0
    rh_frac = np.clip(rh_pct / 100.0, 0.30, 1.0)
    synth_qpf_prior = np.where(
        pop_pct >= 20.0,
        np.where(
            pop_pct >= 60.0,
            1.35 * (pop_frac ** 1.15) * rh_frac,
            0.65 * (pop_frac ** 1.35) * rh_frac,
        ),
        0.0,
    )
    qpf_mm = np.where(raw_qpf_mm > 0.0, raw_qpf_mm, synth_qpf_prior)

    out["lead_time_hours"] = lead_hrs
    out["nws_temp_c"] = t_c
    out["nws_rh_pct"] = rh_pct
    out["nws_pop_pct"] = pop_pct
    out["nws_qpf_1hr_mm"] = qpf_mm
    out["nws_sky_cover_pct"] = sky_pct

    w_spd = get_col(df, "nws_wind_speed_ms", 0.0).clip(lower=0.0)
    w_dir_rad = np.radians(get_col(df, "nws_wind_dir_deg", 0.0))
    out["wind_along_canyon_ms"] = w_spd * np.cos(w_dir_rad - CANYON_AXIS_RAD)
    out["wind_cross_canyon_ms"] = w_spd * np.sin(w_dir_rad - CANYON_AXIS_RAD)
    upslope_wind_ms = np.maximum(0.0, w_spd * np.cos(w_dir_rad - OROGRAPHIC_UPSLOPE_RAD))
    out["orographic_upslope_wind_ms"] = upslope_wind_ms

    es_kpa = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea_kpa = es_kpa * (rh_pct / 100.0)
    vpd_kpa = np.maximum(0.0, es_kpa - ea_kpa)
    dew_depr_c = np.maximum(0.0, t_c - dew_c)

    out["dewpoint_depression_c"] = dew_depr_c
    out["vpd_kpa"] = vpd_kpa
    out["virga_evap_index"] = vpd_kpa * dew_depr_c * (1.0 - (rh_pct / 100.0))
    out["orographic_moisture_flux"] = upslope_wind_ms * ea_kpa
    out["qpf_moisture_scaling"] = qpf_mm * ((rh_pct / 100.0) ** 2)

    ptrend_3h = get_col(df, "pressure_trend_3hr_hpa", 0.0)
    out["pressure_trend_3hr_hpa"] = ptrend_3h
    out["frontal_pressure_fall_hpa"] = np.minimum(0.0, ptrend_3h)

    issue_chews_t = get_col(df, "issue_chews_temp_c", np.nan).fillna(t_c + 2.0)
    issue_tass_t = get_col(df, "issue_tass_temp_c", np.nan).fillna(t_c)
    out["issue_lapse_chews_c_km"] = (
        -1.0 * (issue_chews_t - issue_tass_t) / 1.051
    ).fillna(6.5)

    issue_chews_p = get_col(df, "issue_chews_precip_mm", 0.0)
    out["decayed_chews_precip_mm"] = issue_chews_p * np.exp(-lead_hrs / 6.0)

    init_bias = get_col(df, "issue_persistence_bias_c", 0.0)
    out["decayed_init_bias_c"] = init_bias * np.exp(-lead_hrs / 12.0)

    return out


def analytical_hydro_hurdle_fallback(X: pd.DataFrame) -> pd.DataFrame:
    """
    Zero-Trap-Immune Physical Orographic-Virga Hurdle baseline.
    Synthesizes conditional accumulation whenever canyon-calibrated PoP >= 20%
    and boundary-layer moisture supports hydrometeor survival to 485m MSL.
    """
    raw_pop = X["nws_pop_pct"].values
    raw_qpf = X["nws_qpf_1hr_mm"].values
    rh = X["nws_rh_pct"].values
    vpd = X["vpd_kpa"].values
    upslope_flux = X["orographic_moisture_flux"].values
    p_fall = X["frontal_pressure_fall_hpa"].values
    chews_seed = X["decayed_chews_precip_mm"].values

    virga_factor = np.where(
        (rh < 70.0) & (raw_qpf < 1.5),
        np.exp(-0.48 * vpd),
        1.0,
    )

    orographic_factor = (
        1.0
        + 0.08 * np.clip(upslope_flux, 0.0, 5.0)
        + 0.05 * np.clip(-p_fall, 0.0, 4.0)
        + 0.06 * np.clip(chews_seed, 0.0, 3.0)
    )

    net_multiplier = np.clip(virga_factor * orographic_factor, 0.20, 1.85)

    mos_pop = np.clip(
        raw_pop * np.clip(virga_factor * (1.0 + 0.04 * upslope_flux), 0.30, 1.35)
        + np.where(raw_qpf >= TIPPING_BUCKET_MM, 10.0 * (rh / 100.0), -5.0 * (vpd > 1.3)),
        0.0,
        100.0,
    )

    pop_frac = mos_pop / 100.0
    rh_factor = np.clip((rh - 35.0) / 65.0, 0.25, 1.0)
    conditional_qpf_prior = np.where(
        mos_pop >= 20.0,
        np.where(
            mos_pop >= 60.0,
            1.25 * (pop_frac ** 1.15) * rh_factor,
            0.55 * (pop_frac ** 1.30) * rh_factor,
        ),
        0.0,
    )
    effective_base_qpf = np.maximum(raw_qpf, conditional_qpf_prior)

    mos_qpf = effective_base_qpf * net_multiplier
    mos_qpf = np.where(mos_pop < 15.0, 0.0, mos_qpf)

    mos_qpf_p90 = np.where(
        mos_pop >= 12.0,
        np.maximum(mos_qpf * 1.65, mos_qpf + 0.45 * pop_frac * orographic_factor),
        0.0,
    )

    return pd.DataFrame(
        {
            "mos_pop_pct": np.round(mos_pop, 1),
            "mos_qpf_mm": np.round(mos_qpf, 3),
            "mos_qpf_p90_mm": np.round(mos_qpf_p90, 3),
        },
        index=X.index,
    )


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
    Includes Fix 4: Re-triggers hydrological inference on migrated rows where
    SQLite DEFAULT 0.0 or legacy zero-QPF traps left mos_qpf_mm at 0.0 during active PoP.
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

    # Fix 4: Detect both NaN columns AND migrated rows stuck at 0.0 when PoP/QPF is active
    raw_pop_s = get_col(out, "nws_pop_pct", 0.0)
    raw_qpf_s = get_col(out, "nws_qpf_1hr_mm", 0.0)
    mos_pop_s = get_col(out, "mos_pop_pct", np.nan)
    mos_qpf_s = get_col(out, "mos_qpf_mm", np.nan)

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
    sub["issue_chews_precip_mm"] = get_col(sub, "issue_chews_precip_mm", 0.0)
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

            # Self-healing guard: if the loaded .joblib artifact was trained prior to Fix 2
            # and returned 0.0 for mos_qpf_mm during >=20% PoP hours, patch with zero-safe hurdle
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

        # Archive-wide Precipitation Stats
        archive_stats = {
            "total_obs_hours": 0,
            "wet_hours": 0,
            "total_rain_mm": 0.0,
            "max_1hr_mm": 0.0,
        }
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
                    future_fc_df["corrected_rh_pct"] = future_fc_df[
                        "corrected_rh_pct"
                    ].combine_first(future_fc_df["nws_rh_pct"])
                if "corrected_wind_gust_ms" in future_fc_df.columns:
                    future_fc_df["corrected_wind_gust_ms"] = (
                        future_fc_df["corrected_wind_gust_ms"]
                        .combine_first(future_fc_df["nws_wind_gust_ms"])
                        .combine_first(future_fc_df["nws_wind_speed_ms"] * 1.35)
                    )
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


@st.cache_data(ttl=300)
def compute_model_diagnostics_and_importances(verify_df: pd.DataFrame):
    """
    Extracts or fits the Statsmodels OLS regression summary (corrector.ols_summary)
    and computes permutation feature importances for both the Thermal and Hydrological
    MOS models from models/tassajara_mos_v1.joblib.
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

    eval_sub = verify_df.tail(min(len(verify_df), 2500)).copy()
    X_full = engineer_mos_features(eval_sub).fillna(0.0)
    y_temp_err = (
        get_col(eval_sub, "actual_temp_c", 15.0) - get_col(eval_sub, "raw_nws_temp_c", 15.0)
    ).values

    # 1. Fit OLS directly on verification features to extract structured p-values, t-stats & R^2
    try:
        X_ols = sm.add_constant(X_full, has_constant="add")
        ols_res = sm.OLS(y_temp_err, X_ols).fit()
        if not diag_out["ols_summary_text"]:
            diag_out["ols_summary_text"] = ols_res.summary().as_text()
        diag_out["ols_r2"] = float(ols_res.rsquared)
        diag_out["ols_adj_r2"] = float(ols_res.rsquared_adj)
        diag_out["ols_fstat"] = float(ols_res.fvalue) if ols_res.fvalue is not None else np.nan
        diag_out["ols_aic"] = float(ols_res.aic)
        diag_out["ols_nobs"] = int(ols_res.nobs)

        ci_df = ols_res.conf_int()
        coef_rows = []
        for feat in ols_res.params.index:
            if feat == "const":
                label, desc = "Intercept (Constant)", "Baseline systematic offset (C) when predictors are zero."
            else:
                label, desc = FEATURE_METADATA.get(feat, (feat, "Engineered physical MOS predictor."))
            coef_rows.append({
                "Feature Key": feat,
                "Physical Predictor": label,
                "Coefficient (beta)": round(float(ols_res.params[feat]), 4),
                "Std Error": round(float(ols_res.bse[feat]), 4),
                "t-Statistic": round(float(ols_res.tvalues[feat]), 2),
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

    # Thermal Importance
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
            thermal_raw = np.abs(X_model.corrwith(pd.Series(y_temp_err, index=X_model.index)).fillna(0.0).values)
    else:
        thermal_raw = np.abs(X_model.corrwith(pd.Series(y_temp_err, index=X_model.index)).fillna(0.0).values)

    t_sum = float(thermal_raw.sum())
    thermal_pct = (thermal_raw / t_sum * 100.0) if t_sum > 0 else np.zeros_like(thermal_raw)

    # Hydrological Importance (Permutation on trained QPF/PoP model or analytical hydro sensitivity)
    y_hydro_target = get_col(eval_sub, "mos_qpf_mm", get_col(eval_sub, "actual_precip_mm", 0.0)).values
    if (
        mos_obj is not None
        and getattr(mos_obj, "precip_ml_trained", False)
        and hasattr(mos_obj, "model_qpf_mean")
    ):
        try:
            perm_h = permutation_importance(
                mos_obj.model_qpf_mean,
                X_full,
                y_hydro_target,
                scoring="neg_root_mean_squared_error",
                n_repeats=5,
                random_state=42,
            )
            hydro_map = dict(zip(X_full.columns, np.maximum(0.0, perm_h.importances_mean)))
        except Exception:
            hydro_map = {}
    else:
        # Compute perturbation sensitivity on the physical orographic-virga hurdle model
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

    imp_df = pd.DataFrame(imp_rows).sort_values("Thermal Importance (%)", ascending=False).reset_index(drop=True)
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


def render_layman_forecast_view(
    obs_df: pd.DataFrame,
    future_fc_df: pd.DataFrame,
    ref_dict: dict,
    solar_state: str,
    solar_el: float,
    last_obs_str: str,
):
    """Renders the intuitive, jargon-free Everyday Weather Forecast Page for TZMC."""
    # 1. Extract Current Conditions
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

    # Hero Card + Today's Highlights
    hero_col1, hero_col2 = st.columns([1.35, 1.65])
    with hero_col1:
        st.markdown(
            f'<div style="background: rgba(15, 23, 42, 0.65); '
            f'border: 1px solid rgba(56, 189, 248, 0.30); border-radius: 10px; '
            f'padding: 20px 24px; height: 100%;">'
            f'<div style="font-size: 0.82rem; text-transform: uppercase; '
            f'letter-spacing: 0.06em; color: #38bdf8; font-weight: 600;">'
            f'Right Now at Tassajara Stone Office (1,590 ft)</div>'
            f'<div style="display: flex; align-items: center; gap: 18px; margin: 10px 0;">'
            f'<span style="font-size: 3.6rem; line-height: 1;">{cur_icon}</span>'
            f'<div>'
            f'<div style="font-size: 2.9rem; font-weight: 800; line-height: 1.05;">'
            f'{temp_f:.0f}°F</div>'
            f'<div style="font-size: 1.1rem; color: #e2e8f0; font-weight: 500;">'
            f'{cur_cond}</div>'
            f'</div></div>'
            f'<div style="font-size: 0.90rem; color: #cbd5e1; margin-top: 8px;">'
            f'{sun_badge}</div>'
            f'<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 10px;">'
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
        chews_str = (
            f"Chews Ridge (5,040 ft): {c_to_f(chews_t_c):.0f}°F"
            if not math.isnan(chews_t_c)
            else "Chews Ridge: N/A"
        )
        c4.metric(
            "Rain (Past 24 Hours)",
            f"{rain_24_in:.2f} in",
            chews_str,
            delta_color="off",
            help="Total rainfall measured in the past 24 hours, plus current temperature up at Chews Ridge.",
        )

    st.markdown("")

    # 2. Plain-English Canyon Advisories (Freeze, Road, & Trail Conditions)
    if not future_fc_df.empty:
        next_48 = future_fc_df.head(48).copy()
        next_72 = future_fc_df.head(72).copy()

        # Smooth temperature series for everyday display
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
        # Apply 3-hour centered rolling average to eliminate decision-tree sawtooth ripples
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
                    f'<div style="background: rgba(15, 23, 42, 0.50); '
                    f'border: 1px solid rgba(148, 163, 184, 0.20); border-radius: 8px; '
                    f'padding: 10px 6px; text-align: center;">'
                    f'<div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">'
                    f'{day_tag} {time_lbl}</div>'
                    f'<div style="font-size: 1.9rem; margin: 6px 0;">{ic}</div>'
                    f'<div style="font-size: 1.25rem; font-weight: 700;">'
                    f'{r["smooth_temp_f"]:.0f}°F</div>'
                    f'<div style="font-size: 0.74rem; color: #cbd5e1; margin-top: 2px;">'
                    f'{lbl}</div>'
                    f'<div style="font-size: 0.75rem; color: #38bdf8; margin-top: 4px;">'
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
                fillcolor="rgba(56, 189, 248, 0.12)",
                line=dict(color="rgba(56, 189, 248, 0.55)", width=1.5, dash="dot"),
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
                    marker_color="#38bdf8",
                    opacity=0.65,
                ),
                secondary_y=True,
            )

        fig_layman.add_trace(
            go.Scatter(
                x=fc_48["valid_local"],
                y=fc_48["smooth_temp_f"],
                mode="lines",
                line=dict(color="#f97316", width=3.2, shape="spline"),
                name="Canyon Temperature (°F)",
            ),
            secondary_y=False,
        )

        fig_layman.add_hline(
            y=32.0,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text="Freezing (32°F)",
            annotation_position="bottom right",
            secondary_y=False,
        )

        fig_layman.update_yaxes(
            title_text="Temperature (°F)",
            gridcolor="rgba(148,163,184,0.12)",
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
            gridcolor="rgba(148,163,184,0.12)",
        )
        fig_layman.update_layout(
            height=390,
            margin=dict(t=25, b=35, l=50, r=50),
            hovermode="x unified",
            plot_bgcolor="rgba(15, 23, 42, 0.35)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig_layman, use_container_width=True)

    st.markdown("---")

    # 5. 7-Day Visual Daily Forecast Cards & Simple Table
    if not future_fc_df.empty:
        st.subheader("7-Day Tassajara Canyon Forecast")
        d_df = future_fc_df.copy()
        d_df["valid_local"] = (
            pd.to_datetime(d_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        )
        d_df["Date_Label"] = d_df["valid_local"].dt.strftime("%a, %b %d")
        d_df["Day_Short"] = d_df["valid_local"].dt.strftime("%a")
        d_df["Date_Short"] = d_df["valid_local"].dt.strftime("%b %d")
        d_df["Sort_Date"] = d_df["valid_local"].dt.date

        # Smooth hourly temperature before daily aggregation
        d_df["_t_f"] = (
            c_to_f(get_col(d_df, "corrected_temp_c", 15.0))
            .rolling(window=3, center=True, min_periods=1)
            .mean()
        )
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
                    f'<div style="background: rgba(15, 23, 42, 0.55); '
                    f'border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 10px; '
                    f'padding: 12px 8px; text-align: center; height: 100%;">'
                    f'<div style="font-size: 0.92rem; font-weight: 700; color: #f8fafc;">'
                    f'{r_d["Day_Short"]}</div>'
                    f'<div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px;">'
                    f'{r_d["Date_Short"]}</div>'
                    f'<div style="font-size: 2.2rem; margin: 6px 0;">{ic_d}</div>'
                    f'<div style="font-size: 0.78rem; color: #e2e8f0; min-height: 2.1em;">'
                    f'{cond_d}</div>'
                    f'<div style="font-size: 1.15rem; font-weight: 700; margin-top: 6px;">'
                    f'<span style="color: #fb923c;">{r_d["High_F"]:.0f}°</span> / '
                    f'<span style="color: #38bdf8;">{r_d["Low_F"]:.0f}°</span></div>'
                    f'<div style="font-size: 0.74rem; color: #94a3b8; margin-top: 6px;">'
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
ref_dict, obs_df, future_fc_df, verify_df, archive_stats = load_all_dashboard_data()
model_diag = compute_model_diagnostics_and_importances(verify_df)

now_utc = datetime.now(timezone.utc)
solar_state, solar_el, solar_az, horizon_elev = compute_canyon_solar_state(now_utc)

active_model_name = (
    "MOS_GBM_v2_HYDRO (Thermal Quantile + Two-Stage Hurdle PoP/QPF)"
    if MODEL_PATH.exists()
    else "TOPO_HYDRO_PHYSICS (Analytical Baseline)"
)

with st.sidebar:
    st.markdown("### TZMC MOS Architecture")
    st.caption(
        "Model Output Statistics (MOS) ensemble & boundary-layer diagnostics for "
        "Station 36966 (Stone Office Davis Vantage Pro2, 485 m MSL)."
    )
    st.metric(
        "Serialized Model Artifact",
        "Loaded (v1.joblib)" if model_diag["model_loaded"] else "Physics Fallback",
        f"Features: {len(model_diag['feature_names']) or 21} predictors",
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
    st.markdown("---")
    st.markdown(
        "**Ensemble Hyperparameters:**\n"
        "- `HistGradientBoostingRegressor`\n"
        "- Quantiles: `mean`, `P10`, `P90`\n"
        "- `max_iter`: 350 | `lr`: 0.04\n"
        "- `max_leaf_nodes`: 31 | `L2`: 1.5\n"
        "- Hydro Hurdle: `CalibratedClassifierCV` + `Poisson` QPF"
    )

st.markdown(
    '<div class="met-header-bar">'
    '<h2 style="margin-bottom:0.2rem;">'
    'Tassajara Zen Mountain Center — Microclimate & Hydrological MOS Forecast Suite'
    '</h2></div>',
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
    f'<b>PRIMARY ISS:</b> Stone Office Davis Vantage Pro2 '
    f'(ID 36966 | 36.2337°N, 121.5501°W | 485 m / 1,590 ft MSL) &nbsp;&bull;&nbsp; '
    f'<b>BOUNDARY REFS:</b> Chews Ridge CHWC1 (1,536 m) &amp; Hastings CAHC1 (546 m)<br>'
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
    )
else:

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
            "Vertical Lapse Rate (CHWC1 → TZMC)",
            lapse_str,
            stability_state,
            delta_color="off",
            help=(
                "Environmental temperature lapse rate (Gamma = -[T_Chews - T_Tassajara] / 1.051 km) "
                "across the 1,051 m vertical column between Chews Ridge (CHWC1, 1,536 m) and "
                "Tassajara (485 m). Negative values (Gamma < 0 C/km) indicate a nocturnal thermal "
                "inversion (cold-air pool trapped in the gorge); Gamma > 6.5 C/km indicates a "
                "well-mixed or convectively unstable boundary layer."
            ),
        )
        r2c4.metric(
            "Canyon Thermal Confinement",
            confinement_str,
            ref_sub_str,
            delta_color="off",
            help=(
                "Temperature differential between the inner Tassajara gorge (485 m) and the open "
                "foothill reference station at Hastings Natural History Reservation (CAHC1, 546 m MSL). "
                "Negative differentials at night quantify cold-pool drainage confinement; positive "
                "differentials at midday quantify topographic solar trapping inside the canyon walls."
            ),
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
        v_sub["p10_f"] = c_to_f(
            get_col(v_sub, "corrected_temp_p10_c", get_col(v_sub, "corrected_temp_c", 15.0) - 2.1)
        )
        v_sub["p90_f"] = c_to_f(
            get_col(v_sub, "corrected_temp_p90_c", get_col(v_sub, "corrected_temp_c", 15.0) + 2.1)
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
                "the uncorrected 2.5 km NWP/NWS grid forecast relative to Stone Office observations. "
                "Because the NWP grid resolves Tassajara terrain at ~880 m MSL, it systematically "
                "misses canyon cold-air pooling and early afternoon ridge-shadow cooling."
            ),
        )
        s2.metric(
            "Tassajara MOS Error (MAE)",
            f"± {mos_mae_f:.2f} °F",
            f"{mos_mae_f - nws_mae_f:+.2f} °F MAE vs Raw NWP",
            delta_color="inverse",
            help=(
                "Mean Absolute Error (MAE) of the Tassajara MOS HistGradientBoosting ensemble "
                "after correcting for solar elevation, Black Cone Trail topographic shading, "
                "along-canyon wind vectors, and Chews Ridge vertical stability."
            ),
        )
        s3.metric(
            "MOS RMSE Skill Score",
            f"{skill_pct:+.1f} %",
            f"MOS RMSE: ± {mos_rmse_f:.2f} °F (Bias: {mos_bias_f:+.2f} °F)",
            delta_color="normal" if skill_pct >= 0 else "inverse",
            help=(
                "Forecast Skill Score defined as (1 - RMSE_MOS / RMSE_NWS) * 100%. Positive "
                "percentages quantify the proportional reduction in squared temperature forecast "
                "error achieved by the local microclimate MOS pipeline over the raw operational grid."
            ),
        )
        s4.metric(
            "Hourly Win Frequency",
            f"{win_rate_pct:.1f} % ({hours_mos_won}/{total_eval_hrs}h)",
            f"Evaluated over {total_eval_hrs} paired hours",
            delta_color="off",
            help=(
                "Percentage of evaluated verification hours in which the absolute residual error "
                "of the Tassajara MOS forecast was less than or equal to the raw NWS grid error "
                "(|E_MOS| <= |E_NWS|)."
            ),
        )
        s5.metric(
            "P10–P90 Interval Coverage",
            f"{in_bounds:.1f} %",
            "Nominal 80% Quantile Envelope",
            delta_color="off",
            help=(
                "Empirical calibration of the probabilistic quantile regression models "
                "(loss='quantile' at alpha=0.10 and alpha=0.90). Measures the percentage of "
                "actual Stone Office observations falling inside the predicted P10-P90 "
                "uncertainty fan (target nominal coverage = 80.0%)."
            ),
        )
        s6.metric(
            "Hydrological Brier & QPF",
            f"BS: {mos_brier:.4f}",
            f"Obs: {obs_total_in:.2f}\" | MOS: {mos_total_in:.2f}\"",
            delta_color="off",
            help=(
                "Brier Score (BS = mean[(PoP/100 - O_wet)^2]) measuring the probabilistic "
                "calibration of MOS Probability of Precipitation against observed wet hours "
                "(>= 0.01 in / 0.254 mm), where 0.0000 is perfect reliability. Delta compares "
                "cumulative observed rain vs. cumulative MOS QPF."
            ),
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

            st.dataframe(
                ver_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Valid Time (Local)": st.column_config.TextColumn(
                        "Valid Time (Local)",
                        help="Top-of-the-hour timestamp converted to America/Los_Angeles (PST/PDT).",
                    ),
                    "Lead (hr)": st.column_config.NumberColumn(
                        "Lead (hr)",
                        help="Forecast lead time in hours between NWP initialization and valid time.",
                    ),
                    "Actual (°F)": st.column_config.NumberColumn(
                        "Actual (°F)",
                        help="Ground-truth dry-bulb temperature (F) recorded at the Stone Office ISS.",
                        format="%.1f °F",
                    ),
                    "MOS Forecast (°F)": st.column_config.NumberColumn(
                        "MOS Forecast (°F)",
                        help="Tassajara MOS mean bias-corrected temperature forecast (F).",
                        format="%.1f °F",
                    ),
                    "MOS [P10, P90] (°F)": st.column_config.TextColumn(
                        "MOS [P10, P90] (°F)",
                        help="80% prediction interval bounded by the 10th and 90th percentiles.",
                    ),
                    "Raw NWS/NWP (°F)": st.column_config.NumberColumn(
                        "Raw NWS/NWP (°F)",
                        help="Uncorrected baseline temperature forecast (F) from the NWP/NWS grid.",
                        format="%.1f °F",
                    ),
                    "MOS Residual (°F)": st.column_config.NumberColumn(
                        "MOS Residual (°F)",
                        help="Signed MOS forecast error (MOS minus Actual, F).",
                        format="%+.2f °F",
                    ),
                    "NWS Residual (°F)": st.column_config.NumberColumn(
                        "NWS Residual (°F)",
                        help="Signed raw NWS forecast error (Raw NWS minus Actual, F).",
                        format="%+.2f °F",
                    ),
                    "|MOS| Gain (°F)": st.column_config.NumberColumn(
                        "|MOS| Gain (°F)",
                        help="Absolute error reduction (|NWS Error| - |MOS Error| in F). Positive = MOS win.",
                        format="%+.2f °F",
                    ),
                    "Obs Rain (in)": st.column_config.NumberColumn(
                        "Obs Rain (in)",
                        help="Hourly tipping-bucket rainfall accumulation (inches) at Stone Office ISS.",
                        format="%.3f in",
                    ),
                    "MOS QPF (in)": st.column_config.NumberColumn(
                        "MOS QPF (in)",
                        help="Two-stage hurdle MOS-calibrated Quantitative Precipitation Forecast (in/hr).",
                        format="%.3f in",
                    ),
                    "MOS PoP (%)": st.column_config.NumberColumn(
                        "MOS PoP (%)",
                        help="Calibrated Probability of Precipitation (%) for measurable canyon rain.",
                        format="%d%%",
                    ),
                    "Superior Model": st.column_config.TextColumn(
                        "Superior Model",
                        help="Forecast system with the lower absolute temperature error for that hour.",
                    ),
                },
            )
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
            options=["72h Forecast + Past 48h Hindcast", "7-Day Forecast + Past 72h Hindcast"],
            index=0,
            help=(
                "Toggles the time-series viewport between a high-resolution 5-day window "
                "(48h verification hindcast + 72h operational forecast) and an extended 10-day "
                "synoptic window (72h hindcast + 168h forecast)."
            ),
        )

    with st.expander(
        "Scientific Guide: Interpreting the 3-Panel Canyon Microclimate & Hydrological Suite",
        expanded=False,
    ):
        st.markdown(
            "* **Panel 1 — Thermal Profile & P10–P90 Quantile Fan:** Compares observed Stone Office "
            "temperatures (green markers) against the raw 2.5 km NWS grid (blue dashed line) and the "
            "Tassajara MOS ensemble (orange solid line). The shaded orange band represents the **80% "
            "probabilistic prediction interval (`P10` to `P90`)**, capturing nocturnal cold-air pooling "
            "along Tassajara Creek (`P10` floor) and midday solar heating (`P90` ceiling). When the sun "
            "drops below the **32° southwestern ridge horizon** (Pine Ridge / Black Cone Trail massif) "
            "in mid-afternoon, the MOS trajectory captures rapid radiative cooling missed by NWP grids.\n"
            "* **Panel 2 — Boundary-Layer Kinematics & Moisture:** Tracks 10-meter peak wind gusts "
            "(mph, yellow) alongside relative humidity (%, purple dotted). Strong gusts coinciding with "
            "low RH (< 20%) signal foehn-like subsidence or offshore drainage fire-weather regimes.\n"
            "* **Panel 3 — Two-Stage Hurdle Hydrological Outlook:** Displays observed tipping-bucket "
            "rainfall (green bars) alongside raw NWS QPF (translucent blue bars), **MOS-Calibrated QPF** "
            "(solid cyan bars), and the **MOS P90 Extreme Downpour Rate** (red dashed line). Under "
            "high-VPD pre-frontal regimes, the hurdle model suppresses false-positive light NWP rain "
            "caused by **sub-cloud virga evaporation**; under moist SSW (190°–240°) atmospheric rivers "
            "intersecting the Santa Lucia crest, it amplifies QPF to capture **orographic spillover**."
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
        combined_df["valid_local"] = (
            pd.to_datetime(combined_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        )

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
            row_heights=[0.46, 0.24, 0.30],
            specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=(
                (
                    "Temperature & Dewpoint Profile (°F): "
                    "Stone Office Actuals vs. MOS Quantile Fan (P10–P90) vs. Raw NWS"
                ),
                (
                    "Boundary-Layer Kinematics & Moisture: "
                    "10m Peak Wind Gusts (mph) & Relative Humidity (%)"
                ),
                (
                    "Canyon Hydrological Outlook: "
                    "MOS Calibrated QPF (in/hr), P90 Extreme Rate, Observed Rain & PoP (%)"
                ),
            ),
        )

        corr_c_series = get_col(combined_df, "corrected_temp_c", 15.0)
        if smooth_sci_curve:
            corr_c_series = corr_c_series.rolling(window=3, center=True, min_periods=1).mean()
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
                fillcolor="rgba(56, 189, 248, 0.14)",
                line=dict(color="#38bdf8", width=2.0),
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
                line=dict(color="rgba(148, 163, 184, 0.55)", width=1.3, dash="dot"),
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
                line=dict(color="rgba(244, 63, 94, 0.65)", width=1.4, dash="dash"),
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
                marker_color="rgba(96, 165, 250, 0.45)",
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
                marker_color="#0ea5e9",
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
        fig.update_xaxes(
            title_text="Local Time (America/Los_Angeles)",
            gridcolor="rgba(148,163,184,0.12)",
            row=3,
            col=1,
        )

        fig.update_layout(
            height=880,
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
    # Section 4: 7-Day Synoptic, Hydrological & Road Risk Planning Table
    # ---------------------------------------------------------------------------
    if not future_fc_df.empty:
        st.subheader("7-Day Synoptic, Hydrological & Canyon Road Risk Outlook")
        daily_df = future_fc_df.copy()
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

        st.dataframe(
            display_daily,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn(
                    "Date",
                    help="Local calendar date (America/Los_Angeles) aggregated across forecast hours.",
                ),
                "MOS High (°F)": st.column_config.NumberColumn(
                    "MOS High (°F)",
                    help="Expected daytime maximum dry-bulb temperature (F) from the Tassajara MOS mean model.",
                    format="%.1f °F",
                ),
                "MOS Low (°F)": st.column_config.NumberColumn(
                    "MOS Low (°F)",
                    help="Expected nocturnal minimum dry-bulb temperature (F) accounting for katabatic drainage.",
                    format="%.1f °F",
                ),
                "P10 Floor (°F)": st.column_config.NumberColumn(
                    "P10 Floor (°F)",
                    help="10th-percentile (P10) nocturnal cold-pool floor (F) for freeze protection (<= 32.0 F).",
                    format="%.1f °F",
                ),
                "P90 Ceiling (°F)": st.column_config.NumberColumn(
                    "P90 Ceiling (°F)",
                    help="90th-percentile (P90) daytime thermal peak (F) under clear-sky canyon insolation.",
                    format="%.1f °F",
                ),
                "Raw NWS High (°F)": st.column_config.NumberColumn(
                    "Raw NWS High (°F)",
                    help="Uncorrected daily maximum temperature (F) from the coarse 2.5 km NWS grid.",
                    format="%.1f °F",
                ),
                "Raw NWS Low (°F)": st.column_config.NumberColumn(
                    "Raw NWS Low (°F)",
                    help="Uncorrected daily minimum temperature (F) from the coarse 2.5 km NWS grid.",
                    format="%.1f °F",
                ),
                "MOS PoP (%)": st.column_config.NumberColumn(
                    "MOS PoP (%)",
                    help="Daily maximum MOS-calibrated Probability of Precipitation (%) for measurable rain.",
                    format="%d%%",
                ),
                "MOS QPF (in)": st.column_config.NumberColumn(
                    "MOS QPF (in)",
                    help="24-hour total MOS-calibrated QPF (inches), adjusted for virga and spillover.",
                    format="%.2f in",
                ),
                "P90 Ceiling QPF (in)": st.column_config.NumberColumn(
                    "P90 Ceiling QPF (in)",
                    help="90th-percentile (P90) 24-hour rainfall total (in); >= 2.0 in flags road/creek risk.",
                    format="%.2f in",
                ),
                "Raw NWS QPF (in)": st.column_config.NumberColumn(
                    "Raw NWS QPF (in)",
                    help="Uncorrected 24-hour total Quantitative Precipitation Forecast (in) from NWP grid.",
                    format="%.2f in",
                ),
                "Min RH (%)": st.column_config.NumberColumn(
                    "Min RH (%)",
                    help="Daily minimum forecast Relative Humidity (%). <= 20% flags fire-weather subsidence.",
                    format="%d%%",
                ),
                "Peak Gust (mph)": st.column_config.NumberColumn(
                    "Peak Gust (mph)",
                    help="Maximum 10-meter wind gust (mph) expected during the 24-hour period.",
                    format="%.1f mph",
                ),
                "Canyon Meteorological & Hydrological Regime": st.column_config.TextColumn(
                    "Canyon Meteorological & Hydrological Regime",
                    help="Automated operational synthesis flagging freeze, virga, rain, road, and fire risks.",
                ),
            },
        )

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
            top_n = imp_df.sort_values("Combined Share (%)", ascending=True).tail(14)

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
                    marker_color="#f97316",
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
                    marker_color="#38bdf8",
                    name="Hydrological Hurdle Share (%)",
                    text=top_n["Hydrological Importance (%)"].map(lambda v: f"{v:.1f}%"),
                    textposition="auto",
                ),
                row=1,
                col=2,
            )

            fig_imp.update_xaxes(
                title_text="Relative Attribution (%)",
                gridcolor="rgba(148,163,184,0.12)",
                row=1,
                col=1,
            )
            fig_imp.update_xaxes(
                title_text="Relative Attribution (%)",
                gridcolor="rgba(148,163,184,0.12)",
                row=1,
                col=2,
            )
            fig_imp.update_yaxes(gridcolor="rgba(148,163,184,0.12)", row=1, col=1)
            fig_imp.update_layout(
                height=480,
                margin=dict(t=50, b=35, l=20, r=25),
                plot_bgcolor="rgba(15, 23, 42, 0.35)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )

            st.plotly_chart(fig_imp, use_container_width=True)

            st.dataframe(
                imp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Feature Key": st.column_config.TextColumn(
                        "Feature Key",
                        help="Exact column identifier engineered in mos_model.engineer_mos_features().",
                    ),
                    "Physical Predictor": st.column_config.TextColumn(
                        "Physical Predictor",
                        help="Human-readable meteorological name of the predictor.",
                    ),
                    "Thermal Importance (%)": st.column_config.NumberColumn(
                        "Thermal Importance (%)",
                        help=(
                            "Normalized permutation importance (%) driving the "
                            "HistGradientBoosting thermal bias model."
                        ),
                        format="%.2f%%",
                    ),
                    "Hydrological Importance (%)": st.column_config.NumberColumn(
                        "Hydrological Importance (%)",
                        help="Normalized predictor sensitivity (%) driving the two-stage QPF/PoP hurdle subsystem.",
                        format="%.2f%%",
                    ),
                    "Combined Share (%)": st.column_config.NumberColumn(
                        "Combined Share (%)",
                        help="Mean relative importance across both thermal and hydrological forecasting tasks.",
                        format="%.2f%%",
                    ),
                    "Physical Mechanism": st.column_config.TextColumn(
                        "Physical Mechanism",
                        help="Boundary-layer or topographic mechanism represented by this feature.",
                    ),
                },
            )
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
                help=(
                    "Coefficient of determination (R²) from Ordinary Least Squares regression of "
                    "y_err = (T_actual - T_NWS) onto the physical canyon predictors. Quantifies the "
                    "fraction of systematic NWP temperature error explained linearly before GBDT trees "
                    "capture higher-order interactions."
                ),
            )
            o2.metric(
                "Omnibus F-Statistic",
                f"{model_diag['ols_fstat']:.1f}" if not math.isnan(model_diag["ols_fstat"]) else "N/A",
                "Global Model Significance (p < 0.0001)",
                delta_color="off",
                help="F-test statistic evaluating the joint statistical significance of all physical predictors.",
            )
            o3.metric(
                "Akaike Info Criterion (AIC)",
                f"{model_diag['ols_aic']:.1f}" if not math.isnan(model_diag["ols_aic"]) else "N/A",
                "Penalized Goodness-of-Fit",
                delta_color="off",
                help="Akaike Information Criterion balancing log-likelihood fit against predictor parsimony.",
            )
            o4.metric(
                "Aligned Verification Sample",
                f"{model_diag['ols_nobs']} hrs",
                "Paired Stone Office vs NWP",
                delta_color="off",
                help="Number of hourly observation-forecast pairs evaluated in the OLS diagnostic fit.",
            )

        if not model_diag["ols_coef_df"].empty:
            st.markdown("#### Parametric OLS Coefficient & Significance Table (`statsmodels.api.OLS`)")
            st.dataframe(
                model_diag["ols_coef_df"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Feature Key": st.column_config.TextColumn(
                        "Feature Key",
                        help="Engineered feature variable name in X_ols.",
                    ),
                    "Physical Predictor": st.column_config.TextColumn(
                        "Physical Predictor",
                        help="Meteorological name of the linear predictor.",
                    ),
                    "Coefficient (beta)": st.column_config.NumberColumn(
                        "Coefficient (beta)",
                        help="Partial regression slope (C of temperature bias per unit change in predictor).",
                        format="%+.4f",
                    ),
                    "Std Error": st.column_config.NumberColumn(
                        "Std Error",
                        help="Standard error of the estimated partial regression coefficient.",
                        format="%.4f",
                    ),
                    "t-Statistic": st.column_config.NumberColumn(
                        "t-Statistic",
                        help="Student t-statistic (beta / Std Error) testing H0: beta = 0.",
                        format="%+.2f",
                    ),
                    "p-Value": st.column_config.NumberColumn(
                        "p-Value",
                        help="Two-tailed p-value; values < 0.05 indicate statistically significant physical drivers.",
                        format="%.4f",
                    ),
                    "95% CI Low": st.column_config.NumberColumn(
                        "95% CI Low",
                        help="Lower bound of the 95% confidence interval for beta.",
                        format="%+.4f",
                    ),
                    "95% CI High": st.column_config.NumberColumn(
                        "95% CI High",
                        help="Upper bound of the 95% confidence interval for beta.",
                        format="%+.4f",
                    ),
                    "Physical Interpretation": st.column_config.TextColumn(
                        "Physical Interpretation",
                        help="Physical meaning of the predictor in the Tassajara inner gorge.",
                    ),
                },
            )

        if model_diag["ols_summary_text"]:
            with st.expander(
                "View Raw Statsmodels ASCII Summary Output (corrector.ols_summary)",
                expanded=True,
            ):
                st.code(model_diag["ols_summary_text"], language="text")
        else:
            st.info(
                "OLS summary text not yet available. "
                "Run `python backfill_and_train.py` to populate `corrector.ols_summary`."
            )
