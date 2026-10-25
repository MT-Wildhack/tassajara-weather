#!/usr/bin/env python3
"""
app.py — Tassajara Zen Mountain Center Meteorological & MOS Forecast Monitor
Professional Meteorological Edition:
- Pure meteorological data, forecasts, verification, and sensor telemetry.
- Zero shop operations logs, facility alerts, or corny emojis.
- Professional styling and robust schema-resilient queries.
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
    page_title="Tassajara Meteorological & MOS Forecast Monitor",
    page_icon="📊",
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
        return "Nighttime Katabatic Drainage", el_deg
    if el_deg < 24.0:
        return "Topographic Ridge Shadow", el_deg
    return "Direct Solar Irradiation", el_deg


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
        )

    obs_df = pd.DataFrame()
    ref_df = pd.DataFrame()
    mos_df = pd.DataFrame()
    nws_df = pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        try:
            obs_df = pd.read_sql_query(
                "SELECT * FROM local_observations ORDER BY valid_time_utc DESC LIMIT 200",
                conn,
            )
            if "temp_c" in obs_df.columns:
                obs_df = obs_df[obs_df["temp_c"].notna()].reset_index(drop=True)
        except Exception:
            obs_df = pd.DataFrame()

        try:
            ref_df = pd.read_sql_query(
                "SELECT * FROM reference_observations ORDER BY valid_time_utc DESC LIMIT 50",
                conn,
            )
        except Exception:
            ref_df = pd.DataFrame()

        try:
            mos_df = pd.read_sql_query(
                "SELECT * FROM mos_corrected_forecasts ORDER BY valid_time_utc DESC LIMIT 600",
                conn,
            )
        except Exception:
            mos_df = pd.DataFrame()

        try:
            nws_df = pd.read_sql_query(
                "SELECT * FROM nws_forecasts ORDER BY valid_time_utc DESC LIMIT 1000",
                conn,
            )
        except Exception:
            nws_df = pd.DataFrame()

    ref_dict = {}
    if not ref_df.empty and "station_id" in ref_df.columns:
        for stid, grp in ref_df.groupby("station_id"):
            ref_dict[str(stid)] = grp.iloc[0].to_dict()

    future_fc_df = pd.DataFrame()
    if not mos_df.empty and "issue_time_utc" in mos_df.columns:
        latest_issue = mos_df["issue_time_utc"].max()
        future_fc_df = mos_df[mos_df["issue_time_utc"] == latest_issue].copy()
        future_fc_df = future_fc_df.sort_values("valid_time_utc").reset_index(drop=True)

        if not nws_df.empty and "valid_time_utc" in nws_df.columns:
            sort_col = "issue_time_utc" if "issue_time_utc" in nws_df.columns else "valid_time_utc"
            nws_latest = (
                nws_df.sort_values(sort_col)
                .groupby("valid_time_utc", as_index=False)
                .last()
            )
            extra_cols = [c for c in nws_latest.columns if c not in future_fc_df.columns or c == "valid_time_utc"]
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
        active_renames = {k: v for k, v in rename_map.items() if k in obs_clean.columns}
        obs_clean = obs_clean.rename(columns=active_renames)

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
            if "raw_nws_temp_c" in verify_df.columns and "nws_temp_c" in nws_agg.columns:
                nws_map = nws_agg.set_index("valid_time_utc")["nws_temp_c"]
                verify_df["raw_nws_temp_c"] = verify_df["raw_nws_temp_c"].combine_first(
                    verify_df["valid_time_utc"].map(nws_map)
                )

        if "raw_nws_temp_c" in verify_df.columns:
            mask_valid = verify_df["raw_nws_temp_c"].notna() & verify_df["actual_temp_c"].notna()
            verify_df = verify_df[mask_valid].copy()
            verify_df = verify_df.sort_values("valid_time_utc").tail(72).reset_index(drop=True)

            if "corrected_temp_c" not in verify_df.columns:
                verify_df["corrected_temp_c"] = np.nan
                verify_df["corrected_temp_p10_c"] = np.nan
                verify_df["corrected_temp_p90_c"] = np.nan

            missing_mos = verify_df["corrected_temp_c"].isna()
            if missing_mos.any():
                fb = fallback_physics_mos(verify_df.loc[missing_mos])
                verify_df.loc[missing_mos, "corrected_temp_c"] = fb["corrected_temp_c"]
                verify_df.loc[missing_mos, "corrected_temp_p10_c"] = fb["corrected_temp_p10_c"]
                verify_df.loc[missing_mos, "corrected_temp_p90_c"] = fb["corrected_temp_p90_c"]
        else:
            verify_df = pd.DataFrame()

    return ref_dict, obs_df, future_fc_df, verify_df


# ---------------------------------------------------------------------------
# Header & Telemetry
# ---------------------------------------------------------------------------
ref_dict, obs_df, future_fc_df, verify_df = load_all_dashboard_data()

now_utc = datetime.now(timezone.utc)
solar_state, solar_el = compute_canyon_solar_state(now_utc)

active_model_name = "MOS_GBM_v1 (Trained Ensemble)" if MODEL_PATH.exists() else "TOPO_PHYSICS (Topographic Baseline)"
if not future_fc_df.empty and "model_version" in future_fc_df.columns:
    mv = future_fc_df["model_version"].dropna()
    if not mv.empty:
        active_model_name = str(mv.iloc[0])

st.markdown("## Tassajara Zen Mountain Center — Microclimate & MOS Forecast")

if not obs_df.empty and "valid_time_utc" in obs_df.columns:
    last_obs_dt = pd.to_datetime(obs_df.iloc[0]["valid_time_utc"], utc=True).tz_convert(LOCAL_TZ)
    last_obs_str = last_obs_dt.strftime("%a %b %d, %Y %I:%M %p %Z")
else:
    last_obs_str = "Awaiting telemetry"

st.caption(
    f"Station: Stone Office Davis Vantage Pro2 (1,590 ft / 485 m) | "
    f"Reference Baselines: Chews Ridge CHWC1 (5,040 ft) & Hastings CAHC1 (1,792 ft) | "
    f"Last Observation: {last_obs_str} | Model Engine: {active_model_name}"
)

# ---------------------------------------------------------------------------
# Section 1: Live Meteorological Telemetry Strip
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
    dew_c = safe_float(latest.get("dewpoint_c"), temp_c - 5.0)
    dew_f = c_to_f(dew_c)

    wind_spd_mph = ms_to_mph(safe_float(latest.get("wind_speed_ms"), 0.0))
    wind_gst_mph = ms_to_mph(safe_float(latest.get("wind_gust_ms"), 0.0))
    wind_card = deg_to_cardinal(latest.get("wind_dir_deg"))

    mslp_hpa = safe_float(latest.get("mslp_hpa"), 1013.25)
    mslp_inhg = hpa_to_inhg(mslp_hpa)
    ptrend_hpa = safe_float(latest.get("pressure_trend_3hr_hpa"), 0.0)
    if ptrend_hpa > 0.5:
        ptrend_label = f"+{ptrend_hpa:.1f} hPa (Rising)"
    elif ptrend_hpa < -0.5:
        ptrend_label = f"{ptrend_hpa:.1f} hPa (Falling)"
    else:
        ptrend_label = f"{ptrend_hpa:+.1f} hPa (Steady)"

    rain_24_in = mm_to_in(float(get_col(last_24, "precip_1hr_mm", 0.0).sum()))
    last_hr_rain = mm_to_in(safe_float(latest.get("precip_1hr_mm"), 0.0))
    qc_status = str(latest.get("qc_flag") or "OK")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Stone Office Temperature", f"{temp_f:.1f} °F", f"24h Range: {lo_24_f:.0f}° / {hi_24_f:.0f}°F", delta_color="off")
    k2.metric("Relative Humidity", f"{rh_pct:.0f} %", f"Dewpoint: {dew_f:.1f} °F", delta_color="off")
    k3.metric("Wind Vector", f"{wind_spd_mph:.1f} mph ({wind_card})", f"Peak Gust: {wind_gst_mph:.1f} mph", delta_color="off")
    k4.metric("Barometric Pressure", f"{mslp_inhg:.2f} inHg", ptrend_label, delta_color="off")
    k5.metric("Precipitation (24h)", f"{rain_24_in:.2f} in", f"Last Hour: {last_hr_rain:.2f} in", delta_color="off")
    k6.metric("Sensor Quality Status", qc_status, "Station ID 36966", delta_color="off")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Model Accuracy Scorecard (Verification vs. Standard NWS)
# ---------------------------------------------------------------------------
st.subheader("Forecast Accuracy Verification: Tassajara MOS vs. Standard NWS")

if not verify_df.empty and len(verify_df) >= 3:
    v24 = verify_df.tail(24).copy()
    v24["valid_local"] = pd.to_datetime(v24["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    v24["actual_f"] = c_to_f(get_col(v24, "actual_temp_c", 15.0))
    v24["nws_f"] = c_to_f(get_col(v24, "raw_nws_temp_c", 15.0))
    v24["mos_f"] = c_to_f(get_col(v24, "corrected_temp_c", 15.0))

    v24["nws_abs_err_f"] = (v24["actual_f"] - v24["nws_f"]).abs()
    v24["mos_abs_err_f"] = (v24["actual_f"] - v24["mos_f"]).abs()

    nws_mae_f = float(v24["nws_abs_err_f"].mean())
    mos_mae_f = float(v24["mos_abs_err_f"].mean())
    nws_rmse_f = float(np.sqrt(np.mean((v24["actual_f"] - v24["nws_f"]) ** 2)))
    mos_rmse_f = float(np.sqrt(np.mean((v24["actual_f"] - v24["mos_f"]) ** 2)))

    skill_pct = ((1.0 - (mos_rmse_f / max(nws_rmse_f, 0.01))) * 100.0)
    hours_mos_won = int((v24["mos_abs_err_f"] <= v24["nws_abs_err_f"]).sum())
    total_eval_hrs = len(v24)

    act_hi, act_lo = v24["actual_f"].max(), v24["actual_f"].min()
    mos_hi, mos_lo = v24["mos_f"].max(), v24["mos_f"].min()
    nws_hi, nws_lo = v24["nws_f"].max(), v24["nws_f"].min()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Standard NWS Error (MAE)", f"± {nws_mae_f:.1f} °F", f"RMSE: ± {nws_rmse_f:.1f} °F", delta_color="off")
    s2.metric("Tassajara MOS Error (MAE)", f"± {mos_mae_f:.1f} °F", f"{mos_mae_f - nws_mae_f:+.1f} °F vs NWS", delta_color="inverse")
    s3.metric("Skill Score Improvement", f"{skill_pct:+.1f} %", f"MOS superior in {hours_mos_won}/{total_eval_hrs} hours", delta_color="normal" if skill_pct >= 0 else "inverse")
    s4.metric("24h Extreme Verification", f"Actual: {act_lo:.0f}° / {act_hi:.0f}°F", f"MOS: {mos_lo:.0f}°/{mos_hi:.0f}°F", delta_color="off")

    with st.expander("View Hourly Verification Log (Past 24 Hours)", expanded=False):
        ver_table = pd.DataFrame({
            "Timestamp (Local)": v24["valid_local"].dt.strftime("%a %b %d, %I:%M %p"),
            "Actual Observation (°F)": v24["actual_f"].round(1),
            "MOS Forecast (°F)": v24["mos_f"].round(1),
            "Standard NWS Forecast (°F)": v24["nws_f"].round(1),
            "MOS Error (°F)": (v24["mos_f"] - v24["actual_f"]).round(1),
            "NWS Error (°F)": (v24["nws_f"] - v24["actual_f"]).round(1),
            "Preferred Model": np.where(v24["mos_abs_err_f"] <= v24["nws_abs_err_f"], "Tassajara MOS", "Standard NWS"),
        }).iloc[::-1].reset_index(drop=True)
        st.dataframe(ver_table, use_container_width=True, hide_index=True)
else:
    st.info("Forecast verification scorecard is accumulating paired hourly observations. Run the backfill workflow in GitHub Actions to populate historical verification records.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Microclimate Timeline & Forecast Charts
# ---------------------------------------------------------------------------
ctrl_col1, ctrl_col2 = st.columns([3, 1])
with ctrl_col1:
    st.subheader("Microclimate Timeline: Past 36h Verification & Forecast Series")
with ctrl_col2:
    horizon_choice = st.selectbox("Forecast Horizon", options=["72 Hours + Past 36h", "7 Days + Past 36h"], index=0)

if not future_fc_df.empty:
    max_future_hrs = 72 if "72 Hours" in horizon_choice else 168
    fut_sub = future_fc_df.head(max_future_hrs).copy()

    if not verify_df.empty:
        past_sub = verify_df.tail(36).copy()
        past_sub = past_sub[~past_sub["valid_time_utc"].isin(fut_sub["valid_time_utc"])]
        combined_df = pd.concat([past_sub, fut_sub], ignore_index=True, sort=False)
    else:
        combined_df = fut_sub.copy()

    combined_df = combined_df.sort_values("valid_time_utc").reset_index(drop=True)
    combined_df["valid_local"] = pd.to_datetime(combined_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    if not obs_df.empty and "valid_time_utc" in obs_df.columns:
        obs_dedup = obs_df.drop_duplicates(subset=["valid_time_utc"]).set_index("valid_time_utc")
        for col_dest, col_src in [
            ("actual_temp_c", "temp_c"),
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
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.50, 0.25, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Temperature Profile (°F): Actuals vs. MOS Forecast vs. NWS Grid",
            "Atmospheric Moisture & Wind: Relative Humidity (%) & Peak Gusts (mph)",
            "Precipitation & Probability: Cumulative Rate (in/hr) & PoP (%)",
        ),
    )

    # Panel 1: Temperature
    corr_c_series = get_col(combined_df, "corrected_temp_c", 15.0)
    p90_f = c_to_f(get_col(combined_df, "corrected_temp_p90_c", corr_c_series + 2.0))
    p10_f = c_to_f(get_col(combined_df, "corrected_temp_p10_c", corr_c_series - 2.0))

    fig.add_trace(
        go.Scatter(
            x=pd.concat([combined_df["valid_local"], combined_df["valid_local"][::-1]]),
            y=pd.concat([p90_f, p10_f[::-1]]),
            fill="toself",
            fillcolor="rgba(230, 97, 1, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="MOS Uncertainty Interval (10th-90th)",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=c_to_f(get_col(combined_df, "raw_nws_temp_c", corr_c_series)),
            mode="lines",
            line=dict(color="#4a90e2", width=2, dash="dash"),
            name="Standard NWS Grid Forecast",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=c_to_f(corr_c_series),
            mode="lines",
            line=dict(color="#ff7f0e", width=3),
            name="Tassajara MOS Forecast",
        ),
        row=1, col=1,
    )

    if "actual_temp_c" in combined_df.columns:
        actual_pts = combined_df[combined_df["actual_temp_c"].notna()]
        if not actual_pts.empty:
            act_y = c_to_f(pd.to_numeric(actual_pts["actual_temp_c"], errors="coerce"))
            fig.add_trace(
                go.Scatter(
                    x=actual_pts["valid_local"],
                    y=act_y,
                    mode="lines+markers",
                    line=dict(color="#2ecc71", width=2),
                    marker=dict(color="#2ecc71", size=6, symbol="circle"),
                    name="Stone Office Actuals",
                ),
                row=1, col=1,
            )

    fig.add_hline(
        y=32.0, line_dash="dot", line_color="#3b82f6",
        annotation_text="Freezing Threshold (32 °F)", annotation_position="bottom right",
        row=1, col=1,
    )

    # Panel 2: Wind Gusts & RH
    gust_series_ms = (
        get_col(combined_df, "corrected_wind_gust_ms", np.nan)
        .combine_first(get_col(combined_df, "nws_wind_gust_ms", np.nan))
        .combine_first(get_col(combined_df, "nws_wind_speed_ms", 0.0))
        .fillna(0.0)
    )
    gust_mph_series = gust_series_ms * 2.23694

    rh_series = (
        get_col(combined_df, "corrected_rh_pct", np.nan)
        .combine_first(get_col(combined_df, "nws_rh_pct", 50.0))
        .fillna(50.0)
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"], y=gust_mph_series.round(1),
            mode="lines", line=dict(color="#f59e0b", width=2),
            name="Peak Wind Gust (mph)",
        ),
        row=2, col=1, secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"], y=rh_series.round(0),
            mode="lines", line=dict(color="#a855f7", width=2, dash="dot"),
            name="Relative Humidity (%)",
        ),
        row=2, col=1, secondary_y=True,
    )

    # Panel 3: Precipitation
    nws_rain_in = (get_col(combined_df, "nws_qpf_1hr_mm", 0.0) / 25.4).round(3)
    actual_rain_in = (get_col(combined_df, "actual_precip_mm", 0.0) / 25.4).round(3)
    pop_pct = get_col(combined_df, "nws_pop_pct", 0.0).clip(0.0, 100.0)

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"], y=pop_pct.round(0),
            mode="lines", fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="rgba(56, 189, 248, 0.65)", width=1.5),
            name="Probability of Precipitation (%)",
        ),
        row=3, col=1, secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=combined_df["valid_local"], y=nws_rain_in,
            name="NWS QPF Rain (in/hr)", marker_color="#38bdf8", opacity=0.8,
        ),
        row=3, col=1, secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=combined_df["valid_local"], y=actual_rain_in,
            name="Actual Rain Gauge (in/hr)", marker_color="#2ecc71",
        ),
        row=3, col=1, secondary_y=False,
    )

    if not fut_sub.empty:
        now_marker_ts = pd.to_datetime(fut_sub["valid_time_utc"].iloc[0], utc=True).tz_convert(LOCAL_TZ)
        for r_idx in (1, 2, 3):
            fig.add_vline(
                x=now_marker_ts, line_width=1.5, line_dash="dash",
                line_color="rgba(255, 255, 255, 0.55)",
                row=r_idx, col=1,
            )

    max_rain_val = float(max(nws_rain_in.max(), actual_rain_in.max(), 0.0))
    rain_y_top = max(0.20, round(max_rain_val * 1.25, 2))
    max_gust_val = float(max(gust_mph_series.max(), 15.0))

    fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Wind Speed (mph)", range=[0, max_gust_val * 1.2], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Humidity (%)", range=[0, 105], showgrid=False, row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Precip Rate (in/hr)", range=[0, rain_y_top], row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PoP (%)", range=[0, 105], showgrid=False, row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local Time (America/Los_Angeles)", row=3, col=1)

    fig.update_layout(
        height=820,
        margin=dict(t=55, b=40, l=50, r=50),
        hovermode="x unified",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: 7-Day Meteorological Planning Summary Table
# ---------------------------------------------------------------------------
if not future_fc_df.empty:
    st.subheader("7-Day Meteorological Summary & Outlook")
    daily_df = future_fc_df.copy()
    daily_df["valid_local"] = pd.to_datetime(daily_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    daily_df["Date_Key"] = daily_df["valid_local"].dt.strftime("%a, %b %d")
    daily_df["Sort_Date"] = daily_df["valid_local"].dt.date

    daily_df["_corr_t"] = get_col(daily_df, "corrected_temp_c", 15.0)
    daily_df["_p10_t"] = get_col(daily_df, "corrected_temp_p10_c", daily_df["_corr_t"] - 2.0)
    daily_df["_nws_t"] = get_col(daily_df, "raw_nws_temp_c", daily_df["_corr_t"])
    daily_df["_pop"] = get_col(daily_df, "nws_pop_pct", 0.0)
    daily_df["_qpf"] = get_col(daily_df, "nws_qpf_1hr_mm", 0.0)
    daily_df["_rh"] = get_col(daily_df, "corrected_rh_pct", np.nan).combine_first(get_col(daily_df, "nws_rh_pct", 45.0))
    daily_df["_gust"] = (
        get_col(daily_df, "corrected_wind_gust_ms", np.nan)
        .combine_first(get_col(daily_df, "nws_wind_gust_ms", np.nan))
        .combine_first(get_col(daily_df, "nws_wind_speed_ms", 0.0))
    )

    daily_summary = (
        daily_df.groupby(["Sort_Date", "Date_Key"], as_index=False)
        .agg(
            MOS_High_F=("_corr_t", lambda s: round(c_to_f(float(s.max())), 1)),
            MOS_Low_F=("_corr_t", lambda s: round(c_to_f(float(s.min())), 1)),
            Cold_Pool_Floor_F=("_p10_t", lambda s: round(c_to_f(float(s.min())), 1)),
            NWS_High_F=("_nws_t", lambda s: round(c_to_f(float(s.max())), 1)),
            NWS_Low_F=("_nws_t", lambda s: round(c_to_f(float(s.min())), 1)),
            Max_PoP_Pct=("_pop", lambda s: round(safe_float(s.max(), 0.0), 0)),
            Total_Precip_In=("_qpf", lambda s: round(mm_to_in(float(s.sum())), 2)),
            Min_RH_Pct=("_rh", lambda s: round(safe_float(s.min(), 40.0), 0)),
            Peak_Gust_MPH=("_gust", lambda s: round(ms_to_mph(safe_float(s.max(), 0.0)), 1)),
        )
        .sort_values("Sort_Date")
    )

    def _met_note(row):
        notes = []
        if row["Cold_Pool_Floor_F"] <= 32.0:
            notes.append("Freezing temperatures (≤32°F)")
        elif row["Cold_Pool_Floor_F"] <= 36.0:
            notes.append("Nocturnal thermal inversion / frost risk")
        if row["Total_Precip_In"] >= 0.25 or row["Max_PoP_Pct"] >= 50:
            notes.append("Precipitation expected")
        if row["Min_RH_Pct"] <= 20 and row["Peak_Gust_MPH"] >= 20:
            notes.append("Elevated fire weather indices")
        if not notes:
            notes.append("Stable meteorological conditions")
        return " | ".join(notes)

    daily_summary["Synoptic Regime"] = daily_summary.apply(_met_note, axis=1)

    display_daily = daily_summary[[
        "Date_Key",
        "MOS_High_F",
        "MOS_Low_F",
        "Cold_Pool_Floor_F",
        "NWS_High_F",
        "NWS_Low_F",
        "Max_PoP_Pct",
        "Total_Precip_In",
        "Min_RH_Pct",
        "Peak_Gust_MPH",
        "Synoptic Regime",
    ]].rename(columns={
        "Date_Key": "Date",
        "MOS_High_F": "MOS High (°F)",
        "MOS_Low_F": "MOS Low (°F)",
        "Cold_Pool_Floor_F": "P10 Floor (°F)",
        "NWS_High_F": "NWS High (°F)",
        "NWS_Low_F": "NWS Low (°F)",
        "Max_PoP_Pct": "Max PoP (%)",
        "Total_Precip_In": "QPF Total (in)",
        "Min_RH_Pct": "Min RH (%)",
        "Peak_Gust_MPH": "Peak Gust (mph)",
    })

    st.dataframe(display_daily, use_container_width=True, hide_index=True)
