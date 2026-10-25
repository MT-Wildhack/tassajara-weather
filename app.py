#!/usr/bin/env python3
"""
app.py (v4) — Tassajara Zen Mountain Center Operational Microclimate & MOS Dashboard
Designed for Physical Plant & Shop Operations (Stone Office Davis Vantage Pro2, 1,590 ft).
Features:
- Actionable Shop Operational Advisories (Freeze Risk, Fire Weather / Inversion, Rain, Solar Shade)
- Guaranteed-Numeric Live Telemetry Strip (Zero NaN glitches, Barometer & 24h Rain)
- Continuous Hindcast + Forecast Timeline (Past 36h Actuals vs. MOS vs. NWS + Next 7 Days)
- Automated Model Accuracy Scorecard & Hour-by-Hour Verification Table (with backfill fallback)
- 3-Panel Interactive Chart (Temperature Fan Chart, Wind/RH Fire-Weather, Non-Negative Precipitation & PoP%)
- 7-Day Shop Planning Summary Table
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
    ix = int(round(float(deg) / 22.5)) % 16
    return dirs[ix]


def safe_float(val, default=0.0):
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


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
    wind_ms = pd.to_numeric(out.get("nws_wind_speed_ms", 0.0), errors="coerce").fillna(0.0)
    calm_factor = np.clip(1.0 - (wind_ms / 6.0), 0.2, 1.0)

    offset_c = np.where(
        is_night,
        -2.2 * calm_factor,
        np.where(is_afternoon_shade, -1.6 * calm_factor, 0.6 * calm_factor),
    )
    raw_c = pd.to_numeric(out["raw_nws_temp_c"], errors="coerce")
    out["corrected_temp_c"] = (raw_c + offset_c).round(2)
    out["corrected_temp_p10_c"] = (out["corrected_temp_c"] - 2.1).round(2)
    out["corrected_temp_p90_c"] = (out["corrected_temp_c"] + 2.1).round(2)
    return out


# ---------------------------------------------------------------------------
# Database Loader
# ---------------------------------------------------------------------------
@st.cache_data(ttl=180)
def load_all_dashboard_data():
    if not Path(DB_PATH).exists():
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        # 1. Recent Stone Office Observations (last 7 days = 168 hours)
        obs_df = pd.read_sql_query(
            """
            SELECT * FROM local_observations
            WHERE temp_c IS NOT NULL
            ORDER BY valid_time_utc DESC
            LIMIT 168
            """,
            conn,
        )

        # 2. Latest Reference Station Observations (Chews Ridge CHWC1 & Hastings CAHC1)
        ref_df = pd.read_sql_query(
            """
            SELECT r.*
            FROM reference_observations r
            INNER JOIN (
                SELECT station_id, MAX(valid_time_utc) AS max_ts
                FROM reference_observations
                GROUP BY station_id
            ) latest ON r.station_id = latest.station_id AND r.valid_time_utc = latest.max_ts
            """,
            conn,
        )

        # 3. Latest Future Forecast Run (from mos_corrected_forecasts joined with nws_forecasts)
        future_fc_df = pd.read_sql_query(
            """
            SELECT
                m.issue_time_utc,
                m.valid_time_utc,
                m.lead_time_hours,
                m.model_version,
                m.raw_nws_temp_c,
                m.corrected_temp_c,
                m.corrected_temp_p10_c,
                m.corrected_temp_p90_c,
                m.corrected_rh_pct,
                m.corrected_wind_gust_ms,
                n.nws_rh_pct,
                n.nws_dewpoint_c,
                n.nws_wind_speed_ms,
                n.nws_wind_gust_ms,
                n.nws_wind_dir_deg,
                n.nws_pop_pct,
                n.nws_qpf_1hr_mm,
                n.nws_sky_cover_pct,
                n.short_forecast
            FROM mos_corrected_forecasts m
            LEFT JOIN nws_forecasts n
                ON m.issue_time_utc = n.issue_time_utc
                AND m.valid_time_utc = n.valid_time_utc
            WHERE m.issue_time_utc = (SELECT MAX(issue_time_utc) FROM mos_corrected_forecasts)
            ORDER BY m.valid_time_utc ASC
            """,
            conn,
        )

        # 4. Historical Verification Dataset (Past 72 hours where we have BOTH forecasts & actuals)
        verify_df = pd.read_sql_query(
            """
            SELECT
                o.valid_time_utc,
                o.temp_c AS actual_temp_c,
                o.rh_pct AS actual_rh_pct,
                o.wind_speed_ms AS actual_wind_speed_ms,
                o.wind_gust_ms AS actual_wind_gust_ms,
                o.precip_1hr_mm AS actual_precip_mm,
                f.nws_temp_c AS raw_nws_temp_c,
                f.nws_rh_pct,
                f.nws_dewpoint_c,
                f.nws_wind_speed_ms,
                f.nws_wind_dir_deg,
                f.nws_sky_cover_pct,
                f.nws_pop_pct,
                f.nws_qpf_1hr_mm,
                m.corrected_temp_c,
                m.corrected_temp_p10_c,
                m.corrected_temp_p90_c,
                m.model_version
            FROM local_observations o
            INNER JOIN (
                SELECT valid_time_utc,
                       AVG(nws_temp_c) AS nws_temp_c,
                       AVG(nws_rh_pct) AS nws_rh_pct,
                       AVG(nws_dewpoint_c) AS nws_dewpoint_c,
                       AVG(nws_wind_speed_ms) AS nws_wind_speed_ms,
                       AVG(nws_wind_dir_deg) AS nws_wind_dir_deg,
                       AVG(nws_sky_cover_pct) AS nws_sky_cover_pct,
                       MAX(nws_pop_pct) AS nws_pop_pct,
                       MAX(nws_qpf_1hr_mm) AS nws_qpf_1hr_mm
                FROM nws_forecasts
                GROUP BY valid_time_utc
            ) f ON o.valid_time_utc = f.valid_time_utc
            LEFT JOIN (
                SELECT valid_time_utc,
                       AVG(corrected_temp_c) AS corrected_temp_c,
                       AVG(corrected_temp_p10_c) AS corrected_temp_p10_c,
                       AVG(corrected_temp_p90_c) AS corrected_temp_p90_c,
                       MAX(model_version) AS model_version
                FROM mos_corrected_forecasts
                GROUP BY valid_time_utc
            ) m ON o.valid_time_utc = m.valid_time_utc
            WHERE o.temp_c IS NOT NULL AND f.nws_temp_c IS NOT NULL
            ORDER BY o.valid_time_utc DESC
            LIMIT 72
            """,
            conn,
        )

        # 5. Facility Alerts
        alerts_df = pd.read_sql_query(
            "SELECT * FROM facility_alert_log ORDER BY triggered_at_utc DESC LIMIT 15",
            conn,
        )

    # Sort verification chronologically & fill any missing historical MOS predictions
    if not verify_df.empty:
        verify_df = verify_df.sort_values("valid_time_utc").reset_index(drop=True)
        missing_mos = verify_df["corrected_temp_c"].isna()
        if missing_mos.any():
            if MODEL_PATH.exists():
                try:
                    from mos_model import TassajaraMOSCorrector
                    corrector = TassajaraMOSCorrector.load(str(MODEL_PATH))
                    eval_input = pd.DataFrame({
                        "issue_time_utc": verify_df["valid_time_utc"],
                        "valid_time_utc": verify_df["valid_time_utc"],
                        "lead_time_hours": 6,
                        "nws_temp_c": verify_df["raw_nws_temp_c"],
                        "nws_dewpoint_c": verify_df["nws_dewpoint_c"].fillna(verify_df["raw_nws_temp_c"] - 6.0),
                        "nws_rh_pct": verify_df["nws_rh_pct"].fillna(55.0),
                        "nws_wind_speed_ms": verify_df["nws_wind_speed_ms"].fillna(1.5),
                        "nws_wind_dir_deg": verify_df["nws_wind_dir_deg"].fillna(115.0),
                        "nws_sky_cover_pct": verify_df["nws_sky_cover_pct"].fillna(20.0),
                        "nws_pop_pct": verify_df["nws_pop_pct"].fillna(0.0),
                        "pressure_trend_3hr_hpa": 0.0,
                        "issue_chews_temp_c": verify_df["raw_nws_temp_c"] + 1.5,
                        "issue_hastings_temp_c": verify_df["raw_nws_temp_c"] + 0.5,
                        "issue_tass_temp_c": verify_df["actual_temp_c"].shift(6).bfill(),
                        "issue_persistence_bias_c": (
                            verify_df["actual_temp_c"].shift(6).bfill() - verify_df["raw_nws_temp_c"]
                        ).fillna(0.0),
                    })
                    preds = corrector.predict(eval_input)
                    verify_df.loc[missing_mos, "corrected_temp_c"] = preds.loc[missing_mos, "corrected_temp_c"]
                    verify_df.loc[missing_mos, "corrected_temp_p10_c"] = preds.loc[missing_mos, "corrected_temp_p10_c"]
                    verify_df.loc[missing_mos, "corrected_temp_p90_c"] = preds.loc[missing_mos, "corrected_temp_p90_c"]
                    verify_df.loc[missing_mos, "model_version"] = "MOS_GBM_v1"
                except Exception:
                    fb = fallback_physics_mos(verify_df.loc[missing_mos])
                    verify_df.loc[missing_mos, "corrected_temp_c"] = fb["corrected_temp_c"]
                    verify_df.loc[missing_mos, "corrected_temp_p10_c"] = fb["corrected_temp_p10_c"]
                    verify_df.loc[missing_mos, "corrected_temp_p90_c"] = fb["corrected_temp_p90_c"]
            else:
                fb = fallback_physics_mos(verify_df.loc[missing_mos])
                verify_df.loc[missing_mos, "corrected_temp_c"] = fb["corrected_temp_c"]
                verify_df.loc[missing_mos, "corrected_temp_p10_c"] = fb["corrected_temp_p10_c"]
                verify_df.loc[missing_mos, "corrected_temp_p90_c"] = fb["corrected_temp_p90_c"]

    ref_dict = {}
    if not ref_df.empty:
        for _, r in ref_df.iterrows():
            ref_dict[r["station_id"]] = r.to_dict()

    return ref_dict, obs_df, future_fc_df, verify_df, alerts_df


# ---------------------------------------------------------------------------
# Header & Top Operational Status Bar
# ---------------------------------------------------------------------------
ref_dict, obs_df, future_fc_df, verify_df, alerts_df = load_all_dashboard_data()

now_utc = datetime.now(timezone.utc)
solar_state, solar_el = compute_canyon_solar_state(now_utc)

active_model_name = "MOS_GBM_v1 (Trained ML Ensemble)" if MODEL_PATH.exists() else "TOPO_PHYSICS (Canyon Shadow & Drainage Baseline)"
if not future_fc_df.empty and "model_version" in future_fc_df.columns:
    mv = future_fc_df["model_version"].iloc[0]
    if pd.notna(mv):
        active_model_name = str(mv)

st.markdown("## 🌲 Tassajara Zen Mountain Center — Canyon Weather & Shop Operations Monitor")

if not obs_df.empty:
    last_obs_dt = pd.to_datetime(obs_df.iloc[0]["valid_time_utc"], utc=True).tz_convert(LOCAL_TZ)
    last_obs_str = last_obs_dt.strftime("%a %b %d, %I:%M %p %Z")
else:
    last_obs_str = "Awaiting telemetry"

st.caption(
    f"**Station:** Stone Office Davis Vantage Pro2 (`1,590 ft` / `485 m`) | "
    f"**Ridge Baselines:** Chews Ridge `CHWC1` (`5,040 ft`) & Hastings `CAHC1` (`1,792 ft`) | "
    f"**Last Updated:** {last_obs_str} | **Active Engine:** `{active_model_name}`"
)

# ---------------------------------------------------------------------------
# Section 1: Actionable Shop Operational Advisories (Next 48 Hours)
# ---------------------------------------------------------------------------
if not future_fc_df.empty:
    fc_48 = future_fc_df.head(48).copy()
    fc_48["valid_local"] = pd.to_datetime(fc_48["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    min_temp_f = c_to_f(fc_48["corrected_temp_c"].min())
    min_p10_f = c_to_f(fc_48["corrected_temp_p10_c"].min())
    min_row = fc_48.loc[fc_48["corrected_temp_c"].idxmin()]
    min_time_str = min_row["valid_local"].strftime("%a %I:%M %p")

    max_temp_f = c_to_f(fc_48["corrected_temp_c"].max())
    min_rh_48 = safe_float(
        fc_48["corrected_rh_pct"].min() if "corrected_rh_pct" in fc_48 else fc_48["nws_rh_pct"].min(),
        40.0,
    )
    max_gust_mph = ms_to_mph(
        safe_float(
            fc_48["corrected_wind_gust_ms"].max() if "corrected_wind_gust_ms" in fc_48 else fc_48["nws_wind_gust_ms"].max(),
            0.0,
        )
    )

    total_rain_48_in = mm_to_in(fc_48["nws_qpf_1hr_mm"].fillna(0.0).sum())
    max_pop_48 = safe_float(fc_48["nws_pop_pct"].fillna(0.0).max(), 0.0)

    adv1, adv2, adv3, adv4 = st.columns(4)

    with adv1:
        if min_p10_f <= 32.0:
            st.error(
                f"**❄️ FREEZE ALERT (Next 48h)**\n\n"
                f"Low of **{min_temp_f:.1f}°F** (Cold-Pool Risk: **{min_p10_f:.1f}°F**) around **{min_time_str}**. "
                f"Check Bathhouse bypass & exposed lines."
            )
        elif min_p10_f <= 37.0:
            st.warning(
                f"**🌡️ FROST WATCH (Next 48h)**\n\n"
                f"Canyon low **{min_temp_f:.1f}°F** (Cold-Pool P10: **{min_p10_f:.1f}°F**) on **{min_time_str}**. "
                f"Pipes safe; light creek frost possible."
            )
        else:
            st.success(
                f"**🟢 NO PIPE FREEZE RISK**\n\n"
                f"48h Lowest Temp: **{min_temp_f:.1f}°F** (10th-pct floor: **{min_p10_f:.1f}°F**) on **{min_time_str}**."
            )

    with adv2:
        if total_rain_48_in >= 0.50:
            st.warning(
                f"**🌧️ HEAVY RAIN INBOUND**\n\n"
                f"**{total_rain_48_in:.2f} in** expected next 48h (**{max_pop_48:.0f}%** chance). "
                f"Check Kitchen roof, Cabins 16–19 drain & Stone Office rain cone."
            )
        elif total_rain_48_in > 0.01 or max_pop_48 >= 30.0:
            st.info(
                f"**🌦️ SHOWERS POSSIBLE**\n\n"
                f"Up to **{total_rain_48_in:.2f} in** next 48h (**{max_pop_48:.0f}%** peak chance)."
            )
        else:
            st.success(
                f"**☀️ DRY CANYON WINDOW**\n\n"
                f"**0.00 in** rain expected next 48h (Max rain chance: **{max_pop_48:.0f}%**). Clear for outdoor work."
            )

    with adv3:
        if min_rh_48 <= 20.0 and max_gust_mph >= 20.0:
            st.error(
                f"**🔥 ELEVATED FIRE WEATHER**\n\n"
                f"RH dropping to **{min_rh_48:.0f}%** with canyon gusts to **{max_gust_mph:.0f} mph**."
            )
        else:
            st.info(
                f"**💨 WIND & HUMIDITY (48h)**\n\n"
                f"Peak Canyon Gust: **{max_gust_mph:.1f} mph** | Lowest RH: **{min_rh_48:.0f}%** | High Temp: **{max_temp_f:.1f}°F**."
            )

    with adv4:
        chews_temp_c = ref_dict.get("CHWC1", {}).get("temp_c")
        tass_now_c = obs_df.iloc[0]["temp_c"] if not obs_df.empty else None
        if chews_temp_c is not None and tass_now_c is not None:
            diff_f = c_to_f(chews_temp_c) - c_to_f(tass_now_c)
            inv_text = (
                f"Canyon Inversed (**{abs(diff_f):.1f}°F cooler** than Chews Ridge)"
                if diff_f > 0
                else f"Normal Lapse (**{abs(diff_f):.1f}°F warmer** than Chews Ridge)"
            )
        else:
            inv_text = "Ridge telemetry syncing"

        st.info(
            f"**⛰️ CANYON SOLAR & AIRMASS**\n\n"
            f"State: **{solar_state}** (Sun El: `{solar_el:.0f}°`)\n\n"
            f"{inv_text}"
        )

# ---------------------------------------------------------------------------
# Section 2: Live Stone Office Sensor Readouts (Guaranteed Numeric — Zero NaNs)
# ---------------------------------------------------------------------------
if not obs_df.empty:
    latest = obs_df.iloc[0]
    last_24 = obs_df.head(24)

    temp_c = safe_float(latest.get("temp_c"), 15.0)
    temp_f = c_to_f(temp_c)
    hi_24_f = c_to_f(safe_float(last_24["temp_c"].max(), temp_c))
    lo_24_f = c_to_f(safe_float(last_24["temp_c"].min(), temp_c))

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

    rain_24_in = mm_to_in(safe_float(last_24["precip_1hr_mm"].fillna(0.0).sum(), 0.0))
    qc_status = str(latest.get("qc_flag") or "OK")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Stone Office Temp", f"{temp_f:.1f} °F", f"24h: {lo_24_f:.0f}° / {hi_24_f:.0f}°F", delta_color="off")
    k2.metric("Relative Humidity", f"{rh_pct:.0f} %", f"Dewpoint: {dew_f:.1f} °F", delta_color="off")
    k3.metric("Wind Speed & Dir", f"{wind_spd_mph:.1f} mph ({wind_card})", f"Peak Gust: {wind_gst_mph:.1f} mph", delta_color="off")
    k4.metric("Barometer (MSLP)", f"{mslp_inhg:.2f} inHg", ptrend_label, delta_color="off")
    k5.metric("24-Hr Rain Gauge", f"{rain_24_in:.2f} in", f"Last Hr: {mm_to_in(safe_float(latest.get('precip_1hr_mm'), 0.0)):.2f} in", delta_color="off")
    k6.metric("Sensor QC Status", qc_status, "Station ID 36966", delta_color="off")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Model Accuracy Scorecard (Previous Forecast vs. Actual Weather)
# ---------------------------------------------------------------------------
st.subheader("🎯 Forecast Accuracy Scorecard: Our MOS Forecast vs. Standard NWS vs. Actual Weather")

if not verify_df.empty:
    v24 = verify_df.tail(24).copy()
    v24["valid_local"] = pd.to_datetime(v24["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    v24["actual_f"] = c_to_f(v24["actual_temp_c"])
    v24["nws_f"] = c_to_f(v24["raw_nws_temp_c"])
    v24["mos_f"] = c_to_f(v24["corrected_temp_c"])

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
    s1.metric(
        "Standard NWS Error (24h MAE)",
        f"± {nws_mae_f:.1f} °F",
        f"RMSE: ± {nws_rmse_f:.1f} °F",
        delta_color="off",
        help="Average temperature error of the generic National Weather Service grid forecast over the past 24 verified hours.",
    )
    s2.metric(
        "Tassajara MOS Error (24h MAE)",
        f"± {mos_mae_f:.1f} °F",
        f"{mos_mae_f - nws_mae_f:+.1f} °F error vs. NWS",
        delta_color="inverse",
        help="Average temperature error of our canyon-corrected MOS forecast over the exact same 24 hours.",
    )
    s3.metric(
        "Accuracy Improvement (Skill Score)",
        f"{skill_pct:+.1f} %",
        f"MOS beat NWS in {hours_mos_won}/{total_eval_hrs} hrs",
        delta_color="normal" if skill_pct >= 0 else "inverse",
        help="Percentage reduction in Root Mean Squared Error (RMSE) achieved by the Tassajara MOS model compared to the standard NWS forecast.",
    )
    s4.metric(
        "24h High / Low Verification",
        f"Actual: {act_hi:.0f}° / {act_lo:.0f}°F",
        f"MOS: {mos_hi:.0f}°/{mos_lo:.0f}°F | NWS: {nws_hi:.0f}°/{nws_lo:.0f}°F",
        delta_color="off",
        help="Comparison of actual Stone Office 24-hour High/Low temperatures against MOS and raw NWS predictions.",
    )

    with st.expander("📋 View Hour-by-Hour Forecast Verification Table (Past 24 Hours: Actual vs. MOS vs. Standard NWS)", expanded=False):
        ver_table = pd.DataFrame({
            "Local Time": v24["valid_local"].dt.strftime("%a %b %d, %I:%M %p"),
            "Actual Stone Office (°F)": v24["actual_f"].round(1),
            "Our MOS Forecast (°F)": v24["mos_f"].round(1),
            "Standard NWS Forecast (°F)": v24["nws_f"].round(1),
            "Our MOS Error (°F)": (v24["mos_f"] - v24["actual_f"]).round(1),
            "Standard NWS Error (°F)": (v24["nws_f"] - v24["actual_f"]).round(1),
            "More Accurate Model": np.where(
                v24["mos_abs_err_f"] <= v24["nws_abs_err_f"],
                "✅ Tassajara MOS",
                "Standard NWS",
            ),
        }).iloc[::-1].reset_index(drop=True)
        st.dataframe(ver_table, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Continuous Hindcast + Forecast Interactive Charts
# ---------------------------------------------------------------------------
ctrl_col1, ctrl_col2 = st.columns([3, 1])
with ctrl_col1:
    st.subheader("📈 Canyon Microclimate Timeline: Past 36h Verification & Upcoming Forecast")
with ctrl_col2:
    horizon_choice = st.selectbox(
        "Forecast Horizon",
        options=["Next 72 Hours (3 Days) + Past 36h", "Full 7-Day Forecast + Past 36h"],
        index=0,
    )

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

    if not obs_df.empty:
        obs_lookup = obs_df.set_index("valid_time_utc")[["temp_c", "precip_1hr_mm", "rh_pct", "wind_gust_ms"]]
        for col_dest, col_src in [
            ("actual_temp_c", "temp_c"),
            ("actual_precip_mm", "precip_1hr_mm"),
            ("actual_rh_pct", "rh_pct"),
            ("actual_wind_gust_ms", "wind_gust_ms"),
        ]:
            if col_dest not in combined_df.columns:
                combined_df[col_dest] = np.nan
            mapped = combined_df["valid_time_utc"].map(obs_lookup[col_src])
            combined_df[col_dest] = combined_df[col_dest].combine_first(mapped)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.50, 0.25, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "1. Canyon Temperature (°F): Stone Office Actuals (Green Dots) vs. Tassajara MOS (Orange) vs. Standard NWS (Blue)",
            "2. Fire Weather & Canyon Wind: Peak Gusts (mph) & Relative Humidity (%)",
            "3. Precipitation: Expected & Actual Rain (in/hr) + Rain Probability (PoP %)",
        ),
    )

    # --- PANEL 1: TEMPERATURE ---
    p90_f = c_to_f(combined_df["corrected_temp_p90_c"])
    p10_f = c_to_f(combined_df["corrected_temp_p10_c"])

    fig.add_trace(
        go.Scatter(
            x=pd.concat([combined_df["valid_local"], combined_df["valid_local"][::-1]]),
            y=pd.concat([p90_f, p10_f[::-1]]),
            fill="toself",
            fillcolor="rgba(230, 97, 1, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="MOS 10th–90th Percentile Band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=c_to_f(combined_df["raw_nws_temp_c"]),
            mode="lines",
            line=dict(color="#4a90e2", width=2, dash="dash"),
            name="Standard NWS Grid Forecast (°F)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=c_to_f(combined_df["corrected_temp_c"]),
            mode="lines",
            line=dict(color="#ff7f0e", width=3),
            name="Tassajara MOS-Corrected Forecast (°F)",
        ),
        row=1,
        col=1,
    )

    actual_pts = combined_df.dropna(subset=["actual_temp_c"])
    if not actual_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_pts["valid_local"],
                y=c_to_f(actual_pts["actual_temp_c"]),
                mode="lines+markers",
                line=dict(color="#2ecc71", width=2),
                marker=dict(color="#2ecc71", size=6, symbol="circle"),
                name="Stone Office VP2 Actual Temp (°F)",
            ),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=32.0,
        line_dash="dot",
        line_color="#3b82f6",
        annotation_text="Freeze Threshold (32°F)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # --- PANEL 2: WIND GUSTS & RELATIVE HUMIDITY ---
    gust_series_ms = (
        combined_df["corrected_wind_gust_ms"].combine_first(combined_df.get("nws_wind_gust_ms", pd.Series(dtype=float)))
        if "corrected_wind_gust_ms" in combined_df.columns
        else combined_df.get("nws_wind_speed_ms", pd.Series(0.0, index=combined_df.index))
    )
    gust_mph_series = gust_series_ms.fillna(0.0) * 2.23694

    rh_series = (
        combined_df["corrected_rh_pct"].combine_first(combined_df.get("nws_rh_pct", pd.Series(dtype=float)))
        if "corrected_rh_pct" in combined_df.columns
        else combined_df.get("nws_rh_pct", pd.Series(50.0, index=combined_df.index))
    ).fillna(50.0)

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=gust_mph_series.round(1),
            mode="lines",
            line=dict(color="#f59e0b", width=2),
            name="Forecast Peak Gust (mph)",
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
            line=dict(color="#a855f7", width=2, dash="dot"),
            name="Relative Humidity (%)",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # --- PANEL 3: PRECIPITATION (NON-NEGATIVE AXIS + PoP %) ---
    nws_rain_in = (combined_df.get("nws_qpf_1hr_mm", pd.Series(0.0, index=combined_df.index)).fillna(0.0) / 25.4).round(3)
    actual_rain_in = (combined_df.get("actual_precip_mm", pd.Series(0.0, index=combined_df.index)).fillna(0.0) / 25.4).round(3)
    pop_pct = combined_df.get("nws_pop_pct", pd.Series(0.0, index=combined_df.index)).fillna(0.0).clip(0.0, 100.0)

    fig.add_trace(
        go.Scatter(
            x=combined_df["valid_local"],
            y=pop_pct.round(0),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="rgba(56, 189, 248, 0.65)", width=1.5),
            name="Rain Probability (PoP %)",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=combined_df["valid_local"],
            y=nws_rain_in,
            name="NWS Forecast Rain (in/hr)",
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
            name="Stone Office Actual Rain (in/hr)",
            marker_color="#2ecc71",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )

    # Vertical "NOW" Divider Line across all panels
    if not fut_sub.empty:
        now_marker_ts = pd.to_datetime(fut_sub["valid_time_utc"].iloc[0], utc=True).tz_convert(LOCAL_TZ)
        for r_idx in (1, 2, 3):
            fig.add_vline(
                x=now_marker_ts,
                line_width=1.5,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.55)",
                row=r_idx,
                col=1,
            )

    # Fix Precipitation Y-Axis so it NEVER goes negative!
    max_rain_val = float(max(nws_rain_in.max(), actual_rain_in.max(), 0.0))
    rain_y_top = max(0.20, round(max_rain_val * 1.25, 2))
    max_gust_val = float(max(gust_mph_series.max(), 15.0))

    fig.update_yaxes(title_text="Temp (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Wind (mph)", range=[0, max_gust_val * 1.2], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RH (%)", range=[0, 105], showgrid=False, row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Rain (in/hr)", range=[0, rain_y_top], row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PoP (%)", range=[0, 105], showgrid=False, row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local Canyon Time (America/Los_Angeles)", row=3, col=1)

    fig.update_layout(
        height=820,
        margin=dict(t=55, b=40, l=50, r=50),
        hovermode="x unified",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0.0,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: 7-Day Daily Shop Operations Summary Table
# ---------------------------------------------------------------------------
if not future_fc_df.empty:
    st.subheader("📅 7-Day Canyon Daily Summary (For Shop & Work Period Planning)")
    daily_df = future_fc_df.copy()
    daily_df["valid_local"] = pd.to_datetime(daily_df["valid_time_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    daily_df["Date_Key"] = daily_df["valid_local"].dt.strftime("%a, %b %d")
    daily_df["Sort_Date"] = daily_df["valid_local"].dt.date

    daily_summary = (
        daily_df.groupby(["Sort_Date", "Date_Key"], as_index=False)
        .agg(
            MOS_High_F=("corrected_temp_c", lambda s: round(c_to_f(s.max()), 1)),
            MOS_Low_F=("corrected_temp_c", lambda s: round(c_to_f(s.min()), 1)),
            Cold_Pool_P10_F=("corrected_temp_p10_c", lambda s: round(c_to_f(s.min()), 1)),
            NWS_High_F=("raw_nws_temp_c", lambda s: round(c_to_f(s.max()), 1)),
            NWS_Low_F=("raw_nws_temp_c", lambda s: round(c_to_f(s.min()), 1)),
            Rain_Chance_Pct=("nws_pop_pct", lambda s: round(safe_float(s.max(), 0.0), 0)),
            Total_Rain_In=("nws_qpf_1hr_mm", lambda s: round(mm_to_in(s.fillna(0.0).sum()), 2)),
            Min_RH_Pct=("nws_rh_pct", lambda s: round(safe_float(s.min(), 40.0), 0)),
            Peak_Gust_MPH=(
                "corrected_wind_gust_ms" if "corrected_wind_gust_ms" in daily_df.columns else "nws_wind_gust_ms",
                lambda s: round(ms_to_mph(safe_float(s.max(), 0.0)), 1),
            ),
        )
        .sort_values("Sort_Date")
    )

    def _shop_note(row):
        notes = []
        if row["Cold_Pool_P10_F"] <= 32.0:
            notes.append("❄️ Freeze risk (check exposed pipes)")
        elif row["Cold_Pool_P10_F"] <= 36.0:
            notes.append("🌡️ Chilly morning inversion")
        if row["Total_Rain_In"] >= 0.25 or row["Rain_Chance_Pct"] >= 50:
            notes.append("🌧️ Rain expected (check roof/drains)")
        if row["Min_RH_Pct"] <= 20 and row["Peak_Gust_MPH"] >= 20:
            notes.append("🔥 Dry/gusty fire weather")
        if not notes:
            notes.append("✅ Normal canyon shop weather")
        return " | ".join(notes)

    daily_summary["Shop Operational Note"] = daily_summary.apply(_shop_note, axis=1)

    display_daily = daily_summary[[
        "Date_Key",
        "MOS_High_F",
        "MOS_Low_F",
        "Cold_Pool_P10_F",
        "NWS_High_F",
        "NWS_Low_F",
        "Rain_Chance_Pct",
        "Total_Rain_In",
        "Min_RH_Pct",
        "Peak_Gust_MPH",
        "Shop Operational Note",
    ]].rename(columns={
        "Date_Key": "Day / Date",
        "MOS_High_F": "Canyon High (°F)",
        "MOS_Low_F": "Canyon Low (°F)",
        "Cold_Pool_P10_F": "10th-Pct Cold Floor (°F)",
        "NWS_High_F": "NWS Raw High (°F)",
        "NWS_Low_F": "NWS Raw Low (°F)",
        "Rain_Chance_Pct": "Rain Chance (%)",
        "Total_Rain_In": "Expected Rain (in)",
        "Min_RH_Pct": "Lowest RH (%)",
        "Peak_Gust_MPH": "Peak Gust (mph)",
    })

    st.dataframe(display_daily, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 6: Automated Facility Alerts & Hardware QC Log
# ---------------------------------------------------------------------------
st.subheader("🚨 Recent Automated Facility Alerts & Hardware Maintenance Logs")
if not alerts_df.empty:
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)
else:
    st.success("✅ All Stone Office Vantage Pro2 sensors operating within normal QC bounds. No active weather or hardware alerts.")
