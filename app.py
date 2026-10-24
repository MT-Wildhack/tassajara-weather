#!/usr/bin/env python3
"""
app.py
Streamlit real-time visualization dashboard comparing generic NWS forecasts
against the Tassajara MOS-corrected canyon forecast.
"""

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Tassajara Canyon Microclimate & MOS Forecast",
    layout="wide"
)

DB_PATH = "tassajara_weather.db"


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


@st.cache_data(ttl=300)
def load_dashboard_data():
    with sqlite3.connect(DB_PATH) as conn:
        obs_df = pd.read_sql_query(
            "SELECT * FROM local_observations ORDER BY valid_time_utc DESC LIMIT 96", conn
        )
        fc_df = pd.read_sql_query(
            """
            SELECT m.*, o.temp_c AS actual_temp_c
            FROM mos_corrected_forecasts m
            LEFT JOIN local_observations o ON m.valid_time_utc = o.valid_time_utc
            WHERE m.issue_time_utc = (SELECT MAX(issue_time_utc) FROM mos_corrected_forecasts)
            ORDER BY m.valid_time_utc ASC
            """,
            conn
        )
        alerts_df = pd.read_sql_query(
            "SELECT * FROM facility_alert_log ORDER BY triggered_at_utc DESC LIMIT 10", conn
        )
    return obs_df, fc_df, alerts_df


st.title("Tassajara Zen Mountain Center — Local Forecast & Microclimate Monitor")
st.caption("Stone Office Davis Vantage Pro2 (1,590 ft) | Reference: Chews Ridge CHWC1 (5,040 ft) & Hastings RAWS CAHC1 (1,792 ft)")

try:
    obs_df, fc_df, alerts_df = load_dashboard_data()
except Exception:
    obs_df, fc_df, alerts_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if not obs_df.empty:
    latest = obs_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stone Office Temp", f"{c_to_f(latest['temp_c']):.1f} °F", f"{latest['temp_c']:.1f} °C")
    c2.metric("Relative Humidity", f"{latest['rh_pct']:.0f} %")
    c3.metric("Peak Wind Gust", f"{(latest['wind_gust_ms'] or 0)*2.23694:.1f} mph")
    c4.metric("Sensor QC Status", latest["qc_flag"])

if not fc_df.empty:
    fc_df["valid_local"] = pd.to_datetime(fc_df["valid_time_utc"], utc=True).dt.tz_convert("America/Los_Angeles")

    fig = go.Figure()

    # 10th-90th Percentile Uncertainty Ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["valid_local"], fc_df["valid_local"][::-1]]),
        y=pd.concat([c_to_f(fc_df["corrected_temp_p90_c"]), c_to_f(fc_df["corrected_temp_p10_c"])[::-1]]),
        fill="toself",
        fillcolor="rgba(230, 97, 1, 0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        name="MOS 10th–90th Percentile Band"
    ))

    # Generic NWS Forecast
    fig.add_trace(go.Scatter(
        x=fc_df["valid_local"],
        y=c_to_f(fc_df["raw_nws_temp_c"]),
        mode="lines",
        line=dict(color="#377eb8", width=2, dash="dash"),
        name="Standard NWS Grid Forecast"
    ))

    # Tassajara MOS-Corrected Forecast
    fig.add_trace(go.Scatter(
        x=fc_df["valid_local"],
        y=c_to_f(fc_df["corrected_temp_c"]),
        mode="lines+markers",
        line=dict(color="#e66101", width=3),
        name="Tassajara MOS-Corrected Forecast"
    ))

    # Ground-Truth Vantage Pro2 Actuals
    valid_actuals = fc_df.dropna(subset=["actual_temp_c"])
    if not valid_actuals.empty:
        fig.add_trace(go.Scatter(
            x=valid_actuals["valid_local"],
            y=c_to_f(valid_actuals["actual_temp_c"]),
            mode="markers",
            marker=dict(color="#111111", size=8, symbol="circle"),
            name="Stone Office VP2 Actuals"
        ))

    # Freeze Threshold Line (32 F)
    fig.add_hline(y=32.0, line_dash="dot", line_color="royalblue", annotation_text="Freeze Threshold (32°F)")

    fig.update_layout(
        title="48-Hour Forecast Comparison: Standard NWS vs. Canyon-Corrected MOS (°F)",
        xaxis_title="Local Time (America/Los_Angeles)",
        yaxis_title="Temperature (°F)",
        hovermode="x unified",
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Automated Facility Alerts & Hardware Maintenance Logs")
if not alerts_df.empty:
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.info("No active critical weather or hardware QC alerts triggered.")