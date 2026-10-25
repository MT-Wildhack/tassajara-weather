#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

st.set_page_config(
    page_title="Tassajara Canyon Microclimate & MOS Forecast",
    layout="wide"
)

DB_PATH = "tassajara_weather.db"

def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0

def mm_to_inches(mm):
    if pd.isna(mm) or mm is None:
        return 0.0
    return mm / 25.4

@st.cache_data(ttl=300)
def load_dashboard_data():
    with sqlite3.connect(DB_PATH) as conn:
        obs_df = pd.read_sql_query(
            "SELECT * FROM local_observations ORDER BY valid_time_utc DESC LIMIT 96", conn
        )
        fc_df = pd.read_sql_query(
            '''
            SELECT m.*, o.temp_c AS actual_temp_c, o.precip_1hr_mm,
                   f.nws_qpf_1hr_mm
            FROM mos_corrected_forecasts m
            LEFT JOIN local_observations o ON m.valid_time_utc = o.valid_time_utc
            LEFT JOIN nws_forecasts f ON m.issue_time_utc = f.issue_time_utc AND m.valid_time_utc = f.valid_time_utc
            WHERE m.issue_time_utc = (SELECT MAX(issue_time_utc) FROM mos_corrected_forecasts)
            ORDER BY m.valid_time_utc ASC
            ''',
            conn
        )
        
        # Load the LAST forecast run (to compare historical predictions against current actuals)
        # We look for the forecast issued 24 hours ago
        historical_fc_df = pd.read_sql_query(
            '''
            SELECT m.valid_time_utc, m.raw_nws_temp_c, m.corrected_temp_c, o.temp_c AS actual_temp_c
            FROM mos_corrected_forecasts m
            INNER JOIN local_observations o ON m.valid_time_utc = o.valid_time_utc
            WHERE o.temp_c IS NOT NULL
            AND m.issue_time_utc = (
                SELECT MAX(issue_time_utc) 
                FROM mos_corrected_forecasts 
                WHERE issue_time_utc <= datetime('now', '-24 hours')
            )
            ORDER BY m.valid_time_utc ASC
            ''',
            conn
        )
        
        alerts_df = pd.read_sql_query(
            "SELECT * FROM facility_alert_log ORDER BY triggered_at_utc DESC LIMIT 10", conn
        )
    return obs_df, fc_df, historical_fc_df, alerts_df

st.title("Tassajara Zen Mountain Center — Local Forecast & Microclimate Monitor")
st.caption("Stone Office Davis Vantage Pro2 (1,590 ft) | Reference: Chews Ridge CHWC1 (5,040 ft) & Hastings RAWS CAHC1 (1,792 ft)")

try:
    obs_df, fc_df, historical_fc_df, alerts_df = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    obs_df, fc_df, historical_fc_df, alerts_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if not obs_df.empty:
    latest = obs_df.iloc[0]
    
    # Fix the NaN glitch for wind gusts
    gust_ms = latest['wind_gust_ms']
    gust_mph = (gust_ms * 2.23694) if pd.notna(gust_ms) else 0.0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stone Office Temp", f"{c_to_f(latest['temp_c']):.1f} °F", f"{latest['temp_c']:.1f} °C")
    c2.metric("Relative Humidity", f"{latest['rh_pct']:.0f} %")
    c3.metric("Peak Wind Gust", f"{gust_mph:.1f} mph")
    c4.metric("Sensor QC Status", latest["qc_flag"])

# --- NEW SECTION: MODEL ACCURACY METRICS ---
if not historical_fc_df.empty and len(historical_fc_df) > 5:
    st.markdown("---")
    st.subheader("Model Accuracy Tracker (Trailing 24-Hour Evaluation)")
    
    # Calculate RMSE in Fahrenheit for readability
    actuals_f = c_to_f(historical_fc_df['actual_temp_c'])
    nws_fc_f = c_to_f(historical_fc_df['raw_nws_temp_c'])
    mos_fc_f = c_to_f(historical_fc_df['corrected_temp_c'])
    
    try:
        raw_rmse = root_mean_squared_error(actuals_f, nws_fc_f)
        mos_rmse = root_mean_squared_error(actuals_f, mos_fc_f)
        improvement = 1.0 - (mos_rmse / raw_rmse)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Generic NWS Forecast Error", f"± {raw_rmse:.1f} °F", help="Average error of the standard regional grid forecast over the last 24 hours.")
        m2.metric("Tassajara MOS Error", f"± {mos_rmse:.1f} °F", delta=f"{raw_rmse - mos_rmse:.1f} °F Better", delta_color="normal", help="Average error of our canyon-corrected model.")
        m3.metric("Overall Accuracy Improvement", f"+{improvement * 100:.1f} %", help="How much better the MOS model is performing compared to the generic NWS grid.")
    except Exception as e:
        st.caption("Insufficient clean data to calculate rolling accuracy.")
    st.markdown("---")

if not fc_df.empty:
    fc_df["valid_local"] = pd.to_datetime(fc_df["valid_time_utc"], utc=True).dt.tz_convert("America/Los_Angeles")

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Temperature Comparison (°F)", "Precipitation (Inches / Hr)")
    )

    # --- ROW 1: TEMPERATURE FAN CHART ---
    
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["valid_local"], fc_df["valid_local"][::-1]]),
        y=pd.concat([c_to_f(fc_df["corrected_temp_p90_c"]), c_to_f(fc_df["corrected_temp_p10_c"])[::-1]]),
        fill="toself",
        fillcolor="rgba(230, 97, 1, 0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        name="MOS 10th–90th Percentile Band",
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fc_df["valid_local"],
        y=c_to_f(fc_df["raw_nws_temp_c"]),
        mode="lines",
        line=dict(color="#377eb8", width=2, dash="dash"),
        name="Standard NWS Grid Forecast"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fc_df["valid_local"],
        y=c_to_f(fc_df["corrected_temp_c"]),
        mode="lines+markers",
        line=dict(color="#e66101", width=3),
        name="Tassajara MOS-Corrected Forecast"
    ), row=1, col=1)

    valid_actuals = fc_df.dropna(subset=["actual_temp_c"])
    if not valid_actuals.empty:
        fig.add_trace(go.Scatter(
            x=valid_actuals["valid_local"],
            y=c_to_f(valid_actuals["actual_temp_c"]),
            mode="markers",
            marker=dict(color="#111111", size=8, symbol="circle"),
            name="Stone Office VP2 Actuals"
        ), row=1, col=1)

    fig.add_hline(y=32.0, line_dash="dot", line_color="royalblue", annotation_text="Freeze Threshold (32°F)", row=1, col=1)

    # --- ROW 2: PRECIPITATION BAR CHART ---
    
    nws_rain_in = (fc_df["nws_qpf_1hr_mm"].fillna(0) / 25.4).round(2)
    fig.add_trace(go.Bar(
        x=fc_df["valid_local"],
        y=nws_rain_in,
        name="NWS Forecast Rain (in/hr)",
        marker_color="#377eb8",
        opacity=0.6
    ), row=2, col=1)

    actual_rain_in = (fc_df["precip_1hr_mm"].fillna(0) / 25.4).round(2)
    fig.add_trace(go.Bar(
        x=fc_df["valid_local"],
        y=actual_rain_in,
        name="Stone Office Rain Gauge Actuals",
        marker_color="#111111"
    ), row=2, col=1)

    fig.update_layout(
        title="48-Hour Tassajara Microclimate Forecast",
        hovermode="x unified",
        height=700,
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Rain (in)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Automated Facility Alerts & Hardware Maintenance Logs")
if not alerts_df.empty:
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.info("No active critical weather or hardware QC alerts triggered.")
