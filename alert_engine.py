#!/usr/bin/env python3
"""
alert_engine.py
Evaluates MOS-corrected local forecasts and hardware QC diagnostics against
Tassajara operational thresholds, logging events to SQLite/PostgreSQL and
dispatching notifications via Webhook, SMTP Email, and Twilio SMS.
"""

import os
import json
import sqlite3
import smtplib
import logging
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import requests
import pandas as pd

logger = logging.getLogger("TassajaraAlerts")

# Notification Credentials (from environment)
WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER", "")
ALERT_SMS_TO = os.getenv("ALERT_SMS_TO_NUMBER", "")

# Operational Thresholds for Tassajara Canyon
FREEZE_THRESHOLD_C = -1.1        # 30.0 F hard-freeze pipe protection threshold
FIRE_RH_THRESHOLD_PCT = 18.0     # Critical low RH threshold
FIRE_WIND_GUST_MS = 11.2         # 25.0 mph canyon gust during low RH
HIGH_WIND_HAZARD_MS = 15.6       # 35.0 mph sycamore limb hazard in canyon
COOLDOWN_HOURS = 6               # Suppress duplicate alerts within 6 hours


class FacilityAlertManager:
    def __init__(self, db_path: str = "tassajara_weather.db"):
        self.db_path = db_path

    def _is_in_cooldown(self, alert_category: str) -> bool:
        """Checks if an alert of the same category was already dispatched within COOLDOWN_HOURS."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) FROM facility_alert_log
                WHERE alert_category = ? AND triggered_at_utc >= ?
                """,
                (alert_category, cutoff)
            )
            count = cur.fetchone()[0]
        return count > 0

    def _record_and_dispatch(
        self,
        valid_time_utc: str,
        category: str,
        severity: str,
        metric_val: float,
        threshold_val: float,
        message: str
    ) -> None:
        if self._is_in_cooldown(category):
            logger.info(f"Alert [{category}] suppressed by {COOLDOWN_HOURS}h cooldown hysteresis.")
            return

        channels = []
        # 1. Webhook Dispatch (Slack / Mattermost / Local Ops Channel)
        if WEBHOOK_URL:
            try:
                requests.post(
                    WEBHOOK_URL,
                    json={"text": f"*[{severity} - {category}]* {message} (Valid: {valid_time_utc})"},
                    timeout=10
                )
                channels.append("WEBHOOK")
            except Exception as exc:
                logger.error(f"Webhook dispatch failed: {exc}")

        # 2. Twilio SMS Dispatch for CRITICAL alerts
        if severity == "CRITICAL" and TWILIO_SID and TWILIO_TOKEN and ALERT_SMS_TO:
            try:
                twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
                requests.post(
                    twilio_url,
                    auth=(TWILIO_SID, TWILIO_TOKEN),
                    data={"From": TWILIO_FROM, "To": ALERT_SMS_TO, "Body": f"TASSAJARA {category}: {message}"},
                    timeout=10
                )
                channels.append("TWILIO_SMS")
            except Exception as exc:
                logger.error(f"Twilio SMS dispatch failed: {exc}")

        # 3. Persist to Database Audit Log
        now_iso = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO facility_alert_log (
                    triggered_at_utc, valid_time_utc, alert_category,
                    severity, metric_value, threshold_value, message, dispatched_channels
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now_iso, valid_time_utc, category, severity, metric_val, threshold_val, message, ",".join(channels))
            )
            conn.commit()
        logger.warning(f"Alert logged & dispatched [{category}]: {message}")

    def evaluate_forecasts_and_hardware(
        self,
        mos_df: pd.DataFrame,
        diagnostics: Dict[str, Any],
        chews_obs: Dict[str, Any]
    ) -> None:
        """Evaluates the next 24 hours of MOS-corrected forecasts and current hardware QC."""
        next_24h = mos_df.head(24)

        # 1. Hard Freeze Pipe Protection Check (using 10th percentile lower confidence bound)
        freeze_rows = next_24h[next_24h["corrected_temp_p10_c"] <= FREEZE_THRESHOLD_C]
        if len(freeze_rows) >= 2:
            first_hit = freeze_rows.iloc[0]
            min_p10_c = float(freeze_rows["corrected_temp_p10_c"].min())
            min_p10_f = min_p10_c * 9.0 / 5.0 + 32.0
            raw_nws_f = float(first_hit["raw_nws_temp_c"]) * 9.0 / 5.0 + 32.0
            self._record_and_dispatch(
                valid_time_utc=str(first_hit["valid_time_utc"]),
                category="HARD_FREEZE",
                severity="CRITICAL",
                metric_val=min_p10_c,
                threshold_val=FREEZE_THRESHOLD_C,
                message=(
                    f"Canyon cold-air pool freeze risk: MOS P10 forecast drops to {min_p10_f:.1f}°F "
                    f"({min_p10_c:.1f}°C) for {len(freeze_rows)} hrs (Raw NWS forecast: {raw_nws_f:.1f}°F). "
                    f"Inspect and drain exposed bathhouse/cabin water lines."
                )
            )

        # 2. Fire Weather / Ridge Wind Break-In Warning
        fire_rows = next_24h[
            (next_24h.get("corrected_rh_pct", 50.0) <= FIRE_RH_THRESHOLD_PCT)
            & (next_24h.get("corrected_wind_gust_ms", 0.0) >= FIRE_WIND_GUST_MS)
        ]
        chews_gust_ms = chews_obs.get("wind_gust_ms") or 0.0
        chews_rh = chews_obs.get("rh_pct") or 50.0
        lapse = diagnostics.get("lapse_rate_chews_c_km") or 0.0

        if not fire_rows.empty or (chews_gust_ms >= 17.8 and chews_rh <= 18.0 and lapse >= 7.5):
            valid_t = str(fire_rows.iloc[0]["valid_time_utc"]) if not fire_rows.empty else datetime.now(timezone.utc).isoformat()
            self._record_and_dispatch(
                valid_time_utc=valid_t,
                category="FIRE_WEATHER",
                severity="CRITICAL",
                metric_val=chews_rh,
                threshold_val=FIRE_RH_THRESHOLD_PCT,
                message=(
                    f"Critical Fire Weather conditions detected/forecast in Tassajara Canyon. "
                    f"Chews Ridge gusting {chews_gust_ms*2.237:.0f} mph at {chews_rh:.0f}% RH with mixing lapse rate "
                    f"{lapse:.1f} C/km. Initiate Tassajara fire-watch protocol."
                )
            )

        # 3. High Canyon Wind / Sycamore Canopy Hazard
        wind_rows = next_24h[next_24h.get("corrected_wind_gust_ms", 0.0) >= HIGH_WIND_HAZARD_MS]
        if not wind_rows.empty:
            peak_row = wind_rows.loc[wind_rows["corrected_wind_gust_ms"].idxmax()]
            gust_mph = float(peak_row["corrected_wind_gust_ms"]) * 2.23694
            self._record_and_dispatch(
                valid_time_utc=str(peak_row["valid_time_utc"]),
                category="HIGH_WIND",
                severity="WARNING",
                metric_val=float(peak_row["corrected_wind_gust_ms"]),
                threshold_val=HIGH_WIND_HAZARD_MS,
                message=(
                    f"MOS forecast predicts canyon wind gusts up to {gust_mph:.1f} mph. "
                    f"Secure outdoor fixtures and exercise caution under sycamore canopy near Stone Office."
                )
            )

        # 4. Automated Hardware Maintenance & Sensor QC Ticket
        if diagnostics.get("qc_flag", "OK") != "OK":
            notes_str = " | ".join(diagnostics.get("qc_notes", []))
            self._record_and_dispatch(
                valid_time_utc=datetime.now(timezone.utc).isoformat(),
                category="HARDWARE_QC",
                severity="ADVISORY",
                metric_val=diagnostics.get("lapse_rate_chews_c_km") or 0.0,
                threshold_val=12.5,
                message=f"Automated Vantage Pro2 QC Trigger [{diagnostics['qc_flag']}]: {notes_str}"
            )