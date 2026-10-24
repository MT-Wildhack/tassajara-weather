#!/usr/bin/env python3
"""
ingest_nws.py
Queries the NWS API (api.weather.gov) for Tassajara's exact grid cell
and writes hourly forecasts into the SQLite/PostgreSQL database.
"""

import re
import sqlite3
from datetime import datetime, timezone
import requests

TASS_LAT = 36.2337
TASS_LON = -121.5501
NWS_HEADERS = {
    "User-Agent": "(TassajaraZenMountainCenter-MOS/1.0, weather-eng@sfzc.org)",
    "Accept": "application/geo+json"
}

COMPASS_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5
}


def parse_wind_speed_mph(wind_str: str) -> float:
    nums = [float(x) for x in re.findall(r"\d+", wind_str or "")]
    return max(nums) * 0.44704 if nums else 0.0


def fetch_and_store_nws_hourly(db_path: str = "tassajara_weather.db") -> int:
    # Step 1: Resolve Gridpoint URL
    pt_resp = requests.get(
        f"https://api.weather.gov/points/{TASS_LAT},{TASS_LON}",
        headers=NWS_HEADERS, timeout=15
    )
    pt_resp.raise_for_status()
    hourly_url = pt_resp.json()["properties"]["forecastHourly"]

    # Step 2: Fetch Hourly Forecast Periods (156 hours)
    fc_resp = requests.get(hourly_url, headers=NWS_HEADERS, timeout=15)
    fc_resp.raise_for_status()
    props = fc_resp.json()["properties"]

    issue_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    issue_iso = issue_dt.isoformat()

    rows_to_insert = []
    for period in props.get("periods", []):
        valid_dt = datetime.fromisoformat(period["startTime"]).astimezone(timezone.utc)
        lead_hours = int((valid_dt - issue_dt).total_seconds() // 3600)
        if lead_hours < 0:
            continue

        temp_f = period["temperature"]
        temp_c = round((temp_f - 32.0) * 5.0 / 9.0, 2) if period["temperatureUnit"] == "F" else temp_f
        dew_c = period.get("dewpoint", {}).get("value")
        rh_pct = period.get("relativeHumidity", {}).get("value")
        pop_pct = period.get("probabilityOfPrecipitation", {}).get("value") or 0.0
        w_spd_ms = round(parse_wind_speed_mph(period.get("windSpeed", "")), 2)
        w_dir_deg = COMPASS_TO_DEG.get(period.get("windDirection", "N"), 0.0)

        rows_to_insert.append((
            issue_iso,
            valid_dt.isoformat(),
            lead_hours,
            temp_c,
            round(dew_c, 2) if dew_c is not None else None,
            rh_pct,
            w_spd_ms,
            w_dir_deg,
            pop_pct,
            period.get("shortForecast", "")
        ))

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO nws_forecasts (
                issue_time_utc, valid_time_utc, lead_time_hours,
                nws_temp_c, nws_dewpoint_c, nws_rh_pct,
                nws_wind_speed_ms, nws_wind_dir_deg, nws_pop_pct, short_forecast
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert
        )
        conn.commit()
    return len(rows_to_insert)