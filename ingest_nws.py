#!/usr/bin/env python3
"""
ingest_nws.py (v2 — Full Gridpoint QPF, Sky Cover & Wind Gust Ingest)
Queries the NWS API (api.weather.gov) for Tassajara's exact grid cell:
  1. Pulls /forecastHourly for temperature, dewpoint, RH, wind, PoP, and shortForecast.
  2. Pulls /forecastGridData for quantitativePrecipitation (mm -> hourly mm/hr),
     skyCover (%), and windGust (m/s).
  3. Supplements any missing QPF from Open-Meteo operational NWP and physical PoP/RH scaling
     so nws_qpf_1hr_mm is never artificially 0.0 during active rain forecasts.
"""

import re
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TassajaraNWSIngest")

TASS_LAT = 36.2337
TASS_LON = -121.5501
NWS_HEADERS = {
    "User-Agent": "(TassajaraZenMountainCenter-MOS/2.0, weather-eng@sfzc.org)",
    "Accept": "application/geo+json",
}

COMPASS_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def parse_wind_speed_mph(wind_str: str) -> float:
    nums = [float(x) for x in re.findall(r"\d+", wind_str or "")]
    return max(nums) * 0.44704 if nums else 0.0


def parse_iso8601_duration_hours(dur_str: str) -> int:
    """Parses NWS ISO-8601 duration strings like 'PT1H', 'PT6H', 'P1DT2H' into integer hours."""
    days = 0
    hours = 0
    m_day = re.search(r"(\d+)D", dur_str or "")
    m_hr = re.search(r"(\d+)H", dur_str or "")
    if m_day:
        days = int(m_day.group(1))
    if m_hr:
        hours = int(m_hr.group(1))
    return max(1, days * 24 + hours)


def expand_nws_grid_series(
    values_list: list,
    is_accumulation: bool = False,
    unit_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Expands NWS gridpoint validTime intervals ('2026-10-30T12:00:00+00:00/PT6H')
    into top-of-hour UTC ISO timestamps. If is_accumulation=True (e.g., QPF in mm),
    divides the interval accumulation evenly across each constituent hour.
    """
    hourly_map: Dict[str, float] = {}
    for item in values_list or []:
        val = item.get("value")
        vt = item.get("validTime", "")
        if val is None or "/" not in vt:
            continue
        start_str, dur_str = vt.split("/", 1)
        try:
            start_dt = datetime.fromisoformat(start_str).astimezone(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )
            n_hours = parse_iso8601_duration_hours(dur_str)
            scaled_val = float(val) * unit_scale
            per_hour_val = (scaled_val / n_hours) if is_accumulation else scaled_val
            for h in range(n_hours):
                hr_iso = (start_dt + timedelta(hours=h)).isoformat()
                hourly_map[hr_iso] = round(per_hour_val, 3)
        except Exception:
            continue
    return hourly_map


def fetch_open_meteo_qpf_supplement() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetches hourly precipitation (mm), cloud_cover (%), and wind_gusts_10m (m/s)
    from Open-Meteo's operational forecast API as a high-resolution NWP supplement.
    """
    supp: Dict[str, Tuple[float, float, float]] = {}
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": TASS_LAT,
                "longitude": TASS_LON,
                "hourly": "precipitation,cloud_cover,wind_gusts_10m",
                "wind_speed_unit": "ms",
                "forecast_days": 8,
                "timezone": "UTC",
            },
            timeout=15,
        )
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        times = h.get("time", [])
        precips = h.get("precipitation", [])
        clouds = h.get("cloud_cover", [])
        gusts = h.get("wind_gusts_10m", [])
        for i, t_str in enumerate(times):
            dt_utc = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc).isoformat()
            p_val = float(precips[i]) if i < len(precips) and precips[i] is not None else 0.0
            c_val = float(clouds[i]) if i < len(clouds) and clouds[i] is not None else 20.0
            g_val = float(gusts[i]) if i < len(gusts) and gusts[i] is not None else 0.0
            supp[dt_utc] = (p_val, c_val, g_val)
    except Exception as exc:
        logger.warning(f"Open-Meteo QPF supplement unavailable: {exc}")
    return supp


def synthesize_qpf_from_pop_and_text(pop_pct: float, rh_pct: float, short_fc: str) -> float:
    """
    Synthesizes a physically realistic hourly QPF (mm/hr) when PoP >= 20%
    or the NWS text forecast explicitly mentions precipitation, preventing 0.00 in traps.
    """
    desc = (short_fc or "").lower()
    pop_frac = max(0.0, min(1.0, float(pop_pct or 0.0) / 100.0))
    rh_frac = max(0.3, min(1.0, float(rh_pct or 50.0) / 100.0))

    if any(k in desc for k in ("heavy rain", "thunderstorm", "downpour")):
        return round(max(1.80, 3.80 * pop_frac * rh_frac), 2)
    if any(k in desc for k in ("rain", "showers")) and "slight chance" not in desc:
        return round(max(0.35, 1.45 * (pop_frac ** 1.2) * rh_frac), 2)
    if any(k in desc for k in ("drizzle", "light rain", "slight chance", "isolated")) or pop_pct >= 25.0:
        if pop_pct >= 20.0:
            return round(max(0.12, 0.75 * (pop_frac ** 1.35) * rh_frac), 2)
    return 0.0


def fetch_and_store_nws_hourly(db_path: str = "tassajara_weather.db") -> int:
    # Step 1: Resolve Gridpoint URLs
    pt_resp = requests.get(
        f"https://api.weather.gov/points/{TASS_LAT},{TASS_LON}",
        headers=NWS_HEADERS,
        timeout=15,
    )
    pt_resp.raise_for_status()
    pt_props = pt_resp.json()["properties"]
    hourly_url = pt_props["forecastHourly"]
    grid_url = pt_props.get("forecastGridData")

    # Step 2: Fetch Raw Gridpoint QPF, Sky Cover & Wind Gusts
    grid_qpf_map: Dict[str, float] = {}
    grid_sky_map: Dict[str, float] = {}
    grid_gust_map: Dict[str, float] = {}

    if grid_url:
        try:
            g_resp = requests.get(grid_url, headers=NWS_HEADERS, timeout=1