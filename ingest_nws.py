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
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
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
            start_dt = (
                datetime.fromisoformat(start_str)
                .astimezone(timezone.utc)
                .replace(minute=0, second=0, microsecond=0)
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
            dt_utc = (
                datetime.fromisoformat(t_str)
                .replace(tzinfo=timezone.utc)
                .isoformat()
            )
            p_val = (
                float(precips[i])
                if i < len(precips) and precips[i] is not None
                else 0.0
            )
            c_val = (
                float(clouds[i])
                if i < len(clouds) and clouds[i] is not None
                else 20.0
            )
            g_val = (
                float(gusts[i])
                if i < len(gusts) and gusts[i] is not None
                else 0.0
            )
            supp[dt_utc] = (p_val, c_val, g_val)
    except Exception as exc:
        logger.warning(f"Open-Meteo QPF supplement unavailable: {exc}")
    return supp


def synthesize_qpf_from_pop_and_text(
    pop_pct: float,
    rh_pct: float,
    short_fc: str,
) -> float:
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
    if (
        any(k in desc for k in ("drizzle", "light rain", "slight chance", "isolated"))
        or pop_pct >= 25.0
    ):
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
            g_resp = requests.get(
                grid_url,
                headers=NWS_HEADERS,
                timeout=15,
            )
            g_resp.raise_for_status()
            g_props = g_resp.json().get("properties", {})

            qpf_obj = g_props.get("quantitativePrecipitation", {})
            grid_qpf_map = expand_nws_grid_series(
                qpf_obj.get("values", []),
                is_accumulation=True,
                unit_scale=1.0,
            )

            sky_obj = g_props.get("skyCover", {})
            grid_sky_map = expand_nws_grid_series(
                sky_obj.get("values", []),
                is_accumulation=False,
                unit_scale=1.0,
            )

            gust_obj = g_props.get("windGust", {})
            uom = (gust_obj.get("uom") or "").lower()
            if "km_h" in uom or "km/h" in uom:
                gust_scale = 1.0 / 3.6
            elif "mph" in uom:
                gust_scale = 0.44704
            else:
                gust_scale = 1.0
            grid_gust_map = expand_nws_grid_series(
                gust_obj.get("values", []),
                is_accumulation=False,
                unit_scale=gust_scale,
            )
        except Exception as exc:
            logger.warning(
                f"NWS forecastGridData fetch failed ({exc}); using supplement."
            )

    om_supp = fetch_open_meteo_qpf_supplement()

    # Step 3: Fetch Hourly Forecast Periods (156 hours)
    fc_resp = requests.get(
        hourly_url,
        headers=NWS_HEADERS,
        timeout=15,
    )
    fc_resp.raise_for_status()
    props = fc_resp.json()["properties"]

    issue_dt = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    issue_iso = issue_dt.isoformat()

    rows_to_insert = []
    for period in props.get("periods", []):
        valid_dt = (
            datetime.fromisoformat(period["startTime"])
            .astimezone(timezone.utc)
            .replace(minute=0, second=0, microsecond=0)
        )
        lead_hours = int((valid_dt - issue_dt).total_seconds() // 3600)
        if lead_hours < 0:
            continue

        valid_iso = valid_dt.isoformat()
        temp_f = period["temperature"]
        if period["temperatureUnit"] == "F":
            temp_c = round((temp_f - 32.0) * 5.0 / 9.0, 2)
        else:
            temp_c = float(temp_f)

        dew_c = period.get("dewpoint", {}).get("value")
        rh_pct = float(period.get("relativeHumidity", {}).get("value") or 50.0)
        pop_pct = float(
            period.get("probabilityOfPrecipitation", {}).get("value") or 0.0
        )
        w_spd_ms = round(parse_wind_speed_mph(period.get("windSpeed", "")), 2)
        w_dir_deg = COMPASS_TO_DEG.get(period.get("windDirection", "N"), 0.0)
        short_fc = period.get("shortForecast", "")

        om_qpf, om_sky, om_gust = om_supp.get(valid_iso, (0.0, 20.0, 0.0))

        nws_grid_qpf = grid_qpf_map.get(valid_iso, 0.0)
        synth_qpf = synthesize_qpf_from_pop_and_text(pop_pct, rh_pct, short_fc)
        if nws_grid_qpf > 0.0:
            qpf_1hr_mm = round(nws_grid_qpf, 2)
        elif om_qpf > 0.0:
            qpf_1hr_mm = round(om_qpf, 2)
        else:
            qpf_1hr_mm = synth_qpf

        sky_pct = round(grid_sky_map.get(valid_iso, om_sky), 1)
        w_gust_ms = round(
            max(w_spd_ms * 1.35, grid_gust_map.get(valid_iso, 0.0), om_gust), 2
        )

        rows_to_insert.append(
            (
                issue_iso,
                valid_iso,
                lead_hours,
                temp_c,
                round(float(dew_c), 2) if dew_c is not None else None,
                rh_pct,
                w_spd_ms,
                w_gust_ms,
                w_dir_deg,
                pop_pct,
                qpf_1hr_mm,
                sky_pct,
                short_fc,
            )
        )

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO nws_forecasts (
                issue_time_utc, valid_time_utc, lead_time_hours,
                nws_temp_c, nws_dewpoint_c, nws_rh_pct,
                nws_wind_speed_ms, nws_wind_gust_ms, nws_wind_dir_deg,
                nws_pop_pct, nws_qpf_1hr_mm, nws_sky_cover_pct, short_forecast
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()

    wet_fc_hrs = sum(1 for r in rows_to_insert if r[10] > 0.0)
    logger.info(
        f"Stored {len(rows_to_insert)} NWS hourly periods "
        f"({wet_fc_hrs} hours with QPF > 0 mm)."
    )
    return len(rows_to_insert)


if __name__ == "__main__":
    fetch_and_store_nws_hourly()
