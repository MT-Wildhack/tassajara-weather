#!/usr/bin/env python3
"""
ingest_nws.py (v3.1 — Resilient NOAA NWS /gridpoints/MTR/... + Open-Meteo NWP Ingest)
Queries the NOAA NWS API (api.weather.gov) for the Tassajara grid cell (36.2337, -121.5501):
  1. Resolves /points/36.2337,-121.5501 -> /gridpoints/MTR/{x},{y} (forecastGridData).
  2. Parses all raw NDFD gridpoint series: quantitativePrecipitation (disaggregated to mm/hr),
     probabilityOfPrecipitation (%), temperature (C), dewpoint (C), relativeHumidity (%),
     skyCover (%), windSpeed (m/s), windGust (m/s), and windDirection (deg).
  3. Enriches hourly rows with /forecastHourly text descriptions when available, without
     failing if /forecastHourly returns a transient 500/503 error.
  4. Supplements with Open-Meteo NWP at 880m MSL so nws_qpf_1hr_mm is never 0.0 during storms.
"""

import re
import time
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TassajaraNWSIngest")

TASS_LAT = 36.2337
TASS_LON = -121.5501
NWP_GRID_ELEV_M = 880.0

NWS_HEADERS = {
    "User-Agent": "(TassajaraZenMountainCenter-MOS/3.1, tzmc-weather@tassajara.org)",
    "Accept": "application/geo+json",
}

COMPASS_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

ISO_DUR_REGEX = re.compile(
    r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)


def parse_wind_speed_mph(wind_str: str) -> float:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(wind_str or ""))]
    return max(nums) * 0.44704 if nums else 0.0


def parse_iso8601_duration_hours(dur_str: str) -> int:
    """Parses NWS ISO-8601 duration strings ('PT1H', 'PT3H', 'PT6H', 'P1DT2H') into integer hours."""
    s = (dur_str or "").strip()
    m = ISO_DUR_REGEX.match(s)
    if m:
        days = int(m.group(1) or 0)
        hours = int(m.group(2) or 0)
        mins = int(m.group(3) or 0)
        return max(1, days * 24 + hours + (1 if mins >= 30 else 0))
    m_day = re.search(r"(\d+)D", s)
    m_hr = re.search(r"(\d+)H", s)
    days = int(m_day.group(1)) if m_day else 0
    hours = int(m_hr.group(1)) if m_hr else 0
    return max(1, days * 24 + hours)


def _convert_grid_value(val: float, uom: str, is_accumulation: bool = False) -> float:
    """
    Converts a raw NWS gridpoint value to metric (mm, C, m/s, %, deg) using exact
    unit token matching so 'wmoUnit:mm' is never mistaken for meters ('m').
    """
    u = (uom or "").strip().lower()
    unit_token = u.split(":")[-1].split("/")[-1].strip()
    if is_accumulation:
        if unit_token in ("mm", "kg_m-2", "kg/m2", "kg/m^2"):
            return val
        if unit_token in ("in", "inch", "inches"):
            return val * 25.4
        if unit_token in ("m", "meter", "meters"):
            return val * 1000.0
        if unit_token in ("cm", "centimeter", "centimeters"):
            return val * 10.0
        return val
    if "degf" in unit_token or unit_token == "f":
        return (val - 32.0) * 5.0 / 9.0
    if "degk" in unit_token or unit_token == "k":
        return val - 273.15
    if "km_h-1" in unit_token or "km/h" in unit_token:
        return val / 3.6
    if "mi_h-1" in unit_token or "mph" in unit_token:
        return val * 0.44704
    if unit_token in ("kt", "kn", "knots"):
        return val * 0.514444
    return val


def expand_nws_grid_series(
    prop_or_values: Any,
    is_accumulation: bool = False,
    unit_scale: Optional[float] = None,
) -> Dict[str, float]:
    """
    Expands a NWS /gridpoints/MTR/{x},{y} property dict (or values list) with validTime
    intervals ('2026-11-13T12:00:00+00:00/PT6H') into top-of-hour UTC ISO strings.
    When is_accumulation=True (e.g., quantitativePrecipitation), divides the interval total
    by the interval duration in hours to yield the hourly accumulation rate (mm/hr).
    """
    if isinstance(prop_or_values, dict):
        uom = str(prop_or_values.get("uom") or "")
        values_list = prop_or_values.get("values") or []
    elif isinstance(prop_or_values, list):
        uom = ""
        values_list = prop_or_values
    else:
        return {}

    hourly_map: Dict[str, float] = {}
    for item in values_list:
        if not isinstance(item, dict):
            continue
        vt = str(item.get("validTime") or "")
        if "/" not in vt:
            continue
        raw_val = item.get("value")
        if raw_val is None:
            if is_accumulation:
                raw_val = 0.0
            else:
                continue
        start_str, dur_str = vt.split("/", 1)
        try:
            start_dt = (
                datetime.fromisoformat(start_str)
                .astimezone(timezone.utc)
                .replace(minute=0, second=0, microsecond=0)
            )
            n_hours = parse_iso8601_duration_hours(dur_str)
            val_f = float(raw_val)
            if unit_scale is not None:
                scaled_val = val_f * float(unit_scale)
            else:
                scaled_val = _convert_grid_value(val_f, uom, is_accumulation=is_accumulation)
            per_hour_val = max(0.0, scaled_val / float(n_hours)) if is_accumulation else scaled_val
            for h in range(n_hours):
                hr_iso = (start_dt + timedelta(hours=h)).isoformat()
                hourly_map[hr_iso] = round(per_hour_val, 3)
        except Exception:
            continue
    return hourly_map


def _get_json_with_retries(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 12,
    retries: int = 3,
) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(1.2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    return None


def resolve_nws_endpoints(lat: float = TASS_LAT, lon: float = TASS_LON) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves (forecastGridData_url, forecastHourly_url) for (lat, lon) via /points/{lat},{lon}.
    """
    pt_data = _get_json_with_retries(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers=NWS_HEADERS,
        timeout=12,
        retries=3,
    )
    if not isinstance(pt_data, dict):
        return None, None
    pt_props = pt_data.get("properties") or {}
    grid_url = pt_props.get("forecastGridData")
    hourly_url = pt_props.get("forecastHourly")

    if (
        not grid_url
        and pt_props.get("gridId")
        and pt_props.get("gridX") is not None
        and pt_props.get("gridY") is not None
    ):
        grid_url = (
            f"https://api.weather.gov/gridpoints/"
            f"{pt_props['gridId']}/{pt_props['gridX']},{pt_props['gridY']}"
        )
    elif not grid_url and hourly_url and "/forecast/hourly" in hourly_url:
        grid_url = hourly_url.split("/forecast/hourly")[0]

    if not hourly_url and grid_url:
        hourly_url = f"{grid_url.rstrip('/')}/forecast/hourly"

    return grid_url, hourly_url


def fetch_open_meteo_nwp_map(past_days: int = 1, forecast_days: int = 8) -> Dict[str, Dict[str, float]]:
    """
    Fetches full hourly NWP series at 880m MSL from Open-Meteo as a high-resolution
    supplement or cross-check. Returns {valid_time_utc_iso: dict_of_metrics}.
    """
    supp: Dict[str, Dict[str, float]] = {}
    params = {
        "latitude": TASS_LAT,
        "longitude": TASS_LON,
        "elevation": NWP_GRID_ELEV_M,
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "precipitation_probability,precipitation,cloud_cover,"
            "wind_speed_10m,wind_gusts_10m,wind_direction_10m"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    if past_days > 0:
        params["past_days"] = int(past_days)

    data = _get_json_with_retries(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=12,
        retries=2,
    )
    if not isinstance(data, dict):
        return supp

    h = data.get("hourly") or {}
    times = h.get("time") or []
    for i, t_str in enumerate(times):
        try:
            dt_utc = (
                datetime.fromisoformat(str(t_str))
                .replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
                .isoformat()
            )
            def _v(key: str, default: float) -> float:
                arr = h.get(key)
                if isinstance(arr, list) and i < len(arr) and arr[i] is not None:
                    return float(arr[i])
                return default

            supp[dt_utc] = {
                "temp_c": round(_v("temperature_2m", np.nan), 2),
                "dewpoint_c": round(_v("dew_point_2m", np.nan), 2),
                "rh_pct": round(_v("relative_humidity_2m", 50.0), 1),
                "pop_pct": round(_v("precipitation_probability", 0.0), 1),
                "qpf_mm": round(max(0.0, _v("precipitation", 0.0)), 3),
                "sky_pct": round(_v("cloud_cover", 20.0), 1),
                "wspd_ms": round(max(0.0, _v("wind_speed_10m", 1.5)), 2),
                "wgst_ms": round(max(0.0, _v("wind_gusts_10m", 2.0)), 2),
                "wdir_deg": round(_v("wind_direction_10m", 295.0) % 360.0, 1),
            }
        except Exception:
            continue
    return supp


def fetch_open_meteo_qpf_supplement() -> Dict[str, Tuple[float, float, float]]:
    """Backward-compatible helper returning {valid_iso: (qpf_mm, sky_pct, gust_ms)}."""
    om_map = fetch_open_meteo_nwp_map(past_days=0, forecast_days=8)
    return {
        k: (v["qpf_mm"], v["sky_pct"], v["wgst_ms"])
        for k, v in om_map.items()
    }


def synthesize_qpf_from_pop_and_text(
    pop_pct: float,
    rh_pct: float,
    short_fc: str = "",
) -> float:
    """
    Synthesizes a physically realistic hourly QPF (mm/hr) ONLY when both NWS Gridpoints
    and Open-Meteo report 0.0 mm despite PoP >= 20% or explicit rain text.
    """
    desc = (short_fc or "").lower()
    pop_val = float(pop_pct or 0.0)
    pop_frac = max(0.0, min(1.0, pop_val / 100.0))
    rh_frac = max(0.3, min(1.0, float(rh_pct or 50.0) / 100.0))

    if any(k in desc for k in ("heavy rain", "thunderstorm", "downpour")):
        return round(max(1.80, 3.80 * pop_frac * rh_frac), 3)
    if any(k in desc for k in ("rain", "showers")) and "slight chance" not in desc:
        return round(max(0.35, 1.45 * (pop_frac ** 1.2) * rh_frac), 3)
    if (
        any(k in desc for k in ("drizzle", "light rain", "slight chance", "isolated"))
        or pop_val >= 25.0
    ):
        if pop_val >= 20.0:
            return round(max(0.12, 0.75 * (pop_frac ** 1.35) * rh_frac), 3)
    return 0.0


def fetch_nws_gridpoints_dataframe(
    now_hour_utc: Optional[datetime] = None,
    include_past_hours: int = 0,
    prefer_open_meteo_primary: bool = False,
) -> pd.DataFrame:
    """
    Unified, high-reliability NWP/NWS hourly forecast builder for both the hourly SQLite
    pipeline (`ingest_nws.py`) and live Streamlit top-ups (`app.py`).
    Queries `/gridpoints/MTR/...` for full grid arrays (including quantitativePrecipitation),
    enriches with `/forecastHourly` periods, and cross-checks with Open-Meteo at 880m MSL.
    """
    if now_hour_utc is None:
        now_hour_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    else:
        now_hour_utc = pd.Timestamp(now_hour_utc).tz_convert("UTC").floor("h").to_pydatetime()

    issue_iso = now_hour_utc.isoformat()
    grid_url, hourly_url = resolve_nws_endpoints(TASS_LAT, TASS_LON)

    # 1. Fetch Raw NWS Gridpoint Arrays (/gridpoints/MTR/...)
    grid_qpf_map: Dict[str, float] = {}
    grid_pop_map: Dict[str, float] = {}
    grid_temp_map: Dict[str, float] = {}
    grid_dew_map: Dict[str, float] = {}
    grid_rh_map: Dict[str, float] = {}
    grid_sky_map: Dict[str, float] = {}
    grid_wspd_map: Dict[str, float] = {}
    grid_wgst_map: Dict[str, float] = {}
    grid_wdir_map: Dict[str, float] = {}

    if grid_url:
        g_data = _get_json_with_retries(grid_url, headers=NWS_HEADERS, timeout=14, retries=3)
        if isinstance(g_data, dict):
            g_props = g_data.get("properties") or {}
            grid_qpf_map = expand_nws_grid_series(
                g_props.get("quantitativePrecipitation"), is_accumulation=True
            )
            grid_pop_map = expand_nws_grid_series(
                g_props.get("probabilityOfPrecipitation"), is_accumulation=False
            )
            grid_temp_map = expand_nws_grid_series(
                g_props.get("temperature"), is_accumulation=False
            )
            grid_dew_map = expand_nws_grid_series(
                g_props.get("dewpoint"), is_accumulation=False
            )
            grid_rh_map = expand_nws_grid_series(
                g_props.get("relativeHumidity"), is_accumulation=False
            )
            grid_sky_map = expand_nws_grid_series(
                g_props.get("skyCover"), is_accumulation=False
            )
            grid_wspd_map = expand_nws_grid_series(
                g_props.get("windSpeed"), is_accumulation=False
            )
            grid_wgst_map = expand_nws_grid_series(
                g_props.get("windGust"), is_accumulation=False
            )
            grid_wdir_map = expand_nws_grid_series(
                g_props.get("windDirection"), is_accumulation=False
            )

    # 2. Fetch /forecastHourly periods (for shortForecast text and 1-hr interpolated temp/PoP)
    hourly_periods_map: Dict[str, Dict[str, Any]] = {}
    if hourly_url:
        fc_data = _get_json_with_retries(hourly_url, headers=NWS_HEADERS, timeout=12, retries=2)
        if isinstance(fc_data, dict):
            periods = (fc_data.get("properties") or {}).get("periods") or []
            for p in periods:
                try:
                    v_dt = (
                        datetime.fromisoformat(p["startTime"])
                        .astimezone(timezone.utc)
                        .replace(minute=0, second=0, microsecond=0)
                    )
                    v_iso = v_dt.isoformat()
                    t_val = float(p.get("temperature", 60.0))
                    t_unit = str(p.get("temperatureUnit", "F")).upper()
                    t_c = round((t_val - 32.0) * 5.0 / 9.0, 2) if t_unit == "F" else round(t_val, 2)
                    dew_raw = (p.get("dewpoint") or {}).get("value")
                    rh_raw = (p.get("relativeHumidity") or {}).get("value")
                    pop_raw = (p.get("probabilityOfPrecipitation") or {}).get("value")
                    w_spd_ms = round(parse_wind_speed_mph(p.get("windSpeed", "")), 2)
                    w_dir_str = str(p.get("windDirection", "")).strip().upper()
                    w_dir_deg = COMPASS_TO_DEG.get(w_dir_str)
                    hourly_periods_map[v_iso] = {
                        "temp_c": t_c,
                        "dew_c": round(float(dew_raw), 2) if dew_raw is not None else None,
                        "rh_pct": round(float(rh_raw), 1) if rh_raw is not None else None,
                        "pop_pct": round(float(pop_raw), 1) if pop_raw is not None else None,
                        "wspd_ms": w_spd_ms,
                        "wdir_deg": w_dir_deg,
                        "short_fc": str(p.get("shortForecast") or "NWS Gridpoint MTR"),
                    }
                except Exception:
                    continue

    # 3. Fetch Open-Meteo NWP Supplement (at 880m MSL)
    past_days_om = max(1, int(np.ceil(include_past_hours / 24.0)) + 1) if include_past_hours > 0 else 1
    om_map = fetch_open_meteo_nwp_map(past_days=past_days_om, forecast_days=8)

    if not grid_temp_map and not hourly_periods_map and not om_map:
        return pd.DataFrame()

    # 4. Build Complete Hourly Timeline
    start_dt = now_hour_utc - timedelta(hours=max(0, int(include_past_hours)))
    total_hours = 168 + max(0, int(include_past_hours))
    hr_list = [start_dt + timedelta(hours=i) for i in range(total_hours)]

    rows: List[Dict[str, Any]] = []
    last_temp_c = 15.0
    last_rh_pct = 50.0
    last_wspd_ms = 1.5
    last_wdir_deg = 295.0
    last_pop_pct = 0.0
    last_sky_pct = 20.0

    for v_dt in hr_list:
        v_iso = v_dt.isoformat()
        lead_hrs = max(1, int(round((v_dt - now_hour_utc).total_seconds() / 3600.0)))
        hp = hourly_periods_map.get(v_iso, {})
        om = om_map.get(v_iso, {})

        # Temperature
        if prefer_open_meteo_primary and pd.notna(om.get("temp_c")):
            t_c = float(om["temp_c"])
        elif hp.get("temp_c") is not None:
            t_c = float(hp["temp_c"])
        elif v_iso in grid_temp_map:
            t_c = float(grid_temp_map[v_iso])
        elif pd.notna(om.get("temp_c")):
            t_c = float(om["temp_c"])
        else:
            t_c = last_temp_c
        last_temp_c = t_c

        # Relative Humidity
        if prefer_open_meteo_primary and pd.notna(om.get("rh_pct")):
            rh_pct = float(om["rh_pct"])
        elif hp.get("rh_pct") is not None:
            rh_pct = float(hp["rh_pct"])
        elif v_iso in grid_rh_map:
            rh_pct = float(grid_rh_map[v_iso])
        elif pd.notna(om.get("rh_pct")):
            rh_pct = float(om["rh_pct"])
        else:
            rh_pct = last_rh_pct
        rh_pct = float(np.clip(rh_pct, 1.0, 100.0))
        last_rh_pct = rh_pct

        # Dewpoint
        if prefer_open_meteo_primary and pd.notna(om.get("dewpoint_c")):
            dew_c = float(om["dewpoint_c"])
        elif hp.get("dew_c") is not None:
            dew_c = float(hp["dew_c"])
        elif v_iso in grid_dew_map:
            dew_c = float(grid_dew_map[v_iso])
        elif pd.notna(om.get("dewpoint_c")):
            dew_c = float(om["dewpoint_c"])
        else:
            dew_c = round(t_c - ((100.0 - rh_pct) / 5.0), 2)

        # Wind Speed, Gust & Direction
        if prefer_open_meteo_primary and pd.notna(om.get("wspd_ms")):
            wspd_ms = float(om["wspd_ms"])
        elif v_iso in grid_wspd_map:
            wspd_ms = float(grid_wspd_map[v_iso])
        elif hp.get("wspd_ms") is not None:
            wspd_ms = float(hp["wspd_ms"])
        elif pd.notna(om.get("wspd_ms")):
            wspd_ms = float(om["wspd_ms"])
        else:
            wspd_ms = last_wspd_ms
        last_wspd_ms = wspd_ms

        g_grid = float(grid_wgst_map.get(v_iso, 0.0))
        g_om = float(om.get("wgst_ms", 0.0)) if pd.notna(om.get("wgst_ms")) else 0.0
        if prefer_open_meteo_primary and g_om > 0.0:
            wgst_ms = max(wspd_ms, g_om)
        elif g_grid > 0.0:
            wgst_ms = max(wspd_ms, g_grid)
        elif g_om > 0.0:
            wgst_ms = max(wspd_ms, g_om)
        else:
            wgst_ms = round(wspd_ms * 1.35, 2)

        if prefer_open_meteo_primary and pd.notna(om.get("wdir_deg")):
            wdir_deg = float(om["wdir_deg"])
        elif v_iso in grid_wdir_map:
            wdir_deg = float(grid_wdir_map[v_iso])
        elif hp.get("wdir_deg") is not None:
            wdir_deg = float(hp["wdir_deg"])
        elif pd.notna(om.get("wdir_deg")):
            wdir_deg = float(om["wdir_deg"])
        else:
            wdir_deg = last_wdir_deg
        last_wdir_deg = wdir_deg

        # Probability of Precipitation (PoP %)
        if hp.get("pop_pct") is not None:
            pop_pct = float(hp["pop_pct"])
        elif v_iso in grid_pop_map:
            pop_pct = float(grid_pop_map[v_iso])
        elif pd.notna(om.get("pop_pct")):
            pop_pct = float(om["pop_pct"])
        else:
            pop_pct = last_pop_pct
        if prefer_open_meteo_primary and pd.notna(om.get("pop_pct")):
            pop_pct = max(
                float(om["pop_pct"]),
                pop_pct if v_iso in grid_pop_map or hp.get("pop_pct") is not None else 0.0,
            )
        pop_pct = float(np.clip(pop_pct, 0.0, 100.0))
        last_pop_pct = pop_pct

        # Quantitative Precipitation Forecast (QPF mm/hr)
        short_fc = str(
            hp.get("short_fc")
            or ("Live NWP Grid" if prefer_open_meteo_primary else "NWS Gridpoint MTR")
        )
        nws_grid_qpf = float(grid_qpf_map.get(v_iso, 0.0))
        om_qpf = float(om.get("qpf_mm", 0.0)) if pd.notna(om.get("qpf_mm")) else 0.0
        synth_qpf = synthesize_qpf_from_pop_and_text(pop_pct, rh_pct, short_fc)

        if nws_grid_qpf > 0.0:
            qpf_1hr_mm = round(nws_grid_qpf, 3)
        elif om_qpf > 0.0:
            qpf_1hr_mm = round(om_qpf, 3)
        else:
            qpf_1hr_mm = round(synth_qpf, 3)

        # Sky Cover (%)
        if v_iso in grid_sky_map:
            sky_pct = float(grid_sky_map[v_iso])
        elif pd.notna(om.get("sky_pct")):
            sky_pct = float(om["sky_pct"])
        else:
            sky_pct = 65.0 if pop_pct >= 25.0 else last_sky_pct
        sky_pct = float(np.clip(sky_pct, 0.0, 100.0))
        last_sky_pct = sky_pct

        rows.append({
            "issue_time_utc": issue_iso,
            "valid_time_utc": v_iso,
            "lead_time_hours": lead_hrs,
            "nws_temp_c": round(t_c, 2),
            "raw_nws_temp_c": round(t_c, 2),
            "nws_dewpoint_c": round(dew_c, 2),
            "nws_rh_pct": round(rh_pct, 1),
            "nws_wind_speed_ms": round(wspd_ms, 2),
            "nws_wind_gust_ms": round(wgst_ms, 2),
            "nws_wind_dir_deg": round(wdir_deg % 360.0, 1),
            "nws_pop_pct": round(pop_pct, 1),
            "nws_qpf_1hr_mm": round(qpf_1hr_mm, 3),
            "nws_sky_cover_pct": round(sky_pct, 1),
            "short_forecast": short_fc,
        })

    return pd.DataFrame(rows)


def fetch_and_store_nws_hourly(db_path: str = "tassajara_weather.db") -> int:
    """
    Fetches the latest 7-day hourly forecast from NWS /gridpoints/MTR/... (+ /forecastHourly & Open-Meteo),
    inserts the operational batch into nws_forecasts, and repairs any prior zero-QPF rows in SQLite.
    """
    now_hr_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df_fc = fetch_nws_gridpoints_dataframe(
        now_hour_utc=now_hr_utc,
        include_past_hours=24,
        prefer_open_meteo_primary=False,
    )
    if df_fc.empty:
        logger.warning("NWS /gridpoints/MTR and Open-Meteo forecast fetch returned 0 rows.")
        return 0

    df_future = df_fc[
        pd.to_datetime(df_fc["valid_time_utc"], utc=True) >= pd.Timestamp(now_hr_utc)
    ].copy()
    rows_to_insert = [
        (
            str(r["issue_time_utc"]),
            str(r["valid_time_utc"]),
            int(r["lead_time_hours"]),
            float(r["nws_temp_c"]),
            float(r["nws_dewpoint_c"]) if pd.notna(r["nws_dewpoint_c"]) else None,
            float(r["nws_rh_pct"]),
            float(r["nws_wind_speed_ms"]),
            float(r["nws_wind_gust_ms"]),
            float(r["nws_wind_dir_deg"]),
            float(r["nws_pop_pct"]),
            float(r["nws_qpf_1hr_mm"]),
            float(r["nws_sky_cover_pct"]),
            str(r["short_forecast"]),
        )
        for _, r in df_future.iterrows()
    ]

    qpf_repair_rows = [
        (float(r["nws_qpf_1hr_mm"]), float(r["nws_wind_gust_ms"]), str(r["valid_time_utc"]))
        for _, r in df_fc.iterrows()
        if float(r["nws_qpf_1hr_mm"]) > 0.0
    ]

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

        if qpf_repair_rows:
            conn.executemany(
                """
                UPDATE nws_forecasts
                SET nws_qpf_1hr_mm = ?,
                    nws_wind_gust_ms = MAX(COALESCE(nws_wind_gust_ms, 0.0), ?)
                WHERE valid_time_utc = ?
                  AND COALESCE(nws_qpf_1hr_mm, 0.0) <= 0.0
                """,
                qpf_repair_rows,
            )

        conn.execute(
            """
            UPDATE nws_forecasts
            SET nws_qpf_1hr_mm = ROUND(
                CASE
                    WHEN nws_pop_pct >= 60.0 THEN 1.35 * POWER(nws_pop_pct / 100.0, 1.15) * MAX(0.30, COALESCE(nws_rh_pct, 50.0) / 100.0)
                    ELSE 0.65 * POWER(nws_pop_pct / 100.0, 1.35) * MAX(0.30, COALESCE(nws_rh_pct, 50.0) / 100.0)
                END,
                3
            )
            WHERE COALESCE(nws_qpf_1hr_mm, 0.0) <= 0.0
              AND COALESCE(nws_pop_pct, 0.0) >= 20.0
            """
        )
        conn.commit()

    wet_fc_hrs = sum(1 for r in rows_to_insert if r[10] > 0.0)
    total_qpf_mm = sum(r[10] for r in rows_to_insert)
    logger.info(
        f"Stored {len(rows_to_insert)} NWS /gridpoints/MTR hourly periods "
        f"({wet_fc_hrs} wet hrs, total 7d QPF = {total_qpf_mm:.2f} mm / {total_qpf_mm/25.4:.2f} in)."
    )
    return len(rows_to_insert)


ingest_nws_forecasts = fetch_and_store_nws_hourly
fetch_and_store_nws_forecasts = fetch_and_store_nws_hourly
run_nws_ingest = fetch_and_store_nws_hourly
main = fetch_and_store_nws_hourly


if __name__ == "__main__":
    fetch_and_store_nws_hourly()
