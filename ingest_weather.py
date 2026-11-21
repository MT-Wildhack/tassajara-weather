#!/usr/bin/env python3
"""
ingest_weather.py (v2.0 — Patched for Credential Safety, Retry/Backoff,
Discrete dst=2 Barometric Trend Filtering, True North Wind Preservation,
Synoptic 1-Hr Rain Safety & Reference Source Provenance)

Fetches ground-truth observations from the Tassajara Stone Office Davis Vantage Pro2
via WeatherLink API v2, pulls the 5-station Santa Lucia regional reference network
(Chews Ridge CHWC1, Hastings CAHC1, Arroyo Seco ASRC1, Big Sur PPSC1, Ventana Cone VTCC1)
with per-station fault isolation, and performs spatial anomaly & lapse-rate diagnostics.
"""

import os
import math
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from zoneinfo import ZoneInfo
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TassajaraIngest")


def _get_secret(env_key: str) -> str:
    """Reads a credential from environment variables, falling back to Streamlit secrets if present."""
    val = (os.getenv(env_key) or "").strip()
    if val:
        return val
    try:
        import streamlit as st
        s_val = st.secrets.get(env_key, "")
        return str(s_val).strip() if s_val else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Configuration & Credentials (Loaded strictly from Env Vars / Secrets)
# ---------------------------------------------------------------------------
WEATHERLINK_API_KEY = _get_secret("WL_API_KEY")
WEATHERLINK_API_SECRET = _get_secret("WL_API_SECRET")
WEATHERLINK_BASE_URL = "https://api.weatherlink.com/v2"

# Synoptic Data (MesoWest) public token for RAWS/CNRFC/HADS stations
SYNOPTIC_API_TOKEN = _get_secret("SYNOPTIC_TOKEN")

# Primary Tassajara Station ID (Stone Office Vantage Pro2)
DEFAULT_TASSAJARA_STATION_ID = 36966

# Station Elevations (meters MSL)
ELEV_TASSAJARA_M = 485.0     # Stone Office (1,590 ft)
ELEV_CHEWS_RIDGE_M = 1536.0  # CHWC1 Chews Ridge (5,040 ft)
ELEV_HASTINGS_M = 546.0      # CAHC1 Hastings Natural History Reservation (1,792 ft)
ELEV_ARROYO_SECO_M = 268.0   # ASRC1 Arroyo Seco RAWS (879 ft)
ELEV_BIG_SUR_M = 108.0       # PPSC1 Big Sur / Pfeiffer State Park RAWS (354 ft)
ELEV_VENTANA_CONE_M = 1453.0 # VTCC1 Ventana Cone HADS/CNRFC (4,766 ft)

# Canonical 5-Station Santa Lucia Reference Network Registry
REFERENCE_STATIONS: Dict[str, Dict[str, Any]] = {
    "CHWC1": {
        "name": "Chews Ridge",
        "lat": 36.3060,
        "lon": -121.5650,
        "elev_m": ELEV_CHEWS_RIDGE_M,
        "role": "Interior Crest (NNW)",
        "default_temp_offset_c": -6.83,
    },
    "CAHC1": {
        "name": "Hastings RAWS",
        "lat": 36.3886,
        "lon": -121.5514,
        "elev_m": ELEV_HASTINGS_M,
        "role": "Foothill Valley (N)",
        "default_temp_offset_c": -0.60,
    },
    "ASRC1": {
        "name": "Arroyo Seco RAWS",
        "lat": 36.2355,
        "lon": -121.4799,
        "elev_m": ELEV_ARROYO_SECO_M,
        "role": "Gorge Outlet (6.3km E)",
        "default_temp_offset_c": 1.41,
    },
    "PPSC1": {
        "name": "Big Sur RAWS",
        "lat": 36.2454,
        "lon": -121.7802,
        "elev_m": ELEV_BIG_SUR_M,
        "role": "Coastal Slope (W)",
        "default_temp_offset_c": 2.45,
    },
    "VTCC1": {
        "name": "Ventana Cone",
        "lat": 36.2981,
        "lon": -121.7150,
        "elev_m": ELEV_VENTANA_CONE_M,
        "role": "Ventana Crest (WNW)",
        "default_temp_offset_c": -6.29,
    },
}

# WeatherLink v2 data_structure_types that belong ONLY to indoor console / health sensors
INDOOR_OR_HEALTH_DST = {12, 13, 14, 15, 16, 17, 18, 21, 22, 27}


def first_not_none(*vals):
    """Returns the first argument that is not None (preserving 0 and 0.0)."""
    for v in vals:
        if v is not None:
            return v
    return None


def first_non_negative(*vals) -> Optional[float]:
    """Returns the first numeric argument that is not None and >= 0.0 (filtering out -1 sentinels)."""
    for v in vals:
        if v is not None:
            try:
                fv = float(v)
                if not math.isnan(fv) and fv >= 0.0:
                    return fv
            except (ValueError, TypeError):
                continue
    return None


def f_to_c(temp_f: Optional[float]) -> Optional[float]:
    return round((float(temp_f) - 32.0) * 5.0 / 9.0, 2) if temp_f is not None else None


def mph_to_ms(speed_mph: Optional[float]) -> Optional[float]:
    return round(float(speed_mph) * 0.44704, 2) if speed_mph is not None else None


def in_to_mm(precip_in: Optional[float]) -> Optional[float]:
    return round(float(precip_in) * 25.4, 3) if precip_in is not None else None


def inhg_to_hpa(press_inhg: Optional[float]) -> Optional[float]:
    return round(float(press_inhg) * 33.86389, 2) if press_inhg is not None else None


def compute_vpd_kpa(temp_c: Optional[float], rh_pct: Optional[float]) -> Optional[float]:
    """Computes saturation Vapor Pressure Deficit (kPa) via Tetens equation."""
    if temp_c is None or rh_pct is None:
        return None
    es = 0.61078 * math.exp((17.27 * float(temp_c)) / (float(temp_c) + 237.3))
    ea = es * (max(0.0, min(100.0, float(rh_pct))) / 100.0)
    return round(max(0.0, es - ea), 3)


def extract_archive_rain_mm(rec: Dict[str, Any]) -> float:
    """
    Extracts strictly incremental archive-interval rainfall (mm) from a single
    WeatherLink v2 /historic/ record. Never uses rolling 60-minute fields
    (rainfall_last_60_min_*), which would multiply rain by 4x-12x when summed
    across 5- or 15-minute sub-hourly archive packets.
    """
    rain_in = first_not_none(
        rec.get("rainfall_in"),
        rec.get("rain_in"),
    )
    if rain_in is not None:
        return round(max(0.0, float(rain_in) * 25.4), 3)

    rain_mm = first_not_none(
        rec.get("rainfall_mm"),
        rec.get("rain_mm"),
    )
    if rain_mm is not None:
        return round(max(0.0, float(rain_mm)), 3)

    clicks = first_not_none(
        rec.get("rainfall_clicks"),
        rec.get("rain_clicks"),
        rec.get("rainfall"),
    )
    if clicks is not None:
        return round(max(0.0, float(clicks) * 0.254), 3)

    return 0.0


def extract_record_wind(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extracts (wind_speed_ms, wind_gust_ms, wind_dir_deg) from any WeatherLink v2
    current or archive record across all data_structure_type schemas.
    """
    w_spd_mph = first_non_negative(
        rec.get("wind_speed_avg_last_10_min"),
        rec.get("wind_speed_10_min_avg"),
        rec.get("wind_speed_avg_last_2_min"),
        rec.get("wind_speed_avg_last_1_min"),
        rec.get("wind_speed_avg"),
        rec.get("wind_speed_last"),
        rec.get("wind_speed"),
    )
    w_gst_mph = first_non_negative(
        rec.get("wind_speed_hi_last_10_min"),
        rec.get("wind_gust_10_min"),
        rec.get("wind_speed_hi_last_2_min"),
        rec.get("wind_speed_hi"),
        rec.get("wind_gust_last"),
        rec.get("wind_gust"),
        w_spd_mph,
    )
    w_dir = first_non_negative(
        rec.get("wind_dir_scalar_avg_last_10_min"),
        rec.get("wind_dir_scalar_avg_last_2_min"),
        rec.get("wind_dir_scalar_avg_last_1_min"),
        rec.get("wind_dir_of_prevail"),
        rec.get("wind_dir_scalar_avg"),
        rec.get("wind_dir_last"),
        rec.get("wind_dir"),
        rec.get("wind_dir_at_hi_speed_last_10_min"),
        rec.get("wind_dir_at_hi_speed_last_2_min"),
        rec.get("wind_dir_of_hi"),
        rec.get("wind_speed_hi_dir"),
    )

    spd_ms = mph_to_ms(w_spd_mph) if w_spd_mph is not None else None
    gst_ms = mph_to_ms(w_gst_mph) if w_gst_mph is not None else None
    if spd_ms is not None and gst_ms is not None:
        gst_ms = max(gst_ms, spd_ms)
    dir_deg = float(w_dir) if w_dir is not None else None
    return spd_ms, gst_ms, dir_deg


class WeatherLinkV2Client:
    """Client for Davis WeatherLink v2 API using X-Api-Secret header authentication."""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = (api_key or "").strip() or _get_secret("WL_API_KEY")
        self.api_secret = (api_secret or "").strip() or _get_secret("WL_API_SECRET")
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Secret": self.api_secret,
            "Accept": "application/json",
        })

    def _get_with_retry(self, url: str, params: dict, timeout: int = 15, retries: int = 3):
        """Issues a GET request with exponential backoff on transient 429/5xx errors."""
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "WeatherLink credentials missing: set WL_API_KEY and WL_API_SECRET in environment or Streamlit secrets."
            )
        last_exc = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"WeatherLink GET failed for {url}")

    def get_station_id(self) -> int:
        """Discovers the active station_id associated with the API key, defaulting to 36966."""
        url = f"{WEATHERLINK_BASE_URL}/stations"
        try:
            resp = self._get_with_retry(url, params={"api-key": self.api_key}, timeout=15)
            stations = resp.json().get("stations", [])
            if stations:
                for stn in stations:
                    if int(stn.get("station_id", 0)) == DEFAULT_TASSAJARA_STATION_ID:
                        logger.info(
                            f"Verified Tassajara WeatherLink station: "
                            f"{stn.get('station_name')} (ID: {DEFAULT_TASSAJARA_STATION_ID})"
                        )
                        return DEFAULT_TASSAJARA_STATION_ID
                station_id = int(stations[0]["station_id"])
                logger.info(
                    f"Discovered WeatherLink station: {stations[0].get('station_name')} (ID: {station_id})"
                )
                return station_id
        except Exception as exc:
            logger.warning(
                f"Station discovery query failed ({exc}); using default Station ID {DEFAULT_TASSAJARA_STATION_ID}."
            )
        return DEFAULT_TASSAJARA_STATION_ID

    def fetch_current_observations(self, station_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Pulls current conditions from WeatherLink v2 API (/v2/current/{station_id})
        and normalizes Davis Vantage Pro2 ISS, Console, and Barometer sensor structures.
        Never lets an indoor/barometer sensor overwrite outdoor ISS wind, rain, or timestamp.
        """
        if station_id is None:
            station_id = self.get_station_id()

        url = f"{WEATHERLINK_BASE_URL}/current/{station_id}"
        resp = self._get_with_retry(url, params={"api-key": self.api_key}, timeout=15)
        payload = resp.json()

        parsed: Dict[str, Any] = {
            "station_id": f"TASS_VP2_{station_id}",
            "valid_time_utc": datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat(),
            "temp_c": None,
            "dewpoint_c": None,
            "wetbulb_c": None,
            "rh_pct": None,
            "wind_speed_ms": 0.0,
            "wind_gust_ms": 0.0,
            "wind_dir_deg": None,
            "mslp_hpa": None,
            "pressure_trend_hpa": None,
            "precip_1hr_mm": 0.0,
            "precip_day_mm": 0.0,
            "precip_24hr_mm": 0.0,
            "precip_storm_mm": 0.0,
            "rain_storm_start_utc": None,
            "precip_month_mm": 0.0,
            "precip_year_mm": None,
            "precip_rate_mm_hr": 0.0,
            "solar_rad_wm2": None,
            "uv_index": None,
        }

        max_iss_ts = 0

        for sensor in payload.get("sensors", []):
            dst = int(sensor.get("data_structure_type") or 0)
            data_records = sensor.get("data", [])
            if not data_records:
                continue
            latest = data_records[0]

            # 1. Barometric Pressure (often on Console/Barometer sensor dst 2, 3, 9, 12, 19, 20)
            bar_sl = first_not_none(
                latest.get("bar_sea_level"),
                latest.get("bar"),
                latest.get("pressure_last"),
                latest.get("bar_absolute"),
            )
            if bar_sl is not None:
                parsed["mslp_hpa"] = inhg_to_hpa(bar_sl)
                btrend = first_not_none(latest.get("bar_trend"), latest.get("bar_trend_3_hr"))
                # Legacy Davis dst=2 bar_trend is a discrete 5-state code (-60,-20,0,20,60),
                # not continuous inHg; leave None so run_hourly_mos_pipeline computes true 3h MSLP delta.
                if btrend is not None and dst != 2:
                    bt_val = float(btrend)
                    if abs(bt_val) <= 2.0:
                        parsed["pressure_trend_hpa"] = inhg_to_hpa(bt_val)

            # Skip strictly indoor/health/AirLink AQI sensors for outdoor weather metrics
            stype = int(sensor.get("sensor_type") or 0)
            if (dst in INDOOR_OR_HEALTH_DST or stype in {323, 326, 504, 506}) and "temp_out" not in latest:
                continue
            rec_ts_check = int(latest.get("ts") or 0)
            if rec_ts_check > 0 and max_iss_ts > 0 and (max_iss_ts - rec_ts_check) > 1800 and "temp_out" not in latest:
                continue

            rec_ts = int(latest.get("ts") or 0)
            if rec_ts > max_iss_ts:
                max_iss_ts = rec_ts
                parsed["valid_time_utc"] = datetime.fromtimestamp(
                    rec_ts, tz=timezone.utc
                ).isoformat()

            # 2. Outdoor Temperature & Humidity
            t_f = first_not_none(
                latest.get("temp_out"),
                latest.get("temp"),
                latest.get("temp_last"),
            )
            if t_f is not None:
                parsed["temp_c"] = f_to_c(t_f)
                dew_f = first_not_none(
                    latest.get("dew_point_out"),
                    latest.get("dew_point"),
                    latest.get("dew_point_last"),
                )
                wet_f = first_not_none(
                    latest.get("wet_bulb_out"),
                    latest.get("wet_bulb"),
                    latest.get("wet_bulb_last"),
                )
                hum = first_not_none(
                    latest.get("hum_out"),
                    latest.get("hum"),
                    latest.get("hum_last"),
                )
                if dew_f is not None:
                    parsed["dewpoint_c"] = f_to_c(dew_f)
                if wet_f is not None:
                    parsed["wetbulb_c"] = f_to_c(wet_f)
                if hum is not None:
                    parsed["rh_pct"] = float(hum)

            # 3. Wind Speed, Gust & Direction (preserves True North 0.0 deg vs None)
            spd_ms, gst_ms, dir_deg = extract_record_wind(latest)
            if spd_ms is not None and spd_ms > parsed["wind_speed_ms"]:
                parsed["wind_speed_ms"] = spd_ms
            if gst_ms is not None and gst_ms > parsed["wind_gust_ms"]:
                parsed["wind_gust_ms"] = gst_ms
            parsed["wind_gust_ms"] = max(parsed["wind_gust_ms"], parsed["wind_speed_ms"])
            if dir_deg is not None and 0.0 <= dir_deg <= 360.0:
                parsed["wind_dir_deg"] = dir_deg

            # 4. Rainfall (1-Hour, 24-Hour / Daily, Storm, and Rate)
            rain_1hr_mm = first_not_none(
                latest.get("rainfall_last_60_min_mm"),
                latest.get("rain_60_min_mm"),
                in_to_mm(
                    first_not_none(
                        latest.get("rainfall_last_60_min_in"),
                        latest.get("rain_60_min_in"),
                        latest.get("rainfall_in"),
                        latest.get("rain_in"),
                    )
                ),
            )
            if rain_1hr_mm is None:
                clicks_1h = first_not_none(
                    latest.get("rainfall_last_60_min_clicks"),
                    latest.get("rainfall_last_60_min"),
                    latest.get("rain_60_min"),
                )
                if clicks_1h is not None:
                    rain_1hr_mm = float(clicks_1h) * 0.254
            if rain_1hr_mm is not None:
                parsed["precip_1hr_mm"] = max(parsed["precip_1hr_mm"], round(float(rain_1hr_mm), 3))

            rain_day_mm = first_not_none(
                latest.get("rainfall_daily_mm"),
                latest.get("rain_day_mm"),
                in_to_mm(
                    first_not_none(
                        latest.get("rainfall_daily_in"),
                        latest.get("rain_day_in"),
                    )
                ),
            )
            if rain_day_mm is None:
                clicks_day = first_not_none(
                    latest.get("rainfall_daily_clicks"),
                    latest.get("rainfall_daily"),
                    latest.get("rain_day_clicks"),
                )
                if clicks_day is not None:
                    rain_day_mm = float(clicks_day) * 0.254
            if rain_day_mm is not None:
                parsed["precip_day_mm"] = max(parsed.get("precip_day_mm", 0.0), round(float(rain_day_mm), 3))

            rain_24hr_mm = first_not_none(
                latest.get("rainfall_last_24_hr_mm"),
                latest.get("rain_24_hr_mm"),
                in_to_mm(
                    first_not_none(
                        latest.get("rainfall_last_24_hr_in"),
                        latest.get("rain_24_hr_in"),
                    )
                ),
            )
            if rain_24hr_mm is None:
                clicks_24h = first_not_none(
                    latest.get("rainfall_last_24_hr_clicks"),
                    latest.get("rainfall_last_24_hr"),
                    latest.get("rain_24_hr"),
                )
                if clicks_24h is not None:
                    rain_24hr_mm = float(clicks_24h) * 0.254
            if rain_24hr_mm is None:
                rain_24hr_mm = parsed.get("precip_day_mm", 0.0)
            if rain_24hr_mm is not None:
                parsed["precip_24hr_mm"] = max(
                    parsed["precip_24hr_mm"],
                    parsed.get("precip_day_mm", 0.0),
                    round(float(rain_24hr_mm), 3),
                )

            rain_month_mm = first_not_none(
                latest.get("rainfall_monthly_mm"),
                latest.get("rain_month_mm"),
                in_to_mm(first_not_none(latest.get("rainfall_monthly_in"), latest.get("rain_month_in"))),
            )
            if rain_month_mm is not None:
                parsed["precip_month_mm"] = max(parsed.get("precip_month_mm", 0.0), round(float(rain_month_mm), 3))

            rain_year_mm = first_not_none(
                latest.get("rainfall_year_mm"),
                latest.get("rain_year_mm"),
                in_to_mm(first_not_none(latest.get("rainfall_year_in"), latest.get("rain_year_in"))),
            )
            if rain_year_mm is not None:
                prev_yr = parsed.get("precip_year_mm")
                yr_val = round(float(rain_year_mm), 3)
                parsed["precip_year_mm"] = yr_val if prev_yr is None else max(prev_yr, yr_val)

            rain_storm_mm = first_not_none(
                latest.get("rain_storm_mm"),
                latest.get("rain_storm_last_mm"),
                in_to_mm(
                    first_not_none(
                        latest.get("rain_storm_in"),
                        latest.get("rain_storm_last_in"),
                    )
                ),
            )
            if rain_storm_mm is None:
                clicks_storm = first_not_none(
                    latest.get("rain_storm_clicks"),
                    latest.get("rain_storm"),
                )
                if clicks_storm is not None:
                    rain_storm_mm = float(clicks_storm) * 0.254
            if rain_storm_mm is not None:
                parsed["precip_storm_mm"] = max(parsed["precip_storm_mm"], round(float(rain_storm_mm), 3))

            storm_start_raw = first_non_negative(
                latest.get("rain_storm_start_date"),
                latest.get("rain_storm_start_time"),
                latest.get("rain_storm_last_start_at"),
            )
            if storm_start_raw is not None and storm_start_raw > 946684800:
                parsed["rain_storm_start_utc"] = (
                    datetime.fromtimestamp(int(storm_start_raw), tz=timezone.utc)
                    .replace(minute=0, second=0, microsecond=0)
                    .isoformat()
                )

            rain_rate_mm = first_not_none(
                latest.get("rain_rate_hi_mm"),
                latest.get("rain_rate_last_mm"),
                latest.get("rain_rate_mm"),
                in_to_mm(
                    first_not_none(
                        latest.get("rain_rate_hi_in"),
                        latest.get("rain_rate_last_in"),
                        latest.get("rain_rate_in"),
                    )
                ),
            )
            if rain_rate_mm is not None:
                parsed["precip_rate_mm_hr"] = max(parsed["precip_rate_mm_hr"], round(float(rain_rate_mm), 3))

            sol = first_not_none(latest.get("solar_rad"), latest.get("solar_rad_avg"))
            uv = first_not_none(latest.get("uv_index"), latest.get("uv_index_avg"), latest.get("uv"))
            if sol is not None:
                parsed["solar_rad_wm2"] = float(sol)
            if uv is not None:
                parsed["uv_index"] = float(uv)

        return parsed


def _extract_synoptic_obs(
    obs_dict: Dict[str, Any],
    candidate_keys: List[str],
    default: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Null-safe extractor for Synoptic OBSERVATIONS sub-dictionaries."""
    if not isinstance(obs_dict, dict):
        return default, None
    for key in candidate_keys:
        node = obs_dict.get(key)
        if isinstance(node, dict) and node.get("value") is not None:
            try:
                return float(node["value"]), node.get("date_time")
            except (ValueError, TypeError):
                continue
    return default, None


def _empty_reference_record(stid: str, status: str = "OFFLINE") -> Dict[str, Any]:
    """Returns a schema-compliant skeleton record when a remote station is offline."""
    meta = REFERENCE_STATIONS.get(stid, {})
    return {
        "station_id": stid,
        "name": meta.get("name", stid),
        "elev_m": meta.get("elev_m"),
        "role": meta.get("role", "Reference"),
        "valid_time_utc": None,
        "online": False,
        "is_model_fallback": False,
        "source": "OFFLINE",
        "status": status,
        "temp_c": None,
        "rh_pct": None,
        "wind_speed_ms": None,
        "wind_gust_ms": None,
        "wind_dir_deg": None,
        "precip_1hr_mm": 0.0,
    }


def _resolve_synoptic_token(raw_token: str) -> str:
    """If raw_token is a Synoptic Account Key, exchanges it via /v2/auth for a valid public token."""
    tok = (raw_token or "").strip()
    if not tok:
        return tok
    try:
        r = requests.get("https://api.synopticdata.com/v2/auth", params={"apikey": tok}, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if data.get("TOKEN"):
                return str(data["TOKEN"]).strip()
    except Exception:
        pass
    return tok


def _fetch_nws_station_latest(stid: str) -> Optional[Dict[str, Any]]:
    """Fallback Tier 2: Pulls latest observation from NOAA/NWS API (api.weather.gov) for RAWS/HADS."""
    url = f"https://api.weather.gov/stations/{stid}/observations/latest"
    headers = {
        "User-Agent": "(TassajaraWeatherMOSMonitor/2.0, tzmc-weather@tassajara.org)",
        "Accept": "application/geo+json",
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        props = r.json().get("properties") or {}
        ts = props.get("timestamp")

        def _val(key: str) -> Optional[float]:
            node = props.get(key)
            if isinstance(node, dict) and node.get("value") is not None:
                try:
                    return float(node["value"])
                except (ValueError, TypeError):
                    return None
            return None

        temp_c = _val("temperature")
        rh_pct = _val("relativeHumidity")
        w_spd_kmh = _val("windSpeed")
        w_gst_kmh = _val("windGust")
        w_dir = _val("windDirection")
        p_m = _val("precipitationLastHour")

        w_spd_ms = round(w_spd_kmh / 3.6, 2) if w_spd_kmh is not None else None
        w_gst_ms = round(w_gst_kmh / 3.6, 2) if w_gst_kmh is not None else w_spd_ms
        p_mm = round(max(0.0, p_m * 1000.0), 2) if p_m is not None else 0.0

        if temp_c is None and rh_pct is None:
            return None

        return {
            "valid_time_utc": ts,
            "temp_c": round(temp_c, 2) if temp_c is not None else None,
            "rh_pct": round(rh_pct, 1) if rh_pct is not None else None,
            "wind_speed_ms": w_spd_ms,
            "wind_gust_ms": w_gst_ms,
            "wind_dir_deg": round(w_dir, 1) if w_dir is not None else None,
            "precip_1hr_mm": p_mm,
        }
    except Exception:
        return None



def _fetch_iem_station_latest(stid: str) -> Optional[Dict[str, Any]]:
    """Fallback Tier 2B: Free, no-key Iowa Environmental Mesonet (CA_DCP) live RAWS/HADS feed."""
    url = "https://mesonet.agron.iastate.edu/json/current.py"
    try:
        r = requests.get(
            url,
            params={"network": "CA_DCP", "station": stid},
            headers={"User-Agent": "(TassajaraWeatherMOSMonitor/2.0, tzmc-weather@tassajara.org)"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        ob = (r.json() or {}).get("last_ob") or {}
        if not ob:
            return None

        def _f(key: str) -> Optional[float]:
            v = ob.get(key)
            if v is None:
                return None
            try:
                fv = float(v)
                return None if math.isnan(fv) else fv
            except (ValueError, TypeError):
                return None

        t_f = _f("airtemp[F]")
        td_f = _f("dewpointtemp[F]")
        rh = _f("relh[%]")
        w_kt = _f("windspeed[kt]") if _f("windspeed[kt]") is not None else _f("sknt[kt]")
        g_kt = _f("windgust[kt]") if _f("windgust[kt]") is not None else _f("gust[kt]")
        w_dir = _f("winddirection[deg]") if _f("winddirection[deg]") is not None else _f("drct")
        p_hr_in = _f("phour[in]")
        p_day_in = _f("precip_today[in]")
        ts = ob.get("utc_valid")

        if t_f is None and rh is None and p_day_in is None and p_hr_in is None:
            return None

        t_c = f_to_c(t_f) if t_f is not None else None
        if rh is None and t_c is not None and td_f is not None:
            td_c = f_to_c(td_f)
            if td_c is not None:
                rh = max(
                    0.0,
                    min(
                        100.0,
                        100.0
                        * math.exp((17.625 * td_c) / (243.04 + td_c))
                        / math.exp((17.625 * t_c) / (243.04 + t_c)),
                    ),
                )

        w_spd_ms = round(w_kt * 0.514444, 2) if w_kt is not None else None
        w_gst_ms = round(g_kt * 0.514444, 2) if g_kt is not None else w_spd_ms
        p_hr_mm = round(max(0.0, p_hr_in * 25.4), 2) if p_hr_in is not None else 0.0
        p_day_mm = round(max(0.0, p_day_in * 25.4), 2) if p_day_in is not None else None

        return {
            "valid_time_utc": ts,
            "temp_c": t_c,
            "rh_pct": round(rh, 1) if rh is not None else None,
            "wind_speed_ms": w_spd_ms,
            "wind_gust_ms": w_gst_ms,
            "wind_dir_deg": round(w_dir, 1) if w_dir is not None else None,
            "precip_1hr_mm": p_hr_mm,
            "precip_day_mm": p_day_mm,
        }
    except Exception:
        return None


def _fetch_openmeteo_reference_batch(stids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fallback Tier 3: Pulls real-time HRRR/GFS boundary conditions at exact station lat/lon/elev."""
    out: Dict[str, Dict[str, Any]] = {}
    now_hr_iso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
    for stid in stids:
        meta = REFERENCE_STATIONS.get(stid, {})
        if not meta:
            continue
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": meta["lat"],
                    "longitude": meta["lon"],
                    "elevation": meta["elev_m"],
                    "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "ms",
                    "precipitation_unit": "mm",
                    "timezone": "UTC",
                },
                timeout=10,
            )
            if r.status_code != 200:
                continue
            cur = r.json().get("current") or {}
            t_c = cur.get("temperature_2m")
            if t_c is None:
                continue
            out[stid] = {
                "valid_time_utc": now_hr_iso,
                "temp_c": round(float(t_c), 2),
                "rh_pct": round(float(cur["relative_humidity_2m"]), 1) if cur.get("relative_humidity_2m") is not None else None,
                "wind_speed_ms": round(float(cur.get("wind_speed_10m") or 0.0), 2),
                "wind_gust_ms": round(float(cur.get("wind_gusts_10m") or cur.get("wind_speed_10m") or 0.0), 2),
                "wind_dir_deg": round(float(cur.get("wind_direction_10m")), 1) if cur.get("wind_direction_10m") is not None else None,
                "precip_1hr_mm": round(max(0.0, float(cur.get("precipitation") or 0.0)), 2),
            }
        except Exception:
            continue
    return out


def fetch_regional_reference_stations(
    stids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Pulls latest observations for the 5-station Santa Lucia reference array
    (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) using a 3-tier high-availability cascade:
      1. Synoptic Data (MesoWest) v2 API (strictly 1-hr precip; never water-year precip_accum)
      2. NOAA/NWS Observation API (api.weather.gov/stations/{stid}/observations/latest)
      3. Open-Meteo HRRR/GFS Boundary Analysis (explicitly tagged OPENMETEO_SYNTH / online=False)
    """
    if stids is None:
        stids = list(REFERENCE_STATIONS.keys())

    now_hr_iso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
    references: Dict[str, Dict[str, Any]] = {
        stid: _empty_reference_record(stid, status="MISSING_FROM_PAYLOAD")
        for stid in stids
    }

    # --- Tier 1: Synoptic Data v2 API ---
    syn_tok = _get_secret("SYNOPTIC_TOKEN")
    data = None
    if syn_tok:
        url = "https://api.synopticdata.com/v2/stations/latest"
        params = {
            "stid": ",".join(stids),
            "vars": (
                "air_temp,relative_humidity,wind_speed,wind_gust,"
                "wind_direction,precip_accum_one_hour"
            ),
            "units": "metric",
            "within": 360,
            "token": syn_tok,
        }
        try:
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code in (401, 403):
                resolved_tok = _resolve_synoptic_token(syn_tok)
                if resolved_tok != syn_tok:
                    params["token"] = resolved_tok
                    resp = requests.get(url, params=params, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
        except Exception:
            data = None

    station_list = data.get("STATION") if isinstance(data, dict) else None
    if isinstance(station_list, list):
        for stn in station_list:
            if not isinstance(stn, dict):
                continue
            stid = str(stn.get("STID", "")).strip().upper()
            if not stid or stid not in references:
                continue
            try:
                obs = stn.get("OBSERVATIONS") or {}
                temp_c, temp_ts = _extract_synoptic_obs(obs, ["air_temp_value_1"])
                rh_pct, rh_ts = _extract_synoptic_obs(obs, ["relative_humidity_value_1"])
                w_spd, _ = _extract_synoptic_obs(obs, ["wind_speed_value_1"])
                w_gst, _ = _extract_synoptic_obs(obs, ["wind_gust_value_1"])
                w_dir, _ = _extract_synoptic_obs(obs, ["wind_direction_value_1"])
                # Strictly 1-hour accumulation; NEVER fall back to water-year precip_accum_value_1
                precip_1h, p_ts = _extract_synoptic_obs(
                    obs,
                    ["precip_accum_one_hour_value_1"],
                    default=0.0,
                )
                obs_ts = first_not_none(temp_ts, rh_ts, p_ts) or now_hr_iso
                is_online = (temp_c is not None) or (rh_pct is not None)
                meta = REFERENCE_STATIONS.get(stid, {})
                if is_online:
                    references[stid] = {
                        "station_id": stid,
                        "name": meta.get("name", stn.get("NAME", stid)),
                        "elev_m": meta.get("elev_m"),
                        "role": meta.get("role", "Reference"),
                        "valid_time_utc": obs_ts,
                        "online": True,
                        "is_model_fallback": False,
                        "source": "RAWS_SYNOPTIC",
                        "status": "OK",
                        "temp_c": round(temp_c, 2) if temp_c is not None else None,
                        "rh_pct": round(rh_pct, 1) if rh_pct is not None else None,
                        "wind_speed_ms": round(w_spd, 2) if w_spd is not None else None,
                        "wind_gust_ms": round(w_gst, 2) if w_gst is not None else None,
                        "wind_dir_deg": round(w_dir, 1) if w_dir is not None else None,
                        "precip_1hr_mm": round(max(0.0, precip_1h or 0.0), 2),
                    }
            except Exception:
                continue

    # --- Tier 2: NOAA/NWS Observations API for any offline/missing stations ---
    missing_stids = [s for s in stids if not references[s].get("online") or references[s].get("temp_c") is None]
    for stid in missing_stids:
        nws_rec = _fetch_nws_station_latest(stid)
        if nws_rec and nws_rec.get("temp_c") is not None:
            meta = REFERENCE_STATIONS.get(stid, {})
            references[stid] = {
                "station_id": stid,
                "name": meta.get("name", stid),
                "elev_m": meta.get("elev_m"),
                "role": meta.get("role", "Reference"),
                "valid_time_utc": nws_rec.get("valid_time_utc") or now_hr_iso,
                "online": True,
                "is_model_fallback": False,
                "source": "NWS_OBS",
                "status": "OK_NWS",
                "temp_c": nws_rec["temp_c"],
                "rh_pct": nws_rec["rh_pct"],
                "wind_speed_ms": nws_rec["wind_speed_ms"],
                "wind_gust_ms": nws_rec["wind_gust_ms"],
                "wind_dir_deg": nws_rec["wind_dir_deg"],
                "precip_1hr_mm": nws_rec["precip_1hr_mm"],
            }

    # --- Tier 2B: Iowa Environmental Mesonet (CA_DCP) free live RAWS/HADS feed ---
    iem_missing = [s for s in stids if not references[s].get("online") or references[s].get("temp_c") is None]
    for stid in iem_missing:
        iem_rec = _fetch_iem_station_latest(stid)
        if iem_rec and (iem_rec.get("temp_c") is not None or iem_rec.get("precip_day_mm") is not None):
            meta = REFERENCE_STATIONS.get(stid, {})
            is_precip_only = iem_rec.get("temp_c") is None
            references[stid] = {
                "station_id": stid,
                "name": meta.get("name", stid),
                "elev_m": meta.get("elev_m"),
                "role": meta.get("role", "Reference"),
                "valid_time_utc": iem_rec.get("valid_time_utc") or now_hr_iso,
                "online": True,
                "is_model_fallback": False,
                "source": "IEM_CA_DCP",
                "status": "OK_IEM_PRECIP" if is_precip_only else "OK_IEM",
                "temp_c": iem_rec["temp_c"],
                "rh_pct": iem_rec["rh_pct"],
                "wind_speed_ms": iem_rec["wind_speed_ms"],
                "wind_gust_ms": iem_rec["wind_gust_ms"],
                "wind_dir_deg": iem_rec["wind_dir_deg"],
                "precip_1hr_mm": iem_rec["precip_1hr_mm"],
                "precip_day_mm": iem_rec["precip_day_mm"],
            }

    # --- Tier 3: Open-Meteo Real-Time Boundary Analysis for any remaining gaps ---
    still_missing = [s for s in stids if not references[s].get("online")]
    if still_missing:
        om_batch = _fetch_openmeteo_reference_batch(still_missing)
        for stid, om_rec in om_batch.items():
            meta = REFERENCE_STATIONS.get(stid, {})
            references[stid] = {
                "station_id": stid,
                "name": meta.get("name", stid),
                "elev_m": meta.get("elev_m"),
                "role": meta.get("role", "Reference"),
                "valid_time_utc": om_rec["valid_time_utc"],
                "online": False,
                "is_model_fallback": True,
                "source": "OPENMETEO_SYNTH",
                "status": "FALLBACK_OPENMETEO",
                "temp_c": om_rec["temp_c"],
                "rh_pct": om_rec["rh_pct"],
                "wind_speed_ms": om_rec["wind_speed_ms"],
                "wind_gust_ms": om_rec["wind_gust_ms"],
                "wind_dir_deg": om_rec["wind_dir_deg"],
                "precip_1hr_mm": om_rec["precip_1hr_mm"],
            }

    # Backfill missing RH/wind on CHWC1/VTCC1 if temperature came from HADS/NWS without RH
    for stid in stids:
        if references[stid].get("online") and references[stid].get("rh_pct") is None:
            om_fill = _fetch_openmeteo_reference_batch([stid]).get(stid)
            if om_fill and om_fill.get("rh_pct") is not None:
                references[stid]["rh_pct"] = om_fill["rh_pct"]
            if references[stid].get("wind_speed_ms") is None and om_fill:
                references[stid]["wind_speed_ms"] = om_fill.get("wind_speed_ms")
                references[stid]["wind_gust_ms"] = om_fill.get("wind_gust_ms")
                references[stid]["wind_dir_deg"] = om_fill.get("wind_dir_deg")

    return references


def compute_microclimate_diagnostics(
    tass_obs: Dict[str, Any],
    ref_obs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cross-references Tassajara actuals against the 5-station Santa Lucia network
    (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) to compute boundary-layer stability,
    katabatic gorge drainage, coastal marine-layer gradients, and hardware QC flags.
    """
    diagnostics: Dict[str, Any] = {
        "lapse_rate_chews_c_km": None,
        "lapse_rate_ventana_c_km": None,
        "canyon_confinement_diff_c": None,
        "gorge_drainage_diff_c": None,
        "coastal_inland_grad_c": None,
        "marine_layer_rh_diff_pct": None,
        "vpd_kpa": compute_vpd_kpa(tass_obs.get("temp_c"), tass_obs.get("rh_pct")),
        "inversion_active": False,
        "virga_regime_active": False,
        "online_stations": [k for k, v in ref_obs.items() if v.get("online")],
        "qc_flag": "OK",
        "qc_notes": [],
    }

    t_tass = tass_obs.get("temp_c")
    rh_tass = tass_obs.get("rh_pct")
    chews = ref_obs.get("CHWC1", {})
    hastings = ref_obs.get("CAHC1", {})
    arroyo = ref_obs.get("ASRC1", {})
    bigsur = ref_obs.get("PPSC1", {})
    ventana = ref_obs.get("VTCC1", {})

    if t_tass is not None and chews.get("temp_c") is not None:
        dz_km = (ELEV_CHEWS_RIDGE_M - ELEV_TASSAJARA_M) / 1000.0
        lapse_rate = -1.0 * (chews["temp_c"] - t_tass) / dz_km
        diagnostics["lapse_rate_chews_c_km"] = round(lapse_rate, 2)
        diagnostics["inversion_active"] = bool(lapse_rate < 0.0)

    if t_tass is not None and ventana.get("temp_c") is not None:
        dz_v_km = (ELEV_VENTANA_CONE_M - ELEV_TASSAJARA_M) / 1000.0
        lapse_v = -1.0 * (ventana["temp_c"] - t_tass) / dz_v_km
        diagnostics["lapse_rate_ventana_c_km"] = round(lapse_v, 2)
        if diagnostics["lapse_rate_chews_c_km"] is None:
            diagnostics["inversion_active"] = bool(lapse_v < 0.0)

    active_lapse = first_not_none(
        diagnostics["lapse_rate_chews_c_km"],
        diagnostics["lapse_rate_ventana_c_km"],
    )
    if active_lapse is not None and active_lapse > 12.5 and (tass_obs.get("wind_speed_ms") or 0.0) < 1.5:
        diagnostics["qc_flag"] = "WARN_RADIATION_SHIELD_BIAS"
        diagnostics["qc_notes"].append(
            f"Super-adiabatic lapse rate ({active_lapse:.1f} C/km) under low wind; "
            "Stone Office roof thermal plume or radiation shield fouling suspected."
        )

    if t_tass is not None and hastings.get("temp_c") is not None:
        t_hastings_adj = hastings["temp_c"] + 0.60
        diagnostics["canyon_confinement_diff_c"] = round(t_tass - t_hastings_adj, 2)

    if t_tass is not None and arroyo.get("temp_c") is not None:
        t_arroyo_adj = arroyo["temp_c"] - 1.41
        diagnostics["gorge_drainage_diff_c"] = round(t_tass - t_arroyo_adj, 2)

    if bigsur.get("temp_c") is not None and arroyo.get("temp_c") is not None:
        diagnostics["coastal_inland_grad_c"] = round((bigsur["temp_c"] - 1.04) - arroyo["temp_c"], 2)
    if bigsur.get("rh_pct") is not None and chews.get("rh_pct") is not None:
        diagnostics["marine_layer_rh_diff_pct"] = round(bigsur["rh_pct"] - chews["rh_pct"], 1)

    # Compare 1-hour Stone Office rain (or active rate) against 1-hour crest rain
    tass_p = max(
        float(tass_obs.get("precip_1hr_mm") or 0.0),
        float(tass_obs.get("precip_rate_mm_hr") or 0.0),
    )
    crest_max_p = max(
        float(chews.get("precip_1hr_mm") or 0.0),
        float(ventana.get("precip_1hr_mm") or 0.0),
        float(bigsur.get("precip_1hr_mm") or 0.0),
    )
    wet_stations = [
        stid for stid, d in ref_obs.items()
        if d.get("online") and (d.get("precip_1hr_mm") or 0.0) > 2.0
    ]

    if crest_max_p > 1.0 and tass_p == 0.0 and (rh_tass or 100.0) < 75.0:
        diagnostics["virga_regime_active"] = True
        diagnostics["qc_notes"].append(
            f"Sub-cloud virga active: Santa Lucia crest/coast reporting {crest_max_p:.1f} mm/hr "
            f"while canyon boundary layer remains dry (RH {rh_tass:.0f}%)."
        )
    elif len(wet_stations) >= 2 and tass_p == 0.0 and (rh_tass or 0.0) >= 88.0:
        diagnostics["qc_flag"] = "ERR_RAIN_GAUGE_CLOG"
        diagnostics["qc_notes"].append(
            f"Zero precipitation recorded at Stone Office under saturated conditions (RH >= 88%) "
            f"while {len(wet_stations)} reference stations ({', '.join(wet_stations)}) report >2mm/hr. "
            "Inspect rain cone for sycamore leaf blockage."
        )

    return diagnostics


def get_local_day_boundaries_utc(ref_utc: Optional[datetime] = None) -> Tuple[str, str, str]:
    """
    Returns (yesterday_midnight_utc_iso, today_midnight_utc_iso, now_hour_utc_iso)
    anchored to 00:00:00 America/Los_Angeles (the exact time the Davis Vantage Pro2
    console resets rain_day_in to 0.00).
    """
    tz_local = ZoneInfo("America/Los_Angeles")
    if ref_utc is None:
        ref_utc = datetime.now(timezone.utc)
    now_hr_utc = ref_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    local_dt = now_hr_utc.astimezone(tz_local)
    today_midnight_local = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight_local = today_midnight_local - timedelta(days=1)
    return (
        yesterday_midnight_local.astimezone(timezone.utc).isoformat(),
        today_midnight_local.astimezone(timezone.utc).isoformat(),
        now_hr_utc.isoformat(),
    )