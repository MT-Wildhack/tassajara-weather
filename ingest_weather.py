#!/usr/bin/env python3
"""
ingest_weather.py
Fetches ground-truth observations from the Tassajara Stone Office Davis Vantage Pro2
via WeatherLink API v2, pulls the 5-station Santa Lucia regional reference network
(Chews Ridge CHWC1, Hastings CAHC1, Arroyo Seco ASRC1, Big Sur PPSC1, Ventana Cone VTCC1)
with per-station fault isolation, and performs spatial anomaly & lapse-rate diagnostics.
"""

import os
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TassajaraIngest")

# ---------------------------------------------------------------------------
# Configuration & Credentials (Immune to empty-string "" env vars in CI)
# ---------------------------------------------------------------------------
WEATHERLINK_API_KEY = (os.getenv("WL_API_KEY") or "").strip() or "tigkhjhsweahdrae2cl0uenfuws3kv8k"
WEATHERLINK_API_SECRET = (os.getenv("WL_API_SECRET") or "").strip() or "ffnphp69pdbsqwsahkdewvctewmxagwt"
WEATHERLINK_BASE_URL = "https://api.weatherlink.com/v2"

# Synoptic Data (MesoWest) public token for RAWS/CNRFC/HADS stations
SYNOPTIC_API_TOKEN = (os.getenv("SYNOPTIC_TOKEN") or "").strip() or "ee99ea4962204176a8fccdd037b046d7"

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
        "default_temp_offset_c": 2.0,
    },
    "CAHC1": {
        "name": "Hastings RAWS",
        "lat": 36.3886,
        "lon": -121.5514,
        "elev_m": ELEV_HASTINGS_M,
        "role": "Foothill Valley (N)",
        "default_temp_offset_c": 0.5,
    },
    "ASRC1": {
        "name": "Arroyo Seco RAWS",
        "lat": 36.2355,
        "lon": -121.4799,
        "elev_m": ELEV_ARROYO_SECO_M,
        "role": "Gorge Outlet (6.3km E)",
        "default_temp_offset_c": 1.4,
    },
    "PPSC1": {
        "name": "Big Sur RAWS",
        "lat": 36.2454,
        "lon": -121.7802,
        "elev_m": ELEV_BIG_SUR_M,
        "role": "Coastal Slope (W)",
        "default_temp_offset_c": 0.8,
    },
    "VTCC1": {
        "name": "Ventana Cone",
        "lat": 36.2981,
        "lon": -121.7150,
        "elev_m": ELEV_VENTANA_CONE_M,
        "role": "Ventana Crest (WNW)",
        "default_temp_offset_c": 1.5,
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
    Extracts incremental rainfall (mm) from a single WeatherLink v2 archive record
    across all Davis ISS / WeatherLink Console / WeatherLink Live schemas.
    """
    rain_mm = first_not_none(
        rec.get("rainfall_mm"),
        rec.get("rain_mm"),
        rec.get("rainfall_last_60_min_mm"),
    )
    if rain_mm is not None and float(rain_mm) > 0.0:
        return round(float(rain_mm), 3)

    rain_in = first_not_none(
        rec.get("rainfall_in"),
        rec.get("rain_in"),
        rec.get("rainfall_last_60_min_in"),
    )
    if rain_in is not None and float(rain_in) > 0.0:
        return round(float(rain_in) * 25.4, 3)

    clicks = first_not_none(
        rec.get("rainfall_clicks"),
        rec.get("rain_clicks"),
        rec.get("rainfall"),
    )
    if clicks is not None and float(clicks) > 0.0:
        return round(float(clicks) * 0.254, 3)

    return 0.0


def extract_record_wind(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extracts (wind_speed_ms, wind_gust_ms, wind_dir_deg) from any WeatherLink v2
    current or archive record across all data_structure_type schemas.
    """
    w_spd_mph = first_not_none(
        rec.get("wind_speed_avg_last_10_min"),
        rec.get("wind_speed_10_min_avg"),
        rec.get("wind_speed_avg_last_2_min"),
        rec.get("wind_speed_avg_last_1_min"),
        rec.get("wind_speed_avg"),
        rec.get("wind_speed_last"),
        rec.get("wind_speed"),
    )
    w_gst_mph = first_not_none(
        rec.get("wind_speed_hi_last_10_min"),
        rec.get("wind_gust_10_min"),
        rec.get("wind_speed_hi_last_2_min"),
        rec.get("wind_speed_hi"),
        rec.get("wind_gust_last"),
        rec.get("wind_gust"),
        w_spd_mph,
    )
    w_dir = first_not_none(
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
    dir_deg = float(w_dir) if w_dir is not None else None
    return spd_ms, gst_ms, dir_deg


class WeatherLinkV2Client:
    """Client for Davis WeatherLink v2 API using X-Api-Secret header authentication."""

    def __init__(self, api_key: str = WEATHERLINK_API_KEY, api_secret: str = WEATHERLINK_API_SECRET):
        self.api_key = (api_key or "").strip() or WEATHERLINK_API_KEY
        self.api_secret = (api_secret or "").strip() or WEATHERLINK_API_SECRET
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Secret": self.api_secret,
            "Accept": "application/json",
        })

    def get_station_id(self) -> int:
        """Discovers the active station_id associated with the API key, defaulting to 36966."""
        url = f"{WEATHERLINK_BASE_URL}/stations"
        try:
            resp = self.session.get(url, params={"api-key": self.api_key}, timeout=15)
            resp.raise_for_status()
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
        resp = self.session.get(url, params={"api-key": self.api_key}, timeout=15)
        resp.raise_for_status()
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
            "wind_dir_deg": 0.0,
            "mslp_hpa": None,
            "pressure_trend_hpa": None,
            "precip_1hr_mm": 0.0,
            "precip_24hr_mm": 0.0,
            "precip_storm_mm": 0.0,
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

            # 1. Barometric Pressure (often on Console/Barometer sensor dst 3, 9, 12, 19, 20)
            bar_sl = first_not_none(
                latest.get("bar_sea_level"),
                latest.get("bar"),
                latest.get("pressure_last"),
                latest.get("bar_absolute"),
            )
            if bar_sl is not None:
                parsed["mslp_hpa"] = inhg_to_hpa(bar_sl)
                btrend = first_not_none(latest.get("bar_trend"), latest.get("bar_trend_3_hr"))
                if btrend is not None:
                    bt_val = float(btrend)
                    if abs(bt_val) > 2.0:
                        bt_val = bt_val / 1000.0
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

            # 3. Wind Speed, Gust & Direction
            spd_ms, gst_ms, dir_deg = extract_record_wind(latest)
            if spd_ms is not None and spd_ms > parsed["wind_speed_ms"]:
                parsed["wind_speed_ms"] = spd_ms
            if gst_ms is not None and gst_ms > parsed["wind_gust_ms"]:
                parsed["wind_gust_ms"] = max(gst_ms, parsed["wind_speed_ms"])
            if dir_deg is not None and dir_deg > 0.0:
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

            rain_24hr_mm = first_not_none(
                latest.get("rainfall_last_24_hr_mm"),
                latest.get("rain_24_hr_mm"),
                latest.get("rainfall_daily_mm"),
                latest.get("rain_day_mm"),
                in_to_mm(
                    first_not_none(
                        latest.get("rainfall_last_24_hr_in"),
                        latest.get("rain_24_hr_in"),
                        latest.get("rainfall_daily_in"),
                        latest.get("rain_day_in"),
                    )
                ),
            )
            if rain_24hr_mm is None:
                clicks_24h = first_not_none(
                    latest.get("rainfall_last_24_hr_clicks"),
                    latest.get("rainfall_last_24_hr"),
                    latest.get("rain_24_hr"),
                    latest.get("rainfall_daily_clicks"),
                    latest.get("rainfall_daily"),
                    latest.get("rain_day_clicks"),
                )
                if clicks_24h is not None:
                    rain_24hr_mm = float(clicks_24h) * 0.254
            if rain_24hr_mm is not None:
                parsed["precip_24hr_mm"] = max(parsed["precip_24hr_mm"], round(float(rain_24hr_mm), 3))

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
        "status": status,
        "temp_c": None,
        "rh_pct": None,
        "wind_speed_ms": None,
        "wind_gust_ms": None,
        "wind_dir_deg": None,
        "precip_1hr_mm": 0.0,
    }


def fetch_regional_reference_stations(
    stids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Pulls latest observations for the 5-station Santa Lucia reference array
    (CHWC1, CAHC1, ASRC1, PPSC1, VTCC1) via the Synoptic Data (MesoWest) v2 API.
    """
    if stids is None:
        stids = list(REFERENCE_STATIONS.keys())

    references: Dict[str, Dict[str, Any]] = {
        stid: _empty_reference_record(stid, status="MISSING_FROM_PAYLOAD")
        for stid in stids
    }

    url = "https://api.synopticdata.com/v2/stations/latest"
    params = {
        "stid": ",".join(stids),
        "vars": (
            "air_temp,relative_humidity,wind_speed,wind_gust,"
            "wind_direction,precip_accum_one_hour,precip_accum"
        ),
        "units": "metric",
        "within": 360,
        "token": SYNOPTIC_API_TOKEN,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"Reference station batch HTTP fetch failed ({exc}); returning offline skeletons.")
        return references

    station_list = data.get("STATION")
    if not isinstance(station_list, list):
        logger.warning("Synoptic payload missing 'STATION' array; returning offline skeletons.")
        return references

    for stn in station_list:
        if not isinstance(stn, dict):
            continue
        stid = str(stn.get("STID", "")).strip().upper()
        if not stid:
            continue

        try:
            obs = stn.get("OBSERVATIONS") or {}
            temp_c, temp_ts = _extract_synoptic_obs(obs, ["air_temp_value_1"])
            rh_pct, rh_ts = _extract_synoptic_obs(obs, ["relative_humidity_value_1"])
            w_spd, _ = _extract_synoptic_obs(obs, ["wind_speed_value_1"])
            w_gst, _ = _extract_synoptic_obs(obs, ["wind_gust_value_1"])
            w_dir, _ = _extract_synoptic_obs(obs, ["wind_direction_value_1"])
            precip_1h, p_ts = _extract_synoptic_obs(
                obs,
                [
                    "precip_accum_one_hour_value_1",
                    "precip_intervals_value_1",
                ],
                default=0.0,
            )

            obs_ts = first_not_none(temp_ts, rh_ts, p_ts)
            is_online = (temp_c is not None) or (rh_pct is not None) or (p_ts is not None)
            meta = REFERENCE_STATIONS.get(stid, {})

            references[stid] = {
                "station_id": stid,
                "name": meta.get("name", stn.get("NAME", stid)),
                "elev_m": meta.get(
                    "elev_m",
                    round(float(stn["ELEVATION"]) * 0.3048, 1) if stn.get("ELEVATION") else None,
                ),
                "role": meta.get("role", "Reference"),
                "valid_time_utc": obs_ts,
                "online": is_online,
                "status": "OK" if is_online else "NO_RECENT_SENSORS",
                "temp_c": round(temp_c, 2) if temp_c is not None else None,
                "rh_pct": round(rh_pct, 1) if rh_pct is not None else None,
                "wind_speed_ms": round(w_spd, 2) if w_spd is not None else None,
                "wind_gust_ms": round(w_gst, 2) if w_gst is not None else None,
                "wind_dir_deg": round(w_dir, 1) if w_dir is not None else None,
                "precip_1hr_mm": round(max(0.0, precip_1h or 0.0), 2),
            }
        except Exception as stn_exc:
            logger.error(f"Isolated parsing error for reference station {stid}: {stn_exc}")
            references[stid] = _empty_reference_record(stid, status="PARSE_ERROR")

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

    tass_p = max(tass_obs.get("precip_1hr_mm") or 0.0, tass_obs.get("precip_24hr_mm") or 0.0)
    crest_max_p = max(
        chews.get("precip_1hr_mm") or 0.0,
        ventana.get("precip_1hr_mm") or 0.0,
        bigsur.get("precip_1hr_mm") or 0.0,
    )
    wet_stations = [
        stid for stid, d in ref_obs.items()
        if (d.get("precip_1hr_mm") or 0.0) > 2.0
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