#!/usr/bin/env python3
"""
ingest_weather.py
Fetches ground-truth observations from the Tassajara Stone Office Davis Vantage Pro2
via WeatherLink API v2, pulls regional reference stations (Chews Ridge CHWC1,
Hastings RAWS CAHC1), and performs spatial anomaly detection & lapse-rate calculation.
"""

import os
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TassajaraIngest")

# ---------------------------------------------------------------------------
# Configuration & Credentials (Immune to empty-string "" env vars in CI)
# ---------------------------------------------------------------------------
WEATHERLINK_API_KEY = (os.getenv("WL_API_KEY") or "").strip() or "tigkhjhsweahdrae2cl0uenfuws3kv8k"
WEATHERLINK_API_SECRET = (os.getenv("WL_API_SECRET") or "").strip() or "ffnphp69pdbsqwsahkdewvctewmxagwt"
WEATHERLINK_BASE_URL = "https://api.weatherlink.com/v2"

# Synoptic Data (MesoWest) public token for RAWS/CNRFC stations (CHWC1, CAHC1)
SYNOPTIC_API_TOKEN = (os.getenv("SYNOPTIC_TOKEN") or "").strip() or "959822b9da094eabb0e41b8a5a7dcaa5"

# Primary Tassajara Station ID (Stone Office Vantage Pro2)
DEFAULT_TASSAJARA_STATION_ID = 36966

# Station Elevations (meters MSL)
ELEV_TASSAJARA_M = 485.0    # Stone Office (1,590 ft)
ELEV_CHEWS_RIDGE_M = 1536.0 # CHWC1 Chews Ridge (5,040 ft)
ELEV_HASTINGS_M = 546.0     # CAHC1 Hastings Natural History Reservation (1,792 ft)


def first_not_none(*vals):
    """Returns the first argument that is not None (preserving 0 and 0.0)."""
    for v in vals:
        if v is not None:
            return v
    return None


def f_to_c(temp_f: Optional[float]) -> Optional[float]:
    return round((temp_f - 32.0) * 5.0 / 9.0, 2) if temp_f is not None else None


def mph_to_ms(speed_mph: Optional[float]) -> Optional[float]:
    return round(speed_mph * 0.44704, 2) if speed_mph is not None else None


def in_to_mm(precip_in: Optional[float]) -> Optional[float]:
    return round(precip_in * 25.4, 2) if precip_in is not None else None


def inhg_to_hpa(press_inhg: Optional[float]) -> Optional[float]:
    return round(press_inhg * 33.86389, 2) if press_inhg is not None else None


def compute_vpd_kpa(temp_c: Optional[float], rh_pct: Optional[float]) -> Optional[float]:
    """Computes saturation Vapor Pressure Deficit (kPa) via Tetens equation."""
    if temp_c is None or rh_pct is None:
        return None
    es = 0.61078 * math.exp((17.27 * float(temp_c)) / (float(temp_c) + 237.3))
    ea = es * (max(0.0, min(100.0, float(rh_pct))) / 100.0)
    return round(max(0.0, es - ea), 3)


class WeatherLinkV2Client:
    """Client for Davis WeatherLink v2 API using X-Api-Secret header authentication."""

    def __init__(self, api_key: str = WEATHERLINK_API_KEY, api_secret: str = WEATHERLINK_API_SECRET):
        self.api_key = (api_key or "").strip() or WEATHERLINK_API_KEY
        self.api_secret = (api_secret or "").strip() or WEATHERLINK_API_SECRET
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Secret": self.api_secret,
            "Accept": "application/json"
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
        Pulls current conditions from WeatherLink v2 API and normalizes
        Davis Vantage Pro2 ISS, Barometer, and Internal sensor structures.
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

        for sensor in payload.get("sensors", []):
            data_records = sensor.get("data", [])
            if not data_records:
                continue
            latest = data_records[0]

            if "ts" in latest:
                parsed["valid_time_utc"] = datetime.fromtimestamp(
                    latest["ts"], tz=timezone.utc
                ).isoformat()

            bar_sl = first_not_none(
                latest.get("bar_sea_level"),
                latest.get("bar_absolute"),
                latest.get("bar"),
            )
            if bar_sl is not None:
                parsed["mslp_hpa"] = inhg_to_hpa(bar_sl)
                if latest.get("bar_trend") is not None:
                    parsed["pressure_trend_hpa"] = inhg_to_hpa(latest.get("bar_trend"))

            if "temp" in latest and latest.get("temp") is not None:
                parsed["temp_c"] = f_to_c(latest.get("temp"))
                parsed["dewpoint_c"] = f_to_c(latest.get("dew_point"))
                parsed["wetbulb_c"] = f_to_c(latest.get("wet_bulb"))
                parsed["rh_pct"] = latest.get("hum")

                w_spd = first_not_none(
                    latest.get("wind_speed_avg_last_10_min"),
                    latest.get("wind_speed_last"),
                    0.0,
                )
                w_gst = first_not_none(
                    latest.get("wind_speed_hi_last_10_min"),
                    latest.get("wind_gust_10_min"),
                    w_spd,
                    0.0,
                )
                w_dir = first_not_none(
                    latest.get("wind_dir_scalar_avg_last_10_min"),
                    latest.get("wind_dir_last"),
                    0.0,
                )

                parsed["wind_speed_ms"] = mph_to_ms(w_spd) or 0.0
                parsed["wind_gust_ms"] = mph_to_ms(w_gst) or 0.0
                parsed["wind_dir_deg"] = float(w_dir)

                # 1-Hour Rainfall
                rain_1hr_in = latest.get("rainfall_last_60_min_in")
                if rain_1hr_in is None and latest.get("rainfall_last_60_min") is not None:
                    rain_1hr_in = latest["rainfall_last_60_min"] * 0.01
                parsed["precip_1hr_mm"] = in_to_mm(rain_1hr_in or 0.0) or 0.0

                # 24-Hour & Storm Rainfall
                rain_24hr_in = first_not_none(
                    latest.get("rainfall_last_24_hr_in"),
                    (latest.get("rainfall_last_24_hr") * 0.01) if latest.get("rainfall_last_24_hr") is not None else None,
                    0.0,
                )
                parsed["precip_24hr_mm"] = in_to_mm(rain_24hr_in) or 0.0

                rain_storm_in = first_not_none(
                    latest.get("rain_storm_in"),
                    latest.get("rain_storm_last_in"),
                    0.0,
                )
                parsed["precip_storm_mm"] = in_to_mm(rain_storm_in) or 0.0

                rain_rate_in = first_not_none(
                    latest.get("rain_rate_hi_in"),
                    latest.get("rain_rate_last_in"),
                    0.0,
                )
                parsed["precip_rate_mm_hr"] = in_to_mm(rain_rate_in) or 0.0

                parsed["solar_rad_wm2"] = latest.get("solar_rad")
                parsed["uv_index"] = latest.get("uv_index")

        return parsed


def fetch_regional_reference_stations(stids: List[str] = ["CHWC1", "CAHC1"]) -> Dict[str, Dict[str, Any]]:
    """
    Pulls latest observations for Chews Ridge (CHWC1) and Hastings RAWS (CAHC1)
    via the Synoptic Data (MesoWest) v2 API.
    """
    url = "https://api.synopticdata.com/v2/stations/latest"
    params = {
        "stid": ",".join(stids),
        "vars": "air_temp,relative_humidity,wind_speed,wind_gust,wind_direction,precip_accum_one_hour",
        "units": "metric",
        "within": 120,
        "token": SYNOPTIC_API_TOKEN
    }
    references: Dict[str, Dict[str, Any]] = {}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for stn in data.get("STATION", []):
            stid = stn["STID"]
            obs = stn.get("OBSERVATIONS", {})
            references[stid] = {
                "station_id": stid,
                "temp_c": obs.get("air_temp_value_1", {}).get("value"),
                "rh_pct": obs.get("relative_humidity_value_1", {}).get("value"),
                "wind_speed_ms": obs.get("wind_speed_value_1", {}).get("value"),
                "wind_gust_ms": obs.get("wind_gust_value_1", {}).get("value"),
                "wind_dir_deg": obs.get("wind_direction_value_1", {}).get("value"),
                "precip_1hr_mm": obs.get("precip_accum_one_hour_value_1", {}).get("value", 0.0) or 0.0
            }
    except Exception as exc:
        logger.warning(f"Reference station fetch failed or token unconfigured: {exc}")
    return references


def compute_microclimate_diagnostics(
    tass_obs: Dict[str, Any],
    ref_obs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Cross-references Tassajara actuals against Chews Ridge (CHWC1) and Hastings (CAHC1)
    to compute boundary-layer stability, orographic enhancement ratios, and hardware QC flags.
    """
    diagnostics = {
        "lapse_rate_chews_c_km": None,
        "canyon_confinement_diff_c": None,
        "vpd_kpa": compute_vpd_kpa(tass_obs.get("temp_c"), tass_obs.get("rh_pct")),
        "inversion_active": False,
        "virga_regime_active": False,
        "qc_flag": "OK",
        "qc_notes": []
    }

    t_tass = tass_obs.get("temp_c")
    rh_tass = tass_obs.get("rh_pct")
    chews = ref_obs.get("CHWC1", {})
    hastings = ref_obs.get("CAHC1", {})

    if t_tass is not None and chews.get("temp_c") is not None:
        dz_km = (ELEV_CHEWS_RIDGE_M - ELEV_TASSAJARA_M) / 1000.0  # 1.051 km
        lapse_rate = -1.0 * (chews["temp_c"] - t_tass) / dz_km
        diagnostics["lapse_rate_chews_c_km"] = round(lapse_rate, 2)
        diagnostics["inversion_active"] = bool(lapse_rate < 0.0)

        if lapse_rate > 12.5 and (tass_obs.get("wind_speed_ms") or 0.0) < 1.5:
            diagnostics["qc_flag"] = "WARN_RADIATION_SHIELD_BIAS"
            diagnostics["qc_notes"].append(
                f"Super-adiabatic lapse rate ({lapse_rate:.1f} C/km) under low wind; "
                "Stone Office roof thermal plume or radiation shield fouling suspected."
            )

    if t_tass is not None and hastings.get("temp_c") is not None:
        t_hastings_adj = hastings["temp_c"] + 0.60
        diff_c = round(t_tass - t_hastings_adj, 2)
        diagnostics["canyon_confinement_diff_c"] = diff_c

    chews_p = chews.get("precip_1hr_mm") or 0.0
    hast_p = hastings.get("precip_1hr_mm") or 0.0
    tass_p = tass_obs.get("precip_1hr_mm") or 0.0

    # Distinguish sub-cloud canyon virga (low RH at Stone Office while ridge precipitates)
    # from mechanical tipping-bucket blockage (high RH >= 88% at Stone Office but 0 tips)
    if chews_p > 1.0 and tass_p == 0.0 and (rh_tass or 100.0) < 75.0:
        diagnostics["virga_regime_active"] = True
        diagnostics["qc_notes"].append(
            f"Sub-cloud virga active: Chews Ridge reporting {chews_p:.1f} mm/hr while "
            f"canyon boundary layer remains dry (RH {rh_tass:.0f}%)."
        )
    elif chews_p > 2.5 and hast_p > 2.0 and tass_p == 0.0 and (rh_tass or 0.0) >= 88.0:
        diagnostics["qc_flag"] = "ERR_RAIN_GAUGE_CLOG"
        diagnostics["qc_notes"].append(
            "Zero precipitation recorded at Stone Office under saturated conditions (RH >= 88%) "
            "while both CHWC1 and CAHC1 report >2mm/hr. Inspect rain cone for sycamore leaf blockage."
        )

    return diagnostics


if __name__ == "__main__":
    wl_client = WeatherLinkV2Client(WEATHERLINK_API_KEY, WEATHERLINK_API_SECRET)
    try:
        tass_data = wl_client.fetch_current_observations()
        ref_data = fetch_regional_reference_stations(["CHWC1", "CAHC1"])
        diag = compute_microclimate_diagnostics(tass_data, ref_data)
        logger.info(f"Tassajara Obs: {tass_data}")
        logger.info(f"Microclimate Diagnostics: {diag}")
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")