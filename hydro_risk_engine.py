#!/usr/bin/env python3
"""
hydro_risk_engine.py — Tassajara Zen Mountain Center (TZMC) Early Flood & Road Washout Engine
Production hydrological soil saturation (Antecedent Precipitation Index, API),
nonlinear runoff partitioning, kinematic unit-hydrograph creek routing, and
4-Tier Flood & Tassajara Road Severity classification.

Physical Context:
  - Tassajara Creek Canyon Floor: 485 m (1,590 ft) MSL at Stone Office (Station 36966).
  - Contributing Headwater Massifs: Pine Ridge (~1,450 m), Black Cone (~1,480 m),
    Ventana Cone (VTCC1, 1,453 m), and Chews Ridge (CHWC1, 1,536 m).
  - Effective Upper Watershed Area above TZMC: ~72.0 km^2 of steep, rocky
    Santa Lucia fractured crystalline/metamorphic chaparral & mixed oak-conifer terrain.
"""

import math
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("TassajaraHydroRisk")

# ---------------------------------------------------------------------------
# Physical & Hydrological Constants (Tassajara Creek Watershed & Road Corridor)
# ---------------------------------------------------------------------------
MM_PER_INCH = 25.4
CATCHMENT_AREA_KM2 = 72.0
BASEFLOW_CFS = 6.5  # Typical autumn/winter baseline spring-fed creek discharge (CFS)

# Exponential API Decay Bounds (k in [0.88, 0.94] for rocky Mediterranean chaparral/forest)
API_DECAY_K_DEFAULT = 0.91
API_DECAY_K_MIN = 0.88
API_DECAY_K_MAX = 0.94

# Runoff Coefficient (C) Sigmoid Bounds
RUNOFF_C_MIN = 0.08        # Dry fractured granite / chaparral duff initial abstraction
RUNOFF_C_MAX = 0.86        # Saturated canyon regolith & bedrock sheetflow ceiling
API_MIDPOINT_MM = 32.0     # ~1.26 in API inflection point where macropore flow accelerates
API_SCALE_MM = 11.5        # Sigmoid steepness parameter (mm)

# 3-Hour Steep-Canyon Kinematic Unit Hydrograph Weights (Tc ~ 1.5 - 2.5 hrs off Black Cone/Pine Ridge)
UNIT_HYDROGRAPH_WEIGHTS = (0.48, 0.34, 0.18)

# ---------------------------------------------------------------------------
# Multi-Tier Flood & Road Severity Thresholds (Metric mm & Imperial inches)
# ---------------------------------------------------------------------------
# Tier 1: YELLOW (Advisory)
YELLOW_API_MM = 25.4               # 1.00 in antecedent wetness index
YELLOW_RAIN_24H_MM = 25.4          # 1.00 in / 24h persistent rain
YELLOW_QPF_RATE_MM_HR = 6.35       # 0.25 in/hr moderate shower rate
YELLOW_STAGE_M = 0.75              # Optional stream gauge advisory stage (m)

# Tier 2: ORANGE (Flash Flood Watch / Road Risk)
ORANGE_API_MM = 44.45              # 1.75 in high catchment saturation
ORANGE_SATURATED_FLOOR_MM = 20.32  # 0.80 in primed soil floor for heavy rain band
ORANGE_QPF_RATE_MM_HR = 12.70      # 0.50 in/hr heavy rain band threshold
ORANGE_P90_RATE_MM_HR = 19.05      # 0.75 in/hr P90 downpour threshold
ORANGE_RAIN_24H_MM = 50.80         # 2.00 in / 24h heavy storm accumulation
ORANGE_RUNOFF_MM_HR = 6.50         # Effective surface runoff rate watch threshold
ORANGE_STAGE_M = 1.15              # Optional stream gauge watch stage (m)
ORANGE_RISE_RATE_M_HR = 0.20       # Rapid stage rise >= 0.20 m/hr

# Tier 3: RED (Flash Flood Warning / Road Closure)
RED_QPF_RATE_MM_HR = 25.40         # 1.00 in/hr extreme precipitation intensity
RED_P90_RATE_MM_HR = 31.75         # 1.25 in/hr extreme P90 downpour intensity
RED_RAIN_72H_CRITICAL_MM = 88.90   # 3.50 in / 72h critical 3-day total on wet ground
RED_RAIN_72H_ABSOLUTE_MM = 101.60  # 4.00 in / 72h absolute 3-day washout threshold
RED_API_EXTREME_MM = 63.50         # 2.50 in extreme catchment saturation
RED_RUNOFF_MM_HR = 14.00           # Critical surface runoff rate (bankfull/debris flow)
RED_STAGE_M = 1.60                 # Optional stream gauge bankfull/overtopping stage (m)
RED_RISE_RATE_M_HR = 0.40          # Flash flood surge >= 0.40 m/hr

TIER_METADATA: Dict[str, Dict[str, Any]] = {
    "GREEN": {
        "tier": 0,
        "code": "GREEN",
        "label": "Normal — Creek Stable & Road Passable",
        "color_hex": "#10b981",
        "bg_rgba": "rgba(16, 185, 129, 0.14)",
        "border_rgba": "rgba(16, 185, 129, 0.45)",
        "icon": "🟢",
        "road_status": "Open / Dry to Damp (Standard 4WD/AWD mountain precautions)",
        "creek_status": "Stable Baseflow — Footbridges clear and well above water line",
        "recommended_action": (
            "Routine canyon operations. Standard high-clearance / AWD driving practices "
            "apply along the unpaved 14-mile Tassajara Road."
        ),
    },
    "YELLOW": {
        "tier": 1,
        "code": "YELLOW",
        "label": "Hydrological Advisory — Saturated Catchment & Muddy Road",
        "color_hex": "#eab308",
        "bg_rgba": "rgba(234, 179, 8, 0.16)",
        "border_rgba": "rgba(234, 179, 8, 0.55)",
        "icon": "🟡",
        "road_status": "Caution — Muddy, slick clay grades over Chews Ridge summit; minor rockfall risk",
        "creek_status": "Elevated Flow — Turbid water and moderate rise in Tassajara Creek",
        "recommended_action": (
            "4WD required on Tassajara Road; postpone non-essential low-clearance trips. "
            "Clear leaves/branches from Stone Office & road culverts; monitor creek footbridges."
        ),
    },
    "ORANGE": {
        "tier": 2,
        "code": "ORANGE",
        "label": "Flash Flood Watch & Road Risk — Heavy Rain on Primed Catchment",
        "color_hex": "#f97316",
        "bg_rgba": "rgba(249, 115, 22, 0.18)",
        "border_rgba": "rgba(249, 115, 22, 0.65)",
        "icon": "🟠",
        "road_status": "Hazardous — Travel Strongly Discouraged (Slipout, rutting & rockslide danger)",
        "creek_status": "Rapidly Rising — High velocity flow; woody debris mobilization likely",
        "recommended_action": (
            "Suspend routine travel on Tassajara Road. Inspect lower canyon footbridges for "
            "log/debris damming, secure creek-side equipment, and stage emergency response radio net."
        ),
    },
    "RED": {
        "tier": 3,
        "code": "RED",
        "label": "Flash Flood Warning & Road Closure — Immediate Washout / Overtopping Danger",
        "color_hex": "#ef4444",
        "bg_rgba": "rgba(239, 68, 68, 0.22)",
        "border_rgba": "rgba(239, 68, 68, 0.80)",
        "icon": "🔴",
        "road_status": "CLOSED / IMPASSABLE — Severe washout, debris flow & road slipout danger",
        "creek_status": "BANKFULL / FLOOD STAGE — Footbridge overtopping & log-dam surge hazard",
        "recommended_action": (
            "CLOSE Tassajara Road immediately. Evacuate low-lying creek-bank structures and bathhouse "
            "approaches, prohibit footbridge crossings, and activate TZMC Emergency Response Team."
        ),
    },
}


# ---------------------------------------------------------------------------
# Core Hydrological Equations (API, Runoff Coefficient, Kinematic Routing)
# ---------------------------------------------------------------------------
def compute_dynamic_decay_k(
    rh_pct: Optional[float] = None,
    temp_c: Optional[float] = None,
    base_k: float = API_DECAY_K_DEFAULT,
) -> float:
    """
    Computes the hourly/step Antecedent Precipitation Index recession constant k in [0.88, 0.94],
    tuned for steep rocky Mediterranean chaparral/forest catchments.
    Humid, cool canyon conditions retain soil moisture longer (k -> 0.94);
    warm, dry subsidence accelerates drainage/evapotranspiration (k -> 0.88).
    """
    if rh_pct is None or temp_c is None or pd.isna(rh_pct) or pd.isna(temp_c):
        return float(np.clip(base_k, API_DECAY_K_MIN, API_DECAY_K_MAX))

    t_val = float(temp_c)
    r_val = float(np.clip(rh_pct, 1.0, 100.0))
    es = 0.61078 * math.exp((17.27 * t_val) / (t_val + 237.3))
    ea = es * (r_val / 100.0)
    vpd_kpa = max(0.0, es - ea)

    # Adjust k smoothly between 0.88 (high evaporative demand) and 0.94 (saturated boundary layer)
    k_adj = 0.935 - 0.025 * min(vpd_kpa, 2.2)
    return round(float(np.clip(k_adj, API_DECAY_K_MIN, API_DECAY_K_MAX)), 4)


def compute_api_step(
    prev_api_mm: float,
    precip_mm: float,
    k: float = API_DECAY_K_DEFAULT,
) -> float:
    """
    Exponential decay bucket update:
        API_t = API_{t-1} * k + P_t
    """
    k_clamped = float(np.clip(k, API_DECAY_K_MIN, API_DECAY_K_MAX))
    p_clean = max(0.0, float(precip_mm or 0.0))
    prev_clean = max(0.0, float(prev_api_mm or 0.0))
    return round(prev_clean * k_clamped + p_clean, 3)


def compute_effective_catchment_precip_mm(
    tass_precip_mm: float,
    chews_precip_mm: float = 0.0,
    ventana_precip_mm: float = 0.0,
    bigsur_precip_mm: float = 0.0,
) -> Tuple[float, float]:
    """
    Computes catchment-integrated hourly precipitation (mm) and upstream ridge spillover (mm).
    Because Tassajara Creek is fed by steep tributaries draining Pine Ridge, Black Cone,
    and Chews Ridge (1,450 - 1,536 m MSL), intense crest rainfall runs off directly into
    the canyon even when the 485 m valley floor experiences partial rain-shadowing.
    Returns (effective_precip_mm, upstream_spillover_mm).
    """
    p_tass = max(0.0, float(tass_precip_mm or 0.0))
    p_chews = max(0.0, float(chews_precip_mm or 0.0))
    p_vent = max(0.0, float(ventana_precip_mm or 0.0))
    p_bigsur = max(0.0, float(bigsur_precip_mm or 0.0))

    # Crest average weighted toward Chews Ridge (CHWC1) and Ventana Cone (VTCC1)
    crest_rain_mm = max(p_chews, 0.85 * p_vent, 0.55 * p_bigsur)
    spillover_mm = max(0.0, crest_rain_mm - p_tass)

    # Basin-integrated effective precipitation: 70% valley gauge + 30% upper-ridge excess
    effective_p_mm = max(p_tass, 0.70 * p_tass + 0.30 * crest_rain_mm)
    return round(effective_p_mm, 3), round(spillover_mm, 3)


def compute_runoff_coefficient(
    api_mm: float,
    precip_72hr_mm: float = 0.0,
) -> float:
    """
    Scales the effective runoff coefficient C in [0.08, 0.86] as a nonlinear function
    of current soil wetness state (API_t) and 72-hour deep regolith priming.
    """
    effective_wetness_mm = max(0.0, float(api_mm or 0.0)) + 0.18 * max(0.0, float(precip_72hr_mm or 0.0))
    logistic = 1.0 / (1.0 + math.exp(-(effective_wetness_mm - API_MIDPOINT_MM) / API_SCALE_MM))
    c_val = RUNOFF_C_MIN + (RUNOFF_C_MAX - RUNOFF_C_MIN) * logistic
    return round(float(np.clip(c_val, RUNOFF_C_MIN, RUNOFF_C_MAX)), 3)


def route_runoff_to_discharge_cfs(
    runoff_mm_hr_series: List[float],
    catchment_area_km2: float = CATCHMENT_AREA_KM2,
    baseflow_cfs: float = BASEFLOW_CFS,
) -> List[float]:
    """
    Converts a time series of effective hourly surface runoff depth R_t = C_t * P_t (mm/hr)
    into estimated Tassajara Creek discharge (CFS) using a 3-hour steep-canyon kinematic
    unit hydrograph kernel (w = [0.48, 0.34, 0.18]).
    Note: 1 mm/hr over 1 km^2 = 0.27778 m^3/s = 9.8096 CFS.
    Wait: 1,000 m * 1,000 m * 0.001 m / 3,600 s = 0.27778 m^3/s.
    In steep mountain watersheds with transmission losses and channel storage, the
    effective quickflow delivery factor to the narrow gorge is ~0.22.
    """
    cfs_per_mm_hr = catchment_area_km2 * 0.27778 * 35.3147 * 0.22  # ~155.4 CFS per 1 mm/hr basin-wide
    w0, w1, w2 = UNIT_HYDROGRAPH_WEIGHTS
    out_cfs: List[float] = []
    n = len(runoff_mm_hr_series)
    for i in range(n):
        r0 = max(0.0, float(runoff_mm_hr_series[i]))
        r1 = max(0.0, float(runoff_mm_hr_series[i - 1])) if i >= 1 else r0 * 0.5
        r2 = max(0.0, float(runoff_mm_hr_series[i - 2])) if i >= 2 else r1 * 0.5
        routed_r = w0 * r0 + w1 * r1 + w2 * r2
        q_cfs = baseflow_cfs + routed_r * cfs_per_mm_hr
        out_cfs.append(round(q_cfs, 1))
    return out_cfs


# ---------------------------------------------------------------------------
# Multi-Tier Flood & Road Severity Matrix Classifier
# ---------------------------------------------------------------------------
def evaluate_flood_and_road_tier(
    api_mm: float,
    precip_1hr_mm: float,
    precip_24hr_mm: float,
    precip_72hr_mm: float,
    peak_qpf_1hr_24h_mm: float = 0.0,
    peak_p90_1hr_24h_mm: float = 0.0,
    forecast_qpf_24hr_mm: float = 0.0,
    forecast_qpf_72hr_mm: float = 0.0,
    peak_runoff_24hr_mm_hr: float = 0.0,
    projected_max_api_24h_mm: float = 0.0,
    stream_stage_m: Optional[float] = None,
    rate_of_rise_m_hr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates the 4-Tier Flood & Tassajara Road Severity Matrix (GREEN, YELLOW, ORANGE, RED)
    combining antecedent wetness (API), observed rainfall, forward MOS QPF & P90 rates,
    cumulative 3-day storm totals, and optional physical stream-gauge telemetry.
    """
    eff_api_mm = max(float(api_mm or 0.0), float(projected_max_api_24h_mm or 0.0))
    max_1hr_rate_mm = max(float(precip_1hr_mm or 0.0), float(peak_qpf_1hr_24h_mm or 0.0))
    max_p90_rate_mm = max(max_1hr_rate_mm, float(peak_p90_1hr_24h_mm or 0.0))
    rolling_24h_mm = max(float(precip_24hr_mm or 0.0), float(forecast_qpf_24hr_mm or 0.0))
    total_72h_window_mm = max(
        float(precip_72hr_mm or 0.0),
        float(precip_24hr_mm or 0.0) + float(forecast_qpf_24hr_mm or 0.0),
        float(forecast_qpf_72hr_mm or 0.0),
    )
    soil_primed = (api_mm >= ORANGE_SATURATED_FLOOR_MM) or (eff_api_mm >= YELLOW_API_MM)

    red_reasons: List[str] = []
    orange_reasons: List[str] = []
    yellow_reasons: List[str] = []

    # --- Check RED (Tier 3: Flash Flood Warning / Road Closure) ---
    if max_1hr_rate_mm >= RED_QPF_RATE_MM_HR:
        red_reasons.append(
            f"Extreme rainfall rate {max_1hr_rate_mm / MM_PER_INCH:.2f} in/hr "
            f"(>= 1.00 in/hr flash flood threshold)"
        )
    if max_p90_rate_mm >= RED_P90_RATE_MM_HR and soil_primed:
        red_reasons.append(
            f"Extreme MOS P90 downpour rate {max_p90_rate_mm / MM_PER_INCH:.2f} in/hr "
            f"on primed catchment (API {eff_api_mm / MM_PER_INCH:.2f} in)"
        )
    if total_72h_window_mm >= RED_RAIN_72H_ABSOLUTE_MM:
        red_reasons.append(
            f"Cumulative 3-day storm total {total_72h_window_mm / MM_PER_INCH:.2f} in "
            f"exceeds critical 4.00 in road washout threshold"
        )
    elif total_72h_window_mm >= RED_RAIN_72H_CRITICAL_MM and eff_api_mm >= ORANGE_API_MM:
        red_reasons.append(
            f"3-day rainfall {total_72h_window_mm / MM_PER_INCH:.2f} in combined with "
            f"heavily saturated soil (API {eff_api_mm / MM_PER_INCH:.2f} in)"
        )
    if eff_api_mm >= RED_API_EXTREME_MM and max_1hr_rate_mm >= ORANGE_QPF_RATE_MM_HR:
        red_reasons.append(
            f"Extreme catchment saturation (API {eff_api_mm / MM_PER_INCH:.2f} in) "
            f"with heavy rain ({max_1hr_rate_mm / MM_PER_INCH:.2f} in/hr)"
        )
    if peak_runoff_24hr_mm_hr >= RED_RUNOFF_MM_HR:
        red_reasons.append(
            f"Projected peak surface runoff {peak_runoff_24hr_mm_hr / MM_PER_INCH:.2f} in/hr "
            f"exceeds bankfull canyon threshold"
        )
    if stream_stage_m is not None and not pd.isna(stream_stage_m) and float(stream_stage_m) >= RED_STAGE_M:
        red_reasons.append(f"Creek stage {float(stream_stage_m):.2f} m exceeds bankfull warning level")
    if rate_of_rise_m_hr is not None and not pd.isna(rate_of_rise_m_hr) and float(rate_of_rise_m_hr) >= RED_RISE_RATE_M_HR:
        red_reasons.append(f"Rapid creek surge +{float(rate_of_rise_m_hr):.2f} m/hr")

    # --- Check ORANGE (Tier 2: Flash Flood Watch / Road Risk) ---
    if max_1hr_rate_mm >= ORANGE_QPF_RATE_MM_HR and soil_primed:
        orange_reasons.append(
            f"Heavy rain band ({max_1hr_rate_mm / MM_PER_INCH:.2f} in/hr > 0.50 in/hr) "
            f"on saturated ground (API {eff_api_mm / MM_PER_INCH:.2f} in)"
        )
    elif max_1hr_rate_mm >= 16.5:
        orange_reasons.append(
            f"Intense hourly downpour rate {max_1hr_rate_mm / MM_PER_INCH:.2f} in/hr approaching canyon"
        )
    if max_p90_rate_mm >= ORANGE_P90_RATE_MM_HR and soil_primed:
        orange_reasons.append(
            f"MOS P90 downpour rate {max_p90_rate_mm / MM_PER_INCH:.2f} in/hr "
            f"(> 0.75 in/hr) over wet catchment"
        )
    if rolling_24h_mm >= ORANGE_RAIN_24H_MM:
        orange_reasons.append(
            f"24-hour rainfall accumulation {rolling_24h_mm / MM_PER_INCH:.2f} in (>= 2.00 in)"
        )
    if eff_api_mm >= ORANGE_API_MM and max_1hr_rate_mm >= YELLOW_QPF_RATE_MM_HR:
        orange_reasons.append(
            f"High catchment wetness (API {eff_api_mm / MM_PER_INCH:.2f} in) with active rain"
        )
    if peak_runoff_24hr_mm_hr >= ORANGE_RUNOFF_MM_HR:
        orange_reasons.append(
            f"Elevated surface runoff potential ({peak_runoff_24hr_mm_hr / MM_PER_INCH:.2f} in/hr)"
        )
    if stream_stage_m is not None and not pd.isna(stream_stage_m) and float(stream_stage_m) >= ORANGE_STAGE_M:
        orange_reasons.append(f"Creek stage elevated at {float(stream_stage_m):.2f} m")
    if rate_of_rise_m_hr is not None and not pd.isna(rate_of_rise_m_hr) and float(rate_of_rise_m_hr) >= ORANGE_RISE_RATE_M_HR:
        orange_reasons.append(f"Creek rising at +{float(rate_of_rise_m_hr):.2f} m/hr")

    # --- Check YELLOW (Tier 1: Hydrological Advisory) ---
    if eff_api_mm >= YELLOW_API_MM:
        yellow_reasons.append(
            f"Saturated catchment (API {eff_api_mm / MM_PER_INCH:.2f} in > 1.00 in threshold)"
        )
    if rolling_24h_mm >= YELLOW_RAIN_24H_MM:
        yellow_reasons.append(
            f"Persistent 24-hour rainfall ({rolling_24h_mm / MM_PER_INCH:.2f} in > 1.00 in/24h)"
        )
    if max_1hr_rate_mm >= ORANGE_QPF_RATE_MM_HR:
        yellow_reasons.append(
            f"Brief heavy rain rate ({max_1hr_rate_mm / MM_PER_INCH:.2f} in/hr) on dry soil"
        )
    elif max_p90_rate_mm >= ORANGE_P90_RATE_MM_HR:
        yellow_reasons.append(
            f"P90 downpour risk ({max_p90_rate_mm / MM_PER_INCH:.2f} in/hr) in forecast window"
        )
    if total_72h_window_mm >= 38.1:
        yellow_reasons.append(
            f"72-hour rainfall total {total_72h_window_mm / MM_PER_INCH:.2f} in softening road grades"
        )
    if stream_stage_m is not None and not pd.isna(stream_stage_m) and float(stream_stage_m) >= YELLOW_STAGE_M:
        yellow_reasons.append(f"Creek stage at advisory level ({float(stream_stage_m):.2f} m)")

    if red_reasons:
        code = "RED"
        triggers = red_reasons
    elif orange_reasons:
        code = "ORANGE"
        triggers = orange_reasons
    elif yellow_reasons:
        code = "YELLOW"
        triggers = yellow_reasons
    else:
        code = "GREEN"
        triggers = [
            f"Catchment wetness low (API {api_mm / MM_PER_INCH:.2f} in) and "
            f"24h rainfall ({rolling_24h_mm / MM_PER_INCH:.2f} in) well below advisory thresholds."
        ]

    meta = TIER_METADATA[code].copy()
    meta["trigger_reasons"] = triggers
    meta["trigger_summary"] = " | ".join(triggers)
    return meta


# ---------------------------------------------------------------------------
# Historical API Spin-Up & Forward 72h/168h Hydrograph Projection
# ---------------------------------------------------------------------------
def spinup_historical_api_series(
    obs_df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame] = None,
    base_k: float = API_DECAY_K_DEFAULT,
) -> pd.DataFrame:
    """
    Computes the chronological Antecedent Precipitation Index (API_t), effective
    runoff coefficient (C_t), hourly surface runoff (mm/hr), and routed discharge (CFS)
    across historical observations (sorted oldest -> newest).
    """
    if obs_df is None or obs_df.empty:
        return pd.DataFrame()

    df = obs_df.copy()
    df["_dt_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True, errors="coerce").dt.floor("h")
    df = (
        df.loc[df["_dt_utc"].notna()]
        .sort_values("_dt_utc", ascending=True)
        .drop_duplicates(subset=["_dt_utc"], keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        return pd.DataFrame()

    # Optional join of upstream crest precipitation (CHWC1, VTCC1, PPSC1)
    chews_map: Dict[pd.Timestamp, float] = {}
    vent_map: Dict[pd.Timestamp, float] = {}
    bigsur_map: Dict[pd.Timestamp, float] = {}
    if ref_df is not None and not ref_df.empty and "valid_time_utc" in ref_df.columns:
        r_tmp = ref_df.copy()
        r_tmp["_dt_utc"] = pd.to_datetime(r_tmp["valid_time_utc"], utc=True, errors="coerce").dt.floor("h")
        for stid, target_map in (("CHWC1", chews_map), ("VTCC1", vent_map), ("PPSC1", bigsur_map)):
            sub = r_tmp[r_tmp["station_id"] == stid]
            if not sub.empty and "precip_1hr_mm" in sub.columns:
                for _, r_row in sub.iterrows():
                    if pd.notna(r_row["_dt_utc"]) and pd.notna(r_row["precip_1hr_mm"]):
                        target_map[r_row["_dt_utc"]] = max(0.0, float(r_row["precip_1hr_mm"]))

    api_vals: List[float] = []
    k_vals: List[float] = []
    eff_p_vals: List[float] = []
    spill_vals: List[float] = []
    c_vals: List[float] = []
    runoff_vals: List[float] = []

    current_api = 0.0
    p_series = pd.to_numeric(df.get("precip_1hr_mm", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    rolling_72h_p = p_series.rolling(window=72, min_periods=1).sum().values

    for idx, row in df.iterrows():
        ts = row["_dt_utc"]
        p_tass = float(p_series.iloc[idx])
        p_chews = chews_map.get(ts, 0.0)
        p_vent = vent_map.get(ts, 0.0)
        p_bigsur = bigsur_map.get(ts, 0.0)

        eff_p, spill_p = compute_effective_catchment_precip_mm(
            p_tass, p_chews, p_vent, p_bigsur
        )
        k_t = compute_dynamic_decay_k(row.get("rh_pct"), row.get("temp_c"), base_k=base_k)
        current_api = compute_api_step(current_api, eff_p, k=k_t)
        c_t = compute_runoff_coefficient(current_api, precip_72hr_mm=float(rolling_72h_p[idx]))
        r_t = round(c_t * eff_p, 3)

        api_vals.append(current_api)
        k_vals.append(k_t)
        eff_p_vals.append(eff_p)
        spill_vals.append(spill_p)
        c_vals.append(c_t)
        runoff_vals.append(r_t)

    df["effective_precip_1hr_mm"] = eff_p_vals
    df["upstream_spillover_mm"] = spill_vals
    df["decay_k"] = k_vals
    df["api_mm"] = api_vals
    df["api_in"] = np.round(np.array(api_vals) / MM_PER_INCH, 3)
    df["runoff_coeff_c"] = c_vals
    df["runoff_mm_hr"] = runoff_vals
    df["precip_24hr_rolling_mm"] = p_series.rolling(window=24, min_periods=1).sum().round(3).values
    df["precip_72hr_rolling_mm"] = np.round(rolling_72h_p, 3)
    df["discharge_index_cfs"] = route_runoff_to_discharge_cfs(runoff_vals)
    return df


def project_forward_hydro_series(
    future_fc_df: pd.DataFrame,
    initial_api_mm: float,
    recent_runoff_history_mm_hr: Optional[List[float]] = None,
    recent_precip_48h_mm: float = 0.0,
    base_k: float = API_DECAY_K_DEFAULT,
) -> pd.DataFrame:
    """
    Projects soil saturation (API_t & P90 API_t), effective runoff coefficient (C_t),
    surface runoff rate (mm/hr), routed creek discharge (CFS), and hourly alert tier
    across the forward MOS forecast horizon.
    """
    if future_fc_df is None or future_fc_df.empty:
        return pd.DataFrame()

    fc = future_fc_df.copy()
    fc["_dt_utc"] = pd.to_datetime(fc["valid_time_utc"], utc=True, errors="coerce").dt.floor("h")
    fc = (
        fc.loc[fc["_dt_utc"].notna()]
        .sort_values("_dt_utc", ascending=True)
        .drop_duplicates(subset=["_dt_utc"], keep="first")
        .reset_index(drop=True)
    )
    if fc.empty:
        return pd.DataFrame()

    mos_qpf = pd.to_numeric(
        fc.get("mos_qpf_mm", fc.get("nws_qpf_1hr_mm", 0.0)), errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    mos_p90 = pd.to_numeric(
        fc.get("mos_qpf_p90_mm", mos_qpf * 1.5), errors="coerce"
    ).fillna(mos_qpf * 1.5).clip(lower=0.0)
    mos_p90 = np.maximum(mos_qpf, mos_p90)

    t_series = pd.to_numeric(
        fc.get("corrected_temp_c", fc.get("nws_temp_c", 15.0)), errors="coerce"
    ).fillna(15.0)
    rh_series = pd.to_numeric(
        fc.get("corrected_rh_pct", fc.get("nws_rh_pct", 55.0)), errors="coerce"
    ).fillna(55.0)

    api_exp = max(0.0, float(initial_api_mm or 0.0))
    api_p90 = max(0.0, float(initial_api_mm or 0.0))

    proj_api_mm: List[float] = []
    proj_api_p90_mm: List[float] = []
    proj_c: List[float] = []
    proj_c_p90: List[float] = []
    proj_r_mm: List[float] = []
    proj_r_p90_mm: List[float] = []
    proj_tiers: List[str] = []

    cum_qpf_series = mos_qpf.rolling(window=24, min_periods=1).sum().values
    cum_qpf_72_series = (mos_qpf.rolling(window=72, min_periods=1).sum() + float(recent_precip_48h_mm * 0.5)).values

    for idx in range(len(fc)):
        q_exp = float(mos_qpf.iloc[idx])
        q_p90 = float(mos_p90.iloc[idx])
        k_t = compute_dynamic_decay_k(rh_series.iloc[idx], t_series.iloc[idx], base_k=base_k)

        api_exp = compute_api_step(api_exp, q_exp, k=k_t)
        api_p90 = compute_api_step(api_p90, q_p90, k=k_t)

        c_exp = compute_runoff_coefficient(api_exp, precip_72hr_mm=float(cum_qpf_72_series[idx]))
        c_p90 = compute_runoff_coefficient(api_p90, precip_72hr_mm=float(cum_qpf_72_series[idx]))

        r_exp = round(c_exp * q_exp, 3)
        r_p90 = round(c_p90 * q_p90, 3)

        hr_tier_meta = evaluate_flood_and_road_tier(
            api_mm=api_exp,
            precip_1hr_mm=q_exp,
            precip_24hr_mm=float(cum_qpf_series[idx]),
            precip_72hr_mm=float(cum_qpf_72_series[idx]),
            peak_qpf_1hr_24h_mm=q_exp,
            peak_p90_1hr_24h_mm=q_p90,
            peak_runoff_24hr_mm_hr=r_exp,
        )

        proj_api_mm.append(api_exp)
        proj_api_p90_mm.append(api_p90)
        proj_c.append(c_exp)
        proj_c_p90.append(c_p90)
        proj_r_mm.append(r_exp)
        proj_r_p90_mm.append(r_p90)
        proj_tiers.append(hr_tier_meta["code"])

    seed_hist = list(recent_runoff_history_mm_hr[-2:]) if recent_runoff_history_mm_hr else [0.0, 0.0]
    routed_exp_all = route_runoff_to_discharge_cfs(seed_hist + proj_r_mm)
    routed_p90_all = route_runoff_to_discharge_cfs(seed_hist + proj_r_p90_mm)
    routed_exp = routed_exp_all[len(seed_hist):]
    routed_p90 = routed_p90_all[len(seed_hist):]

    fc["mos_qpf_mm"] = mos_qpf.round(3).values
    fc["mos_qpf_p90_mm"] = np.round(mos_p90, 3)
    fc["projected_api_mm"] = proj_api_mm
    fc["projected_api_in"] = np.round(np.array(proj_api_mm) / MM_PER_INCH, 3)
    fc["projected_api_p90_mm"] = proj_api_p90_mm
    fc["projected_api_p90_in"] = np.round(np.array(proj_api_p90_mm) / MM_PER_INCH, 3)
    fc["projected_runoff_coeff"] = proj_c
    fc["projected_runoff_coeff_p90"] = proj_c_p90
    fc["projected_runoff_mm_hr"] = proj_r_mm
    fc["projected_runoff_in_hr"] = np.round(np.array(proj_r_mm) / MM_PER_INCH, 3)
    fc["projected_runoff_p90_mm_hr"] = proj_r_p90_mm
    fc["projected_runoff_p90_in_hr"] = np.round(np.array(proj_r_p90_mm) / MM_PER_INCH, 3)
    fc["projected_discharge_cfs"] = routed_exp
    fc["projected_discharge_p90_cfs"] = routed_p90
    fc["projected_tier"] = proj_tiers
    return fc


# ---------------------------------------------------------------------------
# End-to-End Evaluation (In-Memory for Streamlit + SQLite Persistence for Cron)
# ---------------------------------------------------------------------------
def ensure_hydro_schema(db_path: str = "tassajara_weather.db") -> None:
    """Idempotently creates the Early Flood & Road Warning tables and view in SQLite."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stream_gauge_observations (
                valid_time_utc TEXT,
                gauge_id TEXT DEFAULT 'TASS_CREEK_FOOTBRIDGE',
                stream_stage_m REAL,
                stream_stage_ft REAL,
                rate_of_rise_m_hr REAL DEFAULT 0.0,
                estimated_discharge_cfs REAL,
                water_temp_c REAL,
                battery_v REAL,
                sensor_health TEXT DEFAULT 'OK',
                qc_flag TEXT DEFAULT 'OK',
                PRIMARY KEY (valid_time_utc, gauge_id)
            );
            CREATE INDEX IF NOT EXISTS idx_stream_gauge_time
                ON stream_gauge_observations(valid_time_utc);

            CREATE TABLE IF NOT EXISTS hydrological_risk_state (
                valid_time_utc TEXT PRIMARY KEY,
                api_mm REAL NOT NULL DEFAULT 0.0,
                api_in REAL NOT NULL DEFAULT 0.0,
                decay_k REAL NOT NULL DEFAULT 0.91,
                runoff_coeff_c REAL NOT NULL DEFAULT 0.08,
                effective_precip_1hr_mm REAL DEFAULT 0.0,
                precip_24hr_mm REAL DEFAULT 0.0,
                precip_72hr_mm REAL DEFAULT 0.0,
                upstream_spillover_mm REAL DEFAULT 0.0,
                current_runoff_mm_hr REAL DEFAULT 0.0,
                peak_runoff_6hr_mm_hr REAL DEFAULT 0.0,
                peak_runoff_24hr_mm_hr REAL DEFAULT 0.0,
                peak_qpf_1hr_24h_mm REAL DEFAULT 0.0,
                peak_p90_1hr_24h_mm REAL DEFAULT 0.0,
                forecast_qpf_24hr_mm REAL DEFAULT 0.0,
                forecast_qpf_72hr_mm REAL DEFAULT 0.0,
                discharge_index_cfs REAL DEFAULT 5.0,
                peak_discharge_24hr_cfs REAL DEFAULT 5.0,
                stream_stage_m REAL,
                rate_of_rise_m_hr REAL,
                alert_tier INTEGER NOT NULL DEFAULT 0,
                alert_code TEXT NOT NULL DEFAULT 'GREEN',
                road_status TEXT,
                creek_status TEXT,
                recommended_action TEXT,
                trigger_summary TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_hydro_risk_time
                ON hydrological_risk_state(valid_time_utc);
            CREATE INDEX IF NOT EXISTS idx_hydro_risk_code
                ON hydrological_risk_state(alert_code, valid_time_utc);

            CREATE TABLE IF NOT EXISTS hydrological_forecast_series (
                issue_time_utc TEXT,
                valid_time_utc TEXT,
                lead_time_hours INTEGER,
                mos_qpf_mm REAL DEFAULT 0.0,
                mos_qpf_p90_mm REAL DEFAULT 0.0,
                projected_api_mm REAL DEFAULT 0.0,
                projected_api_p90_mm REAL DEFAULT 0.0,
                projected_runoff_coeff REAL DEFAULT 0.08,
                projected_runoff_mm_hr REAL DEFAULT 0.0,
                projected_runoff_p90_mm_hr REAL DEFAULT 0.0,
                projected_discharge_cfs REAL DEFAULT 5.0,
                projected_discharge_p90_cfs REAL DEFAULT 5.0,
                projected_tier TEXT DEFAULT 'GREEN',
                PRIMARY KEY (issue_time_utc, valid_time_utc)
            );
            CREATE INDEX IF NOT EXISTS idx_hydro_fc_valid
                ON hydrological_forecast_series(valid_time_utc);
            CREATE INDEX IF NOT EXISTS idx_hydro_fc_issue
                ON hydrological_forecast_series(issue_time_utc);

            DROP VIEW IF EXISTS vw_creek_flood_monitor;
            CREATE VIEW vw_creek_flood_monitor AS
            SELECT
                h.valid_time_utc,
                h.alert_tier,
                h.alert_code,
                h.api_mm,
                h.api_in,
                h.runoff_coeff_c,
                h.effective_precip_1hr_mm,
                h.precip_24hr_mm,
                h.precip_72hr_mm,
                h.upstream_spillover_mm,
                h.current_runoff_mm_hr,
                h.peak_runoff_6hr_mm_hr,
                h.peak_runoff_24hr_mm_hr,
                h.discharge_index_cfs,
                h.peak_discharge_24hr_cfs,
                COALESCE(g.stream_stage_m, h.stream_stage_m) AS stream_stage_m,
                COALESCE(g.rate_of_rise_m_hr, h.rate_of_rise_m_hr) AS rate_of_rise_m_hr,
                g.sensor_health AS gauge_sensor_health,
                h.road_status,
                h.creek_status,
                h.recommended_action,
                h.trigger_summary,
                o.temp_c,
                o.rh_pct,
                o.wind_gust_ms,
                o.mslp_hpa,
                o.pressure_trend_3hr_hpa
            FROM hydrological_risk_state h
            LEFT JOIN local_observations o
                ON h.valid_time_utc = o.valid_time_utc
            LEFT JOIN stream_gauge_observations g
                ON h.valid_time_utc = g.valid_time_utc
               AND g.gauge_id = 'TASS_CREEK_FOOTBRIDGE'
            ORDER BY h.valid_time_utc DESC;
            """
        )
        conn.commit()


def compute_dashboard_hydro_state(
    obs_df: pd.DataFrame,
    future_fc_df: pd.DataFrame,
    ref_dict: Optional[Dict[str, Any]] = None,
    ref_hist_df: Optional[pd.DataFrame] = None,
    stream_gauge_latest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes the complete current hydrological state, 4-tier Flood & Road alert banner
    payload, historical spin-up DataFrame, and 72h/168h forward hydrograph DataFrame.
    Works seamlessly both in the hourly pipeline and live inside Streamlit `app.py`.
    """
    hist_hydro_df = spinup_historical_api_series(obs_df, ref_df=ref_hist_df)

    # Extract current observation & upstream spillover
    chews_p_now = 0.0
    vent_p_now = 0.0
    bigsur_p_now = 0.0
    if isinstance(ref_dict, dict):
        chews_p_now = float(ref_dict.get("CHWC1", {}).get("precip_1hr_mm") or 0.0)
        vent_p_now = float(ref_dict.get("VTCC1", {}).get("precip_1hr_mm") or 0.0)
        bigsur_p_now = float(ref_dict.get("PPSC1", {}).get("precip_1hr_mm") or 0.0)

    if not hist_hydro_df.empty:
        latest_row = hist_hydro_df.iloc[-1]
        valid_time_utc = str(latest_row["valid_time_utc"])
        tass_p_1hr = float(latest_row.get("precip_1hr_mm") or 0.0)
        eff_p_now, spill_now = compute_effective_catchment_precip_mm(
            tass_p_1hr, chews_p_now, vent_p_now, bigsur_p_now
        )
        # Reconcile top hour if live reference station has higher spillover
        prev_api = float(hist_hydro_df.iloc[-2]["api_mm"]) if len(hist_hydro_df) >= 2 else 0.0
        decay_k = float(latest_row.get("decay_k") or API_DECAY_K_DEFAULT)
        api_mm = max(float(latest_row["api_mm"]), compute_api_step(prev_api, eff_p_now, k=decay_k))
        precip_24hr_mm = float(latest_row.get("precip_24hr_rolling_mm") or 0.0)
        precip_72hr_mm = float(latest_row.get("precip_72hr_rolling_mm") or 0.0)
        precip_48hr_mm = float(
            pd.to_numeric(hist_hydro_df.tail(48)["precip_1hr_mm"], errors="coerce").fillna(0.0).sum()
        )
        runoff_coeff_c = compute_runoff_coefficient(api_mm, precip_72hr_mm=precip_72hr_mm)
        current_runoff_mm_hr = round(runoff_coeff_c * eff_p_now, 3)
        discharge_cfs = float(latest_row.get("discharge_index_cfs") or BASEFLOW_CFS)
        recent_runoff_list = hist_hydro_df["runoff_mm_hr"].tail(3).tolist()
    else:
        valid_time_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
        eff_p_now, spill_now = compute_effective_catchment_precip_mm(
            0.0, chews_p_now, vent_p_now, bigsur_p_now
        )
        decay_k = API_DECAY_K_DEFAULT
        api_mm = eff_p_now
        precip_24hr_mm = eff_p_now
        precip_72hr_mm = eff_p_now
        precip_48hr_mm = eff_p_now
        runoff_coeff_c = compute_runoff_coefficient(api_mm, precip_72hr_mm=precip_72hr_mm)
        current_runoff_mm_hr = round(runoff_coeff_c * eff_p_now, 3)
        discharge_cfs = BASEFLOW_CFS
        recent_runoff_list = [current_runoff_mm_hr]

    # Forward Hydrograph & API Projection
    proj_fc_df = project_forward_hydro_series(
        future_fc_df=future_fc_df,
        initial_api_mm=api_mm,
        recent_runoff_history_mm_hr=recent_runoff_list,
        recent_precip_48h_mm=precip_48hr_mm,
        base_k=decay_k,
    )

    if not proj_fc_df.empty:
        fc_6h = proj_fc_df.head(6)
        fc_24h = proj_fc_df.head(24)
        fc_72h = proj_fc_df.head(72)

        peak_runoff_6hr_mm_hr = max(
            current_runoff_mm_hr, float(fc_6h["projected_runoff_mm_hr"].max())
        )
        peak_runoff_24hr_mm_hr = max(
            current_runoff_mm_hr, float(fc_24h["projected_runoff_mm_hr"].max())
        )
        peak_qpf_1hr_24h_mm = float(fc_24h["mos_qpf_mm"].max())
        peak_p90_1hr_24h_mm = float(fc_24h["mos_qpf_p90_mm"].max())
        forecast_qpf_24hr_mm = float(fc_24h["mos_qpf_mm"].sum())
        forecast_qpf_72hr_mm = float(fc_72h["mos_qpf_mm"].sum())
        forecast_p90_72hr_mm = float(fc_72h["mos_qpf_p90_mm"].sum())
        peak_discharge_24hr_cfs = max(
            discharge_cfs, float(fc_24h["projected_discharge_cfs"].max())
        )
        peak_discharge_p90_24hr_cfs = max(
            discharge_cfs, float(fc_24h["projected_discharge_p90_cfs"].max())
        )
        projected_max_api_24h_mm = max(
            api_mm, float(fc_24h["projected_api_mm"].max())
        )
        projected_max_api_72h_mm = max(
            api_mm, float(fc_72h["projected_api_mm"].max())
        )
    else:
        peak_runoff_6hr_mm_hr = current_runoff_mm_hr
        peak_runoff_24hr_mm_hr = current_runoff_mm_hr
        peak_qpf_1hr_24h_mm = 0.0
        peak_p90_1hr_24h_mm = 0.0
        forecast_qpf_24hr_mm = 0.0
        forecast_qpf_72hr_mm = 0.0
        forecast_p90_72hr_mm = 0.0
        peak_discharge_24hr_cfs = discharge_cfs
        peak_discharge_p90_24hr_cfs = discharge_cfs
        projected_max_api_24h_mm = api_mm
        projected_max_api_72h_mm = api_mm

    stage_m = None
    rise_m_hr = None
    gauge_health = "UNINSTALLED (MOS + Tipping Bucket Mode)"
    if isinstance(stream_gauge_latest, dict) and stream_gauge_latest.get("stream_stage_m") is not None:
        stage_m = float(stream_gauge_latest["stream_stage_m"])
        rise_m_hr = float(stream_gauge_latest.get("rate_of_rise_m_hr") or 0.0)
        gauge_health = str(stream_gauge_latest.get("sensor_health") or "OK")

    tier_info = evaluate_flood_and_road_tier(
        api_mm=api_mm,
        precip_1hr_mm=eff_p_now,
        precip_24hr_mm=precip_24hr_mm,
        precip_72hr_mm=precip_72hr_mm,
        peak_qpf_1hr_24h_mm=peak_qpf_1hr_24h_mm,
        peak_p90_1hr_24h_mm=peak_p90_1hr_24h_mm,
        forecast_qpf_24hr_mm=forecast_qpf_24hr_mm,
        forecast_qpf_72hr_mm=forecast_qpf_72hr_mm,
        peak_runoff_24hr_mm_hr=peak_runoff_24hr_mm_hr,
        projected_max_api_24h_mm=projected_max_api_24h_mm,
        stream_stage_m=stage_m,
        rate_of_rise_m_hr=rise_m_hr,
    )

    state_summary = {
        "valid_time_utc": valid_time_utc,
        "api_mm": round(api_mm, 2),
        "api_in": round(api_mm / MM_PER_INCH, 2),
        "projected_max_api_24h_mm": round(projected_max_api_24h_mm, 2),
        "projected_max_api_24h_in": round(projected_max_api_24h_mm / MM_PER_INCH, 2),
        "projected_max_api_72h_mm": round(projected_max_api_72h_mm, 2),
        "projected_max_api_72h_in": round(projected_max_api_72h_mm / MM_PER_INCH, 2),
        "decay_k": round(decay_k, 4),
        "runoff_coeff_c": round(runoff_coeff_c, 3),
        "runoff_coeff_pct": round(runoff_coeff_c * 100.0, 1),
        "effective_precip_1hr_mm": round(eff_p_now, 2),
        "effective_precip_1hr_in": round(eff_p_now / MM_PER_INCH, 2),
        "precip_24hr_mm": round(precip_24hr_mm, 2),
        "precip_24hr_in": round(precip_24hr_mm / MM_PER_INCH, 2),
        "precip_72hr_mm": round(precip_72hr_mm, 2),
        "precip_72hr_in": round(precip_72hr_mm / MM_PER_INCH, 2),
        "upstream_spillover_mm": round(spill_now, 2),
        "upstream_spillover_in": round(spill_now / MM_PER_INCH, 2),
        "current_runoff_mm_hr": round(current_runoff_mm_hr, 3),
        "current_runoff_in_hr": round(current_runoff_mm_hr / MM_PER_INCH, 3),
        "peak_runoff_6hr_mm_hr": round(peak_runoff_6hr_mm_hr, 3),
        "peak_runoff_6hr_in_hr": round(peak_runoff_6hr_mm_hr / MM_PER_INCH, 3),
        "peak_runoff_24hr_mm_hr": round(peak_runoff_24hr_mm_hr, 3),
        "peak_runoff_24hr_in_hr": round(peak_runoff_24hr_mm_hr / MM_PER_INCH, 3),
        "peak_qpf_1hr_24h_mm": round(peak_qpf_1hr_24h_mm, 2),
        "peak_qpf_1hr_24h_in": round(peak_qpf_1hr_24h_mm / MM_PER_INCH, 2),
        "peak_p90_1hr_24h_mm": round(peak_p90_1hr_24h_mm, 2),
        "peak_p90_1hr_24h_in": round(peak_p90_1hr_24h_mm / MM_PER_INCH, 2),
        "forecast_qpf_24hr_mm": round(forecast_qpf_24hr_mm, 2),
        "forecast_qpf_24hr_in": round(forecast_qpf_24hr_mm / MM_PER_INCH, 2),
        "forecast_qpf_72hr_mm": round(forecast_qpf_72hr_mm, 2),
        "forecast_qpf_72hr_in": round(forecast_qpf_72hr_mm / MM_PER_INCH, 2),
        "forecast_p90_72hr_in": round(forecast_p90_72hr_mm / MM_PER_INCH, 2),
        "discharge_index_cfs": round(discharge_cfs, 1),
        "peak_discharge_24hr_cfs": round(peak_discharge_24hr_cfs, 1),
        "peak_discharge_p90_24hr_cfs": round(peak_discharge_p90_24hr_cfs, 1),
        "stream_stage_m": stage_m,
        "rate_of_rise_m_hr": rise_m_hr,
        "gauge_sensor_health": gauge_health,
        "alert_tier": tier_info["tier"],
        "alert_code": tier_info["code"],
        "alert_label": tier_info["label"],
        "color_hex": tier_info["color_hex"],
        "bg_rgba": tier_info["bg_rgba"],
        "border_rgba": tier_info["border_rgba"],
        "icon": tier_info["icon"],
        "road_status": tier_info["road_status"],
        "creek_status": tier_info["creek_status"],
        "recommended_action": tier_info["recommended_action"],
        "trigger_reasons": tier_info["trigger_reasons"],
        "trigger_summary": tier_info["trigger_summary"],
    }

    return {
        "summary": state_summary,
        "hist_hydro_df": hist_hydro_df,
        "proj_fc_df": proj_fc_df,
    }


def evaluate_and_store_hydro_risk(db_path: str = "tassajara_weather.db") -> Dict[str, Any]:
    """
    Pipeline entry point invoked every hour by `run_hourly_mos_pipeline.py`:
      1. Ensures hydrological tables & views exist in SQLite.
      2. Loads the past 14 days (336 hours) of Stone Office and 5-station Santa Lucia
         reference observations to spin up the exact Antecedent Precipitation Index (API_t).
      3. Loads the latest operational MOS forecast batch (`mos_corrected_forecasts` joined
         with `nws_forecasts`).
      4. Projects 6h/24h/72h runoff, creek discharge, and 4-tier Flood & Road alerts.
      5. Persists results to `hydrological_risk_state` and `hydrological_forecast_series`.
    """
    ensure_hydro_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        obs_df = pd.read_sql_query(
            """
            SELECT * FROM local_observations
            WHERE temp_c IS NOT NULL
            ORDER BY valid_time_utc DESC
            LIMIT 336
            """,
            conn,
        )
        ref_hist_df = pd.read_sql_query(
            """
            SELECT * FROM reference_observations
            ORDER BY valid_time_utc DESC
            LIMIT 1680
            """,
            conn,
        )

        # Latest reference snapshot per station
        ref_dict: Dict[str, Any] = {}
        if not ref_hist_df.empty:
            for stid, grp in ref_hist_df.groupby("station_id"):
                ref_dict[str(stid)] = grp.iloc[0].to_dict()

        # Latest optional physical stream gauge reading
        gauge_row = conn.execute(
            """
            SELECT valid_time_utc, gauge_id, stream_stage_m, stream_stage_ft,
                   rate_of_rise_m_hr, estimated_discharge_cfs, sensor_health
            FROM stream_gauge_observations
            WHERE qc_flag = 'OK'
            ORDER BY valid_time_utc DESC
            LIMIT 1
            """
        ).fetchone()
        gauge_latest = None
        if gauge_row:
            gauge_latest = {
                "valid_time_utc": gauge_row[0],
                "gauge_id": gauge_row[1],
                "stream_stage_m": gauge_row[2],
                "stream_stage_ft": gauge_row[3],
                "rate_of_rise_m_hr": gauge_row[4],
                "estimated_discharge_cfs": gauge_row[5],
                "sensor_health": gauge_row[6],
            }

        now_hour_iso = (
            datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
        )
        future_fc_df = pd.read_sql_query(
            """
            SELECT
                m.issue_time_utc,
                m.valid_time_utc,
                COALESCE(f.lead_time_hours, 6) AS lead_time_hours,
                m.corrected_temp_c,
                m.corrected_rh_pct,
                m.mos_pop_pct,
                m.mos_qpf_mm,
                m.mos_qpf_p90_mm
            FROM mos_corrected_forecasts m
            LEFT JOIN nws_forecasts f
                ON m.issue_time_utc = f.issue_time_utc
               AND m.valid_time_utc = f.valid_time_utc
            WHERE m.issue_time_utc = (
                SELECT issue_time_utc
                FROM mos_corrected_forecasts
                ORDER BY issue_time_utc DESC
                LIMIT 1
            )
              AND m.valid_time_utc >= ?
            ORDER BY m.valid_time_utc ASC
            """,
            conn,
            params=(now_hour_iso,),
        )

        result = compute_dashboard_hydro_state(
            obs_df=obs_df,
            future_fc_df=future_fc_df,
            ref_dict=ref_dict,
            ref_hist_df=ref_hist_df,
            stream_gauge_latest=gauge_latest,
        )
        s = result["summary"]
        hist_df = result["hist_hydro_df"]
        proj_df = result["proj_fc_df"]

        # Persist recent 48 hours of spin-up states + current full risk state
        if not hist_df.empty:
            recent_hist_rows = []
            for _, h_row in hist_df.tail(48).iterrows():
                h_ts = str(h_row["valid_time_utc"])
                if h_ts == s["valid_time_utc"]:
                    continue
                h_meta = evaluate_flood_and_road_tier(
                    api_mm=float(h_row["api_mm"]),
                    precip_1hr_mm=float(h_row["effective_precip_1hr_mm"]),
                    precip_24hr_mm=float(h_row["precip_24hr_rolling_mm"]),
                    precip_72hr_mm=float(h_row["precip_72hr_rolling_mm"]),
                )
                recent_hist_rows.append((
                    h_ts,
                    round(float(h_row["api_mm"]), 3),
                    round(float(h_row["api_in"]), 3),
                    round(float(h_row["decay_k"]), 4),
                    round(float(h_row["runoff_coeff_c"]), 3),
                    round(float(h_row["effective_precip_1hr_mm"]), 3),
                    round(float(h_row["precip_24hr_rolling_mm"]), 3),
                    round(float(h_row["precip_72hr_rolling_mm"]), 3),
                    round(float(h_row["upstream_spillover_mm"]), 3),
                    round(float(h_row["runoff_mm_hr"]), 3),
                    round(float(h_row["runoff_mm_hr"]), 3),
                    round(float(h_row["runoff_mm_hr"]), 3),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    round(float(h_row["discharge_index_cfs"]), 1),
                    round(float(h_row["discharge_index_cfs"]), 1),
                    None,
                    None,
                    int(h_meta["tier"]),
                    str(h_meta["code"]),
                    str(h_meta["road_status"]),
                    str(h_meta["creek_status"]),
                    str(h_meta["recommended_action"]),
                    str(h_meta["trigger_summary"]),
                ))
            if recent_hist_rows:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO hydrological_risk_state (
                        valid_time_utc, api_mm, api_in, decay_k, runoff_coeff_c,
                        effective_precip_1hr_mm, precip_24hr_mm, precip_72hr_mm,
                        upstream_spillover_mm, current_runoff_mm_hr,
                        peak_runoff_6hr_mm_hr, peak_runoff_24hr_mm_hr,
                        peak_qpf_1hr_24h_mm, peak_p90_1hr_24h_mm,
                        forecast_qpf_24hr_mm, forecast_qpf_72hr_mm,
                        discharge_index_cfs, peak_discharge_24hr_cfs,
                        stream_stage_m, rate_of_rise_m_hr,
                        alert_tier, alert_code, road_status, creek_status,
                        recommended_action, trigger_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    recent_hist_rows,
                )

        # Upsert current hour's authoritative risk state
        conn.execute(
            """
            INSERT OR REPLACE INTO hydrological_risk_state (
                valid_time_utc, api_mm, api_in, decay_k, runoff_coeff_c,
                effective_precip_1hr_mm, precip_24hr_mm, precip_72hr_mm,
                upstream_spillover_mm, current_runoff_mm_hr,
                peak_runoff_6hr_mm_hr, peak_runoff_24hr_mm_hr,
                peak_qpf_1hr_24h_mm, peak_p90_1hr_24h_mm,
                forecast_qpf_24hr_mm, forecast_qpf_72hr_mm,
                discharge_index_cfs, peak_discharge_24hr_cfs,
                stream_stage_m, rate_of_rise_m_hr,
                alert_tier, alert_code, road_status, creek_status,
                recommended_action, trigger_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                s["valid_time_utc"],
                s["api_mm"],
                s["api_in"],
                s["decay_k"],
                s["runoff_coeff_c"],
                s["effective_precip_1hr_mm"],
                s["precip_24hr_mm"],
                s["precip_72hr_mm"],
                s["upstream_spillover_mm"],
                s["current_runoff_mm_hr"],
                s["peak_runoff_6hr_mm_hr"],
                s["peak_runoff_24hr_mm_hr"],
                s["peak_qpf_1hr_24h_mm"],
                s["peak_p90_1hr_24h_mm"],
                s["forecast_qpf_24hr_mm"],
                s["forecast_qpf_72hr_mm"],
                s["discharge_index_cfs"],
                s["peak_discharge_24hr_cfs"],
                s["stream_stage_m"],
                s["rate_of_rise_m_hr"],
                s["alert_tier"],
                s["alert_code"],
                s["road_status"],
                s["creek_status"],
                s["recommended_action"],
                s["trigger_summary"],
            ),
        )

        # Persist forward projected hydrograph series
        if not proj_df.empty:
            fc_rows = []
            for _, r in proj_df.iterrows():
                fc_rows.append((
                    str(r.get("issue_time_utc") or s["valid_time_utc"]),
                    str(r["valid_time_utc"]),
                    int(r.get("lead_time_hours") or 1),
                    float(r["mos_qpf_mm"]),
                    float(r["mos_qpf_p90_mm"]),
                    float(r["projected_api_mm"]),
                    float(r["projected_api_p90_mm"]),
                    float(r["projected_runoff_coeff"]),
                    float(r["projected_runoff_mm_hr"]),
                    float(r["projected_runoff_p90_mm_hr"]),
                    float(r["projected_discharge_cfs"]),
                    float(r["projected_discharge_p90_cfs"]),
                    str(r["projected_tier"]),
                ))
            conn.executemany(
                """
                INSERT OR REPLACE INTO hydrological_forecast_series (
                    issue_time_utc, valid_time_utc, lead_time_hours,
                    mos_qpf_mm, mos_qpf_p90_mm,
                    projected_api_mm, projected_api_p90_mm,
                    projected_runoff_coeff,
                    projected_runoff_mm_hr, projected_runoff_p90_mm_hr,
                    projected_discharge_cfs, projected_discharge_p90_cfs,
                    projected_tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fc_rows,
            )

        conn.commit()

    logger.info(
        f"Hydrological Risk Evaluated ({s['valid_time_utc']}): "
        f"Tier={s['alert_code']} (Level {s['alert_tier']}), "
        f"API={s['api_in']:.2f} in ({s['api_mm']:.1f} mm), "
        f"C={s['runoff_coeff_c']:.2f}, "
        f"24h_Peak_Discharge={s['peak_discharge_24hr_cfs']:.1f} CFS | {s['trigger_summary']}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = evaluate_and_store_hydro_risk()
    print("Evaluated Hydrological Risk Summary:", res["summary"])
