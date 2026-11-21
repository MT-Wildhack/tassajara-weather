#!/usr/bin/env python3
"""
hydro_risk_engine.py (v2.0 — Calibrated Santa Lucia Hourly-Decay API, Mass-Conserving
Two-Pathway Routing, Non-Compounding P90 Envelope & Coincident 6h/24h/72h Gating)

Physical Context:
  - Tassajara Creek Canyon Floor: 485 m (1,590 ft) MSL at Stone Office (Station 36966).
  - Contributing Headwater Massifs: Pine Ridge (~1,450 m), Black Cone (~1,480 m),
    Ventana Cone (VTCC1, 1,453 m), and Chews Ridge (CHWC1, 1,536 m).
  - Effective Upper Watershed Area above TZMC: ~72.0 km^2 of steep, rocky
    Santa Lucia fractured crystalline/metamorphic chaparral & mixed oak-conifer terrain.
"""

import re
import math
import sqlite3
import logging
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

# Daily API Recession Bounds (converted to exact hourly k_hr = k_day**(1/24))
API_DAILY_K_DEFAULT = 0.91
API_DAILY_K_MIN = 0.86
API_DAILY_K_MAX = 0.94
API_HOURLY_K_DEFAULT = round(API_DAILY_K_DEFAULT ** (1.0 / 24.0), 5)  # 0.99608
API_HOURLY_K_MIN = round(API_DAILY_K_MIN ** (1.0 / 24.0), 5)          # 0.99374
API_HOURLY_K_MAX = round(API_DAILY_K_MAX ** (1.0 / 24.0), 5)          # 0.99742

# Runoff Coefficient (C) Sigmoid Bounds (recalibrated for true multi-day hourly-decayed API)
INITIAL_ABSTRACTION_MM_HR = 1.00  # Canopy interception & fractured-granite initial abstraction
RUNOFF_C_MIN = 0.03               # Dry autumn fractured granite / chaparral duff floor
RUNOFF_C_MAX = 0.68               # Saturated mid-winter regolith & bedrock sheetflow ceiling
API_MIDPOINT_MM = 135.0           # ~5.31 in multi-day API inflection where vadose storage fills
API_SCALE_MM = 32.0               # ~1.26 in sigmoid scale parameter

# Mass-Conserving Two-Pathway Routing (1 mm/hr over 72 km^2 = 20.0 m^3/s = 706.29 CFS)
QUICKFLOW_FRACTION = 0.18         # Fast surface/macropore quickflow delivered within 1-4 hrs
INTERFLOW_FRACTION = 0.57         # Shallow subsurface stormflow (tau ~ 22 hrs)
# Remaining 0.25 of effective runoff enters deep fractured-bedrock groundwater recharge
INTERFLOW_RECESSION_K = 0.955     # Hourly linear reservoir retention (releases 4.5%/hr)
UNIT_HYDROGRAPH_WEIGHTS = (0.32, 0.30, 0.22, 0.16)  # 4-hr steep Santa Lucia canyon kernel

# ---------------------------------------------------------------------------
# Multi-Tier Flood & Road Severity Thresholds (True Multi-Day API Scale)
# ---------------------------------------------------------------------------
# Tier 1: YELLOW (Advisory)
YELLOW_API_MM = 50.80             # 2.00 in multi-day antecedent soil wetness
YELLOW_RAIN_24H_MM = 25.40        # 1.00 in / 24h persistent rain
YELLOW_RAIN_72H_MM = 38.10        # 1.50 in / 72h advance storm prep
YELLOW_QPF_RATE_MM_HR = 3.81      # 0.15 in/hr moderate active/imminent canyon rain
YELLOW_STAGE_M = 0.75

# Tier 2: ORANGE (Flash Flood Watch / Hazardous Road)
ORANGE_API_MM = 88.90             # 3.50 in saturated catchment
ORANGE_PRIMED_FLOOR_MM = 50.80    # 2.00 in primed soil floor for heavy rain band
ORANGE_QPF_RATE_MM_HR = 12.70     # 0.50 in/hr heavy rain band
ORANGE_SUSTAINED_RATE_MM_HR = 7.62 # 0.30 in/hr sustained rain on saturated soil
ORANGE_P90_RATE_MM_HR = 19.05     # 0.75 in/hr P90 downpour threshold
ORANGE_RAIN_24H_MM = 57.15        # 2.25 in / 24h heavy storm accumulation
ORANGE_RAIN_72H_MM = 88.90        # 3.50 in / 72h road slipout accumulation
ORANGE_RUNOFF_MM_HR = 6.50        # Effective runoff rate watch threshold
ORANGE_STAGE_M = 1.15
ORANGE_RISE_RATE_M_HR = 0.20

# Tier 3: RED (Flash Flood Warning / Road Closure — Imminent <= 6h Only)
RED_QPF_RATE_MM_HR = 25.40        # 1.00 in/hr extreme hourly intensity
RED_TORRENTIAL_RATE_MM_HR = 16.51 # 0.65 in/hr torrential intensity on saturated soil
RED_P90_RATE_MM_HR = 31.75        # 1.25 in/hr extreme P90 downpour on saturated soil
RED_RAIN_24H_MM = 88.90           # 3.50 in / 24h extreme storm total on saturated soil
RED_RAIN_72H_MM = 114.30          # 4.50 in / 72h extreme washout total on saturated soil
RED_API_EXTREME_MM = 127.00       # 5.00 in extreme multi-day saturation
RED_RUNOFF_MM_HR = 12.00          # Critical surface runoff rate (bankfull / debris flow)
RED_STAGE_M = 1.60
RED_RISE_RATE_M_HR = 0.40

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
        "label": "Hydrological Advisory — Wetting Catchment & Muddy Road",
        "color_hex": "#eab308",
        "bg_rgba": "rgba(234, 179, 8, 0.16)",
        "border_rgba": "rgba(234, 179, 8, 0.55)",
        "icon": "🟡",
        "road_status": "Caution — Muddy, slick clay grades over Chews Ridge summit; minor rockfall risk",
        "creek_status": "Elevated Flow — Turbid water and moderate rise in Tassajara Creek",
        "recommended_action": (
            "4WD required on Tassajara Road; postpone non-essential low-clearance trips. "
            "Monitor creek footbridges."
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


def get_severity_matrix_rows() -> List[Dict[str, str]]:
    """Single source of truth for the UI Severity Matrix table (always synced to code thresholds)."""
    return [
        {
            "Alert Tier": "🟢 Tier 0 — GREEN (Normal)",
            "Soil Saturation & Rain Triggers": (
                f"API < {YELLOW_API_MM/MM_PER_INCH:.2f} in AND 1h Rate < {YELLOW_QPF_RATE_MM_HR/MM_PER_INCH:.2f} in/hr "
                f"AND 24h Rain < {YELLOW_RAIN_24H_MM/MM_PER_INCH:.2f} in AND 72h Rain < {YELLOW_RAIN_72H_MM/MM_PER_INCH:.2f} in"
            ),
            "Tassajara Creek Response": TIER_METADATA["GREEN"]["creek_status"],
            "Tassajara Road Passability": TIER_METADATA["GREEN"]["road_status"],
            "Operational Protocol": TIER_METADATA["GREEN"]["recommended_action"],
        },
        {
            "Alert Tier": "🟡 Tier 1 — YELLOW (Advisory)",
            "Soil Saturation & Rain Triggers": (
                f"API >= {YELLOW_API_MM/MM_PER_INCH:.2f} in OR Imminent (<=6h) Rate >= {YELLOW_QPF_RATE_MM_HR/MM_PER_INCH:.2f} in/hr "
                f"OR 24h Rain >= {YELLOW_RAIN_24H_MM/MM_PER_INCH:.2f} in OR 72h Rain >= {YELLOW_RAIN_72H_MM/MM_PER_INCH:.2f} in"
            ),
            "Tassajara Creek Response": TIER_METADATA["YELLOW"]["creek_status"],
            "Tassajara Road Passability": TIER_METADATA["YELLOW"]["road_status"],
            "Operational Protocol": TIER_METADATA["YELLOW"]["recommended_action"],
        },
        {
            "Alert Tier": "🟠 Tier 2 — ORANGE (Watch / Road Risk)",
            "Soil Saturation & Rain Triggers": (
                f"Within 24h (coincident hour): 1h Rate >= {ORANGE_QPF_RATE_MM_HR/MM_PER_INCH:.2f} in/hr on primed soil "
                f"(API >= {ORANGE_PRIMED_FLOOR_MM/MM_PER_INCH:.2f} in), OR >= {ORANGE_SUSTAINED_RATE_MM_HR/MM_PER_INCH:.2f} in/hr "
                f"on saturated soil (API >= {ORANGE_API_MM/MM_PER_INCH:.2f} in), OR 24h >= {ORANGE_RAIN_24H_MM/MM_PER_INCH:.2f} in "
                f"with active rain, OR Runoff >= {ORANGE_RUNOFF_MM_HR/MM_PER_INCH:.2f} in/hr, OR Routed Creek Q >= 250 CFS"
            ),
            "Tassajara Creek Response": TIER_METADATA["ORANGE"]["creek_status"],
            "Tassajara Road Passability": TIER_METADATA["ORANGE"]["road_status"],
            "Operational Protocol": TIER_METADATA["ORANGE"]["recommended_action"],
        },
        {
            "Alert Tier": "🔴 Tier 3 — RED (Warning / Road Closure)",
            "Soil Saturation & Rain Triggers": (
                f"Imminent/Active (<=6h, coincident hour): 1h Rate >= {RED_QPF_RATE_MM_HR/MM_PER_INCH:.2f} in/hr, "
                f"OR P90 >= {RED_P90_RATE_MM_HR/MM_PER_INCH:.2f} in/hr on saturated soil (API >= {ORANGE_API_MM/MM_PER_INCH:.2f} in), "
                f"OR Rate >= {RED_TORRENTIAL_RATE_MM_HR/MM_PER_INCH:.2f} in/hr with API >= {RED_API_EXTREME_MM/MM_PER_INCH:.2f} in "
                f"(or 24h >= {RED_RAIN_24H_MM/MM_PER_INCH:.2f} in / 72h >= {RED_RAIN_72H_MM/MM_PER_INCH:.2f} in on saturated soil), "
                f"OR Runoff >= {RED_RUNOFF_MM_HR/MM_PER_INCH:.2f} in/hr, OR Routed Creek Q >= 500 CFS"
            ),
            "Tassajara Creek Response": TIER_METADATA["RED"]["creek_status"],
            "Tassajara Road Passability": TIER_METADATA["RED"]["road_status"],
            "Operational Protocol": TIER_METADATA["RED"]["recommended_action"],
        },
    ]


# ---------------------------------------------------------------------------
# Core Hydrological Equations (Hourly API, Runoff Coefficient, Two-Pathway Routing)
# ---------------------------------------------------------------------------
def compute_dynamic_decay_k(
    rh_pct: Optional[float] = None,
    temp_c: Optional[float] = None,
    base_daily_k: float = API_DAILY_K_DEFAULT,
) -> float:
    """
    Computes the TRUE HOURLY Antecedent Precipitation Index recession constant
    k_hr = k_day**(1/24) in [0.99374, 0.99742] (daily k_day in [0.86, 0.94]).
    """
    if rh_pct is None or temp_c is None or pd.isna(rh_pct) or pd.isna(temp_c):
        k_day = float(np.clip(base_daily_k, API_DAILY_K_MIN, API_DAILY_K_MAX))
        return round(k_day ** (1.0 / 24.0), 5)

    t_val = float(temp_c)
    r_val = float(np.clip(rh_pct, 1.0, 100.0))
    es = 0.61078 * math.exp((17.27 * t_val) / (t_val + 237.3))
    ea = es * (r_val / 100.0)
    vpd_kpa = max(0.0, es - ea)

    k_day = float(np.clip(0.935 - 0.032 * min(vpd_kpa, 2.2), API_DAILY_K_MIN, API_DAILY_K_MAX))
    return round(k_day ** (1.0 / 24.0), 5)


def compute_api_step(
    prev_api_mm: float,
    precip_mm: float,
    k_hr: float = API_HOURLY_K_DEFAULT,
) -> float:
    """Hourly exponential decay bucket update: API_t = API_{t-1} * k_hr + P_t."""
    k_clamped = float(np.clip(k_hr, API_HOURLY_K_MIN, API_HOURLY_K_MAX))
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
    Sanity-caps rogue remote gauge spikes (> 100 mm/hr) to protect against telemetry faults.
    """
    p_tass = min(100.0, max(0.0, float(tass_precip_mm or 0.0)))
    p_chews = min(100.0, max(0.0, float(chews_precip_mm or 0.0)))
    p_vent = min(100.0, max(0.0, float(ventana_precip_mm or 0.0)))
    p_bigsur = min(100.0, max(0.0, float(bigsur_precip_mm or 0.0)))

    crest_rain_mm = max(p_chews, 0.85 * p_vent, 0.55 * p_bigsur)
    spillover_mm = max(0.0, crest_rain_mm - p_tass)
    effective_p_mm = max(p_tass, 0.70 * p_tass + 0.30 * crest_rain_mm)
    return round(effective_p_mm, 3), round(spillover_mm, 3)


def compute_runoff_coefficient(
    api_mm: float,
    precip_72hr_mm: float = 0.0,
) -> float:
    """
    Scales the effective runoff coefficient C in [0.03, 0.68] as a smooth nonlinear
    function of multi-day antecedent soil wetness (API_t).
    """
    api_clean = max(0.0, float(api_mm or 0.0))
    deep_priming_mm = min(api_clean * 0.15, 0.05 * max(0.0, float(precip_72hr_mm or 0.0)))
    effective_wetness_mm = api_clean + deep_priming_mm
    logistic = 1.0 / (1.0 + math.exp(-(effective_wetness_mm - API_MIDPOINT_MM) / API_SCALE_MM))
    c_val = RUNOFF_C_MIN + (RUNOFF_C_MAX - RUNOFF_C_MIN) * logistic
    return round(float(np.clip(c_val, RUNOFF_C_MIN, RUNOFF_C_MAX)), 3)


def compute_hourly_effective_runoff_mm(precip_mm: float, runoff_coeff_c: float) -> float:
    """Applies initial canopy/regolith abstraction before C_t partitioning (no double abstraction)."""
    p_net = max(0.0, float(precip_mm or 0.0) - INITIAL_ABSTRACTION_MM_HR)
    return round(max(0.0, float(runoff_coeff_c) * p_net), 3)


def route_runoff_to_discharge_cfs(
    runoff_mm_hr_series: List[float],
    catchment_area_km2: float = CATCHMENT_AREA_KM2,
    baseflow_cfs: float = BASEFLOW_CFS,
    initial_interflow_storage_mm: float = 0.0,
    return_final_storage: bool = False,
):
    """
    Mass-conserving two-pathway kinematic canyon router:
      - Exact volumetric unit conversion: 1 mm/hr over 72 km^2 = 706.29 CFS.
      - QUICKFLOW_FRACTION (18%) routes through a 4-hour steep-canyon unit hydrograph.
      - INTERFLOW_FRACTION (57%) enters a linear shallow-subsurface reservoir (k=0.955,
        releasing 4.5%/hr) for smooth storm hydrograph recession.
      - Remaining 25% recharges deep fractured-bedrock groundwater (supporting baseflow).
    """
    cfs_per_mm_hr = catchment_area_km2 * (1000.0 / 3600.0) * 35.3147  # 706.29 CFS per mm/hr
    w0, w1, w2, w3 = UNIT_HYDROGRAPH_WEIGHTS
    storage_mm = max(0.0, float(initial_interflow_storage_mm or 0.0))
    out_cfs: List[float] = []

    n = len(runoff_mm_hr_series)
    for i in range(n):
        r0 = max(0.0, float(runoff_mm_hr_series[i]))
        r1 = max(0.0, float(runoff_mm_hr_series[i - 1])) if i >= 1 else 0.0
        r2 = max(0.0, float(runoff_mm_hr_series[i - 2])) if i >= 2 else 0.0
        r3 = max(0.0, float(runoff_mm_hr_series[i - 3])) if i >= 3 else 0.0

        quick_mm_hr = QUICKFLOW_FRACTION * (w0 * r0 + w1 * r1 + w2 * r2 + w3 * r3)
        storage_mm = INTERFLOW_RECESSION_K * storage_mm + INTERFLOW_FRACTION * r0
        interflow_mm_hr = (1.0 - INTERFLOW_RECESSION_K) * storage_mm

        q_cfs = baseflow_cfs + (quick_mm_hr + interflow_mm_hr) * cfs_per_mm_hr
        out_cfs.append(round(q_cfs, 1))

    if return_final_storage:
        return out_cfs, round(storage_mm, 4)
    return out_cfs


# ---------------------------------------------------------------------------
# Coincident Multi-Tier Flood & Road Severity Matrix Classifier
# ---------------------------------------------------------------------------
def evaluate_single_hour_tier(
    api_mm: float,
    precip_1hr_mm: float,
    p90_1hr_mm: float,
    runoff_mm_hr: float,
    rolling_24h_mm: float,
    rolling_72h_mm: float,
    allow_red: bool = True,
    allow_orange: bool = True,
    discharge_cfs: Optional[float] = None,
    stream_stage_m: Optional[float] = None,
    rate_of_rise_m_hr: Optional[float] = None,
) -> Tuple[str, List[str]]:
    """
    Evaluates a single coincident hourly hydrological state (api_t, qpf_t, p90_t,
    runoff_t, rolling_24h_t, rolling_72h_t). Prevents false alarms from pairing
    non-coincident end-of-storm API with earlier peak rain rates.
    """
    api_t = max(0.0, float(api_mm or 0.0))
    p1 = max(0.0, float(precip_1hr_mm or 0.0))
    p90 = max(p1, float(p90_1hr_mm or 0.0))
    r1 = max(0.0, float(runoff_mm_hr or 0.0))
    p24 = max(0.0, float(rolling_24h_mm or 0.0))
    p72 = max(0.0, float(rolling_72h_mm or 0.0))

    primed = api_t >= ORANGE_PRIMED_FLOOR_MM
    saturated = api_t >= ORANGE_API_MM
    extreme_sat = api_t >= RED_API_EXTREME_MM

    red_reasons: List[str] = []
    orange_reasons: List[str] = []
    yellow_reasons: List[str] = []

    if allow_red:
        if p1 >= RED_QPF_RATE_MM_HR:
            red_reasons.append(
                f"Extreme hourly rainfall rate {p1/MM_PER_INCH:.2f} in/hr (>= 1.00 in/hr flash flood threshold)"
            )
        if p90 >= RED_P90_RATE_MM_HR and saturated and p1 >= ORANGE_QPF_RATE_MM_HR:
            red_reasons.append(
                f"Extreme P90 downpour {p90/MM_PER_INCH:.2f} in/hr on saturated catchment (API {api_t/MM_PER_INCH:.2f} in)"
            )
        if extreme_sat and p1 >= RED_TORRENTIAL_RATE_MM_HR:
            red_reasons.append(
                f"Extreme catchment saturation (API {api_t/MM_PER_INCH:.2f} in) with torrential rain ({p1/MM_PER_INCH:.2f} in/hr)"
            )
        if p24 >= RED_RAIN_24H_MM and saturated and p1 >= RED_TORRENTIAL_RATE_MM_HR:
            red_reasons.append(
                f"Extreme 24h storm total {p24/MM_PER_INCH:.2f} in on saturated soil with intense rain ({p1/MM_PER_INCH:.2f} in/hr)"
            )
        elif p72 >= RED_RAIN_72H_MM and saturated and p1 >= RED_TORRENTIAL_RATE_MM_HR:
            red_reasons.append(
                f"Extreme 72h storm total {p72/MM_PER_INCH:.2f} in on saturated soil with intense rain ({p1/MM_PER_INCH:.2f} in/hr)"
            )
        if r1 >= RED_RUNOFF_MM_HR:
            red_reasons.append(
                f"Critical surface runoff {r1/MM_PER_INCH:.2f} in/hr exceeds bankfull canyon threshold"
            )
        if stream_stage_m is not None and not pd.isna(stream_stage_m) and float(stream_stage_m) >= RED_STAGE_M:
            red_reasons.append(f"Creek stage {float(stream_stage_m):.2f} m exceeds bankfull warning level")
        if discharge_cfs is not None and not pd.isna(discharge_cfs) and float(discharge_cfs) >= 500.0:
            red_reasons.append(f"Routed creek discharge {float(discharge_cfs):.0f} CFS (>= 500 CFS flood stage)")
        if rate_of_rise_m_hr is not None and not pd.isna(rate_of_rise_m_hr) and float(rate_of_rise_m_hr) >= RED_RISE_RATE_M_HR:
            red_reasons.append(f"Rapid creek surge +{float(rate_of_rise_m_hr):.2f} m/hr")

    if allow_orange:
        if p1 >= ORANGE_QPF_RATE_MM_HR and primed:
            orange_reasons.append(
                f"Heavy rain band ({p1/MM_PER_INCH:.2f} in/hr >= 0.50 in/hr) on primed catchment (API {api_t/MM_PER_INCH:.2f} in)"
            )
        elif p1 >= RED_TORRENTIAL_RATE_MM_HR:
            orange_reasons.append(
                f"Torrential hourly downpour ({p1/MM_PER_INCH:.2f} in/hr >= 0.65 in/hr)"
            )
        elif saturated and p1 >= ORANGE_SUSTAINED_RATE_MM_HR:
            orange_reasons.append(
                f"Sustained heavy rain ({p1/MM_PER_INCH:.2f} in/hr >= 0.30 in/hr) on saturated soil (API {api_t/MM_PER_INCH:.2f} in)"
            )
        if p90 >= ORANGE_P90_RATE_MM_HR and primed and p1 >= ORANGE_SUSTAINED_RATE_MM_HR:
            orange_reasons.append(
                f"MOS P90 downpour potential {p90/MM_PER_INCH:.2f} in/hr (>= 0.75 in/hr) on primed catchment"
            )
        if p24 >= ORANGE_RAIN_24H_MM and p1 >= YELLOW_QPF_RATE_MM_HR:
            orange_reasons.append(
                f"Heavy 24h storm accumulation {p24/MM_PER_INCH:.2f} in (>= 2.25 in) with active canyon rain"
            )
        elif p72 >= ORANGE_RAIN_72H_MM and primed and p1 >= YELLOW_QPF_RATE_MM_HR:
            orange_reasons.append(
                f"Multi-day storm total {p72/MM_PER_INCH:.2f} in on primed soil elevates road slipout risk"
            )
        if r1 >= ORANGE_RUNOFF_MM_HR:
            orange_reasons.append(f"Elevated effective surface runoff ({r1/MM_PER_INCH:.2f} in/hr)")
        if stream_stage_m is not None and not pd.isna(stream_stage_m) and float(stream_stage_m) >= ORANGE_STAGE_M:
            orange_reasons.append(f"Creek stage elevated at {float(stream_stage_m):.2f} m")
        if discharge_cfs is not None and not pd.isna(discharge_cfs) and float(discharge_cfs) >= 250.0:
            orange_reasons.append(f"High-velocity creek discharge {float(discharge_cfs):.0f} CFS (>= 250 CFS debris/footbridge risk)")
        if rate_of_rise_m_hr is not None and not pd.isna(rate_of_rise_m_hr) and float(rate_of_rise_m_hr) >= ORANGE_RISE_RATE_M_HR:
            orange_reasons.append(f"Creek rising at +{float(rate_of_rise_m_hr):.2f} m/hr")

    if api_t >= YELLOW_API_MM:
        yellow_reasons.append(
            f"Wetted catchment soil (API {api_t/MM_PER_INCH:.2f} in >= 2.00 in)"
        )
    if p1 >= YELLOW_QPF_RATE_MM_HR:
        yellow_reasons.append(
            f"Moderate canyon rain ({p1/MM_PER_INCH:.2f} in/hr >= 0.15 in/hr)"
        )
    if p24 >= YELLOW_RAIN_24H_MM:
        yellow_reasons.append(
            f"24h rainfall {p24/MM_PER_INCH:.2f} in (>= 1.00 in/24h)"
        )
    if p72 >= YELLOW_RAIN_72H_MM:
        yellow_reasons.append(
            f"72h storm total {p72/MM_PER_INCH:.2f} in (>= 1.50 in) softening road grades"
        )
    if stream_stage_m is not None and not pd.isna(stream_stage_m) and float(stream_stage_m) >= YELLOW_STAGE_M:
        yellow_reasons.append(f"Creek stage at advisory level ({float(stream_stage_m):.2f} m)")

    if red_reasons:
        return "RED", red_reasons
    if orange_reasons:
        return "ORANGE", orange_reasons
    if yellow_reasons:
        return "YELLOW", yellow_reasons
    return "GREEN", []


def evaluate_flood_and_road_tier(
    api_mm: float,
    precip_1hr_mm: float,
    precip_24hr_mm: float,
    precip_72hr_mm: float,
    current_runoff_mm_hr: float = 0.0,
    proj_fc_df: Optional[pd.DataFrame] = None,
    current_discharge_cfs: Optional[float] = None,
    stream_stage_m: Optional[float] = None,
    rate_of_rise_m_hr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Aggregates coincident hourly evaluations across three operational horizons:
      - Current / Imminent (0..6h): Full GREEN / YELLOW / ORANGE / RED authority.
      - Near-Term Lookahead (7..24h): Up to ORANGE (Watch) for advance road/bridge prep.
      - Extended Outlook (25..72h): Up to YELLOW (Advisory) for advance storm prep.
    """
    tier_rank = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
    best_code = "GREEN"
    collected: Dict[str, List[str]] = {"RED": [], "ORANGE": [], "YELLOW": []}

    # 1. Current observed hour (t = 0)
    cur_code, cur_reasons = evaluate_single_hour_tier(
        api_mm=api_mm,
        precip_1hr_mm=precip_1hr_mm,
        p90_1hr_mm=precip_1hr_mm,
        runoff_mm_hr=current_runoff_mm_hr,
        rolling_24h_mm=precip_24hr_mm,
        rolling_72h_mm=precip_72hr_mm,
        allow_red=True,
        allow_orange=True,
        discharge_cfs=current_discharge_cfs,
        stream_stage_m=stream_stage_m,
        rate_of_rise_m_hr=rate_of_rise_m_hr,
    )
    if tier_rank[cur_code] > tier_rank[best_code]:
        best_code = cur_code
    if cur_code in collected:
        collected[cur_code].extend(cur_reasons)

    # 2. Forward forecast hours (coincident per-hour state)
    if proj_fc_df is not None and not proj_fc_df.empty:
        for idx, row in proj_fc_df.head(72).iterrows():
            hr_pos = int(idx) + 1
            allow_red = hr_pos <= 6
            allow_orange = hr_pos <= 24
            h_code, h_reasons = evaluate_single_hour_tier(
                api_mm=float(row.get("projected_api_mm", 0.0)),
                precip_1hr_mm=float(row.get("mos_qpf_mm", 0.0)),
                p90_1hr_mm=float(row.get("mos_qpf_p90_mm", 0.0)),
                runoff_mm_hr=float(row.get("projected_runoff_mm_hr", 0.0)),
                rolling_24h_mm=float(row.get("rolling_24h_mm", 0.0)),
                rolling_72h_mm=float(row.get("rolling_72h_mm", 0.0)),
                allow_red=allow_red,
                allow_orange=allow_orange,
                discharge_cfs=float(row.get("projected_discharge_cfs", 0.0)),
            )
            if tier_rank[h_code] > tier_rank[best_code]:
                best_code = h_code
            if h_code in collected:
                window_tag = (
                    "Imminent (<=6h): " if hr_pos <= 6
                    else ("24h Watch: " if hr_pos <= 24 else "72h Outlook: ")
                )
                for r_txt in h_reasons:
                    tagged = f"{window_tag}{r_txt}"
                    if tagged not in collected[h_code]:
                        collected[h_code].append(tagged)

    if best_code != "GREEN" and collected[best_code]:
        deduped = []
        seen_cats = set()
        for msg in collected[best_code]:
            body = msg.split(": ", 1)[1] if ": " in msg else msg
            cat = re.sub(r"\d+(?:\.\d+)?", "#", body).split("(")[0].strip()[:22]
            if cat not in seen_cats:
                seen_cats.add(cat)
                deduped.append(msg)
        triggers = deduped[:4]
    else:
        triggers = [
            f"Catchment wetness low (API {float(api_mm or 0.0)/MM_PER_INCH:.2f} in) and "
            f"24h rainfall ({float(precip_24hr_mm or 0.0)/MM_PER_INCH:.2f} in) below advisory thresholds."
        ]

    meta = TIER_METADATA[best_code].copy()
    meta["trigger_reasons"] = triggers
    meta["trigger_summary"] = " | ".join(triggers)
    return meta


# ---------------------------------------------------------------------------
# Historical API Spin-Up & Forward 72h/168h Hydrograph Projection
# ---------------------------------------------------------------------------
def spinup_historical_api_series(
    obs_df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame] = None,
    base_daily_k: float = API_DAILY_K_DEFAULT,
) -> pd.DataFrame:
    """
    Computes chronological hourly API_t, C_t, effective runoff, and two-pathway routed
    discharge across historical observations (oldest -> newest) on a continuous 1-hr grid.
    Preserves `is_observed_hour` so missing gap hours are never persisted as false GREEN states.
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

    observed_ts_set = set(df["_dt_utc"].tolist())
    full_hr_idx = pd.date_range(
        start=df["_dt_utc"].min(), end=df["_dt_utc"].max(), freq="1h", tz="UTC"
    )
    df = (
        df.set_index("_dt_utc")
        .reindex(full_hr_idx)
        .reset_index()
        .rename(columns={"index": "_dt_utc"})
    )
    df["valid_time_utc"] = df["_dt_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df["is_observed_hour"] = df["_dt_utc"].isin(observed_ts_set)
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].ffill().bfill().fillna("TASS_VP2")
    df["temp_c"] = (
        pd.to_numeric(df.get("temp_c"), errors="coerce")
        .interpolate(method="linear", limit_direction="both")
        .fillna(15.0)
    )
    df["rh_pct"] = (
        pd.to_numeric(df.get("rh_pct"), errors="coerce")
        .interpolate(method="linear", limit_direction="both")
        .fillna(60.0)
    )
    df["precip_1hr_mm"] = (
        pd.to_numeric(df.get("precip_1hr_mm"), errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    chews_map: Dict[pd.Timestamp, float] = {}
    vent_map: Dict[pd.Timestamp, float] = {}
    bigsur_map: Dict[pd.Timestamp, float] = {}
    if ref_df is not None and not ref_df.empty and "valid_time_utc" in ref_df.columns:
        r_tmp = ref_df.copy()
        if "source" in r_tmp.columns:
            r_tmp = r_tmp[~r_tmp["source"].isin(["OPENMETEO_SYNTH", "OFFLINE"])].copy()
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
    p_series = df["precip_1hr_mm"]
    rolling_72h_p = p_series.rolling(window=72, min_periods=1).sum().values

    for idx, row in df.iterrows():
        ts = row["_dt_utc"]
        eff_p, spill_p = compute_effective_catchment_precip_mm(
            float(p_series.iloc[idx]),
            chews_map.get(ts, 0.0),
            vent_map.get(ts, 0.0),
            bigsur_map.get(ts, 0.0),
        )
        k_t = compute_dynamic_decay_k(row.get("rh_pct"), row.get("temp_c"), base_daily_k=base_daily_k)
        current_api = compute_api_step(current_api, eff_p, k_hr=k_t)
        c_t = compute_runoff_coefficient(current_api, precip_72hr_mm=float(rolling_72h_p[idx]))
        r_t = compute_hourly_effective_runoff_mm(eff_p, c_t)

        api_vals.append(current_api)
        k_vals.append(k_t)
        eff_p_vals.append(eff_p)
        spill_vals.append(spill_p)
        c_vals.append(c_t)
        runoff_vals.append(r_t)

    routed_cfs, final_interflow_mm = route_runoff_to_discharge_cfs(
        runoff_vals, return_final_storage=True
    )

    df["effective_precip_1hr_mm"] = eff_p_vals
    df["upstream_spillover_mm"] = spill_vals
    df["decay_k"] = k_vals
    df["api_mm"] = api_vals
    df["api_in"] = np.round(np.array(api_vals) / MM_PER_INCH, 3)
    df["runoff_coeff_c"] = c_vals
    df["runoff_mm_hr"] = runoff_vals
    df["precip_24hr_rolling_mm"] = p_series.rolling(window=24, min_periods=1).sum().round(3).values
    df["precip_72hr_rolling_mm"] = np.round(rolling_72h_p, 3)
    df["discharge_index_cfs"] = routed_cfs
    df.attrs["final_interflow_storage_mm"] = final_interflow_mm
    return df


def project_forward_hydro_series(
    future_fc_df: pd.DataFrame,
    initial_api_mm: float,
    recent_runoff_history_mm_hr: Optional[List[float]] = None,
    initial_interflow_storage_mm: float = 0.0,
    recent_precip_history_mm: Optional[List[float]] = None,
    base_daily_k: float = API_DAILY_K_DEFAULT,
) -> pd.DataFrame:
    """
    Projects API_t, P90 storm-envelope API_t, C_t, effective runoff, two-pathway
    routed discharge (with continuous interflow handoff), and coincident hourly tiers.
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
        fc.get("mos_qpf_p90_mm", mos_qpf * 1.45), errors="coerce"
    ).fillna(mos_qpf * 1.45).clip(lower=0.0)
    mos_p90 = np.maximum(mos_qpf, mos_p90)

    t_series = pd.to_numeric(
        fc.get("corrected_temp_c", fc.get("nws_temp_c", 15.0)), errors="coerce"
    ).fillna(15.0)
    rh_series = pd.to_numeric(
        fc.get("corrected_rh_pct", fc.get("nws_rh_pct", 55.0)), errors="coerce"
    ).fillna(55.0)

    hist_tail_23 = [max(0.0, float(x)) for x in (recent_precip_history_mm or [])[-23:]]
    hist_tail_71 = [max(0.0, float(x)) for x in (recent_precip_history_mm or [])[-71:]]
    combined_24 = pd.Series(hist_tail_23 + mos_qpf.tolist(), dtype="float64")
    combined_72 = pd.Series(hist_tail_71 + mos_qpf.tolist(), dtype="float64")
    rolling_24h_arr = combined_24.rolling(window=24, min_periods=1).sum().iloc[len(hist_tail_23):].values
    rolling_72h_arr = combined_72.rolling(window=72, min_periods=1).sum().iloc[len(hist_tail_71):].values

    api_exp = max(0.0, float(initial_api_mm or 0.0))
    api_p90 = max(0.0, float(initial_api_mm or 0.0))

    proj_api_mm: List[float] = []
    proj_api_p90_mm: List[float] = []
    proj_c: List[float] = []
    proj_c_p90: List[float] = []
    proj_r_mm: List[float] = []
    proj_r_p90_mm: List[float] = []
    proj_tiers: List[str] = []

    for idx in range(len(fc)):
        q_exp = float(mos_qpf.iloc[idx])
        q_p90 = float(mos_p90.iloc[idx])
        # Storm-scale P90 accumulation uses 35% of hourly P90 excess to avoid compounding P99.99 tails
        q_storm_p90 = q_exp + 0.35 * max(0.0, q_p90 - q_exp)
        k_t = compute_dynamic_decay_k(rh_series.iloc[idx], t_series.iloc[idx], base_daily_k=base_daily_k)

        api_exp = compute_api_step(api_exp, q_exp, k_hr=k_t)
        api_p90 = compute_api_step(api_p90, q_storm_p90, k_hr=k_t)

        c_exp = compute_runoff_coefficient(api_exp, precip_72hr_mm=float(rolling_72h_arr[idx]))
        c_p90 = compute_runoff_coefficient(api_p90, precip_72hr_mm=float(rolling_72h_arr[idx]))

        r_exp = compute_hourly_effective_runoff_mm(q_exp, c_exp)
        # P90 surge runoff evaluates P90 burst intensity over storm-envelope wetness
        r_p90 = compute_hourly_effective_runoff_mm(q_p90, 0.5 * (c_exp + c_p90))

        proj_api_mm.append(api_exp)
        proj_api_p90_mm.append(api_p90)
        proj_c.append(c_exp)
        proj_c_p90.append(c_p90)
        proj_r_mm.append(r_exp)
        proj_r_p90_mm.append(r_p90)

    # Seed last 3 historical runoff hours AND historical interflow storage for seamless continuity
    seed_hist = list((recent_runoff_history_mm_hr or [])[-3:])
    while len(seed_hist) < 3:
        seed_hist.insert(0, 0.0)

    # Route forward only by seeding unit-hydrograph taps from seed_hist without double-adding to interflow
    cfs_per_mm_hr = CATCHMENT_AREA_KM2 * (1000.0 / 3600.0) * 35.3147
    w0, w1, w2, w3 = UNIT_HYDROGRAPH_WEIGHTS

    def _route_forward_with_seed(r_fut: List[float]) -> List[float]:
        full = seed_hist + r_fut
        st_mm = max(0.0, float(initial_interflow_storage_mm or 0.0))
        out_q = []
        for j in range(len(seed_hist), len(full)):
            r0, r1, r2, r3 = full[j], full[j - 1], full[j - 2], full[j - 3]
            q_mm = QUICKFLOW_FRACTION * (w0 * r0 + w1 * r1 + w2 * r2 + w3 * r3)
            st_mm = INTERFLOW_RECESSION_K * st_mm + INTERFLOW_FRACTION * r0
            if_mm = (1.0 - INTERFLOW_RECESSION_K) * st_mm
            out_q.append(round(BASEFLOW_CFS + (q_mm + if_mm) * cfs_per_mm_hr, 1))
        return out_q

    routed_exp = _route_forward_with_seed(proj_r_mm)
    routed_p90 = _route_forward_with_seed(proj_r_p90_mm)

    # Classify coincident hourly tiers AFTER routing so 250 / 500 CFS safeguards apply
    for idx in range(len(fc)):
        hr_code, _ = evaluate_single_hour_tier(
            api_mm=proj_api_mm[idx],
            precip_1hr_mm=float(mos_qpf.iloc[idx]),
            p90_1hr_mm=float(mos_p90.iloc[idx]),
            runoff_mm_hr=proj_r_mm[idx],
            rolling_24h_mm=float(rolling_24h_arr[idx]),
            rolling_72h_mm=float(rolling_72h_arr[idx]),
            allow_red=(idx < 6),
            allow_orange=(idx < 24),
            discharge_cfs=routed_exp[idx],
        )
        proj_tiers.append(hr_code)

    fc["mos_qpf_mm"] = mos_qpf.round(3).values
    fc["mos_qpf_p90_mm"] = np.round(mos_p90, 3)
    fc["rolling_24h_mm"] = np.round(rolling_24h_arr, 3)
    fc["rolling_72h_mm"] = np.round(rolling_72h_arr, 3)
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
    with sqlite3.connect(db_path, timeout=30) as conn:
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
                decay_k REAL NOT NULL DEFAULT 0.9961,
                runoff_coeff_c REAL NOT NULL DEFAULT 0.05,
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
                discharge_index_cfs REAL DEFAULT 6.5,
                peak_discharge_24hr_cfs REAL DEFAULT 6.5,
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
                projected_runoff_coeff REAL DEFAULT 0.05,
                projected_runoff_mm_hr REAL DEFAULT 0.0,
                projected_runoff_p90_mm_hr REAL DEFAULT 0.0,
                projected_discharge_cfs REAL DEFAULT 6.5,
                projected_discharge_p90_cfs REAL DEFAULT 6.5,
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
    """
    hist_hydro_df = spinup_historical_api_series(obs_df, ref_df=ref_hist_df)

    chews_p_now = 0.0
    vent_p_now = 0.0
    bigsur_p_now = 0.0
    if isinstance(ref_dict, dict):
        chews_p_now = float(ref_dict.get("CHWC1", {}).get("precip_1hr_mm") or 0.0) if ref_dict.get("CHWC1", {}).get("online") else 0.0
        vent_p_now = float(ref_dict.get("VTCC1", {}).get("precip_1hr_mm") or 0.0) if ref_dict.get("VTCC1", {}).get("online") else 0.0
        bigsur_p_now = float(ref_dict.get("PPSC1", {}).get("precip_1hr_mm") or 0.0) if ref_dict.get("PPSC1", {}).get("online") else 0.0

    if not hist_hydro_df.empty:
        latest_row = hist_hydro_df.iloc[-1]
        valid_time_utc = str(latest_row["valid_time_utc"])
        tass_p_1hr = float(latest_row.get("precip_1hr_mm") or 0.0)
        hist_eff_p = float(latest_row.get("effective_precip_1hr_mm") or 0.0)
        hist_spill = float(latest_row.get("upstream_spillover_mm") or 0.0)
        live_eff_p, live_spill = compute_effective_catchment_precip_mm(
            tass_p_1hr, chews_p_now, vent_p_now, bigsur_p_now
        )
        eff_p_now = max(hist_eff_p, live_eff_p)
        spill_now = max(hist_spill, live_spill)
        prev_api = float(hist_hydro_df.iloc[-2]["api_mm"]) if len(hist_hydro_df) >= 2 else 0.0
        decay_k = float(latest_row.get("decay_k") or API_HOURLY_K_DEFAULT)
        api_mm = max(float(latest_row["api_mm"]), compute_api_step(prev_api, eff_p_now, k_hr=decay_k))
        api_24h_ago_mm = (
            float(hist_hydro_df.iloc[-25]["api_mm"])
            if len(hist_hydro_df) >= 25
            else float(hist_hydro_df.iloc[0]["api_mm"])
        )
        precip_24hr_mm = float(latest_row.get("precip_24hr_rolling_mm") or 0.0)
        precip_72hr_mm = float(latest_row.get("precip_72hr_rolling_mm") or 0.0)
        runoff_coeff_c = compute_runoff_coefficient(api_mm, precip_72hr_mm=precip_72hr_mm)
        current_runoff_mm_hr = compute_hourly_effective_runoff_mm(eff_p_now, runoff_coeff_c)

        last_idx = hist_hydro_df.index[-1]
        hist_hydro_df.loc[last_idx, "effective_precip_1hr_mm"] = eff_p_now
        hist_hydro_df.loc[last_idx, "upstream_spillover_mm"] = spill_now
        hist_hydro_df.loc[last_idx, "api_mm"] = api_mm
        hist_hydro_df.loc[last_idx, "api_in"] = round(api_mm / MM_PER_INCH, 3)
        hist_hydro_df.loc[last_idx, "runoff_coeff_c"] = runoff_coeff_c
        hist_hydro_df.loc[last_idx, "runoff_mm_hr"] = current_runoff_mm_hr
        routed_hist_cfs, final_interflow_mm = route_runoff_to_discharge_cfs(
            hist_hydro_df["runoff_mm_hr"].tolist(), return_final_storage=True
        )
        hist_hydro_df["discharge_index_cfs"] = routed_hist_cfs
        discharge_cfs = float(routed_hist_cfs[-1])
        recent_runoff_list = hist_hydro_df["runoff_mm_hr"].tail(3).tolist()
    else:
        valid_time_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
        eff_p_now, spill_now = compute_effective_catchment_precip_mm(
            0.0, chews_p_now, vent_p_now, bigsur_p_now
        )
        decay_k = API_HOURLY_K_DEFAULT
        api_mm = eff_p_now
        api_24h_ago_mm = 0.0
        precip_24hr_mm = eff_p_now
        precip_72hr_mm = eff_p_now
        runoff_coeff_c = compute_runoff_coefficient(api_mm, precip_72hr_mm=precip_72hr_mm)
        current_runoff_mm_hr = compute_hourly_effective_runoff_mm(eff_p_now, runoff_coeff_c)
        discharge_cfs = BASEFLOW_CFS
        final_interflow_mm = 0.0
        recent_runoff_list = [current_runoff_mm_hr]

    recent_precip_list = (
        pd.to_numeric(hist_hydro_df.tail(71)["precip_1hr_mm"], errors="coerce").fillna(0.0).tolist()
        if not hist_hydro_df.empty and "precip_1hr_mm" in hist_hydro_df.columns
        else []
    )

    # Strictly route hours AFTER the last observed hour so `now` is never double-counted
    fc_strict_future = future_fc_df
    if (
        not hist_hydro_df.empty
        and future_fc_df is not None
        and not future_fc_df.empty
        and "valid_time_utc" in future_fc_df.columns
    ):
        last_obs_dt = pd.to_datetime(valid_time_utc, utc=True).floor("h")
        fc_dt = pd.to_datetime(future_fc_df["valid_time_utc"], utc=True, errors="coerce").dt.floor("h")
        fc_strict_future = future_fc_df.loc[fc_dt > last_obs_dt].copy()

    proj_fc_df = project_forward_hydro_series(
        future_fc_df=fc_strict_future,
        initial_api_mm=api_mm,
        recent_runoff_history_mm_hr=recent_runoff_list,
        initial_interflow_storage_mm=final_interflow_mm,
        recent_precip_history_mm=recent_precip_list,
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
        current_runoff_mm_hr=current_runoff_mm_hr,
        proj_fc_df=proj_fc_df,
        current_discharge_cfs=discharge_cfs,
        stream_stage_m=stage_m,
        rate_of_rise_m_hr=rise_m_hr,
    )

    state_summary = {
        "valid_time_utc": valid_time_utc,
        "api_mm": round(api_mm, 2),
        "api_in": round(api_mm / MM_PER_INCH, 2),
        "api_24h_ago_in": round(api_24h_ago_mm / MM_PER_INCH, 2),
        "projected_max_api_24h_mm": round(projected_max_api_24h_mm, 2),
        "projected_max_api_24h_in": round(projected_max_api_24h_mm / MM_PER_INCH, 2),
        "projected_max_api_72h_mm": round(projected_max_api_72h_mm, 2),
        "projected_max_api_72h_in": round(projected_max_api_72h_mm / MM_PER_INCH, 2),
        "decay_k": round(decay_k, 5),
        "decay_k_daily_equiv": round(decay_k ** 24, 3),
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
    """Pipeline entry point invoked every hour by `run_hourly_mos_pipeline.py`."""
    ensure_hydro_schema(db_path)

    with sqlite3.connect(db_path, timeout=30) as conn:
        obs_df = pd.read_sql_query(
            """
            SELECT * FROM local_observations
            WHERE temp_c IS NOT NULL
            ORDER BY valid_time_utc DESC
            LIMIT 720
            """,
            conn,
        )
        ref_hist_df = pd.read_sql_query(
            """
            SELECT * FROM reference_observations
            ORDER BY valid_time_utc DESC
            LIMIT 3600
            """,
            conn,
        )

        ref_dict: Dict[str, Any] = {}
        anchor_dt = (
            pd.to_datetime(obs_df.iloc[0]["valid_time_utc"], utc=True)
            if not obs_df.empty
            else pd.Timestamp(datetime.now(timezone.utc))
        )
        if not ref_hist_df.empty:
            r_live = ref_hist_df.copy()
            if "source" in r_live.columns:
                r_live = r_live[~r_live["source"].isin(["OPENMETEO_SYNTH", "OFFLINE"])].copy()
            r_live["_dt"] = pd.to_datetime(r_live["valid_time_utc"], utc=True, errors="coerce")
            r_live = r_live.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)
            for stid, grp in r_live.groupby("station_id"):
                row_d = grp.iloc[0].to_dict()
                row_d["online"] = bool(abs((anchor_dt - row_d["_dt"]).total_seconds()) / 3600.0 <= 3.5)
                ref_dict[str(stid)] = row_d

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

        # Persist ONLY genuine observed historical hours (skip synthetic gap-filled slots)
        if not hist_df.empty:
            recent_hist_rows = []
            for _, h_row in hist_df.tail(48).iterrows():
                if not bool(h_row.get("is_observed_hour", True)):
                    continue
                h_ts = str(h_row["valid_time_utc"])
                if h_ts == s["valid_time_utc"]:
                    continue
                h_meta = evaluate_flood_and_road_tier(
                    api_mm=float(h_row["api_mm"]),
                    precip_1hr_mm=float(h_row["effective_precip_1hr_mm"]),
                    precip_24hr_mm=float(h_row["precip_24hr_rolling_mm"]),
                    precip_72hr_mm=float(h_row["precip_72hr_rolling_mm"]),
                    current_runoff_mm_hr=float(h_row["runoff_mm_hr"]),
                    current_discharge_cfs=float(h_row["discharge_index_cfs"]),
                )
                recent_hist_rows.append((
                    h_ts,
                    round(float(h_row["api_mm"]), 3),
                    round(float(h_row["api_in"]), 3),
                    round(float(h_row["decay_k"]), 5),
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
                older_rows = recent_hist_rows[:-6] if len(recent_hist_rows) > 6 else []
                trailing_rows = recent_hist_rows[-6:]
                if older_rows:
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
                        older_rows,
                    )
                conn.executemany(
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
                    trailing_rows,
                )

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

        if not proj_df.empty:
            fc_rows = [
                (
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
                )
                for _, r in proj_df.iterrows()
            ]
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
        f"API={s['api_in']:.2f} in ({s['api_mm']:.1f} mm, k_day={s['decay_k_daily_equiv']:.3f}), "
        f"C={s['runoff_coeff_c']:.2f}, "
        f"24h_Peak_Discharge={s['peak_discharge_24hr_cfs']:.1f} CFS | {s['trigger_summary']}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = evaluate_and_store_hydro_risk()
    print("Evaluated Hydrological Risk Summary:", res["summary"])