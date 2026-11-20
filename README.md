# Tassajara Zen Mountain Center (TZMC) — Microclimate & Hydrological MOS Forecast Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](https://tzmc-weather.streamlit.app)
[![Database](https://img.shields.io/badge/Database-SQLite%20WAL-003B57.svg)](https://sqlite.org)
[![Model Architecture](https://img.shields.io/badge/MOS%20Engine-30--Feature%20GBDT%20%2B%20HAC%20OLS-orange.svg)](#5-mos-machine-learning-architecture)
[![Hydrology](https://img.shields.io/badge/Hydrology-Two--Pathway%20API%20%2B%20CFS%20Router-0284C7.svg)](#6-early-flood--road-washout-warning-system)
[![Offline Ready](https://img.shields.io/badge/Deployment-Cloud%20%2B%20Local%20LAN%20Resilient-16A34A.svg)](#9-local-on-premises-deployment-hardware-selection--offline-resilience)

An operational, publication-grade meteorological **Model Output Statistics (MOS)** forecast suite and **Early Flood & Road Washout Warning System** custom-engineered for the **Tassajara Zen Mountain Center (TZMC)** in the Santa Lucia Mountains of Monterey County, California.

This README serves as both the scientific reference and the complete **Maintainer Handover Manual**, covering cloud automation, database operations, model retraining, and dedicated local Mini PC / Raspberry Pi deployment for uninterrupted storm-mode operation if internet connectivity drops.

---

## Table of Contents
1. [Geographic & Physical Context](#1-geographic--physical-context)
2. [System Architecture & Repository Map](#2-system-architecture--repository-map)
3. [Santa Lucia 5-Station Boundary Array & Fallback Cascade](#3-santa-lucia-5-station-boundary-array--fallback-cascade)
4. [Data Integrity, Provenance & Rainfall Accounting](#4-data-integrity-provenance--rainfall-accounting)
5. [MOS Machine Learning Architecture](#5-mos-machine-learning-architecture)
6. [Early Flood & Road Washout Warning System](#6-early-flood--road-washout-warning-system)
7. [Database Schema & Production Views](#7-database-schema--production-views)
8. [Maintainer Handover: Setup, Secrets & Cloud Operations](#8-maintainer-handover-setup-secrets--cloud-operations)
9. [Local On-Premises Deployment: Hardware Selection & Offline Resilience](#9-local-on-premises-deployment-hardware-selection--offline-resilience)
10. [Streamlit Dashboard UI](#10-streamlit-dashboard-ui)
11. [Hardware Roadmap & Pending Physical Stream Telemetry](#11-hardware-roadmap--pending-physical-stream-telemetry)
12. [Future Directions & Methodological Improvements](#12-future-directions--methodological-improvements)
13. [License & Operational Disclaimer](#13-license--operational-disclaimer)

---

## 1. Geographic & Physical Context

Tassajara Zen Mountain Center sits deep along the canyon floor of Tassajara Creek at **$485\text{ m}$ ($1,590\text{ ft}$) MSL** ($36.2337^\circ\text{N},\, 121.5501^\circ\text{W}$), surrounded by precipitous ridges rising above **$1,500\text{ m}$ ($5,000\text{ ft}$)** (Chews Ridge, Pine Ridge, Black Cone, and Ventana Cone). Access is governed exclusively by a rugged, unpaved 14-mile mountain road crossing the $1,536\text{ m}$ Chews Ridge summit from Jamesburg.

```text
Elevation (m MSL)
1600m ─── [CHWC1: Chews Ridge 1536m] ────────── [VTCC1: Ventana Cone 1453m (Rain Gauge)]
          \                                    /
1200m ─────\────── High-Gradient Mountain Tributaries ──────/
            \                                              /
800m  ───────\── Operational NWS/NWP Grid Reference Elevation (~880m MSL) ────
              \                                          /
485m  ─────────\── [TZMC Stone Office Davis VP2 485m] ──/ (Deep Canyon Cold-Pool)
                \                                      /
200m  ───────────\── [ASRC1: Arroyo Seco 268m] ─────── [PPSC1: Big Sur 108m Coast]
```

### Microclimatic & Hydrological Complexities
* **Topographic Shading:** The southern canyon wall (Black Cone Trail ridge, $32^\circ$ horizon) casts an early afternoon solar shadow, accelerating radiational cooling hours before astronomical sunset.
* **Nocturnal Katabatic Drainage:** Cool air pooling along Tassajara Creek decouples the canyon floor from the free atmosphere, pushing overnight lows $5^\circ\text{F}$ to $15^\circ\text{F}$ colder than regional model grids.
* **Orographic Spillover & Feeder-Seeder Rain:** Pacific atmospheric rivers intercept the coastal Santa Lucia crest, wringing out intense headwater rainfall that spills into upper Tassajara Creek tributaries while the valley floor sits in partial rain-shadow.
* **Flash Floods & Road Slipouts:** Multi-inch storms over steep, fractured granite and clay-rich mountain slopes trigger rapid creek surges, woody debris damming at lower footbridges, and hazardous slipouts along Tassajara Road.

---

## 2. System Architecture & Repository Map

The suite runs on an automated hourly cycle (`:05` past each hour via GitHub Actions and/or a local `systemd` timer), pulling Stone Office telemetry, 5-station boundary observations, and NWS `/gridpoints/MTR` hourly forecasts; running MOS inference and two-pathway hydrological routing; and persisting everything to a WAL-mode SQLite database consumed by the Streamlit UI.

```text
                              ┌───────────────────────────────────────────────┐
                              │  Davis WeatherLink v2 API (Station 36966)     │
                              │  /v2/current (WL_LIVE) + 48h /v2/historic     │
                              │  (Optional LAN fallback: WeatherLink Live IP) │
                              └───────────────────────┬───────────────────────┘
                                                      │
┌───────────────────────────────────────┐             │             ┌──────────────────────────────────────────┐
│ 5-Station Santa Lucia Boundary Array  │             │             │ NWS Monterey /gridpoints/MTR Forecast    │
│ Tier 1: Synoptic API (optional)       │             │             │ 168-hr hourly NWP grid (~880m MSL)       │
│ Tier 2A: NOAA/NWS Obs (NWS_OBS)       │             │             │ (Open-Meteo top-up if NWS grid drops)    │
│ Tier 2B: Free IEM CA_DCP (CHWC1/VTCC1)│             │             └────────────────────┬─────────────────────┘
│ Tier 3: Open-Meteo (flagged fallback) │             │                                  │
└──────────────────┬────────────────────┘             │                                  │
                   │                                  │                                  │
                   ▼                                  ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                run_hourly_mos_pipeline.py (Hourly Orchestrator)                              │
│  - 48-hr WeatherLink Pro WL_HISTORIC catch-up + monotonic rain_year_mm delta tracking (ingest_state)         │
│  - Monotonic IEM precip_today[in] delta tracking for crest ALERT gauges (CHWC1, VTCC1)                       │
│  - Boundary-layer diagnostics (crest lapse rate, gorge confinement, virga VPD, barometric tendency)          │
│  - 30-predictor feature engineering (astronomical, topographic, thermodynamic, and spatial gradients)        │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       mos_model.py (MOS_GBM_v3_HYDRO — TassajaraMOSCorrector Ensemble)                       │
│  1. Thermal Subsystem: HistGradientBoostingRegressor (Mean + P10/P90 Conformal Quantile Envelope)            │
│  2. Hydrological Hurdle: Calibrated PoP Classifier + Stage-2 Wet-Only Poisson QPF & P90 Rate Regressors      │
│  3. Econometric Baseline: Standardized Newey-West HAC OLS Regression (cov_type='HAC', maxlags=12)            │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         hydro_risk_engine.py (Early Flood & Road Washout Warning System)                     │
│  - Dynamic Antecedent Precipitation Index: API_t = API_{t-1} * k + P_eff (k in [0.88, 0.94] via VPD)         │
│  - Nonlinear Sigmoid Runoff Partitioning C(API) in [0.08, 0.86]; filters out OPENMETEO_SYNTH spillover       │
│  - Mass-Conserving Two-Pathway Hydrograph (72 km^2): Quickflow + Shallow Subsurface Interflow (k=0.955)      │
│  - 4-Tier Severity Matrix: GREEN, YELLOW, ORANGE (>= 250 CFS safeguard), RED (>= 500 CFS safeguard)          │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SQLite WAL Database (tassajara_weather.db)                                 │
│  Tables: local_observations, reference_observations, nws_forecasts, mos_corrected_forecasts,                 │
│          hydrological_risk_state, hydrological_forecast_series, stream_gauge_observations, ingest_state      │
│  Views:  vw_mos_training_dataset (deduplicated genuine Davis rows), vw_creek_flood_monitor                   │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          app.py (Streamlit Web Dashboard)                                    │
│  Mode 1: Everyday Forecast (Community & Guest View — plain-language cards, road/creek alerts, freeze lows)   │
│  Mode 2: Scientific & Statistical MOS Suite (8-metric grid, 3-panel Plotly, verification scorecard, OLS)     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Repository File Map

| File / Path | Responsibility |
| :--- | :--- |
| `ingest_weather.py` | WeatherLink v2 API client (Station `36966`), NWS `/gridpoints/MTR` fetcher, and 4-tier 5-station reference fetcher (Synoptic $\rightarrow$ NWS $\rightarrow$ IEM `CA_DCP` $\rightarrow$ Open-Meteo). |
| `run_hourly_mos_pipeline.py` | Main hourly orchestrator: fetches 48h `WL_HISTORIC` + live `WL_LIVE`, tracks monotonic `rain_year_mm`, syncs 5 reference stations, generates 168h `MOS_GBM_v3_HYDRO` forecasts, and runs `hydro_risk_engine.py`. |
| `mos_model.py` | Defines `TassajaraMOSCorrector`, 30-feature engineering, lead-time anomaly decay, purged 5-fold CV, Stage-1/Stage-2 hydrological hurdle, and Newey-West HAC OLS diagnostics. |
| `hydro_risk_engine.py` | Antecedent Precipitation Index ($API$), logistic runoff coefficient ($C$), two-pathway quickflow/interflow creek router (CFS), 4-tier severity matrix, and trigger deduplication. |
| `backfill_and_train.py` | Historical WeatherLink Pro backfill, model training on `vw_mos_training_dataset`, historical verification table population, and artifact serialization to `models/tassajara_mos_v1.joblib`. |
| `app.py` | Dual-mode Streamlit dashboard (`Everyday Forecast` & `Scientific MOS Suite`), live SQLite + API hydration, interactive Plotly visualizations, and paired verification scorecard. |
| `schema.sql` | SQLite WAL schema, indexes, and production views (`vw_mos_training_dataset`, `vw_creek_flood_monitor`). |
| `models/tassajara_mos_v1.joblib` | Serialized production `MOS_GBM_v3_HYDRO` ensemble artifact trained on $17,570$ verified Davis hours. |
| `tassajara_weather.db` | Production SQLite database (WAL mode) storing observations, forecasts, hydro states, and monotonic `ingest_state` counters. |
| `.github/workflows/` | Automated GitHub Actions workflows for hourly ingestion/inference (`hourly_weather.yml`) and scheduled weekly model retraining. |

---

## 3. Santa Lucia 5-Station Boundary Array & Fallback Cascade

The pipeline monitors five regional stations surrounding the Tassajara watershed to resolve vertical thermal stratification, coastal marine intrusion, and headwater rainfall:

| Station ID | Station Name | Elev (m / ft) | Primary Live Source | Sensor Suite | Boundary Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CHWC1`** | Chews Ridge | $1,536\text{ m}$ ($5,039\text{ ft}$) | `IEM_CA_DCP` / `SYNOPTIC` | Temp, Rain (ALERT/RAWS) | **Interior Crest (NNW):** Primary crest lapse rate ($^\circ\text{C/km}$) and summit freeze monitoring. |
| **`VTCC1`** | Ventana Cone | $1,453\text{ m}$ ($4,767\text{ ft}$) | `IEM_CA_DCP` | **Precipitation-Only** (MPWMD ALERT) | **Ventana Crest (WNW):** Windward headwater tipping-bucket gauge (`Online (Rain Gauge)`). |
| **`CAHC1`** | Hastings Reserve | $546\text{ m}$ ($1,791\text{ ft}$) | `NWS_OBS` / `IEM_CA_DCP` | Temp, RH, Wind, Rain | **Foothill Valley (N):** Regional inland valley thermal differential ($\Delta T$ in $^\circ\text{F}$). |
| **`ASRC1`** | Arroyo Seco | $268\text{ m}$ ($879\text{ ft}$) | `NWS_OBS` / `IEM_CA_DCP` | Temp, RH, Wind, Rain | **Gorge Outlet (6.3 km E):** Down-canyon drainage outlet and cold-pool confinement contrast. |
| **`PPSC1`** | Big Sur State Park | $108\text{ m}$ ($354\text{ ft}$) | `NWS_OBS` / `IEM_CA_DCP` | Temp, RH, Wind, Rain | **Coastal Slope (W):** Marine layer inversion cap and coastal-to-canyon thermal gradient. |

### Four-Tier Zero-Cost Resilient Ingestion Cascade
To guarantee $5/5$ live telemetry without requiring a paid Synoptic subscription, `fetch_regional_reference_stations()` cascades automatically:
1. **Tier 1 — Synoptic Data API (`SYNOPTIC`):** Attempted first if `SYNOPTIC_TOKEN` has active Weather Data API access.
2. **Tier 2A — NOAA/NWS Observations API (`NWS_OBS`):** Free, no-key government endpoint providing live observations for `CAHC1`, `ASRC1`, and `PPSC1`.
3. **Tier 2B — Iowa Environmental Mesonet (`IEM_CA_DCP`):** Free, no-key public RAWS/HADS/ALERT JSON feed (`https://mesonet.agron.iastate.edu/json/current.py?network=CA_DCP`) providing live observations for `CHWC1` and precipitation-only `VTCC1`. Computes RH from `dewpointtemp[F]` via the Magnus equation and tracks monotonic hourly rainfall increments from `precip_today[in]`.
4. **Tier 3 — Open-Meteo Real-Time Boundary Analysis (`OPENMETEO_SYNTH`):** Used **only** if a station fails all three live feeds. Marked `online=False` (`Model Fallback / Stale`, flagged with `*` in the UI) and **strictly excluded** from hydrological headwater spillover so synthetic model precipitation never fabricates creek runoff.

### Special Handling for Ventana Cone (`VTCC1`)
`VTCC1` is a Monterey County / MPWMD ALERT precipitation-only station and carries **no physical thermometer**. The pipeline and UI explicitly preserve its live status (`Online (Rain Gauge)`, `Temp: N/A`), protect it from Tier 3 Open-Meteo overwrites, track its live daily tipping-bucket accumulation, and prevent synthetic temperatures from contaminating the `Crest Lapse Rate` card (which reports genuine `CHWC1` lapse rates when `VTCC1` has no thermistor).

---

## 4. Data Integrity, Provenance & Rainfall Accounting

Production storm auditing uncovered and permanently eliminated several subtle telemetry failure modes:

* **Zero Midnight Rain Double-Counting (Monotonic `rain_year_mm` Tracking):** Davis console `rain_day_mm` counters can drift across local midnight (`00:00 America/Los_Angeles`), previously causing unreset prior-day totals to dump into `01:00 PST` as a phantom single-hour cloudburst. Both `run_hourly_mos_pipeline.py` and `app.py` now compute incremental rainfall using the console's monotonic **`rain_year_mm`** counter persisted in `ingest_state` (falling back to monotonic `rain_day_mm` only when `console_day_mm < prev_day_mm`), paired with 48-hour WeatherLink Pro `/v2/historic` (`WL_HISTORIC`) reconciliation.
* **Strict Observation Provenance & Tipping-Bucket Fingerprint:** Every row in `local_observations` carries an explicit `source` tag (`WL_HISTORIC` or `WL_LIVE`). Genuine Davis Vantage Pro2 hourly rainfall totals always fall on exact $0.254\text{ mm}$ ($0.01\text{ in}$) tipping-bucket multiples, allowing immediate audit verification against any synthetic $0.1\text{ mm}$ model data (`OPENMETEO_SYNTH`). Synthetic rows are excluded from `vw_mos_training_dataset` and archive rain statistics.
* **Verification Deduplication:** Historical MOS verification joins (`mos_corrected_forecasts`) use `ROW_NUMBER() OVER (PARTITION BY valid_time_utc, issue_time_utc ORDER BY model_version DESC)` plus `valid_time_utc` deduplication in `app.py`, ensuring exact $72 / 72$ paired evaluation hours in the 72-hour verification scorecard.

---

## 5. MOS Machine Learning Architecture

The production statistical engine (`mos_model.py`, version **`MOS_GBM_v3_HYDRO`**) is trained on **$17,570$ verified Davis Vantage Pro2 hourly pairs** spanning 2 full annual cycles (**$870$ observed wet hours**, **$51.62\text{ in}$** verified tipping-bucket rainfall) with **30 physical predictors**.

### Verified Production Benchmarks (Full 730-Day Purged 5-Fold Out-of-Fold CV)
* **Raw Generic NWP Forecast RMSE:** $2.715^\circ\text{C}$ ($4.89^\circ\text{F}$)
* **Tassajara MOS Corrected RMSE:** **$2.024^\circ\text{C}$ ($3.64^\circ\text{F}$)**
* **Full-Cycle Thermal Skill Improvement:** **$+25.5\%$ RMSE reduction**
* **HAC OLS Diagnostic Baseline ($n = 17,570$):** $R^2 = 0.419$, $\text{Adj } R^2 = 0.418$, matrix condition number $\kappa < 30$ (well-conditioned, zero severe multicollinearity).

### Subsystem Design
1. **Thermal Subsystem (`HistGradientBoostingRegressor`):**
   * Mean error corrector (`loss='squared_error'`, L2-regularized) paired with dual quantile estimators ($\alpha = 0.10$ and $\alpha = 0.90$) generating conformal-calibrated 80% uncertainty bands ($P_{10}$ cold-pool floor to $P_{90}$ daytime ceiling).
   * Issue-time local anomalies (temperature persistence, lapse-rate anomaly, gorge confinement, and barometric pressure tendency) decay exponentially with forecast lead time:
     $$\Gamma_{\text{eff}}(h) = 6.5 + (\Gamma_0 - 6.5)\cdot e^{-h / 12.0}$$
2. **Two-Stage Hydrological Hurdle Subsystem:**
   * **Stage 1 (Calibrated PoP Classifier):** `CalibratedClassifierCV` wrapping `HistGradientBoostingClassifier` estimating $P(\text{Rain} \ge 0.254\text{ mm})$.
   * **Stage 2 (Wet-Only QPF & $P_{90}$ Regressors):** Trained strictly on verified wet hours ($\ge 0.254\text{ mm}$) to prevent dry-hour zero-inflation from suppressing heavy Santa Lucia rain rates.
   * **Virga & Orographic Modulation:** Modulates pre-frontal QPF using boundary-layer Vapor Pressure Deficit (VPD) and upslope moisture flux:
     $$F_{\text{virga}} = \exp(-0.48 \cdot \text{VPD}), \quad F_{\text{oro}} = 1.0 + 0.08 \cdot \text{Flux}_{\text{upslope}} + 0.05 \cdot (-\Delta P_{3\text{h}})$$
3. **Econometric HAC OLS Baseline:**
   * Standardized predictors evaluated via Ordinary Least Squares with 12-hour lag Newey-West Heteroskedasticity and Autocorrelation Consistent standard errors (`cov_type='HAC', maxlags=12`), exposed directly in the Scientific UI for ongoing physical interpretability.

---

## 6. Early Flood & Road Washout Warning System

Implemented in `hydro_risk_engine.py`, this module tracks continuous catchment soil saturation and routes runoff across the $72\text{ km}^2$ upper Tassajara Creek basin (Pine Ridge, Black Cone, Chews Ridge $\rightarrow$ Stone Office at $485\text{ m}$ MSL).

### 1. Dynamic Antecedent Precipitation Index ($API$)
$$API_t = API_{t-1} \cdot k_{\text{hr}} + P_{\text{eff}, t}$$
* **Evaporative Recession ($k_{\text{day}}$):** Modulated dynamically between $0.88$ and $0.94$ from observed Vapor Pressure Deficit (VPD) so saturated winter storms drain slower than dry autumn intervals:
  $$k_{\text{day}} = \text{clip}\!\left(0.935 - 0.025 \cdot \min(\text{VPD},\, 2.2),\, 0.88,\, 0.94\right)$$
* **Effective Catchment Rain ($P_{\text{eff}}$):** Blends Stone Office tipping-bucket rain with **online-only** ridge stations (`CHWC1`, `VTCC1`, `PPSC1`), explicitly ignoring `OPENMETEO_SYNTH` / `OFFLINE` fallback rows.

### 2. Nonlinear Soil-Saturation Runoff Coefficient ($C$)
$$C(API) = C_{\min} + \frac{C_{\max} - C_{\min}}{1 + \exp\!\left(-\dfrac{API + 0.18 \cdot P_{72\text{h}} - API_{\text{mid}}}{s}\right)}$$
Where $C_{\min} = 0.08$ (dry fractured granite / duff abstraction) and $C_{\max} = 0.86$ (saturated bedrock/regolith sheetflow ceiling).

### 3. Mass-Conserving Two-Pathway Creek Hydrograph (CFS)
Rather than snapping abruptly back to dry baseflow after rain pauses, effective runoff ($R_t = C_t \cdot P_{\text{eff}, t}$) over the $72\text{ km}^2$ catchment is partitioned across two physical pathways plus dry-weather spring baseflow ($Q_{\text{base}} = 6.5\text{ CFS}$):
* **Fast Surface Quickflow ($18\%$ of net runoff):** Routed through a 4-hour steep-canyon unit hydrograph kernel to resolve acute storm-band surge peaks.
* **Shallow Subsurface Interflow Reservoir ($57\%$ of net runoff, hourly recession $k_{\text{inter}} = 0.955$):** Sustains realistic elevated post-storm creek flow across $24\text{--}72\text{ hours}$ and transitions seamlessly across the historical-to-forecast `now` divider with zero step discontinuity.

> **Ground-Truth Calibration Note:** Until a physical ultrasonic/radar stage gauge is commissioned at Tassajara Creek, routed `CFS` is a **catchment-scaled hydrological index**, not gauged USGS streamflow. After visually inspecting Tassajara Creek in daylight during a storm, a maintainer can tune `API_MIDPOINT_MM` or `INTERFLOW_FRACTION` in `hydro_risk_engine.py` to anchor modeled CFS directly to observed footbridge clearance.

### 4. Four-Tier Operational Severity Matrix

| Tier | Status Code | Soil Saturation, Rainfall & Routed Discharge Triggers | Tassajara Road Status (14-Mi Unpaved) | Tassajara Creek & Footbridges | Operational Protocol |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **0** | **GREEN** | $API < 2.00"$, $1\text{h} < 0.15"\text{/hr}$, $24\text{h} < 1.00"$, $72\text{h} < 1.50"$, $Q < 250\text{ CFS}$ | **Open / Dry to Damp**<br>Standard mountain precautions | **Stable Baseflow**<br>Footbridges clear and well above water | Routine canyon operations and normal high-clearance travel. |
| **1** | **YELLOW** | $API \ge 2.00"$, OR imminent ($\le 6\text{h}$) $1\text{h} \ge 0.15"\text{/hr}$, OR $24\text{h} \ge 1.00"$, OR $72\text{h} \ge 1.50"$ | **Advisory / Caution**<br>Muddy, slick clay summit grades; minor rockfall risk | **Elevated Flow**<br>Turbid water; moderate creek rise | 4WD required on Tassajara Road. Postpone low-clearance trips; monitor footbridges and culverts. |
| **2** | **ORANGE** | Within $24\text{h}$: $1\text{h} \ge 0.50"\text{/hr}$ on primed soil ($API \ge 2.00"$), OR heavy multi-day rain on primed soil, **OR Routed Creek $Q \ge 250\text{ CFS}$** | **Hazardous / Watch**<br>Travel strongly discouraged (slipout, rutting & rockslide danger) | **Rapidly Rising**<br>High-velocity flow; woody debris mobilization likely | Suspend routine travel on Tassajara Road. Inspect lower footbridges for log damming; secure creek equipment; stage emergency radio net. |
| **3** | **RED** | Imminent/active ($\le 6\text{h}$): $1\text{h} \ge 1.00"\text{/hr}$ (or $P_{90} \ge 1.25"\text{/hr}$), **OR Routed Creek $Q \ge 500\text{ CFS}$** (or physical stage/rise trigger when installed) | **CLOSED / IMPASSABLE**<br>Severe washouts, debris flows, and road cuts | **BANKFULL / FLOOD STAGE**<br>Footbridge overtopping & structural surge hazard | **Immediate road closure.** Evacuate low creek-side structures; prohibit all bridge crossings; notify Monterey County OES as needed. |

---

## 7. Database Schema & Production Views

`tassajara_weather.db` runs in SQLite **Write-Ahead Logging (`WAL`)** mode (`PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`) so background ingestion scripts and the Streamlit web UI can read and write concurrently without lock timeouts.

### Core Tables
* **`local_observations`:** Hourly Stone Office Davis Vantage Pro2 records (Station `36966`) with `source` provenance (`WL_HISTORIC`, `WL_LIVE`).
* **`reference_observations`:** Synchronized hourly records for `CHWC1`, `VTCC1`, `CAHC1`, `ASRC1`, `PPSC1` with `source` (`SYNOPTIC`, `NWS_OBS`, `IEM_CA_DCP`, `OPENMETEO_SYNTH`).
* **`nws_forecasts`:** 168-hour NWS `/gridpoints/MTR` forecast batches stamped by `issue_time_utc` and `valid_time_utc`.
* **`mos_corrected_forecasts`:** Operational and historical MOS predictions (`corrected_temp_c`, `P10`, `P90`, `mos_pop_pct`, `mos_qpf_1hr_mm`, `mos_qpf_p90_mm`, `model_version`).
* **`hydrological_risk_state`:** Historical hourly $API$, $C$, effective runoff, routed discharge (CFS), and alert tier records.
* **`hydrological_forecast_series`:** Forward 72-hour projected hydrograph, $API$, and $P_{90}$ surge envelope.
* **`ingest_state`:** Key-value state table persisting monotonic console counters (`last_console_rain_year_mm`, `last_console_rain_day_mm`, `last_ref_rain_day_mm_<STID>`) to guarantee zero double-counting across restarts and midnight rollovers.
* **`stream_gauge_observations`:** Pre-provisioned table awaiting physical creek stage sensor commissioning.

### Production Views
* **`vw_mos_training_dataset`:** Deduplicated training view pairing genuine Davis observations (`source != 'OPENMETEO_SYNTH'`) with aligned NWP forecasts and issue-time spatial boundary predictors.
* **`vw_creek_flood_monitor`:** Unified operational snapshot uniting live Stone Office meteorology, soil saturation, routed creek discharge, and active alert tier.

---

## 8. Maintainer Handover: Setup, Secrets & Cloud Operations

Everything required for a new technical steward to take over day-to-day operations is listed below.

### 1. Required Credentials & Where They Live
**Never commit live API keys or secrets into Git.** Configure credentials in three places (only WeatherLink keys are strictly required; Synoptic is optional thanks to free `NWS_OBS` + `IEM_CA_DCP`):

| Secret Name | Required? | Source / Purpose |
| :--- | :---: | :--- |
| `WL_API_KEY` | **Required** | Davis WeatherLink v2 API Key (`weatherlink.com` $\rightarrow$ Account $\rightarrow$ API v2) for Station `36966` (Device Tier: Pro). |
| `WL_API_SECRET` | **Required** | Davis WeatherLink v2 API Secret (used to compute HMAC-SHA256 / v2 request headers). |
| `SYNOPTIC_TOKEN` | Optional | Synoptic Data Public Token (`customer.synopticdata.com`). Leave unset or expired without issue—the pipeline seamlessly uses free `NWS_OBS` and `IEM_CA_DCP`. |

Where to configure them:
1. **GitHub Actions:** Repository $\rightarrow$ **Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions** $\rightarrow$ add `WL_API_KEY`, `WL_API_SECRET` (and optional `SYNOPTIC_TOKEN`).
2. **Streamlit Community Cloud:** `share.streamlit.io` $\rightarrow$ App $\rightarrow$ **Settings** $\rightarrow$ **Secrets** (TOML format):
   ```toml
   WL_API_KEY = "your_weatherlink_v2_api_key"
   WL_API_SECRET = "your_weatherlink_v2_api_secret"
   SYNOPTIC_TOKEN = "optional_synoptic_token"
   ```
3. **Local Workstation / On-Premises Server:** Export in shell or store in a protected `.env` file (`chmod 600 .env`).

### 2. Local Development Setup
```bash
git clone https://github.com/MT-Wildhack/tassajara-weather.git
cd tassajara-weather

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Initialize a fresh database only if starting from scratch (otherwise use existing tassajara_weather.db):
# sqlite3 tassajara_weather.db < schema.sql
```

### 3. Essential Operational Commands
```bash
# Run full hourly ingestion, MOS inference, and hydrological risk update:
python3 run_hourly_mos_pipeline.py

# Re-evaluate hydrological risk & 72h hydrograph independently from existing SQLite data:
python3 hydro_risk_engine.py

# Retrain MOS model on all genuine verified Davis data already in SQLite (no API backfill wait):
python3 backfill_and_train.py --skip-fetch

# Full 730-day WeatherLink Pro backfill + retrain (slow; run only for initial rebuilds):
python3 backfill_and_train.py --days 730

# Launch Streamlit dashboard locally:
streamlit run app.py
```

### 4. Handling Hourly GitHub Actions & SQLite Merge Conflicts
Because `.github/workflows/hourly_weather.yml` runs at `:05` past every hour and commits updated `tassajara_weather.db` snapshots to `main`, pushing manual local edits to `tassajara_weather.db` can occasionally trigger a binary merge conflict:
* **If you repaired or retrained `tassajara_weather.db` locally:** In GitHub Desktop (or Git CLI via `git checkout --ours tassajara_weather.db`), select **Use the modified (local) file**, commit the merge, and push.
* **If you only edited Python code (`app.py`, `hydro_risk_engine.py`, etc.):** Keep `tassajara_weather.db` unchecked locally (or `git checkout --theirs tassajara_weather.db`) so you preserve the latest hourly GitHub Actions observations.
* **Keep local scratch files out of Git:** Never stage `.DS_Store`, `__pycache__/`, `tassajara_weather.db-wal`, `tassajara_weather.db-shm`, or local `.env` files.

### 5. Rebooting Streamlit Community Cloud After Model or Code Updates
Streamlit Cloud caches Python modules and `.joblib` models in RAM. After pushing changes to `models/tassajara_mos_v1.joblib`, `hydro_risk_engine.py`, or `ingest_weather.py`:
1. Open `https://tzmc-weather.streamlit.app`.
2. Click **`< Manage app`** in the bottom-right corner.
3. Click the **three-dot menu (`⋮`)** in the top-right of the management drawer and select **Reboot app**.
4. Wait $30\text{--}60\text{ seconds}$ and confirm the sidebar displays the updated model sample size (`n=17570+`), `Archive Wet Hours`, and live `5/5` reference status.

---

## 9. Local On-Premises Deployment: Hardware Selection & Offline Resilience

During severe Santa Lucia atmospheric rivers—**the exact moments when flood and road-washout guidance matters most**—Starlink, microwave backhaul, and grid power at Tassajara are most likely to fail simultaneously. Running a dedicated local node inside the Stone Office converts the suite from a cloud-dependent website into an **offline-resilient, campus-wide emergency warning appliance**.

### 1. Selecting the Optimal Hardware (Mini PC vs. Raspberry Pi)

| Platform | Recommended Spec | Typical Cost | Power Draw | Pros & Cons for Tassajara Stone Office |
| :--- | :--- | :---: | :---: | :--- |
| **Fanless x86 Mini PC** *(Recommended Primary)* | Intel **N100 / N150** (4-core), **16 GB RAM**, **256–512 GB NVMe SSD**, Dual GbE (e.g., Beelink EQ13/EQ14, Asus NUC Essential, or industrial fanless N100) | $\$160\text{--}\$250$ | $6\text{--}12\text{ W}$ idle ($15\text{ W}$ peak) | **Best overall reliability.** Standard `x86_64` wheels for `scikit-learn`/`statsmodels` (zero ARM compilation surprises), fast 730-day MOS retraining ($< 45\text{ s}$), native NVMe longevity for hourly SQLite WAL writes, fanless models resist canyon dust. |
| **Raspberry Pi 5 + NVMe SSD** *(Recommended Ultra-Low-Power / DC Solar)* | **Raspberry Pi 5 (8 GB)** + Official Active Cooler + **M.2 NVMe HAT + 256 GB NVMe SSD** + Official $27\text{ W}$ USB-C PSU | $\$145\text{--}\$185$ | $3\text{--}7\text{ W}$ idle ($12\text{ W}$ peak) | **Best for DC/solar off-grid power budgets.** Easy GPIO/UART expansion later if connecting a local LoRaWAN or SDI-12 creek gauge directly. *Must use NVMe SSD—never run production SQLite on a microSD card.* |
| **Raspberry Pi on microSD** *(Not Recommended)* | Pi 4 / Pi 5 booting solely from microSD | $\$70\text{--}\$110$ | $3\text{--}6\text{ W}$ | **Avoid for life-safety telemetry.** Continuous hourly SQLite WAL writes and OS logs wear out consumer microSD cards within $6\text{--}18$ months, risking database corruption mid-storm. |

#### Mandatory Power Backup (UPS)
Plug **all three** of the following devices into a single dedicated **Uninterruptible Power Supply (UPS)** so telemetry and campus Wi-Fi stay alive during generator transfer or power outages:
1. The Mini PC / Raspberry Pi 5.
2. The **Davis Vantage Pro2 Console / WeatherLink Live (6100)** gateway.
3. The Stone Office **LAN switch / Wi-Fi access point**.

* **AC Option:** Compact sine-wave UPS (e.g., APC Back-UPS Pro 600–1000VA or CyberPower CP1000PFCLCD, giving $4\text{--}8+$ hours of runtime at $\sim 25\text{ W}$ total load).
* **DC / Off-Grid Option:** A $12\text{ V}$ LiFePO4 battery ($20\text{--}50\text{ Ah}$) with a $12\text{ V}$ DC-UPS / buck converter powering a $12\text{ V}$ Mini PC + router + Davis gateway directly for $24\text{--}48+$ hours with zero inverter losses.

---

### 2. How the System Survives an Internet Dropout

When the local node runs on-premises inside the Stone Office LAN, losing external internet triggers a **graceful degradation hierarchy** instead of a blackout:

1. **Live Local Rain & Weather Ingestion Over LAN (Zero Internet Needed):**
   * If Stone Office uses a **Davis WeatherLink Live (Model 6100)** on the local Ethernet/Wi-Fi network, the local node can poll its unauthenticated local HTTP JSON endpoint directly at `http://<WLL_LOCAL_IP>/v1/current_conditions` every 60 seconds (or read a USB/Serial Davis Console locally via `weewx` / `pyserial`) whenever WeatherLink Cloud is unreachable.
   * Assign the WeatherLink Live a **static DHCP reservation** on the Stone Office router (e.g., `192.168.1.50`) so its LAN IP never changes.
2. **Cached 7-Day NWS & MOS Forecast Horizon Remains Intact:**
   * Because the hourly pipeline stores the full **168-hour** `nws_forecasts` and `mos_corrected_forecasts` batches in local `tassajara_weather.db`, an internet outage mid-storm still leaves **days of valid hourly MOS QPF and $P_{90}$ forecasts** already saved on disk.
   * `hydro_risk_engine.py` continues updating $API_t$, runoff readiness $C$, routed creek discharge (CFS), and `GREEN / YELLOW / ORANGE / RED` escalations using **live local Stone Office tipping-bucket clicks + the cached MOS forecast**.
3. **Campus-Wide Local Dashboard Access (`http://tassajara-weather.local:8501`):**
   * Streamlit binds to the Stone Office LAN (`0.0.0.0:8501`) and advertises via mDNS (`avahi-daemon`). Any phone, tablet, or laptop on the Stone Office / Disaster Council Wi-Fi can open `http://tassajara-weather.local:8501` (or `http://<MINI_PC_IP>:8501`) with **zero external internet**.
4. **Automatic Re-Sync Upon Internet Recovery:**
   * The moment Starlink/ISP returns, the next hourly timer automatically pulls fresh NWS `/gridpoints/MTR` grids, 5-station IEM/NWS boundary telemetry, and 48-hour `WL_HISTORIC` verification rows—and optionally pushes updated snapshots to GitHub so the public Streamlit Cloud mirror catches up automatically.

---

### 3. Step-by-Step Local Linux Setup (Ubuntu Server 24.04 LTS or Raspberry Pi OS Lite 64-bit)

#### Step A: Base OS, Security Hardening & mDNS
```bash
# 1. Update OS and install core dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip sqlite3 \
                    avahi-daemon ufw unattended-upgrades util-linux

# 2. Set hostname so campus devices can reach http://tassajara-weather.local:8501
sudo hostnamectl set-hostname tassajara-weather
sudo systemctl enable --now avahi-daemon

# 3. Lock down firewall (LAN-only access — replace 192.168.1.0/24 with your Stone Office subnet)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.1.0/24 to any port 22 proto tcp comment 'SSH from local LAN only'
sudo ufw allow from 192.168.1.0/24 to any port 8501 proto tcp comment 'Streamlit Dashboard LAN only'
sudo ufw enable

# 4. Enable automatic security updates
sudo dpkg-reconfigure -plow unattended-upgrades
```

*(Recommended SSH hardening: disable password login in `/etc/ssh/sshd_config` (`PasswordAuthentication no`, `PermitRootLogin no`) after copying your SSH public key via `ssh-copy-id`.)*

#### Step B: Dedicated Service User, Repository & Protected `.env`
```bash
# Create non-root dedicated system user
sudo useradd -m -s /bin/bash tzmcweather
sudo su - tzmcweather

# Clone repository and install Python dependencies
git clone https://github.com/MT-Wildhack/tassajara-weather.git
cd tassajara-weather
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create restricted environment file (readable ONLY by tzmcweather)
cat <<'EOF' > /home/tzmcweather/tassajara-weather/.env
WL_API_KEY=your_weatherlink_v2_api_key
WL_API_SECRET=your_weatherlink_v2_api_secret
SYNOPTIC_TOKEN=optional_synoptic_token
WLL_LOCAL_IP=192.168.1.50
EOF
chmod 600 /home/tzmcweather/tassajara-weather/.env
exit
```

#### Step C: Always-On Local Streamlit Dashboard (`systemd` Service)
Create `/etc/systemd/system/tzmc-dashboard.service` so the dashboard starts automatically on power-up and restarts if it ever crashes:

```ini
[Unit]
Description=TZMC Microclimate & Flood Warning Streamlit Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=tzmcweather
WorkingDirectory=/home/tzmcweather/tassajara-weather
EnvironmentFile=/home/tzmcweather/tassajara-weather/.env
ExecStart=/home/tzmcweather/tassajara-weather/.venv/bin/streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Step D: Hourly Ingestion & Inference Timer (`systemd` Timer with `flock` Lock)
Using a `systemd` timer instead of bare `cron` provides structured `journalctl` logs, automatic catch-up (`Persistent=true`), and `flock` overlap protection so two runs never collide on SQLite.

Create `/etc/systemd/system/tzmc-pipeline.service`:
```ini
[Unit]
Description=TZMC Hourly Weather Ingestion, MOS Inference & Flood Risk Pipeline
After=network-online.target

[Service]
Type=oneshot
User=tzmcweather
WorkingDirectory=/home/tzmcweather/tassajara-weather
EnvironmentFile=/home/tzmcweather/tassajara-weather/.env
ExecStart=/usr/bin/flock -n /tmp/tzmc_pipeline.lock \
    /home/tzmcweather/tassajara-weather/.venv/bin/python3 \
    /home/tzmcweather/tassajara-weather/run_hourly_mos_pipeline.py
```

Create `/etc/systemd/system/tzmc-pipeline.timer`:
```ini
[Unit]
Description=Run TZMC Hourly MOS Pipeline at :05 past every hour

[Timer]
OnCalendar=*:05:00
Persistent=true
RandomizedDelaySec=15

[Install]
WantedBy=timers.target
```

Create a weekly retraining timer (`/etc/systemd/system/tzmc-retrain.service` & `tzmc-retrain.timer`, running Sundays at `03:30` local time using `backfill_and_train.py --skip-fetch`), then enable everything:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tzmc-dashboard.service
sudo systemctl enable --now tzmc-pipeline.timer

# Verify live status and logs
systemctl status tzmc-dashboard.service
systemctl list-timers --all | grep tzmc
journalctl -u tzmc-pipeline.service -n 50 --no-pager
```

#### Step E: Automated Safe SQLite WAL Backups
Never back up a live WAL-mode SQLite file with plain `cp` (which can produce a torn copy during a write). Schedule a daily safe snapshot using SQLite's online backup API to a secondary USB flash drive or Stone Office NAS:

```bash
# Example daily backup command (run at 04:00 via systemd timer or tzmcweather crontab)
mkdir -p /home/tzmcweather/backups
sqlite3 /home/tzmcweather/tassajara-weather/tassajara_weather.db \
    ".backup '/home/tzmcweather/backups/tassajara_weather_$(date +\%Y\%m\%d).db'"
find /home/tzmcweather/backups -name "tassajara_weather_*.db" -mtime +30 -delete
```

#### Step F: Dual-Mode Cloud + Local Topology (Recommended)
For maximum resilience with zero single point of failure:
* **Local Mini PC (Primary Life-Safety Source inside Canyon):** Serves `http://tassajara-weather.local:8501` over Stone Office Wi-Fi continuously—immune to Starlink/ISP dropouts.
* **GitHub Actions + Streamlit Cloud (Public / Off-Site Mirror):** Continues serving `https://tzmc-weather.streamlit.app` for staff traveling on Tassajara Road, SFZC City Center, and Jamesburg guests. Because both ingest independently from WeatherLink Cloud, NWS, and IEM `CA_DCP`, neither blocks the other.

---

## 10. Streamlit Dashboard UI

Launch locally via `streamlit run app.py` or view online at `https://tzmc-weather.streamlit.app`.

### Mode 1: Everyday Forecast (Community & Guest View)
* **Synchronized 4-Tier Flood & Road Warning Banner:** Plain-language guidance for Tassajara Road travel, lower footbridges, and operational actions.
* **Canyon Current Conditions & Topographic Shading Badge:** Real-time temperature, humidity, 24h rain, and solar state (Direct Sunlight, Black Cone Shadow, Active Storm Overcast, or Nocturnal Katabatic Drainage).
* **Freeze Warning & Creek Cold-Pool Alerts:** Flags nights where $P_{10} \le 32^\circ\text{F}$ to protect exposed pipes and garden beds.
* **24-Hour (3-Hour Cards) & 7-Day Planner:** Day/night-aware weather icons (`🌧️` at night vs. `🌦️` daytime), `100%` PoP badge for today's already-wet hours, and a clean 7-day planner table.
* **Interactive Creek Runoff & Road Passability Preview:** Soil wetness gauge, 72-hour routed hydrograph, and collapsible 4-tier severity matrix.

### Mode 2: Scientific & Statistical MOS Suite
* **8-Metric Telemetry Grid:** Dry-bulb, RH/dewpoint, Stull wet-bulb/VPD, 485 m wind vector, MSLP + 3h tendency, 24h rain + 72h MOS QPF, true crest lapse rate (`CHWC1` $^\circ\text{C/km}$), and gorge/coastal thermal gradients (`ASRC1`, `CAHC1`, `PPSC1` in $^\circ\text{F}$).
* **5-Station Boundary Network Inspector:** Live status (`Online`, `Online (Rain Gauge)` for `VTCC1`, or `Model Fallback / Stale`) with elevation-appropriate gradients.
* **3-Panel Verification & Operational Time Series:**
  * *Panel 1:* Observed Stone Office temperature vs. MOS Quantile Fan ($P_{10}\text{--}P_{90}$) vs. Raw NWS grid.
  * *Panel 2:* Observed canyon gusts ($485\text{ m}$) cleanly separated from forecast ridge gusts ($880\text{ m}$) + RH (%).
  * *Panel 3:* MOS calibrated QPF, $P_{90}$ extreme rates, observed tipping bucket, and calibrated PoP (%).
* **Deduplicated Verification Scorecard (`Past 72 Hours` / `Full Historical Archive`):** Raw NWS vs. MOS MAE/RMSE, RMSE Skill Score, Hourly Win Rate, $P_{10}\text{--}P_{90}$ Empirical Coverage (nominal $80.0\%$), Hydrological Brier Score, and paired residual table.
* **Model Interpretability & HAC OLS Diagnostics:** Permutation feature importances and full standardized Newey-West HAC OLS coefficient table ($R^2$, Wald $F$, $\kappa$, 95% CIs).

---

## 11. Hardware Roadmap & Pending Physical Stream Telemetry

The database schema (`stream_gauge_observations`, `vw_creek_flood_monitor`) and `hydro_risk_engine.py` are pre-engineered for a physical non-contact creek stage sensor at Tassajara Creek (e.g., downstream stringer of the old bathhouse footbridge):

```text
                       CURRENT PRODUCTION MODE (Virtual Two-Pathway Routing)
┌──────────────────────────────────────────────┐
│  Stone Office Davis VP2 (ISS 36966)          │───────► [hydro_risk_engine.py] ───► Routed Creek Index
│  + Live Crest Spillover (CHWC1 / VTCC1)      │          (API + 2-Pathway Router)   (discharge_index_cfs)
└──────────────────────────────────────────────┘

                       PHASE 2 HARDWARE ROADMAP (Physical Ground Truth)
┌──────────────────────────────────────────────┐
│  Non-Contact Radar / Ultrasonic Level Gauge  │───────► [stream_gauge_observations]
│  (e.g., Campbell CS475A / MaxBotix + LoRa)   │          - stream_stage_m / rate_of_rise_m_hr
└──────────────────────────────────────────────┘          - water_temp_c / battery_v
```

### Automatic Physical Failover Already Coded
When physical telemetry begins populating `stream_gauge_observations`, `evaluate_flood_and_road_tier()` automatically activates physical stage thresholds ($0.75\text{ m}$, $1.15\text{ m}$, $1.60\text{ m}$) and rapid-surge triggers ($\ge 0.40\text{ m/hr}$) alongside the rainfall and discharge triggers without requiring dashboard rewrites.

---

## 12. Future Directions & Methodological Improvements

1. **Cloud-Attenuated Midday Insolation (`v2` MOS Feature):** During persistent dense midday rainstorms ($>90\%$ RH, active precipitation), clear-sky geometric insolation can still introduce a warm daytime overshoot. Weighting geometric insolation by forecast cloud cover / RH / PoP will tighten storm-day high temperatures.
2. **Chews Ridge Summit Icing Advisory:** `CHWC1` ($5,039\text{ ft}$) routinely drops near or below $32^\circ\text{F}$ during passing cold fronts while the canyon floor sits in the $40\text{s}$. Adding a dedicated summit freeze/black-ice badge when `CHWC1` $\le 33^\circ\text{F}$ during wet road conditions will further improve Tassajara Road travel safety.
3. **Physical Creek Rating Curve Calibration:** Logging visual footbridge stage observations during winter storms (and eventually installing a radar level gauge) will convert the synthetic `CFS` index into a site-calibrated stage-discharge rating curve.
4. **Native LAN WeatherLink Live Poller Integration:** Extending `ingest_weather.py` with an automatic fallback from WeatherLink Cloud v2 to `http://$WLL_LOCAL_IP/v1/current_conditions` will make local Mini PC storm operation 100% immune to WAN outages out of the box.
5. **Probabilistic Ensemble Forcing (HREF / GEFS) & Burn-Scar Hydrology:** Integrating high-resolution ensemble QPF probabilities and dynamic post-wildfire BAER soil burn severity scaling ($C_{\max}$ adjustment) for long-term watershed resilience.

---

## 13. License & Operational Disclaimer

Maintained for the safety and operational resilience of the **Tassajara Zen Mountain Center (San Francisco Zen Center)** community, staff, students, and guests. Distributed under the **MIT License**.

**Operational Disclaimer:** Forecasts, hydrological indices, and alert tiers (`GREEN`, `YELLOW`, `ORANGE`, `RED`) generated by this software serve strictly as **local advisory decision-support guidance**. They do not constitute an official government warning and never supersede physical ground verification, visual inspection of Tassajara Road and creek footbridges, or official directives issued by the **National Weather Service (NWS Monterey)**, **Monterey County Office of Emergency Services (OES)**, **US Forest Service (USFS)**, or the **TZMC Disaster Council**.
