# Tassajara Zen Mountain Center (TZMC) — Microclimate & Hydrological MOS Forecast Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-tassajara--weather.streamlit.app-FF4B4B.svg)](https://tassajara-weather.streamlit.app)
[![Database](https://img.shields.io/badge/Database-SQLite%20WAL-003B57.svg)](https://sqlite.org)
[![Model Architecture](https://img.shields.io/badge/Architecture-MOS__GBM__v3__HYDRO%20%2B%20HAC%20OLS-orange.svg)](#5-mos-machine-learning-architecture)
[![Hydrology](https://img.shields.io/badge/Hydrology-Two--Pathway%20API%20Routing-0284C7.svg)](#6-early-flood--road-washout-warning-system)

An operational, publication-grade meteorological **Model Output Statistics (MOS)** forecast suite and **Early Flood & Road Washout Warning System** custom-engineered for the **Tassajara Zen Mountain Center (TZMC)** in the Santa Lucia Mountains of Monterey County, California.

* **Live Cloud Dashboard:** [https://tassajara-weather.streamlit.app](https://tassajara-weather.streamlit.app)

This README serves as both the scientific reference and the complete **Maintainer Handover Manual**, covering cloud automation, database operations, model retraining, and dedicated local Mini PC / Raspberry Pi deployment integrated with the Stone Office **Davis Vantage Pro2 Cabled Console** and its attached **WeatherLinkIP (`6555`) Ethernet Data Logger** for uninterrupted storm-mode operation if internet connectivity drops.

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

```
Elevation (m MSL)
1600m ─── [CHWC1: Chews Ridge 1536m] ────────── [VTCC1: Ventana Cone 1453m (Rain Gauge)]
          \                                    /
1200m ─────\────── High-Gradient Mountain Tributaries ──────/
            \                                              /
800m  ───────\── Operational NWS/NWP Grid Reference Elevation (~880m MSL) ────
              \                                          /
485m  ─────────\── [TZMC Stone Office Davis VP2 Cabled 485m] ─/ (Deep Canyon Cold-Pool)
                \      └─ WeatherLinkIP (6555) Ethernet      /
200m  ───────────\── [ASRC1: Arroyo Seco 268m] ───────────── [PPSC1: Big Sur 108m Coast]
```

### Microclimatic & Hydrological Complexities

* **Topographic Shading:** The southern canyon wall (Black Cone Trail ridge, $32^\circ$ horizon) casts an early afternoon solar shadow, accelerating radiational cooling hours before astronomical sunset.
* **Nocturnal Katabatic Drainage:** Cool air pooling along Tassajara Creek decouples the canyon floor from the free atmosphere, pushing overnight lows $5^\circ\text{F}$ to $15^\circ\text{F}$ colder than regional model grids.
* **Orographic Spillover & Feeder-Seeder Rain:** Pacific atmospheric rivers intercept the coastal Santa Lucia crest, wringing out intense headwater rainfall that spills into upper Tassajara Creek tributaries while the valley floor sits in partial rain-shadow.
* **Flash Floods & Road Slipouts:** Multi-inch storms over steep, fractured granite and clay-rich mountain slopes trigger rapid creek surges, woody debris damming at lower footbridges, and hazardous slipouts along Tassajara Road.

---

## 2. System Architecture & Repository Map

The suite runs on an automated hourly cycle (`:05` past each hour via GitHub Actions and/or a local `systemd` timer), pulling Stone Office telemetry from the **Davis Vantage Pro2 Cabled Console** (Station `36966`) via its attached **WeatherLinkIP (`6555`) Ethernet Data Logger**, 5-station boundary observations, and NWS `/gridpoints/MTR` hourly forecasts; executing MOS inference and two-pathway hydrological routing; and persisting everything to a WAL-mode SQLite database consumed by the Streamlit UI.

```
                              ┌──────────────────────────────────────────────────────────┐
                              │  Stone Office Davis Vantage Pro2 Cabled Console (36966)  │
                              │  Attached Logger: Davis WeatherLinkIP (6555) Ethernet    │
                              │  • Normal Mode: WeatherLink v2 Cloud API (current + 48h) │
                              │  • Offline Mode: Direct LAN TCP Socket (Port 22222)      │
                              └────────────────────────────┬─────────────────────────────┘
                                                           │
┌───────────────────────────────────────┐                  │             ┌──────────────────────────────────────────┐
│ 5-Station Santa Lucia Boundary Array  │                  │             │ NWS Monterey /gridpoints/MTR Forecast    │
│ Tier 1: Synoptic API (optional)       │                  │             │ 168-hr hourly NWP grid (~880m MSL)       │
│ Tier 2A: NOAA/NWS Obs (NWS_OBS)       │                  │             │ (Open-Meteo top-up if NWS grid drops)    │
│ Tier 2B: Free IEM CA_DCP (CHWC1/VTCC1)│                  │             └────────────────────┬─────────────────────┘
│ Tier 3: Open-Meteo (flagged fallback) │                  │                                  │
└──────────────────┬────────────────────┘                  │                                  │
                   │                                       │                                  │
                   ▼                                       ▼                                  ▼
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
│                app.py (Streamlit Web Dashboard — https://tassajara-weather.streamlit.app)                    │
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

* **Zero Midnight Rain Double-Counting (Monotonic `rain_year_mm` Tracking):** Davis Vantage Pro2 console `rain_day_mm` counters can drift across local midnight (`00:00 America/Los_Angeles`), previously causing unreset prior-day totals to dump into `01:00 PST` as a phantom single-hour cloudburst. Both `run_hourly_mos_pipeline.py` and `app.py` now compute incremental rainfall using the console's monotonic **`rain_year_mm`** counter persisted in `ingest_state` (falling back to monotonic `rain_day_mm` only when `console_day_mm < prev_day_mm`), paired with 48-hour WeatherLink Pro `/v2/historic` (`WL_HISTORIC`) reconciliation.
* **Strict Observation Provenance & Tipping-Bucket Fingerprint:** Every row in `local_observations` carries an explicit `source` tag (`WL_HISTORIC` or `WL_LIVE`). Genuine Davis Vantage Pro2 hourly rainfall totals always fall on exact $0.254\text{ mm}$ ($0.01\text{ in}$) tipping-bucket multiples, allowing immediate audit verification against any synthetic $0.1\text{ mm}$ model data (`OPENMETEO_SYNTH`). Synthetic rows are excluded from `vw_mos_training_dataset` and archive rain statistics.
* **Verification Deduplication:** Historical MOS verification joins (`mos_corrected_forecasts`) use `ROW_NUMBER() OVER (PARTITION BY valid_time_utc, issue_time_utc ORDER BY model_version DESC)` plus `valid_time_utc` deduplication in `app.py`, ensuring exact $72 / 72$ paired evaluation hours in the 72-hour verification scorecard.

---

## 5. MOS Machine Learning Architecture

The production statistical engine (`mos_model.py`, version **`MOS_GBM_v3_HYDRO`**) is trained on **$17,570$ verified Davis Vantage Pro2 hourly pairs** spanning 2 full annual cycles (**$870$ observed wet hours**, $51.62\text{ in}$ verified tipping-bucket rainfall) with **30 physical predictors**.

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

Rather than snapping abruptly back to dry baseflow after rain pauses, effective runoff ($R_t = C_t \cdot P_{\text{eff}, t}$) splits across two physical storage pathways over the $72\text{ km}^2$ watershed above a $6.5\text{ CFS}$ spring baseflow floor ($Q_{\text{base}} = 6.5\text{ CFS}$):

* **Fast Surface Quickflow ($18\%$):** Routed via a steep-canyon 4-hour unit hydrograph kernel to capture rapid storm-band peaks.
* **Shallow Subsurface Interflow ($57\%$):** Routed through a linear hillslope storage reservoir ($k_{\text{interflow}} = 0.955\text{ hr}^{-1}$) sustaining realistic multi-day post-storm creek recession across the `now` boundary with zero step discontinuity.

> **Calibration Note:** Until a physical stream stage gauge is installed at the bathhouse footbridge, routed `CFS` is an uncalibrated catchment-scaled discharge index. Visual ground-truth logs at the footbridges during storms can be used to tune `API_MIDPOINT_MM` and `INTERFLOW_FRACTION` in `hydro_risk_engine.py`.

### 4. Four-Tier Operational Severity Matrix

| Tier | Status Code | Rain, Soil Saturation & Discharge Triggers | Tassajara Road (14-Mile Unpaved) | Tassajara Creek & Footbridges | Operational Protocol |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **0** | **GREEN** | $API < 2.00"$, $1\text{h Rate} < 0.15"\text{/hr}$, $24\text{h Rain} < 1.00"$, $72\text{h Rain} < 1.50"$, $Q < 250\text{ CFS}$ | **Open / Dry to Damp**<br>Standard mountain precautions | **Stable Baseflow**<br>Footbridges clear | Routine canyon operations. Normal high-clearance travel. |
| **1** | **YELLOW** | $API \ge 2.00"$ OR Imminent ($\le 6\text{h}$) Rate $\ge 0.15"\text{/hr}$ OR $24\text{h Rain} \ge 1.00"$ OR $72\text{h Rain} \ge 1.50"$ | **Advisory / Caution**<br>Muddy, slick clay grades over Chews Ridge; minor rockfall | **Elevated Flow**<br>Turbid water; moderate rise | 4WD required on Tassajara Road. Postpone non-essential low-clearance trips; clear culverts. |
| **2** | **ORANGE** | Within $24\text{h}$: Rate $\ge 0.50"\text{/hr}$ on primed soil ($API \ge 2.00"$), OR multi-day storm total on saturated soil, **OR Routed Creek $Q \ge 250\text{ CFS}$** | **Hazardous / High Risk**<br>Travel strongly discouraged; slipout & rockslide danger | **Rapidly Rising**<br>High velocity; woody debris mobilization likely | Suspend routine travel on Tassajara Road. Inspect lower footbridges for log damming; stage emergency radio net. |
| **3** | **RED** | Imminent/Active ($\le 6\text{h}$): Rate $\ge 1.00"\text{/hr}$ OR $P_{90} \ge 1.25"\text{/hr}$ on saturated soil, **OR Routed Creek $Q \ge 500\text{ CFS}$** | **CLOSED / IMPASSABLE**<br>Severe washouts, debris flows, road cuts | **BANKFULL / FLOOD STAGE**<br>Footbridge overtopping surge hazard | **Immediate road closure.** Evacuate low creek-side structures; prohibit all footbridge crossings. |

---

## 7. Database Schema & Production Views

The SQLite database (`tassajara_weather.db`) runs in **Write-Ahead Logging (`WAL`)** mode so the hourly ingestion pipeline and Streamlit dashboard can read/write concurrently without locking errors:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

### Core Tables

* **`local_observations`:** Ground-truth Stone Office Davis Vantage Pro2 records (`WL_HISTORIC`, `WL_LIVE`). Stores temperature, dewpoint, wetbulb, MSLP, wind vector, solar radiation, and $0.254\text{ mm}$-quantum tipping-bucket rainfall (`precip_1hr_mm`).
* **`reference_observations`:** Synchronized hourly records from the 5 Santa Lucia reference stations (`CHWC1`, `VTCC1`, `CAHC1`, `ASRC1`, `PPSC1`) with explicit `source` provenance (`SYNOPTIC`, `NWS_OBS`, `IEM_CA_DCP`).
* **`nws_forecasts`:** Operational 168-hour NWS `/gridpoints/MTR` forecast grids stored by `issue_time_utc` and `valid_time_utc`.
* **`mos_corrected_forecasts`:** Deduplicated `MOS_GBM_v3_HYDRO` hourly temperature, $P_{10}$/$P_{90}$ bounds, calibrated PoP (%), and expected/$P_{90}$ QPF rates.
* **`hydrological_risk_state`:** Hourly archive of catchment saturation ($API$), runoff coefficient ($C$), peak runoff, routed creek discharge (CFS), and 4-tier alert status.
* **`hydrological_forecast_series`:** Forward 72-hour projected hydrograph and $API$ trajectory.
* **`ingest_state`:** Key-value state table holding monotonic rainfall counters (`last_console_rain_year_mm`, `last_console_rain_day_mm`, `last_ref_rain_day_mm_<STID>`) and `last_obs_hour_iso`.
* **`stream_gauge_observations`:** Physical stream gauge telemetry table (pre-wired for future bathhouse footbridge sensor commissioning).

### Production Views

* **`vw_mos_training_dataset`:** Deduplicated training view joining genuine Davis observations (`source != 'OPENMETEO_SYNTH'`) with aligned forecast grids and boundary reference telemetry.
* **`vw_creek_flood_monitor`:** Operational summary view uniting live Stone Office conditions, catchment wetness ($API$), routed creek discharge, and active alert tiers.

---

## 8. Maintainer Handover: Setup, Secrets & Cloud Operations

### 1. Required Accounts & Credentials Matrix

Never commit API keys, secrets, or physical Davis device `DID`/`KEY` credentials to Git. Configure them in **local shell/systemd `.env`**, **GitHub Repository Secrets**, and **Streamlit Cloud Secrets**:

| Variable | Required? | Where to Obtain / Notes |
| :--- | :---: | :--- |
| `WL_API_KEY` | **Yes** | Davis WeatherLink v2 API Key for Station `36966` (WeatherLink Pro account). |
| `WL_API_SECRET` | **Yes** | Davis WeatherLink v2 API Secret paired with `WL_API_KEY`. |
| `SYNOPTIC_TOKEN` | Optional | Synoptic Data Public Token. If omitted or returning `403`, `ingest_weather.py` automatically falls back to free `NWS_OBS` + `IEM_CA_DCP` for all 5 stations. |
| `WLIP_LOCAL_IP` | Local Only | Static LAN IP of the Stone Office **Davis WeatherLinkIP (`6555`) Ethernet Data Logger** for direct TCP port `22222` offline failover (see Section 9). |

> **Security Note on Physical Hardware Labels:** The WeatherLinkIP (`6555`) Ethernet dongle attached to the back of the Stone Office console carries a physical sticker showing its `DID` (MAC/Device ID) and `KEY`. Treat these as confidential credentials—record them only in the Stone Office IT password vault and never post photos of the label or paste the `DID`/`KEY` into public repositories.

### 2. Local Development & Manual Execution

```bash
git clone https://github.com/MT-Wildhack/tassajara-weather.git
cd tassajara-weather

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Export credentials (or load from a chmod 600 .env file)
export WL_API_KEY="your_weatherlink_v2_api_key"
export WL_API_SECRET="your_weatherlink_v2_api_secret"
export SYNOPTIC_TOKEN="optional_synoptic_token"

# Run full hourly ingestion, MOS inference, and hydrological evaluation
python3 run_hourly_mos_pipeline.py

# Run standalone hydrological evaluation
python3 hydro_risk_engine.py

# Retrain MOS model on existing SQLite archive (no re-fetch)
python3 backfill_and_train.py --skip-fetch

# Launch dashboard locally
streamlit run app.py
```

### 3. Cloud Automation & Streamlit Cloud Operations

* **Hourly Pipeline (GitHub Actions):** `.github/workflows/hourly_weather.yml` runs at `:05` past every hour, executes `run_hourly_mos_pipeline.py`, and commits the updated `tassajara_weather.db` back to `main`. Verify green runs under the repository's **Actions** tab.
* **Rebooting Streamlit Cloud (`https://tassajara-weather.streamlit.app`):** After pushing code, database repairs, or a newly retrained `models/tassajara_mos_v1.joblib`, Streamlit Cloud auto-redeploys within ~60 seconds. If cached modules or old model artifacts persist, click **`Manage app`** (bottom-right) $\rightarrow$ **`⋮`** $\rightarrow$ **`Reboot app`**.
* **Handling `tassajara_weather.db` Merge Conflicts in GitHub Desktop:** Because GitHub Actions commits `tassajara_weather.db` hourly, pushing local edits may trigger a binary merge conflict:
  * If you only edited Python files (`app.py`, `ingest_weather.py`, etc.), choose **Use the version from `main`** for `tassajara_weather.db` so you keep the latest automated observations.
  * If you intentionally repaired rows or retrained locally, choose **Use the modified file** (your local `tassajara_weather.db`), click **Continue Merge**, and push.

---

## 9. Local On-Premises Deployment: Hardware Selection & Offline Resilience

During severe Santa Lucia winter storms, Tassajara's satellite/WAN internet link and grid power are most likely to drop **at the exact moment flood and road-washout guidance is most critical**. Running a dedicated, hardened local appliance inside the Stone Office on the same UPS-backed LAN as the **Davis Vantage Pro2 Cabled Console** and its **WeatherLinkIP (`6555`) Ethernet Data Logger** guarantees 100% uptime for local staff and the Emergency Response team even during a complete internet blackout.

### 1. Recommended Hardware Comparison

| Criterion | **Option A: Fanless Intel N100 / N150 Mini PC (Recommended)** | **Option B: Raspberry Pi 5 (8 GB) + NVMe Hat** |
| :--- | :--- | :--- |
| **Target Models** | Beelink EQ13/EQ14 (N100/N150), MeLE Quieter4C (fanless), Minisforum UN100P | Raspberry Pi 5 (8 GB) + Official Active Cooler + M.2 NVMe Base + 256 GB NVMe SSD |
| **Storage Reliability** | Native M.2 NVMe SSD (high write endurance for hourly SQLite WAL commits) | **Mandatory NVMe SSD** (never run SQLite WAL 24/7 on a microSD card—cards fail within 6–12 months) |
| **Power Draw** | $6\text{–}12\text{ W}$ idle, $15\text{–}20\text{ W}$ peak (easily backed by a small UPS) | $3\text{–}5\text{ W}$ idle, $10\text{–}12\text{ W}$ peak (best if running purely on off-grid DC solar/battery) |
| **MOS Retraining Speed** | Full 730-day (`17,570`-row) 5-fold GBDT + OLS retrain in **~45–75 seconds** | Full 730-day retrain in ~2.5–4 minutes |
| **Dust & Thermal Resilience** | Fanless aluminum chassis (e.g., MeLE Quieter4C) resists canyon dust and summer heat | Requires dust-filtered aluminum case + heatsink |
| **Estimated Cost (w/ UPS)** | ~$220–$300 (Mini PC + 500–600 VA UPS) | ~$200–$260 (Pi 5 + NVMe Hat/SSD + PSU + Case + UPS) |

**Recommendation:** Deploy a **fanless Intel N100/N150 Mini PC (16 GB RAM, 256–512 GB NVMe SSD)** running **Ubuntu Server 24.04 LTS**, backed by a **600–850 VA Sine-Wave UPS** (e.g., APC Back-UPS Pro or CyberPower CP850PFCLCD). Plug all four critical components into the **Battery Backup** outlets of the same UPS:
1. The Mini PC
2. The Stone Office local Ethernet switch / Wi-Fi router
3. The **Davis Vantage Pro2 Cabled Console** AC power adapter *(also keep **3× fresh C-cell alkaline/lithium backup batteries** installed inside the console—replaced every October—so the cabled console and outdoor ISS keep logging even if the UPS exhausts during a multi-day outage)*
4. An **inline RJ-11 telephone/data surge protector** on the copper cable entering the Stone Office from the outdoor Davis ISS to protect the console and Ethernet logger against storm-induced surges

### 2. Confirmed Hardware Integration: Davis Vantage Pro2 Cabled Console + WeatherLinkIP (`6555`) Ethernet

The Stone Office **Davis Vantage Pro2 Cabled Console** is fitted with a **Davis WeatherLinkIP (`6555`) Ethernet Data Logger** plugged directly into the back expansion bay and patched into the Stone Office Ethernet network.

#### Network Setup (One-Time)
1. Log into the Stone Office router/DHCP server and locate the WeatherLinkIP (`6555`) Ethernet logger by its MAC address (matching the `DID` printed on the Ethernet dongle label).
2. Assign a **permanent Static DHCP Reservation** on the local LAN (for example, `192.168.1.50`) and record it as `WLIP_LOCAL_IP="192.168.1.50"` in `/opt/tassajara-weather/.env`.
3. Verify local reachability from the Mini PC (with zero internet required):
   ```bash
   nc -zv -w 3 192.168.1.50 22222
   ```

#### Critical WeatherLinkIP (`6555`) Single-Socket Protocol
Unlike wireless WeatherLink Live hubs, the **WeatherLinkIP (`6555`) Ethernet logger only supports ONE active TCP socket on port `22222` at a time**. While any local client holds port `22222` open, the logger pauses its automatic uploads to `weatherlink.com`.

To prevent disrupting cloud uploads while guaranteeing instant offline survival, follow this strict failover protocol:
* **Normal Mode (Internet Up):** `run_hourly_mos_pipeline.py` pulls Stone Office data from the **WeatherLink v2 Cloud API** (`/v2/current` + 48-hour `/v2/historic`) without touching TCP port `22222`. Both the local appliance and `https://tassajara-weather.streamlit.app` stay continuously synchronized.
* **Storm / Internet-Outage Failover Mode (Internet Down):** Only when the WeatherLink v2 Cloud API request times out or fails DNS resolution does the local appliance open a brief (`< 2 second`) TCP socket to `WLIP_LOCAL_IP:22222`, send the Davis serial wake-up (`\n`) + `LOOP 1\n` (or `DMPAFT` for buffered archive records), parse the binary Davis packet (temperature, RH, barometric pressure, wind, and monotonic `year_rain` clicks), and **immediately close the TCP socket**.
* **Internal Archive Buffer:** The `6555` logger stores up to **2,560 archive records** onboard (over 17 days at 10-minute intervals, or > 53 days at 30-minute intervals). Even if both the internet and the Mini PC go down temporarily, no rainfall clicks are lost—as soon as connectivity returns, `WL_HISTORIC` catches up automatically.

### 3. How the System Survives a Complete Internet Dropout

| Capability | Internet Up (Normal) | Internet Down (Storm / Island Mode) |
| :--- | :--- | :--- |
| **Stone Office Live Telemetry** | WeatherLink v2 Cloud API (fed by `6555` Ethernet) | **100% Live** via brief local LAN TCP poll (`WLIP_LOCAL_IP:22222`) |
| **Hydrological Soil Wetness ($API$) & Creek CFS** | Live Stone Office + 5-station crest spillover | **100% Live** from Stone Office tipping bucket + last cached 168h MOS QPF |
| **Flood & Road Warning Tiers (`GREEN`–`RED`)** | Live real-time escalation | **100% Operational** (escalates immediately on local rain rates, $API$, and routed CFS) |
| **Forecast Horizon (Temp, PoP, QPF)** | Refreshed hourly from NWS `/gridpoints/MTR` | **Continues smoothly for up to 7 days** from the last cached 168-hour `MOS_GBM_v3_HYDRO` batch in `tassajara_weather.db` |
| **5-Station Regional Array** | Live (`NWS_OBS` + `IEM_CA_DCP`) | Gracefully marks remote stations stale (`age_hours > 3.5h`) without crashing or fabricating spillover |
| **Dashboard Access for Staff** | `https://tassajara-weather.streamlit.app` + local LAN | **Local LAN / Wi-Fi:** `http://tassajara-weather.local:8501` (accessible from any phone/laptop on Stone Office Wi-Fi) |

### 4. Hardened Local Linux Setup (Ubuntu Server 24.04 LTS)

#### Step A: Base OS, Dedicated User, mDNS & Firewall
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip sqlite3 git ufw avahi-daemon netcat-openbsd

# Set local hostname so staff can browse to http://tassajara-weather.local:8501
sudo hostnamectl set-hostname tassajara-weather

# Lock down firewall: allow SSH and Streamlit strictly from the local Stone Office LAN subnet
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.0.0/16 to any port 22 proto tcp
sudo ufw allow from 192.168.0.0/16 to any port 8501 proto tcp
sudo ufw enable

# Create dedicated unprivileged service user
sudo useradd -r -m -d /opt/tassajara-weather -s /bin/bash tzmc
sudo -u tzmc git clone https://github.com/MT-Wildhack/tassajara-weather.git /opt/tassajara-weather/app
cd /opt/tassajara-weather/app
sudo -u tzmc python3 -m venv .venv
sudo -u tzmc .venv/bin/pip install --upgrade pip -r requirements.txt
```

#### Step B: Protected Environment File (`/opt/tassajara-weather/.env`)
```bash
sudo tee /opt/tassajara-weather/.env > /dev/null <<'EOF'
WL_API_KEY=your_weatherlink_v2_api_key
WL_API_SECRET=your_weatherlink_v2_api_secret
SYNOPTIC_TOKEN=optional_synoptic_token
WLIP_LOCAL_IP=192.168.1.50
WLIP_LOCAL_PORT=22222
EOF
sudo chown tzmc:tzmc /opt/tassajara-weather/.env
sudo chmod 600 /opt/tassajara-weather/.env
```

#### Step C: Persistent Streamlit Dashboard Service (`/etc/systemd/system/tzmc-dashboard.service`)
```ini
[Unit]
Description=Tassajara Weather & Flood MOS Streamlit Dashboard
After=network-online.target

[Service]
Type=simple
User=tzmc
WorkingDirectory=/opt/tassajara-weather/app
EnvironmentFile=/opt/tassajara-weather/.env
ExecStart=/opt/tassajara-weather/app/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### Step D: Hourly Pipeline Service & Timer (with `flock` Concurrency Protection)
Create `/etc/systemd/system/tzmc-pipeline.service`:
```ini
[Unit]
Description=Tassajara Hourly MOS & Hydrological Pipeline
After=network-online.target

[Service]
Type=oneshot
User=tzmc
WorkingDirectory=/opt/tassajara-weather/app
EnvironmentFile=/opt/tassajara-weather/.env
ExecStart=/usr/bin/flock -n /tmp/tzmc_pipeline.lock /opt/tassajara-weather/app/.venv/bin/python3 run_hourly_mos_pipeline.py
```

Create `/etc/systemd/system/tzmc-pipeline.timer`:
```ini
[Unit]
Description=Run Tassajara Hourly MOS Pipeline at :05 past every hour

[Timer]
OnCalendar=*:05:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start both services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tzmc-dashboard.service tzmc-pipeline.timer
sudo systemctl status tzmc-dashboard.service tzmc-pipeline.timer
```

#### Step E: Safe Daily WAL SQLite Backups
Never copy a live WAL-mode SQLite file with `cp` (which can produce a corrupted snapshot). Schedule a nightly online backup using SQLite's native `.backup` command to a mounted USB drive or NAS:
```bash
sqlite3 /opt/tassajara-weather/app/tassajara_weather.db ".backup '/mnt/backup/tassajara_weather_$(date +%F).db'"
```

#### Step F: BIOS Auto-Power-On After Extended Blackout
In the Mini PC's UEFI/BIOS settings, enable **Restore on AC Power Loss $\rightarrow$ Power On**. If a multi-day storm drains the UPS completely, the appliance will automatically boot, restart the dashboard and hourly timer, and resume flood monitoring the moment generator or grid power returns.

---

## 10. Streamlit Dashboard UI

Hosted live at **[https://tassajara-weather.streamlit.app](https://tassajara-weather.streamlit.app)** (or locally at `http://tassajara-weather.local:8501`), the interface provides two dedicated modes:

### 1. Everyday Forecast (Community & Guest View)
* **Synchronized Flood & Road Warning Banner:** Color-coded operational guidance (`GREEN`, `YELLOW`, `ORANGE`, `RED`) covering Tassajara Road passability, footbridge debris risk, and actionable emergency steps.
* **Topographic Shading & Current Conditions Card:** Plain-language canyon conditions (`Active Storm Overcast`, `Canyon Wall Shadow`, `Clear Canyon Night`), 24-hour extremes, and rolling 24-hour rain.
* **Freeze & Trail Advisories:** Automated freeze alerts when cold-pool $P_{10}$ dips to $\le 32^\circ\text{F}$, plus 3-hour outlook cards (with day/night-aware icons) and a 7-day planner table.
* **Interactive Creek Runoff & Hydrograph Outlook:** Displays catchment saturation ($API$), runoff readiness ($C$), and routed creek discharge alongside the 4-tier severity matrix.

### 2. Scientific & Statistical MOS Suite
* **8-Metric Boundary-Layer Telemetry Grid:** Live dry-bulb, RH/dewpoint, Stull wet-bulb/Tetens VPD, anemometer vector, MSLP + 3-hour barometric tendency, 24h rain + 72h MOS QPF, crest lapse rate ($^\circ\text{C/km}$ for crest stations $\Delta z \ge 0.30\text{ km}$), and gorge/coastal differentials ($\Delta T$ in $^\circ\text{F}$).
* **5-Station Santa Lucia Network Table:** Live status (`Online`, `Online (Rain Gauge)` for `VTCC1`, or `Model Fallback / Stale`), temperature, RH, wind/gust, hourly rain, and vertical/horizontal gradient vs. Stone Office.
* **3-Panel Interactive Plotly Suite:**
  * *Panel 1:* Observed temperature actuals vs. MOS Quantile Fan ($P_{10}$–$P_{90}$) vs. Raw NWS forecast.
  * *Panel 2:* Observed canyon gusts ($485\text{ m}$) cleanly separated from forecast ridge-grid gusts ($880\text{ m}$) + RH (%).
  * *Panel 3:* MOS calibrated QPF, $P_{90}$ extreme downpour rates, observed tipping-bucket rain, and PoP (%).
* **Deduplicated Forecast Accuracy Verification Scorecard:** Evaluates exact paired hours (72h or Full Archive) for Raw NWS vs. MOS MAE/RMSE, RMSE Skill Score (%), Hourly Win Frequency, $P_{10}$–$P_{90}$ empirical coverage, and hydrological Brier Score & QPF bias.
* **Model Interpretability & HAC OLS Diagnostics:** Permutation feature importances and standardized Newey-West HAC OLS regression summary ($R^2$, $F$-stat, $\kappa$, $95\%$ CIs).

---

## 11. Hardware Roadmap & Pending Physical Stream Telemetry

The database schema (`stream_gauge_observations` and `vw_creek_flood_monitor`) and `hydro_risk_engine.py` are pre-wired for a physical stream-stage sensor at the **Tassajara Creek Bathhouse Footbridge**:

* **Current Mode:** `UNINSTALLED (MOS + Tipping Bucket Mode)` — `stream_stage_m`, `rate_of_rise_m_hr`, `water_temp_c`, and `battery_v` remain `NULL`.
* **Target Hardware:** Non-contact SDI-12 / Modbus Radar or Ultrasonic Level Sensor (e.g., Campbell Scientific CS475A) paired with a solar-powered ESP32 / LoRaWAN node transmitting 5-minute stage readings to the Stone Office Mini PC.
* **Automatic Integration:** Once rows populate `stream_gauge_observations`, `evaluate_flood_and_road_tier()` automatically activates physical stage height and surge-rate ($\text{m/hr}$) escalation rules alongside the existing $API$ and `250 / 500 CFS` triggers without requiring code rewrites.

---

## 12. Future Directions & Methodological Improvements

1. **Cloud-Attenuated Insolation (`v2` MOS Retrain):** Attenuating the clear-sky geometric insolation predictor using forecast cloud cover / RH / PoP will eliminate midday warm overshoots during dense winter storm overcast.
2. **Visual Creek Calibration:** Logging observed water depth, turbidity, and bankfull percentage at the footbridges during storms to calibrate `API_MIDPOINT_MM` and `INTERFLOW_FRACTION` against real canyon hydraulics.
3. **Chews Ridge Summit Icing Advisory:** Adding a dedicated road-icing badge when live `CHWC1` ($5,039\text{ ft}$) drops to $\le 33^\circ\text{F}$ during wet road conditions (`API >= 2.00 in`).
4. **Post-Fire Hydrological Scaling:** Ingesting BAER soil burn severity layers if future wildfires impact the upper Ventana Wilderness catchment to dynamically adjust $C_{\max}$ for hydrophobic debris-flow risk.

---

## 13. License & Operational Disclaimer

Maintained for the safety and operational resilience of the **Tassajara Zen Mountain Center (San Francisco Zen Center)** community, staff, students, and guests. Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).

**Operational Disclaimer:** The computations, forecasts, discharge indices, and alert tiers (`GREEN`, `YELLOW`, `ORANGE`, `RED`) generated by this software serve strictly as **advisory decision-support guidance**. They do not constitute an official government warning and do not supersede direct visual inspection of Tassajara Road and Tassajara Creek or official directives issued by the **National Weather Service (NWS Monterey)**, **Monterey County Office of Emergency Services (OES)**, **US Forest Service (USFS)**, or the **TZMC Disaster Council**.
