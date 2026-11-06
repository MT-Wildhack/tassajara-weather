# Tassajara Zen Mountain Center (TZMC) — Microclimate & Hydrological MOS Forecast Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://share.streamlit.io)
[![Database](https://img.shields.io/badge/Database-SQLite%20WAL-003B57.svg)](https://sqlite.org)
[![Model Architecture](https://img.shields.io/badge/Architecture-MOS%20GBDT%20%2B%20HAC%20OLS-orange.svg)](#mos-machine-learning-architecture)
[![Hydrology](https://img.shields.io/badge/Hydrology-API%20Runoff%20Routing-0284C7.svg)](#early-flood--road-washout-warning-system)

An operational, publication-grade meteorological Model Output Statistics (MOS) corrector and Early Flood & Road Washout Warning System custom-engineered for the **Tassajara Zen Mountain Center (TZMC)** in the Santa Lucia Mountains of Monterey County, California.

---

## Table of Contents
1. [Geographic & Physical Context](#geographic--physical-context)
2. [System Architecture Overview](#system-architecture-overview)
3. [Santa Lucia 5-Station Boundary Array](#santa-lucia-5-station-boundary-array)
4. [MOS Machine Learning Architecture](#mos-machine-learning-architecture)
5. [Early Flood & Road Washout Warning System](#early-flood--road-washout-warning-system)
6. [Hardware Roadmap & Pending Sensor Telemetry](#hardware-roadmap--pending-sensor-telemetry)
7. [Database Schema & Migrations](#database-schema--migrations)
8. [Installation & Local Setup](#installation--local-setup)
9. [Operational Execution (Cron & GitHub Actions)](#operational-execution-cron--github-actions)
10. [Streamlit Dashboard UI](#streamlit-dashboard-ui)
11. [Future Directions & Methodological Improvements](#future-directions--methodological-improvements)

---

## Geographic & Physical Context

Tassajara Zen Mountain Center sits deep along the canyon floor of Tassajara Creek at **$485\text{ m}$ ($1,590\text{ ft}$) MSL**, surrounded by precipitous ridges rising to over **$1,500\text{ m}$ ($5,000\text{ ft}$)** (Chews Ridge, Pine Ridge, Black Cone, and Ventana Cone). Access is governed exclusively by a rugged, unpaved 14-mile mountain road crossing a $1,536\text{ m}$ summit from Jamesburg.

```
Elevation (m MSL)
1600m ─── [CHWC1: Chews Ridge 1536m] ────────── [VTCC1: Ventana Cone 1453m]
          \                                    /
1200m ─────\────── High-Gradient Mountain Tributaries ──────/
            \                                              /
800m  ───────\── Operational NWP Model Reference Elevation (~880m MSL) ────
              \                                          /
485m  ─────────\── [TZMC Stone Office Vantage Pro2 485m] ─/ (Deep Canyon Cold-Pool)
                \                                        /
200m  ───────────\── [ASRC1: Arroyo Seco 268m] ───────── [PPSC1: Big Sur 108m Coast]
```

### Microclimatic & Hydrological Complexities
* **Topographic Shading:** The southern canyon wall (Black Cone Trail ridge, 32° horizon) creates an early afternoon solar shadow, accelerating radiational cooling hours before sunset.
* **Nocturnal Katabatic Drainage:** Cool air pooling along Tassajara Creek decouples the canyon floor from the regional boundary layer, driving overnight lows $5^\circ\text{F}$ to $15^\circ\text{F}$ below coarse regional forecasts.
* **Orographic Spillover & Feeder-Seeder Rain:** Pacific atmospheric rivers encounter the coastal Santa Lucia crest, wringing out intense precipitation that spills across the divide into upper Tassajara headwaters while the valley floor sits in partial rain-shadow.
* **Flash Floods & Road Slipouts:** Extreme rainfall on rocky, steep Mediterranean chaparral/oak catchments triggers high-velocity creek surges, footbridge log-jamming, and catastrophic slipouts along Tassajara Road.

Standard Numerical Weather Prediction (NWP) models (e.g., $2.5\text{ km}$ HRRR/GFS grids centered at $\sim 880\text{ m}$ MSL) smooth over these gradients, requiring localized MOS calibration and dynamic hydrological routing.

---

## System Architecture Overview

The system operates on an automated hourly cycle that ingests live telemetry, pulls regional forecasts, executes statistical learning ensembles, routes surface runoff, and updates interactive dashboards.

```
                              ┌───────────────────────────────────────────────┐
                              │  Davis WeatherLink v2 API (/v2/current)       │
                              │  Station 36966 (Stone Office Davis VP2, 485m) │
                              └───────────────────────┬───────────────────────┘
                                                      │
┌───────────────────────────────────────┐             │             ┌──────────────────────────────────────────┐
│ Synoptic Data API (MesoWest v2)       │             │             │ NWS / NWP Hourly Operational Forecast    │
│ 5-Station Santa Lucia Boundary Array  │             │             │ HRRR/GFS Grid @ 880m MSL via Open-Meteo  │
└──────────────────┬────────────────────┘             │             └────────────────────┬─────────────────────┘
                   │                                  │                                  │
                   ▼                                  ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                run_hourly_mos_pipeline.py (Hourly Orchestrator)                              │
│  - 48-hr non-destructive historical catch-up & 24-hr console tipping-bucket shortfall reconciliation         │
│  - Microclimate boundary diagnostics (lapse rate, gorge drainage, virga, radiation shield/cone clog QC)     │
│  - Feature engineering (25 astronomical, topographic, thermodynamic, and spatial gradient predictors)        │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               mos_model.py (TassajaraMOSCorrector Ensemble)                                  │
│  1. Thermal Subsystem: HistGradientBoostingRegressor (Mean + P10/P90 Quantile Bands)                         │
│  2. Hydrological Hurdle: Calibrated HistGradientBoostingClassifier (PoP) + Poisson/P90 QPF Regressors        │
│  3. Econometric Baseline: Standardized Newey-West HAC OLS Regression (cov_type='HAC', maxlags=12)          │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         hydro_risk_engine.py (Early Flood & Road Washout Warning System)                     │
│  - Dynamic Antecedent Precipitation Index: API_t = API_{t-1} * k + P_t (k in [0.88, 0.94] based on VPD)      │
│  - Nonlinear Runoff Partitioning: Sigmoid coefficient C in [0.08, 0.86]                                      │
│  - Kinematic Quickflow Unit Hydrograph: 72 km^2 upper basin routed to Tassajara Creek Discharge (CFS)        │
│  - 4-Tier Operational Severity Matrix: GREEN, YELLOW, ORANGE, RED                                            │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       SQLite Database (tassajara_weather.db)                                 │
│  Tables: local_observations, reference_observations, nws_forecasts, mos_corrected_forecasts,                │
│          hydrological_risk_state, hydrological_forecast_series, stream_gauge_observations                    │
│  Views:  vw_mos_training_dataset, vw_creek_flood_monitor                                                     │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          app.py (Streamlit Web Dashboard)                                    │
│  Mode 1: Everyday Community Forecast (Jargon-free 7-day cards, road advisory badges, smooth curves)          │
│  Mode 2: Scientific & Statistical MOS Suite (8-metric grid, 3-panel Plotly suite, OLS & feature importances) │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Santa Lucia 5-Station Boundary Array

To monitor atmospheric layering, the pipeline continuously tracks a 5-station boundary network via the Synoptic Data API:

| Station ID | Station Name | Elev (m / ft) | Network | Geographic Role & Boundary Function |
| :--- | :--- | :--- | :--- | :--- |
| **`CHWC1`** | Chews Ridge | $1,536\text{ m}$ ($5,040\text{ ft}$) | RAWS | **Interior Crest (NNW):** Free-troposphere lapse rates and feeder-seeder ridge precipitation. |
| **`VTCC1`** | Ventana Cone | $1,453\text{ m}$ ($4,766\text{ ft}$) | CNRFC / HADS | **Ventana Crest (WNW):** Windward crest moisture flux and southwesterly orographic spillover. |
| **`CAHC1`** | Hastings Reserve | $546\text{ m}$ ($1,792\text{ ft}$) | RAWS | **Foothill Valley (N):** Regional inland valley temperature and cold-pool baseline. |
| **`ASRC1`** | Arroyo Seco | $268\text{ m}$ ($879\text{ ft}$) | RAWS | **Gorge Outlet (6.3 km E):** Down-canyon drainage outlet and thermal boundary contrast. |
| **`PPSC1`** | Big Sur State Park | $108\text{ m}$ ($354\text{ ft}$) | RAWS | **Coastal Slope (W):** Marine layer stratus cap index and cross-range marine contrast. |

### Symmetric Cross-Station Fallback Chaining
When remote stations experience power or satellite outages, the pipeline utilizes elevation-scaled cross-station fallbacks:
* $\text{VTCC1} \longleftrightarrow \text{CHWC1}$ ($\Delta z = 83\text{ m}$, offset $0.54^\circ\text{C}$)
* $\text{ASRC1} \longleftrightarrow \text{CAHC1}$ ($\Delta z = 278\text{ m}$, offset $1.80^\circ\text{C}$ to $2.01^\circ\text{C}$)
* If entire ridges lose telemetry, fallbacks anchor strictly to the issue-time Stone Office temperature ($T_{\text{issue}}$) with standard-atmosphere environmental lapse offsets ($\Gamma = +6.5^\circ\text{C/km}$), preventing artificial subsidence inversions from corrupting predictions.

---

## MOS Machine Learning Architecture

The statistical engine (`mos_model.py`) trains on paired historical observations and NWP forecast grids using 25 engineered features.

### 1. Thermal Subsystem
* **GBDT Mean Error Regressor:** `HistGradientBoostingRegressor(loss='squared_error', max_iter=350, learning_rate=0.04, l2_regularization=1.5)`
* **Quantile Uncertainty Envelope:** Dual quantile estimators (`loss='quantile'`, $\alpha = 0.10$ and $\alpha = 0.90$) modeling the 80% confidence interval ($P_{10}$ cold-pool floor to $P_{90}$ thermal ceiling).
* **Lead-Time Decay:** Initialized issue-time anomalies (e.g., $\Delta T_{\text{persis}}$, lapse rate anomalies, cold-pool confinement) decay exponentially toward climatological neutral states:
  $$\Gamma_{\text{eff}} = 6.5 + (\Gamma_0 - 6.5) \cdot e^{-\text{lead\_hrs} / 12.0}$$

### 2. Two-Stage Hydrological Hurdle Subsystem
* **Stage 1 (PoP Classifier):** `CalibratedClassifierCV` wrapping an L2-regularized `HistGradientBoostingClassifier(loss='log_loss')` estimating $P(\text{Rain} \ge 0.254\text{ mm})$.
* **Stage 2 (QPF Regressors):** A Poisson-deviance regressor for conditional accumulation paired with a 90th-percentile extreme quantile regressor for heavy downpour rates.
* **Analytical Virga & Orographic Fallback:** If raw NWP QPF is zero during high-PoP pre-frontal setups, the system synthesizes conditional accumulation modulated by moisture flux and sub-cloud vapor pressure deficit:
  $$F_{\text{virga}} = \exp(-0.48 \cdot \text{VPD}), \quad F_{\text{oro}} = 1.0 + 0.08 \cdot \text{Flux}_{\text{upslope}} + 0.05 \cdot (-\Delta P_{3\text{h}})$$

### 3. Econometric HAC OLS Diagnostic Baseline
To maintain physical interpretability and detect collinearity, predictors are standardized and orthogonalized ($R^2 > 0.96$) to guarantee a design matrix condition number $\kappa < 30$. Coefficients are estimated via Ordinary Least Squares with 12-hour lag Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors.

---

## Early Flood & Road Washout Warning System

Implemented in `hydro_risk_engine.py`, this module models continuous catchment saturation and runoff delivery into Tassajara Creek.

### 1. Antecedent Precipitation Index (API)
The catchment soil wetness state ($API_t$, in mm or inches) is tracked via an exponential recession bucket model:
$$API_t = API_{t-1} \cdot k + P_{\text{eff}, t}$$
* **Recession Constant ($k$):** Dynamically modulated between $0.88$ and $0.94$ based on boundary-layer Vapor Pressure Deficit (VPD):
  $$k = \text{clip}(0.935 - 0.025 \cdot \min(\text{VPD}, 2.2), 0.88, 0.94)$$
* **Effective Catchment Precipitation ($P_{\text{eff}}$):** Blends the valley gauge clicks with ridge-top spillover from Chews Ridge, Ventana Cone, and Big Sur:
  $$P_{\text{eff}} = \max\left(P_{\text{tass}}, \, 0.70 \cdot P_{\text{tass}} + 0.30 \cdot P_{\text{crest}}\right)$$

### 2. Nonlinear Runoff Partitioning ($C$)
The fraction of incoming rainfall converted directly into surface quickflow ($R_t = C_t \cdot P_{\text{eff}, t}$) follows a logistic soil-saturation function:
$$C(API) = C_{\min} + \frac{C_{\max} - C_{\min}}{1 + \exp\left(-\frac{API + 0.18 \cdot P_{72\text{h}} - API_{\text{mid}}}{s}\right)}$$
* $C_{\min} = 0.08$ (Dry fractured granite / chaparral duff abstraction)
* $C_{\max} = 0.86$ (Saturated canyon bedrock / regolith sheetflow ceiling)
* $API_{\text{mid}} = 32.0\text{ mm}$ ($1.26\text{ in}$ inflection point), $s = 11.5\text{ mm}$

### 3. Kinematic Unit Hydrograph Creek Routing
Hourly quickflow depth ($R_t$) across the $72\text{ km}^2$ upper watershed is routed into estimated creek discharge ($Q$, in Cubic Feet per Second) using a 3-hour steep-canyon kernel ($w_0 = 0.48, w_1 = 0.34, w_2 = 0.18$):
$$Q_t = Q_{\text{base}} + 155.4 \cdot \sum_{i=0}^2 w_i \cdot R_{t-i}$$
*(where $Q_{\text{base}} = 6.5\text{ CFS}$ represents standard autumn/winter baseline spring flow).*

### 4. Four-Tier Operational Severity Matrix

| Tier | Status Code | Rain & API Threshold Triggers | Tassajara Road Status (14-Mile Dirt Road) | Tassajara Creek & Footbridges | Operational Protocol |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **0** | **GREEN** | $API < 1.00"$, $24\text{h Rain} < 1.00"$, Rate $< 0.50"\text{/hr}$ | **Open / Dry to Damp**<br>Standard mountain precautions | **Stable Baseflow**<br>Footbridges clear | Routine canyon operations. Normal high-clearance travel. |
| **1** | **YELLOW** | $API \ge 1.00"$ ($25.4\text{ mm}$) OR $24\text{h Rain} \ge 1.00"$ | **Advisory / Caution**<br>Muddy, slick summit grades; minor rockfall | **Elevated Flow**<br>Turbid water; moderate rise | 4WD required on Tassajara Road. Clear leaves from culverts; inspect bridges. |
| **2** | **ORANGE** | Rain $\ge 0.50"\text{/hr}$ (or $P_{90} \ge 0.75"\text{/hr}$) on wet ground OR $24\text{h Rain} \ge 2.00"$ | **Hazardous / High Risk**<br>Travel strongly discouraged; slipouts likely | **Rapidly Rising**<br>Woody debris mobilization | Suspend non-emergency road travel. Inspect lower bridges for log dams. |
| **3** | **RED** | Rain $\ge 1.00"\text{/hr}$ (or $P_{90} \ge 1.25"\text{/hr}$) OR $72\text{h Rain} \ge 3.50"-4.00"$ | **CLOSED / IMPASSABLE**<br>Severe washouts, debris flows, road cuts | **BANKFULL / FLOOD**<br>Bridge overtopping surge hazard | **Immediate road closure.** Evacuate low creek structures; prohibit bridge crossings. |

---

## Hardware Roadmap & Pending Sensor Telemetry

The database schema and UI have been architected with native hooks for downstream physical monitoring hardware. Several fields in `stream_gauge_observations` and `vw_creek_flood_monitor` are explicitly designed to remain unpopulated until hardware commissioning is complete.

```
                                  CURRENT ARCHITECTURE (Virtual Runoff Routing)
┌──────────────────────────────────────────────┐
│  Stone Office Rain Spoon (ISS 36966)         │───────► [hydro_risk_engine.py] ───► Estimated Creek Discharge
│  + 5-Station Crest Spillover (CHWC1/VTCC1)   │          (API + Kinematic Hydrograph)  (discharge_index_cfs)
└──────────────────────────────────────────────┘

                                    FUTURE HARDWARE ROADMAP (Physical Ground Truth)
┌──────────────────────────────────────────────┐
│  Ultrasonic / Radar Non-Contact Level Gauge  │───────► [stream_gauge_observations]
│  Mounted at Tassajara Creek Bathhouse Bridge │          - stream_stage_m
└──────────────────────────────────────────────┘          - rate_of_rise_m_hr
                                                          - water_temp_c / battery_v
```

### Unpopulated Data Fields Awaiting Hardware Commissioning

The following database fields represent planned physical measurement nodes:

* **`stream_stage_m` & `stream_stage_ft`:** Physical creek stage height above the datum. Currently unpopulated (`NULL`). The engine currently operates in **Virtual Kinematic Runoff Mode**, deriving stage hazard from rainfall accumulation and catchment saturation.
* **`rate_of_rise_m_hr`:** Rate of water level change computed between consecutive ultrasonic pings. Currently unpopulated (`NULL`).
* **`water_temp_c`:** In-stream thermistor probe for aquatic temperature tracking. Currently unpopulated (`NULL`).
* **`battery_v` & `sensor_health`:** Battery supply voltage and hardware heartbeat from the remote solar/LoRa telemetry logger. Currently defaults to `"UNINSTALLED (MOS + Tipping Bucket Mode)"`.

### Proposed Equipment Specifications
* **Sensor Node:** Non-contact SDI-12 / Modbus Radar Level Sensor (e.g., Campbell Scientific CS475A or Valarm Ultrasonic) installed on the downstream stringer of the old bathhouse footbridge.
* **Datalogger & Telemetry:** Solar-powered ESP32 / LoRaWAN remote node transmitting stage height at 5-minute intervals to the Stone Office local gateway, persisting directly to `stream_gauge_observations`.
* **Automatic Failover:** When physical stream gauge telemetry goes live, `evaluate_flood_and_road_tier()` will automatically incorporate physical stage thresholds ($0.75\text{ m}$, $1.15\text{ m}$, $1.60\text{ m}$) and surge rates ($\ge 0.40\text{ m/hr}$) directly into the alert arbitration logic without requiring code changes.

---

## Database Schema & Migrations

The SQLite database (`tassajara_weather.db`) operates in **Write-Ahead Logging (`WAL`)** mode for concurrent access between the hourly ingestion background workers and the Streamlit frontend.

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

### Core Tables
* **`local_observations`:** Ground-truth records from Station 36966 (Stone Office Davis VP2). Stores temperature, dewpoint, wetbulb, pressure, wind vector, solar radiation, and tipping-bucket rainfall (`precip_1hr_mm`).
* **`reference_observations`:** Synchronized hourly records from the 5 Santa Lucia reference stations (`CHWC1`, `VTCC1`, `CAHC1`, `ASRC1`, `PPSC1`).
* **`nws_forecasts`:** Operational 7-day hourly NWP forecast grids initialized every 6 hours.
* **`mos_corrected_forecasts`:** MOS-corrected temperature, quantile bounds ($P_{10}$/$P_{90}$), calibrated PoP (%), and expected/P90 QPF rates.
* **`hydrological_risk_state`:** Hourly history of catchment saturation ($API$), runoff coefficients ($C$), peak runoff rates, discharge index (CFS), and 4-tier alert status.
* **`hydrological_forecast_series`:** Forward 72-hour projected routed hydrograph and API progression.
* **`stream_gauge_observations`:** Physical stream gauge telemetry table (configured for future sensor commissioning).

### Production Views
* **`vw_mos_training_dataset`:** Aligns historical observations with valid-time forecasts and issue-time boundary references for continuous model retraining.
* **`vw_creek_flood_monitor`:** Operational view uniting live atmospheric observations, catchment soil moisture ($API$), routed creek discharge, and alert status codes.

---

## Installation & Local Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/MT-Wildhack/tassajara-weather.git
cd tassajara-weather

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Requirements (`requirements.txt`)
```text
streamlit>=1.35.0
pandas>=2.2.0
numpy>=1.26.0
requests>=2.31.0
plotly>=5.20.0
scikit-learn>=1.4.0
joblib>=1.3.0
statsmodels>=0.14.0
```

### 3. Initialize SQLite Schema
```bash
sqlite3 tassajara_weather.db < schema.sql
```

### 4. API Credentials Configuration
Set the following environment variables (or rely on built-in pipeline defaults):
```bash
export WL_API_KEY="your_weatherlink_v2_api_key"
export WL_API_SECRET="your_weatherlink_v2_api_secret"
export SYNOPTIC_TOKEN="your_synoptic_mesowest_token"
```

---

## Operational Execution (Cron & GitHub Actions)

### Hourly Ingestion & Inference Pipeline
To run the end-to-end data acquisition, MOS inference, and hydrological risk evaluation cycle:
```bash
python3 run_hourly_mos_pipeline.py
```

### Standalone Hydrological Risk Evaluation
To calculate catchment saturation ($API$) and update the 4-tier flood alert status independently:
```bash
python3 hydro_risk_engine.py
```

### Model Retraining Pipeline
To retrain the 25-feature MOS gradient-boosted ensemble and refit the Newey-West HAC OLS baseline:
```bash
python3 backfill_and_train.py --days 730
```

---

## Streamlit Dashboard UI

Launch the interactive monitoring suite locally:
```bash
streamlit run app.py
```

The application provides two purpose-built display modes:

### 1. Everyday Forecast (Community & Guest View)
* **High-Visibility Flood & Road Warning Banner:** Clear, color-coded status (Green, Yellow, Orange, Red) with actionable guidance for Tassajara Road travel and creek footbridges.
* **Topographic Shading Indicator:** Real-time solar badge indicating whether the canyon floor is in direct sunlight, nocturnal katabatic drainage, or Black Cone ridge shadow.
* **Simplified 7-Day & 24-Hour Outlook:** Plain-language weather summaries, freeze danger warnings ($P_{10} \le 32^\circ\text{F}$), and 3-hour interval cards.
* **Interactive Creek Runoff & Hydrograph Preview:** Displays expected creek flow alongside precipitation probability.

### 2. Scientific & Statistical MOS Suite
* **8-Metric Telemetry Grid:** Real-time dry bulb, relative humidity, Stull wet-bulb, Tetens vapor pressure deficit, anemometer wind vectors, MSLP tendency, crest lapse rates ($\Gamma$, in $^\circ\text{C/km}$), and gorge confinement differentials.
* **3-Panel Plotly Time-Series Suite:**
  * *Panel 1:* Verification actuals vs. MOS Quantile Fan ($P_{10}$–$P_{90}$) vs. Raw NWP grid curves.
  * *Panel 2:* 10m peak wind gusts (mph) paired with relative humidity (%).
  * *Panel 3:* MOS calibrated QPF, $P_{90}$ extreme downpour rates, observed rain, and calibrated PoP (%).
* **Hydrological Diagnostic Suite:** Soil wetness ($API$) gauge indicator alongside 72-hour routed creek discharge hydrograph (CFS).
* **Model Interpretability & HAC OLS Diagnostics:** Permutation feature importance rankings for both thermal and hydrological subsystems, accompanied by standardized Newey-West HAC regression metrics ($R^2$, condition number $\kappa$, and coefficient confidence intervals).

---

## Future Directions & Methodological Improvements

While the current MOS architecture and kinematic runoff model provide robust, localized guidance, continuous improvement of the Early Flood Warning System is paramount given the extreme vulnerability of the Tassajara canyon floor and access road.

### Methodological Alternatives & Enhancements
* **Deep Learning (LSTMs / Transformers):**
  * *Current Approach:* Gradient Boosted Decision Trees (GBDT).
  * *Alternative:* Long Short-Term Memory (LSTM) networks or time-series Transformers.
  * *Pros/Cons:* Deep learning excels at capturing complex, non-linear temporal sequences (like the lag between rainfall and peak discharge). However, LSTMs require vast amounts of uninterrupted training data—which is difficult to secure with only ~2 years of local station history—and lack the strict physical interpretability offered by GBDT permutation importance and our HAC OLS baseline.
* **Distributed Physically-Based Hydrological Models (e.g., WRF-Hydro, HEC-HMS):**
  * *Current Approach:* Empirical Antecedent Precipitation Index (API) and kinematic unit hydrograph routing.
  * *Alternative:* Fully distributed models resolving 3D soil moisture grids, groundwater percolation, and channel routing.
  * *Pros/Cons:* While physically rigorous, setting up HEC-HMS requires high-resolution bathymetry, soil porosity surveys, and land-cover rasters for the Ventana Wilderness that are currently unavailable at the necessary resolution. The empirical API bucket model remains the most practical and computationally lightweight approach for a single, un-gauged canyon point.
* **Probabilistic Ensemble Forcing (HREF / GEFS):**
  * Transitioning from deterministic single-run NWP inputs (HRRR/GFS) to ensemble datasets (e.g., High-Resolution Ensemble Forecast) would allow the system to natively compute the probability of exceeding critical QPF thresholds rather than relying solely on the single-model $P_{90}$ statistical bound.

### Improving the Creek Flood Warning System (Equipment & Data)
To transition from *virtual* runoff routing to *verified* flood tracking, the following equipment and data integrations are prioritized:

1. **Deploying Physical Stream Telemetry (Stage & Flow):**
   * *Equipment:* SDI-12 non-contact Radar or Ultrasonic level sensors (e.g., Campbell Scientific CS475A) mounted to the old bathhouse footbridge.
   * *Impact:* Replaces theoretical CFS routing with actual stage height ($m$) and rate of rise ($m/hr$). This enables the development of an empirical rating curve to calibrate the runoff coefficient ($C$) perfectly to real-world watershed delivery times.
2. **In-Situ Soil Moisture Probes:**
   * *Equipment:* Deployment of volumetric water content (VWC) sensors at varying soil depths along the canyon slopes.
   * *Impact:* Allows the system to phase out the mathematically derived Antecedent Precipitation Index ($API_t$) in favor of direct soil saturation measurements, radically improving the accuracy of the nonlinear runoff partitioning function.
3. **Upstream Headwater Rain Gauges:**
   * *Equipment:* Placing ruggedized tipping buckets higher up the immediate Black Cone and Pine Ridge catchments.
   * *Impact:* Currently, the system relies on Chews Ridge (`CHWC1`) and Ventana Cone (`VTCC1`) to infer orographic spillover. Localized headwater gauges would capture the exact rain bands falling into the immediate Tassajara Creek tributaries.
4. **Dynamic Fire Hydrology (Burn Scar Ingestion):**
   * *Data Source:* Automated ingestion of post-wildfire burn severity mapping (e.g., BAER soil burn severity grids).
   * *Impact:* Wildfires create hydrophobic soil layers that drastically alter the watershed's infiltration capacity. Incorporating time-decaying burn severity layers would allow the model to dynamically scale the runoff coefficient ($C_{\max}$) to account for post-fire debris flow risks.

---

## License & Operational Responsibility
Maintained for the safety and operational resilience of the **Tassajara Zen Mountain Center (San Francisco Zen Center)** community, staff, students, and guests. Distributed under the MIT License.

**Operational Disclaimer:** The computations, forecasts, and alert tiers (Green, Yellow, Orange, Red) generated by this software serve strictly as **advisory decision-support guidance**. They do not constitute an official warning and do not supersede ground verification, physical road conditions, or official directives issued by the **National Weather Service (NWS)**, **Monterey County Office of Emergency Services (OES)**, **US Forest Service (USFS)**, or the **TZMC Disaster Council**.
