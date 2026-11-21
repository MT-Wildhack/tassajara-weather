# Tassajara Zen Mountain Center (TZMC) — Microclimate & Hydrological MOS Forecast Suite

[![Python 3.11 / 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-tassajara--weather.streamlit.app-FF4B4B.svg)](https://tassajara-weather.streamlit.app)
[![Database](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://sqlite.org)
[![Model Architecture](https://img.shields.io/badge/Architecture-MOS__GBM__v3__HYDRO%20%2B%20HAC%20OLS-orange.svg)](#5-mos-machine-learning-architecture)
[![Hydrology](https://img.shields.io/badge/Hydrology-Two--Pathway%20API%20Routing-0284C7.svg)](#6-early-flood--road-washout-warning-system)

An operational, publication-grade meteorological **Model Output Statistics (MOS)** forecast suite and **Early Flood & Road Washout Warning System** custom-engineered for the **Tassajara Zen Mountain Center (TZMC)** in the Santa Lucia Mountains of Monterey County, California.

* **Live Cloud Dashboard:** [https://tassajara-weather.streamlit.app](https://tassajara-weather.streamlit.app)

This README serves as both the scientific reference and the complete **Maintainer Handover Manual**, covering cloud automation, dual-baseline NWP/MOS architecture, database operations, model retraining, a data-gated future modeling timetable, and dedicated local Mini PC / Raspberry Pi deployment integrated with the Stone Office **Davis Vantage Pro2 Cabled Console** and its attached **WeatherLinkIP (`6555`) Ethernet Data Logger** for uninterrupted storm-mode operation if internet connectivity drops.

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
12. [Future Directions & Model Improvement Timetable](#12-future-directions--model-improvement-timetable)
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
800m  ───────\── NWP Reference Elevation (~880m MSL: Open-Meteo + NWS MTR) ──
              \                                          /
485m  ─────────\── [TZMC Stone Office Davis VP2 Cabled 485m] ─/ (Deep Canyon Cold-Pool)
                \      └─ WeatherLinkIP (6555) Ethernet      /
200m  ───────────\── [ASRC1: Arroyo Seco 268m] ───────────── [PPSC1: Big Sur 108m Coast]
```

### Microclimatic & Hydrological Complexities

* **Topographic Shading:** The southern canyon wall (Black Cone Trail ridge, $32^\circ$ horizon across $145^\circ\text{–}255^\circ$ azimuth) casts an early afternoon solar shadow—and nearly grazes solar noon in deep winter—accelerating radiational cooling hours before astronomical sunset.
* **Nocturnal Katabatic Drainage:** Cool air pooling along Tassajara Creek decouples the canyon floor from the free atmosphere, pushing overnight lows $5^\circ\text{F}$ to $15^\circ\text{F}$ colder than regional model grids on clear nights.
* **Orographic Spillover & Feeder-Seeder Rain:** Pacific atmospheric rivers intercept the coastal Santa Lucia crest, wringing out intense headwater rainfall that spills into upper Tassajara Creek tributaries while the valley floor sits in partial rain-shadow.
* **Flash Floods & Road Slipouts:** Multi-inch storms over steep, fractured granite and clay-rich mountain slopes trigger rapid creek surges, woody debris damming at lower footbridges, and hazardous slipouts along Tassajara Road.

---

## 2. System Architecture & Repository Map

The suite runs on an automated hourly cycle (`:05` past each hour via GitHub Actions and/or a local `systemd` timer, plus a weekly Sunday `11:25 UTC` MOS retrain), pulling Stone Office telemetry from the **Davis Vantage Pro2 Cabled Console** (Station `36966`) via its attached **WeatherLinkIP (`6555`) Ethernet Data Logger**, 5-station boundary observations, and dual-baseline hourly forecasts (official NOAA NWS `/gridpoints/MTR` + matched Open-Meteo `880 m` thermal/wind/sky companion fields); executing MOS inference and two-pathway hydrological routing; and persisting everything to SQLite (`tassajara_weather.db`) consumed read-only by the Streamlit UI.

```
                              ┌──────────────────────────────────────────────────────────┐
                              │  Stone Office Davis Vantage Pro2 Cabled Console (36966)  │
                              │  Attached Logger: Davis WeatherLinkIP (6555) Ethernet    │
                              │  • Normal Mode: WeatherLink v2 Cloud API (current + 48h) │
                              │  • Offline Mode: Direct LAN TCP Socket (Port 22222)      │
                              └────────────────────────────┬─────────────────────────────┘
                                                           │
┌───────────────────────────────────────┐                  │             ┌──────────────────────────────────────────┐
│ 5-Station Santa Lucia Boundary Array  │                  │             │ Dual-Baseline 168-hr Hourly Forecast     │
│ Tier 1: Synoptic API (optional)       │                  │             │ • Official NOAA NWS /gridpoints/MTR      │
│ Tier 2A: NOAA/NWS Obs (<= 3h fresh)   │                  │             │   (benchmark temp + official QPF/PoP)    │
│ Tier 2B: Free IEM CA_DCP (CHWC1/VTCC1)│                  │             │ • Matched Open-Meteo 880m (om_* columns) │
│ Tier 3: Open-Meteo (flagged fallback) │                  │             │   (domain-matched thermal MOS input)     │
└──────────────────┬────────────────────┘                  │             └────────────────────┬─────────────────────┘
                   │                                       │                                  │
                   ▼                                       ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                run_hourly_mos_pipeline.py (Hourly Orchestrator)                              │
│  - 48-hr WL_HISTORIC catch-up + mass-conserving clock-hour rain delta (console_day_mm - prior_today_mm)      │
│  - Guarded same-day (<= 2h) IEM precip_today[in] differencing for crest ALERT gauges (CHWC1, VTCC1)          │
│  - Boundary-layer diagnostics (crest lapse rate, gorge confinement, virga VPD, continuous 3h MSLP tendency)  │
│  - 33-predictor feature engineering (radiative cloud-transmittance, topographic, thermodynamic, spatial)     │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       mos_model.py (MOS_GBM_v3_HYDRO — TassajaraMOSCorrector Ensemble)                       │
│  1. Thermal Subsystem: Dual-baseline parity on Open-Meteo 880m (Mean + P10/P90 lead-inflated CQR envelope)   │
│  2. Hydrological Hurdle: Dry-fold-guarded Calibrated PoP + Stage-2 Bounded Log-Ratio QPF & P90 Regressors    │
│  3. Econometric Baseline: Standardized Newey-West HAC OLS (maxlags=12, collinearity pruned at max_r2=0.85)   │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         hydro_risk_engine.py (Early Flood & Road Washout Warning System)                     │
│  - 30-day (720h) spin-up; hourly API decay k_hr = k_day^(1/24) (k_day in [0.86, 0.94] via VPD)               │
│  - Nonlinear Sigmoid Runoff C(API) in [0.03, 0.68] after 1.00 mm/hr abstraction; online-only crest spillover │
│  - Mass-Conserving Two-Pathway Router (72 km^2): 18% 4-hr Quickflow + 57% Interflow (k=0.955) + 25% Deep GW  │
│  - Post-routing 4-Tier Matrix: GREEN, YELLOW, ORANGE (<= 24h, >= 250 CFS), RED (<= 6h, >= 500 CFS)           │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     SQLite Database (tassajara_weather.db)                                   │
│  Tables: local_observations, reference_observations, nws_forecasts (+ om_*), mos_corrected_forecasts,        │
│          hydrological_risk_state, hydrological_forecast_series, stream_gauge_observations, ingest_state      │
│  Views:  vw_mos_training_dataset (causal, pure Open-Meteo 880m), vw_creek_flood_monitor                      │
└──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                app.py (Streamlit Web Dashboard — [https://tassajara-weather.streamlit.app](https://tassajara-weather.streamlit.app))                    │
│  Read-only SQLite URI (?mode=ro) + @st.cache_resource model loader (visible joblib fallback warning)         │
│  Mode 1: Everyday Forecast (Community & Guest View — plain-language cards, road/creek alerts, freeze lows)   │
│  Mode 2: Scientific & Statistical MOS Suite (8-metric grid, 3-panel Plotly, exact-match verification, OLS)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Repository File Map

| File / Path | Responsibility |
| :--- | :--- |
| `ingest_weather.py` | WeatherLink v2 API client (Station `36966`, order-independent `max_iss_ts`, retry/backoff) and 4-tier 5-station reference fetcher (Synoptic $\rightarrow$ NWS $\rightarrow$ IEM `CA_DCP` $\rightarrow$ Open-Meteo) with 3-hour staleness guards and strict provenance isolation. |
| `ingest_nws.py` | Dual-baseline 168-hour forecast fetcher: resolves NOAA NWS `/gridpoints/MTR` + `/forecastHourly`, disaggregates multi-hour NDFD QPF blocks using Open-Meteo temporal weights, guards against NDFD QPF dropouts, persists matched Open-Meteo `880 m` `om_*` companion columns, and prunes superseded forecast batches. |
| `run_hourly_mos_pipeline.py` | Main hourly orchestrator: fetches 48h `WL_HISTORIC` + live `WL_LIVE`, computes mass-conserving clock-hour rain, syncs 5 reference stations, generates 168h `MOS_GBM_v3_HYDRO` forecasts, evaluates `hydro_risk_engine.py`, and supports `--skip-alerts` for idempotent CI push retries. |
| `mos_model.py` | Defines `TassajaraMOSCorrector`, 33-predictor engineering (including radiative cloud-attenuated insolation), dual-baseline `om_*` routing, fold-internal synthetic lead augmentation, lead-inflated OOF CQR quantiles, bounded log-ratio Stage-2 QPF, Fold-5 OOF permutation importances, and Newey-West HAC OLS diagnostics. |
| `hydro_risk_engine.py` | 30-day historical API spin-up, hourly-decay $API$, logistic runoff coefficient ($C$), two-pathway quickflow/interflow creek router (CFS), strict `now`-divider handoff, post-routing 4-tier severity matrix, and digit-stripped trigger deduplication. |
| `backfill_and_train.py` | Historical WeatherLink Pro backfill (with AirLink filtering and timestamp-exact 3-hour MSLP differencing), causal `vw_mos_training_dataset` view management, model training, OOF-safe (`INSERT OR IGNORE`) verification writes, and serialization to `models/tassajara_mos_v1.joblib`. |
| `app.py` | Dual-mode Streamlit dashboard (`Everyday Forecast` & `Scientific MOS Suite`), read-only SQLite URI access, `@st.cache_resource` model caching, interactive Plotly visualizations, and exact-pair verification scorecard. |
| `requirements.txt` | Python 3.11 / 3.14 wheel-backed dependency bounds ensuring fast cloud builds and `joblib` binary compatibility between GitHub Actions and Streamlit Community Cloud. |
| `models/tassajara_mos_v1.joblib` | Serialized production `MOS_GBM_v3_HYDRO` ensemble artifact (33 predictors). |
| `tassajara_weather.db` | Production SQLite database storing observations, dual-baseline forecasts, MOS predictions, hydro states, and monotonic `ingest_state` counters. |
| `.github/workflows/` | Automated GitHub Actions workflow (`hourly_weather.yml`) running at `:05` hourly and Sunday `11:25 UTC` weekly retrains, with safe push retries (`--skip-alerts`) and `PRAGMA journal_mode=DELETE` checkpointing before Git commits. |

---

## 3. Santa Lucia 5-Station Boundary Array & Fallback Cascade

The pipeline monitors five regional stations surrounding the Tassajara watershed to resolve vertical thermal stratification, coastal marine intrusion, and headwater rainfall:

| Station ID | Station Name | Elev (m / ft) | Primary Live Source | Sensor Suite | Boundary Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CHWC1`** | Chews Ridge | $1,536\text{ m}$ ($5,039\text{ ft}$) | `IEM_CA_DCP` / `SYNOPTIC` | Temp (intermittent on rain-only bulletins), Rain (ALERT/RAWS) | **Interior Crest (NNW):** Primary crest lapse rate ($^\circ\text{C/km}$), summit freeze monitoring, and live headwater tipping-bucket rain. |
| **`VTCC1`** | Ventana Cone | $1,453\text{ m}$ ($4,767\text{ ft}$) | `IEM_CA_DCP` | **Precipitation-Only** (MPWMD ALERT) | **Ventana Crest (WNW):** Windward headwater tipping-bucket gauge (`Online (Rain Gauge)`). |
| **`CAHC1`** | Hastings Reserve | $546\text{ m}$ ($1,791\text{ ft}$) | `NWS_OBS` / `IEM_CA_DCP` | Temp, RH, Wind, Rain | **Foothill Valley (N):** Regional inland valley thermal differential ($\Delta T$ in $^\circ\text{F}$). |
| **`ASRC1`** | Arroyo Seco | $268\text{ m}$ ($879\text{ ft}$) | `NWS_OBS` / `IEM_CA_DCP` | Temp, RH, Wind, Rain | **Gorge Outlet (6.3 km E):** Down-canyon drainage outlet and cold-pool confinement contrast. |
| **`PPSC1`** | Big Sur State Park | $108\text{ m}$ ($354\text{ ft}$) | `NWS_OBS` / `IEM_CA_DCP` | Temp, RH, Wind, Rain | **Coastal Slope (W):** Marine layer inversion cap and coastal-to-canyon thermal gradient. |

### Four-Tier Zero-Cost Resilient Ingestion Cascade

To guarantee $5/5$ live telemetry without requiring a paid Synoptic subscription, `fetch_regional_reference_stations()` cascades automatically:

1. **Tier 1 — Synoptic Data API (`SYNOPTIC`):** Attempted first if `SYNOPTIC_TOKEN` has active Weather Data API access; strictly requests `precip_accum_one_hour` (never water-year cumulative totals).
2. **Tier 2A — NOAA/NWS Observations API (`NWS_OBS`):** Free, no-key government endpoint providing live observations for `CAHC1`, `ASRC1`, and `PPSC1`, rejecting any observation older than **3 hours**.
3. **Tier 2B — Iowa Environmental Mesonet (`IEM_CA_DCP`):** Free, no-key public RAWS/HADS/ALERT JSON feed (`https://mesonet.agron.iastate.edu/json/current.py?network=CA_DCP`) providing live observations for `CHWC1` and precipitation-only `VTCC1`, also rejecting observations older than **3 hours**. Computes RH from `dewpointtemp[F]` via the Magnus equation when available and tracks hourly rainfall increments from `precip_today[in]` **only** when the prior `ingest_state` checkpoint is on the same local calendar day and within **2 hours** (preventing post-gap full-day rain dumps).
4. **Tier 3 — Open-Meteo Real-Time Boundary Analysis (`OPENMETEO_SYNTH`):** Used **only** if a station fails all three live feeds. Marked `online=False` (`Model Fallback / Stale`, flagged with `*` in the UI) and **strictly excluded** from hydrological headwater spillover so synthetic model precipitation never fabricates creek runoff.

### Strict Provenance Isolation for Crest ALERT Gauges (`CHWC1` & `VTCC1`)

* `VTCC1` is a Monterey County / MPWMD ALERT precipitation-only gauge with **no physical thermometer**.
* `CHWC1` reports live temperature via IEM `CA_DCP`, but intermittently omits temperature on rain-only ALERT bulletins.
* Both stations retain their live `IEM_CA_DCP` tipping-bucket rain (`Online (Rain Gauge)` / `"Rain Gauge"` banner label when `temp_c` is null) without ever falling back wholesale to `OPENMETEO_SYNTH` or injecting synthetic Open-Meteo RH/wind values into `IEM_CA_DCP` rows.

---

## 4. Data Integrity, Provenance & Rainfall Accounting

Production storm auditing and first-principles code audits established the following data-integrity invariants across the pipeline:

* **Mass-Conserving Clock-Hour Rain Accounting (Zero Double-Counting):** Running at `:05` past the hour previously risked double-counting prior-hour rainfall into the new clock hour when using rolling `rainfall_last_60_min_mm` or additive console deltas. Both `run_hourly_mos_pipeline.py` and `app.py` now anchor each clock hour to the `WL_HISTORIC` archive base and compute live current-hour rain strictly by daily mass conservation:
  $$P_{\text{this\_hr}} = \max\!\left(P_{\text{hist\_hr}},\; \max\!\left(0,\; P_{\text{console\_day}} - \sum_{\text{midnight} \le h < \text{now}} P_{\text{obs}}(h)\right)\right)$$
  A dedicated midnight-lag guard compares `last_console_rain_day_mm` and monotonic `last_console_rain_year_mm` (supporting clean water-year resets to `0.0`) so an unreset console counter at `00:00 America/Los_Angeles` never dumps yesterday's rain into the new day.
* **Strict Causal Forecast Leads (Zero Target Leakage):** `ingest_nws.py` records `valid_time_utc == issue_time_utc` truthfully as `lead_time_hours = 0` (removing prior `max(1, ...)` clamping), and both `vw_mos_training_dataset` and `app.py` verification queries strictly enforce `lead_time_hours >= 1 AND issue_time_utc < valid_time_utc`.
* **Permanent Preservation of Genuine Operational Verification:** Weekly retrains (`backfill_and_train.py`) write historical verification rows strictly with `INSERT OR IGNORE` using genuine out-of-fold CV predictions (`MOS_GBM_v3_OOF`) while skipping any `(issue_time_utc, valid_time_utc)` pair that already holds a live operational forecast (`MOS_GBM_v3_HYDRO`). Active future batch updates are restricted to `valid_time_utc >= now_utc`.
* **Exact-Match Verification Joins:** `app.py` scores verification strictly on exact `(issue_time_utc, valid_time_utc)` pairs (`o.valid_time_utc < now_hour_utc`, `source != 'OPENMETEO_SYNTH'`), with no mismatched-lead `m_any` fallback and no future `hr_dt` reference-station leakage during hindcast fills.
* **Sensor Filtering & Timestamp-Exact Barometric Trends:** Both live and historic WeatherLink clients filter out indoor/health/AirLink AQI sensors (`stype in {323, 326, 504, 506}`), determine `max_iss_ts` order-independently, ignore legacy discrete `dst=2` console pressure-trend codes, and compute exact 3-hour MSLP tendencies on a regular 1-hour UTC grid so telemetry outages never stretch a 3-row diff across multi-day gaps.
* **NDFD QPF Dropout Protection:** If NOAA `/gridpoints/MTR` temporarily publishes `0.0 mm` across all hours while `max(PoP) >= 65%` and Open-Meteo forecasts `>= 2.0 mm`, `ingest_nws.py` logs a warning and automatically falls back to Open-Meteo QPF for that batch.
* **SQLite Concurrency & CI Safety:** All pipeline writers use `timeout=30`; `app.py` connects strictly via read-only URI (`file:tassajara_weather.db?mode=ro`); GitHub Actions checkpoints and switches to `PRAGMA journal_mode=DELETE` prior to committing `tassajara_weather.db` to Git; and CI push retries pass `--skip-alerts` to prevent duplicate SMS dispatches.

---

## 5. MOS Machine Learning Architecture

The production statistical engine (`mos_model.py`, version **`MOS_GBM_v3_HYDRO`**) uses **33 physical predictors** trained on genuine Davis Vantage Pro2 hourly observations (**$17,513$ domain-pure Open-Meteo `880 m` pairs**, **$876$ observed wet hours**, $>51\text{ in}$ verified $0.254\text{ mm}$ tipping-bucket rainfall).

### Dual-Baseline Thermal Parity Architecture (Option A)

Classical MOS theory (Glahn & Lowry) requires train-vs-inference NWP model parity. Direct diagnostic comparison over overlapping operational hours showed NOAA NWS `/gridpoints/MTR` running **$+4.4^\circ\text{F}$ to $+8.0^\circ\text{F}$ warmer** during afternoon hours (`12:00–15:00 LA`) than the Open-Meteo `880 m` training archive—meaning applying Open-Meteo's afternoon warming bias onto NOAA NWS temperatures double-corrected daytime highs.

To eliminate this mismatch immediately while accumulating a full annual cycle of native NOAA NWS data:
* **Thermal/Wind/Sky MOS Domain:** Both `vw_mos_training_dataset` and live operational thermal inference (`engineer_mos_features()` / `predict()`) run strictly on the matched **Open-Meteo `880 m`** baseline (`om_temp_c`, `om_dewpoint_c`, `om_rh_pct`, `om_wind_speed_ms`, `om_wind_dir_deg`, `om_sky_cover_pct`), with initialization persistence bias (`issue_persistence_bias_c`) computed against `om_temp_c`.
* **Official NOAA NWS Benchmark & Hydrology:** Official NOAA NWS `/gridpoints/MTR` temperatures (`nws_temp_c` $\rightarrow$ `mos_corrected_forecasts.raw_nws_temp_c`) remain stored on every row as the honest operational benchmark in the verification scorecard, and official NOAA `nws_qpf_1hr_mm` / `nws_pop_pct` continue to drive the hydrological hurdle.

### Verified Production Benchmarks (Purged 5-Fold Out-of-Fold CV, 48h Embargo)

* **Raw Generic Open-Meteo `880 m` RMSE:** $2.713^\circ\text{C}$ ($4.88^\circ\text{F}$)
* **Tassajara MOS Corrected OOF RMSE (with fold-internal multi-lead augmentation):** **$2.083^\circ\text{C}$ ($3.75^\circ\text{F}$)**
* **Full-Cycle Thermal Skill Improvement:** **$+23.2\%$ RMSE reduction** (leak-free, causal leads only)
* **HAC OLS Diagnostic Baseline:** $R^2 = 0.434$, $\text{Adj } R^2 = 0.433$, collinearity pruned at `max_r2 = 0.85` ($\text{VIF} \le 6.7$), 12-hour lag Newey-West standard errors (`cov_type='HAC', maxlags=12`).

### Subsystem Design

1. **Thermal Subsystem (`HistGradientBoostingRegressor`):**
   * Mean error corrector (`loss='squared_error'`, L2-regularized) paired with dual quantile estimators ($\alpha = 0.10$ and $\alpha = 0.90$) and asymmetric out-of-fold Conformalized Quantile Regression (CQR) pads.
   * **Radiative-Physics Predictors:** Supplements clear-sky canyon horizon geometry (`effective_insolation`, `in_canyon_shadow`) with three explicit cloud/moisture terms to suppress midday overcast warm bias:
     * `cloud_attenuated_insolation`: Kasten & Czeplak cloud transmittance $(1 - 0.75\,f_{\text{cloud}}^{3.4})\cdot e^{-0.35\,\text{QPF}}$ applied to canyon insolation.
     * `overcast_daytime_cooling_idx`: daytime insolation $\times\, f_{\text{cloud}}^2 \times (\text{RH}/100)$.
     * `clear_nocturnal_cooling_idx`: night/shade clear-sky radiational cooling potential weighted by calm wind confinement $e^{-u/3}$.
   * **Fold-Internal Synthetic Lead Augmentation & Lead-Inflated Uncertainty:** Because archive rows center at Lead 6h, `_augment_training_leads()` injects decayed synthetic `18h`, `36h`, `72h`, and `120h` examples strictly *inside* each training fold (never crossing train/validation splits) so Day 2–7 operational forecasts stay in-distribution. At inference, P10–P90 half-widths scale smoothly with lead time via $\sqrt{1 + 0.012\cdot\max(0,\, h - 6)}$.
   * **Issue-Time Anomaly Decay:** Local anomalies (persistence bias, crest lapse-rate anomaly, gorge confinement, and barometric tendency) decay exponentially with lead time:
     $$\Gamma_{\text{eff}}(h) = 6.5 + (\Gamma_0 - 6.5)\cdot e^{-h / 12.0}$$
2. **Two-Stage Hydrological Hurdle Subsystem:**
   * **Stage 1 (Calibrated PoP Classifier):** `CalibratedClassifierCV` (Platt sigmoid) wrapping `HistGradientBoostingClassifier` estimating $P(\text{Rain} \ge 0.254\text{ mm})$, with an explicit inner-fold guard requiring $\ge 5$ wet hours per train/val split to prevent silent dry-summer calibration failures.
   * **Stage 2 (Bounded Log-Ratio Wet-Only QPF & $P_{90}$ Regressors):** Trained strictly on wet hours ($\ge 0.254\text{ mm}$) targeting the bounded log-ratio bias:
     $$z = \text{clip}\!\left(\log\frac{y_{\text{obs}} + 0.5}{q_{\text{raw}} + 0.5},\; -1.2,\; 0.85\right)$$
     Reconstructing $\hat{q}_{\text{wet}} = \max(0,\, (q_{\text{raw}} + 0.5)\,e^{\hat{z}} - 0.5)$ preserves linear scaling into unprecedented Atmospheric River intensities without tree-leaf extrapolation clipping, blended with the physical virga/orographic fallback.
3. **Out-of-Fold Interpretability & HAC OLS Diagnostics:**
   * Permutation feature importances (`precomputed_importances`) are computed strictly on **Fold 5 OOF** before the final full-sample refit, and standardized Newey-West HAC OLS statistics ($R^2$, $F$-stat, $\kappa$, $95\%$ CIs) are persisted in the artifact for instant rendering in the Scientific UI.

---

## 6. Early Flood & Road Washout Warning System

Implemented in `hydro_risk_engine.py`, this module spins up 30 days (**720 hours**) of continuous catchment soil saturation and routes runoff across the $72.0\text{ km}^2$ upper Tassajara Creek basin (Pine Ridge, Black Cone, Chews Ridge $\rightarrow$ Stone Office at $485\text{ m}$ MSL).

### 1. Dynamic Hourly Antecedent Precipitation Index ($API$)

$$API_t = API_{t-1} \cdot k_{\text{hr}} + P_{\text{eff}, t}, \qquad k_{\text{hr}} = k_{\text{day}}^{1/24} \in [0.99374,\; 0.99742]$$

* **Evaporative Recession ($k_{\text{day}}$):** Modulated dynamically between $0.86$ and $0.94$ from observed Vapor Pressure Deficit (VPD) so saturated winter storms drain slower than dry autumn intervals:
  $$k_{\text{day}} = \text{clip}\!\left(0.935 - 0.032 \cdot \min(\text{VPD},\, 2.2),\; 0.86,\; 0.94\right)$$
* **Effective Catchment Rain ($P_{\text{eff}}$) vs. Console Audit Totals:** `effective_precip_1hr_mm` blends Stone Office tipping-bucket rain with **online-only** ridge stations (`CHWC1`, `VTCC1`, `PPSC1`, verified fresh within $3.5\text{ h}$ in both cron and UI), strictly excluding `OPENMETEO_SYNTH` / `OFFLINE` rows. Rolling `precip_24hr_mm` and `precip_72hr_mm` remain pure Stone Office tipping-bucket totals for 1:1 console auditing.

### 2. Nonlinear Soil-Saturation Runoff Coefficient ($C$)

After applying a **$1.00\text{ mm/hr}$** canopy/fractured-granite initial abstraction ($P_{\text{net}} = \max(0,\, P_{\text{eff}} - 1.00)$), effective runoff scales via a smooth logistic curve in **$[0.03,\; 0.68]$**:

$$C(API) = 0.03 + \frac{0.68 - 0.03}{1 + \exp\!\left(-\dfrac{API + \min(0.15\cdot API,\, 0.05\cdot P_{72\text{h}}) - 135.0}{32.0}\right)}$$

### 3. Mass-Conserving Two-Pathway Creek Hydrograph (CFS)

Over the $72.0\text{ km}^2$ watershed ($1\text{ mm/hr} = 706.29\text{ CFS}$), effective runoff ($R_t = C_t \cdot P_{\text{net}, t}$) routes above a $6.5\text{ CFS}$ spring baseflow floor across two physical storage pathways:

* **Fast Surface Quickflow ($18\%$):** Convolved through a 4-hour steep-canyon unit hydrograph kernel `(0.32, 0.30, 0.22, 0.16)` to capture rapid storm-band peaks.
* **Shallow Subsurface Interflow ($57\%$):** Routed through a linear hillslope storage reservoir ($k_{\text{interflow}} = 0.955\text{ hr}^{-1}$, releasing $4.5\%\text{/hr}$) sustaining realistic multi-day post-storm creek recession.
* **Deep Bedrock Groundwater Recharge ($25\%$):** Recharges fractured crystalline bedrock supporting perennial baseflow.
* **Seamless `now`-Divider Continuity:** Forward projection routes strictly hours **`valid_time_utc > last_obs_ts`** seeded with the exact final historical interflow storage and 3-hour quickflow buffer (eliminating current-hour double-routing across `now`), while `app.py` visually anchors the forecast line to the final observed point.

> **Calibration & Operational Constraints:** Until a physical stream stage gauge is installed at the bathhouse footbridge, routed `CFS` is an uncalibrated catchment-scaled discharge index. Instantaneous 1-hour runoff thresholds remain at **`6.50 mm/hr` (ORANGE)** and **`12.00 mm/hr` (RED)** (each `1 mm/hr` 1-hour impulse contributes `58.8 CFS`), while routed **`250 / 500 CFS`** safeguards govern multi-hour accumulation and dry recessions across both banner tiers and hourly `projected_tier` rows. Formal `TIER 2 ORANGE` lookahead remains capped at **24 hours** (and `RED` at **6 hours**) to prevent false travel suspensions on ungauged Day 2–3 QPF.

### 4. Four-Tier Operational Severity Matrix

| Tier | Status Code | Rain, Soil Saturation & Discharge Triggers | Tassajara Road (14-Mile Unpaved) | Tassajara Creek & Footbridges | Operational Protocol |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **0** | **GREEN** | $API < 2.00"$, $1\text{h Rate} < 0.15"\text{/hr}$, $24\text{h Rain} < 1.00"$, $72\text{h Rain} < 1.50"$, $Q < 250\text{ CFS}$ | **Open / Dry to Damp**<br>Standard mountain precautions *(Everyday UI shows a separate `🚗` 2WD caution if $0.10"+$ light rain is forecast)* | **Stable Baseflow**<br>Footbridges clear | Routine canyon operations. Normal high-clearance travel. |
| **1** | **YELLOW** | $API \ge 2.00"$ OR Imminent ($\le 6\text{h}$) Rate $\ge 0.15"\text{/hr}$ OR $24\text{h Rain} \ge 1.00"$ OR $72\text{h Rain} \ge 1.50"$ | **Advisory / Caution**<br>Muddy, slick clay grades over Chews Ridge; minor rockfall | **Elevated Flow**<br>Turbid water; moderate rise | 4WD required on Tassajara Road. Postpone non-essential low-clearance trips; clear culverts. |
| **2** | **ORANGE** | Within $24\text{h}$ (coincident hour): Rate $\ge 0.50"\text{/hr}$ on primed soil ($API \ge 2.00"$), OR $\ge 0.30"\text{/hr}$ on saturated soil ($API \ge 3.50"$), OR Runoff $\ge 6.50\text{ mm/hr}$, **OR Routed Creek $Q \ge 250\text{ CFS}$** | **Hazardous / High Risk**<br>Travel strongly discouraged; slipout & rockslide danger | **Rapidly Rising**<br>High velocity; woody debris mobilization likely | Suspend routine travel on Tassajara Road. Inspect lower footbridges for log damming; stage emergency radio net. |
| **3** | **RED** | Imminent/Active ($\le 6\text{h}$, coincident hour): Rate $\ge 1.00"\text{/hr}$ OR $P_{90} \ge 1.25"\text{/hr}$ on saturated soil ($API \ge 3.50"$), OR Runoff $\ge 12.00\text{ mm/hr}$, **OR Routed Creek $Q \ge 500\text{ CFS}$** | **CLOSED / IMPASSABLE**<br>Severe washouts, debris flows, road cuts | **BANKFULL / FLOOD STAGE**<br>Footbridge overtopping surge hazard | **Immediate road closure.** Evacuate low creek-side structures; prohibit all footbridge crossings. |

---

## 7. Database Schema & Production Views

* **Cloud/Git Mode:** GitHub Actions runs `PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;` before committing `tassajara_weather.db` so no unmerged `-wal`/`-shm` sidecars exist in Git. `app.py` opens SQLite strictly in read-only URI mode (`file:tassajara_weather.db?mode=ro`).
* **Local On-Premises Mode:** Enable `PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;` on the Stone Office Mini PC for concurrent local cron writes and Streamlit reads.

### Core Tables

* **`local_observations`:** Ground-truth Stone Office Davis Vantage Pro2 records (`WL_HISTORIC`, `WL_LIVE`). Stores temperature, dewpoint, wetbulb, MSLP, 3-hour barometric tendency, wind vector, solar radiation, and $0.254\text{ mm}$-quantum tipping-bucket rainfall (`precip_1hr_mm`).
* **`reference_observations`:** Synchronized hourly records from the 5 Santa Lucia reference stations (`CHWC1`, `VTCC1`, `CAHC1`, `ASRC1`, `PPSC1`) with explicit `source` provenance (`RAWS_SYNOPTIC`, `NWS_OBS`, `IEM_CA_DCP`, `OPENMETEO_SYNTH`).
* **`nws_forecasts`:** Operational 168-hour official NOAA NWS `/gridpoints/MTR` forecast grids stored by `(issue_time_utc, valid_time_utc)`, plus 7 matched Open-Meteo `880 m` companion columns (`om_temp_c`, `om_dewpoint_c`, `om_rh_pct`, `om_wind_speed_ms`, `om_wind_gust_ms`, `om_wind_dir_deg`, `om_sky_cover_pct`).
* **`mos_corrected_forecasts`:** Keyed by `(issue_time_utc, valid_time_utc, model_version)` storing `MOS_GBM_v3_HYDRO` (live operational) and `MOS_GBM_v3_OOF` (out-of-fold backfill) temperature, $P_{10}$/$P_{90}$ bounds, calibrated PoP (%), expected/$P_{90}$ QPF rates, and official NOAA `raw_nws_temp_c`.
* **`hydrological_risk_state`:** Hourly archive of catchment saturation ($API$), runoff coefficient ($C$), peak runoff, routed creek discharge (CFS), and 4-tier alert status (trailing 6 hours refreshed via `INSERT OR REPLACE` as `WL_HISTORIC` finalizes).
* **`hydrological_forecast_series`:** Forward projected hydrograph (`valid_time_utc > last_obs_ts`), $API$ trajectory, and post-routing `projected_tier`.
* **`ingest_state`:** Key-value state table holding monotonic rainfall counters (`last_console_rain_year_mm`, `last_console_rain_day_mm`, `last_ref_rain_day_mm_<STID>`) and `last_obs_hour_iso`.
* **`stream_gauge_observations`:** Physical stream gauge telemetry table (pre-wired for future bathhouse footbridge sensor commissioning).

### Production Views

* **`vw_mos_training_dataset`:** Strictly causal (`lead_time_hours >= 1 AND issue_time_utc < valid_time_utc`), deduplicated training view projecting `COALESCE(om_*, nws_*)` on genuine Davis observations (`source != 'OPENMETEO_SYNTH'`) and excluding operational NWS rows lacking `om_temp_c` so weekly retrains stay 100% Open-Meteo domain-pure.
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
git clone [https://github.com/MT-Wildhack/tassajara-weather.git](https://github.com/MT-Wildhack/tassajara-weather.git)
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

# Retrain MOS model on existing SQLite archive (preserves live verification rows)
python3 backfill_and_train.py --skip-fetch

# Launch dashboard locally
streamlit run app.py
```

### 3. Cloud Automation & Streamlit Cloud Operations

* **Hourly Pipeline & Weekly Retrain (GitHub Actions):** `.github/workflows/hourly_weather.yml` runs at `:05` past every hour (`run_hourly_mos_pipeline.py`) and every Sunday at `11:25 UTC` (`backfill_and_train.py --skip-fetch`), committing `tassajara_weather.db` (and updated `models/tassajara_mos_v1.joblib`) back to `main`. Push retries automatically pass `--skip-alerts` to prevent duplicate SMS notifications.
* **Rebooting Streamlit Cloud (`https://tassajara-weather.streamlit.app`):** After pushing code, database updates, or dependency changes, Streamlit Cloud auto-redeploys within ~60 seconds. If a dependency rebuild hangs or cached model artifacts persist, click **`Manage app`** (bottom-right) $\rightarrow$ **`⋮`** $\rightarrow$ **`Reboot app`**.
* **Handling `tassajara_weather.db` Merge Conflicts:** Because GitHub Actions commits `tassajara_weather.db` hourly, pushing local edits can trigger a binary merge conflict:
  * If you only edited Python files, keep `origin/main`'s database (`git checkout --theirs tassajara_weather.db`) so you never lose automated telemetry hours that arrived while you worked locally.
  * If you need schema views or a fresh forecast batch applied onto `origin/main`'s database, check out `origin/main`'s DB first, run `ensure_schema_and_view()` / `fetch_and_store_nws_hourly()`, checkpoint with `PRAGMA journal_mode=DELETE;`, and push.

---

## 9. Local On-Premises Deployment: Hardware Selection & Offline Resilience

During severe Santa Lucia winter storms, Tassajara's satellite/WAN internet link and grid power are most likely to drop **at the exact moment flood and road-washout guidance is most critical**. Running a dedicated, hardened local appliance inside the Stone Office on the same UPS-backed LAN as the **Davis Vantage Pro2 Cabled Console** and its **WeatherLinkIP (`6555`) Ethernet Data Logger** guarantees 100% uptime for local staff and the Emergency Response team even during a complete internet blackout.

### 1. Recommended Hardware Comparison

| Criterion | **Option A: Fanless Intel N100 / N150 Mini PC (Recommended)** | **Option B: Raspberry Pi 5 (8 GB) + NVMe Hat** |
| :--- | :--- | :--- |
| **Target Models** | Beelink EQ13/EQ14 (N100/N150), MeLE Quieter4C (fanless), Minisforum UN100P | Raspberry Pi 5 (8 GB) + Official Active Cooler + M.2 NVMe Base + 256 GB NVMe SSD |
| **Storage Reliability** | Native M.2 NVMe SSD (high write endurance for hourly SQLite WAL commits) | **Mandatory NVMe SSD** (never run SQLite WAL 24/7 on a microSD card—cards fail within 6–12 months) |
| **Power Draw** | $6\text{–}12\text{ W}$ idle, $15\text{–}20\text{ W}$ peak (easily backed by a small UPS) | $3\text{–}5\text{ W}$ idle, $10\text{–}12\text{ W}$ peak (best if running purely on off-grid DC solar/battery) |
| **MOS Retraining Speed** | Full 730-day (`17,500+`-row) 5-fold GBDT + OLS retrain in **~60–90 seconds** | Full 730-day retrain in ~3–5 minutes |
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
| **Forecast Horizon (Temp, PoP, QPF)** | Refreshed hourly from NOAA NWS + Open-Meteo `880 m` | **Continues smoothly for up to 7 days** from the last cached 168-hour `MOS_GBM_v3_HYDRO` batch in `tassajara_weather.db` |
| **5-Station Regional Array** | Live (`NWS_OBS` + `IEM_CA_DCP`) | Gracefully marks remote stations stale (`age_hours > 3.5h`) without crashing or fabricating spillover |
| **Dashboard Access for Staff** | `https://tassajara-weather.streamlit.app` + local LAN | **Local LAN / Wi-Fi:** `http://tassajara-weather.local:8501` (accessible from any phone/laptop on Stone Office Wi-Fi) |

### 4. Hardened Local Linux Setup (Ubuntu Server 24.04 LTS)

#### Step A: Base OS, Dedicated User, mDNS & Firewall
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip sqlite3 git ufw avahi-daemon netcat-openbsd

# Set local hostname so staff can browse to [http://tassajara-weather.local:8501](http://tassajara-weather.local:8501)
sudo hostnamectl set-hostname tassajara-weather

# Lock down firewall: allow SSH and Streamlit strictly from the local Stone Office LAN subnet
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.0.0/16 to any port 22 proto tcp
sudo ufw allow from 192.168.0.0/16 to any port 8501 proto tcp
sudo ufw enable

# Create dedicated unprivileged service user
sudo useradd -r -m -d /opt/tassajara-weather -s /bin/bash tzmc
sudo -u tzmc git clone [https://github.com/MT-Wildhack/tassajara-weather.git](https://github.com/MT-Wildhack/tassajara-weather.git) /opt/tassajara-weather/app
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

#### Step E: Safe Daily SQLite Backups
Never copy a live WAL-mode SQLite file with `cp` (which can produce a corrupted snapshot). Schedule a nightly online backup using SQLite's native `.backup` command to a mounted USB drive or NAS:
```bash
sqlite3 /opt/tassajara-weather/app/tassajara_weather.db ".backup '/mnt/backup/tassajara_weather_$(date +%F).db'"
```

#### Step F: BIOS Auto-Power-On After Extended Blackout
In the Mini PC's UEFI/BIOS settings, enable **Restore on AC Power Loss $\rightarrow$ Power On**. If a multi-day storm drains the UPS completely, the appliance will automatically boot, restart the dashboard and hourly timer, and resume flood monitoring the moment generator or grid power returns.

---

## 10. Streamlit Dashboard UI

Hosted live at **[https://tassajara-weather.streamlit.app](https://tassajara-weather.streamlit.app)** (or locally at `http://tassajara-weather.local:8501`), the interface uses read-only SQLite access (`?mode=ro`), `@st.cache_resource` model caching with visible fallback warnings, `180s` data caching (`1.25h` SQLite reference freshness window), and Streamlit `width="stretch"` layout compatibility across two modes:

### 1. Everyday Forecast (Community & Guest View)
* **Synchronized Flood & Road Warning Banner:** Color-coded operational guidance (`GREEN`, `YELLOW`, `ORANGE`, `RED`) covering Tassajara Road passability, footbridge debris risk, and actionable emergency steps.
* **Distinct Light-Rain 2WD Road Caution:** Displays a `🚗` light-rain driving caution when `0.10+ in` is forecast while formal creek hydrology remains `GREEN`, avoiding visual collision with `TIER 1 YELLOW`.
* **Topographic Shading & Current Conditions Card:** Plain-language canyon conditions (`Active Storm Overcast`, `Canyon Wall Shadow`, `Clear Canyon Night`), `Past 24h High / Low`, and rolling 24-hour rain.
* **Freeze & Trail Advisories:** Automated freeze alerts when cold-pool $P_{10}$ dips to $\le 32^\circ\text{F}$, plus 3-hour outlook cards (with day/night-aware icons) and a 7-day planner table.
* **Interactive Creek Runoff & Hydrograph Outlook:** Displays catchment saturation ($API$), runoff readiness ($C$), and routed creek discharge seamlessly across `now` alongside `250 / 500 CFS` reference thresholds during high-flow events.

### 2. Scientific & Statistical MOS Suite
* **8-Metric Boundary-Layer Telemetry Grid:** Live dry-bulb, RH/dewpoint, Stull wet-bulb/Tetens VPD, anemometer vector (`10-Min Peak Gust`), MSLP + 3-hour barometric tendency, 24h rain + 72h MOS QPF, crest lapse rate ($^\circ\text{C/km}$), and gorge/coastal differentials ($\Delta T$ in $^\circ\text{F}$).
* **5-Station Santa Lucia Network Table:** Live status (`Online`, `Online (Rain Gauge)`, or `Model Fallback / Stale`), temperature, RH, wind/gust, hourly rain, and vertical/horizontal gradient vs. Stone Office.
* **3-Panel Interactive Plotly Suite:**
  * *Panel 1:* Observed temperature actuals vs. MOS Quantile Fan ($P_{10}$–$P_{90}$) vs. Official Raw NOAA NWS forecast.
  * *Panel 2:* Observed canyon gusts ($485\text{ m}$) cleanly separated from forecast ridge-grid gusts ($880\text{ m}$) + RH (%).
  * *Panel 3:* MOS calibrated QPF, $P_{90}$ extreme downpour rates, observed tipping-bucket rain, and PoP (%).
* **Exact-Match Verification Scorecard:** Evaluates strictly causal `(issue_time_utc, valid_time_utc)` pairs for Official NOAA NWS vs. MOS MAE/RMSE, RMSE Skill Score (%), Hourly Win Frequency, $P_{10}$–$P_{90}$ empirical coverage, and hydrological Brier Score & QPF bias.
* **Model Interpretability & HAC OLS Diagnostics:** Fold-5 OOF permutation feature importances (all 33 predictors) and standardized Newey-West HAC OLS regression summary ($R^2$, $F$-stat, $\kappa$, $95\%$ CIs).

---

## 11. Hardware Roadmap & Pending Physical Stream Telemetry

The database schema (`stream_gauge_observations` and `vw_creek_flood_monitor`) and `hydro_risk_engine.py` are pre-wired for a physical stream-stage sensor at the **Tassajara Creek Bathhouse Footbridge**:

* **Current Mode:** `UNINSTALLED (MOS + Tipping Bucket Mode)` — `stream_stage_m`, `rate_of_rise_m_hr`, `water_temp_c`, and `battery_v` remain `NULL`.
* **Target Hardware:** Non-contact SDI-12 / Modbus Radar or Ultrasonic Level Sensor (e.g., Campbell Scientific CS475A) paired with a solar-powered ESP32 / LoRaWAN node transmitting 5-minute stage readings to the Stone Office Mini PC.
* **Automatic Integration:** Once rows populate `stream_gauge_observations`, `evaluate_flood_and_road_tier()` automatically activates physical stage height and surge-rate ($\text{m/hr}$) escalation rules alongside the existing $API$ and `250 / 500 CFS` triggers without requiring code rewrites.

---

## 12. Future Directions & Model Improvement Timetable

Because `nws_forecasts` now continuously archives **both** official NOAA NWS `/gridpoints/MTR` grids and matched Open-Meteo `880 m` grids—while `mos_corrected_forecasts` permanently preserves genuine causal operational forecasts without weekly overwrite—future model upgrades follow a strict **data-gated timetable** rather than short-window tuning:

| Window & Data Gate | Target Milestone | Scientific Objective & Go/No-Go Validation Gate |
| :--- | :--- | :--- |
| **Phase A: Weeks 1–4**<br>*(Late Nov – Dec 2026)*<br>**~300–700 dual-baseline operational hrs** | **Model Freeze & Clean Operational Baseline Audit** | Freeze `mos_model.py`. Audit live post-Option-A verification (`MOS_GBM_v3_HYDRO` vs. Official NOAA NWS) across clear winter ridges, cold-pool inversions, and early storms. Verify daytime overcast error under `cloud_attenuated_insolation` and Day 1 vs. Day 2–7 empirical $P_{10}\text{–}P_{90}$ coverage. |
| **Phase B: Months 2–3**<br>*(Jan – Feb 2027)*<br>**~1,500–2,200 operational hrs + peak winter storms** | **Bowen-Ratio Soil Wetness, Multi-Lead CQR & Storm Hydrology Calibration** | 1. Test adding antecedent soil wetness (`API_t` / rolling 72h rain) as a thermal predictor to capture post-storm evapotranspiration cooling; adopt only if full 2-year purged OOF CV improves wet-regime (`API >= 50 mm`) daytime RMSE without hurting dry-regime skill.<br>2. Replace the analytical conformal inflation curve $\sqrt{1 + 0.012(h-6)}$ with empirical per-horizon CQR bins (Day 1, Day 2–3, Day 4–7) calibrated on archived multi-lead operational runs.<br>3. Audit bounded log-ratio Stage-2 QPF and `250 / 500 CFS` hydrograph peaks against visual footbridge storm logs (tuning `API_MIDPOINT_MM` and `INTERFLOW_FRACTION`).<br>4. Add a **Chews Ridge Summit Icing Advisory** badge when live `CHWC1` drops to $\le 33^\circ\text{F}$ on wet roads (`API >= 2.00 in`). |
| **Phase C: Months 4–6**<br>*(Mar – May 2027)*<br>**~3,000–4,500 dual-baseline NWS + Open-Meteo pairs** | **Spring Marine-Layer Cross-Model Audit & Horizon Review** | 1. Compare native NOAA NWS `/gridpoints/MTR` vs. Open-Meteo `880 m` error structure across winter frontal and spring marine-stratus transitions.<br>2. Evaluate a lightweight regularized dual-NWP residual blender if NWS SmartInit and Open-Meteo show complementary skill across synoptic regimes.<br>3. Re-evaluate extending `TIER 2 ORANGE` lookahead from `24h` to `36h` only if Day-2 QPF Brier/RMSE and visual creek records demonstrate acceptable false-alarm rates. |
| **Phase D: Months 9–12**<br>*(Aug – Nov 2027)*<br>**~6,500–8,760 hrs (Full 4-Season Native NWS Cycle)** | **Native NOAA NWS MOS & Multi-NWP Super-Ensemble** | 1. Train and benchmark a **native NOAA NWS `/gridpoints/MTR` MOS model** spanning all four Santa Lucia seasons (winter storms, spring stratus, summer canyon heat, autumn offshore winds).<br>2. Graduate to a **Two-Model Multi-NWP Super-Ensemble MOS** (NOAA NWS + Open-Meteo `880 m`), incorporating inter-model temperature/QPF spread (`|nws_temp_c - om_temp_c|`) as an explicit synoptic uncertainty predictor for $P_{10}\text{–}P_{90}$.<br>3. Integrate physical bathhouse stream-gauge rating curves (`stream_gauge_observations`) and BAER post-fire hydrological scaling ($C_{\max}$ adjustment) if wildfire impacts the upper Ventana Wilderness catchment. |

---

## 13. License & Operational Disclaimer

Maintained for the safety and operational resilience of the **Tassajara Zen Mountain Center (San Francisco Zen Center)** community, staff, students, and guests. Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).

**Operational Disclaimer:** The computations, forecasts, discharge indices, and alert tiers (`GREEN`, `YELLOW`, `ORANGE`, `RED`) generated by this software serve strictly as **advisory decision-support guidance**. They do not constitute an official government warning and do not supersede direct visual inspection of Tassajara Road and Tassajara Creek or official directives issued by the **National Weather Service (NWS Monterey)**, **Monterey County Office of Emergency Services (OES)**, **US Forest Service (USFS)**, or the **TZMC Disaster Council**.
