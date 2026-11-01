# Tassajara MOS Microclimate Pipeline 🌤️⛰️

An automated, zero-hardware Model Output Statistics (MOS) pipeline and real-time dashboard for **Tassajara Zen Mountain Center** (36.2337° N, 121.5501° W, Elev: 485 m / 1,590 ft MSL).

This system corrects standard National Weather Service (NWS) grid forecasts to account for the extreme topographic and boundary-layer microclimate of Tassajara Creek Canyon in the Santa Lucia Mountains, incorporating a **5-station Santa Lucia boundary array** (`CHWC1` Chews Ridge, `VTCC1` Ventana Cone, `CAHC1` Hastings, `ASRC1` Arroyo Seco, `PPSC1` Big Sur) and a **Two-Stage Hydrological Hurdle Model** (PoP/QPF calibration).

**Live Dashboard:** [tzmc-weather.streamlit.app](https://tzmc-weather.streamlit.app)

---

## ⛰️ The Problem: Canyon Microclimate Dynamics

Standard Numerical Weather Prediction (NWP) models (like the NWS NDFD or HRRR) operate on 2.5km to 3.0km grids (~880m MSL). In the Ventana Wilderness, these coarse grids smooth steep terrain. Consequently, generic forecasts miss four critical local weather phenomena:

1. **Topographic Solar Cutoff:** The steep ridge of the Black Cone Trail / Pine Ridge corridor (32° SW horizon) eclipses the sun hours before astronomical sunset, causing temperatures to plunge rapidly in the early afternoon.
2. **Nocturnal Cold-Air Pooling (CAP):** Heavy, cold air drains down tributary ravines at night, trapping freezing air on the canyon floor. NWS forecasts frequently overpredict overnight lows by 5°F to 15°F, missing hard-freeze risks to facility plumbing.
3. **Sub-Cloud Virga & Orographic Spillover:** Under pre-frontal arid regimes, VPD >= 1.5 kPa evaporates light precipitation as virga before it reaches the valley floor. Conversely, moist SSW (190°–240°) atmospheric rivers intersecting the 1,536m Chews Ridge and Ventana Cone crests produce intense feeder-seeder orographic enhancement.
4. **Roof Convective Plumes / Radiation Shield Bias:** Under midday sun and low wind, uncalibrated temperature sensors experience artificial thermal heat biases.

---

## 🧠 The Models

Rather than predicting temperature from scratch, this pipeline uses **Model Output Statistics (MOS) Residual Correction**. It takes the physical NWS forecast and predicts the *error* (residual) caused by local canyon topography, preserving the NWS model's ability to see incoming synoptic weather fronts.

### 1. Gradient-Boosted Tree Ensemble & Quantile Uncertainty (`models/tassajara_mos_v1.joblib`)
* **Thermal Subsystem:** `HistGradientBoostingRegressor` trained across 25 engineered physical features using MSE loss for the mean, paired with P10/P90 quantile loss estimators.
  * **P10 (Lower Bound):** Drives automated **Hard Freeze** alerts for pipe and garden protection.
  * **P90 (Upper Bound):** Drives automated **Fire Weather** alerts during hot, dry offshore wind events.
* **Hydrological Hurdle Subsystem:**
  * *Stage 1 (PoP Classifier):* Calibrated `HistGradientBoostingClassifier` predicting $P(\text{Rain} \ge 0.254\,\text{mm})$.
  * *Stage 2 (QPF Regressor):* Poisson deviance and P90 extreme quantile regressors scaling accumulation by Santa Lucia orographic moisture flux and eliminating zero-QPF precipitation traps during high-PoP events.
* **Econometric Baseline:** Z-score standardized, orthogonalized Ordinary Least Squares (OLS) with Newey-West HAC robust standard errors (12-hour lag).

### 2. Features Engineered (25 Predictors)
* Cyclic temporal harmonics (sine/cosine of hour and day-of-year).
* Solar elevation angle (`solar_el_deg`), canyon shadow flag (`in_canyon_shadow`), and effective insolation.
* Along-canyon and cross-canyon wind vector decomposition (rotated to the canyon's 115° ESE primary axis).
* Santa Lucia SSW upslope wind vector and orographic moisture flux (IVT proxy).
* Sub-cloud virga evaporation index (VPD * dewpoint depression * dryness).
* **5-Station Santa Lucia Spatial Gradients:**
  * Upper-column vertical lapse rate across Chews Ridge (`CHWC1`, 1,536m) and Ventana Cone (`VTCC1`, 1,453m) to Tassajara (485m).
  * Gorge and foothill cold-pool thermal confinement against Hastings (`CAHC1`, 546m) and Arroyo Seco (`ASRC1`, 268m).
  * Cross-range coastal-to-interior thermal and humidity gradients against Big Sur RAWS (`PPSC1`, 108m).
  * Decayed upstream feeder-seeder precipitation advection.

---

## ⚙️ Zero-Hardware Architecture

This pipeline requires no local servers or Raspberry Pis at the off-grid monastery:

1. **Ingestion (GitHub Actions / Hourly Pipeline):** Queries the Davis WeatherLink API (`/v2/current` and `/v2/historic` catch-up) and the Synoptic Data 5-station array (`CHWC1`, `VTCC1`, `CAHC1`, `ASRC1`, `PPSC1`), runs MOS inference, and updates database records.
2. **Storage (SQLite):** All raw observations, standard forecasts, corrected forecasts, and alert logs are committed directly back to `tassajara_weather.db` inside this repository.
3. **Visualization (Streamlit):** Streamlit Community Cloud actively watches this repository and updates the public web dashboard in real time.

---

## 🚀 Local Development & Execution

This project uses `uv` for seamless, one-command Python environment management.

```bash
# 1. Clone the repository
git clone [https://github.com/MT-Wildhack/tassajara-weather.git](https://github.com/MT-Wildhack/tassajara-weather.git)
cd tassajara-weather

# 2. Backfill historical reference data and train the 25-feature production MOS model
uv run python backfill_and_train.py --days 730 --skip-weatherlink

# 3. Run a live hourly pipeline ingestion cycle
uv run python run_hourly_mos_pipeline.py

# 4. Launch the local web dashboard
uv run streamlit run app.py
