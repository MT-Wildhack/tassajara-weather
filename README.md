# Tassajara MOS Microclimate Pipeline 🌤️⛰️

An automated, zero-hardware Model Output Statistics (MOS) pipeline and real-time dashboard for **Tassajara Zen Mountain Center** (`36.2337° N, 121.5501° W`, Elev: 1,590 ft). 

This system corrects standard National Weather Service (NWS) grid forecasts to account for the extreme topographic and boundary-layer microclimate of Tassajara Creek Canyon in the Santa Lucia Mountains.

**Live Dashboard:** [tzmc-weather.streamlit.app](https://tzmc-weather.streamlit.app)

---

## ⛰️ The Problem: Canyon Microclimate Dynamics
Standard Numerical Weather Prediction (NWP) models (like the NWS NDFD or HRRR) operate on 2.5km to 3.0km grids. In the Ventana Wilderness, these grids smooth the terrain, assuming Tassajara sits at a broad average elevation of ~3,000 ft. Consequently, generic forecasts miss three critical, high-consequence local weather phenomena:

1. **Topographic Solar Cutoff:** The steep ridge of the Black Cone Trail corridor eclipses the sun hours before astronomical sunset, causing temperatures to plunge rapidly in the early afternoon.
2. **Nocturnal Cold-Air Pooling (CAP):** Heavy, cold air drains down tributary ravines at night, trapping freezing air on the canyon floor. NWS forecasts frequently overpredict overnight lows by 5°F to 15°F, missing hard-freeze risks to facility plumbing.
3. **Roof Convective Plumes:** The local Davis Vantage Pro2 sensor is mounted on a metal/wood roof. Under midday sun and low wind, it experiences a $+1.5^\circ\text{C}$ to $+3.5^\circ\text{C}$ artificial thermal heat bias.

## 🧠 The Models
Rather than predicting temperature from scratch, this pipeline uses **Model Output Statistics (MOS) Residual Correction**. It takes the physical NWS forecast and predicts the *error* (residual) caused by the local canyon topography, preserving the NWS model's ability to see incoming synoptic weather fronts.

The pipeline transitions between two models based on available historical data:

### 1. Topographic Physics Baseline (Cold-Start Mode)
Used during the first 14-30 days of data collection. Because machine learning requires historical data, the pipeline initially applies deterministic physical equations:
* **Solar Geometry:** Computes the exact solar altitude and azimuth. If the sun is mathematically above the horizon but physically blocked by the 32° elevation of the southern ridge, an immediate $-2.5^\circ\text{C}$ shadow offset is applied.
* **Drainage Offset:** Applies a steady negative bias during nighttime hours under low wind conditions to simulate cold-air pooling.

### 2. Gradient-Boosted Tree Ensemble (Production MOS)
Once sufficient hourly observation pairs accumulate in the database, the system shifts to a machine learning approach using Scikit-Learn's `HistGradientBoostingRegressor`.
* **Features Engineered:**
  * Harmonic diurnal and seasonal cycles (sine/cosine of hour and day-of-year).
  * `in_canyon_shadow` flag and `effective_insolation`.
  * Along-canyon and cross-canyon wind vector decomposition (rotated to the canyon's 115° ESE primary axis).
  * Initialization stability (lapse rate against the 5,040 ft Chews Ridge reference station).
  * Exponentially decayed persistence bias.
* **Quantile Uncertainty Bounds:** The pipeline trains three parallel models using Pinball/Quantile Loss to generate the 10th percentile, Mean (Squared Error), and 90th percentile predictions. 
  * *P10 (Lower Bound):* Drives automated **Hard Freeze** alerts for pipe protection.
  * *P90 (Upper Bound):* Drives automated **Fire Weather** alerts during hot, dry offshore wind events.

## ⚙️ Zero-Hardware Architecture
This pipeline requires no local servers or Raspberry Pis at the off-grid monastery.

1. **Ingestion (GitHub Actions):** Every hour at `:05`, a GitHub Actions runner wakes up, queries the local Davis WeatherLink API and the NWS API, runs the MOS inference, and checks for critical facility thresholds.
2. **Storage (SQLite):** All raw observations, standard forecasts, corrected forecasts, and alert logs are committed directly back to `tassajara_weather.db` inside this repository.
3. **Visualization (Streamlit):** Streamlit Community Cloud actively watches this repository and updates the public web dashboard whenever a new database is committed.

## 🚀 Local Development & Execution
This project uses `uv` for seamless, one-command Python environment management.

```bash
# 1. Clone the repository
git clone https://github.com/MT-Wildhack/tassajara-weather.git
cd tassajara-weather

# 2. Run the hourly ingestion and MOS inference pipeline
uv run python run_hourly_mos_pipeline.py

# 3. Launch the local web dashboard
uv run streamlit run app.py
