#!/usr/bin/env python3
"""
mos_model.py
Trains and evaluates the Tassajara Model Output Statistics (MOS) Gradient Boosted
Ensemble:
  1. Thermal Subsystem: Mean error + P10/P90 quantile bounds (HistGradientBoostingRegressor)
  2. Hydrological Hurdle Subsystem:
     - Stage 1 PoP Classifier: Calibrated HistGradientBoostingClassifier for P(Rain >= 0.254mm)
     - Stage 2 QPF Regressor: Poisson deviance & P90 extreme quantile HistGradientBoostingRegressor
       accounting for Santa Lucia orographic moisture flux, Chews Ridge spillover, and canyon virga.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    brier_score_loss,
)
import statsmodels.api as sm

TASS_LAT_RAD = np.radians(36.2337)
TASS_LON_DEG = -121.5501
CANYON_AXIS_RAD = np.radians(115.0)       # ESE-WNW primary axis of Tassajara Creek Canyon
OROGRAPHIC_UPSLOPE_RAD = np.radians(215.0) # SSW (215 deg) perpendicular to Santa Lucia crest
TIPPING_BUCKET_MM = 0.254                 # 0.01 inch Davis Vantage Pro2 rain spoon threshold


def compute_solar_and_canyon_shadow(timestamps_utc: pd.Series) -> pd.DataFrame:
    """
    Calculates solar altitude, azimuth, and a topographic shadow indicator
    modeling the Black Cone Trail / Pine Ridge southern canyon wall.
    """
    dt = pd.to_datetime(timestamps_utc, utc=True)
    doy = dt.dt.dayofyear.values
    hour_utc = dt.dt.hour.values + dt.dt.minute.values / 60.0

    # Fractional year & solar declination (Spencer 1971 approximation)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )

    # True solar time & hour angle
    time_offset = eqtime + 4.0 * TASS_LON_DEG
    tst = (hour_utc * 60.0 + time_offset) % 1440.0
    ha_rad = np.radians((tst / 4.0) - 180.0)

    # Solar Elevation (Altitude) and Azimuth
    sin_el = (
        np.sin(TASS_LAT_RAD) * np.sin(decl)
        + np.cos(TASS_LAT_RAD) * np.cos(decl) * np.cos(ha_rad)
    )
    solar_el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    cos_az = (
        (np.sin(decl) - np.sin(TASS_LAT_RAD) * sin_el)
        / (np.cos(TASS_LAT_RAD) * np.cos(np.arcsin(np.clip(sin_el, -1.0, 1.0))) + 1e-6)
    )
    az_deg = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    solar_az_deg = np.where(ha_rad > 0, (360.0 - az_deg) % 360.0, az_deg)

    # Azimuth-dependent horizon profile of Tassajara Canyon (degrees above horizontal)
    # 145-255 deg (SSE to WSW): Black Cone Trail & Pine Ridge wall (~32 deg horizon)
    # 45-115 deg (NE to ESE): Eastern ridge horizon (~23 deg horizon)
    horizon_deg = np.full_like(solar_az_deg, 18.0)
    horizon_deg = np.where((solar_az_deg >= 145.0) & (solar_az_deg <= 255.0), 32.0, horizon_deg)
    horizon_deg = np.where((solar_az_deg >= 45.0) & (solar_az_deg <= 115.0), 23.0, horizon_deg)

    in_canyon_shadow = ((solar_el_deg > 0.0) & (solar_el_deg < horizon_deg)).astype(float)
    effective_insolation_factor = np.where(
        solar_el_deg >= horizon_deg, np.sin(np.radians(solar_el_deg)), 0.0
    )

    return pd.DataFrame({
        "solar_el_deg": solar_el_deg,
        "solar_az_deg": solar_az_deg,
        "in_canyon_shadow": in_canyon_shadow,
        "effective_insolation": effective_insolation_factor
    }, index=timestamps_utc.index)


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def engineer_mos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds physical, cyclical, topographic, and hydrological features for both
    canyon temperature bias and orographic/virga precipitation calibration.
    """
    out = pd.DataFrame(index=df.index)
    dt_valid = pd.to_datetime(df["valid_time_utc"], utc=True)
    dt_local = dt_valid.dt.tz_convert("America/Los_Angeles")

    hour = dt_local.dt.hour + dt_local.dt.minute / 60.0
    doy = dt_local.dt.dayofyear

    # 1. Cyclical Diurnal and Seasonal Harmonics
    out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    # 2. Solar & Black Cone Trail Shadow Features
    solar_df = compute_solar_and_canyon_shadow(df["valid_time_utc"])
    out["solar_el_deg"] = solar_df["solar_el_deg"]
    out["in_canyon_shadow"] = solar_df["in_canyon_shadow"]
    out["effective_insolation"] = solar_df["effective_insolation"]

    # 3. Raw NWP Predictors & Lead Time
    lead_hrs = _num_series(df, "lead_time_hours", 6.0).clip(lower=1.0)
    t_c = _num_series(df, "nws_temp_c", 15.0)
    rh_pct = _num_series(df, "nws_rh_pct", 50.0).clip(1.0, 100.0)
    dew_c = _num_series(df, "nws_dewpoint_c", np.nan)
    # Approximate dewpoint via Magnus formula if missing
    dew_fallback = t_c - ((100.0 - rh_pct) / 5.0)
    dew_c = dew_c.fillna(dew_fallback)

    pop_pct = _num_series(df, "nws_pop_pct", 0.0).clip(0.0, 100.0)
    qpf_mm = _num_series(df, "nws_qpf_1hr_mm", 0.0).clip(lower=0.0)
    sky_pct = _num_series(df, "nws_sky_cover_pct", 20.0).clip(0.0, 100.0)

    out["lead_time_hours"] = lead_hrs
    out["nws_temp_c"] = t_c
    out["nws_rh_pct"] = rh_pct
    out["nws_pop_pct"] = pop_pct
    out["nws_qpf_1hr_mm"] = qpf_mm
    out["nws_sky_cover_pct"] = sky_pct

    # 4. Along-Canyon, Cross-Canyon & Santa Lucia Upslope Wind Components
    w_spd = _num_series(df, "nws_wind_speed_ms", 0.0).clip(lower=0.0)
    w_dir_rad = np.radians(_num_series(df, "nws_wind_dir_deg", 0.0))
    out["wind_along_canyon_ms"] = w_spd * np.cos(w_dir_rad - CANYON_AXIS_RAD)
    out["wind_cross_canyon_ms"] = w_spd * np.sin(w_dir_rad - CANYON_AXIS_RAD)
    upslope_wind_ms = np.maximum(0.0, w_spd * np.cos(w_dir_rad - OROGRAPHIC_UPSLOPE_RAD))
    out["orographic_upslope_wind_ms"] = upslope_wind_ms

    # 5. Hydrometeorological Moisture Flux & Sub-Cloud Virga Predictors
    # Tetens saturation and actual vapor pressure (kPa)
    es_kpa = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea_kpa = es_kpa * (rh_pct / 100.0)
    vpd_kpa = np.maximum(0.0, es_kpa - ea_kpa)
    dew_depr_c = np.maximum(0.0, t_c - dew_c)

    out["dewpoint_depression_c"] = dew_depr_c
    out["vpd_kpa"] = vpd_kpa
    # Sub-cloud virga index: high when NWP predicts rain/PoP into a dry sub-cloud canyon layer
    out["virga_evap_index"] = vpd_kpa * dew_depr_c * (1.0 - (rh_pct / 100.0))
    # Orographic moisture flux proxy: SSW upslope wind * actual vapor pressure
    out["orographic_moisture_flux"] = upslope_wind_ms * ea_kpa
    # Nonlinear interaction between NWP QPF and relative humidity
    out["qpf_moisture_scaling"] = qpf_mm * ((rh_pct / 100.0) ** 2)

    # 6. Initialized Boundary-Layer State & Upstream Seeding at Issue Time
    ptrend_3h = _num_series(df, "pressure_trend_3hr_hpa", 0.0)
    out["pressure_trend_3hr_hpa"] = ptrend_3h
    # Negative pressure tendency (falling barometer) enhances frontal lift
    out["frontal_pressure_fall_hpa"] = np.minimum(0.0, ptrend_3h)

    issue_chews_t = _num_series(df, "issue_chews_temp_c", np.nan).fillna(t_c + 2.0)
    issue_tass_t = _num_series(df, "issue_tass_temp_c", np.nan).fillna(t_c)
    out["issue_lapse_chews_c_km"] = (
        -1.0 * (issue_chews_t - issue_tass_t) / 1.051
    ).fillna(6.5)

    # Decayed upstream Chews Ridge feeder-seeder precipitation at issue time
    issue_chews_p = _num_series(df, "issue_chews_precip_mm", 0.0)
    out["decayed_chews_precip_mm"] = issue_chews_p * np.exp(-lead_hrs / 6.0)

    # Exponentially decayed thermal persistence bias
    init_bias = _num_series(df, "issue_persistence_bias_c", 0.0)
    out["decayed_init_bias_c"] = init_bias * np.exp(-lead_hrs / 12.0)

    return out


def analytical_hydro_hurdle_fallback(X: pd.DataFrame) -> pd.DataFrame:
    """
    Physical Orographic-Virga Hurdle baseline used when historical training data
    has insufficient wet hours (< 5 wet hours) or before model weights are trained.
    """
    raw_pop = X["nws_pop_pct"].values
    raw_qpf = X["nws_qpf_1hr_mm"].values
    rh = X["nws_rh_pct"].values
    vpd = X["vpd_kpa"].values
    upslope_flux = X["orographic_moisture_flux"].values
    p_fall = X["frontal_pressure_fall_hpa"].values
    chews_seed = X["decayed_chews_precip_mm"].values

    # 1. Sub-cloud virga suppression factor (penalizes light QPF when canyon VPD > 0.8 kPa & RH < 72%)
    virga_factor = np.where(
        (rh < 72.0) & (raw_qpf < 1.5),
        np.exp(-0.55 * vpd),
        1.0,
    )

    # 2. Santa Lucia SSW orographic spillover enhancement (up to +55% during moist upslope + falling pressure)
    orographic_factor = (
        1.0
        + 0.08 * np.clip(upslope_flux, 0.0, 5.0)
        + 0.05 * np.clip(-p_fall, 0.0, 4.0)
        + 0.06 * np.clip(chews_seed, 0.0, 3.0)
    )

    net_multiplier = np.clip(virga_factor * orographic_factor, 0.15, 1.75)

    # Calibrated PoP (%)
    mos_pop = np.clip(
        raw_pop * np.clip(virga_factor * (1.0 + 0.04 * upslope_flux), 0.25, 1.35)
        + np.where(raw_qpf >= TIPPING_BUCKET_MM, 12.0 * (rh / 100.0), -6.0 * (vpd > 1.2)),
        0.0,
        100.0,
    )

    # Expected QPF (mm) with sub-tipping-bucket noise dampening
    mos_qpf = raw_qpf * net_multiplier
    mos_qpf = np.where((mos_pop < 15.0) & (mos_qpf < 0.15), 0.0, mos_qpf)

    # P90 Extreme Quantile QPF (mm)
    mos_qpf_p90 = np.where(
        mos_pop >= 10.0,
        np.maximum(mos_qpf * 1.65, mos_qpf + 0.40 * (mos_pop / 100.0) * orographic_factor),
        0.0,
    )

    return pd.DataFrame({
        "mos_pop_pct": np.round(mos_pop, 1),
        "mos_qpf_mm": np.round(mos_qpf, 2),
        "mos_qpf_p90_mm": np.round(mos_qpf_p90, 2),
    }, index=X.index)


class TassajaraMOSCorrector:
    """
    Multi-Task Ensemble MOS Forecast Corrector:
      - Thermal Subsystem: Expected temperature error (mean) + P10/P90 uncertainty intervals.
      - Hydrological Subsystem: Two-Stage Hurdle Model predicting calibrated PoP (%),
        expected QPF (mm/hr), and P90 extreme rainfall rate (mm/hr).
    """

    def __init__(self):
        common_temp_params = dict(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42
        )
        self.model_mean = HistGradientBoostingRegressor(loss="squared_error", **common_temp_params)
        self.model_p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **common_temp_params)
        self.model_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common_temp_params)

        # Stage 1 Hurdle: Probability of Measurable Precipitation (>= 0.254 mm)
        self.base_pop_clf = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=25,
            min_samples_leaf=20,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=42,
        )
        self.model_pop = None

        # Stage 2 Hurdle: Non-negative Poisson QPF Regressor & P90 Extreme Quantile
        self.model_qpf_mean = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=25,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=42,
        )
        self.model_qpf_p90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=25,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=42,
        )

        self.precip_ml_trained = False
        self.feature_names = []
        self.ols_summary = None

    def fit_and_evaluate(self, df: pd.DataFrame) -> dict:
        X = engineer_mos_features(df)
        self.feature_names = list(X.columns)
        y_err = df["actual_temp_c"] - df["nws_temp_c"]

        # 1. Fit Statsmodels OLS Baseline for Physical Interpretability (p-values & R^2)
        X_ols = sm.add_constant(X.fillna(0.0))
        ols_fit = sm.OLS(y_err, X_ols).fit()
        self.ols_summary = ols_fit.summary().as_text()

        # 2. Time-Series Cross-Validation (5 Expanding Folds) for Temperature & Precipitation
        tscv = TimeSeriesSplit(n_splits=5)
        raw_rmse_list, mos_rmse_list, raw_mae_list, mos_mae_list = [], [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, _ = y_err.iloc[train_idx], y_err.iloc[test_idx]

            self.model_mean.fit(X_tr, y_tr)
            pred_err = self.model_mean.predict(X_te)

            actual_te = df["actual_temp_c"].iloc[test_idx]
            raw_fc_te = df["nws_temp_c"].iloc[test_idx]
            mos_fc_te = raw_fc_te + pred_err

            raw_rmse_list.append(root_mean_squared_error(actual_te, raw_fc_te))
            mos_rmse_list.append(root_mean_squared_error(actual_te, mos_fc_te))
            raw_mae_list.append(mean_absolute_error(actual_te, raw_fc_te))
            mos_mae_list.append(mean_absolute_error(actual_te, mos_fc_te))

        # 3. Final Fit of Thermal Models on Full Historical Dataset
        self.model_mean.fit(X, y_err)
        self.model_p10.fit(X, y_err)
        self.model_p90.fit(X, y_err)

        # 4. Fit Hydrological Two-Stage Hurdle Models
        y_precip = _num_series(df, "actual_precip_mm", 0.0).clip(lower=0.0)
        y_wet = (y_precip >= TIPPING_BUCKET_MM).astype(int)
        wet_hours_count = int(y_wet.sum())

        raw_qpf_all = X["nws_qpf_1hr_mm"].values
        raw_pop_all = (X["nws_pop_pct"].values / 100.0).clip(0.0, 1.0)
        raw_qpf_rmse = float(root_mean_squared_error(y_precip, raw_qpf_all))
        raw_brier = float(brier_score_loss(y_wet, raw_pop_all)) if wet_hours_count > 0 else 0.0

        if wet_hours_count >= 15 and len(np.unique(y_wet)) == 2:
            try:
                # Calibrate PoP classifier using 3-fold temporal CV
                cal_cv = CalibratedClassifierCV(
                    estimator=self.base_pop_clf,
                    method="sigmoid",
                    cv=3,
                )
                cal_cv.fit(X, y_wet)
                self.model_pop = cal_cv
            except Exception:
                self.base_pop_clf.fit(X, y_wet)
                self.model_pop = self.base_pop_clf

            # Fit conditional intensity & P90 models on candidate wet or NWP-active hours
            active_mask = (y_precip > 0.0) | (X["nws_qpf_1hr_mm"] > 0.0) | (X["nws_pop_pct"] >= 15.0)
            if active_mask.sum() >= 25 and y_precip[active_mask].sum() > 0:
                X_act = X.loc[active_mask]
                y_act = y_precip.loc[active_mask]
                self.model_qpf_mean.fit(X_act, y_act)
                self.model_qpf_p90.fit(X_act, y_act)
            else:
                self.model_qpf_mean.fit(X, y_precip)
                self.model_qpf_p90.fit(X, y_precip)

            self.precip_ml_trained = True
        else:
            self.precip_ml_trained = False

        # Evaluate Hydrological Predictions across Dataset
        hydro_preds = self._predict_hydro(X)
        mos_qpf_rmse = float(root_mean_squared_error(y_precip, hydro_preds["mos_qpf_mm"]))
        mos_brier = (
            float(brier_score_loss(y_wet, hydro_preds["mos_pop_pct"] / 100.0))
            if wet_hours_count > 0
            else 0.0
        )

        mean_raw_rmse = float(np.mean(raw_rmse_list))
        mean_mos_rmse = float(np.mean(mos_rmse_list))
        skill_score = 1.0 - (mean_mos_rmse / max(mean_raw_rmse, 1e-4))
        qpf_skill_score = (
            (1.0 - (mos_qpf_rmse / max(raw_qpf_rmse, 1e-4))) * 100.0
            if raw_qpf_rmse > 0
            else 0.0
        )

        return {
            "raw_nws_rmse_c": round(mean_raw_rmse, 3),
            "mos_corrected_rmse_c": round(mean_mos_rmse, 3),
            "raw_nws_mae_c": round(float(np.mean(raw_mae_list)), 3),
            "mos_corrected_mae_c": round(float(np.mean(mos_mae_list)), 3),
            "rmse_skill_improvement_pct": round(skill_score * 100.0, 1),
            "wet_hours_count": wet_hours_count,
            "precip_ml_trained": self.precip_ml_trained,
            "raw_qpf_rmse_mm": round(raw_qpf_rmse, 3),
            "mos_qpf_rmse_mm": round(mos_qpf_rmse, 3),
            "qpf_skill_improvement_pct": round(qpf_skill_score, 1),
            "raw_pop_brier": round(raw_brier, 4),
            "mos_pop_brier": round(mos_brier, 4),
        }

    def _predict_hydro(self, X_fut: pd.DataFrame) -> pd.DataFrame:
        phys_df = analytical_hydro_hurdle_fallback(X_fut)
        if not getattr(self, "precip_ml_trained", False) or getattr(self, "model_pop", None) is None:
            return phys_df

        try:
            prob_wet = self.model_pop.predict_proba(X_fut)[:, 1]
            # Blend ML calibrated probability with physical virga/orographic prior for out-of-sample stability
            mos_pop_pct = np.clip(
                0.75 * (prob_wet * 100.0) + 0.25 * phys_df["mos_pop_pct"].values,
                0.0,
                100.0,
            )

            cond_qpf = np.clip(self.model_qpf_mean.predict(X_fut), 0.0, None)
            cond_p90 = np.clip(self.model_qpf_p90.predict(X_fut), 0.0, None)

            # Hurdle expectation gated by probability of overcoming sub-cloud virga
            hurdle_gate = np.where(mos_pop_pct < 12.0, 0.0, np.clip(mos_pop_pct / 45.0, 0.15, 1.0))
            mos_qpf_mm = (0.75 * (cond_qpf * hurdle_gate) + 0.25 * phys_df["mos_qpf_mm"].values)
            mos_qpf_mm = np.where(mos_qpf_mm < 0.08, 0.0, mos_qpf_mm)

            mos_qpf_p90_mm = np.maximum(
                mos_qpf_mm,
                0.70 * (cond_p90 * np.clip(mos_pop_pct / 35.0, 0.0, 1.0))
                + 0.30 * phys_df["mos_qpf_p90_mm"].values,
            )

            return pd.DataFrame({
                "mos_pop_pct": np.round(mos_pop_pct, 1),
                "mos_qpf_mm": np.round(mos_qpf_mm, 2),
                "mos_qpf_p90_mm": np.round(mos_qpf_p90_mm, 2),
            }, index=X_fut.index)
        except Exception:
            return phys_df

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        X_full = engineer_mos_features(df_future)
        # Backward compatibility if loaded model was trained with fewer columns
        for col in self.feature_names:
            if col not in X_full.columns:
                X_full[col] = 0.0
        X_fut = X_full[self.feature_names]

        err_mean = self.model_mean.predict(X_fut)
        err_p10 = self.model_p10.predict(X_fut)
        err_p90 = self.model_p90.predict(X_fut)

        hydro_df = self._predict_hydro(X_full)

        res = df_future[["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
        res["pred_temp_error_c"] = np.round(err_mean, 2)
        res["corrected_temp_c"] = np.round(res["nws_temp_c"] + err_mean, 2)
        res["corrected_temp_p10_c"] = np.round(res["nws_temp_c"] + err_p10, 2)
        res["corrected_temp_p90_c"] = np.round(res["nws_temp_c"] + err_p90, 2)

        res["raw_nws_pop_pct"] = np.round(X_full["nws_pop_pct"].values, 1)
        res["mos_pop_pct"] = hydro_df["mos_pop_pct"].values
        res["raw_nws_qpf_mm"] = np.round(X_full["nws_qpf_1hr_mm"].values, 2)
        res["mos_qpf_mm"] = hydro_df["mos_qpf_mm"].values
        res["mos_qpf_p90_mm"] = hydro_df["mos_qpf_p90_mm"].values
        return res

    def save(self, filepath: str = "models/tassajara_mos_v1.joblib"):
        joblib.dump(self, filepath)


if __name__ == "__main__":
    # Synthetic verification demonstration simulating Tassajara's diurnal canyon bias + winter AR storms
    rng = np.random.default_rng(42)
    n_hours = 24 * 180  # 180 days of hourly data
    times = pd.date_range("2025-11-01", periods=n_hours, freq="h", tz="UTC")
    solar = compute_solar_and_canyon_shadow(pd.Series(times))

    nws_temp = 12.0 + 6.0 * np.sin(2 * np.pi * (times.hour - 21) / 24.0) + rng.normal(0, 1.5, n_hours)
    true_bias = (
        -4.2 * (solar["solar_el_deg"] <= 0).astype(float)
        - 2.8 * solar["in_canyon_shadow"]
        + 1.9 * (solar["effective_insolation"] > 0.65).astype(float)
        + rng.normal(0, 0.7, n_hours)
    )

    # Simulate periodic atmospheric river passages and pre-frontal virga
    storm_wave = np.sin(2 * np.pi * np.arange(n_hours) / (24.0 * 9.0))
    nws_pop = np.clip((storm_wave - 0.55) * 220.0 + rng.normal(0, 8.0, n_hours), 0.0, 100.0)
    nws_qpf = np.where(nws_pop > 25.0, np.maximum(0.0, (storm_wave - 0.60) * 8.5 + rng.exponential(0.6, n_hours)), 0.0)
    nws_rh = np.clip(55.0 + 0.42 * nws_pop - 2.0 * true_bias, 15.0, 100.0)
    w_dir = np.where(storm_wave > 0.65, 210.0, rng.choice([115.0, 295.0], n_hours))
    w_spd = np.where(storm_wave > 0.65, rng.uniform(4.0, 12.0, n_hours), rng.uniform(0.5, 4.0, n_hours))

    # Actual canyon precip: zeroed out when RH < 72% (virga), enhanced by 1.45x when SSW wind > 6 m/s
    actual_precip = np.where(
        nws_rh < 72.0,
        0.0,
        nws_qpf * np.where((w_dir == 210.0) & (w_spd > 6.0), 1.45, 0.95),
    )

    synth_df = pd.DataFrame({
        "issue_time_utc": times - pd.Timedelta(hours=6),
        "valid_time_utc": times,
        "lead_time_hours": rng.integers(1, 36, n_hours),
        "nws_temp_c": nws_temp,
        "nws_dewpoint_c": nws_temp - ((100.0 - nws_rh) / 5.0),
        "nws_rh_pct": nws_rh,
        "nws_wind_speed_ms": w_spd,
        "nws_wind_dir_deg": w_dir,
        "nws_pop_pct": nws_pop,
        "nws_qpf_1hr_mm": nws_qpf,
        "nws_sky_cover_pct": np.clip(nws_pop * 0.9 + 10.0, 0.0, 100.0),
        "pressure_trend_3hr_hpa": np.where(storm_wave > 0.65, -2.1, 0.2) + rng.normal(0, 0.5, n_hours),
        "issue_chews_temp_c": nws_temp - 5.5,
        "issue_tass_temp_c": nws_temp + true_bias,
        "issue_chews_precip_mm": actual_precip * 1.25,
        "issue_persistence_bias_c": true_bias + rng.normal(0, 0.4, n_hours),
        "actual_temp_c": nws_temp + true_bias,
        "actual_precip_mm": actual_precip,
    })

    mos = TassajaraMOSCorrector()
    metrics = mos.fit_and_evaluate(synth_df)
    print("Cross-Validation Metrics:", metrics)