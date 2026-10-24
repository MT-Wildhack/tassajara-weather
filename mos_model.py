#!/usr/bin/env python3
"""
mos_model.py
Trains and evaluates a Model Output Statistics (MOS) Gradient Boosted Tree ensemble
(with 10th/90th percentile uncertainty bounds) and OLS baseline diagnostics using
scikit-learn and statsmodels.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import statsmodels.api as sm

TASS_LAT_RAD = np.radians(36.2337)
TASS_LON_DEG = -121.5501
CANYON_AXIS_RAD = np.radians(115.0)  # ESE-WNW primary axis of Tassajara Creek Canyon


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
    # 150-250 deg (S to WSW): Black Cone Trail & Pine Ridge wall (~32 deg horizon)
    # 50-110 deg (ENE to ESE): Eastern ridge horizon (~22 deg horizon)
    horizon_deg = np.full_like(solar_az_deg, 18.0)
    horizon_deg = np.where((solar_az_deg >= 145.0) & (solar_az_deg <= 255.0), 32.0, horizon_deg)
    horizon_deg = np.where((solar_az_deg >= 45.0) & (solar_az_deg <= 115.0), 23.0, horizon_deg)

    # Topographic Shadow Flag: Sun is above astronomical horizon (>0) but blocked by canyon wall
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


def engineer_mos_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds physical, cyclical, and topographic features for MOS error prediction."""
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
    out["lead_time_hours"] = df["lead_time_hours"]
    out["nws_temp_c"] = df["nws_temp_c"]
    out["nws_rh_pct"] = df["nws_rh_pct"]
    out["nws_pop_pct"] = df.get("nws_pop_pct", 0.0)
    out["nws_sky_cover_pct"] = df.get("nws_sky_cover_pct", 20.0)

    # 4. Along-Canyon & Cross-Canyon Wind Components
    w_spd = df["nws_wind_speed_ms"].fillna(0.0)
    w_dir_rad = np.radians(df["nws_wind_dir_deg"].fillna(0.0))
    out["wind_along_canyon_ms"] = w_spd * np.cos(w_dir_rad - CANYON_AXIS_RAD)
    out["wind_cross_canyon_ms"] = w_spd * np.sin(w_dir_rad - CANYON_AXIS_RAD)

    # 5. Initialized Boundary-Layer State at Issue Time
    out["pressure_trend_3hr_hpa"] = df.get("pressure_trend_3hr_hpa", 0.0).fillna(0.0)
    out["issue_lapse_chews_c_km"] = (
        -1.0 * (df["issue_chews_temp_c"] - df["issue_tass_temp_c"]) / 1.051
    ).fillna(6.5)

    # Exponentially decayed persistence bias
    init_bias = df.get("issue_persistence_bias_c", 0.0).fillna(0.0)
    out["decayed_init_bias_c"] = init_bias * np.exp(-df["lead_time_hours"] / 12.0)

    return out


class TassajaraMOSCorrector:
    """
    Ensemble MOS Forecast Corrector predicting expected forecast error (mean)
    plus 10th and 90th percentile uncertainty intervals.
    """

    def __init__(self):
        common_params = dict(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42
        )
        self.model_mean = HistGradientBoostingRegressor(loss="squared_error", **common_params)
        self.model_p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **common_params)
        self.model_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common_params)
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

        # 2. Time-Series Cross-Validation (5 Expanding Folds)
        tscv = TimeSeriesSplit(n_splits=5)
        raw_rmse_list, mos_rmse_list, raw_mae_list, mos_mae_list = [], [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_err.iloc[train_idx], y_err.iloc[test_idx]

            self.model_mean.fit(X_tr, y_tr)
            pred_err = self.model_mean.predict(X_te)

            actual_te = df["actual_temp_c"].iloc[test_idx]
            raw_fc_te = df["nws_temp_c"].iloc[test_idx]
            mos_fc_te = raw_fc_te + pred_err

            raw_rmse_list.append(root_mean_squared_error(actual_te, raw_fc_te))
            mos_rmse_list.append(root_mean_squared_error(actual_te, mos_fc_te))
            raw_mae_list.append(mean_absolute_error(actual_te, raw_fc_te))
            mos_mae_list.append(mean_absolute_error(actual_te, mos_fc_te))

        # 3. Final Fit on Full Historical Dataset
        self.model_mean.fit(X, y_err)
        self.model_p10.fit(X, y_err)
        self.model_p90.fit(X, y_err)

        mean_raw_rmse = float(np.mean(raw_rmse_list))
        mean_mos_rmse = float(np.mean(mos_rmse_list))
        skill_score = 1.0 - (mean_mos_rmse / mean_raw_rmse)

        return {
            "raw_nws_rmse_c": round(mean_raw_rmse, 3),
            "mos_corrected_rmse_c": round(mean_mos_rmse, 3),
            "raw_nws_mae_c": round(float(np.mean(raw_mae_list)), 3),
            "mos_corrected_mae_c": round(float(np.mean(mos_mae_list)), 3),
            "rmse_skill_improvement_pct": round(skill_score * 100.0, 1)
        }

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        X_fut = engineer_mos_features(df_future)[self.feature_names]
        err_mean = self.model_mean.predict(X_fut)
        err_p10 = self.model_p10.predict(X_fut)
        err_p90 = self.model_p90.predict(X_fut)

        res = df_future[["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
        res["pred_temp_error_c"] = np.round(err_mean, 2)
        res["corrected_temp_c"] = np.round(res["nws_temp_c"] + err_mean, 2)
        res["corrected_temp_p10_c"] = np.round(res["nws_temp_c"] + err_p10, 2)
        res["corrected_temp_p90_c"] = np.round(res["nws_temp_c"] + err_p90, 2)
        return res

    def save(self, filepath: str = "models/tassajara_mos_v1.joblib"):
        joblib.dump(self, filepath)


if __name__ == "__main__":
    # Synthetic verification demonstration simulating Tassajara's diurnal canyon bias
    rng = np.random.default_rng(42)
    n_hours = 24 * 90  # 90 days of hourly data
    times = pd.date_range("2026-07-25", periods=n_hours, freq="h", tz="UTC")
    solar = compute_solar_and_canyon_shadow(pd.Series(times))

    nws_temp = 18.0 + 7.0 * np.sin(2 * np.pi * (times.hour - 21) / 24.0) + rng.normal(0, 1.2, n_hours)
    # Simulate Tassajara bias: -4.5C nocturnal cold-air pool, -3.0C afternoon shadow drop, +2.0C midday roof plume
    true_bias = (
        -4.2 * (solar["solar_el_deg"] <= 0).astype(float)
        - 2.8 * solar["in_canyon_shadow"]
        + 1.9 * (solar["effective_insolation"] > 0.65).astype(float)
        + rng.normal(0, 0.7, n_hours)
    )
    synth_df = pd.DataFrame({
        "issue_time_utc": times - pd.Timedelta(hours=6),
        "valid_time_utc": times,
        "lead_time_hours": rng.integers(1, 36, n_hours),
        "nws_temp_c": nws_temp,
        "nws_rh_pct": np.clip(60.0 - 2.0 * true_bias, 10, 100),
        "nws_wind_speed_ms": rng.uniform(0.5, 6.0, n_hours),
        "nws_wind_dir_deg": rng.choice([115.0, 295.0, 45.0], n_hours),
        "nws_pop_pct": 0.0,
        "nws_sky_cover_pct": 15.0,
        "pressure_trend_3hr_hpa": rng.normal(0, 0.8, n_hours),
        "issue_chews_temp_c": nws_temp + 2.5,
        "issue_tass_temp_c": nws_temp + true_bias,
        "issue_persistence_bias_c": true_bias + rng.normal(0, 0.4, n_hours),
        "actual_temp_c": nws_temp + true_bias
    })

    mos = TassajaraMOSCorrector()
    metrics = mos.fit_and_evaluate(synth_df)
    print("Cross-Validation Metrics:", metrics)