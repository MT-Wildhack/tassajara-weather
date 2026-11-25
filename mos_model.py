#!/usr/bin/env python3
"""
mos_model.py (v5.0 — Leak-Free 30-Feature Santa Lucia Spatial MOS, Purged Temporal CV,
True OOF Conformal Quantiles & Proper Wet-Only Hydrological Hurdle)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("TassajaraMOS")
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    brier_score_loss,
)
import statsmodels.api as sm

TASS_LAT_RAD = np.radians(36.2337)
TASS_LON_DEG = -121.5501
CANYON_AXIS_RAD = np.radians(115.0)
OROGRAPHIC_UPSLOPE_RAD = np.radians(215.0)
TIPPING_BUCKET_MM = 0.254  # 0.01 inch Davis Vantage Pro2 tipping spoon


def compute_solar_and_canyon_shadow(timestamps_utc: pd.Series) -> pd.DataFrame:
    """Calculates solar altitude, azimuth, and Tassajara Canyon wall shadow indicator."""
    dt = pd.to_datetime(timestamps_utc, utc=True)
    doy = dt.dt.dayofyear.values
    hour_utc = dt.dt.hour.values + dt.dt.minute.values / 60.0

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

    time_offset = eqtime + 4.0 * TASS_LON_DEG
    tst = (hour_utc * 60.0 + time_offset) % 1440.0
    ha_rad = np.radians((tst / 4.0) - 180.0)

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
        "effective_insolation": effective_insolation_factor,
    }, index=timestamps_utc.index)


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _num_series_prefer(df: pd.DataFrame, primary: str, fallback: str, default: float = 0.0) -> pd.Series:
    fb = _num_series(df, fallback, default)
    if primary in df.columns:
        pri = pd.to_numeric(df[primary], errors="coerce")
        return pri.combine_first(fb)
    return fb


def engineer_mos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 30 physical, cyclical, topographic, 5-station spatial gradient,
    and hydrological predictors. All issue-time telemetry (including barometric
    tendency) decays exponentially with lead_time_hours so Day 2-7 forecasts
    are never biased by current weather snapshots.
    """
    out = pd.DataFrame(index=df.index)
    dt_valid = pd.to_datetime(df["valid_time_utc"], utc=True)
    dt_local = dt_valid.dt.tz_convert("America/Los_Angeles")

    hour = dt_local.dt.hour + dt_local.dt.minute / 60.0
    doy = dt_local.dt.dayofyear

    out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    solar_df = compute_solar_and_canyon_shadow(df["valid_time_utc"])
    out["solar_el_deg"] = solar_df["solar_el_deg"]
    out["in_canyon_shadow"] = solar_df["in_canyon_shadow"]
    out["effective_insolation"] = solar_df["effective_insolation"]

    lead_hrs = _num_series(df, "lead_time_hours", 6.0).clip(lower=1.0)
    nws_t_fb = _num_series(df, "nws_temp_c", _num_series(df, "raw_nws_temp_c", 15.0))
    t_c = (
        pd.to_numeric(df["om_temp_c"], errors="coerce").combine_first(nws_t_fb)
        if "om_temp_c" in df.columns
        else nws_t_fb
    )
    rh_pct = _num_series_prefer(df, "om_rh_pct", "nws_rh_pct", 50.0).clip(1.0, 100.0)
    dew_fb = _num_series(df, "nws_dewpoint_c", np.nan)
    dew_c = (
        pd.to_numeric(df["om_dewpoint_c"], errors="coerce").combine_first(dew_fb)
        if "om_dewpoint_c" in df.columns
        else dew_fb
    )
    dew_c = dew_c.fillna(t_c - ((100.0 - rh_pct) / 5.0))

    # Keep official NOAA NWS PoP/QPF for hydrological hurdle
    pop_pct = _num_series(df, "nws_pop_pct", 0.0).clip(0.0, 100.0)
    qpf_mm = _num_series(df, "nws_qpf_1hr_mm", 0.0).clip(lower=0.0)
    sky_pct = _num_series_prefer(df, "om_sky_cover_pct", "nws_sky_cover_pct", 20.0).clip(0.0, 100.0)

    out["lead_time_hours"] = lead_hrs
    out["nws_temp_c"] = t_c
    out["nws_rh_pct"] = rh_pct
    out["nws_pop_pct"] = pop_pct
    out["nws_qpf_1hr_mm"] = qpf_mm
    out["nws_sky_cover_pct"] = sky_pct

    w_spd = _num_series_prefer(df, "om_wind_speed_ms", "nws_wind_speed_ms", 0.0).clip(lower=0.0)
    cloud_frac = np.clip(sky_pct / 100.0, 0.0, 1.0)
    cloud_trans = 1.0 - 0.75 * (cloud_frac ** 3.4)
    rain_atten = np.exp(-0.35 * qpf_mm)
    out["cloud_attenuated_insolation"] = solar_df["effective_insolation"] * cloud_trans * rain_atten
    out["overcast_daytime_cooling_idx"] = (
        solar_df["effective_insolation"] * (cloud_frac ** 2) * (rh_pct / 100.0)
    )
    is_dark_or_shade = (solar_df["effective_insolation"] <= 0.0).astype(float)
    out["clear_nocturnal_cooling_idx"] = (
        is_dark_or_shade * (1.0 - 0.75 * (cloud_frac ** 2)) * np.exp(-w_spd / 3.0)
    )
    w_dir_rad = np.radians(_num_series_prefer(df, "om_wind_dir_deg", "nws_wind_dir_deg", 0.0))
    out["wind_along_canyon_ms"] = w_spd * np.cos(w_dir_rad - CANYON_AXIS_RAD)
    out["wind_cross_canyon_ms"] = w_spd * np.sin(w_dir_rad - CANYON_AXIS_RAD)
    upslope_wind_ms = np.maximum(0.0, w_spd * np.cos(w_dir_rad - OROGRAPHIC_UPSLOPE_RAD))
    out["orographic_upslope_wind_ms"] = upslope_wind_ms

    es_kpa = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea_kpa = es_kpa * (rh_pct / 100.0)
    vpd_kpa = np.maximum(0.0, es_kpa - ea_kpa)
    dew_depr_c = np.maximum(0.0, t_c - dew_c)

    out["dewpoint_depression_c"] = dew_depr_c
    out["vpd_kpa"] = vpd_kpa
    out["virga_evap_index"] = vpd_kpa * dew_depr_c * (1.0 - (rh_pct / 100.0))
    out["orographic_moisture_flux"] = upslope_wind_ms * ea_kpa
    out["qpf_moisture_scaling"] = qpf_mm * ((rh_pct / 100.0) ** 2)

    # Lead-decayed issue-time barometric tendency (prevents Day 2-7 leakage/bias)
    ptrend_3h = _num_series(df, "pressure_trend_3hr_hpa", 0.0) * np.exp(-lead_hrs / 12.0)
    out["pressure_trend_3hr_hpa"] = ptrend_3h
    out["frontal_pressure_fall_hpa"] = np.minimum(0.0, ptrend_3h)

    issue_tass_t = _num_series(df, "issue_tass_temp_c", np.nan).fillna(t_c)

    raw_chews_t = _num_series(df, "issue_chews_temp_c", np.nan)
    raw_ventana_t = _num_series(df, "issue_ventana_temp_c", np.nan)
    issue_chews_t = raw_chews_t.fillna(raw_ventana_t - 0.54).fillna(issue_tass_t - 6.83)
    issue_ventana_t = raw_ventana_t.fillna(issue_chews_t + 0.54)

    raw_hast_t = _num_series(df, "issue_hastings_temp_c", np.nan)
    raw_arroyo_t = _num_series(df, "issue_arroyo_temp_c", np.nan)
    issue_hast_t = raw_hast_t.fillna(raw_arroyo_t - 2.01).fillna(issue_tass_t - 0.60)
    issue_arroyo_t = raw_arroyo_t.fillna(issue_hast_t + 2.01).fillna(issue_tass_t + 1.41)
    issue_bigsur_t = _num_series(df, "issue_bigsur_temp_c", np.nan).fillna(issue_tass_t + 2.45)

    raw_lapse_chews = (-1.0 * (issue_chews_t - issue_tass_t) / 1.051).fillna(6.5)
    raw_lapse_ventana = (-1.0 * (issue_ventana_t - issue_tass_t) / 0.968).fillna(6.5)
    lapse_chews = 6.5 + (raw_lapse_chews - 6.5) * np.exp(-lead_hrs / 12.0)
    lapse_ventana = 6.5 + (raw_lapse_ventana - 6.5) * np.exp(-lead_hrs / 12.0)
    out["issue_lapse_chews_c_km"] = lapse_chews

    confine_hast_c = issue_tass_t - (issue_hast_t + 0.60)
    confine_arroyo_c = issue_tass_t - (issue_arroyo_t - 1.41)
    out["decayed_canyon_confinement_c"] = (
        0.55 * confine_arroyo_c + 0.45 * confine_hast_c
    ) * np.exp(-lead_hrs / 8.0)

    coastal_inland_grad_c = (issue_bigsur_t - 1.04) - issue_arroyo_t
    out["decayed_coastal_inland_grad_c"] = coastal_inland_grad_c * np.exp(-lead_hrs / 12.0)

    issue_chews_rh = _num_series(df, "issue_chews_rh_pct", np.nan).fillna(rh_pct).clip(1.0, 100.0)
    issue_bigsur_rh = _num_series(df, "issue_bigsur_rh_pct", np.nan).fillna(issue_chews_rh).clip(1.0, 100.0)
    rh_coastal_contrast = (issue_bigsur_rh - issue_chews_rh) / 100.0
    inversion_strength = np.maximum(0.0, 6.5 - (0.5 * (lapse_chews + lapse_ventana)))
    out["issue_marine_layer_cap_idx"] = (
        rh_coastal_contrast * inversion_strength * np.exp(-lead_hrs / 10.0)
    )

    issue_chews_p = _num_series(df, "issue_chews_precip_mm", 0.0).clip(lower=0.0)
    issue_ventana_p = _num_series(df, "issue_ventana_precip_mm", np.nan).fillna(issue_chews_p).clip(lower=0.0)
    issue_bigsur_p = _num_series(df, "issue_bigsur_precip_mm", np.nan).fillna(0.0).clip(lower=0.0)

    out["decayed_chews_precip_mm"] = issue_chews_p * np.exp(-lead_hrs / 6.0)

    ssw_weight = np.clip(0.5 + 0.15 * upslope_wind_ms, 0.5, 1.5)
    windward_crest_p = np.maximum(
        issue_chews_p,
        np.maximum(issue_ventana_p, 0.75 * issue_bigsur_p) * ssw_weight,
    )
    out["decayed_ssw_spillover_precip_mm"] = windward_crest_p * np.exp(-lead_hrs / 6.0)

    init_bias = _num_series(df, "issue_persistence_bias_c", 0.0)
    out["decayed_init_bias_c"] = init_bias * np.exp(-lead_hrs / 12.0)

    return out


def build_well_conditioned_ols_matrix(X_df: pd.DataFrame, max_r2: float = 0.85) -> pd.DataFrame:
    """
    Standardizes (Z-scores) predictors and explicitly PRUNES strictly constant or
    near-collinear derived predictors (R^2 > 0.94 / VIF > 16.7) rather than silently
    replacing them with Gram-Schmidt residuals under original feature names.
    """
    X_work = X_df.copy()
    kept_cols = []
    for col in X_work.columns:
        s = pd.to_numeric(X_work[col], errors="coerce").fillna(0.0)
        std_val = float(s.std())
        if std_val > 1e-6:
            X_work[col] = (s - float(s.mean())) / std_val
            kept_cols.append(col)

    X_std = X_work[kept_cols].copy()
    accepted_cols = []
    for col in X_std.columns:
        vec = X_std[col].values
        if accepted_cols:
            B = X_std[accepted_cols].values
            beta, _, _, _ = np.linalg.lstsq(B, vec, rcond=None)
            resid = vec - B @ beta
            r_sq = 1.0 - (np.var(resid) / max(np.var(vec), 1e-12))
            if r_sq > max_r2:
                continue
        accepted_cols.append(col)

    return sm.add_constant(X_std[accepted_cols], has_constant="add")


def analytical_hydro_hurdle_fallback(X: pd.DataFrame) -> pd.DataFrame:
    """
    Physical Orographic-Virga Hurdle baseline for lee-side Tassajara Canyon (485 m MSL).
    Uses smooth logistic virga attenuation (no cliff at RH=70%) and realistic lee-side
    orographic scaling (0.82 lee shadow baseline up to 1.18 max during SSW IVT spillover).
    """
    raw_pop = X["nws_pop_pct"].values
    raw_qpf = X["nws_qpf_1hr_mm"].values
    rh = X["nws_rh_pct"].values
    vpd = X["vpd_kpa"].values
    upslope_flux = X["orographic_moisture_flux"].values
    p_fall = X["frontal_pressure_fall_hpa"].values
    spillover_seed = (
        X["decayed_ssw_spillover_precip_mm"].values
        if "decayed_ssw_spillover_precip_mm" in X.columns
        else X["decayed_chews_precip_mm"].values
    )

    # Smooth continuous virga weight across RH [60%, 85%] and rain intensity [0, 3 mm/hr]
    dry_subcloud_w = 1.0 / (1.0 + np.exp((rh - 72.0) / 5.0))
    light_rain_w = np.exp(-raw_qpf / 1.5)
    virga_factor = 1.0 - (dry_subcloud_w * light_rain_w) * (1.0 - np.exp(-0.45 * vpd))

    # Lee-side Santa Lucia scaling relative to 880 m NWP grid (capped at 1.18 to prevent over-inflation)
    orographic_factor = (
        0.82
        + 0.045 * np.clip(upslope_flux, 0.0, 4.5)
        + 0.03 * np.clip(-p_fall, 0.0, 3.0)
        + 0.035 * np.clip(spillover_seed, 0.0, 3.0)
    )
    net_multiplier = np.clip(virga_factor * orographic_factor, 0.20, 1.18)

    mos_pop = np.clip(
        raw_pop * np.clip(virga_factor * (1.0 + 0.03 * upslope_flux), 0.30, 1.25)
        + np.where(raw_qpf >= TIPPING_BUCKET_MM, 8.0 * (rh / 100.0), -5.0 * (vpd > 1.3)),
        0.0,
        100.0,
    )

    pop_frac = mos_pop / 100.0
    mos_qpf = raw_qpf * net_multiplier
    mos_qpf = np.where((mos_pop < 15.0) & (raw_qpf < TIPPING_BUCKET_MM), 0.0, mos_qpf)

    mos_qpf_p90 = np.where(
        mos_qpf > 0.0,
        np.maximum(mos_qpf * 1.45, mos_qpf + 0.35 * pop_frac * orographic_factor),
        0.0,
    )

    return pd.DataFrame({
        "mos_pop_pct": np.round(mos_pop, 1),
        "mos_qpf_mm": np.round(mos_qpf, 3),
        "mos_qpf_p90_mm": np.round(mos_qpf_p90, 3),
    }, index=X.index)


class TassajaraMOSCorrector:
    """
    Multi-Task Ensemble MOS Forecast Corrector:
      - Thermal Subsystem: Mean + P10/P90 quantile regressors with out-of-fold
        conformalized quantile regression (CQR) pads targeting exact 80% coverage.
      - Hydrological Subsystem: True Two-Stage Hurdle (temporally calibrated PoP
        classifier + wet-only Stage-2 QPF/P90 regressors).
    """

    def __init__(self):
        common_temp_params = dict(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42,
        )
        self.model_mean = HistGradientBoostingRegressor(loss="squared_error", **common_temp_params)
        self.model_p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **common_temp_params)
        self.model_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common_temp_params)

        self.base_pop_clf = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=25,
            min_samples_leaf=20,
            l2_regularization=2.5,
            random_state=42,
        )
        self.model_pop = None

        self.model_qpf_mean = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=20,
            min_samples_leaf=12,
            l2_regularization=2.5,
            random_state=42,
        )
        self.model_qpf_p90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=20,
            min_samples_leaf=12,
            l2_regularization=2.5,
            random_state=42,
        )

        self.precip_ml_trained = False
        self.feature_names = []
        self.ols_summary = None
        self.ols_diagnostics = {}
        self.precomputed_importances = {}
        self.q_scale_low = 1.0
        self.q_scale_high = 1.0
        self.conformal_pad_low_c = 0.0
        self.conformal_pad_high_c = 0.0
        self.qpf_uses_log_ratio = True
        self.oof_predictions_df = None

    def _fit_hydro_models(self, X_tr: pd.DataFrame, y_p_tr: pd.Series):
        y_wet = (y_p_tr >= TIPPING_BUCKET_MM).astype(int)
        if int(y_wet.sum()) < 30 or len(np.unique(y_wet)) < 2:
            return None, None, None

        clf_base = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=25,
            min_samples_leaf=20, l2_regularization=2.5, random_state=42,
        )
        try:
            gap_cal = min(24, max(0, len(X_tr) // 20))
            inner_splits = [
                (tr_i, va_i)
                for tr_i, va_i in TimeSeriesSplit(n_splits=3, gap=gap_cal).split(X_tr)
                if int(y_wet.iloc[tr_i].sum()) >= 5 and int(y_wet.iloc[va_i].sum()) >= 5
            ]
            if len(inner_splits) >= 2:
                cal_cv = CalibratedClassifierCV(
                    estimator=clf_base,
                    method="sigmoid",
                    cv=inner_splits,
                )
                cal_cv.fit(X_tr, y_wet)
                m_pop = cal_cv
            else:
                logger.warning("Skipping Platt calibration: fewer than 2 inner folds with >= 5 wet hours.")
                clf_base.fit(X_tr, y_wet)
                m_pop = clf_base
        except Exception as exc:
            logger.warning(f"CalibratedClassifierCV fallback to raw PoP probabilities: {exc}")
            clf_base.fit(X_tr, y_wet)
            m_pop = clf_base

        wet_mask = y_p_tr >= TIPPING_BUCKET_MM
        X_wet = X_tr.loc[wet_mask]
        y_wet_vals = y_p_tr.loc[wet_mask].values
        q_raw_wet = np.clip(X_wet["nws_qpf_1hr_mm"].values, 0.0, None)
        # Bounded log-ratio target preserves linear scaling into extreme Atmospheric River intensities
        z_target = np.clip(np.log((y_wet_vals + 0.5) / (q_raw_wet + 0.5)), -1.2, 0.85)
        m_qpf = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=300, learning_rate=0.04,
            max_leaf_nodes=20, min_samples_leaf=12, l2_regularization=2.5, random_state=42,
        )
        m_p90 = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.90, max_iter=300, learning_rate=0.04,
            max_leaf_nodes=20, min_samples_leaf=12, l2_regularization=2.5, random_state=42,
        )
        m_qpf.fit(X_wet, z_target)
        m_p90.fit(X_wet, z_target)
        return m_pop, m_qpf, m_p90

    @staticmethod
    def _augment_training_leads(
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        seed: int = 42,
        sample_weight: np.ndarray = None,
    ):
        """Fold-internal synthetic lead augmentation so Day 2-7 decayed features stay in-distribution."""
        if len(X_tr) < 200:
            return (X_tr, y_tr, sample_weight) if sample_weight is not None else (X_tr, y_tr)
        rng = np.random.default_rng(seed)
        n_aug = min(len(X_tr), max(400, len(X_tr) // 3))
        idx_aug = rng.choice(len(X_tr), size=n_aug, replace=False)
        X_aug = X_tr.iloc[idx_aug].copy()
        y_aug = y_tr.iloc[idx_aug].copy()
        orig_lead = np.clip(X_aug["lead_time_hours"].values, 1.0, None)
        syn_lead = rng.choice(np.array([18.0, 36.0, 72.0, 120.0]), size=n_aug)
        X_aug["lead_time_hours"] = syn_lead
        for col, tau in (
            ("decayed_init_bias_c", 12.0),
            ("pressure_trend_3hr_hpa", 12.0),
            ("frontal_pressure_fall_hpa", 12.0),
            ("decayed_coastal_inland_grad_c", 12.0),
            ("decayed_canyon_confinement_c", 8.0),
            ("issue_marine_layer_cap_idx", 10.0),
            ("decayed_chews_precip_mm", 6.0),
            ("decayed_ssw_spillover_precip_mm", 6.0),
        ):
            if col in X_aug.columns:
                X_aug[col] = X_aug[col].values * np.exp(-(syn_lead - orig_lead) / tau)
        if "issue_lapse_chews_c_km" in X_aug.columns:
            X_aug["issue_lapse_chews_c_km"] = 6.5 + (
                X_aug["issue_lapse_chews_c_km"].values - 6.5
            ) * np.exp(-(syn_lead - orig_lead) / 12.0)
        X_out = pd.concat([X_tr, X_aug], ignore_index=True)
        y_out = pd.concat([y_tr, y_aug], ignore_index=True)
        if sample_weight is not None:
            w_out = np.concatenate([sample_weight, sample_weight[idx_aug]])
            return X_out, y_out, w_out
        return X_out, y_out

    def _blend_hydro_predictions(
        self, X_fut: pd.DataFrame, m_pop, m_qpf, m_p90
    ) -> pd.DataFrame:
        phys_df = analytical_hydro_hurdle_fallback(X_fut)
        if m_pop is None or m_qpf is None or m_p90 is None:
            return phys_df

        X_aligned = X_fut[self.feature_names] if self.feature_names else X_fut
        prob_wet = np.clip(m_pop.predict_proba(X_aligned)[:, 1], 0.0, 1.0)
        mos_pop_pct = np.clip(
            0.45 * (prob_wet * 100.0) + 0.55 * phys_df["mos_pop_pct"].values,
            0.0,
            100.0,
        )
        p_wet_blend = mos_pop_pct / 100.0

        raw_qpf_arr = (
            np.clip(X_fut["nws_qpf_1hr_mm"].values, 0.0, None)
            if "nws_qpf_1hr_mm" in X_fut.columns
            else np.zeros(len(X_fut))
        )
        if getattr(self, "qpf_uses_log_ratio", False):
            z_mean = np.clip(m_qpf.predict(X_aligned), -1.2, 0.85)
            z_p90 = np.maximum(z_mean, np.clip(m_p90.predict(X_aligned), -1.2, 1.15))
            cond_qpf = np.maximum(0.0, (raw_qpf_arr + 0.5) * np.exp(z_mean) - 0.5)
            cond_p90 = np.maximum(cond_qpf, (raw_qpf_arr + 0.5) * np.exp(z_p90) - 0.5)
        else:
            cond_qpf = np.clip(m_qpf.predict(X_aligned), 0.0, None)
            cond_p90 = np.maximum(cond_qpf, np.clip(m_p90.predict(X_aligned), 0.0, None))
        active_mask = (mos_pop_pct >= 15.0) | (raw_qpf_arr >= TIPPING_BUCKET_MM)

        # Mathematically exact hurdle expectation: E[Y] = P(wet) * E[Y | wet]
        ml_qpf_mm = np.where(active_mask, p_wet_blend * cond_qpf, 0.0)
        ml_p90_mm = np.where(active_mask, np.maximum(ml_qpf_mm, (p_wet_blend ** 0.5) * cond_p90), 0.0)

        # Blend with physics (allowing ML to learn both lee suppression and spillover)
        w_ml = np.where(raw_qpf_arr >= 5.0, 0.35, 0.55)
        mos_qpf_mm = np.where(
            active_mask,
            w_ml * ml_qpf_mm + (1.0 - w_ml) * phys_df["mos_qpf_mm"].values,
            0.0,
        )
        mos_qpf_p90_mm = np.where(
            mos_qpf_mm > 0.0,
            np.maximum(
                mos_qpf_mm * 1.30,
                0.55 * ml_p90_mm + 0.45 * phys_df["mos_qpf_p90_mm"].values,
            ),
            0.0,
        )

        return pd.DataFrame({
            "mos_pop_pct": np.round(mos_pop_pct, 1),
            "mos_qpf_mm": np.round(mos_qpf_mm, 3),
            "mos_qpf_p90_mm": np.round(mos_qpf_p90_mm, 3),
        }, index=X_fut.index)

    def fit_and_evaluate(self, df: pd.DataFrame) -> dict:
        df_sorted = (
            df.sort_values("valid_time_utc")
            .drop_duplicates(subset=["valid_time_utc"], keep="last")
            .reset_index(drop=True)
        )
        X = engineer_mos_features(df_sorted)
        self.feature_names = list(X.columns)
        y_err = df_sorted["actual_temp_c"] - df_sorted["nws_temp_c"]
        y_precip = _num_series(df_sorted, "actual_precip_mm", 0.0).clip(lower=0.0)
        y_wet = (y_precip >= TIPPING_BUCKET_MM).astype(int)

        # 1. Standardized Newey-West HAC OLS Baseline (explicit collinearity pruning)
        X_ols = build_well_conditioned_ols_matrix(X)
        ols_fit = sm.OLS(y_err.values, X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        self.ols_summary = ols_fit.summary().as_text()
        ci_df = ols_fit.conf_int()
        self.ols_diagnostics = {
            "ols_summary_text": self.ols_summary,
            "ols_r2": float(ols_fit.rsquared),
            "ols_adj_r2": float(ols_fit.rsquared_adj),
            "ols_fstat": float(ols_fit.fvalue) if ols_fit.fvalue is not None else np.nan,
            "ols_aic": float(ols_fit.aic),
            "ols_cond_no": float(ols_fit.condition_number),
            "ols_nobs": int(ols_fit.nobs),
            "coef_records": [
                {
                    "feat": str(feat),
                    "beta": round(float(ols_fit.params[feat]), 4),
                    "bse": round(float(ols_fit.bse[feat]), 4),
                    "tval": round(float(ols_fit.tvalues[feat]), 2),
                    "pval": round(float(ols_fit.pvalues[feat]), 4),
                    "ci_low": round(float(ci_df.loc[feat, 0]), 4),
                    "ci_high": round(float(ci_df.loc[feat, 1]), 4),
                }
                for feat in ols_fit.params.index
            ],
        }

        # 2. Purged 5-Fold Time-Series CV (48-hour embargo) + True OOF CQR & Hydro Evaluation
        embargo = 48 if len(X) >= 400 else 0
        tscv = TimeSeriesSplit(n_splits=5, gap=embargo)
        raw_rmse_list, mos_rmse_list, raw_mae_list, mos_mae_list = [], [], [], []
        cqr_low_scores, cqr_high_scores = [], []
        oof_qpf_true, oof_qpf_raw, oof_qpf_mos = [], [], []
        oof_wet_true, oof_pop_raw, oof_pop_mos = [], [], []
        oof_frames = []
        last_fold_state = None

        dt_valid_all = pd.to_datetime(df_sorted["valid_time_utc"], utc=True)
        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_err.iloc[train_idx], y_err.iloc[test_idx]
            age_days_tr = (dt_valid_all.iloc[train_idx].max() - dt_valid_all.iloc[train_idx]).dt.total_seconds().values / 86400.0
            w_tr = np.maximum(0.30, np.exp(-age_days_tr / 270.0))
            X_tr_aug, y_tr_aug, w_tr_aug = self._augment_training_leads(
                X_tr, y_tr, seed=42 + fold_i, sample_weight=w_tr
            )

            self.model_mean.fit(X_tr_aug, y_tr_aug, sample_weight=w_tr_aug)
            self.model_p10.fit(X_tr_aug, y_tr_aug, sample_weight=w_tr_aug)
            self.model_p90.fit(X_tr_aug, y_tr_aug, sample_weight=w_tr_aug)

            pred_err = self.model_mean.predict(X_te)
            pred_p10 = np.minimum(self.model_p10.predict(X_te), pred_err - 0.35)
            pred_p90 = np.maximum(self.model_p90.predict(X_te), pred_err + 0.35)

            cqr_low_scores.extend((pred_p10 - y_te.values).tolist())
            cqr_high_scores.extend((y_te.values - pred_p90).tolist())

            actual_te = df_sorted["actual_temp_c"].iloc[test_idx]
            raw_fc_te = df_sorted["nws_temp_c"].iloc[test_idx]
            mos_fc_te = raw_fc_te + pred_err

            raw_rmse_list.append(root_mean_squared_error(actual_te, raw_fc_te))
            mos_rmse_list.append(root_mean_squared_error(actual_te, mos_fc_te))
            raw_mae_list.append(mean_absolute_error(actual_te, raw_fc_te))
            mos_mae_list.append(mean_absolute_error(actual_te, mos_fc_te))

            # Out-of-fold hydrological validation
            m_pop_f, m_qpf_f, m_p90_f = self._fit_hydro_models(X_tr, y_precip.iloc[train_idx])
            h_oof = self._blend_hydro_predictions(X_te, m_pop_f, m_qpf_f, m_p90_f)
            oof_qpf_true.extend(y_precip.iloc[test_idx].tolist())
            oof_qpf_raw.extend(X_te["nws_qpf_1hr_mm"].tolist())
            oof_qpf_mos.extend(h_oof["mos_qpf_mm"].tolist())
            oof_wet_true.extend(y_wet.iloc[test_idx].tolist())
            oof_pop_raw.extend((X_te["nws_pop_pct"].values / 100.0).clip(0.0, 1.0).tolist())
            oof_pop_mos.extend((h_oof["mos_pop_pct"].values / 100.0).clip(0.0, 1.0).tolist())

            fold_df = df_sorted.iloc[test_idx][["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
            fold_df["pred_temp_error_c"] = np.round(pred_err, 2)
            fold_df["corrected_temp_c"] = np.round(mos_fc_te.values, 2)
            fold_df["corrected_temp_p10_c"] = np.round(raw_fc_te.values + pred_p10, 2)
            fold_df["corrected_temp_p90_c"] = np.round(raw_fc_te.values + pred_p90, 2)
            fold_df["raw_nws_pop_pct"] = np.round(X_te["nws_pop_pct"].values, 1)
            fold_df["mos_pop_pct"] = h_oof["mos_pop_pct"].values
            fold_df["raw_nws_qpf_mm"] = np.round(X_te["nws_qpf_1hr_mm"].values, 3)
            fold_df["mos_qpf_mm"] = h_oof["mos_qpf_mm"].values
            fold_df["mos_qpf_p90_mm"] = h_oof["mos_qpf_p90_mm"].values
            oof_frames.append(fold_df)
            last_fold_state = (X_te.copy(), y_te.copy(), y_precip.iloc[test_idx].copy(), m_qpf_f)

        self.oof_predictions_df = pd.concat(oof_frames, axis=0) if oof_frames else pd.DataFrame()

        # True OOF Conformalized Quantile Regression (CQR) asymmetric pads for 80% coverage
        self.conformal_pad_low_c = float(np.clip(np.percentile(cqr_low_scores, 90), -0.5, 2.5)) if cqr_low_scores else 0.0
        self.conformal_pad_high_c = float(np.clip(np.percentile(cqr_high_scores, 90), -0.5, 2.5)) if cqr_high_scores else 0.0
        self.q_scale_low = 1.0
        self.q_scale_high = 1.0

        # Compute Fold-5 OOF Permutation Importances BEFORE full-sample refit
        try:
            if last_fold_state is not None:
                X_f5, y_err_f5, y_p_f5, m_qpf_f5 = last_fold_state
                perm_t = permutation_importance(
                    self.model_mean, X_f5, y_err_f5.values,
                    scoring="neg_root_mean_squared_error", n_repeats=4, random_state=42,
                )
                t_raw = np.maximum(0.0, perm_t.importances_mean)
                t_sum = float(t_raw.sum())
                t_pct = (t_raw / t_sum * 100.0) if t_sum > 0 else np.zeros_like(t_raw)

                wet_f5_idx = y_p_f5[y_p_f5 >= TIPPING_BUCKET_MM].index
                if m_qpf_f5 is not None and len(wet_f5_idx) >= 20:
                    q_raw_f5 = np.clip(X_f5.loc[wet_f5_idx, "nws_qpf_1hr_mm"].values, 0.0, None)
                    z_f5 = np.clip(np.log((y_p_f5.loc[wet_f5_idx].values + 0.5) / (q_raw_f5 + 0.5)), -1.2, 0.85)
                    perm_h = permutation_importance(
                        m_qpf_f5, X_f5.loc[wet_f5_idx], z_f5,
                        scoring="neg_root_mean_squared_error", n_repeats=4, random_state=42,
                    )
                    h_raw = np.maximum(0.0, perm_h.importances_mean)
                else:
                    h_raw = np.zeros_like(t_raw)
                h_sum = float(h_raw.sum())
                h_pct = (h_raw / h_sum * 100.0) if h_sum > 0 else np.zeros_like(h_raw)
                self.precomputed_importances = {
                    "thermal_pct": {f: round(float(v), 2) for f, v in zip(self.feature_names, t_pct)},
                    "hydro_pct": {f: round(float(v), 2) for f, v in zip(self.feature_names, h_pct)},
                }
        except Exception:
            self.precomputed_importances = {}

        # Final full-sample refit (with lead-time augmentation)
        age_days_full = (dt_valid_all.max() - dt_valid_all).dt.total_seconds().values / 86400.0
        w_full = np.maximum(0.30, np.exp(-age_days_full / 270.0))
        X_full_aug, y_full_aug, w_full_aug = self._augment_training_leads(
            X, y_err, seed=99, sample_weight=w_full
        )
        self.model_mean.fit(X_full_aug, y_full_aug, sample_weight=w_full_aug)
        self.model_p10.fit(X_full_aug, y_full_aug, sample_weight=w_full_aug)
        self.model_p90.fit(X_full_aug, y_full_aug, sample_weight=w_full_aug)

        m_pop, m_qpf, m_p90 = self._fit_hydro_models(X, y_precip)
        if m_pop is not None:
            self.model_pop = m_pop
            self.model_qpf_mean = m_qpf
            self.model_qpf_p90 = m_p90
            self.precip_ml_trained = True
        else:
            self.precip_ml_trained = False

        # Honest OOF hydrological metrics
        raw_qpf_rmse = float(root_mean_squared_error(oof_qpf_true, oof_qpf_raw)) if oof_qpf_true else 0.0
        mos_qpf_rmse = float(root_mean_squared_error(oof_qpf_true, oof_qpf_mos)) if oof_qpf_true else 0.0
        has_oof_wet = sum(oof_wet_true) > 0
        raw_brier = float(brier_score_loss(oof_wet_true, oof_pop_raw)) if has_oof_wet else 0.0
        mos_brier = float(brier_score_loss(oof_wet_true, oof_pop_mos)) if has_oof_wet else 0.0

        # (Precomputed importances already scored strictly out-of-fold on Fold 5 above)

        mean_raw_rmse = float(np.mean(raw_rmse_list))
        mean_mos_rmse = float(np.mean(mos_rmse_list))
        skill_score = 1.0 - (mean_mos_rmse / max(mean_raw_rmse, 1e-4))
        qpf_skill_score = (
            (1.0 - (mos_qpf_rmse / max(raw_qpf_rmse, 1e-4))) * 100.0
            if raw_qpf_rmse > 0 else 0.0
        )

        return {
            "raw_nws_rmse_c": round(mean_raw_rmse, 3),
            "mos_corrected_rmse_c": round(mean_mos_rmse, 3),
            "raw_nws_mae_c": round(float(np.mean(raw_mae_list)), 3),
            "mos_corrected_mae_c": round(float(np.mean(mos_mae_list)), 3),
            "rmse_skill_improvement_pct": round(skill_score * 100.0, 1),
            "wet_hours_count": int(y_wet.sum()),
            "precip_ml_trained": self.precip_ml_trained,
            "raw_qpf_rmse_mm": round(raw_qpf_rmse, 3),
            "mos_qpf_rmse_mm": round(mos_qpf_rmse, 3),
            "qpf_skill_improvement_pct": round(qpf_skill_score, 1),
            "raw_pop_brier": round(raw_brier, 4),
            "mos_pop_brier": round(mos_brier, 4),
        }

    def _predict_hydro(self, X_fut: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "precip_ml_trained", False):
            return analytical_hydro_hurdle_fallback(X_fut)
        try:
            return self._blend_hydro_predictions(
                X_fut, self.model_pop, self.model_qpf_mean, self.model_qpf_p90
            )
        except Exception:
            return analytical_hydro_hurdle_fallback(X_fut)

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        X_full = engineer_mos_features(df_future)
        for col in self.feature_names:
            if col not in X_full.columns:
                X_full[col] = 0.0
        X_fut = X_full[self.feature_names] if self.feature_names else X_full

        err_mean = self.model_mean.predict(X_fut)
        err_p10 = np.minimum(self.model_p10.predict(X_fut), err_mean - 0.35)
        err_p90 = np.maximum(self.model_p90.predict(X_fut), err_mean + 0.35)

        pad_low = float(getattr(self, "conformal_pad_low_c", 0.0))
        pad_high = float(getattr(self, "conformal_pad_high_c", 0.0))
        lead_hrs = X_full["lead_time_hours"].values if "lead_time_hours" in X_full.columns else np.full(len(X_full), 6.0)
        lead_inflation = np.sqrt(1.0 + 0.012 * np.maximum(0.0, lead_hrs - 6.0))
        half_low = np.maximum(0.35, ((err_mean - err_p10) + pad_low) * lead_inflation)
        half_high = np.maximum(0.35, ((err_p90 - err_mean) + pad_high) * lead_inflation)

        hydro_df = self._predict_hydro(X_fut)

        res = df_future[["issue_time_utc", "valid_time_utc", "nws_temp_c"]].copy()
        base_model_t = X_full["nws_temp_c"].values
        corr_c = base_model_t + err_mean
        official_nws_t = (
            pd.to_numeric(res["nws_temp_c"], errors="coerce")
            .fillna(pd.Series(base_model_t, index=res.index))
            .values
        )
        res["pred_temp_error_c"] = np.round(corr_c - official_nws_t, 2)
        res["corrected_temp_c"] = np.round(corr_c, 2)
        res["corrected_temp_p10_c"] = np.round(corr_c - half_low, 2)
        res["corrected_temp_p90_c"] = np.round(corr_c + half_high, 2)

        res["raw_nws_pop_pct"] = np.round(X_full["nws_pop_pct"].values, 1)
        res["mos_pop_pct"] = hydro_df["mos_pop_pct"].values
        res["raw_nws_qpf_mm"] = np.round(
            _num_series(df_future, "nws_qpf_1hr_mm", 0.0).clip(lower=0.0).values, 3
        )
        res["mos_qpf_mm"] = hydro_df["mos_qpf_mm"].values
        res["mos_qpf_p90_mm"] = hydro_df["mos_qpf_p90_mm"].values
        return res

    def save(self, filepath: str = "models/tassajara_mos_v1.joblib"):
        joblib.dump(self, filepath)


def compute_diurnal_ewma_bias(
    conn,
    anchor_utc_iso: str,
    lookback_days: int = 14,
    half_life_days: float = 5.0,
    shrinkage_n0: float = 6.0,
    max_abs_bias_c: float = 3.5,
) -> dict:
    """
    Causal 14-day Diurnal EWMA residual bias estimator across six 4-hour PST/PDT blocks.
    Evaluates strictly pre-EWMA dual-baseline residuals (actual - unadjusted_mos) on
    valid_time_utc < anchor_utc_iso to prevent feedback oscillation or leakage.
    """
    try:
        m_cols = {r[1] for r in conn.execute("PRAGMA table_info(mos_corrected_forecasts)").fetchall()}
        if "ewma_bias_applied_c" not in m_cols:
            return {}
        anchor_dt = pd.to_datetime(anchor_utc_iso, utc=True)
        start_iso = (anchor_dt - pd.Timedelta(days=int(lookback_days))).isoformat()
        df_res = pd.read_sql_query(
            """
            SELECT
                o.valid_time_utc,
                o.temp_c AS actual_temp_c,
                m.corrected_temp_c,
                COALESCE(m.ewma_bias_applied_c, 0.0) AS ewma_bias_applied_c
            FROM local_observations o
            JOIN (
                SELECT
                    valid_time_utc,
                    issue_time_utc,
                    corrected_temp_c,
                    ewma_bias_applied_c,
                    ROW_NUMBER() OVER (
                        PARTITION BY valid_time_utc
                        ORDER BY
                            ABS((julianday(valid_time_utc) - julianday(issue_time_utc)) * 24.0 - 6.0) ASC,
                            issue_time_utc DESC
                    ) AS rn
                FROM mos_corrected_forecasts
                WHERE model_version = 'MOS_GBM_v4_DUAL'
                  AND issue_time_utc < valid_time_utc
            ) m ON o.valid_time_utc = m.valid_time_utc AND m.rn = 1
            WHERE o.temp_c IS NOT NULL
              AND o.qc_flag = 'OK'
              AND COALESCE(o.source, 'WL_LIVE') != 'OPENMETEO_SYNTH'
              AND o.valid_time_utc < ?
              AND o.valid_time_utc >= ?
            ORDER BY o.valid_time_utc ASC
            """,
            conn,
            params=(anchor_dt.isoformat(), start_iso),
        )
        if df_res.empty:
            return {}
        dt_v = pd.to_datetime(df_res["valid_time_utc"], utc=True)
        dt_loc = dt_v.dt.tz_convert("America/Los_Angeles")
        blocks = (dt_loc.dt.hour // 4).values
        age_days = np.maximum(0.0, (anchor_dt - dt_v).dt.total_seconds().values / 86400.0)
        weights = np.exp(-np.log(2.0) * age_days / float(half_life_days))
        unadj_c = df_res["corrected_temp_c"].values - df_res["ewma_bias_applied_c"].values
        resid_c = df_res["actual_temp_c"].values - unadj_c

        bias_by_block = {}
        for b in range(6):
            mask = blocks == b
            if not np.any(mask):
                continue
            w_b = weights[mask]
            r_b = resid_c[mask]
            val = float(np.sum(w_b * r_b) / (np.sum(w_b) + float(shrinkage_n0)))
            bias_by_block[int(b)] = round(float(np.clip(val, -max_abs_bias_c, max_abs_bias_c)), 3)
        return bias_by_block
    except Exception:
        return {}


def predict_dual_baseline_with_ewma(
    corrector: TassajaraMOSCorrector,
    df_future: pd.DataFrame,
    ewma_bias_by_block: dict = None,
    om_weight: float = 0.50,
) -> pd.DataFrame:
    """
    Dual-Baseline Consensus + Adaptive Diurnal EWMA:
      1. Evaluates MOS on its native Open-Meteo 880m thermal/wind/sky inputs (preserving official NWS PoP/QPF).
      2. Blends MOS_OM 50/50 with official NOAA NWS MTR temperature where both exist.
      3. Applies causal 4-hour-block diurnal EWMA bias correction and preserves calibrated P10-P90 half-widths.
    """
    df_om = df_future.copy()
    raw_nws_t = _num_series(df_future, "nws_temp_c", _num_series(df_future, "raw_nws_temp_c", 15.0)).values

    has_om = pd.Series(False, index=df_future.index)
    if "om_temp_c" in df_future.columns:
        om_t_s = pd.to_numeric(df_future["om_temp_c"], errors="coerce")
        has_om = om_t_s.notna()
        if has_om.any():
            for om_col, nws_col in (
                ("om_temp_c", "nws_temp_c"),
                ("om_dewpoint_c", "nws_dewpoint_c"),
                ("om_rh_pct", "nws_rh_pct"),
                ("om_sky_cover_pct", "nws_sky_cover_pct"),
                ("om_wind_speed_ms", "nws_wind_speed_ms"),
                ("om_wind_dir_deg", "nws_wind_dir_deg"),
            ):
                if om_col in df_future.columns:
                    vals = pd.to_numeric(df_future[om_col], errors="coerce")
                    if nws_col in df_om.columns:
                        df_om[nws_col] = vals.combine_first(pd.to_numeric(df_om[nws_col], errors="coerce"))
                    else:
                        df_om[nws_col] = vals
            if "issue_om_persistence_bias_c" in df_future.columns:
                df_om["issue_persistence_bias_c"] = pd.to_numeric(
                    df_future["issue_om_persistence_bias_c"], errors="coerce"
                ).fillna(0.0).clip(-6.0, 6.0)
            elif "issue_tass_temp_c" in df_future.columns:
                first_om_t = float(om_t_s.dropna().iloc[0])
                orig_bias = pd.to_numeric(df_future.get("issue_persistence_bias_c", 0.0), errors="coerce").fillna(0.0)
                tass_iss = pd.to_numeric(df_future["issue_tass_temp_c"], errors="coerce").fillna(first_om_t)
                df_om["issue_persistence_bias_c"] = np.where(
                    orig_bias.abs() > 1e-6,
                    np.clip(tass_iss - first_om_t, -6.0, 6.0),
                    0.0,
                )

    preds_om = corrector.predict(df_om)
    mos_om_c = preds_om["corrected_temp_c"].values
    half_low = np.maximum(0.35, mos_om_c - preds_om["corrected_temp_p10_c"].values)
    half_high = np.maximum(0.35, preds_om["corrected_temp_p90_c"].values - mos_om_c)

    w_om = float(np.clip(om_weight, 0.0, 1.0))
    blended_c = np.where(has_om.values, w_om * mos_om_c + (1.0 - w_om) * raw_nws_t, mos_om_c)

    dt_loc = pd.to_datetime(df_future["valid_time_utc"], utc=True).dt.tz_convert("America/Los_Angeles")
    blocks = (dt_loc.dt.hour // 4).values
    ewma_map = ewma_bias_by_block or {}
    ewma_arr = np.array([float(ewma_map.get(int(b), 0.0)) for b in blocks], dtype=float)

    final_c = blended_c + ewma_arr
    res = preds_om.copy()
    res["nws_temp_c"] = np.round(raw_nws_t, 2)
    res["pred_temp_error_c"] = np.round(final_c - raw_nws_t, 2)
    res["corrected_temp_c"] = np.round(final_c, 2)
    res["corrected_temp_p10_c"] = np.round(final_c - half_low, 2)
    res["corrected_temp_p90_c"] = np.round(final_c + half_high, 2)
    res["ewma_bias_applied_c"] = np.round(ewma_arr, 3)
    return res
