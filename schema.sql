-- schema.sql — Tassajara Zen Mountain Center (TZMC) Microclimate & Hydrological MOS Schema
-- Supports Station 36966 (Stone Office Vantage Pro2) and the 5-Station Santa Lucia Array
-- (CHWC1 Chews Ridge, VTCC1 Ventana Cone, CAHC1 Hastings, ASRC1 Arroyo Seco, PPSC1 Big Sur)

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS local_observations (
    valid_time_utc TEXT PRIMARY KEY,
    station_id TEXT DEFAULT 'TASS_VP2',
    temp_c REAL,
    dewpoint_c REAL,
    wetbulb_c REAL,
    rh_pct REAL,
    wind_speed_ms REAL DEFAULT 0.0,
    wind_gust_ms REAL DEFAULT 0.0,
    wind_dir_deg REAL DEFAULT 0.0,
    mslp_hpa REAL,
    pressure_trend_3hr_hpa REAL DEFAULT 0.0,
    precip_1hr_mm REAL DEFAULT 0.0,
    solar_rad_wm2 REAL,
    uv_index REAL,
    qc_flag TEXT DEFAULT 'OK'
);

CREATE INDEX IF NOT EXISTS idx_local_obs_time ON local_observations(valid_time_utc);

CREATE TABLE IF NOT EXISTS reference_observations (
    valid_time_utc TEXT,
    station_id TEXT,
    temp_c REAL,
    rh_pct REAL,
    wind_speed_ms REAL DEFAULT 0.0,
    wind_gust_ms REAL DEFAULT 0.0,
    wind_dir_deg REAL DEFAULT 0.0,
    precip_1hr_mm REAL DEFAULT 0.0,
    PRIMARY KEY (valid_time_utc, station_id)
);

CREATE INDEX IF NOT EXISTS idx_ref_obs_stid_time ON reference_observations(station_id, valid_time_utc);

CREATE TABLE IF NOT EXISTS nws_forecasts (
    issue_time_utc TEXT,
    valid_time_utc TEXT,
    lead_time_hours INTEGER,
    nws_temp_c REAL,
    nws_dewpoint_c REAL,
    nws_rh_pct REAL,
    nws_wind_speed_ms REAL DEFAULT 0.0,
    nws_wind_gust_ms REAL DEFAULT 0.0,
    nws_wind_dir_deg REAL DEFAULT 0.0,
    nws_pop_pct REAL DEFAULT 0.0,
    nws_qpf_1hr_mm REAL DEFAULT 0.0,
    nws_sky_cover_pct REAL DEFAULT 20.0,
    short_forecast TEXT,
    PRIMARY KEY (issue_time_utc, valid_time_utc)
);

CREATE INDEX IF NOT EXISTS idx_nws_fc_valid ON nws_forecasts(valid_time_utc);
CREATE INDEX IF NOT EXISTS idx_nws_fc_issue ON nws_forecasts(issue_time_utc);

CREATE TABLE IF NOT EXISTS mos_corrected_forecasts (
    issue_time_utc TEXT,
    valid_time_utc TEXT,
    model_version TEXT,
    raw_nws_temp_c REAL,
    pred_temp_error_c REAL,
    corrected_temp_c REAL,
    corrected_temp_p10_c REAL,
    corrected_temp_p90_c REAL,
    corrected_rh_pct REAL,
    corrected_wind_gust_ms REAL,
    raw_nws_pop_pct REAL,
    mos_pop_pct REAL,
    raw_nws_qpf_mm REAL,
    mos_qpf_mm REAL,
    mos_qpf_p90_mm REAL,
    PRIMARY KEY (issue_time_utc, valid_time_utc)
);

CREATE INDEX IF NOT EXISTS idx_mos_fc_valid ON mos_corrected_forecasts(valid_time_utc);
CREATE INDEX IF NOT EXISTS idx_mos_fc_issue ON mos_corrected_forecasts(issue_time_utc);

DROP VIEW IF EXISTS vw_mos_training_dataset;
CREATE VIEW vw_mos_training_dataset AS
SELECT
    f.issue_time_utc,
    f.valid_time_utc,
    f.lead_time_hours,
    f.nws_temp_c,
    f.nws_dewpoint_c,
    f.nws_rh_pct,
    f.nws_wind_speed_ms,
    COALESCE(f.nws_wind_gust_ms, f.nws_wind_speed_ms * 1.35) AS nws_wind_gust_ms,
    f.nws_wind_dir_deg,
    COALESCE(f.nws_sky_cover_pct, 20.0) AS nws_sky_cover_pct,
    COALESCE(f.nws_pop_pct, 0.0) AS nws_pop_pct,
    COALESCE(f.nws_qpf_1hr_mm, 0.0) AS nws_qpf_1hr_mm,
    o.temp_c AS actual_temp_c,
    o.rh_pct AS actual_rh_pct,
    o.wind_speed_ms AS actual_wind_speed_ms,
    o.wind_gust_ms AS actual_wind_gust_ms,
    o.wind_dir_deg AS actual_wind_dir_deg,
    o.mslp_hpa AS actual_mslp_hpa,
    COALESCE(o.pressure_trend_3hr_hpa, 0.0) AS pressure_trend_3hr_hpa,
    COALESCE(o.precip_1hr_mm, 0.0) AS actual_precip_mm,
    (o.temp_c - f.nws_temp_c) AS temp_error_c,
    (o.rh_pct - f.nws_rh_pct) AS rh_error_pct,
    -- CHWC1: Chews Ridge (1,536m Interior Crest)
    COALESCE(r_chews.temp_c, r_vent.temp_c - 0.54, f.nws_temp_c + 2.0) AS issue_chews_temp_c,
    COALESCE(r_chews.rh_pct, f.nws_rh_pct) AS issue_chews_rh_pct,
    COALESCE(r_chews.precip_1hr_mm, 0.0) AS issue_chews_precip_mm,
    -- CAHC1: Hastings Reservation (546m Foothill Valley)
    COALESCE(r_hast.temp_c, r_arroyo.temp_c - 1.80, f.nws_temp_c + 0.5) AS issue_hastings_temp_c,
    -- ASRC1: Arroyo Seco RAWS (268m Gorge Outlet)
    COALESCE(r_arroyo.temp_c, r_hast.temp_c + 1.80, f.nws_temp_c + 1.4) AS issue_arroyo_temp_c,
    COALESCE(r_arroyo.rh_pct, r_hast.rh_pct, f.nws_rh_pct) AS issue_arroyo_rh_pct,
    -- PPSC1: Big Sur RAWS (108m Coastal Slope)
    COALESCE(r_bigsur.temp_c, f.nws_temp_c + 0.8) AS issue_bigsur_temp_c,
    COALESCE(r_bigsur.rh_pct, f.nws_rh_pct) AS issue_bigsur_rh_pct,
    COALESCE(r_bigsur.precip_1hr_mm, 0.0) AS issue_bigsur_precip_mm,
    -- VTCC1: Ventana Cone (1,453m Windward Crest)
    COALESCE(r_vent.temp_c, r_chews.temp_c + 0.54, f.nws_temp_c + 1.5) AS issue_ventana_temp_c,
    COALESCE(r_vent.precip_1hr_mm, r_chews.precip_1hr_mm, 0.0) AS issue_ventana_precip_mm,
    -- Local Stone Office Initialization State
    COALESCE(o_issue.temp_c, f.nws_temp_c) AS issue_tass_temp_c,
    COALESCE(o_issue.temp_c - f_issue.nws_temp_c, 0.0) AS issue_persistence_bias_c
FROM nws_forecasts f
INNER JOIN local_observations o
    ON f.valid_time_utc = o.valid_time_utc
LEFT JOIN local_observations o_issue
    ON f.issue_time_utc = o_issue.valid_time_utc
LEFT JOIN (
    SELECT valid_time_utc, AVG(nws_temp_c) AS nws_temp_c
    FROM nws_forecasts
    GROUP BY valid_time_utc
) f_issue
    ON f.issue_time_utc = f_issue.valid_time_utc
LEFT JOIN reference_observations r_chews
    ON f.issue_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
LEFT JOIN reference_observations r_hast
    ON f.issue_time_utc = r_hast.valid_time_utc AND r_hast.station_id = 'CAHC1'
LEFT JOIN reference_observations r_arroyo
    ON f.issue_time_utc = r_arroyo.valid_time_utc AND r_arroyo.station_id = 'ASRC1'
LEFT JOIN reference_observations r_bigsur
    ON f.issue_time_utc = r_bigsur.valid_time_utc AND r_bigsur.station_id = 'PPSC1'
LEFT JOIN reference_observations r_vent
    ON f.issue_time_utc = r_vent.valid_time_utc AND r_vent.station_id = 'VTCC1'
WHERE o.qc_flag = 'OK'
  AND o.temp_c IS NOT NULL
  AND f.nws_temp_c IS NOT NULL
  AND f.lead_time_hours >= 1;