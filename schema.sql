-- ============================================================================
-- schema.sql : Tassajara Forecast-Correction & Microclimate Database Schema
-- Compatible with PostgreSQL 15+ and SQLite 3.35+
-- ============================================================================

-- 1. Station Metadata Registry
CREATE TABLE IF NOT EXISTS stations_metadata (
    station_id          TEXT PRIMARY KEY,
    station_name        TEXT NOT NULL,
    latitude            REAL NOT NULL,
    longitude           REAL NOT NULL,
    elevation_m         REAL NOT NULL,
    sensor_height_agl_m REAL NOT NULL,
    wmo_siting_class    INTEGER DEFAULT 5,
    notes               TEXT
);

-- Seed Tassajara and Reference Stations
INSERT INTO stations_metadata (station_id, station_name, latitude, longitude, elevation_m, sensor_height_agl_m, wmo_siting_class, notes)
VALUES
    ('TASS_VP2', 'Tassajara Stone Office Vantage Pro2', 36.2337, -121.5501, 485.0, 4.5, 5, 'Roof-mounted on Stone Office gable; overhanging sycamore canopy.'),
    ('CHWC1', 'Chews Ridge CNRFC/RAWS', 36.3060, -121.5650, 1536.0, 6.1, 2, 'Upper ridge boundary reference (5,040 ft).'),
    ('CAHC1', 'Hastings Natural History Reservation RAWS', 36.3886, -121.5514, 546.0, 6.1, 2, 'Upper Carmel Valley baseline reference (1,792 ft).')
ON CONFLICT (station_id) DO NOTHING;

-- 2. Local Ground-Truth Observations (Stone Office Vantage Pro2)
CREATE TABLE IF NOT EXISTS local_observations (
    valid_time_utc          TIMESTAMP NOT NULL PRIMARY KEY,
    station_id              TEXT NOT NULL REFERENCES stations_metadata(station_id),
    temp_c                  REAL,
    dewpoint_c              REAL,
    wetbulb_c               REAL,
    rh_pct                  REAL,
    wind_speed_ms           REAL,
    wind_gust_ms            REAL,
    wind_dir_deg            REAL,
    mslp_hpa                REAL,
    pressure_trend_3hr_hpa  REAL,
    precip_1hr_mm           REAL DEFAULT 0.0,
    solar_rad_wm2           REAL,
    uv_index                REAL,
    qc_flag                 TEXT DEFAULT 'OK'
);

CREATE INDEX IF NOT EXISTS idx_local_obs_time ON local_observations(valid_time_utc);

-- 3. Regional Reference Observations (CHWC1 & CAHC1)
CREATE TABLE IF NOT EXISTS reference_observations (
    valid_time_utc          TIMESTAMP NOT NULL,
    station_id              TEXT NOT NULL REFERENCES stations_metadata(station_id),
    temp_c                  REAL,
    rh_pct                  REAL,
    wind_speed_ms           REAL,
    wind_gust_ms            REAL,
    wind_dir_deg            REAL,
    precip_1hr_mm           REAL DEFAULT 0.0,
    PRIMARY KEY (valid_time_utc, station_id)
);

-- 4. Raw Third-Party Forecasts (NWS API Gridpoint MTR)
CREATE TABLE IF NOT EXISTS nws_forecasts (
    issue_time_utc          TIMESTAMP NOT NULL,
    valid_time_utc          TIMESTAMP NOT NULL,
    lead_time_hours         INTEGER NOT NULL,
    nws_temp_c              REAL NOT NULL,
    nws_dewpoint_c          REAL,
    nws_rh_pct              REAL,
    nws_wind_speed_ms       REAL,
    nws_wind_gust_ms        REAL,
    nws_wind_dir_deg        REAL,
    nws_pop_pct             REAL,
    nws_qpf_1hr_mm          REAL DEFAULT 0.0,
    nws_sky_cover_pct       REAL,
    short_forecast          TEXT,
    PRIMARY KEY (issue_time_utc, valid_time_utc)
);

CREATE INDEX IF NOT EXISTS idx_nws_valid_time ON nws_forecasts(valid_time_utc);

-- 5. MOS Corrected Forecasts (Model Output)
CREATE TABLE IF NOT EXISTS mos_corrected_forecasts (
    issue_time_utc          TIMESTAMP NOT NULL,
    valid_time_utc          TIMESTAMP NOT NULL,
    model_version           TEXT NOT NULL,
    raw_nws_temp_c          REAL NOT NULL,
    pred_temp_error_c       REAL NOT NULL,
    corrected_temp_c        REAL NOT NULL,
    corrected_temp_p10_c    REAL,  -- 10th percentile lower bound (freeze risk)
    corrected_temp_p90_c    REAL,  -- 90th percentile upper bound (heat/fire risk)
    corrected_rh_pct        REAL,
    corrected_wind_gust_ms  REAL,
    PRIMARY KEY (issue_time_utc, valid_time_utc, model_version)
);

-- 6. Facility Alert & Hardware Maintenance Audit Log
CREATE TABLE IF NOT EXISTS facility_alert_log (
    alert_id                INTEGER PRIMARY KEY AUTOINCREMENT, -- Use SERIAL in PostgreSQL
    triggered_at_utc        TIMESTAMP NOT NULL,
    valid_time_utc          TIMESTAMP NOT NULL,
    alert_category          TEXT NOT NULL, -- 'FIRE_WEATHER', 'HARD_FREEZE', 'HIGH_WIND', 'HARDWARE_QC'
    severity                TEXT NOT NULL, -- 'ADVISORY', 'WARNING', 'CRITICAL'
    metric_value            REAL,
    threshold_value         REAL,
    message                 TEXT NOT NULL,
    dispatched_channels     TEXT
);

-- 7. Analytical View: Aligned Training Dataset (Joins Forecasts, Actuals & Reference Lapse Rates)
CREATE VIEW IF NOT EXISTS vw_mos_training_dataset AS
SELECT
    f.issue_time_utc,
    f.valid_time_utc,
    f.lead_time_hours,
    f.nws_temp_c,
    f.nws_dewpoint_c,
    f.nws_rh_pct,
    f.nws_wind_speed_ms,
    f.nws_wind_dir_deg,
    f.nws_sky_cover_pct,
    f.nws_pop_pct,
    o.temp_c AS actual_temp_c,
    o.rh_pct AS actual_rh_pct,
    o.wind_speed_ms AS actual_wind_speed_ms,
    o.wind_gust_ms AS actual_wind_gust_ms,
    o.wind_dir_deg AS actual_wind_dir_deg,
    o.mslp_hpa AS actual_mslp_hpa,
    o.pressure_trend_3hr_hpa,
    -- Target Forecast Errors (Residuals)
    (o.temp_c - f.nws_temp_c) AS temp_error_c,
    (o.rh_pct - f.nws_rh_pct) AS rh_error_pct,
    -- Reference Station Temps at Forecast Issue Time (Persistence/Stability Predictors)
    r_chews.temp_c AS issue_chews_temp_c,
    r_hast.temp_c AS issue_hastings_temp_c,
    o_issue.temp_c AS issue_tass_temp_c,
    (o_issue.temp_c - f.nws_temp_c) AS issue_persistence_bias_c
FROM nws_forecasts f
INNER JOIN local_observations o
    ON f.valid_time_utc = o.valid_time_utc
LEFT JOIN local_observations o_issue
    ON f.issue_time_utc = o_issue.valid_time_utc
LEFT JOIN reference_observations r_chews
    ON f.issue_time_utc = r_chews.valid_time_utc AND r_chews.station_id = 'CHWC1'
LEFT JOIN reference_observations r_hast
    ON f.issue_time_utc = r_hast.valid_time_utc AND r_hast.station_id = 'CAHC1'
WHERE o.qc_flag = 'OK';