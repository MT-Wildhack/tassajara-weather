-- schema.sql — Tassajara Zen Mountain Center (TZMC) Microclimate & Hydrological MOS Schema
-- Supports Station 36966 (Stone Office Vantage Pro2), the 5-Station Santa Lucia Array
-- (CHWC1 Chews Ridge, VTCC1 Ventana Cone, CAHC1 Hastings, ASRC1 Arroyo Seco, PPSC1 Big Sur),
-- and the Early Flood & Tassajara Road Washout Warning System (API + Kinematic Runoff Routing).

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

-- ============================================================================
-- Early Flood & Road Washout Warning System Tables
-- ============================================================================

-- 1. Optional Future Stream-Gauge Telemetry (Ultrasonic / Radar / Pressure Transducer)
CREATE TABLE IF NOT EXISTS stream_gauge_observations (
    valid_time_utc TEXT,
    gauge_id TEXT DEFAULT 'TASS_CREEK_FOOTBRIDGE',
    stream_stage_m REAL,
    stream_stage_ft REAL,
    rate_of_rise_m_hr REAL DEFAULT 0.0,
    estimated_discharge_cfs REAL,
    water_temp_c REAL,
    battery_v REAL,
    sensor_health TEXT DEFAULT 'OK',
    qc_flag TEXT DEFAULT 'OK',
    PRIMARY KEY (valid_time_utc, gauge_id)
);

CREATE INDEX IF NOT EXISTS idx_stream_gauge_time ON stream_gauge_observations(valid_time_utc);

-- 2. Hourly Catchment Antecedent Wetness (API), Runoff & Alert State History
CREATE TABLE IF NOT EXISTS hydrological_risk_state (
    valid_time_utc TEXT PRIMARY KEY,
    api_mm REAL NOT NULL DEFAULT 0.0,
    api_in REAL NOT NULL DEFAULT 0.0,
    decay_k REAL NOT NULL DEFAULT 0.91,
    runoff_coeff_c REAL NOT NULL DEFAULT 0.08,
    effective_precip_1hr_mm REAL DEFAULT 0.0,
    precip_24hr_mm REAL DEFAULT 0.0,
    precip_72hr_mm REAL DEFAULT 0.0,
    upstream_spillover_mm REAL DEFAULT 0.0,
    current_runoff_mm_hr REAL DEFAULT 0.0,
    peak_runoff_6hr_mm_hr REAL DEFAULT 0.0,
    peak_runoff_24hr_mm_hr REAL DEFAULT 0.0,
    peak_qpf_1hr_24h_mm REAL DEFAULT 0.0,
    peak_p90_1hr_24h_mm REAL DEFAULT 0.0,
    forecast_qpf_24hr_mm REAL DEFAULT 0.0,
    forecast_qpf_72hr_mm REAL DEFAULT 0.0,
    discharge_index_cfs REAL DEFAULT 5.0,
    peak_discharge_24hr_cfs REAL DEFAULT 5.0,
    stream_stage_m REAL,
    rate_of_rise_m_hr REAL,
    alert_tier INTEGER NOT NULL DEFAULT 0,
    alert_code TEXT NOT NULL DEFAULT 'GREEN',
    road_status TEXT,
    creek_status TEXT,
    recommended_action TEXT,
    trigger_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_hydro_risk_time ON hydrological_risk_state(valid_time_utc);
CREATE INDEX IF NOT EXISTS idx_hydro_risk_code ON hydrological_risk_state(alert_code, valid_time_utc);

-- 3. Forward Hourly Routed Hydrograph & API Projection Series
CREATE TABLE IF NOT EXISTS hydrological_forecast_series (
    issue_time_utc TEXT,
    valid_time_utc TEXT,
    lead_time_hours INTEGER,
    mos_qpf_mm REAL DEFAULT 0.0,
    mos_qpf_p90_mm REAL DEFAULT 0.0,
    projected_api_mm REAL DEFAULT 0.0,
    projected_api_p90_mm REAL DEFAULT 0.0,
    projected_runoff_coeff REAL DEFAULT 0.08,
    projected_runoff_mm_hr REAL DEFAULT 0.0,
    projected_runoff_p90_mm_hr REAL DEFAULT 0.0,
    projected_discharge_cfs REAL DEFAULT 5.0,
    projected_discharge_p90_cfs REAL DEFAULT 5.0,
    projected_tier TEXT DEFAULT 'GREEN',
    PRIMARY KEY (issue_time_utc, valid_time_utc)
);

CREATE INDEX IF NOT EXISTS idx_hydro_fc_valid ON hydrological_forecast_series(valid_time_utc);
CREATE INDEX IF NOT EXISTS idx_hydro_fc_issue ON hydrological_forecast_series(issue_time_utc);

-- 4. Operational Creek & Road Monitoring View
DROP VIEW IF EXISTS vw_creek_flood_monitor;
CREATE VIEW vw_creek_flood_monitor AS
SELECT
    h.valid_time_utc,
    h.alert_tier,
    h.alert_code,
    h.api_mm,
    h.api_in,
    h.runoff_coeff_c,
    h.effective_precip_1hr_mm,
    h.precip_24hr_mm,
    h.precip_72hr_mm,
    h.upstream_spillover_mm,
    h.current_runoff_mm_hr,
    h.peak_runoff_6hr_mm_hr,
    h.peak_runoff_24hr_mm_hr,
    h.discharge_index_cfs,
    h.peak_discharge_24hr_cfs,
    COALESCE(g.stream_stage_m, h.stream_stage_m) AS stream_stage_m,
    COALESCE(g.rate_of_rise_m_hr, h.rate_of_rise_m_hr) AS rate_of_rise_m_hr,
    g.sensor_health AS gauge_sensor_health,
    h.road_status,
    h.creek_status,
    h.recommended_action,
    h.trigger_summary,
    o.temp_c,
    o.rh_pct,
    o.wind_gust_ms,
    o.mslp_hpa,
    o.pressure_trend_3hr_hpa
FROM hydrological_risk_state h
LEFT JOIN local_observations o
    ON h.valid_time_utc = o.valid_time_utc
LEFT JOIN stream_gauge_observations g
    ON h.valid_time_utc = g.valid_time_utc
   AND g.gauge_id = 'TASS_CREEK_FOOTBRIDGE'
ORDER BY h.valid_time_utc DESC;

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
    COALESCE(r_chews_v.temp_c, r_chews_i.temp_c, r_vent_v.temp_c - 0.54, f.nws_temp_c + 2.0) AS issue_chews_temp_c,
    COALESCE(r_chews_v.rh_pct, r_chews_i.rh_pct, f.nws_rh_pct) AS issue_chews_rh_pct,
    COALESCE(r_chews_v.precip_1hr_mm, r_chews_i.precip_1hr_mm, 0.0) AS issue_chews_precip_mm,
    -- CAHC1: Hastings Reservation (546m Foothill Valley)
    COALESCE(r_hast_v.temp_c, r_hast_i.temp_c, r_arroyo_v.temp_c - 1.80, f.nws_temp_c + 0.5) AS issue_hastings_temp_c,
    -- ASRC1: Arroyo Seco RAWS (268m Gorge Outlet)
    COALESCE(r_arroyo_v.temp_c, r_arroyo_i.temp_c, r_hast_v.temp_c + 1.80, f.nws_temp_c + 1.4) AS issue_arroyo_temp_c,
    COALESCE(r_arroyo_v.rh_pct, r_arroyo_i.rh_pct, r_hast_v.rh_pct, f.nws_rh_pct) AS issue_arroyo_rh_pct,
    -- PPSC1: Big Sur RAWS (108m Coastal Slope)
    COALESCE(r_bigsur_v.temp_c, r_bigsur_i.temp_c, f.nws_temp_c + 0.8) AS issue_bigsur_temp_c,
    COALESCE(r_bigsur_v.rh_pct, r_bigsur_i.rh_pct, f.nws_rh_pct) AS issue_bigsur_rh_pct,
    COALESCE(r_bigsur_v.precip_1hr_mm, r_bigsur_i.precip_1hr_mm, 0.0) AS issue_bigsur_precip_mm,
    -- VTCC1: Ventana Cone (1,453m Windward Crest)
    COALESCE(r_vent_v.temp_c, r_vent_i.temp_c, r_chews_v.temp_c + 0.54, f.nws_temp_c + 2.54) AS issue_ventana_temp_c,
    COALESCE(r_vent_v.precip_1hr_mm, r_vent_i.precip_1hr_mm, r_chews_v.precip_1hr_mm, 0.0) AS issue_ventana_precip_mm,
    -- Local Stone Office Initialization State
    COALESCE(o_issue.temp_c, f.nws_temp_c) AS issue_tass_temp_c,
    COALESCE(o_issue.temp_c - f_issue.nws_temp_c, 0.0) AS issue_persistence_bias_c
FROM (
    SELECT
        issue_time_utc,
        valid_time_utc,
        lead_time_hours,
        nws_temp_c,
        nws_dewpoint_c,
        nws_rh_pct,
        nws_wind_speed_ms,
        nws_wind_gust_ms,
        nws_wind_dir_deg,
        nws_sky_cover_pct,
        nws_pop_pct,
        nws_qpf_1hr_mm,
        short_forecast,
        ROW_NUMBER() OVER (
            PARTITION BY valid_time_utc
            ORDER BY
                CASE WHEN short_forecast != 'Archived NWP Grid' THEN 0 ELSE 1 END ASC,
                CASE WHEN lead_time_hours BETWEEN 1 AND 12 THEN 0 ELSE 1 END ASC,
                ABS(lead_time_hours - 6) ASC,
                issue_time_utc DESC
        ) AS rn
    FROM nws_forecasts
    WHERE nws_temp_c IS NOT NULL
      AND lead_time_hours >= 1
) f
INNER JOIN local_observations o
    ON f.valid_time_utc = o.valid_time_utc
LEFT JOIN local_observations o_issue
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = o_issue.valid_time_utc
LEFT JOIN (
    SELECT valid_time_utc, AVG(nws_temp_c) AS nws_temp_c
    FROM nws_forecasts
    GROUP BY valid_time_utc
) f_issue
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = f_issue.valid_time_utc
LEFT JOIN reference_observations r_chews_v
    ON f.valid_time_utc = r_chews_v.valid_time_utc AND r_chews_v.station_id = 'CHWC1'
LEFT JOIN reference_observations r_chews_i
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = r_chews_i.valid_time_utc AND r_chews_i.station_id = 'CHWC1'
LEFT JOIN reference_observations r_hast_v
    ON f.valid_time_utc = r_hast_v.valid_time_utc AND r_hast_v.station_id = 'CAHC1'
LEFT JOIN reference_observations r_hast_i
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = r_hast_i.valid_time_utc AND r_hast_i.station_id = 'CAHC1'
LEFT JOIN reference_observations r_arroyo_v
    ON f.valid_time_utc = r_arroyo_v.valid_time_utc AND r_arroyo_v.station_id = 'ASRC1'
LEFT JOIN reference_observations r_arroyo_i
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = r_arroyo_i.valid_time_utc AND r_arroyo_i.station_id = 'ASRC1'
LEFT JOIN reference_observations r_bigsur_v
    ON f.valid_time_utc = r_bigsur_v.valid_time_utc AND r_bigsur_v.station_id = 'PPSC1'
LEFT JOIN reference_observations r_bigsur_i
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = r_bigsur_i.valid_time_utc AND r_bigsur_i.station_id = 'PPSC1'
LEFT JOIN reference_observations r_vent_v
    ON f.valid_time_utc = r_vent_v.valid_time_utc AND r_vent_v.station_id = 'VTCC1'
LEFT JOIN reference_observations r_vent_i
    ON (substr(f.issue_time_utc, 1, 13) || ':00:00+00:00') = r_vent_i.valid_time_utc AND r_vent_i.station_id = 'VTCC1'
WHERE f.rn = 1
  AND o.qc_flag = 'OK'
  AND o.temp_c IS NOT NULL;
