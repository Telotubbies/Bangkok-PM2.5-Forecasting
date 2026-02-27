-- BKK PM2.5 Forecasting — Supabase Schema
-- Run this in the Supabase SQL Editor to create all tables.

-- ─── Stations ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stations (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  name_th     TEXT NOT NULL DEFAULT '',
  lat         DOUBLE PRECISION NOT NULL,
  lon         DOUBLE PRECISION NOT NULL,
  area        TEXT NOT NULL DEFAULT '',
  province    TEXT NOT NULL DEFAULT 'Bangkok',
  is_active   BOOLEAN NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_stations_active ON stations (is_active) WHERE is_active = TRUE;

-- ─── Readings (hourly observations) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS readings (
  id              BIGSERIAL PRIMARY KEY,
  station_id      TEXT NOT NULL REFERENCES stations(id),
  timestamp_utc   TIMESTAMPTZ NOT NULL,
  pm25            REAL,
  pm10            REAL,
  temperature     REAL,
  humidity        REAL,
  pressure        REAL,
  wind_speed      REAL,
  wind_dir        REAL,
  wind_u10        REAL,
  wind_v10        REAL,
  precipitation   REAL,
  visibility      REAL,
  cloud_cover     REAL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE (station_id, timestamp_utc)
);

CREATE INDEX idx_readings_station_time ON readings (station_id, timestamp_utc DESC);
CREATE INDEX idx_readings_time ON readings (timestamp_utc DESC);

-- ─── Forecasts (model predictions) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS forecasts (
  id              BIGSERIAL PRIMARY KEY,
  station_id      TEXT NOT NULL REFERENCES stations(id),
  model_name      TEXT NOT NULL DEFAULT 'stunn',
  issued_at       TIMESTAMPTZ NOT NULL,
  horizon_hours   INTEGER NOT NULL,
  target_time     TIMESTAMPTZ NOT NULL,
  pm25_predicted  REAL NOT NULL,
  confidence      REAL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_forecasts_station_target ON forecasts (station_id, target_time DESC);
CREATE INDEX idx_forecasts_model ON forecasts (model_name, issued_at DESC);

-- ─── Hotspots (NASA FIRMS fire detections) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS hotspots (
  id              BIGSERIAL PRIMARY KEY,
  lat             DOUBLE PRECISION NOT NULL,
  lon             DOUBLE PRECISION NOT NULL,
  frp             REAL NOT NULL,
  confidence      TEXT NOT NULL DEFAULT 'medium',
  satellite       TEXT NOT NULL DEFAULT 'VIIRS',
  detected_at     TIMESTAMPTZ NOT NULL,
  country_code    TEXT NOT NULL DEFAULT 'TH',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hotspots_time ON hotspots (detected_at DESC);
CREATE INDEX idx_hotspots_geo ON hotspots (lat, lon);

-- ─── RPC: latest reading per active station ──────────────────────────────────
CREATE OR REPLACE FUNCTION latest_readings_per_station()
RETURNS TABLE (
  id          TEXT,
  name        TEXT,
  name_th     TEXT,
  lat         DOUBLE PRECISION,
  lon         DOUBLE PRECISION,
  area        TEXT,
  province    TEXT,
  is_active   BOOLEAN,
  created_at  TIMESTAMPTZ,
  pm25        REAL,
  wind_speed  REAL,
  wind_dir    REAL,
  temperature REAL,
  humidity    REAL
) LANGUAGE sql STABLE AS $$
  SELECT DISTINCT ON (s.id)
    s.id, s.name, s.name_th, s.lat, s.lon, s.area, s.province,
    s.is_active, s.created_at,
    r.pm25, r.wind_speed, r.wind_dir, r.temperature, r.humidity
  FROM stations s
  LEFT JOIN readings r ON r.station_id = s.id
  WHERE s.is_active = TRUE
  ORDER BY s.id, r.timestamp_utc DESC;
$$;

-- ─── Row Level Security ──────────────────────────────────────────────────────
ALTER TABLE stations  ENABLE ROW LEVEL SECURITY;
ALTER TABLE readings  ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE hotspots  ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read stations"  ON stations  FOR SELECT USING (TRUE);
CREATE POLICY "Public read readings"  ON readings  FOR SELECT USING (TRUE);
CREATE POLICY "Public read forecasts" ON forecasts FOR SELECT USING (TRUE);
CREATE POLICY "Public read hotspots"  ON hotspots  FOR SELECT USING (TRUE);
