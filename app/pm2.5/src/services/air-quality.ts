import { supabase } from "../lib/supabase";
import type { Station, Reading, Forecast, Hotspot } from "../types/database";

/**
 * Air quality data service.
 * Reads from Supabase when configured, otherwise returns mock data.
 */

export interface StationReading extends Station {
  pm25: number;
  wind_speed: number;
  wind_dir: number;
  temperature: number | null;
  humidity: number | null;
}

export interface ForecastPoint {
  target_time: string;
  pm25_predicted: number;
  model_name: string;
  confidence: number | null;
}

const USE_LIVE = supabase !== null;

export async function fetchStations(): Promise<Station[]> {
  if (!USE_LIVE) return MOCK_STATIONS;

  const { data, error } = await supabase!
    .from("stations")
    .select("*")
    .eq("is_active", true)
    .order("name");

  if (error) {
    console.error("[air-quality] fetchStations error:", error);
    return MOCK_STATIONS;
  }
  return data;
}

export async function fetchLatestReadings(): Promise<StationReading[]> {
  if (!USE_LIVE) return MOCK_STATION_READINGS;

  const { data, error } = await supabase!
    .rpc("latest_readings_per_station");

  if (error) {
    console.error("[air-quality] fetchLatestReadings error:", error);
    return MOCK_STATION_READINGS;
  }
  return data as StationReading[];
}

export async function fetchForecasts(
  stationId: string,
  modelName = "stunn",
): Promise<ForecastPoint[]> {
  if (!USE_LIVE) return MOCK_FORECASTS;

  const { data, error } = await supabase!
    .from("forecasts")
    .select("target_time, pm25_predicted, model_name, confidence")
    .eq("station_id", stationId)
    .eq("model_name", modelName)
    .gte("target_time", new Date().toISOString())
    .order("target_time")
    .limit(72);

  if (error) {
    console.error("[air-quality] fetchForecasts error:", error);
    return MOCK_FORECASTS;
  }
  return data;
}

export async function fetchHotspots(): Promise<Hotspot[]> {
  if (!USE_LIVE) return MOCK_HOTSPOTS as Hotspot[];

  const oneDayAgo = new Date(Date.now() - 24 * 3600_000).toISOString();
  const { data, error } = await supabase!
    .from("hotspots")
    .select("*")
    .gte("detected_at", oneDayAgo)
    .order("frp", { ascending: false })
    .limit(200);

  if (error) {
    console.error("[air-quality] fetchHotspots error:", error);
    return MOCK_HOTSPOTS as Hotspot[];
  }
  return data;
}

// ── Mock data for development (no Supabase) ─────────────────────────────────

const MOCK_STATIONS: Station[] = [
  { id: "s01", name: "Din Daeng",   name_th: "ดินแดง",   lat: 13.7649, lon: 100.5440, area: "Din Daeng",   province: "Bangkok", is_active: true, created_at: "" },
  { id: "s02", name: "Ratchathewi", name_th: "ราชเทวี",  lat: 13.7583, lon: 100.5316, area: "Ratchathewi", province: "Bangkok", is_active: true, created_at: "" },
  { id: "s03", name: "Bang Na",     name_th: "บางนา",    lat: 13.6673, lon: 100.6048, area: "Bang Na",     province: "Bangkok", is_active: true, created_at: "" },
  { id: "s04", name: "Lat Phrao",   name_th: "ลาดพร้าว", lat: 13.8034, lon: 100.5702, area: "Lat Phrao",   province: "Bangkok", is_active: true, created_at: "" },
  { id: "s05", name: "Thon Buri",   name_th: "ธนบุรี",   lat: 13.7220, lon: 100.4872, area: "Thon Buri",   province: "Bangkok", is_active: true, created_at: "" },
  { id: "s06", name: "Chatuchak",   name_th: "จตุจักร",  lat: 13.8200, lon: 100.5536, area: "Chatuchak",   province: "Bangkok", is_active: true, created_at: "" },
  { id: "s07", name: "Bang Kapi",   name_th: "บางกะปิ",  lat: 13.7658, lon: 100.6456, area: "Bang Kapi",   province: "Bangkok", is_active: true, created_at: "" },
  { id: "s08", name: "Khlong Toei", name_th: "คลองเตย",  lat: 13.7130, lon: 100.5578, area: "Khlong Toei", province: "Bangkok", is_active: true, created_at: "" },
  { id: "s09", name: "Prawet",      name_th: "ประเวศ",   lat: 13.6870, lon: 100.6890, area: "Prawet",      province: "Bangkok", is_active: true, created_at: "" },
  { id: "s10", name: "Nong Khaem",  name_th: "หนองแขม",  lat: 13.7040, lon: 100.3530, area: "Nong Khaem",  province: "Bangkok", is_active: true, created_at: "" },
];

const MOCK_STATION_READINGS: StationReading[] = MOCK_STATIONS.map((s, i) => ({
  ...s,
  pm25: [72, 65, 58, 82, 48, 76, 55, 44, 39, 95][i],
  wind_speed: [8, 6, 12, 5, 14, 7, 10, 16, 18, 4][i],
  wind_dir: [45, 50, 30, 40, 55, 35, 60, 25, 20, 70][i],
  temperature: [32, 33, 31, 34, 30, 33, 32, 31, 30, 35][i],
  humidity: [55, 52, 60, 48, 65, 50, 58, 62, 68, 42][i],
}));

const MOCK_FORECASTS: ForecastPoint[] = Array.from({ length: 24 }, (_, i) => {
  const t = new Date(Date.now() + i * 3600_000);
  const base = 65 + Math.sin((i / 24) * Math.PI * 2) * 25;
  return {
    target_time: t.toISOString(),
    pm25_predicted: Math.round(base + (Math.random() - 0.5) * 10),
    model_name: "stunn",
    confidence: 0.85 + Math.random() * 0.1,
  };
});

const MOCK_HOTSPOTS = [
  { id: 1, lat: 14.02, lon: 100.45, frp: 85,  confidence: "high",   satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 2, lat: 13.95, lon: 100.70, frp: 120, confidence: "high",   satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 3, lat: 13.60, lon: 100.35, frp: 60,  confidence: "medium", satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 4, lat: 14.10, lon: 100.65, frp: 95,  confidence: "high",   satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 5, lat: 13.55, lon: 100.60, frp: 45,  confidence: "low",    satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 6, lat: 14.25, lon: 100.80, frp: 70,  confidence: "medium", satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 7, lat: 13.85, lon: 100.30, frp: 110, confidence: "high",   satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
  { id: 8, lat: 13.45, lon: 100.50, frp: 55,  confidence: "medium", satellite: "VIIRS", detected_at: new Date().toISOString(), country_code: "TH", created_at: "" },
];
