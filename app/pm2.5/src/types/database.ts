/**
 * Supabase database schema types for BKK PM2.5 Forecasting.
 *
 * Tables:
 *   stations       — Air quality monitoring stations metadata
 *   readings       — Hourly PM2.5 + weather observations
 *   forecasts      — Model-generated PM2.5 predictions
 *   hotspots       — NASA FIRMS fire detection data
 *
 * Run `supabase gen types typescript` after schema changes to regenerate.
 */
export interface Database {
  public: {
    Tables: {
      stations: {
        Row: {
          id: string;
          name: string;
          name_th: string;
          lat: number;
          lon: number;
          area: string;
          province: string;
          is_active: boolean;
          created_at: string;
        };
        Insert: Omit<Database["public"]["Tables"]["stations"]["Row"], "created_at">;
        Update: Partial<Database["public"]["Tables"]["stations"]["Insert"]>;
      };

      readings: {
        Row: {
          id: number;
          station_id: string;
          timestamp_utc: string;
          pm25: number | null;
          pm10: number | null;
          temperature: number | null;
          humidity: number | null;
          pressure: number | null;
          wind_speed: number | null;
          wind_dir: number | null;
          wind_u10: number | null;
          wind_v10: number | null;
          precipitation: number | null;
          visibility: number | null;
          cloud_cover: number | null;
          created_at: string;
        };
        Insert: Omit<Database["public"]["Tables"]["readings"]["Row"], "id" | "created_at">;
        Update: Partial<Database["public"]["Tables"]["readings"]["Insert"]>;
      };

      forecasts: {
        Row: {
          id: number;
          station_id: string;
          model_name: string;
          issued_at: string;
          horizon_hours: number;
          target_time: string;
          pm25_predicted: number;
          confidence: number | null;
          created_at: string;
        };
        Insert: Omit<Database["public"]["Tables"]["forecasts"]["Row"], "id" | "created_at">;
        Update: Partial<Database["public"]["Tables"]["forecasts"]["Insert"]>;
      };

      hotspots: {
        Row: {
          id: number;
          lat: number;
          lon: number;
          frp: number;
          confidence: string;
          satellite: string;
          detected_at: string;
          country_code: string;
          created_at: string;
        };
        Insert: Omit<Database["public"]["Tables"]["hotspots"]["Row"], "id" | "created_at">;
        Update: Partial<Database["public"]["Tables"]["hotspots"]["Insert"]>;
      };
    };

    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
  };
}

export type Station = Database["public"]["Tables"]["stations"]["Row"];
export type Reading = Database["public"]["Tables"]["readings"]["Row"];
export type Forecast = Database["public"]["Tables"]["forecasts"]["Row"];
export type Hotspot = Database["public"]["Tables"]["hotspots"]["Row"];
