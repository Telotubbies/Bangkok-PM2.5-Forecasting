import { useState, useMemo, useEffect, useCallback } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import { Flame, Wind, Layers, Sparkles } from "lucide-react";
import L from "leaflet";
import "leaflet.heat";

type Layer = "pm25" | "wind" | "hotspot";

interface Station {
  name: string;
  lat: number;
  lon: number;
  pm25: number;
  wind_speed: number;
  wind_dir: number;
}

interface HotspotPoint {
  lat: number;
  lon: number;
  frp: number;
  label: string;
}

interface TimeSlot {
  offset: number;
  hour: number;
  label: string;
  isFuture: boolean;
  isCurrent: boolean;
}

const BKK_CENTER: [number, number] = [13.7563, 100.5018];
const DEFAULT_ZOOM = 11;

const STATIONS: Station[] = [
  { name: "Din Daeng",     lat: 13.7649, lon: 100.5440, pm25: 72, wind_speed: 8,  wind_dir: 45  },
  { name: "Ratchathewi",   lat: 13.7583, lon: 100.5316, pm25: 65, wind_speed: 6,  wind_dir: 50  },
  { name: "Bang Na",       lat: 13.6673, lon: 100.6048, pm25: 58, wind_speed: 12, wind_dir: 30  },
  { name: "Lat Phrao",     lat: 13.8034, lon: 100.5702, pm25: 82, wind_speed: 5,  wind_dir: 40  },
  { name: "Thon Buri",     lat: 13.7220, lon: 100.4872, pm25: 48, wind_speed: 14, wind_dir: 55  },
  { name: "Chatuchak",     lat: 13.8200, lon: 100.5536, pm25: 76, wind_speed: 7,  wind_dir: 35  },
  { name: "Bang Kapi",     lat: 13.7658, lon: 100.6456, pm25: 55, wind_speed: 10, wind_dir: 60  },
  { name: "Khlong Toei",   lat: 13.7130, lon: 100.5578, pm25: 44, wind_speed: 16, wind_dir: 25  },
  { name: "Prawet",        lat: 13.6870, lon: 100.6890, pm25: 39, wind_speed: 18, wind_dir: 20  },
  { name: "Nong Khaem",    lat: 13.7040, lon: 100.3530, pm25: 95, wind_speed: 4,  wind_dir: 70  },
];

const HOTSPOTS: HotspotPoint[] = [
  { lat: 14.02, lon: 100.45, frp: 85,  label: "นครปฐม" },
  { lat: 13.95, lon: 100.70, frp: 120, label: "มีนบุรี" },
  { lat: 13.60, lon: 100.35, frp: 60,  label: "สมุทรสาคร" },
  { lat: 14.10, lon: 100.65, frp: 95,  label: "ปทุมธานี" },
  { lat: 13.55, lon: 100.60, frp: 45,  label: "สมุทรปราการ" },
  { lat: 14.25, lon: 100.80, frp: 70,  label: "นครนายก" },
  { lat: 13.85, lon: 100.30, frp: 110, label: "นนทบุรี" },
  { lat: 13.45, lon: 100.50, frp: 55,  label: "สมุทรสงคราม" },
];

function getPollutionColor(pm25: number): string {
  if (pm25 <= 25) return "#2ECC71";
  if (pm25 <= 50) return "#F1C40F";
  if (pm25 <= 75) return "#E67E22";
  if (pm25 <= 100) return "#E74C3C";
  return "#8B5CF6";
}

function getPollutionLabel(pm25: number): string {
  if (pm25 <= 25) return "ดี";
  if (pm25 <= 50) return "ปานกลาง";
  if (pm25 <= 75) return "มีผลต่อผู้อ่อนไหว";
  if (pm25 <= 100) return "ไม่ดีต่อสุขภาพ";
  return "อันตราย";
}

function getThaiNow(): Date {
  const utcMs = Date.now() + new Date().getTimezoneOffset() * 60_000;
  return new Date(utcMs + 7 * 3_600_000);
}

function fmtHHMM(hour: number): string {
  return `${String(hour).padStart(2, "0")}:00`;
}

const PAST_COUNT = 3;
const FUTURE_COUNT = 12;
const TOTAL_SLOTS = PAST_COUNT + 1 + FUTURE_COUNT;
const NOW_INDEX = PAST_COUNT;

// PM2.5 heatmap gradient: green → yellow → orange → red → purple
const PM25_GRADIENT: Record<number, string> = {
  0.0: "#1a9641",
  0.25: "#a6d96a",
  0.4: "#f4f466",
  0.55: "#fdae61",
  0.7: "#f46d43",
  0.85: "#d73027",
  1.0: "#7b2d8e",
};

// Wind speed heatmap gradient: calm blue → strong cyan-white
const WIND_GRADIENT: Record<number, string> = {
  0.0: "#0d1b2a",
  0.2: "#1b3a5c",
  0.4: "#2980B9",
  0.6: "#5DADE2",
  0.8: "#85C1E9",
  1.0: "#D6EAF8",
};

// Hotspot/fire heatmap gradient: dark → yellow → orange → red
const HOTSPOT_GRADIENT: Record<number, string> = {
  0.0: "#1a0500",
  0.2: "#7a2800",
  0.4: "#cc5500",
  0.6: "#ff8800",
  0.8: "#ff4400",
  1.0: "#ff0000",
};

function PM25HeatmapLayer({ stations }: { stations: Station[] }) {
  const map = useMap();

  useEffect(() => {
    const points: [number, number, number][] = stations.map((s) => [
      s.lat,
      s.lon,
      Math.min(s.pm25 / 120, 1),
    ]);

    const heat = (L as any).heatLayer(points, {
      radius: 55,
      blur: 40,
      maxZoom: 15,
      max: 1.0,
      minOpacity: 0.35,
      gradient: PM25_GRADIENT,
    });

    heat.addTo(map);
    return () => { map.removeLayer(heat); };
  }, [map, stations]);

  return null;
}

function WindHeatmapLayer({ stations }: { stations: Station[] }) {
  const map = useMap();

  useEffect(() => {
    const points: [number, number, number][] = stations.map((s) => [
      s.lat,
      s.lon,
      Math.min(s.wind_speed / 25, 1),
    ]);

    const heat = (L as any).heatLayer(points, {
      radius: 50,
      blur: 35,
      maxZoom: 15,
      max: 1.0,
      minOpacity: 0.3,
      gradient: WIND_GRADIENT,
    });

    heat.addTo(map);
    return () => { map.removeLayer(heat); };
  }, [map, stations]);

  return null;
}

function WindArrowOverlay({ stations }: { stations: Station[] }) {
  const map = useMap();

  useEffect(() => {
    const markers: L.Marker[] = [];

    stations.forEach((s) => {
      const rad = ((s.wind_dir - 90) * Math.PI) / 180;
      const len = 14 + (s.wind_speed / 20) * 10;
      const dx = Math.cos(rad) * len;
      const dy = Math.sin(rad) * len;
      const opacity = 0.6 + (s.wind_speed / 20) * 0.4;

      const svg = `
        <svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg">
          <line x1="${22 - dx}" y1="${22 + dy}" x2="${22 + dx}" y2="${22 - dy}"
            stroke="rgba(255,255,255,${opacity})" stroke-width="2.5" stroke-linecap="round"/>
          <polygon points="${22 + dx},${22 - dy} ${22 + dx - 5 * Math.sin(rad)},${22 - dy - 5 * Math.cos(rad)} ${22 + dx + 5 * Math.sin(rad)},${22 - dy + 5 * Math.cos(rad)}"
            fill="rgba(255,255,255,${opacity})"/>
          <text x="22" y="40" text-anchor="middle" fill="rgba(255,255,255,0.7)" font-size="8" font-weight="600">${s.wind_speed}</text>
        </svg>`;

      const icon = L.divIcon({
        html: svg,
        className: "",
        iconSize: [44, 44],
        iconAnchor: [22, 22],
      });

      const marker = L.marker([s.lat, s.lon], { icon, interactive: false });
      marker.addTo(map);
      markers.push(marker);
    });

    return () => {
      markers.forEach((m) => map.removeLayer(m));
    };
  }, [map, stations]);

  return null;
}

function HotspotHeatmapLayer({ hotspots }: { hotspots: HotspotPoint[] }) {
  const map = useMap();

  useEffect(() => {
    const points: [number, number, number][] = hotspots.map((h) => [
      h.lat,
      h.lon,
      Math.min(h.frp / 130, 1),
    ]);

    const heat = (L as any).heatLayer(points, {
      radius: 40,
      blur: 30,
      maxZoom: 15,
      max: 1.0,
      minOpacity: 0.4,
      gradient: HOTSPOT_GRADIENT,
    });

    heat.addTo(map);
    return () => { map.removeLayer(heat); };
  }, [map, hotspots]);

  return null;
}

function HotspotMarkerOverlay({ hotspots }: { hotspots: HotspotPoint[] }) {
  const map = useMap();

  useEffect(() => {
    const markers: L.Marker[] = [];

    hotspots.forEach((h) => {
      const size = 14 + (h.frp / 130) * 12;
      const color = h.frp >= 100 ? "#FF2200" : h.frp >= 60 ? "#FF8800" : "#FFCC00";

      const html = `
        <div style="
          width: ${size}px; height: ${size}px; border-radius: 50%;
          background: ${color}55; border: 2px solid ${color};
          animation: pulse 1.5s ease-in-out infinite;
          display: flex; align-items: center; justify-content: center;
        ">
          <div style="width: ${size * 0.35}px; height: ${size * 0.35}px; border-radius: 50%; background: ${color};"></div>
        </div>`;

      const icon = L.divIcon({
        html,
        className: "",
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      });

      const marker = L.marker([h.lat, h.lon], { icon });
      marker.bindPopup(
        `<div style="font-size:13px;font-weight:600;">🔥 ${h.label}</div>
         <div style="font-size:12px;color:${color};margin-top:2px;">FRP: ${h.frp} MW</div>`,
      );
      marker.addTo(map);
      markers.push(marker);
    });

    return () => {
      markers.forEach((m) => map.removeLayer(m));
    };
  }, [map, hotspots]);

  return null;
}

export function MapScreen() {
  const [activeLayer, setActiveLayer] = useState<Layer>("pm25");
  const [timeIndex, setTimeIndex] = useState(NOW_INDEX);
  const [thaiNow, setThaiNow] = useState(getThaiNow());

  useEffect(() => {
    const id = setInterval(() => setThaiNow(getThaiNow()), 60_000);
    return () => clearInterval(id);
  }, []);

  const slots: TimeSlot[] = useMemo(() => {
    const nowHour = thaiNow.getHours();
    return Array.from({ length: TOTAL_SLOTS }, (_, i) => {
      const offset = i - PAST_COUNT;
      const hour = ((nowHour + offset) % 24 + 24) % 24;
      return {
        offset,
        hour,
        label: offset === 0 ? "ตอนนี้" : fmtHHMM(hour),
        isFuture: offset > 0,
        isCurrent: offset === 0,
      };
    });
  }, [thaiNow]);

  const selectedSlot = slots[timeIndex];

  const stationsForTime = useMemo(() => {
    const hourOffset = selectedSlot.offset;
    const seed = hourOffset * 137;
    return STATIONS.map((s, idx) => {
      const pseudoRandom = Math.sin(seed + idx * 7.3) * 0.5 + 0.5;
      const drift = hourOffset * (pseudoRandom * 4 - 2);
      return {
        ...s,
        pm25: Math.max(5, Math.round(s.pm25 + drift)),
        wind_speed: Math.max(1, Math.round(s.wind_speed + drift * 0.3)),
      };
    });
  }, [selectedSlot.offset]);

  const handleTimeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setTimeIndex(Number(e.target.value));
  }, []);

  const nowPct = (NOW_INDEX / (TOTAL_SLOTS - 1)) * 100;
  const curPct = (timeIndex / (TOTAL_SLOTS - 1)) * 100;
  const trackBg =
    timeIndex <= NOW_INDEX
      ? `linear-gradient(to right, #38BDF8 0%, #38BDF8 ${curPct}%, rgba(255,255,255,0.13) ${curPct}%, rgba(255,255,255,0.13) 100%)`
      : `linear-gradient(to right, #38BDF8 0%, #38BDF8 ${nowPct}%, #818CF8 ${nowPct}%, #818CF8 ${curPct}%, rgba(255,255,255,0.13) ${curPct}%, rgba(255,255,255,0.13) 100%)`;

  const tickSlots = slots.filter((_, i) => i % 3 === 0 || i === NOW_INDEX);

  const thaiDateStr = thaiNow.toLocaleDateString("th-TH", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  const thaiTimeStr = thaiNow.toLocaleTimeString("th-TH", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  return (
    <div className="w-full h-full flex flex-col overflow-hidden" style={{ background: "#0A1628" }}>
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 pt-12 pb-3 shrink-0"
        style={{ background: "rgba(10,22,40,0.92)", backdropFilter: "blur(14px)" }}
      >
        <div>
          <h1 style={{ color: "white", fontSize: 18, fontWeight: 700 }}>แผนที่คุณภาพอากาศ</h1>
          <p style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>
            กรุงเทพฯ · {thaiDateStr} · {thaiTimeStr} น.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {selectedSlot.isFuture && (
            <div
              className="px-2 py-1 rounded-xl flex items-center gap-1"
              style={{
                background: "rgba(129,140,248,0.15)",
                border: "1px solid rgba(129,140,248,0.35)",
              }}
            >
              <Sparkles size={11} style={{ color: "#818CF8" }} />
              <span style={{ color: "#818CF8", fontSize: 10, fontWeight: 700 }}>AI</span>
            </div>
          )}
          {!selectedSlot.isFuture && (
            <div
              className="px-2.5 py-1 rounded-xl flex items-center gap-1.5"
              style={{
                background: "rgba(231,76,60,0.15)",
                border: "1px solid rgba(231,76,60,0.3)",
              }}
            >
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: "#E74C3C", boxShadow: "0 0 6px #E74C3C" }}
              />
              <span style={{ color: "#E74C3C", fontSize: 12, fontWeight: 600 }}>LIVE</span>
            </div>
          )}
        </div>
      </div>

      {/* Layer Toggles */}
      <div className="flex gap-2 px-5 py-2 shrink-0">
        {([
          { key: "pm25" as const, icon: Layers, label: "PM2.5" },
          { key: "wind" as const, icon: Wind, label: "ลม" },
          { key: "hotspot" as const, icon: Flame, label: "จุดความร้อน" },
        ]).map((l) => {
          const active = activeLayer === l.key;
          const colors: Record<Layer, string> = {
            pm25: "#E74C3C",
            wind: "#38BDF8",
            hotspot: "#F1C40F",
          };
          const c = colors[l.key];
          return (
            <button
              key={l.key}
              onClick={() => setActiveLayer(l.key)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl"
              style={{
                background: active ? `${c}1E` : "rgba(255,255,255,0.05)",
                border: `1px solid ${active ? c + "55" : "rgba(255,255,255,0.08)"}`,
                transition: "all 0.2s",
              }}
            >
              <l.icon size={13} style={{ color: active ? c : "rgba(255,255,255,0.45)" }} />
              <span
                style={{
                  fontSize: 11,
                  fontWeight: active ? 700 : 400,
                  color: active ? c : "rgba(255,255,255,0.45)",
                }}
              >
                {l.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Leaflet Map */}
      <div className="flex-1 relative">
        <MapContainer
          center={BKK_CENTER}
          zoom={DEFAULT_ZOOM}
          className="w-full h-full"
          zoomControl={false}
          attributionControl={true}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          />

          {/* PM2.5 layer: heatmap + station markers */}
          {activeLayer === "pm25" && (
            <>
              <PM25HeatmapLayer stations={stationsForTime} />
              {stationsForTime.map((s) => {
                const color = getPollutionColor(s.pm25);
                return (
                  <CircleMarker
                    key={s.name}
                    center={[s.lat, s.lon]}
                    radius={8}
                    pathOptions={{
                      color: "#fff",
                      fillColor: color,
                      fillOpacity: 0.9,
                      weight: 2,
                      opacity: 0.8,
                    }}
                  >
                    <Popup>
                      <div style={{ minWidth: 160 }}>
                        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>{s.name}</div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                          <span style={{ fontSize: 28, fontWeight: 900, color }}>{s.pm25}</span>
                          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>µg/m³</span>
                        </div>
                        <div style={{ fontSize: 12, color, fontWeight: 600, marginTop: 2 }}>
                          {getPollutionLabel(s.pm25)}
                        </div>
                        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginTop: 6 }}>
                          💨 {s.wind_speed} กม./ชม. · ทิศ {s.wind_dir}°
                        </div>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </>
          )}

          {/* Wind layer: heatmap + direction arrows */}
          {activeLayer === "wind" && (
            <>
              <WindHeatmapLayer stations={stationsForTime} />
              <WindArrowOverlay stations={stationsForTime} />
            </>
          )}

          {/* Hotspot layer: heatmap + pulsing markers */}
          {activeLayer === "hotspot" && (
            <>
              <HotspotHeatmapLayer hotspots={HOTSPOTS} />
              <HotspotMarkerOverlay hotspots={HOTSPOTS} />
              {stationsForTime.map((s) => {
                const color = getPollutionColor(s.pm25);
                return (
                  <CircleMarker
                    key={`hs-${s.name}`}
                    center={[s.lat, s.lon]}
                    radius={5}
                    pathOptions={{
                      color: color,
                      fillColor: color,
                      fillOpacity: 0.7,
                      weight: 1.5,
                      opacity: 0.6,
                    }}
                  >
                    <Popup>
                      <div style={{ minWidth: 140 }}>
                        <div style={{ fontSize: 13, fontWeight: 700 }}>{s.name}</div>
                        <div style={{ fontSize: 12, color, marginTop: 2 }}>
                          PM2.5: {s.pm25} µg/m³
                        </div>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </>
          )}
        </MapContainer>

        {/* Legend overlays */}
        {activeLayer === "pm25" && (
          <div
            className="absolute top-3 right-3 px-3 py-2.5 rounded-xl z-[1000]"
            style={{
              background: "rgba(10,22,40,0.92)",
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(12px)",
            }}
          >
            <p style={{ color: "rgba(255,255,255,0.7)", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
              PM2.5 (µg/m³)
            </p>
            <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>0</span>
              <div
                style={{
                  width: 100,
                  height: 10,
                  borderRadius: 5,
                  background: "linear-gradient(to right, #1a9641, #a6d96a, #f4f466, #fdae61, #f46d43, #d73027, #7b2d8e)",
                }}
              />
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>120+</span>
            </div>
            <div className="flex justify-between mt-1.5" style={{ width: 120 }}>
              {[
                { color: "#2ECC71", label: "ดี" },
                { color: "#F1C40F", label: "ปานกลาง" },
                { color: "#E67E22", label: "อ่อนไหว" },
                { color: "#E74C3C", label: "ไม่ดี" },
                { color: "#8B5CF6", label: "อันตราย" },
              ].map((item) => (
                <div
                  key={item.label}
                  className="w-2 h-2 rounded-full"
                  title={item.label}
                  style={{ background: item.color, boxShadow: `0 0 3px ${item.color}` }}
                />
              ))}
            </div>
          </div>
        )}

        {activeLayer === "wind" && (
          <div
            className="absolute top-3 right-3 px-3 py-2.5 rounded-xl z-[1000]"
            style={{
              background: "rgba(10,22,40,0.92)",
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(12px)",
            }}
          >
            <p style={{ color: "#93C5FD", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
              ความเร็วลม (กม./ชม.)
            </p>
            <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>0</span>
              <div
                style={{
                  width: 100,
                  height: 10,
                  borderRadius: 5,
                  background: "linear-gradient(to right, #0d1b2a, #1b3a5c, #2980B9, #5DADE2, #85C1E9, #D6EAF8)",
                }}
              />
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>25+</span>
            </div>
            <p style={{ color: "rgba(255,255,255,0.35)", fontSize: 9, marginTop: 4 }}>
              ลูกศร = ทิศทางลม · ตัวเลข = กม./ชม.
            </p>
          </div>
        )}

        {activeLayer === "hotspot" && (
          <div
            className="absolute top-3 right-3 px-3 py-2.5 rounded-xl z-[1000]"
            style={{
              background: "rgba(10,22,40,0.92)",
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(12px)",
            }}
          >
            <p style={{ color: "#F1C40F", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
              จุดความร้อน — FRP (MW)
            </p>
            <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>0</span>
              <div
                style={{
                  width: 100,
                  height: 10,
                  borderRadius: 5,
                  background: "linear-gradient(to right, #1a0500, #7a2800, #cc5500, #ff8800, #ff4400, #ff0000)",
                }}
              />
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>130+</span>
            </div>
            <div className="flex items-center gap-3 mt-2">
              {[
                { color: "#FFCC00", label: "< 60" },
                { color: "#FF8800", label: "60–100" },
                { color: "#FF2200", label: "≥ 100" },
              ].map((h) => (
                <div key={h.label} className="flex items-center gap-1">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ background: h.color, boxShadow: `0 0 3px ${h.color}` }}
                  />
                  <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 9 }}>{h.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div
        className="px-4 pt-3 pb-3 shrink-0"
        style={{
          background: "rgba(10,22,40,0.97)",
          borderTop: "1px solid rgba(255,255,255,0.07)",
          backdropFilter: "blur(12px)",
        }}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span style={{ color: "rgba(255,255,255,0.45)", fontSize: 11 }}>เวลา (ICT)</span>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1">
                <div className="w-3 h-1 rounded-full" style={{ background: "#38BDF8" }} />
                <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 9 }}>ข้อมูลจริง</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-1 rounded-full" style={{ background: "#818CF8" }} />
                <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 9 }}>พยากรณ์</span>
              </div>
            </div>
          </div>

          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl"
            style={{
              background: selectedSlot.isFuture
                ? "rgba(129,140,248,0.15)"
                : "rgba(56,189,248,0.15)",
              border: `1px solid ${selectedSlot.isFuture ? "rgba(129,140,248,0.4)" : "rgba(56,189,248,0.4)"}`,
            }}
          >
            {selectedSlot.isFuture && <Sparkles size={11} style={{ color: "#818CF8" }} />}
            <span
              style={{
                color: selectedSlot.isFuture ? "#818CF8" : "#38BDF8",
                fontSize: 12,
                fontWeight: 700,
              }}
            >
              {selectedSlot.label}
            </span>
            {selectedSlot.isFuture && (
              <span style={{ color: "rgba(129,140,248,0.7)", fontSize: 10 }}>
                +{selectedSlot.offset}ชม.
              </span>
            )}
          </div>
        </div>

        <div className="relative">
          <div
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-full pointer-events-none z-10"
            style={{
              left: `calc(${nowPct}% - 1px)`,
              background: "rgba(255,255,255,0.5)",
              boxShadow: "0 0 4px rgba(255,255,255,0.4)",
            }}
          />
          <input
            type="range"
            min={0}
            max={TOTAL_SLOTS - 1}
            step={1}
            value={timeIndex}
            onChange={handleTimeChange}
            className="w-full h-1.5 rounded-full appearance-none"
            style={{ background: trackBg, cursor: "pointer" }}
          />
        </div>

        <div className="relative mt-2" style={{ height: 20 }}>
          {tickSlots.map((slot) => {
            const pct = ((slot.offset + PAST_COUNT) / (TOTAL_SLOTS - 1)) * 100;
            return (
              <span
                key={slot.offset}
                className="absolute -translate-x-1/2 whitespace-nowrap"
                style={{
                  left: `${pct}%`,
                  top: 0,
                  fontSize: slot.isCurrent ? 10 : 9,
                  fontWeight: slot.isCurrent ? 800 : 400,
                  color: slot.isCurrent
                    ? "#38BDF8"
                    : slot.isFuture
                      ? "rgba(129,140,248,0.6)"
                      : "rgba(255,255,255,0.3)",
                }}
              >
                {slot.label}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
