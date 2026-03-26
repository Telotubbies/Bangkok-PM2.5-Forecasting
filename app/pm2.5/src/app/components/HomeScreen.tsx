import { useState, useRef, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Bell,
  ChevronDown,
  Wind,
  Eye,
  Droplets,
  Sparkles,
  ShieldCheck,
  Clock,
  MapPin,
  TrendingUp,
  TrendingDown,
  HeartPulse,
  Activity,
} from "lucide-react";

const CITIES: Record<
  string,
  {
    pm25: number;
    status: string;
    safeHours: string;
    maskLevel: string;
    forecast: { time: string; value: number }[];
    aiInsight: string;
    humidity: number;
    wind: number;
    visibility: number;
  }
> = {
  "Chiang Mai": {
    pm25: 87,
    status: "Unhealthy",
    safeHours: "< 1",
    maskLevel: "N95 Required",
    humidity: 45,
    wind: 8,
    visibility: 4.2,
    forecast: [
      { time: "Now", value: 87 },
      { time: "1H", value: 92 },
      { time: "2H", value: 98 },
      { time: "3H", value: 89 },
      { time: "4H", value: 81 },
      { time: "5H", value: 76 },
    ],
    aiInsight:
      "AI predicts peak pollution at 2PM due to low wind & dry conditions. Stay indoors. Air quality may improve after 6PM.",
  },
  Bangkok: {
    pm25: 42,
    status: "Moderate",
    safeHours: "2–3 hrs",
    maskLevel: "Recommended",
    humidity: 72,
    wind: 14,
    visibility: 8.5,
    forecast: [
      { time: "Now", value: 42 },
      { time: "1H", value: 45 },
      { time: "2H", value: 51 },
      { time: "3H", value: 48 },
      { time: "4H", value: 44 },
      { time: "5H", value: 39 },
    ],
    aiInsight:
      "Pollution expected to rise slightly during evening rush hour (5–7PM). Morning is the best time for outdoor exercise.",
  },
  Phuket: {
    pm25: 18,
    status: "Good",
    safeHours: "All day",
    maskLevel: "Not Required",
    humidity: 85,
    wind: 22,
    visibility: 15,
    forecast: [
      { time: "Now", value: 18 },
      { time: "1H", value: 16 },
      { time: "2H", value: 14 },
      { time: "3H", value: 15 },
      { time: "4H", value: 17 },
      { time: "5H", value: 19 },
    ],
    aiInsight:
      "Excellent air quality expected throughout the day. Sea breeze keeping pollution minimal. Great day for outdoor activities.",
  },
  "Chiang Rai": {
    pm25: 65,
    status: "Sensitive",
    safeHours: "1 hr",
    maskLevel: "Recommended",
    humidity: 38,
    wind: 6,
    visibility: 5.8,
    forecast: [
      { time: "Now", value: 65 },
      { time: "1H", value: 70 },
      { time: "2H", value: 74 },
      { time: "3H", value: 68 },
      { time: "4H", value: 60 },
      { time: "5H", value: 55 },
    ],
    aiInsight:
      "Wildfire smoke detected 40km north. Sensitive groups should limit outdoor exposure. Conditions may worsen by noon.",
  },
  Pattaya: {
    pm25: 28,
    status: "Good",
    safeHours: "4–5 hrs",
    maskLevel: "Optional",
    humidity: 78,
    wind: 18,
    visibility: 12,
    forecast: [
      { time: "Now", value: 28 },
      { time: "1H", value: 30 },
      { time: "2H", value: 32 },
      { time: "3H", value: 29 },
      { time: "4H", value: 26 },
      { time: "5H", value: 24 },
    ],
    aiInsight:
      "Coastal breeze maintaining good air quality. Light traffic pollution increase expected around 5–7PM.",
  },
};

function getAQIInfo(pm25: number) {
  if (pm25 <= 25)
    return {
      color: "#2ECC71",
      gradBg: "linear-gradient(160deg, #071a0e 0%, #0d2a17 60%, #0F172A 100%)",
      glowColor: "rgba(46,204,113,0.25)",
      label: "Good",
    };
  if (pm25 <= 50)
    return {
      color: "#F1C40F",
      gradBg: "linear-gradient(160deg, #1a1500 0%, #2a2200 60%, #0F172A 100%)",
      glowColor: "rgba(241,196,15,0.25)",
      label: "Moderate",
    };
  if (pm25 <= 100)
    return {
      color: "#E67E22",
      gradBg: "linear-gradient(160deg, #1a0c00 0%, #2a1400 60%, #0F172A 100%)",
      glowColor: "rgba(230,126,34,0.25)",
      label: "Unhealthy",
    };
  if (pm25 <= 150)
    return {
      color: "#E74C3C",
      gradBg: "linear-gradient(160deg, #1a0404 0%, #2a0808 60%, #0F172A 100%)",
      glowColor: "rgba(231,76,60,0.25)",
      label: "Very Unhealthy",
    };
  return {
    color: "#8B5CF6",
    gradBg: "linear-gradient(160deg, #0d0718 0%, #160d2a 60%, #0F172A 100%)",
    glowColor: "rgba(139,92,246,0.25)",
    label: "Hazardous",
  };
}

function getMaskIcon(level: string) {
  if (level === "Not Required") return "✅";
  if (level === "Optional") return "😷";
  if (level === "Recommended") return "😷";
  return "🛡️";
}

const NEARBY_STATIONS: Record<string, { name: string; pm25: number; distance: string }[]> = {
  "Chiang Mai": [
    { name: "ดินแดง", pm25: 92, distance: "1.2 กม." },
    { name: "ราชเทวี", pm25: 85, distance: "2.8 กม." },
    { name: "แม่ริม", pm25: 78, distance: "5.1 กม." },
  ],
  Bangkok: [
    { name: "ดินแดง", pm25: 45, distance: "0.8 กม." },
    { name: "บางนา", pm25: 38, distance: "3.2 กม." },
    { name: "คลองเตย", pm25: 41, distance: "2.1 กม." },
  ],
  Phuket: [
    { name: "ภูเก็ตทาวน์", pm25: 16, distance: "1.5 กม." },
    { name: "ป่าตอง", pm25: 19, distance: "4.3 กม." },
    { name: "ถลาง", pm25: 14, distance: "6.7 กม." },
  ],
  "Chiang Rai": [
    { name: "แม่สาย", pm25: 70, distance: "2.0 กม." },
    { name: "เมืองพาน", pm25: 62, distance: "4.5 กม." },
    { name: "เชียงของ", pm25: 58, distance: "7.2 กม." },
  ],
  Pattaya: [
    { name: "นาเกลือ", pm25: 26, distance: "1.8 กม." },
    { name: "จอมเทียน", pm25: 30, distance: "3.0 กม." },
    { name: "บางละมุง", pm25: 24, distance: "5.4 กม." },
  ],
};

function generate24hTrend(currentPm25: number): { hour: string; value: number }[] {
  const data: { hour: string; value: number }[] = [];
  for (let i = 23; i >= 0; i--) {
    const variation = Math.sin((i / 24) * Math.PI * 2) * 15 + (Math.random() - 0.5) * 10;
    const value = Math.max(5, Math.round(currentPm25 + variation));
    const hour = ((24 - i) % 24).toString().padStart(2, "0");
    data.push({ hour: `${hour}:00`, value });
  }
  return data;
}

function getHealthAdvice(pm25: number): {
  icon: typeof HeartPulse;
  title: string;
  advice: string;
  color: string;
  bgColor: string;
} {
  if (pm25 <= 25)
    return {
      icon: HeartPulse,
      title: "ปลอดภัย",
      advice: "ออกกำลังกายกลางแจ้งได้ตามปกติ",
      color: "#2ECC71",
      bgColor: "rgba(46,204,113,0.12)",
    };
  if (pm25 <= 50)
    return {
      icon: Activity,
      title: "ระวังสุขภาพ",
      advice: "กลุ่มเสี่ยงควรลดกิจกรรมกลางแจ้ง",
      color: "#F1C40F",
      bgColor: "rgba(241,196,15,0.12)",
    };
  if (pm25 <= 100)
    return {
      icon: ShieldCheck,
      title: "สวม N95",
      advice: "สวมหน้ากาก N95 เมื่อออกนอกอาคาร",
      color: "#E67E22",
      bgColor: "rgba(230,126,34,0.12)",
    };
  return {
    icon: ShieldCheck,
    title: "อยู่ในอาคาร",
    advice: "หลีกเลี่ยงกิจกรรมกลางแจ้งทั้งหมด",
    color: "#E74C3C",
    bgColor: "rgba(231,76,60,0.12)",
  };
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const v = payload[0].value;
    const info = getAQIInfo(v);
    return (
      <div
        style={{
          background: "rgba(30,41,59,0.95)",
          border: `1px solid ${info.color}44`,
          borderRadius: 10,
          padding: "6px 12px",
          backdropFilter: "blur(12px)",
        }}
      >
        <p style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>{label}</p>
        <p style={{ color: info.color, fontSize: 16, fontWeight: 700 }}>{v} µg/m³</p>
      </div>
    );
  }
  return null;
};

export function HomeScreen() {
  const [selectedCity, setSelectedCity] = useState("Chiang Mai");
  const [showCityPicker, setShowCityPicker] = useState(false);
  const widgetScrollRef = useRef<HTMLDivElement>(null);

  const city = CITIES[selectedCity];
  const aqi = getAQIInfo(city.pm25);
  const now = new Date();
  const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

  const trendData = useMemo(() => generate24hTrend(city.pm25), [city.pm25]);
  const trendMin = useMemo(() => Math.min(...trendData.map((d) => d.value)), [trendData]);
  const trendMax = useMemo(() => Math.max(...trendData.map((d) => d.value)), [trendData]);
  const nearby = NEARBY_STATIONS[selectedCity] ?? NEARBY_STATIONS["Bangkok"];
  const health = getHealthAdvice(city.pm25);

  return (
    <div
      className="relative w-full h-full overflow-y-auto overflow-x-hidden"
      style={{ background: aqi.gradBg, transition: "background 0.8s ease" }}
    >
      {/* Ambient glow blob */}
      <div
        className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none"
        style={{
          width: 320,
          height: 320,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${aqi.glowColor} 0%, transparent 70%)`,
          filter: "blur(40px)",
          top: 60,
        }}
      />

      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-12 pb-2 relative z-10">
        {/* City Selector */}
        <div className="relative">
          <button
            onClick={() => setShowCityPicker(!showCityPicker)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl"
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.12)",
              backdropFilter: "blur(12px)",
            }}
          >
            <span style={{ color: "white", fontWeight: 600, fontSize: 15 }}>{selectedCity}</span>
            <ChevronDown size={16} style={{ color: "rgba(255,255,255,0.6)" }} />
          </button>

          {/* City Dropdown */}
          {showCityPicker && (
            <div
              className="absolute top-full left-0 mt-2 w-44 overflow-hidden rounded-2xl z-50"
              style={{
                background: "rgba(15,23,42,0.98)",
                border: "1px solid rgba(255,255,255,0.12)",
                backdropFilter: "blur(20px)",
                boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
              }}
            >
              {Object.keys(CITIES).map((city) => {
                const info = getAQIInfo(CITIES[city].pm25);
                return (
                  <button
                    key={city}
                    onClick={() => {
                      setSelectedCity(city);
                      setShowCityPicker(false);
                    }}
                    className="w-full flex items-center justify-between px-4 py-3"
                    style={{
                      background: selectedCity === city ? "rgba(255,255,255,0.06)" : "transparent",
                      borderBottom: "1px solid rgba(255,255,255,0.05)",
                    }}
                  >
                    <span style={{ color: "white", fontSize: 14 }}>{city}</span>
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: info.color,
                      }}
                    >
                      {CITIES[city].pm25}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right icons */}
        <div className="flex items-center gap-2">
          <button
            className="w-9 h-9 flex items-center justify-center rounded-xl"
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            <Bell size={18} style={{ color: "rgba(255,255,255,0.7)" }} />
          </button>
        </div>
      </div>

      {/* Last Updated */}
      <div className="flex items-center gap-1.5 px-5 mt-1 relative z-10">
        <div
          className="w-1.5 h-1.5 rounded-full"
          style={{
            background: "#2ECC71",
            boxShadow: "0 0 6px #2ECC71",
            animation: "pulse 2s infinite",
          }}
        />
        <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>
          Updated {timeStr} · AI Forecasting Active
        </span>
      </div>

      {/* Main PM2.5 Orb */}
      <div className="flex flex-col items-center mt-6 relative z-10">
        <div
          className="relative flex items-center justify-center"
          style={{
            width: 200,
            height: 200,
            borderRadius: "50%",
            background: `radial-gradient(circle at 38% 35%, ${aqi.color}22 0%, ${aqi.color}08 50%, transparent 70%)`,
            border: `1.5px solid ${aqi.color}30`,
            boxShadow: `0 0 60px ${aqi.glowColor}, 0 0 120px ${aqi.glowColor}`,
          }}
        >
          {/* Inner ring */}
          <div
            className="absolute inset-3 rounded-full"
            style={{
              border: `1px solid ${aqi.color}18`,
            }}
          />
          <div className="text-center relative z-10">
            <div
              style={{
                fontSize: 68,
                fontWeight: 900,
                color: aqi.color,
                lineHeight: 1,
                letterSpacing: "-2px",
                textShadow: `0 0 30px ${aqi.color}88`,
              }}
            >
              {city.pm25}
            </div>
            <div style={{ color: "rgba(255,255,255,0.45)", fontSize: 13, marginTop: 2 }}>
              µg/m³ · PM2.5
            </div>
          </div>
        </div>

        {/* Status Badge */}
        <div
          className="mt-4 px-5 py-1.5 rounded-full"
          style={{
            background: `${aqi.color}18`,
            border: `1px solid ${aqi.color}40`,
          }}
        >
          <span style={{ color: aqi.color, fontWeight: 700, fontSize: 15 }}>
            {city.status}
          </span>
        </div>
      </div>

      {/* Stats Row */}
      <div className="flex gap-3 px-5 mt-5 relative z-10">
        {[
          { icon: Droplets, label: "Humidity", value: `${city.humidity}%`, color: "#38BDF8" },
          { icon: Wind, label: "Wind", value: `${city.wind} km/h`, color: "#818CF8" },
          { icon: Eye, label: "Visibility", value: `${city.visibility} km`, color: "#34D399" },
        ].map((stat) => (
          <div
            key={stat.label}
            className="flex-1 rounded-2xl px-3 py-2.5"
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              backdropFilter: "blur(12px)",
            }}
          >
            <stat.icon size={16} style={{ color: stat.color }} />
            <div
              style={{ color: "white", fontSize: 14, fontWeight: 700, marginTop: 4 }}
            >
              {stat.value}
            </div>
            <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* Health Advice Cards */}
      <div className="flex gap-3 px-5 mt-3 relative z-10">
        {/* Mask */}
        <div
          className="flex-1 flex items-center gap-2.5 rounded-2xl px-3 py-3"
          style={{
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(12px)",
          }}
        >
          <span style={{ fontSize: 24 }}>{getMaskIcon(city.maskLevel)}</span>
          <div>
            <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Mask</div>
            <div style={{ color: "white", fontSize: 12, fontWeight: 600 }}>
              {city.maskLevel}
            </div>
          </div>
        </div>

        {/* Safe time */}
        <div
          className="flex-1 flex items-center gap-2.5 rounded-2xl px-3 py-3"
          style={{
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(12px)",
          }}
        >
          <Clock size={22} style={{ color: "#F59E0B" }} />
          <div>
            <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Safe Outdoor</div>
            <div style={{ color: "white", fontSize: 12, fontWeight: 600 }}>
              {city.safeHours}
            </div>
          </div>
        </div>

        {/* Shield */}
        <div
          className="flex-1 flex items-center gap-2.5 rounded-2xl px-3 py-3"
          style={{
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(12px)",
          }}
        >
          <ShieldCheck size={22} style={{ color: "#A78BFA" }} />
          <div>
            <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Protection</div>
            <div style={{ color: "white", fontSize: 12, fontWeight: 600 }}>
              {city.pm25 > 75 ? "High" : city.pm25 > 40 ? "Medium" : "Low"}
            </div>
          </div>
        </div>
      </div>

      {/* ── Widget Strip (horizontally scrollable) ── */}
      <div
        ref={widgetScrollRef}
        className="flex gap-3 px-5 mt-4 relative z-10 overflow-x-auto scrollbar-hide"
        style={{ scrollSnapType: "x mandatory", WebkitOverflowScrolling: "touch" }}
      >
        {/* Widget 1: Nearby Stations */}
        <div
          className="shrink-0 w-[200px] rounded-2xl p-3"
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(12px)",
            scrollSnapAlign: "start",
          }}
        >
          <div className="flex items-center gap-2 mb-2.5">
            <div
              className="w-6 h-6 flex items-center justify-center rounded-lg"
              style={{ background: "rgba(56,189,248,0.15)" }}
            >
              <MapPin size={13} style={{ color: "#38BDF8" }} />
            </div>
            <span style={{ color: "rgba(255,255,255,0.7)", fontSize: 12, fontWeight: 600 }}>
              สถานีใกล้เคียง
            </span>
          </div>
          {nearby.map((st) => {
            const c = getAQIInfo(st.pm25).color;
            return (
              <div key={st.name} className="flex items-center justify-between py-1.5">
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ background: c, boxShadow: `0 0 4px ${c}` }}
                  />
                  <span style={{ color: "rgba(255,255,255,0.6)", fontSize: 11 }}>{st.name}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span style={{ color: c, fontSize: 12, fontWeight: 700 }}>{st.pm25}</span>
                  <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 9 }}>{st.distance}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Widget 2: 24h Trend Sparkline */}
        <div
          className="shrink-0 w-[200px] rounded-2xl p-3"
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(12px)",
            scrollSnapAlign: "start",
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <div
              className="w-6 h-6 flex items-center justify-center rounded-lg"
              style={{ background: "rgba(129,140,248,0.15)" }}
            >
              {trendData[trendData.length - 1]?.value >= trendData[0]?.value ? (
                <TrendingUp size={13} style={{ color: "#818CF8" }} />
              ) : (
                <TrendingDown size={13} style={{ color: "#818CF8" }} />
              )}
            </div>
            <span style={{ color: "rgba(255,255,255,0.7)", fontSize: 12, fontWeight: 600 }}>
              แนวโน้ม 24 ชม.
            </span>
          </div>
          <ResponsiveContainer width="100%" height={60}>
            <LineChart data={trendData} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
              <Line
                type="monotone"
                dataKey="value"
                stroke="#818CF8"
                strokeWidth={1.5}
                dot={false}
              />
              <ReferenceLine y={50} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center justify-between mt-1.5">
            <span style={{ color: "#2ECC71", fontSize: 10, fontWeight: 600 }}>
              ต่ำสุด {trendMin}
            </span>
            <span style={{ color: "#E74C3C", fontSize: 10, fontWeight: 600 }}>
              สูงสุด {trendMax}
            </span>
          </div>
        </div>

        {/* Widget 3: Health Alert */}
        <div
          className="shrink-0 w-[200px] rounded-2xl p-3"
          style={{
            background: health.bgColor,
            border: `1px solid ${health.color}30`,
            backdropFilter: "blur(12px)",
            scrollSnapAlign: "start",
          }}
        >
          <div className="flex items-center gap-2 mb-3">
            <div
              className="w-8 h-8 flex items-center justify-center rounded-xl"
              style={{ background: `${health.color}20` }}
            >
              <health.icon size={18} style={{ color: health.color }} />
            </div>
            <div>
              <div style={{ color: health.color, fontSize: 14, fontWeight: 700 }}>
                {health.title}
              </div>
              <div style={{ color: "rgba(255,255,255,0.3)", fontSize: 10 }}>คำแนะนำ</div>
            </div>
          </div>
          <p
            style={{
              color: "rgba(255,255,255,0.6)",
              fontSize: 12,
              lineHeight: 1.5,
            }}
          >
            {health.advice}
          </p>
        </div>
      </div>

      {/* 6H Forecast */}
      <div
        className="mx-5 mt-4 rounded-2xl p-4 relative z-10"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          backdropFilter: "blur(12px)",
        }}
      >
        <div className="flex items-center justify-between mb-3">
          <span style={{ color: "rgba(255,255,255,0.7)", fontSize: 13, fontWeight: 600 }}>
            6-Hour Forecast
          </span>
          <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 11 }}>µg/m³</span>
        </div>
        <ResponsiveContainer width="100%" height={90}>
          <LineChart data={city.forecast} margin={{ top: 8, right: 4, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor={aqi.color} stopOpacity={0.6} />
                <stop offset="50%" stopColor={aqi.color} stopOpacity={1} />
                <stop offset="100%" stopColor={aqi.color} stopOpacity={0.6} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis hide domain={["auto", "auto"]} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={50} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="value"
              stroke="url(#lineGrad)"
              strokeWidth={2.5}
              dot={{ fill: aqi.color, r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5, fill: aqi.color, strokeWidth: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* AI Insight Card */}
      <div
        className="mx-5 mt-3 mb-6 rounded-2xl p-4 relative z-10 overflow-hidden"
        style={{
          background: "rgba(56,189,248,0.06)",
          border: "1px solid rgba(56,189,248,0.2)",
          backdropFilter: "blur(12px)",
        }}
      >
        {/* Glow accent */}
        <div
          className="absolute -top-6 -right-6 w-20 h-20 rounded-full pointer-events-none"
          style={{
            background: "rgba(56,189,248,0.15)",
            filter: "blur(20px)",
          }}
        />
        <div className="flex items-start gap-3">
          <div
            className="w-8 h-8 flex items-center justify-center rounded-xl shrink-0"
            style={{ background: "rgba(56,189,248,0.15)" }}
          >
            <Sparkles size={16} style={{ color: "#38BDF8" }} />
          </div>
          <div>
            <div
              style={{ color: "#38BDF8", fontSize: 12, fontWeight: 700, marginBottom: 4 }}
            >
              AI Insight
            </div>
            <p style={{ color: "rgba(255,255,255,0.65)", fontSize: 13, lineHeight: 1.5 }}>
              {city.aiInsight}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
