import { useState } from "react";
import { Sparkles, TrendingDown, TrendingUp, Minus } from "lucide-react";

interface DayForecast {
  day: string;
  date: string;
  avg: number;
  peak: number;
  icon: string;
  trend: "up" | "down" | "same";
  advice: string;
  detail: string;
}

const FORECAST_DATA: DayForecast[] = [
  {
    day: "Today",
    date: "Fri, Feb 27",
    avg: 87,
    peak: 105,
    icon: "🌫️",
    trend: "up",
    advice: "Avoid outdoors",
    detail: "High pollution from wildfire smoke. Sensitive groups must stay indoors. N95 mask required if going out.",
  },
  {
    day: "Tomorrow",
    date: "Sat, Feb 28",
    avg: 93,
    peak: 118,
    icon: "🔥",
    trend: "up",
    advice: "Stay indoors",
    detail: "Peak pollution day. Wind speed drops to near zero. Fire activity expected in northern districts.",
  },
  {
    day: "Sunday",
    date: "Sun, Mar 1",
    avg: 78,
    peak: 95,
    icon: "🌁",
    trend: "down",
    advice: "Limit exposure",
    detail: "Slight improvement expected after rainfall in adjacent provinces. Morning still hazardous.",
  },
  {
    day: "Monday",
    date: "Mon, Mar 2",
    avg: 55,
    peak: 72,
    icon: "🌤️",
    trend: "down",
    advice: "Short walks ok",
    detail: "Wind pattern shifting. Air quality improving. Sensitive groups should still use masks outdoors.",
  },
  {
    day: "Tuesday",
    date: "Tue, Mar 3",
    avg: 38,
    peak: 52,
    icon: "⛅",
    trend: "down",
    advice: "Moderate caution",
    detail: "Continued improvement. Light outdoor activity acceptable for healthy individuals. Check real-time levels.",
  },
  {
    day: "Wednesday",
    date: "Wed, Mar 4",
    avg: 22,
    peak: 35,
    icon: "🌬️",
    trend: "down",
    advice: "Good to go",
    detail: "Rain expected. Air quality returns to acceptable levels. Great day for outdoor exercise.",
  },
  {
    day: "Thursday",
    date: "Thu, Mar 5",
    avg: 18,
    peak: 28,
    icon: "☀️",
    trend: "same",
    advice: "Enjoy outdoors",
    detail: "Excellent air quality. Sea breeze and cleansed atmosphere. No restrictions needed.",
  },
];

function getRiskInfo(pm25: number) {
  if (pm25 <= 25) return { label: "Good", color: "#2ECC71", bg: "rgba(46,204,113,0.12)", border: "rgba(46,204,113,0.3)" };
  if (pm25 <= 50) return { label: "Moderate", color: "#F1C40F", bg: "rgba(241,196,15,0.12)", border: "rgba(241,196,15,0.3)" };
  if (pm25 <= 75) return { label: "Sensitive", color: "#E67E22", bg: "rgba(230,126,34,0.12)", border: "rgba(230,126,34,0.3)" };
  if (pm25 <= 100) return { label: "Unhealthy", color: "#E74C3C", bg: "rgba(231,76,60,0.12)", border: "rgba(231,76,60,0.3)" };
  return { label: "Hazardous", color: "#8B5CF6", bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.3)" };
}

function MiniBar({ value, max = 130 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const info = getRiskInfo(value);
  return (
    <div
      className="h-1 rounded-full overflow-hidden"
      style={{ background: "rgba(255,255,255,0.08)", width: 60 }}
    >
      <div
        className="h-full rounded-full"
        style={{
          width: `${pct}%`,
          background: info.color,
          boxShadow: `0 0 6px ${info.color}80`,
        }}
      />
    </div>
  );
}

export function ForecastScreen() {
  const [expanded, setExpanded] = useState<number | null>(0);

  const worstDay = FORECAST_DATA.reduce((a, b) => (a.avg > b.avg ? a : b));
  const bestDay = FORECAST_DATA.reduce((a, b) => (a.avg < b.avg ? a : b));

  return (
    <div
      className="w-full h-full overflow-y-auto"
      style={{ background: "#0F172A" }}
    >
      {/* Header */}
      <div className="px-5 pt-12 pb-4">
        <h1 style={{ color: "white", fontSize: 22, fontWeight: 800 }}>7-Day Forecast</h1>
        <p style={{ color: "rgba(255,255,255,0.4)", fontSize: 13, marginTop: 2 }}>
          Chiang Mai · AI-Powered Prediction
        </p>
      </div>

      {/* Summary strip */}
      <div className="flex gap-3 px-5 mb-4">
        <div
          className="flex-1 rounded-2xl p-3"
          style={{
            background: "rgba(231,76,60,0.08)",
            border: "1px solid rgba(231,76,60,0.2)",
          }}
        >
          <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>Worst Day</div>
          <div style={{ color: "#E74C3C", fontSize: 15, fontWeight: 700 }}>
            {worstDay.day}
          </div>
          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>
            Avg {worstDay.avg} µg/m³
          </div>
        </div>
        <div
          className="flex-1 rounded-2xl p-3"
          style={{
            background: "rgba(46,204,113,0.08)",
            border: "1px solid rgba(46,204,113,0.2)",
          }}
        >
          <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>Best Day</div>
          <div style={{ color: "#2ECC71", fontSize: 15, fontWeight: 700 }}>
            {bestDay.day}
          </div>
          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>
            Avg {bestDay.avg} µg/m³
          </div>
        </div>
      </div>

      {/* AI Summary */}
      <div
        className="mx-5 mb-4 rounded-2xl p-3 flex items-start gap-3"
        style={{
          background: "rgba(56,189,248,0.06)",
          border: "1px solid rgba(56,189,248,0.18)",
        }}
      >
        <div
          className="w-7 h-7 flex items-center justify-center rounded-xl shrink-0 mt-0.5"
          style={{ background: "rgba(56,189,248,0.15)" }}
        >
          <Sparkles size={14} style={{ color: "#38BDF8" }} />
        </div>
        <p style={{ color: "rgba(255,255,255,0.6)", fontSize: 12, lineHeight: 1.5 }}>
          <span style={{ color: "#38BDF8", fontWeight: 700 }}>AI Summary: </span>
          Pollution peaks Sat–Sun due to dry conditions and low wind. Significant improvement expected mid-week after
          rainfall. Best window for outdoor activities: Wednesday onwards.
        </p>
      </div>

      {/* Day Cards */}
      <div className="px-5 pb-6 flex flex-col gap-2">
        {FORECAST_DATA.map((day, idx) => {
          const risk = getRiskInfo(day.avg);
          const peakRisk = getRiskInfo(day.peak);
          const isExpanded = expanded === idx;

          return (
            <div
              key={idx}
              className="rounded-2xl overflow-hidden"
              style={{
                background: isExpanded
                  ? "rgba(255,255,255,0.06)"
                  : "rgba(255,255,255,0.04)",
                border: isExpanded
                  ? `1px solid ${risk.color}30`
                  : "1px solid rgba(255,255,255,0.07)",
                transition: "all 0.25s ease",
              }}
            >
              {/* Main Row */}
              <button
                className="w-full flex items-center gap-3 px-4 py-3"
                onClick={() => setExpanded(isExpanded ? null : idx)}
              >
                {/* Day icon */}
                <div
                  className="w-10 h-10 flex items-center justify-center rounded-xl shrink-0"
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    fontSize: 20,
                  }}
                >
                  {day.icon}
                </div>

                {/* Day info */}
                <div className="flex-1 text-left">
                  <div style={{ color: "white", fontSize: 14, fontWeight: 600 }}>
                    {day.day}
                  </div>
                  <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>
                    {day.date}
                  </div>
                </div>

                {/* PM2.5 avg + bar */}
                <div className="flex flex-col items-end gap-1">
                  <div className="flex items-center gap-1.5">
                    {/* Trend icon */}
                    {day.trend === "up" && (
                      <TrendingUp size={13} style={{ color: "#E74C3C" }} />
                    )}
                    {day.trend === "down" && (
                      <TrendingDown size={13} style={{ color: "#2ECC71" }} />
                    )}
                    {day.trend === "same" && (
                      <Minus size={13} style={{ color: "#F1C40F" }} />
                    )}
                    <span
                      style={{
                        color: risk.color,
                        fontSize: 17,
                        fontWeight: 800,
                        letterSpacing: "-0.5px",
                      }}
                    >
                      {day.avg}
                    </span>
                    <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 10 }}>
                      µg/m³
                    </span>
                  </div>
                  <MiniBar value={day.avg} />
                </div>

                {/* Risk badge */}
                <div
                  className="px-2 py-1 rounded-lg ml-1 shrink-0"
                  style={{
                    background: risk.bg,
                    border: `1px solid ${risk.border}`,
                  }}
                >
                  <span style={{ color: risk.color, fontSize: 10, fontWeight: 700 }}>
                    {risk.label}
                  </span>
                </div>
              </button>

              {/* Expanded content */}
              {isExpanded && (
                <div
                  className="px-4 pb-4 pt-1"
                  style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}
                >
                  {/* Stats row */}
                  <div className="flex gap-3 mb-3">
                    <div
                      className="flex-1 rounded-xl p-2.5"
                      style={{ background: "rgba(255,255,255,0.04)" }}
                    >
                      <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 10 }}>
                        Daily Average
                      </div>
                      <div
                        style={{
                          color: risk.color,
                          fontSize: 18,
                          fontWeight: 800,
                          marginTop: 1,
                        }}
                      >
                        {day.avg}
                        <span
                          style={{ color: "rgba(255,255,255,0.3)", fontSize: 10, fontWeight: 400 }}
                        >
                          {" "}µg/m³
                        </span>
                      </div>
                    </div>
                    <div
                      className="flex-1 rounded-xl p-2.5"
                      style={{ background: "rgba(255,255,255,0.04)" }}
                    >
                      <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 10 }}>
                        Peak Value
                      </div>
                      <div
                        style={{
                          color: peakRisk.color,
                          fontSize: 18,
                          fontWeight: 800,
                          marginTop: 1,
                        }}
                      >
                        {day.peak}
                        <span
                          style={{ color: "rgba(255,255,255,0.3)", fontSize: 10, fontWeight: 400 }}
                        >
                          {" "}µg/m³
                        </span>
                      </div>
                    </div>
                    <div
                      className="flex-1 rounded-xl p-2.5"
                      style={{ background: "rgba(255,255,255,0.04)" }}
                    >
                      <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 10 }}>
                        Advice
                      </div>
                      <div
                        style={{
                          color: "white",
                          fontSize: 11,
                          fontWeight: 600,
                          marginTop: 1,
                          lineHeight: 1.3,
                        }}
                      >
                        {day.advice}
                      </div>
                    </div>
                  </div>

                  {/* Detail text */}
                  <p style={{ color: "rgba(255,255,255,0.55)", fontSize: 12, lineHeight: 1.6 }}>
                    {day.detail}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
