import { Outlet, useNavigate, useLocation } from "react-router";
import { Home, Map, BarChart3 } from "lucide-react";

export function Layout() {
  const navigate = useNavigate();
  const location = useLocation();

  const tabs = [
    { path: "/", icon: Home, label: "Home" },
    { path: "/map", icon: Map, label: "Map" },
    { path: "/forecast", icon: BarChart3, label: "Forecast" },
  ];

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ background: "#060D1A", fontFamily: "'Inter', sans-serif" }}
    >
      {/* Phone Frame */}
      <div
        className="relative flex flex-col overflow-hidden"
        style={{
          width: "100%",
          maxWidth: 390,
          height: "100svh",
          maxHeight: 844,
          background: "#0F172A",
          borderRadius: "clamp(0px, 2vw, 40px)",
          boxShadow: "0 40px 120px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.05)",
        }}
      >
        {/* Screen Content */}
        <div className="flex-1 overflow-hidden relative">
          <Outlet />
        </div>

        {/* Bottom Navigation */}
        <div
          style={{
            background: "rgba(15,23,42,0.95)",
            backdropFilter: "blur(20px)",
            borderTop: "1px solid rgba(255,255,255,0.06)",
            paddingBottom: "env(safe-area-inset-bottom, 0px)",
          }}
        >
          <div className="flex items-center justify-around px-6 py-3">
            {tabs.map((tab) => {
              const active = location.pathname === tab.path;
              return (
                <button
                  key={tab.path}
                  onClick={() => navigate(tab.path)}
                  className="flex flex-col items-center gap-1 px-4 py-1 relative"
                >
                  {active && (
                    <div
                      className="absolute inset-0 rounded-xl"
                      style={{ background: "rgba(56,189,248,0.08)" }}
                    />
                  )}
                  <tab.icon
                    size={22}
                    style={{
                      color: active ? "#38BDF8" : "rgba(255,255,255,0.35)",
                      transition: "color 0.2s",
                    }}
                    strokeWidth={active ? 2.5 : 1.8}
                  />
                  <span
                    style={{
                      fontSize: 10,
                      color: active ? "#38BDF8" : "rgba(255,255,255,0.35)",
                      fontWeight: active ? 600 : 400,
                      transition: "color 0.2s",
                    }}
                  >
                    {tab.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
