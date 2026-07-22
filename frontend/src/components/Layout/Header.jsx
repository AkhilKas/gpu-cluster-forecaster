import { Cpu } from "lucide-react";
import { COLORS } from "../../styles/colors.js";

export default function Header({ status, horizon, onHorizonChange, subtitle }) {
  const dotColor = status === "connected" ? COLORS.success : COLORS.accent;
  const label = status === "connected" ? "API Connected" : "Connecting…";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 16,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div
          style={{
            background: "linear-gradient(135deg, #6366f1, #06b6d4)",
            borderRadius: 14,
            padding: 10,
          }}
        >
          <Cpu size={24} color="#fff" />
        </div>
        <div>
          <h1
            style={{
              margin: 0,
              fontSize: 22,
              fontWeight: 800,
              background: "linear-gradient(135deg, #f1f5f9, #94a3b8)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            GPU Cluster Forecaster
          </h1>
          <p style={{ margin: 0, color: "#64748b", fontSize: 13 }}>{subtitle}</p>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            background: "#0f111a",
            borderRadius: 8,
            padding: "6px 14px",
            border: "1px solid #ffffff10",
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: dotColor,
              boxShadow: `0 0 8px ${dotColor}`,
            }}
          />
          <span style={{ fontSize: 12, color: "#94a3b8" }}>{label}</span>
        </div>
        <select
          value={horizon}
          onChange={(e) => onHorizonChange(+e.target.value)}
          style={{
            background: "#0f111a",
            border: "1px solid #ffffff15",
            color: "#f1f5f9",
            borderRadius: 8,
            padding: "8px 12px",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          <option value={6}>30 min forecast</option>
          <option value={12}>60 min forecast</option>
          <option value={24}>2 hour forecast</option>
        </select>
      </div>
    </div>
  );
}
