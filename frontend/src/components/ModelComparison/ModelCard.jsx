import ModelMetrics from "./ModelMetrics.jsx";
import { COLORS } from "../../styles/colors.js";

export default function ModelCard({ name, subtitle, metrics, overload, comingSoon }) {
  return (
    <div
      style={{
        background: "linear-gradient(135deg, #141627, #1a1d35)",
        borderRadius: 16,
        padding: 24,
        border: "1px solid #ffffff08",
        opacity: comingSoon ? 0.6 : 1,
        position: "relative",
      }}
    >
      {comingSoon && (
        <div
          style={{
            position: "absolute",
            top: 12,
            right: 12,
            background: "#6366f130",
            color: COLORS.primary,
            fontSize: 11,
            fontWeight: 700,
            padding: "4px 10px",
            borderRadius: 6,
          }}
        >
          COMING SOON
        </div>
      )}
      <h3 style={{ margin: "0 0 4px", fontSize: 16, fontWeight: 700 }}>{name}</h3>
      <p style={{ color: "#64748b", fontSize: 12, margin: "0 0 20px" }}>{subtitle}</p>
      <ModelMetrics metrics={metrics} />
      {overload !== undefined && overload !== null && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            background: "#0f111a",
            borderRadius: 10,
            border: "1px solid #ffffff08",
          }}
        >
          <p style={{ fontSize: 12, color: "#94a3b8", margin: 0 }}>
            <span style={{ color: COLORS.success, fontWeight: 700 }}>
              {(overload * 100).toFixed(1)}%
            </span>{" "}
            Overload Prediction Accuracy
          </p>
        </div>
      )}
    </div>
  );
}
