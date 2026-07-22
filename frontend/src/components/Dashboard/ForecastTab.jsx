import MultiHorizonForecast from "../Charts/MultiHorizonForecast.jsx";
import { COLORS } from "../../styles/colors.js";

function summaryColor(v) {
  if (v > 80) return COLORS.danger;
  if (v > 60) return COLORS.accent;
  return COLORS.success;
}

export default function ForecastTab({ machineId, forecast }) {
  const summarySteps = [1, 3, 6, 12];
  return (
    <>
      <MultiHorizonForecast machineId={machineId} data={forecast} />
      <div
        style={{
          background: "#141627",
          borderRadius: 16,
          padding: 24,
          border: "1px solid #ffffff08",
        }}
      >
        <h3
          style={{
            margin: "0 0 16px",
            fontSize: 14,
            fontWeight: 700,
            color: "#94a3b8",
          }}
        >
          FORECAST SUMMARY
        </h3>
        <div
          style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}
        >
          {summarySteps.map((step) => {
            const point = forecast[step - 1];
            const cpu = point?.cpu_usage ?? 0;
            return (
              <div
                key={step}
                style={{
                  background: "#0f111a",
                  borderRadius: 12,
                  padding: 16,
                  textAlign: "center",
                  border: "1px solid #ffffff08",
                }}
              >
                <p style={{ color: "#64748b", fontSize: 12, margin: 0 }}>
                  +{step * 5}m
                </p>
                <p
                  style={{
                    color: summaryColor(cpu),
                    fontSize: 28,
                    fontWeight: 700,
                    margin: "8px 0 4px",
                  }}
                >
                  {cpu.toFixed(1)}%
                </p>
                <p style={{ color: "#475569", fontSize: 11, margin: 0 }}>GPU Util</p>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}
