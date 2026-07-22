import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartTooltip from "./ChartTooltip.jsx";
import { COLORS } from "../../styles/colors.js";

/**
 * `data` is a list of {step, value, forecast?} entries. `historyLength`
 * is the number of leading rows that came from real history — used only
 * to render the historical vs forecast legend hint.
 */
export default function UtilizationForecast({ machineId, data }) {
  return (
    <div
      style={{
        background: "linear-gradient(135deg, #141627 0%, #1a1d35 100%)",
        borderRadius: 16,
        padding: 24,
        border: "1px solid #ffffff08",
        marginBottom: 24,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 20,
        }}
      >
        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>
          GPU {machineId} — Utilization &amp; Forecast
        </h3>
        <div style={{ display: "flex", gap: 16, fontSize: 12 }}>
          <span style={{ color: COLORS.primary }}>● Historical</span>
          <span style={{ color: COLORS.accent }}>● LSTM Forecast</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="uf-grad1" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.primary} stopOpacity={0.3} />
              <stop offset="100%" stopColor={COLORS.primary} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="uf-grad2" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.accent} stopOpacity={0.2} />
              <stop offset="100%" stopColor={COLORS.accent} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
          <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
          <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
          <Tooltip content={<ChartTooltip />} />
          <Area
            type="monotone"
            dataKey="value"
            stroke={COLORS.primary}
            fill="url(#uf-grad1)"
            strokeWidth={2}
            name="Utilization"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke={COLORS.accent}
            strokeWidth={2}
            strokeDasharray="6 3"
            name="LSTM Forecast"
            dot={false}
            connectNulls
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
