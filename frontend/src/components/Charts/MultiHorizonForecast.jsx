import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartTooltip from "./ChartTooltip.jsx";
import { COLORS } from "../../styles/colors.js";

export default function MultiHorizonForecast({ machineId, data }) {
  return (
    <div
      style={{
        background: "linear-gradient(135deg, #141627, #1a1d35)",
        borderRadius: 16,
        padding: 24,
        border: "1px solid #ffffff08",
        marginBottom: 24,
      }}
    >
      <h3 style={{ margin: "0 0 8px", fontSize: 18, fontWeight: 700 }}>
        Multi-Horizon Forecast — GPU {machineId}
      </h3>
      <p style={{ color: "#64748b", fontSize: 13, margin: "0 0 20px" }}>
        LSTM predictions across the requested horizon
      </p>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
          <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
          <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
          <Tooltip content={<ChartTooltip />} />
          <Legend />
          <Line
            type="monotone"
            dataKey="cpu_usage"
            stroke={COLORS.accent}
            strokeWidth={2.5}
            name="CPU Utilization"
            dot={{ fill: COLORS.accent, r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="memory_usage"
            stroke={COLORS.secondary}
            strokeWidth={2}
            strokeDasharray="6 3"
            name="Memory"
            dot={{ fill: COLORS.secondary, r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
