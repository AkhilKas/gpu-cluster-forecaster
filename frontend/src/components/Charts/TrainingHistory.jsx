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

export default function TrainingHistory({ trainLoss = [], valLoss = [] }) {
  const data = trainLoss.map((t, i) => ({
    epoch: i + 1,
    train: +t.toFixed(4),
    val: +(valLoss[i] ?? 0).toFixed(4),
  }));
  return (
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
        TRAINING HISTORY
      </h3>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
          <XAxis
            dataKey="epoch"
            stroke="#64748b"
            fontSize={11}
            label={{ value: "Epoch", position: "bottom", fill: "#64748b", fontSize: 11 }}
          />
          <YAxis stroke="#64748b" fontSize={11} />
          <Tooltip content={<ChartTooltip unit="" />} />
          <Legend />
          <Line
            type="monotone"
            dataKey="train"
            stroke={COLORS.primary}
            strokeWidth={2}
            name="Train Loss"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="val"
            stroke={COLORS.accent}
            strokeWidth={2}
            name="Val Loss"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
