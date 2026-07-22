import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartTooltip from "./ChartTooltip.jsx";
import { COLORS } from "../../styles/colors.js";

export default function MemoryChart({ data }) {
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
        MEMORY USAGE
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="mem-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.secondary} stopOpacity={0.3} />
              <stop offset="100%" stopColor={COLORS.secondary} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
          <XAxis dataKey="time" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} />
          <Tooltip content={<ChartTooltip />} />
          <Area
            type="monotone"
            dataKey="value"
            stroke={COLORS.secondary}
            fill="url(#mem-grad)"
            strokeWidth={2}
            name="Memory"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
