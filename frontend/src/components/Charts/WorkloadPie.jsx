import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { COLORS } from "../../styles/colors.js";

const SLICE_COLORS = [COLORS.primary, COLORS.secondary, COLORS.accent, COLORS.muted];

export default function WorkloadPie({ categories }) {
  const data = (categories ?? []).map((c) => ({ name: c.name, value: c.percent }));
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
        WORKLOAD DISTRIBUTION
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            dataKey="value"
            stroke="none"
          >
            {data.map((_e, i) => (
              <Cell key={i} fill={SLICE_COLORS[i % SLICE_COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(v) => `${v}%`} />
          <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
