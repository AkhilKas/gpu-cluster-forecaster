import { COLORS, GPU_COLORS } from "../../styles/colors.js";

export default function GPUSelector({ machines, activeId, onSelect }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {machines.map((m, i) => {
        const color = GPU_COLORS[i % GPU_COLORS.length];
        const active = m.id === activeId;
        const cpu = m.latest_cpu ?? 0;
        const utilColor =
          cpu > 80 ? COLORS.danger : cpu > 60 ? COLORS.accent : COLORS.success;
        return (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            style={{
              background: active ? `${color}25` : "#0f111a",
              border: `1px solid ${active ? color : "#ffffff10"}`,
              color: active ? color : "#94a3b8",
              borderRadius: 10,
              padding: "8px 14px",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 600,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              minWidth: 70,
            }}
          >
            <span>GPU {m.id}</span>
            <span style={{ fontSize: 11, color: utilColor }}>
              {cpu.toFixed(1)}%
            </span>
          </button>
        );
      })}
    </div>
  );
}
