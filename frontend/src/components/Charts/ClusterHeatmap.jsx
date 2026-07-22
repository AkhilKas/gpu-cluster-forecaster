import { statusLabel } from "../../styles/colors.js";

function hueFor(pct) {
  if (pct > 80) return 0;
  if (pct > 60) return 35;
  if (pct > 30) return 150;
  return 200;
}

export default function ClusterHeatmap({ machines, onSelect }) {
  return (
    <div
      style={{
        background: "#141627",
        borderRadius: 16,
        padding: 24,
        border: "1px solid #ffffff08",
      }}
    >
      <h3 style={{ margin: "0 0 20px", fontSize: 18, fontWeight: 700 }}>
        Cluster GPU Heatmap
      </h3>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 16,
        }}
      >
        {machines.map((m) => {
          const cpu = m.latest_cpu ?? 0;
          const hue = hueFor(cpu);
          return (
            <div
              key={m.id}
              onClick={() => onSelect?.(m.id)}
              style={{
                background: `linear-gradient(135deg, hsl(${hue}, 60%, 12%), hsl(${hue}, 50%, 8%))`,
                border: `1px solid hsl(${hue}, 60%, 25%)`,
                borderRadius: 16,
                padding: 20,
                cursor: onSelect ? "pointer" : "default",
                transition: "all 0.3s",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 12,
                }}
              >
                <span style={{ fontWeight: 700, fontSize: 15 }}>GPU {m.id}</span>
                <span
                  style={{
                    fontSize: 11,
                    color: `hsl(${hue}, 70%, 60%)`,
                    background: `hsl(${hue}, 60%, 15%)`,
                    padding: "3px 8px",
                    borderRadius: 6,
                    fontWeight: 600,
                  }}
                >
                  {statusLabel(cpu)}
                </span>
              </div>
              <p
                style={{
                  fontSize: 32,
                  fontWeight: 800,
                  margin: "0 0 12px",
                  color: `hsl(${hue}, 70%, 65%)`,
                }}
              >
                {cpu.toFixed(1)}%
              </p>
              <div
                style={{
                  fontSize: 12,
                  color: "#94a3b8",
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 6,
                }}
              >
                <span>Mem: {(m.latest_memory ?? 0).toFixed(1)}%</span>
                <span>Jobs: {m.latest_jobs ?? 0}</span>
                <span>Temp: {(m.latest_temperature ?? 0).toFixed(1)}°C</span>
                <span>Power: {(m.latest_power ?? 0).toFixed(0)}W</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
