export default function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  trend,
  color,
  alert,
}) {
  return (
    <div
      style={{
        background: alert
          ? "linear-gradient(135deg, #1e1215 0%, #2d1520 100%)"
          : "linear-gradient(135deg, #141627 0%, #1a1d35 100%)",
        border: `1px solid ${alert ? "#ef444440" : "#ffffff10"}`,
        borderRadius: 16,
        padding: "20px 24px",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {alert && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 3,
            background: "linear-gradient(90deg, #ef4444, #f59e0b)",
          }}
        />
      )}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <p style={{ color: "#94a3b8", fontSize: 13, margin: 0, fontWeight: 500 }}>
            {label}
          </p>
          <p
            style={{
              color: "#f1f5f9",
              fontSize: 32,
              margin: "8px 0 4px",
              fontWeight: 700,
              letterSpacing: -1,
            }}
          >
            {value}
            <span
              style={{ fontSize: 16, color: "#64748b", fontWeight: 400, marginLeft: 4 }}
            >
              {unit}
            </span>
          </p>
          {trend !== undefined && trend !== null && (
            <span
              style={{
                fontSize: 12,
                color: trend > 0 ? "#f59e0b" : "#10b981",
                fontWeight: 600,
              }}
            >
              {trend > 0 ? "▲" : "▼"} {Math.abs(trend)}% vs 1h ago
            </span>
          )}
        </div>
        <div style={{ background: `${color}20`, borderRadius: 12, padding: 10 }}>
          <Icon size={22} color={color} />
        </div>
      </div>
    </div>
  );
}
