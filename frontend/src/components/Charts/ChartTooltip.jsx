export default function ChartTooltip({ active, payload, label, unit = "%" }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#1a1d35ee",
        border: "1px solid #ffffff15",
        borderRadius: 10,
        padding: "10px 14px",
        backdropFilter: "blur(10px)",
      }}
    >
      <p style={{ color: "#94a3b8", fontSize: 11, margin: 0 }}>{label}</p>
      {payload.map((p, i) => (
        <p
          key={i}
          style={{ color: p.color, fontSize: 13, margin: "4px 0 0", fontWeight: 600 }}
        >
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
          {unit}
        </p>
      ))}
    </div>
  );
}
