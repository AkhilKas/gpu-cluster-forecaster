export default function ModelMetrics({ metrics }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
      {metrics.map((m, i) => (
        <div
          key={i}
          style={{
            background: "#0f111a",
            borderRadius: 12,
            padding: 16,
            textAlign: "center",
            border: "1px solid #ffffff08",
          }}
        >
          <p
            style={{
              color: "#64748b",
              fontSize: 11,
              margin: 0,
              textTransform: "uppercase",
              letterSpacing: 1,
            }}
          >
            {m.label}
          </p>
          <p
            style={{
              color: m.color || "#f1f5f9",
              fontSize: 24,
              margin: "8px 0 0",
              fontWeight: 700,
            }}
          >
            {m.value}
          </p>
        </div>
      ))}
    </div>
  );
}
