export default function Footer() {
  return (
    <div
      style={{
        textAlign: "center",
        marginTop: 32,
        padding: 20,
        borderTop: "1px solid #ffffff08",
      }}
    >
      <p style={{ color: "#475569", fontSize: 12, margin: 0 }}>
        GPU Cluster Forecaster · Built with PyTorch + FastAPI + React · Google
        Cluster Dataset
      </p>
    </div>
  );
}
