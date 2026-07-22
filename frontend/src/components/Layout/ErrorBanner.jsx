import { AlertTriangle } from "lucide-react";
import { COLORS } from "../../styles/colors.js";
import { api } from "../../services/api.js";

export default function ErrorBanner({ error }) {
  const detail = error?.message || String(error);
  const status = error?.status;
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div
        style={{
          maxWidth: 560,
          background: "linear-gradient(135deg, #1e1215, #2d1520)",
          border: `1px solid ${COLORS.danger}40`,
          borderRadius: 16,
          padding: "32px 40px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: "50%",
            background: `${COLORS.danger}20`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 20px",
          }}
        >
          <AlertTriangle size={28} color={COLORS.danger} />
        </div>
        <h2 style={{ margin: "0 0 12px", color: "#f1f5f9", fontSize: 22 }}>
          Backend unreachable
        </h2>
        <p style={{ color: "#94a3b8", fontSize: 14, margin: "0 0 20px" }}>
          Could not load data from the API at <code>{api.baseUrl}</code>.
        </p>
        <div
          style={{
            background: "#0f111a",
            border: "1px solid #ffffff10",
            borderRadius: 10,
            padding: "12px 16px",
            textAlign: "left",
            fontSize: 12,
            color: "#94a3b8",
            fontFamily: "monospace",
            marginBottom: 20,
          }}
        >
          {status ? `HTTP ${status} — ` : ""}
          {detail}
        </div>
        <p style={{ color: "#64748b", fontSize: 12, margin: 0 }}>
          Start the backend with <code>make serve</code> (from{" "}
          <code>backend/</code>), or set <code>VITE_API_BASE_URL</code> to point at
          a reachable instance.
        </p>
      </div>
    </div>
  );
}
