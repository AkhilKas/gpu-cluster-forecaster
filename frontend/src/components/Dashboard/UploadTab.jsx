import { useState } from "react";
import { AlertTriangle, Download, Upload } from "lucide-react";
import { api } from "../../services/api.js";
import { COLORS } from "../../styles/colors.js";
import { utilizationChartData } from "../../utils/transform.js";
import UtilizationForecast from "../Charts/UtilizationForecast.jsx";

function makeSampleCsv() {
  const header =
    "machine_id,cpu_usage,memory_usage,assigned_memory,cycles_per_instruction";
  const rows = [header];
  for (let m = 0; m < 2; m++) {
    for (let i = 0; i < 70; i++) {
      const cpu = Math.max(
        5,
        Math.min(95, 40 + Math.sin(i / 10) * 15 + (Math.random() - 0.5) * 10),
      ).toFixed(2);
      const mem = Math.max(
        5,
        Math.min(95, 30 + Math.cos(i / 12) * 10 + (Math.random() - 0.5) * 8),
      ).toFixed(2);
      rows.push(`gpu-${m},${cpu},${mem},64,1.02`);
    }
  }
  return rows.join("\n");
}

function downloadSampleCsv() {
  const csv = makeSampleCsv();
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sample_gpu_telemetry.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function WarningList({ warnings }) {
  if (!warnings?.length) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      {warnings.map((w, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            color: COLORS.accent,
            fontSize: 12,
            padding: "4px 0",
          }}
        >
          <AlertTriangle size={14} /> {w}
        </div>
      ))}
    </div>
  );
}

function MachineForecastCard({ machine }) {
  const chartData = utilizationChartData(machine);
  const hasForecast = machine.forecast.length > 0;
  return (
    <div
      style={{
        background: "#141627",
        borderRadius: 16,
        padding: 20,
        border: "1px solid #ffffff08",
        marginBottom: 20,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 12,
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>
            {machine.machine_id}
          </h3>
          <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: 12 }}>
            {machine.num_input_rows} rows uploaded
          </p>
        </div>
      </div>
      <WarningList warnings={machine.warnings} />
      {hasForecast ? (
        <UtilizationForecast machineId={machine.machine_id} data={chartData} />
      ) : (
        <p style={{ color: "#64748b", fontSize: 13 }}>
          No forecast — see warnings above.
        </p>
      )}
    </div>
  );
}

export default function UploadTab() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.uploadCsv(file);
      setResult(data);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div
        style={{
          background: "linear-gradient(135deg, #141627, #1a1d35)",
          borderRadius: 16,
          padding: 24,
          border: "1px solid #ffffff08",
          marginBottom: 24,
        }}
      >
        <h2 style={{ margin: "0 0 8px", fontSize: 20, fontWeight: 700 }}>
          Forecast on your cluster
        </h2>
        <p style={{ color: "#94a3b8", fontSize: 14, margin: "0 0 20px" }}>
          Upload a CSV of your GPU telemetry and the model will forecast the next{" "}
          <strong>60 minutes</strong> for each machine.
        </p>

        <div
          style={{
            background: "#0f111a",
            border: "1px dashed #ffffff20",
            borderRadius: 12,
            padding: 20,
            marginBottom: 16,
          }}
        >
          <p style={{ color: "#94a3b8", fontSize: 13, margin: "0 0 8px" }}>
            CSV format:
          </p>
          <ul
            style={{
              color: "#94a3b8",
              fontSize: 12,
              margin: "0 0 12px",
              paddingLeft: 20,
            }}
          >
            <li>
              <strong style={{ color: "#f1f5f9" }}>Required:</strong>{" "}
              <code>cpu_usage</code>, <code>memory_usage</code> (percentages,
              0–100)
            </li>
            <li>
              <strong style={{ color: "#f1f5f9" }}>Optional:</strong>{" "}
              <code>machine_id</code> (groups by machine), plus{" "}
              <code>assigned_memory</code> and <code>cycles_per_instruction</code>
            </li>
            <li>
              At least <strong style={{ color: "#f1f5f9" }}>60 rows</strong> per
              machine (the model&apos;s input window length)
            </li>
          </ul>
          <button
            onClick={downloadSampleCsv}
            style={{
              background: "transparent",
              border: `1px solid ${COLORS.primary}80`,
              color: COLORS.primary,
              borderRadius: 8,
              padding: "6px 12px",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 600,
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <Download size={13} /> Download sample CSV
          </button>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            style={{ color: "#94a3b8", fontSize: 13 }}
          />
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            style={{
              background:
                !file || loading
                  ? "#1f2333"
                  : "linear-gradient(135deg, #6366f1, #4f46e5)",
              border: "none",
              color: !file || loading ? "#64748b" : "#fff",
              borderRadius: 10,
              padding: "10px 20px",
              cursor: !file || loading ? "not-allowed" : "pointer",
              fontSize: 13,
              fontWeight: 600,
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <Upload size={14} />
            {loading ? "Forecasting…" : "Upload & Forecast"}
          </button>
        </div>

        {error && (
          <div
            style={{
              marginTop: 16,
              padding: 12,
              background: "#2d1520",
              border: `1px solid ${COLORS.danger}40`,
              borderRadius: 10,
              color: COLORS.danger,
              fontSize: 13,
            }}
          >
            <strong>Upload failed</strong>
            <div style={{ marginTop: 4, fontFamily: "monospace", fontSize: 12 }}>
              {error.status ? `HTTP ${error.status} — ` : ""}
              {error.message}
            </div>
          </div>
        )}
      </div>

      {result && (
        <>
          <WarningList warnings={result.warnings} />
          <p style={{ color: "#94a3b8", fontSize: 13, margin: "0 0 16px" }}>
            Forecast from model <code>{result.model}</code> ·{" "}
            {result.num_machines} machine{result.num_machines === 1 ? "" : "s"}
          </p>
          {result.machines.map((m) => (
            <MachineForecastCard key={m.machine_id} machine={m} />
          ))}
        </>
      )}
    </>
  );
}
