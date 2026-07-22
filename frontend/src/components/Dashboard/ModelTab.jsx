import ModelCard from "../ModelComparison/ModelCard.jsx";
import TrainingHistory from "../Charts/TrainingHistory.jsx";
import { COLORS } from "../../styles/colors.js";

function fmt(x, digits = 2, suffix = "") {
  if (x === null || x === undefined) return "—";
  return `${Number(x).toFixed(digits)}${suffix}`;
}

export default function ModelTab({ metrics, comparison }) {
  const lstm = metrics ?? {};
  const overload = lstm.overload?.cpu_usage?.accuracy;
  const training = lstm.training_history ?? {};

  return (
    <>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 20,
          marginBottom: 24,
        }}
      >
        <ModelCard
          name={`LSTM Model${lstm.name ? ` — ${lstm.name}` : ""}`}
          subtitle="Trained on Google Cluster / synthetic data"
          metrics={[
            {
              label: "MAE",
              value: fmt(lstm.overall?.mae, 4),
              color: COLORS.success,
            },
            {
              label: "RMSE",
              value: fmt(lstm.overall?.rmse, 4),
              color: COLORS.accent,
            },
            {
              label: "MAPE",
              value: fmt(lstm.overall?.mape, 1, "%"),
              color: COLORS.primary,
            },
          ]}
          overload={overload}
        />
        <ModelCard
          name="Transformer (PatchTST)"
          subtitle="Patch-based attention · Multi-horizon"
          metrics={[
            { label: "MAE", value: "—" },
            { label: "RMSE", value: "—" },
            { label: "MAPE", value: "—" },
          ]}
          comingSoon
        />
      </div>
      <TrainingHistory
        trainLoss={training.train_loss}
        valLoss={training.val_loss}
      />
      {comparison && comparison.length > 1 && (
        <div
          style={{
            marginTop: 24,
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
            MODEL COMPARISON
          </h3>
          <table style={{ width: "100%", fontSize: 13, color: "#f1f5f9" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "#64748b" }}>
                <th style={{ padding: "8px 0" }}>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>MAPE</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.name} style={{ borderTop: "1px solid #ffffff08" }}>
                  <td style={{ padding: "8px 0" }}>{row.name}</td>
                  <td>{fmt(row.mae, 4)}</td>
                  <td>{fmt(row.rmse, 4)}</td>
                  <td>{fmt(row.mape, 1, "%")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
