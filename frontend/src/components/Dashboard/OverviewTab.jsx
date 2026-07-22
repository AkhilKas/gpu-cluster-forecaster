import { Activity, AlertTriangle, Thermometer, Zap } from "lucide-react";
import MetricCard from "./MetricCard.jsx";
import GPUSelector from "./GPUSelector.jsx";
import UtilizationForecast from "../Charts/UtilizationForecast.jsx";
import MemoryChart from "../Charts/MemoryChart.jsx";
import WorkloadPie from "../Charts/WorkloadPie.jsx";
import { COLORS } from "../../styles/colors.js";

export default function OverviewTab({
  machines,
  activeMachine,
  activeId,
  onSelectMachine,
  utilization,
  memory,
  workload,
  overloadPct,
}) {
  const clusterUtil =
    machines.length > 0
      ? machines.reduce((s, m) => s + (m.latest_cpu ?? 0), 0) / machines.length
      : 0;
  const totalPowerKW =
    machines.reduce((s, m) => s + (m.latest_power ?? 0), 0) / 1000;
  const latestTemp = activeMachine?.latest_temperature ?? 0;

  return (
    <>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 16,
          marginBottom: 24,
        }}
      >
        <MetricCard
          icon={Activity}
          label="Cluster Utilization"
          value={clusterUtil.toFixed(1)}
          unit="%"
          color={COLORS.primary}
        />
        <MetricCard
          icon={Thermometer}
          label="Avg Temperature"
          value={latestTemp.toFixed(1)}
          unit="°C"
          color={COLORS.secondary}
        />
        <MetricCard
          icon={Zap}
          label="Total Power Draw"
          value={totalPowerKW.toFixed(1)}
          unit="kW"
          color={COLORS.accent}
        />
        <MetricCard
          icon={AlertTriangle}
          label="Overload Risk"
          value={overloadPct.toFixed(1)}
          unit="%"
          color={overloadPct > 30 ? COLORS.danger : COLORS.accent}
          alert={overloadPct > 30}
        />
      </div>

      <div
        style={{
          background: "#141627",
          borderRadius: 16,
          padding: 20,
          border: "1px solid #ffffff08",
          marginBottom: 24,
        }}
      >
        <h3
          style={{
            margin: "0 0 14px",
            fontSize: 14,
            fontWeight: 700,
            color: "#94a3b8",
          }}
        >
          SELECT GPU
        </h3>
        <GPUSelector
          machines={machines}
          activeId={activeId}
          onSelect={onSelectMachine}
        />
      </div>

      <UtilizationForecast machineId={activeId} data={utilization} />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <MemoryChart data={memory} />
        <WorkloadPie categories={workload?.categories} />
      </div>
    </>
  );
}
