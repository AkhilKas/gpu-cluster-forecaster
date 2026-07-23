import { useEffect, useMemo, useState } from "react";
import Header from "./components/Layout/Header.jsx";
import TabNav from "./components/Layout/TabNav.jsx";
import Footer from "./components/Layout/Footer.jsx";
import ErrorBanner from "./components/Layout/ErrorBanner.jsx";
import OverviewTab from "./components/Dashboard/OverviewTab.jsx";
import ForecastTab from "./components/Dashboard/ForecastTab.jsx";
import ClusterTab from "./components/Dashboard/ClusterTab.jsx";
import ModelTab from "./components/Dashboard/ModelTab.jsx";
import UploadTab from "./components/Dashboard/UploadTab.jsx";
import {
  useForecast,
  useHealth,
  useMachines,
  useModelComparison,
  useModelMetrics,
  useModels,
  useWorkload,
} from "./hooks/useDashboardData.js";
import {
  memoryChartData,
  multiHorizonChartData,
  utilizationChartData,
} from "./utils/transform.js";

function Loading() {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#64748b",
        fontSize: 14,
      }}
    >
      Loading GPU cluster data…
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("overview");
  const [activeId, setActiveId] = useState(null);
  const [horizon, setHorizon] = useState(12);

  const health = useHealth();
  const machines = useMachines();
  const workload = useWorkload();
  const models = useModels();
  const modelComparison = useModelComparison();

  // Default active machine to the first one when the list arrives.
  useEffect(() => {
    if (!activeId && machines.data && machines.data.length > 0) {
      setActiveId(machines.data[0].id);
    }
  }, [activeId, machines.data]);

  const forecast = useForecast(activeId);
  const defaultModelName = models.data?.[0]?.name;
  const modelMetrics = useModelMetrics(defaultModelName);

  const activeMachine = useMemo(
    () => machines.data?.find((m) => m.id === activeId),
    [machines.data, activeId],
  );

  // Fail loud if the backend is unreachable.
  if (health.error) return <ErrorBanner error={health.error} />;
  if (health.loading) return <Loading />;

  // Any other required-fetch error also shows the banner.
  const criticalError = machines.error || workload.error || models.error;
  if (criticalError) return <ErrorBanner error={criticalError} />;

  const utilization = utilizationChartData(forecast.data);
  const memory = memoryChartData(forecast.data);
  const multiHorizon = multiHorizonChartData(forecast.data);
  const overloadPct =
    (forecast.data?.forecast ?? []).reduce(
      (max, p) => Math.max(max, p.values.cpu_usage ?? 0),
      0,
    );

  return (
    <div
      style={{
        background: "#0a0b14",
        minHeight: "100vh",
        color: "#f1f5f9",
      }}
    >
      <div
        style={{
          background: "linear-gradient(180deg, #12142a 0%, #0a0b14 100%)",
          borderBottom: "1px solid #ffffff08",
          padding: "20px 32px",
        }}
      >
        <Header
          status="connected"
          horizon={horizon}
          onHorizonChange={setHorizon}
          subtitle={`Real-time utilization prediction · ${machines.data?.length ?? 0} GPUs · ${defaultModelName ?? "no model"}`}
        />
        <TabNav active={tab} onSelect={setTab} />
      </div>

      <div style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>
        {tab === "overview" && activeMachine && (
          <OverviewTab
            machines={machines.data ?? []}
            activeMachine={activeMachine}
            activeId={activeId}
            onSelectMachine={setActiveId}
            utilization={utilization}
            memory={memory}
            workload={workload.data}
            overloadPct={overloadPct}
          />
        )}
        {tab === "forecast" && activeMachine && (
          <ForecastTab machineId={activeId} forecast={multiHorizon} />
        )}
        {tab === "cluster" && (
          <ClusterTab
            machines={machines.data ?? []}
            onSelectMachine={(id) => {
              setActiveId(id);
              setTab("overview");
            }}
          />
        )}
        {tab === "model" && (
          <ModelTab
            metrics={modelMetrics.data}
            comparison={modelComparison.data}
          />
        )}
        {tab === "upload" && <UploadTab />}
        <Footer />
      </div>
    </div>
  );
}
