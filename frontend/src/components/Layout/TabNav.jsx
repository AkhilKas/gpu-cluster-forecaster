import { Activity, Cpu, Server, TrendingUp } from "lucide-react";

const TABS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "forecast", label: "Forecast", icon: TrendingUp },
  { id: "cluster", label: "Cluster Map", icon: Server },
  { id: "model", label: "Model Performance", icon: Cpu },
];

function TabButton({ tab, active, onClick, Icon }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active
          ? "linear-gradient(135deg, #6366f1, #4f46e5)"
          : "transparent",
        border: active ? "none" : "1px solid #ffffff15",
        color: active ? "#fff" : "#94a3b8",
        borderRadius: 10,
        padding: "10px 20px",
        cursor: "pointer",
        fontSize: 13,
        fontWeight: 600,
        display: "flex",
        alignItems: "center",
        gap: 8,
        transition: "all 0.2s",
      }}
    >
      <Icon size={15} />
      {tab.label}
    </button>
  );
}

export default function TabNav({ active, onSelect }) {
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 20 }}>
      {TABS.map((tab) => (
        <TabButton
          key={tab.id}
          tab={tab}
          Icon={tab.icon}
          active={active === tab.id}
          onClick={() => onSelect(tab.id)}
        />
      ))}
    </div>
  );
}
