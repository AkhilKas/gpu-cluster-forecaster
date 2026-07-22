import ClusterHeatmap from "../Charts/ClusterHeatmap.jsx";

export default function ClusterTab({ machines, onSelectMachine }) {
  return <ClusterHeatmap machines={machines} onSelect={onSelectMachine} />;
}
