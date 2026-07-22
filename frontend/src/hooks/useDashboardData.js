import { useApi } from "./useApi.js";
import { api } from "../services/api.js";

export function useHealth() {
  return useApi(() => api.health(), []);
}

export function useMachines() {
  return useApi(() => api.listMachines(), []);
}

export function useForecast(machineId) {
  return useApi(
    () => (machineId ? api.forecast(machineId) : Promise.resolve(null)),
    [machineId],
  );
}

export function useWorkload() {
  return useApi(() => api.workload(), []);
}

export function useModels() {
  return useApi(() => api.listModels(), []);
}

export function useModelMetrics(name) {
  return useApi(
    () => (name ? api.modelMetrics(name) : Promise.resolve(null)),
    [name],
  );
}

export function useModelComparison() {
  return useApi(() => api.compareModels(), []);
}
