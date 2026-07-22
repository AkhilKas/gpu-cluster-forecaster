/**
 * Thin fetch wrapper around the FastAPI backend.
 * Reads base URL from VITE_API_BASE_URL (default: http://localhost:8000).
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, status, url) {
    super(message);
    this.status = status;
    this.url = url;
  }
}

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  let response;
  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (e) {
    throw new ApiError(`Network error: ${e.message}`, 0, url);
  }
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text().catch(() => "");
    }
    throw new ApiError(
      detail || `HTTP ${response.status}`,
      response.status,
      url,
    );
  }
  return response.json();
}

export const api = {
  baseUrl: BASE_URL,
  health: () => request("/health"),
  listMachines: () => request("/machines"),
  history: (id, steps = 60) =>
    request(`/machines/${encodeURIComponent(id)}/history?steps=${steps}`),
  forecast: (id, model) => {
    const q = model ? `?model=${encodeURIComponent(model)}` : "";
    return request(`/machines/${encodeURIComponent(id)}/forecast${q}`);
  },
  workload: () => request("/machines/workload"),
  listModels: () => request("/models"),
  modelMetrics: (name) =>
    request(`/models/${encodeURIComponent(name)}/metrics`),
  compareModels: () => request("/models/compare"),
  predict: (window, model) =>
    request("/predict", {
      method: "POST",
      body: JSON.stringify({ window, model }),
    }),
};
