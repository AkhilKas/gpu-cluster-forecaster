/**
 * Thin fetch wrapper around the FastAPI backend.
 *
 * BASE_URL resolution:
 *   1. If VITE_API_BASE_URL is set (including empty string), use it.
 *   2. Otherwise, in production builds → "" (relative, same-origin as the
 *      served bundle — matches the unified Render deploy).
 *   3. Otherwise, in dev → http://localhost:8000.
 */

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.PROD ? "" : "http://localhost:8000");

export class ApiError extends Error {
  constructor(message, status, url) {
    super(message);
    this.status = status;
    this.url = url;
  }
}

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const headers = { ...(options.headers || {}) };
  // Only send Content-Type when we actually have a body. Adding it to GETs
  // makes them "non-simple" CORS requests and triggers a preflight even for
  // same-origin — which is what happens on Render deploys.
  // Skip for FormData — the browser sets it with the multipart boundary.
  const isFormData = options.body instanceof FormData;
  if (options.body != null && !isFormData && !("Content-Type" in headers)) {
    headers["Content-Type"] = "application/json";
  }
  let response;
  try {
    response = await fetch(url, { ...options, headers });
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
  uploadCsv: (file, model = null) => {
    const form = new FormData();
    form.append("file", file);
    if (model) form.append("model", model);
    return request("/predict/upload", { method: "POST", body: form });
  },
};
