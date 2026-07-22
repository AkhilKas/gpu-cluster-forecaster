/**
 * Shape API payloads into the arrays each chart expects.
 * Kept here rather than inline so the tab components stay lean.
 */

function stepLabel(offset) {
  // Offset counted in 5-minute steps from "now".
  const mins = offset * 5;
  if (mins === 0) return "now";
  return `${mins > 0 ? "+" : ""}${mins}m`;
}

/**
 * Merge a forecast payload's history + forecast into a single array with
 * `time`, `value`, and (for future steps) `forecast` keys.
 */
export function utilizationChartData(forecastPayload) {
  if (!forecastPayload) return [];
  const history = forecastPayload.history ?? [];
  const forecast = forecastPayload.forecast ?? [];
  const hist = history.map((p, i) => ({
    time: stepLabel(i - history.length + 1),
    value: p.values.cpu_usage ?? 0,
    forecast: null,
  }));
  const anchor = hist.length ? hist[hist.length - 1].value : null;
  const fwd = forecast.map((p, i) => ({
    time: stepLabel(i + 1),
    value: null,
    forecast: p.values.cpu_usage ?? 0,
  }));
  // Prepend the last history point to the forecast so the line connects.
  if (fwd.length && anchor !== null) {
    fwd.unshift({ time: hist[hist.length - 1].time, value: null, forecast: anchor });
  }
  return [...hist, ...fwd];
}

/** History as memory-only chart rows. */
export function memoryChartData(forecastPayload) {
  const history = forecastPayload?.history ?? [];
  return history.map((p, i) => ({
    time: stepLabel(i - history.length + 1),
    value: p.values.memory_usage ?? 0,
  }));
}

/** Forecast rows with both cpu and memory columns for the Forecast tab. */
export function multiHorizonChartData(forecastPayload) {
  const forecast = forecastPayload?.forecast ?? [];
  return forecast.map((p, i) => ({
    time: stepLabel(i + 1),
    cpu_usage: p.values.cpu_usage ?? 0,
    memory_usage: p.values.memory_usage ?? 0,
  }));
}
