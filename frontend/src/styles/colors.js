export const COLORS = {
  primary: "#6366f1",
  secondary: "#06b6d4",
  accent: "#f59e0b",
  danger: "#ef4444",
  success: "#10b981",
  muted: "#64748b",
};

export const GPU_COLORS = [
  "#6366f1", "#06b6d4", "#f59e0b", "#10b981",
  "#f43f5e", "#8b5cf6", "#ec4899", "#14b8a6",
];

export function utilizationColor(pct) {
  if (pct > 80) return COLORS.danger;
  if (pct > 60) return COLORS.accent;
  if (pct > 30) return COLORS.success;
  return COLORS.muted;
}

export function statusLabel(pct) {
  if (pct > 80) return "HIGH";
  if (pct > 60) return "MEDIUM";
  if (pct > 30) return "NORMAL";
  return "LOW";
}
