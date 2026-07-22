"""
Derived operational metrics.

The training data (CPU + memory) doesn't include GPU temperature, power draw,
running-jobs count, or workload categorization. These are computed server-side
from the observed cpu_usage using the same formulas that
`ClusterDataLoader.generate_synthetic` uses to generate its test data — so
the shape and range match what the model was trained against.
"""

from app.api.schemas import WorkloadCategory, WorkloadDistribution


def temperature_from_cpu(cpu_pct: float) -> float:
    """0-100 % CPU → 40-80 °C, matching synthetic-data generator."""
    return round(40.0 + cpu_pct * 0.4, 1)


def power_from_cpu(cpu_pct: float) -> float:
    """0-100 % CPU → 80-330 W."""
    return round(80.0 + cpu_pct * 2.5, 1)


def jobs_from_cpu(cpu_pct: float) -> int:
    """0-100 % CPU → 0-4 concurrent jobs (clipped at 10)."""
    return int(min(10, max(0, cpu_pct / 25)))


def status_from_cpu(cpu_pct: float) -> str:
    """Utilization band for at-a-glance UI badges."""
    if cpu_pct > 80:
        return "high"
    if cpu_pct > 60:
        return "medium"
    if cpu_pct > 30:
        return "normal"
    return "low"


def workload_distribution(avg_cluster_cpu: float) -> WorkloadDistribution:
    """
    Cluster-wide workload categorization derived from average utilization.

    Simple heuristic: at higher load, more work is training/inference; at low
    load, most GPUs are idle. Percentages sum to 100.
    """
    activity = max(0.0, min(1.0, avg_cluster_cpu / 100.0))
    training = round(activity * 55, 1)
    inference = round(activity * 25, 1)
    data_proc = round(activity * 15, 1)
    idle = round(max(0.0, 100.0 - training - inference - data_proc), 1)
    return WorkloadDistribution(
        avg_cluster_cpu=round(avg_cluster_cpu, 1),
        categories=[
            WorkloadCategory(name="Training", percent=training),
            WorkloadCategory(name="Inference", percent=inference),
            WorkloadCategory(name="Data Proc", percent=data_proc),
            WorkloadCategory(name="Idle", percent=idle),
        ],
    )
