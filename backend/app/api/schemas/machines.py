from pydantic import BaseModel


class MachineInfo(BaseModel):
    id: str
    num_windows: int
    latest_cpu: float | None = None
    latest_memory: float | None = None
    latest_temperature: float | None = None
    latest_power: float | None = None
    latest_jobs: int | None = None
    status: str | None = None  # "high" | "medium" | "normal" | "low"


class HistoryPoint(BaseModel):
    step: int
    values: dict[str, float]


class HistoryResponse(BaseModel):
    machine_id: str
    denormalized: bool
    columns: list[str]
    steps: list[HistoryPoint]


class ForecastResponse(BaseModel):
    machine_id: str
    model: str
    horizon: int
    target_columns: list[str]
    history: list[HistoryPoint]
    forecast: list[HistoryPoint]


class WorkloadCategory(BaseModel):
    name: str
    percent: float


class WorkloadDistribution(BaseModel):
    avg_cluster_cpu: float
    categories: list[WorkloadCategory]
