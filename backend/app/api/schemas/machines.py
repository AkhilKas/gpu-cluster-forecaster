from pydantic import BaseModel


class MachineInfo(BaseModel):
    id: str
    num_windows: int
    latest_cpu: float | None = None
    latest_memory: float | None = None


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
