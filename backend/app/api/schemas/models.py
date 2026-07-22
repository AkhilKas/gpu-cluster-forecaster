from typing import Any

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    num_params: int
    config: dict[str, Any]
    has_metrics: bool
    invokable: bool = True


class ModelMetrics(BaseModel):
    name: str
    overall: dict[str, float] | None = None
    per_target: dict[str, dict[str, float]] | None = None
    per_horizon: list[dict[str, Any]] | None = None
    overload: dict[str, dict[str, Any]] | None = None
    training_history: dict[str, list[float]] | None = None


class ModelCompareRow(BaseModel):
    name: str
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
