from pydantic import BaseModel, Field

from .machines import HistoryPoint


class PredictRequest(BaseModel):
    window: list[list[float]] = Field(
        ...,
        description="Input window shaped (seq_len, num_features). Must be normalized.",
    )
    model: str | None = Field(
        None, description="Model name. If omitted, the default model is used."
    )


class PredictResponse(BaseModel):
    model: str
    horizon: int
    target_columns: list[str]
    forecast: list[list[float]] = Field(
        ..., description="Forecast shaped (horizon, num_targets)."
    )


class BatchPredictRequest(BaseModel):
    windows: list[list[list[float]]] = Field(
        ...,
        description="Batch of input windows, shape (batch, seq_len, num_features).",
    )
    model: str | None = None


class BatchPredictResponse(BaseModel):
    model: str
    horizon: int
    target_columns: list[str]
    forecasts: list[list[list[float]]]


class UploadedMachine(BaseModel):
    machine_id: str
    num_input_rows: int
    history: list[HistoryPoint] = []
    forecast: list[HistoryPoint] = []
    warnings: list[str] = []


class UploadPredictResponse(BaseModel):
    model: str
    num_machines: int
    target_columns: list[str]
    machines: list[UploadedMachine]
    warnings: list[str] = []
