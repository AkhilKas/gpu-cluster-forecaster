from pydantic import BaseModel, Field


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
