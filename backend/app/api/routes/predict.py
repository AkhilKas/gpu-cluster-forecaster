import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_predictor, get_registry
from app.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from app.inference import ModelRegistry, Predictor

router = APIRouter(prefix="/predict", tags=["predict"])


def _resolved_name(registry: ModelRegistry, requested: str | None) -> str:
    name = requested or registry.default_name()
    if name is None:
        raise HTTPException(status_code=503, detail="No models loaded on the server.")
    if name not in registry.models:
        raise HTTPException(
            status_code=404, detail=f"Model {name!r} not found or not invokable."
        )
    return name


@router.post("", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    registry: ModelRegistry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    name = _resolved_name(registry, body.model)
    try:
        window = np.asarray(body.window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("`window` must be 2D: (seq_len, num_features)")
        forecast = predictor.predict(window, name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PredictResponse(
        model=name,
        horizon=int(forecast.shape[0]),
        target_columns=predictor.config.target_columns,
        forecast=forecast.tolist(),
    )


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(
    body: BatchPredictRequest,
    registry: ModelRegistry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> BatchPredictResponse:
    name = _resolved_name(registry, body.model)
    try:
        windows = np.asarray(body.windows, dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError("`windows` must be 3D: (batch, seq_len, num_features)")
        forecasts = predictor.predict_batch(windows, name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return BatchPredictResponse(
        model=name,
        horizon=int(forecasts.shape[1]),
        target_columns=predictor.config.target_columns,
        forecasts=forecasts.tolist(),
    )
