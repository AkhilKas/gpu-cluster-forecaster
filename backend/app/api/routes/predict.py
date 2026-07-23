import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_predictor, get_registry
from app.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    UploadPredictResponse,
)
from app.inference import ModelRegistry, Predictor, UploadPredictService

router = APIRouter(prefix="/predict", tags=["predict"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


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


@router.post("/upload", response_model=UploadPredictResponse)
async def predict_upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    registry: ModelRegistry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> UploadPredictResponse:
    """
    Upload a CSV of GPU telemetry and get per-machine forecasts.

    Required columns: cpu_usage, memory_usage.
    Optional: machine_id, assigned_memory, cycles_per_instruction.
    Each machine needs at least `sequence_length` (default 60) rows.
    """
    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must have a .csv extension.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)} bytes; max {MAX_UPLOAD_BYTES}).",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}") from e

    service = UploadPredictService(predictor, registry, predictor.config)
    return service.process(df, model_name=model)
