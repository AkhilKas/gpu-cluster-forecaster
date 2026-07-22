import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_data_store, get_predictor
from app.api.schemas import (
    ForecastResponse,
    HistoryPoint,
    HistoryResponse,
    MachineInfo,
)
from app.inference import DataStore, Predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/machines", tags=["machines"])


@router.get("", response_model=list[MachineInfo])
async def list_machines(
    data_store: DataStore = Depends(get_data_store),
    predictor: Predictor = Depends(get_predictor),
) -> list[MachineInfo]:
    infos: list[MachineInfo] = []
    for mid in sorted(data_store.machines.keys()):
        splits = data_store.machines[mid]
        num_windows = (
            int(splits.get("X_test", []).shape[0]) if "X_test" in splits else 0
        )
        # Latest denormalized CPU / memory from the tail window's last step
        latest_cpu = latest_mem = None
        try:
            window = data_store.get_latest_window(mid)
            denorm = predictor._denormalize_features(mid, window)
            feature_cols = predictor.config.feature_columns
            last_row = denorm[-1]
            if "cpu_usage" in feature_cols:
                latest_cpu = float(last_row[feature_cols.index("cpu_usage")])
            if "memory_usage" in feature_cols:
                latest_mem = float(last_row[feature_cols.index("memory_usage")])
        except Exception as e:
            logger.warning(f"Could not compute latest values for machine {mid}: {e}")
        infos.append(
            MachineInfo(
                id=mid,
                num_windows=num_windows,
                latest_cpu=latest_cpu,
                latest_memory=latest_mem,
            )
        )
    return infos


@router.get("/{machine_id}/history", response_model=HistoryResponse)
async def get_history(
    machine_id: str,
    steps: int = Query(60, ge=1, le=1000),
    denormalized: bool = Query(True),
    data_store: DataStore = Depends(get_data_store),
    predictor: Predictor = Depends(get_predictor),
) -> HistoryResponse:
    if not data_store.has_machine(machine_id):
        raise HTTPException(
            status_code=404, detail=f"Machine {machine_id!r} not found."
        )

    window = data_store.get_latest_window(machine_id)  # (seq_len, num_features)
    n = min(steps, window.shape[0])
    tail = window[-n:]
    values = predictor._denormalize_features(machine_id, tail) if denormalized else tail

    feature_cols = predictor.config.feature_columns[: values.shape[1]]
    history = [
        HistoryPoint(
            step=i,
            values={col: float(values[i, j]) for j, col in enumerate(feature_cols)},
        )
        for i in range(values.shape[0])
    ]
    return HistoryResponse(
        machine_id=machine_id,
        denormalized=denormalized,
        columns=feature_cols,
        steps=history,
    )


@router.get("/{machine_id}/forecast", response_model=ForecastResponse)
async def get_forecast(
    machine_id: str,
    model: str | None = Query(None),
    data_store: DataStore = Depends(get_data_store),
    predictor: Predictor = Depends(get_predictor),
) -> ForecastResponse:
    if not data_store.has_machine(machine_id):
        raise HTTPException(
            status_code=404, detail=f"Machine {machine_id!r} not found."
        )
    result = predictor.forecast_from_machine(machine_id, model)
    result["history"] = [HistoryPoint(**h) for h in result["history"]]
    result["forecast"] = [HistoryPoint(**f) for f in result["forecast"]]
    return ForecastResponse(**result)
