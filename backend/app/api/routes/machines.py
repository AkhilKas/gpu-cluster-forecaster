import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_data_store, get_predictor
from app.api.schemas import (
    ForecastResponse,
    HistoryPoint,
    HistoryResponse,
    MachineInfo,
    WorkloadDistribution,
)
from app.inference import DataStore, Predictor
from app.inference.derive import (
    jobs_from_cpu,
    power_from_cpu,
    status_from_cpu,
    temperature_from_cpu,
    workload_distribution,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/machines", tags=["machines"])


def _latest_denorm(data_store: DataStore, predictor: Predictor, mid: str):
    """Fetch and denormalize the latest window for a machine. Returns None on error."""
    try:
        window = data_store.get_latest_window(mid)
        denorm = predictor._denormalize_features(mid, window)
        return denorm
    except Exception as e:
        logger.warning(f"Could not compute latest window for machine {mid}: {e}")
        return None


def _extract(denorm_row, feature_cols, col_name):
    """Pull a named column out of a denormalized row."""
    if col_name in feature_cols:
        return float(denorm_row[feature_cols.index(col_name)])
    return None


@router.get("", response_model=list[MachineInfo])
async def list_machines(
    data_store: DataStore = Depends(get_data_store),
    predictor: Predictor = Depends(get_predictor),
) -> list[MachineInfo]:
    feature_cols = predictor.config.feature_columns
    infos: list[MachineInfo] = []
    for mid in sorted(data_store.machines.keys()):
        splits = data_store.machines[mid]
        num_windows = (
            int(splits.get("X_test", []).shape[0]) if "X_test" in splits else 0
        )
        latest_cpu = latest_mem = None
        latest_temp = latest_power = None
        latest_jobs = None
        status = None
        denorm = _latest_denorm(data_store, predictor, mid)
        if denorm is not None:
            last_row = denorm[-1]
            latest_cpu = _extract(last_row, feature_cols, "cpu_usage")
            latest_mem = _extract(last_row, feature_cols, "memory_usage")
            if latest_cpu is not None:
                latest_temp = temperature_from_cpu(latest_cpu)
                latest_power = power_from_cpu(latest_cpu)
                latest_jobs = jobs_from_cpu(latest_cpu)
                status = status_from_cpu(latest_cpu)
        infos.append(
            MachineInfo(
                id=mid,
                num_windows=num_windows,
                latest_cpu=latest_cpu,
                latest_memory=latest_mem,
                latest_temperature=latest_temp,
                latest_power=latest_power,
                latest_jobs=latest_jobs,
                status=status,
            )
        )
    return infos


@router.get("/workload", response_model=WorkloadDistribution)
async def get_workload_distribution(
    data_store: DataStore = Depends(get_data_store),
    predictor: Predictor = Depends(get_predictor),
) -> WorkloadDistribution:
    """Cluster-wide workload distribution derived from average utilization."""
    feature_cols = predictor.config.feature_columns
    cpus = []
    for mid in data_store.machines:
        denorm = _latest_denorm(data_store, predictor, mid)
        if denorm is None:
            continue
        val = _extract(denorm[-1], feature_cols, "cpu_usage")
        if val is not None:
            cpus.append(val)
    avg = float(np.mean(cpus)) if cpus else 0.0
    return workload_distribution(avg)


def _augment_point(values: dict[str, float]) -> dict[str, float]:
    """Add derived operational fields to a values dict if cpu_usage is present."""
    cpu = values.get("cpu_usage")
    if cpu is not None:
        values = {
            **values,
            "temperature": temperature_from_cpu(cpu),
            "power_draw": power_from_cpu(cpu),
            "jobs_count": float(jobs_from_cpu(cpu)),
        }
    return values


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
    steps_out: list[HistoryPoint] = []
    for i in range(values.shape[0]):
        row = {col: float(values[i, j]) for j, col in enumerate(feature_cols)}
        if denormalized:
            row = _augment_point(row)
        steps_out.append(HistoryPoint(step=i, values=row))

    columns = list(feature_cols)
    if denormalized and "cpu_usage" in feature_cols:
        columns = [*columns, "temperature", "power_draw", "jobs_count"]

    return HistoryResponse(
        machine_id=machine_id,
        denormalized=denormalized,
        columns=columns,
        steps=steps_out,
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
    result["history"] = [
        HistoryPoint(step=h["step"], values=_augment_point(h["values"]))
        for h in result["history"]
    ]
    result["forecast"] = [
        HistoryPoint(step=f["step"], values=_augment_point(f["values"]))
        for f in result["forecast"]
    ]
    return ForecastResponse(**result)
