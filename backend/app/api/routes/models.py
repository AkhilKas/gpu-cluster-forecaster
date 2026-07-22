from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_registry
from app.api.schemas import ModelCompareRow, ModelInfo, ModelMetrics
from app.inference import ModelRegistry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
) -> list[ModelInfo]:
    known = registry.known_names()
    infos = []
    for name in sorted(known):
        model = registry.models.get(name)
        infos.append(
            ModelInfo(
                name=name,
                num_params=model.count_parameters() if model else 0,
                config=model._get_config() if model else {},
                has_metrics=name in registry.metrics_paths,
                invokable=model is not None,
            )
        )
    return infos


@router.get("/compare", response_model=list[ModelCompareRow])
async def compare_models(
    registry: ModelRegistry = Depends(get_registry),
) -> list[ModelCompareRow]:
    rows: list[ModelCompareRow] = []
    for name in sorted(registry.known_names()):
        metrics = registry.load_metrics(name)
        overall = (metrics or {}).get("overall", {})
        rows.append(
            ModelCompareRow(
                name=name,
                mae=overall.get("mae"),
                rmse=overall.get("rmse"),
                mape=overall.get("mape"),
            )
        )
    return rows


@router.get("/{name}/metrics", response_model=ModelMetrics)
async def get_metrics(
    name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelMetrics:
    if name not in registry.known_names():
        raise HTTPException(status_code=404, detail=f"Model {name!r} not found.")
    metrics = registry.load_metrics(name) or {}
    history = registry.load_history(name)
    return ModelMetrics(
        name=name,
        overall=metrics.get("overall"),
        per_target=metrics.get("per_target"),
        per_horizon=metrics.get("per_horizon"),
        overload=metrics.get("overload"),
        training_history=history,
    )
