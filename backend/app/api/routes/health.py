from fastapi import APIRouter, Depends

from app.api.deps import get_data_store, get_registry
from app.api.schemas import HealthResponse
from app.inference import DataStore, ModelRegistry

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(
    registry: ModelRegistry = Depends(get_registry),
    data_store: DataStore = Depends(get_data_store),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=sorted(registry.models.keys()),
        machines_available=sorted(data_store.machines.keys()),
    )
