from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    machines_available: list[str]
