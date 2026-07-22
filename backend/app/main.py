"""
FastAPI entrypoint.

Usage (dev):
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Programmatic (tests):
    from app.main import create_app
    app = create_app(weights_dir=..., processed_dir=...)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.middleware import configure_cors, configure_error_handlers
from app.api.routes import health as health_routes
from app.api.routes import machines as machine_routes
from app.api.routes import models as model_routes
from app.api.routes import predict as predict_routes
from app.config import BACKEND_DIR, DATA_PROCESSED, WEIGHTS_DIR, DataConfig
from app.inference import DataStore, ModelRegistry, Predictor
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# frontend/dist sits next to backend/ at the repo root.
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


def create_app(
    weights_dir: Path | None = None,
    processed_dir: Path | None = None,
    data_config: DataConfig | None = None,
) -> FastAPI:
    """Build the FastAPI app. Paths default to project constants."""
    weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    processed_dir = Path(processed_dir) if processed_dir else DATA_PROCESSED
    data_config = data_config or DataConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging("INFO")
        logger.info(f"Starting up: weights={weights_dir}, processed={processed_dir}")

        registry = ModelRegistry(device="cpu")
        registry.load_all(weights_dir)

        data_store = DataStore()
        data_store.load(processed_dir)

        predictor = Predictor(registry, data_store, data_config)

        app.state.registry = registry
        app.state.data_store = data_store
        app.state.predictor = predictor
        app.state.data_config = data_config

        logger.info(
            f"Ready: {len(registry.models)} model(s), "
            f"{len(data_store.machines)} machine(s) loaded."
        )
        yield
        logger.info("Shutting down.")

    app = FastAPI(
        title="GPU Cluster Forecaster API",
        version="0.1.0",
        description="Inference API for the LSTM GPU-utilization forecaster.",
        lifespan=lifespan,
    )
    configure_cors(app)
    configure_error_handlers(app)
    app.include_router(health_routes.router)
    app.include_router(model_routes.router)
    app.include_router(machine_routes.router)
    app.include_router(predict_routes.router)

    # Serve the built frontend from the same origin when available.
    # Registered last so API routes above take precedence.
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
        logger.info(f"Serving frontend bundle from {FRONTEND_DIST}")
    else:
        logger.info(
            f"No frontend build at {FRONTEND_DIST}; static mount skipped "
            "(run `npm run build` in frontend/ to enable)."
        )

    return app


app = create_app()
