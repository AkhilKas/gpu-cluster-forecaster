import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.inference.model_registry import ModelLoadError

logger = logging.getLogger(__name__)


def configure_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(_: Request, exc: ModelLoadError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def key_error_handler(_: Request, exc: KeyError):
        # KeyError is used by DataStore for "not found"
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {type(exc).__name__}"},
        )
