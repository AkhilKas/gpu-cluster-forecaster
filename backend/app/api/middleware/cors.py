import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEFAULT_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]


def configure_cors(app: FastAPI) -> None:
    """Register CORS middleware. Extra origins can be added via CORS_ORIGINS env var."""
    origins = list(DEFAULT_ORIGINS)
    extra = os.getenv("CORS_ORIGINS", "")
    origins.extend([o.strip() for o in extra.split(",") if o.strip()])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
