import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEFAULT_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# Cover Render + Vercel deploy URLs by default so a fresh deploy works with
# zero env-var configuration. Extra hosts can be added via CORS_ORIGINS or
# a custom CORS_ORIGIN_REGEX.
DEFAULT_ORIGIN_REGEX = r"https://.*\.(onrender\.com|vercel\.app)"


def configure_cors(app: FastAPI) -> None:
    """Register CORS middleware.

    - `CORS_ORIGINS` env var (comma-separated) adds explicit origins.
    - `CORS_ORIGIN_REGEX` env var overrides the built-in Render/Vercel regex.
    """
    origins = list(DEFAULT_ORIGINS)
    extra = os.getenv("CORS_ORIGINS", "")
    origins.extend([o.strip() for o in extra.split(",") if o.strip()])

    origin_regex = os.getenv("CORS_ORIGIN_REGEX", DEFAULT_ORIGIN_REGEX)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
