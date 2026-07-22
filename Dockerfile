# syntax=docker/dockerfile:1.6

# ─── Stage 1: build the Vite frontend ────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend .
# Empty base URL → same-origin, matches the unified deploy.
ENV VITE_API_BASE_URL=""
RUN npm run build


# ─── Stage 2: python runtime with backend + trained model ────────
FROM python:3.11-slim
WORKDIR /app

# torch needs OpenMP at runtime on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only wheel of torch first (much smaller than the default),
# then let requirements.txt pick up everything else.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code + frontend bundle.
COPY backend backend
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Bake the trained model + processed data into the image so the API is ready
# on first boot. Uses synthetic data → deterministic, no network at runtime.
RUN cd backend \
    && python scripts/download_data.py --synthetic --process \
    && python scripts/train.py --model lstm --data synthetic --epochs 15

# Render provides $PORT at runtime.
ENV PORT=8000
EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
