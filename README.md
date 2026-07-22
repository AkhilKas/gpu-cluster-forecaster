# GPU Cluster Forecaster

A deep-learning system that predicts near-future GPU utilization and memory usage across an ML cluster, so a scheduler can place new work **before** load actually arrives — reducing idle time and heading off overloads. Ships as a full stack: PyTorch training pipeline, FastAPI inference server, and a React dashboard.

## What it does

Given the last five hours of per-machine telemetry (60 five-minute intervals), the model predicts the next hour of utilization (12 more intervals) for each GPU in the cluster. The dashboard visualizes the result:

- Current utilization, temperature, and power draw across the fleet
- Historical trends per GPU with the LSTM's forecast overlaid
- Multi-horizon forecasts (5 min out through 60 min)
- A cluster-wide heatmap for at-a-glance load spotting
- Model performance metrics (MAE, RMSE, MAPE, overload-detection accuracy, training curves)

The current v0 uses the public **Google Cluster Data 2019** dataset. CPU utilization stands in for GPU utilization since the public dataset doesn't include GPU-specific traces — the pipeline is architected so real GPU telemetry can drop in later without changes to the model or API contract.

## How it works

```
Raw traces  →  Preprocess         →  LSTM          →  FastAPI        →  React
(CSV shards    (resample, scale,      (60-step        (loads model      dashboard
 from GCS)      slide windows,         input,          + scalers,        (Vite +
                per-machine split)     12-step         serves 9          Recharts)
                                       forecast)      endpoints)
```

1. **Ingest.** `scripts/download_data.py` fetches a few shards of Google Cluster Data 2019 (or generates synthetic telemetry for dev).
2. **Preprocess.** Resample the raw event stream to 5-minute bins, forward-fill small gaps, min-max scale on the training portion only, and turn the time series into `(60-step input, 12-step target)` sliding-window pairs — one set per machine. The scaler is saved so the API can reverse the transform at inference time.
3. **Train.** An LSTM (or a linear / moving-average baseline for comparison) is trained with Adam + MSE + gradient clipping + early stopping. The best checkpoint is saved under `backend/weights/` along with metrics and training-history JSON.
4. **Serve.** FastAPI loads every `*_best.pt` at startup, plus the fitted scalers, and exposes nine endpoints for listing machines, listing models, running predictions, and fetching per-machine history + forecast. Responses are denormalized back to real percentages so the frontend never has to know about scaling.
5. **Visualize.** The React dashboard hits those endpoints, reshapes the payloads for Recharts, and renders four tabs (Overview, Forecast, Cluster Map, Model Performance).

## Tech stack

| Layer      | Choices                                                        |
|------------|----------------------------------------------------------------|
| Data       | Google Cluster Data 2019 (with a synthetic fallback for dev)   |
| ML         | Python 3.10+, PyTorch, scikit-learn, NumPy, pandas             |
| Backend    | FastAPI + Uvicorn                                              |
| Frontend   | React 18, Vite, Recharts, lucide-react                         |
| Tooling    | pytest, Black, Ruff, ESLint, GitHub Actions CI                 |

## Getting started

### Prerequisites
- Python 3.10 or newer
- Node 20 or newer
- ~500 MB free disk if using real cluster data; synthetic mode is much smaller

### 1. Clone and set up the backend

```bash
git clone https://github.com/AkhilKas/gpu-cluster-forecaster.git
cd gpu-cluster-forecaster/backend

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt # for tests + linting
```

### 2. Generate data and train a model

```bash
# Synthetic data — fast, no download (recommended for a first run)
python scripts/download_data.py --synthetic --process

# Or use real Google Cluster Data (~3 shards, hundreds of MB)
# python scripts/download_data.py --shards 3 --process

python scripts/train.py --model lstm --data synthetic --epochs 20
```

This step:
- writes preprocessed `.npy` splits to `data/processed/machine_*/`
- saves a checkpoint at `backend/weights/lstm_*_best.pt`
- writes `..._metrics.json` and `..._history.json` sidecars used by the dashboard

### 3. Start the API

```bash
make serve
# or: uvicorn app.main:app --reload --port 8000
```

Interactive OpenAPI docs live at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Start the dashboard

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The dashboard fails loud with an error banner if the API is unreachable — start the backend first.

### Running the tests

```bash
# Backend
cd backend && pytest tests/         # 48 tests

# Frontend
cd frontend && npm run build && npm run lint
```

## Project status

- [x] Data pipeline (download, preprocess, windowing)
- [x] LSTM model (+ Seq2Seq variant, linear / moving-average baselines, training loop, evaluator)
- [ ] Transformer / PatchTST model (Phase 2 — comparison model)
- [x] FastAPI backend (9 endpoints, auto-loads models + scalers, CORS wired for Vite)
- [x] React dashboard (Vite + Recharts, 4 tabs wired to real API)
- [ ] Deployment (Render for backend, Vercel for frontend)

## Repository layout

```
backend/
  app/
    config.py           Data + train config (dataclasses)
    data/               Loaders, preprocessor, PyTorch Dataset
    models/             LSTM, LSTM-Seq2Seq, linear + moving-average baselines
    training/           Trainer, callbacks, evaluator
    inference/          Model registry, data store, prediction service
    api/                FastAPI routes, schemas, middleware
    main.py             App factory + startup lifespan
  scripts/              CLI: download_data.py, train.py
  tests/                pytest suite (data / models / training / api)
  weights/              Saved checkpoints + metrics/history JSON (gitignored)
  Makefile              format / lint / test / train / serve
frontend/
  src/
    App.jsx             Composition + top-level state
    components/         Layout, Dashboard, Charts, ModelComparison
    hooks/              useApi + endpoint-specific hooks
    services/api.js     Fetch wrapper for all 9 endpoints
    utils/transform.js  API payload → chart data
    styles/colors.js    Shared palette
data/
  raw/                  Downloaded shards (gitignored)
  processed/            .npy windows + scalers.pkl (gitignored)
.github/workflows/      Backend CI (lint + tests) and Frontend CI (lint + build)
```

## Design notes

- **Chronological splits, always.** Time series need ordered splits — the preprocessor never shuffles raw timesteps. Windows can be shuffled inside the train DataLoader because a shuffled window is still internally ordered.
- **Per-machine scalers.** Every GPU has its own `MinMaxScaler` fit on **its** training portion only, saved to `scalers.pkl` so the API can return predictions in real percentages instead of `[0,1]` normalized values.
- **Model registry as a factory.** On startup, FastAPI scans `backend/weights/*_best.pt` and rebuilds each model from the checkpoint's saved config. Adding a new model is a matter of subclassing `BaseForecaster` — the registry picks it up automatically.
- **Derived operational metrics.** Temperature, power draw, running-jobs count, and workload distribution are computed server-side from CPU utilization (using the same formulas as the synthetic-data generator). Keeps the dashboard visually rich even though the source data only carries CPU and memory.

## API surface

Once `make serve` is running:

| Method | Path                                | Purpose                                              |
|--------|-------------------------------------|------------------------------------------------------|
| GET    | `/health`                           | Loaded models + machines + status                    |
| GET    | `/models`                           | List models with params + rebuilt config             |
| GET    | `/models/compare`                   | Overall metrics side-by-side                         |
| GET    | `/models/{name}/metrics`            | Full metrics + training history                      |
| GET    | `/machines`                         | List with latest cpu / memory / temp / power / jobs  |
| GET    | `/machines/workload`                | Cluster-wide workload distribution (pie chart)       |
| GET    | `/machines/{id}/history?steps=N`    | Denormalized recent window                           |
| GET    | `/machines/{id}/forecast`           | Latest window → forecast, denormalized               |
| POST   | `/predict`                          | `{window, model?}` → forecast                        |
| POST   | `/predict/batch`                    | `{windows[], model?}` → forecasts                    |

## Deploying to Render

The whole stack — FastAPI backend, trained LSTM, and the built React dashboard — deploys to a single Render web service via the included `Dockerfile` and `render.yaml`. Backend and frontend share the same origin, so there's no CORS to configure.

**One-time setup:**

1. Sign in to [render.com](https://render.com) and connect your GitHub account.
2. Click **New → Blueprint**, point it at this repo, pick a branch (usually `main`), and confirm. Render reads `render.yaml`, spins up the Docker build, and gives you a URL like `https://gpu-cluster-forecaster.onrender.com`.

**What happens during the build (5-10 min on the free tier):**

- Stage 1: `node:20-alpine` runs `npm ci` + `npm run build` — Vite bundles the React dashboard with `VITE_API_BASE_URL=""` (same-origin).
- Stage 2: `python:3.11-slim` installs the CPU-only PyTorch wheel + the rest of `requirements.txt`, copies the built frontend into `frontend/dist/`, then runs `download_data.py --synthetic --process` and `train.py` to bake a trained LSTM + preprocessed splits + scalers into the image.
- Startup: `uvicorn` binds to `$PORT`, FastAPI mounts `frontend/dist/` at `/`, and `/health` is Render's health check.

**Free-tier caveats:**

- The service sleeps after 15 min of inactivity. First request after sleep takes ~30 s to warm up.
- 512 MB RAM. The LSTM at default size is small (~500 KB); the runtime is comfortably under the limit.
- Every push to `main` auto-deploys. To change that, edit `autoDeploy: true` in `render.yaml`.

**Local Docker test:**

```bash
docker build -t gpu-cluster-forecaster .
docker run -p 8000:8000 gpu-cluster-forecaster
# → http://localhost:8000 serves the dashboard, /docs shows the API
```

## Contributing / development

Common tasks are wired into `backend/Makefile`:

```bash
make format    # black + ruff --fix
make lint      # black --check + ruff (no fix)
make test      # pytest with coverage
make train     # scripts/train.py --model lstm --data synthetic
make serve     # uvicorn app.main:app --reload
```

The GitHub Actions workflows (`.github/workflows/`) run lint + tests on every PR touching `backend/**`, and lint + build on every PR touching `frontend/**`.
