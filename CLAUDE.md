# CLAUDE.md

Project context for AI coding assistants working on **gpu-cluster-forecaster**.

## What this project is

A deep learning system that forecasts GPU utilization and memory usage across ML clusters. The goal is to enable proactive scheduling and reduce idle GPU time. Given historical cluster telemetry, the model should predict near-future GPU load so a scheduler can place work more efficiently.

## Tech stack

- **Backend**: Python, PyTorch, FastAPI
- **Frontend**: React, Recharts, Vite
- **Data**: Google Cluster Data 2019 (with a synthetic-data fallback)
- **Models**: LSTM (Phase 1), then Transformer / PatchTST (Phase 2)

## Repo layout

```
.github/workflows   CI config
backend             Python: data pipeline, models, API (Python is ~75% of the repo)
data                datasets / processed data
docs                documentation
frontend            React dashboard (JavaScript is ~25% of the repo)
notebooks           exploration / experiments
README.md
```

## Data pipeline (the only part currently built)

A download/preprocess script lives under `backend/scripts/` and is driven from the CLI:

```bash
cd backend
pip install -r requirements.txt

# Synthetic data path
python scripts/download_data.py --synthetic --process

# Real Google Cluster Data path
python scripts/download_data.py --shards 3 --process
```

The pipeline covers download, preprocessing, and windowing (turning raw time series into fixed-length input windows for supervised training).

## Current status

- [x] Data pipeline (download, preprocess, windowing)
- [x] LSTM model (+ Seq2Seq variant, linear/moving-avg baselines, training loop, evaluator)
- [ ] Transformer model
- [ ] FastAPI backend
- [ ] React dashboard (a demo `App.jsx` mockup exists but has no build config and no API wiring)
- [ ] Deployment (Render for backend + Vercel for frontend)

## Suggested next step

Build the **FastAPI backend**. The trained LSTM in `backend/app/models/lstm.py` needs an inference endpoint before the frontend can be wired up. `backend/app/api/{routes,middleware,schemas}/` and `backend/app/inference/` already exist as empty scaffolding.

## Data contract (as implemented)

The pipeline is built, so these values are fixed:

- **Window shapes**: `X = (n, sequence_length=60, num_features=4)`, `y = (n, forecast_horizon=12, num_targets=2)`. Multi-step forecasting.
- **Features / targets**: features = `[cpu_usage, memory_usage, assigned_memory, cycles_per_instruction]`; targets = `[cpu_usage, memory_usage]`. CPU is used as a proxy for GPU utilization — there is no real GPU-specific data in the current pipeline.
- **Normalization**: per-machine `MinMaxScaler` fit on the training portion only, applied to all rows. Scalers are pickled to `data/processed/scalers.pkl` — any inference path must load and reuse them.
- **Split**: chronological 70/15/15, no shuffling of raw timesteps (windows are shuffled inside the train DataLoader only).
- **Processed data layout**: `data/processed/machine_{id}/{X,y}_{train,val,test}.npy`.
- **Config**: see `backend/app/config.py` (`DataConfig`, `TrainConfig` dataclasses). Logging goes through `app/utils/logger.setup_logging`.

## Important: verify before you build (Phase 2 / Transformer)

Before adding new models or an inference API, re-confirm anything above that could have drifted, and additionally:

- **Scaler round-trip**: `Preprocessor.inverse_transform` builds a dummy full-feature array — verify it still matches the target-column layout if you change features.
- **Model interface**: new models must subclass `BaseForecaster` (`app/models/base.py`) and implement `forward`, `model_name`, and `_get_config` so `Trainer` checkpointing keeps working.
- **Dependencies**: `requirements.txt` pins are unversioned lower bounds; check what's actually installed before adding anything (e.g. `torch>=2.0.0`).
