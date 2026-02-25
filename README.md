# GPU Cluster Forecaster

Deep learning system that forecasts GPU utilization and memory usage in ML clusters, enabling proactive scheduling and reduced idle time.

## Tech Stack
- **Backend**: Python, PyTorch, FastAPI
- **Frontend**: React, Recharts, Vite
- **Data**: Google Cluster Data 2019
- **Models**: LSTM (Phase 1), Transformer/PatchTST (Phase 2)

## Quick Start
```bash
cd backend
pip install -r requirements.txt

# Generate synthetic data and preprocess
python scripts/download_data.py --synthetic --process

# Use real Google Cluster Data
python scripts/download_data.py --shards 3 --process
```

## Project Status
- [x] Data pipeline (download, preprocess, windowing)
- [ ] LSTM model
- [ ] Transformer model
- [ ] FastAPI backend
- [ ] React dashboard
- [ ] Deployment (Render + Vercel)
