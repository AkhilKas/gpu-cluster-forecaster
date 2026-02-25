from pathlib import Path
from dataclasses import dataclass, field
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
DATA_RAW = ROOT_DIR.parent / "data" / "raw"
DATA_PROCESSED = ROOT_DIR.parent / "data" / "processed"
WEIGHTS_DIR = ROOT_DIR / "weights"

# Ensure dirs exist
for d in [DATA_RAW, DATA_PROCESSED, WEIGHTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    # Google Cluster Data 2019 settings
    gcs_bucket: str = "clusterdata_2019_a"
    tables: List[str] = field(default_factory=lambda: [
        "instance_usage",
        "instance_events",
        "machine_events",
        "collection_events",
    ])
    # How many shards to download per table (full dataset is huge)
    # Each table has ~500 shards; start small for dev
    max_shards_per_table: int = 3

    # Preprocessing
    resample_interval: str = "5min"       # Aggregate to 5-min windows
    sequence_length: int = 60             # 60 steps = 5 hours of history
    forecast_horizon: int = 12            # 12 steps = 1 hour ahead
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Features used for modeling
    feature_columns: List[str] = field(default_factory=lambda: [
        "cpu_usage",
        "memory_usage",
        "assigned_memory",
        "cycles_per_instruction",
    ])
    target_columns: List[str] = field(default_factory=lambda: [
        "cpu_usage",        # proxy for gpu_utilization
        "memory_usage",     # proxy for gpu_memory
    ])

    # Synthetic data (for dev/testing without downloading)
    synthetic_num_machines: int = 8
    synthetic_num_timesteps: int = 5000


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    patience: int = 10           # Early stopping patience
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    device: str = "auto"         # "auto", "cuda", "cpu"