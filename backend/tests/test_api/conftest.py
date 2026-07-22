"""
Fixtures for API tests: builds a temp weights/processed directory with a
real (small) trained LSTM, then boots the FastAPI app against it.

Scoped 'module' so we train once per test file, not once per test.
"""

import json
import pickle

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.config import DataConfig, TrainConfig
from app.data.dataset import create_dataloaders
from app.data.loader import ClusterDataLoader
from app.data.preprocessor import Preprocessor
from app.main import create_app
from app.models.lstm import LSTMForecaster


@pytest.fixture(scope="module")
def api_data_config() -> DataConfig:
    """Small, fast config that still exercises the real code paths."""
    return DataConfig(
        synthetic_num_machines=2,
        synthetic_num_timesteps=500,
        sequence_length=30,
        forecast_horizon=6,
    )


@pytest.fixture(scope="module")
def api_train_config() -> TrainConfig:
    return TrainConfig(
        batch_size=16,
        learning_rate=1e-3,
        epochs=2,
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        device="cpu",
    )


@pytest.fixture(scope="module")
def api_env(tmp_path_factory, api_data_config, api_train_config):
    """
    Provision a temp environment: processed splits for two machines + scalers +
    a briefly-trained LSTM + metrics/history JSON. Returns paths + metadata.
    """
    root = tmp_path_factory.mktemp("api_env")
    weights_dir = root / "weights"
    processed_dir = root / "processed"
    weights_dir.mkdir()
    processed_dir.mkdir()

    # 1. Generate + preprocess synthetic data for two machines.
    machines = ClusterDataLoader.generate_synthetic(api_data_config)
    preprocessor = Preprocessor(api_data_config)

    machine_ids = sorted(machines.keys())
    first_splits = None
    for mid in machine_ids:
        splits = preprocessor.process_machine(machines[mid], str(mid))
        if first_splits is None:
            first_splits = splits
        machine_dir = processed_dir / f"machine_{mid}"
        machine_dir.mkdir()
        for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            np.save(machine_dir / f"{key}.npy", splits[key])

    with open(processed_dir / "scalers.pkl", "wb") as f:
        pickle.dump(preprocessor.scalers, f)

    # 2. Build + briefly train an LSTM on the first machine's data.
    loaders = create_dataloaders(first_splits, api_train_config)
    meta = loaders["meta"]
    model = LSTMForecaster(
        input_dim=meta["input_dim"],
        output_dim=meta["output_dim"],
        horizon=meta["horizon"],
        hidden_size=api_train_config.hidden_size,
        num_layers=api_train_config.num_layers,
        dropout=api_train_config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=api_train_config.learning_rate)
    criterion = torch.nn.MSELoss()
    for _ in range(api_train_config.epochs):
        model.train()
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # 3. Save checkpoint + sidecar JSON files.
    model.eval()
    checkpoint_path = weights_dir / f"{model.model_name}_best.pt"
    model.save(checkpoint_path)

    metrics = {
        "overall": {"mae": 0.1234, "rmse": 0.2345, "mape": 5.67},
        "per_target": {
            "cpu_usage": {"mae": 0.11, "rmse": 0.22, "mape": 5.0},
            "memory_usage": {"mae": 0.14, "rmse": 0.25, "mape": 6.3},
        },
        "per_horizon": [
            {"step": i + 1, "mae": 0.1 + i * 0.01, "rmse": 0.2 + i * 0.01}
            for i in range(meta["horizon"])
        ],
        "overload": {
            "cpu_usage": {
                "precision": 0.9,
                "recall": 0.85,
                "f1": 0.87,
                "accuracy": 0.92,
                "total_overload_events": 10,
                "predicted_overload_events": 11,
            }
        },
    }
    with open(weights_dir / f"{model.model_name}_metrics.json", "w") as f:
        json.dump(metrics, f)

    history = {
        "train_loss": [0.5, 0.3],
        "val_loss": [0.6, 0.35],
        "lr": [1e-3, 1e-3],
    }
    with open(weights_dir / f"{model.model_name}_history.json", "w") as f:
        json.dump(history, f)

    return {
        "weights_dir": weights_dir,
        "processed_dir": processed_dir,
        "data_config": api_data_config,
        "model_name": model.model_name,
        "machine_ids": [str(m) for m in machine_ids],
        "input_dim": meta["input_dim"],
        "output_dim": meta["output_dim"],
        "seq_len": meta["seq_len"],
        "horizon": meta["horizon"],
    }


@pytest.fixture(scope="module")
def client(api_env):
    """FastAPI TestClient bound to the temp environment. Runs lifespan on enter."""
    app = create_app(
        weights_dir=api_env["weights_dir"],
        processed_dir=api_env["processed_dir"],
        data_config=api_env["data_config"],
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def empty_client(tmp_path_factory, api_data_config):
    """A TestClient booted against empty weights/processed dirs — for 503/404 tests."""
    root = tmp_path_factory.mktemp("api_empty")
    (root / "weights").mkdir()
    (root / "processed").mkdir()
    app = create_app(
        weights_dir=root / "weights",
        processed_dir=root / "processed",
        data_config=api_data_config,
    )
    with TestClient(app) as c:
        yield c
