"""Shared test fixtures."""

import pytest

from app.config import DataConfig, TrainConfig
from app.data.dataset import create_dataloaders
from app.data.loader import ClusterDataLoader
from app.data.preprocessor import Preprocessor


@pytest.fixture
def data_config():
    """Small config for fast tests."""
    return DataConfig(
        synthetic_num_machines=2,
        synthetic_num_timesteps=500,
        sequence_length=30,
        forecast_horizon=6,
    )


@pytest.fixture
def train_config():
    return TrainConfig(
        batch_size=16,
        learning_rate=1e-3,
        epochs=3,
        patience=2,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        device="cpu",
    )


@pytest.fixture
def synthetic_machines(data_config):
    """Generate small synthetic dataset."""
    return ClusterDataLoader.generate_synthetic(data_config)


@pytest.fixture
def processed_splits(synthetic_machines, data_config):
    """Preprocessed splits for first machine."""
    preprocessor = Preprocessor(data_config)
    machine_id = sorted(synthetic_machines.keys())[0]
    return preprocessor.process_machine(synthetic_machines[machine_id], str(machine_id))


@pytest.fixture
def dataloaders(processed_splits, train_config):
    """Ready-to-use DataLoaders."""
    return create_dataloaders(processed_splits, train_config)
