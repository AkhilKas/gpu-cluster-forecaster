"""
PyTorch Dataset and DataLoader creation for GPU forecast training.
"""
import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from app.config import TrainConfig

logger = logging.getLogger(__name__)


class GPUForecastDataset(Dataset):
    """
    PyTorch Dataset wrapping windowed time series arrays.

    Args:
        X: Input sequences (num_samples, seq_len, num_features)
        y: Target sequences (num_samples, forecast_horizon, num_targets)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return self.X.shape[-1]

    @property
    def output_dim(self) -> int:
        """Number of target features."""
        return self.y.shape[-1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[1]

    @property
    def horizon(self) -> int:
        return self.y.shape[1]


def create_dataloaders(
    splits: dict,
    train_config: TrainConfig = None,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from preprocessed splits.

    Args:
        splits: Dict with X_train, y_train, X_val, y_val, X_test, y_test
        train_config: Training configuration

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders and metadata
    """
    config = train_config or TrainConfig()

    train_ds = GPUForecastDataset(splits["X_train"], splits["y_train"])
    val_ds = GPUForecastDataset(splits["X_val"], splits["y_val"])
    test_ds = GPUForecastDataset(splits["X_test"], splits["y_test"])

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,           # OK to shuffle windows (not raw timesteps)
            num_workers=0,           # Set >0 for faster loading on multi-core
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        # Metadata for model initialization
        "meta": {
            "input_dim": train_ds.input_dim,
            "output_dim": train_ds.output_dim,
            "seq_len": train_ds.seq_len,
            "horizon": train_ds.horizon,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "feature_columns": splits.get("feature_columns", []),
            "target_columns": splits.get("target_columns", []),
        },
    }

    logger.info(
        f"DataLoaders ready → "
        f"train: {len(train_ds)} samples ({len(loaders['train'])} batches), "
        f"val: {len(val_ds)}, test: {len(test_ds)} | "
        f"input_dim={train_ds.input_dim}, output_dim={train_ds.output_dim}, "
        f"seq={train_ds.seq_len}→{train_ds.horizon}"
    )
    return loaders