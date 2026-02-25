"""
PyTorch Dataset and DataLoader creation for GPU forecast training.
"""
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.config import TrainConfig

logger = logging.getLogger(__name__)


class GPUForecastDataset(Dataset):
    """
    PyTorch Dataset wrapping windowed time series arrays.

    Args:
        x_data: Input sequences (num_samples, seq_len, num_features)
        y_data: Target sequences (num_samples, forecast_horizon, num_targets)
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        assert len(x_data) == len(y_data), (
            f"X/y length mismatch: {len(x_data)} vs {len(y_data)}"
        )
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]

    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return self.x_data.shape[-1]

    @property
    def output_dim(self) -> int:
        """Number of target features."""
        return self.y_data.shape[-1]

    @property
    def seq_len(self) -> int:
        return self.x_data.shape[1]

    @property
    def horizon(self) -> int:
        return self.y_data.shape[1]


def create_dataloaders(
    splits: dict,
    train_config: TrainConfig | None = None,
) -> dict[str, DataLoader]:
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
        f"DataLoaders ready -> "
        f"train: {len(train_ds)} samples ({len(loaders['train'])} batches), "
        f"val: {len(val_ds)}, test: {len(test_ds)} | "
        f"input_dim={train_ds.input_dim}, output_dim={train_ds.output_dim}, "
        f"seq={train_ds.seq_len}->{train_ds.horizon}"
    )
    return loaders
