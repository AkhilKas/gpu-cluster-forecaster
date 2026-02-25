"""
Training callbacks: early stopping, checkpointing, logging.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement. Best val loss: {self.best_loss:.6f}"
                )
        return self.should_stop


class ModelCheckpoint:
    """Save model when validation loss improves."""

    def __init__(self, save_dir: Path, model_name: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_loss = float("inf")
        self.best_path = None

    def step(self, val_loss: float, model, epoch: int) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_path = self.save_dir / f"{self.model_name}_best.pt"
            model.save(self.best_path)
            logger.info(
                f"Checkpoint saved: {self.best_path.name} "
                f"(val_loss={val_loss:.6f}, epoch={epoch})"
            )
            return True
        return False

    def load_best(self, model, device: str = "cpu"):
        """Load the best saved checkpoint."""
        if self.best_path and self.best_path.exists():
            model.load(self.best_path, device=device)
            logger.info(f"Loaded best model from {self.best_path.name}")
        else:
            logger.warning("No checkpoint found to load.")
