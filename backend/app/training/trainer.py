"""
Training loop with logging, early stopping, and checkpointing.
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.config import WEIGHTS_DIR, TrainConfig
from app.models.base import BaseForecaster

from .callbacks import EarlyStopping, ModelCheckpoint
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the full training lifecycle:
        1. Training loop with gradient clipping
        2. Validation each epoch
        3. Early stopping
        4. Best model checkpointing
        5. Final test evaluation
    """

    def __init__(
        self,
        model: BaseForecaster,
        config: TrainConfig | None = None,
        device: str | None = None,
    ):
        self.config = config or TrainConfig()
        self.device = self._resolve_device(device or self.config.device)
        self.model = model.to(self.device)

        # Loss & optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Callbacks
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self.checkpoint = ModelCheckpoint(WEIGHTS_DIR, model.model_name)

        # History for plotting
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        logger.info(
            f"Trainer initialized: {model.model_name} | "
            f"{model.count_parameters():,} params | device={self.device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item() * len(x_batch)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            total_loss += loss.item() * len(x_batch)

        return total_loss / len(loader.dataset)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
    ) -> dict:
        """
        Full training loop.

        Returns:
            Training history dict with losses per epoch.
        """
        epochs = epochs or self.config.epochs
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_loss = self._validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Logging
            if epoch % 5 == 0 or epoch <= 3:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss: {train_loss:.6f} | "
                    f"val_loss: {val_loss:.6f} | "
                    f"lr: {current_lr:.2e} | "
                    f"time: {epoch_time:.1f}s"
                )

            # Checkpoint best model
            self.checkpoint.step(val_loss, self.model, epoch)

            # Early stopping check
            if self.early_stopping.step(val_loss):
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training complete in {total_time:.1f}s ({total_time/60:.1f}min) | "
            f"Best val_loss: {self.checkpoint.best_loss:.6f}"
        )

        # Load best model for evaluation
        self.checkpoint.load_best(self.model, device=str(self.device))

        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        target_names: list | None = None,
    ) -> dict:
        """
        Run full evaluation on test set.

        Returns:
            Comprehensive metrics dict from Evaluator.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(self.device)
            preds = self.model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        return Evaluator.evaluate_all(y_true, y_pred, target_names)

    @torch.no_grad()
    def predict(self, x_input: torch.Tensor) -> np.ndarray:
        """Run inference on raw input tensor."""
        self.model.eval()
        x_input = x_input.to(self.device)
        return self.model(x_input).cpu().numpy()
