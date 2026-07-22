"""
Prediction service: wraps a ModelRegistry + DataStore and handles the
normalize/inverse-transform plumbing so callers can work in real units.
"""

import logging

import numpy as np
import torch

from app.config import DataConfig

from .data_store import DataStore
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class Predictor:
    """
    High-level inference service.

    - `predict` / `predict_batch` are pure model calls (normalized in, normalized out).
    - `forecast_from_machine` pulls the latest observed window from DataStore,
      runs the model, and denormalizes both history and forecast for display.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        data_store: DataStore,
        data_config: DataConfig | None = None,
    ):
        self.registry = registry
        self.data_store = data_store
        self.config = data_config or DataConfig()

    def predict(self, window: np.ndarray, model_name: str | None = None) -> np.ndarray:
        """
        Run one forecast.

        Args:
            window: (seq_len, num_features) normalized input.
            model_name: optional model to use. Defaults to the first loaded.

        Returns:
            (horizon, num_targets) normalized forecast.
        """
        model = self.registry.get(model_name)
        x = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        return out.squeeze(0).cpu().numpy()

    def predict_batch(
        self, windows: np.ndarray, model_name: str | None = None
    ) -> np.ndarray:
        """
        Batched forecast.

        Args:
            windows: (batch, seq_len, num_features).

        Returns:
            (batch, horizon, num_targets)
        """
        model = self.registry.get(model_name)
        x = torch.as_tensor(windows, dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
        return out.cpu().numpy()

    def _all_cols(self) -> list[str]:
        """Feature+target column order used at fit time (see preprocessor.process_machine)."""
        return list(
            dict.fromkeys(self.config.feature_columns + self.config.target_columns)
        )

    def _denormalize_features(
        self, machine_id: str, normalized: np.ndarray
    ) -> np.ndarray:
        """Denormalize a (seq_len, num_features) window using the machine's scaler."""
        scaler = self.data_store.get_scaler(machine_id)
        if scaler is None:
            logger.warning(
                f"No scaler for machine {machine_id!r}; returning normalized values."
            )
            return normalized
        return scaler.inverse_transform(normalized)

    def _denormalize_targets(
        self, machine_id: str, normalized: np.ndarray, target_columns: list[str]
    ) -> np.ndarray:
        """
        Denormalize a (horizon, num_targets) forecast.

        Reconstructs the dummy full-feature array the scaler expects so we can
        map target-column outputs back to real units.
        """
        scaler = self.data_store.get_scaler(machine_id)
        if scaler is None:
            logger.warning(
                f"No scaler for machine {machine_id!r}; returning normalized values."
            )
            return normalized

        all_cols = self._all_cols()
        n_features = len(scaler.scale_)
        dummy = np.zeros((len(normalized), n_features))
        for i, col in enumerate(target_columns):
            if col in all_cols:
                dummy[:, all_cols.index(col)] = normalized[:, i]
        inversed = scaler.inverse_transform(dummy)
        target_indices = [all_cols.index(c) for c in target_columns if c in all_cols]
        return inversed[:, target_indices]

    def forecast_from_machine(
        self, machine_id: str, model_name: str | None = None
    ) -> dict:
        """
        Full-service: pull latest window for a machine, predict, denormalize both
        history and forecast for the dashboard.

        Returns a dict shaped like ForecastResponse.
        """
        model = self.registry.get(model_name)
        resolved_name = next(
            (n for n, m in self.registry.models.items() if m is model), "unknown"
        )

        window = self.data_store.get_latest_window(
            machine_id
        )  # (seq_len, num_features)
        forecast_norm = self.predict(window, model_name)  # (horizon, num_targets)

        history_real = self._denormalize_features(machine_id, window)
        feature_cols = self.config.feature_columns
        target_cols = self.config.target_columns
        forecast_real = self._denormalize_targets(
            machine_id, forecast_norm, target_cols
        )

        history = [
            {
                "step": i,
                "values": {
                    col: float(history_real[i, j])
                    for j, col in enumerate(feature_cols[: history_real.shape[1]])
                },
            }
            for i in range(history_real.shape[0])
        ]
        forecast = [
            {
                "step": i,
                "values": {
                    col: float(forecast_real[i, j])
                    for j, col in enumerate(target_cols[: forecast_real.shape[1]])
                },
            }
            for i in range(forecast_real.shape[0])
        ]

        return {
            "machine_id": machine_id,
            "model": resolved_name,
            "horizon": forecast_real.shape[0],
            "target_columns": target_cols,
            "history": history,
            "forecast": forecast,
        }
