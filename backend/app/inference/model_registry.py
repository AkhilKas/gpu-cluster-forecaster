"""
Loads trained model checkpoints at startup and serves them by name.

Checkpoint layout (see BaseForecaster.save):
    {"model_state_dict": ..., "model_name": str, "config": dict}

The `model_name` prefix decides which class rebuilds the model.
"""

import json
import logging
from pathlib import Path

import torch

from app.models.base import BaseForecaster
from app.models.baseline import LinearBaseline, MovingAverageBaseline
from app.models.lstm import LSTMForecaster, LSTMSeq2Seq

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when a checkpoint cannot be rebuilt into a model."""


def _rebuild(model_name: str, config: dict) -> BaseForecaster:
    """Instantiate the correct model class from a checkpoint's model_name + config."""
    if model_name.startswith("lstm_seq2seq"):
        return LSTMSeq2Seq(**config)
    if model_name.startswith("lstm"):
        return LSTMForecaster(**config)
    if model_name.startswith("linear_baseline"):
        return LinearBaseline(**config)
    if model_name.startswith("moving_avg"):
        return MovingAverageBaseline(**config)
    raise ModelLoadError(f"Unknown model_name prefix: {model_name!r}")


class ModelRegistry:
    """
    Scans a weights directory for `*_best.pt` files and keeps the loaded
    models in memory keyed by model_name.

    Also indexes companion files: `<name>_metrics.json`, `<name>_history.json`.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models: dict[str, BaseForecaster] = {}
        self.metrics_paths: dict[str, Path] = {}
        self.history_paths: dict[str, Path] = {}

    def load_all(self, weights_dir: Path) -> None:
        """Load every `*_best.pt` in `weights_dir`. Skips files that fail to rebuild."""
        weights_dir = Path(weights_dir)
        if not weights_dir.exists():
            logger.warning(f"Weights dir does not exist: {weights_dir}")
            return

        checkpoints = sorted(weights_dir.glob("*_best.pt"))
        if not checkpoints:
            logger.warning(f"No `*_best.pt` files found in {weights_dir}")
            return

        for ckpt_path in checkpoints:
            try:
                checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )
                model_name = checkpoint["model_name"]
                config = checkpoint.get("config", {})
                if not config:
                    logger.warning(
                        f"Checkpoint {ckpt_path.name} has empty config; "
                        "cannot rebuild. Skipping."
                    )
                    continue
                model = _rebuild(model_name, config)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                model.to(self.device)
                self.models[model_name] = model
                logger.info(
                    f"Loaded model '{model_name}' from {ckpt_path.name} "
                    f"({model.count_parameters():,} params)"
                )
            except Exception as e:
                logger.error(f"Failed to load {ckpt_path.name}: {e}")

        # Index sidecar files
        for metrics_path in weights_dir.glob("*_metrics.json"):
            name = metrics_path.stem.removesuffix("_metrics")
            self.metrics_paths[name] = metrics_path
        for history_path in weights_dir.glob("*_history.json"):
            name = history_path.stem.removesuffix("_history")
            self.history_paths[name] = history_path

    def get(self, name: str | None = None) -> BaseForecaster:
        """Fetch a model by name. If name is None, returns the default (first loaded)."""
        if not self.models:
            raise ModelLoadError("No models loaded.")
        if name is None:
            return next(iter(self.models.values()))
        if name not in self.models:
            raise ModelLoadError(f"Model {name!r} not found.")
        return self.models[name]

    def default_name(self) -> str | None:
        """Name of the model returned by `get(None)`."""
        return next(iter(self.models), None)

    def known_names(self) -> set[str]:
        """All model names — loaded ones and those with a metrics sidecar only."""
        return set(self.models) | set(self.metrics_paths)

    def load_metrics(self, name: str) -> dict | None:
        """Read the metrics JSON for a model, if present."""
        path = self.metrics_paths.get(name)
        if path is None:
            return None
        with open(path) as f:
            return json.load(f)

    def load_history(self, name: str) -> dict | None:
        """Read the training-history JSON for a model, if present."""
        path = self.history_paths.get(name)
        if path is None:
            return None
        with open(path) as f:
            return json.load(f)
