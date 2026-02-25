"""
Abstract base class for all forecasting models.
Ensures consistent interface across LSTM, Transformer, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseForecaster(ABC, nn.Module):
    """
    All forecasters must implement:
        - forward(x) -> predictions
        - predict(x) -> numpy predictions (no grad)
        - model_name -> string identifier
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for this model."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: (batch, horizon, output_dim)
        """
        ...

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference without gradient tracking."""
        self.eval()
        return self.forward(x)

    def save(self, path: Path):
        """Save model weights + config."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_name": self.model_name,
                "config": self._get_config(),
            },
            path,
        )

    def load(self, path: Path, device: str = "cpu"):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _get_config(self) -> dict:
        """Override to save model hyperparameters."""
        return {}

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
