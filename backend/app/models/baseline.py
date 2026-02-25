"""
Baseline models for comparison.
Every ML project needs baselines to prove the DL model adds value.
"""
import torch
import torch.nn as nn

from .base import BaseForecaster


class LinearBaseline(BaseForecaster):
    """Simple linear projection from last seq_len steps to horizon."""

    def __init__(self, input_dim: int, output_dim: int, seq_len: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.fc = nn.Linear(seq_len * input_dim, horizon * output_dim)

    @property
    def model_name(self) -> str:
        return "linear_baseline"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        flat = x.view(batch, -1)
        out = self.fc(flat)
        return out.view(batch, self.horizon, self.output_dim)


class MovingAverageBaseline(BaseForecaster):
    """
    Predicts future = mean of last N steps.
    No trainable params — pure heuristic baseline.
    """

    def __init__(self, output_dim: int, horizon: int, window: int = 12):
        super().__init__()
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        # Dummy param so PyTorch treats this as a module
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def model_name(self) -> str:
        return f"moving_avg_{self.window}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average last `window` steps, repeat for entire horizon
        avg = x[:, -self.window:, :self.output_dim].mean(dim=1, keepdim=True)
        return avg.expand(-1, self.horizon, -1)