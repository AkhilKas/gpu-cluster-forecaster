"""
LSTM Forecaster for GPU utilization prediction.

Architecture:
    Input → LSTM (2-layer, 128 hidden) → Dropout → FC → Output

    Input:  (batch, seq_len=60, input_dim=4)
    Output: (batch, horizon=12, output_dim=2)
"""
import torch
import torch.nn as nn

from .base import BaseForecaster


class LSTMForecaster(BaseForecaster):
    """
    Multi-step LSTM forecaster with configurable architecture.

    Uses the final hidden state to predict the full forecast horizon
    in one shot via a fully-connected projection layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Layer norm on LSTM output for training stability
        lstm_output_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # Dropout before projection
        self.dropout = nn.Dropout(dropout)

        # Project last hidden state → full forecast horizon
        # Output shape: (batch, horizon * output_dim) → reshaped to (batch, horizon, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * output_dim),
        )

    @property
    def model_name(self) -> str:
        bi = "_bi" if self.bidirectional else ""
        return f"lstm{bi}_{self.num_layers}x{self.hidden_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, horizon, output_dim)
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last timestep output (captures full sequence context)
        last_out = lstm_out[:, -1, :]           # (batch, hidden * directions)

        # Normalize + dropout
        last_out = self.layer_norm(last_out)
        last_out = self.dropout(last_out)

        # Project to forecast
        forecast = self.fc(last_out)            # (batch, horizon * output_dim)

        # Reshape to (batch, horizon, output_dim)
        forecast = forecast.view(-1, self.horizon, self.output_dim)

        return forecast

    def _get_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
        }


class LSTMSeq2Seq(BaseForecaster):
    """
    Alternative: Encoder-Decoder LSTM for longer horizons.
    Decoder autoregressively generates each future step.

    Better for longer forecast horizons (30-60 min).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def model_name(self) -> str:
        return f"lstm_seq2seq_{self.num_layers}x{self.hidden_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, horizon, output_dim)
        """
        batch_size = x.size(0)

        # Encode input sequence
        _, (h, c) = self.encoder(x)

        # Decoder: start with zeros, autoregressively generate
        decoder_input = torch.zeros(
            batch_size, 1, self.output_dim, device=x.device
        )
        outputs = []

        for t in range(self.horizon):
            decoder_out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.output_proj(self.dropout(decoder_out))
            outputs.append(pred)
            decoder_input = pred  # Feed prediction as next input

        return torch.cat(outputs, dim=1)  # (batch, horizon, output_dim)

    def _get_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }