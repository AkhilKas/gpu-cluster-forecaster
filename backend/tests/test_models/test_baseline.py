"""Tests for baseline models."""

import torch

from app.models.baseline import LinearBaseline, MovingAverageBaseline


class TestLinearBaseline:
    def test_output_shape(self):
        model = LinearBaseline(input_dim=2, output_dim=2, seq_len=30, horizon=6)
        x = torch.randn(8, 30, 2)
        out = model(x)
        assert out.shape == (8, 6, 2)


class TestMovingAverageBaseline:
    def test_output_shape(self):
        model = MovingAverageBaseline(output_dim=2, horizon=6, window=12)
        x = torch.randn(8, 30, 2)
        out = model(x)
        assert out.shape == (8, 6, 2)

    def test_no_trainable_params(self):
        model = MovingAverageBaseline(output_dim=2, horizon=6)
        assert model.count_parameters() == 0
