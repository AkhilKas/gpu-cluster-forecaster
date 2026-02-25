"""Tests for LSTM model."""

import pytest
import torch

from app.models.lstm import LSTMForecaster, LSTMSeq2Seq


class TestLSTMForecaster:
    @pytest.fixture
    def model(self):
        return LSTMForecaster(
            input_dim=2,
            output_dim=2,
            horizon=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
        )

    def test_output_shape(self, model):
        x = torch.randn(8, 30, 2)  # (batch, seq_len, input_dim)
        out = model(x)
        assert out.shape == (8, 6, 2)  # (batch, horizon, output_dim)

    def test_predict_no_grad(self, model):
        x = torch.randn(4, 30, 2)
        out = model.predict(x)
        assert out.shape == (4, 6, 2)

    def test_model_name(self, model):
        assert "lstm" in model.model_name

    def test_parameter_count_positive(self, model):
        assert model.count_parameters() > 0

    def test_save_and_load(self, model, tmp_path):
        path = tmp_path / "test_model.pt"
        model.save(path)
        assert path.exists()

        new_model = LSTMForecaster(
            input_dim=2,
            output_dim=2,
            horizon=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
        )
        new_model.load(path)

        # Weights should match
        x = torch.randn(4, 30, 2)
        torch.testing.assert_close(model.predict(x), new_model.predict(x))

    def test_batch_size_one(self, model):
        x = torch.randn(1, 30, 2)
        out = model(x)
        assert out.shape == (1, 6, 2)


class TestLSTMSeq2Seq:
    @pytest.fixture
    def model(self):
        return LSTMSeq2Seq(
            input_dim=2,
            output_dim=2,
            horizon=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
        )

    def test_output_shape(self, model):
        x = torch.randn(8, 30, 2)
        out = model(x)
        assert out.shape == (8, 6, 2)

    def test_model_name(self, model):
        assert "seq2seq" in model.model_name
