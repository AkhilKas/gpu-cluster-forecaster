"""Tests for training loop."""

import numpy as np

from app.models.lstm import LSTMForecaster
from app.training.evaluator import Evaluator
from app.training.trainer import Trainer


class TestTrainer:
    def test_train_runs_without_error(self, dataloaders, train_config):
        model = LSTMForecaster(
            input_dim=dataloaders["meta"]["input_dim"],
            output_dim=dataloaders["meta"]["output_dim"],
            horizon=dataloaders["meta"]["horizon"],
            hidden_size=train_config.hidden_size,
            num_layers=train_config.num_layers,
            dropout=train_config.dropout,
        )
        trainer = Trainer(model, train_config)
        history = trainer.train(dataloaders["train"], dataloaders["val"], epochs=2)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_evaluate_returns_metrics(self, dataloaders, train_config):
        model = LSTMForecaster(
            input_dim=dataloaders["meta"]["input_dim"],
            output_dim=dataloaders["meta"]["output_dim"],
            horizon=dataloaders["meta"]["horizon"],
            hidden_size=train_config.hidden_size,
            num_layers=train_config.num_layers,
            dropout=train_config.dropout,
        )
        trainer = Trainer(model, train_config)
        trainer.train(dataloaders["train"], dataloaders["val"], epochs=2)
        metrics = trainer.evaluate(dataloaders["test"])
        assert "overall" in metrics
        assert "mae" in metrics["overall"]
        assert metrics["overall"]["mae"] >= 0

    def test_loss_decreases(self, dataloaders, train_config):
        """Loss should generally decrease over a few epochs."""
        train_config.epochs = 10
        model = LSTMForecaster(
            input_dim=dataloaders["meta"]["input_dim"],
            output_dim=dataloaders["meta"]["output_dim"],
            horizon=dataloaders["meta"]["horizon"],
            hidden_size=train_config.hidden_size,
            num_layers=train_config.num_layers,
            dropout=train_config.dropout,
        )
        trainer = Trainer(model, train_config)
        history = trainer.train(dataloaders["train"], dataloaders["val"], epochs=10)
        # Last loss should be lower than first
        assert history["train_loss"][-1] < history["train_loss"][0]


class TestEvaluator:
    def test_mae_zero_for_perfect_predictions(self):
        y = np.random.rand(100, 6, 2)
        assert Evaluator.mae(y, y) == 0.0

    def test_rmse_zero_for_perfect_predictions(self):
        y = np.random.rand(100, 6, 2)
        assert Evaluator.rmse(y, y) == 0.0

    def test_overload_accuracy_perfect(self):
        y = np.array([0.5, 0.9, 0.3, 0.85, 0.1])
        result = Evaluator.overload_accuracy(y, y, threshold=0.8)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
