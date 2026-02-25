"""
Evaluation metrics for time series forecasting.
Computes both regression metrics and operational metrics.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Compute forecast evaluation metrics.

    Regression:  MAE, RMSE, MAPE
    Operational: Overload prediction accuracy, scheduling improvement
    """

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        denom = np.maximum(np.abs(y_true), epsilon)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    @staticmethod
    def overload_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.8,
    ) -> dict[str, float]:
        """
        Accuracy of predicting overload events (utilization > threshold).

        Returns precision, recall, f1, and accuracy.
        """
        true_overload = (y_true > threshold).astype(int)
        pred_overload = (y_pred > threshold).astype(int)

        tp = np.sum((pred_overload == 1) & (true_overload == 1))
        fp = np.sum((pred_overload == 1) & (true_overload == 0))
        fn = np.sum((pred_overload == 0) & (true_overload == 1))
        tn = np.sum((pred_overload == 0) & (true_overload == 0))

        total = tp + fp + fn + tn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(total, 1)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "total_overload_events": int(np.sum(true_overload)),
            "predicted_overload_events": int(np.sum(pred_overload)),
        }

    @staticmethod
    def per_horizon_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> list:
        """
        Compute MAE/RMSE for each forecast step.
        Shows how accuracy degrades with longer horizons.

        Args:
            y_true: (num_samples, horizon, output_dim)
            y_pred: (num_samples, horizon, output_dim)

        Returns:
            List of dicts, one per horizon step.
        """
        horizon = y_true.shape[1]
        results = []
        for t in range(horizon):
            yt = y_true[:, t, :]
            yp = y_pred[:, t, :]
            results.append(
                {
                    "step": t + 1,
                    "mae": round(float(np.mean(np.abs(yt - yp))), 4),
                    "rmse": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
                }
            )
        return results

    @classmethod
    def evaluate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list | None = None,
    ) -> dict:
        """
        Run full evaluation suite.

        Args:
            y_true: (num_samples, horizon, output_dim)
            y_pred: (num_samples, horizon, output_dim)
            target_names: Names for each output dimension

        Returns:
            Comprehensive metrics dict.
        """
        if target_names is None:
            target_names = [f"target_{i}" for i in range(y_true.shape[-1])]

        results = {
            "overall": {
                "mae": cls.mae(y_true, y_pred),
                "rmse": cls.rmse(y_true, y_pred),
                "mape": cls.mape(y_true, y_pred),
            },
            "per_target": {},
            "per_horizon": cls.per_horizon_metrics(y_true, y_pred),
            "overload": {},
        }

        # Per-target metrics
        for i, name in enumerate(target_names):
            yt = y_true[:, :, i]
            yp = y_pred[:, :, i]
            results["per_target"][name] = {
                "mae": cls.mae(yt, yp),
                "rmse": cls.rmse(yt, yp),
                "mape": cls.mape(yt, yp),
            }
            # Overload detection for utilization targets
            if "cpu" in name or "util" in name:
                results["overload"][name] = cls.overload_accuracy(
                    yt.flatten(), yp.flatten()
                )

        # Log summary
        logger.info(
            f"Evaluation -> MAE: {results['overall']['mae']:.4f}, "
            f"RMSE: {results['overall']['rmse']:.4f}, "
            f"MAPE: {results['overall']['mape']:.2f}%"
        )
        return results
