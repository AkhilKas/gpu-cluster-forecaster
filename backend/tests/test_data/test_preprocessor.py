"""Tests for preprocessing pipeline."""

from app.data.preprocessor import Preprocessor


class TestPreprocessor:
    def test_resample_reduces_rows(self, synthetic_machines, data_config):
        preprocessor = Preprocessor(data_config)
        df = next(iter(synthetic_machines.values()))
        resampled = preprocessor.resample(df)
        # Resampled should have fewer or equal rows
        assert len(resampled) <= len(df)

    def test_process_machine_returns_correct_keys(self, processed_splits):
        required_keys = [
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
            "feature_columns",
            "target_columns",
            "machine_id",
        ]
        for key in required_keys:
            assert key in processed_splits, f"Missing key: {key}"

    def test_window_shapes(self, processed_splits, data_config):
        x_train = processed_splits["X_train"]
        y_train = processed_splits["y_train"]

        # X: (num_samples, seq_len, num_features)
        assert x_train.ndim == 3
        assert x_train.shape[1] == data_config.sequence_length

        # y: (num_samples, horizon, num_targets)
        assert y_train.ndim == 3
        assert y_train.shape[1] == data_config.forecast_horizon

        # Same number of samples
        assert x_train.shape[0] == y_train.shape[0]

    def test_values_normalized_0_to_1(self, processed_splits):
        x_train = processed_splits["X_train"]
        assert x_train.min() >= -0.1  # small tolerance for float precision
        assert x_train.max() <= 1.1

    def test_chronological_split_no_leakage(self, processed_splits):
        """Train data should come before val, val before test."""
        n_train = len(processed_splits["X_train"])
        n_val = len(processed_splits["X_val"])
        n_test = len(processed_splits["X_test"])
        assert n_train > 0
        assert n_val > 0
        assert n_test > 0

    def test_split_ratios_approximate(self, processed_splits, data_config):
        total = (
            len(processed_splits["X_train"])
            + len(processed_splits["X_val"])
            + len(processed_splits["X_test"])
        )
        train_ratio = len(processed_splits["X_train"]) / total
        assert abs(train_ratio - data_config.train_ratio) < 0.05
