"""
Preprocessor: resample, normalize, window, and split time series
for model training.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.config import DataConfig, DATA_PROCESSED

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Transforms raw per-machine DataFrames into model-ready numpy arrays.

    Pipeline:
        1. Resample to fixed interval (5 min)
        2. Forward-fill gaps, drop remaining NaN
        3. Normalize features with MinMaxScaler (fit on train only)
        4. Create sliding windows (X, y) pairs
        5. Chronological train/val/test split
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.scalers: Dict[str, MinMaxScaler] = {}  # per-machine scalers
        self._is_fitted = False

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to fixed time intervals and interpolate gaps."""
        df = df.set_index("timestamp").sort_index()

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df[numeric_cols]

        # Resample to fixed interval
        df_resampled = df_numeric.resample(self.config.resample_interval).mean()

        # Forward fill small gaps (up to 3 missing intervals = 15 min)
        df_resampled = df_resampled.ffill(limit=3)

        # Drop any remaining NaN rows
        df_resampled = df_resampled.dropna()

        logger.debug(
            f"Resampled: {len(df)} → {len(df_resampled)} rows "
            f"({self.config.resample_interval} intervals)"
        )
        return df_resampled.reset_index()

    def normalize(
        self,
        df: pd.DataFrame,
        machine_id: str,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Min-Max normalize feature columns to [0, 1].

        Args:
            fit: If True, fit scaler on this data (use for train set only).
                 If False, use previously fitted scaler (for val/test).
        """
        feature_cols = [
            c for c in self.config.feature_columns if c in df.columns
        ]
        if not feature_cols:
            raise ValueError(
                f"No feature columns found. Available: {df.columns.tolist()}"
            )

        if fit:
            scaler = MinMaxScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers[machine_id] = scaler
            self._is_fitted = True
        else:
            if machine_id not in self.scalers:
                raise RuntimeError(
                    f"No fitted scaler for machine {machine_id}. "
                    "Call normalize with fit=True on training data first."
                )
            df[feature_cols] = self.scalers[machine_id].transform(df[feature_cols])

        return df

    def create_windows(
        self,
        data: np.ndarray,
        feature_indices: List[int],
        target_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for time series forecasting.

        Args:
            data: 2D array (timesteps, features)
            feature_indices: Column indices used as input features
            target_indices: Column indices to predict

        Returns:
            X: (num_windows, seq_len, num_features)
            y: (num_windows, forecast_horizon, num_targets)
        """
        seq_len = self.config.sequence_length
        horizon = self.config.forecast_horizon
        total_len = seq_len + horizon

        if len(data) < total_len:
            raise ValueError(
                f"Data length ({len(data)}) < required ({total_len}). "
                "Need more data or shorter sequence/horizon."
            )

        n_windows = len(data) - total_len + 1
        X = np.zeros((n_windows, seq_len, len(feature_indices)))
        y = np.zeros((n_windows, horizon, len(target_indices)))

        for i in range(n_windows):
            X[i] = data[i : i + seq_len][:, feature_indices]
            y[i] = data[i + seq_len : i + total_len][:, target_indices]

        logger.debug(f"Windows created: X={X.shape}, y={y.shape}")
        return X, y

    def split_chronological(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Split data chronologically (NEVER shuffle time series).

        Returns:
            dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
        """
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        splits = {
            "X_train": X[:train_end],
            "y_train": y[:train_end],
            "X_val": X[train_end:val_end],
            "y_val": y[train_end:val_end],
            "X_test": X[val_end:],
            "y_test": y[val_end:],
        }

        logger.info(
            f"Split sizes → train: {len(splits['X_train'])}, "
            f"val: {len(splits['X_val'])}, test: {len(splits['X_test'])}"
        )
        return splits

    def process_machine(
        self,
        df: pd.DataFrame,
        machine_id: str,
    ) -> dict:
        """
        Full pipeline for a single machine: resample → normalize → window → split.

        Returns:
            dict with train/val/test arrays ready for PyTorch DataLoader
        """
        logger.info(f"Processing machine {machine_id}...")

        # Step 1: Resample
        df = self.resample(df)

        # Step 2: Identify feature/target columns present in data
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        target_cols = [c for c in self.config.target_columns if c in df.columns]
        all_cols = list(dict.fromkeys(feature_cols + target_cols))  # unique, ordered

        if not feature_cols or not target_cols:
            raise ValueError(
                f"Missing columns. Features: {feature_cols}, Targets: {target_cols}"
            )

        # Step 3: Normalize (fit on full data first, then refit on train only)
        # For proper practice: split first, then fit scaler on train only
        numeric_data = df[all_cols].values
        n = len(numeric_data)
        train_end = int(n * self.config.train_ratio)

        # Fit scaler ONLY on training portion
        scaler = MinMaxScaler()
        scaler.fit(numeric_data[:train_end])
        self.scalers[machine_id] = scaler
        self._is_fitted = True

        normalized = scaler.transform(numeric_data)

        # Step 4: Create sliding windows
        feature_indices = [all_cols.index(c) for c in feature_cols]
        target_indices = [all_cols.index(c) for c in target_cols]

        X, y = self.create_windows(normalized, feature_indices, target_indices)

        # Step 5: Chronological split
        splits = self.split_chronological(X, y)
        splits["feature_columns"] = feature_cols
        splits["target_columns"] = target_cols
        splits["machine_id"] = machine_id

        return splits

    def process_all_machines(
        self,
        machines: Dict[str, pd.DataFrame],
    ) -> Dict[str, dict]:
        """Process all machines and return dict of splits."""
        results = {}
        for mid, df in machines.items():
            try:
                results[mid] = self.process_machine(df, str(mid))
            except Exception as e:
                logger.warning(f"Skipping machine {mid}: {e}")
        return results

    def inverse_transform(
        self,
        data: np.ndarray,
        machine_id: str,
        columns: List[str],
    ) -> np.ndarray:
        """
        Reverse normalization for predictions back to original scale.
        Useful for computing real-unit metrics and display.
        """
        if machine_id not in self.scalers:
            raise RuntimeError(f"No scaler for machine {machine_id}")

        scaler = self.scalers[machine_id]
        # Build a dummy full-feature array to inverse transform target columns
        feature_cols = [c for c in self.config.feature_columns if True]
        n_features = len(scaler.scale_)
        dummy = np.zeros((len(data), n_features))

        all_cols = list(dict.fromkeys(
            self.config.feature_columns + self.config.target_columns
        ))
        for i, col in enumerate(columns):
            if col in all_cols:
                idx = all_cols.index(col)
                dummy[:, idx] = data[:, i] if data.ndim > 1 else data

        inversed = scaler.inverse_transform(dummy)
        result_indices = [all_cols.index(c) for c in columns if c in all_cols]
        return inversed[:, result_indices]

    def save_scalers(self, path: Path = None):
        """Save fitted scalers for inference."""
        path = path or DATA_PROCESSED / "scalers.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Scalers saved to {path}")

    def load_scalers(self, path: Path = None):
        """Load scalers for inference."""
        path = path or DATA_PROCESSED / "scalers.pkl"
        with open(path, "rb") as f:
            self.scalers = pickle.load(f)
        self._is_fitted = True
        logger.info(f"Scalers loaded from {path}")