"""
Loads processed per-machine data + fitted scalers into memory at startup.

Expected layout (produced by scripts/download_data.py --process):

    data/processed/
        scalers.pkl
        machine_<id>/
            X_train.npy  y_train.npy
            X_val.npy    y_val.npy
            X_test.npy   y_test.npy
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class DataStore:
    """In-memory cache of processed splits + per-machine scalers."""

    def __init__(self):
        self.machines: dict[str, dict[str, np.ndarray]] = {}
        self.scalers: dict[str, MinMaxScaler] = {}

    def load(self, processed_dir: Path) -> None:
        """Load scalers.pkl and every machine_*/ subdirectory."""
        processed_dir = Path(processed_dir)
        if not processed_dir.exists():
            logger.warning(f"Processed data dir does not exist: {processed_dir}")
            return

        scalers_path = processed_dir / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, "rb") as f:
                self.scalers = pickle.load(f)
            logger.info(
                f"Loaded {len(self.scalers)} scaler(s) from {scalers_path.name}"
            )
        else:
            logger.warning(f"No scalers.pkl at {processed_dir}")

        for machine_dir in sorted(processed_dir.glob("machine_*")):
            machine_id = machine_dir.name.removeprefix("machine_")
            try:
                splits = {}
                for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
                    npy_path = machine_dir / f"{key}.npy"
                    if npy_path.exists():
                        splits[key] = np.load(npy_path)
                if splits:
                    self.machines[machine_id] = splits
                    logger.info(
                        f"Loaded machine {machine_id}: "
                        f"X_test={splits.get('X_test', np.array([])).shape}"
                    )
            except Exception as e:
                logger.error(f"Failed to load {machine_dir}: {e}")

    def has_machine(self, machine_id: str) -> bool:
        return machine_id in self.machines

    def get_latest_window(self, machine_id: str) -> np.ndarray:
        """
        Return the most recent input window observed for a machine.

        Shape: (seq_len, num_features), still in normalized [0, 1] space.
        """
        if machine_id not in self.machines:
            raise KeyError(f"Machine {machine_id!r} not found.")
        splits = self.machines[machine_id]
        # Prefer the tail of X_test, fall back to X_val or X_train
        for split_name in ["X_test", "X_val", "X_train"]:
            x = splits.get(split_name)
            if x is not None and len(x) > 0:
                return x[-1]
        raise KeyError(f"Machine {machine_id!r} has no data.")

    def get_scaler(self, machine_id: str) -> MinMaxScaler | None:
        """Fetch the fitted MinMaxScaler for a machine, if available."""
        # scalers were keyed by str(machine_id) during preprocessing
        return self.scalers.get(str(machine_id))
