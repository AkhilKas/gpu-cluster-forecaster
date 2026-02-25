"""
Load and parse Google Cluster Data into clean DataFrames.

"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from app.config import DataConfig, DATA_RAW

logger = logging.getLogger(__name__)

# Column names for instance_usage (no header in raw CSVs)
INSTANCE_USAGE_COLUMNS = [
    "start_time", "end_time", "collection_id", "instance_index",
    "machine_id", "alloc_collection_id",
    "avg_cpu", "avg_memory", "max_cpu", "max_memory",
    "sample_cpu", "sample_memory", "assigned_memory",
    "cycles_per_instruction", "memory_accesses_per_instruction",
    "sample_rate",
]


class ClusterDataLoader:
    """Load Google Cluster trace data into analysis-ready DataFrames."""

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.raw_dir = DATA_RAW / "google_cluster_2019"

    def load_instance_usage(
        self,
        csv_paths: List[Path] = None,
        max_rows_per_shard: int = 500_000,
    ) -> pd.DataFrame:
        """
        Load instance_usage shards into a single DataFrame.

        Args:
            csv_paths: Specific CSV files to load. If None, loads all in raw dir.
            max_rows_per_shard: Limit rows per file to control memory.

        Returns:
            DataFrame with parsed timestamps and renamed columns.
        """
        if csv_paths is None:
            usage_dir = self.raw_dir / "instance_usage"
            if not usage_dir.exists():
                raise FileNotFoundError(
                    f"No data found at {usage_dir}. "
                    "Run `python scripts/download_data.py` first."
                )
            csv_paths = sorted(usage_dir.glob("*.csv"))

        if not csv_paths:
            raise FileNotFoundError("No CSV files found to load.")

        logger.info(f"Loading {len(csv_paths)} instance_usage shards...")
        frames = []

        for path in csv_paths:
            try:
                df = pd.read_csv(
                    path,
                    header=None,
                    names=INSTANCE_USAGE_COLUMNS,
                    nrows=max_rows_per_shard,
                    na_values=["", " "],
                )
                frames.append(df)
                logger.debug(f"  Loaded {len(df)} rows from {path.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {path.name}: {e}")

        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Total rows loaded: {len(df):,}")

        # ── Clean & transform ──
        df = self._clean_instance_usage(df)
        return df

    def _clean_instance_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw instance_usage data."""
        # Convert microsecond timestamps to datetime
        # Google uses microseconds since trace start
        df["timestamp"] = pd.to_datetime(df["start_time"], unit="us")

        # Rename to project-standard column names
        df = df.rename(columns={
            "avg_cpu": "cpu_usage",
            "avg_memory": "memory_usage",
            "max_cpu": "max_cpu_usage",
            "max_memory": "max_memory_usage",
        })

        # CPU/memory are normalized 0-1 in Google data → scale to percentage
        for col in ["cpu_usage", "memory_usage", "max_cpu_usage", "max_memory_usage"]:
            if col in df.columns:
                df[col] = (df[col] * 100).clip(0, 100)

        # Drop rows with missing critical values
        df = df.dropna(subset=["cpu_usage", "memory_usage", "machine_id"])

        # Keep only columns we need
        keep_cols = [
            "timestamp", "machine_id", "collection_id",
            "cpu_usage", "memory_usage", "max_cpu_usage", "max_memory_usage",
            "assigned_memory", "cycles_per_instruction",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Sort by time
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Cleaned data: {len(df):,} rows, "
            f"{df['machine_id'].nunique()} unique machines, "
            f"time range: {df['timestamp'].min()} → {df['timestamp'].max()}"
        )
        return df

    def load_per_machine(
        self,
        df: pd.DataFrame = None,
        top_n_machines: int = 8,
    ) -> dict:
        """
        Group data by machine and return top N most active machines.

        Returns:
            dict mapping machine_id → DataFrame of that machine's time series
        """
        if df is None:
            df = self.load_instance_usage()

        # Aggregate all instances on a machine at each timestamp
        machine_df = (
            df.groupby(["timestamp", "machine_id"])
            .agg({
                "cpu_usage": "sum",
                "memory_usage": "sum",
                "max_cpu_usage": "max",
                "max_memory_usage": "max",
                "assigned_memory": "sum",
                "collection_id": "nunique",  # → running_jobs_count
            })
            .reset_index()
            .rename(columns={"collection_id": "running_jobs_count"})
        )

        # Clip aggregated CPU (sum of instances can exceed 100%)
        machine_df["cpu_usage"] = machine_df["cpu_usage"].clip(0, 100)
        machine_df["memory_usage"] = machine_df["memory_usage"].clip(0, 100)

        # Select top N machines by data volume
        top_machines = (
            machine_df["machine_id"]
            .value_counts()
            .head(top_n_machines)
            .index.tolist()
        )

        result = {}
        for mid in top_machines:
            mdf = (
                machine_df[machine_df["machine_id"] == mid]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            result[mid] = mdf
            logger.info(f"  Machine {mid}: {len(mdf)} records")

        return result

    @staticmethod
    def generate_synthetic(config: DataConfig = None) -> dict:
        """
        Generate synthetic GPU-like telemetry for development/testing.
        No download required — useful for building the pipeline first.

        Returns:
            dict mapping gpu_id → DataFrame
        """
        config = config or DataConfig()
        n_machines = config.synthetic_num_machines
        n_steps = config.synthetic_num_timesteps
        rng = np.random.default_rng(42)

        logger.info(
            f"Generating synthetic data: {n_machines} GPUs × {n_steps} timesteps"
        )

        timestamps = pd.date_range(
            start="2024-01-01",
            periods=n_steps,
            freq="5min",
        )

        machines = {}
        for gpu_id in range(n_machines):
            # Base utilization pattern (daily cycle + noise)
            t = np.arange(n_steps)
            daily_cycle = 20 * np.sin(2 * np.pi * t / 288)  # 288 = 24h / 5min
            base_util = 45 + gpu_id * 5
            trend = t * 0.001 * (gpu_id % 3 - 1)

            # Random workload spikes
            spikes = np.zeros(n_steps)
            n_spikes = rng.integers(10, 30)
            for _ in range(n_spikes):
                start = rng.integers(0, n_steps - 100)
                duration = rng.integers(20, 100)
                intensity = rng.uniform(15, 40)
                spikes[start:start + duration] += intensity

            cpu_usage = np.clip(
                base_util + daily_cycle + trend + spikes
                + rng.normal(0, 5, n_steps),
                0, 100,
            )

            # Memory correlates with CPU but is smoother
            memory_usage = np.clip(
                cpu_usage * 0.7 + 15 + rng.normal(0, 3, n_steps),
                0, 100,
            )

            # Simulated derived features
            temperature = 40 + cpu_usage * 0.4 + rng.normal(0, 2, n_steps)
            power_draw = 80 + cpu_usage * 2.5 + rng.normal(0, 10, n_steps)
            jobs_count = np.clip(
                (cpu_usage / 25).astype(int) + rng.integers(0, 2, n_steps),
                0, 10,
            )

            machines[gpu_id] = pd.DataFrame({
                "timestamp": timestamps,
                "machine_id": gpu_id,
                "cpu_usage": np.round(cpu_usage, 2),
                "memory_usage": np.round(memory_usage, 2),
                "max_cpu_usage": np.round(np.clip(cpu_usage + rng.uniform(5, 15, n_steps), 0, 100), 2),
                "max_memory_usage": np.round(np.clip(memory_usage + rng.uniform(3, 10, n_steps), 0, 100), 2),
                "temperature": np.round(temperature, 1),
                "power_draw": np.round(power_draw, 1),
                "running_jobs_count": jobs_count,
            })

        logger.info(f"Synthetic data ready: {n_machines} machines")
        return machines