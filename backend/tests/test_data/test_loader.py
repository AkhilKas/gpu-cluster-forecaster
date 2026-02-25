"""Tests for data loading and synthetic generation."""

import pandas as pd

from app.data.loader import ClusterDataLoader


class TestSyntheticDataGeneration:
    def test_generates_correct_num_machines(self, data_config):
        machines = ClusterDataLoader.generate_synthetic(data_config)
        assert len(machines) == data_config.synthetic_num_machines

    def test_generates_correct_num_timesteps(self, data_config):
        machines = ClusterDataLoader.generate_synthetic(data_config)
        for _, df in machines.items():
            assert len(df) == data_config.synthetic_num_timesteps

    def test_has_required_columns(self, synthetic_machines):
        required = ["timestamp", "machine_id", "cpu_usage", "memory_usage"]
        for _, df in synthetic_machines.items():
            for col in required:
                assert col in df.columns, f"Missing column: {col}"

    def test_values_in_valid_range(self, synthetic_machines):
        for _, df in synthetic_machines.items():
            assert df["cpu_usage"].between(0, 100).all()
            assert df["memory_usage"].between(0, 100).all()

    def test_timestamps_are_sorted(self, synthetic_machines):
        for _, df in synthetic_machines.items():
            assert df["timestamp"].is_monotonic_increasing

    def test_reproducible_with_same_config(self, data_config):
        m1 = ClusterDataLoader.generate_synthetic(data_config)
        m2 = ClusterDataLoader.generate_synthetic(data_config)
        key = sorted(m1.keys())[0]
        pd.testing.assert_frame_equal(m1[key], m2[key])
