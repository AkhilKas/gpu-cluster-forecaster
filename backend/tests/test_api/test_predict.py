"""End-to-end + smoke tests for /predict."""

import io

import numpy as np
import pandas as pd


class TestPredict:
    def test_returns_correct_shape(self, client, api_env):
        window = np.random.rand(api_env["seq_len"], api_env["input_dim"]).tolist()
        r = client.post("/predict", json={"window": window})
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == api_env["model_name"]
        assert body["horizon"] == api_env["horizon"]
        assert len(body["forecast"]) == api_env["horizon"]
        assert len(body["forecast"][0]) == api_env["output_dim"]

    def test_explicit_model_name(self, client, api_env):
        window = np.random.rand(api_env["seq_len"], api_env["input_dim"]).tolist()
        r = client.post(
            "/predict", json={"window": window, "model": api_env["model_name"]}
        )
        assert r.status_code == 200

    def test_invalid_shape_returns_422(self, client, api_env):
        # 1D input where a 2D window is expected.
        r = client.post("/predict", json={"window": [1.0, 2.0, 3.0]})
        assert r.status_code == 422

    def test_unknown_model_returns_404(self, client, api_env):
        window = np.random.rand(api_env["seq_len"], api_env["input_dim"]).tolist()
        r = client.post("/predict", json={"window": window, "model": "no_such_model"})
        assert r.status_code == 404

    def test_no_models_loaded_returns_503(self, empty_client, api_env):
        window = np.random.rand(api_env["seq_len"], api_env["input_dim"]).tolist()
        r = empty_client.post("/predict", json={"window": window})
        assert r.status_code == 503


class TestPredictBatch:
    def test_returns_correct_shape(self, client, api_env):
        batch = np.random.rand(3, api_env["seq_len"], api_env["input_dim"]).tolist()
        r = client.post("/predict/batch", json={"windows": batch})
        assert r.status_code == 200
        body = r.json()
        assert body["horizon"] == api_env["horizon"]
        assert len(body["forecasts"]) == 3
        assert len(body["forecasts"][0]) == api_env["horizon"]

    def test_invalid_batch_shape_returns_422(self, client):
        # 2D where 3D is expected.
        r = client.post("/predict/batch", json={"windows": [[1.0, 2.0], [3.0, 4.0]]})
        assert r.status_code == 422


def _make_csv(n_rows: int, machines=("gpu-0",), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    rows = []
    for mid in machines:
        base = 40 + rng.integers(0, 20)
        for _ in range(n_rows):
            rows.append(
                {
                    "machine_id": mid,
                    "cpu_usage": float(np.clip(base + rng.normal(0, 8), 0, 100)),
                    "memory_usage": float(np.clip(30 + rng.normal(0, 6), 0, 100)),
                    "assigned_memory": 64.0,
                    "cycles_per_instruction": 1.0 + rng.normal(0, 0.05),
                }
            )
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


class TestPredictUpload:
    def test_happy_path_returns_forecasts(self, client, api_env):
        # api_env's data_config has sequence_length=30, horizon=6.
        csv_bytes = _make_csv(n_rows=60, machines=("gpu-a", "gpu-b"))
        files = {"file": ("upload.csv", csv_bytes, "text/csv")}
        r = client.post("/predict/upload", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["num_machines"] == 2
        assert body["model"] == api_env["model_name"]
        ids = sorted(m["machine_id"] for m in body["machines"])
        assert ids == ["gpu-a", "gpu-b"]
        for m in body["machines"]:
            assert len(m["forecast"]) == api_env["horizon"]
            assert "cpu_usage" in m["forecast"][0]["values"]
            assert m["warnings"] == []

    def test_no_machine_id_treats_as_single(self, client, api_env):
        # Drop the machine_id column so the service falls back to "uploaded".
        df = pd.DataFrame(
            {
                "cpu_usage": np.random.rand(40) * 100,
                "memory_usage": np.random.rand(40) * 100,
                "assigned_memory": [64] * 40,
                "cycles_per_instruction": [1.0] * 40,
            }
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["num_machines"] == 1
        assert body["machines"][0]["machine_id"] == "uploaded"
        assert any("machine_id" in w for w in body["warnings"])

    def test_only_required_columns_sufficient(self, client, api_env):
        # The fixture's LSTM has input_dim=2, so cpu_usage + memory_usage is
        # exactly what the model needs. No padding warnings should fire.
        df = pd.DataFrame(
            {
                "machine_id": ["gpu-x"] * 40,
                "cpu_usage": np.random.rand(40) * 100,
                "memory_usage": np.random.rand(40) * 100,
            }
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["num_machines"] == 1
        assert len(body["machines"][0]["forecast"]) == api_env["horizon"]
        assert body["warnings"] == []

    def test_extra_columns_ignored_with_warning(self, client, api_env):
        # Uploading more columns than the 2-feature model expects should
        # emit a warning naming the ignored columns.
        n = 40
        df = pd.DataFrame(
            {
                "machine_id": ["gpu-y"] * n,
                "cpu_usage": np.random.rand(n) * 100,
                "memory_usage": np.random.rand(n) * 100,
                "assigned_memory": [64.0] * n,
                "cycles_per_instruction": [1.0] * n,
            }
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert any("Model expects 2 features" in w for w in body["warnings"])

    def test_missing_required_column_returns_422(self, client):
        df = pd.DataFrame({"cpu_usage": [1, 2, 3]})  # missing memory_usage
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 422
        assert "memory_usage" in r.text

    def test_short_history_returns_warning(self, client):
        # 10 rows but sequence_length=30 in the api_env config.
        df = pd.DataFrame(
            {
                "machine_id": ["gpu-tiny"] * 10,
                "cpu_usage": [10.0] * 10,
                "memory_usage": [20.0] * 10,
            }
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["machines"][0]["forecast"] == []
        assert any("Not enough rows" in w for w in body["machines"][0]["warnings"])

    def test_non_csv_extension_returns_400(self, client):
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.txt", b"cpu,mem\n1,2\n", "text/plain")},
        )
        assert r.status_code == 400

    def test_empty_file_returns_400(self, client):
        r = client.post(
            "/predict/upload",
            files={"file": ("upload.csv", b"", "text/csv")},
        )
        assert r.status_code == 400

    def test_no_models_loaded_returns_503(self, empty_client):
        csv_bytes = _make_csv(n_rows=40)
        r = empty_client.post(
            "/predict/upload",
            files={"file": ("upload.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 503
