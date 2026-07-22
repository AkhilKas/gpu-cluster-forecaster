"""End-to-end + smoke tests for /predict."""

import numpy as np


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
