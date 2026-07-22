"""Smoke tests for /models routes."""


class TestListModels:
    def test_lists_loaded_model(self, client, api_env):
        r = client.get("/models")
        assert r.status_code == 200
        body = r.json()
        assert len(body) >= 1
        entry = next(m for m in body if m["name"] == api_env["model_name"])
        assert entry["num_params"] > 0
        assert entry["invokable"] is True
        assert entry["has_metrics"] is True
        assert "input_dim" in entry["config"]


class TestGetMetrics:
    def test_returns_metrics_and_history(self, client, api_env):
        r = client.get(f"/models/{api_env['model_name']}/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == api_env["model_name"]
        assert body["overall"]["mae"] == 0.1234
        assert body["training_history"]["train_loss"] == [0.5, 0.3]
        assert len(body["per_horizon"]) == api_env["horizon"]

    def test_missing_model_returns_404(self, client):
        r = client.get("/models/does_not_exist/metrics")
        assert r.status_code == 404


class TestCompareModels:
    def test_returns_row_per_model(self, client, api_env):
        r = client.get("/models/compare")
        assert r.status_code == 200
        body = r.json()
        assert any(row["name"] == api_env["model_name"] for row in body)
        row = next(row for row in body if row["name"] == api_env["model_name"])
        assert row["mae"] == 0.1234
        assert row["rmse"] == 0.2345
