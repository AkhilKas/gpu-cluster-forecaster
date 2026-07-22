"""Smoke tests for /machines routes."""


class TestListMachines:
    def test_returns_all_machines(self, client, api_env):
        r = client.get("/machines")
        assert r.status_code == 200
        body = r.json()
        assert sorted(m["id"] for m in body) == sorted(api_env["machine_ids"])
        for m in body:
            assert m["num_windows"] > 0
            # Denormalized cpu_usage should be back in the 0-100 range.
            assert 0 <= (m["latest_cpu"] or 0) <= 100


class TestHistory:
    def test_returns_denormalized_window(self, client, api_env):
        mid = api_env["machine_ids"][0]
        r = client.get(f"/machines/{mid}/history?steps=10")
        assert r.status_code == 200
        body = r.json()
        assert body["machine_id"] == mid
        assert body["denormalized"] is True
        assert len(body["steps"]) == 10
        first = body["steps"][0]
        assert "cpu_usage" in first["values"]

    def test_missing_machine_returns_404(self, client):
        r = client.get("/machines/no_such_machine/history")
        assert r.status_code == 404


class TestForecast:
    def test_end_to_end_forecast(self, client, api_env):
        mid = api_env["machine_ids"][0]
        r = client.get(f"/machines/{mid}/forecast")
        assert r.status_code == 200
        body = r.json()
        assert body["machine_id"] == mid
        assert body["horizon"] == api_env["horizon"]
        assert len(body["forecast"]) == api_env["horizon"]
        assert len(body["history"]) == api_env["seq_len"]
        # Forecast values should be denormalized (percentages, not [0,1]).
        first_forecast = body["forecast"][0]["values"]
        assert "cpu_usage" in first_forecast
        # Denormalized cpu should sit inside a plausible % range.
        assert -20 <= first_forecast["cpu_usage"] <= 200

    def test_forecast_missing_machine(self, client):
        r = client.get("/machines/no_such_machine/forecast")
        assert r.status_code == 404
