"""Smoke tests for GET /health."""


class TestHealth:
    def test_returns_ok(self, client, api_env):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert api_env["model_name"] in body["models_loaded"]
        assert sorted(body["machines_available"]) == sorted(api_env["machine_ids"])

    def test_health_when_empty(self, empty_client):
        r = empty_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models_loaded"] == []
        assert body["machines_available"] == []
