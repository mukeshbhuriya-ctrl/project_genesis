"""
Integration tests for Deployment and Topology REST API endpoints (Phase 7).
"""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_deployment_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestDeploymentAPI:
    """Test /api/deployment routes."""

    def test_get_topology(self, client: TestClient):
        res = client.get("/api/deployment/topology")
        assert res.status_code == 200
        data = res.json()
        assert "hostname" in data
        assert "devices" in data
        assert "compute_backend" in data
        assert isinstance(data["devices"], list)
        assert len(data["devices"]) >= 1

    def test_calculate_layer_plan(self, client: TestClient):
        payload = {
            "model_name": "qwen2.5:7b",
            "total_layers": 28,
            "parameter_size_b": 7.0,
            "quant_bits": 4,
            "context_tokens": 4096,
            "safety_reserve_mb": 1024.0,
        }
        res = client.post("/api/deployment/layer-plan", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "qwen2.5:7b"
        assert data["total_layers"] == 28
        assert "strategy" in data
        assert "device_layers" in data
        assert isinstance(data["device_layers"], dict)

    def test_job_submission_and_management(self, client: TestClient):
        # 1. Invalid job_type returns 400
        bad_res = client.post(
            "/api/deployment/jobs",
            json={"job_type": "invalid_type", "payload": {}},
        )
        assert bad_res.status_code == 400

        # 2. Submit valid job
        submit_res = client.post(
            "/api/deployment/jobs",
            json={
                "job_type": "custom",
                "payload": {"action": "api_test_echo"},
                "priority": 25,
            },
        )
        assert submit_res.status_code == 200
        job = submit_res.json()
        job_id = job["id"]
        assert job["status"] == "queued"
        assert job["priority"] == 25

        # 3. List jobs
        list_res = client.get("/api/deployment/jobs?status=queued")
        assert list_res.status_code == 200
        jobs = list_res.json()
        assert any(j["id"] == job_id for j in jobs)

        # 4. Get specific job
        get_res = client.get(f"/api/deployment/jobs/{job_id}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == job_id

        # 5. Cancel job
        cancel_res = client.post(f"/api/deployment/jobs/{job_id}/cancel")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["success"] is True
        assert cancel_res.json()["job"]["status"] == "cancelled"

        # 6. Non-existent job
        not_found_res = client.get("/api/deployment/jobs/non-existent-uuid")
        assert not_found_res.status_code == 404
