"""Tests for the FastAPI health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    """Provide a test client for the API."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_ok(self, client: TestClient):
        """Health check should return 200 with status healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_includes_version(self, client: TestClient):
        """Health response should include the application version."""
        from genesis import __version__
        response = client.get("/health")
        assert response.json()["version"] == __version__
