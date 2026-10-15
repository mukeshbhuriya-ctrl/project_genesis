"""
Integration tests for Tools and Tasks REST API endpoints.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core.database import initialize_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")

    app = create_app()
    return TestClient(app)


class TestToolsAPI:
    """Test /api/tools endpoints."""

    def test_list_tools(self, client: TestClient):
        res = client.get("/api/tools")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] >= 5
        tool_names = [t["name"] for t in data["tools"]]
        assert "calculator" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_execute_calculator_tool(self, client: TestClient):
        payload = {
            "tool_name": "calculator",
            "arguments": {"expression": "2 * 21"},
        }
        res = client.post("/api/tools/execute", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["output"] == 42
        assert data["execution_time_ms"] >= 0

    def test_execute_nonexistent_tool(self, client: TestClient):
        payload = {
            "tool_name": "quantum_teleporter",
            "arguments": {},
        }
        res = client.post("/api/tools/execute", json=payload)
        assert res.status_code == 404


class TestTasksAPI:
    """Test /api/tasks endpoints."""

    def test_run_task_with_explicit_steps(self, client: TestClient, tmp_path: Path):
        report_file = "data/api_test_report.txt"
        payload = {
            "goal": "Generate calculation report",
            "steps": [
                {
                    "description": "Compute value",
                    "tool_name": "calculator",
                    "tool_arguments": {"expression": "256 * 4"},
                },
                {
                    "description": "Write to file",
                    "tool_name": "write_file",
                    "tool_arguments": {"path": report_file, "content": "Value: 1024"},
                },
            ],
        }
        run_res = client.post("/api/tasks/run", json=payload)
        assert run_res.status_code == 200
        data = run_res.json()
        assert data["status"] == "completed"
        assert len(data["steps"]) == 2
        task_id = data["id"]

        # Get task by ID
        get_res = client.get(f"/api/tasks/{task_id}")
        assert get_res.status_code == 200
        task_data = get_res.json()
        assert task_data["id"] == task_id
        assert task_data["status"] == "completed"

        # List tasks
        list_res = client.get("/api/tasks")
        assert list_res.status_code == 200
        assert list_res.json()["count"] >= 1

        # Verify task
        verify_payload = {
            "criteria": {
                "file_exists": report_file,
                "file_contains": {report_file: "Value: 1024"},
            }
        }
        try:
            verify_res = client.post(f"/api/tasks/{task_id}/verify", json=verify_payload)
            assert verify_res.status_code == 200
            verify_data = verify_res.json()
            assert verify_data["verified"] is True
        finally:
            p = Path(report_file)
            if p.exists():
                p.unlink()
