"""
Integration tests for Meta-Improvement, Router, and Benchmark Expansion API endpoints (Phase 8).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner, get_standard_benchmarks
from genesis.evaluation.models import EvalRun, RunStatus


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_meta_test.db"
    initialize_database(db_file)
    runner = BenchmarkRunner(db_path=db_file)
    for bm in get_standard_benchmarks():
        runner.save_benchmark(bm)

    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestMetaAPI:
    """Test /api/meta/* routes."""

    def test_list_routes(self, client: TestClient):
        res = client.get("/api/meta/routes")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        categories = {r["task_category"] for r in data}
        assert "coding" in categories
        assert "reasoning" in categories
        assert "fast_validation" in categories

    def test_update_route(self, client: TestClient):
        payload = {
            "preferred_model": "qwen2.5-coder:14b",
            "fallback_model": "qwen2.5-coder:7b",
            "min_vram_mb": 9000.0,
            "metadata": {"custom_priority": True},
        }
        res = client.put("/api/meta/routes/coding", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task_category"] == "coding"
        assert data["preferred_model"] == "qwen2.5-coder:14b"
        assert data["min_vram_mb"] == 9000.0

        # Invalid category returns 400
        bad = client.put("/api/meta/routes/invalid-category", json=payload)
        assert bad.status_code == 400

    def test_route_task(self, client: TestClient):
        # Coding prompt
        payload = {"task_prompt": "def calculate_hash(data: bytes) -> str:\n    return hashlib.sha256(data).hexdigest()"}
        res = client.post("/api/meta/route", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task_category"] == "coding"
        assert "selected_model" in data
        assert "confidence" in data
        assert data["confidence"] > 0.6

    def test_expand_benchmark(self, client: TestClient):
        payload = {
            "benchmark_id": "bm-coding-v1",
            "domain": "coding",
            "count": 2,
        }
        res = client.post("/api/meta/benchmark/expand", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["benchmark_id"] == "bm-coding-v1"
        assert data["added_cases_count"] >= 1
        assert "cases" in data

    def test_meta_cycles_lifecycle(self, client: TestClient):
        # 1. Initially cycles list is empty or seeded
        list_res = client.get("/api/meta/cycles")
        assert list_res.status_code == 200
        assert isinstance(list_res.json(), list)

        # 2. Mock run_benchmark_async so test runs quickly without live model inference
        mock_run = EvalRun(
            id="test-run-fast",
            benchmark_id="bm-coding-v1",
            model_id="qwen2.5:7b",
            status=RunStatus.COMPLETED,
            total_cases=3,
            passed_cases=3,
            failed_cases=0,
            score=1.0,
        )

        with patch("genesis.evaluation.benchmarks.BenchmarkRunner.run_benchmark_async", new=AsyncMock(return_value=mock_run)):
            run_payload = {
                "benchmark_id": "bm-coding-v1",
                "model_id": "qwen2.5:7b",
                "name": "API Meta Cycle Test",
                "total_train_steps": 2,
            }
            res = client.post("/api/meta/cycles/run", json=run_payload)
            assert res.status_code == 200
            data = res.json()
            assert data["name"] == "API Meta Cycle Test"
            assert data["status"] == "completed"
            assert data["verdict"] == "promoted"
            assert len(data["stages"]) == 5
            cycle_id = data["id"]

            # 3. Retrieve specific cycle
            get_res = client.get(f"/api/meta/cycles/{cycle_id}")
            assert get_res.status_code == 200
            get_data = get_res.json()
            assert get_data["id"] == cycle_id
            assert get_data["verdict"] == "promoted"

            # 4. 404 for unknown cycle
            missing = client.get("/api/meta/cycles/non-existent-id")
            assert missing.status_code == 404
