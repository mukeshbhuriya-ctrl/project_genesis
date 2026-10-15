"""
Integration tests for Evaluation and Benchmarking REST API endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import EvalResult, EvalRun, RunStatus


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_eval_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestEvaluationAPI:
    """Test /api/evaluation routes."""

    def test_list_benchmarks(self, client: TestClient):
        res = client.get("/api/evaluation/benchmarks")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] >= 3
        bm_ids = [b["id"] for b in data["benchmarks"]]
        assert "bm-reasoning-v1" in bm_ids
        assert "bm-coding-v1" in bm_ids
        assert "bm-tools-v1" in bm_ids

    def test_get_benchmark_detail(self, client: TestClient):
        res = client.get("/api/evaluation/benchmarks/bm-reasoning-v1")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "bm-reasoning-v1"
        assert data["category"] == "reasoning"
        assert len(data["cases"]) >= 3

    def test_get_nonexistent_benchmark(self, client: TestClient):
        res = client.get("/api/evaluation/benchmarks/non-existent-benchmark")
        assert res.status_code == 404

    def test_runs_and_reports_workflow(self, client: TestClient, tmp_path: Path):
        db_file = tmp_path / "api_eval_test.db"
        runner = BenchmarkRunner(db_path=db_file)

        # Seed a completed run into DB
        run = EvalRun(
            id="run-api-test-1",
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5:7b",
            status=RunStatus.COMPLETED,
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            score=100.0,
            duration_ms=450.0,
            started_at="2026-10-15T02:00:00Z",
            results=[
                EvalResult(case_id="case-reas-1", passed=True, score=1.0, latency_ms=200.0),
                EvalResult(case_id="case-reas-2", passed=True, score=1.0, latency_ms=250.0),
            ],
        )
        runner.save_run(run)

        # 1. List runs
        res = client.get("/api/evaluation/runs")
        assert res.status_code == 200
        runs_data = res.json()
        assert runs_data["count"] >= 1
        assert any(r["id"] == "run-api-test-1" for r in runs_data["runs"])

        # 2. Get specific run
        res = client.get("/api/evaluation/runs/run-api-test-1")
        assert res.status_code == 200
        run_data = res.json()
        assert run_data["id"] == "run-api-test-1"
        assert run_data["score"] == 100.0
        assert len(run_data["results"]) == 2

        # 3. Get run report
        res = client.get("/api/evaluation/runs/run-api-test-1/report")
        assert res.status_code == 200
        rep_data = res.json()
        assert "markdown" in rep_data
        assert "# Evaluation Report:" in rep_data["markdown"]
        assert rep_data["run_id"] == "run-api-test-1"

        # 4. Compare run (establishes baseline if first)
        compare_payload = {"candidate_run_id": "run-api-test-1"}
        res = client.post("/api/evaluation/compare", json=compare_payload)
        assert res.status_code == 200
        comp_data = res.json()
        assert "comparison" in comp_data
        assert comp_data["comparison"]["verdict"] == "BASELINE_ESTABLISHED"
        assert "report_markdown" in comp_data
