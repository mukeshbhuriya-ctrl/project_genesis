"""
Integration tests for Phase 14 REST APIs:
- Structural Pruning & NAS (/api/pruning/...)
- Hardware-Targeted Micro-Quantization (/api/quantization/...)
- Self-Speculative Decoding Engine (/api/speculative/...)
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


# ── Structural Pruning & NAS API Tests ──

def test_api_pruning_architectures(client):
    """Test GET /api/pruning/architectures."""
    res = client.get("/api/pruning/architectures")
    assert res.status_code == 200
    data = res.json()
    assert "qwen2.5:7b" in data
    assert "qwen2.5-coder:1.5b-base" in data
    assert data["qwen2.5:7b"]["layers"] == 28


def test_api_pruning_plan_and_execute(client):
    """Test POST /api/pruning/plan and POST /api/pruning/execute."""
    # Plan
    plan_req = {
        "model_name": "qwen2.5:7b",
        "target_vram_mb": 4000.0,
        "strategy": "structured_head",
    }
    plan_res = client.post("/api/pruning/plan", json=plan_req)
    assert plan_res.status_code == 200
    plan = plan_res.json()
    assert plan["model_name"] == "qwen2.5:7b"
    assert plan["parameter_reduction_pct"] > 0
    assert plan["estimated_vram_mb"] < plan["original_vram_mb"]

    # Execute
    test_id = f"test-exp-nas-{uuid.uuid4().hex[:6]}"
    exec_req = {
        "model_name": "qwen2.5:7b",
        "target_vram_mb": 4000.0,
        "strategy": "structured_head",
        "experiment_id": test_id,
    }
    exec_res = client.post("/api/pruning/execute", json=exec_req)
    assert exec_res.status_code == 200
    record = exec_res.json()
    assert record["id"] == test_id
    assert record["status"] in ("verified", "planned")

    # List experiments
    list_res = client.get("/api/pruning/experiments")
    assert list_res.status_code == 200
    experiments = list_res.json()
    assert any(e["id"] == test_id for e in experiments)

    # Get single experiment
    get_res = client.get(f"/api/pruning/experiments/{test_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == test_id

    # Not found
    nf_res = client.get("/api/pruning/experiments/nonexistent-id")
    assert nf_res.status_code == 404


# ── Micro-Quantization API Tests ──

def test_api_quantization_strategies(client):
    """Test GET /api/quantization/strategies."""
    res = client.get("/api/quantization/strategies")
    assert res.status_code == 200
    data = res.json()
    assert "strategies" in data
    ids = [s["id"] for s in data["strategies"]]
    assert "awq_outlier_salient" in ids
    assert "symmetric" in ids
    assert "asymmetric" in ids


def test_api_quantization_profile_and_manage(client):
    """Test POST /api/quantization/profile and GET /api/quantization/profiles."""
    profile_req = {
        "model_name": "qwen2.5:7b",
        "bits": 4,
        "strategy": "awq_outlier_salient",
        "block_size": 64,
        "outlier_channel_ratio": 0.01,
    }
    prof_res = client.post("/api/quantization/profile", json=profile_req)
    assert prof_res.status_code == 200
    prof = prof_res.json()
    prof_id = prof["id"]
    assert prof["bits"] == 4
    assert prof["compression_ratio"] > 3.0
    assert prof["cosine_similarity"] > 0.90

    # List profiles
    list_res = client.get("/api/quantization/profiles")
    assert list_res.status_code == 200
    profiles = list_res.json()
    assert any(p["id"] == prof_id for p in profiles)

    # Get profile
    get_res = client.get(f"/api/quantization/profiles/{prof_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == prof_id

    # Not found
    assert client.get("/api/quantization/profiles/nonexistent-prof").status_code == 404


# ── Self-Speculative Decoding API Tests ──

def test_api_speculative_configs(client):
    """Test GET /api/speculative/configs."""
    res = client.get("/api/speculative/configs")
    assert res.status_code == 200
    data = res.json()
    assert "recommended_pairings" in data
    assert len(data["recommended_pairings"]) >= 2


def test_api_speculative_run_and_manage(client):
    """Test POST /api/speculative/run and GET /api/speculative/runs."""
    run_test_id = f"test-spec-api-{uuid.uuid4().hex[:6]}"
    run_req = {
        "prompt": "Test speculative decoding acceleration.",
        "target_model": "qwen2.5:7b",
        "draft_model": "qwen2.5-coder:1.5b-base",
        "lookahead_k": 4,
        "run_id": run_test_id,
    }
    run_res = client.post("/api/speculative/run", json=run_req)
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["id"] == run_test_id
    assert run_data["speedup_factor"] >= 1.0
    assert run_data["status"] in ("completed", "fallback_autoregressive")

    # List runs
    list_res = client.get("/api/speculative/runs")
    assert list_res.status_code == 200
    runs = list_res.json()
    assert any(r["id"] == run_test_id for r in runs)

    # Get run
    get_res = client.get(f"/api/speculative/runs/{run_test_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == run_test_id

    # Not found
    assert client.get("/api/speculative/runs/nonexistent-run").status_code == 404
