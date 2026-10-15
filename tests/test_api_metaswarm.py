"""
Tests for Metacognition, Swarm Consensus, Self-Compiler & Pareto REST API Routes (Phase 16).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_metaswarm_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


def test_metacognition_api_flow(client: TestClient):
    """Verify metacognitive introspection, listing, and summary stats via REST API."""
    # 1. Introspect request
    intro_res = client.post(
        "/api/metacognition/introspect",
        json={
            "prompt": "Evaluate invariant safety across concurrent AST mutations.",
            "candidate_output": "All invariant checks passed without deadlocks.",
            "token_log_probs": [-0.1, -0.2],
        },
    )
    assert intro_res.status_code == 200
    data = intro_res.json()
    assert data["id"].startswith("metacog-")
    assert "confidence_score" in data
    assert "decision" in data
    assert data["decision"] in ("fast_path", "reflective_repair", "swarm_escalation")

    # 2. List evaluations
    list_res = client.get("/api/metacognition/evaluations?limit=10")
    assert list_res.status_code == 200
    evals = list_res.json()
    assert len(evals) >= 1
    assert evals[0]["id"] == data["id"]

    # 3. Get summary
    sum_res = client.get("/api/metacognition/summary")
    assert sum_res.status_code == 200
    summary = sum_res.json()
    assert summary["total_evaluations"] >= 1
    assert "decisions" in summary


def test_swarm_api_flow(client: TestClient):
    """Verify swarm nodes registration, Byzantine consensus execution, and session inspection."""
    # 1. List default nodes
    nodes_res = client.get("/api/swarm/nodes")
    assert nodes_res.status_code == 200
    nodes = nodes_res.json()
    assert len(nodes) >= 4

    # 2. Register custom node
    reg_res = client.post(
        "/api/swarm/nodes",
        json={
            "name": "API-Reasoner-Gamma",
            "model_name": "qwen2.5:7b",
            "role": "critic",
            "reputation": 1.7,
        },
    )
    assert reg_res.status_code == 201
    custom_node = reg_res.json()
    assert custom_node["name"] == "API-Reasoner-Gamma"

    # 3. Execute consensus session
    cons_res = client.post(
        "/api/swarm/consensus",
        json={
            "task_prompt": "Synthesize verified LRU cache eviction logic.",
            "max_rounds": 2,
        },
    )
    assert cons_res.status_code == 200
    session = cons_res.json()
    assert session["id"].startswith("swarm-sess-")
    assert session["quorum_reached"] is True
    assert len(session["proof_chain_hash"]) == 64

    # 4. List and fetch sessions
    sess_list = client.get("/api/swarm/sessions?limit=5")
    assert sess_list.status_code == 200
    assert len(sess_list.json()) >= 1

    fetch_res = client.get(f"/api/swarm/sessions/{session['id']}")
    assert fetch_res.status_code == 200
    assert fetch_res.json()["id"] == session["id"]


def test_self_compiler_api_flow(client: TestClient):
    """Verify listing targets, compiling hot paths, and querying target metadata."""
    # 1. Compile single target
    comp_res = client.post(
        "/api/self-compiler/compile",
        json={"target_name": "cosine_similarity"},
    )
    assert comp_res.status_code == 200
    target = comp_res.json()
    assert target["target_name"] == "cosine_similarity"
    assert target["status"] in ("compiled_native", "fallback_python")

    # 2. List targets
    list_res = client.get("/api/self-compiler/targets")
    assert list_res.status_code == 200
    targets = list_res.json()
    assert len(targets) >= 1

    # 3. Fetch specific target
    fetch_res = client.get("/api/self-compiler/targets/cosine_similarity")
    assert fetch_res.status_code == 200
    assert fetch_res.json()["target_name"] == "cosine_similarity"


def test_pareto_api_flow(client: TestClient):
    """Verify executing Pareto evolutionary optimization and querying frontiers."""
    # 1. Run optimization
    opt_res = client.post(
        "/api/pareto/optimize",
        json={
            "name": "API-Test-Pareto",
            "population_size": 10,
            "generations": 2,
        },
    )
    assert opt_res.status_code == 200
    data = opt_res.json()
    assert "experiment" in data
    assert "frontier" in data
    exp_id = data["experiment"]["id"]
    assert len(data["frontier"]) >= 1

    # 2. List experiments
    exp_list = client.get("/api/pareto/experiments?limit=5")
    assert exp_list.status_code == 200
    assert len(exp_list.json()) >= 1

    # 3. Get frontiers by experiment id
    front_res = client.get(f"/api/pareto/experiments/{exp_id}/frontiers")
    assert front_res.status_code == 200
    frontiers = front_res.json()
    assert len(frontiers) == len(data["frontier"])
    assert frontiers[0]["latency_ms"] > 0


def test_api_404s(client: TestClient):
    """Verify proper 404 error handling for missing resources."""
    res1 = client.get("/api/swarm/sessions/nonexistent-session")
    assert res1.status_code == 404

    res2 = client.get("/api/self-compiler/targets/nonexistent_target")
    assert res2.status_code == 404
