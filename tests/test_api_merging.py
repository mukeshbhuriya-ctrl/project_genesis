"""
Integration tests for Cross-Fleet Merging, Kernel Synthesis, and Steering APIs (Phase 12).
"""

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_api_list_merge_methods(client):
    """Test GET /api/merging/methods returns supported algorithms."""
    res = client.get("/api/merging/methods")
    assert res.status_code == 200
    data = res.json()
    assert "methods" in data
    ids = [m["id"] for m in data["methods"]]
    assert "linear" in ids
    assert "slerp" in ids
    assert "ties" in ids
    assert "dare" in ids


def test_api_execute_and_manage_merge(client):
    """Test POST /api/merging/execute, GET /api/merging/merges, and DELETE."""
    payload = {
        "name": "api-ties-test",
        "merge_method": "ties",
        "target_model": "qwen2.5:7b",
        "source_adapters": ["adapter-node-a", "adapter-node-b"],
        "weights": [0.5, 0.5],
        "density": 0.8,
        "metadata": {"test_run": True},
    }
    exec_res = client.post("/api/merging/execute", json=payload)
    assert exec_res.status_code == 200
    recipe = exec_res.json()
    merge_id = recipe["id"]
    assert recipe["name"] == "api-ties-test"
    assert recipe["status"] == "completed"

    # List merges
    list_res = client.get("/api/merging/merges")
    assert list_res.status_code == 200
    merges = list_res.json()
    assert any(m["id"] == merge_id for m in merges)

    # Get merge
    get_res = client.get(f"/api/merging/merges/{merge_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "api-ties-test"

    # Delete merge
    del_res = client.delete(f"/api/merging/merges/{merge_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"


def test_api_kernel_synthesis_templates_and_run(client):
    """Test /api/kernel-synthesis endpoints."""
    # List templates
    tmpl_res = client.get("/api/kernel-synthesis/templates")
    assert tmpl_res.status_code == 200
    data = tmpl_res.json()
    assert "templates" in data
    assert "dot_product" in data["templates"]

    # Run synthesis
    run_payload = {
        "kernel_name": "dot_product",
        "operation_type": "reduction",
        "max_iterations": 2,
        "tolerance": 1e-3,
    }
    run_res = client.post("/api/kernel-synthesis/run", json=run_payload)
    assert run_res.status_code == 200
    exp_data = run_res.json()
    assert exp_data["kernel_name"] == "dot_product"
    assert exp_data["status"] == "success"
    assert exp_data["passed_numerical_check"] is True

    # List experiments
    exp_list_res = client.get("/api/kernel-synthesis/experiments")
    assert exp_list_res.status_code == 200
    assert len(exp_list_res.json()) >= 1


def test_api_steering_directives(client):
    """Test /api/steering/directives endpoints."""
    # Emit directive
    directive_payload = {
        "target_type": "campaign",
        "target_id": "camp-test-01",
        "action": "critique",
        "payload": {"feedback": "Focus specifically on KV-cache memory saving on 8GB VRAM."},
    }
    emit_res = client.post("/api/steering/directives", json=directive_payload)
    assert emit_res.status_code == 200
    event_data = emit_res.json()
    event_id = event_data["id"]
    assert event_data["status"] == "received"
    assert event_data["action"] == "critique"

    # List directives
    list_res = client.get("/api/steering/directives")
    assert list_res.status_code == 200
    events = list_res.json()
    assert any(e["id"] == event_id for e in events)

    # Mark applied
    apply_res = client.post(f"/api/steering/directives/{event_id}/apply")
    assert apply_res.status_code == 200
    assert apply_res.json()["status"] == "applied"
