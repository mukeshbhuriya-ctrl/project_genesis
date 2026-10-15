"""
Tests for World Model, Latent Dreams & MCTS REST API Routes (Phase 15).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_world_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


def test_predict_and_record_transition_api(client: TestClient):
    """Verify predicting and recording state transitions over REST API."""
    predict_payload = {
        "current_state": {
            "files": {"main.py": "x = 1\n"},
            "ast_symbols": {"main.py": []},
        },
        "action": {
            "action_type": "file_write",
            "target": "service.py",
            "payload": "def handle_request():\n    return {'status': 'ok'}\n",
        },
    }

    resp = client.post("/api/world/predict-transition", json=predict_payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    assert "service.py" in pred_data["predicted_state"]["files"]
    assert pred_data["safety_score"] == 1.0

    # Record transition
    record_payload = {
        "session_id": "api-sess-001",
        "action": predict_payload["action"],
        "prediction": pred_data,
        "actual_state": pred_data["predicted_state"],
        "record_id": "rec-api-001",
    }
    rec_resp = client.post("/api/world/record-transition", json=record_payload)
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["id"] == "rec-api-001"
    assert rec_data["transition_error_score"] == 0.0

    # List transitions
    list_resp = client.get("/api/world/transitions?session_id=api-sess-001")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    # Get specific transition
    get_resp = client.get("/api/world/transitions/rec-api-001")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == "rec-api-001"


def test_dream_rollout_api(client: TestClient):
    """Verify latent dream rollout and replay buffer export over REST API."""
    rollout_payload = {
        "prompt": "Synthesize distributed consensus barrier algorithm",
        "depth": 3,
        "export_to_replay": True,
        "replay_domain": "reasoning",
    }
    resp = client.post("/api/world/dream-rollout", json=rollout_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert len(data["steps"]) >= 3
    assert data["exported_to_replay_count"] >= 1
    traj_id = data["id"]

    # List trajectories
    list_resp = client.get("/api/world/trajectories")
    assert list_resp.status_code == 200
    assert any(t["id"] == traj_id for t in list_resp.json())

    # Get trajectory
    get_resp = client.get(f"/api/world/trajectories/{traj_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == traj_id


def test_align_modalities_and_retrieve_api(client: TestClient):
    """Verify multi-modal representation alignment and cross-modal retrieval via REST."""
    align_payload = {
        "text_prompt": "Interactive data table with sorting and filtering",
        "ast_code": "class SortableTable:\n    def sort_by(self, col: str):\n        pass\n",
        "visual_spec": {"component": "DataTable", "features": ["sort", "filter"]},
        "record_id": "align-api-001",
    }
    resp = client.post("/api/world/align-modalities", json=align_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "align-api-001"
    assert -1.0 <= data["overall_alignment_score"] <= 1.0

    # List alignments
    list_resp = client.get("/api/world/alignments")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

    # Cross-modal retrieval
    ret_payload = {
        "query_modality": "text",
        "query_repr": "sortable table",
        "candidates": [
            {"modality": "ast", "content": "class SortableTable:\n    pass\n"},
            {"modality": "visual", "content": "NavBar component"},
        ],
    }
    ret_resp = client.post("/api/world/cross-modal-retrieve", json=ret_payload)
    assert ret_resp.status_code == 200
    ranked = ret_resp.json()
    assert len(ranked) == 2
    assert ranked[0]["modality"] == "ast"


def test_mcts_search_api(client: TestClient):
    """Verify MCTS tree search execution and querying over REST API."""
    mcts_payload = {
        "root_task": "Refactor token encoder with SIMD vectorization",
        "initial_files": {"encoder.py": "def encode(text):\n    return [ord(c) for c in text]\n"},
        "iterations": 8,
        "max_depth": 3,
        "run_id": "mcts-api-run-001",
    }
    resp = client.post("/api/world/mcts-search", json=mcts_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "mcts-api-run-001"
    assert data["nodes_explored"] >= 2
    assert len(data["best_path"]) >= 1

    # List runs
    list_resp = client.get("/api/world/mcts-runs")
    assert list_resp.status_code == 200
    assert any(r["id"] == "mcts-api-run-001" for r in list_resp.json())

    # Get run
    get_resp = client.get("/api/world/mcts-runs/mcts-api-run-001")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == "mcts-api-run-001"


def test_world_model_404s(client: TestClient):
    """Verify 404 responses for non-existent entities."""
    assert client.get("/api/world/transitions/nonexistent").status_code == 404
    assert client.get("/api/world/trajectories/nonexistent").status_code == 404
    assert client.get("/api/world/mcts-runs/nonexistent").status_code == 404
