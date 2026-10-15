"""
Integration tests for Continual Learning & Neuro-Symbolic REST API routes.
"""

import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import get_settings, reset_settings
from genesis.core.database import initialize_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_continual_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


def test_api_replay_buffer_flow(client):
    # 1. Add replay sample
    add_res = client.post(
        "/api/continual/replay/add",
        json={
            "sample_input": "def reverse_string(s):",
            "sample_target": "return s[::-1]",
            "task_domain": "coding",
            "priority": 2.0,
            "metadata": {"test": True},
        },
    )
    assert add_res.status_code == 201
    sample = add_res.json()
    assert sample["id"].startswith("replay-")
    assert sample["task_domain"] == "coding"

    # 2. List samples
    list_res = client.get("/api/continual/replay/samples?task_domain=coding")
    assert list_res.status_code == 200
    samples = list_res.json()
    assert len(samples) == 1
    assert samples[0]["id"] == sample["id"]

    # 3. Stats
    stats_res = client.get("/api/continual/replay/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["domains"]["coding"] == 1

    # 4. Batch sampling
    batch_res = client.get("/api/continual/replay/batch?batch_size=2")
    assert batch_res.status_code == 200
    assert len(batch_res.json()) == 1

    # 5. Delete sample
    del_res = client.delete(f"/api/continual/replay/samples/{sample['id']}")
    assert del_res.status_code == 200


def test_api_ewc_consolidation_and_penalty(client):
    weights = {"fc1.weight": [0.4, -0.1, 0.9], "fc2.weight": [0.2, 0.5, -0.3]}

    # 1. Consolidate EWC
    res = client.post(
        "/api/continual/ewc/consolidate",
        json={
            "task_name": "task_math_v1",
            "weights": weights,
            "importance_lambda": 500.0,
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["id"].startswith("ewc-")
    assert data["parameter_count"] == 6

    # 2. List checkpoints
    list_res = client.get("/api/continual/ewc/checkpoints")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 3. Compute penalty with perturbed weights
    perturbed = {"fc1.weight": [0.5, -0.1, 0.8], "fc2.weight": [0.2, 0.5, -0.3]}
    pen_res = client.post(
        "/api/continual/ewc/penalty",
        json={"checkpoint_id": data["id"], "current_weights": perturbed},
    )
    assert pen_res.status_code == 200
    assert pen_res.json()["ewc_penalty_loss"] > 0.0


def test_api_prefix_generation(client):
    res = client.post(
        "/api/continual/prefix/generate",
        json={"task_name": "synthetic_reasoning", "virtual_tokens": 8, "prefix_dim": 32},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["id"].startswith("prefix-")
    assert data["task_name"] == "synthetic_reasoning"
    assert data["virtual_tokens"] == 8


def test_api_symbolic_verification_routes(client):
    # 1. Code verification: detect infinite loop
    res_code = client.post(
        "/api/symbolic/verify/code",
        json={"code_str": "while True:\n    pass"},
    )
    assert res_code.status_code == 200
    data_code = res_code.json()
    assert data_code["status"] == "violated"
    assert len(data_code["violations"]) >= 1

    # 2. Math verification: non-negativity
    res_math = client.post(
        "/api/symbolic/verify/math",
        json={
            "expression_str": "x ** 2 + 1.0",
            "variable_ranges": {"x": [-2.0, 2.0]},
            "constraints": [{"constraint_type": "non_negativity"}],
        },
    )
    assert res_math.status_code == 200
    assert res_math.json()["status"] == "verified"

    # 3. Logic verification: Modus Ponens
    res_logic = client.post(
        "/api/symbolic/verify/logic",
        json={"premises": ["Rain", "Rain -> Wet"], "hypothesis": "Wet"},
    )
    assert res_logic.status_code == 200
    assert res_logic.json()["status"] == "verified"

    # 4. List verifications
    res_list = client.get("/api/symbolic/verifications")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 3

    # 5. Fetch single record
    rec_id = res_code.json()["id"]
    res_get = client.get(f"/api/symbolic/verifications/{rec_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == rec_id
