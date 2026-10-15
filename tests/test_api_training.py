"""
Integration tests for Training and Dataset REST API endpoints.
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
    db_file = tmp_path / "api_training_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestTrainingAPI:
    """Test /api/training routes."""

    def test_dataset_lifecycle(self, client: TestClient):
        # 1. Initial listing
        res = client.get("/api/training/datasets")
        assert res.status_code == 200
        assert res.json()["count"] == 0

        # 2. Create dataset
        samples = [
            {"instruction": "What is SGD?", "output": "Stochastic Gradient Descent."},
            {"instruction": "What is Adam?", "output": "Adaptive Moment Estimation optimizer."},
            {"instruction": "What is SGD?", "output": "Stochastic Gradient Descent."},  # duplicate
        ]
        res = client.post(
            "/api/training/datasets/create",
            json={
                "name": "opt-theory-v1",
                "description": "Optimization fundamentals for reasoning",
                "format": "instruction",
                "version": "1.0.0",
                "samples": samples,
            },
        )
        assert res.status_code == 200
        ds = res.json()
        assert ds["name"] == "opt-theory-v1"
        assert ds["sample_count"] == 3
        dataset_id = ds["id"]

        # 3. Retrieve dataset detail
        res = client.get(f"/api/training/datasets/{dataset_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["dataset"]["id"] == dataset_id
        assert data["statistics"]["sample_count"] == 3
        assert len(data["preview"]) == 3

        # 4. Preprocess dataset
        res = client.post(
            f"/api/training/datasets/{dataset_id}/preprocess",
            json={
                "min_tokens": 2,
                "max_tokens": 1024,
                "deduplicate": True,
                "validate_syntax": True,
                "check_contamination": True,
            },
        )
        assert res.status_code == 200
        prep_data = res.json()
        report = prep_data["report"]
        assert report["duplicates_removed"] == 1
        assert report["final_count"] == 2
        assert "cleaned_dataset" in prep_data
        assert prep_data["cleaned_dataset"]["sample_count"] == 2

    def test_experiment_and_training_lifecycle(self, client: TestClient):
        # 1. Register a dataset first
        samples = [
            {"instruction": f"Instruction sample {i}", "output": f"Output response {i}"}
            for i in range(12)
        ]
        res = client.post(
            "/api/training/datasets/create",
            json={
                "name": "exp-ds",
                "description": "Dataset for training run",
                "samples": samples,
            },
        )
        assert res.status_code == 200
        dataset_id = res.json()["id"]

        # 2. Create experiment
        res = client.post(
            "/api/training/experiments/create",
            json={
                "name": "lora-qwen-reasoning",
                "base_model": "qwen2.5:7b",
                "dataset_id": dataset_id,
                "hyperparameters": {
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "epochs": 2,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 2e-4,
                },
                "total_steps": 6,
            },
        )
        assert res.status_code == 200
        exp = res.json()
        exp_id = exp["id"]
        assert exp["status"] == "pending"
        assert exp["base_model"] == "qwen2.5:7b"

        # 3. Retrieve experiment detail
        res = client.get(f"/api/training/experiments/{exp_id}")
        assert res.status_code == 200
        assert res.json()["id"] == exp_id

        # 4. List experiments
        res = client.get("/api/training/experiments")
        assert res.status_code == 200
        assert res.json()["count"] >= 1

        # 5. Start training
        res = client.post(f"/api/training/experiments/{exp_id}/start?step_delay_s=0.0")
        assert res.status_code == 200
        completed = res.json()
        assert completed["status"] == "completed"
        assert completed["current_step"] == completed["total_steps"]
        assert completed["current_step"] > 0
        assert completed["training_loss"] is not None

        # 6. List checkpoints
        res = client.get(f"/api/training/experiments/{exp_id}/checkpoints")
        assert res.status_code == 200
        chk_data = res.json()
        assert chk_data["count"] >= 1
        best_chk_id = chk_data["best_checkpoint_id"]
        assert best_chk_id is not None

        # 7. Deploy checkpoint
        res = client.post(
            f"/api/training/checkpoints/{best_chk_id}/deploy",
            json={
                "deployed_name": "qwen2.5-tuned:v1",
                "system_prompt": "You are Genesis Tuned Reasoning Model.",
                "parameters": {"temperature": 0.2},
            },
        )
        assert res.status_code == 200
        deploy_data = res.json()
        assert deploy_data["status"] == "deployed"
        assert deploy_data["deployed_name"] == "qwen2.5-tuned:v1"

        # 8. Rollback model
        res = client.post(
            f"/api/training/experiments/{exp_id}/rollback",
            json={"baseline_model": "qwen2.5:7b"},
        )
        assert res.status_code == 200
        rb_data = res.json()
        assert rb_data["status"] == "rolled_back"
        assert rb_data["active_baseline"] == "qwen2.5:7b"

    def test_not_found_errors(self, client: TestClient):
        res = client.get("/api/training/datasets/nonexistent-ds-id")
        assert res.status_code == 404

        res = client.post(
            "/api/training/datasets/nonexistent-ds-id/preprocess",
            json={"min_tokens": 1},
        )
        assert res.status_code == 404

        res = client.get("/api/training/experiments/nonexistent-exp-id")
        assert res.status_code == 404

        res = client.post(
            "/api/training/checkpoints/nonexistent-chk-id/deploy",
            json={"deployed_name": "test-name"},
        )
        assert res.status_code == 404
