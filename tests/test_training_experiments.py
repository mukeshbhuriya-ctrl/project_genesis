"""
Tests for ExperimentTracker, CheckpointManager, Modelfile export, and FineTuningTrainer.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.training.datasets import DatasetFormat, DatasetManager, DatasetSample
from genesis.training.experiments import (
    ExperimentStatus,
    ExperimentTracker,
    Hyperparameters,
    TrainingMetricPoint,
)
from genesis.training.export import CheckpointManager, TrainingCheckpoint
from genesis.training.trainer import FineTuningTrainer


class TestExperimentTrackerAndCheckpoints:
    """Tests for ExperimentTracker and CheckpointManager persistence."""

    def test_experiment_lifecycle_and_metrics(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        mgr = DatasetManager(db_path=tmp_db_path)
        tracker = ExperimentTracker(db_path=tmp_db_path)

        ds = mgr.register_dataset(
            name="test-ds",
            samples=[DatasetSample(instruction=f"Instr {i}", output=f"Out {i}") for i in range(10)],
        )

        hp = Hyperparameters(lora_r=8, lora_alpha=16, epochs=2, batch_size=2)
        exp = tracker.create_experiment(
            name="exp-test-qwen",
            base_model="qwen2.5:7b",
            dataset_id=ds.id,
            hyperparameters=hp,
            total_steps=10,
        )

        assert exp.status == ExperimentStatus.PENDING
        assert exp.base_model == "qwen2.5:7b"
        assert exp.total_epochs == 2
        assert exp.total_steps == 10

        # Record metrics
        m1 = TrainingMetricPoint(step=1, epoch=0.2, loss=2.31, learning_rate=2e-4)
        m2 = TrainingMetricPoint(step=2, epoch=0.4, loss=1.95, eval_loss=1.98, learning_rate=1.8e-4)
        tracker.record_metric(exp.id, m1)
        tracker.record_metric(exp.id, m2)

        reloaded = tracker.get_experiment(exp.id)
        assert reloaded is not None
        assert reloaded.current_step == 2
        assert reloaded.training_loss == 1.95
        assert reloaded.eval_loss == 1.98
        assert len(reloaded.metrics) == 2

        # Update status
        tracker.update_status(exp.id, ExperimentStatus.COMPLETED, duration_ms=5400.0)
        final_exp = tracker.get_experiment(exp.id)
        assert final_exp.status == ExperimentStatus.COMPLETED
        assert final_exp.duration_ms == 5400.0
        assert final_exp.completed_at is not None

    def test_checkpoint_management_and_best_selection(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        mgr = DatasetManager(db_path=tmp_db_path)
        tracker = ExperimentTracker(db_path=tmp_db_path)
        chk_mgr = CheckpointManager(db_path=tmp_db_path)

        ds = mgr.register_dataset(
            name="test-ds-chk",
            samples=[DatasetSample(instruction="Q", output="A")],
        )

        exp = tracker.create_experiment(
            name="test-exp-chk",
            base_model="qwen2.5:7b",
            dataset_id=ds.id,
            hyperparameters=Hyperparameters(),
            total_steps=15,
        )

        c1 = TrainingCheckpoint(experiment_id=exp.id, step=5, epoch=1.0, training_loss=1.80, eval_loss=1.85, file_path="chk5")
        c2 = TrainingCheckpoint(experiment_id=exp.id, step=10, epoch=2.0, training_loss=1.20, eval_loss=1.25, file_path="chk10", is_best=True)
        c3 = TrainingCheckpoint(experiment_id=exp.id, step=15, epoch=3.0, training_loss=1.35, eval_loss=1.40, file_path="chk15")

        chk_mgr.save_checkpoint(c1)
        chk_mgr.save_checkpoint(c2)
        chk_mgr.save_checkpoint(c3)

        checkpoints = chk_mgr.list_checkpoints(exp.id)
        assert len(checkpoints) == 3

        best = chk_mgr.get_best_checkpoint(exp.id)
        assert best is not None
        assert best.step == 10
        assert best.is_best is True

    def test_build_modelfile_and_rollback(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        mgr = DatasetManager(db_path=tmp_db_path)
        tracker = ExperimentTracker(db_path=tmp_db_path)
        chk_mgr = CheckpointManager(db_path=tmp_db_path)

        modelfile_str = chk_mgr.build_modelfile(
            base_model="qwen2.5:7b",
            system_prompt="You are a specialized Genesis fine-tuned reasoning agent.",
            parameters={"temperature": 0.1, "top_p": 0.85},
        )

        assert "FROM qwen2.5:7b" in modelfile_str
        assert "PARAMETER temperature 0.1" in modelfile_str
        assert "PARAMETER top_p 0.85" in modelfile_str
        assert "You are a specialized Genesis fine-tuned reasoning agent." in modelfile_str

        # Test rollback
        ds = mgr.register_dataset(
            name="test-ds-rollback",
            samples=[DatasetSample(instruction="Q", output="A")],
        )
        exp = tracker.create_experiment(
            name="test-exp-rollback",
            base_model="qwen2.5:7b",
            dataset_id=ds.id,
            hyperparameters=Hyperparameters(),
            total_steps=5,
        )
        chk = TrainingCheckpoint(experiment_id=exp.id, step=5, epoch=1.0, training_loss=1.0, file_path="chk", deployed_name="candidate-model")
        chk_mgr.save_checkpoint(chk)

        rb_result = chk_mgr.rollback_model(exp.id, baseline_model="qwen2.5:7b")
        assert rb_result["status"] == "rolled_back"
        assert rb_result["active_baseline"] == "qwen2.5:7b"


class TestFineTuningTrainer:
    """Tests for FineTuningTrainer execution loop."""

    @pytest.mark.asyncio
    async def test_run_training_loop_mock(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        mgr = DatasetManager(db_path=tmp_db_path)
        tracker = ExperimentTracker(db_path=tmp_db_path)
        trainer = FineTuningTrainer(db_path=tmp_db_path)

        # Create dataset
        ds = mgr.register_dataset(
            name="training-ds",
            samples=[DatasetSample(instruction=f"Instr {i}", output=f"Out {i}") for i in range(20)],
        )

        # Create experiment: 2 epochs, batch size 2, gradient accumulation 2 -> 4 effective batch size
        hp = Hyperparameters(lora_r=8, lora_alpha=16, epochs=2, batch_size=2, gradient_accumulation_steps=2)
        exp = tracker.create_experiment(
            name="live-test-exp",
            base_model="qwen2.5:7b",
            dataset_id=ds.id,
            hyperparameters=hp,
            total_steps=8,
        )

        metrics_received = []
        completed_exp = await trainer.run_training_async(
            experiment_id=exp.id,
            checkpoint_interval=4,
            step_delay_s=0.01,
            metric_callback=lambda m: metrics_received.append(m),
        )

        assert completed_exp.status == ExperimentStatus.COMPLETED
        assert completed_exp.current_step == 8
        assert completed_exp.training_loss is not None
        assert completed_exp.training_loss < 2.45  # Loss decreased from initial 2.45
        assert len(metrics_received) == 8

        # Check that checkpoints were generated
        chk_mgr = CheckpointManager(db_path=tmp_db_path)
        checkpoints = chk_mgr.list_checkpoints(exp.id)
        assert len(checkpoints) >= 2  # Step 4 and Step 8
        best = chk_mgr.get_best_checkpoint(exp.id)
        assert best is not None
