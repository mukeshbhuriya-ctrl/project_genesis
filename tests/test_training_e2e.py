"""
End-to-End integration test for Phase 5 Dataset & Fine-Tuning Subsystem,
integrated with Phase 4 Regression Detection and automated Rollback.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import EvalResult, EvalRun, RunStatus
from genesis.evaluation.regression import RegressionDetector
from genesis.training.datasets import DatasetFormat, DatasetManager, DatasetSample
from genesis.training.experiments import (
    ExperimentStatus,
    ExperimentTracker,
    Hyperparameters,
)
from genesis.training.export import CheckpointManager
from genesis.training.preprocessing import DataPreprocessor
from genesis.training.trainer import FineTuningTrainer


class TestTrainingE2EPipeline:
    """End-to-end integration tests for data curation, training, checkpointing, and evaluation defense."""

    @pytest.mark.asyncio
    async def test_full_finetuning_and_evaluation_guard_lifecycle(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)

        # 1. Initialize subsystem components with isolated database
        dataset_mgr = DatasetManager(db_path=tmp_db_path)
        preprocessor = DataPreprocessor(db_path=tmp_db_path)
        tracker = ExperimentTracker(db_path=tmp_db_path)
        trainer = FineTuningTrainer(db_path=tmp_db_path)
        chk_mgr = CheckpointManager(db_path=tmp_db_path)
        bench_runner = BenchmarkRunner(db_path=tmp_db_path)
        regression_detector = RegressionDetector(db_path=tmp_db_path)

        # 2. Ingest raw multi-format dataset containing duplicates, invalid syntax, and clean samples
        raw_samples = [
            # Clean instruction sample
            DatasetSample(
                instruction="Explain the difference between L1 and L2 regularization.",
                output="L1 regularization adds absolute weights penalty (sparsity), while L2 adds squared weights penalty.",
            ),
            # Duplicate sample (to test deduplication)
            DatasetSample(
                instruction="Explain the difference between L1 and L2 regularization.",
                output="L1 regularization adds absolute weights penalty (sparsity), while L2 adds squared weights penalty.",
            ),
            # Broken code sample (to test AST syntax validation)
            DatasetSample(
                instruction="Write a faulty function",
                output="```python\ndef broken(:\n    pass\n```",
            ),
            # Valid code sample
            DatasetSample(
                instruction="Implement quicksort partition",
                output="```python\ndef partition(arr, low, high):\n    pivot = arr[high]\n    i = low - 1\n    return i\n```",
            ),
            # Benchmark contaminated sample (matching bm-reasoning-v1)
            DatasetSample(
                instruction="If all transformers use self-attention, and Model Alpha is a transformer, does Model Alpha use self-attention?",
                output="Yes, it does.",
            ),
        ]
        for i in range(15):
            raw_samples.append(
                DatasetSample(
                    instruction=f"Reasoning query {i} on graph algorithms",
                    output=f"Algorithmic answer {i} detailing edge traversal and node complexity.",
                )
            )

        raw_dataset = dataset_mgr.register_dataset(
            name="genesis-reasoning-raw",
            samples=raw_samples,
            description="Raw dataset before quality preprocessing",
        )
        assert raw_dataset.sample_count == len(raw_samples)

        # 3. Clean and preprocess dataset
        cleaned_samples, report = preprocessor.clean_and_filter(
            samples=raw_samples,
            deduplicate=True,
            validate_syntax=True,
            check_contamination=True,
        )

        assert report.duplicates_removed >= 1
        assert report.syntax_errors_filtered >= 1
        assert report.contaminated_removed >= 1
        assert len(cleaned_samples) < len(raw_samples)

        clean_dataset = dataset_mgr.register_dataset(
            name="genesis-reasoning-clean",
            samples=cleaned_samples,
            description="Quality filtered, deduplicated, uncontaminated training dataset",
        )

        # 4. Partition dataset into deterministic splits
        split = dataset_mgr.split_dataset(cleaned_samples, train_ratio=0.8, val_ratio=0.2, seed=42)
        assert len(split.train) > 0
        assert len(split.val) > 0
        assert len(split.train) + len(split.val) + len(split.test) == len(cleaned_samples)

        # 5. Create fine-tuning experiment
        hp = Hyperparameters(
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=2,
            epochs=2,
            quantization="4bit",
        )
        experiment = tracker.create_experiment(
            name="exp-lora-reasoning-v1",
            base_model="qwen2.5:7b",
            dataset_id=clean_dataset.id,
            hyperparameters=hp,
        )
        assert experiment.status == ExperimentStatus.PENDING

        # 6. Execute fine-tuning loop asynchronously
        metrics_log = []
        completed_experiment = await trainer.run_training_async(
            experiment_id=experiment.id,
            checkpoint_interval=2,
            step_delay_s=0.005,
            metric_callback=lambda m: metrics_log.append(m),
        )

        assert completed_experiment.status == ExperimentStatus.COMPLETED
        assert completed_experiment.current_step == completed_experiment.total_steps
        assert completed_experiment.training_loss is not None
        assert completed_experiment.training_loss < 2.45
        assert len(metrics_log) == completed_experiment.total_steps

        # 7. Checkpoint selection and Modelfile export
        checkpoints = chk_mgr.list_checkpoints(experiment.id)
        assert len(checkpoints) >= 1
        best_chk = chk_mgr.get_best_checkpoint(experiment.id)
        assert best_chk is not None
        assert best_chk.is_best is True

        modelfile_path = chk_mgr.deploy_to_ollama(
            checkpoint_id=best_chk.id,
            deployed_name="qwen2.5-genesis:v1",
            base_model="qwen2.5:7b",
            system_prompt="You are an expert Genesis fine-tuned reasoning model.",
            parameters={"temperature": 0.2, "top_p": 0.9},
        )
        assert Path(modelfile_path).is_file()
        content = Path(modelfile_path).read_text(encoding="utf-8")
        assert "FROM qwen2.5:7b" in content
        assert "PARAMETER temperature 0.2" in content
        assert "PARAMETER top_p 0.9" in content
        assert "You are an expert Genesis fine-tuned reasoning model." in content

        # 8. Evaluate Candidate vs Baseline via Phase 4 Regression Detector
        bm = bench_runner.get_benchmark("bm-reasoning-v1")
        assert bm is not None

        # Simulate baseline run (score = 100.0)
        baseline_run = EvalRun(
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5:7b",
            status=RunStatus.COMPLETED,
            total_cases=len(bm.cases),
            passed_cases=len(bm.cases),
            failed_cases=0,
            score=100.0,
            duration_ms=1200.0,
            results=[
                EvalResult(case_id=c.id, passed=True, actual_output=c.expected_output, latency_ms=400.0)
                for c in bm.cases
            ],
        )
        bench_runner.save_run(baseline_run)

        # Simulate a candidate run with a regression on 1 test case (score drops to 66.7)
        candidate_run = EvalRun(
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5-genesis:v1",
            status=RunStatus.COMPLETED,
            total_cases=len(bm.cases),
            passed_cases=len(bm.cases) - 1,
            failed_cases=1,
            score=66.7,
            duration_ms=1300.0,
            results=[
                EvalResult(
                    case_id=bm.cases[0].id,
                    passed=False,
                    actual_output="Incorrect calculation",
                    latency_ms=450.0,
                ),
                EvalResult(case_id=bm.cases[1].id, passed=True, actual_output="Yes", latency_ms=420.0),
                EvalResult(case_id=bm.cases[2].id, passed=True, actual_output="Yes", latency_ms=430.0),
            ],
        )
        bench_runner.save_run(candidate_run)

        comparison = regression_detector.compare_runs(baseline_run, candidate_run)
        assert comparison.regressed is True
        assert comparison.verdict == "REGRESSED"
        assert len(comparison.regressed_cases) == 1

        # 9. Automated Rollback upon Regression Detection
        rollback_result = chk_mgr.rollback_model(
            experiment_id=experiment.id,
            baseline_model="qwen2.5:7b",
        )
        assert rollback_result["status"] == "rolled_back"
        assert rollback_result["active_baseline"] == "qwen2.5:7b"
        assert rollback_result["deployed_name"] == "qwen2.5-genesis:v1"
