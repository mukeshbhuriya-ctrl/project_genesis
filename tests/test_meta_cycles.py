"""Unit tests for the autonomous meta-improvement closed-loop orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner, get_standard_benchmarks
from genesis.evaluation.models import Benchmark, EvalRun, RunStatus
from genesis.evaluation.regression import RegressionComparison, RegressionDetector
from genesis.research.meta import CycleStage, MetaCycleRecord, MetaResearchOrchestrator
from genesis.training.export import CheckpointManager


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_genesis.db"
    initialize_database(db_file)
    runner = BenchmarkRunner(db_path=db_file)
    for bm in get_standard_benchmarks():
        runner.save_benchmark(bm)
    return db_file


@pytest.fixture
def mock_runner(test_db: Path) -> BenchmarkRunner:
    runner = BenchmarkRunner(db_path=test_db)
    # Mock run_benchmark_async to return realistic EvalRun
    baseline_run = EvalRun(
        id="run-baseline-1",
        benchmark_id="bm-coding-v1",
        model_id="qwen2.5:7b",
        status=RunStatus.COMPLETED,
        total_cases=3,
        passed_cases=2,
        failed_cases=1,
        score=0.67,
    )
    runner.run_benchmark_async = AsyncMock(return_value=baseline_run)
    return runner


class TestMetaResearchOrchestrator:
    """Tests for closed-loop autonomous meta-improvement cycles."""

    @pytest.mark.asyncio
    async def test_run_successful_meta_cycle_promotion(self, test_db: Path, mock_runner: BenchmarkRunner):
        orchestrator = MetaResearchOrchestrator(
            db_path=test_db,
            runner=mock_runner,
        )

        cycle = await orchestrator.run_cycle_async(
            benchmark_id="bm-coding-v1",
            model_id="qwen2.5:7b",
            name="Autonomous Coding Improvement v1",
            total_train_steps=3,
        )

        assert isinstance(cycle, MetaCycleRecord)
        assert cycle.status == "completed"
        assert cycle.verdict == "promoted"
        assert cycle.dataset_id is not None
        assert cycle.experiment_id is not None
        assert cycle.checkpoint_id is not None
        assert len(cycle.stages) == 5

        # Check stage progression
        stage_names = [s["stage"] for s in cycle.stages]
        assert stage_names == ["diagnosis", "curation", "fine_tuning", "validation", "decision"]
        assert all(s["status"] == "completed" for s in cycle.stages)

        # Check SQLite persistence
        fetched = orchestrator.get_cycle(cycle.id)
        assert fetched is not None
        assert fetched.id == cycle.id
        assert fetched.status == "completed"
        assert fetched.verdict == "promoted"

    @pytest.mark.asyncio
    async def test_run_meta_cycle_regression_defense(self, test_db: Path, mock_runner: BenchmarkRunner):
        # Create orchestrator with mock regression detector that triggers a regression
        mock_regression = MagicMock(spec=RegressionDetector)
        mock_regression.compare_runs = MagicMock(
            return_value=RegressionComparison(
                baseline_run_id="run-1",
                candidate_run_id="run-2",
                benchmark_id="bm-coding-v1",
                baseline_model="qwen2.5:7b",
                candidate_model="candidate-1",
                baseline_score=0.80,
                candidate_score=0.50,
                score_delta=-0.30,
                regressed=True,
                verdict="REGRESSED",
                summary="Significant regression detected",
            )
        )

        orchestrator = MetaResearchOrchestrator(
            db_path=test_db,
            runner=mock_runner,
            regression_detector=mock_regression,
        )

        cycle = await orchestrator.run_cycle_async(
            benchmark_id="bm-coding-v1",
            model_id="qwen2.5:7b",
            name="Regressed Candidate Cycle",
            total_train_steps=2,
        )

        assert cycle.status == "regressed"
        assert cycle.verdict == "rolled_back"
        assert cycle.stages[-1]["stage"] == "decision"
        assert cycle.stages[-1]["details"]["decision"] == "rolled_back"

        # Verify persistence
        reloaded = orchestrator.get_cycle(cycle.id)
        assert reloaded is not None
        assert reloaded.verdict == "rolled_back"

    @pytest.mark.asyncio
    async def test_run_cycle_missing_benchmark(self, test_db: Path, mock_runner: BenchmarkRunner):
        orchestrator = MetaResearchOrchestrator(
            db_path=test_db,
            runner=mock_runner,
        )

        cycle = await orchestrator.run_cycle_async(
            benchmark_id="non-existent-benchmark",
            model_id="qwen2.5:7b",
        )

        assert cycle.status == "failed"
        assert cycle.verdict == "failed"

    def test_list_cycles(self, test_db: Path):
        orchestrator = MetaResearchOrchestrator(db_path=test_db)
        c1 = MetaCycleRecord(id="cycle-1", name="Cycle 1", benchmark_id="bm-coding-v1", model_id="qwen2.5:7b")
        c2 = MetaCycleRecord(id="cycle-2", name="Cycle 2", benchmark_id="bm-reasoning-v1", model_id="qwen2.5:7b")
        orchestrator._save_cycle(c1)
        orchestrator._save_cycle(c2)

        cycles = orchestrator.list_cycles(limit=10)
        assert len(cycles) >= 2
        ids = {c.id for c in cycles}
        assert "cycle-1" in ids
        assert "cycle-2" in ids
