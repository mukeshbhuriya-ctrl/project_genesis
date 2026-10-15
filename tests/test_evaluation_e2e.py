"""
Live End-to-End Test for Phase 4 Acceptance Criteria:
"The platform can execute repeatable evaluation benchmarks against a local model,
classify failure modes into an actionable error taxonomy, detect regressions against
historical baselines, and output structured comparison reports."
"""

from __future__ import annotations

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import Benchmark, BenchmarkCategory, EvalCase, EvalType, RunStatus
from genesis.evaluation.regression import RegressionDetector
from genesis.evaluation.reports import EvaluationReporter
from genesis.models.ollama_backend import OllamaProvider


@pytest.mark.asyncio
async def test_live_model_evaluation_and_regression_e2e(tmp_path: Path):
    """
    Execute live benchmark against real local Ollama model (qwen2.5:7b or qwen2.5-coder:7b),
    record results in SQLite, test regression analysis, and generate Markdown & JSON reports.
    """
    provider = OllamaProvider()
    if not await provider.is_available():
        pytest.skip("Local Ollama service is not running on 127.0.0.1:11434")

    models = await provider.list_models()
    model_names = [m.name for m in models]
    target_model = None
    for candidate in ["qwen2.5:7b", "qwen2.5-coder:7b"]:
        if any(candidate in name for name in model_names):
            target_model = candidate
            break

    if not target_model:
        pytest.skip("Neither qwen2.5:7b nor qwen2.5-coder:7b is installed locally.")

    # 1. Initialize temporary isolated SQLite database
    db_file = tmp_path / "live_eval_test.db"
    initialize_database(db_file)

    runner = BenchmarkRunner(model_provider=provider, db_path=db_file)
    detector = RegressionDetector(db_path=db_file)

    # 2. Define a focused, fast live benchmark suite for real model grading
    live_bm = Benchmark(
        id="bm-live-verify-v1",
        name="Genesis Live Verification Suite",
        category=BenchmarkCategory.REASONING,
        description="Fast live reasoning verification for Project Genesis Phase 4.",
        cases=[
            EvalCase(
                id="case-live-1",
                benchmark_id="bm-live-verify-v1",
                name="Logical Deduction",
                prompt="If all transformers use self-attention, and Model Alpha is a transformer, does Model Alpha use self-attention? Answer with Yes or No.",
                expected_output="Yes",
                eval_type=EvalType.CONTAINS,
            ),
            EvalCase(
                id="case-live-2",
                benchmark_id="bm-live-verify-v1",
                name="Tool Math Calculation",
                prompt="What is (100 * 25) / 5? Reply with only the single final number.",
                expected_output="500",
                eval_type=EvalType.CONTAINS,
            ),
        ],
    )
    runner.save_benchmark(live_bm)

    # 3. Execute benchmark suite against live local model
    # Use max_tokens=60 for ultra-fast response
    run1 = await runner.run_benchmark_async(
        benchmark=live_bm,
        model_id=target_model,
        temperature=0.1,
        max_tokens=60,
    )

    assert run1.status == RunStatus.COMPLETED
    assert run1.total_cases == 2
    assert len(run1.results) == 2
    assert run1.duration_ms > 0
    assert run1.score >= 50.0  # Real 7B models score 50%-100% on these clear prompts

    # 4. Verify SQLite persistence
    stored_run = runner.get_run(run1.id)
    assert stored_run is not None
    assert stored_run.id == run1.id
    assert stored_run.model_id == target_model
    assert len(stored_run.results) == 2

    # 5. Verify Failure Taxonomy and Summary
    failure_summary = stored_run.metadata.get("failure_summary", {})
    assert "total_failures" in failure_summary
    assert "dominant_failure" in failure_summary
    assert "recommendation" in failure_summary

    # 6. Verify Regression Detection against baseline
    comparison = detector.detect_regression_for_run(run1.id)
    assert comparison.verdict in ("BASELINE_ESTABLISHED", "PASSED", "IMPROVED")

    # 7. Verify Markdown & JSON report generation
    report_md = EvaluationReporter.generate_run_markdown(stored_run, benchmark_name=live_bm.name)
    assert f"# Evaluation Report: {live_bm.name}" in report_md
    assert f"**Target Model:** `{target_model}`" in report_md
    assert "Overall Score" in report_md

    comp_md = EvaluationReporter.generate_comparison_markdown(comparison)
    assert "# Benchmark Regression Comparison:" in comp_md

    report_json = EvaluationReporter.export_run_json(stored_run)
    assert run1.id in report_json
