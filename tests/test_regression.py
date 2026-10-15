"""
Tests for regression detection, benchmark comparisons, and evaluation reporting.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import (
    Benchmark,
    BenchmarkCategory,
    EvalCase,
    EvalResult,
    EvalRun,
    EvalType,
    FailureCategory,
    RunStatus,
)
from genesis.evaluation.regression import RegressionComparison, RegressionDetector
from genesis.evaluation.reports import EvaluationReporter


class TestRegressionDetector:
    """Tests for comparing candidate evaluation runs against baseline runs."""

    def test_compare_runs_identical_passes(self):
        detector = RegressionDetector()
        res1 = EvalResult(case_id="c1", passed=True, score=1.0, latency_ms=100.0)
        res2 = EvalResult(case_id="c2", passed=True, score=1.0, latency_ms=100.0)

        base = EvalRun(
            id="base-1",
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5:7b",
            score=100.0,
            duration_ms=200.0,
            results=[res1, res2],
        )

        cand = EvalRun(
            id="cand-1",
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5:7b",
            score=100.0,
            duration_ms=210.0,
            results=[res1, res2],
        )

        comp = detector.compare_runs(base, cand)
        assert comp.regressed is False
        assert comp.verdict == "PASSED"
        assert comp.score_delta == 0.0
        assert len(comp.regressed_cases) == 0

    def test_compare_runs_detects_regression(self):
        detector = RegressionDetector()
        b_res1 = EvalResult(case_id="c1", passed=True, score=1.0)
        b_res2 = EvalResult(case_id="c2", passed=True, score=1.0)

        c_res1 = EvalResult(case_id="c1", passed=True, score=1.0)
        c_res2 = EvalResult(
            case_id="c2",
            passed=False,
            score=0.0,
            failure_category=FailureCategory.INCORRECT_ANSWER,
            actual_output="wrong answer",
        )

        base = EvalRun(
            id="base-1",
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5:7b",
            score=100.0,
            duration_ms=200.0,
            results=[b_res1, b_res2],
        )

        cand = EvalRun(
            id="cand-1",
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5-coder:7b",
            score=50.0,
            duration_ms=220.0,
            results=[c_res1, c_res2],
        )

        comp = detector.compare_runs(base, cand)
        assert comp.regressed is True
        assert comp.verdict == "REGRESSED"
        assert comp.score_delta == -50.0
        assert len(comp.regressed_cases) == 1
        assert comp.regressed_cases[0]["case_id"] == "c2"

    def test_compare_runs_detects_improvement(self):
        detector = RegressionDetector()
        b_res1 = EvalResult(case_id="c1", passed=True, score=1.0)
        b_res2 = EvalResult(case_id="c2", passed=False, score=0.0)

        c_res1 = EvalResult(case_id="c1", passed=True, score=1.0)
        c_res2 = EvalResult(case_id="c2", passed=True, score=1.0)

        base = EvalRun(
            id="base-1",
            benchmark_id="bm-reasoning-v1",
            model_id="model-v1",
            score=50.0,
            duration_ms=300.0,
            results=[b_res1, b_res2],
        )

        cand = EvalRun(
            id="cand-1",
            benchmark_id="bm-reasoning-v1",
            model_id="model-v2",
            score=100.0,
            duration_ms=250.0,
            results=[c_res1, c_res2],
        )

        comp = detector.compare_runs(base, cand)
        assert comp.regressed is False
        assert comp.verdict == "IMPROVED"
        assert comp.score_delta == 50.0
        assert len(comp.fixed_cases) == 1
        assert comp.fixed_cases[0]["case_id"] == "c2"

    def test_auto_detect_baseline_from_db(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        detector = RegressionDetector(db_path=tmp_db_path)

        bm = runner.get_benchmark("bm-reasoning-v1")
        assert bm is not None

        # First run establishes baseline
        run1 = EvalRun(
            id="run-base-1",
            benchmark_id=bm.id,
            model_id="qwen2.5:7b",
            status=RunStatus.COMPLETED,
            total_cases=1,
            passed_cases=1,
            score=100.0,
            started_at="2026-10-15T01:00:00Z",
            results=[EvalResult(case_id="case-reas-1", passed=True, score=1.0)],
        )
        runner.save_run(run1)

        # First run compared against itself or no prior -> BASELINE_ESTABLISHED
        comp1 = detector.detect_regression_for_run("run-base-1")
        assert comp1.verdict == "BASELINE_ESTABLISHED"

        # Second run should auto-detect run1 as baseline
        run2 = EvalRun(
            id="run-cand-2",
            benchmark_id=bm.id,
            model_id="qwen2.5-coder:7b",
            status=RunStatus.COMPLETED,
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            score=0.0,
            started_at="2026-10-15T02:00:00Z",
            results=[EvalResult(case_id="case-reas-1", passed=False, score=0.0)],
        )
        runner.save_run(run2)

        comp2 = detector.detect_regression_for_run("run-cand-2")
        assert comp2.baseline_run_id == "run-base-1"
        assert comp2.candidate_run_id == "run-cand-2"
        assert comp2.regressed is True
        assert comp2.verdict == "REGRESSED"


class TestEvaluationReporter:
    """Tests for Markdown, JSON, and disk report generation."""

    def test_generate_run_markdown(self):
        run = EvalRun(
            id="run-test-md",
            benchmark_id="bm-reasoning-v1",
            model_id="qwen2.5:7b",
            status=RunStatus.COMPLETED,
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            score=50.0,
            duration_ms=450.0,
            started_at="2026-10-15T01:00:00Z",
            metadata={
                "failure_summary": {
                    "dominant_failure": "incorrect_answer",
                    "recommendation": "Augment memory context.",
                    "distribution": {"incorrect_answer": 1},
                    "distribution_percentages": {"incorrect_answer": 100.0},
                }
            },
            results=[
                EvalResult(case_id="c1", passed=True, score=1.0, latency_ms=200.0),
                EvalResult(
                    case_id="c2",
                    passed=False,
                    score=0.0,
                    latency_ms=250.0,
                    failure_category=FailureCategory.INCORRECT_ANSWER,
                    error="Wrong answer",
                ),
            ],
        )

        md = EvaluationReporter.generate_run_markdown(run, benchmark_name="Reasoning Test")
        assert "# Evaluation Report: Reasoning Test" in md
        assert "**Overall Score** | **50.00%**" in md
        assert "`incorrect_answer`" in md
        assert "Augment memory context." in md
        assert "❌ Fail" in md

    def test_generate_comparison_markdown(self):
        comp = RegressionComparison(
            baseline_run_id="base-1",
            candidate_run_id="cand-1",
            benchmark_id="bm-tools-v1",
            baseline_model="qwen2.5:7b",
            candidate_model="qwen2.5:7b",
            baseline_score=100.0,
            candidate_score=50.0,
            score_delta=-50.0,
            regressed=True,
            verdict="REGRESSED",
            summary="Regression detected.",
            regressed_cases=[
                {
                    "case_id": "case-tool-1",
                    "baseline_score": 1.0,
                    "candidate_score": 0.0,
                    "failure_category": "format_error",
                    "actual_output": "bad json",
                }
            ],
        )

        md = EvaluationReporter.generate_comparison_markdown(comp)
        assert "# Benchmark Regression Comparison: `bm-tools-v1`" in md
        assert "🚨 REGRESSION DETECTED" in md
        assert "-50.00%" in md
        assert "`case-tool-1`" in md
