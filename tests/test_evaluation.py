"""
Unit and integration tests for Genesis Evaluation Subsystem.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from genesis.core.database import initialize_database
from genesis.evaluation.analysis import FailureAnalyzer
from genesis.evaluation.benchmarks import BenchmarkRunner, get_standard_benchmarks
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
from genesis.models import GenerationResponse, ModelProvider


class DummyModelProvider(ModelProvider):
    """Test model provider returning configurable responses."""

    def __init__(self, response_map: dict[str, str] | None = None, default_response: str = "40"):
        self.response_map = response_map or {}
        self.default_response = default_response

    async def generate(self, request):
        prompt = request.messages[-1]["content"]
        content = self.response_map.get(prompt, self.default_response)
        return GenerationResponse(
            content=content,
            model=request.model or "dummy-model",
            token_count_prompt=10,
            token_count_completion=5,
        )

    async def list_models(self):
        return []

    async def is_available(self):
        return True

    async def generate_stream(self, request):
        yield self.default_response

    async def get_model_info(self, model_id: str):
        return None

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


class TestFailureAnalyzer:
    """Tests for FailureAnalyzer error taxonomy classification."""

    def test_classify_timeout(self):
        case = EvalCase(name="Timeout Test", eval_type=EvalType.CONTAINS)
        cat = FailureAnalyzer.classify_failure(case, "", error="Operation timed out after 30 seconds")
        assert cat == FailureCategory.TIMEOUT

    def test_classify_syntax_error(self):
        case = EvalCase(name="Code Test", eval_type=EvalType.CONTAINS)
        cat = FailureAnalyzer.classify_failure(case, "", error="SyntaxError: invalid syntax at line 2")
        assert cat == FailureCategory.SYNTAX_ERROR

    def test_classify_tool_misuse(self):
        case = EvalCase(name="Tool Test", eval_type=EvalType.TOOL_CALL)
        cat = FailureAnalyzer.classify_failure(case, "I think you should use a calculator", error=None)
        assert cat == FailureCategory.TOOL_MISUSE

    def test_classify_format_error_on_empty(self):
        case = EvalCase(name="Empty Test", eval_type=EvalType.EXACT_MATCH)
        cat = FailureAnalyzer.classify_failure(case, "   ", error=None)
        assert cat == FailureCategory.FORMAT_ERROR

    def test_classify_numeric_format_error(self):
        case = EvalCase(name="Numeric Test", eval_type=EvalType.NUMERIC_TOLERANCE)
        cat = FailureAnalyzer.classify_failure(case, "no number here at all", error=None)
        assert cat == FailureCategory.FORMAT_ERROR

    def test_summarize_failures(self):
        results = [
            EvalResult(passed=True, failure_category=FailureCategory.NONE),
            EvalResult(passed=False, failure_category=FailureCategory.SYNTAX_ERROR),
            EvalResult(passed=False, failure_category=FailureCategory.SYNTAX_ERROR),
            EvalResult(passed=False, failure_category=FailureCategory.TIMEOUT),
        ]
        summary = FailureAnalyzer.summarize_failures(results)
        assert summary["total_failures"] == 3
        assert summary["dominant_failure"] == FailureCategory.SYNTAX_ERROR.value
        assert summary["distribution"][FailureCategory.SYNTAX_ERROR.value] == 2
        assert "syntax verification" in summary["recommendation"]


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner execution, grading, and persistence."""

    def test_standard_benchmarks_seeded(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        benchmarks = runner.list_benchmarks()

        assert len(benchmarks) >= 3
        bm_ids = {bm.id for bm in benchmarks}
        assert "bm-reasoning-v1" in bm_ids
        assert "bm-coding-v1" in bm_ids
        assert "bm-tools-v1" in bm_ids

    def test_grading_exact_match(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        case = EvalCase(expected_output="Genesis", eval_type=EvalType.EXACT_MATCH)

        passed, score = runner.grade_response(case, "Genesis")
        assert passed is True
        assert score == 1.0

        passed, score = runner.grade_response(case, "genesis")
        assert passed is True  # case insensitive
        assert score == 1.0

        passed, score = runner.grade_response(case, "Genesis Platform")
        assert passed is False
        assert score == 0.0

    def test_grading_contains(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        case = EvalCase(expected_output="40", eval_type=EvalType.CONTAINS)

        passed, score = runner.grade_response(case, "The answer is 40 GB of VRAM.")
        assert passed is True
        assert score == 1.0

        passed, score = runner.grade_response(case, "The answer is 32.")
        assert passed is False
        assert score == 0.0

    def test_grading_regex(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        case = EvalCase(expected_output=r"def\s+fib\(n\):", eval_type=EvalType.REGEX)

        passed, score = runner.grade_response(case, "def  fib(n):\n    return n")
        assert passed is True
        assert score == 1.0

        passed, score = runner.grade_response(case, "function fib(n) { return n; }")
        assert passed is False
        assert score == 0.0

    def test_grading_numeric_tolerance(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        case = EvalCase(expected_output="10.0", eval_type=EvalType.NUMERIC_TOLERANCE, criteria={"tolerance": 0.5})

        passed, score = runner.grade_response(case, "Computed value: 10.3")
        assert passed is True
        assert score == 1.0

        passed, score = runner.grade_response(case, "Computed value: 11.2")
        assert passed is False
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_run_benchmark_async_mock(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        provider = DummyModelProvider(
            response_map={
                "A server cluster has 4 nodes. Each node has 2 GPUs with 8 GB VRAM each. If 3 GPUs are dedicated to training and the rest to inference, how many GB of VRAM are available for inference? Reply with only the final number.": "40",
                "If all transformers use self-attention, and Model Alpha is a transformer, does Model Alpha use self-attention? Answer with Yes or No.": "Yes, absolutely.",
                "Box A is heavier than Box B. Box B is heavier than Box C. Is Box C lighter than Box A? Answer with Yes or No.": "Yes.",
            }
        )
        runner = BenchmarkRunner(model_provider=provider, db_path=tmp_db_path)
        bm = runner.get_benchmark("bm-reasoning-v1")
        assert bm is not None

        run = await runner.run_benchmark_async(bm, model_id="dummy-test-model")
        assert run.status == RunStatus.COMPLETED
        assert run.total_cases == 3
        assert run.passed_cases == 3
        assert run.failed_cases == 0
        assert run.score == 100.0

        # Verify run persistence in SQLite
        saved_run = runner.get_run(run.id)
        assert saved_run is not None
        assert saved_run.id == run.id
        assert saved_run.score == 100.0
        assert len(saved_run.results) == 3
        assert all(r.passed for r in saved_run.results)
