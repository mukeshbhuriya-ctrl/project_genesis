"""Unit tests for the automated benchmark expansion module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genesis.core.database import get_sync_connection, initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner, get_standard_benchmarks
from genesis.evaluation.expansion import BenchmarkExpander, ExpansionResult
from genesis.evaluation.models import Benchmark, BenchmarkCategory, EvalCase, EvalType


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_genesis.db"
    initialize_database(db_file)
    # Seed standard benchmarks into database
    runner = BenchmarkRunner(db_path=db_file)
    for bm in get_standard_benchmarks():
        runner.save_benchmark(bm)
    return db_file


class TestBenchmarkExpander:
    """Tests for mining failures and expanding evaluation benchmarks."""

    def test_synthesize_deterministic_cases(self, test_db: Path):
        expander = BenchmarkExpander(db_path=test_db)
        cases = expander.synthesize_deterministic_cases(domain="coding", count=3, target_benchmark_id="bm-coding-v1")

        assert len(cases) == 3
        for c in cases:
            assert c.benchmark_id == "bm-coding-v1"
            assert c.prompt != ""
            assert c.expected_output != ""
            assert isinstance(c.eval_type, EvalType)

    def test_extract_and_synthesize_from_code_failure(self, test_db: Path):
        # Insert a simulated failed code experiment in SQLite
        conn = get_sync_connection(test_db)
        try:
            conn.execute(
                """
                INSERT INTO code_experiments (id, name, objective, language, initial_code, final_code, tests, status, error)
                VALUES ('exp-fail-1', 'Binary Search Boundary', 'Implement a function named `binary_search` with empty list checks', 'python', 'def binary_search(): pass', '', 'assert binary_search([], 5) == -1', 'failed', 'AssertionError: assert None == -1')
                """
            )
            conn.commit()
        finally:
            conn.close()

        expander = BenchmarkExpander(db_path=test_db)
        failures = expander.extract_failure_edge_cases(limit=5)
        assert len(failures) >= 1
        code_fail = [f for f in failures if f.get("source") == "code_experiment"][0]

        eval_case = expander.synthesize_case_from_failure(code_fail, target_benchmark_id="bm-coding-v1")
        assert eval_case is not None
        assert "binary_search" in eval_case.prompt
        assert eval_case.expected_output == "def binary_search"
        assert eval_case.eval_type == EvalType.CONTAINS

    def test_expand_existing_benchmark(self, test_db: Path):
        expander = BenchmarkExpander(db_path=test_db)
        initial_bm = expander.runner.get_benchmark("bm-coding-v1")
        assert initial_bm is not None
        initial_case_count = len(initial_bm.cases)

        new_cases = [
            EvalCase(
                id="case-new-edge-1",
                benchmark_id="bm-coding-v1",
                name="Custom Boundary Edge Case",
                prompt="Write a Python function named `clip(x, low, high)`. Return only the code.",
                expected_output="def clip",
                eval_type=EvalType.CONTAINS,
            )
        ]

        result = expander.expand_benchmark("bm-coding-v1", new_cases)
        assert isinstance(result, ExpansionResult)
        assert result.added_cases_count == 1
        assert result.total_cases_count == initial_case_count + 1

        # Verify persistence in SQLite
        reloaded_bm = expander.runner.get_benchmark("bm-coding-v1")
        assert reloaded_bm is not None
        assert len(reloaded_bm.cases) == initial_case_count + 1
        assert any(c.name == "Custom Boundary Edge Case" for c in reloaded_bm.cases)

    def test_create_expanded_suite(self, test_db: Path):
        expander = BenchmarkExpander(db_path=test_db)
        extra_cases = expander.synthesize_deterministic_cases(domain="reasoning", count=2, target_benchmark_id="bm-reasoning-expanded")

        suite = expander.create_expanded_suite(
            source_benchmark_id="bm-reasoning-v1",
            new_suite_id="bm-reasoning-expanded",
            new_suite_name="Genesis Reasoning Expanded Benchmark",
            extra_cases=extra_cases,
        )

        assert suite.id == "bm-reasoning-expanded"
        assert len(suite.cases) == 3 + 2  # 3 standard + 2 extra
        assert suite.category == BenchmarkCategory.REASONING

        # Check retrieval from DB
        fetched = expander.runner.get_benchmark("bm-reasoning-expanded")
        assert fetched is not None
        assert len(fetched.cases) == 5
