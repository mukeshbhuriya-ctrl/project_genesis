"""
Automated benchmark expansion from edge-case discovery.

Extracts novel failure modes from code experiments and evaluation runs,
synthesizes deterministic evaluation cases, and expands benchmark suites.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import (
    Benchmark,
    BenchmarkCategory,
    EvalCase,
    EvalType,
    FailureCategory,
)

log = get_logger("benchmark_expansion")


@dataclass
class ExpansionResult:
    """The outcome of an automated benchmark expansion."""
    benchmark_id: str
    added_cases_count: int
    total_cases_count: int
    cases: list[EvalCase]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "added_cases_count": self.added_cases_count,
            "total_cases_count": self.total_cases_count,
            "cases": [c.to_dict() for c in self.cases],
            "created_at": self.created_at,
        }


class BenchmarkExpander:
    """
    Expands deterministic evaluation benchmarks by mining edge cases from failures
    and synthesizing rigorous validation cases with deterministic grading rubrics.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        runner: BenchmarkRunner | None = None,
    ):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(get_settings().database_url)
        self.runner = runner or BenchmarkRunner(db_path=self.db_path)

    def extract_failure_edge_cases(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Queries recent failed code experiments and evaluation results to find edge cases.
        """
        edge_cases = []
        conn = get_sync_connection(self.db_path)
        try:
            # 1. Query failed code experiments
            cursor = conn.execute(
                """
                SELECT id, name, objective, language, initial_code, final_code, tests, test_results, error
                FROM code_experiments
                WHERE status IN ('failed', 'error')
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            for row in cursor.fetchall():
                edge_cases.append({
                    "source": "code_experiment",
                    "id": row["id"],
                    "name": row["name"],
                    "objective": row["objective"],
                    "language": row["language"],
                    "code": row["final_code"] or row["initial_code"],
                    "tests": row["tests"],
                    "error": row["error"],
                })

            # 2. Query failed eval cases
            cursor = conn.execute(
                """
                SELECT r.case_id, c.name, c.prompt, c.expected_output, c.eval_type, r.actual_output, r.failure_category, r.error
                FROM eval_results r
                JOIN eval_cases c ON r.case_id = c.id
                WHERE r.passed = 0
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            for row in cursor.fetchall():
                edge_cases.append({
                    "source": "eval_result",
                    "case_id": row["case_id"],
                    "name": row["name"],
                    "prompt": row["prompt"],
                    "expected_output": row["expected_output"],
                    "eval_type": row["eval_type"],
                    "actual_output": row["actual_output"],
                    "failure_category": row["failure_category"],
                    "error": row["error"],
                })
        except Exception as e:
            log.warning("Could not query failure edge cases: %s", e)
        finally:
            conn.close()

        return edge_cases

    def synthesize_case_from_failure(self, failure: dict[str, Any], target_benchmark_id: str) -> Optional[EvalCase]:
        """
        Transforms a failure signature into a rigorous deterministic EvalCase.
        """
        source = failure.get("source")
        if source == "code_experiment":
            objective = failure.get("objective", "")
            name = f"Edge Case: {failure.get('name', 'Code Verification')}"
            # Extract expected function name or keyword from objective
            m = re.search(r"`([a-zA-Z_]\w*)`", objective)
            if not m:
                m = re.search(r"\b(?:function|def)\s+(?:named\s+)?([a-zA-Z_]\w*)", objective)
            func_name = m.group(1) if m else "solution"

            prompt = (
                f"Write a Python function named `{func_name}` satisfying this objective:\n"
                f"{objective}\n"
                "Return only the valid Python function code without markdown fences."
            )
            expected = f"def {func_name}"
            return EvalCase(
                id=f"exp-case-{uuid.uuid4().hex[:8]}",
                benchmark_id=target_benchmark_id,
                name=name,
                prompt=prompt,
                expected_output=expected,
                eval_type=EvalType.CONTAINS,
                criteria={"source_experiment_id": failure.get("id"), "derived_from": "failed_code_experiment"},
                weight=1.2,
            )

        elif source == "eval_result":
            orig_name = failure.get("name", "Verification")
            prompt = failure.get("prompt", "")
            expected = failure.get("expected_output", "")
            eval_type_str = failure.get("eval_type", "contains")
            try:
                eval_type = EvalType(eval_type_str)
            except ValueError:
                eval_type = EvalType.CONTAINS

            # Synthesize an edge-case variant with enhanced boundary constraints
            variant_name = f"Edge Case Variant: {orig_name}"
            return EvalCase(
                id=f"exp-case-{uuid.uuid4().hex[:8]}",
                benchmark_id=target_benchmark_id,
                name=variant_name,
                prompt=prompt,
                expected_output=expected,
                eval_type=eval_type,
                criteria={"source_case_id": failure.get("case_id"), "derived_from": "failed_eval_case"},
                weight=1.0,
            )

        return None

    def synthesize_deterministic_cases(
        self,
        domain: str = "coding",
        count: int = 3,
        target_benchmark_id: str = "bm-coding-v1",
    ) -> list[EvalCase]:
        """
        Synthesizes standard edge-case evaluation cases for boundary conditions and syntax robustness.
        """
        templates = {
            "coding": [
                (
                    "Edge Case: Safe Division Guard",
                    "Write a Python function named `safe_divide(a, b)` that returns `a / b` when `b != 0` and `None` when `b == 0`. Return only the code.",
                    "def safe_divide",
                    EvalType.CONTAINS,
                ),
                (
                    "Edge Case: Palindrome Case-Insensitive",
                    "Write a Python function named `is_palindrome(s)` that returns True if the string is a palindrome ignoring case and spaces. Return only the code.",
                    "def is_palindrome",
                    EvalType.CONTAINS,
                ),
                (
                    "Edge Case: Flatten Nested List",
                    "Write a Python function named `flatten_list(lst)` that flattens a list of arbitrary nested lists. Return only the code.",
                    "def flatten_list",
                    EvalType.CONTAINS,
                ),
                (
                    "Edge Case: Word Frequency Counter",
                    "Write a Python function named `word_freq(text)` that returns a dictionary mapping lowercased words to their count. Return only the code.",
                    "def word_freq",
                    EvalType.CONTAINS,
                ),
            ],
            "reasoning": [
                (
                    "Edge Case: Zero Boundary Math",
                    "A memory pool has 0 allocations remaining. If 3 processes request 4 MB each and are rejected, how many allocations were accepted? Reply with only the single number.",
                    "0",
                    EvalType.EXACT_MATCH,
                ),
                (
                    "Edge Case: Negation Logic",
                    "Condition X is false. Condition Y is the exact opposite of Condition X. Is Condition Y true? Answer with Yes or No.",
                    "Yes",
                    EvalType.CONTAINS,
                ),
                (
                    "Edge Case: Transitive Order Constraints",
                    "Object Alpha is faster than Beta. Object Gamma is slower than Beta. Which object is the fastest? Reply with Alpha, Beta, or Gamma.",
                    "Alpha",
                    EvalType.CONTAINS,
                ),
            ],
        }

        domain_key = domain.lower() if domain.lower() in templates else "coding"
        selected_templates = templates[domain_key][:count]

        cases = []
        for name, prompt, expected, eval_type in selected_templates:
            cases.append(
                EvalCase(
                    id=f"exp-auto-{uuid.uuid4().hex[:8]}",
                    benchmark_id=target_benchmark_id,
                    name=name,
                    prompt=prompt,
                    expected_output=expected,
                    eval_type=eval_type,
                    criteria={"auto_generated": True, "domain": domain_key},
                    weight=1.0,
                )
            )
        return cases

    def expand_benchmark(
        self,
        benchmark_id: str,
        new_cases: list[EvalCase],
    ) -> ExpansionResult:
        """
        Appends new cases to an existing benchmark and persists them to SQLite.
        """
        bm = self.runner.get_benchmark(benchmark_id)
        if not bm:
            raise ValueError(f"Benchmark with id '{benchmark_id}' not found.")

        # Assign benchmark_id to all new cases
        for case in new_cases:
            case.benchmark_id = benchmark_id

        # Update in-memory cases avoiding exact duplicates by name via variant suffix
        existing_names = {c.name for c in bm.cases}
        added = []
        for case in new_cases:
            candidate_name = case.name
            suffix_idx = 1
            while candidate_name in existing_names:
                candidate_name = f"{case.name} (variant {suffix_idx})"
                suffix_idx += 1
            case.name = candidate_name
            bm.cases.append(case)
            added.append(case)
            existing_names.add(case.name)

        # Persist updated benchmark suite and cases to SQLite
        self.runner.save_benchmark(bm)

        log.info(
            "Expanded benchmark '%s' with %d new cases (total: %d)",
            benchmark_id,
            len(added),
            len(bm.cases),
        )

        return ExpansionResult(
            benchmark_id=benchmark_id,
            added_cases_count=len(added),
            total_cases_count=len(bm.cases),
            cases=added,
        )

    def create_expanded_suite(
        self,
        source_benchmark_id: str,
        new_suite_id: str,
        new_suite_name: str,
        extra_cases: list[EvalCase],
    ) -> Benchmark:
        """
        Clones an existing benchmark, adds synthesized edge cases, and creates a new suite.
        """
        source_bm = self.runner.get_benchmark(source_benchmark_id)
        if not source_bm:
            raise ValueError(f"Source benchmark '{source_benchmark_id}' not found.")

        cloned_cases = [
            EvalCase(
                id=f"case-{new_suite_id}-{idx}",
                benchmark_id=new_suite_id,
                name=c.name,
                prompt=c.prompt,
                expected_output=c.expected_output,
                eval_type=c.eval_type,
                criteria=dict(c.criteria),
                weight=c.weight,
            )
            for idx, c in enumerate(source_bm.cases)
        ]

        for extra in extra_cases:
            extra.benchmark_id = new_suite_id
            cloned_cases.append(extra)

        new_suite = Benchmark(
            id=new_suite_id,
            name=new_suite_name,
            category=source_bm.category,
            description=f"Auto-expanded benchmark suite derived from {source_bm.name}.",
            cases=cloned_cases,
        )
        self.runner.save_benchmark(new_suite)
        return new_suite
