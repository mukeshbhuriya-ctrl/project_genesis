"""
Standard baseline benchmarks and evaluation test harness for Project Genesis.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.evaluation.analysis import FailureAnalyzer
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
from genesis.models import GenerationRequest, ModelProvider
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("evaluation")


# ── Built-in Standard Baseline Benchmarks ──

def get_standard_benchmarks() -> list[Benchmark]:
    """Return standard baseline benchmark suites for Project Genesis."""
    # 1. Reasoning Baseline Benchmark
    b_reasoning = Benchmark(
        id="bm-reasoning-v1",
        name="Genesis Reasoning Baseline v1",
        category=BenchmarkCategory.REASONING,
        description="Core multi-step logical deduction, spatial reasoning, and constraint satisfaction.",
        cases=[
            EvalCase(
                id="case-reas-1",
                benchmark_id="bm-reasoning-v1",
                name="Multi-step Math",
                prompt="A server cluster has 4 nodes. Each node has 2 GPUs with 8 GB VRAM each. If 3 GPUs are dedicated to training and the rest to inference, how many GB of VRAM are available for inference? Reply with only the final number.",
                expected_output="40",
                eval_type=EvalType.CONTAINS,
            ),
            EvalCase(
                id="case-reas-2",
                benchmark_id="bm-reasoning-v1",
                name="Logical Deduction",
                prompt="If all transformers use self-attention, and Model Alpha is a transformer, does Model Alpha use self-attention? Answer with Yes or No.",
                expected_output="Yes",
                eval_type=EvalType.CONTAINS,
            ),
            EvalCase(
                id="case-reas-3",
                benchmark_id="bm-reasoning-v1",
                name="Syllogism Constraints",
                prompt="Box A is heavier than Box B. Box B is heavier than Box C. Is Box C lighter than Box A? Answer with Yes or No.",
                expected_output="Yes",
                eval_type=EvalType.CONTAINS,
            ),
        ],
    )

    # 2. Coding Baseline Benchmark
    b_coding = Benchmark(
        id="bm-coding-v1",
        name="Genesis Python Coding Baseline v1",
        category=BenchmarkCategory.CODING,
        description="Algorithmic Python implementations, string transformations, and data structures.",
        cases=[
            EvalCase(
                id="case-code-1",
                benchmark_id="bm-coding-v1",
                name="Fibonacci Function",
                prompt="Write a Python function named `fib` that returns the n-th Fibonacci number (where fib(0)=0, fib(1)=1, fib(6)=8). Return only the function code.",
                expected_output="def fib",
                eval_type=EvalType.CONTAINS,
            ),
            EvalCase(
                id="case-code-2",
                benchmark_id="bm-coding-v1",
                name="Palindrome Checker",
                prompt="Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is identical forwards and backwards. Return only the function definition.",
                expected_output="def is_palindrome",
                eval_type=EvalType.CONTAINS,
            ),
            EvalCase(
                id="case-code-3",
                benchmark_id="bm-coding-v1",
                name="List Deduplication",
                prompt="Write a single-line Python expression using a set or dict to deduplicate a list `lst` while preserving order. Mention `dict.fromkeys` or similar.",
                expected_output="fromkeys",
                eval_type=EvalType.CONTAINS,
            ),
        ],
    )

    # 3. Tool Invocation Baseline Benchmark
    b_tools = Benchmark(
        id="bm-tools-v1",
        name="Genesis Tool Invocation Baseline v1",
        category=BenchmarkCategory.TOOLS,
        description="Tool command formatting, argument structuring, and math calculation requests.",
        cases=[
            EvalCase(
                id="case-tool-1",
                benchmark_id="bm-tools-v1",
                name="Calculator Parameterization",
                prompt="Calculate (256 * 1024) / 4. Output the result directly as a single integer.",
                expected_output="65536",
                eval_type=EvalType.CONTAINS,
            ),
            EvalCase(
                id="case-tool-2",
                benchmark_id="bm-tools-v1",
                name="Filesystem Command Identification",
                prompt="Which tool in Project Genesis is used to inspect files without mutating them: read_file or write_file? Output only the tool name.",
                expected_output="read_file",
                eval_type=EvalType.CONTAINS,
            ),
        ],
    )

    return [b_reasoning, b_coding, b_tools]


# ── Benchmark Evaluation Runner & Persistence ──

class BenchmarkRunner:
    """
    Executes benchmark suites against model backends, grades responses, and records metrics.
    """

    def __init__(
        self,
        model_provider: ModelProvider | None = None,
        db_path: Path | str | None = None,
    ):
        self.provider = model_provider or OllamaProvider()
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(get_settings().database_url)

        self._ensure_standard_benchmarks()

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    def _ensure_standard_benchmarks(self) -> None:
        """Seed built-in standard benchmarks into SQLite if not already present."""
        try:
            for bm in get_standard_benchmarks():
                self.save_benchmark(bm)
        except Exception as exc:
            log.warning(f"Could not seed standard benchmarks: {exc}")

    # ── Benchmark CRUD ──

    def save_benchmark(self, bm: Benchmark) -> None:
        """Insert or update a benchmark suite and its test cases in SQLite."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO eval_benchmarks (id, name, category, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """,
                (bm.id, bm.name, bm.category.value, bm.description, bm.created_at, bm.updated_at),
            )

            for case in bm.cases:
                conn.execute(
                    """
                    INSERT INTO eval_cases (
                        id, benchmark_id, name, prompt, expected_output, eval_type, criteria, weight, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        prompt = excluded.prompt,
                        expected_output = excluded.expected_output,
                        eval_type = excluded.eval_type,
                        criteria = excluded.criteria,
                        weight = excluded.weight
                    """,
                    (
                        case.id,
                        bm.id,
                        case.name,
                        case.prompt,
                        case.expected_output,
                        case.eval_type.value,
                        json.dumps(case.criteria),
                        case.weight,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_benchmark(self, benchmark_id_or_name: str) -> Benchmark | None:
        """Retrieve a benchmark by ID or Name along with its cases."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM eval_benchmarks WHERE id = ? OR name = ?",
                (benchmark_id_or_name, benchmark_id_or_name),
            )
            row = cursor.fetchone()
            if not row:
                return None

            bm = Benchmark(
                id=row["id"],
                name=row["name"],
                category=BenchmarkCategory(row["category"]),
                description=row["description"] or "",
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

            case_cursor = conn.execute(
                "SELECT * FROM eval_cases WHERE benchmark_id = ? ORDER BY id ASC",
                (bm.id,),
            )
            cases = []
            for c_row in case_cursor.fetchall():
                cases.append(
                    EvalCase(
                        id=c_row["id"],
                        benchmark_id=c_row["benchmark_id"],
                        name=c_row["name"],
                        prompt=c_row["prompt"],
                        expected_output=c_row["expected_output"] or "",
                        eval_type=EvalType(c_row["eval_type"]),
                        criteria=json.loads(c_row["criteria"] or "{}"),
                        weight=c_row["weight"],
                    )
                )
            bm.cases = cases
            return bm
        finally:
            conn.close()

    def list_benchmarks(self) -> list[Benchmark]:
        """List all available benchmarks."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT id FROM eval_benchmarks ORDER BY name ASC")
            ids = [row["id"] for row in cursor.fetchall()]
        finally:
            conn.close()

        benchmarks = []
        for b_id in ids:
            bm = self.get_benchmark(b_id)
            if bm:
                benchmarks.append(bm)
        return benchmarks

    # ── Evaluation Execution ──

    def grade_response(self, case: EvalCase, actual_output: str) -> tuple[bool, float]:
        """
        Grade a model response against the case expectations.
        Returns (passed: bool, score: float between 0.0 and 1.0).
        """
        clean_actual = actual_output.strip()
        expected = case.expected_output.strip()

        if case.eval_type == EvalType.EXACT_MATCH:
            passed = clean_actual.lower() == expected.lower()
            return passed, 1.0 if passed else 0.0

        elif case.eval_type == EvalType.CONTAINS:
            passed = expected.lower() in clean_actual.lower()
            return passed, 1.0 if passed else 0.0

        elif case.eval_type == EvalType.REGEX:
            match = re.search(expected, clean_actual, re.IGNORECASE)
            passed = bool(match)
            return passed, 1.0 if passed else 0.0

        elif case.eval_type == EvalType.NUMERIC_TOLERANCE:
            tolerance = float(case.criteria.get("tolerance", 0.01))
            try:
                # Find first number in actual output
                nums = re.findall(r"[-+]?\d*\.?\d+", clean_actual)
                if not nums:
                    return False, 0.0
                val = float(nums[0])
                exp_val = float(expected)
                passed = abs(val - exp_val) <= tolerance
                return passed, 1.0 if passed else 0.0
            except ValueError:
                return False, 0.0

        elif case.eval_type == EvalType.TOOL_CALL:
            expected_tool = case.criteria.get("tool_name", expected)
            passed = expected_tool.lower() in clean_actual.lower()
            return passed, 1.0 if passed else 0.0

        return False, 0.0

    async def run_benchmark_async(
        self,
        benchmark: Benchmark,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 250,
    ) -> EvalRun:
        """
        Execute an entire benchmark suite against a model, grade cases, and persist results.
        """
        run = EvalRun(
            benchmark_id=benchmark.id,
            model_id=model_id,
            status=RunStatus.RUNNING,
            total_cases=len(benchmark.cases),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.save_run(run)

        total_score_weight = 0.0
        earned_score_weight = 0.0
        results: list[EvalResult] = []
        run_start_time = time.perf_counter()

        for case in benchmark.cases:
            t0 = time.perf_counter()
            actual_output = ""
            error_msg: str | None = None
            tokens_gen = 0

            try:
                req = GenerationRequest(
                    messages=[{"role": "user", "content": case.prompt}],
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = await self.provider.generate(req)
                actual_output = response.content
                tokens_gen = response.token_count_completion
            except Exception as exc:
                error_msg = str(exc)
                log.error(f"Error evaluating case '{case.name}': {exc}")

            duration_ms = (time.perf_counter() - t0) * 1000

            passed, score = self.grade_response(case, actual_output) if not error_msg else (False, 0.0)
            
            # Failure classification
            fail_cat = FailureCategory.NONE if passed else FailureAnalyzer.classify_failure(case, actual_output, error_msg)

            res = EvalResult(
                run_id=run.id,
                case_id=case.id,
                passed=passed,
                score=score,
                actual_output=actual_output,
                error=error_msg,
                latency_ms=duration_ms,
                tokens_generated=tokens_gen,
                failure_category=fail_cat,
            )
            results.append(res)

            total_score_weight += case.weight
            earned_score_weight += (score * case.weight)

        run_duration_ms = (time.perf_counter() - run_start_time) * 1000

        run.results = results
        run.passed_cases = sum(1 for r in results if r.passed)
        run.failed_cases = sum(1 for r in results if not r.passed)
        run.score = (earned_score_weight / total_score_weight * 100.0) if total_score_weight > 0 else 0.0
        run.duration_ms = run_duration_ms
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc).isoformat()

        # Add failure summary into metadata
        run.metadata = {
            "failure_summary": FailureAnalyzer.summarize_failures(results),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        self.save_run(run)
        return run

    # ── Run Persistence ──

    def save_run(self, run: EvalRun) -> None:
        """Save evaluation run and all case results into SQLite."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO eval_runs (
                    id, benchmark_id, model_id, status, total_cases, passed_cases, failed_cases,
                    score, duration_ms, started_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    total_cases = excluded.total_cases,
                    passed_cases = excluded.passed_cases,
                    failed_cases = excluded.failed_cases,
                    score = excluded.score,
                    duration_ms = excluded.duration_ms,
                    completed_at = excluded.completed_at,
                    metadata = excluded.metadata
                """,
                (
                    run.id,
                    run.benchmark_id,
                    run.model_id,
                    run.status.value,
                    run.total_cases,
                    run.passed_cases,
                    run.failed_cases,
                    run.score,
                    run.duration_ms,
                    run.started_at,
                    run.completed_at,
                    json.dumps(run.metadata),
                ),
            )

            for res in run.results:
                conn.execute(
                    """
                    INSERT INTO eval_results (
                        id, run_id, case_id, passed, score, actual_output, error, latency_ms,
                        tokens_generated, failure_category, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        passed = excluded.passed,
                        score = excluded.score,
                        actual_output = excluded.actual_output,
                        error = excluded.error,
                        latency_ms = excluded.latency_ms,
                        tokens_generated = excluded.tokens_generated,
                        failure_category = excluded.failure_category
                    """,
                    (
                        res.id,
                        run.id,
                        res.case_id,
                        1 if res.passed else 0,
                        res.score,
                        res.actual_output,
                        res.error,
                        res.latency_ms,
                        res.tokens_generated,
                        res.failure_category.value,
                        res.created_at,
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    def get_run(self, run_id: str) -> EvalRun | None:
        """Retrieve an evaluation run with all results by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None

            run = EvalRun(
                id=row["id"],
                benchmark_id=row["benchmark_id"],
                model_id=row["model_id"],
                status=RunStatus(row["status"]),
                total_cases=row["total_cases"],
                passed_cases=row["passed_cases"],
                failed_cases=row["failed_cases"],
                score=row["score"],
                duration_ms=row["duration_ms"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )

            res_cursor = conn.execute("SELECT * FROM eval_results WHERE run_id = ?", (run.id,))
            results = []
            for r_row in res_cursor.fetchall():
                results.append(
                    EvalResult(
                        id=r_row["id"],
                        run_id=r_row["run_id"],
                        case_id=r_row["case_id"],
                        passed=bool(r_row["passed"]),
                        score=r_row["score"],
                        actual_output=r_row["actual_output"] or "",
                        error=r_row["error"],
                        latency_ms=r_row["latency_ms"],
                        tokens_generated=r_row["tokens_generated"],
                        failure_category=FailureCategory(r_row["failure_category"]),
                        created_at=r_row["created_at"],
                    )
                )
            run.results = results
            return run
        finally:
            conn.close()

    def list_runs(self, limit: int = 50) -> list[EvalRun]:
        """List historical evaluation runs ordered by date descending."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT id FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,))
            ids = [row["id"] for row in cursor.fetchall()]
        finally:
            conn.close()

        runs = []
        for r_id in ids:
            r = self.get_run(r_id)
            if r:
                runs.append(r)
        return runs
