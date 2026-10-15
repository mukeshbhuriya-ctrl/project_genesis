"""
Automated regression detection and performance comparison between evaluation runs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import EvalResult, EvalRun, RunStatus

log = get_logger("evaluation.regression")


@dataclass
class RegressionComparison:
    """Comparison results between a candidate evaluation run and a baseline run."""
    baseline_run_id: str
    candidate_run_id: str
    benchmark_id: str
    baseline_model: str
    candidate_model: str
    baseline_score: float
    candidate_score: float
    score_delta: float
    regressed: bool
    verdict: str  # "PASSED", "IMPROVED", "REGRESSED", "BASELINE_ESTABLISHED"
    regressed_cases: list[dict[str, Any]] = field(default_factory=list)
    fixed_cases: list[dict[str, Any]] = field(default_factory=list)
    common_cases_count: int = 0
    baseline_duration_ms: float = 0.0
    candidate_duration_ms: float = 0.0
    latency_change_percent: float = 0.0
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "benchmark_id": self.benchmark_id,
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "baseline_score": round(self.baseline_score, 2),
            "candidate_score": round(self.candidate_score, 2),
            "score_delta": round(self.score_delta, 2),
            "regressed": self.regressed,
            "verdict": self.verdict,
            "regressed_cases": self.regressed_cases,
            "fixed_cases": self.fixed_cases,
            "common_cases_count": self.common_cases_count,
            "baseline_duration_ms": round(self.baseline_duration_ms, 2),
            "candidate_duration_ms": round(self.candidate_duration_ms, 2),
            "latency_change_percent": round(self.latency_change_percent, 1),
            "summary": self.summary,
            "created_at": self.created_at,
        }


class RegressionDetector:
    """
    Detects regressions in model capabilities, accuracy drops, or latency spikes
    across benchmark runs.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(get_settings().database_url)
        self.runner = BenchmarkRunner(db_path=self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    def compare_runs(
        self,
        baseline: EvalRun,
        candidate: EvalRun,
        score_drop_tolerance: float = 0.0,
        latency_spike_threshold: float = 50.0,
    ) -> RegressionComparison:
        """
        Compare candidate evaluation run against a baseline run.

        Parameters
        ----------
        baseline : EvalRun
            The reference baseline run.
        candidate : EvalRun
            The newly evaluated candidate run.
        score_drop_tolerance : float
            Allowable score drop percentage (default: 0.0%).
        latency_spike_threshold : float
            Percentage increase in total latency that triggers regression warning (default: 50.0%).
        """
        score_delta = candidate.score - baseline.score

        # Map results by case_id
        base_results: dict[str, EvalResult] = {r.case_id: r for r in baseline.results}
        cand_results: dict[str, EvalResult] = {r.case_id: r for r in candidate.results}

        regressed_cases: list[dict[str, Any]] = []
        fixed_cases: list[dict[str, Any]] = []
        common_cases = 0

        for case_id, c_res in cand_results.items():
            if case_id in base_results:
                common_cases += 1
                b_res = base_results[case_id]

                # Case passed in baseline but failed in candidate -> Regression
                if b_res.passed and not c_res.passed:
                    regressed_cases.append({
                        "case_id": case_id,
                        "baseline_score": b_res.score,
                        "candidate_score": c_res.score,
                        "failure_category": c_res.failure_category.value,
                        "actual_output": c_res.actual_output,
                        "error": c_res.error,
                        "baseline_latency_ms": b_res.latency_ms,
                        "candidate_latency_ms": c_res.latency_ms,
                    })

                # Case failed in baseline but passed in candidate -> Improvement
                elif not b_res.passed and c_res.passed:
                    fixed_cases.append({
                        "case_id": case_id,
                        "baseline_score": b_res.score,
                        "candidate_score": c_res.score,
                        "actual_output": c_res.actual_output,
                    })

        # Latency calculation
        lat_change_pct = 0.0
        if baseline.duration_ms > 0:
            lat_change_pct = ((candidate.duration_ms - baseline.duration_ms) / baseline.duration_ms) * 100.0

        # Determine if regressed
        is_regressed = False
        if score_delta < -score_drop_tolerance:
            is_regressed = True
        elif len(regressed_cases) > 0:
            is_regressed = True
        elif lat_change_pct > latency_spike_threshold and candidate.score <= baseline.score:
            is_regressed = True

        # Determine verdict
        if is_regressed:
            verdict = "REGRESSED"
            summary = (
                f"Regression detected: Candidate score {candidate.score:.1f}% vs baseline "
                f"{baseline.score:.1f}% (delta: {score_delta:+.1f}%). "
                f"{len(regressed_cases)} previously passing case(s) failed."
            )
        elif score_delta > 0 or len(fixed_cases) > 0:
            verdict = "IMPROVED"
            summary = (
                f"Improvement observed: Candidate score {candidate.score:.1f}% vs baseline "
                f"{baseline.score:.1f}% (delta: {score_delta:+.1f}%). "
                f"{len(fixed_cases)} case(s) newly resolved."
            )
        else:
            verdict = "PASSED"
            summary = (
                f"Candidate matched baseline: Score {candidate.score:.1f}% vs {baseline.score:.1f}%. "
                f"Zero regressions across {common_cases} test cases."
            )

        return RegressionComparison(
            baseline_run_id=baseline.id,
            candidate_run_id=candidate.id,
            benchmark_id=candidate.benchmark_id,
            baseline_model=baseline.model_id,
            candidate_model=candidate.model_id,
            baseline_score=baseline.score,
            candidate_score=candidate.score,
            score_delta=score_delta,
            regressed=is_regressed,
            verdict=verdict,
            regressed_cases=regressed_cases,
            fixed_cases=fixed_cases,
            common_cases_count=common_cases,
            baseline_duration_ms=baseline.duration_ms,
            candidate_duration_ms=candidate.duration_ms,
            latency_change_percent=lat_change_pct,
            summary=summary,
        )

    def detect_regression_for_run(
        self,
        candidate_run_id: str,
        baseline_run_id: str | None = None,
        score_drop_tolerance: float = 0.0,
    ) -> RegressionComparison:
        """
        Compare a run from the database against a specific or auto-detected baseline.
        """
        candidate = self.runner.get_run(candidate_run_id)
        if not candidate:
            raise ValueError(f"Candidate run '{candidate_run_id}' not found in database.")

        if baseline_run_id:
            baseline = self.runner.get_run(baseline_run_id)
            if not baseline:
                raise ValueError(f"Baseline run '{baseline_run_id}' not found in database.")
        else:
            # Auto-detect baseline: most recent completed run of the same benchmark prior to candidate
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT id FROM eval_runs
                    WHERE benchmark_id = ?
                      AND status = 'completed'
                      AND id != ?
                      AND started_at <= ?
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (candidate.benchmark_id, candidate.id, candidate.started_at),
                )
                row = cursor.fetchone()
                if not row:
                    # If no earlier run, look for any completed run for this benchmark
                    cursor = conn.execute(
                        """
                        SELECT id FROM eval_runs
                        WHERE benchmark_id = ?
                          AND status = 'completed'
                          AND id != ?
                        ORDER BY started_at DESC
                        LIMIT 1
                        """,
                        (candidate.benchmark_id, candidate.id),
                    )
                    row = cursor.fetchone()
            finally:
                conn.close()

            if not row:
                # No baseline exists yet; this run establishes the baseline
                return RegressionComparison(
                    baseline_run_id=candidate.id,
                    candidate_run_id=candidate.id,
                    benchmark_id=candidate.benchmark_id,
                    baseline_model=candidate.model_id,
                    candidate_model=candidate.model_id,
                    baseline_score=candidate.score,
                    candidate_score=candidate.score,
                    score_delta=0.0,
                    regressed=False,
                    verdict="BASELINE_ESTABLISHED",
                    regressed_cases=[],
                    fixed_cases=[],
                    common_cases_count=candidate.total_cases,
                    baseline_duration_ms=candidate.duration_ms,
                    candidate_duration_ms=candidate.duration_ms,
                    latency_change_percent=0.0,
                    summary=f"First completed run for benchmark '{candidate.benchmark_id}'. Baseline established at {candidate.score:.1f}%.",
                )
            baseline = self.runner.get_run(row["id"])
            if not baseline:
                raise ValueError(f"Failed to load baseline run '{row['id']}'.")

        return self.compare_runs(baseline, candidate, score_drop_tolerance=score_drop_tolerance)
