"""
API routes for the Evaluation & Benchmarking Subsystem.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.evaluation.models import RunStatus
from genesis.evaluation.regression import RegressionDetector
from genesis.evaluation.reports import EvaluationReporter

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


def _get_runner() -> BenchmarkRunner:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return BenchmarkRunner(db_path=db_path)


def _get_detector() -> RegressionDetector:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return RegressionDetector(db_path=db_path)


class RunBenchmarkRequest(BaseModel):
    benchmark_id: str = Field(..., description="ID or name of benchmark suite to execute")
    model: str | None = Field(default=None, description="Model ID to evaluate (defaults to active configured model)")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=250, ge=1, le=4096, description="Max tokens per case")


class CompareRunsRequest(BaseModel):
    candidate_run_id: str = Field(..., description="Candidate evaluation run ID")
    baseline_run_id: str | None = Field(default=None, description="Optional baseline evaluation run ID")
    score_drop_tolerance: float = Field(default=0.0, ge=0.0, le=100.0, description="Tolerance for score drop")


@router.get("/benchmarks")
async def list_benchmarks():
    """List all available evaluation benchmarks."""
    runner = _get_runner()
    benchmarks = runner.list_benchmarks()
    return {
        "count": len(benchmarks),
        "benchmarks": [bm.to_dict() for bm in benchmarks],
    }


@router.get("/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """Retrieve full benchmark details and test cases."""
    runner = _get_runner()
    bm = runner.get_benchmark(benchmark_id)
    if not bm:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_id}' not found.")
    return bm.to_dict()


@router.post("/run")
async def run_benchmark(req: RunBenchmarkRequest):
    """Execute a benchmark suite against a target model."""
    runner = _get_runner()
    bm = runner.get_benchmark(req.benchmark_id)
    if not bm:
        raise HTTPException(status_code=404, detail=f"Benchmark '{req.benchmark_id}' not found.")

    model_name = req.model or get_settings().default_model
    run = await runner.run_benchmark_async(
        benchmark=bm,
        model_id=model_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return run.to_dict()


@router.get("/runs")
async def list_runs(limit: int = Query(50, ge=1, le=200)):
    """List historical evaluation runs ordered by recency."""
    runner = _get_runner()
    runs = runner.list_runs(limit=limit)
    return {
        "count": len(runs),
        "runs": [r.to_dict() for r in runs],
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Retrieve full evaluation run details, test case scores, and failure classifications."""
    runner = _get_runner()
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found.")
    return run.to_dict()


@router.get("/runs/{run_id}/report")
async def get_run_report(run_id: str):
    """Generate Markdown and JSON reports for a single evaluation run."""
    runner = _get_runner()
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found.")

    bm = runner.get_benchmark(run.benchmark_id)
    bm_name = bm.name if bm else run.benchmark_id
    markdown_content = EvaluationReporter.generate_run_markdown(run, benchmark_name=bm_name)

    return {
        "run_id": run.id,
        "markdown": markdown_content,
        "json": run.to_dict(),
    }


@router.post("/compare")
async def compare_runs(req: CompareRunsRequest):
    """Compare a candidate evaluation run against a baseline run or auto-detect prior baseline."""
    detector = _get_detector()
    try:
        comparison = detector.detect_regression_for_run(
            candidate_run_id=req.candidate_run_id,
            baseline_run_id=req.baseline_run_id,
            score_drop_tolerance=req.score_drop_tolerance,
        )
        report_md = EvaluationReporter.generate_comparison_markdown(comparison)
        return {
            "comparison": comparison.to_dict(),
            "report_markdown": report_md,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
