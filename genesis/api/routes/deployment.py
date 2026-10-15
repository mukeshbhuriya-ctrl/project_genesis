"""
API routes for Hardware and Deployment Evolution Subsystem (Phase 7).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.core.device import DeviceOrchestrator
from genesis.core.queue import JobQueue

router = APIRouter(prefix="/deployment", tags=["Deployment"])


def _get_orchestrator() -> DeviceOrchestrator:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return DeviceOrchestrator(db_path=db_path)


def _get_queue() -> JobQueue:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return JobQueue(db_path=db_path)


class LayerPlanRequest(BaseModel):
    model_name: str = Field(default="qwen2.5:7b", description="Model identifier")
    total_layers: int = Field(default=28, ge=1, le=128, description="Number of transformer layers")
    parameter_size_b: float = Field(default=7.0, ge=0.5, le=140.0, description="Model parameter size in billions")
    quant_bits: int = Field(default=4, ge=2, le=16, description="Quantization bit depth")
    context_tokens: int = Field(default=4096, ge=512, le=131072, description="Target context window length")
    safety_reserve_mb: float = Field(default=1024.0, ge=0.0, description="Reserved VRAM buffer in MB")


class SubmitJobRequest(BaseModel):
    job_type: str = Field(
        ...,
        description="Type of background job: 'fine_tune', 'eval_benchmark', 'code_synthesis', 'research_synthesis', or 'custom'",
    )
    payload: dict[str, Any] = Field(default_factory=dict, description="Job execution payload parameters")
    priority: int = Field(default=10, ge=0, le=100, description="Priority integer (higher executed first)")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum automated retry attempts")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional custom metadata tags")


@router.get("/topology")
async def get_device_topology(
    refresh: bool = Query(default=False, description="Force hardware rescan instead of cached snapshot")
) -> dict[str, Any]:
    """Retrieve discovered GPU and CPU compute device topology and memory metrics."""
    orchestrator = _get_orchestrator()
    if not refresh:
        latest = orchestrator.get_latest_topology()
        if latest:
            return latest.to_dict()

    topology = orchestrator.discover_topology()
    return topology.to_dict()


@router.post("/layer-plan")
async def calculate_layer_distribution_plan(req: LayerPlanRequest) -> dict[str, Any]:
    """Calculate optimal model layer split across available GPUs and CPU offload."""
    orchestrator = _get_orchestrator()
    plan = orchestrator.calculate_layer_distribution(
        model_name=req.model_name,
        total_layers=req.total_layers,
        parameter_size_b=req.parameter_size_b,
        quant_bits=req.quant_bits,
        context_tokens=req.context_tokens,
        safety_reserve_mb=req.safety_reserve_mb,
    )
    return plan.to_dict()


@router.post("/jobs")
async def enqueue_work_job(req: SubmitJobRequest) -> dict[str, Any]:
    """Enqueue a long-running background job into the persistent worker queue."""
    valid_types = {"fine_tune", "eval_benchmark", "code_synthesis", "research_synthesis", "custom"}
    if req.job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job_type '{req.job_type}'. Must be one of: {sorted(valid_types)}",
        )

    queue = _get_queue()
    job = queue.submit(
        job_type=req.job_type,
        payload=req.payload,
        priority=req.priority,
        max_retries=req.max_retries,
        metadata=req.metadata,
    )
    return job.to_dict()


@router.get("/jobs")
async def list_work_jobs(
    status: str | None = Query(default=None, description="Filter by status ('queued', 'leased', 'running', 'completed', 'failed', 'cancelled')"),
    job_type: str | None = Query(default=None, description="Filter by job_type"),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """List work jobs with optional status and job type filtering."""
    queue = _get_queue()
    jobs = queue.list_jobs(status=status, job_type=job_type, limit=limit)
    return [j.to_dict() for j in jobs]


@router.get("/jobs/{job_id}")
async def get_work_job(job_id: str) -> dict[str, Any]:
    """Retrieve details, progress, and result of a specific work job."""
    queue = _get_queue()
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
async def cancel_work_job(job_id: str) -> dict[str, Any]:
    """Cancel a pending, leased, or active work job."""
    queue = _get_queue()
    cancelled = queue.cancel(job_id)
    if not cancelled:
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job '{job_id}' with status '{job.status}'.",
        )

    updated_job = queue.get_job(job_id)
    return {
        "success": True,
        "message": f"Job {job_id} cancelled successfully.",
        "job": updated_job.to_dict() if updated_job else None,
    }
