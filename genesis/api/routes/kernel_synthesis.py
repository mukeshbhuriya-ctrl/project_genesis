"""
REST API routes for Autonomous LLM-Driven Kernel Synthesis & Verification (Phase 12).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.research.kernel_synthesis import (
    SEED_TEMPLATES,
    KernelSynthesisEngine,
)

router = APIRouter(prefix="/kernel-synthesis", tags=["Autonomous Kernel Synthesis"])


class RunSynthesisRequest(BaseModel):
    kernel_name: str = Field(..., description="Unique kernel identifier (e.g. gelu, rmsnorm, swiglu, dot_product)")
    operation_type: str = Field(..., description="Operation type: activation, quantization, normalization, reduction, gemm")
    initial_c_source: Optional[str] = Field(default=None, description="Initial C implementation or template override")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Max iterative repair passes")
    tolerance: float = Field(default=1e-4, ge=1e-8, le=1.0, description="Maximum allowed absolute error against ground truth")


@router.get("/templates")
async def list_seed_templates():
    """List available seed C kernel templates."""
    return {
        "templates": list(SEED_TEMPLATES.keys()),
        "sources": SEED_TEMPLATES,
    }


@router.post("/run")
async def run_kernel_synthesis(req: RunSynthesisRequest):
    """Run closed-loop kernel synthesis, compilation, verification, and reflective repair."""
    engine = KernelSynthesisEngine()
    try:
        experiment = engine.run_synthesis_loop(
            kernel_name=req.kernel_name,
            operation_type=req.operation_type,
            initial_c_source=req.initial_c_source,
            max_iterations=req.max_iterations,
            tolerance=req.tolerance,
        )
        return experiment.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kernel synthesis failed: {e}")


@router.get("/experiments")
async def list_kernel_experiments(limit: int = Query(default=50, ge=1, le=100)):
    """List historical kernel synthesis runs."""
    engine = KernelSynthesisEngine()
    experiments = engine.list_experiments(limit=limit)
    return [exp.to_dict() for exp in experiments]
