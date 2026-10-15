"""
API routes for autonomous meta-improvement cycles, dynamic routing, and benchmark expansion.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.evaluation.expansion import BenchmarkExpander
from genesis.models.router import ModelRouter, TASK_CATEGORIES
from genesis.research.meta import MetaResearchOrchestrator

router = APIRouter(prefix="/meta", tags=["Meta-Improvement"])


# ── Request / Response Schemas ──

class RunCycleRequest(BaseModel):
    benchmark_id: str = Field(default="bm-coding-v1", description="Target benchmark suite ID")
    model_id: Optional[str] = Field(default=None, description="Base model identifier to evaluate and improve")
    name: Optional[str] = Field(default=None, description="Optional custom name for this cycle")
    total_train_steps: int = Field(default=5, ge=1, le=50, description="Steps for LoRA fine-tuning")


class RouteRequest(BaseModel):
    task_prompt: str = Field(default="", description="Prompt or description of the task to route")
    task_category: Optional[str] = Field(default=None, description="Optional explicit category override")
    require_vram_mb: float = Field(default=0.0, description="Minimum free VRAM required in MB")


class UpdateRouteRequest(BaseModel):
    preferred_model: str
    fallback_model: str
    min_vram_mb: float = 0.0
    metadata: Optional[dict[str, Any]] = None


class ExpandBenchmarkRequest(BaseModel):
    benchmark_id: str = Field(default="bm-coding-v1", description="Benchmark ID to expand")
    domain: str = Field(default="coding", description="Capability domain: coding or reasoning")
    count: int = Field(default=3, ge=1, le=10, description="Number of edge cases to synthesize")


# ── Dependency Helpers ──

def _get_orchestrator() -> MetaResearchOrchestrator:
    return MetaResearchOrchestrator()


def _get_router() -> ModelRouter:
    return ModelRouter()


def _get_expander() -> BenchmarkExpander:
    return BenchmarkExpander()


# ── Meta-Improvement Cycles ──

@router.post("/cycles/run")
async def run_meta_cycle(req: RunCycleRequest):
    """Trigger a closed-loop autonomous meta-improvement cycle."""
    orchestrator = _get_orchestrator()
    try:
        cycle = await orchestrator.run_cycle_async(
            benchmark_id=req.benchmark_id,
            model_id=req.model_id,
            name=req.name,
            total_train_steps=req.total_train_steps,
        )
        return cycle.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cycle execution failed: {e}")


@router.get("/cycles")
def list_meta_cycles(limit: int = Query(default=20, ge=1, le=100)):
    """List historical autonomous meta-improvement cycles."""
    orchestrator = _get_orchestrator()
    cycles = orchestrator.list_cycles(limit=limit)
    return [c.to_dict() for c in cycles]


@router.get("/cycles/{cycle_id}")
def get_meta_cycle(cycle_id: str):
    """Get detailed status, metrics, and stage audit trace of a meta-cycle."""
    orchestrator = _get_orchestrator()
    cycle = orchestrator.get_cycle(cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail=f"Meta cycle '{cycle_id}' not found.")
    return cycle.to_dict()


# ── Dynamic Model Routing ──

@router.post("/route")
async def route_task(req: RouteRequest):
    """Dynamically route a task to the optimal model based on intent and VRAM headroom."""
    router_inst = _get_router()
    try:
        decision = await router_inst.route(
            task_prompt=req.task_prompt,
            task_category=req.task_category,
            require_vram_mb=req.require_vram_mb,
        )
        return decision.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {e}")


@router.get("/routes")
def list_model_routes():
    """List all configured model routes across task categories."""
    router_inst = _get_router()
    routes = router_inst.list_routes()
    return [r.to_dict() for r in routes]


@router.put("/routes/{category}")
def update_model_route(category: str, req: UpdateRouteRequest):
    """Update preferred model, fallback, and VRAM requirement for a task category."""
    if category not in TASK_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be one of {TASK_CATEGORIES}",
        )
    router_inst = _get_router()
    try:
        updated = router_inst.set_route(
            task_category=category,
            preferred_model=req.preferred_model,
            fallback_model=req.fallback_model,
            min_vram_mb=req.min_vram_mb,
            metadata=req.metadata,
        )
        return updated.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update route: {e}")


# ── Benchmark Expansion ──

@router.post("/benchmark/expand")
def expand_benchmark_suite(req: ExpandBenchmarkRequest):
    """Mine failure edge cases and expand benchmark with deterministic test cases."""
    expander = _get_expander()
    try:
        # First check if any real failed experiments exist to mine
        failures = expander.extract_failure_edge_cases(limit=req.count)
        mined_cases = []
        for fail in failures:
            c = expander.synthesize_case_from_failure(fail, target_benchmark_id=req.benchmark_id)
            if c:
                mined_cases.append(c)

        # Supplement with synthesized edge-case templates if needed
        needed = max(0, req.count - len(mined_cases))
        if needed > 0:
            synth_cases = expander.synthesize_deterministic_cases(
                domain=req.domain,
                count=needed,
                target_benchmark_id=req.benchmark_id,
            )
            mined_cases.extend(synth_cases)

        result = expander.expand_benchmark(
            benchmark_id=req.benchmark_id,
            new_cases=mined_cases[:req.count],
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark expansion failed: {e}")
