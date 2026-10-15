"""
API routes for the Research & Capability Expansion Subsystem (Phase 6).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.models.context import ContextManager
from genesis.models.profiler import InferenceProfiler
from genesis.research.coding import ReflectiveCodeSynthesizer
from genesis.research.synthesis import AdaptiveKnowledgeSynthesizer

router = APIRouter(prefix="/research", tags=["Research"])


def _get_context_manager() -> ContextManager:
    settings = get_settings()
    return ContextManager(default_model=settings.default_model)


def _get_profiler() -> InferenceProfiler:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return InferenceProfiler(db_path=db_path)


def _get_code_synthesizer() -> ReflectiveCodeSynthesizer:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    # Prefer coding model if installed, else fallback to default
    return ReflectiveCodeSynthesizer(db_path=db_path, model_id="qwen2.5-coder:7b")


def _get_synthesizer() -> AdaptiveKnowledgeSynthesizer:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return AdaptiveKnowledgeSynthesizer(db_path=db_path, model_id=settings.default_model)


class ProfileInferenceRequest(BaseModel):
    model_id: str | None = Field(default=None, description="Model to profile (defaults to active)")
    prompt: str = Field(
        default="Explain the QuickSort algorithm and its average time complexity in 3 short sentences.",
        description="Prompt to benchmark",
    )
    max_tokens: int = Field(default=128, ge=16, le=2048, description="Max tokens to generate")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class SynthesizeCodeRequest(BaseModel):
    objective: str = Field(..., description="Coding task objective to implement and verify")
    name: str | None = Field(default=None, description="Optional experiment name")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max self-repair attempts")
    export_dataset_id: str | None = Field(
        default=None, description="Optional Phase 5 training dataset ID to append verified code"
    )


class KnowledgeSynthesisRequest(BaseModel):
    topic: str = Field(..., description="Research topic or question to synthesize")
    query: str | None = Field(default=None, description="Optional search query override")
    max_sources: int = Field(default=5, ge=1, le=10, description="Max grounding sources to retrieve")
    save_to_memory: bool = Field(default=True, description="Save synthesis brief to long-term memory")


# ── 1. Context & VRAM Budgeting ──

@router.get("/context-budget")
async def get_context_budget(
    model_id: str | None = Query(default=None, description="Model ID to evaluate"),
    context_tokens: int = Query(default=4096, ge=512, le=65536, description="Target context tokens"),
    available_vram_mb: float | None = Query(default=None, description="Available GPU VRAM in MB"),
):
    """Calculate KV-cache VRAM consumption, headroom, and safety status."""
    cm = _get_context_manager()
    budget = cm.get_budget(
        context_tokens=context_tokens,
        model_name=model_id,
        available_vram_mb=available_vram_mb,
    )
    return budget.to_dict()


# ── 2. Inference Profiler ──

@router.post("/profile")
async def run_inference_profile(req: ProfileInferenceRequest):
    """Run live inference profiling measuring TTFT, throughput, inter-token latency, and VRAM."""
    settings = get_settings()
    target_model = req.model_id or settings.default_model
    profiler = _get_profiler()

    result = await profiler.profile(
        model_id=target_model,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return result.to_dict()


@router.get("/profiles")
async def list_inference_profiles(limit: int = Query(default=20, ge=1, le=100)):
    """List recent inference profiling runs."""
    profiler = _get_profiler()
    profiles = profiler.list_profiles(limit=limit)
    return {
        "count": len(profiles),
        "profiles": [p.to_dict() for p in profiles],
    }


# ── 3. Code Synthesis & Self-Repair ──

@router.post("/coding/synthesize")
async def synthesize_code(req: SynthesizeCodeRequest):
    """Execute autonomous code synthesis and reflective sandbox repair."""
    synthesizer = _get_code_synthesizer()
    record = await synthesizer.synthesize_and_repair(
        objective=req.objective,
        name=req.name,
        max_iterations=req.max_iterations,
        export_dataset_id=req.export_dataset_id,
    )
    return record.to_dict()


@router.get("/coding/experiments")
async def list_code_experiments(limit: int = Query(default=20, ge=1, le=100)):
    """List code synthesis experiments and their sandbox test results."""
    synthesizer = _get_code_synthesizer()
    experiments = synthesizer.list_experiments(limit=limit)
    return {
        "count": len(experiments),
        "experiments": [e.to_dict() for e in experiments],
    }


# ── 4. Knowledge Synthesis & Citations ──

@router.post("/synthesize")
async def run_knowledge_synthesis(req: KnowledgeSynthesisRequest):
    """Execute multi-source research synthesis with citation verification."""
    synthesizer = _get_synthesizer()
    brief = await synthesizer.synthesize(
        topic=req.topic,
        query=req.query,
        max_sources=req.max_sources,
        save_to_memory=req.save_to_memory,
    )
    return brief.to_dict()


@router.get("/syntheses")
async def list_knowledge_syntheses(limit: int = Query(default=20, ge=1, le=100)):
    """List past research synthesis briefs."""
    synthesizer = _get_synthesizer()
    briefs = synthesizer.list_syntheses(limit=limit)
    return {
        "count": len(briefs),
        "syntheses": [b.to_dict() for b in briefs],
    }
