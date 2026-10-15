"""
REST API routes for Autonomous Recursive Self-Architecting & Block Pruning (Phase 14).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.models.pruning import (
    KNOWN_ARCH_CONFIGS,
    PruningStrategy,
    RecursiveModelArchitect,
)

router = APIRouter(prefix="/pruning", tags=["Autonomous Structural Pruning & NAS"])


class CreatePlanRequest(BaseModel):
    model_name: str = Field(default="qwen2.5:7b", description="Model architecture identifier")
    target_vram_mb: Optional[float] = Field(default=None, description="Target VRAM ceiling in MB (e.g. 5500 for RTX 2000 Ada)")
    strategy: PruningStrategy = Field(default="structured_head", description="Head sensitivity ranking strategy")
    head_prune_ratio: float = Field(default=0.25, ge=0.0, lt=1.0, description="Fraction of attention heads to prune if target VRAM not specified")
    mlp_prune_ratio: float = Field(default=0.85, gt=0.0, le=1.0, description="SwiGLU intermediate dimension scaling ratio")


class ExecutePruningRequest(BaseModel):
    model_name: str = Field(default="qwen2.5:7b", description="Model architecture identifier")
    target_vram_mb: Optional[float] = Field(default=None, description="Target VRAM ceiling in MB")
    strategy: PruningStrategy = Field(default="structured_head", description="Pruning strategy")
    head_prune_ratio: float = Field(default=0.25, ge=0.0, lt=1.0, description="Fraction of attention heads to prune")
    mlp_prune_ratio: float = Field(default=0.85, gt=0.0, le=1.0, description="Intermediate dimension scaling ratio")
    experiment_id: Optional[str] = Field(default=None, description="Optional custom experiment ID")


@router.get("/architectures")
async def list_known_architectures():
    """List supported model architectures and baseline configurations."""
    return {
        name: {
            "name": cfg.name,
            "layers": cfg.layers,
            "hidden_dim": cfg.hidden_dim,
            "attention_heads": cfg.attention_heads,
            "kv_heads": cfg.kv_heads,
            "head_dim": cfg.head_dim,
            "intermediate_dim": cfg.intermediate_dim,
            "vocab_size": cfg.vocab_size,
            "bytes_per_param": cfg.bytes_per_param,
        }
        for name, cfg in KNOWN_ARCH_CONFIGS.items()
    }


@router.post("/plan")
async def create_pruning_plan(req: CreatePlanRequest):
    """Generate a structured pruning blueprint with parameter and VRAM estimates."""
    architect = RecursiveModelArchitect()
    try:
        plan = architect.create_pruning_plan(
            model_name=req.model_name,
            target_vram_mb=req.target_vram_mb,
            strategy=req.strategy,
            head_prune_ratio=req.head_prune_ratio,
            mlp_prune_ratio=req.mlp_prune_ratio,
        )
        return plan.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate pruning plan: {e}")


@router.post("/execute")
async def execute_pruning(req: ExecutePruningRequest):
    """Execute and persist a model structural pruning experiment."""
    architect = RecursiveModelArchitect()
    try:
        plan = architect.create_pruning_plan(
            model_name=req.model_name,
            target_vram_mb=req.target_vram_mb,
            strategy=req.strategy,
            head_prune_ratio=req.head_prune_ratio,
            mlp_prune_ratio=req.mlp_prune_ratio,
        )
        record = architect.execute_pruning(plan=plan, experiment_id=req.experiment_id)
        return record.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pruning execution failed: {e}")


@router.get("/experiments")
async def list_pruning_experiments(limit: int = Query(default=20, ge=1, le=100)):
    """List historical structural pruning experiments."""
    architect = RecursiveModelArchitect()
    records = architect.list_experiments(limit=limit)
    return [r.to_dict() for r in records]


@router.get("/experiments/{exp_id}")
async def get_pruning_experiment(exp_id: str):
    """Retrieve details for a specific pruning experiment."""
    architect = RecursiveModelArchitect()
    record = architect.get_experiment(exp_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Pruning experiment '{exp_id}' not found")
    return record.to_dict()
