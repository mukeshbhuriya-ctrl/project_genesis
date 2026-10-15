"""
REST API routes for Cross-Fleet Model Parameter Merging (Phase 12).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.training.merging import ModelMergeManager

router = APIRouter(prefix="/merging", tags=["Cross-Fleet Model Parameter Merging"])


class ExecuteMergeRequest(BaseModel):
    name: str = Field(..., description="Unique label for the merged model adapter")
    merge_method: str = Field(..., description="Merge algorithm: linear, slerp, ties, dare")
    target_model: str = Field(..., description="Base model name (e.g. qwen2.5:7b, qwen2.5-coder:7b)")
    source_adapters: list[str] = Field(..., min_length=1, description="List of adapter IDs to merge")
    weights: Optional[list[float]] = Field(default=None, description="Interpolation weights per adapter")
    density: Optional[float] = Field(default=1.0, ge=0.01, le=1.0, description="Pruning density parameter for TIES/DARE")
    adapter_payloads: Optional[list[dict[str, list[float]]]] = Field(default=None, description="Direct parameter tensors (optional)")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")


@router.get("/methods")
async def list_merge_methods():
    """List supported parameter merging strategies and documentation."""
    return {
        "methods": [
            {
                "id": "linear",
                "name": "Linear Weighted Averaging",
                "description": "Standard convex combination of model weights: theta_merged = sum(w_i * theta_i).",
                "requires_density": False,
                "min_sources": 1,
            },
            {
                "id": "slerp",
                "name": "Spherical Linear Interpolation (SLERP)",
                "description": "Geometric interpolation along hypersphere preserving vector magnitude and angles.",
                "requires_density": False,
                "min_sources": 2,
            },
            {
                "id": "ties",
                "name": "TIES-Merging (Trimming, Electing Signs & Disjoint Merge)",
                "description": "Resolves interference by trimming redundant delta weights, electing consensus signs, and disjoint merging.",
                "requires_density": True,
                "min_sources": 2,
            },
            {
                "id": "dare",
                "name": "DARE (Drop And REscale)",
                "description": "Randomly drops low-magnitude weights and rescales surviving parameters by 1/p before combining.",
                "requires_density": True,
                "min_sources": 2,
            },
        ]
    }


@router.post("/execute")
async def execute_model_merge(req: ExecuteMergeRequest):
    """Execute model parameter merge operation."""
    mgr = ModelMergeManager()
    try:
        recipe = mgr.execute_merge(
            name=req.name,
            merge_method=req.merge_method,
            target_model=req.target_model,
            source_adapters=req.source_adapters,
            weights=req.weights,
            density=req.density or 1.0,
            adapter_payloads=req.adapter_payloads,
            metadata=req.metadata,
        )
        return recipe.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge execution failed: {e}")


@router.get("/merges")
async def list_model_merges(limit: int = Query(default=50, ge=1, le=100)):
    """List historical parameter merge recipes."""
    mgr = ModelMergeManager()
    merges = mgr.list_merges(limit=limit)
    return [m.to_dict() for m in merges]


@router.get("/merges/{merge_id}")
async def get_model_merge(merge_id: str):
    """Get details of a specific parameter merge."""
    mgr = ModelMergeManager()
    recipe = mgr.get_merge(merge_id)
    if not recipe:
        raise HTTPException(status_code=404, detail=f"Merge recipe '{merge_id}' not found")
    return recipe.to_dict()


@router.delete("/merges/{merge_id}")
async def delete_model_merge(merge_id: str):
    """Evict a merge recipe and delete its disk artifact."""
    mgr = ModelMergeManager()
    deleted = mgr.delete_merge(merge_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Merge recipe '{merge_id}' not found")
    return {"id": merge_id, "status": "deleted"}
