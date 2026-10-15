"""
Continual Learning and EWC REST API endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.training.continual import (
    EWCConsolidator,
    PrefixTuningManager,
    ReplayBufferManager,
)

router = APIRouter(prefix="/api/continual", tags=["continual-learning"])


class AddReplaySampleRequest(BaseModel):
    sample_input: str
    sample_target: str
    task_domain: str = Field(default="general")
    priority: float = Field(default=1.0, ge=0.01)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EWCConsolidateRequest(BaseModel):
    task_name: str
    weights: dict[str, list[float]]
    importance_lambda: float = Field(default=1000.0, ge=0.1)
    experiment_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EWCPenaltyRequest(BaseModel):
    checkpoint_id: str
    current_weights: dict[str, list[float]]


class PrefixGenerateRequest(BaseModel):
    task_name: str
    virtual_tokens: int = Field(default=16, ge=1, le=128)
    prefix_dim: int = Field(default=64, ge=16, le=1024)


def _get_replay_manager() -> ReplayBufferManager:
    return ReplayBufferManager()


def _get_ewc_consolidator() -> EWCConsolidator:
    return EWCConsolidator()


@router.get("/replay/samples")
async def list_replay_samples(
    task_domain: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List exemplars in experience replay buffer."""
    mgr = _get_replay_manager()
    samples = mgr.list_samples(task_domain=task_domain, limit=limit, offset=offset)
    return [s.to_dict() for s in samples]


@router.post("/replay/add", status_code=status.HTTP_201_CREATED)
async def add_replay_sample(req: AddReplaySampleRequest):
    """Add a new sample to the streaming experience replay buffer."""
    mgr = _get_replay_manager()
    try:
        sample = mgr.add_sample(
            sample_input=req.sample_input,
            sample_target=req.sample_target,
            task_domain=req.task_domain,
            priority=req.priority,
            metadata=req.metadata,
        )
        return sample.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/replay/batch")
async def sample_replay_batch(
    batch_size: int = Query(16, ge=1, le=128),
    balanced: bool = Query(True),
):
    """Draw a priority-weighted rehearsal batch."""
    mgr = _get_replay_manager()
    batch = mgr.sample_batch(batch_size=batch_size, balanced=balanced)
    return [s.to_dict() for s in batch]


@router.get("/replay/stats")
async def get_replay_stats():
    """Retrieve sample count distribution across task domains."""
    mgr = _get_replay_manager()
    return {"domains": mgr.get_domain_stats()}


@router.delete("/replay/samples/{sample_id}")
async def delete_replay_sample(sample_id: str):
    """Delete a sample from the replay buffer."""
    mgr = _get_replay_manager()
    deleted = mgr.delete_sample(sample_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"message": "Sample deleted", "id": sample_id}


@router.get("/ewc/checkpoints")
async def list_ewc_checkpoints():
    """List registered EWC reference checkpoints."""
    consolidator = _get_ewc_consolidator()
    checkpoints = consolidator.list_checkpoints()
    return [c.to_dict() for c in checkpoints]


@router.post("/ewc/consolidate", status_code=status.HTTP_201_CREATED)
async def consolidate_ewc_checkpoint(req: EWCConsolidateRequest):
    """Estimate Fisher information and persist an EWC reference checkpoint."""
    consolidator = _get_ewc_consolidator()
    fisher = consolidator.compute_fisher_information(req.weights)
    record = consolidator.save_checkpoint(
        task_name=req.task_name,
        optimal_weights=req.weights,
        fisher_diagonal=fisher,
        importance_lambda=req.importance_lambda,
        experiment_id=req.experiment_id,
        metadata=req.metadata,
    )
    return record.to_dict()


@router.post("/ewc/penalty")
async def compute_ewc_penalty(req: EWCPenaltyRequest):
    """Compute EWC loss penalty for current weights against reference checkpoint."""
    consolidator = _get_ewc_consolidator()
    record = consolidator.get_checkpoint(req.checkpoint_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"EWC checkpoint '{req.checkpoint_id}' not found")

    loss = consolidator.compute_ewc_loss(
        current_weights=req.current_weights,
        optimal_weights=record.optimal_weights,
        fisher_diagonal=record.fisher_diagonal,
        importance_lambda=record.importance_lambda,
    )
    return {"checkpoint_id": req.checkpoint_id, "ewc_penalty_loss": loss}


@router.post("/prefix/generate")
async def generate_task_prefix(req: PrefixGenerateRequest):
    """Generate normalized soft prompt prefix vectors for task conditioning."""
    mgr = PrefixTuningManager()
    config = mgr.generate_random_prefix(
        task_name=req.task_name,
        virtual_tokens=req.virtual_tokens,
        prefix_dim=req.prefix_dim,
    )
    return config.to_dict()
