"""
REST API routes for World Model, Latent Dream Exploration & MCTS Subsystems (Phase 15).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.world_model.alignment import CrossModalAligner
from genesis.world_model.dream import LatentDreamEngine
from genesis.world_model.mcts import MCTSSearchEngine
from genesis.world_model.transition import (
    Action,
    EnvironmentState,
    StateTransitionModel,
    TransitionPrediction,
)

router = APIRouter(prefix="/world", tags=["World Model & Latent Dreams"])


# --- Request & Response Schemas ---

class PredictTransitionRequest(BaseModel):
    current_state: dict[str, Any] = Field(default_factory=dict, description="Current workspace state snapshot")
    action: dict[str, Any] = Field(..., description="Action to simulate: action_type, target, payload, params")


class RecordTransitionRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    action: dict[str, Any] = Field(..., description="Action executed")
    prediction: dict[str, Any] = Field(..., description="Predicted transition outcome")
    actual_state: dict[str, Any] = Field(..., description="Actual ground-truth state")
    record_id: Optional[str] = Field(default=None, description="Optional custom record ID")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata")


class DreamRolloutRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Goal prompt to simulate rollouts for")
    seed_steps: Optional[list[str]] = Field(default=None, description="Optional explicit seed action steps")
    depth: int = Field(default=4, ge=1, le=10, description="Mental rollout trajectory depth")
    prune_threshold: float = Field(default=0.20, ge=-1.0, le=1.0, description="Branch pruning threshold")
    trajectory_id: Optional[str] = Field(default=None, description="Optional custom trajectory ID")
    export_to_replay: bool = Field(default=False, description="Automatically export high-reward steps to replay buffer")
    replay_domain: str = Field(default="reasoning", description="Replay buffer domain category")


class AlignModalitiesRequest(BaseModel):
    text_prompt: str = Field(..., description="Natural language specification or reasoning query")
    ast_code: str = Field(..., description="Symbolic code syntax or AST snippet")
    visual_spec: Any = Field(..., description="Visual wireframe, UI layout, or element dictionary")
    record_id: Optional[str] = Field(default=None, description="Optional custom alignment ID")


class CrossModalRetrieveRequest(BaseModel):
    query_modality: str = Field(..., description="'text', 'ast', or 'visual'")
    query_repr: str = Field(..., description="Query content string")
    candidates: list[dict[str, str]] = Field(..., description="List of dicts with 'modality' and 'content'")


class MCTSSearchRequest(BaseModel):
    root_task: str = Field(..., min_length=1, description="Target task to explore with MCTS")
    initial_files: Optional[dict[str, str]] = Field(default=None, description="Initial workspace files")
    iterations: int = Field(default=15, ge=1, le=100, description="MCTS iteration budget")
    max_depth: int = Field(default=4, ge=1, le=10, description="Maximum tree depth")
    run_id: Optional[str] = Field(default=None, description="Optional custom run ID")


# --- Endpoints ---

@router.post("/predict-transition")
async def predict_transition(req: PredictTransitionRequest):
    """Simulate and predict workspace state transitions and safety invariants before execution."""
    model = StateTransitionModel()
    env_state = EnvironmentState.from_dict(req.current_state)
    action_obj = Action.from_dict(req.action)
    prediction = model.predict_transition(env_state, action_obj)
    return prediction.to_dict()


@router.post("/record-transition")
async def record_transition(req: RecordTransitionRequest):
    """Persist prediction vs actual execution comparison and compute error score."""
    model = StateTransitionModel()
    action_obj = Action.from_dict(req.action)
    actual_state = EnvironmentState.from_dict(req.actual_state)

    pred_state = EnvironmentState.from_dict(req.prediction.get("predicted_state", {}))
    pred_obj = TransitionPrediction(
        predicted_state=pred_state,
        delta_files=req.prediction.get("delta_files", []),
        delta_ast=req.prediction.get("delta_ast", []),
        confidence=float(req.prediction.get("confidence", 1.0)),
        safety_score=float(req.prediction.get("safety_score", 1.0)),
        safety_violations=req.prediction.get("safety_violations", []),
        predicted_exit_code=int(req.prediction.get("predicted_exit_code", 0)),
    )

    record = model.record_transition(
        session_id=req.session_id,
        action=action_obj,
        prediction=pred_obj,
        actual_state=actual_state,
        record_id=req.record_id,
        metadata=req.metadata,
    )
    return record.to_dict()


@router.get("/transitions")
async def list_transitions(
    session_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Retrieve historical transition verification records."""
    model = StateTransitionModel()
    records = model.list_transitions(session_id=session_id, limit=limit)
    return [r.to_dict() for r in records]


@router.get("/transitions/{record_id}")
async def get_transition(record_id: str):
    """Fetch specific transition record by ID."""
    model = StateTransitionModel()
    record = model.get_transition(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Transition record '{record_id}' not found.")
    return record.to_dict()


@router.post("/dream-rollout")
async def simulate_dream_rollout(req: DreamRolloutRequest):
    """Simulate a multi-step latent dream problem-solving trajectory in memory."""
    engine = LatentDreamEngine()
    traj = engine.simulate_dream(
        prompt=req.prompt,
        seed_steps=req.seed_steps,
        depth=req.depth,
        prune_threshold=req.prune_threshold,
        trajectory_id=req.trajectory_id,
    )
    exported_count = 0
    if req.export_to_replay:
        exported_count = engine.export_to_replay_buffer(traj, domain=req.replay_domain)

    res = traj.to_dict()
    res["exported_to_replay_count"] = exported_count
    return res


@router.get("/trajectories")
async def list_dream_trajectories(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List imagined dream trajectories."""
    engine = LatentDreamEngine()
    trajs = engine.list_trajectories(status=status, limit=limit)
    return [t.to_dict() for t in trajs]


@router.get("/trajectories/{trajectory_id}")
async def get_dream_trajectory(trajectory_id: str):
    """Fetch specific dream trajectory by ID."""
    engine = LatentDreamEngine()
    traj = engine.get_trajectory(trajectory_id)
    if not traj:
        raise HTTPException(status_code=404, detail=f"Dream trajectory '{trajectory_id}' not found.")
    return traj.to_dict()


@router.post("/align-modalities")
async def align_modalities(req: AlignModalitiesRequest):
    """Project text, code AST, and visual UI layouts into joint latent space."""
    aligner = CrossModalAligner()
    result = aligner.align_modalities(
        text_prompt=req.text_prompt,
        ast_code=req.ast_code,
        visual_spec=req.visual_spec,
        record_id=req.record_id,
    )
    return result.to_dict()


@router.get("/alignments")
async def list_alignments(limit: int = Query(default=50, ge=1, le=200)):
    """List multi-modal alignment records sorted by similarity."""
    aligner = CrossModalAligner()
    alignments = aligner.list_alignments(limit=limit)
    return [a.to_dict() for a in alignments]


@router.post("/cross-modal-retrieve")
async def cross_modal_retrieve(req: CrossModalRetrieveRequest):
    """Retrieve and rank cross-modal artifacts against a query representation."""
    aligner = CrossModalAligner()
    candidates = [(c.get("modality", "text"), c.get("content", "")) for c in req.candidates]
    ranked = aligner.cross_modal_retrieve(
        query_modality=req.query_modality,
        query_repr=req.query_repr,
        candidates=candidates,
    )
    return [
        {"modality": mod, "content": content, "similarity_score": sim}
        for mod, content, sim in ranked
    ]


@router.post("/mcts-search")
async def run_mcts_search(req: MCTSSearchRequest):
    """Execute speculative Monte Carlo Tree Search across candidate mutations."""
    engine = MCTSSearchEngine()
    result = engine.search(
        root_task=req.root_task,
        initial_files=req.initial_files,
        iterations=req.iterations,
        max_depth=req.max_depth,
        run_id=req.run_id,
    )
    return result.to_dict()


@router.get("/mcts-runs")
async def list_mcts_runs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List MCTS exploration runs."""
    engine = MCTSSearchEngine()
    runs = engine.list_runs(status=status, limit=limit)
    return [r.to_dict() for r in runs]


@router.get("/mcts-runs/{run_id}")
async def get_mcts_run(run_id: str):
    """Fetch specific MCTS run by ID."""
    engine = MCTSSearchEngine()
    run = engine.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"MCTS run '{run_id}' not found.")
    return run.to_dict()
