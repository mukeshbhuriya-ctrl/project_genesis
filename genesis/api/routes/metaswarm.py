"""
REST API Endpoints for Metacognition, Swarm Consensus, Self-Compiler & Pareto (Phase 16).
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.self_compiler import SelfHostingCompilerEngine
from genesis.models.metacognition import MetacognitiveIntrospector
from genesis.research.pareto import ParetoOptimizer
from genesis.tasks.swarm import SwarmConsensusEngine

router = APIRouter()


# ── Metacognition Request Models ──
class IntrospectRequest(BaseModel):
    prompt: str
    candidate_output: str | None = None
    token_log_probs: list[float] | None = None
    historical_accuracy: float | None = None


# ── Swarm Request Models ──
class RegisterNodeRequest(BaseModel):
    name: str
    model_name: str
    role: str = "reasoner"
    reputation: float = Field(default=1.0, ge=0.1, le=5.0)


class ExecuteConsensusRequest(BaseModel):
    task_prompt: str
    proposals_override: dict[str, str] | None = None
    max_rounds: int = Field(default=2, ge=1, le=5)


# ── Self-Compiler Request Models ──
class CompileTargetRequest(BaseModel):
    target_name: str | None = None  # If None, compile all targets


# ── Pareto Request Models ──
class RunParetoRequest(BaseModel):
    name: str = "Workstation-Pareto-Run"
    population_size: int = Field(default=16, ge=4, le=64)
    generations: int = Field(default=4, ge=1, le=20)


# ── Metacognition Endpoints ──
@router.post("/api/metacognition/introspect")
def introspect_prompt(req: IntrospectRequest):
    """Perform introspective confidence evaluation and reasoning budget allocation."""
    introspector = MetacognitiveIntrospector()
    eval_res = introspector.introspect(
        prompt=req.prompt,
        candidate_output=req.candidate_output,
        token_log_probs=req.token_log_probs,
        historical_accuracy=req.historical_accuracy,
    )
    return eval_res.model_dump()


@router.get("/api/metacognition/evaluations")
def list_metacognitive_evaluations(limit: int = Query(50, ge=1, le=200)):
    """List historical metacognitive introspection logs."""
    introspector = MetacognitiveIntrospector()
    evals = introspector.list_evaluations(limit=limit)
    return [e.model_dump() for e in evals]


@router.get("/api/metacognition/summary")
def get_calibration_summary():
    """Retrieve aggregate calibration error and decision statistics."""
    introspector = MetacognitiveIntrospector()
    return introspector.get_calibration_summary()


# ── Swarm Consensus Endpoints ──
@router.get("/api/swarm/nodes")
def list_swarm_nodes():
    """List all registered peer swarm reasoning nodes."""
    engine = SwarmConsensusEngine()
    nodes = engine.list_nodes()
    return [n.model_dump() for n in nodes]


@router.post("/api/swarm/nodes", status_code=201)
def register_swarm_node(req: RegisterNodeRequest):
    """Register a new peer swarm reasoning node."""
    engine = SwarmConsensusEngine()
    node = engine.register_node(
        name=req.name,
        model_name=req.model_name,
        role=req.role,
        reputation=req.reputation,
    )
    return node.model_dump()


@router.post("/api/swarm/consensus")
def execute_swarm_consensus(req: ExecuteConsensusRequest):
    """Execute Byzantine fault-tolerant consensus across peer nodes."""
    engine = SwarmConsensusEngine()
    session = engine.execute_consensus(
        task_prompt=req.task_prompt,
        proposals_override=req.proposals_override,
        max_rounds=req.max_rounds,
    )
    return session.model_dump()


@router.get("/api/swarm/sessions")
def list_swarm_sessions(limit: int = Query(50, ge=1, le=200)):
    """List historical swarm consensus sessions."""
    engine = SwarmConsensusEngine()
    sessions = engine.list_sessions(limit=limit)
    return [s.model_dump() for s in sessions]


@router.get("/api/swarm/sessions/{session_id}")
def get_swarm_session(session_id: str):
    """Retrieve specific swarm consensus session by ID."""
    engine = SwarmConsensusEngine()
    session = engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Consensus session '{session_id}' not found")
    return session.model_dump()


# ── Self-Hosting Compiler Endpoints ──
@router.get("/api/self-compiler/targets")
def list_compiler_targets():
    """List all registered self-compiled hot-path targets."""
    engine = SelfHostingCompilerEngine()
    targets = engine.list_targets()
    return [t.model_dump() for t in targets]


@router.post("/api/self-compiler/compile")
def compile_target(req: CompileTargetRequest):
    """Compile specific target or all targets and benchmark speedups."""
    engine = SelfHostingCompilerEngine()
    if req.target_name:
        try:
            target = engine.compile_target(req.target_name)
            return target.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    targets = engine.compile_all()
    return [t.model_dump() for t in targets]


@router.get("/api/self-compiler/targets/{target_name}")
def get_compiler_target(target_name: str):
    """Fetch compiled target metadata by name."""
    engine = SelfHostingCompilerEngine()
    target = engine.get_target(target_name)
    if not target:
        raise HTTPException(status_code=404, detail=f"Compiler target '{target_name}' not found")
    return target.model_dump()


# ── Pareto Optimization Endpoints ──
@router.post("/api/pareto/optimize")
def run_pareto_optimization(req: RunParetoRequest):
    """Execute continuous multi-objective Pareto optimization search."""
    optimizer = ParetoOptimizer()
    exp, front = optimizer.run_optimization(
        name=req.name,
        population_size=req.population_size,
        generations=req.generations,
    )
    return {
        "experiment": exp.model_dump(),
        "frontier": [p.model_dump() for p in front],
    }


@router.get("/api/pareto/experiments")
def list_pareto_experiments(limit: int = Query(20, ge=1, le=100)):
    """List historical Pareto optimization runs."""
    optimizer = ParetoOptimizer()
    experiments = optimizer.list_experiments(limit=limit)
    return [e.model_dump() for e in experiments]


@router.get("/api/pareto/experiments/{experiment_id}/frontiers")
def get_pareto_frontiers(experiment_id: str):
    """Fetch Pareto-optimal points for a given experiment."""
    optimizer = ParetoOptimizer()
    frontiers = optimizer.get_frontiers(experiment_id)
    return [p.model_dump() for p in frontiers]
