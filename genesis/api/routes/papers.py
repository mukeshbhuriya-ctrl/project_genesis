"""
REST API routes for Autonomous Research Daemon & Publications (Phase 10).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.research.daemon import AutonomousResearchDaemon

router = APIRouter(prefix="/research/daemon", tags=["Autonomous Research Daemon"])


# ── Schemas ──

class RunDaemonCycleRequest(BaseModel):
    topic: str = Field(..., description="Research problem, proposition, or architectural question to investigate")
    context: Optional[str] = Field(default="", description="Domain constraints, target runtime, or background context")
    rounds: int = Field(default=2, ge=1, le=4, description="Deliberation rounds for multi-agent consensus debate")
    campaign_id: Optional[str] = Field(default=None, description="Optional parent campaign ID to associate with")


# ── Routes ──

@router.post("/cycle")
async def run_daemon_cycle(req: RunDaemonCycleRequest):
    """
    Launch an autonomous research cycle:
    1. Multi-agent dialectic consensus debate
    2. Sandboxed prototype code execution
    3. Formal research paper synthesis
    4. Knowledge graph & memory weaving
    """
    daemon = AutonomousResearchDaemon()
    result = await daemon.run_cycle(
        topic=req.topic,
        context=req.context or "",
        rounds=req.rounds,
        campaign_id=req.campaign_id,
    )
    return result


@router.get("/papers")
async def list_papers(limit: int = Query(default=50, ge=1, le=200)):
    """List published autonomous research papers."""
    daemon = AutonomousResearchDaemon()
    return daemon.list_papers(limit=limit)


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Retrieve full text and metadata of a published research paper."""
    daemon = AutonomousResearchDaemon()
    paper = daemon.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Research paper '{paper_id}' not found")
    return paper
