"""
REST API routes for multi-agent teams, dialectic consensus debates, and research campaigns (Phase 9).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.research.campaigns import CampaignManager
from genesis.tasks.agents import (
    ARCHITECT_PERSONA,
    IMPLEMENTER_PERSONA,
    RESEARCHER_PERSONA,
    REVIEWER_PERSONA,
    SYNTHESIZER_PERSONA,
    AgentRole,
    AgentTeam,
)
from genesis.tasks.consensus import ConsensusEngine

router = APIRouter(prefix="/agents", tags=["Multi-Agent & Campaigns"])


# ── Schemas ──

class CreateTeamRequest(BaseModel):
    name: str = Field(..., description="Team name")
    description: Optional[str] = Field(default="", description="Team purpose or description")
    roles: list[str] = Field(default=["researcher", "architect", "implementer", "reviewer"], description="List of role identifiers")
    metadata: Optional[dict[str, Any]] = None


class RunDebateRequest(BaseModel):
    topic: str = Field(..., description="Objective or design decision to debate")
    context: Optional[str] = Field(default=None, description="Optional domain context or constraints")
    max_rounds: int = Field(default=2, ge=1, le=5, description="Number of deliberation rounds")


class CreateCampaignRequest(BaseModel):
    title: str = Field(..., description="Title of the research campaign")
    objective: str = Field(..., description="Comprehensive objective statement")
    hypotheses: Optional[list[str]] = Field(default=None, description="Initial empirical hypotheses")
    custom_milestones: Optional[list[str]] = Field(default=None, description="Optional custom milestone steps")
    metadata: Optional[dict[str, Any]] = None


# ── Dependency Helpers ──

def _get_consensus_engine() -> ConsensusEngine:
    return ConsensusEngine()


def _get_campaign_manager() -> CampaignManager:
    return CampaignManager()


# ── Endpoints ──

@router.get("/roles")
def list_available_roles():
    """List available agent roles, cognitive personas, and default routing categories."""
    personas = [
        RESEARCHER_PERSONA,
        ARCHITECT_PERSONA,
        IMPLEMENTER_PERSONA,
        REVIEWER_PERSONA,
        SYNTHESIZER_PERSONA,
    ]
    return [p.to_dict() for p in personas]


@router.get("/teams")
def list_agent_teams():
    """List configured agent teams."""
    from genesis.core.database import get_db_path_from_url, get_sync_connection
    from genesis.core import get_settings
    import json

    db_path = get_db_path_from_url(get_settings().database_url)
    conn = get_sync_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM agent_teams ORDER BY created_at DESC")
        teams = []
        for r in cur.fetchall():
            teams.append({
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "roles": json.loads(r["roles_json"] or "[]"),
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata"] or "{}"),
            })
        return teams
    finally:
        conn.close()


@router.post("/teams")
def create_agent_team(req: CreateTeamRequest):
    """Register a new custom multi-agent team."""
    import uuid
    import json
    from genesis.core.database import get_db_path_from_url, get_sync_connection
    from genesis.core import get_settings

    team_id = f"team-{uuid.uuid4().hex[:8]}"
    db_path = get_db_path_from_url(get_settings().database_url)
    conn = get_sync_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO agent_teams (id, name, description, roles_json, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                team_id,
                req.name,
                req.description or "",
                json.dumps(req.roles),
                json.dumps(req.metadata or {}),
            ),
        )
        conn.commit()
        return {
            "id": team_id,
            "name": req.name,
            "description": req.description,
            "roles": req.roles,
            "metadata": req.metadata or {},
        }
    finally:
        conn.close()


# ── Dialectic Debate Endpoints ──

@router.post("/debate/run")
async def run_consensus_debate(req: RunDebateRequest):
    """Run a multi-turn dialectic debate across the agent team to reach consensus."""
    engine = _get_consensus_engine()
    try:
        session = await engine.run_debate(
            topic=req.topic,
            context=req.context,
            max_rounds=req.max_rounds,
        )
        return session.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate execution failed: {e}")


@router.get("/debates")
def list_debate_sessions(limit: int = Query(default=20, ge=1, le=100)):
    """List recent dialectic debate sessions."""
    engine = _get_consensus_engine()
    sessions = engine.list_sessions(limit=limit)
    return [s.to_dict() for s in sessions]


@router.get("/debates/{session_id}")
def get_debate_session(session_id: str):
    """Get detailed transcript, turns, and consensus verdict for a debate session."""
    engine = _get_consensus_engine()
    session = engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Debate session '{session_id}' not found")
    return session.to_dict()


# ── Long-Horizon Research Campaigns ──

@router.post("/campaigns")
def create_research_campaign(req: CreateCampaignRequest):
    """Create a new long-horizon autonomous research campaign."""
    manager = _get_campaign_manager()
    try:
        campaign = manager.create_campaign(
            title=req.title,
            objective=req.objective,
            hypotheses=req.hypotheses,
            custom_milestones=req.custom_milestones,
            metadata=req.metadata,
        )
        return campaign.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign creation failed: {e}")


@router.get("/campaigns")
def list_research_campaigns(limit: int = Query(default=20, ge=1, le=100)):
    """List active and historical research campaigns."""
    manager = _get_campaign_manager()
    campaigns = manager.list_campaigns(limit=limit)
    return [c.to_dict() for c in campaigns]


@router.get("/campaigns/{campaign_id}")
def get_research_campaign(campaign_id: str):
    """Get detailed state, milestones, and findings of a research campaign."""
    manager = _get_campaign_manager()
    campaign = manager.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Research campaign '{campaign_id}' not found")
    return campaign.to_dict()


@router.post("/campaigns/{campaign_id}/advance")
async def advance_research_campaign(campaign_id: str):
    """Advance a research campaign by executing its next pending milestone."""
    manager = _get_campaign_manager()
    try:
        campaign = await manager.advance_campaign(campaign_id)
        return campaign.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to advance campaign: {e}")
