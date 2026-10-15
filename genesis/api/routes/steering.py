"""
REST API routes for Interactive Human-in-the-Loop Steering Controls (Phase 12).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.research.steering import SteeringController

router = APIRouter(prefix="/steering", tags=["Human-in-the-Loop Steering"])


class EmitDirectiveRequest(BaseModel):
    target_type: str = Field(..., description="Target process: daemon_cycle, campaign, debate, kernel_synthesis")
    target_id: str = Field(..., description="Identifier of target cycle, campaign, or session")
    action: str = Field(..., description="Action: pause, resume, critique, override_hypothesis, abort, inject_guidance")
    payload: Optional[dict[str, Any]] = Field(default_factory=dict, description="Operator payload or guidance message")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")


@router.post("/directives")
async def emit_steering_directive(req: EmitDirectiveRequest):
    """Emit an operator steering directive for an active autonomous process."""
    ctrl = SteeringController()
    try:
        event = ctrl.emit_directive(
            target_type=req.target_type,
            target_id=req.target_id,
            action=req.action,
            payload=req.payload,
            metadata=req.metadata,
        )
        return event.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to emit directive: {e}")


@router.get("/directives")
async def list_steering_directives(limit: int = Query(default=50, ge=1, le=100)):
    """List recent human-in-the-loop steering directives."""
    ctrl = SteeringController()
    events = ctrl.list_events(limit=limit)
    return [e.to_dict() for e in events]


@router.post("/directives/{event_id}/apply")
async def mark_directive_applied(event_id: str):
    """Mark a directive as applied by the autonomous system."""
    ctrl = SteeringController()
    ctrl.mark_applied(event_id)
    return {"id": event_id, "status": "applied"}
