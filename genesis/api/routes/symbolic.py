"""
Neuro-Symbolic Constraint Verification REST API endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.research.symbolic import (
    NeuroSymbolicVerifier,
    SymbolicConstraint,
)

router = APIRouter(prefix="/api/symbolic", tags=["neuro-symbolic"])


class VerifyCodeRequest(BaseModel):
    code_str: str
    constraints: list[dict[str, Any]] = Field(default_factory=list)


class VerifyMathRequest(BaseModel):
    expression_str: str
    variable_ranges: dict[str, list[float]] = Field(default_factory=dict)
    constraints: list[dict[str, Any]] = Field(default_factory=list)


class VerifyLogicRequest(BaseModel):
    premises: list[str]
    hypothesis: str


def _get_verifier() -> NeuroSymbolicVerifier:
    return NeuroSymbolicVerifier()


@router.post("/verify/code")
async def verify_code(req: VerifyCodeRequest):
    """Verify static AST invariants (termination, purity, bounds) on Python code."""
    verifier = _get_verifier()
    constraints = [
        SymbolicConstraint(
            constraint_type=c.get("constraint_type", "purity_check"),
            target=c.get("target", ""),
            expression=c.get("expression", ""),
            params=c.get("params", {}),
        )
        for c in req.constraints
    ]
    res = verifier.verify_code_invariants(req.code_str, constraints=constraints)
    return res.to_dict()


@router.post("/verify/math")
async def verify_math(req: VerifyMathRequest):
    """Verify mathematical bounds and invariants over continuous domains."""
    verifier = _get_verifier()
    ranges = {k: (v[0], v[1]) for k, v in req.variable_ranges.items() if len(v) >= 2}
    constraints = [
        SymbolicConstraint(
            constraint_type=c.get("constraint_type", "range_bound"),
            target=c.get("target", ""),
            expression=c.get("expression", ""),
            params=c.get("params", {}),
        )
        for c in req.constraints
    ]
    res = verifier.verify_mathematical_bounds(req.expression_str, ranges, constraints)
    return res.to_dict()


@router.post("/verify/logic")
async def verify_logic(req: VerifyLogicRequest):
    """Verify logical hypothesis against premises using resolution refutation."""
    verifier = _get_verifier()
    res = verifier.verify_propositional_logic(req.premises, req.hypothesis)
    return res.to_dict()


@router.get("/verifications")
async def list_verifications(
    target_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List historical neuro-symbolic verification runs."""
    verifier = _get_verifier()
    records = verifier.list_verifications(target_type=target_type, status=status, limit=limit)
    return [r.to_dict() for r in records]


@router.get("/verifications/{record_id}")
async def get_verification(record_id: str):
    """Fetch verification details by record ID."""
    verifier = _get_verifier()
    res = verifier.get_verification(record_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Verification record '{record_id}' not found")
    return res.to_dict()
