"""
REST API routes for Multi-Modal Vision Processing and Visual Inspection (Phase 10).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.models.vision import VisionInspector, VisionRequest

router = APIRouter(prefix="/multimodal", tags=["Multi-Modal Vision"])


# ── Schemas ──

class InspectArtifactRequest(BaseModel):
    artifact_name: str = Field(..., description="Name of the artifact or file")
    artifact_type: str = Field(default="diagram", description="Type: diagram, ui_screenshot, chart, loss_curve, image")
    image_data: str = Field(..., description="Raw base64-encoded image string, data URI, or raw SVG text")
    prompt: Optional[str] = Field(default="Inspect and analyze this visual artifact.", description="Evaluation prompt")
    expected_elements: Optional[list[str]] = Field(default_factory=list, description="Required visual or textual components")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")


# ── Routes ──

@router.post("/inspect")
async def inspect_artifact(req: InspectArtifactRequest):
    """Inspect and evaluate a visual artifact (architecture diagram, UI layout, or loss chart)."""
    inspector = VisionInspector()
    vision_req = VisionRequest(
        image=req.image_data,
        artifact_type=req.artifact_type,
        artifact_name=req.artifact_name,
        prompt=req.prompt or "Analyze this visual artifact.",
        expected_elements=req.expected_elements or [],
        metadata=req.metadata or {},
    )

    resp = await inspector.inspect_artifact(vision_req)
    return {
        "inspection_id": resp.inspection_id,
        "artifact_name": resp.artifact_name,
        "artifact_type": resp.artifact_type,
        "passed": resp.passed,
        "elements_detected": resp.elements_detected,
        "missing_elements": resp.missing_elements,
        "dimensions": resp.dimensions,
        "format": resp.format,
        "analysis_report": resp.content,
    }


@router.get("/inspections")
async def list_inspections(
    artifact_type: Optional[str] = Query(default=None, description="Filter by artifact type"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List historical visual inspections."""
    inspector = VisionInspector()
    return inspector.list_inspections(limit=limit, artifact_type=artifact_type)


@router.get("/inspections/{inspection_id}")
async def get_inspection(inspection_id: str):
    """Get single visual inspection by ID."""
    inspector = VisionInspector()
    res = inspector.get_inspection(inspection_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Inspection '{inspection_id}' not found")
    return res
