"""
REST API routes for Distributed Multi-Workstation Fleet Synchronization (Phase 11).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.deployment.fleet import FleetManager

router = APIRouter(prefix="/fleet", tags=["Distributed Fleet Synchronization"])


# ── Schemas ──

class RegisterFleetNodeRequest(BaseModel):
    hostname: str = Field(..., description="Machine hostname or label")
    address: str = Field(..., description="IP address or DNS hostname reachable on LAN")
    port: int = Field(default=8000, ge=1, le=65535, description="Port serving Genesis API")
    gpu_device: Optional[str] = Field(default="", description="GPU hardware identifier (e.g. RTX 2000 Ada)")
    vram_gb: Optional[float] = Field(default=0.0, ge=0.0, description="Available GPU VRAM in GB")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Custom node metadata")


class ImportDeltaBundleRequest(BaseModel):
    bundle: dict[str, Any] = Field(..., description="Complete exported delta bundle")
    source_node_id: Optional[str] = Field(default="manual", description="Originating peer node ID")


# ── Routes ──

@router.get("/nodes")
async def list_fleet_nodes():
    """List all registered workstation nodes in the distributed fleet."""
    mgr = FleetManager()
    nodes = mgr.list_nodes()
    return [n.to_dict() for n in nodes]


@router.post("/nodes")
async def register_fleet_node(req: RegisterFleetNodeRequest):
    """Register a new peer workstation node in the fleet."""
    mgr = FleetManager()
    node = mgr.register_node(
        hostname=req.hostname,
        address=req.address,
        port=req.port,
        gpu_device=req.gpu_device or "",
        vram_gb=req.vram_gb or 0.0,
        metadata=req.metadata or {},
    )
    return node.to_dict()


@router.get("/nodes/{node_id}")
async def get_fleet_node(node_id: str):
    """Get details of a specific fleet node."""
    mgr = FleetManager()
    node = mgr.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Fleet node '{node_id}' not found")
    return node.to_dict()


@router.delete("/nodes/{node_id}")
async def remove_fleet_node(node_id: str):
    """Remove a peer workstation node from the fleet."""
    mgr = FleetManager()
    removed = mgr.remove_node(node_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Fleet node '{node_id}' not found")
    return {"id": node_id, "status": "deleted"}


@router.post("/nodes/{node_id}/ping")
async def ping_fleet_node(node_id: str):
    """Update heartbeat status of a peer workstation node."""
    mgr = FleetManager()
    node = mgr.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Fleet node '{node_id}' not found")
    mgr.update_heartbeat(node_id, status="online")
    return {"id": node_id, "status": "online", "message": "Heartbeat updated successfully"}


@router.get("/export-delta")
async def export_delta_bundle(since: Optional[str] = Query(default="", description="ISO timestamp cursor")):
    """Export recent delta bundle for peer synchronization."""
    mgr = FleetManager()
    bundle = mgr.export_delta_bundle(since=since or "")
    return bundle


@router.post("/import-delta")
async def import_delta_bundle(req: ImportDeltaBundleRequest):
    """Import and merge an incoming delta bundle into the local database."""
    mgr = FleetManager()
    result = mgr.import_delta_bundle(req.bundle, source_node_id=req.source_node_id or "manual")
    return result.to_dict()


@router.post("/sync/{node_id}")
async def sync_with_peer(node_id: str):
    """Trigger delta synchronization with a peer fleet node."""
    mgr = FleetManager()
    result = await mgr.sync_with_peer(node_id)
    return result.to_dict()


@router.get("/logs")
async def list_sync_logs(limit: int = Query(default=50, ge=1, le=200)):
    """List historical fleet synchronization transactions."""
    mgr = FleetManager()
    return mgr.list_sync_logs(limit=limit)
