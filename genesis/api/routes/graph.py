"""
REST API routes for Cross-Campaign Meta-Memory Knowledge Graph (Phase 10).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.memory.graph import MetaMemoryGraph
from genesis.research.campaigns import CampaignManager

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


# ── Schemas ──

class NodeCreateRequest(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Node category (concept, hypothesis, debate, experiment, campaign, metric, finding, paper, component)")
    label: str = Field(..., description="Human-readable node label")
    properties: Optional[dict[str, Any]] = Field(default_factory=dict, description="Custom node properties")
    embedding: Optional[list[float]] = Field(default=None, description="Optional 768-dim vector embedding")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Metadata dictionary")


class EdgeCreateRequest(BaseModel):
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relation_type: str = Field(..., description="Semantic relation type")
    weight: float = Field(default=1.0, ge=0.0, description="Edge weight / strength")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Metadata dictionary")
    edge_id: Optional[str] = Field(default=None, description="Optional custom edge ID")


# ── Routes ──

@router.get("/nodes")
async def list_nodes(
    node_type: Optional[str] = Query(default=None, description="Filter by node type"),
    limit: int = Query(default=100, ge=1, le=500),
):
    """List knowledge graph nodes."""
    graph = MetaMemoryGraph()
    nodes = graph.list_nodes(node_type=node_type, limit=limit)
    return [
        {
            "id": n.id,
            "type": n.node_type,
            "label": n.label,
            "properties": n.properties,
            "has_embedding": n.embedding is not None,
            "created_at": n.created_at,
            "updated_at": n.updated_at,
        }
        for n in nodes
    ]


@router.post("/nodes")
async def create_node(req: NodeCreateRequest):
    """Create or overwrite a knowledge graph node."""
    graph = MetaMemoryGraph()
    node = graph.add_node(
        id=req.id,
        node_type=req.node_type,
        label=req.label,
        properties=req.properties,
        embedding=req.embedding,
        metadata=req.metadata,
    )
    return {
        "id": node.id,
        "type": node.node_type,
        "label": node.label,
        "properties": node.properties,
        "created_at": node.created_at,
    }


@router.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Retrieve details for a single knowledge node."""
    graph = MetaMemoryGraph()
    node = graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Knowledge node '{node_id}' not found")
    return {
        "id": node.id,
        "type": node.node_type,
        "label": node.label,
        "properties": node.properties,
        "has_embedding": node.embedding is not None,
        "created_at": node.created_at,
        "updated_at": node.updated_at,
        "metadata": node.metadata,
    }


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a knowledge node and all cascading edges."""
    graph = MetaMemoryGraph()
    deleted = graph.delete_node(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Knowledge node '{node_id}' not found")
    return {"deleted": True, "node_id": node_id}


@router.get("/edges")
async def list_edges(
    node_id: Optional[str] = Query(default=None, description="Filter by connected node"),
    relation_type: Optional[str] = Query(default=None, description="Filter by relation type"),
    limit: int = Query(default=150, ge=1, le=500),
):
    """List knowledge graph edges."""
    graph = MetaMemoryGraph()
    edges = graph.get_edges(node_id=node_id, relation_type=relation_type, limit=limit)
    return [
        {
            "id": e.id,
            "source": e.source_id,
            "target": e.target_id,
            "relation": e.relation_type,
            "weight": e.weight,
            "created_at": e.created_at,
        }
        for e in edges
    ]


@router.post("/edges")
async def create_edge(req: EdgeCreateRequest):
    """Create a directed edge connecting two knowledge nodes."""
    graph = MetaMemoryGraph()
    # Verify nodes exist
    if not graph.get_node(req.source_id):
        raise HTTPException(status_code=400, detail=f"Source node '{req.source_id}' does not exist")
    if not graph.get_node(req.target_id):
        raise HTTPException(status_code=400, detail=f"Target node '{req.target_id}' does not exist")

    edge = graph.add_edge(
        source_id=req.source_id,
        target_id=req.target_id,
        relation_type=req.relation_type,
        weight=req.weight,
        metadata=req.metadata,
        edge_id=req.edge_id,
    )
    return {
        "id": edge.id,
        "source": edge.source_id,
        "target": edge.target_id,
        "relation": edge.relation_type,
        "weight": edge.weight,
        "created_at": edge.created_at,
    }


@router.delete("/edges/{edge_id}")
async def delete_edge(edge_id: str):
    """Delete an edge by ID."""
    graph = MetaMemoryGraph()
    deleted = graph.delete_edge(edge_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Edge '{edge_id}' not found")
    return {"deleted": True, "edge_id": edge_id}


@router.get("/neighbors/{node_id}")
async def get_neighbors(
    node_id: str,
    direction: str = Query(default="both", description="'in', 'out', or 'both'"),
    relation_type: Optional[str] = Query(default=None),
):
    """Get adjacent nodes and connecting edges."""
    graph = MetaMemoryGraph()
    if not graph.get_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    neighbors = graph.get_neighbors(node_id=node_id, direction=direction, relation_type=relation_type)
    return [
        {
            "node": {
                "id": node.id,
                "type": node.node_type,
                "label": node.label,
                "properties": node.properties,
            },
            "edge": {
                "id": edge.id,
                "source": edge.source_id,
                "target": edge.target_id,
                "relation": edge.relation_type,
                "weight": edge.weight,
            },
        }
        for node, edge in neighbors
    ]


@router.get("/path")
async def find_path(
    source_id: str = Query(..., description="Start node ID"),
    target_id: str = Query(..., description="End node ID"),
    max_depth: int = Query(default=4, ge=1, le=10),
):
    """Find the shortest path connecting source_id to target_id."""
    graph = MetaMemoryGraph()
    path = graph.find_path(source_id=source_id, target_id=target_id, max_depth=max_depth)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No path found between '{source_id}' and '{target_id}' within depth {max_depth}")
    return {"source": source_id, "target": target_id, "path": path, "length": len(path) - 1}


@router.get("/subgraph")
async def get_subgraph(
    centers: str = Query(..., description="Comma-separated list of center node IDs"),
    depth: int = Query(default=1, ge=1, le=4),
):
    """Extract sub-network around specified center nodes."""
    center_ids = [c.strip() for c in centers.split(",") if c.strip()]
    if not center_ids:
        raise HTTPException(status_code=400, detail="At least one center node ID required")

    graph = MetaMemoryGraph()
    subgraph = graph.get_subgraph(center_node_ids=center_ids, depth=depth)
    return {
        "centers": center_ids,
        "depth": depth,
        "nodes": [
            {"id": n.id, "type": n.node_type, "label": n.label, "properties": n.properties}
            for n in subgraph.nodes
        ],
        "edges": [
            {"id": e.id, "source": e.source_id, "target": e.target_id, "relation": e.relation_type, "weight": e.weight}
            for e in subgraph.edges
        ],
    }


@router.post("/sync/campaign/{campaign_id}")
async def sync_campaign_to_graph(campaign_id: str):
    """Synchronize an existing research campaign and its artifacts into the knowledge graph."""
    cm = CampaignManager()
    campaign = cm.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Research campaign '{campaign_id}' not found")

    graph = MetaMemoryGraph()
    result = graph.sync_from_campaign(campaign.to_dict())
    return {
        "status": "synchronized",
        "campaign_id": campaign_id,
        "nodes_created": result["nodes_created"],
        "edges_created": result["edges_created"],
    }


@router.get("/export")
async def export_graph():
    """Export complete knowledge graph for UI visualization."""
    graph = MetaMemoryGraph()
    return graph.export_graph()
