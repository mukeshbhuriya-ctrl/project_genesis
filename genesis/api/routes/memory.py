"""Memory API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.memory import MemoryStore

router = APIRouter()


def _get_store() -> MemoryStore:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return MemoryStore(str(db_path))


class MemoryCreateRequest(BaseModel):
    content: str
    category: str = "general"
    source: str = ""
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    category: str | None = None
    confidence: float | None = None
    metadata: dict | None = None


@router.post("/memories")
async def create_memory(req: MemoryCreateRequest) -> dict:
    """Create a new memory record."""
    store = _get_store()
    record = store.create(
        content=req.content,
        category=req.category,
        source=req.source,
        confidence=req.confidence,
        metadata=req.metadata,
    )
    return {"id": record.id, "category": record.category, "version": record.version}


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    """Get a specific memory by ID."""
    store = _get_store()
    record = store.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record.__dict__


@router.put("/memories/{memory_id}")
async def update_memory(memory_id: str, req: MemoryUpdateRequest) -> dict:
    """Update a memory record."""
    store = _get_store()
    record = store.update(
        memory_id,
        content=req.content,
        category=req.category,
        confidence=req.confidence,
        metadata=req.metadata,
    )
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"id": record.id, "version": record.version, "updated_at": record.updated_at}


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """Delete a memory record."""
    store = _get_store()
    if not store.delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@router.get("/memories")
async def search_memories(
    q: str = "",
    category: str | None = None,
    limit: int = 20,
) -> dict:
    """Search memories by text and/or category."""
    store = _get_store()
    records = store.search(query=q, category=category, limit=limit)
    return {
        "count": len(records),
        "results": [r.__dict__ for r in records],
    }


@router.get("/memories/categories")
async def list_categories() -> dict:
    """List all memory categories."""
    store = _get_store()
    return {"categories": store.list_categories()}
