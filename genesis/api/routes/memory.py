"""Memory API endpoints with semantic and hybrid retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.memory import MemoryStore
from genesis.memory.embeddings import EmbeddingEngine
from genesis.memory.consolidation import consolidate_memory

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
    auto_embed: bool = True
    consolidate: bool = False


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    category: str | None = None
    confidence: float | None = None
    metadata: dict | None = None


@router.post("/memories")
async def create_memory(req: MemoryCreateRequest) -> dict:
    """Create a new memory record, with optional automatic embedding and consolidation."""
    store = _get_store()
    vector = None

    if req.auto_embed:
        try:
            engine = EmbeddingEngine()
            vector = await engine.get_embedding(req.content)
        except Exception:
            vector = None

    if req.consolidate and vector:
        record, is_new = consolidate_memory(
            store=store,
            content=req.content,
            category=req.category,
            source=req.source,
            confidence=req.confidence,
            metadata=req.metadata,
            vector=vector,
        )
        return {
            "id": record.id,
            "category": record.category,
            "version": record.version,
            "consolidated": not is_new,
        }

    record = store.create(
        content=req.content,
        category=req.category,
        source=req.source,
        confidence=req.confidence,
        metadata=req.metadata,
        embedding=vector,
    )
    return {"id": record.id, "category": record.category, "version": record.version, "has_embedding": bool(vector)}


@router.get("/memories/categories")
async def list_categories() -> dict:
    """List all memory categories."""
    store = _get_store()
    return {"categories": store.list_categories()}


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    """Get a specific memory by ID."""
    store = _get_store()
    record = store.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    res = dict(record.__dict__)
    res["has_embedding"] = store.get_embedding(memory_id) is not None
    return res


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
    mode: str = Query(default="hybrid", pattern="^(hybrid|semantic|lexical)$"),
    alpha: float = Query(default=0.6, ge=0.0, le=1.0),
) -> dict:
    """Search memories using hybrid (vector + lexical), semantic, or lexical matching."""
    store = _get_store()

    if not q:
        records = store.search(query="", category=category, limit=limit)
        return {"count": len(records), "mode": "all", "results": [r.__dict__ for r in records]}

    if mode == "lexical":
        records = store.search(query=q, category=category, limit=limit)
        return {
            "count": len(records),
            "mode": "lexical",
            "results": [{"memory": r.__dict__, "score": 1.0} for r in records],
        }

    # Generate query embedding for semantic / hybrid modes
    try:
        engine = EmbeddingEngine()
        query_vector = await engine.get_embedding(q)
    except Exception:
        query_vector = None

    if mode == "semantic" and query_vector:
        scored = store.search_semantic(query_vector=query_vector, category=category, limit=limit)
        return {
            "count": len(scored),
            "mode": "semantic",
            "results": [{"memory": r.__dict__, "similarity": round(s, 4)} for r, s in scored],
        }

    # Default to hybrid search
    scored = store.search_hybrid(
        query=q,
        query_vector=query_vector,
        category=category,
        limit=limit,
        alpha=alpha,
    )
    return {
        "count": len(scored),
        "mode": "hybrid" if query_vector else "lexical_fallback",
        "results": [{"memory": r.__dict__, "score": s} for r, s in scored],
    }
