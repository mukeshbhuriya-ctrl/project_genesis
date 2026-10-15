"""Knowledge indexing and document search API endpoints."""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.knowledge.indexer import KnowledgeIndexer

router = APIRouter()


def _get_indexer() -> KnowledgeIndexer:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return KnowledgeIndexer(str(db_path))


class IndexFileRequest(BaseModel):
    file_path: str
    title: str | None = None
    mime_type: str = "text/plain"


@router.post("/knowledge/index-file")
async def index_file(req: IndexFileRequest) -> dict:
    """Index a local text or markdown file into the knowledge repository."""
    indexer = _get_indexer()
    try:
        doc = await indexer.index_file(
            file_path=req.file_path,
            title=req.title,
            mime_type=req.mime_type,
        )
        return {
            "success": True,
            "document_id": doc.id,
            "title": doc.title,
            "chunk_count": doc.chunk_count,
            "checksum": doc.checksum,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@router.get("/knowledge/documents")
async def list_documents() -> dict:
    """List all indexed knowledge documents."""
    indexer = _get_indexer()
    docs = indexer.list_documents()
    return {"count": len(docs), "documents": docs}


@router.delete("/knowledge/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Delete an indexed document and its chunk vectors."""
    indexer = _get_indexer()
    if not indexer.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@router.get("/knowledge/search")
async def search_knowledge(
    q: str = Query(..., description="Natural language search query"),
    limit: int = Query(default=5, ge=1, le=20),
    min_score: float = Query(default=0.25, ge=0.0, le=1.0),
) -> dict:
    """Search knowledge chunks using dense vector semantic similarity."""
    indexer = _get_indexer()
    results = await indexer.search(query=q, limit=limit, min_score=min_score)
    return {"count": len(results), "query": q, "results": results}
