"""
Document indexing and semantic passage retrieval.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core.logging import get_logger
from genesis.memory.embeddings import EmbeddingEngine, pack_vector, unpack_vector, cosine_similarity

log = get_logger("knowledge.indexer")


@dataclass
class DocumentInfo:
    """Metadata for an indexed document source."""
    id: str
    title: str
    source_path: str
    mime_type: str = "text/plain"
    file_size_bytes: int = 0
    chunk_count: int = 0
    checksum: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeChunk:
    """A single semantic text chunk from an indexed document."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int = 0
    embedding: list[float] | None = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_file_checksum(file_path: Path | str) -> str:
    """Calculate SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """
    Split text into semantic passages with overlap.
    Prefers paragraph breaks, falls back to sentence or character boundaries.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    # Split by double newline paragraphs first
    paragraphs = cleaned.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len + 2 <= chunk_size:
            current_chunk.append(p)
            current_len += p_len + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            # Start new chunk with overlap if possible
            current_chunk = [p]
            current_len = p_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Fallback if any chunk is still larger than chunk_size * 1.5
    final_chunks: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final_chunks.append(c)
        else:
            # Simple window chunking
            start = 0
            while start < len(c):
                end = start + chunk_size
                final_chunks.append(c[start:end])
                start += chunk_size - chunk_overlap

    return final_chunks


class KnowledgeIndexer:
    """Manages document ingestion, chunking, embedding generation, and retrieval."""

    def __init__(self, db_path: str, engine: EmbeddingEngine | None = None):
        self.db_path = db_path
        self.engine = engine or EmbeddingEngine()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    async def index_file(
        self,
        file_path: Path | str,
        title: str | None = None,
        mime_type: str = "text/plain",
    ) -> DocumentInfo:
        """
        Index a local text or markdown document:
        - Computes checksum to skip if unchanged
        - Chunks content
        - Generates dense vector embeddings
        - Persists document and chunk records
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        checksum = compute_file_checksum(path)
        doc_title = title or path.stem
        now = datetime.now(timezone.utc).isoformat()
        file_size = path.stat().st_size

        conn = self._conn()
        try:
            # Check existing document by path
            cursor = conn.execute(
                "SELECT * FROM knowledge_documents WHERE source_path = ?",
                (str(path),),
            )
            existing = cursor.fetchone()
            if existing and existing["checksum"] == checksum:
                log.info(f"Document '{doc_title}' already indexed and unchanged (checksum={checksum[:8]}).")
                return DocumentInfo(
                    id=existing["id"],
                    title=existing["title"],
                    source_path=existing["source_path"],
                    mime_type=existing["mime_type"],
                    file_size_bytes=existing["file_size_bytes"],
                    chunk_count=existing["chunk_count"],
                    checksum=existing["checksum"],
                    created_at=existing["created_at"],
                    updated_at=existing["updated_at"],
                )

            # Read content
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            chunks_text = chunk_text(content)
            if not chunks_text:
                chunks_text = [content]

            # Generate embeddings
            vectors = await self.engine.get_embeddings(chunks_text)

            doc_id = existing["id"] if existing else str(uuid.uuid4())

            # Delete old chunks if updating existing document
            if existing:
                conn.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (doc_id,))
                conn.execute(
                    """
                    UPDATE knowledge_documents
                    SET title = ?, checksum = ?, file_size_bytes = ?, chunk_count = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (doc_title, checksum, file_size, len(chunks_text), now, doc_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO knowledge_documents (
                        id, title, source_path, mime_type, file_size_bytes,
                        chunk_count, checksum, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, doc_title, str(path), mime_type, file_size, len(chunks_text), checksum, now, now),
                )

            # Insert chunks with embeddings
            for idx, (text_chunk, vec) in enumerate(zip(chunks_text, vectors)):
                chunk_id = str(uuid.uuid4())
                packed = pack_vector(vec) if vec else None
                conn.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        id, document_id, chunk_index, content, token_count, embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, idx, text_chunk, len(text_chunk) // 4, packed, now),
                )

            conn.commit()
            log.info(f"Indexed document '{doc_title}' -> {len(chunks_text)} chunks.")

            return DocumentInfo(
                id=doc_id,
                title=doc_title,
                source_path=str(path),
                mime_type=mime_type,
                file_size_bytes=file_size,
                chunk_count=len(chunks_text),
                checksum=checksum,
                created_at=now,
                updated_at=now,
            )
        finally:
            conn.close()

    async def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Search indexed knowledge chunks using semantic vector similarity.
        """
        if not query.strip():
            return []

        query_vector = await self.engine.get_embedding(query)
        if not query_vector:
            return []

        conn = self._conn()
        try:
            sql = """
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding,
                       d.title as doc_title, d.source_path
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            """
            cursor = conn.execute(sql)
            results: list[dict[str, Any]] = []

            for row in cursor.fetchall():
                blob = row["embedding"]
                vec = unpack_vector(blob)
                score = cosine_similarity(query_vector, vec)
                if score >= min_score:
                    results.append({
                        "chunk_id": row["id"],
                        "document_id": row["document_id"],
                        "document_title": row["doc_title"],
                        "source_path": row["source_path"],
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                        "similarity": round(score, 4),
                    })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
        finally:
            conn.close()

    def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "SELECT id, title, source_path, file_size_bytes, chunk_count, updated_at "
                "FROM knowledge_documents ORDER BY updated_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_document(self, document_id: str) -> bool:
        """Delete an indexed document and its chunks."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM knowledge_documents WHERE id = ?", (document_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
