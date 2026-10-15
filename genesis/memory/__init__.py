"""
Persistent memory subsystem.

Provides CRUD operations for structured memory records with
categories, metadata, versioning, and search.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis.core.logging import get_logger

log = get_logger("memory")


@dataclass
class MemoryRecord:
    """A single memory entry."""
    id: str = ""
    category: str = "general"
    content: str = ""
    source: str = ""
    confidence: float = 1.0
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """SQLite-backed persistent memory storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row to a MemoryRecord."""
        meta = {}
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return MemoryRecord(
            id=row["id"],
            category=row["category"],
            content=row["content"],
            source=row["source"] or "",
            confidence=row["confidence"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            metadata=meta,
        )

    def create(
        self,
        content: str,
        category: str = "general",
        source: str = "",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        embedding_model: str = "nomic-embed-text",
    ) -> MemoryRecord:
        """Create a new memory record and optionally persist its vector embedding."""
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO memories (id, category, content, source, confidence,
                                      version, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (record_id, category, content, source, confidence, now, now, meta_json),
            )

            if embedding:
                from genesis.memory.embeddings import pack_vector
                blob = pack_vector(embedding)
                conn.execute(
                    """
                    INSERT INTO memory_embeddings (memory_id, embedding, dimension, model, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (record_id, blob, len(embedding), embedding_model, now),
                )

            conn.commit()

            log.info(f"Created memory {record_id[:8]}... [{category}] (has_embedding={bool(embedding)})")
            return MemoryRecord(
                id=record_id,
                category=category,
                content=content,
                source=source,
                confidence=confidence,
                version=1,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
        finally:
            conn.close()

    def store_embedding(
        self,
        memory_id: str,
        embedding: list[float],
        model: str = "nomic-embed-text",
    ) -> bool:
        """Store or update vector embedding for an existing memory."""
        from genesis.memory.embeddings import pack_vector
        blob = pack_vector(embedding)
        now = datetime.now(timezone.utc).isoformat()

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO memory_embeddings (memory_id, embedding, dimension, model, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    model = excluded.model,
                    created_at = excluded.created_at
                """,
                (memory_id, blob, len(embedding), model, now),
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            log.warning(f"Failed to store embedding for memory {memory_id}: {e}")
            return False
        finally:
            conn.close()

    def get_embedding(self, memory_id: str) -> list[float] | None:
        """Retrieve the unpacked float embedding for a memory."""
        from genesis.memory.embeddings import unpack_vector
        conn = self._conn()
        try:
            cursor = conn.execute(
                "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row and row["embedding"]:
                return unpack_vector(row["embedding"])
            return None
        finally:
            conn.close()


    def get(self, memory_id: str) -> MemoryRecord | None:
        """Retrieve a memory by ID."""
        conn = self._conn()
        try:
            cursor = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None
        finally:
            conn.close()

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        """Update an existing memory record. Increments version."""
        existing = self.get(memory_id)
        if not existing:
            return None

        now = datetime.now(timezone.utc).isoformat()
        new_content = content if content is not None else existing.content
        new_category = category if category is not None else existing.category
        new_confidence = confidence if confidence is not None else existing.confidence
        new_metadata = metadata if metadata is not None else existing.metadata
        new_version = existing.version + 1

        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE memories
                SET content = ?, category = ?, confidence = ?,
                    version = ?, updated_at = ?, metadata = ?
                WHERE id = ?
                """,
                (
                    new_content, new_category, new_confidence,
                    new_version, now, json.dumps(new_metadata),
                    memory_id,
                ),
            )
            conn.commit()
            log.info(f"Updated memory {memory_id[:8]}... to v{new_version}")

            existing.content = new_content
            existing.category = new_category
            existing.confidence = new_confidence
            existing.version = new_version
            existing.updated_at = now
            existing.metadata = new_metadata
            return existing
        finally:
            conn.close()

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                log.info(f"Deleted memory {memory_id[:8]}...")
            return deleted
        finally:
            conn.close()

    def search(
        self,
        query: str = "",
        category: str | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Search memories by text content and/or category."""
        conn = self._conn()
        try:
            conditions = []
            params: list[Any] = []

            if query:
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")
            if category:
                conditions.append("category = ?")
                params.append(category)

            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_record(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def list_categories(self) -> list[str]:
        """List all distinct memory categories."""
        conn = self._conn()
        try:
            cursor = conn.execute("SELECT DISTINCT category FROM memories ORDER BY category")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def count(self, category: str | None = None) -> int:
        """Count memories, optionally filtered by category."""
        conn = self._conn()
        try:
            if category:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE category = ?", (category,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def search_semantic(
        self,
        query_vector: list[float],
        category: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.2,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search memories using vector cosine similarity."""
        from genesis.memory.embeddings import unpack_vector, cosine_similarity

        conn = self._conn()
        try:
            sql = """
                SELECT m.*, e.embedding
                FROM memories m
                JOIN memory_embeddings e ON m.id = e.memory_id
            """
            params: list[Any] = []
            if category:
                sql += " WHERE m.category = ?"
                params.append(category)

            cursor = conn.execute(sql, params)
            scored: list[tuple[MemoryRecord, float]] = []

            for row in cursor.fetchall():
                record = self._row_to_record(row)
                vec = unpack_vector(row["embedding"])
                sim = cosine_similarity(query_vector, vec)
                if sim >= min_similarity:
                    scored.append((record, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        finally:
            conn.close()

    def search_hybrid(
        self,
        query: str,
        query_vector: list[float] | None = None,
        category: str | None = None,
        limit: int = 10,
        alpha: float = 0.6,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Hybrid search combining semantic vector similarity and lexical matching.
        alpha: weight for vector similarity (1.0 = purely vector, 0.0 = purely lexical).
        """
        if not query_vector:
            lexical_results = self.search(query=query, category=category, limit=limit)
            return [(r, 1.0) for r in lexical_results]

        semantic_candidates = self.search_semantic(
            query_vector=query_vector, category=category, limit=limit * 2, min_similarity=0.0
        )
        semantic_map = {rec.id: (rec, score) for rec, score in semantic_candidates}

        lexical_results = self.search(query=query, category=category, limit=limit * 2)

        combined_scores: dict[str, tuple[MemoryRecord, float]] = {}

        for rank, rec in enumerate(lexical_results):
            lexical_score = 1.0 / (1.0 + rank * 0.1)
            sem_score = semantic_map[rec.id][1] if rec.id in semantic_map else 0.0
            fused = (alpha * sem_score) + ((1.0 - alpha) * lexical_score)
            fused *= rec.confidence
            combined_scores[rec.id] = (rec, round(fused, 4))

        for rec_id, (rec, sem_score) in semantic_map.items():
            if rec_id not in combined_scores:
                fused = alpha * sem_score * rec.confidence
                combined_scores[rec_id] = (rec, round(fused, 4))

        results = list(combined_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

