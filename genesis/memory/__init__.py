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
    ) -> MemoryRecord:
        """Create a new memory record."""
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
            conn.commit()

            log.info(f"Created memory {record_id[:8]}... [{category}]")
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
