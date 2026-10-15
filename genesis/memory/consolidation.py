"""
Memory consolidation, deduplication, and contradiction resolution.
"""

from __future__ import annotations

from typing import Any

from genesis.core.logging import get_logger
from genesis.memory import MemoryRecord, MemoryStore

log = get_logger("memory.consolidation")


def find_duplicate(
    store: MemoryStore,
    vector: list[float],
    threshold: float = 0.90,
    category: str | None = None,
) -> tuple[MemoryRecord, float] | None:
    """
    Search for an existing memory record that is semantically near-identical.
    Returns (MemoryRecord, similarity_score) if above threshold, else None.
    """
    candidates = store.search_semantic(
        query_vector=vector,
        category=category,
        limit=1,
        min_similarity=threshold,
    )
    if candidates:
        return candidates[0]
    return None


def consolidate_memory(
    store: MemoryStore,
    content: str,
    category: str = "general",
    source: str = "",
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
    vector: list[float] | None = None,
    threshold: float = 0.90,
) -> tuple[MemoryRecord, bool]:
    """
    Consolidate a candidate memory into the store:
    - If a near-identical memory exists (similarity >= threshold), update its version,
      timestamp, confidence, and metadata without inserting a duplicate.
    - Otherwise, create a new record with vector embedding.

    Returns:
        tuple[MemoryRecord, bool]: (memory_record, is_new)
    """
    if vector:
        duplicate_match = find_duplicate(store, vector, threshold=threshold, category=category)
        if duplicate_match:
            existing_record, similarity = duplicate_match
            log.info(
                f"Consolidation: match found for '{content[:40]}...' -> "
                f"existing {existing_record.id[:8]} (sim={similarity:.3f}). Updating existing."
            )
            # Merge metadata
            merged_meta = dict(existing_record.metadata)
            if metadata:
                merged_meta.update(metadata)
            merged_meta["last_duplicate_suppressed_at"] = existing_record.updated_at
            merged_meta["consolidation_match_similarity"] = similarity

            # Average or max confidence
            updated_confidence = max(existing_record.confidence, confidence)

            updated = store.update(
                memory_id=existing_record.id,
                content=content if confidence > existing_record.confidence else existing_record.content,
                confidence=updated_confidence,
                metadata=merged_meta,
            )
            if updated:
                return updated, False

    # No duplicate found or no vector provided -> insert fresh
    created = store.create(
        content=content,
        category=category,
        source=source,
        confidence=confidence,
        metadata=metadata,
        embedding=vector,
    )
    return created, True
