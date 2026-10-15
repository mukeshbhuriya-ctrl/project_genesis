"""
Tests for Phase 2 Memory Embeddings, Hybrid Retrieval, and Consolidation.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.memory import MemoryStore
from genesis.memory.embeddings import pack_vector, unpack_vector, cosine_similarity
from genesis.memory.consolidation import consolidate_memory, find_duplicate


class TestVectorMath:
    """Test vector packing, unpacking, and similarity calculation."""

    def test_pack_and_unpack_roundtrip(self):
        original = [0.123456, -0.987654, 0.0, 1.0, 42.5]
        packed = pack_vector(original)
        assert isinstance(packed, bytes)
        assert len(packed) == len(original) * 4  # 4 bytes per float32

        unpacked = unpack_vector(packed)
        assert len(unpacked) == len(original)
        for orig, unp in zip(original, unpacked):
            assert abs(orig - unp) < 1e-5

    def test_cosine_similarity_identical(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_cosine_similarity_opposite(self):
        v1 = [1.0, 2.0]
        v2 = [-1.0, -2.0]
        assert abs(cosine_similarity(v1, v2) - (-1.0)) < 1e-6

    def test_cosine_similarity_empty_or_mismatched(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


@pytest.fixture
def memory_db(tmp_path: Path) -> MemoryStore:
    """Provide an initialized MemoryStore with Schema v2."""
    reset_settings()
    db_file = tmp_path / "test_memory_embeddings.db"
    initialize_database(db_file)
    return MemoryStore(str(db_file))


class TestMemoryVectorStore:
    """Test vector storage, semantic search, and hybrid ranking."""

    def test_create_with_embedding_and_retrieve(self, memory_db: MemoryStore):
        vec = [0.1] * 768
        record = memory_db.create(
            content="Local first AI development on Windows",
            category="architecture",
            embedding=vec,
            embedding_model="nomic-embed-text",
        )
        assert record.id

        stored_vec = memory_db.get_embedding(record.id)
        assert stored_vec is not None
        assert len(stored_vec) == 768
        assert abs(stored_vec[0] - 0.1) < 1e-5

    def test_store_embedding_existing(self, memory_db: MemoryStore):
        record = memory_db.create(content="Memory without embedding initially")
        assert memory_db.get_embedding(record.id) is None

        vec = [0.5] * 128
        success = memory_db.store_embedding(record.id, vec, model="test-embed")
        assert success is True

        retrieved = memory_db.get_embedding(record.id)
        assert retrieved is not None
        assert len(retrieved) == 128

    def test_search_semantic_ranking(self, memory_db: MemoryStore):
        # Record 1: close to query vector
        memory_db.create(
            content="GPU accelerated transformer inference",
            category="ai",
            embedding=[1.0, 0.0, 0.0],
        )
        # Record 2: somewhat related
        memory_db.create(
            content="Database indexing and query planning",
            category="db",
            embedding=[0.7071, 0.7071, 0.0],
        )
        # Record 3: orthogonal / unrelated
        memory_db.create(
            content="Making artisan sourdough bread",
            category="cooking",
            embedding=[0.0, 1.0, 0.0],
        )

        query_vec = [1.0, 0.0, 0.0]
        results = memory_db.search_semantic(query_vec, limit=3, min_similarity=0.1)

        assert len(results) == 2
        top_rec, top_score = results[0]
        assert top_rec.content == "GPU accelerated transformer inference"
        assert abs(top_score - 1.0) < 1e-5

        second_rec, second_score = results[1]
        assert second_rec.content == "Database indexing and query planning"
        assert abs(second_score - 0.7071) < 1e-3

    def test_search_hybrid(self, memory_db: MemoryStore):
        # Matching both keyword and vector
        memory_db.create(
            content="SQLite WAL mode offers fast concurrency",
            category="storage",
            embedding=[0.9, 0.1, 0.0],
        )
        # Matching vector only
        memory_db.create(
            content="High throughput persistent disk storage",
            category="storage",
            embedding=[0.95, 0.0, 0.0],
        )
        # Matching lexical keyword only
        memory_db.create(
            content="Concurrency techniques in Python async",
            category="dev",
            embedding=[0.0, 0.9, 0.1],
        )

        query_vec = [1.0, 0.0, 0.0]
        results = memory_db.search_hybrid(
            query="concurrency",
            query_vector=query_vec,
            limit=3,
            alpha=0.6,
        )

        assert len(results) == 3
        # Top result should balance both semantic proximity and lexical hit
        top_rec, score = results[0]
        assert "SQLite WAL mode" in top_rec.content


class TestMemoryConsolidation:
    """Test memory deduplication and suppression."""

    def test_deduplication_suppresses_duplicate(self, memory_db: MemoryStore):
        vec1 = [0.5, 0.5, 0.5]
        rec1, is_new1 = consolidate_memory(
            store=memory_db,
            content="Default model is Qwen 2.5 7B",
            category="config",
            vector=vec1,
            threshold=0.95,
        )
        assert is_new1 is True
        assert rec1.version == 1

        # Near-identical vector (similarity > 0.99)
        vec2 = [0.501, 0.499, 0.5]
        rec2, is_new2 = consolidate_memory(
            store=memory_db,
            content="Default model is Qwen 2.5 7B",
            category="config",
            vector=vec2,
            threshold=0.95,
        )
        assert is_new2 is False
        assert rec2.id == rec1.id
        assert rec2.version == 2
        assert memory_db.count() == 1
