"""
Tests for Document Ingestion, Chunking, and Knowledge Retrieval.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock
import pytest

from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.knowledge.indexer import KnowledgeIndexer, chunk_text, compute_file_checksum
from genesis.memory.embeddings import EmbeddingEngine


class TestTextChunking:
    """Test text splitting and chunk boundaries."""

    def test_chunk_short_text(self):
        text = "Short single paragraph document."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_paragraphs(self):
        p1 = "First paragraph containing introductory concepts."
        p2 = "Second paragraph elaborating on implementation details."
        p3 = "Third paragraph summarizing findings."
        doc = f"{p1}\n\n{p2}\n\n{p3}"

        chunks = chunk_text(doc, chunk_size=80, chunk_overlap=10)
        assert len(chunks) >= 2


@pytest.fixture
def knowledge_indexer(tmp_path: Path) -> KnowledgeIndexer:
    """Provide a KnowledgeIndexer instance with an initialized test database."""
    reset_settings()
    db_file = tmp_path / "test_knowledge.db"
    initialize_database(db_file)

    # Mock embedding engine to be fast and deterministic in unit tests
    mock_engine = AsyncMock(spec=EmbeddingEngine)
    mock_engine.get_embedding.side_effect = lambda text: [0.1] * 768
    mock_engine.get_embeddings.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

    return KnowledgeIndexer(str(db_file), engine=mock_engine)


class TestKnowledgeIndexer:
    """Test document indexing lifecycle and retrieval."""

    @pytest.mark.asyncio
    async def test_index_file_and_retrieve_chunks(self, tmp_path: Path, knowledge_indexer: KnowledgeIndexer):
        # Create sample document
        sample_file = tmp_path / "architecture_notes.md"
        sample_file.write_text(
            "# Architecture Notes\n\nProject Genesis uses modular subsystems.\n\n"
            "The memory subsystem provides vector embeddings and persistent storage.",
            encoding="utf-8",
        )

        doc = await knowledge_indexer.index_file(sample_file, title="Architecture Guide")
        assert doc.id
        assert doc.title == "Architecture Guide"
        assert doc.chunk_count >= 1

        docs = knowledge_indexer.list_documents()
        assert len(docs) == 1
        assert docs[0]["title"] == "Architecture Guide"

        # Search
        results = await knowledge_indexer.search("vector embeddings", limit=3, min_score=0.1)
        assert len(results) >= 1
        assert results[0]["document_title"] == "Architecture Guide"
        assert "subsystems" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_skip_indexing_unchanged_file(self, tmp_path: Path, knowledge_indexer: KnowledgeIndexer):
        sample_file = tmp_path / "static_doc.txt"
        sample_file.write_text("Immutable specification document content.", encoding="utf-8")

        doc1 = await knowledge_indexer.index_file(sample_file)
        # Re-index same file without modifications
        doc2 = await knowledge_indexer.index_file(sample_file)

        assert doc1.id == doc2.id
        assert doc1.checksum == doc2.checksum

    @pytest.mark.asyncio
    async def test_delete_document_cascades(self, tmp_path: Path, knowledge_indexer: KnowledgeIndexer):
        sample_file = tmp_path / "temporary.md"
        sample_file.write_text("Temporary notes to be purged.", encoding="utf-8")

        doc = await knowledge_indexer.index_file(sample_file)
        assert len(knowledge_indexer.list_documents()) == 1

        deleted = knowledge_indexer.delete_document(doc.id)
        assert deleted is True
        assert len(knowledge_indexer.list_documents()) == 0

        # Search should return empty
        results = await knowledge_indexer.search("Temporary notes")
        assert len(results) == 0
