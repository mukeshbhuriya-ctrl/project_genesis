"""Tests for the memory subsystem."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.core.database import initialize_database
from genesis.memory import MemoryStore


@pytest.fixture
def memory_store(tmp_db_path: Path) -> MemoryStore:
    """Provide an initialized memory store."""
    initialize_database(tmp_db_path)
    return MemoryStore(str(tmp_db_path))


class TestMemoryStore:
    """Test memory CRUD operations."""

    def test_create_and_get(self, memory_store: MemoryStore):
        """Create a memory and retrieve it."""
        record = memory_store.create(
            content="Python is a programming language",
            category="fact",
            source="test",
        )

        assert record.id
        assert record.content == "Python is a programming language"
        assert record.category == "fact"
        assert record.version == 1

        # Retrieve
        fetched = memory_store.get(record.id)
        assert fetched is not None
        assert fetched.content == record.content

    def test_update_increments_version(self, memory_store: MemoryStore):
        """Update should increment the version number."""
        record = memory_store.create(content="Initial content")
        updated = memory_store.update(record.id, content="Updated content")

        assert updated is not None
        assert updated.content == "Updated content"
        assert updated.version == 2

    def test_delete(self, memory_store: MemoryStore):
        """Delete should remove the memory."""
        record = memory_store.create(content="To be deleted")
        assert memory_store.delete(record.id) is True
        assert memory_store.get(record.id) is None

    def test_delete_nonexistent(self, memory_store: MemoryStore):
        """Deleting a nonexistent memory should return False."""
        assert memory_store.delete("nonexistent-id") is False

    def test_search_by_text(self, memory_store: MemoryStore):
        """Search should find memories by text content."""
        memory_store.create(content="The sky is blue", category="fact")
        memory_store.create(content="Water is wet", category="fact")
        memory_store.create(content="The sky is vast", category="observation")

        results = memory_store.search(query="sky")
        assert len(results) == 2

    def test_search_by_category(self, memory_store: MemoryStore):
        """Search should filter by category."""
        memory_store.create(content="Fact one", category="fact")
        memory_store.create(content="Preference one", category="preference")

        results = memory_store.search(category="fact")
        assert len(results) == 1
        assert results[0].category == "fact"

    def test_list_categories(self, memory_store: MemoryStore):
        """list_categories should return distinct categories."""
        memory_store.create(content="A", category="fact")
        memory_store.create(content="B", category="preference")
        memory_store.create(content="C", category="fact")

        categories = memory_store.list_categories()
        assert set(categories) == {"fact", "preference"}

    def test_count(self, memory_store: MemoryStore):
        """count should return the number of memories."""
        assert memory_store.count() == 0
        memory_store.create(content="One")
        memory_store.create(content="Two")
        assert memory_store.count() == 2

    def test_count_by_category(self, memory_store: MemoryStore):
        """count with category should filter correctly."""
        memory_store.create(content="Fact", category="fact")
        memory_store.create(content="Pref", category="preference")
        assert memory_store.count(category="fact") == 1

    def test_update_nonexistent(self, memory_store: MemoryStore):
        """Update of nonexistent memory should return None."""
        assert memory_store.update("nope", content="x") is None

    def test_metadata_round_trip(self, memory_store: MemoryStore):
        """Metadata should survive create/get cycle."""
        meta = {"tags": ["important", "test"], "score": 0.95}
        record = memory_store.create(content="With metadata", metadata=meta)
        fetched = memory_store.get(record.id)
        assert fetched is not None
        assert fetched.metadata == meta

    def test_confidence(self, memory_store: MemoryStore):
        """Confidence values should be stored correctly."""
        record = memory_store.create(content="Uncertain", confidence=0.5)
        fetched = memory_store.get(record.id)
        assert fetched is not None
        assert fetched.confidence == 0.5
