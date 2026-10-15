"""Tests for the database module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from genesis.core.database import (
    initialize_database,
    get_sync_connection,
    get_db_path_from_url,
)


class TestDatabaseInitialization:
    """Test database schema creation and migration."""

    def test_initialize_creates_tables(self, tmp_db_path: Path):
        """Initialize should create all required tables."""
        initialize_database(tmp_db_path)

        conn = get_sync_connection(tmp_db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row["name"] for row in cursor.fetchall()}

            assert "conversations" in tables
            assert "messages" in tables
            assert "memories" in tables
            assert "models" in tables
            assert "experiments" in tables
            assert "schema_migrations" in tables
            assert "memory_embeddings" in tables
            assert "knowledge_documents" in tables
            assert "knowledge_chunks" in tables
            assert "tasks" in tables
            assert "task_steps" in tables
            assert "eval_benchmarks" in tables
            assert "eval_cases" in tables
            assert "eval_runs" in tables
            assert "eval_results" in tables
        finally:
            conn.close()

    def test_initialize_is_idempotent(self, tmp_db_path: Path):
        """Running initialize twice should not error."""
        initialize_database(tmp_db_path)
        initialize_database(tmp_db_path)  # Should not raise

        conn = get_sync_connection(tmp_db_path)
        try:
            cursor = conn.execute("SELECT MAX(version) FROM schema_migrations")
            assert cursor.fetchone()[0] >= 1
        finally:
            conn.close()

    def test_schema_version_tracking(self, tmp_db_path: Path):
        """Schema version should be tracked in schema_migrations."""
        initialize_database(tmp_db_path)

        conn = get_sync_connection(tmp_db_path)
        try:
            cursor = conn.execute(
                "SELECT version, description FROM schema_migrations ORDER BY version"
            )
            rows = cursor.fetchall()
            assert len(rows) >= 1
            assert rows[0]["version"] == 1
            assert "Initial schema" in rows[0]["description"]
        finally:
            conn.close()

    def test_foreign_keys_enabled(self, tmp_db_path: Path):
        """Foreign keys should be enforced."""
        initialize_database(tmp_db_path)

        conn = get_sync_connection(tmp_db_path)
        try:
            cursor = conn.execute("PRAGMA foreign_keys")
            assert cursor.fetchone()[0] == 1
        finally:
            conn.close()

    def test_wal_mode(self, tmp_db_path: Path):
        """Database should use WAL journal mode."""
        initialize_database(tmp_db_path)

        conn = get_sync_connection(tmp_db_path)
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            assert cursor.fetchone()[0] == "wal"
        finally:
            conn.close()


class TestDbPathFromUrl:
    """Test URL parsing utility."""

    def test_sqlite_url(self):
        path = get_db_path_from_url("sqlite:///d:/data/genesis.db")
        assert path == Path("d:/data/genesis.db")

    def test_plain_path(self):
        path = get_db_path_from_url("genesis.db")
        assert path == Path("genesis.db")
