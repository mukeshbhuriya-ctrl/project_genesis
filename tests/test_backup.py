"""
Tests for Atomic Backup and Disaster Recovery.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest

from genesis.core.backup import create_backup, verify_backup, restore_backup, list_backups
from genesis.core.database import initialize_database


@pytest.fixture
def test_db_with_data(tmp_path: Path) -> Path:
    """Create a populated test database."""
    db_file = tmp_path / "origin.db"
    initialize_database(db_file)

    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "INSERT INTO conversations (id, title, model_id) VALUES (?, ?, ?)",
        ("conv-123", "Backup Verification Dialogue", "qwen2.5:7b"),
    )
    conn.commit()
    conn.close()
    return db_file


class TestBackupAndRecovery:
    """Test backup snapshot creation, verification, and restore."""

    def test_create_and_verify_backup(self, tmp_path: Path, test_db_with_data: Path):
        backup_dir = tmp_path / "backups"
        backup_file = create_backup(db_path=test_db_with_data, backup_dir=backup_dir)

        assert backup_file.is_file()
        assert backup_file.stat().st_size > 0
        assert verify_backup(backup_file) is True

        backups = list_backups(backup_dir=backup_dir)
        assert len(backups) == 1
        assert backups[0]["filename"] == backup_file.name

    def test_restore_backup(self, tmp_path: Path, test_db_with_data: Path):
        backup_dir = tmp_path / "backups"
        backup_file = create_backup(db_path=test_db_with_data, backup_dir=backup_dir)

        # Restore into a brand new destination
        restored_db = tmp_path / "restored.db"
        assert restore_backup(backup_file, restored_db) is True

        # Verify restored contents
        conn = sqlite3.connect(str(restored_db))
        cursor = conn.execute("SELECT title FROM conversations WHERE id = 'conv-123'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "Backup Verification Dialogue"

    def test_verify_invalid_backup_returns_false(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.db"
        corrupt_file.write_text("NOT A VALID SQLITE DATABASE", encoding="utf-8")
        assert verify_backup(corrupt_file) is False
