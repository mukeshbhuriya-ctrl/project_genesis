"""
Atomic backup and recovery for Project Genesis SQLite databases.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from genesis.core import get_settings
from genesis.core.logging import get_logger

log = get_logger("core.backup")


def create_backup(
    db_path: Path | str | None = None,
    backup_dir: Path | str | None = None,
) -> Path:
    """
    Create a verified, point-in-time atomic snapshot of the SQLite database.
    Uses SQLite's online backup API so operations can proceed concurrently.
    """
    settings = get_settings()
    source_path = Path(db_path) if db_path else Path(settings.data_dir / "genesis.db")
    target_dir = Path(backup_dir) if backup_dir else Path(settings.backup_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = target_dir / f"genesis_backup_{timestamp}.db"

    source_conn = sqlite3.connect(str(source_path))
    dest_conn = sqlite3.connect(str(backup_path))

    try:
        source_conn.backup(dest_conn)
        log.info(f"Database backed up to {backup_path}")
    finally:
        dest_conn.close()
        source_conn.close()

    # Integrity verification on backup
    if not verify_backup(backup_path):
        raise RuntimeError(f"Backup verification failed for {backup_path}")

    return backup_path


def verify_backup(backup_path: Path | str) -> bool:
    """Verify that a backup database passes SQLite integrity check."""
    path = Path(backup_path)
    if not path.is_file():
        return False

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.execute("PRAGMA integrity_check")
        rows = cursor.fetchall()
        return len(rows) == 1 and rows[0][0] == "ok"
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def restore_backup(backup_path: Path | str, target_db_path: Path | str) -> bool:
    """Restore database from a verified backup snapshot."""
    src = Path(backup_path)
    target = Path(target_db_path)

    if not verify_backup(src):
        log.error(f"Cannot restore: backup {src} failed integrity verification.")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src))
    target_conn = sqlite3.connect(str(target))

    try:
        src_conn.backup(target_conn)
        log.info(f"Database restored successfully from {src} to {target}")
        return True
    finally:
        target_conn.close()
        src_conn.close()


def list_backups(backup_dir: Path | str | None = None) -> list[dict]:
    """List all available backup snapshots ordered by newest first."""
    settings = get_settings()
    target_dir = Path(backup_dir) if backup_dir else Path(settings.backup_dir)
    if not target_dir.exists():
        return []

    backups = []
    for file in sorted(target_dir.glob("genesis_backup_*.db"), reverse=True):
        stat = file.stat()
        backups.append({
            "path": str(file),
            "filename": file.name,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return backups
