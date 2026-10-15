"""Backup API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from genesis.core import get_settings
from genesis.core.backup import create_backup, list_backups, verify_backup
from genesis.core.database import get_db_path_from_url

router = APIRouter()


@router.get("/backups")
async def get_backups() -> dict:
    """List all available SQLite point-in-time backup snapshots."""
    settings = get_settings()
    backups = list_backups(settings.backup_dir)
    return {"count": len(backups), "backups": backups}


@router.post("/backups")
async def trigger_backup() -> dict:
    """Trigger an immediate atomic online backup with integrity verification."""
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    try:
        backup_path = create_backup(db_path=db_path, backup_dir=settings.backup_dir)
        is_valid = verify_backup(backup_path)
        return {
            "success": is_valid,
            "filename": backup_path.name,
            "path": str(backup_path),
            "size_bytes": backup_path.stat().st_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")
