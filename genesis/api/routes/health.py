"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from genesis import __version__

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check."""
    return {
        "status": "healthy",
        "version": __version__,
        "service": "project-genesis",
    }


@router.get("/health/detailed")
async def detailed_health() -> dict:
    """Detailed health check including subsystem status."""
    from genesis.core import get_settings
    from genesis.core.database import get_db_path_from_url
    from genesis.models.ollama_backend import OllamaProvider

    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())

    # Check Ollama
    ollama = OllamaProvider()
    ollama_ok = await ollama.is_available()

    return {
        "status": "healthy",
        "version": __version__,
        "subsystems": {
            "database": {"ok": db_path.exists(), "path": str(db_path)},
            "ollama": {"ok": ollama_ok},
            "data_dir": {"ok": settings.data_dir.exists(), "path": str(settings.data_dir)},
        },
    }
