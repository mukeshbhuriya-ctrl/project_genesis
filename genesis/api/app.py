"""
FastAPI application factory for Project Genesis.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from genesis import __version__
    from genesis.core import get_settings
    from genesis.core.logging import setup_logging
    from genesis.api.routes import health, chat, memory, models

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.data_dir / "logs")

    app = FastAPI(
        title="Project Genesis",
        description="Local-first AI research and model development platform",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — local development only by default
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(memory.router, prefix="/api", tags=["Memory"])
    app.include_router(models.router, prefix="/api", tags=["Models"])

    return app
