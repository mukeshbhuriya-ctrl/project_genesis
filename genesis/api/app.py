"""
FastAPI application factory for Project Genesis.
"""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from genesis import __version__
    from genesis.core import get_settings
    from genesis.core.logging import setup_logging
    from genesis.api.routes import health, chat, memory, models, knowledge, backup, tools, tasks, evaluation, training, research, deployment

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.data_dir / "logs")

    app = FastAPI(
        title="Project Genesis",
        description="Local-first AI research and model development platform",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — open locally for dev/tools
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(memory.router, prefix="/api", tags=["Memory"])
    app.include_router(models.router, prefix="/api", tags=["Models"])
    app.include_router(knowledge.router, prefix="/api", tags=["Knowledge"])
    app.include_router(backup.router, prefix="/api", tags=["Backups"])
    app.include_router(tools.router, prefix="/api", tags=["Tools"])
    app.include_router(tasks.router, prefix="/api", tags=["Tasks"])
    app.include_router(evaluation.router, prefix="/api", tags=["Evaluation"])
    app.include_router(training.router, prefix="/api", tags=["Training"])
    app.include_router(research.router, prefix="/api", tags=["Research"])
    app.include_router(deployment.router, prefix="/api", tags=["Deployment"])


    # Static Web UI Assets
    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def serve_ui():
            index_path = web_dir / "index.html"
            if index_path.is_file():
                return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
            return HTMLResponse(content="<h1>Project Genesis API Active</h1><p>UI not found.</p>")

    return app
