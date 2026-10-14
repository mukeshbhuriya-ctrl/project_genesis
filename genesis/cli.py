"""
Project Genesis CLI entry point.

Commands:
    init      — Initialize database and data directories
    serve     — Start the API server
    hardware  — Print hardware diagnostic report
    status    — Show project status
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force UTF-8 output on Windows to avoid cp1252 encoding errors with
# Unicode characters used by Rich (checkmarks, emoji, box-drawing, etc.).
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Ensure stdout/stderr use UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console(force_terminal=True)


@click.group()
@click.version_option(package_name="project-genesis")
def main() -> None:
    """Project Genesis -- Local-first AI research platform."""
    pass


@main.command()
@click.option("--data-dir", type=click.Path(), default=None, help="Data directory path")
def init(data_dir: str | None) -> None:
    """Initialize the database and data directories."""
    from genesis.core import get_settings
    from genesis.core.database import initialize_database, get_db_path_from_url

    settings = get_settings()
    if data_dir:
        settings.data_dir = Path(data_dir)

    console.print("[bold blue]Initializing Project Genesis...[/]")

    # Create directories
    settings.ensure_directories()
    console.print(f"  [green]OK[/] Data directories created at [cyan]{settings.data_dir}[/]")

    # Initialize database
    db_path = get_db_path_from_url(settings.get_db_url())
    initialize_database(db_path)
    console.print(f"  [green]OK[/] Database initialized at [cyan]{db_path}[/]")

    console.print("\n[bold green]Initialization complete![/]")


@main.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the Genesis API server."""
    from genesis.core import get_settings

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port

    console.print(
        Panel(
            f"[bold]Project Genesis[/] API Server\n"
            f"Listening on [cyan]http://{bind_host}:{bind_port}[/]",
            title="Starting",
            border_style="blue",
        )
    )

    try:
        import uvicorn
        uvicorn.run(
            "genesis.api.app:create_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=reload,
            log_level="info",
        )
    except ImportError:
        console.print("[red]Error: uvicorn not installed. Run: pip install uvicorn[standard][/]")
        sys.exit(1)


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def hardware(as_json: bool) -> None:
    """Show hardware capability report."""
    from genesis.core.hardware import build_report
    from genesis.core import get_settings

    settings = get_settings()
    report = build_report(data_dir=settings.data_dir)

    if as_json:
        import json
        console.print_json(json.dumps(report.to_dict(), indent=2))
    else:
        console.print(Panel(report.summary(), title="Hardware Report", border_style="green"))


@main.command()
def status() -> None:
    """Show current project status."""
    from genesis.core import get_settings
    from genesis.core.database import get_db_path_from_url

    settings = get_settings()

    table = Table(title="Project Genesis Status", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Check database
    db_path = get_db_path_from_url(settings.get_db_url())
    if db_path.exists():
        table.add_row("Database", "[green]Ready[/]", str(db_path))
    else:
        table.add_row("Database", "[red]Not initialized[/]", "Run: genesis init")

    # Check data directories
    if settings.data_dir.exists():
        table.add_row("Data Directory", "[green]Exists[/]", str(settings.data_dir))
    else:
        table.add_row("Data Directory", "[red]Missing[/]", "Run: genesis init")

    # Check Ollama
    from genesis.core.hardware import _run_cmd
    ollama_ver = _run_cmd(["ollama", "--version"])
    if ollama_ver:
        table.add_row("Ollama", "[green]Installed[/]", ollama_ver)
    else:
        table.add_row("Ollama", "[red]Not installed[/]", "winget install Ollama.Ollama")

    # Check for models
    if ollama_ver:
        model_list = _run_cmd(["ollama", "list"])
        if model_list and len(model_list.strip().splitlines()) > 1:
            count = len(model_list.strip().splitlines()) - 1  # subtract header
            table.add_row("Models", f"[green]{count} available[/]", "")
        else:
            table.add_row("Models", "[red]None pulled[/]", "ollama pull qwen2.5:7b")

    console.print(table)


if __name__ == "__main__":
    main()
