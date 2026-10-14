"""
Structured logging for Project Genesis.

Provides both human-readable console output (via Rich) and
machine-parseable JSON log files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


_configured = False


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    json_logs: bool = False,
) -> logging.Logger:
    """
    Configure the root logger for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If None, logs go to console only.
        json_logs: If True, file logs are JSON-formatted.

    Returns:
        The configured root logger.
    """
    global _configured
    if _configured:
        return logging.getLogger("genesis")

    root = logging.getLogger("genesis")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # ── Console handler (Rich) ──
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    # ── File handler (optional) ──
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "genesis.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        if json_logs:
            # JSON-lines format for machine parsing
            file_fmt = logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","message":"%(message)s"}'
            )
        else:
            file_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        file_handler.setFormatter(file_fmt)
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the genesis namespace."""
    return logging.getLogger(f"genesis.{name}")
