"""
Core configuration for Project Genesis.

Loads settings from environment variables and .env file.
All paths and tunables are centralized here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> str | None:
    """Walk up from CWD to find a .env file."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return str(candidate)
    return None


class GenesisSettings(BaseSettings):
    """Application-wide settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_prefix="GENESIS_",
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──
    env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    secret_key: str = "change-me-to-a-random-secret"

    # ── Paths ──
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    model_dir: Path = Field(default_factory=lambda: Path("data/models"))
    dataset_dir: Path = Field(default_factory=lambda: Path("data/datasets"))
    experiment_dir: Path = Field(default_factory=lambda: Path("data/experiments"))
    backup_dir: Path = Field(default_factory=lambda: Path("data/backups"))

    # ── Database ──
    db_url: str = ""

    # ── Model runtime ──
    default_backend: str = "ollama"
    default_model: str = ""
    embedding_model: str = ""

    # ── Hardware hints ──
    gpu_enabled: str = "auto"
    max_ram_gb: float | None = None
    max_vram_gb: float | None = None

    @property
    def project_root(self) -> Path:
        """Root directory of the Genesis project."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def database_url(self) -> str:
        """Return the effective database URL."""
        return self.get_db_url()

    def get_db_url(self) -> str:
        """Return the database URL, defaulting to a SQLite file inside data_dir."""
        if self.db_url:
            return self.db_url
        db_path = self.data_dir / "genesis.db"
        return f"sqlite:///{db_path}"

    def ensure_directories(self) -> None:
        """Create all configured data directories if they don't exist."""
        for directory in [
            self.data_dir,
            self.model_dir,
            self.dataset_dir,
            self.experiment_dir,
            self.backup_dir,
            self.data_dir / "knowledge",
            self.data_dir / "knowledge" / "documents",
            self.data_dir / "knowledge" / "index",
            self.data_dir / "logs",
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
_settings: GenesisSettings | None = None


def get_settings() -> GenesisSettings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = GenesisSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
