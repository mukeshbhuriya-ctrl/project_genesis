"""Tests for the configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from genesis.core import GenesisSettings, get_settings, reset_settings


class TestGenesisSettings:
    """Test configuration loading and defaults."""

    def setup_method(self):
        reset_settings()

    def test_default_values(self):
        """Settings should have sensible defaults."""
        settings = GenesisSettings()
        assert settings.env == "development"
        assert settings.log_level == "INFO"
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.default_backend == "ollama"

    def test_db_url_default(self):
        """Default DB URL should be SQLite in the data directory."""
        settings = GenesisSettings()
        db_url = settings.get_db_url()
        assert db_url.startswith("sqlite:///")
        assert "genesis.db" in db_url

    def test_custom_db_url(self):
        """Custom DB URL should override the default."""
        settings = GenesisSettings(db_url="sqlite:///custom.db")
        assert settings.get_db_url() == "sqlite:///custom.db"

    def test_ensure_directories(self, tmp_path: Path):
        """ensure_directories should create all required subdirectories."""
        settings = GenesisSettings(data_dir=tmp_path / "testdata")
        settings.ensure_directories()

        assert (tmp_path / "testdata").exists()
        assert (tmp_path / "testdata" / "knowledge").exists()
        assert (tmp_path / "testdata" / "logs").exists()

    def test_gpu_enabled_default(self):
        """GPU should default to auto-detection."""
        settings = GenesisSettings()
        assert settings.gpu_enabled == "auto"

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("GENESIS_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GENESIS_PORT", "9000")
        reset_settings()
        settings = GenesisSettings()
        assert settings.log_level == "DEBUG"
        assert settings.port == 9000
