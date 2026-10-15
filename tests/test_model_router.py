"""Unit tests for the dynamic task router and multi-model dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.database import initialize_database
from genesis.core.device import ComputeDevice, DeviceTopology, DeviceOrchestrator
from genesis.models.ollama_backend import OllamaProvider
from genesis.models.router import ModelRouter, RouteDecision, TASK_CATEGORIES


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_genesis.db"
    initialize_database(db_file)
    return db_file


@pytest.fixture
def mock_provider() -> OllamaProvider:
    provider = MagicMock(spec=OllamaProvider)
    # Mock list_models to return typical installed local models
    m1 = MagicMock()
    m1.id = "qwen2.5:7b"
    m1.name = "qwen2.5:7b"
    m2 = MagicMock()
    m2.id = "qwen2.5-coder:7b"
    m2.name = "qwen2.5-coder:7b"
    m3 = MagicMock()
    m3.id = "qwen2.5-coder:1.5b-base"
    m3.name = "qwen2.5-coder:1.5b-base"
    m4 = MagicMock()
    m4.id = "nomic-embed-text:latest"
    m4.name = "nomic-embed-text:latest"
    provider.list_models = AsyncMock(return_value=[m1, m2, m3, m4])
    return provider


@pytest.fixture
def mock_topology_mgr(test_db: Path) -> DeviceOrchestrator:
    mgr = MagicMock(spec=DeviceOrchestrator)
    devices = [
        ComputeDevice(
            id="gpu:0",
            name="NVIDIA RTX 2000 Ada",
            device_type="cuda",
            vram_total_mb=8192.0,
            vram_free_mb=6500.0,
            is_primary=True,
        )
    ]
    topo = DeviceTopology(
        id="test-topo",
        hostname="test-host",
        devices=devices,
        total_vram_mb=8192.0,
        total_ram_gb=32.0,
        compute_backend="cuda",
    )
    mgr.discover_topology = MagicMock(return_value=topo)
    return mgr


class TestModelRouterClassification:
    """Tests for classifying prompts into task categories."""

    def test_classify_coding_prompts(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        cat, conf = router.classify_task("def calculate_fibonacci(n: int) -> int:\n    return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)")
        assert cat == "coding"
        assert conf > 0.7

        cat2, _ = router.classify_task("Please debug this python script, pytest test harness failed with syntax error")
        assert cat2 == "coding"

    def test_classify_fast_validation_prompts(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        cat, conf = router.classify_task("check if this is valid json syntax and verify schema")
        assert cat == "fast_validation"
        assert conf > 0.65

    def test_classify_reasoning_prompts(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        cat, conf = router.classify_task("Explain step by step why we choose WAL mode and analyze trade-offs vs rollback journal")
        assert cat == "reasoning"
        assert conf > 0.65

    def test_classify_embedding_prompts(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        cat, conf = router.classify_task("Generate dense vector embedding and calculate cosine similarity")
        assert cat == "embedding"
        assert conf >= 0.9

    def test_classify_general_fallback(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        cat, conf = router.classify_task("Good morning, what is the agenda for today?")
        assert cat == "general"


class TestModelRouterRoutesCRUD:
    """Tests for SQLite persistence of model routes."""

    def test_list_seeded_routes(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)
        routes = router.list_routes()
        categories = {r.task_category for r in routes}
        assert set(TASK_CATEGORIES) == categories

    def test_get_and_set_route(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        # Update route
        updated = router.set_route(
            task_category="coding",
            preferred_model="qwen2.5-coder:14b",
            fallback_model="qwen2.5-coder:7b",
            min_vram_mb=9000.0,
            metadata={"notes": "experimental 14b upgrade"},
        )
        assert updated.preferred_model == "qwen2.5-coder:14b"
        assert updated.min_vram_mb == 9000.0

        retrieved = router.get_route("coding")
        assert retrieved is not None
        assert retrieved.preferred_model == "qwen2.5-coder:14b"
        assert retrieved.metadata.get("notes") == "experimental 14b upgrade"


class TestModelRouterDispatch:
    """Tests for dynamic model routing with availability and VRAM checks."""

    @pytest.mark.asyncio
    async def test_route_preferred_when_available(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        decision = await router.route("Write a python function to compute primes with unit tests")
        assert decision.task_category == "coding"
        assert decision.selected_model == "qwen2.5-coder:7b"
        assert decision.is_fallback is False

    @pytest.mark.asyncio
    async def test_route_fallback_when_preferred_not_installed(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        # Restrict installed models to only qwen2.5:7b
        m = MagicMock()
        m.id = "qwen2.5:7b"
        m.name = "qwen2.5:7b"
        mock_provider.list_models = AsyncMock(return_value=[m])

        decision = await router.route("Write a python function to compute primes with unit tests")
        assert decision.task_category == "coding"
        assert decision.selected_model == "qwen2.5:7b"
        assert decision.is_fallback is True
        assert "not installed" in decision.reason

    @pytest.mark.asyncio
    async def test_route_fallback_when_insufficient_vram(self, test_db: Path, mock_provider: OllamaProvider, mock_topology_mgr: DeviceOrchestrator):
        router = ModelRouter(db_path=test_db, provider=mock_provider, topology_mgr=mock_topology_mgr)

        # Set low free VRAM (e.g. 1500 MB free, but coding requires 4000 MB)
        low_vram_device = ComputeDevice(
            id="gpu:0",
            name="NVIDIA GPU",
            device_type="cuda",
            vram_total_mb=8192.0,
            vram_free_mb=1200.0,
            is_primary=True,
        )
        mock_topology_mgr.discover_topology = MagicMock(
            return_value=DeviceTopology(
                id="low-vram-topo",
                hostname="test-host",
                devices=[low_vram_device],
                total_vram_mb=8192.0,
                total_ram_gb=32.0,
                compute_backend="cuda",
            )
        )

        decision = await router.route(
            "Write a python function to compute primes",
            require_vram_mb=5000.0,
        )
        assert decision.is_fallback is True
        assert "VRAM headroom" in decision.reason
