"""
Unit tests for Device Topology and Heterogeneous Compute Orchestration (genesis/core/device.py).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.core.device import (
    ComputeDevice,
    DeviceOrchestrator,
    DeviceTopology,
    LayerDistributionPlan,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_topology.db"
    initialize_database(db)
    return db


class TestDeviceTopology:
    """Test device discovery, layer allocation calculations, and persistence."""

    def test_discover_topology_live(self, test_db: Path):
        orchestrator = DeviceOrchestrator(db_path=test_db)
        topo = orchestrator.discover_topology()

        assert topo.id is not None
        assert topo.hostname != ""
        assert len(topo.devices) >= 1
        assert topo.total_ram_gb > 0.0
        assert topo.compute_backend in ("cuda", "cpu")

        # Verify device schema
        primary_dev = topo.primary_device
        assert primary_dev is not None
        assert primary_dev.id.startswith(("gpu:", "cpu:"))
        assert primary_dev.name != ""

    def test_layer_distribution_single_gpu(self, test_db: Path):
        orchestrator = DeviceOrchestrator(db_path=test_db)
        # Small model (1.5B, 4-bit) on 8GB GPU
        plan = orchestrator.calculate_layer_distribution(
            model_name="qwen2.5-coder:1.5b-base",
            total_layers=28,
            parameter_size_b=1.5,
            quant_bits=4,
            context_tokens=4096,
            safety_reserve_mb=1024.0,
        )

        assert plan.model_name == "qwen2.5-coder:1.5b-base"
        assert plan.total_layers == 28
        assert plan.strategy in ("single_gpu", "cpu_only")
        if plan.strategy == "single_gpu":
            assert plan.is_full_offload is True
            assert "gpu:0" in plan.device_layers
            assert plan.device_layers["gpu:0"] == 28
            assert plan.estimated_vram_usage_mb["gpu:0"] > 0

    def test_layer_distribution_cpu_offload_fallback(self, test_db: Path):
        orchestrator = DeviceOrchestrator(db_path=test_db)
        # Enormous model (70B, 16-bit FP16 ~140 GB weights) that cannot fit into 8 GB GPU
        plan = orchestrator.calculate_layer_distribution(
            model_name="llama-3-70b-fp16",
            total_layers=80,
            parameter_size_b=70.0,
            quant_bits=16,
            context_tokens=8192,
            safety_reserve_mb=1024.0,
        )

        assert plan.total_layers == 80
        if "gpu:0" in plan.device_layers:
            assert plan.strategy == "cpu_offload"
            assert plan.is_full_offload is False
            assert "cpu:0" in plan.device_layers
            assert plan.device_layers["gpu:0"] + plan.device_layers["cpu:0"] == 80

    def test_layer_distribution_multi_gpu_simulation(self, test_db: Path, monkeypatch):
        orchestrator = DeviceOrchestrator(db_path=test_db)

        # Mock multi-GPU topology
        mock_devices = [
            ComputeDevice(id="gpu:0", name="NVIDIA RTX 4090", device_type="cuda", vram_total_mb=24576.0, is_primary=True),
            ComputeDevice(id="gpu:1", name="NVIDIA RTX 4090", device_type="cuda", vram_total_mb=24576.0, is_primary=False),
            ComputeDevice(id="cpu:0", name="Host CPU", device_type="cpu", vram_total_mb=65536.0),
        ]
        mock_topo = DeviceTopology(
            id="mock-multi-gpu",
            hostname="deep-thought",
            devices=mock_devices,
            total_vram_mb=49152.0,
            total_ram_gb=64.0,
            compute_backend="cuda",
        )
        monkeypatch.setattr(orchestrator, "discover_topology", lambda: mock_topo)

        plan = orchestrator.calculate_layer_distribution(
            model_name="qwen2.5:32b",
            total_layers=64,
            parameter_size_b=32.0,
            quant_bits=4,
            context_tokens=8192,
        )

        assert plan.strategy == "pipeline_parallel"
        assert plan.is_full_offload is True
        assert "gpu:0" in plan.device_layers
        assert "gpu:1" in plan.device_layers
        assert plan.device_layers["gpu:0"] + plan.device_layers["gpu:1"] == 64
        assert plan.metadata.get("gpu_count") == 2

    def test_topology_persistence_and_query(self, test_db: Path):
        orchestrator = DeviceOrchestrator(db_path=test_db)
        created = orchestrator.discover_topology()

        latest = orchestrator.get_latest_topology()
        assert latest is not None
        assert latest.id == created.id
        assert latest.hostname == created.hostname
        assert len(latest.devices) == len(created.devices)
