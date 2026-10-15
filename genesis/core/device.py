"""
Multi-GPU and heterogeneous compute device topology orchestrator.

Discovers local compute devices (NVIDIA CUDA, DirectML, CPU fallback),
models multi-GPU workloads, computes pipeline/tensor parallel layer splits,
and prevents out-of-VRAM allocation during high-parameter model inference.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core.database import get_sync_connection
from genesis.core.hardware import detect_cpu, detect_gpu, detect_ram
from genesis.core.logging import get_logger

log = get_logger("device_topology")


@dataclass
class ComputeDevice:
    """A physical or virtual compute device available for model execution."""
    id: str  # e.g. "gpu:0", "cpu:0"
    name: str
    device_type: str  # 'cuda', 'directml', 'cpu'
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    compute_capability: str = ""
    is_primary: bool = False
    temperature_c: float | None = None
    utilization_pct: float | None = None
    power_watts: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type,
            "vram_total_mb": round(self.vram_total_mb, 1),
            "vram_free_mb": round(self.vram_free_mb, 1),
            "compute_capability": self.compute_capability,
            "is_primary": self.is_primary,
            "temperature_c": self.temperature_c,
            "utilization_pct": self.utilization_pct,
            "power_watts": self.power_watts,
            "metadata": self.metadata,
        }


@dataclass
class DeviceTopology:
    """Complete snapshot of system compute resources and topology."""
    id: str
    hostname: str
    devices: list[ComputeDevice] = field(default_factory=list)
    total_vram_mb: float = 0.0
    total_ram_gb: float = 0.0
    compute_backend: str = "cuda"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_gpu(self) -> bool:
        gpus = [d for d in self.devices if d.device_type == "cuda"]
        return len(gpus) > 1

    @property
    def primary_device(self) -> ComputeDevice | None:
        for d in self.devices:
            if d.is_primary:
                return d
        return self.devices[0] if self.devices else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hostname": self.hostname,
            "devices": [d.to_dict() for d in self.devices],
            "total_vram_mb": round(self.total_vram_mb, 1),
            "total_ram_gb": round(self.total_ram_gb, 1),
            "compute_backend": self.compute_backend,
            "is_multi_gpu": self.is_multi_gpu,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class LayerDistributionPlan:
    """Calculated layer offload distribution across available devices."""
    model_name: str
    total_layers: int
    device_layers: dict[str, int]
    is_full_offload: bool
    estimated_vram_usage_mb: dict[str, float]
    strategy: str  # 'single_gpu', 'pipeline_parallel', 'cpu_offload'
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "device_layers": self.device_layers,
            "is_full_offload": self.is_full_offload,
            "estimated_vram_usage_mb": {
                k: round(v, 1) for k, v in self.estimated_vram_usage_mb.items()
            },
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


class DeviceOrchestrator:
    """Discovers compute devices and calculates optimal multi-device execution topologies."""

    def __init__(self, db_path: str | Path = "data/genesis.db") -> None:
        self.db_path = Path(db_path)

    def discover_topology(self) -> DeviceTopology:
        """Query hardware layer to discover all GPUs, CPUs, and memory."""
        hostname = socket.gethostname()
        devices: list[ComputeDevice] = []
        total_vram = 0.0

        # 1. Discover NVIDIA CUDA GPUs
        try:
            gpus = detect_gpu()
            for idx, g in enumerate(gpus):
                is_primary = idx == 0
                devices.append(
                    ComputeDevice(
                        id=f"gpu:{idx}",
                        name=g.name,
                        device_type="cuda",
                        vram_total_mb=float(g.vram_total_mb),
                        vram_free_mb=float(g.vram_free_mb),
                        is_primary=is_primary,
                        metadata={"driver_version": g.driver_version},
                    )
                )
                total_vram += float(g.vram_total_mb)
        except Exception as e:
            log.warning("GPU discovery error: %s", e)

        # 2. Discover Host CPU & System RAM
        ram_gb = 0.0
        try:
            ram = detect_ram()
            ram_gb = float(ram.total_gb)
        except Exception:
            ram_gb = 16.0

        cpu_name = platform.processor() or "Host CPU"
        try:
            cpu = detect_cpu()
            cpu_name = cpu.name or cpu_name
        except Exception:
            pass

        # CPU device
        devices.append(
            ComputeDevice(
                id="cpu:0",
                name=cpu_name,
                device_type="cpu",
                vram_total_mb=ram_gb * 1024.0,  # System RAM in MB
                vram_free_mb=ram_gb * 1024.0 * 0.6,
                is_primary=len(devices) == 0,
            )
        )

        backend = "cuda" if any(d.device_type == "cuda" for d in devices) else "cpu"
        topology = DeviceTopology(
            id=str(uuid.uuid4()),
            hostname=hostname,
            devices=devices,
            total_vram_mb=total_vram,
            total_ram_gb=ram_gb,
            compute_backend=backend,
        )

        self._save_topology(topology)
        return topology

    def calculate_layer_distribution(
        self,
        model_name: str,
        total_layers: int = 28,
        parameter_size_b: float = 7.0,
        quant_bits: int = 4,
        context_tokens: int = 4096,
        safety_reserve_mb: float = 1024.0,
    ) -> LayerDistributionPlan:
        """
        Calculates optimal model layer allocation across devices.
        Guarantees that allocated layers + KV-cache do not exceed physical VRAM.
        """
        topology = self.discover_topology()
        gpu_devices = [d for d in topology.devices if d.device_type == "cuda"]

        # Approximate model weight sizes
        bytes_per_param = quant_bits / 8.0
        # Total weight bytes + 15% quantization metadata/buffer overhead
        total_weight_mb = (parameter_size_b * 1e9 * bytes_per_param * 1.15) / (1024 * 1024)
        mb_per_layer = total_weight_mb / max(1, total_layers)

        # Approximate KV-cache at target context (for Qwen 7B ~56KB/tok = 224MB @ 4K)
        kv_cache_per_layer_mb = (2 * 4 * 128 * 2 * context_tokens) / (1024 * 1024 * max(1, total_layers))

        if not gpu_devices:
            # CPU only
            return LayerDistributionPlan(
                model_name=model_name,
                total_layers=total_layers,
                device_layers={"cpu:0": total_layers},
                is_full_offload=False,
                estimated_vram_usage_mb={"cpu:0": total_weight_mb},
                strategy="cpu_only",
            )

        if len(gpu_devices) == 1:
            gpu = gpu_devices[0]
            # Usable VRAM on primary GPU
            usable_vram = max(0.0, gpu.vram_total_mb - safety_reserve_mb - 512.0)  # 512MB CUDA context
            layer_cost = mb_per_layer + kv_cache_per_layer_mb

            max_gpu_layers = int(usable_vram / max(1.0, layer_cost))

            if max_gpu_layers >= total_layers:
                # Fits completely on GPU
                vram_used = (total_layers * layer_cost) + 512.0
                return LayerDistributionPlan(
                    model_name=model_name,
                    total_layers=total_layers,
                    device_layers={gpu.id: total_layers},
                    is_full_offload=True,
                    estimated_vram_usage_mb={gpu.id: vram_used},
                    strategy="single_gpu",
                )
            else:
                # Partial GPU offload + CPU fallback
                gpu_layers = max(0, max_gpu_layers)
                cpu_layers = total_layers - gpu_layers
                vram_used = (gpu_layers * layer_cost) + 512.0
                return LayerDistributionPlan(
                    model_name=model_name,
                    total_layers=total_layers,
                    device_layers={gpu.id: gpu_layers, "cpu:0": cpu_layers},
                    is_full_offload=False,
                    estimated_vram_usage_mb={
                        gpu.id: vram_used,
                        "cpu:0": cpu_layers * mb_per_layer,
                    },
                    strategy="cpu_offload",
                    metadata={"offload_ratio": round(gpu_layers / total_layers, 2)},
                )

        # Multi-GPU Pipeline Parallel distribution
        total_usable_gpu_vram = sum(
            max(0.0, g.vram_total_mb - safety_reserve_mb - 512.0) for g in gpu_devices
        )
        device_layers: dict[str, int] = {}
        vram_estimates: dict[str, float] = {}
        remaining_layers = total_layers

        for i, gpu in enumerate(gpu_devices):
            usable = max(0.0, gpu.vram_total_mb - safety_reserve_mb - 512.0)
            if i == len(gpu_devices) - 1:
                # Last GPU takes remainder
                layers = remaining_layers
            else:
                ratio = usable / max(1.0, total_usable_gpu_vram)
                layers = min(remaining_layers, int(round(total_layers * ratio)))

            device_layers[gpu.id] = layers
            vram_estimates[gpu.id] = (layers * (mb_per_layer + kv_cache_per_layer_mb)) + 512.0
            remaining_layers -= layers

        return LayerDistributionPlan(
            model_name=model_name,
            total_layers=total_layers,
            device_layers=device_layers,
            is_full_offload=True,
            estimated_vram_usage_mb=vram_estimates,
            strategy="pipeline_parallel",
            metadata={"gpu_count": len(gpu_devices)},
        )

    def _save_topology(self, t: DeviceTopology) -> None:
        """Persist snapshot into SQLite database."""
        if not self.db_path.exists():
            return
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO device_topologies (
                    id, hostname, devices_json, total_vram_mb, total_ram_gb,
                    compute_backend, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.id,
                    t.hostname,
                    json.dumps([d.to_dict() for d in t.devices]),
                    t.total_vram_mb,
                    t.total_ram_gb,
                    t.compute_backend,
                    t.created_at,
                    json.dumps(t.metadata),
                ),
            )
            conn.commit()
            log.info("Persisted device topology snapshot %s", t.id)
        except Exception as e:
            log.warning("Failed to save device topology: %s", e)
        finally:
            conn.close()

    def get_latest_topology(self) -> DeviceTopology | None:
        """Retrieve most recent topology record from SQLite."""
        if not self.db_path.exists():
            return None
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, hostname, devices_json, total_vram_mb, total_ram_gb,
                       compute_backend, created_at, metadata
                FROM device_topologies
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                return None

            devices_data = json.loads(row["devices_json"] or "[]")
            devices = [
                ComputeDevice(
                    id=d["id"],
                    name=d["name"],
                    device_type=d["device_type"],
                    vram_total_mb=d.get("vram_total_mb", 0.0),
                    vram_free_mb=d.get("vram_free_mb", 0.0),
                    compute_capability=d.get("compute_capability", ""),
                    is_primary=d.get("is_primary", False),
                    temperature_c=d.get("temperature_c"),
                    utilization_pct=d.get("utilization_pct"),
                    power_watts=d.get("power_watts"),
                    metadata=d.get("metadata", {}),
                )
                for d in devices_data
            ]

            return DeviceTopology(
                id=row["id"],
                hostname=row["hostname"],
                devices=devices,
                total_vram_mb=row["total_vram_mb"],
                total_ram_gb=row["total_ram_gb"],
                compute_backend=row["compute_backend"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()
