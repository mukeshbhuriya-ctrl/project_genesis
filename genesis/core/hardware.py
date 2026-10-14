"""
Hardware discovery and capability reporting.

Detects CPU, RAM, GPU, disk, and runtime information.
All detection is best-effort — missing or inaccessible hardware
is reported gracefully, never raises.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genesis.core.logging import get_logger

log = get_logger("hardware")


@dataclass
class GpuInfo:
    """Information about a detected NVIDIA GPU."""
    name: str = "Unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    driver_version: str = "Unknown"
    cuda_version: str = "Unknown"


@dataclass
class DiskInfo:
    """Disk space for a particular path."""
    path: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0


@dataclass
class HardwareReport:
    """Complete hardware capability report."""
    # System
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    hostname: str = ""

    # CPU
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0

    # Memory
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpus: list[GpuInfo] = field(default_factory=list)
    gpu_available: bool = False

    # Storage
    disks: list[DiskInfo] = field(default_factory=list)

    # Runtime
    python_version: str = ""
    pip_version: str = ""
    git_version: str = ""
    node_version: str = ""
    ollama_available: bool = False
    ollama_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "═══ Hardware Report ═══",
            f"OS:      {self.os_name} {self.os_version} ({self.architecture})",
            f"CPU:     {self.cpu_name} ({self.cpu_cores_physical}C/{self.cpu_cores_logical}T)",
            f"RAM:     {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB available",
        ]
        if self.gpus:
            for i, gpu in enumerate(self.gpus):
                lines.append(
                    f"GPU {i}:   {gpu.name} — {gpu.vram_total_mb} MB VRAM "
                    f"({gpu.vram_free_mb} MB free), Driver {gpu.driver_version}"
                )
        else:
            lines.append("GPU:     None detected")

        for disk in self.disks:
            lines.append(f"Disk:    {disk.path} — {disk.free_gb:.1f} / {disk.total_gb:.1f} GB free")

        lines.append(f"Python:  {self.python_version}")
        lines.append(f"Ollama:  {'Yes (' + self.ollama_version + ')' if self.ollama_available else 'Not installed'}")
        return "\n".join(lines)


def _run_cmd(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def detect_gpu() -> list[GpuInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    output = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []

    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(GpuInfo(
                name=parts[0],
                vram_total_mb=int(float(parts[1])),
                vram_free_mb=int(float(parts[2])),
                driver_version=parts[3],
            ))
    return gpus


def detect_disk(paths: list[str | Path] | None = None) -> list[DiskInfo]:
    """Detect disk space for given paths (defaults to CWD)."""
    if paths is None:
        paths = [Path.cwd()]
    disks = []
    for p in paths:
        try:
            usage = shutil.disk_usage(str(p))
            disks.append(DiskInfo(
                path=str(p),
                total_gb=usage.total / (1024 ** 3),
                free_gb=usage.free / (1024 ** 3),
                used_gb=usage.used / (1024 ** 3),
            ))
        except OSError:
            pass
    return disks


def detect_ram() -> tuple[float, float]:
    """Detect total and available RAM in GB. Returns (total, available)."""
    try:
        if sys.platform == "win32":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            total = mem.ullTotalPhys / (1024 ** 3)
            avail = mem.ullAvailPhys / (1024 ** 3)
            return total, avail
    except Exception:
        pass
    return 0.0, 0.0


def detect_cpu() -> tuple[str, int, int]:
    """Detect CPU name, physical cores, and logical cores."""
    name = platform.processor() or "Unknown"

    # Try to get a better name on Windows
    if sys.platform == "win32":
        wmic_name = _run_cmd(["wmic", "cpu", "get", "name", "/value"])
        if wmic_name:
            for line in wmic_name.splitlines():
                if line.startswith("Name="):
                    name = line.split("=", 1)[1].strip()
                    break

    logical = os.cpu_count() or 0
    # Physical cores: best-effort via OS
    physical = logical
    if sys.platform == "win32":
        wmic_cores = _run_cmd(["wmic", "cpu", "get", "NumberOfCores", "/value"])
        if wmic_cores:
            for line in wmic_cores.splitlines():
                if line.startswith("NumberOfCores="):
                    try:
                        physical = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                    break

    return name, physical, logical


def build_report(data_dir: Path | None = None) -> HardwareReport:
    """Build a complete hardware capability report."""
    cpu_name, cpu_phys, cpu_logical = detect_cpu()
    ram_total, ram_avail = detect_ram()
    gpus = detect_gpu()

    disk_paths = [Path.cwd()]
    if data_dir and data_dir.exists():
        disk_paths.append(data_dir)

    report = HardwareReport(
        os_name=platform.system(),
        os_version=platform.version(),
        architecture=platform.machine(),
        hostname=platform.node(),
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_phys,
        cpu_cores_logical=cpu_logical,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpus=gpus,
        gpu_available=len(gpus) > 0,
        disks=detect_disk(disk_paths),
        python_version=platform.python_version(),
        pip_version=_run_cmd([sys.executable, "-m", "pip", "--version"]) or "Unknown",
        git_version=_run_cmd(["git", "--version"]) or "Not installed",
        node_version=_run_cmd(["node", "--version"]) or "Not installed",
    )

    # Ollama check
    ollama_ver = _run_cmd(["ollama", "--version"])
    if ollama_ver:
        report.ollama_available = True
        report.ollama_version = ollama_ver
    else:
        report.ollama_available = False

    return report
