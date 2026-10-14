"""Tests for the hardware detection module."""

from __future__ import annotations

import pytest

from genesis.core.hardware import (
    HardwareReport,
    GpuInfo,
    DiskInfo,
    build_report,
    detect_disk,
    detect_ram,
    detect_cpu,
)


class TestHardwareDetection:
    """Test hardware discovery (best-effort, non-GPU tests)."""

    def test_build_report_completes(self):
        """build_report should return a valid report without crashing."""
        report = build_report()
        assert isinstance(report, HardwareReport)
        assert report.os_name  # Should always have an OS name
        assert report.python_version  # Should always have Python version

    def test_detect_ram(self):
        """detect_ram should return positive values on any OS."""
        total, avail = detect_ram()
        # On Windows this should work; on other OS might return 0
        import sys
        if sys.platform == "win32":
            assert total > 0
            assert avail > 0

    def test_detect_cpu(self):
        """detect_cpu should return core counts."""
        name, phys, logical = detect_cpu()
        assert logical > 0

    def test_detect_disk(self):
        """detect_disk should return disk info for CWD."""
        disks = detect_disk()
        assert len(disks) > 0
        assert disks[0].total_gb > 0
        assert disks[0].free_gb > 0

    def test_report_summary_is_string(self):
        """summary() should return a readable string."""
        report = build_report()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Hardware Report" in summary

    def test_report_to_dict(self):
        """to_dict should return a serializable dictionary."""
        report = build_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "cpu_name" in d
        assert "gpus" in d


class TestGpuInfo:
    """Test GPU info data class."""

    def test_default_values(self):
        gpu = GpuInfo()
        assert gpu.name == "Unknown"
        assert gpu.vram_total_mb == 0


class TestDiskInfo:
    """Test disk info data class."""

    def test_default_values(self):
        disk = DiskInfo()
        assert disk.total_gb == 0.0
