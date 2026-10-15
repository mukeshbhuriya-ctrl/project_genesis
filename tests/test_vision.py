"""
Tests for Multi-Modal Vision Processing and Visual Inspection Engine (Phase 10).
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

from genesis.core.database import initialize_database
from genesis.models.vision import VisionInspector, VisionRequest


@pytest.fixture
def initialized_inspector(tmp_db_path: Path) -> VisionInspector:
    """Provide an initialized VisionInspector with ready database."""
    initialize_database(tmp_db_path)
    return VisionInspector(db_path=tmp_db_path)


def _make_sample_png_bytes(width: int = 640, height: int = 480) -> bytes:
    """Generate a minimal valid 24-byte PNG header with width and height in IHDR."""
    header = b"\x89PNG\r\n\x1a\n"
    # Chunk: length (4 bytes), chunk type 'IHDR' (4 bytes), w (4 bytes), h (4 bytes), bit depth + color (5 bytes), CRC (4 bytes)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"
    return header + ihdr_chunk


class TestVisionInspector:
    """Test suite for VisionInspector."""

    @pytest.mark.asyncio
    async def test_vision_inspector_decodes_png_dimensions(self, initialized_inspector: VisionInspector):
        """Inspector should decode PNG binary headers and extract dimensions."""
        png_bytes = _make_sample_png_bytes(800, 600)

        req = VisionRequest(
            image=png_bytes,
            artifact_type="diagram",
            artifact_name="system_topology.png",
        )
        resp = await initialized_inspector.inspect_artifact(req)

        assert resp.format == "png"
        assert resp.dimensions == (800, 600)
        assert resp.passed is True
        assert resp.inspection_id.startswith("vis-")

    @pytest.mark.asyncio
    async def test_vision_inspector_svg_diagram_verification(self, initialized_inspector: VisionInspector):
        """Inspector should analyze SVG architecture diagrams and verify expected components."""
        svg_content = """
        <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
            <rect width="800" height="600" fill="#0f172a"/>
            <text x="50" y="50">ModelRouter</text>
            <text x="50" y="150">SafeCodeSandbox</text>
            <text x="50" y="250">ConsensusEngine</text>
        </svg>
        """

        resp = await initialized_inspector.verify_architecture_diagram(
            image=svg_content,
            required_components=["ModelRouter", "SafeCodeSandbox", "ConsensusEngine"],
            artifact_name="core_architecture.svg",
        )

        assert resp.format == "svg"
        assert resp.dimensions == (800, 600)
        assert resp.passed is True
        assert len(resp.missing_elements) == 0
        assert "ModelRouter" in resp.elements_detected
        assert "SafeCodeSandbox" in resp.elements_detected
        assert "ConsensusEngine" in resp.elements_detected

    @pytest.mark.asyncio
    async def test_vision_inspector_missing_element_detection(self, initialized_inspector: VisionInspector):
        """Inspector should fail verification when required components are missing."""
        svg_content = """
        <svg width="600" height="400">
            <text x="20" y="30">ModelRouter</text>
        </svg>
        """

        resp = await initialized_inspector.verify_architecture_diagram(
            image=svg_content,
            required_components=["ModelRouter", "SafeCodeSandbox", "QuantumOptimizer"],
            artifact_name="partial_diagram.svg",
        )

        assert resp.passed is False
        assert "QuantumOptimizer" in resp.missing_elements
        assert "SafeCodeSandbox" in resp.missing_elements
        assert "ModelRouter" in resp.elements_detected

    @pytest.mark.asyncio
    async def test_vision_inspector_ui_layout_verification(self, initialized_inspector: VisionInspector):
        """Inspector should verify UI layout screenshots from base64 data."""
        raw_png = _make_sample_png_bytes(1024, 768)
        b64_str = "data:image/png;base64," + base64.b64encode(raw_png).decode("ascii")

        resp = await initialized_inspector.verify_ui_layout(
            image=b64_str,
            required_elements=[],
            artifact_name="web_dashboard_preview.png",
        )

        assert resp.passed is True
        assert resp.format == "png"
        assert resp.dimensions == (1024, 768)

    @pytest.mark.asyncio
    async def test_vision_inspector_sqlite_persistence(self, initialized_inspector: VisionInspector):
        """Inspector should persist audit trail in SQLite and allow retrieval."""
        svg_content = '<svg width="400" height="300"><text x="10" y="20">TestCanvas</text></svg>'

        resp = await initialized_inspector.inspect_artifact(
            VisionRequest(
                image=svg_content,
                artifact_name="audit_test.svg",
                artifact_type="chart",
            )
        )

        # Retrieve list
        inspections = initialized_inspector.list_inspections(limit=10)
        assert len(inspections) >= 1
        assert inspections[0]["id"] == resp.inspection_id
        assert inspections[0]["artifact_type"] == "chart"

        # Retrieve single
        single = initialized_inspector.get_inspection(resp.inspection_id)
        assert single is not None
        assert single["artifact_name"] == "audit_test.svg"
        assert single["passed"] is True

