"""
Integration tests for Knowledge Graph, Multimodal Vision, and Autonomous Research Daemon API routes (Phase 10).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_graph_test.db"
    initialize_database(db_file)

    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestGraphAPI:
    """Test /api/graph/* routes."""

    def test_node_and_edge_crud_flow(self, client: TestClient):
        # 1. Create nodes
        n1 = client.post("/api/graph/nodes", json={
            "id": "node-model-router",
            "node_type": "component",
            "label": "Dynamic Model Router",
            "properties": {"routes": 5},
        })
        assert n1.status_code == 200
        assert n1.json()["id"] == "node-model-router"

        n2 = client.post("/api/graph/nodes", json={
            "id": "node-lora-tuning",
            "node_type": "experiment",
            "label": "LoRA Parameter Tuning",
            "properties": {"rank": 16},
        })
        assert n2.status_code == 200

        # 2. List nodes
        nodes_res = client.get("/api/graph/nodes")
        assert nodes_res.status_code == 200
        ids = [n["id"] for n in nodes_res.json()]
        assert "node-model-router" in ids
        assert "node-lora-tuning" in ids

        # 3. Create edge
        e1 = client.post("/api/graph/edges", json={
            "source_id": "node-model-router",
            "target_id": "node-lora-tuning",
            "relation_type": "relates_to",
            "weight": 1.25,
        })
        assert e1.status_code == 200
        assert e1.json()["source"] == "node-model-router"

        # 4. Neighbors
        neighbors = client.get("/api/graph/neighbors/node-model-router?direction=out")
        assert neighbors.status_code == 200
        assert len(neighbors.json()) == 1
        assert neighbors.json()[0]["node"]["id"] == "node-lora-tuning"

        # 5. Path
        path_res = client.get("/api/graph/path?source_id=node-model-router&target_id=node-lora-tuning")
        assert path_res.status_code == 200
        assert path_res.json()["path"] == ["node-model-router", "node-lora-tuning"]

        # 6. Subgraph
        subgraph = client.get("/api/graph/subgraph?centers=node-model-router&depth=1")
        assert subgraph.status_code == 200
        assert len(subgraph.json()["nodes"]) == 2

        # 7. Export
        export = client.get("/api/graph/export")
        assert export.status_code == 200
        assert export.json()["stats"]["total_nodes"] >= 2
        assert export.json()["stats"]["total_edges"] >= 1


class TestMultimodalAPI:
    """Test /api/multimodal/* routes."""

    def test_inspect_svg_artifact(self, client: TestClient):
        svg_diagram = """
        <svg width="600" height="400">
            <text x="10" y="20">ModelRouter</text>
            <text x="10" y="80">SafeCodeSandbox</text>
        </svg>
        """

        res = client.post("/api/multimodal/inspect", json={
            "artifact_name": "system_design.svg",
            "artifact_type": "diagram",
            "image_data": svg_diagram,
            "expected_elements": ["ModelRouter", "SafeCodeSandbox"],
        })
        assert res.status_code == 200
        data = res.json()
        assert data["passed"] is True
        assert data["format"] == "svg"
        assert "ModelRouter" in data["elements_detected"]

        # Test list
        history = client.get("/api/multimodal/inspections")
        assert history.status_code == 200
        assert len(history.json()) >= 1
        assert history.json()[0]["artifact_name"] == "system_design.svg"


class TestDaemonPapersAPI:
    """Test /api/research/daemon/* routes."""

    @patch("genesis.research.daemon.ConsensusEngine.run_debate", new_callable=AsyncMock)
    @patch("genesis.research.daemon.CodeSandbox.execute")
    def test_daemon_cycle_and_papers(self, mock_sandbox_exec, mock_run_debate, client: TestClient):
        from genesis.research.coding import SandboxResult

        mock_run_debate.return_value = {
            "id": "deb-test-api",
            "topic": "Autonomous Vision Inspection",
            "status": "concluded",
            "rounds": 2,
            "consensus_verdict": "Consensus: Visual structural verification defends against corrupted UI renders.",
            "dissenting_views": [],
            "turns": [
                {"round_num": 1, "speaker_role": "researcher", "content": "Visual tests verify rendered layouts."},
            ],
        }

        mock_sandbox_exec.return_value = SandboxResult(
            passed=True,
            exit_code=0,
            stdout="Verification passed.\n",
            stderr="",
            traceback="",
            duration_ms=25.0,
        )

        res = client.post("/api/research/daemon/cycle", json={
            "topic": "Visual inspection for local AI agents",
            "context": "Local 8GB VRAM workstation",
            "rounds": 1,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["paper_id"].startswith("paper-")
        assert data["tests_passed"] is True

        # List papers
        papers_res = client.get("/api/research/daemon/papers")
        assert papers_res.status_code == 200
        assert len(papers_res.json()) >= 1
        paper_id = papers_res.json()[0]["id"]

        # Get single paper
        single = client.get(f"/api/research/daemon/papers/{paper_id}")
        assert single.status_code == 200
        assert "## 1. Introduction" in single.json()["content_markdown"]
