"""
Tests for Autonomous Background Research Daemon & Paper Synthesis (Phase 10).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from genesis.core.database import initialize_database
from genesis.memory import MemoryStore
from genesis.memory.graph import MetaMemoryGraph
from genesis.research.coding import CodeSandbox, SandboxResult
from genesis.research.daemon import AutonomousResearchDaemon
from genesis.tasks.consensus import ConsensusEngine


@pytest.fixture
def initialized_daemon(tmp_db_path: Path) -> AutonomousResearchDaemon:
    """Provide an initialized AutonomousResearchDaemon."""
    initialize_database(tmp_db_path)
    graph = MetaMemoryGraph(db_path=tmp_db_path)
    memory_store = MemoryStore(str(tmp_db_path))

    # Mock consensus engine to avoid slow multi-round LLM prompts in unit tests
    mock_consensus = ConsensusEngine(db_path=tmp_db_path)
    mock_consensus.run_debate = AsyncMock(return_value={
        "id": "deb-test-mock",
        "topic": "FlashAttention KV Cache Quantization",
        "status": "concluded",
        "rounds": 2,
        "consensus_verdict": "Consensus reached: Adopt per-channel asymmetric int8 quantization with outlier clipping.",
        "dissenting_views": ["Reviewer noted outlier clipping requires calibration thresholds."],
        "turns": [
            {"round_num": 1, "speaker_role": "researcher", "content": "Proposal: int8 quantization preserves perplexity."},
            {"round_num": 1, "speaker_role": "architect", "content": "Architect: Confirms memory layout compatibility with 8GB VRAM."},
            {"round_num": 2, "speaker_role": "reviewer", "content": "Reviewer: Need to verify outlier clipping overhead."},
            {"round_num": 2, "speaker_role": "synthesizer", "content": "Synthesizer: Consensus achieved on calibrated asymmetric scaling."},
        ],
    })

    # Mock sandbox to run synchronously and reliably
    mock_sandbox = CodeSandbox()
    mock_sandbox.run = AsyncMock(return_value=SandboxResult(
        passed=True,
        exit_code=0,
        stdout="Empirical validation result: True\nCompression factor: 1.41x\n",
        stderr="",
        traceback="",
        duration_ms=42.5,
    ))

    return AutonomousResearchDaemon(
        db_path=tmp_db_path,
        graph=graph,
        consensus_engine=mock_consensus,
        sandbox=mock_sandbox,
        memory_store=memory_store,
    )


class TestAutonomousResearchDaemon:
    """Test suite for AutonomousResearchDaemon."""

    @pytest.mark.asyncio
    async def test_run_autonomous_research_cycle(self, initialized_daemon: AutonomousResearchDaemon):
        """Daemon should orchestrate debate, sandbox execution, paper generation, and graph persistence."""
        res = await initialized_daemon.run_cycle(
            topic="Should Genesis use int8 KV cache quantization on 8GB VRAM?",
            context="Target: RTX 2000 Ada local GPU.",
            rounds=2,
        )

        assert res["paper_id"].startswith("paper-")
        assert "Autonomous Investigation:" in res["title"]
        assert "Consensus reached:" in res["consensus_verdict"]
        assert len(res["dissenting_views"]) == 1
        assert res["tests_passed"] is True

        # Verify paper is stored in SQLite
        papers = initialized_daemon.list_papers()
        assert len(papers) >= 1
        assert papers[0]["id"] == res["paper_id"]

        # Verify full paper retrieval with IMRaD markdown content
        paper = initialized_daemon.get_paper(res["paper_id"])
        assert paper is not None
        assert "## Abstract" in paper["content_markdown"]
        assert "## 1. Introduction & Research Problem" in paper["content_markdown"]
        assert "## 2. Multi-Agent Dialectic Debate" in paper["content_markdown"]
        assert "## 3. Empirical Sandbox Validation" in paper["content_markdown"]
        assert "## 4. Discussion & Implications" in paper["content_markdown"]
        assert "## 5. Conclusion" in paper["content_markdown"]

        # Verify memory entry was created
        memories = initialized_daemon.memory_store.search(category="research_paper")
        assert len(memories) >= 1
        assert res["paper_id"] in memories[0].source

        # Verify graph nodes were linked
        graph_nodes = initialized_daemon.graph.list_nodes()
        node_labels = [n.label for n in graph_nodes]
        assert any("Autonomous Investigation:" in label for label in node_labels)

    @pytest.mark.asyncio
    async def test_list_and_get_papers_missing(self, initialized_daemon: AutonomousResearchDaemon):
        """Nonexistent paper should return None."""
        assert initialized_daemon.get_paper("paper-nonexistent") is None
