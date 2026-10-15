"""
Unit tests for long-horizon autonomous research campaigns (Phase 9).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.database import initialize_database
from genesis.memory import MemoryStore
from genesis.research.campaigns import CampaignManager, ResearchCampaign, ResearchMilestone
from genesis.research.coding import CodeExperimentRecord
from genesis.tasks.agents import AgentTeam
from genesis.tasks.consensus import DebateSession, DebateTurn


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "campaign_test.db"
    initialize_database(db_file)
    return db_file


@pytest.fixture
def mock_campaign_manager(test_db: Path) -> CampaignManager:
    manager = CampaignManager(db_path=test_db)

    # Mock consensus engine
    mock_debate = DebateSession(
        id="deb-mock-1",
        topic="Architecture topic",
        team_id="team-genesis-core",
        status="consensus_reached",
        rounds=2,
        consensus_verdict="Consensus architecture plan verified.",
    )
    manager.consensus_engine.run_debate = AsyncMock(return_value=mock_debate)

    # Mock code synthesizer
    mock_exp = CodeExperimentRecord(
        id="exp-mock-1",
        name="Prototype Exp",
        objective="Implement algorithm",
        status="verified",
        test_results={"passed": True},
    )
    manager.code_synthesizer.synthesize_with_repair = AsyncMock(return_value=mock_exp)

    return manager


class TestCampaignManager:
    """Tests for long-horizon research campaign orchestration."""

    def test_create_and_load_campaign(self, test_db: Path):
        manager = CampaignManager(db_path=test_db)
        campaign = manager.create_campaign(
            title="KV-Cache Compaction Strategy",
            objective="Evaluate and prototype dynamic token eviction under 8GB VRAM boundaries",
            hypotheses=["Evicting attention sinks above threshold retains >= 95% perplexity."],
        )

        assert campaign is not None
        assert campaign.id.startswith("camp-")
        assert campaign.title == "KV-Cache Compaction Strategy"
        assert campaign.status == "active"
        assert campaign.current_milestone == 0
        assert len(campaign.milestones) == 4

        # Verify load from DB
        loaded = manager.get_campaign(campaign.id)
        assert loaded is not None
        assert loaded.title == campaign.title
        assert loaded.current_milestone == 0
        assert len(loaded.milestones) == 4

        # List campaigns
        all_campaigns = manager.list_campaigns()
        assert any(c.id == campaign.id for c in all_campaigns)

    @pytest.mark.asyncio
    async def test_full_campaign_milestone_progression(self, mock_campaign_manager: CampaignManager, test_db: Path):
        campaign = mock_campaign_manager.create_campaign(
            title="Autonomous AST Optimization Pass",
            objective="Design and verify an automated AST visitor for memory reduction",
        )
        assert campaign.current_milestone == 0

        # Advance Milestone 1: Grounding
        c1 = await mock_campaign_manager.advance_campaign(campaign.id)
        assert c1.current_milestone == 1
        assert c1.milestones[0].status == "completed"
        assert len(c1.findings) == 1

        # Advance Milestone 2: Consensus Architecture
        c2 = await mock_campaign_manager.advance_campaign(campaign.id)
        assert c2.current_milestone == 2
        assert c2.milestones[1].status == "completed"
        assert len(c2.findings) == 2

        # Advance Milestone 3: Code Synthesis
        c3 = await mock_campaign_manager.advance_campaign(campaign.id)
        assert c3.current_milestone == 3
        assert c3.milestones[2].status == "completed"
        assert len(c3.findings) == 3

        # Advance Milestone 4: Findings Consolidation
        c4 = await mock_campaign_manager.advance_campaign(campaign.id)
        assert c4.current_milestone == 4
        assert c4.status == "completed"
        assert c4.completed_at is not None

        # Verify memory record was written
        mem_store = MemoryStore(db_path=test_db)
        memories = mem_store.search(category="research_campaign")
        assert len(memories) >= 1
        assert "RESEARCH CAMPAIGN REPORT" in memories[0].content
