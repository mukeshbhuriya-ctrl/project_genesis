"""
Unit tests for role-specialized autonomous agents and teams (Phase 9).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.database import initialize_database
from genesis.tasks.agents import (
    AgentRole,
    AgentTeam,
    ArchitectAgent,
    ImplementerAgent,
    ResearcherAgent,
    ReviewerAgent,
    SpecializedAgent,
    SynthesizerAgent,
)
from genesis.tasks.agents.models import ARCHITECT_PERSONA


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "agents_test.db"
    initialize_database(db_file)
    return db_file


class TestSpecializedAgents:
    """Tests for individual agent specialization and role behaviors."""

    def test_agent_role_enumeration(self):
        assert AgentRole.RESEARCHER == "researcher"
        assert AgentRole.ARCHITECT == "architect"
        assert AgentRole.IMPLEMENTER == "implementer"
        assert AgentRole.REVIEWER == "reviewer"
        assert AgentRole.SYNTHESIZER == "synthesizer"

    def test_specialized_agent_model_resolution(self, test_db: Path):
        agent = SpecializedAgent(persona=ARCHITECT_PERSONA, db_path=test_db)
        resolved = agent.resolve_model()
        assert resolved in ("qwen2.5:7b", "qwen2.5-coder:7b")

        # Explicit override takes priority
        agent_custom = SpecializedAgent(persona=ARCHITECT_PERSONA, model_id="custom-model:latest", db_path=test_db)
        assert agent_custom.resolve_model() == "custom-model:latest"

    def test_role_subclasses_instantiation(self, test_db: Path):
        researcher = ResearcherAgent(db_path=test_db)
        architect = ArchitectAgent(db_path=test_db)
        implementer = ImplementerAgent(db_path=test_db)
        reviewer = ReviewerAgent(db_path=test_db)
        synthesizer = SynthesizerAgent(db_path=test_db)

        assert researcher.role == AgentRole.RESEARCHER
        assert architect.role == AgentRole.ARCHITECT
        assert implementer.role == AgentRole.IMPLEMENTER
        assert reviewer.role == AgentRole.REVIEWER
        assert synthesizer.role == AgentRole.SYNTHESIZER

        assert researcher.persona.default_task_category == "reasoning"
        assert implementer.persona.default_task_category == "coding"
        assert reviewer.persona.default_task_category == "coding"

    @pytest.mark.asyncio
    async def test_agent_generation_mock(self, test_db: Path):
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Synthesized architecture plan with 3 modules.")

        agent = ArchitectAgent(db_path=test_db, provider=mock_provider)
        response = await agent.generate("Design the state persistence layer.")
        assert "Synthesized architecture plan" in response
        assert mock_provider.generate.called

    def test_agent_to_dict(self, test_db: Path):
        agent = ImplementerAgent(db_path=test_db)
        data = agent.to_dict()
        assert data["role"] == "implementer"
        assert data["name"] == "Implementer"
        assert "resolved_model" in data
        assert "persona" in data


class TestAgentTeam:
    """Tests for agent ensemble management."""

    def test_team_creation_and_lookup(self, test_db: Path):
        team = AgentTeam(id="team-test", name="Test Team", description="Testing team ensemble")
        architect = ArchitectAgent(db_path=test_db)
        reviewer = ReviewerAgent(db_path=test_db)

        team.add_member(architect)
        team.add_member(reviewer)

        assert team.get_member(AgentRole.ARCHITECT) is architect
        assert team.get_member("reviewer") is reviewer
        assert team.get_member(AgentRole.RESEARCHER) is None

        members = team.list_members()
        assert len(members) == 2

    def test_create_default_ensemble(self, test_db: Path):
        team = AgentTeam.create_default_ensemble(db_path=test_db)
        assert team.id == "team-genesis-core"
        assert len(team.members) == 5
        assert team.get_member(AgentRole.RESEARCHER) is not None
        assert team.get_member(AgentRole.ARCHITECT) is not None
        assert team.get_member(AgentRole.IMPLEMENTER) is not None
        assert team.get_member(AgentRole.REVIEWER) is not None
        assert team.get_member(AgentRole.SYNTHESIZER) is not None

        d = team.to_dict()
        assert d["id"] == "team-genesis-core"
        assert len(d["members"]) == 5
