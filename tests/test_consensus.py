"""
Unit tests for dialectic debate and consensus engine (Phase 9).
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
    SynthesizerAgent,
)
from genesis.tasks.consensus import ConsensusEngine, DebateSession, DebateTurn


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "consensus_test.db"
    initialize_database(db_file)
    return db_file


@pytest.fixture
def mock_team(test_db: Path) -> AgentTeam:
    team = AgentTeam(id="team-mock", name="Mock Team", description="Mocked ensemble for consensus testing")

    def make_mock_agent(agent_cls, response_text):
        provider = MagicMock()
        provider.generate = AsyncMock(return_value=response_text)
        return agent_cls(db_path=test_db, provider=provider)

    team.add_member(make_mock_agent(ResearcherAgent, "Researcher: Found 3 relevant architecture benchmarks."))
    team.add_member(make_mock_agent(ArchitectAgent, "Architect: Proposed SQLite WAL mode with indexed query plan."))
    team.add_member(make_mock_agent(ImplementerAgent, "Implementer: Confirmed schema migration path and async context manager."))
    team.add_member(make_mock_agent(ReviewerAgent, "Reviewer: Identified concurrency lock contention risk under heavy writes."))
    team.add_member(make_mock_agent(SynthesizerAgent, "Synthesizer: Consensus achieved: Use WAL mode with connection pooling. Trade-off: Write contention."))
    return team


class TestConsensusEngine:
    """Tests for multi-agent dialectic consensus deliberation."""

    @pytest.mark.asyncio
    async def test_run_debate_full_cycle(self, test_db: Path, mock_team: AgentTeam):
        engine = ConsensusEngine(team=mock_team, db_path=test_db)
        topic = "Decide database connection pooling strategy for high-throughput task worker"

        session = await engine.run_debate(topic=topic, max_rounds=2)

        assert session is not None
        assert session.id.startswith("deb-")
        assert session.status == "consensus_reached"
        assert len(session.turns) >= 3
        assert session.consensus_verdict is not None
        assert "Consensus achieved" in session.consensus_verdict

        # Verify turn stances and roles
        stances = [t.stance for t in session.turns]
        assert "propose" in stances
        assert "critique" in stances
        assert "rebut" in stances

        # Verify critique target linking
        critique_turn = next(t for t in session.turns if t.stance == "critique")
        rebut_turn = next(t for t in session.turns if t.stance == "rebut")
        assert critique_turn.critique_target_id is not None
        assert rebut_turn.critique_target_id == critique_turn.id

    def test_get_and_list_sessions(self, test_db: Path, mock_team: AgentTeam):
        engine = ConsensusEngine(team=mock_team, db_path=test_db)

        # Manually save a session
        sess = DebateSession(
            id="deb-manual-1",
            topic="Evaluate vector index chunking size",
            team_id=mock_team.id,
            status="consensus_reached",
            rounds=2,
            consensus_verdict="Consensus: 512 token chunking with 64 token overlap.",
            dissenting_opinions=["Slightly increased storage footprint."],
            turns=[
                DebateTurn(
                    id="turn-1",
                    session_id="deb-manual-1",
                    round_num=1,
                    agent_role=AgentRole.ARCHITECT.value,
                    model_id="qwen2.5:7b",
                    stance="propose",
                    content="Propose 512 token chunk size.",
                )
            ],
        )
        engine._save_session(sess)
        engine._save_turn(sess.turns[0])

        loaded = engine.get_session("deb-manual-1")
        assert loaded is not None
        assert loaded.topic == "Evaluate vector index chunking size"
        assert len(loaded.turns) == 1
        assert loaded.turns[0].agent_role == "architect"
        assert loaded.dissenting_opinions == ["Slightly increased storage footprint."]

        sessions = engine.list_sessions()
        assert any(s.id == "deb-manual-1" for s in sessions)
