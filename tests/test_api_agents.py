"""
Integration tests for Multi-Agent, Debate Consensus, and Research Campaign API routes (Phase 9).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.tasks.consensus import DebateSession, DebateTurn


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_agents_test.db"
    initialize_database(db_file)

    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestAgentsAPI:
    """Test /api/agents/* routes."""

    def test_list_roles(self, client: TestClient):
        res = client.get("/api/agents/roles")
        assert res.status_code == 200
        roles = res.json()
        assert isinstance(roles, list)
        role_names = {r["role"] for r in roles}
        assert "researcher" in role_names
        assert "architect" in role_names
        assert "implementer" in role_names
        assert "reviewer" in role_names
        assert "synthesizer" in role_names

    def test_list_and_create_teams(self, client: TestClient):
        # 1. List seeded teams
        res = client.get("/api/agents/teams")
        assert res.status_code == 200
        teams = res.json()
        assert len(teams) >= 1
        assert any(t["id"] == "team-genesis-core" for t in teams)

        # 2. Create custom team
        payload = {
            "name": "Security Audit Swarm",
            "description": "Specialized team auditing code vulnerabilities",
            "roles": ["architect", "reviewer"],
            "metadata": {"specialty": "cybersecurity"},
        }
        create_res = client.post("/api/agents/teams", json=payload)
        assert create_res.status_code == 200
        created = create_res.json()
        assert created["name"] == "Security Audit Swarm"
        assert created["roles"] == ["architect", "reviewer"]

        # Verify in list
        list_res = client.get("/api/agents/teams")
        assert any(t["id"] == created["id"] for t in list_res.json())

    def test_debate_workflow(self, client: TestClient):
        mock_debate = DebateSession(
            id="deb-api-test-1",
            topic="Test consensus API topic",
            team_id="team-genesis-core",
            status="consensus_reached",
            rounds=2,
            consensus_verdict="API test consensus agreement.",
            dissenting_opinions=[],
            turns=[
                DebateTurn(
                    id="turn-api-1",
                    session_id="deb-api-test-1",
                    round_num=1,
                    agent_role="architect",
                    model_id="qwen2.5:7b",
                    stance="propose",
                    content="Proposal text",
                )
            ],
        )

        with patch("genesis.tasks.consensus.ConsensusEngine.run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_debate

            # 1. Trigger debate run
            res = client.post(
                "/api/agents/debate/run",
                json={"topic": "Test consensus API topic", "max_rounds": 2},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["id"] == "deb-api-test-1"
            assert data["status"] == "consensus_reached"

        # 2. Query debates list (after manual insert)
        from genesis.tasks.consensus import ConsensusEngine
        engine = ConsensusEngine()
        engine._save_session(mock_debate)
        engine._save_turn(mock_debate.turns[0])

        list_res = client.get("/api/agents/debates")
        assert list_res.status_code == 200
        assert any(d["id"] == "deb-api-test-1" for d in list_res.json())

        # 3. Get specific debate details
        detail_res = client.get("/api/agents/debates/deb-api-test-1")
        assert detail_res.status_code == 200
        detail = detail_res.json()
        assert detail["id"] == "deb-api-test-1"
        assert len(detail["turns"]) == 1

    def test_campaign_workflow(self, client: TestClient):
        # 1. Create research campaign
        payload = {
            "title": "Autonomous Cache Optimization",
            "objective": "Explore and verify dynamic page caching strategies",
            "hypotheses": ["Dynamic caching reduces latency by 20%."],
        }
        res = client.post("/api/agents/campaigns", json=payload)
        assert res.status_code == 200
        camp = res.json()
        assert camp["title"] == "Autonomous Cache Optimization"
        assert camp["status"] == "active"
        camp_id = camp["id"]

        # 2. List campaigns
        list_res = client.get("/api/agents/campaigns")
        assert list_res.status_code == 200
        assert any(c["id"] == camp_id for c in list_res.json())

        # 3. Get campaign detail
        get_res = client.get(f"/api/agents/campaigns/{camp_id}")
        assert get_res.status_code == 200
        detail = get_res.json()
        assert detail["id"] == camp_id
        assert detail["current_milestone"] == 0
        assert len(detail["milestones"]) == 4

        # 4. Advance milestone 1
        adv_res = client.post(f"/api/agents/campaigns/{camp_id}/advance")
        assert adv_res.status_code == 200
        adv_data = adv_res.json()
        assert adv_data["current_milestone"] == 1
        assert adv_data["milestones"][0]["status"] == "completed"
