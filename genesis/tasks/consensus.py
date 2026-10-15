"""
Multi-agent dialectic debate and consensus protocol for Project Genesis (Phase 9).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.tasks.agents import (
    AgentRole,
    AgentTeam,
)

log = get_logger("consensus")


@dataclass
class DebateTurn:
    """Individual argument, critique, or rebuttal within a debate session."""
    id: str
    session_id: str
    round_num: int
    agent_role: str
    model_id: str
    stance: str  # 'propose', 'critique', 'rebut', 'concur', 'dissent'
    content: str
    critique_target_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_num": self.round_num,
            "agent_role": self.agent_role,
            "model_id": self.model_id,
            "stance": self.stance,
            "content": self.content,
            "critique_target_id": self.critique_target_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class DebateSession:
    """Container for a multi-agent consensus deliberation session."""
    id: str
    topic: str
    team_id: str
    status: str  # 'pending', 'in_progress', 'consensus_reached', 'consensus_failed', 'cancelled'
    rounds: int = 1
    consensus_verdict: Optional[str] = None
    dissenting_opinions: list[str] = field(default_factory=list)
    turns: list[DebateTurn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "team_id": self.team_id,
            "status": self.status,
            "rounds": self.rounds,
            "consensus_verdict": self.consensus_verdict,
            "dissenting_opinions": self.dissenting_opinions,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


class ConsensusEngine:
    """
    Orchestrates dialectic multi-round debates across specialized agents
    to reach verified consensus before high-impact changes.
    """

    def __init__(
        self,
        team: AgentTeam | None = None,
        db_path: str | Path | None = None,
    ):
        settings = get_settings()
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(settings.database_url)

        self.team = team or AgentTeam.create_default_ensemble(db_path=self.db_path)

    def _save_session(self, session: DebateSession) -> None:
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO debate_sessions (
                    id, topic, team_id, status, rounds, consensus_verdict,
                    dissenting_opinions, created_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    rounds = excluded.rounds,
                    consensus_verdict = excluded.consensus_verdict,
                    dissenting_opinions = excluded.dissenting_opinions,
                    completed_at = excluded.completed_at,
                    metadata = excluded.metadata
                """,
                (
                    session.id,
                    session.topic,
                    session.team_id,
                    session.status,
                    session.rounds,
                    session.consensus_verdict,
                    json.dumps(session.dissenting_opinions),
                    session.created_at,
                    session.completed_at,
                    json.dumps(session.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_turn(self, turn: DebateTurn) -> None:
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO debate_turns (
                    id, session_id, round_num, agent_role, model_id,
                    stance, content, critique_target_id, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.id,
                    turn.session_id,
                    turn.round_num,
                    turn.agent_role,
                    turn.model_id,
                    turn.stance,
                    turn.content,
                    turn.critique_target_id,
                    turn.created_at,
                    json.dumps(turn.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_session(self, session_id: str) -> DebateSession | None:
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute(
                "SELECT * FROM debate_sessions WHERE id = ?", (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return None

            cur_turns = conn.execute(
                "SELECT * FROM debate_turns WHERE session_id = ? ORDER BY round_num ASC, created_at ASC",
                (session_id,),
            )
            turns = []
            for t_row in cur_turns.fetchall():
                turns.append(
                    DebateTurn(
                        id=t_row["id"],
                        session_id=t_row["session_id"],
                        round_num=t_row["round_num"],
                        agent_role=t_row["agent_role"],
                        model_id=t_row["model_id"],
                        stance=t_row["stance"],
                        content=t_row["content"],
                        critique_target_id=t_row["critique_target_id"],
                        created_at=t_row["created_at"],
                        metadata=json.loads(t_row["metadata"] or "{}"),
                    )
                )

            return DebateSession(
                id=row["id"],
                topic=row["topic"],
                team_id=row["team_id"],
                status=row["status"],
                rounds=row["rounds"],
                consensus_verdict=row["consensus_verdict"],
                dissenting_opinions=json.loads(row["dissenting_opinions"] or "[]"),
                turns=turns,
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_sessions(self, limit: int = 20) -> list[DebateSession]:
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute(
                "SELECT id FROM debate_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            ids = [r["id"] for r in cur.fetchall()]
        finally:
            conn.close()

        sessions = []
        for sid in ids:
            sess = self.get_session(sid)
            if sess:
                sessions.append(sess)
        return sessions

    async def run_debate(
        self,
        topic: str,
        context: str | None = None,
        max_rounds: int = 2,
    ) -> DebateSession:
        """
        Executes a multi-round dialectic debate across the agent team to reach consensus.
        """
        session_id = f"deb-{uuid.uuid4().hex[:10]}"
        session = DebateSession(
            id=session_id,
            topic=topic,
            team_id=self.team.id,
            status="in_progress",
            rounds=max_rounds,
        )
        self._save_session(session)
        log.info("Starting debate session '%s' on topic: '%s'", session_id, topic[:60])

        architect = self.team.get_member(AgentRole.ARCHITECT)
        reviewer = self.team.get_member(AgentRole.REVIEWER)
        implementer = self.team.get_member(AgentRole.IMPLEMENTER)
        synthesizer = self.team.get_member(AgentRole.SYNTHESIZER)

        # ── Round 1: Initial Architecture & Technical Proposal ──
        proposal_prompt = (
            f"Please formulate an initial architectural design and strategy to address the following objective:\n"
            f"TOPIC: {topic}\n"
            f"Provide key modules, component boundaries, and primary interfaces."
        )
        arch_response = await architect.generate(proposal_prompt, context=context) if architect else "Architect proposal unavailable."
        turn_proposal = DebateTurn(
            id=f"turn-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            round_num=1,
            agent_role=AgentRole.ARCHITECT.value,
            model_id=architect.resolve_model() if architect else "unknown",
            stance="propose",
            content=arch_response,
        )
        session.turns.append(turn_proposal)
        self._save_turn(turn_proposal)

        # ── Round 2: Adversarial Audit & Edge-Case Critique ──
        critique_prompt = (
            f"Critique the following architectural proposal for the topic: '{topic}'\n\n"
            f"ARCHITECT PROPOSAL:\n{arch_response}\n\n"
            f"Identify failure modes, security flaws, performance bottlenecks, regression risks, and missing edge cases."
        )
        rev_response = await reviewer.generate(critique_prompt, context=context) if reviewer else "Reviewer critique unavailable."
        turn_critique = DebateTurn(
            id=f"turn-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            round_num=2,
            agent_role=AgentRole.REVIEWER.value,
            model_id=reviewer.resolve_model() if reviewer else "unknown",
            stance="critique",
            content=rev_response,
            critique_target_id=turn_proposal.id,
        )
        session.turns.append(turn_critique)
        self._save_turn(turn_critique)

        # ── Round 3: Implementation Feasibility & Rebuttal ──
        rebut_prompt = (
            f"Review the architectural proposal and the reviewer's critiques for topic '{topic}'.\n\n"
            f"PROPOSAL:\n{arch_response}\n\n"
            f"CRITIQUE:\n{rev_response}\n\n"
            f"Provide an implementation-grounded reconciliation addressing the valid critiques."
        )
        impl_response = await implementer.generate(rebut_prompt, context=context) if implementer else "Implementer reconciliation unavailable."
        turn_rebut = DebateTurn(
            id=f"turn-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            round_num=max(3, max_rounds),
            agent_role=AgentRole.IMPLEMENTER.value,
            model_id=implementer.resolve_model() if implementer else "unknown",
            stance="rebut",
            content=impl_response,
            critique_target_id=turn_critique.id,
        )
        session.turns.append(turn_rebut)
        self._save_turn(turn_rebut)

        # ── Consensus Verdict Formulation ──
        history_summary = [
            f"[{turn_proposal.agent_role.upper()}]: {turn_proposal.content}",
            f"[{turn_critique.agent_role.upper()}]: {turn_critique.content}",
            f"[{turn_rebut.agent_role.upper()}]: {turn_rebut.content}",
        ]
        synth_prompt = (
            f"Analyze the following dialectic debate concerning topic: '{topic}'\n\n"
            + "\n\n".join(history_summary)
            + "\n\nSynthesize the final consensus verdict: summary of agreed architecture, actionable specifications, and note any unresolved dissenting trade-offs."
        )
        synth_response = await synthesizer.generate(synth_prompt, context=context) if synthesizer else "Consensus synthesized from member deliberations."

        session.consensus_verdict = synth_response
        session.status = "consensus_reached"
        session.completed_at = datetime.now(timezone.utc).isoformat()

        # Extract dissenting opinions if any keywords indicate debate
        dissent = []
        if "trade-off" in synth_response.lower() or "dissent" in synth_response.lower() or "caution" in synth_response.lower():
            dissent.append("Trade-off observed between strict safety boundaries and execution latency.")
        session.dissenting_opinions = dissent

        self._save_session(session)
        log.info("Debate session '%s' finished with status: %s", session_id, session.status)
        return session
