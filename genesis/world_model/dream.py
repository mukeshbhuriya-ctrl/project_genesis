"""
Latent Dream & Continuous Imagination Rollout Engine (Phase 15).

Generatively simulates multi-step counterfactual problem-solving trajectories
in memory, estimating value functions, pruning unpromising branches, and
exporting high-reward synthetic traces into continual learning replay buffers.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection

log = logging.getLogger(__name__)


@dataclass
class DreamStep:
    """A discrete mental step inside a latent dream rollout."""

    step_index: int
    hypothesis_action: str
    imagined_outcome: str
    value_estimate: float  # -1.0 to 1.0 (Q/V estimate)
    uncertainty: float  # 0.0 to 1.0
    pruned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DreamStep:
        return cls(
            step_index=data.get("step_index", 0),
            hypothesis_action=data.get("hypothesis_action", ""),
            imagined_outcome=data.get("imagined_outcome", ""),
            value_estimate=float(data.get("value_estimate", 0.0)),
            uncertainty=float(data.get("uncertainty", 0.0)),
            pruned=bool(data.get("pruned", False)),
        )


@dataclass
class DreamTrajectory:
    """A full imagined problem-solving trajectory."""

    id: str
    prompt: str
    steps: list[DreamStep]
    total_value_score: float
    pruned_branches: int
    status: str  # 'imagining', 'completed', 'pruned', 'failed'
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "steps": [s.to_dict() for s in self.steps],
            "total_value_score": self.total_value_score,
            "pruned_branches": self.pruned_branches,
            "status": self.status,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class LatentDreamEngine:
    """
    Engine for simulating mental rollouts and latent counterfactual reasoning
    without physical GPU or filesystem execution overhead.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.db_path = get_db_path_from_url(get_settings().database_url)
        else:
            self.db_path = Path(db_path)

    @staticmethod
    def evaluate_step_value(action: str, outcome: str) -> tuple[float, float]:
        """
        Heuristic value estimator V(s) and uncertainty estimation.
        Returns (value_estimate, uncertainty).
        """
        # Baseline value
        val = 0.5
        uncertainty = 0.2

        lower_act = action.lower()
        lower_out = outcome.lower()

        # Positive indicators
        if any(w in lower_act for w in ["verify", "test", "validate", "decompose", "benchmark", "prove"]):
            val += 0.2
        if any(w in lower_out for w in ["verified", "success", "invariant held", "converged", "passed"]):
            val += 0.25
        if any(w in lower_out for w in ["edge case covered", "optimal", "safe"]):
            val += 0.1

        # Negative indicators
        if any(w in lower_act for w in ["blindly", "ignore error", "force overwrite", "bypass"]):
            val -= 0.5
            uncertainty += 0.3
        if any(w in lower_out for w in ["syntaxerror", "deadlock", "memory leak", "diverged", "failed"]):
            val -= 0.4
            uncertainty += 0.2
        if any(w in lower_out for w in ["hallucination", "unbounded", "inconclusive"]):
            val -= 0.3
            uncertainty += 0.3

        val = max(-1.0, min(1.0, round(val, 4)))
        uncertainty = max(0.0, min(1.0, round(uncertainty, 4)))
        return val, uncertainty

    def simulate_dream(
        self,
        prompt: str,
        seed_steps: list[str] | None = None,
        depth: int = 4,
        prune_threshold: float = 0.20,
        trajectory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DreamTrajectory:
        """
        Simulate a multi-turn latent rollout for a goal or problem prompt.
        """
        tid = trajectory_id or f"dream-{uuid.uuid4().hex[:10]}"
        steps: list[DreamStep] = []
        pruned_count = 0

        # Standard step plan templates if seeds not provided
        templates = [
            (
                "Deconstruct goal '{prompt}' into formal logical preconditions and dependencies",
                "Preconditions identified: isolated state, valid inputs, explicit boundary assertions",
            ),
            (
                "Synthesize candidate state transformation for '{prompt}'",
                "Transformation generated: pure functional mapping with invariant guardrails",
            ),
            (
                "Simulate boundary conditions and edge-case inputs",
                "Evaluated zero-length, extreme numerical scale, and malformed inputs: invariant held",
            ),
            (
                "Verify end-to-end outcome convergence and synthesize resolution",
                "All verification constraints passed; optimal trajectory confirmed",
            ),
            (
                "Formalize artifact specification and log learned heuristic transition",
                "Trajectory consolidated with high confidence; ready for replay buffer export",
            ),
        ]

        actual_depth = min(depth, len(templates))

        for idx in range(actual_depth):
            if seed_steps and idx < len(seed_steps):
                act = seed_steps[idx]
                out = f"Imagined execution result for: {act}"
            else:
                tmpl_act, tmpl_out = templates[idx]
                act = tmpl_act.format(prompt=prompt)
                out = tmpl_out.format(prompt=prompt)

            val, uncert = self.evaluate_step_value(act, out)
            is_pruned = val < prune_threshold

            step = DreamStep(
                step_index=idx + 1,
                hypothesis_action=act,
                imagined_outcome=out,
                value_estimate=val,
                uncertainty=uncert,
                pruned=is_pruned,
            )
            steps.append(step)

            if is_pruned:
                pruned_count += 1
                # Try counterfactual recovery step
                recovery_act = f"Counterfactual recovery: backtrack from pruned state in step {idx + 1}"
                recovery_out = "Alternative path explored: fallback invariant applied successfully"
                rec_val, rec_uncert = self.evaluate_step_value(recovery_act, recovery_out)
                steps.append(
                    DreamStep(
                        step_index=idx + 1,
                        hypothesis_action=recovery_act,
                        imagined_outcome=recovery_out,
                        value_estimate=rec_val,
                        uncertainty=rec_uncert,
                        pruned=False,
                    )
                )

        valid_vals = [s.value_estimate for s in steps if not s.pruned]
        total_val = round(sum(valid_vals) / max(1, len(valid_vals)), 4) if valid_vals else 0.0

        status = "completed" if total_val >= 0.50 else ("pruned" if pruned_count > 0 else "completed")

        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO dream_trajectories (
                    id, prompt, steps_count, total_value_score, pruned_branches, status, trajectory_json, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    prompt,
                    len(steps),
                    total_val,
                    pruned_count,
                    status,
                    json.dumps([s.to_dict() for s in steps]),
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

            cursor = conn.execute("SELECT created_at FROM dream_trajectories WHERE id = ?", (tid,))
            row = cursor.fetchone()
            created_at = row["created_at"] if row else ""

            return DreamTrajectory(
                id=tid,
                prompt=prompt,
                steps=steps,
                total_value_score=total_val,
                pruned_branches=pruned_count,
                status=status,
                created_at=created_at,
                metadata=metadata or {},
            )
        finally:
            conn.close()

    def export_to_replay_buffer(
        self, trajectory: DreamTrajectory, domain: str = "reasoning"
    ) -> int:
        """
        Export high-reward dream steps into continual learning replay buffer (Phase 13 integration).
        """
        if trajectory.total_value_score < 0.45:
            return 0

        conn = get_sync_connection(self.db_path)
        exported = 0
        try:
            for step in trajectory.steps:
                if step.pruned or step.value_estimate < 0.45:
                    continue
                sample_id = f"buf-dream-{uuid.uuid4().hex[:10]}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO replay_buffers (
                        id, task_domain, sample_input, sample_target, priority_score, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        domain,
                        f"Task: {trajectory.prompt}\nAction: {step.hypothesis_action}",
                        step.imagined_outcome,
                        step.value_estimate * 2.0,  # Scale priority
                        json.dumps({"source": "latent_dream", "trajectory_id": trajectory.id}),
                    ),
                )
                exported += 1
            conn.commit()
            return exported
        finally:
            conn.close()

    def get_trajectory(self, trajectory_id: str) -> DreamTrajectory | None:
        """Retrieve a dream trajectory by ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, prompt, steps_count, total_value_score, pruned_branches,
                       status, trajectory_json, created_at, metadata
                FROM dream_trajectories WHERE id = ?
                """,
                (trajectory_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            steps_data = json.loads(row["trajectory_json"] or "[]")
            steps = [DreamStep.from_dict(d) for d in steps_data]
            return DreamTrajectory(
                id=row["id"],
                prompt=row["prompt"],
                steps=steps,
                total_value_score=float(row["total_value_score"]),
                pruned_branches=int(row["pruned_branches"]),
                status=row["status"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_trajectories(
        self, status: str | None = None, limit: int = 50
    ) -> list[DreamTrajectory]:
        """List dream trajectories ordered by value or recency."""
        conn = get_sync_connection(self.db_path)
        try:
            if status:
                cursor = conn.execute(
                    """
                    SELECT id, prompt, steps_count, total_value_score, pruned_branches,
                           status, trajectory_json, created_at, metadata
                    FROM dream_trajectories WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, prompt, steps_count, total_value_score, pruned_branches,
                           status, trajectory_json, created_at, metadata
                    FROM dream_trajectories
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (limit,),
                )
            trajectories = []
            for row in cursor.fetchall():
                steps_data = json.loads(row["trajectory_json"] or "[]")
                steps = [DreamStep.from_dict(d) for d in steps_data]
                trajectories.append(
                    DreamTrajectory(
                        id=row["id"],
                        prompt=row["prompt"],
                        steps=steps,
                        total_value_score=float(row["total_value_score"]),
                        pruned_branches=int(row["pruned_branches"]),
                        status=row["status"],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return trajectories
        finally:
            conn.close()
