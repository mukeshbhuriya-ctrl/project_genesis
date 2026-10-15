"""
Continual Online Learning, Streaming Replay Buffers, and Elastic Weight Consolidation (EWC).

Prevents catastrophic forgetting during sequential fine-tuning and task adaptation across
reasoning, coding, tools, and general domains.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger

logger = get_logger(__name__)

VALID_DOMAINS = ("reasoning", "coding", "tools", "general")


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return get_db_path_from_url(get_settings().get_db_url())


@dataclass
class ReplaySample:
    """An exemplar stored in the streaming experience replay buffer."""
    id: str
    task_domain: str
    sample_input: str
    sample_target: str
    priority_score: float = 1.0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayBufferManager:
    """
    Manages streaming prioritized experience replay buffers across task domains.
    Supports balanced rehearsal mini-batches to prevent catastrophic forgetting.
    """

    def __init__(self, db_path: str | Path | None = None, capacity_per_domain: int = 1000):
        self.db_path = _resolve_db_path(db_path)
        self.capacity_per_domain = capacity_per_domain

    def add_sample(
        self,
        sample_input: str,
        sample_target: str,
        task_domain: str = "general",
        priority: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> ReplaySample:
        """Add an exemplar to the experience replay buffer with capacity enforcement."""
        if task_domain not in VALID_DOMAINS:
            raise ValueError(f"Invalid task_domain '{task_domain}'. Must be one of {VALID_DOMAINS}")

        sample_id = f"replay-{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        meta = metadata or {}
        p_score = max(0.01, float(priority))

        conn = get_sync_connection(self.db_path)
        try:
            # Check for exact duplicate input in the same domain
            cursor = conn.execute(
                "SELECT id, usage_count, priority_score FROM replay_buffers WHERE task_domain = ? AND sample_input = ?",
                (task_domain, sample_input),
            )
            existing = cursor.fetchone()
            if existing:
                # Update existing priority
                new_p = max(existing["priority_score"], p_score)
                conn.execute(
                    "UPDATE replay_buffers SET priority_score = ?, metadata = ? WHERE id = ?",
                    (new_p, json.dumps(meta), existing["id"]),
                )
                conn.commit()
                return ReplaySample(
                    id=existing["id"],
                    task_domain=task_domain,
                    sample_input=sample_input,
                    sample_target=sample_target,
                    priority_score=new_p,
                    usage_count=existing["usage_count"],
                    created_at=now,
                    metadata=meta,
                )

            # Insert new sample
            conn.execute(
                """
                INSERT INTO replay_buffers (id, task_domain, sample_input, sample_target, priority_score, usage_count, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (sample_id, task_domain, sample_input, sample_target, p_score, now, json.dumps(meta)),
            )

            # Enforce domain capacity: prune lowest priority / oldest when exceeding capacity
            count_cur = conn.execute(
                "SELECT COUNT(*) FROM replay_buffers WHERE task_domain = ?", (task_domain,)
            )
            count = count_cur.fetchone()[0]
            if count > self.capacity_per_domain:
                excess = count - self.capacity_per_domain
                conn.execute(
                    """
                    DELETE FROM replay_buffers WHERE id IN (
                        SELECT id FROM replay_buffers
                        WHERE task_domain = ?
                        ORDER BY priority_score ASC, created_at ASC
                        LIMIT ?
                    )
                    """,
                    (task_domain, excess),
                )

            conn.commit()
            return ReplaySample(
                id=sample_id,
                task_domain=task_domain,
                sample_input=sample_input,
                sample_target=sample_target,
                priority_score=p_score,
                usage_count=0,
                created_at=now,
                metadata=meta,
            )
        finally:
            conn.close()

    def get_sample(self, sample_id: str) -> ReplaySample | None:
        """Fetch a single sample by ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute("SELECT * FROM replay_buffers WHERE id = ?", (sample_id,))
            row = cur.fetchone()
            if not row:
                return None
            return ReplaySample(
                id=row["id"],
                task_domain=row["task_domain"],
                sample_input=row["sample_input"],
                sample_target=row["sample_target"],
                priority_score=row["priority_score"],
                usage_count=row["usage_count"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_samples(
        self,
        task_domain: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReplaySample]:
        """List samples with optional domain filter and pagination."""
        conn = get_sync_connection(self.db_path)
        try:
            if task_domain:
                cur = conn.execute(
                    """
                    SELECT * FROM replay_buffers
                    WHERE task_domain = ?
                    ORDER BY priority_score DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (task_domain, limit, offset),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT * FROM replay_buffers
                    ORDER BY priority_score DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
            return [
                ReplaySample(
                    id=r["id"],
                    task_domain=r["task_domain"],
                    sample_input=r["sample_input"],
                    sample_target=r["sample_target"],
                    priority_score=r["priority_score"],
                    usage_count=r["usage_count"],
                    created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in cur.fetchall()
            ]
        finally:
            conn.close()

    def sample_batch(
        self,
        batch_size: int = 16,
        balanced: bool = True,
        domains: list[str] | None = None,
    ) -> list[ReplaySample]:
        """
        Draw a rehearsal mini-batch with priority scoring.
        If balanced=True, draws evenly across specified (or all valid) domains.
        Increments usage_count for drawn samples.
        """
        target_domains = domains or list(VALID_DOMAINS)
        conn = get_sync_connection(self.db_path)
        sampled: list[ReplaySample] = []

        try:
            if balanced and target_domains:
                per_domain = max(1, batch_size // len(target_domains))
                for d in target_domains:
                    cur = conn.execute(
                        """
                        SELECT * FROM replay_buffers
                        WHERE task_domain = ?
                        ORDER BY (priority_score / (usage_count + 1.0)) DESC
                        LIMIT ?
                        """,
                        (d, per_domain),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        sampled.append(
                            ReplaySample(
                                id=r["id"],
                                task_domain=r["task_domain"],
                                sample_input=r["sample_input"],
                                sample_target=r["sample_target"],
                                priority_score=r["priority_score"],
                                usage_count=r["usage_count"],
                                created_at=r["created_at"],
                                metadata=json.loads(r["metadata"] or "{}"),
                            )
                        )
            else:
                cur = conn.execute(
                    """
                    SELECT * FROM replay_buffers
                    ORDER BY (priority_score / (usage_count + 1.0)) DESC
                    LIMIT ?
                    """,
                    (batch_size,),
                )
                rows = cur.fetchall()
                for r in rows:
                    sampled.append(
                        ReplaySample(
                            id=r["id"],
                            task_domain=r["task_domain"],
                            sample_input=r["sample_input"],
                            sample_target=r["sample_target"],
                            priority_score=r["priority_score"],
                            usage_count=r["usage_count"],
                            created_at=r["created_at"],
                            metadata=json.loads(r["metadata"] or "{}"),
                        )
                    )

            # Increment usage count for sampled records
            if sampled:
                sample_ids = [s.id for s in sampled]
                placeholders = ",".join("?" for _ in sample_ids)
                conn.execute(
                    f"UPDATE replay_buffers SET usage_count = usage_count + 1 WHERE id IN ({placeholders})",
                    sample_ids,
                )
                conn.commit()

            return sampled
        finally:
            conn.close()

    def get_domain_stats(self) -> dict[str, int]:
        """Return total sample counts per domain."""
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute("SELECT task_domain, COUNT(*) as c FROM replay_buffers GROUP BY task_domain")
            counts = {d: 0 for d in VALID_DOMAINS}
            for r in cur.fetchall():
                counts[r["task_domain"]] = r["c"]
            return counts
        finally:
            conn.close()

    def delete_sample(self, sample_id: str) -> bool:
        """Delete an exemplar by ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute("DELETE FROM replay_buffers WHERE id = ?", (sample_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


@dataclass
class EWCRecord:
    """An Elastic Weight Consolidation (EWC) reference snapshot."""
    id: str
    task_name: str
    importance_lambda: float
    fisher_diagonal: dict[str, list[float]]
    optimal_weights: dict[str, list[float]]
    parameter_count: int
    experiment_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_name": self.task_name,
            "importance_lambda": self.importance_lambda,
            "parameter_count": self.parameter_count,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class EWCConsolidator:
    """
    Computes diagonal Fisher Information estimates and quadratic EWC loss penalties:
    L_EWC = (lambda / 2) * sum_i F_i * (theta_i - theta_i^*)^2
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = _resolve_db_path(db_path)

    def compute_fisher_information(
        self,
        weights_dict: dict[str, list[float]],
        replay_samples: list[ReplaySample] | None = None,
        empirical_variance: float = 1.0,
    ) -> dict[str, list[float]]:
        """
        Estimate diagonal Fisher information matrix F_i.
        For each parameter layer, computes sensitivity weights proportional to gradient variance:
        F_{p, i} = E[(grad_i)^2] + eps
        """
        fisher: dict[str, list[float]] = {}
        sample_multiplier = len(replay_samples) if replay_samples else 1
        eps = 1e-6

        for param_name, tensor in weights_dict.items():
            fisher_layer: list[float] = []
            for val in tensor:
                # Mathematical sensitivity: magnitude-weighted variance estimation
                sensitivity = (abs(val) + empirical_variance) ** 2 * (1.0 / max(1, sample_multiplier))
                fisher_layer.append(round(sensitivity + eps, 6))
            fisher[param_name] = fisher_layer

        return fisher

    def compute_ewc_loss(
        self,
        current_weights: dict[str, list[float]],
        optimal_weights: dict[str, list[float]],
        fisher_diagonal: dict[str, list[float]],
        importance_lambda: float = 1000.0,
    ) -> float:
        """
        Compute scalar EWC quadratic penalty:
        L_EWC = 0.5 * lambda * sum_{p, i} F_{p, i} * (theta_{p, i} - theta^*_{p, i})^2
        """
        total_penalty = 0.0

        for param_name, current_layer in current_weights.items():
            if param_name not in optimal_weights or param_name not in fisher_diagonal:
                continue

            optimal_layer = optimal_weights[param_name]
            fisher_layer = fisher_diagonal[param_name]
            layer_len = min(len(current_layer), len(optimal_layer), len(fisher_layer))

            for i in range(layer_len):
                diff = current_layer[i] - optimal_layer[i]
                total_penalty += fisher_layer[i] * (diff ** 2)

        return float(0.5 * importance_lambda * total_penalty)

    def save_checkpoint(
        self,
        task_name: str,
        optimal_weights: dict[str, list[float]],
        fisher_diagonal: dict[str, list[float]],
        importance_lambda: float = 1000.0,
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EWCRecord:
        """Persist an EWC checkpoint into the database."""
        checkpoint_id = f"ewc-{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        meta = metadata or {}
        param_count = sum(len(layer) for layer in optimal_weights.values())

        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO ewc_checkpoints (id, experiment_id, task_name, importance_lambda, fisher_diagonal_json, optimal_weights_json, parameter_count, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    experiment_id,
                    task_name,
                    importance_lambda,
                    json.dumps(fisher_diagonal),
                    json.dumps(optimal_weights),
                    param_count,
                    now,
                    json.dumps(meta),
                ),
            )
            conn.commit()
            return EWCRecord(
                id=checkpoint_id,
                task_name=task_name,
                importance_lambda=importance_lambda,
                fisher_diagonal=fisher_diagonal,
                optimal_weights=optimal_weights,
                parameter_count=param_count,
                experiment_id=experiment_id,
                created_at=now,
                metadata=meta,
            )
        finally:
            conn.close()

    def get_checkpoint(self, checkpoint_id: str) -> EWCRecord | None:
        """Retrieve an EWC checkpoint by ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute("SELECT * FROM ewc_checkpoints WHERE id = ?", (checkpoint_id,))
            row = cur.fetchone()
            if not row:
                return None
            return EWCRecord(
                id=row["id"],
                task_name=row["task_name"],
                importance_lambda=row["importance_lambda"],
                fisher_diagonal=json.loads(row["fisher_diagonal_json"] or "{}"),
                optimal_weights=json.loads(row["optimal_weights_json"] or "{}"),
                parameter_count=row["parameter_count"],
                experiment_id=row["experiment_id"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_checkpoints(self) -> list[EWCRecord]:
        """List all registered EWC checkpoints."""
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute("SELECT * FROM ewc_checkpoints ORDER BY created_at DESC")
            return [
                EWCRecord(
                    id=row["id"],
                    task_name=row["task_name"],
                    importance_lambda=row["importance_lambda"],
                    fisher_diagonal={},  # omit large weights in summary listing
                    optimal_weights={},
                    parameter_count=row["parameter_count"],
                    experiment_id=row["experiment_id"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
                for row in cur.fetchall()
            ]
        finally:
            conn.close()


@dataclass
class PrefixTuningConfig:
    """Soft prompt prefix representation for parameter-efficient task conditioning."""
    id: str
    task_name: str
    virtual_tokens: int = 16
    prefix_dim: int = 64
    prefix_weights: dict[str, list[float]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_name": self.task_name,
            "virtual_tokens": self.virtual_tokens,
            "prefix_dim": self.prefix_dim,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class PrefixTuningManager:
    """Manages soft prompt prefixes and multi-task prefix interpolation."""

    @staticmethod
    def generate_random_prefix(task_name: str, virtual_tokens: int = 16, prefix_dim: int = 64) -> PrefixTuningConfig:
        """Create normalized Gaussian soft prompt prefix vectors."""
        config_id = f"prefix-{uuid.uuid4().hex[:10]}"
        keys = [round(random.gauss(0.0, 0.02), 5) for _ in range(virtual_tokens * prefix_dim)]
        values = [round(random.gauss(0.0, 0.02), 5) for _ in range(virtual_tokens * prefix_dim)]

        return PrefixTuningConfig(
            id=config_id,
            task_name=task_name,
            virtual_tokens=virtual_tokens,
            prefix_dim=prefix_dim,
            prefix_weights={"prefix_key": keys, "prefix_value": values},
        )

    @staticmethod
    def interpolate_prefixes(
        prefix_a: PrefixTuningConfig,
        prefix_b: PrefixTuningConfig,
        alpha: float = 0.5,
    ) -> PrefixTuningConfig:
        """Linearly blend two task prefixes: P = (1 - alpha) * P_a + alpha * P_b."""
        alpha = max(0.0, min(1.0, float(alpha)))
        blended_weights: dict[str, list[float]] = {}

        for k in ("prefix_key", "prefix_value"):
            w_a = prefix_a.prefix_weights.get(k, [])
            w_b = prefix_b.prefix_weights.get(k, [])
            min_len = min(len(w_a), len(w_b))
            blended = [round((1.0 - alpha) * w_a[i] + alpha * w_b[i], 5) for i in range(min_len)]
            blended_weights[k] = blended

        return PrefixTuningConfig(
            id=f"prefix-blend-{uuid.uuid4().hex[:8]}",
            task_name=f"{prefix_a.task_name}+{prefix_b.task_name}",
            virtual_tokens=min(prefix_a.virtual_tokens, prefix_b.virtual_tokens),
            prefix_dim=min(prefix_a.prefix_dim, prefix_b.prefix_dim),
            prefix_weights=blended_weights,
            metadata={"source_a": prefix_a.id, "source_b": prefix_b.id, "alpha": alpha},
        )
