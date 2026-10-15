"""
Interactive Human-in-the-Loop Steering Subsystem (Phase 12).

Enables human operators and supervisory systems to inspect, pause, resume,
critique, or steer ongoing autonomous research cycles, dialectic debates,
research campaigns, and kernel synthesis loops.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger

log = get_logger("research.steering")


@dataclass
class SteeringEvent:
    """An operator intervention or directive for an active autonomous process."""
    id: str
    target_type: str  # daemon_cycle, campaign, debate, kernel_synthesis
    target_id: str
    action: str  # pause, resume, critique, override_hypothesis, abort, inject_guidance
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "received"  # received, applied, rejected
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "action": self.action,
            "payload": self.payload,
            "status": self.status,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class SteeringController:
    """
    Manages operator steering directives and provides synchronization hooks
    for autonomous engines.
    """

    def __init__(self, db_path: str | Path | None = None):
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else get_db_path_from_url(settings.database_url)

    def _conn(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    def emit_directive(
        self,
        target_type: str,
        target_id: str,
        action: str,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SteeringEvent:
        """
        Record a new steering directive from an operator.
        """
        valid_targets = ("daemon_cycle", "campaign", "debate", "kernel_synthesis")
        if target_type not in valid_targets:
            raise ValueError(f"Invalid target_type '{target_type}'. Must be one of {valid_targets}")

        valid_actions = ("pause", "resume", "critique", "override_hypothesis", "abort", "inject_guidance")
        if action not in valid_actions:
            raise ValueError(f"Invalid action '{action}'. Must be one of {valid_actions}")

        event_id = f"steer-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        payload_data = payload or {}
        meta = metadata or {}

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO steering_events (
                    id, target_type, target_id, action, payload_json,
                    status, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    target_type,
                    target_id,
                    action,
                    json.dumps(payload_data),
                    "received",
                    now,
                    json.dumps(meta),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        log.info(f"Operator steering directive emitted: {event_id} ({action} on {target_type}:{target_id})")
        return SteeringEvent(
            id=event_id,
            target_type=target_type,
            target_id=target_id,
            action=action,
            payload=payload_data,
            status="received",
            created_at=now,
            metadata=meta,
        )

    def get_pending_directives(self, target_type: str, target_id: str) -> list[SteeringEvent]:
        """Fetch all unhandled directives for a specific autonomous process."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, target_type, target_id, action, payload_json, status, created_at, metadata
                FROM steering_events
                WHERE target_type = ? AND target_id = ? AND status = 'received'
                ORDER BY created_at ASC
                """,
                (target_type, target_id),
            )
            rows = cursor.fetchall()
            return [
                SteeringEvent(
                    id=r["id"],
                    target_type=r["target_type"],
                    target_id=r["target_id"],
                    action=r["action"],
                    payload=json.loads(r["payload_json"] or "{}"),
                    status=r["status"],
                    created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def mark_applied(self, event_id: str) -> None:
        """Mark a steering directive as applied by the autonomous engine."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE steering_events SET status = 'applied' WHERE id = ?",
                (event_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def list_events(self, limit: int = 50) -> list[SteeringEvent]:
        """List recent steering events."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, target_type, target_id, action, payload_json, status, created_at, metadata
                FROM steering_events
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                SteeringEvent(
                    id=r["id"],
                    target_type=r["target_type"],
                    target_id=r["target_id"],
                    action=r["action"],
                    payload=json.loads(r["payload_json"] or "{}"),
                    status=r["status"],
                    created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in rows
            ]
        finally:
            conn.close()
