"""
Persistent, crash-resilient work job queue for decoupled process execution.

Supports lease-based locking, worker heartbeats, automated recovery of orphaned jobs,
progress tracking, and priority scheduling backed by SQLite WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core.database import get_sync_connection
from genesis.core.logging import get_logger

log = get_logger("job_queue")


@dataclass
class WorkJob:
    """A background workload execution unit."""
    id: str
    job_type: str
    payload: dict[str, Any]
    status: str = "queued"
    priority: int = 10
    worker_id: str | None = None
    progress: float = 0.0
    max_retries: int = 3
    retries: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    leased_at: str | None = None
    completed_at: str | None = None
    heartbeat_at: str | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "payload": self.payload,
            "status": self.status,
            "priority": self.priority,
            "worker_id": self.worker_id,
            "progress": round(self.progress, 2),
            "max_retries": self.max_retries,
            "retries": self.retries,
            "created_at": self.created_at,
            "leased_at": self.leased_at,
            "completed_at": self.completed_at,
            "heartbeat_at": self.heartbeat_at,
            "error": self.error,
            "result": self.result,
            "metadata": self.metadata,
        }


class JobQueue:
    """SQLite-backed persistent queue managing job dispatch and lease lifecycles."""

    def __init__(self, db_path: str | Path = "data/genesis.db") -> None:
        self.db_path = Path(db_path)

    def submit(
        self,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 10,
        max_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> WorkJob:
        """Enqueue a new job for background processing."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        job = WorkJob(
            id=job_id,
            job_type=job_type,
            payload=payload,
            status="queued",
            priority=priority,
            max_retries=max_retries,
            created_at=now,
            metadata=metadata or {},
        )

        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO work_jobs (
                    id, job_type, payload, status, priority, worker_id,
                    progress, max_retries, retries, created_at, result, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.job_type,
                    json.dumps(job.payload),
                    job.status,
                    job.priority,
                    job.worker_id,
                    job.progress,
                    job.max_retries,
                    job.retries,
                    job.created_at,
                    json.dumps(job.result),
                    json.dumps(job.metadata),
                ),
            )
            conn.commit()
            log.info("Submitted work job %s (%s, priority=%d)", job_id, job_type, priority)
            return job
        finally:
            conn.close()

    def lease(
        self,
        worker_id: str,
        lease_timeout_seconds: float = 300.0,
    ) -> WorkJob | None:
        """
        Atomically lease the next available job for worker execution:
        1. Checks for orphaned jobs (leased/running with expired heartbeats) and reclaims them.
        2. Leases the highest-priority pending queued job.
        """
        conn = get_sync_connection(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()

            # 1. Crash recovery: reclaim orphaned jobs
            cursor = conn.execute(
                """
                SELECT id, heartbeat_at, retries, max_retries
                FROM work_jobs
                WHERE status IN ('leased', 'running')
                """
            )
            for row in cursor.fetchall():
                hb = row["heartbeat_at"]
                if hb:
                    try:
                        hb_dt = datetime.fromisoformat(hb)
                        elapsed = (datetime.now(timezone.utc) - hb_dt).total_seconds()
                        if elapsed > lease_timeout_seconds:
                            # Worker died or hung: retry or fail
                            if row["retries"] < row["max_retries"]:
                                log.warning(
                                    "Reclaiming orphaned job %s (no heartbeat for %.1fs, retrying)",
                                    row["id"], elapsed
                                )
                                conn.execute(
                                    """
                                    UPDATE work_jobs
                                    SET status = 'queued', worker_id = NULL, retries = retries + 1
                                    WHERE id = ?
                                    """,
                                    (row["id"],),
                                )
                            else:
                                log.error("Orphaned job %s exceeded max retries, marking failed", row["id"])
                                conn.execute(
                                    """
                                    UPDATE work_jobs
                                    SET status = 'failed', error = 'Heartbeat timeout: worker abandoned job', completed_at = ?
                                    WHERE id = ?
                                    """,
                                    (now, row["id"]),
                                )
                    except Exception as ex:
                        log.debug("Error parsing heartbeat: %s", ex)

            conn.commit()

            # 2. Claim next available job atomically
            cursor = conn.execute(
                """
                SELECT id FROM work_jobs
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                return None

            job_id = row["id"]
            conn.execute(
                """
                UPDATE work_jobs
                SET status = 'leased', worker_id = ?, leased_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (worker_id, now, now, job_id),
            )
            conn.commit()

            return self.get_job(job_id)
        finally:
            conn.close()

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        progress: float | None = None,
    ) -> bool:
        """Update job heartbeat timestamp and optional progress float (0.0 to 1.0)."""
        now = datetime.now(timezone.utc).isoformat()
        conn = get_sync_connection(self.db_path)
        try:
            if progress is not None:
                cursor = conn.execute(
                    """
                    UPDATE work_jobs
                    SET status = 'running', heartbeat_at = ?, progress = ?
                    WHERE id = ? AND worker_id = ? AND status IN ('leased', 'running')
                    """,
                    (now, min(1.0, max(0.0, progress)), job_id, worker_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE work_jobs
                    SET status = 'running', heartbeat_at = ?
                    WHERE id = ? AND worker_id = ? AND status IN ('leased', 'running')
                    """,
                    (now, job_id, worker_id),
                )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark job as successfully finished."""
        now = datetime.now(timezone.utc).isoformat()
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                UPDATE work_jobs
                SET status = 'completed', progress = 1.0, completed_at = ?, result = ?
                WHERE id = ? AND worker_id = ?
                """,
                (now, json.dumps(result or {}), job_id, worker_id),
            )
            conn.commit()
            log.info("Job %s marked completed by worker %s", job_id, worker_id)
            return cursor.rowcount > 0
        finally:
            conn.close()

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        allow_retry: bool = True,
    ) -> bool:
        """Record job failure, triggering retry if attempts remain."""
        now = datetime.now(timezone.utc).isoformat()
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT retries, max_retries FROM work_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            retries, max_retries = row["retries"], row["max_retries"]

            if allow_retry and retries + 1 < max_retries:
                conn.execute(
                    """
                    UPDATE work_jobs
                    SET status = 'queued', worker_id = NULL, retries = retries + 1, error = ?
                    WHERE id = ?
                    """,
                    (error, job_id),
                )
                log.warning("Job %s failed (%s); requeued for retry %d/%d", job_id, error, retries + 1, max_retries)
            else:
                conn.execute(
                    """
                    UPDATE work_jobs
                    SET status = 'failed', completed_at = ?, error = ?
                    WHERE id = ?
                    """,
                    (now, error, job_id),
                )
                log.error("Job %s failed permanently: %s", job_id, error)

            conn.commit()
            return True
        finally:
            conn.close()

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or active job."""
        now = datetime.now(timezone.utc).isoformat()
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                UPDATE work_jobs
                SET status = 'cancelled', completed_at = ?
                WHERE id = ? AND status IN ('queued', 'leased', 'running')
                """,
                (now, job_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_job(self, job_id: str) -> WorkJob | None:
        """Retrieve job record by ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, job_type, payload, status, priority, worker_id, progress,
                       max_retries, retries, created_at, leased_at, completed_at,
                       heartbeat_at, error, result, metadata
                FROM work_jobs
                WHERE id = ?
                """,
                (job_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_job(row)
        finally:
            conn.close()

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 20,
    ) -> list[WorkJob]:
        """Query jobs with optional status and type filters."""
        conn = get_sync_connection(self.db_path)
        try:
            conditions = []
            params: list[Any] = []
            if status:
                conditions.append("status = ?")
                params.append(status)
            if job_type:
                conditions.append("job_type = ?")
                params.append(job_type)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            sql = f"""
                SELECT id, job_type, payload, status, priority, worker_id, progress,
                       max_retries, retries, created_at, leased_at, completed_at,
                       heartbeat_at, error, result, metadata
                FROM work_jobs
                {where}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_job(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> WorkJob:
        return WorkJob(
            id=row["id"],
            job_type=row["job_type"],
            payload=json.loads(row["payload"] or "{}"),
            status=row["status"],
            priority=row["priority"],
            worker_id=row["worker_id"],
            progress=row["progress"],
            max_retries=row["max_retries"],
            retries=row["retries"],
            created_at=row["created_at"],
            leased_at=row["leased_at"],
            completed_at=row["completed_at"],
            heartbeat_at=row["heartbeat_at"],
            error=row["error"],
            result=json.loads(row["result"] or "{}"),
            metadata=json.loads(row["metadata"] or "{}"),
        )
