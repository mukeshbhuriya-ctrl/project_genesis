"""
Unit tests for the SQLite-backed Decoupled Job Queue (genesis/core/queue.py).
"""

from __future__ import annotations

import datetime
from pathlib import Path
import pytest

from genesis.core.database import initialize_database, get_sync_connection
from genesis.core.queue import JobQueue, WorkJob


@pytest.fixture
def queue_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_queue.db"
    initialize_database(db)
    return db


class TestJobQueue:
    """Test job lifecycle: submit, lease, heartbeat, complete, fail, retry, orphan recovery."""

    def test_submit_and_get_job(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)
        job = queue.submit(
            job_type="custom",
            payload={"task": "test_echo", "val": 42},
            priority=20,
            metadata={"source": "pytest"},
        )

        assert job.id is not None
        assert job.job_type == "custom"
        assert job.status == "queued"
        assert job.priority == 20
        assert job.payload["val"] == 42
        assert job.metadata["source"] == "pytest"

        retrieved = queue.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.status == "queued"

    def test_atomic_lease_priority_order(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)

        # Submit low priority job first, then high priority job
        j1 = queue.submit(job_type="custom", payload={"job": "low"}, priority=5)
        j2 = queue.submit(job_type="custom", payload={"job": "high"}, priority=50)

        # Lease next job
        leased = queue.lease(worker_id="worker-001")
        assert leased is not None
        assert leased.id == j2.id  # Higher priority leased first
        assert leased.status == "leased"
        assert leased.worker_id == "worker-001"
        assert leased.leased_at is not None

        # Next lease gets j1
        leased_second = queue.lease(worker_id="worker-002")
        assert leased_second is not None
        assert leased_second.id == j1.id

        # Queue now empty
        assert queue.lease(worker_id="worker-003") is None

    def test_heartbeat_and_progress(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)
        job = queue.submit(job_type="custom", payload={})
        queue.lease(worker_id="w1")

        # Heartbeat updates status to running and sets progress
        success = queue.heartbeat(job.id, worker_id="w1", progress=0.45)
        assert success is True

        updated = queue.get_job(job.id)
        assert updated.status == "running"
        assert updated.progress == 0.45
        assert updated.heartbeat_at is not None

    def test_complete_job(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)
        job = queue.submit(job_type="custom", payload={})
        queue.lease(worker_id="w1")

        ok = queue.complete(job.id, worker_id="w1", result={"answer": 42})
        assert ok is True

        completed = queue.get_job(job.id)
        assert completed.status == "completed"
        assert completed.progress == 1.0
        assert completed.completed_at is not None
        assert completed.result["answer"] == 42

    def test_failure_retry_and_exhaustion(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)
        job = queue.submit(job_type="custom", payload={}, max_retries=2)
        queue.lease(worker_id="w1")

        # 1st failure -> requeued for retry (retries becomes 1 < 2)
        queue.fail(job.id, worker_id="w1", error="Transient GPU reset")
        j_after_1 = queue.get_job(job.id)
        assert j_after_1.status == "queued"
        assert j_after_1.retries == 1

        # Re-lease and fail a 2nd time -> retries + 1 == 2 >= max_retries -> marked failed permanently
        queue.lease(worker_id="w1")
        queue.fail(job.id, worker_id="w1", error="Fatal OOM")
        j_after_2 = queue.get_job(job.id)
        assert j_after_2.status == "failed"
        assert "Fatal OOM" in j_after_2.error

    def test_cancel_job(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)
        job = queue.submit(job_type="custom", payload={})
        assert queue.cancel(job.id) is True

        cancelled = queue.get_job(job.id)
        assert cancelled.status == "cancelled"

    def test_orphan_crash_recovery(self, queue_db: Path):
        queue = JobQueue(db_path=queue_db)
        job = queue.submit(job_type="custom", payload={})
        queue.lease(worker_id="w_crashed")

        # Simulate old heartbeat from a dead worker 600s in the past
        past_time = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600)
        ).isoformat()
        conn = get_sync_connection(queue_db)
        conn.execute(
            "UPDATE work_jobs SET heartbeat_at = ?, status = 'running' WHERE id = ?",
            (past_time, job.id),
        )
        conn.commit()
        conn.close()

        # New worker leases from queue: queue.lease() checks for orphans with timeout 300s
        recovered = queue.lease(worker_id="w_alive", lease_timeout_seconds=300.0)
        assert recovered is not None
        assert recovered.id == job.id
        assert recovered.worker_id == "w_alive"
        assert recovered.retries == 1
