"""
Unit tests for the Genesis JobWorker daemon (genesis/tasks/worker.py).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.core.queue import JobQueue
from genesis.tasks.worker import JobWorker


@pytest.fixture
def worker_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_worker.db"
    initialize_database(db)
    return db


@pytest.mark.asyncio
class TestJobWorker:
    """Test standalone JobWorker daemon processing."""

    async def test_worker_run_once_custom_job(self, worker_db: Path):
        queue = JobQueue(db_path=worker_db)
        job = queue.submit(
            job_type="custom",
            payload={"action": "test_echo", "duration": 0.05},
        )

        worker = JobWorker(worker_id="test-worker-1", queue=queue, db_path=worker_db)
        processed = await worker.run_once()
        assert processed is True

        updated = queue.get_job(job.id)
        assert updated.status == "completed"
        assert updated.worker_id == "test-worker-1"
        assert updated.result["action"] == "test_echo"

    async def test_worker_run_once_empty_queue(self, worker_db: Path):
        queue = JobQueue(db_path=worker_db)
        worker = JobWorker(queue=queue, db_path=worker_db)
        processed = await worker.run_once()
        assert processed is False

    async def test_worker_handles_unregistered_job_type(self, worker_db: Path):
        queue = JobQueue(db_path=worker_db)
        job = queue.submit(job_type="custom", payload={})

        worker = JobWorker(queue=queue, db_path=worker_db)
        # Deliberately remove the default handler to test missing handler branch
        del worker._handlers["custom"]

        processed = await worker.run_once()
        assert processed is True

        updated = queue.get_job(job.id)
        assert updated.status == "failed"
        assert "No handler registered" in updated.error

    async def test_worker_custom_handler_registration(self, worker_db: Path):
        queue = JobQueue(db_path=worker_db)
        job = queue.submit(
            job_type="custom",
            payload={"a": 12, "b": 30},
        )

        worker = JobWorker(queue=queue, db_path=worker_db)

        def _multiply_handler(j):
            a = j.payload["a"]
            b = j.payload["b"]
            return {"product": a * b}

        # Override handler for custom type
        worker.register_handler("custom", _multiply_handler)
        processed = await worker.run_once()
        assert processed is True

        updated = queue.get_job(job.id)
        assert updated.status == "completed"
        assert updated.result["product"] == 360

    async def test_worker_heartbeat_loop_fires(self, worker_db: Path):
        queue = JobQueue(db_path=worker_db)
        job = queue.submit(
            job_type="custom",
            payload={},
        )

        # Worker with 0.05s heartbeat interval
        worker = JobWorker(
            queue=queue,
            db_path=worker_db,
            heartbeat_interval=0.05,
        )

        async def _slow_handler(j):
            await asyncio.sleep(0.2)
            return {"status": "ok"}

        worker.register_handler("custom", _slow_handler)
        await worker.run_once()

        updated = queue.get_job(job.id)
        assert updated.status == "completed"
        assert updated.heartbeat_at is not None
