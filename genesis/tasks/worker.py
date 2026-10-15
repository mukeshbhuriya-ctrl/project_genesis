"""
Standalone background job execution worker.

Decouples long-running model training, benchmark suites, and synthesis tasks
from the primary API server process into dedicated, crash-resilient worker loops.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable


from genesis.core.logging import get_logger
from genesis.core.queue import JobQueue, WorkJob

log = get_logger("job_worker")


class JobWorker:
    """
    Independent worker daemon that leases jobs from SQLite work_jobs,
    executes them asynchronously, emits heartbeats, and records completion.
    """

    def __init__(
        self,
        worker_id: str | None = None,
        queue: JobQueue | None = None,
        db_path: str | Path = "data/genesis.db",
        poll_interval: float = 2.0,
        heartbeat_interval: float = 10.0,
    ) -> None:
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.db_path = Path(db_path)
        self.queue = queue or JobQueue(db_path=self.db_path)
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.running = False
        self._handlers: dict[str, Callable[[WorkJob], Any]] = {}
        self._register_default_handlers()

    def register_handler(self, job_type: str, handler: Callable[[WorkJob], Any]) -> None:
        """Register a custom processing function for a job type."""
        self._handlers[job_type] = handler

    def _register_default_handlers(self) -> None:
        """Attach default subsystem execution handlers."""
        self._handlers["custom"] = self._handle_custom
        self._handlers["code_synthesis"] = self._handle_code_synthesis
        self._handlers["research_synthesis"] = self._handle_research_synthesis
        self._handlers["fine_tune"] = self._handle_fine_tune
        self._handlers["eval_benchmark"] = self._handle_eval_benchmark

    async def _handle_custom(self, job: WorkJob) -> dict[str, Any]:
        """Simple execution handler for custom or test jobs."""
        action = job.payload.get("action", "noop")
        duration = float(job.payload.get("duration", 0.1))
        await asyncio.sleep(duration)
        return {"action": action, "executed_by": self.worker_id}

    async def _handle_code_synthesis(self, job: WorkJob) -> dict[str, Any]:
        from genesis.research.coding import ReflectiveCodeSynthesizer
        synthesizer = ReflectiveCodeSynthesizer(db_path=self.db_path)
        objective = job.payload.get("objective", "")
        max_iter = int(job.payload.get("max_iterations", 3))
        rec = await synthesizer.synthesize_and_repair(
            objective=objective,
            name=job.payload.get("name"),
            max_iterations=max_iter,
            export_dataset_id=job.payload.get("export_dataset_id"),
        )
        return rec.to_dict()

    async def _handle_research_synthesis(self, job: WorkJob) -> dict[str, Any]:
        from genesis.research.synthesis import AdaptiveKnowledgeSynthesizer
        synthesizer = AdaptiveKnowledgeSynthesizer(db_path=self.db_path)
        brief = await synthesizer.synthesize(
            topic=job.payload.get("topic", ""),
            query=job.payload.get("query"),
            max_sources=int(job.payload.get("max_sources", 5)),
            save_to_memory=bool(job.payload.get("save_to_memory", True)),
        )
        return brief.to_dict()

    async def _handle_fine_tune(self, job: WorkJob) -> dict[str, Any]:
        from genesis.training.experiments import ExperimentTracker
        from genesis.training.trainer import FineTuningTrainer
        exp_id = job.payload.get("experiment_id")
        if not exp_id:
            raise ValueError("fine_tune payload missing experiment_id")
        trainer = FineTuningTrainer(db_path=self.db_path)
        exp = await trainer.train(exp_id)
        return exp.to_dict()

    async def _handle_eval_benchmark(self, job: WorkJob) -> dict[str, Any]:
        from genesis.evaluation.benchmarks import BenchmarkRunner
        runner = BenchmarkRunner(db_path=self.db_path)
        bm_id = job.payload.get("benchmark_id", "bm-reasoning-v1")
        model = job.payload.get("model")
        run = await runner.run_benchmark(benchmark_id=bm_id, model=model)
        return run.to_dict()

    async def run_once(self) -> bool:
        """
        Poll for and process a single job from the queue.
        Returns True if a job was found and executed, False if queue was empty.
        """
        job = self.queue.lease(worker_id=self.worker_id)
        if not job:
            return False

        log.info("Worker %s acquired job %s (%s)", self.worker_id, job.id, job.job_type)
        handler = self._handlers.get(job.job_type)

        if not handler:
            self.queue.fail(job.id, self.worker_id, f"No handler registered for job type: {job.job_type}", allow_retry=False)
            return True


        # Heartbeat background task
        hb_stop_event = asyncio.Event()

        async def _heartbeat_loop():
            while not hb_stop_event.is_set():
                await asyncio.sleep(self.heartbeat_interval)
                if not hb_stop_event.is_set():
                    self.queue.heartbeat(job.id, self.worker_id)

        hb_task = asyncio.create_task(_heartbeat_loop())

        try:
            self.queue.heartbeat(job.id, self.worker_id, progress=0.05)
            # Execute handler
            if inspect.iscoroutinefunction(handler):
                result = await handler(job)

            else:
                result = handler(job)

            self.queue.complete(job.id, self.worker_id, result=result if isinstance(result, dict) else {"output": str(result)})
            log.info("Worker %s completed job %s successfully", self.worker_id, job.id)
        except Exception as e:
            log.error("Worker %s error executing job %s: %s", self.worker_id, job.id, e)
            self.queue.fail(job.id, self.worker_id, str(e))
        finally:
            hb_stop_event.set()
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass

        return True

    async def start(self) -> None:
        """Start long-running worker daemon loop."""
        self.running = True
        log.info("Starting Genesis Job Worker %s (polling every %.1fs)", self.worker_id, self.poll_interval)

        while self.running:
            try:
                processed = await self.run_once()
                if not processed:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Unexpected error in worker loop: %s", e)
                await asyncio.sleep(self.poll_interval)

        log.info("Worker %s stopped cleanly.", self.worker_id)

    def stop(self) -> None:
        """Signal worker to stop after completing active turn."""
        self.running = False
