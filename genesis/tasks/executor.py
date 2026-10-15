"""
Task execution engine with persistent step tracking and audit logging in SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.tasks.models import StepStatus, Task, TaskStatus, TaskStep

if TYPE_CHECKING:
    from genesis.tools.executor import ToolExecutor
    from genesis.tools.registry import ToolRegistry

log = get_logger("task_executor")


class TaskExecutor:
    """
    Executes tasks step-by-step, running tools and persisting full audit logs.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        db_path: Path | str | None = None,
    ):
        if registry is None:
            from genesis.tools.registry import default_registry
            self.registry = default_registry
        else:
            self.registry = registry

        if tool_executor is None:
            from genesis.tools.executor import ToolExecutor
            self.tool_executor = ToolExecutor()
        else:
            self.tool_executor = tool_executor

        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(get_settings().database_url)

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    # ── Database Persistence ──

    def save_task(self, task: Task) -> None:
        """Insert or update a task and its steps in the database."""
        conn = self._get_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            task.updated_at = now

            conn.execute(
                """
                INSERT INTO tasks (
                    id, goal, status, plan, current_step, total_steps, result, error, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    current_step = excluded.current_step,
                    total_steps = excluded.total_steps,
                    result = excluded.result,
                    error = excluded.error,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    task.id,
                    task.goal,
                    task.status.value,
                    json.dumps(task.plan),
                    task.current_step,
                    task.total_steps,
                    task.result,
                    task.error,
                    task.created_at,
                    task.updated_at,
                    json.dumps(task.metadata),
                ),
            )

            for step in task.steps:
                conn.execute(
                    """
                    INSERT INTO task_steps (
                        id, task_id, step_number, description, tool_name, tool_arguments,
                        tool_result, status, error, started_at, completed_at, duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        tool_result = excluded.tool_result,
                        status = excluded.status,
                        error = excluded.error,
                        started_at = excluded.started_at,
                        completed_at = excluded.completed_at,
                        duration_ms = excluded.duration_ms
                    """,
                    (
                        step.id,
                        task.id,
                        step.step_number,
                        step.description,
                        step.tool_name,
                        json.dumps(step.tool_arguments),
                        json.dumps(step.tool_result),
                        step.status.value,
                        step.error,
                        step.started_at,
                        step.completed_at,
                        step.duration_ms,
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    def get_task(self, task_id: str) -> Task | None:
        """Retrieve a task and all associated steps by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            task_row = cursor.fetchone()
            if not task_row:
                return None

            task = Task(
                id=task_row["id"],
                goal=task_row["goal"],
                status=TaskStatus(task_row["status"]),
                plan=json.loads(task_row["plan"] or "[]"),
                current_step=task_row["current_step"],
                total_steps=task_row["total_steps"],
                result=task_row["result"],
                error=task_row["error"],
                created_at=task_row["created_at"],
                updated_at=task_row["updated_at"],
                metadata=json.loads(task_row["metadata"] or "{}"),
            )

            # Retrieve steps
            step_cursor = conn.execute(
                "SELECT * FROM task_steps WHERE task_id = ? ORDER BY step_number ASC",
                (task_id,),
            )
            steps = []
            for s_row in step_cursor.fetchall():
                step = TaskStep(
                    id=s_row["id"],
                    task_id=s_row["task_id"],
                    step_number=s_row["step_number"],
                    description=s_row["description"],
                    tool_name=s_row["tool_name"],
                    tool_arguments=json.loads(s_row["tool_arguments"] or "{}"),
                    tool_result=json.loads(s_row["tool_result"] or "{}"),
                    status=StepStatus(s_row["status"]),
                    error=s_row["error"],
                    started_at=s_row["started_at"],
                    completed_at=s_row["completed_at"],
                    duration_ms=s_row["duration_ms"] or 0.0,
                )
                steps.append(step)

            task.steps = steps
            return task
        finally:
            conn.close()

    def list_tasks(self, limit: int = 50) -> list[Task]:
        """List tasks ordered by creation date descending."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            tasks = []
            for row in cursor.fetchall():
                t = Task(
                    id=row["id"],
                    goal=row["goal"],
                    status=TaskStatus(row["status"]),
                    plan=json.loads(row["plan"] or "[]"),
                    current_step=row["current_step"],
                    total_steps=row["total_steps"],
                    result=row["result"],
                    error=row["error"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
                tasks.append(t)
            return tasks
        finally:
            conn.close()

    # ── Step & Task Execution ──

    def execute_step(self, step: TaskStep) -> bool:
        """
        Execute an individual task step synchronously.
        Returns True if successful, False if failed.
        """
        step.started_at = datetime.now(timezone.utc).isoformat()
        step.status = StepStatus.RUNNING
        t0 = time.perf_counter()

        log.info(f"Executing step {step.step_number}: {step.description}")

        if not step.tool_name:
            # Reasoning or no-op step
            step.duration_ms = (time.perf_counter() - t0) * 1000
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.status = StepStatus.SUCCESS
            step.tool_result = {"info": "No tool required for step"}
            return True

        tool = self.registry.get(step.tool_name)
        if not tool:
            step.duration_ms = (time.perf_counter() - t0) * 1000
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.status = StepStatus.FAILED
            step.error = f"Tool '{step.tool_name}' not found in registry."
            return False

        # Execute tool via executor
        res = self.tool_executor.execute(tool, **step.tool_arguments)
        step.duration_ms = (time.perf_counter() - t0) * 1000
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.tool_result = res.to_dict()

        if res.success:
            step.status = StepStatus.SUCCESS
            return True
        else:
            step.status = StepStatus.FAILED
            step.error = res.error
            return False

    async def execute_step_async(self, step: TaskStep, timeout: float = 30.0) -> bool:
        """
        Execute an individual task step asynchronously with timeout.
        """
        step.started_at = datetime.now(timezone.utc).isoformat()
        step.status = StepStatus.RUNNING
        t0 = time.perf_counter()

        log.info(f"Executing step {step.step_number} async: {step.description}")

        if not step.tool_name:
            step.duration_ms = (time.perf_counter() - t0) * 1000
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.status = StepStatus.SUCCESS
            step.tool_result = {"info": "No tool required for step"}
            return True

        tool = self.registry.get(step.tool_name)
        if not tool:
            step.duration_ms = (time.perf_counter() - t0) * 1000
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.status = StepStatus.FAILED
            step.error = f"Tool '{step.tool_name}' not found in registry."
            return False

        res = await self.tool_executor.execute_async(tool, timeout=timeout, **step.tool_arguments)
        step.duration_ms = (time.perf_counter() - t0) * 1000
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.tool_result = res.to_dict()

        if res.success:
            step.status = StepStatus.SUCCESS
            return True
        else:
            step.status = StepStatus.FAILED
            step.error = res.error
            return False

    def run_task(self, task: Task) -> Task:
        """
        Run all steps of a task sequentially, persisting state after each step.
        """
        task.status = TaskStatus.RUNNING
        self.save_task(task)

        for step in task.steps:
            task.current_step = step.step_number
            success = self.execute_step(step)
            self.save_task(task)

            if not success:
                task.status = TaskStatus.FAILED
                task.error = f"Step {step.step_number} failed: {step.error}"
                self.save_task(task)
                return task

        task.status = TaskStatus.COMPLETED
        task.result = f"Successfully completed all {len(task.steps)} steps."
        self.save_task(task)
        return task

    async def run_task_async(self, task: Task, step_timeout: float = 30.0) -> Task:
        """
        Run all steps of a task asynchronously, persisting state after each step.
        """
        task.status = TaskStatus.RUNNING
        self.save_task(task)

        for step in task.steps:
            task.current_step = step.step_number
            success = await self.execute_step_async(step, timeout=step_timeout)
            self.save_task(task)

            if not success:
                task.status = TaskStatus.FAILED
                task.error = f"Step {step.step_number} failed: {step.error}"
                self.save_task(task)
                return task

        task.status = TaskStatus.COMPLETED
        task.result = f"Successfully completed all {len(task.steps)} steps."
        self.save_task(task)
        return task
