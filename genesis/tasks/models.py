"""
Data models and state representations for Task Orchestration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle state of an overarching task."""
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Lifecycle state of an individual task step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """A single discrete step within a planned task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    step_number: int = 1
    description: str = ""
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_number": self.step_number,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class Task:
    """An orchestrated goal with structured steps and execution trace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    plan: list[dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    result: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[TaskStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "plan": self.plan,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }
