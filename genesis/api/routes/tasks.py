"""
API routes for Task Orchestration, bounded planning, and verification.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.tasks.executor import TaskExecutor
from genesis.tasks.planner import TaskPlanner
from genesis.tasks.verifier import TaskVerifier

router = APIRouter(prefix="/tasks", tags=["tasks"])

planner = TaskPlanner()
executor = TaskExecutor()
verifier = TaskVerifier()


class PlanRequest(BaseModel):
    goal: str = Field(..., description="Abstract goal to decompose into tool steps")
    model: str | None = Field(default=None, description="Optional specific model override")


class RunTaskRequest(BaseModel):
    goal: str = Field(..., description="Goal of the task")
    steps: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional explicit step definitions. If omitted, the planner decomposes the goal.",
    )
    model: str | None = Field(default=None, description="Model to use for planning if steps omitted")


class VerifyTaskRequest(BaseModel):
    criteria: dict[str, Any] = Field(..., description="Acceptance criteria to verify against")


@router.get("")
async def list_tasks(limit: int = Query(50, ge=1, le=200)):
    """List recent tasks and their execution states."""
    tasks = executor.list_tasks(limit=limit)
    return {
        "count": len(tasks),
        "tasks": [t.to_dict() for t in tasks],
    }


@router.post("/plan")
async def plan_task(req: PlanRequest):
    """Decompose a goal into bounded steps using local model planning."""
    task = await planner.plan_goal_async(req.goal, model_name=req.model)
    executor.save_task(task)
    return task.to_dict()


@router.post("/run")
async def run_task(req: RunTaskRequest):
    """Plan and run a task end-to-end, or run explicitly provided steps."""
    if req.steps is not None and len(req.steps) > 0:
        task = planner.create_deterministic_plan(req.goal, req.steps)
    else:
        task = await planner.plan_goal_async(req.goal, model_name=req.model)

    completed_task = await executor.run_task_async(task)
    return completed_task.to_dict()


@router.get("/{task_id}")
async def get_task(task_id: str):
    """Retrieve full task details, steps, and audit logs."""
    task = executor.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task.to_dict()


@router.post("/{task_id}/verify")
async def verify_task(task_id: str, req: VerifyTaskRequest):
    """Verify task outcomes and generated artifacts against acceptance criteria."""
    task = executor.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    passed, details = verifier.verify_task(task, req.criteria)
    return {
        "task_id": task_id,
        "verified": passed,
        "details": details,
    }
