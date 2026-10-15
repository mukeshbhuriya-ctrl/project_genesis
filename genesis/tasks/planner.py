"""
Task decomposition and plan generation for Project Genesis.
"""

from __future__ import annotations

import json
import re
from typing import Any

from genesis.core.logging import get_logger
from genesis.models import GenerationRequest
from genesis.models.ollama_backend import OllamaProvider
from genesis.tasks.models import StepStatus, Task, TaskStatus, TaskStep
from genesis.tools.registry import ToolRegistry, default_registry

log = get_logger("task_planner")


class TaskPlanner:
    """
    Decomposes an abstract goal into a bounded sequence of executable tool steps.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model_provider: OllamaProvider | None = None,
        max_steps: int = 10,
    ):
        self.registry = registry or default_registry
        self.provider = model_provider or OllamaProvider()
        self.max_steps = max_steps

    def create_deterministic_plan(
        self,
        goal: str,
        steps_data: list[dict[str, Any]],
    ) -> Task:
        """
        Create a Task directly from an explicit list of step specifications.
        """
        task = Task(goal=goal, status=TaskStatus.PENDING)
        task_steps = []

        for idx, s in enumerate(steps_data[: self.max_steps], start=1):
            tool_name = s.get("tool_name")
            tool_args = s.get("tool_arguments", {})
            desc = s.get("description", f"Step {idx}: Execute {tool_name}")

            task_steps.append(
                TaskStep(
                    task_id=task.id,
                    step_number=idx,
                    description=desc,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                    status=StepStatus.PENDING,
                )
            )

        task.steps = task_steps
        task.total_steps = len(task_steps)
        task.plan = [s.to_dict() for s in task_steps]
        return task

    async def plan_goal_async(
        self,
        goal: str,
        model_name: str | None = None,
    ) -> Task:
        """
        Use the local LLM to decompose a goal into discrete, valid tool steps.
        Falls back to a safe default single-step or query plan if the LLM output is malformed.
        """
        task = Task(goal=goal, status=TaskStatus.PLANNING)

        # Prepare tool specifications
        tools_summary = []
        for tool_def in self.registry.list_tools():
            param_desc = ", ".join(
                f"{p.name}: {p.type} ({'required' if p.required else 'optional'})"
                for p in tool_def.parameters
            )
            tools_summary.append(f"- {tool_def.name}: {tool_def.description} | Params: [{param_desc}]")

        tools_text = "\n".join(tools_summary)

        system_prompt = (
            "You are the task planner for Project Genesis, a local AI research platform.\n"
            "Your job is to decompose the user's goal into a bounded list of executable steps using available tools.\n\n"
            f"Available tools:\n{tools_text}\n\n"
            f"Rules:\n"
            f"1. Maximum {self.max_steps} steps.\n"
            f"2. Each step must use a valid tool name from the list, or null if it is a reasoning/verification step.\n"
            f"3. Return ONLY a valid JSON array of objects. No markdown formatting, no commentary.\n"
            f"Example schema:\n"
            f'[\n  {{"step_number": 1, "description": "Calculate dimension", "tool_name": "calculator", "tool_arguments": {{"expression": "768 * 4"}}}}\n]'
        )

        user_prompt = f"Goal: {goal}"

        try:
            req = GenerationRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_name or "",
                temperature=0.1,  # Low temperature for deterministic planning
                max_tokens=250,
            )
            response = await self.provider.generate(req)

            raw_text = response.content.strip()
            log.info(f"Model plan response:\n{raw_text}")

            # Parse JSON from response
            steps_data = self._extract_json_steps(raw_text)
            if not steps_data:
                log.warning("Could not extract JSON step array from model response. Generating fallback plan.")
                steps_data = [{"step_number": 1, "description": f"Evaluate goal: {goal}", "tool_name": None, "tool_arguments": {}}]

            # Validate tools and construct TaskStep objects
            task_steps = []
            for idx, s in enumerate(steps_data[: self.max_steps], start=1):
                tool_name = s.get("tool_name")
                # Verify tool exists in registry
                if tool_name and tool_name not in self.registry:
                    log.warning(f"Planner suggested unknown tool '{tool_name}'; marking tool as None")
                    tool_name = None

                tool_args = s.get("tool_arguments", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                desc = s.get("description", f"Step {idx}")

                task_steps.append(
                    TaskStep(
                        task_id=task.id,
                        step_number=idx,
                        description=desc,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        status=StepStatus.PENDING,
                    )
                )

            task.steps = task_steps
            task.total_steps = len(task_steps)
            task.plan = [s.to_dict() for s in task_steps]
            task.status = TaskStatus.PENDING
            return task

        except Exception as exc:
            log.error(f"Planning failed with error: {exc}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = f"Planning failed: {exc}"
            return task

    def _extract_json_steps(self, text: str) -> list[dict[str, Any]] | None:
        """Extract JSON array of steps from text, handling potential markdown blocks."""
        # Clean markdown code blocks if present
        text_clean = text.strip()
        if text_clean.startswith("```"):
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text_clean, re.DOTALL)
            if match:
                text_clean = match.group(1)

        # Look for [ ... ]
        bracket_match = re.search(r"(\[.*\])", text_clean, re.DOTALL)
        if bracket_match:
            try:
                parsed = json.loads(bracket_match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None
