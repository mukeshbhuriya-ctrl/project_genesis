"""
Reflective Code Synthesis tool for autonomous coding and sandbox self-repair.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any

from genesis.models.ollama_backend import OllamaProvider
from genesis.research.coding import ReflectiveCodeSynthesizer
from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult


class CodeSynthesizerTool(BaseTool):
    """
    Autonomously decomposes coding objectives, generates implementations and tests,
    executes them in an isolated sandbox, and performs reflective self-repair on failure.
    """

    def __init__(
        self,
        db_path: str | Path = "data/genesis.db",
        provider: OllamaProvider | None = None,
        model_id: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.provider = provider
        self.model_id = model_id

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="synthesize_code",
            description=(
                "Autonomously synthesize Python code to satisfy a programming objective, "
                "verify it in an isolated subprocess sandbox with unit tests, and iteratively "
                "repair it on failure."
            ),
            permission=ToolPermission.MUTATING,
            parameters=[
                ToolParameter(
                    name="objective",
                    type="string",
                    description="Clear programming goal or function to implement in Python.",
                    required=True,
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="Optional short identifier for this code experiment.",
                    required=False,
                    default=None,
                ),
                ToolParameter(
                    name="max_iterations",
                    type="integer",
                    description="Maximum number of self-repair iterations (default: 3).",
                    required=False,
                    default=3,
                ),
                ToolParameter(
                    name="export_dataset_id",
                    type="string",
                    description="Optional Phase 5 training dataset ID to record verified code sample into.",
                    required=False,
                    default=None,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.execute_async(**kwargs))).result()
        return asyncio.run(self.execute_async(**kwargs))

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        objective = kwargs.get("objective")
        if not objective or not objective.strip():
            return ToolResult(success=False, error="Parameter 'objective' is required.")

        name = kwargs.get("name")
        max_iterations = int(kwargs.get("max_iterations", 3))
        export_dataset_id = kwargs.get("export_dataset_id")

        try:
            synthesizer = ReflectiveCodeSynthesizer(
                provider=self.provider,
                db_path=self.db_path,
                model_id=self.model_id or "qwen2.5-coder:7b",
            )
            record = await synthesizer.synthesize_and_repair(
                objective=objective.strip(),
                name=name,
                max_iterations=max_iterations,
                export_dataset_id=export_dataset_id,
            )

            success = record.status == "passed"
            output = {
                "id": record.id,
                "status": record.status,
                "iterations": record.iterations,
                "final_code": record.final_code,
                "test_harness": record.tests,
                "error": record.error,

            }
            return ToolResult(
                success=success,
                output=output,
                error=record.error if not success else None,
                metadata=record.to_dict(),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Code synthesis failed: {exc}",
            )
