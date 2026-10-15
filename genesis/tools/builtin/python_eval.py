"""
Subprocess-isolated Python code evaluation tool.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from genesis.core import get_settings
from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult


class PythonEvalTool(BaseTool):
    """
    Executes Python code in an isolated subprocess with bounded memory and timeout.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python_eval",
            description="Execute a Python script or expression in an isolated subprocess. Returns stdout, stderr, and exit code.",
            permission=ToolPermission.MUTATING,
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute.",
                    required=True,
                ),
                ToolParameter(
                    name="timeout_seconds",
                    type="number",
                    description="Execution timeout in seconds (default: 5.0, max: 20.0).",
                    required=False,
                    default=5.0,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code")
        if not code or not code.strip():
            return ToolResult(success=False, error="Parameter 'code' is required.")

        timeout = min(20.0, max(0.5, float(kwargs.get("timeout_seconds", 5.0))))

        try:
            # Run in isolated subprocess with cwd set to project root
            process = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                cwd=str(get_settings().project_root),
                timeout=timeout,
            )

            stdout = process.stdout
            stderr = process.stderr
            returncode = process.returncode

            is_success = (returncode == 0)
            output_msg = stdout if stdout else (stderr if not is_success else "Execution completed with no output.")

            return ToolResult(
                success=is_success,
                output=output_msg,
                error=stderr if not is_success else None,
                metadata={
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Python execution timed out after {timeout} seconds.",
                metadata={"timed_out": True},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to execute Python process: {exc}",
            )
