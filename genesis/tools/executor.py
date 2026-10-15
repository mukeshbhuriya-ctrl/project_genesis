"""
Tool execution engine with timeout enforcement, sandboxing, and execution metrics.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from genesis.core.logging import get_logger
from genesis.tools.base import BaseTool, ToolResult
from genesis.tools.permissions import PermissionManager, SecurityError

log = get_logger("tool_executor")


class ToolExecutor:
    """
    Executes tools safely with permission checks, timeout limits, and error isolation.
    """

    def __init__(
        self,
        permission_manager: PermissionManager | None = None,
        default_timeout: float = 30.0,
    ):
        self.permission_manager = permission_manager or PermissionManager()
        self.default_timeout = default_timeout

    def execute(self, tool: BaseTool, **kwargs: Any) -> ToolResult:
        """
        Synchronously execute a tool with permission checking and timing.
        """
        start_time = time.perf_counter()
        tool_name = tool.name

        try:
            # 1. Security & permission check
            self.permission_manager.check_permission(tool.permission, tool_name)

            log.info(f"Executing tool '{tool_name}' with arguments: {list(kwargs.keys())}")
            result = tool.execute(**kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = duration_ms
            log.info(f"Tool '{tool_name}' completed in {duration_ms:.2f}ms (success={result.success})")
            return result

        except SecurityError as sec_err:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.warning(f"Security error executing tool '{tool_name}': {sec_err}")
            return ToolResult(
                success=False,
                error=str(sec_err),
                execution_time_ms=duration_ms,
                metadata={"security_violation": True},
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.error(f"Error executing tool '{tool_name}': {exc}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {str(exc)}",
                execution_time_ms=duration_ms,
            )

    async def execute_async(
        self,
        tool: BaseTool,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Asynchronously execute a tool with timeout enforcement.
        """
        start_time = time.perf_counter()
        tool_name = tool.name
        eff_timeout = timeout if timeout is not None else self.default_timeout

        try:
            # 1. Security & permission check
            self.permission_manager.check_permission(tool.permission, tool_name)

            log.info(f"Executing tool '{tool_name}' async (timeout={eff_timeout}s)")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute_async(**kwargs),
                timeout=eff_timeout,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = duration_ms
            return result

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Tool '{tool_name}' execution timed out after {eff_timeout} seconds."
            log.warning(err_msg)
            return ToolResult(
                success=False,
                error=err_msg,
                execution_time_ms=duration_ms,
                metadata={"timed_out": True},
            )

        except SecurityError as sec_err:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.warning(f"Security error executing tool '{tool_name}': {sec_err}")
            return ToolResult(
                success=False,
                error=str(sec_err),
                execution_time_ms=duration_ms,
                metadata={"security_violation": True},
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.error(f"Error executing tool '{tool_name}': {exc}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {str(exc)}",
                execution_time_ms=duration_ms,
            )
