"""
Safe file reading tool confined to the Genesis workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult
from genesis.tools.permissions import PermissionManager


class ReadFileTool(BaseTool):
    """Safely reads file contents from the workspace."""

    def __init__(self, permission_manager: PermissionManager | None = None):
        self.permission_manager = permission_manager or PermissionManager()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read content from a text file within the workspace. Supports line range selection.",
            permission=ToolPermission.READ_ONLY,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative or absolute path to the file to read (must be inside workspace).",
                    required=True,
                ),
                ToolParameter(
                    name="start_line",
                    type="integer",
                    description="1-based starting line number to read from (optional, default: 1).",
                    required=False,
                    default=1,
                ),
                ToolParameter(
                    name="max_lines",
                    type="integer",
                    description="Maximum number of lines to read (optional, default: 500, max: 2000).",
                    required=False,
                    default=500,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs.get("path")
        if not path_str:
            return ToolResult(success=False, error="Parameter 'path' is required.")

        start_line = max(1, int(kwargs.get("start_line", 1)))
        max_lines = min(2000, max(1, int(kwargs.get("max_lines", 500))))

        try:
            resolved_path = self.permission_manager.validate_workspace_path(path_str)
            if not resolved_path.exists():
                return ToolResult(success=False, error=f"File not found: {path_str}")
            if resolved_path.is_dir():
                return ToolResult(success=False, error=f"Path is a directory, not a file: {path_str}")

            # Read lines safely
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            selected_lines = all_lines[start_line - 1 : start_line - 1 + max_lines]
            content = "".join(selected_lines)

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "file_path": str(resolved_path),
                    "total_lines": total_lines,
                    "lines_returned": len(selected_lines),
                    "start_line": start_line,
                    "end_line": min(total_lines, start_line + len(selected_lines) - 1),
                },
            )

        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to read file '{path_str}': {exc}")
