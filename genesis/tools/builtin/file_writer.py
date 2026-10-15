"""
Safe file writing tool confined to the Genesis workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult
from genesis.tools.permissions import PermissionManager


class WriteFileTool(BaseTool):
    """Safely writes content to files within the workspace."""

    def __init__(self, permission_manager: PermissionManager | None = None):
        self.permission_manager = permission_manager or PermissionManager()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Write content to a file within the workspace. Creates parent directories if needed.",
            permission=ToolPermission.MUTATING,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative or absolute path to the file to write (must be inside workspace).",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Text content to write into the file.",
                    required=True,
                ),
                ToolParameter(
                    name="overwrite",
                    type="boolean",
                    description="Whether to overwrite if file already exists (default: true).",
                    required=False,
                    default=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs.get("path")
        if not path_str:
            return ToolResult(success=False, error="Parameter 'path' is required.")

        content = kwargs.get("content")
        if content is None:
            return ToolResult(success=False, error="Parameter 'content' is required.")

        overwrite = bool(kwargs.get("overwrite", True))

        try:
            resolved_path = self.permission_manager.validate_workspace_path(path_str, allow_create=True)
            if resolved_path.exists() and not overwrite:
                return ToolResult(
                    success=False,
                    error=f"File already exists and overwrite is set to False: {path_str}",
                )

            # Ensure parent directories exist
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(content)

            bytes_written = len(content.encode("utf-8"))

            return ToolResult(
                success=True,
                output=f"Successfully wrote {bytes_written} bytes to {resolved_path.name}",
                metadata={
                    "file_path": str(resolved_path),
                    "bytes_written": bytes_written,
                    "overwrite": overwrite,
                },
            )

        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to write file '{path_str}': {exc}")
