"""
Safe directory listing tool confined to the Genesis workspace.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult
from genesis.tools.permissions import PermissionManager


class ListDirectoryTool(BaseTool):
    """Safely lists files and folders in a workspace directory."""

    def __init__(self, permission_manager: PermissionManager | None = None):
        self.permission_manager = permission_manager or PermissionManager()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List files and subdirectories within a workspace path. Returns names, sizes, and types.",
            permission=ToolPermission.READ_ONLY,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to directory (relative or absolute, inside workspace). Default: current directory.",
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="Whether to list subdirectories recursively (default: false).",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="max_items",
                    type="integer",
                    description="Maximum number of items to return (default: 100, max: 500).",
                    required=False,
                    default=100,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        path_str = kwargs.get("path", ".")
        recursive = bool(kwargs.get("recursive", False))
        max_items = min(500, max(1, int(kwargs.get("max_items", 100))))

        try:
            resolved_path = self.permission_manager.validate_workspace_path(path_str)
            if not resolved_path.exists():
                return ToolResult(success=False, error=f"Directory not found: {path_str}")
            if not resolved_path.is_dir():
                return ToolResult(success=False, error=f"Path is not a directory: {path_str}")

            items = []
            if recursive:
                for root, dirs, files in os.walk(resolved_path):
                    # Filter out heavy/git folders
                    dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__", "node_modules"}]
                    rel_root = Path(root).relative_to(resolved_path)
                    for d in dirs:
                        if len(items) >= max_items:
                            break
                        items.append({
                            "name": str(rel_root / d) if str(rel_root) != "." else d,
                            "type": "directory",
                            "size_bytes": 0,
                        })
                    for f in files:
                        if len(items) >= max_items:
                            break
                        file_path = Path(root) / f
                        try:
                            size = file_path.stat().st_size
                        except OSError:
                            size = 0
                        items.append({
                            "name": str(rel_root / f) if str(rel_root) != "." else f,
                            "type": "file",
                            "size_bytes": size,
                        })
                    if len(items) >= max_items:
                        break
            else:
                for entry in sorted(resolved_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                    if entry.name in {".git", "__pycache__"}:
                        continue
                    if len(items) >= max_items:
                        break
                    is_dir = entry.is_dir()
                    try:
                        size = 0 if is_dir else entry.stat().st_size
                    except OSError:
                        size = 0
                    items.append({
                        "name": entry.name,
                        "type": "directory" if is_dir else "file",
                        "size_bytes": size,
                    })

            return ToolResult(
                success=True,
                output=items,
                metadata={
                    "directory": str(resolved_path),
                    "total_returned": len(items),
                    "recursive": recursive,
                },
            )

        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to list directory '{path_str}': {exc}")
