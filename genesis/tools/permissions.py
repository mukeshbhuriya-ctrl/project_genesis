"""
Permission model and filesystem sandboxing for Project Genesis tools.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from genesis.core import get_settings
from genesis.tools.base import ToolPermission


class SecurityError(Exception):
    """Raised when a tool operation violates security boundaries."""
    pass


class PermissionManager:
    """
    Governs permission checks and filesystem access confinement.
    
    Default policy:
    - READ_ONLY operations permitted automatically.
    - MUTATING operations permitted within allowed workspace directories.
    - DANGEROUS operations blocked unless allow_dangerous is explicitly enabled.
    """

    def __init__(
        self,
        allowed_permissions: Sequence[ToolPermission] | None = None,
        workspace_roots: Sequence[Path | str] | None = None,
        allow_dangerous: bool = False,
    ):
        if allowed_permissions is not None:
            self.allowed_permissions = set(allowed_permissions)
        else:
            self.allowed_permissions = {ToolPermission.READ_ONLY, ToolPermission.MUTATING}
            if allow_dangerous:
                self.allowed_permissions.add(ToolPermission.DANGEROUS)

        if workspace_roots is not None:
            self.workspace_roots = [Path(r).resolve() for r in workspace_roots]
        else:
            # Default to project root
            self.workspace_roots = [get_settings().project_root.resolve()]

    def is_permission_allowed(self, permission: ToolPermission) -> bool:
        """Check if a tool permission level is permitted under current policy."""
        return permission in self.allowed_permissions

    def check_permission(self, permission: ToolPermission, tool_name: str) -> None:
        """Raise SecurityError if permission level is not allowed."""
        if not self.is_permission_allowed(permission):
            raise SecurityError(
                f"Permission denied: Tool '{tool_name}' requires permission '{permission.value}', "
                f"which is not granted in the current security policy."
            )

    def validate_workspace_path(self, path: Path | str, allow_create: bool = False) -> Path:
        """
        Validate that the target path is strictly confined within one of the workspace roots.
        Prevents directory traversal, symlink escapes, and access to system directories.
        """
        raw_path = Path(path)
        
        # If relative, resolve relative to primary workspace root
        if not raw_path.is_absolute():
            resolved = (self.workspace_roots[0] / raw_path).resolve()
        else:
            resolved = raw_path.resolve()

        # Check against allowed roots
        is_contained = False
        for root in self.workspace_roots:
            try:
                resolved.relative_to(root)
                is_contained = True
                break
            except ValueError:
                continue

        if not is_contained:
            raise SecurityError(
                f"Access denied: Path '{path}' resolves to '{resolved}', which is outside "
                f"the permitted workspace boundaries ({[str(r) for r in self.workspace_roots]})."
            )

        # Block root drive direct modifications or access to sensitive files
        if resolved == resolved.anchor:
            raise SecurityError(f"Access denied: Cannot target drive root '{resolved}'.")

        # Disallow sensitive system folders or hidden sensitive paths
        parts = [p.lower() for p in resolved.parts]
        forbidden_parts = {"windows", "system32", "program files", "program files (x86)", ".ssh", ".gnupg"}
        if any(part in forbidden_parts for part in parts):
            raise SecurityError(f"Access denied: Path '{resolved}' accesses a protected system directory.")

        return resolved
