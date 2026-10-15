"""
Verification utilities for ensuring task artifacts and outputs meet acceptance criteria.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from genesis.core.logging import get_logger
from genesis.tasks.models import Task, TaskStep
from genesis.tools.permissions import PermissionManager

log = get_logger("task_verifier")


class TaskVerifier:
    """Verifies that executed task outputs and artifacts meet real criteria."""

    def __init__(self, permission_manager: PermissionManager | None = None):
        self.permission_manager = permission_manager or PermissionManager()

    def verify_file_exists(self, path: str | Path, min_size_bytes: int = 1) -> tuple[bool, str]:
        """Verify that a specified file exists and is not empty."""
        try:
            resolved = self.permission_manager.validate_workspace_path(path)
            if not resolved.exists():
                return False, f"File does not exist: {path}"
            if not resolved.is_file():
                return False, f"Path is not a regular file: {path}"

            size = resolved.stat().st_size
            if size < min_size_bytes:
                return False, f"File size ({size} bytes) is less than required minimum ({min_size_bytes} bytes)."

            return True, f"Verified file exists ({size} bytes): {resolved.name}"
        except Exception as exc:
            return False, f"Verification failed with error: {exc}"

    def verify_file_contains(self, path: str | Path, expected_text: str) -> tuple[bool, str]:
        """Verify that a file exists and contains the expected substring."""
        exists, msg = self.verify_file_exists(path)
        if not exists:
            return False, msg

        try:
            resolved = self.permission_manager.validate_workspace_path(path)
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if expected_text not in content:
                return False, f"File '{resolved.name}' does not contain expected text '{expected_text}'"

            return True, f"Verified file '{resolved.name}' contains expected text."
        except Exception as exc:
            return False, f"Verification failed with error: {exc}"

    def verify_numeric_result(
        self,
        actual: float | int,
        expected: float | int,
        tolerance: float = 1e-4,
    ) -> tuple[bool, str]:
        """Verify a numerical result within an allowable tolerance."""
        try:
            diff = abs(actual - expected)
            if diff <= tolerance:
                return True, f"Numerical result {actual} matches expected {expected} (diff: {diff:.6f})."
            return False, f"Numerical result {actual} does not match expected {expected} (diff: {diff:.6f} > {tolerance})."
        except Exception as exc:
            return False, f"Numerical verification error: {exc}"

    def verify_task(self, task: Task, criteria: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Verify multiple criteria for a completed task.
        Supported criteria keys:
        - 'file_exists': str or list of paths
        - 'file_contains': dict mapping path to expected substring
        - 'require_success_steps': bool
        """
        results: list[str] = []
        all_passed = True

        if criteria.get("require_success_steps", True):
            failed_steps = [s for s in task.steps if s.status.value == "failed"]
            if failed_steps:
                all_passed = False
                results.append(f"Task contains {len(failed_steps)} failed steps.")
            else:
                results.append("All executed steps succeeded.")

        # Check file_exists
        files_to_check = criteria.get("file_exists")
        if files_to_check:
            if isinstance(files_to_check, (str, Path)):
                files_to_check = [files_to_check]
            for file_path in files_to_check:
                ok, msg = self.verify_file_exists(file_path)
                results.append(msg)
                if not ok:
                    all_passed = False

        # Check file_contains
        contains_dict = criteria.get("file_contains")
        if contains_dict and isinstance(contains_dict, dict):
            for file_path, substr in contains_dict.items():
                ok, msg = self.verify_file_contains(file_path, substr)
                results.append(msg)
                if not ok:
                    all_passed = False

        return all_passed, results
