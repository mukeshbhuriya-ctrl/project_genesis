"""
Unit tests for the Genesis Tool Subsystem (genesis/tools/).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from genesis.tools.base import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolPermission,
    ToolResult,
)
from genesis.tools.builtin.calculator import CalculatorTool
from genesis.tools.builtin.dir_lister import ListDirectoryTool
from genesis.tools.builtin.file_reader import ReadFileTool
from genesis.tools.builtin.file_writer import WriteFileTool
from genesis.tools.builtin.python_eval import PythonEvalTool
from genesis.tools.executor import ToolExecutor
from genesis.tools.permissions import PermissionManager, SecurityError
from genesis.tools.registry import ToolRegistry, create_default_registry


class TestToolPermissions:
    """Test permission validation and workspace path sandboxing."""

    def test_permission_levels(self):
        pm = PermissionManager(
            allowed_permissions=[ToolPermission.READ_ONLY],
            allow_dangerous=False,
        )
        assert pm.is_permission_allowed(ToolPermission.READ_ONLY)
        assert not pm.is_permission_allowed(ToolPermission.MUTATING)
        assert not pm.is_permission_allowed(ToolPermission.DANGEROUS)

        with pytest.raises(SecurityError):
            pm.check_permission(ToolPermission.MUTATING, "test_tool")

    def test_path_confinement_inside_workspace(self, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])

        # Relative path inside workspace
        resolved = pm.validate_workspace_path("subfolder/file.txt")
        assert resolved == (tmp_path / "subfolder" / "file.txt").resolve()

        # Absolute path inside workspace
        child = tmp_path / "data" / "test.json"
        resolved_child = pm.validate_workspace_path(child)
        assert resolved_child == child.resolve()

    def test_path_confinement_blocks_traversal_and_external(self, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])

        # Traversal outside workspace
        with pytest.raises(SecurityError):
            pm.validate_workspace_path("../outside.txt")

        # Windows root or external directory
        with pytest.raises(SecurityError):
            pm.validate_workspace_path("C:\\Windows\\System32\\cmd.exe")


class TestCalculatorTool:
    """Test safe AST mathematical calculation tool."""

    def test_basic_arithmetic(self):
        calc = CalculatorTool()
        res = calc.execute(expression="2 + 3 * 4")
        assert res.success
        assert res.output == 14

    def test_math_functions(self):
        calc = CalculatorTool()
        res = calc.execute(expression="sqrt(144) + abs(-8)")
        assert res.success
        assert res.output == 20

    def test_float_division(self):
        calc = CalculatorTool()
        res = calc.execute(expression="10 / 4")
        assert res.success
        assert res.output == 2.5

    def test_rejects_unsafe_code(self):
        calc = CalculatorTool()
        res = calc.execute(expression="__import__('os').system('dir')")
        assert not res.success
        assert (
            "Only direct math function calls are permitted" in res.error
            or "not in the safe mathematical whitelist" in res.error
            or "not recognized" in res.error
        )

    def test_empty_expression(self):
        calc = CalculatorTool()
        res = calc.execute(expression="   ")
        assert not res.success


class TestFileTools:
    """Test file reader, writer, and directory listing tools."""

    def test_write_and_read_file(self, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])
        writer = WriteFileTool(permission_manager=pm)
        reader = ReadFileTool(permission_manager=pm)

        target_file = tmp_path / "notes" / "experiment.txt"
        content = "Line 1: Model initialised\nLine 2: Loss 0.042\nLine 3: Benchmark pass\n"

        # Write
        write_res = writer.execute(path=str(target_file), content=content)
        assert write_res.success
        assert target_file.exists()

        # Read all
        read_res = reader.execute(path=str(target_file))
        assert read_res.success
        assert read_res.output == content
        assert read_res.metadata["total_lines"] == 3

        # Read line range
        read_slice = reader.execute(path=str(target_file), start_line=2, max_lines=1)
        assert read_slice.success
        assert read_slice.output == "Line 2: Loss 0.042\n"

    def test_read_nonexistent_file(self, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])
        reader = ReadFileTool(permission_manager=pm)
        res = reader.execute(path=str(tmp_path / "nonexistent.txt"))
        assert not res.success
        assert "not found" in res.error.lower()

    def test_list_directory(self, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])
        lister = ListDirectoryTool(permission_manager=pm)

        (tmp_path / "file1.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "dir_a").mkdir()
        (tmp_path / "dir_a" / "file2.txt").write_text("world", encoding="utf-8")

        # Flat listing
        res_flat = lister.execute(path=str(tmp_path), recursive=False)
        assert res_flat.success
        names = {item["name"] for item in res_flat.output}
        assert "file1.txt" in names
        assert "dir_a" in names

        # Recursive listing
        res_rec = lister.execute(path=str(tmp_path), recursive=True)
        assert res_rec.success
        rec_names = {item["name"] for item in res_rec.output}
        assert any("file2.txt" in n for n in rec_names)


class TestPythonEvalTool:
    """Test subprocess Python evaluation tool."""

    def test_eval_simple_script(self):
        evaluator = PythonEvalTool()
        code = "a = 6; b = 7; print(a * b)"
        res = evaluator.execute(code=code)
        assert res.success
        assert "42" in res.output

    def test_eval_syntax_error(self):
        evaluator = PythonEvalTool()
        res = evaluator.execute(code="def broken(")
        assert not res.success
        assert "SyntaxError" in res.error


class TestToolRegistryAndExecutor:
    """Test tool registration, schema export, and safe execution."""

    def test_default_registry_has_core_tools(self):
        reg = create_default_registry()
        assert len(reg) >= 5
        assert "calculator" in reg
        assert "read_file" in reg
        assert "write_file" in reg
        assert "list_directory" in reg
        assert "python_eval" in reg

    def test_to_ollama_tools_format(self):
        reg = create_default_registry()
        schemas = reg.to_ollama_tools(["calculator"])
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert fn["name"] == "calculator"
        assert "expression" in fn["parameters"]["properties"]
        assert "expression" in fn["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_executor_async_timeout(self):
        class SlowTool(BaseTool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(name="slow_tool", description="Slow tool")

            def execute(self, **kwargs):
                return ToolResult(success=True)

            async def execute_async(self, **kwargs):
                await asyncio.sleep(2.0)
                return ToolResult(success=True)

        executor = ToolExecutor(default_timeout=0.1)
        res = await executor.execute_async(SlowTool(), timeout=0.1)
        assert not res.success
        assert "timed out" in res.error.lower()
