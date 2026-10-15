"""
Central registry for tools in Project Genesis.
"""

from __future__ import annotations

from typing import Any, Iterable

from genesis.core.logging import get_logger
from genesis.tools.base import BaseTool, ToolDefinition
from genesis.tools.builtin.calculator import CalculatorTool
from genesis.tools.builtin.code_synthesizer import CodeSynthesizerTool
from genesis.tools.builtin.dir_lister import ListDirectoryTool
from genesis.tools.builtin.file_reader import ReadFileTool
from genesis.tools.builtin.file_writer import WriteFileTool
from genesis.tools.builtin.knowledge_synthesizer import KnowledgeSynthesizerTool
from genesis.tools.builtin.python_eval import PythonEvalTool
from genesis.tools.permissions import PermissionManager

log = get_logger("tool_registry")



class ToolRegistry:
    """Manages registration, lookup, and schema export for tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        name = tool.name
        if name in self._tools:
            log.warning(f"Overwriting previously registered tool '{name}'")
        self._tools[name] = tool
        log.info(f"Registered tool '{name}' (permission: {tool.permission.value})")

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            log.info(f"Unregistered tool '{name}'")
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tool definitions."""
        return [tool.definition for tool in self._tools.values()]

    def to_ollama_tools(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """
        Format registered tools as Ollama / OpenAI function tool calling schemas.
        """
        if names is not None:
            name_set = set(names)
            return [
                tool.definition.to_ollama_tool()
                for name, tool in self._tools.items()
                if name in name_set
            ]
        return [tool.definition.to_ollama_tool() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def create_default_registry(permission_manager: PermissionManager | None = None) -> ToolRegistry:
    """Create and populate a ToolRegistry with all standard built-in tools."""
    pm = permission_manager or PermissionManager()
    registry = ToolRegistry()

    registry.register(ReadFileTool(permission_manager=pm))
    registry.register(WriteFileTool(permission_manager=pm))
    registry.register(ListDirectoryTool(permission_manager=pm))
    registry.register(CalculatorTool())
    registry.register(PythonEvalTool())
    registry.register(CodeSynthesizerTool())
    registry.register(KnowledgeSynthesizerTool())

    return registry



# Global default registry instance
default_registry = create_default_registry()
