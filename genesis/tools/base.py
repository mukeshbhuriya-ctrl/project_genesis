"""
Core tool models, definitions, permissions, and base classes for Project Genesis.
"""

from __future__ import annotations

import inspect
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolPermission(str, Enum):
    """Permission level governing tool execution safety."""
    READ_ONLY = "read_only"      # Safe read operations, math, queries
    MUTATING = "mutating"        # Workspace file writes, memory updates
    DANGEROUS = "dangerous"      # Shell commands, system configuration


@dataclass
class ToolParameter:
    """Specification of a parameter accepted by a tool."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None

    def to_schema_property(self) -> dict[str, Any]:
        """Convert to JSON schema property descriptor."""
        prop: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            prop["enum"] = self.enum
        if self.default is not None:
            prop["default"] = self.default
        return prop


@dataclass
class ToolDefinition:
    """Complete specification of a tool."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    permission: ToolPermission = ToolPermission.READ_ONLY

    def to_ollama_tool(self) -> dict[str, Any]:
        """Format as Ollama / OpenAI compatible tool function specification."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            properties[p.name] = p.to_schema_property()
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert definition to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
        }


@dataclass
class ToolResult:
    """Result returned after tool execution."""
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """Abstract base class for all Genesis tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition metadata."""
        pass

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def permission(self) -> ToolPermission:
        return self.definition.permission

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Synchronously execute the tool with given arguments."""
        pass

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        """Asynchronously execute the tool. Defaults to calling synchronous execute."""
        return self.execute(**kwargs)
