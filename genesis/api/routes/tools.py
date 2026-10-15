"""
API routes for Tool inspection and execution.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from genesis.tools.executor import ToolExecutor
from genesis.tools.registry import default_registry

router = APIRouter(prefix="/tools", tags=["tools"])
executor = ToolExecutor()


class ToolExecuteRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")


@router.get("")
async def list_tools():
    """List all registered tools, their parameters, and permission levels."""
    tools = default_registry.list_tools()
    return {
        "count": len(tools),
        "tools": [t.to_dict() for t in tools],
    }


@router.post("/execute")
async def execute_tool(req: ToolExecuteRequest):
    """Safely execute a tool with the provided arguments."""
    tool = default_registry.get(req.tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found.")

    res = await executor.execute_async(tool, **req.arguments)
    return res.to_dict()
