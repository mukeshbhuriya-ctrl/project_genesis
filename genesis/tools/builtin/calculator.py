"""
Safe AST-based mathematical calculation tool.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult

# Safe operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Safe math functions
SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST expression using only whitelisted operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Literal constant '{node.value}' of type {type(node.value)} is not allowed.")

    elif isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Variable or identifier '{node.id}' is not recognized.")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operator '{op_type.__name__}' is not supported.")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        
        # Guard against absurd power explosions (e.g. 9999 ** 9999)
        if op_type is ast.Pow and (left > 10000 or right > 1000):
            raise ValueError("Exponent calculation exceeds safety threshold.")

        return SAFE_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unary operator '{op_type.__name__}' is not supported.")
        operand = _safe_eval(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct math function calls are permitted.")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not in the safe mathematical whitelist.")

        args = [_safe_eval(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)

    raise ValueError(f"Expression element '{type(node).__name__}' is not permitted.")


class CalculatorTool(BaseTool):
    """Safely evaluates mathematical expressions without eval()."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Evaluate mathematical expressions securely using AST parsing. Supports standard arithmetic (+, -, *, /, %, **), math constants (pi, e), and functions (sqrt, log, sin, cos, abs, round, min, max).",
            permission=ToolPermission.READ_ONLY,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate, e.g., 'sqrt(144) + 2 * pi' or '(1024 * 768 * 4) / 1024**2'.",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get("expression")
        if not expression or not expression.strip():
            return ToolResult(success=False, error="Parameter 'expression' is required.")

        clean_expr = expression.strip()
        try:
            parsed = ast.parse(clean_expr, mode="eval")
            result = _safe_eval(parsed.body)
            # Format nicely
            if isinstance(result, float) and result.is_integer():
                result = int(result)

            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": clean_expr},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Evaluation error for '{clean_expr}': {exc}",
            )
