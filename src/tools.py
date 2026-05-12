from __future__ import annotations

import ast
import operator as op
from typing import Any, Dict


# ----------------------------
# Errors
# ----------------------------

class ToolError(Exception):
    """Raised when a tool call is invalid or unsafe."""


# ----------------------------
# Tool: math_eval (safe math only)
# ----------------------------

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # For Python <3.8 compatibility (not needed for 3.10, but harmless)
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)  # type: ignore[attr-defined]

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        val = _eval_node(node.operand)
        return float(_ALLOWED_UNARYOPS[type(node.op)](val))

    raise ToolError("Unsafe or unsupported expression. Allowed: numbers and + - * / // % ** ( )")


def math_eval(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a simple arithmetic expression.
    Example: "2*(3+4) - 5/2"
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression must be a non-empty string")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError("Invalid math syntax")

    value = _eval_node(tree)
    return {"ok": True, "expression": expression, "value": value}


# ----------------------------
# Registry
# ----------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "math_eval": {
        "fn": math_eval,
        "description": "Evaluate safe arithmetic. args: {\"expression\": \"2+2*3\"}",
    },
}
