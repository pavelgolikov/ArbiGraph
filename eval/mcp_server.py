import ast
import json
import math
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field


mcp = FastMCP("ArbiGraph Calculator", log_level="WARNING")

# Functions callable from an expression, either bare (`sqrt(2)`) or through the
# `math` module (`math.sqrt(2)`).
ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "pow": pow,
    "math.pow": math.pow,
}


def _resolve_function_name(func_node):
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name):
        if func_node.value.id == "math":
            if func_node.attr == "pow":
                return "math.pow"
            return func_node.attr
    return None


def safe_calc(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    try:
        node = ast.parse(expression, mode="eval").body

        def _eval(n):
            if isinstance(n, ast.Constant):
                return n.value
            if isinstance(n, ast.Call):
                name = _resolve_function_name(n.func)
                if name not in ALLOWED_FUNCTIONS:
                    raise TypeError(f"Unsupported function: {ast.unparse(n.func)}")
                if n.keywords:
                    raise TypeError(f"Keyword arguments are not supported for {name}")
                return ALLOWED_FUNCTIONS[name](*(_eval(arg) for arg in n.args))
            if isinstance(n, ast.UnaryOp):
                operand = _eval(n.operand)
                if isinstance(n.op, ast.USub):
                    return -operand
                if isinstance(n.op, ast.UAdd):
                    return +operand
                raise TypeError(f"Unsupported unary op: {type(n.op)}")
            if isinstance(n, ast.BinOp):
                if (
                    isinstance(n.op, ast.Mod)
                    and isinstance(n.left, ast.BinOp)
                    and isinstance(n.left.op, ast.Pow)
                ):
                    base, exponent, modulus = _eval(n.left.left), _eval(n.left.right), _eval(n.right)
                    if all(isinstance(value, int) for value in (base, exponent, modulus)) and exponent >= 0:
                        # Avoid constructing the full integer for (base ** exponent) % modulus.
                        return pow(base, exponent, modulus)
                    return (base ** exponent) % modulus
                left, right = _eval(n.left), _eval(n.right)
                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                if isinstance(n.op, ast.Div):
                    return left / right
                if isinstance(n.op, ast.FloorDiv):
                    return left // right
                if isinstance(n.op, ast.Mod):
                    return left % right
                if isinstance(n.op, ast.Pow):
                    return left ** right
                raise TypeError(f"Unsupported binop: {type(n.op)}")
            raise TypeError(f"Unsupported node: {type(n)}")

        return str(_eval(node))
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def calculator(
    expressions: Annotated[
        list[str],
        Field(
            description=(
                "Mathematical expressions to evaluate in order, e.g. "
                "['17 * 43 + 5', '100 // 7', 'round(sqrt(2), 3)']. "
                "Operators: + - * / // % **. Functions: sqrt, abs, round, pow. "
                "Use pow(base, exponent, modulus) for efficient modular powers, "
                "or pow(base, -1, modulus) for modular inverses."
            )
        ),
    ],
) -> str:
    """Evaluate arithmetic expressions in order using + - * / // % **, sqrt, abs, round, and pow.

    Use pow(base, exponent, modulus) for modular exponentiation, e.g. pow(3, 998244351, 998244353).
    Use pow(base, -1, modulus) for a modular inverse when one exists.
    Integer powers followed by % are also evaluated efficiently. Batch independent expressions in one call.
    """
    results = []
    for expression_text in expressions:
        if not isinstance(expression_text, str):
            results.append(f"Error: expression must be a string, got {type(expression_text).__name__}")
        else:
            results.append(safe_calc(expression_text))
    return json.dumps({"results": results})


if __name__ == "__main__":
    mcp.run()
