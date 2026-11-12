"""
lumped.fom.expr

Translate BMS3-toolbox parameter expressions (MATLAB syntax in x and T) into
callables for NumPy or PyBaMM, without eval.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Callable

import numpy as np
import pybamm

# Functions an expression may call, by backend.
_NUMPY = {"exp": np.exp, "sqrt": np.sqrt, "tanh": np.tanh, "log": np.log, "abs": np.abs}
_PYBAMM = {"exp": pybamm.exp, "sqrt": pybamm.sqrt, "tanh": pybamm.tanh, "log": pybamm.log, "abs": pybamm.AbsoluteValue}
FUNCTIONS = tuple(_NUMPY)

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}


class ExpressionError(ValueError):
    """An expression outside the supported grammar."""


def to_python(body: str) -> str:
    """
    MATLAB expression text as Python expression text.

    Elementwise operators map to their scalar forms; x(:) and T(:) become x and T.
    """
    s = body.strip()
    s = re.sub(r"\b([xT])\(:\)", r"\1", s)
    s = s.replace(".*", "*").replace("./", "/").replace(".^", "**").replace("^", "**")
    return s


def parse(body: str) -> ast.Expression:
    """Parse and validate an expression; raise ExpressionError outside the grammar."""
    text = to_python(body)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as err:
        raise ExpressionError(f"Cannot parse expression '{body}': {err.msg}.") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop)):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(node, ast.Name) and node.id in ("x", "T") + FUNCTIONS:
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS \
                and len(node.args) == 1 and not node.keywords:
            continue
        raise ExpressionError(
            f"Expression '{body}' uses '{ast.unparse(node)}'; expressions may use x, T, numbers, "
            f"+ - * / ^ and {', '.join(FUNCTIONS)}."
        )
    return tree


def compile_expression(body: str, backend: str = "numpy") -> Callable[[Any, Any], Any]:
    """
    Callable f(x, T) for an expression body.

    Parameters
    ----------
    body : str
        Expression in x and T (MATLAB syntax, without the @(x,T) header).
    backend : {'numpy', 'pybamm'}
        Function set used for exp, sqrt, tanh, log and abs.

    Returns
    -------
    f : callable
        f(x, T) evaluating the expression with the chosen backend.
    """
    tree = parse(body)
    funcs = _NUMPY if backend == "numpy" else _PYBAMM

    def walk(node, env):
        if isinstance(node, ast.Expression):
            return walk(node.body, env)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            v = walk(node.operand, env)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](walk(node.left, env), walk(node.right, env))
        if isinstance(node, ast.Call):
            return funcs[node.func.id](walk(node.args[0], env))
        raise ExpressionError(f"Unexpected node {ast.dump(node)}.")  # parse() admits none

    return lambda x, T: walk(tree, {"x": x, "T": T})
