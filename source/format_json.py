"""Render a topology or dataset as readable JSON.

A list of scalars stays on one line however long it is; a list holding objects takes
one element per line. Datasets are mostly long runs of numbers -- a node's input, its
output, every alternative in its bank -- and json.dumps(indent=2) puts each number on
its own line, which roughly doubles the file and buries the structure. Keeping those
runs inline halves it back, at about half a second per twenty-five megabytes.

Inlining every list regardless of what it holds is not an option: a dataset's samples
are a list of objects, so it would put the whole file on one line.
"""

from __future__ import annotations

import json
from typing import Any


def format_json(data: Any) -> str:
    parts: list[str] = []
    _render(data, 0, parts)
    parts.append("\n")
    return "".join(parts)


# JSON coerces a non-string key exactly as it renders that scalar on its own.
def _render_key(key: Any) -> str:
    return json.dumps(key if isinstance(key, str) else json.dumps(key))


def _render(node: Any, indent: int, parts: list[str], step: int = 2) -> None:
    if isinstance(node, dict):
        if not node:
            parts.append("{}")
            return
        pad = " " * (indent + step)
        parts.append("{\n")
        for position, (key, value) in enumerate(node.items()):
            parts.append(pad + _render_key(key) + ": ")
            _render(value, indent + step, parts, step)
            parts.append(",\n" if position < len(node) - 1 else "\n")
        parts.append(" " * indent + "}")
        return

    if isinstance(node, list):
        if not node:
            parts.append("[]")
            return
        if not any(isinstance(item, (dict, list)) for item in node):
            parts.append(json.dumps(node))
            return
        pad = " " * (indent + step)
        parts.append("[\n")
        for position, item in enumerate(node):
            parts.append(pad)
            _render(item, indent + step, parts, step)
            parts.append(",\n" if position < len(node) - 1 else "\n")
        parts.append(" " * indent + "]")
        return

    parts.append(json.dumps(node))
