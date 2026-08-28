"""Small, non-Turing-complete expression renderer for compiled L0 contracts."""

from __future__ import annotations

import re
from typing import Any


class ExpressionError(ValueError):
    pass


_EXPRESSION = re.compile(r"\$\{\s*([^{}]+?)\s*\}")
_PATH = re.compile(
    r"^(arguments|preflight|plan|intent|verification)(?:\.[A-Za-z0-9_-]+)+$",
)


def _lookup(path: str, context: dict[str, Any]) -> Any:
    if not _PATH.fullmatch(path):
        raise ExpressionError(f"unsupported L0 expression {path!r}")
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ExpressionError(f"unresolved L0 expression {path!r}")
        value = value[part]
    return value


def render_template(value: Any, context: dict[str, Any]) -> Any:
    """Render whitelisted paths; no calls, indexing, operators, or code exist."""
    if isinstance(value, dict):
        return {key: render_template(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [render_template(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(render_template(item, context) for item in value)
    if not isinstance(value, str):
        return value
    matches = list(_EXPRESSION.finditer(value))
    if not matches:
        return value
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return _lookup(matches[0].group(1), context)
    rendered = value
    for match in reversed(matches):
        replacement = _lookup(match.group(1), context)
        if isinstance(replacement, (dict, list, tuple)):
            raise ExpressionError("structured values cannot be interpolated into a string")
        rendered = rendered[:match.start()] + str(replacement) + rendered[match.end():]
    return rendered


def render_effect_request(
    request: dict[str, Any], arguments: dict[str, Any],
) -> dict[str, Any]:
    """Render a provider request while omitting absent direct optional arguments."""
    output: dict[str, Any] = {}
    for name, value in request.items():
        direct = _EXPRESSION.fullmatch(value) if isinstance(value, str) else None
        if direct and direct.group(1).startswith("arguments."):
            argument = direct.group(1).split(".", 1)[1]
            if argument not in arguments:
                continue
        output[name] = render_template(value, {"arguments": arguments})
    return output
