"""
schema/prompt.py
----------------
Render an ArgSchema as a compact human/LLM-readable string for inclusion
in tool descriptions in the system prompt.

Two render styles:
  - "compact" (default): single-line, ideal for top-K tool listings
       Args: name (string, required), timeout (integer, default 5)
  - "verbose": multi-line, with descriptions and examples
       Args:
         - name (string, required): Service name to query
         - timeout (integer, default 5): Seconds to wait
       Example: {"name": "auth-svc", "timeout": 10}

Both styles produce stable output ordering (required first, then declared order).
"""
from __future__ import annotations

import json
from typing import Optional

from .types import ArgField, ArgSchema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_args_for_prompt(
    schema:    ArgSchema,
    style:     str = "compact",
    *,
    max_chars: Optional[int] = None,
) -> str:
    """Format `schema` as a string suitable for injection into a tool
    description block in the system prompt.

    Args:
        schema:    the ArgSchema to render
        style:     "compact" (default) or "verbose"
        max_chars: optional truncation cap; suffix '…' appended if truncated.
                   Useful when the prompt-builder is enforcing a token budget.
    """
    if not schema.fields:
        return "Args: (no arguments)"

    if style == "verbose":
        result = _render_verbose(schema)
    else:
        result = _render_compact(schema)

    if max_chars and len(result) > max_chars:
        return result[:max_chars - 1] + "…"
    return result


# ---------------------------------------------------------------------------
# Compact: single line
# ---------------------------------------------------------------------------

def _render_compact(schema: ArgSchema) -> str:
    """E.g. `Args: id (string, required), limit (integer, default 10)`"""
    parts: list[str] = []
    # Required first, then optional, both in declaration order
    for name, fld in _ordered(schema):
        chunks = [name]
        chunks.append(f"({fld.type}")
        if fld.required:
            chunks[-1] += ", required"
        elif fld.default is not None:
            chunks[-1] += f", default {_render_default(fld.default)}"
        if fld.enum:
            chunks[-1] += f", one of {fld.enum}"
        chunks[-1] += ")"
        parts.append(" ".join(chunks))
    return "Args: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Verbose: multi-line with descriptions
# ---------------------------------------------------------------------------

def _render_verbose(schema: ArgSchema) -> str:
    lines = ["Args:"]
    for name, fld in _ordered(schema):
        spec = fld.type
        if fld.required:
            spec += ", required"
        elif fld.default is not None:
            spec += f", default {_render_default(fld.default)}"
        if fld.enum:
            spec += f", one of {fld.enum}"
        line = f"  - {name} ({spec})"
        if fld.description:
            line += f": {fld.description}"
        lines.append(line)

        # Show item type for arrays, sub-fields for objects
        if fld.type == "array" and fld.items:
            lines.append(f"    items: {fld.items.type}")
        elif fld.type == "object" and fld.properties:
            for sub_name, sub_fld in fld.properties.items():
                sub_spec = sub_fld.type + (", required" if sub_fld.required else "")
                sub_line = f"    .{sub_name} ({sub_spec})"
                if sub_fld.description:
                    sub_line += f": {sub_fld.description}"
                lines.append(sub_line)

    if schema.examples:
        ex = schema.examples[0]
        lines.append(f"Example: {json.dumps(ex, ensure_ascii=False)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordered(schema: ArgSchema) -> list[tuple[str, ArgField]]:
    """Required fields first (declaration order), then optional fields."""
    req = [(n, f) for n, f in schema.fields.items() if f.required]
    opt = [(n, f) for n, f in schema.fields.items() if not f.required]
    return req + opt


def _render_default(val) -> str:
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)
