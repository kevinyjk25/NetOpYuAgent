"""
schema/validator.py
-------------------
validate_and_coerce(schema, args) → CoerceResult

Walks an ArgSchema and an args dict; for each field:
  - check presence (required)
  - check type     (with lossless coercion attempts)
  - check enum     (if set)
  - recurse into items (arrays) and properties (nested objects)
  - apply defaults (missing optional fields)

Coercion ladder (most permissive last):
  exact-type-match
    → ints from numeric strings ("3" → 3)
    → strings from primitives (3 → "3")
    → arrays from single values ("x" → ["x"]) when target is array
    → arrays of dicts → dict via shallow merge (useful for LLM list-of-dict pattern)
    → dicts from list-of-strings → {add: [...]} (the mock_tools `changes` case)

The goal is "accept what the LLM produces, normalise to canonical shape, warn
about which coercions happened so observability stays high."
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .types import ArgField, ArgSchema, CoerceResult, FieldType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_and_coerce(
    schema: ArgSchema,
    args:   Optional[dict[str, Any]],
) -> CoerceResult:
    """Validate `args` against `schema`, returning a normalised dict.

    Args:
        schema: the ArgSchema for the target tool
        args:   the arguments produced by the LLM (or any caller). May be None,
                empty dict, or partially specified.

    Returns:
        CoerceResult with .ok telling the caller whether to proceed.
        On ok=False the caller should pass .errors back to the LLM as a
        structured error so the LLM can self-correct on the next turn.
    """
    args = dict(args or {})
    errors:   list[str] = []
    warnings: list[str] = []
    out:      dict[str, Any] = {}

    # 1. Check unknown keys (only emit a warning unless schema is strict and
    #    we have schema-level config to drop them).
    known = set(schema.fields.keys())
    for k in list(args.keys()):
        if k not in known:
            warnings.append(f"unknown argument {k!r} (kept as-is)")
            out[k] = args[k]

    # 2. For each declared field
    for name, field in schema.fields.items():
        present = name in args
        if not present:
            if field.required:
                errors.append(f"missing required argument {name!r} (expected {field.type})")
                continue
            if field.default is not None:
                out[name] = field.default
            continue

        raw = args[name]
        coerced, sub_errors, sub_warnings = _coerce_value(raw, field, path=name)
        if sub_errors:
            errors.extend(sub_errors)
            continue
        warnings.extend(sub_warnings)
        out[name] = coerced

    return CoerceResult(
        ok=(not errors),
        args=out,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Recursive coercion
# ---------------------------------------------------------------------------

def _coerce_value(
    raw:   Any,
    field: ArgField,
    path:  str,
) -> tuple[Any, list[str], list[str]]:
    """Coerce a single value to match `field`. Returns (value, errors, warnings)."""
    # any-type accepts everything verbatim
    if field.type == "any":
        return raw, [], []

    # null
    if field.type == "null":
        if raw is None:
            return None, [], []
        return None, [f"{path!r}: expected null, got {type(raw).__name__}"], []

    # boolean
    if field.type == "boolean":
        if isinstance(raw, bool):
            return raw, [], []
        if isinstance(raw, str) and raw.lower() in ("true", "false", "yes", "no"):
            return raw.lower() in ("true", "yes"), [], [f"{path!r}: coerced str→bool"]
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw), [], [f"{path!r}: coerced num→bool"]
        return None, [f"{path!r}: expected boolean, got {type(raw).__name__}"], []

    # integer / number
    if field.type == "integer":
        if isinstance(raw, bool):  # bool is subclass of int — reject explicitly
            return None, [f"{path!r}: expected integer, got boolean"], []
        if isinstance(raw, int):
            v = raw
        elif isinstance(raw, float) and raw.is_integer():
            v = int(raw); _w = [f"{path!r}: coerced float→int"]
            return _check_enum(v, field, path) or (v, [], _w)
        elif isinstance(raw, str):
            try:
                v = int(raw, 10)
                return _check_enum(v, field, path) or (v, [], [f"{path!r}: coerced str→int"])
            except ValueError:
                return None, [f"{path!r}: expected integer, got non-numeric string {raw!r}"], []
        else:
            return None, [f"{path!r}: expected integer, got {type(raw).__name__}"], []
        return _check_enum(v, field, path) or (v, [], [])

    if field.type == "number":
        if isinstance(raw, bool):
            return None, [f"{path!r}: expected number, got boolean"], []
        if isinstance(raw, (int, float)):
            return _check_enum(raw, field, path) or (raw, [], [])
        if isinstance(raw, str):
            try:
                v = float(raw)
                return _check_enum(v, field, path) or (v, [], [f"{path!r}: coerced str→float"])
            except ValueError:
                return None, [f"{path!r}: expected number, got non-numeric string {raw!r}"], []
        return None, [f"{path!r}: expected number, got {type(raw).__name__}"], []

    # string
    if field.type == "string":
        if isinstance(raw, str):
            return _check_enum(raw, field, path) or (raw, [], [])
        if isinstance(raw, (int, float, bool)):
            v = str(raw)
            return _check_enum(v, field, path) or (v, [], [f"{path!r}: coerced {type(raw).__name__}→str"])
        return None, [f"{path!r}: expected string, got {type(raw).__name__}"], []

    # array
    if field.type == "array":
        if not isinstance(raw, list):
            # Lossless single-value-to-list coercion
            v = [raw]
            warning = [f"{path!r}: wrapped single value into list"]
            errors:   list[str] = []
            warnings: list[str] = warning
        else:
            v = list(raw)
            errors, warnings = [], []
        # Recurse into items
        if field.items:
            new_v = []
            for i, elem in enumerate(v):
                ev, e_err, e_warn = _coerce_value(elem, field.items, f"{path}[{i}]")
                errors.extend(e_err); warnings.extend(e_warn)
                if not e_err:
                    new_v.append(ev)
            v = new_v
        return v, errors, warnings

    # object
    if field.type == "object":
        if isinstance(raw, dict):
            v = dict(raw)
            errors:   list[str] = []
            warnings: list[str] = []
        elif isinstance(raw, list):
            # Two LLM-friendly fallbacks — both flagged as warnings:
            # 1) list of dicts → shallow-merge into single dict
            # 2) list of strings → {"add": [strings]}
            warnings = [f"{path!r}: coerced list→object"]
            errors:   list[str] = []
            if all(isinstance(x, dict) for x in raw):
                merged: dict[str, Any] = {}
                for d in raw:
                    for k, val in d.items():
                        if k in merged and isinstance(merged[k], list) and isinstance(val, list):
                            merged[k].extend(val)
                        elif k in merged and isinstance(merged[k], list):
                            merged[k].append(val)
                        else:
                            merged[k] = val
                v = merged
            elif all(isinstance(x, str) for x in raw):
                v = {"add": list(raw)}
            else:
                # Mixed list — best effort
                v = {"add": [str(x) if not isinstance(x, dict) else x for x in raw]}
        elif isinstance(raw, str):
            # Single string → {"add": [it]} so LLM "raw IOS line" usage works
            v = {"add": [raw]}
            warnings = [f"{path!r}: coerced str→object"]
            errors = []
        else:
            return None, [f"{path!r}: expected object, got {type(raw).__name__}"], []

        # Recurse into declared properties (if schema specifies them)
        if field.properties:
            new_v: dict[str, Any] = {}
            # Validate each declared sub-field
            for sub_name, sub_field in field.properties.items():
                if sub_name in v:
                    sub_v, sub_err, sub_warn = _coerce_value(
                        v[sub_name], sub_field, f"{path}.{sub_name}",
                    )
                    errors.extend(sub_err); warnings.extend(sub_warn)
                    if not sub_err:
                        new_v[sub_name] = sub_v
                elif sub_field.required:
                    errors.append(f"{path!r}.{sub_name}: missing required sub-field")
                elif sub_field.default is not None:
                    new_v[sub_name] = sub_field.default
            # Keep extras unless schema is strict
            if field.additional_properties:
                for k, val in v.items():
                    if k not in field.properties:
                        new_v[k] = val
            v = new_v
        return v, errors, warnings

    # Unknown type — pass through with a warning (forward compat)
    return raw, [], [f"{path!r}: unknown schema type {field.type!r}, passed through"]


def _check_enum(value: Any, field: ArgField, path: str):
    """Return (value, [], []) tuple if value is in enum, or (None, [error], []) if not.
    Returns None when no enum constraint; caller treats as no-op.
    """
    if not field.enum:
        return None
    if value in field.enum:
        return value, [], []
    return None, [f"{path!r}: value {value!r} not in allowed values {field.enum}"], []
