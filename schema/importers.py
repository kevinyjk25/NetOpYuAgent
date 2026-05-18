"""
schema/importers.py
-------------------
Convert various external formats into ArgSchema:

  from_dict_metadata(name, meta)        — current ToolLoader format
  from_mcp_input_schema(name, mcp_spec) — MCP `inputSchema` (JSON Schema)
  from_openapi_operation(name, op_spec) — OpenAPI 3 operation (parameters + requestBody)

These are the THREE bridges that make the unified schema framework actually
unified. Once data is in ArgSchema form, validate/coerce/render are identical
regardless of source.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .types import ArgField, ArgSchema, FieldType, VALID_TYPES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Dict-style metadata (the existing ToolLoader format)
# ---------------------------------------------------------------------------
# Existing format:
#   {
#     "description": "...",
#     "parameters": {"timeout": "timeout in seconds", "host": "..."},
#     "example": {"timeout": 3, "host": "10.0.0.5"},
#     "examples": [...],
#     "returns_large": False,
#     "returns": "...",
#   }
#
# This format has NO type info — only descriptions. We:
#   - Type-infer from `example` / `examples[0]` if present
#   - Default to "any" otherwise
#   - Treat all parameters as optional unless name appears in
#     metadata['required'] (a forward-compat key not used today)


def from_dict_metadata(tool_name: str, meta: dict[str, Any]) -> ArgSchema:
    """Convert dict-style metadata (the existing ToolLoader format) to ArgSchema.

    Type inference is done by scanning ALL provided examples (both `example`
    and `examples[]`). When a parameter appears in any example, the first
    type seen wins — but if subsequent examples have inconsistent types,
    the field is widened to "any" so coercion handles all variants.

    This is critical because some tools (like edit_device_config) accept
    multiple call shapes for the same parameter (e.g. `changes` may be a
    dict OR a list; `config_lines` may be a list OR a single string).
    """
    params      = meta.get("parameters") or {}
    primary     = meta.get("example") or {}
    extra       = meta.get("examples") or []
    all_examples = ([primary] if primary else []) + [e for e in extra if isinstance(e, dict)]
    required    = set(meta.get("required") or [])
    description = meta.get("description") or ""

    fields: dict[str, ArgField] = {}
    for pname, pdesc in params.items():
        # Scan all examples for this parameter
        seen_types: set[FieldType] = set()
        sample_values: list[Any] = []
        for ex in all_examples:
            if pname in ex:
                t = _infer_type(ex[pname])
                if t != "any":
                    seen_types.add(t)
                sample_values.append(ex[pname])

        if len(seen_types) == 1:
            inferred_type: FieldType = seen_types.pop()
        elif len(seen_types) > 1:
            # Multiple types seen — widen to "any" so coercer handles each shape
            inferred_type = "any"
        else:
            inferred_type = "any"

        fields[pname] = ArgField(
            name        = pname,
            type        = inferred_type,
            description = pdesc if isinstance(pdesc, str) else str(pdesc),
            required    = pname in required,
            examples    = sample_values,
        )

    return ArgSchema(
        tool_name   = tool_name,
        fields      = fields,
        description = description,
        examples    = all_examples,
    )


# ---------------------------------------------------------------------------
# 2. MCP inputSchema (JSON Schema directly)
# ---------------------------------------------------------------------------
# MCP tool spec shape:
#   {
#     "name": "tool_name",
#     "description": "...",
#     "inputSchema": {
#       "type": "object",
#       "properties": {
#         "field_a": {"type": "string", "description": "..."},
#         "field_b": {"type": "integer", "default": 5},
#       },
#       "required": ["field_a"],
#     }
#   }
#
# This is the common JSON-Schema-object shape. We walk it directly.


def from_mcp_input_schema(tool_name: str, spec: dict[str, Any]) -> ArgSchema:
    """Convert an MCP tool spec (with .inputSchema field) to ArgSchema.

    Accepts either:
      - A full MCP spec dict (with `name`, `description`, `inputSchema`)
      - The inputSchema object directly (when caller already extracted it)
    """
    description = spec.get("description") or ""

    # Accept both wrapped and unwrapped shapes
    if "inputSchema" in spec:
        json_schema = spec["inputSchema"] or {}
    else:
        json_schema = spec

    fields = _properties_to_fields(
        json_schema.get("properties", {}),
        required=set(json_schema.get("required", [])),
    )

    return ArgSchema(
        tool_name   = tool_name,
        fields      = fields,
        description = description,
    )


# ---------------------------------------------------------------------------
# 3. OpenAPI 3 operation (parameters + requestBody)
# ---------------------------------------------------------------------------
# OpenAPI shape:
#   {
#     "operationId": "do_thing",
#     "parameters": [
#       {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
#       {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
#     ],
#     "requestBody": {
#       "content": {
#         "application/json": {
#           "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
#         }
#       }
#     }
#   }


def from_openapi_operation(tool_name: str, op: dict[str, Any]) -> ArgSchema:
    """Convert an OpenAPI 3 operation object to ArgSchema.

    Both query/path parameters and JSON request body are flattened into
    a single field set — matching the agent's tool-call convention where
    everything is one args dict.
    """
    description = op.get("description") or op.get("summary") or ""
    fields: dict[str, ArgField] = {}

    # Parameters (query / path / header)
    for p in op.get("parameters", []) or []:
        pname    = p.get("name", "")
        if not pname:
            continue
        pschema  = p.get("schema") or {}
        ptype    = _normalise_type(pschema.get("type") or "any")
        required = bool(p.get("required", False))
        fields[pname] = ArgField(
            name        = pname,
            type        = ptype,
            description = p.get("description") or "",
            required    = required,
            default     = pschema.get("default"),
            enum        = pschema.get("enum"),
        )

    # Request body (application/json)
    rb = op.get("requestBody") or {}
    content = rb.get("content") or {}
    json_part = content.get("application/json") or {}
    body_schema = json_part.get("schema") or {}
    if body_schema.get("type") == "object" or body_schema.get("properties"):
        body_required = set(body_schema.get("required", []))
        body_fields = _properties_to_fields(
            body_schema.get("properties", {}),
            required=body_required,
        )
        # Merge body fields into top-level (request body is the args dict)
        for fname, fschema in body_fields.items():
            if fname not in fields:
                fields[fname] = fschema

    return ArgSchema(
        tool_name   = tool_name,
        fields      = fields,
        description = description,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _properties_to_fields(
    properties: dict[str, Any],
    required:   set[str],
) -> dict[str, ArgField]:
    """Convert a JSON-Schema `properties` dict into ArgField map. Recurses for
    nested object/array schemas."""
    fields: dict[str, ArgField] = {}
    for fname, fschema in (properties or {}).items():
        if not isinstance(fschema, dict):
            continue
        ftype = _normalise_type(fschema.get("type") or "any")
        item_field: Optional[ArgField] = None
        sub_props: Optional[dict[str, ArgField]] = None

        if ftype == "array" and "items" in fschema:
            items_schema = fschema["items"]
            item_type = _normalise_type(items_schema.get("type") or "any") if isinstance(items_schema, dict) else "any"
            sub_sub: Optional[dict[str, ArgField]] = None
            if item_type == "object" and isinstance(items_schema, dict):
                sub_sub = _properties_to_fields(
                    items_schema.get("properties", {}),
                    required=set(items_schema.get("required", [])),
                )
            item_field = ArgField(
                name        = "",       # element field has no name
                type        = item_type,
                description = items_schema.get("description", "") if isinstance(items_schema, dict) else "",
                properties  = sub_sub,
            )

        if ftype == "object" and "properties" in fschema:
            sub_props = _properties_to_fields(
                fschema.get("properties", {}),
                required=set(fschema.get("required", [])),
            )

        fields[fname] = ArgField(
            name        = fname,
            type        = ftype,
            description = fschema.get("description") or "",
            required    = fname in required,
            default     = fschema.get("default"),
            enum        = fschema.get("enum"),
            items       = item_field,
            properties  = sub_props,
            additional_properties = fschema.get("additionalProperties", True),
            examples    = fschema.get("examples") or
                          ([fschema["example"]] if "example" in fschema else []),
        )
    return fields


def _normalise_type(t: Any) -> FieldType:
    """Normalise a JSON Schema type field to our supported subset.
    JSON Schema allows arrays of types (e.g. ['string','null']); we pick the
    first non-null type and warn."""
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if non_null:
            return _normalise_type(non_null[0])
        return "null"
    if not isinstance(t, str):
        return "any"
    if t in VALID_TYPES:
        return t
    return "any"


def _infer_type(value: Any) -> FieldType:
    """Best-effort type inference from a single example value (used by
    from_dict_metadata since old metadata has no explicit types)."""
    if value is None:
        return "any"
    if isinstance(value, bool):     # check bool BEFORE int (bool is subclass of int)
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "any"
