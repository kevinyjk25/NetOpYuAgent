"""
schema/ollama_export.py
-----------------------
Convert ArgSchema → Ollama / OpenAI native-tools JSON schema.

The Ollama Chat API (≥ 0.4) supports an OpenAI-style `tools` parameter:

    tools: [
      {
        "type": "function",
        "function": {
          "name": "edit_device_config",
          "description": "...",
          "parameters": {
            "type": "object",
            "properties": {
              "device_id":     {"type": "string", "description": "..."},
              "config_lines":  {"type": "array",  "items": {"type": "string"}},
              "reason":        {"type": "string"},
            },
            "required": ["device_id", "config_lines"]
          }
        }
      },
      ...
    ]

When this is passed alongside `messages`, the model emits a structured
`tool_calls` field in its response instead of (or in addition to) free-text.
Each tool_call has `function.name` + `function.arguments` (JSON-encoded).

This eliminates the "model put the device_id in the wrong field" / "model
forgot a quote and the JSON didn't parse" failure modes that plague the
[TOOL:name] {...} text protocol — args come back as a real dict, not as
a JSON blob the model had to type by hand.

Why a separate module: keeping the conversion out of registry.py keeps
the registry dependency-free (it doesn't need to know about Ollama or
OpenAI), and lets us add more exporters (Anthropic, vLLM, etc.) without
bloating the core.

Lives in `schema/` (not `integrations/clients/`) because it's a SCHEMA
transformation, not a client. Per ARCHITECTURE.md's separation: schema
defines shapes, integrations wire shapes to external systems.
"""
from __future__ import annotations

from typing import Any

from schema.types import ArgField, ArgSchema


def _field_to_json_schema(field: ArgField) -> dict[str, Any]:
    """One ArgField → one property in a JSON Schema `properties` map.

    Handles nested arrays (via `items`) and nested objects (via `properties`).
    Maps `type="any"` → no type constraint (omitted from the output), which
    is how JSON Schema represents an unconstrained value.
    """
    out: dict[str, Any] = {}
    if field.type and field.type != "any":
        out["type"] = field.type
    if field.description:
        out["description"] = field.description
    if field.enum:
        out["enum"] = list(field.enum)
    if field.examples:
        # JSON Schema 2020-12 supports `examples`; older drafts ignore it.
        # Ollama passes the spec through to the model verbatim, so the model
        # gets useful demonstration values either way.
        out["examples"] = list(field.examples)

    # Arrays: descend into items
    if field.type == "array" and field.items is not None:
        out["items"] = _field_to_json_schema(field.items)

    # Objects: descend into properties
    if field.type == "object" and field.properties:
        nested_props: dict[str, dict[str, Any]] = {}
        nested_required: list[str] = []
        for sub_name, sub_field in field.properties.items():
            nested_props[sub_name] = _field_to_json_schema(sub_field)
            if sub_field.required:
                nested_required.append(sub_name)
        out["properties"] = nested_props
        if nested_required:
            out["required"] = nested_required
        # additional_properties=False means "reject unknown keys" — pass
        # through so the model sees the constraint.
        if not field.additional_properties:
            out["additionalProperties"] = False

    return out


def arg_schema_to_ollama_tool(schema: ArgSchema) -> dict[str, Any]:
    """One ArgSchema → one entry in Ollama's `tools` array."""
    properties: dict[str, dict[str, Any]] = {}
    for fname, fspec in schema.fields.items():
        properties[fname] = _field_to_json_schema(fspec)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if schema.required_names:
        parameters["required"] = schema.required_names

    return {
        "type": "function",
        "function": {
            "name":        schema.tool_name,
            "description": schema.description or f"Tool: {schema.tool_name}",
            "parameters":  parameters,
        },
    }


def export_for_ollama(
    schemas: list[ArgSchema],
    *,
    allowed_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a batch of ArgSchema → Ollama tools array.

    If `allowed_names` is provided, only schemas whose tool_name is in the
    set are exported. This lets the LLM engine ship only the top-K retrieved
    tools per turn (matching the text-protocol's prompt-budget approach)
    rather than every registered tool (which can be 40+).
    """
    out: list[dict[str, Any]] = []
    for s in schemas:
        if allowed_names is not None and s.tool_name not in allowed_names:
            continue
        out.append(arg_schema_to_ollama_tool(s))
    return out
