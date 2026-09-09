"""Expose the existing binding grammar before generation, not only at compile.

Authoring-size bounds are not runtime limits or permission/semantic guarantees.
The original compiler still checks types, aliases, scope and all expressions.
"""

from __future__ import annotations

from evaluation.structured_authoring import _obj

MAX_BLOCK_STEPS = 8


def omit_schema_titles(node):
    """Drop schema annotations only, never literal payloads or property names."""
    if not isinstance(node, dict):
        return node
    result = {k: v for k, v in node.items() if k != "title"}
    for key in ("properties", "$defs", "patternProperties", "dependentSchemas"):
        if key in result:
            result[key] = {k: omit_schema_titles(v) for k, v in result[key].items()}
    for key in ("items", "additionalProperties", "contains", "not", "if", "then", "else", "propertyNames"):
        if key in result:
            result[key] = omit_schema_titles(result[key])
    for key in ("oneOf", "anyOf", "allOf", "prefixItems"):
        if key in result:
            result[key] = [omit_schema_titles(v) for v in result[key]]
    return result


def tighten(schema, tool_names):
    definitions = schema["$defs"]
    reference = _obj({"kind": {"const": "reference"}, "source": {"type": "string"}, "pointer": {"type": "string"}})
    value = {"$ref": "#/$defs/BindingExpression"}
    definitions["BindingReference"] = reference
    definitions["BindingExpression"] = {"oneOf": [
        _obj({"kind": {"const": "literal"}, "value": {}}),
        {"$ref": "#/$defs/BindingReference"},
        _obj({"kind": {"const": "object"}, "fields": {"type": "object", "additionalProperties": value}}),
        _obj({"kind": {"const": "array"}, "items": {"type": "array", "items": value}}),
        _obj({"kind": {"const": "column_rows"}, "source": {"type": "string"}, "pointer": {"type": "string"},
              "fields": {"type": "array", "minItems": 1, "maxItems": 32, "uniqueItems": True,
                         "items": {"type": "string", "minLength": 1, "maxLength": 128}},
              "max_rows": {"type": "integer", "minimum": 1, "maximum": 256},
              "max_columns": {"type": "integer", "minimum": 1, "maximum": 128}})]}
    definitions["StructuredTreeRead"]["properties"]["arguments"] = value
    definitions["StructuredTreeRead"]["properties"]["tool"] = {"enum": list(tool_names)}
    definitions["StructuredTreeEffect"]["properties"]["arguments"] = value
    branch = definitions["StructuredTreeIf"]["properties"]
    branch["left"] = {"$ref": "#/$defs/BindingReference"}
    branch["equals"] = {"type": ["string", "number", "boolean", "null"]}
    for field in ("when_equal", "otherwise"):
        branch[field]["maxItems"] = MAX_BLOCK_STEPS
    tree = definitions["StructuredFlowTree"]["properties"]
    tree["steps"]["maxItems"] = MAX_BLOCK_STEPS
    tree["purpose"]["maxLength"] = 400
    definitions["StructuredTreeEnd"]["properties"]["explanation"]["maxLength"] = 400
    return schema
