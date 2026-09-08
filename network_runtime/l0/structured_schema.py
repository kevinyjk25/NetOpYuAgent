"""Bounded, lossless JSON data schemas for inactive structured bindings.

This profile is separate from historical ReadObjectSchema. Unsupported schema
keywords are errors, not discarded constraints. No external references, regex
evaluation, coercion, default insertion, provider calls or authority decisions.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

PROFILE = "netopyu.io/structured-data/v1"
MAX_BYTES = 1024 * 1024
MAX_DEPTH = 32
MAX_NODES = 16384
MAX_COLLECTION = 1024
MAX_SCHEMA_NODES = 512
MAX_SCHEMA_DEPTH = 16
TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
ANNOTATIONS = {"title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly"}
KEYWORDS = ANNOTATIONS | {
    "$schema", "$defs", "$ref", "type", "properties", "required", "additionalProperties",
    "items", "enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
}


class DataBindingError(ValueError):
    """Diagnostics carry paths and categories, never rejected payload contents."""

    def __init__(self, code: str, pointer: str, detail: str):
        self.code, self.pointer, self.detail = code, pointer, detail
        super().__init__(f"{code} at {pointer or '<root>'}: {detail}")

    def as_dict(self) -> dict:
        return {"code": self.code, "pointer": self.pointer, "detail": self.detail}


def join_pointer(base: str, key: str | int) -> str:
    return base + "/" + str(key).replace("~", "~0").replace("/", "~1")


def pointer_parts(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or len(pointer) > 4096 or (pointer and not pointer.startswith("/")):
        raise DataBindingError("invalid_pointer", "", "use an explicit JSON Pointer")
    if pointer == "":
        return []
    parts = pointer[1:].split("/")
    if len(parts) > MAX_DEPTH or any(re.search(r"~(?![01])", part) for part in parts):
        raise DataBindingError("invalid_pointer", pointer, "invalid escape or excessive depth")
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def snapshot_json(value: Any) -> Any:
    """Reject cycles/non-JSON/nonfinite/oversized data before serialization."""
    count, ancestors = 0, set()

    def visit(item, path, depth):
        nonlocal count
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise DataBindingError("data_budget", path, "JSON node/depth budget exceeded")
        if type(item) in {dict, list}:
            if id(item) in ancestors or len(item) > MAX_COLLECTION:
                raise DataBindingError("data_budget", path, "cyclic or oversized collection")
            ancestors.add(id(item))
            entries = item.items() if type(item) is dict else enumerate(item)
            for key, child in entries:
                if type(item) is dict and type(key) is not str:
                    raise DataBindingError("non_json_value", path, "object keys must be strings")
                visit(child, join_pointer(path, key), depth + 1)
            ancestors.remove(id(item))
        elif type(item) not in {str, int, float, bool, type(None)} or (type(item) is float and not math.isfinite(item)):
            raise DataBindingError("non_json_value", path, "finite JSON values required")

    visit(value, "", 0)
    try:
        wire = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(wire.encode("utf-8")) > MAX_BYTES:
            raise DataBindingError("data_budget", "", "JSON byte budget exceeded")
    except (UnicodeError, ValueError) as error:
        if isinstance(error, DataBindingError):
            raise
        raise DataBindingError("non_json_value", "", "invalid JSON text or number") from None
    return json.loads(wire)


def pointer_value(value: Any, pointer: str) -> Any:
    current = value
    for key in pointer_parts(pointer):
        if type(current) is dict and key in current:
            current = current[key]
        elif type(current) is list and re.fullmatch(r"0|[1-9][0-9]*", key) and int(key) < len(current):
            current = current[int(key)]
        else:
            raise DataBindingError("missing_source_value", pointer, "field or fixed array index is absent")
    return current


def schema_types(schema: dict) -> set[str]:
    kind = schema["type"]
    return set(kind) if isinstance(kind, list) else {kind}


def _resolve(schema: dict, root: dict) -> dict:
    for _ in range(MAX_SCHEMA_DEPTH + 1):
        if "$ref" not in schema:
            return schema
        schema = pointer_value(root, schema["$ref"][1:])
    raise DataBindingError("schema_budget", "", "reference depth exceeded")


def checked_schema(value: Any) -> dict:
    """Validate an explicit versioned subset; retain the original schema exactly."""
    root = snapshot_json(value)
    inventory, edges = {}, {}

    def walk(node, path, depth):
        if depth > MAX_SCHEMA_DEPTH or len(inventory) >= MAX_SCHEMA_NODES:
            raise DataBindingError("schema_budget", path, "schema node/depth budget exceeded")
        if not isinstance(node, dict):
            raise DataBindingError("untyped_schema", path, "an explicit typed schema is required")
        inventory[path], edges[path] = node, []
        for key in node:
            if key not in KEYWORDS:
                raise DataBindingError("unsupported_schema_keyword", join_pointer(path, key), "constraint is retained but unsupported")
        if "$schema" in node and node["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            raise DataBindingError("unsupported_schema_dialect", path, "only draft 2020-12 is accepted")
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/") or "%" in ref:
                raise DataBindingError("unsupported_schema_reference", join_pointer(path, "$ref"), "only explicit local schema pointers are accepted")
            pointer_parts(ref[1:])
            if set(node) - (ANNOTATIONS | {"$ref", "$schema", "$defs"}):
                raise DataBindingError("unsupported_ref_siblings", path, "reference assertion siblings require a later profile")
        else:
            kind = node.get("type")
            kinds = kind if isinstance(kind, list) else [kind]
            if not kinds or any(not isinstance(k, str) or k not in TYPES for k in kinds) or len(set(kinds)) != len(kinds):
                raise DataBindingError("untyped_schema", path, "explicit supported types required")
            if "array" in kinds and "items" not in node:
                raise DataBindingError("untyped_schema", path, "array items need an explicit schema")
        for key in ("properties", "$defs"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise DataBindingError("invalid_schema", join_pointer(path, key), "schema map required")
                for name, child in node[key].items():
                    at = join_pointer(join_pointer(path, key), name)
                    edges[path].append(at)
                    walk(child, at, depth + 1)
        for key in ("items", "additionalProperties"):
            if key in node and not (key == "additionalProperties" and type(node[key]) is bool):
                at = join_pointer(path, key)
                edges[path].append(at)
                walk(node[key], at, depth + 1)

    walk(root, "", 0)
    try:
        Draft202012Validator.check_schema(root)
    except SchemaError as error:
        path = ""
        for part in error.absolute_path:
            path = join_pointer(path, part)
        raise DataBindingError("invalid_json_schema", path, "declared JSON Schema is invalid") from None
    for path, node in inventory.items():
        if "$ref" in node:
            target = node["$ref"][1:]
            if target not in inventory:
                raise DataBindingError("unknown_schema_reference", path, "reference does not identify a declared schema")
            edges[path].append(target)
    visiting, heights = set(), {}

    def height(path):
        if path in visiting:
            raise DataBindingError("recursive_schema", path, "recursive schemas are outside this bounded profile")
        if path in heights:
            return heights[path]
        visiting.add(path)
        result = 1 + max((height(child) for child in edges[path]), default=0)
        visiting.remove(path)
        if result > MAX_SCHEMA_DEPTH:
            raise DataBindingError("schema_budget", path, "expanded schema depth exceeded")
        heights[path] = result
        return result

    height("")
    return root


def schema_location(root: dict, pointer: str) -> tuple[dict, bool]:
    """Locate a typed field; bool denotes presence implied by the source schema.

    Unknown/open-object keys and unions with ambiguous traversal are not guessed.
    Even a required path's value must still be validated at materialization time.
    """
    node, guaranteed = _resolve(root, root), True
    for key in pointer_parts(pointer):
        kinds = schema_types(node)
        containers = kinds & {"object", "array"}
        if len(containers) != 1:
            raise DataBindingError("ambiguous_schema_path", pointer, "path needs exactly one container type")
        guaranteed = guaranteed and kinds <= containers
        if "object" in containers:
            guaranteed = guaranteed and key in node.get("required", [])
            child = node.get("properties", {}).get(key, node.get("additionalProperties", True))
            if not isinstance(child, dict):
                raise DataBindingError("unknown_schema_path", pointer, "field is forbidden or has no declared type")
        else:
            if not re.fullmatch(r"0|[1-9][0-9]*", key) or int(key) >= node.get("maxItems", MAX_COLLECTION):
                raise DataBindingError("unknown_schema_path", pointer, "invalid or statically out-of-bounds index")
            guaranteed = guaranteed and int(key) < node.get("minItems", 0)
            child = node["items"]
        node = _resolve(child, root)
    return node, guaranteed


def validate_data(schema: dict, value: Any, *, root: dict | None = None) -> Any:
    """Non-coercing structural check only; defaults remain annotations."""
    # Public calls validate the full profile before jsonschema can see any $ref.
    if root is None:
        root = checked_schema(schema)
        schema = root
    else:
        root = checked_schema(root)
        # Fragment must originate from this already-checked document.
        if not any(schema == node for node in _schema_nodes(root)):
            raise DataBindingError("unbound_schema_fragment", "", "fragment is not part of the checked schema")
    copied = snapshot_json(value)
    validator = Draft202012Validator(root).evolve(schema=schema)
    error = next(validator.iter_errors(copied), None)
    if error:
        path = ""
        for key in error.absolute_path:
            path = join_pointer(path, key)
        raise DataBindingError("value_constraint", path, f"failed {error.validator} validation")
    return copied


def _schema_nodes(node):
    yield node
    for key in ("properties", "$defs"):
        for child in node.get(key, {}).values():
            yield from _schema_nodes(child)
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            yield from _schema_nodes(node[key])
