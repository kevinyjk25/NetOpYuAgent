"""Versioned, inactive structured-data binding plans; not another executor.

The operations are literals, JSON Pointers, objects, arrays and bounded column
projection. Every materialization rechecks source values and the complete target.
No interpolation, coercion, implicit selection, general loop or tool invocation.
Shape checks neither prove source semantics nor authenticate observations.
"""

from __future__ import annotations

import re

from network_runtime.contracts import sha256_json

from .column_rows import check_projection_schema, decode_column_rows
from .structured_schema import (
    PROFILE, DataBindingError, checked_schema, join_pointer, pointer_value,
    schema_location, schema_types, snapshot_json, validate_data,
)

API_VERSION = "netopyu.io/l0-data-binding/v1"
ARRAY_LENGTH_API_VERSION = "netopyu.io/l0-data-binding/v2"


def _seal(body: dict, field: str) -> dict:
    return {**body, field: sha256_json(body)}


def _overlap(source: set[str], target: set[str]) -> bool:
    # JSON Schema integer is a subset of number, never of boolean.
    def numbers(kinds):
        return kinds | ({"integer"} if "number" in kinds else set())
    return bool(numbers(source) & numbers(target))


def compile_binding(
    source_schemas: dict, target_schema: dict, expression: dict, *, source_bundle_digest: str | None = None,
) -> dict:
    """Qualify a data plan, not a reusable whole-Skill or authorized invocation.

    Overlapping types can still fail on enum/range/required constraints. We do
    not claim schema-subtyping proof: actual source and target values are always
    revalidated, including when a declared path is optional or an array is empty.
    """
    source_schemas = snapshot_json(source_schemas)
    if (not isinstance(source_schemas, dict) or len(source_schemas) > 32
            or any(not key.strip() for key in source_schemas)):
        raise DataBindingError("invalid_sources", "", "at most 32 named source schemas required")
    sources = {key: checked_schema(value) for key, value in source_schemas.items()}
    target = checked_schema(target_schema)
    expression = snapshot_json(expression)
    if source_bundle_digest is not None and (not isinstance(source_bundle_digest, str)
                                            or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_bundle_digest)):
        raise DataBindingError("invalid_source_digest", "", "expected a source bundle digest")
    mappings, count = [], 0

    def walk(expr, target_path, expression_path, depth):
        nonlocal count
        count += 1
        if count > 512 or depth > 16:
            raise DataBindingError("binding_budget", expression_path, "binding node/depth budget exceeded")
        if not isinstance(expr, dict):
            raise DataBindingError("invalid_binding", expression_path, "explicit binding expression required")
        shapes = {"literal": {"kind", "value"}, "reference": {"kind", "source", "pointer"},
                  "array_length": {"kind", "source", "pointer"},
                  "object": {"kind", "fields"}, "array": {"kind", "items"},
                  "column_rows": {"kind", "source", "pointer", "fields", "max_rows", "max_columns"}}
        kind = expr.get("kind")
        if not isinstance(kind, str) or kind not in shapes or set(expr) != shapes[kind]:
            raise DataBindingError("invalid_binding", expression_path, "unknown kind or fields")
        expected, _ = schema_location(target, target_path)
        kinds = schema_types(expected)
        base = {"targetPointer": target_path, "expressionPointer": expression_path, "kind": kind}
        if kind == "literal":
            validate_data(expected, expr["value"], root=target)
            mappings.append({**base, "literalDigest": sha256_json(expr["value"])})
        elif kind in {"reference", "array_length"}:
            if not isinstance(expr["source"], str) or expr["source"] not in sources:
                raise DataBindingError("unknown_binding_source", expression_path, "source schema is not declared")
            actual, guaranteed = schema_location(sources[expr["source"]], expr["pointer"])
            actual_kinds = schema_types(actual)
            if kind == "array_length":
                if actual_kinds != {"array"}:
                    raise DataBindingError("binding_type_mismatch", expression_path, "array_length requires an exclusively array source")
                actual_kinds = {"integer"}
            if not _overlap(actual_kinds, kinds):
                raise DataBindingError("binding_type_mismatch", expression_path, "source and target types do not overlap")
            mappings.append({**base, "source": expr["source"], "sourcePointer": expr["pointer"],
                             "sourcePathGuaranteedPresent": guaranteed, "valueValidationRequired": True,
                             "schemaSubtypingProven": False})
        elif kind == "column_rows":
            if not isinstance(expr["source"], str) or expr["source"] not in sources:
                raise DataBindingError("unknown_binding_source", expression_path, "source schema is not declared")
            check_projection_schema(sources[expr["source"]], expr["pointer"], target, target_path,
                                    expr["fields"], expr["max_rows"], expr["max_columns"])
            mappings.append({**base, "source": expr["source"], "sourcePointer": expr["pointer"],
                             "selectedFields": expr["fields"], "indexResolution": "per_response_columns_metadata",
                             "maxRows": expr["max_rows"], "maxColumns": expr["max_columns"],
                             "valueValidationRequired": True, "schemaSubtypingProven": False})
        elif kind == "object":
            fields = expr["fields"]
            if "object" not in kinds or not isinstance(fields, dict):
                raise DataBindingError("binding_type_mismatch", expression_path, "object construction requires an object target")
            if set(expected.get("required", [])) - fields.keys():
                raise DataBindingError("missing_target_binding", expression_path, "required target fields must be explicitly bound")
            for key, child in fields.items():
                walk(child, join_pointer(target_path, key), join_pointer(join_pointer(expression_path, "fields"), key), depth + 1)
        else:
            items = expr["items"]
            if "array" not in kinds or not isinstance(items, list):
                raise DataBindingError("binding_type_mismatch", expression_path, "array construction requires an array target")
            if not expected.get("minItems", 0) <= len(items) <= expected.get("maxItems", 1024):
                raise DataBindingError("binding_array_length", expression_path, "constructed length violates target bounds")
            for index, child in enumerate(items):
                walk(child, join_pointer(target_path, index), join_pointer(join_pointer(expression_path, "items"), index), depth + 1)

    walk(expression, "", "", 0)
    body = {
        "apiVersion": ARRAY_LENGTH_API_VERSION if any(m["kind"] == "array_length" for m in mappings) else API_VERSION,
        "schemaProfile": PROFILE,
        "sourceSchemas": sources, "targetSchema": target, "expression": expression,
        "sourceBundleDigest": source_bundle_digest, "mappings": mappings,
        "requiredSources": sorted({m["source"] for m in mappings if m["kind"] in {"reference", "column_rows", "array_length"}}),
        "status": "typed_data_binding_not_authorized", "requiresPerInstanceValidation": True,
        "semanticAlignmentProven": False, "sourceAuthenticityVerified": False, "runtimeAuthorityGranted": False,
    }
    return snapshot_json(_seal(body, "bindingDigest"))


def verify_binding(plan: dict) -> dict:
    plan = snapshot_json(plan)
    if not isinstance(plan, dict) or plan.get("apiVersion") not in {API_VERSION, ARRAY_LENGTH_API_VERSION}:
        raise DataBindingError("invalid_binding_plan", "", "unknown data binding version")
    required = {"sourceSchemas", "targetSchema", "expression", "sourceBundleDigest"}
    if not required <= plan.keys():
        raise DataBindingError("invalid_binding_plan", "", "incomplete binding plan")
    rebuilt = compile_binding(plan["sourceSchemas"], plan["targetSchema"], plan["expression"],
                              source_bundle_digest=plan["sourceBundleDigest"])
    if rebuilt != plan:
        raise DataBindingError("binding_plan_drift", "", "binding, mappings or authority flags changed")
    return rebuilt


def materialize_binding(plan: dict, source_values: dict) -> dict:
    """Create validated data only; never trusts a passed flag or prior shape check."""
    plan = verify_binding(plan)
    supplied = snapshot_json(source_values)
    if not isinstance(supplied, dict) or set(supplied) != set(plan["requiredSources"]):
        raise DataBindingError("binding_source_set", "", "provide exactly the referenced sources")
    values = {}
    for key, value in supplied.items():
        try:
            values[key] = validate_data(plan["sourceSchemas"][key], value)
        except DataBindingError as error:
            raise DataBindingError(error.code, join_pointer("/sources", key) + error.pointer, error.detail) from None

    def resolve(expr):
        if expr["kind"] == "literal":
            return expr["value"]
        if expr["kind"] in {"reference", "column_rows", "array_length"}:
            try:
                source = pointer_value(values[expr["source"]], expr["pointer"])
                if expr["kind"] == "reference":
                    return source
                if expr["kind"] == "array_length":
                    if type(source) is not list:
                        raise DataBindingError("binding_type_mismatch", expr["pointer"], "array_length requires an actual array")
                    return len(source)
                try:
                    return decode_column_rows(source, expr["fields"], max_rows=expr["max_rows"], max_columns=expr["max_columns"])
                except DataBindingError as error:
                    raise DataBindingError(error.code, expr["pointer"] + error.pointer, error.detail) from None
            except DataBindingError as error:
                raise DataBindingError(error.code, join_pointer("/sources", expr["source"]) + error.pointer, error.detail) from None
        if expr["kind"] == "object":
            return {key: resolve(child) for key, child in expr["fields"].items()}
        return [resolve(child) for child in expr["items"]]

    assembled = resolve(plan["expression"])
    try:
        result = validate_data(plan["targetSchema"], assembled)
    except DataBindingError as error:
        raise DataBindingError(error.code, "/target" + error.pointer, error.detail) from None
    body = {
        "kind": "StructuredArgumentsDraft", "bindingDigest": plan["bindingDigest"], "arguments": result,
        "sourceValueDigests": {key: sha256_json(value) for key, value in sorted(values.items())},
        "argumentDigest": sha256_json(result), "mappings": plan["mappings"],
        "shapeValid": True, "businessCorrectnessProven": False,
        "sourceAuthenticityVerified": False, "runtimeAuthorityGranted": False,
    }
    return _seal(body, "draftDigest")


def compile_tool_binding(catalog: dict, tool_name: str, source_schemas: dict, expression: dict, **kwargs) -> dict:
    """Bind to the original tool declaration, with no read-only or permission inference."""
    catalog = snapshot_json(catalog)
    tools = catalog.get("tools") if isinstance(catalog, dict) else None
    if not isinstance(tools, list) or not 1 <= len(tools) <= 128:
        raise DataBindingError("invalid_host_catalog", "", "nonempty bounded tools list required")
    names = [tool.get("name") if isinstance(tool, dict) else None for tool in tools]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(names) != len(set(names)):
        raise DataBindingError("invalid_host_catalog", "/tools", "unique nonblank tool names required")
    if tool_name not in names:
        raise DataBindingError("unknown_host_tool", "/tools", "tool must be selected from the supplied catalog")
    tool = tools[names.index(tool_name)]
    if "inputSchema" not in tool:
        raise DataBindingError("missing_host_schema", "/inputSchema", "target input schema is required")
    plan = compile_binding(source_schemas, tool["inputSchema"], expression, **kwargs)
    body = {"kind": "HostStructuredBindingDraft", "catalogDigest": sha256_json(catalog),
            "toolDeclaration": tool, "toolDigest": sha256_json(tool), "binding": plan,
            "readOnlyProven": False, "runtimeAuthorityGranted": False}
    return snapshot_json(_seal(body, "hostBindingDigest"))


def materialize_tool_binding(host_plan: dict, catalog: dict, source_values: dict) -> dict:
    """Require the exact catalog again; hashes bind content, not provider identity."""
    host_plan = snapshot_json(host_plan)
    if not isinstance(host_plan, dict) or not isinstance(host_plan.get("binding"), dict):
        raise DataBindingError("invalid_host_binding", "", "host binding draft required")
    plan = verify_binding(host_plan["binding"])
    tool = host_plan.get("toolDeclaration", {})
    if not isinstance(tool, dict):
        raise DataBindingError("invalid_host_binding", "", "original tool declaration required")
    rebuilt = compile_tool_binding(catalog, tool.get("name"), plan["sourceSchemas"], plan["expression"],
                                    source_bundle_digest=plan["sourceBundleDigest"])
    if rebuilt != host_plan:
        raise DataBindingError("host_binding_drift", "", "host declaration, catalog or binding changed")
    return {"hostBindingDigest": host_plan["hostBindingDigest"], "tool": tool["name"],
            "draft": materialize_binding(plan, source_values), "runtimeAuthorityGranted": False}
