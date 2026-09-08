"""Mechanical type/binding checks, not semantic Skill or Runtime experiments."""

import copy
import json
from pathlib import Path

import pytest

from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_bindings import (
    compile_binding, compile_tool_binding, materialize_binding, materialize_tool_binding, verify_binding,
)
from network_runtime.l0.structured_schema import (
    DataBindingError, checked_schema, join_pointer, pointer_parts, snapshot_json, validate_data,
)


def obj(properties, required=None, **kw):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False, **kw}


def ref(source, pointer):
    return {"kind": "reference", "source": source, "pointer": pointer}


def literal(value):
    return {"kind": "literal", "value": value}


def object_expr(**fields):
    return {"kind": "object", "fields": fields}


def test_previous_catalog_preserved_and_nested_arguments_materialized():
    catalog = json.loads(Path("examples/translation-intake/mcp-catalog.json").read_text())
    before = copy.deepcopy(catalog)
    tool = catalog["tools"][0]
    for schema in (tool["inputSchema"], tool["outputSchema"]):
        assert checked_schema(schema) == schema
    source = obj({"设备/id": {"type": "string"}, "states": {"type": "array", "items": {"type": "string"}}})
    expr = object_expr(deviceId=ref("input", "/设备~1id"), selections=object_expr(states=ref("input", "/states")))
    host = compile_tool_binding(catalog, tool["name"], {"input": source}, expr)
    result = materialize_tool_binding(host, catalog, {"input": {"设备/id": "r1", "states": ["down"]}})
    assert result["draft"]["arguments"] == {"deviceId": "r1", "selections": {"states": ["down"]}}
    assert not host["readOnlyProven"] and not result["runtimeAuthorityGranted"]
    assert host["toolDeclaration"] == tool and catalog == before
    with pytest.raises(DataBindingError, match="enum"):
        materialize_tool_binding(host, catalog, {"input": {"设备/id": "r1", "states": ["unknown"]}})


def test_full_output_validated_before_nested_reference_and_whole_array_binding():
    schema = obj({"ports": {"type": "array", "items": obj({"if.Name": {"type": "string"}, "up": {"type": "boolean"}})}})
    target = obj({"first": {"type": "string"}, "all": schema["properties"]["ports"]})
    plan = compile_binding({"status": schema}, target,
                           object_expr(first=ref("status", "/ports/0/if.Name"), all=ref("status", "/ports")))
    values = {"status": {"ports": [{"if.Name": "eth0", "up": True}]}}
    result = materialize_binding(plan, values)
    assert result["arguments"] == {"first": "eth0", "all": values["status"]["ports"]}
    assert not next(m for m in plan["mappings"] if m["targetPointer"] == "/first")["sourcePathGuaranteedPresent"]
    values["status"]["ports"][0]["up"] = "true"
    with pytest.raises(DataBindingError) as caught:
        materialize_binding(plan, values)
    assert caught.value.pointer == "/sources/status/ports/0/up"
    assert result["arguments"]["all"][0]["up"] is True


def test_literal_array_assembly_null_and_no_implicit_defaults_or_interpolation():
    target = obj({"items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": ["string", "null"]}},
                  "optional": {"type": "string", "default": "never inserted"}}, required=["items"])
    expr = object_expr(items={"kind": "array", "items": [literal("${not_executed}"), literal(None)]})
    plan = compile_binding({}, target, expr)
    assert materialize_binding(plan, {})["arguments"] == {"items": ["${not_executed}", None]}


def test_local_refs_retain_constraints_and_validate_fragments():
    schema = obj({"port": {"$ref": "#/$defs/Port"}}, **{"$defs": {"Port": obj({
        "id": {"type": "integer", "minimum": 1, "maximum": 4094}})}})
    assert checked_schema(schema) == schema
    plan = compile_binding({}, schema, object_expr(port=literal({"id": 10})))
    assert materialize_binding(plan, {})["arguments"] == {"port": {"id": 10}}
    with pytest.raises(DataBindingError, match="maximum"):
        compile_binding({}, schema, object_expr(port=literal({"id": 5000})))
    plan2 = compile_binding({"read": schema}, {"type": "integer"}, ref("read", "/port/id"))
    assert materialize_binding(plan2, {"read": {"port": {"id": 10}}})["arguments"] == 10


@pytest.mark.parametrize("source", [None, [], {}])
def test_absent_optional_or_empty_array_never_becomes_default_success(source):
    schema = obj({"ports": {"type": ["array", "null"], "items": obj({"id": {"type": "string"}})}}, required=[])
    plan = compile_binding({"r": schema}, {"type": "string"}, ref("r", "/ports/0/id"))
    payload = {} if source == {} else {"ports": source}
    with pytest.raises(DataBindingError, match="missing_source_value"):
        materialize_binding(plan, {"r": payload})


@pytest.mark.parametrize("schema,value,keyword", [
    ({"type": "integer"}, True, "type"), ({"type": "integer"}, "1", "type"),
    ({"type": "string", "minLength": 2}, "a", "minLength"),
    ({"type": "number", "exclusiveMinimum": 0}, 0, "exclusiveMinimum"),
    ({"type": "integer", "maximum": 10}, 11, "maximum"),
    ({"type": "array", "items": {"type": "string"}, "uniqueItems": True}, ["a", "a"], "uniqueItems"),
    ({"type": "string", "enum": ["up", "down"]}, "UP", "enum"),
    ({"type": "string", "const": "only"}, "other", "const"),
    (obj({"x": {"type": "integer"}}), {}, "required"),
    (obj({}), {"x": 1}, "additionalProperties"),
])
def test_constraints_reject_without_coercion_or_payload_leak(schema, value, keyword):
    with pytest.raises(DataBindingError) as caught:
        validate_data(schema, value)
    assert caught.value.detail == f"failed {keyword} validation"


@pytest.mark.parametrize("schema,code", [
    ({"type": "string", "pattern": "(a+)+$"}, "unsupported_schema_keyword"),
    ({"type": "string", "format": "uri"}, "unsupported_schema_keyword"),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "unsupported_schema_keyword"),
    ({"type": "array"}, "untyped_schema"),
    ({"$ref": "https://example.invalid/schema"}, "unsupported_schema_reference"),
    ({"$ref": "file:///etc/passwd"}, "unsupported_schema_reference"),
    ({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}}, "recursive_schema"),
    ({"$ref": "#/$defs/absent"}, "unknown_schema_reference"),
    ({"type": "string", "$schema": "https://json-schema.org/draft-07/schema"}, "unsupported_schema_dialect"),
    ({"$ref": "#/$defs/A", "type": "string", "$defs": {"A": {"type": "string"}}}, "unsupported_ref_siblings"),
])
def test_unsupported_constraints_not_silently_discarded(schema, code, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("no reference retrieval"))
    before = copy.deepcopy(schema)
    with pytest.raises(DataBindingError) as caught:
        checked_schema(schema)
    assert caught.value.code == code and schema == before


@pytest.mark.parametrize("key", ["camelCase", "UPPER", "vlan-id", "a.b", "a/b", "a~b", "设备", "", "0"])
def test_field_name_metamorphism_keeps_values_and_pointers_exact(key):
    source = obj({key: {"type": "integer"}})
    pointer = join_pointer("", key)
    assert pointer_parts(pointer) == [key]
    plan = compile_binding({"input": source}, source, {"kind": "object", "fields": {key: ref("input", pointer)}})
    assert materialize_binding(plan, {"input": {key: 23}})["arguments"] == {key: 23}


@pytest.mark.parametrize("pointer", ["field", "/bad~2escape", "/arr/-", "/arr/01", "/arr/9999"])
def test_invalid_or_unbounded_selection_not_guessed(pointer):
    schema = obj({"arr": {"type": "array", "items": {"type": "string"}}})
    with pytest.raises(DataBindingError):
        compile_binding({"input": schema}, {"type": "string"}, ref("input", pointer))


def test_static_shape_compatibility_does_not_skip_instance_constraints():
    plan = compile_binding({"input": {"type": ["integer", "null"]}}, {"type": "integer", "maximum": 3}, ref("input", ""))
    assert not plan["mappings"][0]["schemaSubtypingProven"]
    assert materialize_binding(plan, {"input": 3})["arguments"] == 3
    for value in (None, 4):
        with pytest.raises(DataBindingError, match="value_constraint"):
            materialize_binding(plan, {"input": value})
    with pytest.raises(DataBindingError, match="type_mismatch"):
        compile_binding({"input": {"type": "boolean"}}, {"type": "integer"}, ref("input", ""))


@pytest.mark.parametrize("mutation", ["runtimeAuthorityGranted", "mappings", "expression", "schemaProfile"])
def test_resealed_plan_mutation_cannot_bypass_recompilation(mutation):
    plan = compile_binding({}, {"type": "string"}, literal("x"))
    plan[mutation] = True if mutation == "runtimeAuthorityGranted" else []
    plan["bindingDigest"] = sha256_json({k: v for k, v in plan.items() if k != "bindingDigest"})
    with pytest.raises(DataBindingError):
        verify_binding(plan)


def test_catalog_drift_and_extra_or_missing_source_fail():
    catalog = {"tools": [{"name": "x", "inputSchema": obj({"id": {"type": "string"}})}]}
    host = compile_tool_binding(catalog, "x", {"input": {"type": "string"}}, object_expr(id=ref("input", "")))
    for values in ({}, {"input": "a", "unknown": "b"}):
        with pytest.raises(DataBindingError, match="binding_source_set"):
            materialize_tool_binding(host, catalog, values)
    catalog["tools"][0]["description"] = "changed source"
    with pytest.raises(DataBindingError, match="host_binding_drift"):
        materialize_tool_binding(host, catalog, {"input": "a"})


@pytest.mark.parametrize("value", [float("inf"), float("nan"), {1: "not a string key"}, (1, 2)])
def test_non_json_inputs_fail_before_validation(value):
    with pytest.raises(DataBindingError, match="non_json_value"):
        snapshot_json(value)


def test_recursive_or_excessive_data_and_schema_fail_bounded():
    value = []
    value.append(value)
    with pytest.raises(DataBindingError, match="data_budget"):
        snapshot_json(value)
    with pytest.raises(DataBindingError, match="data_budget"):
        snapshot_json([0] * 1025)
    schema = {"type": "string"}
    for _ in range(20):
        schema = {"type": "array", "items": schema}
    with pytest.raises(DataBindingError, match="schema_budget"):
        checked_schema(schema)


def test_unknown_fields_operations_and_forbidden_or_untyped_targets_fail():
    with pytest.raises(DataBindingError, match="invalid_binding"):
        compile_binding({}, {"type": "string"}, {"kind": "eval", "value": "x"})
    with pytest.raises(DataBindingError, match="missing_target_binding"):
        compile_binding({}, obj({"required": {"type": "string"}}), object_expr())
    for schema in (obj({}), {"type": "object"}):
        with pytest.raises(DataBindingError, match="unknown_schema_path"):
            compile_binding({}, schema, object_expr(unknown=literal("x")))


def test_typed_additional_properties_and_array_length_checks():
    schema = {"type": "object", "additionalProperties": {"type": "integer"}, "maxProperties": 2}
    plan = compile_binding({}, schema, object_expr(vlan=literal(12)))
    assert materialize_binding(plan, {})["arguments"] == {"vlan": 12}
    with pytest.raises(DataBindingError, match="binding_array_length"):
        compile_binding({}, {"type": "array", "items": {"type": "string"}, "minItems": 1}, {"kind": "array", "items": []})


def test_target_field_named_sources_does_not_confuse_error_origin():
    target = obj({"sources": obj({"x": {"type": "integer", "maximum": 3}})})
    plan = compile_binding({"input": {"type": "integer"}}, target,
                           object_expr(sources=object_expr(x=ref("input", ""))))
    with pytest.raises(DataBindingError) as caught:
        materialize_binding(plan, {"input": 4})
    assert caught.value.pointer == "/target/sources/x"
