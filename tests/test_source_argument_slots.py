"""Mechanical shape/type checks; same-type business choices are never inferred."""
import copy

import pytest

from evaluation import source_argument_slots as slots
from evaluation.structured_flow_demo import object_schema
from network_runtime.l0.structured_bindings import compile_binding, materialize_binding


def example(target=None):
    identifier = {"type": "string"}
    source = object_schema({"first": identifier, "second": identifier,
        "rows": {"type": "array", "maxItems": 4, "items": object_schema({"name": identifier})}})
    target = target if target is not None else object_schema({"device": object_schema({"id": identifier})})
    packet = {"inputSchema": source, "catalog": {"tools": [{"name": "inspect", "inputSchema": target}]}}
    plan = {"reportDigest": "plan", "reads": []}
    slot = {"treePointer": "/steps/0", "tool": "inspect"}
    navigation = {"navigationTruncated": False, "paths": [
        {"reference": {"kind": "reference", "source": "input", "pointer": path}, "types": [kind]}
        for path, kind in [("", "object"), ("/first", "string"), ("/second", "string"), ("/rows/0/name", "string")]]}
    return slots.build(packet, plan, slot, navigation, {"input": "input"})


def answer(blueprint, bindings):
    return {"mode": "slot_arguments", "planDigest": blueprint["planDigest"], "readPointer": blueprint["readPointer"],
            "slotPacketDigest": blueprint["reportDigest"], "bindings": bindings}


def test_type_navigation_keeps_all_same_type_roles_and_does_not_choose():
    blueprint = example()
    choices = blueprint["slots"][0]["referenceChoices"]
    assert set(choices) == {"input#/first", "input#/second", "input#/rows/0/name"}
    assert not blueprint["semanticMappingInferred"]
    expression, audit = slots.lower(blueprint, answer(blueprint, {"/device/id": "input#/second"}))
    assert expression["fields"]["device"]["fields"]["id"] == choices["input#/second"]
    assert not audit["businessSourceSelectionByCode"] and not audit["runtimeAuthorityGranted"]
    assert audit["literalOriginCheckStillRequired"]
    # Both same-type choices remain possible; correctness requires source review.
    assert slots.lower(blueprint, answer(blueprint, {"/device/id": "input#/first"}))[0] != expression


def test_unlisted_array_index_preserves_original_pointer_and_instance_checks():
    blueprint = example()
    value = answer(blueprint, {"/device/id": {"reference": {"source": "input", "pointer": "/rows/2/name"}}})
    expression, _ = slots.lower(blueprint, value)
    binding = compile_binding(blueprint["sourceSchemas"], blueprint["targetSchema"], expression)
    actual = {"first": "a", "second": "b", "rows": [{"name": "x"}, {"name": "y"}, {"name": "z"}]}
    assert materialize_binding(binding, {"input": actual})["arguments"] == {"device": {"id": "z"}}
    with pytest.raises(ValueError):
        materialize_binding(binding, {"input": {**actual, "rows": []}})
    value["bindings"]["/device/id"]["reference"]["pointer"] = "/rows/4/name"
    with pytest.raises(ValueError):
        slots.lower(blueprint, value)


@pytest.mark.parametrize("selection", [
    "input#", "unknown#/first", {"reference": {"source": "future", "pointer": "/first"}},
    {"reference": {"source": "input", "pointer": ""}}, {"reference": {"source": "input", "pointer": "/missing"}},
    {"reference": {"source": "input", "pointer": "/first", "extra": True}},
    {"literal": 1, "origin": {"kind": "task", "quote": "1"}},
])
def test_invalid_or_wrong_typed_choices_do_not_repair_or_execute(selection):
    blueprint = example()
    with pytest.raises(ValueError):
        slots.lower(blueprint, answer(blueprint, {"/device/id": selection}))


def test_dynamic_literal_is_not_executed_or_interpreted_as_reference():
    blueprint = example()
    raw = {"literal": "{{input.first}}", "origin": {"kind": "task", "quote": "unproven"}}
    expression, audit = slots.lower(blueprint, answer(blueprint, {"/device/id": raw}))
    assert expression["fields"]["device"]["fields"]["id"]["value"] == "{{input.first}}"
    assert audit["literalOriginCheckStillRequired"]  # existing source-origin checker rejects unsupported literals


@pytest.mark.parametrize("target", [
    {"type": "array", "items": {"type": "string"}},
    {"type": "object", "properties": {"optional": {"type": "string"}}, "additionalProperties": False},
    {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}},
    {"type": ["object", "null"], "properties": {}, "additionalProperties": False},
])
def test_complex_shapes_retain_generic_authoring(target):
    assert example(target) is None


def test_empty_object_and_escaped_keys_are_exact_schema_shape():
    empty = example(object_schema({}))
    assert slots.lower(empty, answer(empty, {}))[0] == {"kind": "object", "fields": {}}
    escaped = example(object_schema({"a/b~c": {"type": "string"}}))
    expression, _ = slots.lower(escaped, answer(escaped, {"/a~1b~0c": "input#/first"}))
    assert "a/b~c" in expression["fields"]


def test_original_host_constant_origin_is_exact_not_inferred_from_source():
    blueprint = example(object_schema({"mode": {"type": "string", "const": "status"}}))
    alternatives = slots.schema(blueprint)["properties"]["bindings"]["properties"]["/mode"]["oneOf"]
    assert alternatives[-1]["properties"]["origin"] == {"const": {"kind": "host_schema", "pointer": "/properties/mode/const"}}


def test_slot_bindings_are_frozen_and_complete():
    blueprint = example()
    with pytest.raises(ValueError):
        slots.lower(blueprint, answer(blueprint, {}))
    broken = copy.deepcopy(blueprint)
    broken["layout"] = {}
    with pytest.raises(ValueError, match="drift"):
        slots.lower(broken, answer(blueprint, {"/device/id": "input#/first"}))
