import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from evaluation.flow_translation import local_sources
from evaluation.flow_tree_capabilities import bounded_request, host_schema, validate_bounded
from network_runtime.l0.flow import EffectTarget


def end():
    return {"business_source_ids": ["s0001"], "steps": [{"kind": "end", "source_id": "s0001", "outcome": "needs_l1"}], "issues": []}


def test_empty_host_removes_unavailable_constructors_recursively():
    source = local_sources().model_copy(update={"reads": {}, "effects": {}})
    schema = host_schema(source)
    assert "TreeRead" not in json.dumps(schema) and "TreeEffect" not in json.dumps(schema)
    Draft202012Validator(schema).validate(end())
    assert not validate_bounded(source, end())["runtimeAuthorityGranted"]


@pytest.mark.parametrize("kind", ["read", "effect_candidate"])
def test_missing_constructor_rejected_even_if_decoder_ignores_constraints(kind):
    source = local_sources().model_copy(update={"reads": {}, "effects": {}})
    raw = end()
    raw["steps"][0] = ({"kind": "read", "source_id": "s0001", "tool": "invented", "bind": "result", "arguments": {}}
                       if kind == "read" else {"kind": kind, "source_id": "s0001", "binding_id": "needs_l1", "arguments": {}})
    with pytest.raises(ValidationError):
        validate_bounded(source, raw)

def test_effect_targets_come_only_from_host_not_read_aliases():
    source = local_sources()
    target = EffectTarget(profile="local", tool="write", skill_id="local.example", contract_hash="sha256:" + "1" * 64,
                          input_schema=source.input_schema)
    source = source.model_copy(update={"effects": {"host-change": target}})
    schema = host_schema(source)
    assert schema["$defs"]["TreeEffect"]["properties"]["binding_id"]["enum"] == ["host-change"]
    raw = end()
    raw["steps"][0] = {"kind": "effect_candidate", "source_id": "s0001", "binding_id": "inventory_result", "arguments": {}}
    with pytest.raises(ValidationError):
        validate_bounded(source, raw)
    raw["steps"][0]["binding_id"] = "host-change"
    raw["steps"][0]["arguments"] = {"device_id": {"kind": "reference", "source": "input", "field": "device_id"}}
    result = validate_bounded(source, raw)
    assert result["flow"]["nodes"][0]["binding_id"] == "host-change"
    assert not result["runtimeAuthorityGranted"]


def test_tool_enum_and_compiler_parameter_gate_remain():
    source = local_sources()
    assert host_schema(source)["$defs"]["TreeRead"]["properties"]["tool"]["enum"] == ["read_inventory_device"]
    raw = end()
    raw["steps"].insert(0, {"kind": "read", "source_id": "s0001", "tool": "read_inventory_device", "bind": "result", "arguments": {}})
    Draft202012Validator(host_schema(source)).validate(raw)
    with pytest.raises(ValueError):
        validate_bounded(source, raw)


def test_schema_visible_and_no_label_dependent_answer_repair():
    source = local_sources()
    wire = bounded_request(source)
    payload = json.loads(wire["messages"][1]["content"])
    assert payload["outputSchema"] == wire["format"] == host_schema(source)
    assert payload["constructorBoundary"]["effectTargetIds"] == []
    assert bounded_request(source.model_copy(update={"source_path": "different-hidden-label"})) == wire
    assert "expectedObject" not in payload


def test_semantically_wrong_terminal_is_not_auto_repaired_or_authorized():
    source = local_sources()
    raw = end()  # Compiler cannot infer full source fidelity from a cited title.
    report = validate_bounded(source, raw)
    assert report["flow"]["nodes"][0]["outcome"] == "needs_l1"
    assert report["status"] == "compiled_pending_source_review_not_executable"
    assert not report["runtimeAuthorityGranted"]
