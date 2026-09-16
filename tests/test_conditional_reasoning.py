import copy

import pytest
from jsonschema import Draft202012Validator

from evaluation.hybrid_continuation import candidate_schema, selection_transport_schema
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from tests.test_governed_hybrid import setup, execute, ref
from tests.test_hybrid_live_demo import packet_and_compilation


def conditional_setup():
    values = setup()
    raw = values[0]
    raw["nodes"][1].update(kind="reason_if", condition={
        "left": ref("observe", "/outcome"), "equals": "read_path_completed",
        "value_schema": {"type": "string", "enum": ["read_path_completed", "needs_l1"]}},
        otherwise={"kind": "literal", "value": {"summary": "Retained unverified draft. No new evidence."}})
    return values


@pytest.mark.parametrize("invoke_model", [False, True])
def test_condition_controls_model_invocation_without_faking_a_call_or_promoting_output(invoke_model):
    raw, reads, bindings, reasoners, ctx, calls, requests = conditional_setup()
    if invoke_model:
        raw["nodes"][1]["condition"]["equals"] = "needs_l1"
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "governed_graph_completed"
    assert len(calls) == 1 and len(requests) == int(invoke_model)
    assert result["modelCallsReserved"] == int(invoke_model)
    assert result["outputs"]["draft"]["role"] == "model_candidate"
    if not invoke_model:
        event = next(e for e in result["trace"] if e.get("modelInvoked") is False)
        assert event["origin"].startswith("host_bound_retained_candidate")
        assert "model" not in event and not event["semanticCorrectnessProven"]
        assert result["outputs"]["draft"]["value"] == raw["nodes"][1]["otherwise"]["value"]


@pytest.mark.parametrize("mutation", ["missing_dependency", "unknown_source", "wrong_literal", "array_condition", "wrong_retained_type"])
def test_conditional_bindings_are_frozen_typed_and_dependency_checked(mutation):
    raw, reads, *_ = conditional_setup()
    node = raw["nodes"][1]
    if mutation == "missing_dependency":
        node["depends_on"] = []
    elif mutation == "unknown_source":
        node["condition"]["left"] = ref("not-a-parent")
    elif mutation == "wrong_literal":
        node["condition"]["equals"] = True
    elif mutation == "array_condition":
        node["condition"]["value_schema"] = {"type": "array", "items": {"type": "string"}}
    else:
        node["otherwise"]["value"] = {"summary": 123}
    with pytest.raises(ValueError):
        qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)


def test_conditional_reuse_still_requires_host_binding_and_cannot_bypass_strict_admission():
    raw, reads, bindings, _, ctx, calls, requests = conditional_setup()
    with pytest.raises(PermissionError):
        execute(raw, reads, bindings, {}, ctx)
    assert not calls and not requests
    raw["nodes"].append({**copy.deepcopy(raw["nodes"][0]), "id": "unsafe", "depends_on": ["draft"], "inputs": ref("draft")})
    raw["outputs"] = ["unsafe"]
    with pytest.raises(ValueError, match="candidate cannot directly"):
        qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)


def test_retained_candidate_byte_budget_is_enforced_without_callback():
    raw, reads, bindings, reasoners, ctx, calls, requests = conditional_setup()
    raw["nodes"][1]["max_output_bytes"] = 32
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "blocked" and not requests and len(calls) == 1


def test_selection_decoder_discriminates_empty_non_read_branch_but_runtime_gate_is_unchanged():
    packet, _, _ = packet_and_compilation()
    schema = selection_transport_schema(packet)
    validator = Draft202012Validator(schema)
    before = candidate_schema(packet)
    validator.validate({"decision": "clarify", "requests": {}, "message": "Which device should be inspected?"})
    bad = {"decision": "clarify", "requests": {"get_interfaces": {}}, "message": "Which device should be inspected?"}
    assert list(validator.iter_errors(bad))
    validator.validate({"decision": "get_interfaces", "requests": {"get_interfaces": {"device": {"id": "lab-sw1"}}},
                        "message": "Inspect this declared device's interfaces."})
    assert candidate_schema(packet) == before
