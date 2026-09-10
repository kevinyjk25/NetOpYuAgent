"""Closed syntax tests are hand-authored mechanics, not model accuracy data."""
import copy

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_closed_program as closed, source_program_lines as lines


def terminal():
    return {"op": "complete", "source_id": "original-line"}


def program():
    return {"op": "read", "name": "obs0", "tool": "fetch", "source_id": "original-line",
        "next": {"op": "if_equal", "source_id": "original-line",
            "value": {"kind": "field", "source": "obs0", "pointer": "/ready"}, "equals": True,
            "when_equal": terminal(), "otherwise": {"op": "handoff", "source_id": "original-line",
                "outcome": "needs_l1",
                "duties": [{"when": "When not ready", "requirement": "Explain this observation only.", "source_id": "original-line"}],
                "restrictions": []}}}


def model_schema():
    return closed.schema({"tools": [{"name": "fetch"}]}, evidence_ids=["original-line"],
        observation_names=lines.OBSERVATION_SLOTS,
        value_paths=[{"types": ["boolean"], "pointer": "/ready"}])


def test_closed_tree_lowers_every_node_without_repair_or_inferred_success():
    node = program()
    original = copy.deepcopy(node)
    assert Draft202012Validator(model_schema()).is_valid(node)
    lowered = closed.lower(node)
    assert node == original
    assert lowered["nodeCount"] == len(lowered["statementOrigins"]) == 4
    assert lowered["statementsDiscarded"] == 0
    assert lowered["statements"][0] == {k: v for k, v in node.items() if k != "next"}
    assert lowered["statements"][1]["otherwise"] == [{**node["next"]["otherwise"],
                                                     "explanation": closed.TERMINAL_LABELS["handoff"]}]
    assert lowered["statementOrigins"][1] == {"originalPointer": "/program/next", "normalizedPointer": "/program/1"}
    assert not lowered["businessConditionsInferred"] and not lowered["runtimeAuthorityGranted"]


def test_read_origin_is_declared_before_adapter_and_continuation():
    read = model_schema()["$defs"]["ProgramNode"]["oneOf"][0]
    keys = list(read["properties"])
    assert keys.index("source_id") < keys.index("tool") < keys.index("next")
    assert read["properties"]["source_id"] == {"$ref": "#/$defs/ProgramSourceId"}


def test_all_node_origins_precede_operands_and_terminal_labels_cannot_claim_business_success():
    for variant in model_schema()["$defs"]["ProgramNode"]["oneOf"]:
        properties = variant["properties"]
        assert list(properties)[:2] == ["op", "source_id"]
        if properties["op"]["const"] in closed.TERMINAL_LABELS:
            assert "explanation" not in properties
    node = terminal()
    node["explanation"] = "The entire service is healthy and all repairs succeeded."
    assert not Draft202012Validator(model_schema()).is_valid(node)
    with pytest.raises(ValueError, match="no model-authored"):
        closed.lower(node)


def test_returned_audit_cannot_mutate_future_control_status_labels():
    audit = closed.lower(terminal())
    original = audit["statements"][0]["explanation"]
    audit["terminalLabels"]["complete"] = "Forged successful business outcome."
    assert closed.lower(terminal())["statements"][0]["explanation"] == original


@pytest.mark.parametrize("mutation", ["terminal_next", "branch_next", "missing_next", "missing_arm", "array_arm", "unknown"])
def test_model_schema_and_lowerer_reject_invalid_control_shapes(mutation):
    node = program()
    if mutation == "terminal_next":
        node["next"]["when_equal"]["next"] = terminal()
    elif mutation == "branch_next":
        node["next"]["next"] = terminal()
    elif mutation == "missing_next":
        node.pop("next")
    elif mutation == "missing_arm":
        node["next"].pop("otherwise")
    elif mutation == "array_arm":
        node["next"]["otherwise"] = [terminal()]
    else:
        node["op"] = "exec"
    assert not Draft202012Validator(model_schema()).is_valid(node)
    with pytest.raises(ValueError):
        closed.lower(node)


def test_over_budget_and_cyclic_inputs_fail_before_recursive_validation():
    node = terminal()
    for i in range(32):
        node = {"op": "read", "name": f"obs{i}", "tool": "fetch", "source_id": "original-line", "next": node}
    with pytest.raises(ValueError, match="budget"):
        closed.lower(node)
    node["next"] = node
    with pytest.raises(ValueError, match="budget"):
        closed.lower(node)


def test_closed_syntax_does_not_choose_business_value_or_drop_remaining_duty():
    node = program()
    node["next"]["equals"] = False
    node["next"]["otherwise"]["duties"][0]["requirement"] = "A deliberately incorrect but source-shaped proposed duty."
    assert Draft202012Validator(model_schema()).is_valid(node)
    lowered = closed.lower(node)
    assert lowered["statements"][1]["equals"] is False
    assert lowered["statements"][1]["otherwise"][0]["duties"] == node["next"]["otherwise"]["duties"]
    assert not lowered["semanticEntailmentProven"]
