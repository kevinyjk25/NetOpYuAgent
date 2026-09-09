"""Compact authoring cannot invent aliases, edges, parameters or authority."""

import copy
import json

import pytest
from jsonschema import ValidationError

from evaluation.flow_behavior import BehaviorSuite, check_behavior
from evaluation.flow_read_region import compile_proposal, request
from evaluation.flow_semantic_examples import followup_cases
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree


def example():
    case = next(c for c in followup_cases() if c["id"] == "helpdesk")
    sources = FlowSources.model_validate(case["sources"])
    args = dict(ticket_key=dict(kind="reference", source="input", field="ticket_key"))
    raw = dict(status="read_region", purpose_source_id="s0003",
        facts=dict(tool="ticket_access_facts", source_id="s0006", arguments=args),
        target=dict(tool="ticket_evidence", source_id="s0009", arguments=args),
        condition="confirmed and (not sensitive or delegated)", condition_source_ids=["s0007"],
        stop_source_id="s0008", completion_source_id="s0009", on_failure="unsupported")
    return case, sources, raw


def test_compact_condition_and_data_compile_through_existing_executor():
    case, sources, raw = example()
    result = compile_proposal(sources, raw)
    assert result["status"] == "inactive_region_pending_source_review"
    tree = FlowTree.model_validate(result["tree"])
    assert check_behavior(sources, tree, BehaviorSuite.model_validate(case["suite"]))["passed"] == 11
    assert not result["runtimeAuthorityGranted"] and result["semanticAccuracy"] is None
    assert result["fullSourceReview"] == "required_not_run"
    payload = json.loads(request(sources)["messages"][1]["content"])
    assert not {"oracle", "reference", "observations", "rejectedProposal", "tree"} & payload.keys()
    assert "expected_calls" not in json.dumps(payload)
    assert "compilerCapabilities" in payload


@pytest.mark.parametrize("bad", ["unknown-tool", "alias", "edge", "argument", "before-read", "citation", "code"])
def test_compact_rejections_do_not_repair_from_expected_data(bad):
    _, sources, raw = example()
    raw = copy.deepcopy(raw)
    if bad == "unknown-tool":
        raw["target"]["tool"] = "invented"
    elif bad == "alias":
        raw["target"]["bind"] = "r0"
    elif bad == "edge":
        raw["next"] = 42
    elif bad == "argument":
        raw["target"]["arguments"] = {}
    elif bad == "before-read":
        raw["facts"]["arguments"]["ticket_key"]["source"] = "facts"
    elif bad == "citation":
        raw["purpose_source_id"] = "s0001"
    else:
        raw["condition"] = "__import__('os').system('false')"
    with pytest.raises((ValueError, ValidationError)):
        compile_proposal(sources, raw)


def test_unresolved_or_contradictory_condition_never_returns_unchecked_skeleton():
    _, sources, raw = example()
    raw["condition"] = "confirmed and not confirmed"
    assert "tree" not in compile_proposal(sources, raw)
    result = compile_proposal(sources, dict(status="unresolved", source_id="s0007", kind="source_ambiguity",
        question="Additional workflow semantics require separate review."))
    assert result["status"] == "unresolved_no_candidate" and "tree" not in result


def test_exact_facts_namespace_is_data_not_python_attribute_access():
    _, sources, raw = example()
    raw["condition"] = "facts.confirmed and (not facts.sensitive or facts.delegated)"
    result = compile_proposal(sources, raw)
    assert result["normalizedCondition"] == "confirmed and (not sensitive or delegated)"
    for invalid in ("facts.missing", "facts.confirmed.__class__", "other.confirmed", "facts.confirmed()"):
        raw["condition"] = invalid
        with pytest.raises(ValueError):
            compile_proposal(sources, raw)
