"""Concrete condition diagnostics preserve source ambiguity and never edit trees."""

import copy
import json

import pytest
from jsonschema import ValidationError

from evaluation.flow_joint_conditions import compare, region, request
from evaluation.flow_joint_lowering import propose
from evaluation.flow_behavior import BehaviorSuite, check_behavior
from evaluation.flow_semantic_examples import cases
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree


def fixture(key):
    case = next(c for c in cases() if c["id"] == key)
    return case, FlowSources.model_validate(case["sources"]), FlowTree.model_validate(case["reference"])


def answers_for(case, sources, tree):
    context = region(sources, tree)
    result = {}
    for key, values in context["assignments"].items():
        # Independent fixture labels are used only by this TEST. Never call the
        # model with this mapping, or build source answers from candidate guards.
        scenario = next(s for s in case["suite"]["scenarios"]
            if all(s["observations"][0]["payload"].get(k) is v for k, v in values.items())
            and s["id"] not in ("provider-error", "missing-fields", "access-denied"))
        result[key] = dict(outcome="read" if len(scenario["expected_calls"]) == 2 else scenario["expected_status"],
            source_ids=["s0007"], reason="Finite test-only source classification fixture.")
    return result


@pytest.mark.parametrize("key", ["invoice", "release", "dispatch", "contacts", "assets"])
def test_reference_agrees_without_mutation_or_authority(key):
    case, sources, tree = fixture(key)
    before = tree.model_dump(mode="json")
    table = answers_for(case, sources, tree)
    report = compare(sources, tree, table)
    assert report["disagreements"] == report["unknowns"] == 0
    assert report["status"] == "finite_agreement_pending_source_review"
    assert tree.model_dump(mode="json") == before
    assert report["treeMutated"] is report["runtimeAuthorityGranted"] is False
    assert report["semanticAccuracy"] is None
    wire = request(sources, tree)
    payload = json.loads(wire["messages"][1]["content"])
    assert not {"draft", "tree", "suite", "oracle", "observations", "reference", "expectedStatus"} & payload.keys()
    assert payload["targetSkillSpans"] and payload["targetArguments"]
    assert "source_id" not in payload["targetArguments"]
    assert "source_draft_disagreement" not in json.dumps(payload)


def test_detects_lost_alternative_and_polarity_with_specific_trace():
    case, sources, tree = fixture("invoice")
    table = answers_for(case, sources, tree)
    # Source answers are independent of the mutation, so a lost alternative is
    # diagnosable instead of turning the faulty graph into its own oracle.
    raw = tree.model_dump(mode="json")
    raw["steps"][1]["otherwise"] = [dict(kind="end", source_id="s0008", outcome="unsupported")]
    mutated = FlowTree.model_validate(raw)
    report = compare(sources, mutated, table)
    assert report["disagreements"] == 1
    failure = next(r for r in report["rows"] if r["decision"] == "source_draft_disagreement")
    assert failure["assignment"] == dict(settled=False, waiver_exists=True)
    assert failure["trace"][-1]["treePointer"] == "/steps/1/otherwise/0"
    assert failure["citations"]


def test_unknown_never_converted_to_agreement():
    case, sources, tree = fixture("invoice")
    table = answers_for(case, sources, tree)
    table["v11"]["outcome"] = "unknown"
    result = compare(sources, tree, table)
    assert result["status"] == "needs_source_resolution"
    assert result["unknowns"] == 1


@pytest.mark.parametrize("mutation", ["missing", "extra", "citation", "outcome"])
def test_bad_answers_fail_closed(mutation):
    case, sources, tree = fixture("invoice")
    table = answers_for(case, sources, tree)
    if mutation == "missing":
        del table["v11"]
    elif mutation == "extra":
        table["v111"] = copy.deepcopy(table["v11"])
    elif mutation == "citation":
        table["v11"]["source_ids"] = ["fabricated"]
    else:
        table["v11"]["outcome"] = "activated"
    with pytest.raises(ValidationError):
        compare(sources, tree, table)


def test_no_capability_stop_to_fake_joint_coverage():
    _, sources, tree = fixture("archive")
    with pytest.raises(ValueError, match="two-read"):
        request(sources, tree)


def test_non_boolean_condition_remains_outside_scope():
    _, sources, tree = fixture("assets")
    raw = tree.model_dump(mode="json")
    raw["steps"][1]["left"]["field"] = "canonical_key"
    raw["steps"][1]["equals"]["value"] = "example"
    with pytest.raises(ValueError, match="non-Boolean"):
        region(sources, FlowTree.model_validate(raw))


@pytest.mark.parametrize("key", ["invoice", "release", "dispatch", "contacts", "assets"])
def test_explicit_revision_matches_existing_executor_and_preserves_inputs(key):
    case, sources, tree = fixture(key)
    table = answers_for(case, sources, tree)
    raw = tree.model_dump(mode="json")
    raw["steps"][1]["equals"]["value"] = not raw["steps"][1]["equals"]["value"]
    broken = FlowTree.model_validate(raw)
    before = broken.model_dump(mode="json")
    result = propose(sources, broken, table)
    assert result["status"] == "revised_inactive_pending_source_review"
    fixed = FlowTree.model_validate(result["tree"])
    assert check_behavior(sources, fixed, BehaviorSuite.model_validate(case["suite"]))["behavior"] == "matched_finite_oracle"
    assert broken.model_dump(mode="json") == before
    assert not result["runtimeAuthorityGranted"]
    assert result["fullSourceReview"] == "required_not_run"
    pointers = {o["treePointer"] for o in result["compilation"]["origins"]}
    assert all(o["treePointer"] in pointers for o in result["tableOrigins"])


def test_revision_is_explicit_and_unknown_or_no_read_witness_cannot_force_accept():
    case, sources, tree = fixture("invoice")
    table = answers_for(case, sources, tree)
    assert propose(sources, tree, table)["status"] == "unchanged_finite_agreement"
    table["v11"]["outcome"] = "unknown"
    assert "tree" not in propose(sources, tree, table)
    for row in table.values():
        row["outcome"] = "unsupported"
    assert propose(sources, tree, table)["status"] == "no_read_witness_requires_source_resolution"


@pytest.mark.parametrize("mask", range(16))
def test_all_two_fact_boolean_functions_lower_exactly_or_remain_unresolved(mask):
    case, sources, tree = fixture("invoice")
    table = answers_for(case, sources, tree)
    for index, row in enumerate(table.values()):
        row["outcome"] = "read" if mask & (1 << index) else "unsupported"
    result = propose(sources, tree, table)
    if not mask:
        assert "tree" not in result
    elif result["status"] == "unchanged_finite_agreement":
        assert compare(sources, tree, table)["disagreements"] == 0
    else:
        assert result["finiteVerification"]["disagreements"] == 0
        assert not result["runtimeAuthorityGranted"]
