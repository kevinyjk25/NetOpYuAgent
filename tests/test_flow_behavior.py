"""Real inert Runtime traces and behavioral mutants; never synthetic AI verdicts."""

import copy
import json

import pytest

from evaluation.flow_behavior import BehaviorSuite, behavior_request, check_behavior
from evaluation.flow_behavior_examples import cases, end, read, ref
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree


def case_at(identifier):
    return next(c for c in cases() if c["id"] == identifier)


def check(case, tree=None):
    return check_behavior(FlowSources.model_validate(case["sources"]),
        FlowTree.model_validate(tree or case["reference"]), BehaviorSuite.model_validate(case["suite"]))


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_manual_witness_is_expressible_and_matches_private_finite_oracle(case):
    result = check(case)
    assert result["representation"] == "compiled"
    assert result["behavior"] == "matched_finite_oracle"
    assert result["inMemoryRuntimeRuns"] == len(case["suite"]["scenarios"])
    assert result["externalProviderCalls"] == result["sourceScriptsExecuted"] == 0
    assert not result["runtimeAuthorityGranted"] and result["semanticAccuracy"] is None
    assert result["fullSourceReview"] == "required_not_run"
    assert result["sourceArchive"] == case["sources"]["source_text"]


def test_actual_reversed_branch_and_wrong_argument_have_counterexamples_without_judge():
    case = case_at("inverted-branch")
    tree = copy.deepcopy(case["reference"])
    branch = tree["steps"][1]
    branch["when_equal"], branch["otherwise"] = branch["otherwise"], branch["when_equal"]
    result = check(case, tree)
    assert result["representation"] == "compiled" and result["behavior"] == "counterexample_found"
    assert not result["scenarios"][0]["passed"]
    assert all("treePointer" in s and "l0Pointer" in s for s in result["scenarios"][0]["trace"])
    tree = copy.deepcopy(case["reference"])
    tree["steps"][1]["otherwise"][0]["arguments"]["device_id"] = ref("input", "device_id")
    result = check(case, tree)
    counterexample = next(r for r in result["scenarios"] if r["id"] == "idc")
    assert counterexample["differences"][0]["callIndex"] == 1
    assert counterexample["differences"][0]["actual"][0]["arguments"] != counterexample["differences"][0]["expected"][0]["arguments"]


def test_stop_everywhere_does_not_score_as_success():
    case = case_at("direct-read")
    tree = {**case["reference"], "steps": [end("unsupported")]}
    result = check(case, tree)
    assert result["passed"] == 0 and result["behavior"] == "counterexample_found"
    assert result["scenarios"][0]["differences"][0]["facet"] == "operation_order_arguments_or_count"


def test_extra_dependency_work_before_missing_prerequisite_stop_is_detected():
    case = case_at("backup-prerequisite")
    tree = {**case["reference"], "steps": [read("read_backup_metadata", "unexpected", {}, "s0002"), end("unsupported", "s0002")]}
    result = check(case, tree)
    assert result["representation"] == "compiled" and result["passed"] == 0
    failure = result["scenarios"][0]
    assert failure["actualCalls"] == [dict(tool="read_backup_metadata", arguments={})]
    assert failure["expectedCalls"] == []  # Unexpected tool access is recorded even without a fixture response.
    assert result["scope"] == "safe_partial_stop" and result["wholeSkillTranslations"] == 0


def test_missing_owner_guard_fails_truth_table_even_when_types_and_citations_pass():
    case = case_at("access-guide")
    tree = copy.deepcopy(case["reference"])
    # Skip available/current, keeping only granted; all nodes still type-check.
    tree["steps"][1] = tree["steps"][1]["when_equal"][0]["when_equal"][0]
    result = check(case, tree)
    assert result["representation"] == "compiled"
    assert result["behavior"] == "counterexample_found"
    assert not next(r for r in result["scenarios"] if r["id"] == "0-0-1")["passed"]
    assert next(r for r in result["scenarios"] if r["id"] == "1-1-1")["passed"]


def test_valid_alpha_renaming_is_not_an_exact_json_answer_oracle():
    case = case_at("inverted-branch")
    tree = json.loads(json.dumps(case["reference"]).replace('"snapshot"', '"different_alias"'))
    assert check(case, tree)["behavior"] == "matched_finite_oracle"


def test_request_has_only_source_and_host_not_reference_oracle_or_prose_guards():
    case = case_at("access-guide")
    source = FlowSources.model_validate(case["sources"])
    original = behavior_request(source)
    case["suite"]["scenarios"][0]["expected_status"] = "needs_l1"
    case["reference"]["steps"] = [end("unsupported")]
    assert behavior_request(source) == original
    payload = json.loads(original["messages"][1]["content"])
    assert "expected_calls" not in json.dumps(payload) and "reference" not in payload and "suite" not in payload
    assert original["format"]["title"] == "FlowTree"
    assert "GuardedDuty" not in json.dumps(original["format"])
    assert original["model"] == "qwen3.5:9b" and original["think"] is False


def test_oracle_drift_uncovered_requirement_and_duplicate_scenario_reject():
    for field in ("digest", "requirement", "duplicate"):
        case = case_at("direct-read")
        if field == "digest":
            case["sources"]["source_text"] += "\nRead twice instead."
        elif field == "requirement":
            case["suite"]["requirements"]["uncovered"] = case["sources"]["source_text"].splitlines()[0]
        else:
            case["suite"]["scenarios"].append(case["suite"]["scenarios"][0])
        with pytest.raises(ValueError):
            check(case)


def test_invalid_graph_is_representation_failure_not_wrong_semantic_percentage():
    case = case_at("direct-read")
    tree = {**case["reference"], "steps": [end(), read("read_inventory_device", "late", dict(device_id=ref("input", "device_id")))]}
    result = check(case, tree)
    assert result["representation"] == "invalid_candidate" and result["behavior"] == "not_run"
    assert result["scenarios"] == []
