"""Mechanism regression on declared fixtures, not additional independent Skills."""

import copy
import json

import pytest
from jsonschema import ValidationError

from evaluation import flow_checkpoint as checkpoint
from evaluation import flow_semantic_probe as probe
from evaluation.flow_behavior_probe import evaluate
from evaluation.flow_contract_authoring import author_candidate, citable_source_ids, constructor_request, lower_constructors
from evaluation.flow_semantic_examples import cases, followup_cases
from evaluation.flow_translation import FlowSources
from tests.test_flow_contract_authoring import renamed


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_new_package_reference_is_feasible_with_existing_executor(case):
    assert evaluate(case, case["reference"])["behavior"] == "matched_finite_oracle"
    source = FlowSources.model_validate(case["sources"])
    wire = constructor_request(source)
    assert wire["model"] == "qwen3.5:9b" and wire["think"] is False
    text = json.dumps(wire)
    assert "expected_calls" not in text and "oracle_author" not in text and "observations" not in text
    assert source.source_text == case["sources"]["source_text"]
    assert not lower_constructors(source, renamed(case["reference"]))["runtimeAuthorityGranted"]


def invoice_macro():
    case = cases()[0]
    tree = renamed(case["reference"])
    first = tree["steps"][1]
    second = first["otherwise"][0]
    tests = [{k: node[k] for k in ("source_id", "left", "equals")} for node in (first, second)]
    proposal = dict(business_source_ids=tree["business_source_ids"], issues=[], steps=[tree["steps"][0],
        dict(kind="require_any", tests=tests, failure_source_id=first["false_source_id"], on_failure="unsupported"),
        *first["when_equal"]])
    return case, FlowSources.model_validate(case["sources"]), proposal


def test_alternatives_are_not_conjunctions_and_common_read_is_not_duplicated():
    case, source, raw = invoice_macro()
    compiled = lower_constructors(source, raw)
    assert evaluate(case, compiled["tree"])["passed"] == 7
    assert len([n for n in compiled["compilation"]["flow"]["nodes"] if n["kind"] == "read"]) == 2
    assert {x["treePointer"] for x in compiled["constructorOrigins"] if x["constructorPointer"] == "/steps/1"} == {
        "/steps/1", "/steps/1/otherwise/0"}
    wrong = copy.deepcopy(raw)
    wrong["steps"][1]["kind"] = "require_all"
    assert evaluate(case, lower_constructors(source, wrong)["tree"])["passed"] == 5


def test_alternative_predicates_can_require_false_without_inverting_failure():
    _, source, raw = invoice_macro()
    raw["steps"][1]["tests"][0]["equals"]["value"] = False
    result = lower_constructors(source, raw)
    first = result["tree"]["steps"][1]
    assert first["equals"]["value"] is False and first["when_equal"] == []
    assert first["otherwise"][0]["equals"]["value"] is True


@pytest.mark.parametrize("mutation", ["empty", "unknown-field", "unknown-source", "wrong-type"])
def test_alternatives_do_not_weaken_schema_or_source_checks(mutation):
    _, source, raw = invoice_macro()
    tests = raw["steps"][1]["tests"]
    if mutation == "empty":
        tests.clear()
    elif mutation == "unknown-field":
        tests[0]["left"]["field"] = "invented"
    elif mutation == "unknown-source":
        tests[0]["source_id"] = "invented"
    else:
        tests[0]["equals"]["value"] = "not-a-boolean"
    with pytest.raises((ValueError, ValidationError)):
        lower_constructors(source, raw)


def test_probe_is_budgeted_replayable_and_cannot_expand_to_a_bulk_run(tmp_path, monkeypatch):
    case = cases()[-1]  # Declared missing runner, no second model phase.
    source_file = tmp_path / "cases.json"
    source_file.write_text(json.dumps([case]))
    monkeypatch.setattr(checkpoint.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b"})
    root = tmp_path / "probe"
    probe.freeze(source_file, root)
    calls = []

    def send(arm, wire):
        calls.append(wire)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(renamed(case["reference"]))), prompt_eval_count=1, eval_count=1)))

    monkeypatch.setattr(checkpoint, "send", send)
    probe.run(root, 0)
    assert not calls
    probe.run(root, 1)
    result = probe.report(root)
    probe.run(root, 0)
    assert result == probe.report(root) and len(calls) == 1
    assert result["metrics"]["firstBehavior"]["safeStopMatched"] == 1
    assert not result["runtimeAuthorityGranted"] and result["semanticAccuracy"] is None
    (root / case["id"] / "generation" / "receipt.json").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        probe.run(root, 1)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="bulk"):
        probe.validate_cases([case] * 13)


@pytest.mark.parametrize("case", followup_cases(), ids=lambda c: c["id"])
def test_followup_references_and_request_isolation(case):
    assert evaluate(case, case["reference"])["passed"] == len(case["suite"]["scenarios"])
    payload = json.dumps(constructor_request(FlowSources.model_validate(case["sources"])))
    assert "expected_calls" not in payload and "oracle_author" not in payload
    assert "observations" not in payload


def test_quote_constraints_are_enforced_before_generation_not_as_semantic_truth():
    case = followup_cases()[3]  # Preserve the real failed source shape, not its old answer.
    source = FlowSources.model_validate(case["sources"])
    assert "s0001" not in citable_source_ids(source)  # Markdown delimiter `---`.
    wire = constructor_request(source)
    assert "s0001" in json.loads(wire["messages"][1]["content"])["targetSkillSpans"]
    tree = renamed(case["reference"])
    tree["business_source_ids"] = ["s0001"]
    with pytest.raises(ValidationError):
        lower_constructors(source, tree)


def test_compiler_feedback_is_explicit_and_never_takes_an_oracle(tmp_path, monkeypatch):
    case = cases()[0]
    source = FlowSources.model_validate(case["sources"])
    valid = renamed(case["reference"])
    broken = copy.deepcopy(valid)
    first = broken["steps"][1]
    first["otherwise"][0]["when_equal"][0]["bind"] = first["when_equal"][0]["bind"]
    monkeypatch.setattr(checkpoint.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b"})
    calls = []

    def send(arm, wire):
        calls.append(wire)
        payload = json.loads(wire["messages"][1]["content"])
        assert "globally unique" in payload["compilerRejection"]
        assert payload["rejectedProposal"] == broken
        assert "expected_calls" not in json.dumps(payload)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(valid)), prompt_eval_count=1, eval_count=1)))

    monkeypatch.setattr(checkpoint, "send", send)
    root = tmp_path / "explicit-repair"
    with pytest.raises(ValueError, match="genuinely rejected"):
        author_candidate(source, root, max_new_calls=1, rejected_proposal=valid)
    assert not calls
    result = author_candidate(source, root, max_new_calls=1, rejected_proposal=broken)
    assert result == author_candidate(source, root, rejected_proposal=broken)
    assert len(calls) == 1 and result["result"]["candidateStatus"] == "inactive_candidate"
    with pytest.raises(ValueError, match="drift"):
        author_candidate(source, root)
