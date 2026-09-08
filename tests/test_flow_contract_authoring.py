"""Compiler invariants; witnesses are not measurements of model accuracy."""

import copy
import json

import pytest
from jsonschema import ValidationError

from evaluation import flow_contract_probe as probe
from evaluation.flow_behavior_examples import cases
from evaluation.flow_contract_authoring import constructor_request, literal_catalog, lower_constructors
from evaluation.flow_translation import FlowSources


def renamed(tree):
    tree = copy.deepcopy(tree)
    names = {}

    def visit(value):
        if isinstance(value, dict):
            if value.get("kind") == "read":
                names[value["bind"]] = "r" + str(len(names))
                value["bind"] = names[value["bind"]]
            if value.get("kind") == "reference" and value["source"] != "input":
                value["source"] = names[value["source"]]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(tree)
    return tree


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_existing_witnesses_keep_identical_behavior(case):
    sources = FlowSources.model_validate(case["sources"])
    result = lower_constructors(sources, renamed(case["reference"]))
    actual = probe.parent.evaluate(case, result["tree"])
    assert actual["behavior"] == "matched_finite_oracle"
    assert not result["semanticAlignmentProven"] and not result["runtimeAuthorityGranted"]
    wire = constructor_request(sources)
    payload = json.loads(wire["messages"][1]["content"])
    assert "expected_calls" not in json.dumps(payload)
    assert payload["targetSkillSpans"]
    for literal in literal_catalog(sources):
        text = payload["targetSkillSpans"][literal["source_id"]][literal["start"]:literal["end"]]
        assert text == literal["value"] or json.loads(text) == literal["value"]


def access():
    case = cases()[-1]
    return case, FlowSources.model_validate(case["sources"]), renamed(case["reference"])


def test_flat_conjunction_lowers_without_dropping_any_precondition():
    case, sources, tree = access()
    branch = tree["steps"][1]
    tests = []
    while branch["kind"] == "if_equal":
        tests.append({k: branch[k] for k in ("source_id", "left", "equals")})
        following = branch["when_equal"]
        branch = following[0]
    tree["steps"] = [tree["steps"][0], dict(kind="require_all", tests=tests,
        failure_source_id="s0004", on_failure="unsupported"), *following]
    result = lower_constructors(sources, tree)
    assert [step["left"]["field"] for step in result["tree"]["steps"] if step["kind"] == "if_equal"] == ["available", "current", "granted"]
    assert probe.parent.evaluate(case, result["tree"])["passed"] == 11
    assert len([r for r in result["constructorOrigins"] if r["constructorPointer"] == "/steps/1"]) == 3
    assert all(r["lexicalWitnesses"] for r in result["argumentBindings"])


@pytest.mark.parametrize("value", [dict(kind="reference", source="input", field="request_id"),
    dict(kind="constant", value="invented-not-in-source"), dict(kind="constant", value=42),
    dict(kind="reference", source="hostInputSchema", field="request_id")])
def test_empty_caller_schema_cannot_gain_an_input_or_invent_literal(value):
    _, sources, tree = access()
    tree["steps"][0]["arguments"]["request_id"] = value
    with pytest.raises(ValidationError):
        lower_constructors(sources, tree)


def test_tool_arguments_and_available_aliases_remain_checked():
    _, sources, original = access()
    for arguments in ({}, dict(request_id=dict(kind="constant", value="Q6"), extra=dict(kind="constant", value="Q6"))):
        tree = copy.deepcopy(original)
        tree["steps"][0]["arguments"] = arguments
        with pytest.raises(ValidationError):
            lower_constructors(sources, tree)
    original["steps"][0]["arguments"]["request_id"] = dict(kind="reference", source="r9", field="value")
    with pytest.raises(ValueError, match="lexical scope"):
        lower_constructors(sources, original)


def test_missing_capability_is_an_explicit_terminal_not_silent_repair():
    case = cases()[-2]
    sources = FlowSources.model_validate(case["sources"])
    tree = dict(business_source_ids=["s0002"], steps=[dict(kind="unavailable", source_id="s0002",
        question="The required reference and its runner have not been supplied.")], issues=[])
    result = lower_constructors(sources, tree)
    assert result["tree"]["issues"][0]["kind"] == "missing_host_capability"
    assert probe.parent.evaluate(case, result["tree"])["passed"] == 1
    tree["steps"].append(dict(kind="read", source_id="s0002", tool="read_backup_metadata", bind="r0", arguments={}))
    with pytest.raises(ValueError, match="unreachable"):
        lower_constructors(sources, tree)


def test_literal_names_are_not_special_cased():
    _, sources, tree = access()
    raw = sources.model_dump()
    raw["source_text"] = raw["source_text"].replace("Q6", "tenant-72")
    sources = FlowSources.model_validate(raw)
    with pytest.raises(ValidationError):
        lower_constructors(sources, tree)
    raw_tree = json.loads(json.dumps(tree).replace('"Q6"', '"tenant-72"'))
    result = lower_constructors(sources, raw_tree)
    assert result["argumentBindings"][0]["lexicalWitnesses"][0]["value"] == "tenant-72"


def test_checkpoint_is_idempotent_and_preserves_original_oracle(tmp_path, monkeypatch):
    from tests.test_flow_behavior_probe import setup
    parent_root, case = setup(tmp_path, monkeypatch)
    probe.parent.freeze(parent_root)
    sent = []

    def send(arm, wire):
        sent.append(wire)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(renamed(case["reference"]))), prompt_eval_count=2, eval_count=3)))
    monkeypatch.setattr(probe.parent, "send", send)
    root = tmp_path / "constructors"
    probe.run(root, parent_root, 1)
    probe.run(root, parent_root, 0)
    result = probe.report(root, parent_root)
    assert result["matched"] == 1 and len(sent) == 1
    assert result["rows"][0]["behavior"]["suiteDigest"] == probe.parent.evaluate(case, case["reference"])["suiteDigest"]
    (root / case["id"] / "receipt.json").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        probe.run(root, parent_root, 0)
    assert len(sent) == 1
