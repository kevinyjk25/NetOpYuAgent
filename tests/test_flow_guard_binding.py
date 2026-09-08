"""Explicit repairs, complete dependency slots, no automatic authority."""

import copy
import json

import pytest
from jsonschema import ValidationError

from evaluation.flow_behavior_examples import cases
from evaluation.flow_behavior_probe import evaluate
from evaluation.flow_guard_binding import author_guards, bind_guards, guard_request, normalize_repeated_stop, slots_for
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree


def access_without_guards():
    case = cases()[-1]
    raw = copy.deepcopy(case["reference"])
    node = raw["steps"][1]
    while node["kind"] == "if_equal":
        tail = node["when_equal"]
        node = tail[0]
    raw["steps"] = [raw["steps"][0], *tail]
    return case, FlowSources.model_validate(case["sources"]), FlowTree.model_validate(raw)


def test_required_decisions_preserve_all_three_facts_and_repair_missing_guards():
    case, sources, tree = access_without_guards()
    slots = slots_for(sources, tree)
    assert [s["reference"]["field"] for s in slots] == ["available", "current", "granted"]
    decisions = {s["id"]: dict(decision="require_true", source_id="s0003", on_failure="unsupported") for s in slots}
    result = bind_guards(sources, tree, decisions)
    assert evaluate(case, result["tree"])["passed"] == 11
    assert not result["semanticAlignmentProven"] and not result["runtimeAuthorityGranted"]
    with pytest.raises(ValidationError):
        bind_guards(sources, tree, {k: v for k, v in decisions.items() if k != "g001"})
    wire = guard_request(sources, tree)
    assert wire["model"] == "qwen3.5:9b" and wire["think"] is False
    assert "expected_calls" not in json.dumps(wire) and "observations" not in json.dumps(wire)


def test_false_not_required_and_unknown_are_not_changed_into_all_true():
    _, sources, tree = access_without_guards()
    decisions = {s["id"]: dict(decision="not_individually_required", source_id="s0004") for s in slots_for(sources, tree)}
    assert bind_guards(sources, tree, decisions)["tree"] == tree.model_dump(mode="json")
    decisions["g000"] = dict(decision="require_false", source_id="s0004", on_failure="needs_l1")
    result = bind_guards(sources, tree, decisions)
    assert result["tree"]["steps"][1]["equals"]["value"] is False
    assert result["tree"]["steps"][1]["otherwise"][0]["outcome"] == "needs_l1"
    decisions["g001"] = dict(decision="unresolved", source_id="s0003")
    blocked = bind_guards(sources, tree, decisions)
    assert blocked["status"] == "unresolved_not_executable" and "tree" not in blocked


def test_no_boolean_slots_do_not_request_another_model_call():
    case = cases()[0]
    sources, tree = FlowSources.model_validate(case["sources"]), FlowTree.model_validate(case["reference"])
    assert slots_for(sources, tree) == []
    with pytest.raises(ValueError, match="no Boolean"):
        guard_request(sources, tree)


def test_only_redundant_stop_is_normalized_with_both_original_citations():
    case = cases()[-2]
    sources = FlowSources.model_validate(case["sources"])
    first = dict(kind="unavailable", source_id="s0002", question="Required reference and runner have not been supplied.")
    second = dict(kind="end", source_id="s0003", outcome="unsupported")
    proposal = dict(business_source_ids=["s0002"], steps=[first, second], issues=[])
    original = copy.deepcopy(proposal)
    result = normalize_repeated_stop(sources, proposal)
    assert proposal == original and len(result["edits"]) == 1
    assert result["edits"][0]["redundantStatement"] == second
    assert evaluate(case, result["lowering"]["tree"])["passed"] == 1
    for tail in (dict(kind="end", source_id="s0003", outcome="read_path_completed"),
            dict(kind="read", source_id="s0002", tool="read_backup_metadata", bind="r0", arguments={})):
        with pytest.raises(ValueError, match="unreachable"):
            normalize_repeated_stop(sources, {**proposal, "steps": [first, tail]})


def test_source_guard_generation_replays_and_never_retries_partial_checkpoint(tmp_path, monkeypatch):
    from evaluation import flow_checkpoint as parent
    _, sources, tree = access_without_guards()
    decisions = {s["id"]: dict(decision="require_true", source_id="s0003", on_failure="unsupported")
        for s in slots_for(sources, tree)}
    sent = []
    monkeypatch.setattr(parent.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b"})

    def send(arm, wire):
        sent.append(wire)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(decisions)), prompt_eval_count=2, eval_count=3)))
    monkeypatch.setattr(parent, "send", send)
    root = tmp_path / "guards"
    with pytest.raises(ValueError, match="budget"):
        author_guards(sources, tree, root)
    result = author_guards(sources, tree, root, max_new_calls=1)
    assert result == author_guards(sources, tree, root)
    assert len(sent) == 1
    (root / "receipt.json").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        author_guards(sources, tree, root, max_new_calls=1)
    assert len(sent) == 1
