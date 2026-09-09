"""Necessity derivation is deterministic; these tests are not model accuracy."""

import copy
import itertools
import json

import pytest
from jsonschema import ValidationError

from evaluation import flow_behavior_probe as parent
from evaluation import flow_checkpoint as checkpoint
from evaluation import flow_guard_counterfactual as cf
from evaluation.flow_guard_binding import slots_for
from tests.test_flow_guard_binding import access_without_guards


def answer(status):
    return dict(status=status, source_id="s0003", **(dict(on_failure="unsupported") if status == "forbidden" else {}))


@pytest.mark.parametrize("no,yes", list(itertools.product(("possible", "forbidden", "unknown"), repeat=2)))
def test_truth_pair_derivation_is_total_and_never_guesses(no, yes):
    _, sources, tree = access_without_guards()
    answers = {s["id"]: dict(if_false=answer(no), if_true=answer(yes)) for s in slots_for(sources, tree)}
    result = cf.bind(sources, tree, answers)
    expected = {("forbidden", "possible"): "require_true", ("possible", "forbidden"): "require_false",
        ("possible", "possible"): "not_individually_required"}.get((no, yes), "unresolved")
    assert all(r["decision"]["decision"] == expected for r in result["derivations"])
    assert not result["sufficiencyProven"] and not result["runtimeAuthorityGranted"]
    if expected == "unresolved":
        assert "tree" not in result["binding"]


def test_false_impossibility_binds_existing_guards_and_old_oracle_still_checks_behavior():
    case, sources, tree = access_without_guards()
    answers = {s["id"]: dict(if_false=answer("forbidden"), if_true=answer("possible")) for s in slots_for(sources, tree)}
    result = cf.bind(sources, tree, answers)
    assert parent.evaluate(case, result["binding"]["tree"])["passed"] == 11
    with pytest.raises(ValidationError):
        cf.bind(sources, tree, {k: v for k, v in answers.items() if k != "g001"})
    bad = copy.deepcopy(answers)
    bad["g001"]["if_false"]["source_id"] = "not-original-source"
    with pytest.raises(ValidationError):
        cf.bind(sources, tree, bad)


def test_request_has_exact_host_and_full_source_but_no_oracle_or_prior_verdict():
    _, sources, tree = access_without_guards()
    wire = cf.request(sources, tree)
    payload = json.loads(wire["messages"][1]["content"])
    assert wire["model"] == "qwen3.5:9b" and wire["think"] is False
    assert wire["options"]["num_predict"] == 4096
    assert all(line in payload["targetSkillSpans"].values() for line in sources.source_text.splitlines() if line.strip())
    assert set(payload) == {"hostInputSchema", "hostReadTools", "hostEffectTargets", "draft", "dependencySlots", "targetSkillSpans"}
    for slot in payload["dependencySlots"]:
        original = sources.reads[slot["factTool"]].spec.output_schema.properties[slot["reference"]["field"]]
        assert slot["factDefinition"]["schema"] == original.model_dump(mode="json")


def test_completed_author_replays_without_a_call_and_failures_never_retry(tmp_path, monkeypatch):
    _, sources, tree = access_without_guards()
    answers = {s["id"]: dict(if_false=answer("forbidden"), if_true=answer("possible")) for s in slots_for(sources, tree)}
    sent = []
    monkeypatch.setattr(checkpoint.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b"})

    def send(arm, wire):
        sent.append(wire)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True, done_reason="stop",
            message=dict(content=json.dumps(answers)), prompt_eval_count=2, eval_count=3)))
    monkeypatch.setattr(checkpoint, "send", send)
    root = tmp_path / "counterfactual"
    with pytest.raises(ValueError, match="budget"):
        cf.author(sources, tree, root)
    result = cf.author(sources, tree, root, max_new_calls=1)
    assert result == cf.author(sources, tree, root) and len(sent) == 1
    (root / "receipt.json").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        cf.author(sources, tree, root, max_new_calls=1)
    assert len(sent) == 1
