"""Checkpoint mechanics, not model-quality evidence."""

import copy
import json

import pytest

from evaluation import flow_behavior_probe as pilot
from evaluation.flow_behavior_examples import cases


def setup(tmp_path, monkeypatch):
    case = cases()[0]
    monkeypatch.setattr(pilot, "cases", lambda: [copy.deepcopy(case)])
    monkeypatch.setattr(pilot, "implementation", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot, "environment", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight",
        lambda self: {"model": "qwen3.5:9b", "modelArtifactDigest": "test"})
    return tmp_path / "batch", case


def test_frozen_oracle_never_sent_and_completed_run_never_repeated(tmp_path, monkeypatch):
    root, case = setup(tmp_path, monkeypatch)
    manifest = pilot.freeze(root)
    assert not pilot.report(root)["completed"]
    sent = []

    def send(arm, wire):
        sent.append(wire)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(case["reference"])), prompt_eval_count=2, eval_count=3)))

    monkeypatch.setattr(pilot, "send", send)
    pilot.run(root, 1)
    pilot.run(root, 0)
    assert sent == [manifest["cases"][0]["request"]]
    result = pilot.report(root)
    assert result["completed"] and result["groups"]["executable_fragment"]["matched"] == 1
    assert result["wholeSkillTranslations"] == 0 and not result["runtimeAuthorityGranted"]
    assert result == pilot.report(root)


def test_invalid_reference_stops_before_model_preflight(tmp_path, monkeypatch):
    root, case = setup(tmp_path, monkeypatch)
    case["reference"]["steps"] = [dict(kind="end", source_id="s0003", outcome="unsupported")]
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("no model preflight"))
    with pytest.raises(ValueError, match="manual feasibility"):
        pilot.freeze(root)
    assert not root.exists()


def test_budget_and_uncertain_checkpoint_cannot_trigger_retry(tmp_path, monkeypatch):
    root, case = setup(tmp_path, monkeypatch)
    pilot.freeze(root)
    monkeypatch.setattr(pilot, "send", lambda *a: pytest.fail("no model call"))
    with pytest.raises(ValueError, match="budget"):
        pilot.run(root, 0)
    (root / case["id"]).mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        pilot.run(root, 1)


def test_transport_error_not_reported_as_semantic_counterexample(tmp_path, monkeypatch):
    root, _ = setup(tmp_path, monkeypatch)
    pilot.freeze(root)
    monkeypatch.setattr(pilot, "send", lambda *a: dict(httpStatus=503, latencyMs=1, body="unavailable"))
    with pytest.raises(ValueError, match="transport"):
        pilot.run(root, 1)
    row = pilot.report(root)["rows"][0]
    assert row["status"] == "transport_error" and row["behavior"] is None


def test_compact_request_preserves_source_host_types_and_existing_ir():
    from jsonschema import Draft202012Validator
    from evaluation.flow_behavior import behavior_request
    from evaluation.flow_behavior_compact import compact_request
    from evaluation.flow_translation import FlowSources
    for case in cases():
        source = FlowSources.model_validate(case["sources"])
        old, new = behavior_request(source), compact_request(source)
        old_payload, new_payload = (json.loads(w["messages"][1]["content"]) for w in (old, new))
        assert new_payload["targetSkillSpans"] == old_payload["targetSkillSpans"]
        assert new_payload["hostInputSchema"] == old_payload["hostInputSchema"]
        assert new_payload["runtimePolicy"] == old_payload["runtimePolicy"]
        assert new["options"] == old["options"] and new["think"] == old["think"]
        assert list(new["format"]["properties"]) == ["steps", "issues", "business_source_ids"]
        assert new["format"]["$defs"] == old["format"]["$defs"]
        Draft202012Validator(new["format"]).validate(case["reference"])


def test_compact_reuses_same_oracle_preserves_parent_and_replays_without_calls(tmp_path, monkeypatch):
    from evaluation import flow_behavior_compact as compact
    root, case = setup(tmp_path, monkeypatch)
    pilot.freeze(root)
    original = (root / "manifest.json").read_bytes()
    calls_sent = []

    def send(arm, wire):
        calls_sent.append(wire)
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(case["reference"])), prompt_eval_count=2, eval_count=3)))

    monkeypatch.setattr(pilot, "send", send)
    new_root = tmp_path / "compact"
    compact.run(new_root, root, 1, version=2)
    compact.run(new_root, root, 0)
    assert len(calls_sent) == 1 and (root / "manifest.json").read_bytes() == original
    result = compact.report(new_root, root)
    assert result["groups"]["executable_fragment"]["matched"] == 1
    assert result["rows"][0]["behavior"]["suiteDigest"] == pilot.evaluate(case, case["reference"])["suiteDigest"]
    assert result == compact.report(new_root, root)


def test_thinking_arm_changes_only_the_think_switch():
    from evaluation.flow_behavior import behavior_request
    from evaluation.flow_behavior_compact import compact_request
    from evaluation.flow_translation import FlowSources
    for case in cases():
        source = FlowSources.model_validate(case["sources"])
        assert compact_request(source, 3) == {**behavior_request(source), "think": True}
