import json

import pytest

from evaluation import semantic_witness_probe as probe
from evaluation import hybrid_semantic_witness as w
from evaluation.flow_tree_authoring import receipt
from evaluation.structured_binding_probe import write_artifacts
from tests.test_hybrid_draft_review import inputs


def setup(tmp_path, monkeypatch):
    original = tmp_path / "original"
    write_artifacts(original, {"request.json": {"governedRequest": {"inputs": inputs()}}})
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"example": str(original / "request.json")}))
    # Only the historical source fixture lacks a live receipt. New model calls
    # below get genuine byte receipts, but synthetic responses, never HTTP.
    real_verify = probe.verify_receipt
    monkeypatch.setattr(probe, "verify_receipt", lambda folder: None if folder == original else real_verify(folder))
    monkeypatch.setattr(probe, "fingerprint", lambda: {})
    calls = []

    def offline_once(root, supplied, derive, **kwargs):
        wire = supplied["wireRequest"]
        calls.append(wire)
        content = json.loads(wire["messages"][-1]["content"])
        if wire["messages"][0]["content"] == w.PLAN_SYSTEM:
            source = content["originalSourceContext"]
            raw = {"source_digest": source["reportDigest"], "requirements": [{
                "task_quote": source["originalTask"], "required_content_or_boundary": "Synthetic requirement."}],
                "observations": {s["source_span_id"]: {"usable_content": "Synthetic", "limits": "Unverified"}
                                 for s in source["sourceSpans"] if s["kind"] == "observation"}}
        else:
            view = content["witnessInput"]
            raw = {"input_digest": view["reportDigest"], "requirements": {r["id"]: {
                "artifact_location": "", "artifact_quote": "", "explanation": "No semantic proof.", "outcome": "unknown", "correction": ""}
                for r in view["unverifiedPlan"]["requirements"]},
                "observation_reconciliation": {s["source_span_id"]: {"candidate_location": "", "candidate_quote": "",
                    "source_quote": "", "explanation": "No semantic proof.", "outcome": "unknown", "correction": ""}
                    for s in view["originalSourceContext"]["sourceSpans"] if s["kind"] == "observation"},
                "other_problems": [], "scope_note": "Synthetic provider fixture only."}
        response = {"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5, "message": {"content": json.dumps(raw)}})}
        files, result = derive(response)
        write_artifacts(root, {"request.json": {**supplied, "model": {}}, "response.json": response, "result.json": result, **files})
        (root / "receipt.json").write_text(json.dumps(receipt(root)))
        return {"result": result, **files}

    monkeypatch.setattr(probe, "author_once", offline_once)
    return manifest, calls


def stopped_after_plan(manifest, output, monkeypatch):
    real_call = probe.call

    def interrupted(folder, wire, binder):
        if folder.name == "check":
            raise RuntimeError("simulated local interruption before HTTP")
        return real_call(folder, wire, binder)

    with monkeypatch.context() as scoped:
        scoped.setattr(probe, "call", interrupted)
        with pytest.raises(RuntimeError):
            probe.run(manifest, output, max_model_calls=2)


def test_two_phase_run_saves_distinct_input_directories_and_exact_receipts(tmp_path, monkeypatch):
    manifest, calls = setup(tmp_path, monkeypatch)
    output = tmp_path / "first"
    report = probe.run(manifest, output, max_model_calls=2)
    assert len(calls) == report["actualChatAttempts"] == 2
    assert (output / "example/inputs/source-input.json").is_file()
    assert (output / "example/check-inputs/check-input.json").is_file()
    assert not report["stageExitMet"]
    with pytest.raises(FileExistsError):
        probe.run(manifest, output, max_model_calls=2)


def test_explicit_plan_reuse_never_repeats_completed_analysis(tmp_path, monkeypatch):
    manifest, calls = setup(tmp_path, monkeypatch)
    old = tmp_path / "old"
    stopped_after_plan(manifest, old, monkeypatch)
    report = probe.run(manifest, tmp_path / "recovery", max_model_calls=1, reuse_plans=old)
    assert len(calls) == 2 and report["actualChatAttempts"] == 1
    assert report["retainedPriorChatAttempts"] == 1 and report["retainedPriorInputTokens"] == 10
    assert report["rows"][0]["planReusedWithoutCall"]


def test_model_prompt_change_cannot_reuse_plan(tmp_path, monkeypatch):
    manifest, _ = setup(tmp_path, monkeypatch)
    old = tmp_path / "old"
    stopped_after_plan(manifest, old, monkeypatch)
    monkeypatch.setattr(w, "PLAN_SYSTEM", w.PLAN_SYSTEM + "\nChanged experiment.")
    with pytest.raises(ValueError, match="drift"):
        probe.run(manifest, tmp_path / "recovery", max_model_calls=1, reuse_plans=old)


def test_reuse_does_not_repeat_an_already_attempted_comparison(tmp_path, monkeypatch):
    manifest, calls = setup(tmp_path, monkeypatch)
    old = tmp_path / "old"
    probe.run(manifest, old, max_model_calls=2)
    with pytest.raises(ValueError, match="already attempted"):
        probe.run(manifest, tmp_path / "recovery", max_model_calls=1, reuse_plans=old)
    assert len(calls) == 2


def test_default_is_zero_call_preflight(tmp_path, monkeypatch):
    manifest, calls = setup(tmp_path, monkeypatch)
    report = probe.run(manifest, tmp_path / "dry")
    assert not calls and report["actualChatAttempts"] == 0
