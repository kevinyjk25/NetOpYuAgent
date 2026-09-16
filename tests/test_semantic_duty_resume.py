import json
import pytest

from evaluation import semantic_duty_resume as resume
from evaluation.structured_binding_probe import write_artifacts
from tests.test_hybrid_duty_contract import fixture


def test_failed_contract_is_quarantined_not_partially_promoted(tmp_path, monkeypatch):
    raw, _, _, _, _, _ = fixture()
    monkeypatch.setattr(resume, "verify_receipt", lambda root: None)
    root = tmp_path / "example" / "plan/model/plan"
    source = resume.duty.source_input(resume.probe.build_review_input(raw))
    write_artifacts(root, {"request.json": {"wireRequest": {}, "governedRequest": {"inputs": source}}, "invalid.json": {"text": "a malformed contract"}})
    payload, state = resume.prepare_case(tmp_path, "example", raw)
    assert payload["candidate"]["draft"] == raw["candidate"]["draft"]
    assert state["plan"]["duties"] == [] and state["plan"]["overflow"]
    assert not state["plan"]["authorityGranted"]
    assert state["failedNodesNotRetried"] == ["plan/plan"]
    assert state["newFocusedCalls"] == 1 and state["priorAttempts"] == 1
    changed = {**raw, "original_task": "A different task with a recomputed freeze digest."}
    with pytest.raises(ValueError, match="actual model request"):
        resume.prepare_case(tmp_path, "example", changed)


def test_notes_projection_preserves_sources_and_separates_target():
    _, _, _, _, _, ctx = fixture()
    projected = resume.note_input(ctx)
    assert projected["sourceContext"] == ctx["sourceContext"]
    assert projected["readOnlyAnswer"] == ctx["completeCandidate"]["draft"]
    assert set(projected["notesToInspect"]) == {"n000"}
    assert not any(k.startswith("d") for k in projected["notesToInspect"])


def test_failed_edit_not_reinterpreted_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "verify_receipt", lambda root: None)
    tmp_path = tmp_path / "call"
    request = {"inputs": {"x": 1}, "outputSchema": {"type": "object"}, "instructions": "No authority."}
    write_artifacts(tmp_path, {"request.json": {"governedRequest": request}, "invalid.json": {"text": "bad"}})
    spec = {"input": request["inputs"], "schema": request["outputSchema"], "system": request["instructions"],
            "bind": lambda raw: (_ for _ in ()).throw(AssertionError("Must not rederive failed output"))}
    assert resume.admitted_old_output(tmp_path, spec) is None
    spec["input"] = {"x": 2}
    with pytest.raises(ValueError, match="frozen contract"):
        resume.admitted_old_output(tmp_path, spec)


def test_recovery_of_failed_plan_notes_only_and_one_shot_claim(tmp_path, monkeypatch):
    raw, _, _, _, _, _ = fixture()
    parent = tmp_path / "parent"
    monkeypatch.setattr(resume, "verify_receipt", lambda root: None)
    monkeypatch.setattr(resume.probe, "fingerprint", lambda: {})
    keys = ("actualChatAttempts", "inputTokens", "outputTokens", "callsWithUnknownInputUsage", "callsWithUnknownOutputUsage",
            "p50CallLatencyMs", "p95CallLatencyMs")
    monkeypatch.setattr(resume, "collect", lambda root: resume.seal({k: 0 for k in keys}))
    freeze = resume.seal({"profile": resume.duty.PROFILE, "live": True, "maxCallsPerTask": 17, "inputs": {"example": raw}})
    write_artifacts(parent / "freeze", {"manifest.json": freeze})
    write_artifacts(parent / "example/checkpoint", {"report.json": resume.seal({"status": "stopped_without_retry"})})
    source = resume.duty.source_input(resume.probe.build_review_input(raw))
    write_artifacts(parent / "example/plan/model/plan", {"request.json": {"wireRequest": {}, "governedRequest": {"inputs": source}}})
    calls = []

    def fake_phase(folder, specs, pin):
        calls.extend((folder.name, s["id"]) for s in specs)
        if folder.name == "focused":
            return {"notes": {"notes": {"n000": {"needsInspection": True, "atoms": []}}}}
        if folder.name == "edit":
            return {s["id"]: s["bind"]({"operation": "keep", "replacement": "", "source_span_ids": [],
                                       "rationale": "Valid uncertainty remains unchanged."}) for s in specs}
        return {"review": {"notes": {}, "semanticApproval": False}}

    monkeypatch.setattr(resume.probe, "phase", fake_phase)
    report = resume.run(parent, tmp_path / "resumed", live=True)
    assert report["rows"][0]["status"] == "independent_siblings_completed_not_original_pass"
    assert calls == [("focused", "notes"), ("edit", "u000"), ("final", "review")]
    assert not report["stageExitMet"] and not report["failedCallsRetried"]
    assert json.loads((tmp_path / "resumed/example/materialized/candidate.json").read_text()) == raw["candidate"]
    with pytest.raises(FileExistsError):
        resume.run(parent, tmp_path / "another-output", live=True)
