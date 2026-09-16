from copy import deepcopy
import json
import pytest

from evaluation import semantic_duty_probe as probe
from evaluation.hybrid_draft_review import build_review_input
from tests.test_hybrid_duty_contract import fixture


def test_phase_uses_actual_runtime_and_stops_at_invalid_binding(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "fingerprint", lambda: {})

    def fake(folder, request, derive, **kwargs):
        calls.append(request["governedRequest"]["nodeId"])
        envelope = {"model": probe.author.MODEL, "message": {"content": '{"ok":true}'},
                    "prompt_eval_count": 1, "eval_count": 1, "done": True, "done_reason": "stop"}
        files, cost = derive({"body": json.dumps(envelope), "latencyMs": 1, "httpStatus": 200})
        return {**files, "result": cost}

    monkeypatch.setattr(probe, "author_once", fake)
    spec = {"id": "a", "input": {"inert": True}, "schema": probe.duty.obj({"ok": {"type": "boolean"}}),
            "system": "No actions.", "bind": lambda raw: raw}
    assert probe.phase(tmp_path / "ok", [spec], {}) == {"a": {"ok": True}}
    bad = {**spec, "bind": lambda raw: (_ for _ in ()).throw(ValueError("invalid binding"))}
    with pytest.raises(ValueError, match="phase stopped"):
        probe.phase(tmp_path / "bad", [bad, {**spec, "id": "b"}], {})
    assert calls == ["a", "a"]


def test_note_repair_not_gated_by_positive_body_review_and_shared_budget():
    raw, _, _, _, plan, _ = fixture()
    raw["candidate"]["notes"] = [f"Unverified note {i}" for i in range(10)]
    payload = build_review_input(raw)
    reviewed_notes = {"notes": {f"n{i:03d}": {"needsInspection": True, "atoms": []} for i in range(10)}}
    specs, owned, _, deferred = probe.edit_specs(payload, plan, [], reviewed_notes, {"checks": []})
    assert len(specs) == 8 and len(deferred) == 2
    assert all(s["edit"] and owned[s["id"]]["kind"] == "note" for s in specs)
    assert "ownedNote" in specs[0]["input"]["editTarget"]


def test_local_failure_schedules_exact_body_fragment_without_llm_negative():
    _, payload, _, _, plan, _ = fixture()
    span = payload["draftSpans"][1]
    checks = {"checks": [{"status": "fail", "location": span["draft_span_id"], "artifactQuote": span["exactQuote"],
                           "explanation": "Test check only.", "check": "fixture"}]}
    specs, owned, _, _ = probe.edit_specs(payload, plan, [], {"notes": {}}, checks)
    assert specs and all(owned[s["id"]]["kind"] == "body" for s in specs)


def test_materialization_preserves_values_and_original_notes(tmp_path):
    raw, payload, _, _, plan, _ = fixture()
    specs, owned, editable, _ = probe.edit_specs(payload, plan, [], {"notes": {"n000": {"needsInspection": True, "atoms": []}}}, {"checks": []})
    edit = {"operation": "replace", "replacement": "A corrected uncertainty.", "source_span_ids": ["s000"],
            "rationale": "Scoped single-note fixture repair."}
    bound = specs[0]["bind"](edit)
    candidate, application = probe.materialize(payload, raw["candidate"]["values"], editable, owned, {specs[0]["id"]: bound})
    assert candidate["values"] == raw["candidate"]["values"] and candidate["draft"] == raw["candidate"]["draft"]
    assert candidate["notes"] == ["A corrected uncertainty."]
    assert application["originalNotes"] == raw["candidate"]["notes"] and not application["hostDutiesCleared"]


def test_unknown_final_positive_never_admits():
    _, _, _, _, plan, ctx = fixture()
    raw = {"duties": {"r000": {"outcome": "met", "artifact_location": "", "artifact_quote": "",
        "explanation": "Unsupported positive.", "issues": []}}, "notes": {"n000": {"atoms": [{
            "quote": ctx["candidateLocations"]["n000"], "status": "supported", "evidence": [], "explanation": "No source."}]}}}
    result = probe.bind_final(ctx, plan, deepcopy(raw))
    assert result["duties"]["r000"]["outcome"] == "unknown"
    assert result["notes"]["notes"]["n000"]["needsInspection"]
    assert not result["semanticApproval"]


def test_complete_control_path_materializes_and_rechecks_without_clearing_duties(tmp_path, monkeypatch):
    raw, _, _, _, _, _ = fixture()
    seen = []

    def fake_phase(folder, specs, pin):
        out = {}
        for spec in specs:
            seen.append((folder.name, spec["id"]))
            if folder.name == "plan":
                reply = {"duties": [{"task_quote": raw["original_task"], "description": "Read and draft."}],
                         "overflow": False, "unrepresented_duties": ""}
            elif spec.get("edit"):
                reply = {"operation": "keep", "replacement": "", "source_span_ids": [],
                         "rationale": "Keep a valid uncertainty; no granted authority."}
            else:
                ctx = spec["input"]
                note_reply = {k: {"atoms": [{"quote": v, "status": "unknown", "evidence": [], "explanation": "Mechanical fixture only."}]}
                              for k, v in ctx["candidateLocations"].items() if k.startswith("n")}
                check_reply = {"outcome": "unknown", "artifact_location": "", "artifact_quote": "", "issues": [],
                               "explanation": "Unknown semantics, never task approval."}
                reply = {"duties": {"r000": check_reply}, "notes": note_reply} if folder.name == "final" else note_reply if spec["id"] == "notes" else check_reply
            out[spec["id"]] = spec["bind"](reply)
        return out

    monkeypatch.setattr(probe, "phase", fake_phase)
    report = probe.case_run(tmp_path, raw, {})
    assert seen == [("plan", "plan"), ("focused", "r000"), ("focused", "notes"), ("edit", "u000"), ("final", "review")]
    materialized = json.loads((tmp_path / "materialized/candidate.json").read_text())
    assert materialized == raw["candidate"]
    assert report["final"]["duties"]["r000"]["outcome"] == "unknown"
    assert not report["stageExitMet"] and not report["completeAnswerApproved"]


def test_existing_output_and_foreign_answer_manifest_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"test": {"gold": "injected"}}))
    with pytest.raises(ValueError, match="no expected answers"):
        probe.run(manifest, tmp_path / "output")
    with pytest.raises(FileExistsError):
        probe.run(manifest, tmp_path)


def test_successful_checkpoint_keeps_linked_and_self_digests_separate(tmp_path, monkeypatch):
    raw, _, _, _, _, _ = fixture()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"case": {"source": "/fixture/source", "request": "/fixture/request", "taskScope": []}}))
    monkeypatch.setattr(probe, "import_candidate_context", lambda *args: (None, {}, raw, None))
    monkeypatch.setattr(probe, "fingerprint", lambda: {})
    case_report = probe.seal({"status": "fixture_only", "semanticApproval": False})
    result = probe.run(manifest, tmp_path / "out", live=True, case_runner=lambda *args: case_report)
    checkpoint = json.loads((tmp_path / "out/case/checkpoint/report.json").read_text())
    assert checkpoint["caseReportDigest"] == case_report["reportDigest"]
    probe.duty.verify(checkpoint)
    assert result["actualChatAttempts"] == 0
