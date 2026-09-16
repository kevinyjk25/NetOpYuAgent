import json
import pytest

from evaluation import semantic_typed_duty_probe as p
from evaluation.structured_binding_probe import write_artifacts
from tests.test_hybrid_typed_duties import fixture


def test_call_budget_unchanged_and_no_plan_model_call():
    assert sum(p.PHASE_CAPS.values()) == 16 <= 17
    assert p.PHASE_CAPS["sharedBodyNotesEdit"] == 8
    raw, _, _ = fixture()
    preflight = p.preflight(raw)
    assert preflight["hostContractModelCalls"] == 0 and preflight["taskAnchorCount"] == 4
    assert len(preflight["taskCheckBudgets"]) == 3


def test_independent_failed_note_never_retried_or_blocks_body(tmp_path, monkeypatch):
    attempts = []

    def phase(folder, specs, pin, **kwargs):
        key = specs[0]["id"]
        attempts.append(key)
        if key == "note":
            write_artifacts(folder / "summary", {"report.json": {"status": "failed"}})
            raise ValueError("failed binding")
        return {key: {"kept": True}}

    monkeypatch.setattr(p.probe, "phase", phase)
    monkeypatch.setattr(p, "fingerprint", lambda: {})
    failures = []
    result = p.independent(tmp_path, [{"id": "note"}, {"id": "body"}], {}, failures)
    assert attempts == ["note", "body"] and result == {"body": {"kept": True}}
    assert len(failures) == 1 and not failures[0]["retried"]


def test_source_drift_cannot_be_treated_as_independent_failure(tmp_path, monkeypatch):
    def phase(folder, specs, pin, **kwargs):
        raise ValueError("drift")
    monkeypatch.setattr(p.probe, "phase", phase)
    monkeypatch.setattr(p, "fingerprint", lambda: {"changed": True})
    with pytest.raises(ValueError, match="source drift"):
        p.independent(tmp_path, [{"id": "a"}, {"id": "b"}], {}, [])


def test_unretained_phase_error_not_hidden(tmp_path, monkeypatch):
    def phase(*args, **kwargs):
        raise ValueError("pre-execution error")
    monkeypatch.setattr(p.probe, "phase", phase)
    monkeypatch.setattr(p, "fingerprint", lambda: {})
    with pytest.raises(ValueError, match="pre-execution"):
        p.independent(tmp_path, [{"id": "a"}], {}, [])


@pytest.mark.parametrize("specs", [[{"id": "a"}] * 2, [{"id": str(i)} for i in range(9)]])
def test_independent_batch_cannot_expand_budget_or_repeat_node(tmp_path, specs):
    with pytest.raises(ValueError, match="unique"):
        p.independent(tmp_path, specs, {}, [])


def test_complete_path_preserves_unknowns_and_does_not_self_admit(tmp_path, monkeypatch):
    raw, _, _ = fixture()
    seen = []

    def phase(folder, specs, pin, **kwargs):
        out = {}
        for spec in specs:
            seen.append(str(folder.relative_to(tmp_path)))
            key, inp = spec["id"], spec["input"]
            check = {"outcome": "unknown", "artifact_location": "", "artifact_quote": "", "issues": [], "explanation": "Fixture only."}
            if key == "extract":
                reply = {k: {"claims": [{"quote": v, "proposition": v, "connective_context": ""}], "overflow": False} for k, v in inp["notes"].items()}
            elif key == "evidence":
                reply = {k: {"interpretation": "faithful", "evidence_relation": "not_addressed", "evidence": []} for k in inp["unverifiedPredicates"]}
            elif spec.get("edit"):
                reply = {"operation": "keep", "replacement": "", "source_span_ids": [], "rationale": "Unknowns remain open, no new authority."}
            elif key == "review":
                reply = {d["id"]: check for d in inp["unverifiedContract"]["duties"]}
            else:
                reply = check
            out[key] = spec["bind"](reply)
        return out

    monkeypatch.setattr(p.probe, "phase", phase)
    report = p.case_run(tmp_path, raw, {})
    assert len(seen) == 9 and not any("plan" in s for s in seen)
    assert len([s for s in seen if "extract" in s]) == 2
    candidate = json.loads((tmp_path / "materialized/candidate.json").read_text())
    assert candidate == raw["candidate"] and report["allScheduledNodesBound"]
    assert report["notesAfter"]["notes"]["n000"]["needsInspection"]
    assert not report["completeAnswerApproved"] and not report["stageExitMet"]


def test_evidence_first_uses_runtime_phases_then_quote_only_notes_without_body_edit(tmp_path, monkeypatch):
    raw, _, _ = fixture()
    seen = []

    def phase(folder, specs, pin, **kwargs):
        out = {}
        for spec in specs:
            key, inp = spec["id"], spec["input"]
            seen.append(key)
            if key == "extract":
                reply = {k: {"claims": [{"quote": v, "proposition": v, "connective_context": ""}], "overflow": False} for k, v in inp["notes"].items()}
            elif key == "locate":
                eid = next(w["evidence_id"] for s in inp["evidenceSources"] if s["kind"] == "observation" for w in s["windows"])
                reply = {cid: [eid] for cid in inp["unverifiedPredicates"]}
            elif key == "compare":
                reply = {cid: {"interpretation": "faithful", "relations": {eid: "different_predicate" for eid in ids}} for cid, ids in inp["selectedByPredicate"].items()}
            else:
                assert spec["edit"] and set(spec["schema"]["properties"]) == {"operation", "evidence_ids"}
                reply = {"operation": "quote_observations", "evidence_ids": list(inp["readOnlyContext"]["availableEvidence"])[:1]}
            out[key] = spec["bind"](reply)
        return out

    monkeypatch.setattr(p.probe, "phase", phase)
    report = p.evidence_case_run(tmp_path, raw, {})
    assert seen == ["extract", "locate", "compare", "n000"]
    assert report["bodyUnchanged"] and report["valuesUnchanged"] and report["freeTextNoteEdits"] == 0
    assert report["sourceProjectionIsNotReasoningSuccess"] and not report["completeAnswerApproved"]
    assert report["comparison"]["notes"]["n000"]["needsInspection"]
    assert report["projections"]["n000"]["sourceProjectionProven"]
    assert sum(p.EVIDENCE_CAPS.values()) <= 17


def test_evidence_first_failed_locator_never_triggers_comparison_or_projection(tmp_path, monkeypatch):
    raw, _, _ = fixture()
    seen = []

    def independent(folder, specs, pin, failures, **kwargs):
        spec = specs[0]
        seen.append(spec["id"])
        if spec["id"] == "locate":
            return {}
        inp = spec["input"]
        reply = {k: {"claims": [{"quote": v, "proposition": v, "connective_context": ""}], "overflow": False} for k, v in inp["notes"].items()}
        return {spec["id"]: spec["bind"](reply)}

    monkeypatch.setattr(p, "independent", independent)
    with pytest.raises(ValueError, match="dependent phases not scheduled"):
        p.evidence_case_run(tmp_path, raw, {})
    assert seen == ["extract", "locate"]
