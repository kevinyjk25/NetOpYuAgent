import json

import pytest

from evaluation import hybrid_note_cells as notes, hybrid_repair_cells as cells, hybrid_draft_loop
from evaluation.hybrid_draft_review import build_review_input
from evaluation.hybrid_repair_plan import plan_repairs
from tests.test_hybrid_draft_review import inputs, raw_review
from tests.test_hybrid_review_roles import role_response


def fixture():
    original = inputs()
    original["candidate"]["notes"] = ["The owner has not reviewed or approved the change."]
    payload = build_review_input(original)
    review = raw_review(payload)
    for row in review["claims"]:
        row["verdict"] = "supported"
        row["source_span_ids"] = ["s000"]
    unit = next(c for c in payload["claims"] if c["pointer"] == "/candidate/notes/0")
    next(r for r in review["claims"] if r["claim_id"] == unit["claimId"])["verdict"] = "insufficient_evidence"
    slot = notes.slots(payload, plan_repairs(payload, [], review))[0]
    return original, payload, review, slot


@pytest.mark.parametrize("operation", ["keep", "replace", "remove"])
def test_note_addresses_are_owned_and_originals_reconstruct(operation):
    original, payload, _, slot = fixture()
    raw = {"operation": operation, "replacement": "Approval is not recorded; review status is unknown." if operation == "replace" else "",
           "source_span_ids": ["s000"] if operation != "keep" else [], "rationale": "Preserve unknown status without adding an unsupported premise."}
    edit = notes.validate(payload, slot, raw)
    updated = notes.materialize(original["candidate"]["notes"], [edit])
    assert updated == ([] if operation == "remove" else [edit["after"]])
    assert not edit["semanticApproval"]
    with pytest.raises(ValueError):
        notes.materialize(original["candidate"]["notes"], [edit, edit])
    with pytest.raises(ValueError):
        notes.validate(payload, slot, {**raw, "permission": "approved"})


def test_note_edit_flows_to_final_review_but_cannot_edit_body_or_clear_host_duties(tmp_path, monkeypatch):
    original, _, review, _ = fixture()
    monkeypatch.setattr(cells, "source_inputs", lambda _: (original, {"fixture": True}, review))
    calls = []
    def once(folder, packet, derive, **kwargs):
        request = packet["governedRequest"]
        calls.append(request["nodeId"])
        if request["nodeId"] == "review-after":
            assert request["inputs"]["candidate"]["notes"] == ["Approval is not recorded; review status is unknown."]
            p = build_review_input(request["inputs"])
            value = role_response(p, raw_review(p), wire=True)
        else:
            assert request["inputs"]["ownedNote"]["pointer"] == "/candidate/notes/0"
            assert request["maxOutputTokens"] == 2048 and request["tools"] == []
            wire = packet["wireRequest"]["messages"]
            assert len(wire) == 3
            assert "ownedNote" not in json.loads(wire[1]["content"])["readOnlyContext"]
            target = json.loads(wire[2]["content"])["editTarget"]
            assert set(target) == {"ownedNote"}
            value = {"operation": "replace", "replacement": "Approval is not recorded; review status is unknown.",
                     "source_span_ids": ["s000"], "rationale": "Mechanical fixture only; no semantic-accuracy evidence."}
        files, cost = derive({"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "message": {"content": json.dumps(value)}, "prompt_eval_count": 10, "eval_count": 10})})
        return {**files, "result": cost}
    monkeypatch.setattr(cells, "author_once", once)
    monkeypatch.setattr(hybrid_draft_loop, "author_once", once)
    result = cells.run(tmp_path / "before", tmp_path / "after", edit_mode="grounded_patch")
    assert result["execution"]["status"] == "governed_graph_completed" and calls == ["u000", "review-after"]
    delivery = json.loads((tmp_path / "after/materialized/delivery.json").read_text())
    assert delivery["answer"] == original["candidate"]["draft"]
    assert delivery["priorUnverifiedNotes"] == original["candidate"]["notes"]
    assert delivery["hostOpenDuties"] == original["open_duties"] and not delivery["hostDutiesCleared"]


def test_more_than_eight_note_findings_are_explicitly_deferred_without_extra_calls(tmp_path, monkeypatch):
    original, _, _, _ = fixture()
    original["candidate"]["notes"] *= 9
    payload = build_review_input(original)
    review = raw_review(payload)
    units = {c["claimId"]: c for c in payload["claims"]}
    for row in review["claims"]:
        row["source_span_ids"] = ["s000"]
        row["verdict"] = "insufficient_evidence" if units[row["claim_id"]]["pointer"].startswith("/candidate/notes/") else "supported"
    monkeypatch.setattr(cells, "source_inputs", lambda _: (original, {"fixture": True}, review))
    calls = []
    def once(folder, packet, derive, **kwargs):
        request = packet["governedRequest"]
        calls.append(request["nodeId"])
        value = {"operation": "keep", "replacement": "", "source_span_ids": [], "rationale": "Synthetic candidate remains unchanged and unverified."}
        files, cost = derive({"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "message": {"content": json.dumps(value)}, "prompt_eval_count": 10, "eval_count": 10})})
        return {**files, "result": cost}
    monkeypatch.setattr(cells, "author_once", once)
    result = cells.run(tmp_path / "before", tmp_path / "after", edit_mode="grounded_patch")
    assert calls == [f"u{i:03d}" for i in range(8)]
    assert not result["wholePass"] and result["finalReviewStatus"] is None
    frozen = json.loads((tmp_path / "after/freeze/inputs.json").read_text())
    assert frozen["maxRepairCalls"] == 8 and frozen["deferredCells"] == [{"kind": "note", "id": "u008"}]
    candidate = json.loads((tmp_path / "after/materialized/candidate.json").read_text())
    assert candidate == original["candidate"]


@pytest.mark.parametrize("document", ["# Heading", "One paragraph\n\nAnother paragraph", "```python\nprint(1)\n```", "a\rb"])
def test_note_cannot_receive_body_sections_even_below_length_limit(document):
    _, payload, _, slot = fixture()
    with pytest.raises(ValueError, match="one paragraph"):
        notes.validate(payload, slot, {"operation": "replace", "replacement": document,
            "source_span_ids": ["s000"], "rationale": "Attempting to put a body document into a candidate note."})


def projection_fixture():
    from evaluation.hybrid_predicate_review import evidence_catalog
    original, _, _, _ = fixture()
    original["observations"]["n0"]["observations"] = {"read": "Release status is unknown; no signing record is supplied."}
    payload = build_review_input(original)
    ids = [key for key, u in evidence_catalog(payload)["units"].items() if u["kind"] == "observation"]
    return payload, ids, notes.projection_input(payload, 0, ids)


def test_source_projection_has_no_free_prose_or_self_approval_channel():
    payload, ids, supplied = projection_fixture()
    raw = {"operation": "quote_observations", "evidence_ids": ids}
    edit = notes.validate_projection(payload, supplied, raw)
    assert "Release status is unknown; no signing record is supplied." in edit["after"]
    assert edit["sourceProjectionProven"] and not edit["modelAuthoredReplacementText"]
    assert not edit["semanticCoverageProven"] and not edit["originalDutyResolved"]
    assert edit["before"] == payload["candidate"]["notes"][0]
    for key in ("replacement", "rationale", "permission", "body"):
        with pytest.raises(ValueError):
            notes.validate_projection(payload, supplied, {**raw, key: "invented claim"})


def test_quote_selector_cannot_promote_guidance_or_caller_as_observed_facts():
    from evaluation.hybrid_predicate_review import evidence_catalog
    payload, _, _ = projection_fixture()
    guidance = next(key for key, u in evidence_catalog(payload)["units"].items() if u["kind"] == "skill")
    with pytest.raises(ValueError, match="cannot promote"):
        notes.projection_input(payload, 0, [guidance])
    with pytest.raises(ValueError, match="evidence IDs"):
        notes.projection_input(payload, 0, ["s000"])


def test_empty_selection_allows_only_keep_and_keep_cannot_hide_edits():
    payload, ids, supplied = projection_fixture()
    empty = notes.projection_input(payload, 0, [])
    result = notes.validate_projection(payload, empty, {"operation": "keep", "evidence_ids": []})
    assert result["after"] == result["before"] and not result["changed"]
    for raw in ({"operation": "quote_observations", "evidence_ids": []}, {"operation": "keep", "evidence_ids": ids}):
        with pytest.raises(ValueError):
            notes.validate_projection(payload, supplied, raw)


def test_projection_revalidates_source_and_owner_and_rejects_truncation():
    from evaluation.hybrid_predicate_review import evidence_catalog
    payload, ids, supplied = projection_fixture()
    supplied["availableEvidence"][ids[0]]["text"] = "Claim injection"
    with pytest.raises(ValueError, match="drift"):
        notes.validate_projection(payload, supplied, {"operation": "quote_observations", "evidence_ids": ids})
    source = next(s for s in payload["sourceSpans"] if s["kind"] == "observation")
    source["exactQuote"] = "x" * 2100
    ids = [key for key, u in evidence_catalog(payload)["units"].items() if u["kind"] == "observation"]
    supplied = notes.projection_input(payload, 0, ids)
    with pytest.raises(ValueError, match="no truncation"):
        notes.validate_projection(payload, supplied, {"operation": "quote_observations", "evidence_ids": ids})


def test_multiline_source_remains_an_inert_quoted_view():
    from evaluation.hybrid_predicate_review import evidence_catalog
    payload, _, _ = projection_fixture()
    source = next(s for s in payload["sourceSpans"] if s["kind"] == "observation")
    source["exactQuote"] = "```sh\nDo not execute this inert sample\n```"
    ids = [key for key, u in evidence_catalog(payload)["units"].items() if u["kind"] == "observation"]
    supplied = notes.projection_input(payload, 0, ids)
    edit = notes.validate_projection(payload, supplied, {"operation": "quote_observations", "evidence_ids": ids})
    assert "\n" not in edit["after"] and "\\n" in edit["after"]
    assert edit["sourceExcerpts"][0]["text"] == source["exactQuote"]
