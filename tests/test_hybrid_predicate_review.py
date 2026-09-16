from copy import deepcopy
import pytest

from evaluation import hybrid_predicate_review as p
from evaluation.hybrid_draft_review import build_review_input
from evaluation.structured_authoring import seal
from tests.test_hybrid_draft_review import inputs


def fixture():
    raw = inputs()
    raw["candidate"]["notes"] = ["Management has not reviewed or approved the request."]
    payload = build_review_input(raw)
    supplied = p.extract_input(payload)
    extracted = p.bind_extract(supplied, {"n000": {"claims": [
        {"quote": "not reviewed", "proposition": "Management has not reviewed the request.", "connective_context": "Shared not over reviewed or approved."},
        {"quote": "approved", "proposition": "Management has not approved the request.", "connective_context": "Shared not over reviewed or approved."}], "overflow": False}})
    return raw, payload, supplied, extracted


def decisions(payload, extracted):
    source = payload["sourceSpans"][0]
    return {cid: {"interpretation": "faithful", "evidence_relation": "direct_support",
        "evidence": [{"source_id": source["source_span_id"], "quote": source["exactQuote"]}]} for cid in extracted["claims"]}


def test_extraction_sees_only_notes_not_answer_source_task_or_old_review():
    raw, _, supplied, _ = fixture()
    raw["candidate"]["draft"] = "body canary"
    raw["original_task"] += " task canary"
    raw["open_duties"] = "review canary"
    assert p.extract_input(build_review_input(raw)) == supplied
    assert set(supplied["notes"]) == {"n000"}
    assert not supplied["sourcesVisible"] and not supplied["answerBodyVisible"]


@pytest.mark.parametrize("quote", ["unrelated body paragraph", " ", "Management is approved"])
def test_extractor_cannot_address_body_or_invent_note_quote(quote):
    _, _, supplied, _ = fixture()
    with pytest.raises(ValueError, match="owned note"):
        p.bind_extract(supplied, {"n000": {"claims": [{"quote": quote, "proposition": "Unverified.", "connective_context": ""}], "overflow": False}})


@pytest.mark.parametrize("relation,status", [("direct_support", "supported"), ("contradiction", "contradicted"),
    ("different_predicate", "unknown"), ("not_addressed", "unknown"), ("derived_only", "unknown"), ("ambiguous", "unknown")])
def test_host_aggregates_each_predicate_not_global_opinion(relation, status):
    _, payload, _, extraction = fixture()
    raw = decisions(payload, extraction)
    raw["n000:p00"]["evidence_relation"] = relation
    report = p.bind_review(payload, extraction, raw)
    note = report["notes"]["n000"]
    assert [a["status"] for a in note["atoms"]] == [status, "supported"]
    assert note["needsInspection"] == (status != "supported")
    assert not note["allPredicatesEnumeratedProven"] and not report["semanticApproval"]
    assert report["allSupportedDoesNotProveNoteTruth"]


@pytest.mark.parametrize("mutation", ["no_evidence", "wrong_quote", "uncertain", "changed"])
def test_positive_downgraded_without_binding_or_faithful_interpretation(mutation):
    _, payload, _, extraction = fixture()
    raw = decisions(payload, extraction)
    row = raw["n000:p00"]
    if mutation == "no_evidence":
        row["evidence"] = []
    elif mutation == "wrong_quote":
        row["evidence"][0]["quote"] = "not in source"
    else:
        row["interpretation"] = mutation
    report = p.bind_review(payload, extraction, raw)
    assert report["notes"]["n000"]["atoms"][0]["status"] == "unknown"


def test_global_supported_cannot_override_unresolved_predicate():
    _, payload, _, extraction = fixture()
    raw = decisions(payload, extraction)
    raw["supported"] = True
    with pytest.raises(ValueError):
        p.bind_review(payload, extraction, raw)


def test_every_predicate_requires_decision_and_overflow_remains_open():
    _, payload, _, extraction = fixture()
    raw = decisions(payload, extraction)
    del raw["n000:p01"]
    with pytest.raises(ValueError):
        p.bind_review(payload, extraction, raw)
    # Build a valid, explicit capacity warning from the extraction protocol.
    supplied = p.extract_input(payload)
    extraction = p.bind_extract(supplied, {"n000": {"claims": [{"quote": supplied["notes"]["n000"],
        "proposition": "Unverified multi-predicate claim.", "connective_context": "unexpanded"}], "overflow": True}})
    report = p.bind_review(payload, extraction, decisions(payload, extraction))
    assert report["notes"]["n000"]["needsInspection"]


@pytest.mark.parametrize("mutation", ["payload_note", "forged_note", "forged_claim_id"])
def test_resealed_or_stale_extraction_cannot_be_rebound(mutation):
    _, payload, _, extraction = fixture()
    extraction = deepcopy(extraction)
    if mutation == "payload_note":
        payload["candidate"]["notes"][0] += " Changed."
    elif mutation == "forged_note":
        extraction["notes"]["n000"] = "Unrelated body."
    else:
        extraction["claims"]["n000:p00"]["id"] = "n999:p00"
    extraction = seal({k: v for k, v in extraction.items() if k != "reportDigest"})
    with pytest.raises(ValueError, match="drift"):
        p.review_input(payload, extraction)


def located_fixture():
    raw, _, _, _ = fixture()
    raw["observations"]["n0"]["observations"] = {"read": "Operator Ke has not authorized release; assessment status is unstated.\n\nA different source says authorized."}
    payload = build_review_input(raw)
    supplied = p.extract_input(payload)
    extraction = p.bind_extract(supplied, {"n000": {"claims": [{"quote": supplied["notes"]["n000"],
        "proposition": "Unverified fixture proposition.", "connective_context": ""}], "overflow": False}})
    ctx = p.locate_input(payload, extraction)
    units = [key for key, u in ctx["evidenceCatalog"]["units"].items() if u["kind"] == "observation"]
    return raw, payload, extraction, ctx, units


def test_evidence_directory_excludes_task_and_caller_and_wire_preserves_every_character():
    _, payload, _, ctx, _ = located_fixture()
    assert all(u["kind"] in {"skill", "observation"} for u in ctx["evidenceCatalog"]["units"].values())
    view = p.locate_view(ctx)
    for source, rendered in zip(ctx["sourceContext"]["sourceSpans"], view["evidenceSources"], strict=True):
        rebuilt = "".join(w["gapBefore"] + w["text"] for w in rendered["windows"]) + rendered["tail"]
        assert rebuilt == source["exactQuote"]
    for u in ctx["evidenceCatalog"]["units"].values():
        if "Operator Ke" in u["text"]:
            assert "assessment status is unstated" in u["text"]  # Shared semicolon stays intact.
    assert view["originalNotes"] == p.extract_input(payload)["notes"]
    assert not ctx["evidenceCatalog"]["semanticSegmentationProven"]


@pytest.mark.parametrize("ids", [["s000"], ["ev:invented:000"]])
def test_locator_cannot_supply_task_or_invented_reference(ids):
    _, payload, extraction, _, _ = located_fixture()
    with pytest.raises(ValueError):
        p.bind_locate(payload, extraction, {"n000:p00": ids})


def test_locator_cannot_repeat_or_expand_capacity():
    _, payload, extraction, _, ids = located_fixture()
    for selected in ([ids[0], ids[0]], [ids[0]] * 5):
        with pytest.raises(ValueError):
            p.bind_locate(payload, extraction, {"n000:p00": selected})


def test_missing_selection_is_unknown_and_does_not_permit_a_positive_model_key():
    _, payload, extraction, _, _ = located_fixture()
    located = p.bind_locate(payload, extraction, {"n000:p00": []})
    checked = p.bind_compare(payload, extraction, located, {})
    assert checked["notes"]["n000"]["atoms"][0]["status"] == "unknown"
    with pytest.raises(ValueError):
        p.bind_compare(payload, extraction, located, {"n000:p00": {"interpretation": "faithful", "relations": {}}})


@pytest.mark.parametrize("relation,status", [("direct_support", "supported"), ("contradiction", "contradicted"),
    ("different_predicate", "unknown"), ("derived_only", "unknown"), ("unclear", "unknown")])
def test_pairwise_judgment_is_bound_to_selected_source_without_retyped_quotes(relation, status):
    _, payload, extraction, _, ids = located_fixture()
    located = p.bind_locate(payload, extraction, {"n000:p00": [ids[0]]})
    raw = {"n000:p00": {"interpretation": "faithful", "relations": {ids[0]: relation}}}
    checked = p.bind_compare(payload, extraction, located, raw)
    atom = checked["notes"]["n000"]["atoms"][0]
    assert atom["status"] == status and atom["bound"]
    assert atom["evidence"][0]["quote"] == p.evidence_catalog(payload)["units"][ids[0]]["text"]
    assert not checked["semanticApproval"] and not checked["sourceSelectionExhaustive"]
    assert checked["allSupportedDoesNotProveNoteTruth"]  # A wrong model opinion still cannot self-admit.


def test_comparison_cannot_change_selection_or_ignore_conflicting_evidence():
    _, payload, extraction, _, ids = located_fixture()
    located = p.bind_locate(payload, extraction, {"n000:p00": ids[:2]})
    raw = {"n000:p00": {"interpretation": "faithful", "relations": {ids[0]: "direct_support", ids[1]: "contradiction"}}}
    checked = p.bind_compare(payload, extraction, located, raw)
    assert checked["notes"]["n000"]["atoms"][0]["status"] == "unknown"
    assert checked["bindingIssues"][0]["code"] == "selected_evidence_conflict"
    del raw["n000:p00"]["relations"][ids[1]]
    with pytest.raises(ValueError):
        p.bind_compare(payload, extraction, located, raw)


@pytest.mark.parametrize("mutation", ["candidate", "source", "selection"])
def test_comparison_rejects_stale_source_candidate_or_resealed_selection(mutation):
    _, payload, extraction, _, ids = located_fixture()
    located = p.bind_locate(payload, extraction, {"n000:p00": [ids[0]]})
    if mutation == "candidate":
        payload["completeCandidateDigest"] = "sha256:changed"
    elif mutation == "source":
        next(s for s in payload["sourceSpans"] if s["kind"] == "observation")["exactQuote"] += " Change."
    else:
        located["selected"]["n000:p00"] = [ids[1]]
        located = seal({k: v for k, v in located.items() if k != "reportDigest"})
    with pytest.raises(ValueError, match="drift"):
        p.compare_input(payload, extraction, located)


def test_compare_includes_complete_parents_and_discloses_unselected_sources():
    _, payload, extraction, _, ids = located_fixture()
    located = p.bind_locate(payload, extraction, {"n000:p00": [ids[0]]})
    ctx = p.compare_input(payload, extraction, located)
    source = next(s for s in payload["sourceSpans"] if s["kind"] == "observation")
    assert ctx["completeSelectedParents"][source["source_span_id"]]["exactQuote"] == source["exactQuote"]
    assert "different source" in source["exactQuote"]
    assert ctx["sourceSelectionIsNotExhaustive"]
