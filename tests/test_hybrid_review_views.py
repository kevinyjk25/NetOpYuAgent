import copy
import json

import pytest

from evaluation.hybrid_draft_review import build_review_input
from evaluation.hybrid_review_views import review_view, editor_view
from evaluation.hybrid_draft_slots import editing_slots
from evaluation.hybrid_repair_cells import lines_for, source_units, relation_index
from tests.test_hybrid_draft_review import inputs


@pytest.mark.parametrize("label", ["1.", "27.", "999999999.", "  3.", "# 1.", "### 12.", "###### 7."])
def test_numbered_markdown_label_stays_with_body_without_losing_claims(label):
    source = inputs()
    first = label + " **Owner:** Quinn reviews at 16:20."
    draft = first + " NOT approved.\n\n2. Follow-up requires new permission."
    source["candidate"]["draft"] = draft
    payload = build_review_input(source)
    spans = payload["draftSpans"]
    assert [s["exactQuote"] for s in spans] == [first, "NOT approved.", "2. Follow-up requires new permission."]
    assert payload["candidate"]["draft"] == draft
    for span in spans:
        assert draft[span["start"]:span["end"]] == span["exactQuote"]
    claims = [c for c in payload["claims"] if c["facet"] == "all_claims_in_text_block"]
    assert len(claims) == len(spans)  # Merge boundaries, never drop body assertions.
    assert payload["runtimeAuthorityGranted"] is False


def test_numbered_heading_retains_heading_and_following_negation():
    from evaluation.hybrid_draft_review import _blocks

    text = "## 3. Limits\nNo write is permitted. Another task remains open."
    assert [s for _, _, s in _blocks(text)] == ["## 3. Limits\nNo write is permitted.", "Another task remains open."]


def test_nonlabel_numeric_sentence_ending_is_not_merged():
    from evaluation.hybrid_draft_review import _blocks

    text = "The loss count is 1. Investigate it.\n\nVersion 1. Another statement."
    assert [s for _, _, s in _blocks(text)] == ["The loss count is 1.", "Investigate it.", "Version 1.", "Another statement."]


def test_empty_numbered_item_does_not_absorb_another_paragraph():
    from evaluation.hybrid_draft_review import _blocks

    text = "1.\n\nApproval remains unknown."
    assert [s for _, _, s in _blocks(text)] == ["1.", "Approval remains unknown."]


def test_review_worksheet_preserves_all_original_text_and_concrete_targets_not_compliance_slogans():
    original = inputs()
    original["candidate"]["draft"] = "Logging is disabled. This causes every event to be discarded."
    original["candidate"]["notes"] = ["Whether events were received before they were discarded."]
    payload = build_review_input(original)
    before = copy.deepcopy(payload)
    view = review_view(payload)
    assert payload == before
    assert view["originalTask"] == original["original_task"] and view["candidate"] == payload["candidate"]
    assert view["hostOpenDuties"] == payload["hostOpenDuties"]
    assert [(s["source_span_id"], s["kind"], s["path"], s["exactQuote"]) for s in view["sourceSpans"]] == [
        (s["source_span_id"], s["kind"], s["path"], s["exactQuote"]) for s in payload["sourceSpans"]]
    assert [c["claimId"] for c in view["claims"]] == [c["claimId"] for c in payload["claims"]]
    text_checks = [c for c in view["claims"] if c["subject"] == "answer_text"]
    assert len(text_checks) == 2 and text_checks[-1]["textToInspect"] == "This causes every event to be discarded."
    note = next(c for c in view["claims"] if c["subject"] == "unverified_candidate_note")
    assert "presupposes" in view["reviewRules"][note["subject"]]
    assert view["claims"][-1]["subject"] == "whole_answer_check"
    assert "self-issued declaration" in view["claims"][-1]["question"]


def test_prose_units_keep_exact_offsets_and_do_not_split_code_or_decimal_literals():
    original = inputs()
    draft = "Rate is 1.25%. A later clause is unverified.\n\n```python\n# One sentence. Another comment.\n\nx = 'A. B.'\n```\n\nRun `print('A. B.')` only in an example.\n注意条件。不要执行。"
    original["candidate"]["draft"] = draft
    spans = build_review_input(original)["draftSpans"]
    for span in spans:
        assert draft[span["start"]:span["end"]] == span["exactQuote"]
    assert any(span["exactQuote"].startswith("Rate is 1.25%.") for span in spans)
    code = [span for span in spans if span["exactQuote"].startswith("```python")]
    assert len(code) == 1 and "x = 'A. B.'" in code[0]["exactQuote"] and code[0]["exactQuote"].rstrip().endswith("```")
    assert any("`print('A. B.')`" in span["exactQuote"] for span in spans)


def test_editor_projection_keeps_full_sources_task_draft_and_located_edit_lines():
    original = inputs()
    payload = build_review_input(original)
    slots = editing_slots(payload, complete_sections=True)
    supplied = {"sourceSpans": payload["sourceSpans"], "originalTask": payload["originalTask"],
        "hostOpenDuties": payload["hostOpenDuties"], "completePriorDraftReadOnly": original["candidate"]["draft"],
        "ownedFragment": slots["slots"][0], "editableLines": lines_for(slots["slots"][0]),
        "sourceUnitCatalog": source_units(payload, slots), "sourceRelationIndex": relation_index(payload, slots),
        "reportedConcerns": [], "repairFocus": []}
    before = copy.deepcopy(supplied)
    view = editor_view(supplied)
    assert supplied == before and view["ownedFragment"]["text"] == supplied["ownedFragment"]["text"]
    assert "readOnlySurroundingDraft" not in view and "readOnlyCompleteDraft" not in view
    assert view["editableLines"] == supplied["editableLines"] and view["originalTask"] == payload["originalTask"]
    rendered = json.dumps(view, ensure_ascii=False)
    for source in payload["sourceSpans"]:
        assert json.dumps(source["exactQuote"], ensure_ascii=False)[1:-1] in rendered


def test_other_candidate_prose_is_not_a_hidden_edit_target_but_remains_in_parent_audit():
    from network_runtime.contracts import sha256_json

    source = inputs()
    source["candidate"]["draft"] = "Intro sentence.\n\n## Other\n\nA foreign invented scenario absent from the business observations.\n"
    payload = build_review_input(source)
    slots = editing_slots(payload, complete_sections=True)
    supplied = {"sourceSpans": payload["sourceSpans"], "originalTask": payload["originalTask"], "hostOpenDuties": payload["hostOpenDuties"],
        "completePriorDraftReadOnly": source["candidate"]["draft"], "ownedFragment": slots["slots"][0],
        "editableLines": lines_for(slots["slots"][0]), "sourceUnitCatalog": [], "sourceRelationIndex": [], "reportedConcerns": [], "repairFocus": []}
    view = editor_view(supplied)
    assert "foreign invented scenario" not in json.dumps(view)
    assert view["parentDraftDigest"] == sha256_json(source["candidate"]["draft"])
    assert view["readOnlyOtherSections"][0]["label"] == "## Other"
    assert "foreign invented scenario" in supplied["completePriorDraftReadOnly"]
