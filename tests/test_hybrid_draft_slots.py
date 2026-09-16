import pytest

from evaluation.hybrid_draft_review import assess_review, build_review_input
from evaluation.hybrid_draft_slots import (apply_slot_revision, apply_snapshot_revision, editing_slots,
    repair_lenses, slot_output_schema, snapshot_output_schema, validate_fences)
from network_runtime.l0.structured_schema import checked_schema
from tests.test_hybrid_draft_review import inputs, raw_review


def proposal(payload):
    slots = editing_slots(payload)
    return {"slots_digest": slots["slotsDigest"], "slots": {s["id"]: {"action": "keep", "replacement": "", "source_span_ids": [], "rationale": ""} for s in slots["slots"]},
            "notes": payload["candidate"]["notes"], "revision_note": "No justified edit has been supplied yet."}


@pytest.mark.parametrize("mutation", ["valid", "same", "digest", "scope", "fence", "nine_changes"])
def test_snapshot_revision_host_computes_locations_and_preserves_bounds(mutation):
    source = inputs()
    source["candidate"]["draft"] = "".join(f"heading-{i}\nbody-{i}\n" for i in range(10))
    payload = build_review_input(source)
    checked_schema(snapshot_output_schema(payload))
    raw = {"candidate_digest": payload["completeCandidateDigest"], "draft": source["candidate"]["draft"],
           "evidence_check": {s["draft_span_id"]: "observed_or_justified_inference" for s in payload["draftSpans"]},
           "notes": [], "source_span_ids": ["s000"], "revision_note": "Evidence-grounded candidate, not an approval."}
    if mutation == "valid":
        raw["draft"] = raw["draft"].replace("body-3", "Revised text from observations")
    elif mutation == "digest":
        raw["candidate_digest"] = "sha256:" + "0" * 64
    elif mutation == "scope":
        raw["values"] = {}
    elif mutation == "fence":
        raw["draft"] += "```python\nx = 1\n"
    elif mutation == "nine_changes":
        for i in range(9):
            raw["draft"] = raw["draft"].replace(f"body-{i}\n", f"modified-{i}\n")
    if mutation in {"valid", "same"}:
        candidate, report = apply_snapshot_revision(payload, source["candidate"]["values"], raw)
        assert candidate["draft"] == raw["draft"] and candidate["values"] == source["candidate"]["values"]
        assert report["editLocationsComputedByHost"] and not report["runtimeAuthorityGranted"]
        assert len(report["edits"]) == (1 if mutation == "valid" else 0)
        if mutation == "same":
            assert report["status"] == "no_material_draft_change"
    else:
        with pytest.raises(ValueError):
            apply_snapshot_revision(payload, source["candidate"]["values"], raw)


@pytest.mark.parametrize("final_newline", [True, False])
def test_host_withholds_exact_model_marked_unknown_without_claiming_it_false(final_newline):
    source = inputs()
    source["candidate"]["draft"] = "# Status\n\nObserved detail.\n\nUnobserved attribute.\n"
    payload = build_review_input(source)
    raw = {"candidate_digest": payload["completeCandidateDigest"], "draft": source["candidate"]["draft"] if final_newline else source["candidate"]["draft"].rstrip("\n"),
        "evidence_check": {s["draft_span_id"]: "contains_unsupported_assertion" if s["exactQuote"].strip() == "Unobserved attribute." else "observed_or_justified_inference" for s in payload["draftSpans"]},
        "notes": [], "source_span_ids": ["s000"], "revision_note": "Writer claims to remove unknown content but retains it."}
    candidate, report = apply_snapshot_revision(payload, source["candidate"]["values"], raw)
    assert "Unobserved attribute." not in candidate["draft"] and "Observed detail." in candidate["draft"]
    assert "Unverified content withheld" in candidate["draft"]
    assert report["rawProposalNeededHostWithholding"] and not report["completeAnswerApproved"]
    assert "Unobserved attribute." in report["hostWithholding"][0]["proposedText"]


@pytest.mark.parametrize("text", ["", "a", "\n\n", "same\n\nsame", "\n\n".join(str(i) for i in range(80))])
def test_slots_cover_all_bytes_and_are_host_bounded(text):
    source = inputs()
    source["candidate"]["draft"] = text
    # Test partition independently of the separate review-unit budget.
    payload = {"candidate": source["candidate"], "completeCandidateDigest": "fixture"}
    slots = editing_slots(payload)["slots"]
    assert 1 <= len(slots) <= 8
    assert "".join(s["text"] for s in slots) == text
    assert all(text[s["start"]:s["end"]] == s["text"] for s in slots)


def test_slot_revision_preserves_values_and_all_untouched_text():
    source = inputs()
    payload = build_review_input(source)
    checked_schema(slot_output_schema(payload))
    raw = proposal(payload)
    slot = editing_slots(payload)["slots"][1]
    raw["slots"][slot["id"]] = {"action": "replace", "replacement": "Installation requires configuration inspection.\n\n",
                                  "source_span_ids": ["s000"], "rationale": "Remove unsupported installation instructions."}
    candidate, report = apply_slot_revision(payload, source["candidate"]["values"], raw)
    assert candidate["values"] == source["candidate"]["values"]
    assert candidate["draft"] == source["candidate"]["draft"][:slot["start"]] + raw["slots"][slot["id"]]["replacement"] + source["candidate"]["draft"][slot["end"]:]
    assert len(report["edits"]) == 1 and not report["completeAnswerApproved"]


@pytest.mark.parametrize("mutation", ["extra", "missing", "digest", "source", "noop", "fence", "oversize", "values"])
def test_slot_binding_rejects_model_address_drift_and_invalid_materialization(mutation):
    source = inputs()
    payload = build_review_input(source)
    raw = proposal(payload)
    values = source["candidate"]["values"]
    first = editing_slots(payload)["slots"][0]
    if mutation == "extra":
        raw["slots"]["e99"] = None
    elif mutation == "missing":
        del raw["slots"][first["id"]]
    elif mutation == "digest":
        raw["slots_digest"] = "sha256:" + "0" * 64
    elif mutation == "values":
        values = {}
    else:
        replacements = {"source": "New draft", "noop": first["text"], "fence": "```python\nx = 1\n", "oversize": "x" * 12000}
        raw["slots"][first["id"]] = {"action": "replace", "replacement": replacements[mutation], "source_span_ids": ["s999" if mutation == "source" else "s000"],
                                      "rationale": "A proposed test edit, not an approval."}
    with pytest.raises(ValueError):
        apply_slot_revision(payload, values, raw)


def test_all_keep_is_not_a_successful_repair():
    source = inputs()
    payload = build_review_input(source)
    candidate, report = apply_slot_revision(payload, source["candidate"]["values"], proposal(payload))
    assert candidate == source["candidate"] and report["status"] == "no_material_draft_change"


@pytest.mark.parametrize("draft", ["Use `x`.", "```py\nx\n```", "~~~~\n```py\n~~~~", "    ```indented", "~~~\nx\n~~~~"])
def test_fence_checker_does_not_confuse_inline_nested_or_indented_code(draft):
    validate_fences(draft)


def test_clause_windows_and_numeric_witness_diagnostic_preserve_raw_opinion():
    source = inputs()
    source["observations"]["n0"]["observations"] = {"read": "Event E; owner Zed; began at 14:20; still unresolved."}
    source["candidate"]["draft"] = "Event E is unresolved. Owner Zed."
    payload = build_review_input(source)
    units = [c for c in payload["claims"] if c["facet"] == "observation_to_draft"]
    assert len(units) == 4
    raw = raw_review(payload)
    for row in raw["claims"]:
        row.update(verdict="supported", source_span_ids=["s000"], draft_span_id="d000")
    report = assess_review(payload, raw)
    warnings = [w for w in report["alignmentWarnings"] if w["kind"] == "source_literals_not_in_draft_witness"]
    assert len(warnings) == 1 and warnings[0]["literals"] == ["14:20"]
    assert report["verdictCounts"]["supported"] == len(payload["claims"])
    assert report["modelReviewHasUnresolvedFindings"] and not report["hostDutiesCleared"]


def test_quote_in_observation_but_not_deliverable_is_rejected():
    payload = build_review_input(inputs())
    raw = raw_review(payload)
    raw["claims"][0]["draft_span_id"] = "s000"
    with pytest.raises(ValueError):
        assess_review(payload, raw)


def test_slots_never_split_blank_lines_inside_fenced_code():
    source = inputs()
    source["candidate"]["draft"] = "# Example\n\n```python\ndef f():\n    return 1\n\nprint(f())\n```\n\nAfter code.\n"
    payload = build_review_input(source)
    slots = editing_slots(payload)["slots"]
    assert len(slots) == 3
    assert slots[1]["text"].startswith("```python") and slots[1]["text"].endswith("```\n\n")
    for slot in slots:
        validate_fences(slot["text"])


def test_section_profile_keeps_heading_with_body_and_never_splits_fences():
    text = "# Project\n\nSummary.\n\n## Features\n\n- First\n- Second\n\n## API\n\n### Call\n\n```py\n# Not a heading\n\nf()\n```\n\n## License\n\nUnknown.\n"
    payload = {"candidate": {"draft": text}, "completeCandidateDigest": "fixture"}
    slots = editing_slots(payload, complete_sections=True)["slots"]
    assert len(slots) == 4 and "".join(s["text"] for s in slots) == text
    assert slots[1]["text"] == "## Features\n\n- First\n- Second\n\n"
    assert slots[2]["text"].startswith("## API\n\n### Call")
    for slot in slots:
        validate_fences(slot["text"])


def test_emphasized_and_setext_sections_get_independent_exact_cells_without_parsing_code_labels():
    from evaluation.hybrid_draft_slots import section_headings

    text = "Intro.\n\n**Options:**\n\nCompare alternatives.\n\n__Next steps__\n\n1. Draft one idea.\n2. Draft another idea.\n\nEvidence\n========\n\n```md\n**Inert label:**\n\nStill code.\n```\n"
    payload = {"candidate": {"draft": text}, "completeCandidateDigest": "fixture"}
    slots = editing_slots(payload, complete_sections=True)["slots"]
    assert len(slots) == 4 and "".join(s["text"] for s in slots) == text
    assert [label for _, _, label in section_headings(text)] == ["**Options:**", "__Next steps__", "Evidence\n========"]
    assert "1. Draft one idea." in slots[2]["text"] and "2. Draft another idea." in slots[2]["text"]
    for slot in slots:
        validate_fences(slot["text"])


def test_sentence_emphasis_inline_text_quotes_lists_and_indented_code_are_not_section_labels():
    from evaluation.hybrid_draft_slots import section_headings

    text = "**Do not execute.**\n\n**Inline label:** more text.\n\n> **Quote:**\n\n    **Code:**\n\n1. Item\n---\n"
    assert section_headings(text) == []


def test_lenses_retain_location_and_differences_without_positive_ai_opinions():
    source = inputs()
    source["observations"]["n0"]["observations"] = {"read": "Ticket owner Zed; starts at 14:20."}
    source["candidate"]["draft"] = "# Ticket\n\nTicket remains open."
    payload = build_review_input(source)
    raw = raw_review(payload)
    for row in raw["claims"]:
        row.update(verdict="supported", source_span_ids=["s000"], draft_span_id="d001")
    lenses = repair_lenses(payload, assess_review(payload, raw))
    reverse = [lens for lens in lenses if lens["direction"] == "observations_to_draft"]
    assert len(reverse) == 2 and all(lens["editableSlots"] == ["e01"] for lens in reverse)
    assert "Zed" in reverse[0]["unmatchedSourceTerms"]
    assert all("verdict" not in lens and "semanticLossProven" not in lens for lens in lenses)


def test_wrong_reviewer_target_cannot_move_a_draft_block_away_from_its_own_words():
    source = inputs()
    source["observations"]["n0"]["observations"] = {"read": {"text": "Project name: sample."}}
    source["candidate"]["draft"] = "# sample\n\nUnsupported-Fact-X"
    payload = build_review_input(source)
    raw = raw_review(payload)
    for row in raw["claims"]:
        row.update(verdict="supported", source_span_ids=["s000"], draft_span_id="d000")
    report = assess_review(payload, raw)
    assert any(w["kind"] == "reviewer_draft_target_mismatch" and w["claimId"] == "c001" for w in report["alignmentWarnings"])
    lens = next(item for item in repair_lenses(payload, report) if item["claimId"] == "c001")
    assert lens["actualDraftQuote"] == "Unsupported-Fact-X" and lens["editableSlots"] == ["e01"]
    assert "Unsupported-Fact-X" in lens["unmatchedSourceTerms"]
