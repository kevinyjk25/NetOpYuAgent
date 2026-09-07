"""Grounding and review integrity fixtures, never semantic accuracy labels."""

import json

import pytest

from evaluation.flow_grounded_translation import (
    CitedDraft, assess_cited, cited_review_input, project, request,
)
from evaluation.flow_translation import local_sources
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_source_alignment import ClaimAssessment, SourceAssessment


def fixture():
    sources = local_sources().model_copy(update={"source_text":
        "Read device_id with read_inventory_device.\n"
        "For site campus finish the read path; otherwise hand off needs_l1.\n"
        "Local planned inventory, not live health; no authority is granted.\n"})
    quoted = CitedDraft.model_validate({
        "purpose_quotes": ["Read device_id with read_inventory_device.",
                           "Local planned inventory, not live health; no authority is granted."],
        "entry": 0, "issues": [], "steps": [
            {"operation": {"kind": "read", "tool": "read_inventory_device", "arguments": {
                "device_id": {"kind": "reference", "source": "input", "field": "device_id"}}, "next": 1},
             "source_quote": "Read device_id with read_inventory_device.", "requires": [], "true_quote": None, "false_quote": None},
            {"operation": {"kind": "branch", "left": {"kind": "reference", "source": 0, "field": "site"},
                "equals": {"kind": "constant", "value": "campus"}, "on_true": 2, "on_false": 3},
             "source_quote": "For site campus finish the read path; otherwise hand off needs_l1.",
             "true_quote": "For site campus finish the read path", "false_quote": "otherwise hand off needs_l1", "requires": [0]},
            {"operation": {"kind": "end", "outcome": "read_path_completed"},
             "source_quote": "For site campus finish the read path", "requires": [0, 1], "true_quote": None, "false_quote": None},
            {"operation": {"kind": "end", "outcome": "needs_l1"},
             "source_quote": "otherwise hand off needs_l1", "requires": [0, 1], "true_quote": None, "false_quote": None},
        ]})
    return sources, quoted


def mechanical_review(packet, verdict="supported"):
    return ReadL05Review(reviewer_id="fixture-not-semantic-review", reviewer_kind="test_fixture",
        assessment=SourceAssessment(input_digest=packet["inputDigest"], scope_note="Protocol fixture only; not a semantic judgment.",
            claims=tuple(ClaimAssessment(claim_id=c["claimId"], verdict=verdict,
                source_span_ids=tuple(c.get("requiredCitationId", "skill-0001") if r == "skill" else "host-0001"
                                     for r in c["requiredEvidenceKinds"]),
                rationale="Mechanical plumbing fixture, no semantic truth claim.", suggested_revision="Inspect the original source.")
                for c in packet["claims"])))


def test_project_only_selects_text_and_renames_indices():
    source, cited = fixture()
    draft = project(source, cited)
    assert draft.purpose == "\n".join(cited.purpose_quotes)
    assert draft.nodes[1].on_true == "node-2"
    assert draft.nodes[3].explanation == cited.steps[3].source_quote
    packet = cited_review_input(source, cited)
    assert len([c for c in packet["claims"] if c["facet"].startswith("source_entails")]) == 4
    assert not assess_cited(source, cited, mechanical_review(packet))["runtimeAuthorityGranted"]


@pytest.mark.parametrize("edit", ["invented_quote", "ambiguous_quote", "empty_quote", "missing_true", "non_branch_quote", "self_dependency", "unreachable_dependency", "duplicate_dependency"])
def test_source_and_dependency_fail_closed(edit):
    source, cited = fixture()
    raw = cited.model_dump(mode="json")
    if edit == "invented_quote":
        raw["purpose_quotes"][0] = "Invented but syntactically valid quote."
    elif edit == "ambiguous_quote":
        source = source.model_copy(update={"source_text": source.source_text * 2})
    elif edit == "empty_quote":
        raw["steps"][0]["source_quote"] = " " * 8
    elif edit == "missing_true":
        raw["steps"][1]["true_quote"] = None
    elif edit == "non_branch_quote":
        raw["steps"][0]["true_quote"] = raw["purpose_quotes"][0]
    elif edit == "self_dependency":
        raw["steps"][0]["requires"] = [0]
    elif edit == "unreachable_dependency":
        raw["steps"][2]["requires"] = [3]
    else:
        raw["steps"][1]["requires"] = [0, 0]
    with pytest.raises(ValueError):
        project(source, CitedDraft.model_validate(raw))


def test_exact_quotes_do_not_prove_polarity():
    source, cited = fixture()
    raw = cited.model_dump(mode="json")
    raw["steps"][1]["operation"].update(on_true=3, on_false=2)
    wrong = CitedDraft.model_validate(raw)
    packet = cited_review_input(source, wrong)  # Quote lookup is NOT entailment.
    assert project(source, wrong).nodes[1].on_true == "node-3"  # Never silently repaired.
    result = assess_cited(source, wrong, mechanical_review(packet, "contradicted"))
    assert result["status"] == "blocked" and not result["semanticAlignmentProven"]


def test_changed_grounding_invalidates_review_even_when_graph_unchanged():
    source, cited = fixture()
    review = mechanical_review(cited_review_input(source, cited))
    raw = cited.model_dump(mode="json")
    raw["steps"][1]["requires"] = []  # Structurally optional evidence, still reviewed.
    changed = CitedDraft.model_validate(raw)
    assert project(source, cited) == project(source, changed)
    with pytest.raises(ValueError, match="digest"):
        assess_cited(source, changed, review)


def test_open_issue_still_blocks_and_gets_a_review_claim():
    source, cited = fixture()
    raw = cited.model_dump(mode="json")
    raw["issues"] = [{"kind": "source_ambiguity", "source_quote": raw["purpose_quotes"][0],
                       "question": "Is the source device identifier ambiguous?"}]
    cited = CitedDraft.model_validate(raw)
    packet = cited_review_input(source, cited)
    assert packet["claims"][-1]["facet"] == "actual_unresolved_fact_not_model_commentary"
    assert assess_cited(source, cited, mechanical_review(packet))["status"] == "blocked"


def test_request_includes_real_contract_purpose_not_case_labels():
    source, _ = fixture()
    wire = request(source)
    payload = json.loads(wire["messages"][1]["content"])
    assert payload["hostReadTools"][0]["businessDescription"] == source.reads["read_inventory_device"].metadata.description
    assert "not live telemetry" in payload["hostReadTools"][0]["businessDescription"]
    assert payload["hostReadTools"][0]["sourceDeclarations"]
    assert "source_path" not in wire["messages"][1]["content"]
    assert request(source.model_copy(update={"source_path": "hidden-answer-label"})) == wire


def test_incomplete_review_cannot_ignore_grounding_claims():
    source, cited = fixture()
    packet = cited_review_input(source, cited)
    raw = mechanical_review(packet).model_dump(mode="json")
    raw["assessment"]["claims"].pop()
    with pytest.raises(ValueError):
        assess_cited(source, cited, ReadL05Review.model_validate(raw))
