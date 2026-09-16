import copy
import json

import pytest

from evaluation.hybrid_draft_review import (
    REVIEW_SCHEMA, apply_revision_patch, assess_review, build_review_input, located_output_schema, revision_context,
)
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import checked_schema, validate_data


def inputs():
    return {"original_task": "Read the project and draft a README without executing code.", "caller": {"path": "/sandbox/project"},
        "source_material": json.dumps({"p000": {"path": "SKILL.md", "text": "Inspect code before drafting. Example: pip install demo.",
            "start": 0, "end": 56, "sourceDigest": "sha256:" + "a" * 64}}),
        "observations": {"n0": {"observations": {"read": {"entries": ["pyproject.toml", "src"]}}, "outcome": "read_path_completed"}},
        "candidate": {"values": {"entries": ["pyproject.toml", "src"]}, "draft": "# Example\n\nInstall using `pip install demo`.\n\nRun `pytest`.",
                      "notes": ["Configuration contents have not been read."]},
        "open_duties": "Configuration inspection and supported drafting remain open."}


def raw_review(payload):
    return {"input_digest": payload["inputDigest"], "claims": [{"claim_id": c["claimId"], "verdict": "insufficient_evidence",
        "source_span_ids": [], "rationale": "This unit requires closer source and observation review.",
        "suggested_revision": "Read missing evidence or withhold unsupported assertions.", "draft_span_id": ""} for c in payload["claims"]],
        "scope_note": "Synthetic bookkeeping review only, not language accuracy or authority."}


def test_no_read_outer_map_is_not_a_business_observation_but_actual_empty_read_is():
    source = inputs()
    source["observations"]["n0"]["observations"] = {}
    assert not [s for s in build_review_input(source)["sourceSpans"] if s["kind"] == "observation"]
    source["observations"]["n0"]["observations"] = {"read": {}}
    assert [s["exactQuote"] for s in build_review_input(source)["sourceSpans"] if s["kind"] == "observation"] == ["{}"]


def test_actionless_negative_finding_is_retained_never_upgraded_to_support():
    from evaluation.translation_source_alignment import SourceAssessment, evaluate_source_assessment
    payload = build_review_input(inputs())
    raw = raw_review(payload)
    raw["claims"][0]["suggested_revision"] = ""
    assessed = assess_review(payload, raw)
    assert assessed["nonActionableFindings"] == [raw["claims"][0]["claim_id"]]
    assert assessed["modelReviewHasUnresolvedFindings"] and not assessed["completeAnswerApproved"]
    strict = SourceAssessment.model_validate({**raw, "claims": [
        {k: v for k, v in row.items() if k != "draft_span_id"} for row in raw["claims"]]})
    with pytest.raises(ValueError):
        evaluate_source_assessment(payload, strict)


def test_exact_block_locations_and_all_fields_retained_without_observation_laundering():
    source = inputs()
    payload = build_review_input(source)
    assert payload["candidate"] == {k: source["candidate"][k] for k in ("draft", "notes")}
    assert payload["completeCandidateDigest"] == sha256_json(source["candidate"])
    for unit in payload["claims"]:
        if unit["facet"] == "all_claims_in_text_block":
            span = next(s for s in payload["draftSpans"] if s["draft_span_id"] == unit["declaredValue"]["draftSpanId"])
            assert source["candidate"]["draft"][unit["start"]:unit["end"]] == span["exactQuote"]
    actual = [s for s in payload["sourceSpans"] if s["kind"] == "observation"]
    assert [s["exactQuote"] for s in actual] == ["pyproject.toml", "src"]
    assert all("pip" not in s["exactQuote"] for s in actual)
    assert payload["claims"][-3]["facet"] == "task_relevant_duty_coverage_and_omissions"


def test_reverse_observation_windows_keep_context_offsets_and_no_candidate_credit():
    source = inputs()
    note = "Report: Zed owns ticket X. If loss persists, Rui checks cable C at 14:20.\n尚未恢复。不得写成已恢复。"
    source["observations"]["n0"]["observations"] = {"read": {"notes": note}}
    source["candidate"]["values"] = {"copied": note}
    payload = build_review_input(source)
    assert note not in json.dumps(payload["candidate"], ensure_ascii=False)
    units = [u for u in payload["claims"] if u["facet"] == "observation_to_draft"]
    assert len(units) == 4
    spans = {s["source_span_id"]: s for s in payload["sourceSpans"]}
    for unit in units:
        parent = spans[unit["declaredValue"]["sourceSpanId"]]
        assert parent["exactQuote"] == note
        assert parent["exactQuote"][unit["start"]:unit["end"]] == unit["declaredValue"]["exactQuote"]
        assert unit["declaredValue"]["targetPointer"] == "/candidate/draft"


def test_located_transport_schema_limits_ids_but_never_semantic_verdicts():
    payload = build_review_input(inputs())
    schema = checked_schema(located_output_schema(payload))
    raw = raw_review(payload)
    assert validate_data(schema, raw) == raw
    row = schema["properties"]["claims"]["items"]["properties"]
    assert row["verdict"] == REVIEW_SCHEMA["properties"]["claims"]["items"]["properties"]["verdict"]
    raw["claims"][0]["source_span_ids"] = ["s999"]
    with pytest.raises(ValueError):
        validate_data(schema, raw)


def test_copied_values_remain_hash_bound_but_are_not_reviewed_as_delivered_text():
    source = inputs()
    old = build_review_input(source)
    raw = raw_review(old)
    source["candidate"]["values"]["entries"] = []
    changed = build_review_input(source)
    assert old["candidate"] == changed["candidate"]
    with pytest.raises(ValueError, match="digest"):
        assess_review(changed, raw)


def test_revision_projection_keeps_all_original_text_and_feedback_locations():
    payload = build_review_input(inputs())
    projected = revision_context(payload)
    for field in ("sourceSpans", "candidate", "originalTask", "hostOpenDuties", "completeCandidateDigest"):
        assert projected[field] == payload[field]
    assert projected["originalReviewInputDigest"] == payload["inputDigest"]
    assert len(projected["reviewedLocations"]) == len(payload["claims"])
    for original, location in zip(payload["claims"], projected["reviewedLocations"], strict=True):
        for field in ("claimId", "facet", "pointer"):
            assert location[field] == original[field]
        if original["facet"] == "all_claims_in_text_block":
            span = next(s for s in payload["draftSpans"] if s["draft_span_id"] == original["declaredValue"]["draftSpanId"])
            assert projected["candidate"]["draft"][location["start"]:location["end"]] == span["exactQuote"]
        elif original["facet"] == "observation_to_draft":
            span = next(s for s in projected["sourceSpans"] if s["source_span_id"] == location["sourceSpanId"])
            assert span["exactQuote"][location["start"]:location["end"]] == original["declaredValue"]["exactQuote"]


def patch_for(source):
    return {"draft_digest": sha256_json(source["candidate"]["draft"]), "edits": [{
        "expected_text": "Install using `pip install demo`.", "replacement": "Installation command is not yet observed.",
        "source_span_ids": ["s000"], "rationale": "Withhold an install command until configuration has been inspected."}],
        "notes": source["candidate"]["notes"], "revision_note": "One anchored edit only, not a verified complete answer."}


def test_anchored_patch_preserves_every_unedited_byte_values_and_non_authority():
    source = inputs()
    patch = patch_for(source)
    candidate, report = apply_revision_patch(build_review_input(source), source["candidate"]["values"], patch)
    assert candidate["draft"] == source["candidate"]["draft"].replace(patch["edits"][0]["expected_text"], patch["edits"][0]["replacement"])
    assert candidate["values"] == source["candidate"]["values"]
    assert not report["completeAnswerApproved"] and not report["runtimeAuthorityGranted"]


@pytest.mark.parametrize("mutation", ["digest", "missing", "ambiguous", "overlap", "no_op", "invented_source", "authority", "values"])
def test_patch_cannot_change_contract_guess_anchors_or_hide_no_op(mutation):
    source = inputs()
    patch = patch_for(source)
    if mutation == "digest":
        patch["draft_digest"] = "sha256:" + "0" * 64
    elif mutation == "missing":
        patch["edits"][0]["expected_text"] = "Text never written."
    elif mutation == "ambiguous":
        source["candidate"]["draft"] += "\n\n" + patch["edits"][0]["expected_text"]
        patch["draft_digest"] = sha256_json(source["candidate"]["draft"])
    elif mutation == "overlap":
        patch["edits"].append({**patch["edits"][0], "expected_text": "pip install demo", "replacement": "withheld"})
    elif mutation == "no_op":
        patch["edits"][0]["replacement"] = patch["edits"][0]["expected_text"]
    elif mutation == "invented_source":
        patch["edits"][0]["source_span_ids"] = ["s999"]
    elif mutation == "authority":
        patch["approved"] = True
    else:
        patch["values"] = {"entries": ["invented"]}
    with pytest.raises(ValueError):
        apply_revision_patch(build_review_input(source), source["candidate"]["values"], patch)


def test_empty_patch_is_explicit_no_material_change_not_a_successful_repair():
    source = inputs()
    patch = patch_for(source)
    patch["edits"] = []
    candidate, report = apply_revision_patch(build_review_input(source), source["candidate"]["values"], patch)
    assert candidate == source["candidate"]
    assert report["status"] == "no_material_draft_change" and not report["completeAnswerApproved"]


def test_patch_materialization_cannot_substitute_other_mapped_values():
    source = inputs()
    with pytest.raises(ValueError, match="binding changed"):
        apply_revision_patch(build_review_input(source), {"entries": ["different"]}, patch_for(source))


def test_overlapping_occurrences_are_not_a_unique_anchor():
    source = inputs()
    source["candidate"]["draft"] = "aaa"
    patch = patch_for(source)
    patch["edits"][0]["expected_text"] = "aa"
    with pytest.raises(ValueError, match="exactly once"):
        apply_revision_patch(build_review_input(source), source["candidate"]["values"], patch)


def test_review_contract_uses_original_assessor_and_preserves_non_authority():
    payload = build_review_input(inputs())
    raw = validate_data(checked_schema(REVIEW_SCHEMA), raw_review(payload))
    report = assess_review(payload, raw)
    assert report["allTextUnitsReviewed"] and report["claimCoverage"] == 1
    assert not report["semanticClaimCoverageProven"] and not report["completeAnswerApproved"]
    assert report["modelReviewHasUnresolvedFindings"] and not report["hostDutiesCleared"]


@pytest.mark.parametrize("mutation", ["omitted", "duplicate", "invented_unit", "invented_source", "no_citation", "digest", "authority"])
def test_review_cannot_drop_units_invent_sources_or_self_approve(mutation):
    payload = build_review_input(inputs())
    raw = raw_review(payload)
    if mutation == "omitted":
        raw["claims"].pop()
    elif mutation == "duplicate":
        raw["claims"][-1] = copy.deepcopy(raw["claims"][0])
    elif mutation == "invented_unit":
        raw["claims"][0]["claim_id"] = "imaginary"
    elif mutation == "invented_source":
        raw["claims"][0]["source_span_ids"] = ["s999"]
    elif mutation == "no_citation":
        raw["claims"][0]["verdict"] = "supported"
    elif mutation == "digest":
        raw["input_digest"] = "sha256:" + "0" * 64
    else:
        raw["approved"] = True
    with pytest.raises(ValueError):
        assess_review(payload, raw)


def test_even_all_supported_ai_opinions_do_not_clear_duties_or_authorize():
    payload = build_review_input(inputs())
    raw = raw_review(payload)
    for row in raw["claims"]:
        row.update(verdict="supported", source_span_ids=["s000"])
    result = assess_review(payload, raw)
    assert result["modelReviewHasUnresolvedFindings"]  # No actual draft witnesses supplied.
    assert result["alignmentWarnings"]
    assert not result["completeAnswerApproved"] and not result["runtimeAuthorityGranted"]
    assert not result["hostDutiesCleared"] and not result["semanticEntailmentProven"]


def test_oversized_reviews_fail_instead_of_silently_discarding_tail():
    source = inputs()
    source["candidate"]["draft"] = "\n\n".join("Paragraph " + str(i) for i in range(49))
    with pytest.raises(ValueError, match="never discard"):
        build_review_input(source)
    source = inputs()
    source["observations"]["n0"]["observations"]["read"]["entries"] = list(range(130))
    with pytest.raises(ValueError, match="never truncate"):
        build_review_input(source)


def test_candidate_change_invalidates_preceding_review_binding():
    before = build_review_input(inputs())
    raw = raw_review(before)
    changed = inputs()
    changed["candidate"]["draft"] += "\n\nAll changes applied."
    with pytest.raises(ValueError, match="digest"):
        assess_review(build_review_input(changed), raw)
