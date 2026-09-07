"""Source-mapping protocol tests, not independent semantic judgments."""

import json

import pytest

from evaluation.read_l05_review import (
    ReadL05Review, assess_read_l05, build_read_review_input, main,
)
from evaluation.translation_source_alignment import ClaimAssessment, SourceAssessment
from network_runtime.l0.compiler import parse_document
from network_runtime.l0.read_l05 import ReadL05Proposal, scaffold_read_l05
from tests.test_l0_read_contracts import EXAMPLE, _parameterized, _raw


def _proposal(parameterized: bool = False) -> ReadL05Proposal:
    scaffold = scaffold_read_l05(parse_document(_parameterized() if parameterized else _raw()))
    # Synthetic all-supported protocol fixture, not a semantic review verdict.
    return scaffold.model_copy(update={"unresolved_questions": ()})


def _review(proposal: ReadL05Proposal, verdict: str = "supported") -> ReadL05Review:
    payload = build_read_review_input(proposal)
    return ReadL05Review(
        reviewer_id="mechanical-protocol-test-only", reviewer_kind="test_fixture",
        assessment=SourceAssessment(
            input_digest=payload["inputDigest"],
            claims=tuple(ClaimAssessment(
                claim_id=claim["claimId"], verdict=verdict,
                source_span_ids=tuple(role + "-0001" for role in claim["requiredEvidenceKinds"]),
                rationale="Synthetic protocol fixture only; this does not establish semantic truth.",
                suggested_revision="Inspect genuine source evidence before claiming support.",
            ) for claim in payload["claims"]),
            scope_note="Mechanical review fixture, not an independent human or model accuracy result.",
        ),
    )


def test_reverse_scaffold_preserves_unresolved_status_and_all_fields() -> None:
    source = parse_document(_raw())
    proposal = scaffold_read_l05(source)
    assert proposal.to_manifest() == source
    report = assess_read_l05(proposal, _review(proposal))
    assert report["status"] == "blocked"
    assert report["compiledContract"] is None
    assert report["blockers"][-1]["l05Pointer"] == "/unresolvedQuestions"


def test_all_supported_never_means_authorized_or_whole_skill_translated() -> None:
    proposal = _proposal()
    report = assess_read_l05(proposal, _review(proposal))
    assert report["status"] == "review_supported_inactive_candidate"
    assert report["compiledContract"]["spec"] == proposal.operation.model_dump(by_alias=True, mode="json")
    assert report["compiledContract"]["metadata"]["description"] == proposal.purpose
    for field in ("runtimeAuthorityGranted", "semanticAlignmentProven", "wholeSkillCoverageProven",
                  "goldAuthored", "humanIndependentEvidence", "sourceAuthenticityVerified"):
        assert report[field] is False


def test_every_scalar_facet_and_schema_has_bidirectional_location() -> None:
    payload = build_read_review_input(_proposal(True))
    rows = payload["claims"]
    for side, name in (("inputSchema", "device"), ("outputSchema", "healthy")):
        selected = [row for row in rows if row["l05Pointer"].startswith(f"/operation/{side}")]
        assert len(selected) == 4
        assert {row["facet"] for row in selected} == {
            "complete_" + side, "field_existence_and_description", "field_type", "field_requiredness",
        }
        assert all(row["l0Pointer"].startswith(f"/spec/{side}") for row in selected)
        assert any(row["declaredValue"] == {"name": name, "required": True} for row in selected)
    for span in payload["sourceSpans"]:
        source = next(item for item in _proposal(True).operation.sources if item.role == span["kind"])
        assert source.text[span["start"]:span["end"]] == span["exactQuote"]
        assert source.sha256 == span["sourceDigest"]


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient_evidence"])
def test_non_supported_findings_point_to_fix_locations(verdict: str) -> None:
    proposal = _proposal()
    report = assess_read_l05(proposal, _review(proposal, verdict))
    assert report["status"] == "blocked" and report["compiledContract"] is None
    assert len(report["blockers"]) == len(build_read_review_input(proposal)["claims"])
    for blocker in report["blockers"]:
        assert blocker["l05Pointer"] and blocker["suggestedRevision"] and blocker["explanation"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "wrong_kind", "empty_revision"])
def test_review_cannot_skip_or_relabel_required_evidence(mutation: str) -> None:
    proposal = _proposal()
    raw = _review(proposal).model_dump()
    rows = list(raw["assessment"]["claims"])
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "unknown":
        rows[0]["source_span_ids"] = ("made-up",)
    elif mutation == "wrong_kind":
        rows[0]["source_span_ids"] = ("tool-0001",)
    else:
        rows[0].update(verdict="insufficient_evidence", suggested_revision="")
    raw["assessment"]["claims"] = rows
    with pytest.raises(ValueError):
        assess_read_l05(proposal, ReadL05Review.model_validate(raw))


def test_proposal_or_source_changes_invalidate_prior_review() -> None:
    proposal = _proposal()
    review = _review(proposal)
    changed = proposal.model_copy(update={"unresolved_questions": ("new source question",)})
    with pytest.raises(ValueError, match="input digest"):
        assess_read_l05(changed, review)
    altered = proposal.model_copy(update={"purpose": "Completely different purpose"})
    with pytest.raises(ValueError, match="purpose"):
        assess_read_l05(altered, review)


def test_model_copy_cannot_inject_unrecognized_fields_in_assessment() -> None:
    proposal = _proposal()
    review = _review(proposal).model_copy(update={"reviewer_kind": "independent_human"})
    with pytest.raises(ValueError):
        assess_read_l05(proposal, review)


def test_empty_evidence_is_allowed_only_as_insufficient_and_remains_blocked() -> None:
    proposal = _proposal()
    raw = _review(proposal, "insufficient_evidence").model_dump()
    for row in raw["assessment"]["claims"]:
        row["source_span_ids"] = ()
    report = assess_read_l05(proposal, ReadL05Review.model_validate(raw))
    assert report["status"] == "blocked"
    assert not report["assessment"]["supportedClaimFraction"]


def test_cli_roundtrip_exports_unresolved_scaffold_and_never_overwrites(tmp_path) -> None:
    proposal_file, packet_file = tmp_path / "l05.json", tmp_path / "packet.json"
    assert main(["scaffold", str(EXAMPLE), "--output", str(proposal_file)]) == 0
    assert main(["review-input", str(proposal_file), "--output", str(packet_file)]) == 0
    assert json.loads(packet_file.read_text())["unresolvedQuestions"]
    before = packet_file.read_bytes()
    with pytest.raises(FileExistsError):
        main(["review-input", str(proposal_file), "--output", str(packet_file)])
    assert packet_file.read_bytes() == before
