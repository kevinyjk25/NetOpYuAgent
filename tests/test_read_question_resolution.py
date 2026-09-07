"""Revision protocol integrity, never synthetic-review semantic accuracy."""

import pytest

from evaluation.read_l05_review import ReadL05Review
from evaluation.read_question_resolution import (
    QuestionAnswer, ReadQuestionResolution, ResolutionCitation, assess_resolution, prepare_resolution,
)
from evaluation.translation_source_alignment import ClaimAssessment, SourceAssessment
from network_runtime.contracts import sha256_json
from tests.test_read_l05_review import _proposal, _review


def _parent():
    return _proposal().model_copy(update={"unresolved_questions": ("What does it read?", "Does it write?")})


def _resolution(parent):
    source = next(item for item in parent.operation.sources if item.role == "skill")
    return ReadQuestionResolution(
        parent_proposal_digest=sha256_json(parent.model_dump(by_alias=True, mode="json")),
        author_id="protocol-fixture-only", author_kind="test_fixture",
        answers=tuple(QuestionAnswer(
            question_index=index, question=question, answer="Mechanical answer fixture, not semantic proof.",
            citations=(ResolutionCitation(role="skill", start=0, end=len(source.text), exact_quote=source.text),),
        ) for index, question in enumerate(parent.unresolved_questions)),
    )


def _new_review(payload, verdict="supported"):
    return ReadL05Review(
        reviewer_id="protocol-fixture-not-independent", reviewer_kind="test_fixture",
        assessment=SourceAssessment(
            input_digest=payload["inputDigest"], scope_note="Synthetic protocol review only; no source entailment claim.",
            claims=tuple(ClaimAssessment(
                claim_id=row["claimId"], verdict=verdict,
                source_span_ids=tuple(row.get("requiredCitationIds") or [role + "-0001" for role in row["requiredEvidenceKinds"]]),
                rationale="Mechanical protocol fixture, not independently reviewed semantic truth.",
                suggested_revision="Obtain a source-backed answer and renew the complete review.",
            ) for row in payload["claims"]),
        ),
    )


def test_revision_changes_only_questions_and_retains_original() -> None:
    parent = _parent()
    original = parent.model_dump_json(by_alias=True)
    resolution = _resolution(parent)
    child, packet = prepare_resolution(parent, resolution)
    assert parent.model_dump_json(by_alias=True) == original
    assert not child.unresolved_questions
    assert child.to_manifest() == parent.to_manifest()
    assert packet["originalQuestions"] == list(parent.unresolved_questions)
    report = assess_resolution(parent, resolution, _new_review(packet))
    assert report["status"] == "review_supported_inactive_candidate"
    assert report["compiledContract"] and report["onlyUnresolvedQuestionsChanged"]
    assert not report["runtimeAuthorityGranted"] and not report["semanticAlignmentProven"]
    assert report["parentProposalDigest"] != report["childProposalDigest"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_index", "changed_question", "wrong_parent"])
def test_cannot_silently_drop_or_rewrite_questions(mutation):
    parent = _parent()
    data = _resolution(parent).model_dump(mode="json")
    if mutation == "missing":
        data["answers"].pop()
    elif mutation == "duplicate":
        data["answers"][1] = data["answers"][0]
    elif mutation == "unknown_index":
        data["answers"][0]["question_index"] = 100
    elif mutation == "changed_question":
        data["answers"][0]["question"] = "A different question?"
    else:
        data["parent_proposal_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        prepare_resolution(parent, ReadQuestionResolution.model_validate(data))


@pytest.mark.parametrize("mutation", ["quote", "offset", "role", "duplicate"])
def test_citations_must_match_pinned_source(mutation):
    parent = _parent()
    data = _resolution(parent).model_dump(mode="json")
    citation = data["answers"][0]["citations"][0]
    if mutation == "quote":
        citation["exact_quote"] = "Fabricated source evidence"
    elif mutation == "offset":
        citation["start"] += 1
    elif mutation == "role":
        citation["role"] = "tool"
    else:
        data["answers"][0]["citations"].append(citation.copy())
    with pytest.raises(ValueError):
        prepare_resolution(parent, ReadQuestionResolution.model_validate(data))


def test_old_review_cannot_authorize_resolution() -> None:
    parent = _parent()
    with pytest.raises(ValueError, match="input digest"):
        assess_resolution(parent, _resolution(parent), _review(parent))


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient_evidence"])
def test_evidence_reviewer_disagreement_blocks_revision(verdict):
    parent = _parent()
    resolution = _resolution(parent)
    _, packet = prepare_resolution(parent, resolution)
    report = assess_resolution(parent, resolution, _new_review(packet, verdict))
    assert report["status"] == "blocked" and report["compiledContract"] is None


def test_exact_resolution_citations_cannot_be_replaced_with_topic_only_source() -> None:
    parent = _parent()
    resolution = _resolution(parent)
    _, packet = prepare_resolution(parent, resolution)
    raw = _new_review(packet).model_dump()
    for row in raw["assessment"]["claims"]:
        if row["claim_id"].startswith("resolution-"):
            row["source_span_ids"] = ("skill-0001",)
    with pytest.raises(ValueError, match="exact proposed evidence"):
        assess_resolution(parent, resolution, ReadL05Review.model_validate(raw))


def test_changed_answer_requires_another_review() -> None:
    parent = _parent()
    resolution = _resolution(parent)
    _, packet = prepare_resolution(parent, resolution)
    answer = resolution.answers[0].model_copy(update={"answer": "A different answer to the source question."})
    changed = resolution.model_copy(update={"answers": (answer, resolution.answers[1])})
    with pytest.raises(ValueError, match="input digest"):
        assess_resolution(parent, changed, _new_review(packet))


def test_resolution_cannot_modify_operation_or_claim_human_authority() -> None:
    parent = _parent()
    for patch in ({"operation": {}}, {"author_kind": "independent_human"}):
        with pytest.raises(ValueError):
            ReadQuestionResolution.model_validate({**_resolution(parent).model_dump(), **patch})
