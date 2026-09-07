"""Evidence-bound resolution sidecars; originals stay immutable and reviews renew.

Exact quotes prove attribution, not entailment. Answers are reviewable proposals,
never an automatic instruction to clear questions or grant execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.read_l05_review import ReadL05Review, build_read_review_input, implementation_digest
from evaluation.translation_source_alignment import evaluate_source_assessment
from network_runtime.contracts import sha256_json
from network_runtime.l0.compiler import compile_documents
from network_runtime.l0.read_l05 import ReadL05Proposal


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResolutionCitation(StrictModel):
    role: Literal["skill", "tool", "adapter"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    exact_quote: str = Field(min_length=1)


class QuestionAnswer(StrictModel):
    question_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=12, max_length=2400)
    citations: tuple[ResolutionCitation, ...] = Field(min_length=1, max_length=8)


class ReadQuestionResolution(StrictModel):
    parent_proposal_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    author_id: str = Field(min_length=1)
    author_kind: Literal["ai_role_simulation", "test_fixture"]
    answers: tuple[QuestionAnswer, ...] = Field(min_length=1, max_length=64)


def prepare_resolution(parent: ReadL05Proposal, resolution: ReadQuestionResolution) -> tuple[ReadL05Proposal, dict]:
    parent = ReadL05Proposal.model_validate(parent.model_dump(by_alias=True))
    resolution = ReadQuestionResolution.model_validate(resolution.model_dump())
    parent_digest = sha256_json(parent.model_dump(by_alias=True, mode="json"))
    if resolution.parent_proposal_digest != parent_digest:
        raise ValueError("resolution parent proposal digest mismatch")
    indices = [item.question_index for item in resolution.answers]
    if sorted(indices) != list(range(len(parent.unresolved_questions))):
        raise ValueError("resolve every parent question exactly once; do not drop or duplicate questions")
    source_by_role = {source.role: source for source in parent.operation.sources}
    citations = []
    answer_claims = []
    for answer in sorted(resolution.answers, key=lambda item: item.question_index):
        if parent.unresolved_questions[answer.question_index] != answer.question:
            raise ValueError("resolution question text differs from parent")
        quoted_ids = []
        unique = set()
        for index, citation in enumerate(answer.citations):
            source = source_by_role[citation.role]
            key = (citation.role, citation.start, citation.end)
            if key in unique:
                raise ValueError("duplicate resolution citation")
            unique.add(key)
            if citation.end > len(source.text) or source.text[citation.start:citation.end] != citation.exact_quote:
                raise ValueError("resolution quote/offset differs from pinned source")
            identifier = f"answer-{answer.question_index:04d}-{index:02d}"
            quoted_ids.append(identifier)
            citations.append({
                "source_span_id": identifier, "kind": citation.role, "path": source.origin,
                "start": citation.start, "end": citation.end, "exactQuote": citation.exact_quote,
                "sourceDigest": source.sha256,
            })
        answer_claims.append({
            "claimId": f"resolution-{answer.question_index:04d}",
            "pointer": f"/answers/{indices.index(answer.question_index)}/answer",
            "l05Pointer": f"/unresolvedQuestions/{answer.question_index}", "l0Pointer": None,
            "facet": "question_answer_entailment_and_sufficiency",
            "declaredValue": {"question": answer.question, "answer": answer.answer},
            "requiredEvidenceKinds": sorted({item.role for item in answer.citations}),
            "requiredCitationIds": quoted_ids,
        })
    # A candidate child is an artifact for review, not automatically admitted.
    child = parent.model_copy(update={"unresolved_questions": ()}, deep=True)
    body = build_read_review_input(child)
    body.pop("inputDigest")
    body["inputProtocol"] = "effect-runtime.io/read-question-resolution/v1"
    body["parentProposalDigest"] = parent_digest
    body["resolutionDigest"] = sha256_json(resolution.model_dump(mode="json"))
    body["originalQuestions"] = list(parent.unresolved_questions)
    body["sourceSpans"].extend(citations)
    body["claims"].extend(answer_claims)
    if len(body["claims"]) > 256:
        raise ValueError("resolution review exceeds 256 claims")
    body["reviewInstruction"] += (
        " Review every proposed answer for semantic entailment AND whether it resolves the exact question. "
        "Cite all requiredCitationIds for supported/contradicted answer claims. A real quote alone is not enough. "
        "If insufficient, the child is blocked; all base claims need a fresh review too."
    )
    return child, {**body, "inputDigest": sha256_json(body)}


def assess_resolution(parent: ReadL05Proposal, resolution: ReadQuestionResolution, review: ReadL05Review) -> dict:
    child, payload = prepare_resolution(parent, resolution)
    review = ReadL05Review.model_validate(review.model_dump())
    result = evaluate_source_assessment(payload, review.assessment)
    for row in result["rows"]:
        if row["verdict"] != "insufficient_evidence" and not set(row.get("requiredCitationIds", [])) <= set(row["source_span_ids"]):
            raise ValueError("answer review must cite the exact proposed evidence, not just its source kind")
    blockers = [row for row in result["rows"] if row["verdict"] != "supported"]
    compiled = None if blockers else compile_documents([child.to_manifest()])[0].model_dump(by_alias=True, mode="json")
    body = {
        "apiVersion": payload["inputProtocol"], "inputDigest": payload["inputDigest"],
        "parentProposalDigest": payload["parentProposalDigest"], "childProposalDigest": payload["proposalDigest"],
        "resolutionDigest": payload["resolutionDigest"], "reviewDigest": sha256_json(review.model_dump(mode="json")),
        "reviewerKind": review.reviewer_kind, "reviewerId": review.reviewer_id,
        "status": "blocked" if blockers else "review_supported_inactive_candidate",
        "assessment": result, "blockers": blockers, "compiledContract": compiled,
        "revisedProposal": child.model_dump(by_alias=True, mode="json"),
        "originalQuestions": payload["originalQuestions"],
        "onlyUnresolvedQuestionsChanged": True,
        "implementationDigest": sha256_json({
            "review": implementation_digest(),
            "resolution": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }),
        "runtimeAuthorityGranted": False, "semanticAlignmentProven": False,
        "wholeSkillCoverageProven": False, "humanIndependentEvidence": False, "goldAuthored": False,
    }
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("packet", "assess"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("resolution", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = ReadL05Proposal.model_validate_json(args.parent.read_text())
    resolution = ReadQuestionResolution.model_validate_json(args.resolution.read_text())
    if args.command == "packet":
        value = prepare_resolution(parent, resolution)[1]
    else:
        if args.review is None:
            parser.error("assess requires --review")
        value = assess_resolution(parent, resolution, ReadL05Review.model_validate_json(args.review.read_text()))
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
