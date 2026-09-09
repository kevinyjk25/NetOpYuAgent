"""Source-cited flow proposals. Citation validity is not semantic entailment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field

from evaluation.flow_translation import (
    FlowDraft, FlowSources, Index, IndexedBranch, IndexedDraft, IndexedEffect,
    IndexedEnd, IndexedRead, _write, author_request, lower, review_input,
)
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from evaluation.translation_source_alignment import evaluate_source_assessment
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import StrictModel

PROTOCOL = "source-cited-flow/v1"
Quote = Annotated[str, Field(min_length=8, max_length=1600)]


class CitedEnd(StrictModel):
    kind: Literal["end"]
    outcome: Literal["read_path_completed", "needs_l1", "unsupported"]


class CitedStep(StrictModel):
    operation: Annotated[IndexedRead | IndexedBranch | IndexedEffect | CitedEnd, Field(discriminator="kind")]
    source_quote: Quote
    requires: tuple[Index, ...] = Field(max_length=64, description="Mandatory predecessor step indices. Never include self or successors.")
    true_quote: Quote | None
    false_quote: Quote | None


class OpenIssue(StrictModel):
    kind: Literal["source_ambiguity", "missing_host_capability", "unsupported_control_flow"]
    source_quote: Quote
    question: str = Field(min_length=12, max_length=600,
        description="One actual unresolved fact or capability. If nothing is unresolved, use issues=[]; never add None/no issues commentary.")


class CitedDraft(StrictModel):
    purpose_quotes: tuple[Quote, ...] = Field(min_length=1, max_length=2,
        description="Exact source excerpts expressing BUSINESS objective and data/authority limits, never translator instructions.")
    entry: Index
    steps: tuple[CitedStep, ...] = Field(min_length=1, max_length=64)
    issues: tuple[OpenIssue, ...] = Field(max_length=16)


def _quote(source: str, quote: str) -> None:
    if not quote.strip() or source.count(quote) != 1:
        raise ValueError("quote must be an exact unique nonblank source excerpt")


def project(sources: FlowSources, proposal: CitedDraft) -> FlowDraft:
    """Project selected text and alpha-rename only; never repair edges or clear issues."""
    proposal = CitedDraft.model_validate(proposal.model_dump())
    quotes = list(proposal.purpose_quotes) + [issue.source_quote for issue in proposal.issues]
    nodes = []
    for step in proposal.steps:
        quotes.append(step.source_quote)
        if isinstance(step.operation, IndexedBranch):
            if step.true_quote is None or step.false_quote is None:
                raise ValueError("branch requires separate true/false source evidence")
            quotes.extend((step.true_quote, step.false_quote))
        elif step.true_quote is not None or step.false_quote is not None:
            raise ValueError("non-branch cannot declare branch evidence")
        node = step.operation.model_dump(mode="json")
        if isinstance(step.operation, CitedEnd):
            node = IndexedEnd(**node, explanation=step.source_quote).model_dump(mode="json")
        nodes.append(node)
    for quote in quotes:
        _quote(sources.source_text, quote)
    draft = IndexedDraft(purpose="\n".join(proposal.purpose_quotes), entry=proposal.entry,
        nodes=nodes, unresolved_questions=tuple(issue.question for issue in proposal.issues)).named()
    lower(sources, draft)  # Existing type/reference/DAG rules, no second executor.

    # Declared prerequisites must dominate the step on EVERY possible path.
    ids = set(range(len(nodes)))
    predecessors = {index: set() for index in ids}
    for index, node in enumerate(nodes):
        for edge in ("next", "on_true", "on_false"):
            if edge in node:
                predecessors[node[edge]].add(index)
    dominators = {index: {index} if index == proposal.entry else set(ids) for index in ids}
    changed = True
    while changed:
        changed = False
        for index in ids - {proposal.entry}:
            updated = {index} | set.intersection(*(dominators[parent] for parent in predecessors[index]))
            if updated != dominators[index]:
                dominators[index] = updated
                changed = True
    for index, step in enumerate(proposal.steps):
        if len(step.requires) != len(set(step.requires)) or not set(step.requires) <= dominators[index] - {index}:
            raise ValueError("declared prerequisite must dominate step on every path")
    return draft


def cited_review_input(sources: FlowSources, proposal: CitedDraft) -> dict:
    draft = project(sources, proposal)
    packet = review_input(sources, draft)
    packet.pop("inputDigest")
    packet["groundingProtocol"] = PROTOCOL
    packet["citedProposal"] = proposal.model_dump(mode="json")
    for index, step in enumerate(proposal.steps):
        pointer = f"/steps/{index}"
        # The judgment must verify meaning, not merely find the quote substring.
        packet["claims"].append({"claimId": f"claim-{len(packet['claims']) + 1:04d}",
            "pointer": pointer, "l05Pointer": pointer, "l0Pointer": f"/nodes/{index}",
            "facet": "source_entails_operation_polarity_and_complete_prerequisites",
            "declaredValue": step.model_dump(mode="json"), "requiredEvidenceKinds": ["skill", "host"]})
    for index, issue in enumerate(proposal.issues):
        pointer = f"/issues/{index}"
        packet["claims"].append({"claimId": f"claim-{len(packet['claims']) + 1:04d}",
            "pointer": pointer, "l05Pointer": pointer, "l0Pointer": None,
            "facet": "actual_unresolved_fact_not_model_commentary", "declaredValue": issue.model_dump(mode="json"),
            "requiredEvidenceKinds": ["skill", "host"]})
    if len(packet["claims"]) > 256:
        raise ValueError("review exceeds 256 claims; never truncate")
    return {**packet, "inputDigest": sha256_json(packet)}


def assess_cited(sources: FlowSources, proposal: CitedDraft, review: ReadL05Review) -> dict:
    review = ReadL05Review.model_validate(review.model_dump())
    packet = cited_review_input(sources, proposal)
    assessment = evaluate_source_assessment(packet, review.assessment)
    judgments = {item.claim_id: item for item in review.assessment.claims}
    for claim in packet["claims"]:
        judgment = judgments[claim["claimId"]]
        if judgment.verdict == "supported" and claim.get("requiredCitationId") and claim["requiredCitationId"] not in judgment.source_span_ids:
            raise ValueError("coverage must cite the exact source requirement")
    accepted = all(item.verdict == "supported" for item in review.assessment.claims) and not proposal.issues
    body = {"inputDigest": packet["inputDigest"], "reviewDigest": sha256_json(review.model_dump(mode="json")),
        "status": "review_supported_inactive_flow" if accepted else "blocked", "assessment": assessment,
        "runtimeAuthorityGranted": False, "semanticAlignmentProven": False,
        "issues": [issue.model_dump(mode="json") for issue in proposal.issues]}
    return {**body, "reportDigest": sha256_json(body)}


def request(sources: FlowSources) -> dict:
    wire = author_request(sources)
    payload = json.loads(wire["messages"][1]["content"])
    for row in payload["hostReadTools"]:
        contract = sources.reads[row["name"]]
        row["businessDescription"] = contract.metadata.description
        row["sourceDeclarations"] = [source.text for source in contract.spec.sources if source.role == "skill"]
    payload["proposalRules"] = {
        "purpose": "Select exact source business-objective and limitation excerpts; do not describe translation.",
        "steps": "operation uses zero-based steps indices; source_quote must justify THIS operation. For an end, the quote becomes its explanation verbatim.",
        "conditions": "true_quote must support the on_true destination and false_quote the on_false destination. For non-branches both are null. Same scalar type never establishes the business meaning of a field.",
        "dependencies": "requires lists all mandatory predecessor steps, not only value references. If a prerequisite capability is absent, stop unsupported BEFORE any dependent operation. Never invent a substitute fact/tool/condition.",
        "unsupported": "Use a reachable unsupported end at the unavailable step; no fake loops or successful completion. needs_l1 is reasoning handoff, not missing-tool recovery or authorization.",
        "issues": "Use [] when no actual issue exists. Otherwise state a concise source-grounded missing fact/capability, not None, self-analysis, or draft repair commentary.",
        "citation": "Copy exact unique excerpts from sourceSkill only. Host declarations explain tool semantics, not permission and not sourceSkill quotes.",
    }
    wire["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    wire["format"] = CitedDraft.model_json_schema()
    # Same decoder compatibility as the frozen indexed protocol; full validation remains.
    def compatible(value):
        if isinstance(value, dict):
            return {key: compatible(item) for key, item in value.items() if key not in {"minLength", "maxLength"}}
        if isinstance(value, list):
            return [compatible(item) for item in value]
        return value
    wire["format"] = compatible(wire["format"])
    return wire


def author_cited(sources: FlowSources, output: Path) -> dict:
    wire = request(sources)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "sources.json", sources.model_dump(mode="json"))
    status = {}
    try:
        model = OllamaAnchoredAuthorAdapter().preflight()
        _write(output / "request.json", {"wireRequest": wire, "model": model, "draftProtocol": PROTOCOL})
        started = time.monotonic()
        with httpx.Client(timeout=240, trust_env=False) as client:
            response = client.post("http://127.0.0.1:11434/api/chat", json=wire)
        _write(output / "response.json", {"httpStatus": response.status_code, "body": response.text,
            "latencyMs": (time.monotonic() - started) * 1000, "timingScope": "post_including_wait_excluding_preflight"})
        response.raise_for_status()
        proposal = CitedDraft.model_validate_json(response.json()["message"]["content"])
        _write(output / "cited-proposal.json", proposal.model_dump(mode="json"))
        draft = project(sources, proposal)
        _write(output / "draft.json", draft.model_dump(mode="json"))
        packet = cited_review_input(sources, proposal)
        _write(output / "review-input.json", packet)
        status = {"status": "awaiting_source_review", "inputDigest": packet["inputDigest"]}
    except Exception as error:
        status = {"status": "blocked", "errorType": type(error).__name__, "error": str(error)}
    _write(output / "status.json", status)
    return status


def load_cited(root: Path) -> tuple[FlowSources, CitedDraft]:
    sources = FlowSources.model_validate_json((root / "sources.json").read_text())
    saved = json.loads((root / "request.json").read_text())
    if saved["draftProtocol"] != PROTOCOL or saved["wireRequest"] != request(sources):
        raise ValueError("cited protocol/request drift")
    response = json.loads((root / "response.json").read_text())
    if response["httpStatus"] != 200:
        raise ValueError("model request failed")
    proposal = CitedDraft.model_validate_json(json.loads(response["body"])["message"]["content"])
    if proposal.model_dump(mode="json") != json.loads((root / "cited-proposal.json").read_text()):
        raise ValueError("cited proposal differs from raw model response")
    if project(sources, proposal).model_dump(mode="json") != json.loads((root / "draft.json").read_text()):
        raise ValueError("projected graph differs from raw proposal")
    if cited_review_input(sources, proposal) != json.loads((root / "review-input.json").read_text()):
        raise ValueError("cited review drift")
    return sources, proposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("author")
    generate.add_argument("sources", type=Path)
    generate.add_argument("output", type=Path)
    review = sub.add_parser("assess")
    review.add_argument("root", type=Path)
    review.add_argument("review", type=Path)
    review.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "author":
        result = author_cited(FlowSources.model_validate_json(args.sources.read_text()), args.output)
        print(json.dumps(result, ensure_ascii=False))
    else:
        _write(args.output, assess_cited(*load_cited(args.root), ReadL05Review.model_validate_json(args.review.read_text())))


if __name__ == "__main__":
    main()
