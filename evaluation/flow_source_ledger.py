"""Total source disposition ledger with mutually exclusive mapping shapes."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from jsonschema import Draft202012Validator
from pydantic import Field

from evaluation.flow_semantics import Constraint, RULES, SemanticFlow, compile_semantic, request as semantic_request
from evaluation.flow_source_selection import SelectedIssue, spans
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import Statement
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_source_alignment import evaluate_source_assessment
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import StrictModel

PROTOCOL = "total-source-ledger-flow/v1"


class Note(StrictModel):
    explanation: str = Field(min_length=12, max_length=600)


class OperationUse(Note):
    kind: Literal["operation"]
    node_pointers: tuple[str, ...] = Field(min_length=1, max_length=64)


class DocumentationUse(Note):
    kind: Literal["documentation"]


class HostUse(Note):
    kind: Literal["host_rule_reference"]
    host_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=8)


class FlowUse(Note):
    kind: Literal["flow_reference"]
    node_pointers: tuple[str, ...] = Field(min_length=1, max_length=64)


class UnresolvedUse(Note):
    kind: Literal["unresolved"]


Disposition = Annotated[OperationUse | DocumentationUse | HostUse | FlowUse | UnresolvedUse, Field(discriminator="kind")]


class LedgerFlow(StrictModel):
    objective_source_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_dispositions: dict[str, Annotated[tuple[Disposition, ...], Field(min_length=1, max_length=8)]]
    steps: tuple[Statement, ...] = Field(min_length=1, max_length=64)
    issues: tuple[SelectedIssue, ...] = Field(max_length=16)


def body_ids(sources: FlowSources) -> list[str]:
    return [key for key, text in spans(sources).items() if not text.lstrip().startswith("#")]


def request(sources: FlowSources) -> dict:
    wire = semantic_request(sources)
    schema = wire["format"]
    model_schema = LedgerFlow.model_json_schema()
    schema["title"] = "LedgerFlow"
    schema["properties"].pop("constraints")
    schema["$defs"].pop("Constraint")
    schema["required"] = model_schema["required"]
    for name in ("OperationUse", "DocumentationUse", "HostUse", "FlowUse", "UnresolvedUse"):
        schema["$defs"][name] = model_schema["$defs"][name]
    schema["$defs"]["HostUse"]["properties"]["host_rule_ids"]["items"]["enum"] = list(RULES)
    ids = body_ids(sources)
    entry = model_schema["properties"]["source_dispositions"]["additionalProperties"]
    schema["properties"]["source_dispositions"] = {"type": "object", "properties": {key: entry for key in ids},
        "required": ids, "additionalProperties": False}

    def compatible(value):
        if isinstance(value, dict):
            return {k: compatible(v) for k, v in value.items() if k not in {"minLength", "maxLength"}}
        return [compatible(v) for v in value] if isinstance(value, list) else value
    schema = compatible(schema)
    Draft202012Validator.check_schema(schema)
    # Preserve general tree instructions, replace only the superseded constraint contract.
    instruction = wire["messages"][0]["content"].split(" Classify actual restrictions using constraints,")[0]
    instruction = instruction.replace("constraints separately preserve all applicable restrictions.", "source_dispositions accounts for every non-heading source paragraph.")
    instruction += (
        " source_dispositions must contain every supplied body source ID, each with one or more precise dispositions. "
        "operation maps a source operation to actual tree node_pointers. documentation preserves interpretation limits without claiming checks. "
        "host_rule_reference cites only actual host_rule_ids for the specific clauses they check, not live health or business truth. "
        "flow_reference cites actual node_pointers for control-flow constraints. unresolved retains a missing mapping and blocks activation. "
        "Split paragraphs with mixed clauses into multiple dispositions when needed. Each shape permits only its own fields. "
        "Do not label an execution requirement as documentation to bypass it, or cite an unrelated node. "
        "Bilingual equivalents still need separate source entries; reuse correct mappings instead of dropping a paragraph. "
        "When a mandatory capability is missing, map its requirement to the explicit unsupported node and preserve a real issue. "
        "The compiler retains all text; a complete ledger is only accounting, not semantic truth or execution proof. Return only the schema object."
    )
    payload = json.loads(wire["messages"][1]["content"])
    payload["outputSchema"] = schema
    payload["requiredDispositionSourceIds"] = ids
    wire.update(format=schema, messages=[{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    return wire


def compile_ledger(sources: FlowSources, proposal: LedgerFlow) -> dict:
    proposal = LedgerFlow.model_validate(proposal.model_dump())
    Draft202012Validator(request(sources)["format"]).validate(proposal.model_dump(mode="json"))
    if set(proposal.source_dispositions) != set(body_ids(sources)):
        raise ValueError("every body source requires explicit disposition; no missing/extra rows")
    constraints = []
    for key, uses in proposal.source_dispositions.items():
        for use in uses:
            if use.kind != "operation":
                constraints.append(Constraint(source_ids=[key], handling=use.kind,
                    host_rule_ids=getattr(use, "host_rule_ids", ()), node_pointers=getattr(use, "node_pointers", ()), explanation=use.explanation))
    semantic = SemanticFlow(objective_source_ids=proposal.objective_source_ids,
        constraints=constraints, steps=proposal.steps, issues=proposal.issues)
    compiled = compile_semantic(sources, semantic)
    pointers = {row["treePointer"] for row in compiled["origins"]}
    packet = compiled["reviewInput"]
    packet.pop("inputDigest")
    catalog = spans(sources)
    citation_ids = dict(zip(catalog, (row["source_span_id"] for row in packet["sourceSpans"] if row["kind"] == "skill"), strict=True))
    for key, uses in proposal.source_dispositions.items():
        for index, use in enumerate(uses):
            if isinstance(use, OperationUse):
                if len(set(use.node_pointers)) != len(use.node_pointers) or not set(use.node_pointers) <= pointers:
                    raise ValueError("operation disposition requires exact existing tree nodes")
            pointer = f"/source_dispositions/{key}/{index}"
            packet["claims"].append({"claimId": f"claim-{len(packet['claims']) + 1:04d}", "pointer": pointer,
                "l05Pointer": pointer, "l0Pointer": None, "facet": "source_disposition_is_faithful_not_merely_present",
                "declaredValue": use.model_dump(mode="json"), "requiredEvidenceKinds": ["skill", "host"] if isinstance(use, HostUse) else ["skill"],
                "requiredCitationIds": [citation_ids[key]] + (["host-rules"] if isinstance(use, HostUse) else [])})
    if len(packet["claims"]) > 256:
        raise ValueError("ledger review exceeds 256 claims; split scope, never truncate")
    packet.update(ledgerProtocol=PROTOCOL, ledgerProposal=proposal.model_dump(mode="json"))
    packet["inputDigest"] = sha256_json(packet)
    compiled.pop("reportDigest")
    compiled.update(protocol=PROTOCOL, reviewInput=packet, sourceDispositionComplete=True, sourceDispositionTruthProven=False)
    compiled["reportDigest"] = sha256_json(compiled)
    return compiled


def assess_ledger(sources: FlowSources, proposal: LedgerFlow, review: ReadL05Review) -> dict:
    review = ReadL05Review.model_validate(review.model_dump())
    packet = compile_ledger(sources, proposal)["reviewInput"]
    assessment = evaluate_source_assessment(packet, review.assessment)
    judgments = {row.claim_id: row for row in review.assessment.claims}
    for claim in packet["claims"]:
        needed = claim.get("requiredCitationIds", []) + ([claim["requiredCitationId"]] if "requiredCitationId" in claim else [])
        item = judgments[claim["claimId"]]
        if item.verdict == "supported" and not set(needed) <= set(item.source_span_ids):
            raise ValueError("supported ledger must cite its own source and rule evidence")
    unresolved = any(use.kind == "unresolved" for uses in proposal.source_dispositions.values() for use in uses)
    accepted = not proposal.issues and not unresolved and all(row.verdict == "supported" for row in review.assessment.claims)
    body = {"status": "review_supported_inactive_flow" if accepted else "blocked", "assessment": assessment,
        "inputDigest": packet["inputDigest"], "reviewDigest": sha256_json(review.model_dump(mode="json")),
        "reviewerId": review.reviewer_id, "reviewerKind": review.reviewer_kind,
        "runtimeAuthorityGranted": False, "allRequirementsImplemented": False, "semanticAlignmentProven": False}
    return {**body, "reportDigest": sha256_json(body)}
