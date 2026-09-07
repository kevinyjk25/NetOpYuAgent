"""Separate source retention, declared constraint mappings and actual enforcement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import Field

from evaluation.flow_grounded_translation import project
from evaluation.flow_source_selection import SelectedIssue, expand, spans
from evaluation.flow_translation import FlowDraft, FlowSources, lower, review_input
from evaluation.flow_tree import FlowTree, Statement, compile_tree, tree_review_input
from evaluation.flow_tree_capabilities import bounded_request, host_schema
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_source_alignment import evaluate_source_assessment
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import StrictModel

PROTOCOL = "source-constraint-flow/v1"
ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "read_input_shape": {"path": "network_runtime/l0/read_contracts.py", "meaning": "Validate declared scalar input names, types and requiredness; not user intent."},
    "read_access": {"path": "network_runtime/l0/read_execution.py", "meaning": "Explicit host identity, exact binding, role/scope/resource authorization before provider read; source text grants no authority."},
    "read_result_shape": {"path": "network_runtime/l0/read_contracts.py", "meaning": "Validate declared returned shape; not live health, provenance or business truth."},
    "observation_error_blocks": {"path": "network_runtime/l0/flow.py", "meaning": "Read/input/access/reference/local-age failures block, never choose false branch as recovery; no proof of source freshness."},
}


def rule_catalog() -> dict:
    return {key: {**row, "implementationDigest": "sha256:" + hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()}
            for key, row in RULES.items()}


class Constraint(StrictModel):
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    handling: Literal["documentation", "host_rule_reference", "flow_reference", "unresolved"]
    host_rule_ids: tuple[str, ...] = Field(max_length=8)
    node_pointers: tuple[str, ...] = Field(max_length=64)
    explanation: str = Field(min_length=12, max_length=600)


class SemanticFlow(StrictModel):
    objective_source_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    constraints: tuple[Constraint, ...] = Field(max_length=64)
    steps: tuple[Statement, ...] = Field(min_length=1, max_length=64)
    issues: tuple[SelectedIssue, ...] = Field(max_length=16)


def schema(sources: FlowSources) -> dict:
    result = host_schema(sources)
    semantic = SemanticFlow.model_json_schema()
    result["title"] = "SemanticFlow"
    result["properties"].pop("business_source_ids")
    result["properties"].update({key: semantic["properties"][key] for key in ("objective_source_ids", "constraints")})
    result["required"] = list(semantic["required"])
    result["$defs"]["Constraint"] = semantic["$defs"]["Constraint"]
    catalog = spans(sources)
    body_ids = [key for key, text in catalog.items() if not text.lstrip().startswith("#")]
    if not body_ids:
        raise ValueError("source has no non-heading operation evidence")
    result["properties"]["objective_source_ids"]["items"]["enum"] = list(catalog)
    fields = result["$defs"]["Constraint"]["properties"]
    fields["source_ids"]["items"]["enum"] = body_ids
    fields["host_rule_ids"]["items"]["enum"] = list(RULES)
    for name in ("TreeRead", "TreeIf", "TreeEnd", "TreeEffect"):
        for key, value in result["$defs"].get(name, {}).get("properties", {}).items():
            if key.endswith("source_id"):
                value["enum"] = body_ids

    def decoder_compatible(value):
        if isinstance(value, dict):
            return {k: decoder_compatible(v) for k, v in value.items() if k not in {"minLength", "maxLength"}}
        return [decoder_compatible(v) for v in value] if isinstance(value, list) else value
    result = decoder_compatible(result)
    Draft202012Validator.check_schema(result)
    return result


def request(sources: FlowSources) -> dict:
    wire = bounded_request(sources)
    payload = json.loads(wire["messages"][1]["content"])
    payload["outputSchema"] = schema(sources)
    payload["hostRuleCatalog"] = rule_catalog()
    wire["format"] = payload["outputSchema"]
    system = wire["messages"][0]["content"]
    system = system.replace("business_source_ids selects the business objective AND its applicable limitations.",
        "objective_source_ids selects only the business objective; constraints separately preserve all applicable restrictions.")
    system += (
        " Classify actual restrictions using constraints, with exact non-heading source_ids and an explanation of each clause. "
        "documentation retains a data/interpretation restriction without claiming automatic enforcement; it has no host_rule_ids or node_pointers. "
        "host_rule_reference names only actual catalog rules and has no node_pointers; rule existence is not execution evidence. "
        "flow_reference links actual tree JSON pointers and has no host_rule_ids; it must explain the control-flow restriction. "
        "unresolved retains a constraint without an available mapping and has no references. Do not label an executable requirement as documentation to bypass it. "
        "A paragraph can contain several restrictions: use separate rows when handling differs. Do not omit data/authority limitations. "
        "Operation and constraint references must cite supporting body text, not headings. The compiler archives all source text, "
        "but archive presence never proves requirement coverage. Every original requirement still undergoes source review. "
        "Do not invent source requirements or claim any rule has already been executed."
    )
    wire["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    return wire


def compile_semantic(sources: FlowSources, proposal: SemanticFlow) -> dict:
    sources = FlowSources.model_validate(sources.model_dump())
    proposal = SemanticFlow.model_validate(proposal.model_dump())
    Draft202012Validator(schema(sources)).validate(proposal.model_dump(mode="json"))
    catalog = spans(sources)
    if len(set(proposal.objective_source_ids)) != len(proposal.objective_source_ids):
        raise ValueError("duplicate objective citation")
    # Bootstrap existing wiring only; all objectives/constraints are rendered below,
    # never silently truncated to the legacy two-span purpose field.
    tree = FlowTree(business_source_ids=proposal.objective_source_ids[:1], steps=proposal.steps, issues=proposal.issues)
    selected, origins = compile_tree(sources, tree)
    pointers = {row["treePointer"] for row in origins}
    for binding in proposal.constraints:
        if any(len(set(values)) != len(values) for values in (binding.source_ids, binding.host_rule_ids, binding.node_pointers)):
            raise ValueError("duplicate constraint references")
        if binding.handling == "host_rule_reference":
            if not binding.host_rule_ids or binding.node_pointers:
                raise ValueError("host rule mapping requires only host rule references")
        elif binding.handling == "flow_reference":
            if not binding.node_pointers or binding.host_rule_ids or not set(binding.node_pointers) <= pointers:
                raise ValueError("flow mapping requires existing tree pointers only")
        elif binding.host_rule_ids or binding.node_pointers:
            raise ValueError("documentation/unresolved rows cannot claim enforcement references")
    purpose = "\n".join(["Business objective:", *(catalog[key] for key in proposal.objective_source_ids),
        "Retained constraints (references are not execution receipts):",
        *(f"[{row.handling}] " + "\n".join(catalog[key] for key in row.source_ids) for row in proposal.constraints)])
    original = project(sources, expand(sources, selected))
    draft = FlowDraft.model_validate({**original.model_dump(mode="json"), "purpose": purpose})
    flow, qualified = lower(sources, draft)
    packet = review_input(sources, draft)
    packet.pop("inputDigest")
    old_packet = tree_review_input(sources, tree)
    # Keep the existing full bidirectional review and append, never replace, checks.
    extra = [claim for claim in old_packet["claims"] if claim["facet"] in {
        "source_entails_operation_polarity_and_complete_prerequisites", "actual_unresolved_fact_not_model_commentary"}]
    packet["claims"].extend(extra)
    rules = rule_catalog()
    rule_text = json.dumps(rules, ensure_ascii=False)
    packet["sourceSpans"].append({"source_span_id": "host-rules", "kind": "host", "path": "implementation-bound-rule-catalog",
        "start": 0, "end": len(rule_text), "exactQuote": rule_text, "sourceDigest": sha256_json(rules)})
    # Both catalogs preserve nonblank line order, including duplicate line text.
    citation_ids = dict(zip(catalog, (row["source_span_id"] for row in packet["sourceSpans"] if row["kind"] == "skill"), strict=True))
    for index, binding in enumerate(proposal.constraints):
        pointer = f"/constraints/{index}"
        packet["claims"].append({"pointer": pointer, "l05Pointer": pointer, "l0Pointer": "/purpose",
            "facet": "constraint_fidelity_and_handling_not_retention_as_enforcement",
            "declaredValue": binding.model_dump(mode="json"),
            "requiredEvidenceKinds": ["skill", "host"] if binding.handling == "host_rule_reference" else ["skill"],
            "requiredCitationIds": [citation_ids[key] for key in binding.source_ids] + (["host-rules"] if binding.handling == "host_rule_reference" else [])})
    if len(packet["claims"]) > 256:
        raise ValueError("review exceeds 256 claims; split scope, never truncate")
    for index, claim in enumerate(packet["claims"]):
        claim["claimId"] = f"claim-{index + 1:04d}"
    archive = {"text": sources.source_text, "sourcePath": sources.source_path, "spans": catalog,
        "textDigest": "sha256:" + hashlib.sha256(sources.source_text.encode()).hexdigest(), "executable": False}
    packet.update(semanticProtocol=PROTOCOL, semanticProposal=proposal.model_dump(mode="json"),
        sourceArchive=archive, origins=origins, ruleCatalog=rules)
    packet["inputDigest"] = sha256_json(packet)
    body = {"protocol": PROTOCOL, "flow": flow.model_dump(mode="json"), "flowDigest": qualified["flowDigest"],
        "sourceArchive": archive, "origins": origins, "reviewInput": packet,
        "constraintMappings": [{**row.model_dump(mode="json"), "executionStatus": "not_executed", "semanticSupport": "pending_review"} for row in proposal.constraints],
        "status": "compiled_pending_source_review_not_executable", "runtimeAuthorityGranted": False,
        "allSourceTextRetained": True, "allRequirementsImplemented": False}
    return {**body, "reportDigest": sha256_json(body)}


def assess_semantic(sources: FlowSources, proposal: SemanticFlow, review: ReadL05Review) -> dict:
    review = ReadL05Review.model_validate(review.model_dump())
    packet = compile_semantic(sources, proposal)["reviewInput"]
    assessment = evaluate_source_assessment(packet, review.assessment)
    judgments = {row.claim_id: row for row in review.assessment.claims}
    for claim in packet["claims"]:
        needed = claim.get("requiredCitationIds", []) + ([claim["requiredCitationId"]] if "requiredCitationId" in claim else [])
        item = judgments[claim["claimId"]]
        if item.verdict == "supported" and not set(needed) <= set(item.source_span_ids):
            raise ValueError("supported coverage/mapping must cite exact requirement and rule evidence")
    accepted = not proposal.issues and not any(row.handling == "unresolved" for row in proposal.constraints) and all(row.verdict == "supported" for row in review.assessment.claims)
    body = {"status": "review_supported_inactive_flow" if accepted else "blocked", "assessment": assessment,
        "inputDigest": packet["inputDigest"], "reviewDigest": sha256_json(review.model_dump(mode="json")),
        "reviewerId": review.reviewer_id, "reviewerKind": review.reviewer_kind,
        "runtimeAuthorityGranted": False, "allRequirementsImplemented": False, "semanticAlignmentProven": False}
    return {**body, "reportDigest": sha256_json(body)}
