"""Host-bound result obligations for mixed flows, not a natural-language judge.

Exact observed-value projection and actual read completion are machine checked.
Open duties remain open; source entailment and completeness of the host's duty
mapping are NOT proved. Neither a model note nor this assessment grants action
authority. Source references reuse existing review/dossier locators, not a new
automatic duty extractor.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping

from pydantic import Field

from network_runtime.contracts import sha256_json

from .flow import StructuredReadNode
from .hybrid import DIGEST, ID, GovernedHybridFlow, ReasoningTask, StrictRegion
from .models import StrictModel
from .structured_reads import read_schema
from .structured_schema import DataBindingError, checked_schema, pointer_value, schema_location, snapshot_json


class Duty(StrictModel):
    id: str = Field(pattern=ID)
    statement: str = Field(min_length=8, max_length=1800)
    source_ref: str = Field(min_length=1, max_length=512)


class ObservedValueDuty(Duty):
    kind: Literal["observed_value"]
    region: str = Field(pattern=ID)
    read_node: str = Field(min_length=1, max_length=128)
    pointer: str = Field(max_length=4096)
    field: str = Field(pattern=ID)


class ReadCompletedDuty(Duty):
    kind: Literal["read_completed"]
    region: str = Field(pattern=ID)
    read_node: str = Field(min_length=1, max_length=128)


class OpenDuty(Duty):
    kind: Literal["open_semantics"]
    reason: Literal["unverified_reasoning", "unsupported_control", "missing_observation", "needs_clarification"]


ResultDuty = Annotated[ObservedValueDuty | ReadCompletedDuty | OpenDuty, Field(discriminator="kind")]


class ResultContract(StrictModel):
    api_version: Literal["netopyu.io/hybrid-result/v1"]
    task_digest: str = Field(pattern=DIGEST)
    source_digest: str = Field(pattern=DIGEST)
    # Digest of the previously reviewed source/duty mapping. A hash is binding,
    # NOT an assertion that the source has been exhaustively/accurately mapped.
    mapping_digest: str = Field(pattern=DIGEST)
    candidate_node: str = Field(pattern=ID)
    duties: tuple[ResultDuty, ...] = Field(min_length=1, max_length=64)


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _standalone(schema, root):
    """Resolve the already validated local schema without rebasing its refs."""
    visited = 0

    def expand(node, depth=0):
        nonlocal visited
        visited += 1
        if depth > 16 or visited > 512:
            raise ValueError("projected result schema exceeds bounded expansion")
        if "$ref" in node:
            return expand(pointer_value(root, node["$ref"][1:]), depth + 1)
        value = {k: snapshot_json(v) for k, v in node.items() if k != "$defs"}
        if "properties" in value:
            value["properties"] = {k: expand(v, depth + 1) for k, v in value["properties"].items()}
        for key in ("items", "additionalProperties"):
            if isinstance(value.get(key), dict):
                value[key] = expand(value[key], depth + 1)
        return value

    return checked_schema(expand(schema))


def _resolve_duties(contract, proposal, reads):
    contract = ResultContract.model_validate(snapshot_json(contract.model_dump(mode="json")))
    if contract.task_digest != proposal.task_digest or contract.source_digest != proposal.source_digest:
        raise ValueError("result contract task/source drift")
    nodes = {n.id: n for n in proposal.nodes}
    candidate = nodes.get(contract.candidate_node)
    if not isinstance(candidate, ReasoningTask) or contract.candidate_node not in proposal.outputs:
        raise ValueError("result contract must name an explicit reasoning output")
    if len({d.id for d in contract.duties}) != len(contract.duties):
        raise ValueError("result duty IDs must be unique")
    fields = {}
    for duty in contract.duties:
        if isinstance(duty, OpenDuty):
            continue
        region = nodes.get(duty.region)
        if not isinstance(region, StrictRegion):
            raise ValueError("result evidence must come from an original strict region, not model/source text")
        selected = [n for n in region.flow.nodes if n.id == duty.read_node and isinstance(n, StructuredReadNode)]
        if len(selected) != 1:
            raise ValueError("result evidence needs an exact declared read node")
        if isinstance(duty, ObservedValueDuty):
            if duty.field in fields:
                raise ValueError("each evidence field must have one unambiguous host binding")
            output = read_schema(reads[selected[0].contract_hash], "output")
            location, _ = schema_location(output, duty.pointer)
            fields[duty.field] = _standalone(location, output)
    return contract, fields


def result_candidate_schema(contract: ResultContract, proposal: GovernedHybridFlow, reads: Mapping) -> dict:
    """Values may be omitted rather than invented; omission remains a duty gap.

    Candidate notes preserve open L1 work, but are never published as verified
    facts. Model-written completion, confidence and evidence locators are absent.
    """
    _, fields = _resolve_duties(contract, proposal, reads)
    return checked_schema(_object({
        "values": _object(fields, []),
        "draft": {"type": "string", "maxLength": 12000},
        "notes": {"type": "array", "maxItems": 6, "items": {"type": "string", "minLength": 1, "maxLength": 2000}},
    }, ["values", "draft", "notes"]))


def bind_result_candidate(contract: ResultContract, proposal: GovernedHybridFlow, reads: Mapping) -> GovernedHybridFlow:
    """Build an INACTIVE candidate-schema revision; caller must obtain new consent.

    The original instructions, graph edges and tool contracts remain intact.
    Only the model output protocol changes. Expected observation values are never
    included; the model gets the host's field-to-observation mapping, not Gold.
    """
    schema = result_candidate_schema(contract, proposal, reads)
    raw = proposal.model_dump(mode="json")
    node = next(n for n in raw["nodes"] if n["id"] == contract.candidate_node)
    node["output_schema"] = schema
    mapping = [{"field": d.field, "region": d.region, "readNode": d.read_node, "pointer": d.pointer}
               for d in contract.duties if isinstance(d, ObservedValueDuty)]
    import json
    node["instructions"] += (
        "\nHOST RESULT PROTOCOL: The requiredOutputSchema replaces any earlier draft output format. "
        "Copy each available mapped observation exactly into values; omit absent values instead of guessing. "
        "Use draft to perform the original open task, including its requested handoff/document format, not merely "
        "to list observations. When observations are insufficient, provide only the supported partial draft. "
        "Use notes for at most six concise limitations; neither draft nor notes is verified by these field checks. "
        "Do not assert overall completion "
        "or introduce evidence references, confidence, approvals or tool calls. The mapping identifies strict "
        "read regions in the supplied observations, not source examples or caller assertions. Field bindings: "
        + json.dumps(mapping, ensure_ascii=False))
    return GovernedHybridFlow.model_validate(raw)


def qualify_result_contract(contract: ResultContract, packet: dict, reads: Mapping) -> dict:
    proposal = GovernedHybridFlow.model_validate(packet["proposal"])
    contract, _ = _resolve_duties(contract, proposal, reads)
    schema = result_candidate_schema(contract, proposal, reads)
    candidate = next(n for n in proposal.nodes if n.id == contract.candidate_node)
    if candidate.output_schema != schema:
        raise ValueError("model output must use the exact host-bound result candidate schema")
    for duty in contract.duties:
        if not isinstance(duty, OpenDuty) and duty.region not in packet["ancestors"][candidate.id]:
            raise ValueError("result evidence must be a completed candidate ancestor")
    body = {"contract": contract.model_dump(mode="json"), "graphDigest": packet["graphDigest"],
            "sourceEntailmentProven": False, "wholeTaskDutyCoverageProven": False}
    return {**body, "contractDigest": sha256_json(body)}


def assess_result(qualified: dict, values: dict, *, graph_status: str) -> dict[str, Any]:
    """Called by the scheduler with its own validated node values, not LLM receipts.

    No I/O, model calls, authorization, string matching of free-form prose or
    inferred task completion. Accepted fields are projections of observations,
    not attestations that an upstream ticket's assertions are true.
    """
    contract = ResultContract.model_validate(qualified["contract"])
    candidate = values.get(contract.candidate_node)
    fields = candidate["values"] if candidate is not None else {}
    rows, evidence, hard_errors = [], [], False
    for duty in contract.duties:
        row = {"dutyId": duty.id, "kind": duty.kind, "statement": duty.statement, "sourceRef": duty.source_ref,
               "contractPointer": "/duties/" + str(len(rows)), "status": "unresolved"}
        if isinstance(duty, OpenDuty):
            row.update(code=duty.reason, repairTarget="reasoning_or_authoring", semanticReviewRequired=True)
            rows.append(row)
            continue
        payloads = values.get(duty.region, {}).get("observations", {})
        row["evidenceLocation"] = {"region": duty.region, "readNode": duty.read_node}
        if duty.read_node not in payloads:
            row.update(code="required_read_not_observed", repairTarget="flow_or_host_binding")
        elif isinstance(duty, ReadCompletedDuty):
            row.update(status="satisfied", code="read_receipt_present", repairTarget=None)
        else:
            row["evidenceLocation"]["pointer"] = duty.pointer
            row["candidatePointer"] = "/values/" + duty.field
            try:
                observed = pointer_value(payloads[duty.read_node], duty.pointer)
            except DataBindingError:
                row.update(code="required_observation_field_missing", repairTarget="observation_or_binding")
            else:
                if duty.field not in fields:
                    row.update(code="candidate_omitted_observed_field", repairTarget="reasoning_output")
                elif sha256_json(fields[duty.field]) != sha256_json(observed):
                    row.update(status="rejected", code="candidate_value_differs_from_observation", repairTarget="reasoning_output_or_binding",
                               candidateValueDigest=sha256_json(fields[duty.field]), observedValueDigest=sha256_json(observed))
                    hard_errors = True
                else:
                    row.update(status="satisfied", code="exact_observed_value", repairTarget=None)
                    evidence.append({"field": duty.field, "label": duty.statement, "value": snapshot_json(observed),
                        "dutyId": duty.id, "evidenceLocation": row["evidenceLocation"],
                        "role": "observed_snapshot_value_not_independent_truth"})
        rows.append(row)
    satisfied = sum(r["status"] == "satisfied" for r in rows)
    complete = graph_status == "governed_graph_completed" and satisfied == len(rows)
    status = ("blocked" if graph_status != "governed_graph_completed" else "rejected" if hard_errors else
              "declared_contract_satisfied" if complete else "partial" if satisfied else "unverified")
    body = {"apiVersion": "netopyu.io/hybrid-result-assessment/v1", "graphDigest": qualified["graphDigest"],
        "contractDigest": qualified["contractDigest"], "mappingDigest": contract.mapping_digest,
        "candidateDigest": sha256_json(candidate) if candidate is not None else None,
        "status": status, "declaredDutyCount": len(rows), "satisfiedDutyCount": satisfied,
        "declaredObligationsSatisfied": complete, "rows": rows, "observedFields": evidence,
        "unverifiedCandidateDraft": candidate["draft"] if candidate is not None else "",
        "unverifiedCandidateNotes": candidate["notes"] if candidate is not None else [],
        "modelDraftVerified": False, "modelNotesVerified": False, "completeAnswerApproved": False, "wholeSkillCorrectnessProven": False,
        "sourceEntailmentProven": False, "wholeTaskDutyCoverageProven": False, "runtimeAuthorityGranted": False,
        "effectAuthorized": False, "freshActionEvidence": False,
        "claimBoundary": "Only declared read receipts and exact observation projections checked; open prose and upstream assertions remain unverified."}
    return {**body, "reportDigest": sha256_json(body)}
