"""Immutable mixed dependency plans over the original L0 read-flow engine.

Qualification is structural, never semantic approval. Model outputs stay
candidates; a separately installed host gate is needed before using one in a
strict region. This module neither invokes models nor adds an Effect executor.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping

from pydantic import Field

from network_runtime.contracts import sha256_json

from .flow import StructuredFlowProposal, StructuredReadNode, qualify_flow
from .models import StrictModel
from .structured_bindings import compile_binding
from .structured_reads import read_schema
from .structured_schema import checked_schema, snapshot_json

ID = r"^[a-z][a-z0-9_-]{0,47}$"
DIGEST = r"^sha256:[0-9a-f]{64}$"


class NodeBase(StrictModel):
    id: str = Field(pattern=ID)
    depends_on: tuple[str, ...] = Field(max_length=16)
    inputs: dict[str, Any]


class StrictRegion(NodeBase):
    kind: Literal["strict_region"]
    flow: StructuredFlowProposal


class ReasoningTask(NodeBase):
    kind: Literal["reason"]
    instructions: str = Field(min_length=8, max_length=24000)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    binding_id: str = Field(pattern=ID)
    model: str = Field(min_length=1, max_length=100)
    configuration_digest: str = Field(pattern=DIGEST)
    timeout_seconds: float = Field(gt=0, le=360, allow_inf_nan=False)
    max_input_bytes: int = Field(ge=64, le=131072, strict=True)
    max_output_bytes: int = Field(ge=32, le=65536, strict=True)
    max_output_tokens: int = Field(ge=32, le=8192, strict=True)


class CandidateAdmission(NodeBase):
    kind: Literal["admit_candidate"]
    candidate: str = Field(pattern=ID)
    input_schema: dict[str, Any]
    gate_id: str = Field(pattern=ID)
    policy_digest: str = Field(pattern=DIGEST)


class RequiredJoin(NodeBase):
    kind: Literal["join"]
    rule: Literal["all_succeeded"]
    input_schema: dict[str, Any]


HybridNode = Annotated[StrictRegion | ReasoningTask | CandidateAdmission | RequiredJoin, Field(discriminator="kind")]


class GovernedHybridFlow(StrictModel):
    api_version: Literal["netopyu.io/governed-hybrid/v1"]
    source_digest: str = Field(pattern=DIGEST)
    task_digest: str = Field(pattern=DIGEST)
    purpose: str = Field(min_length=1, max_length=1200)
    input_schema: dict[str, Any]
    nodes: tuple[HybridNode, ...] = Field(min_length=1, max_length=16)
    outputs: tuple[str, ...] = Field(min_length=1, max_length=16)
    max_parallel: int = Field(ge=1, le=4, strict=True)
    max_model_calls: int = Field(ge=0, le=8, strict=True)
    timeout_seconds: float = Field(gt=0, le=1800, allow_inf_nan=False)
    failure_policy: Literal["stop_no_downstream"]


def _sources(expression):
    """Let the original binder validate syntax; only collect dependency names."""
    found, pending = set(), [expression]
    while pending:
        item = pending.pop()
        if not isinstance(item, dict):
            continue
        if item.get("kind") in {"reference", "array_length", "column_rows"}:
            found.add(item.get("source"))
        elif item.get("kind") == "object" and isinstance(item.get("fields"), dict):
            pending.extend(item["fields"].values())
        elif item.get("kind") == "array" and isinstance(item.get("items"), list):
            pending.extend(item["items"])
    return found


def qualify_hybrid(proposal: GovernedHybridFlow, reads: Mapping) -> dict:
    proposal = GovernedHybridFlow.model_validate(snapshot_json(proposal.model_dump(mode="json")))
    checked_schema(proposal.input_schema)
    nodes = {n.id: n for n in proposal.nodes}
    if len(nodes) != len(proposal.nodes) or "input" in nodes:
        raise ValueError("hybrid node IDs must be unique and cannot shadow input")
    if len(set(proposal.outputs)) != len(proposal.outputs) or set(proposal.outputs) - nodes.keys():
        raise ValueError("outputs require unique declared nodes")
    for node in nodes.values():
        if len(set(node.depends_on)) != len(node.depends_on) or set(node.depends_on) - nodes.keys() or node.id in node.depends_on:
            raise ValueError("invalid or self-referencing hybrid dependencies")
    order, ancestors = [], {}
    while len(order) != len(nodes):
        ready = [key for key, n in nodes.items() if key not in ancestors and set(n.depends_on) <= ancestors.keys()]
        if not ready:
            raise ValueError("hybrid dependency cycle")
        for key in ready:
            ancestors[key] = set(nodes[key].depends_on)
            for dep in nodes[key].depends_on:
                ancestors[key].update(ancestors[dep])
            order.append(key)
    required = set(proposal.outputs)
    for key in proposal.outputs:
        required.update(ancestors[key])
    if required != nodes.keys():
        raise ValueError("every scheduled node must contribute to an explicit output")
    if sum(isinstance(n, ReasoningTask) for n in nodes.values()) > proposal.max_model_calls:
        raise ValueError("graph exceeds its model-call budget")

    schemas, roles, bindings, regions = {"input": proposal.input_schema}, {"input": "caller_input"}, {}, {}
    for key in order:
        node = nodes[key]
        available = {name: schema for name, schema in schemas.items() if name == "input" or name in ancestors[key]}
        sources = _sources(node.inputs)
        if sources - available.keys():
            raise ValueError("input data requires an explicit completed dependency")
        if isinstance(node, StrictRegion):
            if node.flow.source_digest != proposal.source_digest:
                raise ValueError("strict region must bind the original source bundle")
            if any(roles[s] == "model_candidate" for s in sources):
                raise ValueError("model candidate cannot directly become strict-region arguments")
            packet = qualify_flow(node.flow, reads, {})
            if any(n.kind == "effect_candidate" for n in node.flow.nodes):
                raise ValueError("mixed read prototype cannot dispatch or silently consume an Effect candidate")
            regions[key] = packet
            observed = {n.id: read_schema(reads[n.contract_hash], "output")
                        for n in node.flow.nodes if isinstance(n, StructuredReadNode)}
            schemas[key] = {"type": "object", "properties": {
                "observations": {"type": "object", "properties": observed, "required": [], "additionalProperties": False},
                "outcome": {"type": "string", "enum": ["read_path_completed", "needs_l1"]}},
                "required": ["observations", "outcome"], "additionalProperties": False}
            roles[key], target = "observed_region", node.flow.input_schema
        elif isinstance(node, ReasoningTask):
            schemas[key] = checked_schema(node.output_schema)
            roles[key], target = "model_candidate", node.input_schema
        elif isinstance(node, CandidateAdmission):
            if node.candidate not in ancestors[key] or roles.get(node.candidate) != "model_candidate":
                raise ValueError("admission needs a declared model-candidate dependency")
            if any(roles[s] not in {"caller_input", "observed_region", "observed_join"} for s in sources):
                raise ValueError("candidate cannot serve as its own independent admission evidence")
            schemas[key] = schemas[node.candidate]
            roles[key], target = "admitted_candidate", node.input_schema
        else:
            schemas[key] = checked_schema(node.input_schema)
            # A join never upgrades candidate data to observed evidence or
            # consolidates several admission grants into a broader grant.
            roles[key] = "model_candidate" if any(roles[s] in {"model_candidate", "admitted_candidate"} for s in sources) else "observed_join"
            target = node.input_schema
        bindings[key] = compile_binding({s: available[s] for s in sorted(sources)}, target, node.inputs,
                                        source_bundle_digest=proposal.source_digest)
        checked_schema(schemas[key])
    body = {"proposal": proposal.model_dump(mode="json"), "order": order,
        "ancestors": {k: sorted(v) for k, v in ancestors.items()}, "roles": roles,
        "outputSchemas": schemas, "inputBindings": bindings, "regions": regions,
        "status": "structurally_qualified_not_semantically_proven", "runtimeAuthorityGranted": False,
        "wholeGraphDeterministic": not any(isinstance(n, ReasoningTask) for n in nodes.values())}
    return {**body, "graphDigest": sha256_json(body)}
