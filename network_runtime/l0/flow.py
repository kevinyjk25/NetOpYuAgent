"""Bounded business-flow authoring and host-authorized read evaluation.

This is a control layer over existing read contracts, not a replacement Effect
executor. Effect and reasoning leaves stop with a candidate; neither is success
nor permission to invoke a provider. No expression evaluation or script loading.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Mapping

from pydantic import Field

from network_runtime.access import ObservationAccessContext
from network_runtime.contracts import sha256_json

from .models import CompiledAtomicRead, ReadObjectSchema, StrictModel
from .read_contracts import _validate_values, _verified_contract
from .read_execution import HostReadBinding, execute_host_read


class Reference(StrictModel):
    kind: Literal["reference"]
    source: str  # "input" or an earlier read node id
    field: str


class Constant(StrictModel):
    kind: Literal["constant"]
    value: Any


Value = Annotated[Reference | Constant, Field(discriminator="kind")]


class ReadNode(StrictModel):
    kind: Literal["read"]
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    contract_hash: str
    arguments: dict[str, Value]
    next: str


class BranchNode(StrictModel):
    kind: Literal["branch"]
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    left: Reference
    equals: Constant
    on_true: str
    on_false: str


class EndNode(StrictModel):
    kind: Literal["end"]
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    outcome: Literal["read_path_completed", "needs_l1", "unsupported"]
    explanation: str = Field(min_length=1)


class EffectCandidateNode(StrictModel):
    kind: Literal["effect_candidate"]
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    binding_id: str
    arguments: dict[str, Value]


Node = Annotated[ReadNode | BranchNode | EndNode | EffectCandidateNode, Field(discriminator="kind")]


class FlowProposal(StrictModel):
    api_version: Literal["netopyu.io/l0-flow-proposal/v1"]
    source_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    purpose: str = Field(min_length=1)
    input_schema: ReadObjectSchema
    entry: str
    nodes: tuple[Node, ...] = Field(min_length=1, max_length=64)
    max_read_age_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)


class EffectTarget(StrictModel):
    """Host-supplied candidate contract, not an executable binding."""

    profile: str
    tool: str
    skill_id: str
    contract_hash: str
    input_schema: ReadObjectSchema


def _scalar_type(value: Any) -> str:
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float and math.isfinite(value):
        return "number"
    if type(value) is str:
        return "string"
    raise ValueError("flow constants require finite JSON scalars")


def _compatible(actual: str, expected: str) -> bool:
    return actual == expected or (actual == "integer" and expected == "number")


def qualify_flow(
    proposal: FlowProposal, reads: Mapping[str, CompiledAtomicRead],
    effects: Mapping[str, EffectTarget],
) -> dict[str, Any]:
    """Type/graph qualification only, never semantic review or activation."""
    proposal = FlowProposal.model_validate(proposal.model_dump())
    nodes = {node.id: node for node in proposal.nodes}
    if len(nodes) != len(proposal.nodes) or "input" in nodes or proposal.entry not in nodes:
        raise ValueError("flow ids must be unique, nonreserved and include entry")
    successors: dict[str, set[str]] = {}
    predecessors: dict[str, set[str]] = {key: set() for key in nodes}
    for node in nodes.values():
        targets = {node.next} if isinstance(node, ReadNode) else (
            {node.on_true, node.on_false} if isinstance(node, BranchNode) else set()
        )
        if targets - nodes.keys():
            raise ValueError("flow has an unknown successor")
        successors[node.id] = targets
        for target in targets:
            predecessors[target].add(node.id)
    seen: set[str] = set()
    visiting: set[str] = set()
    order: list[str] = []

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("flow cycles are unsupported")
        if key in seen:
            return
        visiting.add(key)
        for target in sorted(successors[key]):
            visit(target)
        visiting.remove(key)
        seen.add(key)
        order.append(key)

    visit(proposal.entry)
    if seen != nodes.keys():
        raise ValueError("unreachable nodes must not disappear from Skill coverage")
    order.reverse()
    schemas = {"input": proposal.input_schema}
    used_reads = {}
    used_effects = {}
    for node in nodes.values():
        if isinstance(node, ReadNode):
            if node.contract_hash not in reads:
                raise ValueError("host read contract is missing")
            contract = _verified_contract(reads[node.contract_hash])
            if contract.contract_hash != node.contract_hash:
                raise ValueError("read key differs from contract hash")
            schemas[node.id] = contract.spec.output_schema
            used_reads[node.contract_hash] = contract
        elif isinstance(node, EffectCandidateNode):
            if node.binding_id not in effects:
                raise ValueError("host effect target is missing")
            used_effects[node.binding_id] = EffectTarget.model_validate(effects[node.binding_id].model_dump())
    dominators: dict[str, set[str]] = {}
    for key in order:
        parents = predecessors[key]
        before = set.intersection(*(dominators[parent] | {parent} for parent in parents)) if parents else set()
        dominators[key] = before

        def source_type(value: Reference | Constant) -> str:
            if isinstance(value, Constant):
                return _scalar_type(value.value)
            if value.source != "input" and value.source not in before:
                raise ValueError("step output must be available on every incoming path")
            if value.source not in schemas or value.field not in schemas[value.source].properties:
                raise ValueError("unknown typed source field")
            return schemas[value.source].properties[value.field].type

        node = nodes[key]
        if isinstance(node, BranchNode):
            left, right = source_type(node.left), source_type(node.equals)
            if not (_compatible(left, right) or _compatible(right, left)):
                raise ValueError("branch equality requires compatible scalar types")
        elif isinstance(node, (ReadNode, EffectCandidateNode)):
            schema = used_reads[node.contract_hash].spec.input_schema if isinstance(node, ReadNode) else used_effects[node.binding_id].input_schema
            if set(node.arguments) - schema.properties.keys() or set(schema.required) - node.arguments.keys():
                raise ValueError("flow arguments differ from required/declared tool fields")
            for name, value in node.arguments.items():
                if not _compatible(source_type(value), schema.properties[name].type):
                    raise ValueError("flow reference type cannot satisfy tool field")
    body = {
        "proposal": proposal.model_dump(mode="json"),
        "readContracts": {key: value.model_dump(by_alias=True, mode="json") for key, value in sorted(used_reads.items())},
        "effectTargets": {key: value.model_dump(mode="json") for key, value in sorted(used_effects.items())},
        "status": "structurally_qualified_not_semantically_proven",
        "runtimeAuthorityGranted": False,
    }
    return {**body, "flowDigest": sha256_json(body)}


@dataclass(frozen=True)
class HostFlowConsent:
    """Trusted in-process host choice, never accepted from a model/request body."""

    flow_digest: str
    arguments_digest: str


def run_read_flow(
    proposal: FlowProposal, arguments: dict[str, Any], *,
    reads: Mapping[str, CompiledAtomicRead], effects: Mapping[str, EffectTarget],
    bindings: Mapping[str, HostReadBinding], context: ObservationAccessContext,
    consent: HostFlowConsent,
) -> dict[str, Any]:
    packet = qualify_flow(proposal, reads, effects)
    # Snapshot nested mutable proposal/contract objects before provider calls.
    proposal = FlowProposal.model_validate(packet["proposal"])
    reads = {key: CompiledAtomicRead.model_validate(value) for key, value in packet["readContracts"].items()}
    effects = {key: EffectTarget.model_validate(value) for key, value in packet["effectTargets"].items()}
    arguments = _validate_values(arguments, proposal.input_schema, inputs=True)
    if consent != HostFlowConsent(packet["flowDigest"], sha256_json(arguments)):
        raise PermissionError("host consent must bind exact flow and request")
    bindings = dict(bindings)
    if any(key not in bindings or bindings[key].approved_contract_hash != key for key in reads):
        raise PermissionError("all declared read paths require exact host bindings")
    values = {"input": arguments}
    completed_at: dict[str, float] = {}
    trace: list[dict[str, Any]] = []
    nodes = {node.id: node for node in proposal.nodes}
    key = proposal.entry

    def resolve(value: Reference | Constant) -> Any:
        if isinstance(value, Constant):
            return value.value
        if value.source != "input":
            age = time.monotonic() - completed_at[value.source]
            if not 0 <= age <= proposal.max_read_age_seconds:
                raise ValueError("flow read evidence expired or clock regressed")
        if value.field not in values[value.source]:
            raise ValueError("referenced optional field is absent; condition is unknown")
        return values[value.source][value.field]

    def finish(status: str, **extra: Any) -> dict[str, Any]:
        body = {"status": status, "flowDigest": packet["flowDigest"],
                "argumentsDigest": sha256_json(arguments), "trace": trace,
                "effectExecuted": False, "wholeSkillCorrectnessProven": False, **extra}
        return {**body, "reportDigest": sha256_json(body)}

    while True:
        node = nodes[key]
        try:
            if isinstance(node, ReadNode):
                args = {name: resolve(value) for name, value in node.arguments.items()}
                receipt = execute_host_read(reads[node.contract_hash], args, context, bindings[node.contract_hash])
                # Receipt and cached payload must not alias provider-owned mutable data.
                receipt = json.loads(json.dumps(receipt, allow_nan=False))
                values[key] = receipt["payload"]
                completed_at[key] = time.monotonic()
                trace.append({"node": key, "kind": "read", "receipt": receipt})
                key = node.next
            elif isinstance(node, BranchNode):
                left, right = resolve(node.left), resolve(node.equals)
                matched = left == right
                trace.append({"node": key, "kind": "branch", "matched": matched,
                              "source": node.left.model_dump(), "selected": node.on_true if matched else node.on_false})
                key = node.on_true if matched else node.on_false
            elif isinstance(node, EffectCandidateNode):
                target = effects[node.binding_id]
                args = _validate_values({name: resolve(value) for name, value in node.arguments.items()}, target.input_schema, inputs=True)
                trace.append({"node": key, "kind": "effect_candidate"})
                return finish("awaiting_effect_admission", candidate={"target": target.model_dump(mode="json"), "arguments": args},
                              boundary="No write authorization. Branch evidence must be rebound/revalidated by Effect admission.")
            else:
                trace.append({"node": key, "kind": "end"})
                return finish(node.outcome, explanation=node.explanation)
        except Exception as error:
            # Do not expose provider exception text (it can contain credentials).
            trace.append({"node": key, "kind": node.kind, "errorType": type(error).__name__})
            return finish("blocked", blockedAt=key, reasonType=type(error).__name__)
