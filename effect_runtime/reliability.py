"""EnsuredSkill reliability primitives.

This module is the domain-neutral kernel described by the EnsuredSkill design.
It deliberately contains no Harness, MCP, network-device, identity-provider, or
deployment logic.  A reasoning plane may propose an operation; only these
contracts and the Runtime decide whether it is eligible to execute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class Reversibility(StrEnum):
    STRONG = "strongly_reversible"
    CONDITIONAL = "conditionally_reversible"
    IRREVERSIBLE = "irreversible"


class ExecutionPhase(StrEnum):
    BEGIN = "begin"
    SNAPSHOT = "snapshot"
    PRECHECK = "precheck"
    AWAITING_APPROVAL = "awaiting_approval"
    REVALIDATE = "revalidate"
    EXECUTE = "execute"
    VERIFY = "verify"
    COMMIT = "commit"
    RECONCILE = "reconcile"
    COMPENSATE = "compensate"
    VERIFY_RECOVERY = "verify_recovery"
    ABORT = "abort"
    ESCALATE = "escalate"


TERMINAL_PHASES = {
    ExecutionPhase.COMMIT,
    ExecutionPhase.ABORT,
    ExecutionPhase.ESCALATE,
}


_TRANSITIONS: dict[ExecutionPhase, frozenset[ExecutionPhase]] = {
    ExecutionPhase.BEGIN: frozenset({ExecutionPhase.SNAPSHOT}),
    ExecutionPhase.SNAPSHOT: frozenset({ExecutionPhase.PRECHECK, ExecutionPhase.ABORT}),
    ExecutionPhase.PRECHECK: frozenset({
        ExecutionPhase.AWAITING_APPROVAL,
        ExecutionPhase.REVALIDATE,
        ExecutionPhase.ABORT,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.AWAITING_APPROVAL: frozenset({
        ExecutionPhase.REVALIDATE,
        ExecutionPhase.ABORT,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.REVALIDATE: frozenset({ExecutionPhase.EXECUTE, ExecutionPhase.ABORT}),
    ExecutionPhase.EXECUTE: frozenset({
        ExecutionPhase.VERIFY,
        ExecutionPhase.RECONCILE,
        ExecutionPhase.COMPENSATE,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.VERIFY: frozenset({
        ExecutionPhase.COMMIT,
        ExecutionPhase.COMPENSATE,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.RECONCILE: frozenset({
        ExecutionPhase.VERIFY,
        ExecutionPhase.COMPENSATE,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.COMPENSATE: frozenset({
        ExecutionPhase.VERIFY_RECOVERY,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.VERIFY_RECOVERY: frozenset({
        ExecutionPhase.ABORT,
        ExecutionPhase.ESCALATE,
    }),
    ExecutionPhase.COMMIT: frozenset(),
    ExecutionPhase.ABORT: frozenset(),
    ExecutionPhase.ESCALATE: frozenset(),
}


@dataclass
class TransactionStateMachine:
    """Small auditable state machine; infrastructure calls live outside it."""

    phase: ExecutionPhase = ExecutionPhase.BEGIN
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, target: ExecutionPhase | str, *, evidence: str = "") -> None:
        target = ExecutionPhase(target)
        if target not in _TRANSITIONS[self.phase]:
            raise ValueError(f"illegal EnsuredSkill transition {self.phase.value} -> {target.value}")
        self.history.append({
            "from": self.phase.value,
            "to": target.value,
            "evidence": evidence,
        })
        self.phase = target


@dataclass(frozen=True)
class EvidenceRequirement:
    id: str
    semantic_type: str
    source_capability: str
    phase: ExecutionPhase
    max_age_seconds: int
    scope: tuple[str, ...] = ()
    associated_action: str = ""
    required_predicates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.semantic_type or not self.source_capability:
            raise ValueError("evidence requirement identity, type and source are required")
        if self.max_age_seconds <= 0:
            raise ValueError("evidence max_age_seconds must be positive")


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    semantic_type: str
    source_capability: str
    collector_identity: str
    collected_at: str
    scope: tuple[str, ...]
    associated_action: str
    payload: Mapping[str, Any]
    payload_digest: str
    valid: bool = True
    parent_evidence_ids: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        id: str,
        semantic_type: str,
        source_capability: str,
        collector_identity: str,
        collected_at: str,
        scope: Sequence[str],
        associated_action: str,
        payload: Mapping[str, Any],
        valid: bool = True,
        parent_evidence_ids: Sequence[str] = (),
    ) -> "EvidenceRecord":
        return cls(
            id=id,
            semantic_type=semantic_type,
            source_capability=source_capability,
            collector_identity=collector_identity,
            collected_at=collected_at,
            scope=tuple(scope),
            associated_action=associated_action,
            payload=dict(payload),
            payload_digest=_digest(payload),
            valid=valid,
            parent_evidence_ids=tuple(parent_evidence_ids),
        )

    def verify_integrity(self) -> bool:
        return self.payload_digest == _digest(self.payload)


@dataclass(frozen=True)
class Guard:
    id: str
    field: str
    operator: str
    expected: Any = None
    evidence_requirement_id: str | None = None


@dataclass(frozen=True)
class ResourceSet:
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap = set(self.reads) & set(self.writes)
        if len(self.writes) != len(set(self.writes)):
            raise ValueError("write resources must be unique")
        if overlap:
            # A resource may be observed before mutation, but the contract must
            # name the snapshot read separately so conflict analysis is exact.
            raise ValueError("read/write resources must use distinct phase-qualified identities")


@dataclass(frozen=True)
class ReliabilityContract:
    operation: str
    version: str
    inputs_schema_digest: str
    preconditions: tuple[Guard, ...]
    evidence: tuple[EvidenceRequirement, ...]
    guards: tuple[Guard, ...]
    postconditions: tuple[Guard, ...]
    resources: ResourceSet
    reversibility: Reversibility
    idempotent: bool
    compensation_operation: str | None
    service_criticality: int = 1
    contract_digest: str = ""

    @classmethod
    def create(cls, **values: Any) -> "ReliabilityContract":
        candidate = cls(**values)
        candidate.validate()
        stable = asdict(candidate)
        stable.pop("contract_digest")
        return cls(**{**values, "contract_digest": _digest(stable)})

    def validate(self) -> None:
        if not self.operation or not self.version:
            raise ValueError("contract operation and version are required")
        evidence_ids = [item.id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence requirement ids must be unique")
        known = set(evidence_ids)
        for guard in (*self.preconditions, *self.guards, *self.postconditions):
            if guard.evidence_requirement_id and guard.evidence_requirement_id not in known:
                raise ValueError(f"guard {guard.id!r} references unknown evidence")
        if self.reversibility != Reversibility.IRREVERSIBLE and not self.compensation_operation:
            raise ValueError("reversible operations require a compensation operation")
        if self.reversibility == Reversibility.IRREVERSIBLE and self.compensation_operation:
            raise ValueError("irreversible operations cannot claim compensation")
        if not 0 <= self.service_criticality <= 4:
            raise ValueError("service_criticality must be between 0 and 4")


@dataclass(frozen=True)
class OperationNode:
    id: str
    phase: ExecutionPhase
    depends_on: tuple[str, ...] = ()
    side_effect: bool = False


@dataclass(frozen=True)
class TypedExecutionGraph:
    nodes: tuple[OperationNode, ...]
    graph_digest: str = ""

    @classmethod
    def create(cls, nodes: Iterable[OperationNode]) -> "TypedExecutionGraph":
        graph = cls(tuple(nodes))
        graph.validate()
        stable = [asdict(item) for item in graph.nodes]
        return cls(graph.nodes, _digest(stable))

    def validate(self) -> None:
        ids = [item.id for item in self.nodes]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("execution graph node ids must be non-empty and unique")
        known = set(ids)
        for node in self.nodes:
            if set(node.depends_on) - known:
                raise ValueError(f"node {node.id!r} has an unknown dependency")
            if node.id in node.depends_on:
                raise ValueError(f"node {node.id!r} depends on itself")
            if node.side_effect and node.phase not in {
                ExecutionPhase.EXECUTE, ExecutionPhase.COMPENSATE,
            }:
                raise ValueError(
                    f"side-effect node {node.id!r} has an invalid phase"
                )
        execute_effects = [
            node for node in self.nodes
            if node.side_effect and node.phase == ExecutionPhase.EXECUTE
        ]
        compensation_effects = [
            node for node in self.nodes
            if node.side_effect and node.phase == ExecutionPhase.COMPENSATE
        ]
        if len(execute_effects) != 1:
            raise ValueError("execution graph requires exactly one primary effect node")
        if len(compensation_effects) > 1:
            raise ValueError("execution graph permits at most one compensation effect node")
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        remaining = {item.id: set(item.depends_on) for item in self.nodes}
        order: list[str] = []
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError("execution graph contains a dependency cycle")
            order.extend(ready)
            for key in ready:
                remaining.pop(key)
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(order)


def build_transaction_graph(*, compensatable: bool) -> TypedExecutionGraph:
    nodes = [
        OperationNode("snapshot", ExecutionPhase.SNAPSHOT),
        OperationNode("precheck", ExecutionPhase.PRECHECK, ("snapshot",)),
        OperationNode("approval", ExecutionPhase.AWAITING_APPROVAL, ("precheck",)),
        OperationNode("revalidate", ExecutionPhase.REVALIDATE, ("approval",)),
        OperationNode("execute", ExecutionPhase.EXECUTE, ("revalidate",), side_effect=True),
        OperationNode("verify", ExecutionPhase.VERIFY, ("execute",)),
        OperationNode("commit", ExecutionPhase.COMMIT, ("verify",)),
        OperationNode("reconcile", ExecutionPhase.RECONCILE, ("execute",)),
    ]
    if compensatable:
        nodes.extend([
            OperationNode("compensate", ExecutionPhase.COMPENSATE, ("execute",), side_effect=True),
            OperationNode("verify_recovery", ExecutionPhase.VERIFY_RECOVERY, ("compensate",)),
            OperationNode("abort", ExecutionPhase.ABORT, ("verify_recovery",)),
        ])
    nodes.append(OperationNode("escalate", ExecutionPhase.ESCALATE, ("reconcile",)))
    return TypedExecutionGraph.create(nodes)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    missing: tuple[str, ...]
    invalid: tuple[str, ...]
    stale: tuple[str, ...]
    mismatched: tuple[str, ...]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    unresolved: tuple[str, ...]


def evaluate_evidence(
    requirements: Sequence[EvidenceRequirement],
    records: Sequence[EvidenceRecord],
    *,
    now: datetime | None = None,
) -> GateResult:
    """Evaluate provenance and freshness before any semantic guard executes."""
    now = now or datetime.now(timezone.utc)
    by_type: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        by_type.setdefault(record.semantic_type, []).append(record)
    missing: list[str] = []
    invalid: list[str] = []
    stale: list[str] = []
    mismatched: list[str] = []
    for requirement in requirements:
        candidates = by_type.get(requirement.semantic_type, [])
        if not candidates:
            missing.append(requirement.id)
            continue
        accepted = False
        for record in candidates:
            if not record.valid or not record.verify_integrity():
                invalid.append(requirement.id)
                continue
            observed = datetime.fromisoformat(record.collected_at)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            age = (now - observed).total_seconds()
            if age < 0 or age > requirement.max_age_seconds:
                stale.append(requirement.id)
                continue
            if (
                record.source_capability != requirement.source_capability
                or (requirement.associated_action and record.associated_action != requirement.associated_action)
                or (requirement.scope and not set(requirement.scope).issubset(record.scope))
            ):
                mismatched.append(requirement.id)
                continue
            accepted = True
            break
        if not accepted and requirement.id not in (*invalid, *stale, *mismatched):
            missing.append(requirement.id)
    return GateResult(
        allowed=not (missing or invalid or stale or mismatched),
        missing=tuple(sorted(set(missing))),
        invalid=tuple(sorted(set(invalid))),
        stale=tuple(sorted(set(stale))),
        mismatched=tuple(sorted(set(mismatched))),
    )


def _field(value: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    def resolve(root: Any) -> tuple[bool, Any]:
        current = root
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return False, None
            current = current[part]
        return True, current

    found, current = resolve(value)
    if found:
        return True, current
    # Runtime Evidence envelopes keep provider state under ``facts`` while L0
    # predicates are authored against the provider's logical result fields.
    # Resolve that typed wrapper deterministically; do not inspect prose.
    facts = value.get("facts")
    return resolve(facts) if isinstance(facts, Mapping) else (False, None)


def _predicate(operator: str, exists: bool, actual: Any, expected: Any) -> bool:
    if operator == "exists":
        return exists
    if not exists:
        return False
    if operator in {"equals", "exact_snapshot"}:
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "in":
        return actual in expected if isinstance(expected, (list, tuple, set)) else False
    if operator == "contains":
        return expected in actual if isinstance(actual, (str, list, tuple, set, dict)) else False
    try:
        if operator == "greater_than":
            return actual > expected
        if operator == "greater_than_or_equal":
            return actual >= expected
        if operator == "less_than":
            return actual < expected
        if operator == "less_than_or_equal":
            return actual <= expected
    except TypeError:
        return False
    raise ValueError(f"unsupported EnsuredSkill guard operator {operator!r}")


def evaluate_guards(
    guards: Sequence[Guard],
    requirements: Sequence[EvidenceRequirement],
    records: Sequence[EvidenceRecord],
) -> GuardResult:
    """Evaluate deterministic predicates only over already typed Evidence.

    Evidence provenance/freshness must be gated first. Unknown references,
    missing records, fields, or operators fail closed rather than being left to
    model interpretation.
    """
    requirement_by_id = {item.id: item for item in requirements}
    passed: list[str] = []
    failed: list[str] = []
    unresolved: list[str] = []
    for guard in guards:
        requirement = requirement_by_id.get(str(guard.evidence_requirement_id or ""))
        if requirement is None:
            unresolved.append(guard.id)
            continue
        record = next((
            item for item in records
            if item.semantic_type == requirement.semantic_type
            and item.source_capability == requirement.source_capability
            and item.valid
            and item.verify_integrity()
        ), None)
        if record is None:
            unresolved.append(guard.id)
            continue
        exists, actual = _field(record.payload, guard.field)
        try:
            accepted = _predicate(guard.operator, exists, actual, guard.expected)
        except ValueError:
            unresolved.append(guard.id)
            continue
        (passed if accepted else failed).append(guard.id)
    return GuardResult(
        allowed=not failed and not unresolved,
        passed=tuple(sorted(set(passed))),
        failed=tuple(sorted(set(failed))),
        unresolved=tuple(sorted(set(unresolved))),
    )


class AutonomyDecision(StrEnum):
    EXECUTE = "execute"
    ASK_HUMAN = "ask_human"
    REJECT = "reject"


@dataclass(frozen=True)
class RiskFactors:
    change_scope: int
    blast_radius: int
    evidence_confidence: float
    reversibility: Reversibility
    historical_success: float
    service_criticality: int


@dataclass(frozen=True)
class RiskAssessment:
    score: int
    decision: AutonomyDecision
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RiskPolicy:
    auto_threshold: int = 4
    reject_threshold: int = 13

    def evaluate(self, factors: RiskFactors) -> RiskAssessment:
        evidence = min(1.0, max(0.0, factors.evidence_confidence))
        history = min(1.0, max(0.0, factors.historical_success))
        reversibility_cost = {
            Reversibility.STRONG: 0,
            Reversibility.CONDITIONAL: 2,
            Reversibility.IRREVERSIBLE: 5,
        }[factors.reversibility]
        contributions = {
            "change_scope": min(3, max(0, factors.change_scope - 1)),
            "blast_radius": min(4, max(0, factors.blast_radius - 1)),
            "evidence_uncertainty": round((1.0 - evidence) * 4),
            "reversibility": reversibility_cost,
            "historical_failure": round((1.0 - history) * 2),
            "service_criticality": min(4, max(0, factors.service_criticality)),
        }
        score = sum(contributions.values())
        if evidence < 0.5:
            decision = AutonomyDecision.REJECT
        elif score <= self.auto_threshold:
            decision = AutonomyDecision.EXECUTE
        elif score < self.reject_threshold:
            decision = AutonomyDecision.ASK_HUMAN
        else:
            decision = AutonomyDecision.REJECT
        reasons = tuple(f"{key}={value}" for key, value in contributions.items())
        return RiskAssessment(score=score, decision=decision, reasons=reasons)


def _runtime_value(value: Any, arguments: Mapping[str, Any]) -> Any:
    if (
        isinstance(value, str)
        and value.startswith("${arguments.")
        and value.endswith("}")
    ):
        name = value[len("${arguments."):-1]
        if not arguments:
            return value
        if name not in arguments:
            raise ValueError(f"guard references missing runtime argument {name!r}")
        return arguments[name]
    if isinstance(value, list):
        return [_runtime_value(item, arguments) for item in value]
    if isinstance(value, tuple):
        return tuple(_runtime_value(item, arguments) for item in value)
    if isinstance(value, dict):
        return {key: _runtime_value(item, arguments) for key, item in value.items()}
    return value


def contract_from_compiled_l0(
    compiled: Any,
    arguments: Mapping[str, Any] | None = None,
) -> ReliabilityContract:
    """Project the existing reviewed L0 v2 artifact into the new core contract."""
    spec = compiled.spec
    runtime_arguments = arguments or {}
    preflight = tuple(
        EvidenceRequirement(
            id=f"preflight:{item.id}",
            semantic_type=item.capability,
            source_capability=item.capability,
            phase=ExecutionPhase.PRECHECK,
            max_age_seconds=300,
            associated_action=spec.effect.capability,
            required_predicates=tuple(predicate.field for predicate in item.predicates),
        )
        for item in spec.preflight
    )
    verification = EvidenceRequirement(
        id="postcondition",
        semantic_type=spec.verification.capability,
        source_capability=spec.verification.capability,
        phase=ExecutionPhase.VERIFY,
        max_age_seconds=300,
        associated_action=spec.effect.capability,
        required_predicates=tuple(item.field for item in spec.verification.predicates),
    )
    compensation = spec.compensation
    reversibility = (
        Reversibility.CONDITIONAL if compensation is not None else Reversibility.IRREVERSIBLE
    )
    risk = {"low": 0, "medium": 1, "high": 2, "critical": 4}[spec.approval.risk]
    return ReliabilityContract.create(
        operation=spec.effect.capability,
        version=compiled.metadata.version,
        inputs_schema_digest=_digest(
            {key: value.model_dump(by_alias=True, mode="json") for key, value in spec.parameters.items()}
        ),
        preconditions=(),
        evidence=(*preflight, verification),
        guards=tuple(
            Guard(
                id=f"preflight:{observation.id}:{index}",
                field=predicate.field,
                operator=predicate.operator,
                expected=_runtime_value(predicate.expected, runtime_arguments),
                evidence_requirement_id=f"preflight:{observation.id}",
            )
            for observation in spec.preflight
            for index, predicate in enumerate(observation.predicates)
        ),
        postconditions=tuple(
            Guard(
                id=f"postcondition:{index}",
                field=item.field,
                operator=item.operator,
                expected=_runtime_value(item.expected, runtime_arguments),
                evidence_requirement_id="postcondition",
            )
            for index, item in enumerate(spec.verification.predicates)
        ),
        resources=ResourceSet(
            reads=tuple(
                f"precheck:{item.capability}" for item in spec.preflight
            ) + (f"verify:{spec.verification.capability}",),
            writes=(f"effect:{spec.effect.capability}",),
        ),
        reversibility=reversibility,
        idempotent=False,
        compensation_operation=(compensation.capability if compensation else None),
        service_criticality=risk,
    )


__all__ = [
    "AutonomyDecision", "EvidenceRecord", "EvidenceRequirement", "ExecutionPhase",
    "GateResult", "Guard", "GuardResult", "OperationNode", "ReliabilityContract", "ResourceSet",
    "Reversibility", "RiskAssessment", "RiskFactors", "RiskPolicy",
    "TERMINAL_PHASES", "TransactionStateMachine", "TypedExecutionGraph",
    "build_transaction_graph", "contract_from_compiled_l0", "evaluate_evidence",
    "evaluate_guards",
]
