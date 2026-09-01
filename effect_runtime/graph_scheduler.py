"""Deterministic scheduler for a reviewed :class:`TypedExecutionGraph`.

The scheduler owns node eligibility and branch transitions. Infrastructure
code may supply a node implementation, but it cannot start a node out of order
or turn a failed/indeterminate predecessor into a success path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from .reliability import (
    ExecutionPhase,
    OperationNode,
    TERMINAL_PHASES,
    TransactionStateMachine,
    TypedExecutionGraph,
)


class GraphScheduleError(ValueError):
    """The requested graph node would violate the reviewed schedule."""


class NodeOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class NodeResult:
    node_id: str
    phase: ExecutionPhase
    outcome: NodeOutcome
    evidence_ids: tuple[str, ...] = ()


def graph_from_step_contract(
    step_contract: Iterable[Mapping[str, Any]],
) -> TypedExecutionGraph:
    """Rebuild and validate the exact graph bound into a PreparedPlan."""
    nodes: list[OperationNode] = []
    for raw in step_contract:
        nodes.append(OperationNode(
            id=str(raw.get("step_id") or "").strip(),
            phase=ExecutionPhase(str(raw.get("phase") or "")),
            depends_on=tuple(str(item) for item in raw.get("depends_on") or ()),
            side_effect=bool(raw.get("side_effect", False)),
        ))
    return TypedExecutionGraph.create(nodes)


class TypedGraphScheduler:
    """Stateful, fail-closed execution cursor over one immutable graph."""

    def __init__(self, graph: TypedExecutionGraph) -> None:
        graph.validate()
        self.graph = graph
        self.machine = TransactionStateMachine()
        self._nodes = {node.id: node for node in graph.nodes}
        self._results: dict[str, NodeResult] = {}
        self._result_order: list[str] = []
        self._active: str | None = None
        self._side_effect_started: set[str] = set()

    @property
    def active_node_id(self) -> str | None:
        return self._active

    @property
    def results(self) -> tuple[NodeResult, ...]:
        return tuple(self._results[node_id] for node_id in self._result_order)

    def _validate_branch(self, node: OperationNode, previous: NodeResult | None) -> None:
        if previous is None:
            if node.phase != ExecutionPhase.SNAPSHOT:
                raise GraphScheduleError("execution graph must start at snapshot")
            return
        outcome = previous.outcome
        phase = node.phase
        if phase == ExecutionPhase.RECONCILE:
            valid = (
                previous.phase == ExecutionPhase.EXECUTE
                and outcome in {NodeOutcome.SUCCEEDED, NodeOutcome.INDETERMINATE}
            ) or (
                previous.phase == ExecutionPhase.VERIFY
                and outcome in {NodeOutcome.FAILED, NodeOutcome.INDETERMINATE}
            )
            if not valid:
                raise GraphScheduleError(
                    "reconcile requires an unresolved Effect or verification boundary"
                )
            return
        if phase == ExecutionPhase.COMPENSATE:
            if outcome not in {NodeOutcome.FAILED, NodeOutcome.INDETERMINATE}:
                raise GraphScheduleError(
                    "compensation requires a failed or indeterminate predecessor"
                )
            return
        if phase == ExecutionPhase.VERIFY_RECOVERY:
            if previous.phase != ExecutionPhase.COMPENSATE or outcome != NodeOutcome.SUCCEEDED:
                raise GraphScheduleError(
                    "recovery verification requires completed compensation"
                )
            return
        if phase == ExecutionPhase.COMMIT:
            if previous.phase != ExecutionPhase.VERIFY or outcome != NodeOutcome.SUCCEEDED:
                raise GraphScheduleError(
                    "commit requires successful independent verification"
                )
            return
        if phase == ExecutionPhase.ABORT:
            valid = (
                not self._side_effect_started
                and previous.phase in {
                    ExecutionPhase.SNAPSHOT,
                    ExecutionPhase.PRECHECK,
                    ExecutionPhase.AWAITING_APPROVAL,
                    ExecutionPhase.REVALIDATE,
                }
            ) or (
                previous.phase == ExecutionPhase.VERIFY_RECOVERY
                and outcome == NodeOutcome.SUCCEEDED
            )
            if not valid:
                raise GraphScheduleError(
                    "abort requires a pre-Effect branch or verified recovery"
                )
            return
        if phase == ExecutionPhase.ESCALATE:
            if outcome not in {NodeOutcome.FAILED, NodeOutcome.INDETERMINATE}:
                raise GraphScheduleError(
                    "escalation requires unresolved failure evidence"
                )
            return
        if outcome != NodeOutcome.SUCCEEDED:
            raise GraphScheduleError(
                f"{phase.value} requires a successful predecessor, got {outcome.value}"
            )

    def start(self, node_id: str, *, evidence: str = "") -> OperationNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise GraphScheduleError(f"node {node_id!r} is not in the approved graph")
        if node_id in self._results or self._active == node_id:
            raise GraphScheduleError(f"graph node {node_id!r} cannot run more than once")
        if self._active is not None:
            raise GraphScheduleError(f"graph node {self._active!r} is still active")
        previous = self.results[-1] if self.results else None
        self._validate_branch(node, previous)
        if node.phase not in {ExecutionPhase.ABORT, ExecutionPhase.ESCALATE}:
            unsettled = [
                dependency for dependency in node.depends_on
                if dependency not in self._results
            ]
            if unsettled:
                raise GraphScheduleError(
                    f"graph node {node_id!r} has unsettled dependencies: {unsettled}"
                )
        if node.side_effect:
            if node_id in self._side_effect_started:
                raise GraphScheduleError(f"side-effect node {node_id!r} is one-shot")
            self._side_effect_started.add(node_id)
        try:
            self.machine.transition(node.phase, evidence=evidence)
        except ValueError as error:
            raise GraphScheduleError(str(error)) from error
        self._active = node_id
        return node

    def recover_start_indeterminate_effect(self, *, evidence: str) -> OperationNode:
        """Restore a possible Effect boundary after a durable execution claim.

        This recovery-only transition never invokes Effect.  It is permitted
        only when revalidation was explicitly recorded as skipped/unknown due
        to a crash, and its Effect result must subsequently be indeterminate.
        """
        if not evidence.strip():
            raise GraphScheduleError("recovery requires durable execution-claim evidence")
        node = self._nodes.get("execute")
        if node is None or self._active is not None or "execute" in self._results:
            raise GraphScheduleError("Effect boundary cannot be recovered from this graph state")
        previous = self.results[-1] if self.results else None
        if (
            previous is None
            or previous.phase != ExecutionPhase.REVALIDATE
            or previous.outcome not in {NodeOutcome.SKIPPED, NodeOutcome.INDETERMINATE}
        ):
            raise GraphScheduleError(
                "recovered Effect requires an explicitly unknown revalidation result"
            )
        unsettled = [
            dependency for dependency in node.depends_on
            if dependency not in self._results
        ]
        if unsettled:
            raise GraphScheduleError(
                f"recovered Effect has unsettled dependencies: {unsettled}"
            )
        try:
            self.machine.transition(node.phase, evidence=evidence)
        except ValueError as error:
            raise GraphScheduleError(str(error)) from error
        self._side_effect_started.add(node.id)
        self._active = node.id
        return node

    def finish(
        self,
        node_id: str,
        outcome: NodeOutcome | str,
        *,
        evidence_ids: Iterable[str] = (),
    ) -> NodeResult:
        if self._active != node_id:
            raise GraphScheduleError(
                f"graph node {node_id!r} is not the active node {self._active!r}"
            )
        node = self._nodes[node_id]
        outcome = NodeOutcome(outcome)
        if node.phase in TERMINAL_PHASES and outcome != NodeOutcome.SUCCEEDED:
            raise GraphScheduleError("terminal graph nodes must complete successfully")
        result = NodeResult(
            node_id=node.id,
            phase=node.phase,
            outcome=outcome,
            evidence_ids=tuple(sorted(set(str(item) for item in evidence_ids if item))),
        )
        self._results[node_id] = result
        self._result_order.append(node_id)
        self._active = None
        return result


__all__ = [
    "GraphScheduleError",
    "NodeOutcome",
    "NodeResult",
    "TypedGraphScheduler",
    "graph_from_step_contract",
]
