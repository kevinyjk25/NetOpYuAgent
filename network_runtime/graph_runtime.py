"""Journal-backed execution of an immutable EnsuredSkill Typed Graph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from effect_runtime.graph_scheduler import (
    GraphScheduleError,
    NodeOutcome,
    TypedGraphScheduler,
    graph_from_step_contract,
)

from .contracts import PreparedPlan, sha256_json, utc_now
from .journal import NetworkJournal


GRAPH_EVENT_SCHEMA = "netopyu.io/typed-graph-node-event/v1"


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


class JournalGraphScheduler:
    """Reconstruct, gate, and hash-chain one plan's graph execution."""

    def __init__(self, plan: PreparedPlan, journal: NetworkJournal) -> None:
        self.plan = plan
        self.journal = journal
        self.scheduler = TypedGraphScheduler(graph_from_step_contract(plan.step_contract))
        self._started_at: dict[str, str] = {}
        self._input_evidence: dict[str, tuple[str, ...]] = {}
        self._replay(journal.events(plan.plan_id))

    @property
    def graph_digest(self) -> str:
        return self.scheduler.graph.graph_digest

    def summary(self) -> dict[str, Any]:
        results = self.scheduler.results
        terminal = self.scheduler.machine.phase.value in {
            "commit", "abort", "escalate",
        }
        return {
            "schema": "netopyu.io/typed-graph-execution/v1",
            "graph_digest": self.graph_digest,
            "active_node_id": self.scheduler.active_node_id,
            "terminal_phase": self.scheduler.machine.phase.value if terminal else None,
            "complete": terminal and self.scheduler.active_node_id is None,
            "results": [
                {
                    "node_id": item.node_id,
                    "phase": item.phase.value,
                    "outcome": item.outcome.value,
                    "evidence_ids": list(item.evidence_ids),
                }
                for item in results
            ],
            "claim_boundary": (
                "Graph conformance proves scheduled control flow, not external-state truth."
            ),
        }

    def _replay(self, events: Iterable[dict[str, Any]]) -> None:
        for event in events:
            payload = event.get("payload") or {}
            event_type = event.get("event_type")
            if event_type not in {"graph_node_started", "graph_node_finished"}:
                continue
            if payload.get("schema") != GRAPH_EVENT_SCHEMA:
                raise GraphScheduleError("graph event schema is missing or unsupported")
            if payload.get("graph_digest") != self.graph_digest:
                raise GraphScheduleError("graph event does not match the approved graph digest")
            node_id = str(payload.get("node_id") or "")
            expected = next(
                (item for item in self.scheduler.graph.nodes if item.id == node_id),
                None,
            )
            if expected is None or payload.get("phase") != expected.phase.value:
                raise GraphScheduleError("graph event node phase is not contract-bound")
            if event_type == "graph_node_started":
                details = payload.get("details") or {}
                if (
                    node_id == "execute"
                    and details.get("source") == "crash_recovery"
                ):
                    self.scheduler.recover_start_indeterminate_effect(
                        evidence=str(details.get("execution_claim_evidence") or "journal_replay"),
                    )
                else:
                    self.scheduler.start(node_id, evidence="journal_replay")
                self._started_at[node_id] = str(
                    payload.get("started_at") or event.get("created_at") or utc_now()
                )
                self._input_evidence[node_id] = tuple(
                    str(item) for item in payload.get("input_evidence_ids") or ()
                )
            else:
                self.scheduler.finish(
                    node_id,
                    str(payload.get("outcome") or ""),
                    evidence_ids=payload.get("output_evidence_ids") or (),
                )

    def start(
        self,
        node_id: str,
        *,
        input_evidence_ids: Iterable[str] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        node = self.scheduler.start(node_id, evidence="journal_bound")
        self._persist_start(
            node,
            input_evidence_ids=input_evidence_ids,
            details=details,
        )

    def _persist_start(
        self,
        node: Any,
        *,
        input_evidence_ids: Iterable[str] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        dependency_evidence = {
            evidence_id
            for result in self.scheduler.results
            if result.node_id in node.depends_on
            for evidence_id in result.evidence_ids
        }
        inputs = tuple(sorted(
            dependency_evidence
            | {str(item) for item in input_evidence_ids if item}
        ))
        started_at = utc_now()
        self._started_at[node.id] = started_at
        self._input_evidence[node.id] = inputs
        self.journal.append_event(self.plan.plan_id, "graph_node_started", {
            "schema": GRAPH_EVENT_SCHEMA,
            "graph_digest": self.graph_digest,
            "node_id": node.id,
            "phase": node.phase.value,
            "side_effect": node.side_effect,
            "started_at": started_at,
            "input_evidence_ids": list(inputs),
            "details": details or {},
        })

    def recover_indeterminate_effect_boundary(
        self,
        *,
        execution_claim_evidence: str,
        approval_proof_id: str = "",
    ) -> None:
        """Normalize a crash-interrupted graph to a read-only reconcile path."""
        active = self.scheduler.active_node_id
        if active == "approval":
            self.finish(
                "approval",
                "succeeded",
                output_evidence_ids=(approval_proof_id,) if approval_proof_id else (),
                details={"source": "durable_execution_claim_recovery"},
            )
            active = None
        if active == "revalidate":
            self.finish(
                "revalidate",
                "skipped",
                details={"reason": "worker_crashed_before_revalidation_was_durable"},
            )
            active = None

        results = self.scheduler.results
        previous = results[-1] if results else None
        if previous is not None and previous.phase.value == "awaiting_approval":
            self.start(
                "revalidate",
                details={"source": "crash_recovery"},
            )
            self.finish(
                "revalidate",
                "skipped",
                details={"reason": "worker_crashed_before_revalidation_was_durable"},
            )
            previous = self.scheduler.results[-1]
        if (
            previous is not None
            and previous.phase.value == "revalidate"
            and previous.outcome.value in {"skipped", "indeterminate"}
        ):
            node = self.scheduler.recover_start_indeterminate_effect(
                evidence=execution_claim_evidence,
            )
            self._persist_start(
                node,
                details={
                    "source": "crash_recovery",
                    "write_replayed": False,
                    "execution_claim_evidence": execution_claim_evidence,
                },
            )
            self.finish(
                "execute",
                "indeterminate",
                details={
                    "reason": "worker_crashed_at_possible_effect_boundary",
                    "write_replayed": False,
                },
            )
            return
        if self.scheduler.active_node_id == "execute":
            self.finish(
                "execute",
                "indeterminate",
                details={
                    "reason": "worker_crashed_during_effect_boundary",
                    "write_replayed": False,
                },
            )
            return
        if self.scheduler.active_node_id == "verify":
            self.finish(
                "verify",
                "failed",
                details={"reason": "worker_crashed_during_read_only_verification"},
            )

    def finish(
        self,
        node_id: str,
        outcome: NodeOutcome | str,
        *,
        output_evidence_ids: Iterable[str] = (),
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        output_ids = tuple(sorted(set(
            str(item) for item in output_evidence_ids if item
        )))
        result = self.scheduler.finish(
            node_id, outcome, evidence_ids=output_ids,
        )
        finished_at = utc_now()
        if duration_ms is None:
            duration_ms = max(0.0, (
                _timestamp(finished_at) - _timestamp(self._started_at[node_id])
            ).total_seconds() * 1000.0)
        node_evidence_id = sha256_json({
            "plan_id": self.plan.plan_id,
            "plan_hash": self.plan.plan_hash,
            "graph_digest": self.graph_digest,
            "node_id": node_id,
            "outcome": result.outcome.value,
            "input_evidence_ids": list(self._input_evidence.get(node_id, ())),
            "output_evidence_ids": list(output_ids),
        })
        self.journal.append_event(self.plan.plan_id, "graph_node_finished", {
            "schema": GRAPH_EVENT_SCHEMA,
            "graph_digest": self.graph_digest,
            "node_id": result.node_id,
            "phase": result.phase.value,
            "outcome": result.outcome.value,
            "started_at": self._started_at[node_id],
            "finished_at": finished_at,
            "duration_ms": round(float(duration_ms), 3),
            "input_evidence_ids": list(self._input_evidence.get(node_id, ())),
            "output_evidence_ids": list(output_ids),
            "node_evidence_id": node_evidence_id,
            "details": details or {},
        })
        return node_evidence_id

    def fail_closed(self, error: BaseException, *, effect_dispatched: bool) -> None:
        """Close an unexpected infrastructure path without inventing success.

        This is deliberately conservative: a failure before the durable Effect
        boundary aborts; once Effect may have been dispatched, any missing
        verification becomes an escalation.  The helper never retries Effect.
        """
        details = {
            "reason": "runtime_exception",
            "error_type": type(error).__name__,
        }
        active = self.scheduler.active_node_id
        if active is not None:
            phase = next(
                item.phase.value for item in self.scheduler.graph.nodes
                if item.id == active
            )
            self.finish(
                active,
                "indeterminate" if phase == "execute" else "failed",
                details=details,
            )
        if self.scheduler.machine.phase.value in {"commit", "abort", "escalate"}:
            return

        results = self.scheduler.results
        executed = any(item.phase.value == "execute" for item in results)
        if not effect_dispatched and not executed:
            self.start("abort", details=details)
            self.finish("abort", "succeeded", details=details)
            return

        # If the exception landed between durable nodes, create the missing
        # negative observation rather than jumping from success to escalation.
        previous = self.scheduler.results[-1]
        if previous.outcome.value == "succeeded":
            if previous.phase.value == "execute":
                self.start("verify", details=details)
                self.finish("verify", "failed", details=details)
            elif previous.phase.value == "compensate":
                self.start("verify_recovery", details=details)
                self.finish("verify_recovery", "failed", details=details)
        self.start("escalate", details=details)
        self.finish("escalate", "succeeded", details=details)


def stage_latency_summary(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Project per-plan graph timings without conflating Agent latency."""
    stages: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event_type") != "graph_node_finished":
            continue
        payload = event.get("payload") or {}
        phase = str(payload.get("phase") or "unknown")
        stages.setdefault(phase, []).append({
            "node_id": str(payload.get("node_id") or ""),
            "outcome": str(payload.get("outcome") or ""),
            "duration_ms": float(payload.get("duration_ms") or 0.0),
        })
    rendered: dict[str, Any] = {}
    runtime_active_ms = 0.0
    approval_wait_ms = 0.0
    for phase, observations in sorted(stages.items()):
        total = round(sum(item["duration_ms"] for item in observations), 3)
        rendered[phase] = {
            "count": len(observations),
            "total_ms": total,
            "observations": observations,
        }
        if phase == "awaiting_approval":
            approval_wait_ms += total
        else:
            runtime_active_ms += total
    return {
        "schema": "netopyu.io/runtime-stage-latency/v1",
        "stages": rendered,
        "runtime_active_ms": round(runtime_active_ms, 3),
        "approval_wait_ms": round(approval_wait_ms, 3),
        "claim_boundary": (
            "Runtime graph timing excludes Reasoning/LLM latency and is not an end-to-end SLO."
        ),
    }


__all__ = [
    "GRAPH_EVENT_SCHEMA",
    "JournalGraphScheduler",
    "stage_latency_summary",
]
