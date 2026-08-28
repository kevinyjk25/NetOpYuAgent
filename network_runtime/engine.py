"""Deterministic prepare/approve/execute/verify/rollback network runtime.

The model and DSH choose *what* to request.  This module owns whether a write
is safe to execute and is the only Python path allowed to invoke mutating
network tools.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from dsh_adapter.backend import BackendSession, open_backend

from .contracts import (
    ApprovalError,
    Evidence,
    ExecutionOutcome,
    NetworkRuntimeError,
    OutcomeIndeterminateError,
    PlanIntegrityError,
    PlanState,
    PreparedPlan,
    TERMINAL_STATES,
    canonical_json,
    sha256_json,
    utc_now,
)
from .compensators import REGISTRY as COMPENSATORS, compensate_operation
from .access import ObservationAccessContext, ObservationPolicy
from .capabilities import CapabilityKind
from .evidence import (
    bounded as _bounded,
    failed_output as _failed_output,
    render as _render,
    same_snapshot,
    typed_evidence as _evidence_value,
)
from .journal import NetworkJournal
from .l0_skills import REGISTRY as L0_SKILLS, L0SkillContract, compile_intent
from .policies import ToolContract, project_arguments, resolve_contract
from .validation import assess_risk, compile_parameters
from .verifiers import REGISTRY as VERIFIERS, verify_operation
from .workflows import WorkflowRuntime


FaultHook = Callable[[str, PreparedPlan], Any]
BackendFactory = Callable[[str], Awaitable[BackendSession]]

def default_journal_path() -> Path:
    configured = (
        os.environ.get("NETOPYU_NETWORK_RUNTIME_STORE")
        or os.environ.get("NETOPYU_DSH_NETWORK_RUNTIME_STORE")
    )
    return Path(configured).expanduser() if configured else Path("data/network_runtime.sqlite")

class NetworkRuntime:
    """One-shot execution engine with durable plan and evidence journaling."""

    def __init__(
        self,
        journal_path: str | Path | None = None,
        *,
        backend_factory: BackendFactory = open_backend,
        plan_ttl_seconds: int = 300,
        execution_timeout_seconds: float | None = None,
        fault_hook: FaultHook | None = None,
        observation_policy: ObservationPolicy | None = None,
    ) -> None:
        if not 30 <= plan_ttl_seconds <= 3600:
            raise ValueError("plan_ttl_seconds must be between 30 and 3600")
        self.journal_path = Path(journal_path or default_journal_path())
        self.backend_factory = backend_factory
        self.plan_ttl_seconds = plan_ttl_seconds
        self.execution_timeout_seconds = execution_timeout_seconds or float(
            os.environ.get("NETOPYU_EXECUTION_TIMEOUT")
            or os.environ.get("NETOPYU_DSH_EXECUTION_TIMEOUT", "90")
        )
        self.fault_hook = fault_hook
        self.observation_policy = observation_policy or ObservationPolicy()

    async def _fault(self, stage: str, plan: PreparedPlan) -> None:
        if self.fault_hook is None:
            return
        value = self.fault_hook(stage, plan)
        if inspect.isawaitable(value):
            await value

    async def prepare(
        self,
        profile_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        session_id: str | None = None,
        l0_skill_id: str | None = None,
    ) -> dict[str, Any]:
        """Compile a request into an immutable plan before approval is shown."""
        backend = await self.backend_factory(profile_id)
        try:
            metadata = backend.metadata.get(tool_name)
            tool = backend.callables.get(tool_name)
            if metadata is None or tool is None:
                return {"ok": False, "status": "rejected", "errors": [
                    f"unknown tool {tool_name!r} in {backend.mode} backend"
                ]}
            action_type = str(metadata.get("action_type", "read_only"))
            requires_approval = bool(metadata.get("hitl")) or action_type != "read_only"
            expected_l0 = (
                L0_SKILLS.for_tool(
                    backend.profile_id, tool_name, skill_id=l0_skill_id,
                )
                if requires_approval else None
            )
            compiled = compile_parameters(
                profile=backend.profile_id,
                mode=backend.mode,
                tool_name=tool_name,
                metadata=metadata,
                arguments=arguments,
            )
            if compiled.missing:
                return {
                    "ok": False,
                    "status": "clarification_required",
                    "missing": list(compiled.missing),
                    "errors": list(compiled.errors),
                    "clarification": {
                        "code": "intent_incomplete",
                        "l0_skill_id": expected_l0.skill_id if expected_l0 else None,
                        "questions": [
                            f"Provide one exact value for required field {name!r}."
                            for name in compiled.missing
                        ],
                    },
                }
            if compiled.errors:
                ambiguous = tuple(
                    error for error in compiled.errors
                    if "does not resolve uniquely" in error
                )
                if ambiguous and len(ambiguous) == len(compiled.errors):
                    return {
                        "ok": False,
                        "status": "clarification_required",
                        "missing": [],
                        "errors": list(ambiguous),
                        "clarification": {
                            "code": "intent_target_ambiguous",
                            "l0_skill_id": expected_l0.skill_id if expected_l0 else None,
                            "questions": [
                                "Select one exact target identifier from current inventory."
                            ],
                        },
                    }
                return {
                    "ok": False,
                    "status": "rejected",
                    "errors": list(compiled.errors),
                }
            with WorkflowRuntime(self.journal_path) as workflow_runtime:
                workflow_decision = workflow_runtime.authorize(
                    session_id=session_id,
                    tool_name=tool_name,
                    arguments=compiled.arguments,
                    mutating=requires_approval,
                )
            if not workflow_decision["allowed"]:
                return {
                    "ok": False,
                    "status": "rejected",
                    "errors": [str(workflow_decision["reason"])],
                }
            workflow_context = workflow_decision.get("workflow")
            source = backend.sources.get(tool_name, "unknown")
            contract = resolve_contract(
                tool_name,
                action_type=action_type,
                requires_approval=requires_approval,
                mode=backend.mode,
                source=source,
                metadata=metadata,
            )
            if contract is None:
                return {
                    "ok": False,
                    "status": "rejected",
                    "errors": [
                        f"no reviewed execution contract for {tool_name} from source={source}; fail closed"
                    ],
                }
            l0_contract: L0SkillContract | None = None
            intent = None
            if requires_approval:
                candidates = L0_SKILLS.candidates_for_tool(backend.profile_id, tool_name)
                l0_contract = L0_SKILLS.for_tool(
                    backend.profile_id, tool_name, skill_id=l0_skill_id,
                )
                if l0_contract is None:
                    if l0_skill_id is None and len({item.skill_id for item in candidates}) > 1:
                        return {
                            "ok": False,
                            "status": "rejected",
                            "errors": [
                                "write tool has multiple L0 semantic contracts; "
                                "an exact L0 Skill id is required"
                            ],
                            "candidate_l0_skill_ids": sorted({
                                item.skill_id for item in candidates
                            }),
                        }
                    return {
                        "ok": False,
                        "status": "rejected",
                        "errors": [f"write tool {tool_name!r} has no registered Network L0 Skill"],
                    }
                if l0_skill_id != l0_contract.skill_id:
                    return {
                        "ok": False,
                        "status": "rejected",
                        "errors": [
                            "write preparation requires the exact registered Network L0 Skill id"
                        ],
                        "expected_l0_skill_id": l0_contract.skill_id,
                    }
                if l0_contract.tool_contract_id != contract.contract_id:
                    return {
                        "ok": False,
                        "status": "rejected",
                        "errors": ["Network L0 Skill references a stale tool contract; fail closed"],
                    }
                intent = compile_intent(
                    l0_contract,
                    profile=backend.profile_id,
                    tool_name=tool_name,
                    arguments=compiled.arguments,
                    provenance=compiled.provenance,
                    targets=compiled.targets,
                )
            if requires_approval and contract.verifier not in VERIFIERS.contract_ids():
                return {
                    "ok": False,
                    "status": "rejected",
                    "errors": [f"verifier contract {contract.verifier!r} is not registered; fail closed"],
                }
            if contract.rollback_tool and (
                not contract.compensator
                or contract.compensator not in COMPENSATORS.contract_ids()
            ):
                return {
                    "ok": False,
                    "status": "rejected",
                    "errors": ["rollback tool has no registered compensation contract; fail closed"],
                }
            if not requires_approval:
                return {
                    "ok": True,
                    "status": "read_ready",
                    "arguments": compiled.arguments,
                    "targets": list(compiled.targets),
                    "contract_id": contract.contract_id,
                }

            preflight = await self._preflight(backend, contract, compiled.arguments)
            if any(item.passed is False for item in preflight):
                return {
                    "ok": False,
                    "status": "rejected",
                    "errors": ["preflight did not prove the target is safe and reachable"],
                    "preflight": [item.to_dict() for item in preflight],
                }
            risk, reasons = assess_risk(tool_name, metadata, compiled.arguments)
            created = datetime.now(timezone.utc)
            expires = created + timedelta(seconds=self.plan_ttl_seconds)
            plan = PreparedPlan.create(
                plan_id=str(uuid.uuid4()),
                profile=backend.profile_id,
                tool_name=tool_name,
                tool_version=contract.contract_id,
                action_type=action_type,
                provider_identity=str(metadata.get("provider_identity") or source),
                input_schema_digest=str(
                    metadata.get("input_schema_digest")
                    or sha256_json(metadata.get("parameters") or {})
                ),
                output_schema_digest=str(
                    metadata.get("output_schema_digest")
                    or sha256_json(metadata.get("output_schema") or {})
                ),
                capability_id=str(metadata.get("capability_id") or l0_contract.skill_id),
                capability_version=str(
                    metadata.get("capability_version") or l0_contract.version
                ),
                provider_role=str(metadata.get("provider_role") or "actor"),
                arguments=compiled.arguments,
                argument_provenance=compiled.provenance,
                targets=compiled.targets,
                risk_level=risk,
                risk_reasons=reasons,
                preflight=preflight,
                verification_contract=contract.verifier,
                rollback_contract=contract.compensator,
                l0_skill_id=l0_contract.skill_id,
                l0_skill_version=l0_contract.version,
                l0_contract_hash=l0_contract.contract_hash,
                intent_spec=intent.to_dict(),
                intent_hash=intent.intent_hash,
                step_contract=tuple(step.to_dict() for step in l0_contract.steps),
                workflow_run_id=(workflow_context or {}).get("run_id"),
                workflow_template_hash=(workflow_context or {}).get("template_hash"),
                created_at=created.isoformat(),
                expires_at=expires.isoformat(),
            )
            nonce = secrets.token_urlsafe(32)
            with NetworkJournal(self.journal_path) as journal:
                journal.create(plan, nonce)
                for step_id in ("validate_parameters", "compile_intent", "preflight"):
                    journal.append_event(plan.plan_id, "l0_step_completed", {
                        "step_id": step_id,
                        "l0_contract_hash": l0_contract.contract_hash,
                        "intent_hash": intent.intent_hash,
                    })
            return {
                "ok": True,
                "status": "plan_ready",
                "plan": plan.to_dict(),
                "execution_nonce": nonce,
                "approval_summary": self._approval_summary(plan, source, backend.mode),
            }
        finally:
            await backend.close()

    async def invoke_read(
        self,
        profile_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        access_context: ObservationAccessContext | dict[str, Any] | None = None,
    ) -> str:
        """Strictly validate and invoke a read-only tool without a write lease."""
        backend = await self.backend_factory(profile_id)
        try:
            metadata = backend.metadata.get(tool_name)
            tool = backend.callables.get(tool_name)
            if metadata is None or tool is None:
                raise KeyError(f"unknown tool {tool_name!r} in {backend.mode} backend")
            capability = backend.describe_capability(tool_name)
            if capability.kind != CapabilityKind.OBSERVATION:
                raise ApprovalError(
                    f"direct write invocation of {tool_name} is retired; use runtime prepare/execute"
                )
            subject = ObservationAccessContext.from_value(
                access_context, profile=backend.profile_id,
            )
            decision = self.observation_policy.authorize(
                capability, arguments, subject,
            )
            if not decision.allowed:
                raise ApprovalError(
                    f"observation authorization denied ({decision.code}): {decision.reason}"
                )
            compiled = compile_parameters(
                profile=backend.profile_id,
                mode=backend.mode,
                tool_name=tool_name,
                metadata=metadata,
                arguments=arguments,
            )
            if compiled.missing:
                raise ValueError("missing required parameters: " + ", ".join(compiled.missing))
            if compiled.errors:
                raise ValueError("; ".join(compiled.errors))
            result = _render(await backend.invoke_observation(tool_name, compiled.arguments))
            if _failed_output(result):
                raise NetworkRuntimeError(f"read tool returned a failure: {_bounded(result, 500)}")
            return self._store_large_result(backend, tool_name, result)
        finally:
            await backend.close()

    async def execute(
        self,
        *,
        plan_id: str,
        plan_hash: str,
        execution_nonce: str,
        approval_request_id: str,
        approval_actor: str,
        allow_destructive: bool,
    ) -> ExecutionOutcome:
        """Consume a plan once, execute once, and never infer success sans evidence."""
        if not allow_destructive:
            raise ApprovalError("destructive runtime execution is disabled")
        with NetworkJournal(self.journal_path) as journal:
            candidate = journal.get(plan_id)
            if candidate.plan_hash != plan_hash:
                raise PlanIntegrityError("approved plan hash does not match stored plan")
            if candidate.workflow_run_id and candidate.workflow_template_hash:
                with WorkflowRuntime(self.journal_path) as workflow_runtime:
                    workflow_runtime.validate_plan_binding(
                        candidate.workflow_run_id, candidate.workflow_template_hash,
                    )
            plan = journal.claim_for_execution(
                plan_id=plan_id,
                plan_hash=plan_hash,
                execution_nonce=execution_nonce,
                approval_request_id=approval_request_id,
                approval_actor=approval_actor,
            )
            journal.append_event(plan.plan_id, "l0_step_completed", {
                "step_id": "approval",
                "l0_skill_id": plan.l0_skill_id,
                "intent_hash": plan.intent_hash,
            })
            backend: BackendSession | None = None
            result: str | None = None
            effect_dispatched = False
            try:
                backend = await self.backend_factory(plan.profile)
                contract = self._revalidate_contract(plan, backend)
                if plan.workflow_run_id and plan.workflow_template_hash:
                    with WorkflowRuntime(self.journal_path) as workflow_runtime:
                        workflow_runtime.validate_plan_binding(
                            plan.workflow_run_id, plan.workflow_template_hash,
                        )
                current_preflight = await self._preflight(backend, contract, plan.arguments)
                if not self._preflight_equivalent(plan.preflight, current_preflight):
                    drift_evidence = tuple(Evidence(
                        evidence_type="execution_precondition",
                        source=item.source,
                        target=item.target,
                        observed_at=item.observed_at,
                        value=item.value,
                        passed=False,
                        predicate="execution-time state equals the approved preflight state",
                        expected=(
                            plan.preflight[index].value
                            if index < len(plan.preflight) else {"missing": True}
                        ),
                    ) for index, item in enumerate(current_preflight))
                    journal.transition(
                        plan.plan_id, PlanState.PRECONDITION_CHANGED,
                        "execution_precondition_changed",
                        {"evidence": [item.to_dict() for item in drift_evidence]},
                    )
                    outcome = ExecutionOutcome(
                        plan.plan_id, plan.plan_hash, PlanState.PRECONDITION_CHANGED,
                        None, drift_evidence,
                        "target state changed after approval; write was not sent",
                    )
                    journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
                    journal.release_locks(plan.plan_id)
                    journal.append_event(plan.plan_id, "l0_step_failed", {
                        "step_id": "revalidate", "reason": "precondition_changed",
                    })
                    self._skip_compensation(journal, plan, "write_not_sent")
                    self._complete_audit(journal, plan, outcome.state)
                    return outcome
                journal.append_event(plan.plan_id, "l0_step_completed", {
                    "step_id": "revalidate",
                    "l0_skill_id": plan.l0_skill_id,
                    "intent_hash": plan.intent_hash,
                })
                await self._fault("before_send", plan)
                journal.append_event(plan.plan_id, "l0_step_started", {
                    "step_id": "execute", "tool_name": plan.tool_name,
                })
                try:
                    effect_dispatched = True
                    result = _render(await asyncio.wait_for(
                        backend.invoke_effect(
                            plan.tool_name, plan.arguments, plan=plan, phase="execute",
                        ),
                        timeout=self.execution_timeout_seconds,
                    ))
                    journal.append_event(plan.plan_id, "l0_step_completed", {
                        "step_id": "execute", "result_received": True,
                    })
                    await self._fault("after_send_before_verify", plan)
                except (asyncio.TimeoutError, OutcomeIndeterminateError) as error:
                    journal.append_event(plan.plan_id, "l0_step_indeterminate", {
                        "step_id": "execute", "reason": type(error).__name__,
                    })
                    journal.transition(
                        plan.plan_id, PlanState.OUTCOME_INDETERMINATE,
                        "write_outcome_indeterminate", {"error": str(error)},
                    )
                    return await self._reconcile_indeterminate(journal, backend, plan, contract, result, error)
                except Exception as error:
                    # Once the callable started, a generic transport exception
                    # cannot prove that the remote system made no change.
                    journal.append_event(plan.plan_id, "l0_step_indeterminate", {
                        "step_id": "execute", "reason": type(error).__name__,
                    })
                    journal.transition(
                        plan.plan_id, PlanState.OUTCOME_INDETERMINATE,
                        "write_raised_after_send", {"error": f"{type(error).__name__}: {error}"},
                    )
                    return await self._reconcile_indeterminate(journal, backend, plan, contract, result, error)

                journal.transition(plan.plan_id, PlanState.VERIFYING, "verification_started", {})
                journal.append_event(plan.plan_id, "l0_step_started", {
                    "step_id": "verify", "contract": plan.verification_contract,
                })
                await self._fault("before_verify", plan)
                evidence, passed, internal_rollback, verify_error = await self._verify(
                    backend, plan, contract, result,
                )
                journal.append_event(plan.plan_id, "l0_step_completed", {
                    "step_id": "verify", "passed": passed,
                    "internal_rollback": internal_rollback,
                })
                if passed:
                    journal.transition(
                        plan.plan_id, PlanState.VERIFIED_SUCCESS,
                        "verification_passed", {"evidence": [item.to_dict() for item in evidence]},
                    )
                    rendered = self._store_large_result(backend, plan.tool_name, result)
                    outcome = ExecutionOutcome(
                        plan.plan_id, plan.plan_hash, PlanState.VERIFIED_SUCCESS,
                        rendered, evidence,
                    )
                    journal.store_outcome(plan.plan_id, outcome.to_dict())
                    journal.release_locks(plan.plan_id)
                    self._skip_compensation(journal, plan, "verification_passed")
                    self._complete_audit(journal, plan, outcome.state)
                    return outcome
                if internal_rollback:
                    journal.transition(plan.plan_id, PlanState.ROLLING_BACK, "tool_reported_rollback", {})
                    journal.append_event(plan.plan_id, "l0_step_started", {
                        "step_id": "compensate", "mode": "tool_internal_rollback_verification",
                    })
                    restored = await self._verify_restored_preflight(backend, plan, contract)
                    evidence = (*evidence, restored)
                    if restored.passed is not True:
                        journal.transition(
                            plan.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
                            "internal_rollback_not_independently_verified",
                            {"evidence": [item.to_dict() for item in evidence]},
                        )
                        outcome = ExecutionOutcome(
                            plan.plan_id, plan.plan_hash, PlanState.MANUAL_INTERVENTION_REQUIRED,
                            result, evidence, "tool reported rollback but an independent read did not prove restoration",
                        )
                        journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
                        journal.append_event(plan.plan_id, "l0_step_failed", {
                            "step_id": "compensate", "reason": "restoration_not_verified",
                        })
                        self._complete_audit(journal, plan, outcome.state)
                        return outcome
                    journal.append_event(plan.plan_id, "l0_step_completed", {
                        "step_id": "compensate", "mode": "tool_internal_rollback",
                        "passed": True,
                    })
                    journal.transition(
                        plan.plan_id, PlanState.ROLLBACK_VERIFIED,
                        "internal_rollback_verified", {"evidence": [item.to_dict() for item in evidence]},
                    )
                    outcome = ExecutionOutcome(
                        plan.plan_id, plan.plan_hash, PlanState.ROLLBACK_VERIFIED,
                        result, evidence, verify_error or "write was rolled back",
                    )
                    journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
                    journal.release_locks(plan.plan_id)
                    self._complete_audit(journal, plan, outcome.state)
                    return outcome
                return await self._rollback_or_escalate(
                    journal, backend, plan, contract, result, evidence, verify_error,
                )
            except (ApprovalError, PlanIntegrityError):
                raise
            except Exception as error:
                current = journal.get(plan.plan_id)
                if current.state == PlanState.EXECUTING:
                    journal.transition(
                        plan.plan_id, PlanState.EXECUTION_FAILED,
                        "runtime_failed_before_verification", {"error": str(error)},
                    )
                    current = journal.get(plan.plan_id)
                if current.state in {
                    PlanState.EXECUTION_FAILED,
                    PlanState.OUTCOME_INDETERMINATE,
                    PlanState.VERIFYING,
                    PlanState.ROLLING_BACK,
                }:
                    journal.transition(
                        plan.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
                        "runtime_exception_escalated", {"error": f"{type(error).__name__}: {error}"},
                    )
                outcome = ExecutionOutcome(
                    plan.plan_id, plan.plan_hash, PlanState.MANUAL_INTERVENTION_REQUIRED,
                    result, error=f"{type(error).__name__}: {error}",
                )
                journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
                # Keep resource locks for manual intervention.
                self._skip_compensation(journal, plan, "runtime_exception_requires_operator")
                self._complete_audit(journal, plan, outcome.state)
                return outcome
            finally:
                if backend is not None:
                    if effect_dispatched:
                        current = journal.get(plan.plan_id)
                        if current.state in TERMINAL_STATES:
                            try:
                                finalized = await backend.finalize_effect(
                                    plan, current.state.value,
                                )
                                if finalized is not None:
                                    journal.append_event(plan.plan_id, "actor_finalized", {
                                        "state": current.state.value,
                                    })
                            except Exception as finalize_error:
                                journal.append_event(plan.plan_id, "actor_finalize_failed", {
                                    "error": f"{type(finalize_error).__name__}: {finalize_error}",
                                })
                    await backend.close()

    def inspect(self, plan_id: str) -> dict[str, Any]:
        with NetworkJournal(self.journal_path) as journal:
            return {
                "plan": journal.get(plan_id).to_dict(),
                "events": journal.events(plan_id),
                "audit": journal.verify_event_chain(plan_id),
                "record": journal.record(plan_id),
            }

    def audit(self, plan_id: str) -> dict[str, Any]:
        with NetworkJournal(self.journal_path) as journal:
            return journal.verify_event_chain(plan_id)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with NetworkJournal(self.journal_path) as journal:
            return journal.recent(limit)

    async def recover_inflight(self) -> list[dict[str, Any]]:
        """Reconcile crash-marked plans by reads only; never replay a write."""
        outcomes: list[dict[str, Any]] = []
        with NetworkJournal(self.journal_path) as journal:
            plan_ids = journal.plan_ids_in_state(PlanState.OUTCOME_INDETERMINATE)
            for plan_id in plan_ids:
                plan = journal.get(plan_id)
                backend: BackendSession | None = None
                try:
                    backend = await self.backend_factory(plan.profile)
                    contract = self._revalidate_contract(plan, backend)
                    outcome = await self._reconcile_indeterminate(
                        journal, backend, plan, contract, None,
                        OutcomeIndeterminateError("runtime restarted during side-effect processing"),
                    )
                    if outcome.state in TERMINAL_STATES:
                        finalized = await backend.finalize_effect(
                            plan, outcome.state.value,
                        )
                        if finalized is not None:
                            journal.append_event(plan.plan_id, "actor_finalized", {
                                "state": outcome.state.value,
                                "source": "startup_reconciliation",
                            })
                    outcomes.append(outcome.to_dict())
                except Exception as error:
                    current = journal.get(plan_id)
                    if current.state == PlanState.OUTCOME_INDETERMINATE:
                        journal.transition(
                            plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
                            "startup_reconciliation_failed",
                            {"error": f"{type(error).__name__}: {error}"},
                        )
                    outcome = ExecutionOutcome(
                        plan.plan_id, plan.plan_hash,
                        PlanState.MANUAL_INTERVENTION_REQUIRED, None,
                        error=f"startup reconciliation failed: {type(error).__name__}: {error}",
                    )
                    journal.store_outcome(plan_id, outcome.to_dict(), outcome.error)
                    outcomes.append(outcome.to_dict())
                finally:
                    if backend is not None:
                        await backend.close()
            await self._recover_actor_finalizers(journal)
        return outcomes

    async def _recover_actor_finalizers(self, journal: NetworkJournal) -> None:
        """Retry a terminal Runtime-to-Actor commit lost during process shutdown."""
        plan_ids = {
            plan_id
            for state in TERMINAL_STATES
            for plan_id in journal.plan_ids_in_state(state)
        }
        for plan_id in sorted(plan_ids):
            plan = journal.get(plan_id)
            if ":netopyu.network-actor@" not in plan.provider_identity:
                continue
            events = journal.events(plan_id)
            if any(item["event_type"] == "actor_finalized" for item in events):
                continue
            if not any(
                item["event_type"] == "l0_step_started"
                and item["payload"].get("step_id") == "execute"
                for item in events
            ):
                continue
            backend: BackendSession | None = None
            try:
                backend = await self.backend_factory(plan.profile)
                self._revalidate_contract(plan, backend)
                finalized = await backend.finalize_effect(plan, plan.state.value)
                if finalized is None:
                    raise RuntimeError("approved Network Actor finalizer is unavailable")
                journal.append_event(plan_id, "actor_finalized", {
                    "state": plan.state.value,
                    "source": "startup_finalizer_recovery",
                })
            except Exception as error:
                journal.append_event(plan_id, "actor_finalize_failed", {
                    "source": "startup_finalizer_recovery",
                    "error": f"{type(error).__name__}: {error}",
                })
            finally:
                if backend is not None:
                    await backend.close()

    def reject(self, *, plan_id: str, plan_hash: str, reason: str) -> dict[str, Any]:
        with NetworkJournal(self.journal_path) as journal:
            plan = journal.get(plan_id)
            if plan.plan_hash != plan_hash:
                raise PlanIntegrityError("rejected plan hash does not match stored plan")
            if plan.state == PlanState.REJECTED:
                return {"ok": True, "plan_id": plan_id, "state": plan.state.value}
            if plan.state != PlanState.PLAN_READY:
                raise NetworkRuntimeError(
                    f"cannot reject plan {plan_id} from state {plan.state.value}"
                )
            updated = journal.transition(
                plan_id, PlanState.REJECTED, "approval_rejected",
                {"reason": _bounded(reason, 1000)},
            )
            journal.store_outcome(plan_id, {
                "ok": False,
                "plan_id": plan_id,
                "plan_hash": plan_hash,
                "state": updated.state.value,
                "error": _bounded(reason, 1000),
            }, _bounded(reason, 1000))
            journal.release_locks(plan_id)
            journal.append_event(plan_id, "l0_step_failed", {
                "step_id": "approval", "reason": "operator_rejected",
            })
            self._skip_compensation(journal, plan, "write_not_sent")
            self._complete_audit(journal, plan, updated.state)
            return {"ok": True, "plan_id": plan_id, "state": updated.state.value}

    async def _preflight(
        self,
        backend: BackendSession,
        contract: ToolContract,
        arguments: dict[str, Any],
    ) -> tuple[Evidence, ...]:
        if not contract.preflight_tool:
            return (Evidence(
                evidence_type="contract_preflight",
                source="reviewed-local-contract",
                target="contract",
                observed_at=utc_now(),
                value={"contract_id": contract.contract_id},
                passed=True,
                predicate="reviewed simulator contract exists",
                expected=True,
            ),)
        tool = backend.callables.get(contract.preflight_tool)
        if tool is None:
            return (Evidence(
                evidence_type="preflight",
                source=contract.preflight_tool,
                target="contract",
                observed_at=utc_now(),
                value={"error_code": "preflight_tool_unavailable"},
                passed=False,
                predicate="preflight tool is registered",
                expected=True,
            ),)
        projected = project_arguments(arguments, contract.preflight_fields)
        rendered = _render(await tool(projected))
        value = _evidence_value(contract.preflight_tool, rendered)
        passed = (
            not _failed_output(rendered)
            and value.get("facts", {}).get("ok") is not False
        )
        return (Evidence(
            evidence_type="preflight",
            source=contract.preflight_tool,
            target=canonical_json(projected),
            observed_at=utc_now(),
            value=value,
            passed=passed,
            predicate="preflight read succeeds without an error marker",
            expected=True,
        ),)

    def _revalidate_contract(self, plan: PreparedPlan, backend: BackendSession) -> ToolContract:
        metadata = backend.metadata.get(plan.tool_name)
        if metadata is None or plan.tool_name not in backend.callables:
            raise PlanIntegrityError("approved tool is no longer registered")
        action = str(metadata.get("action_type", "read_only"))
        requires_approval = bool(metadata.get("hitl")) or action != "read_only"
        contract = resolve_contract(
            plan.tool_name,
            action_type=action,
            requires_approval=requires_approval,
            mode=backend.mode,
            source=backend.sources.get(plan.tool_name, "unknown"),
            metadata=metadata,
        )
        current_source = backend.sources.get(plan.tool_name, "unknown")
        current_provider = str(metadata.get("provider_identity") or current_source)
        current_input_digest = str(
            metadata.get("input_schema_digest")
            or sha256_json(metadata.get("parameters") or {})
        )
        current_output_digest = str(
            metadata.get("output_schema_digest")
            or sha256_json(metadata.get("output_schema") or {})
        )
        current_l0 = L0_SKILLS.for_tool(
            plan.profile,
            plan.tool_name,
            skill_id=plan.l0_skill_id,
            version=plan.l0_skill_version,
        )
        current_capability_id = str(
            metadata.get("capability_id")
            or (current_l0.skill_id if current_l0 is not None else "")
        )
        current_capability_version = str(
            metadata.get("capability_version")
            or (current_l0.version if current_l0 is not None else "")
        )
        current_provider_role = str(metadata.get("provider_role") or "actor")
        if (
            contract is None
            or contract.contract_id != plan.tool_version
            or contract.verifier != plan.verification_contract
            or contract.compensator != plan.rollback_contract
            or action != plan.action_type
            or not requires_approval
            or current_provider != plan.provider_identity
            or current_input_digest != plan.input_schema_digest
            or current_output_digest != plan.output_schema_digest
            or current_capability_id != plan.capability_id
            or current_capability_version != plan.capability_version
            or current_provider_role != plan.provider_role
        ):
            raise PlanIntegrityError("runtime tool contract changed after approval")
        if contract.verifier not in VERIFIERS.contract_ids():
            raise PlanIntegrityError("approved verifier contract is no longer registered")
        if contract.compensator and contract.compensator not in COMPENSATORS.contract_ids():
            raise PlanIntegrityError("approved compensation contract is no longer registered")
        l0_contract = current_l0
        if (
            l0_contract is None
            or plan.l0_skill_id != l0_contract.skill_id
            or plan.l0_skill_version != l0_contract.version
            or plan.l0_contract_hash != l0_contract.contract_hash
            or plan.step_contract != tuple(step.to_dict() for step in l0_contract.steps)
            or l0_contract.tool_contract_id != contract.contract_id
        ):
            raise PlanIntegrityError("Network L0 Skill contract changed after approval")
        intent = compile_intent(
            l0_contract,
            profile=plan.profile,
            tool_name=plan.tool_name,
            arguments=plan.arguments,
            provenance=plan.argument_provenance,
            targets=plan.targets,
        )
        if intent.intent_hash != plan.intent_hash or intent.to_dict() != plan.intent_spec:
            raise PlanIntegrityError("approved intent is not bound to the L0 Skill plan")
        return contract

    @staticmethod
    def _preflight_equivalent(
        approved: tuple[Evidence, ...], current: tuple[Evidence, ...],
    ) -> bool:
        if len(approved) != len(current):
            return False
        for before, after in zip(approved, current, strict=True):
            if (
                before.source != after.source
                or before.target != after.target
                or before.passed is not True
                or after.passed is not True
            ):
                return False
            if not same_snapshot(before.value, after.value):
                return False
        return True

    async def _verify(
        self,
        backend: BackendSession,
        plan: PreparedPlan,
        contract: ToolContract,
        result: str | None,
    ) -> tuple[tuple[Evidence, ...], bool, bool, str | None]:
        verification = await verify_operation(backend, plan, contract, result)
        return (
            verification.evidence,
            verification.passed,
            verification.internal_rollback,
            verification.error,
        )

    async def _reconcile_indeterminate(
        self,
        journal: NetworkJournal,
        backend: BackendSession,
        plan: PreparedPlan,
        contract: ToolContract,
        result: str | None,
        error: BaseException,
    ) -> ExecutionOutcome:
        journal.transition(plan.plan_id, PlanState.VERIFYING, "reconciliation_started", {})
        journal.append_event(plan.plan_id, "l0_step_started", {
            "step_id": "verify", "mode": "indeterminate_reconciliation",
        })
        try:
            evidence, passed, internal_rollback, verify_error = await self._verify(
                backend, plan, contract, result,
            )
        except Exception as verify_exception:
            evidence = ()
            passed = internal_rollback = False
            verify_error = f"reconciliation failed: {type(verify_exception).__name__}: {verify_exception}"
        journal.append_event(plan.plan_id, "l0_step_completed", {
            "step_id": "verify", "passed": passed,
            "internal_rollback": internal_rollback,
            "mode": "indeterminate_reconciliation",
        })
        if passed:
            journal.transition(
                plan.plan_id, PlanState.VERIFIED_SUCCESS,
                "indeterminate_reconciled_success", {"evidence": [item.to_dict() for item in evidence]},
            )
            outcome = ExecutionOutcome(
                plan.plan_id, plan.plan_hash, PlanState.VERIFIED_SUCCESS,
                result, evidence,
            )
            journal.store_outcome(plan.plan_id, outcome.to_dict())
            journal.release_locks(plan.plan_id)
            self._skip_compensation(journal, plan, "reconciliation_verified_success")
            self._complete_audit(journal, plan, outcome.state)
            return outcome
        if internal_rollback:
            journal.transition(plan.plan_id, PlanState.ROLLING_BACK, "indeterminate_internal_rollback", {})
            journal.append_event(plan.plan_id, "l0_step_started", {
                "step_id": "compensate", "mode": "indeterminate_internal_rollback_verification",
            })
            restored = await self._verify_restored_preflight(backend, plan, contract)
            evidence = (*evidence, restored)
            if restored.passed is not True:
                journal.transition(
                    plan.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
                    "indeterminate_rollback_not_verified",
                    {"evidence": [item.to_dict() for item in evidence]},
                )
                outcome = ExecutionOutcome(
                    plan.plan_id, plan.plan_hash, PlanState.MANUAL_INTERVENTION_REQUIRED,
                    result, evidence,
                    "indeterminate write reported rollback but restoration was not independently proven",
                )
                journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
                journal.append_event(plan.plan_id, "l0_step_failed", {
                    "step_id": "compensate", "reason": "restoration_not_verified",
                })
                self._complete_audit(journal, plan, outcome.state)
                return outcome
            journal.append_event(plan.plan_id, "l0_step_completed", {
                "step_id": "compensate", "mode": "indeterminate_internal_rollback",
                "passed": True,
            })
            journal.transition(
                plan.plan_id, PlanState.ROLLBACK_VERIFIED, "rollback_verified",
                {"evidence": [item.to_dict() for item in evidence]},
            )
            outcome = ExecutionOutcome(
                plan.plan_id, plan.plan_hash, PlanState.ROLLBACK_VERIFIED,
                result, evidence, verify_error,
            )
            journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
            journal.release_locks(plan.plan_id)
            self._complete_audit(journal, plan, outcome.state)
            return outcome
        journal.transition(
            plan.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
            "indeterminate_not_reconciled", {"error": str(error), "verification": verify_error},
        )
        outcome = ExecutionOutcome(
            plan.plan_id, plan.plan_hash, PlanState.MANUAL_INTERVENTION_REQUIRED,
            result, evidence,
            f"write outcome could not be reconciled; no retry performed: {error}",
        )
        journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
        self._skip_compensation(journal, plan, "indeterminate_requires_operator")
        self._complete_audit(journal, plan, outcome.state)
        return outcome

    async def _rollback_or_escalate(
        self,
        journal: NetworkJournal,
        backend: BackendSession,
        plan: PreparedPlan,
        contract: ToolContract,
        result: str | None,
        evidence: tuple[Evidence, ...],
        verify_error: str | None,
    ) -> ExecutionOutcome:
        if not contract.compensator:
            journal.transition(
                plan.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
                "verification_failed_no_safe_rollback", {"error": verify_error},
            )
            outcome = ExecutionOutcome(
                plan.plan_id, plan.plan_hash, PlanState.MANUAL_INTERVENTION_REQUIRED,
                result, evidence, verify_error or "postcondition not verified",
            )
            journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
            self._complete_audit(journal, plan, outcome.state)
            return outcome
        journal.transition(plan.plan_id, PlanState.ROLLING_BACK, "rollback_started", {})
        journal.append_event(plan.plan_id, "l0_step_started", {
            "step_id": "compensate", "contract": plan.rollback_contract,
        })
        try:
            await self._fault("before_rollback", plan)
            compensation = await compensate_operation(
                backend, plan, contract, self.execution_timeout_seconds,
            )
            all_evidence = (*evidence, *compensation.evidence)
            journal.transition(
                plan.plan_id, PlanState.ROLLBACK_VERIFIED,
                "rollback_verified", {"evidence": [item.to_dict() for item in all_evidence]},
            )
            outcome = ExecutionOutcome(
                plan.plan_id, plan.plan_hash, PlanState.ROLLBACK_VERIFIED,
                result, all_evidence, verify_error or "write verification failed and was rolled back",
            )
            journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
            journal.release_locks(plan.plan_id)
            journal.append_event(plan.plan_id, "l0_step_completed", {
                "step_id": "compensate", "contract": plan.rollback_contract,
                "passed": True,
            })
            self._complete_audit(journal, plan, outcome.state)
            return outcome
        except Exception as rollback_error:
            journal.transition(
                plan.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
                "rollback_failed", {"error": f"{type(rollback_error).__name__}: {rollback_error}"},
            )
            outcome = ExecutionOutcome(
                plan.plan_id, plan.plan_hash, PlanState.MANUAL_INTERVENTION_REQUIRED,
                result, evidence,
                f"{verify_error or 'verification failed'}; rollback failed: {rollback_error}",
            )
            journal.store_outcome(plan.plan_id, outcome.to_dict(), outcome.error)
            journal.append_event(plan.plan_id, "l0_step_failed", {
                "step_id": "compensate", "contract": plan.rollback_contract,
                "reason": type(rollback_error).__name__,
            })
            self._complete_audit(journal, plan, outcome.state)
            return outcome

    async def _verify_restored_preflight(
        self,
        backend: BackendSession,
        plan: PreparedPlan,
        contract: ToolContract,
    ) -> Evidence:
        if not contract.preflight_tool or not plan.preflight:
            return Evidence(
                evidence_type="rollback_postcondition",
                source=contract.preflight_tool or "none",
                target=",".join(plan.targets), observed_at=utc_now(),
                value={"error_code": "no_independent_preflight_contract"},
                passed=False, predicate="preflight state restored exactly", expected=True,
            )
        output = _render(await backend.callables[contract.preflight_tool](
            project_arguments(plan.arguments, contract.preflight_fields)
        ))
        before = plan.preflight[0].value
        after = _evidence_value(contract.preflight_tool, output)
        passed = isinstance(before, dict) and same_snapshot(before, after)
        return Evidence(
            evidence_type="rollback_postcondition",
            source=contract.preflight_tool,
            target=canonical_json(project_arguments(plan.arguments, contract.preflight_fields)),
            observed_at=utc_now(), value=after, passed=passed,
            predicate="fresh independent read exactly matches typed preflight state",
            expected=before,
        )

    @staticmethod
    def _skip_compensation(
        journal: NetworkJournal, plan: PreparedPlan, reason: str,
    ) -> None:
        if any(step.get("step_id") == "compensate" for step in plan.step_contract):
            journal.append_event(plan.plan_id, "l0_step_skipped", {
                "step_id": "compensate", "reason": reason,
            })

    @staticmethod
    def _complete_audit(
        journal: NetworkJournal, plan: PreparedPlan, state: PlanState,
    ) -> None:
        journal.append_event(plan.plan_id, "l0_step_completed", {
            "step_id": "audit", "terminal_state": state.value,
        })

    @staticmethod
    def _store_large_result(backend: BackendSession, tool_name: str, result: str) -> str:
        if tool_name in {"read_stored_result", "process_stored_chunks"}:
            return result
        return backend._tool_store.store(tool_name, result)

    @staticmethod
    def _approval_summary(plan: PreparedPlan, source: str, mode: str) -> str:
        args = canonical_json(plan.arguments)
        return (
            f"Effect plan {plan.plan_id}\n"
            f"Tool: {plan.tool_name} ({plan.action_type}, risk={plan.risk_level.value})\n"
            f"Target(s): {', '.join(plan.targets)}\n"
            f"Arguments: {args}\n"
            f"Source: {source}; backend={mode}; contract={plan.tool_version}\n"
            f"Provider: {plan.provider_identity}\n"
            f"Capability: {plan.capability_id}@{plan.capability_version} "
            f"({plan.provider_role})\n"
            f"Schemas: input={plan.input_schema_digest}; output={plan.output_schema_digest}\n"
            f"L0 Skill: {plan.l0_skill_id}@{plan.l0_skill_version} "
            f"({plan.l0_contract_hash})\n"
            f"Intent: {plan.intent_spec.get('intent_kind')} "
            f"desired={canonical_json(plan.intent_spec.get('desired_state', {}))}\n"
            f"Intent hash: {plan.intent_hash}\n"
            f"Verification: {plan.verification_contract}; rollback: {plan.rollback_contract or 'none'}\n"
            f"Workflow: {plan.workflow_run_id or 'standalone'}"
            f"{f' ({plan.workflow_template_hash})' if plan.workflow_template_hash else ''}\n"
            f"Expires: {plan.expires_at}\n"
            f"Plan hash: {plan.plan_hash}"
        )
