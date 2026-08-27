"""Versioned compensation registry for independently verified rollback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from dsh_adapter.backend import BackendSession

from .contracts import Evidence, PreparedPlan, canonical_json, sha256_json, utc_now
from .evidence import failed_output, render, same_snapshot, typed_evidence
from .policies import ToolContract, project_arguments


@dataclass(frozen=True)
class CompensationResult:
    result: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class CompensationContext:
    backend: BackendSession
    plan: PreparedPlan
    contract: ToolContract
    timeout_seconds: float


Compensator = Callable[[CompensationContext], Awaitable[CompensationResult]]


class CompensatorRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Compensator] = {}

    def register(self, contract_id: str) -> Callable[[Compensator], Compensator]:
        def decorator(handler: Compensator) -> Compensator:
            if contract_id in self._handlers:
                raise RuntimeError(f"duplicate compensator contract {contract_id}")
            self._handlers[contract_id] = handler
            return handler
        return decorator

    async def compensate(self, context: CompensationContext) -> CompensationResult:
        contract_id = context.contract.compensator
        handler = self._handlers.get(contract_id or "")
        if handler is None:
            raise RuntimeError(f"unsupported compensator {contract_id or 'none'}")
        return await handler(context)

    def contract_ids(self) -> frozenset[str]:
        return frozenset(self._handlers)


REGISTRY = CompensatorRegistry()


async def compensate_operation(
    backend: BackendSession,
    plan: PreparedPlan,
    contract: ToolContract,
    timeout_seconds: float,
) -> CompensationResult:
    return await REGISTRY.compensate(CompensationContext(
        backend=backend,
        plan=plan,
        contract=contract,
        timeout_seconds=timeout_seconds,
    ))


@REGISTRY.register("inverse-tool-v1")
async def _inverse_tool(context: CompensationContext) -> CompensationResult:
    plan, contract = context.plan, context.contract
    if not contract.rollback_tool or not contract.preflight_tool or not plan.preflight:
        raise RuntimeError("inverse-tool compensation requires rollback and preflight contracts")
    rollback = context.backend.callables.get(contract.rollback_tool)
    preflight = context.backend.callables.get(contract.preflight_tool)
    if rollback is None or preflight is None:
        raise RuntimeError("rollback or independent preflight tool is unavailable")

    rollback_args = project_arguments(plan.arguments, contract.rollback_fields)
    rollback_args["reason"] = f"automatic rollback of network runtime plan {plan.plan_id}"
    rollback_result = render(await asyncio.wait_for(
        rollback(rollback_args), timeout=context.timeout_seconds,
    ))
    if failed_output(rollback_result):
        raise RuntimeError(rollback_result)

    read_args = project_arguments(plan.arguments, contract.preflight_fields)
    read_result = render(await preflight(read_args))
    after = typed_evidence(contract.preflight_tool, read_result)
    before = plan.preflight[0].value
    restored = isinstance(before, dict) and same_snapshot(before, after)
    command_evidence = Evidence(
        evidence_type="rollback",
        source=contract.rollback_tool,
        target=canonical_json(rollback_args),
        observed_at=utc_now(),
        value={
            "digest": sha256_json(rollback_result),
            "bytes": len(rollback_result),
            "facts": {"rollback_command_accepted": True},
        },
        passed=True,
        predicate="versioned compensation command completed without an error marker",
        expected=True,
    )
    restore_evidence = Evidence(
        evidence_type="rollback_postcondition",
        source=contract.preflight_tool,
        target=canonical_json(read_args),
        observed_at=utc_now(),
        value=after,
        passed=restored,
        predicate="fresh independent read exactly matches typed preflight state",
        expected=before,
    )
    if not restored:
        raise RuntimeError("rollback command completed but preflight state was not restored")
    return CompensationResult(rollback_result, (command_evidence, restore_evidence))
