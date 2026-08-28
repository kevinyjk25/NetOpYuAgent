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


@REGISTRY.register("device-config-snapshot-v1")
async def _restore_device_snapshot(context: CompensationContext) -> CompensationResult:
    """Restore the provider-owned execution snapshot and prove exact state."""
    plan, contract = context.plan, context.contract
    if not contract.rollback_tool or not contract.preflight_tool or not plan.preflight:
        raise RuntimeError("device snapshot compensation contract is incomplete")
    restore = context.backend.callables.get(contract.rollback_tool)
    preflight = context.backend.callables.get(contract.preflight_tool)
    if restore is None or preflight is None:
        raise RuntimeError("device provider does not expose reviewed snapshot restoration")
    rollback_args = project_arguments(plan.arguments, contract.rollback_fields)
    rollback_result = render(await asyncio.wait_for(
        restore(rollback_args), timeout=context.timeout_seconds,
    ))
    if failed_output(rollback_result):
        raise RuntimeError(rollback_result)
    read_args = project_arguments(plan.arguments, contract.preflight_fields)
    read_result = render(await preflight(read_args))
    after = typed_evidence(contract.preflight_tool, read_result)
    before = plan.preflight[0].value
    restored = isinstance(before, dict) and same_snapshot(before, after)
    evidence = (
        Evidence(
            evidence_type="rollback",
            source=contract.rollback_tool,
            target=canonical_json(rollback_args),
            observed_at=utc_now(),
            value={
                "digest": sha256_json(rollback_result),
                "bytes": len(rollback_result),
                "facts": {"snapshot_restore_completed": True},
            },
            passed=True,
            predicate="reviewed provider snapshot restore completed",
            expected=True,
        ),
        Evidence(
            evidence_type="rollback_postcondition",
            source=contract.preflight_tool,
            target=canonical_json(read_args),
            observed_at=utc_now(),
            value=after,
            passed=restored,
            predicate="fresh normalized configuration exactly matches approved preflight",
            expected=before,
        ),
    )
    if not restored:
        raise RuntimeError("snapshot restore completed but exact preflight state was not restored")
    return CompensationResult(rollback_result, evidence)


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


@REGISTRY.register("fabric-access-vlan-snapshot-v1")
async def _restore_fabric_access_vlan(context: CompensationContext) -> CompensationResult:
    """Restore the provider-owned bridge/PVID snapshot and prove exact equality."""
    plan, contract = context.plan, context.contract
    if not contract.rollback_tool or not contract.preflight_tool or not plan.preflight:
        raise RuntimeError("fabric access VLAN compensation contract is incomplete")
    restore = context.backend.callables.get(contract.rollback_tool)
    preflight = context.backend.callables.get(contract.preflight_tool)
    if restore is None or preflight is None:
        raise RuntimeError("fabric provider does not expose reviewed snapshot restoration")
    rollback_args = project_arguments(plan.arguments, contract.rollback_fields)
    rollback_result = render(await asyncio.wait_for(
        restore(rollback_args), timeout=context.timeout_seconds,
    ))
    if failed_output(rollback_result):
        raise RuntimeError(rollback_result)
    read_args = project_arguments(plan.arguments, contract.preflight_fields)
    read_result = render(await preflight(read_args))
    after = typed_evidence(contract.preflight_tool, read_result)
    before = plan.preflight[0].value
    restored = isinstance(before, dict) and same_snapshot(before, after)
    evidence = (
        Evidence(
            evidence_type="rollback",
            source=contract.rollback_tool,
            target=canonical_json(rollback_args),
            observed_at=utc_now(),
            value={
                "digest": sha256_json(rollback_result),
                "bytes": len(rollback_result),
                "facts": {"snapshot_restore_completed": True},
            },
            passed=True,
            predicate="reviewed provider access-port snapshot restore completed",
            expected=True,
        ),
        Evidence(
            evidence_type="rollback_postcondition",
            source=contract.preflight_tool,
            target=canonical_json(read_args),
            observed_at=utc_now(),
            value=after,
            passed=restored,
            predicate="fresh bridge/PVID read exactly matches approved preflight",
            expected=before,
        ),
    )
    if not restored:
        raise RuntimeError("access VLAN restore completed but exact preflight state was not restored")
    return CompensationResult(rollback_result, evidence)


@REGISTRY.register("service-entitlement-snapshot-v1")
async def _restore_service_entitlement(context: CompensationContext) -> CompensationResult:
    """Restore exact desired roles through the trusted Service MCP compensator."""
    plan, contract = context.plan, context.contract
    if not contract.rollback_tool or not contract.preflight_tool or not plan.preflight:
        raise RuntimeError("service entitlement compensation contract is incomplete")
    restore = context.backend.callables.get(contract.rollback_tool)
    read = context.backend.callables.get(contract.preflight_tool)
    if restore is None or read is None:
        raise RuntimeError("trusted Service MCP restore/read tools are unavailable")
    before = plan.preflight[0].value
    if not isinstance(before, dict) or not isinstance(before.get("concurrency_token"), int):
        raise RuntimeError("approved entitlement snapshot has no concurrency token")
    before_facts = before.get("facts") or {}
    rollback_args = {
        "user_id": plan.arguments["user_id"],
        "app_id": plan.arguments["app_id"],
        "roles": list(before_facts.get("roles") or []),
        "change_id": plan.arguments["change_id"],
        "expected_revision": before["concurrency_token"] + 1,
        "reason": f"automatic rollback of effect-runtime plan {plan.plan_id}",
        "correlation_id": plan.plan_id,
    }
    rollback_result = render(await asyncio.wait_for(
        restore(rollback_args), timeout=context.timeout_seconds,
    ))
    if failed_output(rollback_result):
        raise RuntimeError(rollback_result)
    read_args = project_arguments(plan.arguments, contract.preflight_fields)
    read_result = render(await read(read_args))
    after = typed_evidence(contract.preflight_tool, read_result)
    restored = same_snapshot(before, after)
    evidence = (
        Evidence(
            evidence_type="rollback", source=contract.rollback_tool,
            target=canonical_json(read_args), observed_at=utc_now(),
            value={
                "digest": sha256_json(rollback_result), "bytes": len(rollback_result),
                "facts": {"service_snapshot_restore_completed": True},
            },
            passed=True,
            predicate="trusted Service MCP accepted the exact snapshot restoration",
            expected=True,
        ),
        Evidence(
            evidence_type="rollback_postcondition", source=contract.preflight_tool,
            target=canonical_json(read_args), observed_at=utc_now(), value=after,
            passed=restored,
            predicate="fresh Service MCP read exactly matches approved role facts",
            expected=before,
        ),
    )
    if not restored:
        raise RuntimeError("Service MCP restore completed but entitlement facts were not restored")
    return CompensationResult(rollback_result, evidence)


@REGISTRY.register("service-platform-snapshot-v1")
async def _restore_platform_service(context: CompensationContext) -> CompensationResult:
    """Restore the exact approved service version and health snapshot."""
    plan, contract = context.plan, context.contract
    if not contract.rollback_tool or not contract.preflight_tool or not plan.preflight:
        raise RuntimeError("platform service compensation contract is incomplete")
    restore = context.backend.callables.get(contract.rollback_tool)
    read = context.backend.callables.get(contract.preflight_tool)
    if restore is None or read is None:
        raise RuntimeError("trusted platform MCP restore/read tools are unavailable")
    before = plan.preflight[0].value
    if not isinstance(before, dict) or not isinstance(before.get("concurrency_token"), int):
        raise RuntimeError("approved service snapshot has no concurrency token")
    before_facts = before.get("facts") or {}
    rollback_args = {
        "service": plan.arguments["service"],
        "environment": plan.arguments["environment"],
        "version": before_facts.get("version"),
        "change_id": plan.arguments["change_id"],
        "expected_revision": before["concurrency_token"] + 1,
        "reason": f"automatic rollback of effect-runtime plan {plan.plan_id}",
        "correlation_id": plan.plan_id,
    }
    if not rollback_args["version"]:
        raise RuntimeError("approved platform snapshot has no version")
    rollback_result = render(await asyncio.wait_for(
        restore(rollback_args), timeout=context.timeout_seconds,
    ))
    if failed_output(rollback_result):
        raise RuntimeError(rollback_result)
    read_args = project_arguments(plan.arguments, contract.preflight_fields)
    read_result = render(await read(read_args))
    after = typed_evidence(contract.preflight_tool, read_result)
    restored = same_snapshot(before, after)
    evidence = (
        Evidence(
            evidence_type="rollback", source=contract.rollback_tool,
            target=canonical_json(read_args), observed_at=utc_now(),
            value={
                "digest": sha256_json(rollback_result), "bytes": len(rollback_result),
                "facts": {"service_snapshot_restore_completed": True},
            },
            passed=True,
            predicate="trusted platform MCP accepted the exact snapshot restoration",
            expected=True,
        ),
        Evidence(
            evidence_type="rollback_postcondition", source=contract.preflight_tool,
            target=canonical_json(read_args), observed_at=utc_now(), value=after,
            passed=restored,
            predicate="fresh platform MCP read exactly matches approved service facts",
            expected=before,
        ),
    )
    if not restored:
        raise RuntimeError("platform MCP restore completed but service facts were not restored")
    return CompensationResult(rollback_result, evidence)


@REGISTRY.register("network-app-enforcement-snapshot-v1")
async def _restore_network_app_enforcement(context: CompensationContext) -> CompensationResult:
    """Restore the exact preflight network enforcement verdict."""
    plan, contract = context.plan, context.contract
    if not contract.rollback_tool or not contract.preflight_tool or not plan.preflight:
        raise RuntimeError("network application enforcement compensation is incomplete")
    restore = context.backend.callables.get(contract.rollback_tool)
    read = context.backend.callables.get(contract.preflight_tool)
    if restore is None or read is None:
        raise RuntimeError("network enforcement restore/read tools are unavailable")
    before = plan.preflight[0].value
    facts = before.get("facts") if isinstance(before, dict) else None
    if not isinstance(facts, dict) or not isinstance(facts.get("allowed"), bool):
        raise RuntimeError("approved network enforcement snapshot has no boolean verdict")
    read_args = project_arguments(plan.arguments, contract.preflight_fields)
    rollback_args = {
        **read_args,
        "allowed": facts["allowed"],
        "change_id": plan.arguments["change_id"],
        "reason": f"automatic rollback of effect-runtime plan {plan.plan_id}",
    }
    rollback_result = render(await asyncio.wait_for(
        restore(rollback_args), timeout=context.timeout_seconds,
    ))
    if failed_output(rollback_result):
        raise RuntimeError(rollback_result)
    read_result = render(await read(read_args))
    after = typed_evidence(contract.preflight_tool, read_result)
    restored = same_snapshot(before, after)
    evidence = (
        Evidence(
            evidence_type="rollback", source=contract.rollback_tool,
            target=canonical_json(read_args), observed_at=utc_now(),
            value={
                "digest": sha256_json(rollback_result), "bytes": len(rollback_result),
                "facts": {"network_enforcement_restore_completed": True},
            }, passed=True,
            predicate="manifest-bound provider accepted snapshot restoration", expected=True,
        ),
        Evidence(
            evidence_type="rollback_postcondition", source=contract.preflight_tool,
            target=canonical_json(read_args), observed_at=utc_now(), value=after,
            passed=restored,
            predicate="fresh network enforcement read equals approved preflight facts",
            expected=before,
        ),
    )
    if not restored:
        raise RuntimeError("network enforcement restore completed but facts were not restored")
    return CompensationResult(rollback_result, evidence)
