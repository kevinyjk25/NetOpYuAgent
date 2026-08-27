"""Versioned verifier registry for Network Runtime provider contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from dsh_adapter.backend import BackendSession

from .contracts import Evidence, PreparedPlan, sha256_json, utc_now
from .evidence import failed_output, render, typed_evidence
from .policies import ToolContract


@dataclass(frozen=True)
class VerificationResult:
    evidence: tuple[Evidence, ...]
    passed: bool
    internal_rollback: bool = False
    error: str | None = None


@dataclass(frozen=True)
class VerificationContext:
    backend: BackendSession
    plan: PreparedPlan
    contract: ToolContract
    result: str


Verifier = Callable[[VerificationContext], Awaitable[VerificationResult]]


class VerifierRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Verifier] = {}

    def register(self, contract_id: str) -> Callable[[Verifier], Verifier]:
        def decorator(handler: Verifier) -> Verifier:
            if contract_id in self._handlers:
                raise RuntimeError(f"duplicate verifier contract {contract_id}")
            self._handlers[contract_id] = handler
            return handler
        return decorator

    async def verify(self, context: VerificationContext) -> VerificationResult:
        handler = self._handlers.get(context.contract.verifier)
        if handler is None:
            evidence = (_adapter_evidence(
                context.plan, context.result, False, "registered verifier contract exists",
            ),)
            return VerificationResult(
                evidence, False, error=f"unsupported verifier {context.contract.verifier}",
            )
        return await handler(context)

    def contract_ids(self) -> frozenset[str]:
        return frozenset(self._handlers)


REGISTRY = VerifierRegistry()


def _adapter_evidence(plan: PreparedPlan, value: str, passed: bool, predicate: str) -> Evidence:
    return Evidence(
        evidence_type="adapter_result", source=plan.tool_name,
        target=",".join(plan.targets), observed_at=utc_now(),
        value={
            "digest": sha256_json(value), "bytes": len(value),
            "facts": {"predicate_satisfied": passed},
        },
        passed=passed, predicate=predicate, expected=True,
    )


def _read_evidence(
    source: str, target: str, output: str, expected: object, passed: bool,
) -> Evidence:
    return Evidence(
        evidence_type="postcondition", source=source, target=str(target),
        observed_at=utc_now(), value=typed_evidence(source, output), passed=passed,
        predicate="fresh read yields the expected typed postcondition", expected=expected,
    )


def _result(evidence: Evidence, passed: bool, error: str | None = None) -> VerificationResult:
    return VerificationResult((evidence,), passed, error=error)


async def verify_operation(
    backend: BackendSession, plan: PreparedPlan, contract: ToolContract, result: str | None,
) -> VerificationResult:
    rendered = result or ""
    if "[CRITICAL]" in rendered:
        return _result(
            _adapter_evidence(plan, rendered, False, "tool must not report CRITICAL"),
            False, "tool reported failed verification and rollback",
        )
    if "[ROLLED BACK]" in rendered:
        return VerificationResult(
            (_adapter_evidence(plan, rendered, True, "tool reports rollback completed"),),
            False, internal_rollback=True, error="tool performed automatic rollback",
        )
    return await REGISTRY.verify(VerificationContext(backend, plan, contract, rendered))


@REGISTRY.register("device-config")
async def _device_config(context: VerificationContext) -> VerificationResult:
    plan, rendered = context.plan, context.result
    if plan.arguments.get("dry_run") is True:
        passed = "Validation complete — no changes written" in rendered and "Errors: 0" in rendered
        return _result(
            _adapter_evidence(plan, rendered, passed, "dry-run reports no write and zero errors"),
            passed, None if passed else "dry-run validation did not pass",
        )
    verify_text = render(await context.backend.callables["get_device_config"]({
        "device_id": plan.arguments["device_id"],
        **({"section": plan.arguments["section"]} if plan.arguments.get("section") else {}),
    }))
    passed = _config_matches(plan.arguments, verify_text)
    value = typed_evidence("get_device_config", verify_text)
    value["facts"]["requested_config_matched"] = passed
    evidence = Evidence(
        evidence_type="postcondition", source="get_device_config",
        target=str(plan.arguments["device_id"]), observed_at=utc_now(),
        value=value, passed=passed,
        predicate="running config reflects every compiled change", expected=True,
    )
    return _result(evidence, passed, None if passed else "running config does not prove requested change")


@REGISTRY.register("lan-access-granted")
@REGISTRY.register("lan-access-revoked")
async def _lan_access(context: VerificationContext) -> VerificationResult:
    output = render(await context.backend.callables["get_user_access"]({
        "user_id": context.plan.arguments["user_id"],
    }))
    expected = {"admitted": context.contract.verifier.endswith("granted")}
    passed = typed_evidence("get_user_access", output)["facts"].get("admitted") == expected["admitted"]
    evidence = _read_evidence("get_user_access", context.plan.arguments["user_id"], output, expected, passed)
    return _result(evidence, passed, None if passed else f"postcondition {expected!r} not observed")


@REGISTRY.register("dc-access-granted")
@REGISTRY.register("dc-access-revoked")
async def _dc_access(context: VerificationContext) -> VerificationResult:
    output = render(await context.backend.callables["dc_check_user_app_access"]({
        "user_id": context.plan.arguments["user_id"],
        "app_id": context.plan.arguments["app_id"],
    }))
    expected = {"allowed": context.contract.verifier.endswith("granted")}
    passed = typed_evidence("dc_check_user_app_access", output)["facts"].get("allowed") == expected["allowed"]
    evidence = _read_evidence(
        "dc_check_user_app_access", context.plan.arguments["app_id"], output, expected, passed,
    )
    return _result(evidence, passed, None if passed else f"postcondition {expected!r} not observed")


@REGISTRY.register("dc-config")
async def _dc_config(context: VerificationContext) -> VerificationResult:
    plan = context.plan
    config_output = render(await context.backend.callables["dc_get_applied_config"]({
        "node": plan.arguments["node"],
    }))
    bgp_output = render(await context.backend.callables["dc_bgp_evpn_status"]({
        "node": plan.arguments["node"],
    }))
    config_value = typed_evidence("dc_get_applied_config", config_output)
    bgp_value = typed_evidence("dc_bgp_evpn_status", bgp_output)
    applied = config_value["facts"].get("applied", [])
    config_passed = all(line in applied for line in plan.arguments["config_lines"])
    bgp_passed = (
        bgp_value["facts"].get("readable") is True
        and bgp_value["facts"].get("flapping_neighbors") == 0
    )
    evidence = (
        Evidence(
            evidence_type="postcondition",
            source="dc_get_applied_config",
            target=str(plan.arguments["node"]),
            observed_at=utc_now(),
            value=config_value,
            passed=config_passed,
            predicate="fresh config state contains every compiled line",
            expected=list(plan.arguments["config_lines"]),
        ),
        Evidence(
            evidence_type="postcondition",
            source="dc_bgp_evpn_status",
            target=str(plan.arguments["node"]),
            observed_at=utc_now(),
            value=bgp_value,
            passed=bgp_passed,
            predicate="BGP EVPN is readable with zero flapping neighbors",
            expected={"readable": True, "flapping_neighbors": 0},
        ),
    )
    passed = config_passed and bgp_passed
    return VerificationResult(evidence, passed, error=None if passed else "DC config or BGP postcondition failed")


@REGISTRY.register("wan-failover")
async def _wan_failover(context: VerificationContext) -> VerificationResult:
    plan = context.plan
    output = render(await context.backend.callables["wan_tunnel_status"]({}))
    observed = typed_evidence("wan_tunnel_status", output)["facts"].get("tunnels", {}).get(
        plan.arguments["tunnel"], {},
    )
    expected = {"transport": plan.arguments["to_transport"], "state": "up"}
    passed = all(observed.get(key) == value for key, value in expected.items())
    evidence = _read_evidence("wan_tunnel_status", plan.arguments["tunnel"], output, expected, passed)
    return _result(evidence, passed, None if passed else f"postcondition {expected!r} not observed")


@REGISTRY.register("service-health")
async def _service_health(context: VerificationContext) -> VerificationResult:
    plan, rendered = context.plan, context.result
    output = render(await context.backend.callables["service_health"]({
        "service": plan.arguments["service"], "environment": plan.arguments["environment"],
    }))
    expected = {"status": "healthy", "replicas_ready": 3, "replicas_desired": 3}
    result_ok = "Health check: PASS" in rendered and (
        "Rollout complete" in rendered or "Rollback complete" in rendered
    )
    passed = result_ok and typed_evidence("service_health", output)["facts"] == expected
    evidence = _read_evidence("service_health", plan.arguments["service"], output, expected, passed)
    return _result(evidence, passed, None if passed else f"postcondition {expected!r} not observed")


@REGISTRY.register("mock-state")
async def _mock_state(context: VerificationContext) -> VerificationResult:
    plan, rendered = context.plan, context.result
    resource_id = str(
        plan.arguments.get("deploy_id") or plan.arguments.get("node_id")
        or plan.arguments.get("resource_id") or ""
    )
    status_text = render(await context.backend.callables["mock_operation_status"]({
        "operation": plan.tool_name, "resource_id": resource_id,
    }))
    value = typed_evidence("mock_operation_status", status_text)
    state = value["facts"].get("operations", {}).get(plan.tool_name, {})
    expected_by_tool = {
        "rollback_deploy": {"rolled_back": True, "services_healthy": True},
        "drain_node": {"drained": True, "schedulable": False, "pending": 0, "failed": 0},
        "failover": {"primary": plan.arguments.get("target"), "healthy": True},
        "delete_resource": {"exists": False},
    }
    expected = expected_by_tool.get(plan.tool_name, {})
    passed = (
        not failed_output(rendered) and context.backend.mode == "mock" and bool(expected)
        and all(state.get(key) == expected_value for key, expected_value in expected.items())
    )
    evidence = Evidence(
        evidence_type="postcondition",
        source="mock_operation_status",
        target=resource_id,
        observed_at=utc_now(),
        value=value,
        passed=passed,
        predicate="independent simulator state equals reviewed typed postcondition",
        expected=expected,
    )
    return _result(evidence, passed, None if passed else "simulator state did not prove requested effect")


def _config_matches(arguments: dict[str, object], output: str) -> bool:
    lines = arguments.get("config_lines") or []
    if not lines and isinstance(arguments.get("config_text"), str):
        lines = [line for line in str(arguments["config_text"]).splitlines() if line.strip()]
    changes = arguments.get("changes") or {}
    section = str(arguments.get("section") or "").lower()
    expected: list[str] = []
    for line in lines if isinstance(lines, list) else []:
        lowered = str(line).strip().lower()
        timeout = re.search(r"radius-server.*timeout\s+(\d+)", lowered)
        if timeout:
            expected.append(f"timeout {timeout.group(1)}")
        elif lowered.startswith("no ntp server "):
            if lowered.removeprefix("no ") in output.lower():
                return False
        elif lowered.startswith("ntp server "):
            expected.append(lowered)
        elif "access-list" in lowered or "access-group" in lowered:
            expected.append("access-list")
        elif lowered:
            expected.append(lowered)
    if isinstance(changes, dict):
        if section in {"radius", "aaa"} and "timeout" in changes:
            expected.append(f"timeout {changes['timeout']}")
        if section in {"ntp", "time"}:
            expected.extend(f"ntp server {item}" for item in changes.get("servers", []))
    return bool(expected) and all(item.lower() in output.lower() for item in expected)
