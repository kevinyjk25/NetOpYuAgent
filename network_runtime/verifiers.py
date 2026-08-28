"""Versioned verifier registry for Network Runtime provider contracts."""

from __future__ import annotations

import json
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
    evidence_items = [evidence]
    probe_id = plan.arguments.get("verification_probe_id")
    if probe_id:
        probe_tool = context.backend.callables.get("lab_probe")
        if probe_tool is None:
            return VerificationResult(
                tuple(evidence_items), False,
                error="verification probe requested but lab_probe is unavailable",
            )
        probe_output = render(await probe_tool({"probe_id": str(probe_id)}))
        try:
            probe_value = json.loads(probe_output)
        except json.JSONDecodeError:
            probe_value = {"ok": False, "invalid_json": True}
        probe_passed = (
            probe_value.get("ok") is True
            and probe_value.get("probe_id") == str(probe_id)
            and int(probe_value.get("transmitted", 0)) > 0
            and probe_value.get("received") == probe_value.get("transmitted")
        )
        evidence_items.append(Evidence(
            evidence_type="traffic_postcondition",
            source="lab_probe",
            target=str(probe_id),
            observed_at=utc_now(),
            value=probe_value,
            passed=probe_passed,
            predicate="predeclared independent traffic probe has zero packet loss",
            expected=True,
        ))
        passed = passed and probe_passed
    return VerificationResult(
        tuple(evidence_items), passed,
        error=None if passed else "running config or independent traffic probe did not prove the change",
    )


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


@REGISTRY.register("fabric-access-vlan")
async def _fabric_access_vlan(context: VerificationContext) -> VerificationResult:
    """Prove the real bridge/PVID change and optional independent traffic result."""
    plan = context.plan
    output = render(await context.backend.callables["lab_get_access_vlan"]({
        "device_id": plan.arguments["device_id"],
        "interface": plan.arguments["interface"],
    }))
    value = typed_evidence("lab_get_access_vlan", output)
    facts = value["facts"]
    expected = {
        "ok": True,
        "device_id": plan.arguments["device_id"],
        "interface": plan.arguments["interface"],
        "current_vlan": plan.arguments["vlan_id"],
    }
    vlan_passed = all(facts.get(key) == item for key, item in expected.items())
    evidence_items = [Evidence(
        evidence_type="postcondition",
        source="lab_get_access_vlan",
        target=f"{plan.arguments['device_id']}:{plan.arguments['interface']}",
        observed_at=utc_now(),
        value=value,
        passed=vlan_passed,
        predicate="fresh Linux bridge read proves the approved PVID and bridge membership",
        expected=expected,
    )]
    passed = vlan_passed
    probe_id = plan.arguments.get("verification_probe_id")
    if probe_id:
        probe = context.backend.callables.get("lab_probe")
        if probe is None:
            return VerificationResult(
                tuple(evidence_items), False,
                error="verification probe requested but lab_probe is unavailable",
            )
        probe_output = render(await probe({"probe_id": str(probe_id)}))
        try:
            probe_value = json.loads(probe_output)
        except json.JSONDecodeError:
            probe_value = {"ok": False, "invalid_json": True}
        probe_passed = (
            isinstance(probe_value, dict)
            and probe_value.get("ok") is True
            and probe_value.get("probe_id") == str(probe_id)
            and int(probe_value.get("transmitted", 0)) > 0
            and probe_value.get("received") == probe_value.get("transmitted")
        )
        evidence_items.append(Evidence(
            evidence_type="traffic_postcondition",
            source="lab_probe",
            target=str(probe_id),
            observed_at=utc_now(),
            value=probe_value,
            passed=probe_passed,
            predicate="predeclared independent traffic probe has zero packet loss",
            expected=True,
        ))
        passed = passed and probe_passed
    return VerificationResult(
        tuple(evidence_items), passed,
        error=None if passed else "access VLAN state or independent traffic probe failed",
    )


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


@REGISTRY.register("service-entitlement-granted")
@REGISTRY.register("service-entitlement-revoked")
async def _service_entitlement(context: VerificationContext) -> VerificationResult:
    plan = context.plan
    output = render(await context.backend.callables["access_policy_get_entitlement"]({
        "user_id": plan.arguments["user_id"],
        "app_id": plan.arguments["app_id"],
    }))
    value = typed_evidence("access_policy_get_entitlement", output)
    facts = value["facts"]
    if context.contract.verifier.endswith("granted"):
        expected = {"allowed": True, "role": plan.arguments["role"]}
        passed = facts.get("allowed") is True and plan.arguments["role"] in facts.get("roles", [])
    else:
        expected = {"allowed": False, "roles": []}
        passed = facts.get("allowed") is False and facts.get("roles") == []
    evidence = Evidence(
        evidence_type="postcondition",
        source="access_policy_get_entitlement",
        target=f"{plan.arguments['user_id']}:{plan.arguments['app_id']}",
        observed_at=utc_now(),
        value=value,
        passed=passed,
        predicate="fresh Service MCP read equals the reviewed entitlement postcondition",
        expected=expected,
    )
    return _result(evidence, passed, None if passed else f"postcondition {expected!r} not observed")


@REGISTRY.register("service-platform-healthy")
async def _service_platform_healthy(context: VerificationContext) -> VerificationResult:
    plan = context.plan
    output = render(await context.backend.callables["platform_get_service_health"]({
        "service": plan.arguments["service"],
        "environment": plan.arguments["environment"],
    }))
    value = typed_evidence("platform_get_service_health", output)
    facts = value["facts"]
    expected = {
        "status": "healthy",
        "replicas_ready": 3,
        "replicas_desired": 3,
    }
    if plan.tool_name == "platform_rollback_service":
        expected["version"] = plan.arguments["version"]
    passed = all(facts.get(key) == expected_value for key, expected_value in expected.items())
    evidence = Evidence(
        evidence_type="postcondition",
        source="platform_get_service_health",
        target=f"{plan.arguments['service']}:{plan.arguments['environment']}",
        observed_at=utc_now(),
        value=value,
        passed=passed,
        predicate="fresh Service MCP health read equals the reviewed platform postcondition",
        expected=expected,
    )
    return _result(evidence, passed, None if passed else f"postcondition {expected!r} not observed")


@REGISTRY.register("network-app-enforcement-granted")
@REGISTRY.register("network-app-enforcement-revoked")
async def _network_app_enforcement(context: VerificationContext) -> VerificationResult:
    plan = context.plan
    output = render(await context.backend.callables["network_get_app_enforcement"]({
        "user_id": plan.arguments["user_id"], "app_id": plan.arguments["app_id"],
    }))
    value = typed_evidence("network_get_app_enforcement", output)
    expected = {"allowed": context.contract.verifier.endswith("granted")}
    passed = (
        value["facts"].get("ok") is True
        and value["facts"].get("allowed") == expected["allowed"]
        and value["facts"].get("implementation") == "server-source-blackhole-route"
    )
    evidence = Evidence(
        evidence_type="postcondition", source="network_get_app_enforcement",
        target=f"{plan.arguments['user_id']}:{plan.arguments['app_id']}",
        observed_at=utc_now(), value=value, passed=passed,
        predicate="fresh Containerlab enforcement read equals the approved network state",
        expected=expected,
    )
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
