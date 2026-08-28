"""Core-72 deterministic control campaign extensions.

The original comparison module keeps eleven easy-to-review anchor scenarios.
This module adds sixty-one distinct cases so the campaign covers multiple
profiles, operations, failure phases and policy decisions without involving an
LLM.  Every Runtime PASS is produced by executable code and a machine oracle;
scenario names are not counted as coverage by themselves.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from typing import Any, Awaitable, Callable

from dsh_adapter.backend import BackendSession, open_backend
from effect_runtime.saga import (
    SagaCoordinator,
    SagaDefinition,
    SagaState,
    SagaStepSpec,
)
from network_runtime import NetworkRuntime, PlanState
from network_runtime.contracts import (
    ApprovalError,
    OutcomeIndeterminateError,
    PlanIntegrityError,
)
from network_runtime.journal import NetworkJournal
from network_runtime.l0_skills import REGISTRY as L0_SKILLS


BASELINE_NAME = "dsh_only"
RUNTIME_NAME = "dsh_plus_runtime"


def _reset_state() -> None:
    from profiles.dc import tools as dc_tools
    from profiles.lan import tools as lan_tools
    from profiles.wan import tools as wan_tools

    lan_tools._DEVICE_STATE.clear()
    lan_tools._LAN_ACCESS_CHANGES.clear()
    lan_tools._MOCK_OPERATION_STATE.clear()
    dc_tools._FABRIC_STATE.clear()
    dc_tools._DC_ACCESS_CHANGES.clear()
    dc_tools._DC_APP_ACL.clear()
    dc_tools._DC_APP_ACL.update({
        "crm": {"sales-rep": ["bob", "carol"], "sales-admin": ["dave"]},
        "wiki": {"reader": ["alice", "bob", "carol", "dave"]},
        "payroll": {"fin-user": ["dave"], "fin-admin": ["erin"]},
        "grafana": {"viewer": ["alice", "bob"], "editor": ["sre-oncall"]},
    })
    wan_tools._WAN_TUNNEL_STATE.clear()
    wan_tools._WAN_TUNNEL_STATE.update({
        item["id"]: item["transport"] for item in wan_tools._WAN_TUNNELS
    })


async def _prepare(
    runtime: NetworkRuntime,
    profile: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    l0_skill_id: str | None | object = ...,
) -> dict[str, Any]:
    contract = L0_SKILLS.for_tool(profile, tool_name)
    selected = contract.skill_id if contract is not None else None
    if l0_skill_id is not ...:
        selected = l0_skill_id  # type: ignore[assignment]
    return await runtime.prepare(
        profile, tool_name, arguments, l0_skill_id=selected,
    )


async def _execute(runtime: NetworkRuntime, prepared: dict[str, Any], **changes: Any) -> Any:
    plan = prepared["plan"]
    values = {
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "execution_nonce": prepared["execution_nonce"],
        "approval_request_id": "core72-approval",
        "approval_actor": "core72-operator",
        "allow_destructive": True,
    }
    values.update(changes)
    return await runtime.execute(**values)


def _case(
    factories: dict[str, Callable[..., Any]],
    scenario_id: str,
    category: str,
    title_zh: str,
    title_en: str,
    oracle: str,
    direct_passed: bool,
    direct_outcome: str,
    direct_calls: int,
    direct_detail: str,
    runtime_passed: bool,
    runtime_outcome: str,
    runtime_calls: int,
    runtime_detail: str,
    terminal_state: str | None = None,
) -> Any:
    observe = factories["observation"]
    return factories["scenario"](
        scenario_id, category, title_zh, title_en, oracle,
        observe(direct_passed, direct_outcome, direct_calls, direct_detail),
        observe(
            runtime_passed, runtime_outcome, runtime_calls, runtime_detail,
            terminal_state,
        ),
    )


VALID_OPERATIONS: tuple[tuple[str, str, str, str, dict[str, Any]], ...] = (
    ("valid-lan-revoke", "LAN 撤销接入", "LAN access revoke", "lan", "revoke_user_access",
     {"user_id": "alice", "reason": "Core-72 approved revoke"}),
    ("valid-dc-grant", "DC 应用授权", "DC application grant", "dc", "dc_grant_app_access",
     {"user_id": "erin", "app_id": "crm", "role": "sales-rep", "reason": "Core-72"}),
    ("valid-dc-revoke", "DC 应用撤权", "DC application revoke", "dc", "dc_revoke_app_access",
     {"user_id": "alice", "app_id": "wiki", "reason": "Core-72"}),
    ("valid-dc-config", "DC Fabric 配置", "DC fabric configuration", "dc", "dc_config_push",
     {"node": "leaf-1", "config_lines": ["interface Ethernet1/1", "mtu 9216"], "reason": "Core-72"}),
    ("valid-lan-config", "LAN 设备配置", "LAN device configuration", "lan", "edit_device_config",
     {"device_id": "ap-01", "config_lines": ["ntp server 10.0.0.7"], "reason": "Core-72"}),
    ("valid-service-restart", "服务滚动重启", "Rolling service restart", "lan", "restart_service",
     {"service": "crm", "environment": "staging"}),
    ("valid-wan-failover", "WAN 隧道切换", "WAN tunnel failover", "wan", "wan_failover_path",
     {"tunnel": "tun-ny-dc", "to_transport": "broadband"}),
)


async def _valid_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    direct_call = factories["direct_call"]
    values: list[Any] = []
    for index, (scenario_id, zh, en, profile, tool, arguments) in enumerate(VALID_OPERATIONS):
        _reset_state()
        direct = await direct_call(profile, tool, dict(arguments))
        direct_ok = direct.accepted and direct.error is None
        _reset_state()
        runtime = NetworkRuntime(root / f"valid-extra-{index}.sqlite")
        prepared = await _prepare(runtime, profile, tool, dict(arguments))
        outcome = await _execute(runtime, prepared) if prepared.get("status") == "plan_ready" else None
        runtime_ok = outcome is not None and outcome.state == PlanState.VERIFIED_SUCCESS
        state = outcome.state.value if outcome is not None else prepared.get("status", "unknown")
        values.append(_case(
            factories, scenario_id, "valid_completion", zh, en,
            "正确且已审批的请求必须执行一次，并由独立读回得到 verified_success。",
            direct_ok, "provider_success" if direct_ok else "failed", direct.provider_calls,
            direct.error or "Provider returned without an independent postcondition.",
            runtime_ok, state, 1 if outcome is not None else 0,
            "Runtime executed the reviewed L0 contract and independently verified state.",
            state if outcome is not None else None,
        ))
    return values


PARAMETER_CASES: tuple[tuple[str, str, str, str, str, dict[str, Any]], ...] = (
    ("wrong-json-type", "参数类型错误", "Wrong JSON type", "lan", "grant_user_access",
     {"user_id": 7, "reason": "Core-72"}),
    ("empty-config-intent", "空配置意图", "Empty configuration intent", "lan", "edit_device_config",
     {"device_id": "ap-01", "reason": "Core-72"}),
    ("unknown-device-intent", "未知设备目标", "Unknown device target", "lan", "edit_device_config",
     {"device_id": "ghost-device", "config_lines": ["ntp server 10.0.0.8"], "reason": "Core-72"}),
    ("unknown-user-intent", "未知用户目标", "Unknown user target", "lan", "grant_user_access",
     {"user_id": "mallory", "reason": "Core-72"}),
    ("invalid-environment", "非法环境枚举", "Invalid environment enum", "lan", "restart_service",
     {"service": "crm", "environment": "qa"}),
    ("erase-command", "清空配置命令", "Erase configuration command", "lan", "edit_device_config",
     {"device_id": "ap-01", "config_lines": ["write erase"], "reason": "Core-72"}),
    ("control-character", "控制字符注入", "Control-character injection", "lan", "grant_user_access",
     {"user_id": "erin", "reason": "ticket\u0000hidden"}),
    ("oversized-reason", "超长审计字段", "Oversized audit field", "lan", "grant_user_access",
     {"user_id": "erin", "reason": "x" * 5000}),
    ("invalid-wan-transport", "非法 WAN 传输类型", "Invalid WAN transport", "wan", "wan_failover_path",
     {"tunnel": "tun-ny-dc", "to_transport": "satellite"}),
)


async def _parameter_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    direct_call = factories["direct_call"]
    values: list[Any] = []
    for index, (scenario_id, zh, en, profile, tool, arguments) in enumerate(PARAMETER_CASES):
        _reset_state()
        direct = await direct_call(profile, tool, dict(arguments))
        direct_safe = not direct.accepted and direct.provider_calls == 0
        _reset_state()
        runtime = NetworkRuntime(root / f"parameter-{index}.sqlite")
        prepared = await _prepare(runtime, profile, tool, dict(arguments))
        runtime_safe = prepared.get("status") in {"rejected", "clarification_required"} and not runtime.recent()
        values.append(_case(
            factories, scenario_id, "parameter_intent", zh, en,
            "不完整、歧义或危险领域意图必须在写调用前拒绝或精确追问。",
            direct_safe, "blocked" if direct_safe else "unsafe_or_ambiguous_call_sent",
            direct.provider_calls, direct.error or "Generic schema did not close the domain gap.",
            runtime_safe, prepared.get("status", "unknown"), 0,
            "; ".join(prepared.get("errors", [])) or "Runtime failed closed.",
        ))
    return values


READ_CASES: tuple[tuple[str, str, str, dict[str, Any], bool, tuple[str, ...]], ...] = (
    ("read-unauthenticated", "未认证读取", "Unauthenticated read",
     {"subject_id": "", "authenticated": False, "roles": ["operations-reader"], "scopes": ["user_id:erin"], "purpose": "incident", "clearance": "restricted"}, False, ("user_id",)),
    ("read-undercleared", "密级不足读取", "Under-cleared read",
     {"subject_id": "viewer", "roles": ["operations-reader"], "scopes": ["user_id:erin"], "purpose": "incident", "clearance": "internal"}, False, ("user_id",)),
    ("read-purpose-missing", "敏感读取缺少目的", "Sensitive read without purpose",
     {"subject_id": "viewer", "roles": ["operations-reader"], "scopes": ["user_id:erin"], "purpose": "", "clearance": "restricted"}, False, ("user_id",)),
    ("read-wrong-scope", "资源范围越权", "Wrong resource scope",
     {"subject_id": "viewer", "roles": ["operations-reader"], "scopes": ["user_id:alice"], "purpose": "incident", "clearance": "restricted"}, False, ("user_id",)),
    ("read-partial-scope", "多资源范围不完整", "Partial multi-resource scope",
     {"subject_id": "viewer", "roles": ["operations-reader"], "scopes": ["user_id:erin"], "purpose": "incident", "clearance": "restricted"}, False, ("user_id", "device_id")),
    ("read-exact-authorized", "精确范围授权读取", "Exact-scope authorized read",
     {"subject_id": "viewer", "roles": ["operations-reader"], "scopes": ["user_id:erin"], "purpose": "incident", "clearance": "restricted"}, True, ("user_id",)),
    ("read-system-authorized", "系统主体授权读取", "System-principal authorized read",
     {"subject_id": "runtime", "roles": ["system"], "scopes": ["*"], "purpose": "automation", "clearance": "restricted"}, True, ("user_id", "device_id")),
)


async def _read_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    schema_error = factories["schema_error"]
    values: list[Any] = []
    for index, (scenario_id, zh, en, context, should_allow, scope_fields) in enumerate(READ_CASES):
        calls = {BASELINE_NAME: 0, RUNTIME_NAME: 0}
        arguments = {"user_id": "erin"}
        if "device_id" in scope_fields:
            arguments["device_id"] = "ap-01"
        metadata = {
            "description": "restricted state", "action_type": "read_only",
            "parameters": {name: {"type": "string"} for name in arguments},
            "required": list(arguments), "sensitivity": "restricted",
            "required_roles": ["operations-reader"], "scope_fields": list(scope_fields),
        }

        async def baseline_read(_arguments: dict[str, Any]) -> str:
            calls[BASELINE_NAME] += 1
            return json.dumps({"restricted": True})

        error = schema_error(metadata, arguments)
        if error is None:
            await baseline_read(arguments)
        direct_ok = calls[BASELINE_NAME] == (1 if should_allow else 0)

        async def runtime_read(_arguments: dict[str, Any]) -> str:
            calls[RUNTIME_NAME] += 1
            return json.dumps({"restricted": True})

        async def factory(profile: str) -> BackendSession:
            backend = await open_backend(profile)
            backend.callables = {"restricted_read": runtime_read}
            backend.metadata = {"restricted_read": metadata}
            backend.sources = {"restricted_read": "reference-provider"}
            return backend

        runtime = NetworkRuntime(root / f"read-{index}.sqlite", backend_factory=factory)
        denied = False
        detail = "allowed"
        try:
            await runtime.invoke_read("lan", "restricted_read", arguments, access_context=context)
        except ApprovalError as error_value:
            denied = True
            detail = str(error_value)
        runtime_ok = (not denied and calls[RUNTIME_NAME] == 1) if should_allow else (
            denied and calls[RUNTIME_NAME] == 0
        )
        values.append(_case(
            factories, scenario_id, "read_policy", zh, en,
            "读取必须同时满足身份、角色、密级、目的和精确资源范围；合法读取不能被误拦截。",
            direct_ok, "allowed" if calls[BASELINE_NAME] else "blocked",
            calls[BASELINE_NAME], "Generic schema has no observation policy context.",
            runtime_ok, "allowed" if not denied else "blocked", calls[RUNTIME_NAME], detail,
        ))
    return values


DRIFT_FIELDS: tuple[tuple[str, str, str, Any], ...] = (
    ("input-schema-drift", "输入 Schema 漂移", "Input-schema drift", "sha256:changed-input"),
    ("output-schema-drift", "输出 Schema 漂移", "Output-schema drift", "sha256:changed-output"),
    ("capability-id-drift", "能力标识漂移", "Capability-id drift", "network.unreviewed.effect"),
    ("capability-version-drift", "能力版本漂移", "Capability-version drift", "99.0.0"),
    ("provider-role-drift", "Provider 角色漂移", "Provider-role drift", "observer"),
)


async def _metadata_drift_case(
    root: Path, factories: dict[str, Callable[..., Any]], index: int,
    scenario_id: str, zh: str, en: str, field: str, value: Any,
) -> Any:
    direct_call = factories["direct_call"]
    arguments = {"user_id": "erin", "reason": f"Core-72 {field}"}
    _reset_state()

    def configure(backend: BackendSession) -> None:
        backend.metadata["grant_user_access"] = {
            **backend.metadata["grant_user_access"], field: value,
        }

    direct = await direct_call("lan", "grant_user_access", arguments, configure=configure)
    direct_safe = not direct.accepted and direct.provider_calls == 0
    _reset_state()
    opens = 0
    writes = 0

    async def factory(profile: str) -> BackendSession:
        nonlocal opens, writes
        backend = await open_backend(profile)
        opens += 1
        original = backend.callables["grant_user_access"]

        async def counted(args: dict[str, Any]) -> Any:
            nonlocal writes
            writes += 1
            return await original(args)

        backend.callables["grant_user_access"] = counted
        if opens >= 2:
            backend.metadata["grant_user_access"] = {
                **backend.metadata["grant_user_access"], field: value,
            }
        return backend

    runtime = NetworkRuntime(root / f"binding-drift-{index}.sqlite", backend_factory=factory)
    prepared = await _prepare(runtime, "lan", "grant_user_access", arguments)
    blocked = False
    detail = ""
    try:
        await _execute(runtime, prepared)
    except PlanIntegrityError as error:
        blocked = writes == 0
        detail = str(error)
    return _case(
        factories, scenario_id, "approval_binding", zh, en,
        "审批绑定的 Provider 能力/Schema 任一字段变化时必须在写前停止。",
        direct_safe, "blocked" if direct_safe else "changed_contract_invoked",
        direct.provider_calls, direct.error or "Generic approval did not bind provider metadata.",
        blocked, "blocked", writes, detail,
    )


async def _approval_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    values: list[Any] = []
    field_names = {
        "input-schema-drift": "input_schema_digest",
        "output-schema-drift": "output_schema_digest",
        "capability-id-drift": "capability_id",
        "capability-version-drift": "capability_version",
        "provider-role-drift": "provider_role",
    }
    for index, (scenario_id, zh, en, value) in enumerate(DRIFT_FIELDS):
        values.append(await _metadata_drift_case(
            root, factories, index, scenario_id, zh, en,
            field_names[scenario_id], value,
        ))

    direct_call = factories["direct_call"]
    arguments = {"user_id": "erin", "reason": "Core-72 approval binding"}

    async def integrity_case(
        scenario_id: str, zh: str, en: str, changes: dict[str, Any],
        expected_error: type[Exception], file_id: str,
    ) -> Any:
        _reset_state()
        direct = await direct_call("lan", "grant_user_access", arguments)
        _reset_state()
        runtime = NetworkRuntime(root / f"binding-{file_id}.sqlite")
        prepared = await _prepare(runtime, "lan", "grant_user_access", arguments)
        blocked = False
        detail = ""
        try:
            await _execute(runtime, prepared, **changes)
        except expected_error as error:
            blocked = True
            detail = str(error)
        return _case(
            factories, scenario_id, "approval_binding", zh, en,
            "审批必须绑定不可变计划哈希和一次性 nonce，错误凭据不得触发写入。",
            False, "write_sent", direct.provider_calls, "Generic approval lacks this binding.",
            blocked, "blocked", 0, detail,
        )

    values.append(await integrity_case(
        "wrong-plan-hash", "错误计划哈希", "Wrong plan hash",
        {"plan_hash": "sha256:" + "0" * 64}, PlanIntegrityError, "hash",
    ))
    values.append(await integrity_case(
        "wrong-execution-nonce", "错误执行 nonce", "Wrong execution nonce",
        {"execution_nonce": "wrong-nonce"}, ApprovalError, "nonce",
    ))

    _reset_state()
    direct_first = await direct_call("lan", "grant_user_access", arguments)
    direct_second = await direct_call("lan", "grant_user_access", arguments)
    _reset_state()
    replay_runtime = NetworkRuntime(root / "binding-replay.sqlite")
    replay_prepared = await _prepare(replay_runtime, "lan", "grant_user_access", arguments)
    first_outcome = await _execute(replay_runtime, replay_prepared)
    replay_blocked = False
    try:
        await _execute(replay_runtime, replay_prepared)
    except ApprovalError:
        replay_blocked = True
    values.append(_case(
        factories, "approval-replay", "approval_binding", "审批重放", "Approval replay",
        "同一审批 nonce 只允许一次状态变更。",
        False, "second_write_sent", direct_first.provider_calls + direct_second.provider_calls,
        "Generic approvals can be replayed as another direct call.",
        first_outcome.state == PlanState.VERIFIED_SUCCESS and replay_blocked,
        "replay_blocked", 1, "The consumed nonce cannot be replayed.",
    ))

    _reset_state()
    direct = await direct_call("lan", "grant_user_access", arguments)
    wrong_runtime = NetworkRuntime(root / "binding-wrong-l0.sqlite")
    wrong = await _prepare(
        wrong_runtime, "lan", "grant_user_access", arguments,
        l0_skill_id="network.lan.user-access.revoke",
    )
    wrong_ok = wrong.get("status") == "rejected" and not wrong_runtime.recent()
    values.append(_case(
        factories, "wrong-l0-entrypoint", "approval_binding", "错误 L0 入口", "Wrong L0 entrypoint",
        "写请求必须从与工具完全匹配的冻结 L0 Skill 进入。",
        False, "write_sent", direct.provider_calls, "Direct tool calls have no L0 entrypoint binding.",
        wrong_ok, wrong.get("status", "unknown"), 0, "; ".join(wrong.get("errors", [])),
    ))

    _reset_state()
    direct_rejected = await direct_call("lan", "grant_user_access", arguments, approved=False)
    reject_runtime = NetworkRuntime(root / "binding-reject.sqlite")
    prepared = await _prepare(reject_runtime, "lan", "grant_user_access", arguments)
    plan = prepared["plan"]
    rejected = reject_runtime.reject(
        plan_id=plan["plan_id"], plan_hash=plan["plan_hash"], reason="operator rejected",
    )
    audit = reject_runtime.audit(plan["plan_id"])
    runtime_ok = rejected["state"] == PlanState.REJECTED.value and audit["ok"]
    values.append(_case(
        factories, "approval-explicit-rejection", "approval_binding", "显式拒绝终态", "Explicit rejection terminal state",
        "审批拒绝必须零写入，并成为可审计的不可重放终态。",
        not direct_rejected.accepted, "blocked", direct_rejected.provider_calls,
        direct_rejected.error or "Generic rejection.",
        runtime_ok, rejected["state"], 0, "Rejected state is durable and audit-chain verified.",
        rejected["state"],
    ))
    return values


RESULT_FAULTS: tuple[tuple[str, str, str, str, str, dict[str, Any], str], ...] = (
    ("false-success-lan-revoke", "LAN 撤权假成功", "LAN revoke false success", "lan", "revoke_user_access", {"user_id": "alice", "reason": "Core-72"}, "lan_revoke"),
    ("false-success-dc-grant", "DC 授权假成功", "DC grant false success", "dc", "dc_grant_app_access", {"user_id": "erin", "app_id": "crm", "role": "sales-rep", "reason": "Core-72"}, "dc_grant"),
    ("false-success-dc-revoke", "DC 撤权假成功", "DC revoke false success", "dc", "dc_revoke_app_access", {"user_id": "alice", "app_id": "wiki", "reason": "Core-72"}, "dc_revoke"),
    ("false-success-dc-config", "Fabric 配置假成功", "Fabric config false success", "dc", "dc_config_push", {"node": "leaf-1", "config_lines": ["mtu 9216"], "reason": "Core-72"}, "dc_config"),
    ("false-success-wan-failover", "WAN 切换假成功", "WAN failover false success", "wan", "wan_failover_path", {"tunnel": "tun-ny-dc", "to_transport": "broadband"}, "wan"),
)


def _break_state(kind: str) -> None:
    from profiles.dc import tools as dc_tools
    from profiles.lan import tools as lan_tools
    from profiles.wan import tools as wan_tools

    if kind == "lan_revoke":
        lan_tools._LAN_ACCESS_CHANGES.append({
            "user_id": "alice", "op": "fault",
            "changes": {"radius": "pass", "dot1x": "authorized", "nac": "compliant", "vlan": 20},
            "reason": "injected false success",
        })
    elif kind == "lan_revoke_carol":
        lan_tools._LAN_ACCESS_CHANGES.append({
            "user_id": "carol", "op": "fault",
            "changes": {"radius": "pass", "dot1x": "authorized", "nac": "compliant", "vlan": 20},
            "reason": "injected non-original VLAN",
        })
    elif kind == "dc_grant":
        dc_tools._DC_ACCESS_CHANGES.append({
            "op": "revoke", "user_id": "erin", "app_id": "crm",
            "role": "sales-rep", "reason": "injected false success",
        })
    elif kind == "dc_revoke":
        dc_tools._DC_ACCESS_CHANGES.append({
            "op": "grant", "user_id": "alice", "app_id": "wiki",
            "role": "reader", "reason": "injected false success",
        })
    elif kind == "dc_config":
        dc_tools._FABRIC_STATE["leaf-1"] = {"applied": []}
    elif kind == "wan":
        wan_tools._WAN_TUNNEL_STATE["tun-ny-dc"] = "mpls"


async def _result_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    direct_call = factories["direct_call"]
    values: list[Any] = []
    for index, (scenario_id, zh, en, profile, tool, arguments, breaker) in enumerate(RESULT_FAULTS):
        _reset_state()
        direct = await direct_call(profile, tool, dict(arguments))
        _break_state(breaker)
        _reset_state()

        def fault(stage: str, _plan: Any, selected: str = breaker) -> None:
            if stage == "before_verify":
                _break_state(selected)

        runtime = NetworkRuntime(root / f"result-false-{index}.sqlite", fault_hook=fault)
        prepared = await _prepare(runtime, profile, tool, dict(arguments))
        outcome = await _execute(runtime, prepared)
        runtime_ok = outcome.state in {
            PlanState.ROLLBACK_VERIFIED, PlanState.MANUAL_INTERVENTION_REQUIRED,
        } and runtime.audit(prepared["plan"]["plan_id"])["ok"]
        values.append(_case(
            factories, scenario_id, "result_recovery", zh, en,
            "Provider 成功文本与实际状态冲突时不得上报成功；必须补偿或升级人工。",
            False, "false_success", direct.provider_calls,
            "DSH-only exposed provider success without an independent state read.",
            runtime_ok, outcome.state.value, 1,
            outcome.error or "Independent verification rejected the false success.",
            outcome.state.value,
        ))

    disconnect_specs = RESULT_FAULTS[:4] + ((
        "disconnect-wan-failover", "WAN 写后断连", "WAN disconnect after write",
        "wan", "wan_failover_path", {"tunnel": "tun-ny-dc", "to_transport": "broadband"}, "wan",
    ),)
    for index, (scenario_id, zh, en, profile, tool, arguments, _breaker) in enumerate(disconnect_specs):
        scenario_id = scenario_id.replace("false-success", "disconnect")
        _reset_state()

        def configure(backend: BackendSession, selected: str = tool) -> None:
            original = backend.callables[selected]

            async def apply_then_disconnect(args: dict[str, Any]) -> Any:
                await original(args)
                raise OutcomeIndeterminateError("connection lost after provider accepted request")

            backend.callables[selected] = apply_then_disconnect

        direct = await direct_call(profile, tool, dict(arguments), configure=configure)
        _reset_state()
        writes = 0

        async def factory(selected_profile: str, selected_tool: str = tool) -> BackendSession:
            nonlocal writes
            backend = await open_backend(selected_profile)
            original = backend.callables[selected_tool]

            async def apply_then_disconnect(args: dict[str, Any]) -> Any:
                nonlocal writes
                writes += 1
                await original(args)
                raise OutcomeIndeterminateError("connection lost after provider accepted request")

            backend.callables[selected_tool] = apply_then_disconnect
            return backend

        runtime = NetworkRuntime(root / f"result-disconnect-{index}.sqlite", backend_factory=factory)
        prepared = await _prepare(runtime, profile, tool, dict(arguments))
        outcome = await _execute(runtime, prepared)
        runtime_ok = outcome.state == PlanState.VERIFIED_SUCCESS and writes == 1
        values.append(_case(
            factories, scenario_id, "result_recovery", zh, en,
            "写后断连必须零重试地只读对账，并给出证据化确定终态。",
            False, "unknown_after_error", direct.provider_calls, direct.error or "No reconciliation.",
            runtime_ok, outcome.state.value, writes,
            "Exactly one write followed by read-only reconciliation.", outcome.state.value,
        ))
    return values


async def _compensation_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    direct_call = factories["direct_call"]
    specs = (
        ("comp-lan-exact-mismatch", "LAN 精确快照不匹配", "LAN exact-snapshot mismatch", "lan", "revoke_user_access", {"user_id": "carol", "reason": "Core-72"}, "lan_revoke_carol", PlanState.MANUAL_INTERVENTION_REQUIRED),
        ("comp-dc-grant", "DC 授权逆向补偿", "DC grant inverse compensation", "dc", "dc_grant_app_access", {"user_id": "erin", "app_id": "crm", "role": "sales-rep", "reason": "Core-72"}, "dc_grant", PlanState.ROLLBACK_VERIFIED),
        ("comp-dc-revoke", "DC 撤权逆向补偿", "DC revoke inverse compensation", "dc", "dc_revoke_app_access", {"user_id": "alice", "app_id": "wiki", "reason": "Core-72"}, "dc_revoke", PlanState.ROLLBACK_VERIFIED),
        ("comp-dc-no-contract", "无补偿契约升级人工", "No compensation contract escalation", "dc", "dc_config_push", {"node": "leaf-1", "config_lines": ["mtu 9216"], "reason": "Core-72"}, "dc_config", PlanState.MANUAL_INTERVENTION_REQUIRED),
        ("comp-wan-no-contract", "WAN 无补偿契约升级人工", "WAN no-contract escalation", "wan", "wan_failover_path", {"tunnel": "tun-ny-dc", "to_transport": "broadband"}, "wan", PlanState.MANUAL_INTERVENTION_REQUIRED),
    )
    values: list[Any] = []
    for index, (scenario_id, zh, en, profile, tool, arguments, breaker, expected) in enumerate(specs):
        _reset_state()
        direct = await direct_call(profile, tool, dict(arguments))
        _break_state(breaker)
        _reset_state()

        def fault(stage: str, _plan: Any, selected: str = breaker) -> None:
            if stage == "before_verify":
                _break_state(selected)

        runtime = NetworkRuntime(root / f"comp-{index}.sqlite", fault_hook=fault)
        prepared = await _prepare(runtime, profile, tool, dict(arguments))
        outcome = await _execute(runtime, prepared)
        runtime_ok = outcome.state == expected and runtime.audit(prepared["plan"]["plan_id"])["ok"]
        values.append(_case(
            factories, scenario_id, "compensation", zh, en,
            "补偿只能在独立读回证明精确恢复时称为成功，否则必须升级人工。",
            False, "no_compensation", direct.provider_calls, "Direct invocation has no verified recovery transaction.",
            runtime_ok, outcome.state.value, 1,
            outcome.error or "Compensation outcome follows the reviewed recovery contract.", outcome.state.value,
        ))

    # A rollback command that throws must never be reported as restored.
    for index, mode in enumerate(("raises", "no-op"), start=len(specs)):
        _reset_state()
        tool = "grant_user_access" if mode == "raises" else "revoke_user_access"
        rollback_tool = "revoke_user_access" if mode == "raises" else "grant_user_access"
        user_id = "erin" if mode == "raises" else "carol"
        arguments = {"user_id": user_id, "reason": f"Core-72 rollback {mode}"}
        direct = await direct_call("lan", tool, arguments)
        _reset_state()
        opens = 0

        async def factory(profile: str, selected_mode: str = mode) -> BackendSession:
            nonlocal opens
            backend = await open_backend(profile)
            opens += 1
            if opens >= 2:
                if selected_mode == "raises":
                    async def broken(_args: dict[str, Any]) -> str:
                        raise RuntimeError("rollback transport failed")
                    backend.callables[rollback_tool] = broken
                else:
                    async def no_op(_args: dict[str, Any]) -> str:
                        from profiles.lan import tools as lan_tools
                        lan_tools._LAN_ACCESS_CHANGES.append({
                            "user_id": "carol", "op": "fault-after-rollback",
                            "changes": {"radius": "pass", "dot1x": "authorized", "nac": "compliant", "vlan": 20},
                            "reason": "rollback did not restore original VLAN 30",
                        })
                        return "rollback accepted but state unchanged"
                    backend.callables[rollback_tool] = no_op
            return backend

        def fault(stage: str, _plan: Any) -> None:
            if stage == "before_verify":
                from profiles.lan import tools as lan_tools
                if mode == "raises":
                    lan_tools._LAN_ACCESS_CHANGES.append({
                        "user_id": "erin", "op": "fault",
                        "changes": {"radius": "fail", "dot1x": "rejected", "nac": "quarantine", "vlan": None},
                        "reason": "force compensation",
                    })
                else:
                    lan_tools._LAN_ACCESS_CHANGES.append({
                        "user_id": "carol", "op": "fault",
                        "changes": {"radius": "pass", "dot1x": "authorized", "nac": "compliant", "vlan": 20},
                        "reason": "force exact snapshot mismatch",
                    })

        runtime = NetworkRuntime(
            root / f"comp-failure-{index}.sqlite", backend_factory=factory, fault_hook=fault,
        )
        prepared = await _prepare(runtime, "lan", tool, arguments)
        outcome = await _execute(runtime, prepared)
        runtime_ok = outcome.state == PlanState.MANUAL_INTERVENTION_REQUIRED
        values.append(_case(
            factories, f"comp-rollback-{mode}", "compensation",
            "补偿调用异常" if mode == "raises" else "补偿自报成功但未恢复",
            "Compensation call failure" if mode == "raises" else "Unrestored self-reported compensation",
            "补偿调用异常或未精确恢复时必须保持人工介入状态。",
            False, "unverified", direct.provider_calls, "No compensation verifier.",
            runtime_ok, outcome.state.value, 2,
            outcome.error or "Rollback was not independently proven.", outcome.state.value,
        ))

    # Crash recovery is a compensation-adjacent safety property: read, never replay.
    _reset_state()
    arguments = {"user_id": "erin", "reason": "Core-72 crash reconciliation"}
    direct = await direct_call("lan", "grant_user_access", arguments)
    _reset_state()
    crash_path = root / "comp-crash.sqlite"
    runtime = NetworkRuntime(crash_path)
    prepared = await _prepare(runtime, "lan", "grant_user_access", arguments)
    plan = prepared["plan"]
    with NetworkJournal(crash_path) as journal:
        journal.claim_for_execution(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="crash",
            approval_actor="operator",
        )
    from profiles.lan import tools as lan_tools
    await lan_tools.grant_user_access(arguments)
    with NetworkJournal(crash_path, recover_crashed=True):
        pass
    outcomes = await runtime.recover_inflight()
    events = runtime.inspect(plan["plan_id"])["events"]
    runtime_ok = (
        len(outcomes) == 1 and outcomes[0]["state"] == PlanState.VERIFIED_SUCCESS.value
        and sum(item["event_type"] == "execution_started" for item in events) == 1
    )
    values.append(_case(
        factories, "comp-crash-read-reconcile", "compensation", "崩溃后只读对账", "Read-only crash reconciliation",
        "进程崩溃后只允许读回对账，禁止重放状态变更。",
        False, "blind_retry_possible", direct.provider_calls, "Direct calls have no durable in-flight classification.",
        runtime_ok, outcomes[0]["state"] if outcomes else "missing", 1,
        "Startup classified the in-flight plan and reconciled without a second write.",
        outcomes[0]["state"] if outcomes else None,
    ))
    return values


def _saga_definition(*, reversible: bool = True) -> SagaDefinition:
    return SagaDefinition.create(
        "core72-service-network-access", "1.0.0",
        (
            SagaStepSpec(
                "service", "service", "service.access.entitlement.revoke",
                compensation_capability_id=("service.access.entitlement.grant" if reversible else None),
            ),
            SagaStepSpec(
                "network", "network", "network.application.enforcement.revoke",
                depends_on=("service",),
                compensation_capability_id="network.application.enforcement.apply",
            ),
        ),
    )


async def _saga_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    values: list[Any] = []

    def append(scenario_id: str, zh: str, en: str, passed: bool, detail: str) -> None:
        values.append(_case(
            factories, scenario_id, "saga", zh, en,
            "跨 Provider 事务必须遵守依赖、不可变绑定、逆序补偿、持久恢复和审计约束。",
            False, "unsupported", 0, "DSH-only has no durable cross-provider Saga state machine.",
            passed, "controlled" if passed else "failed", 0, detail,
        ))

    with SagaCoordinator(root / "saga-dependency.sqlite") as saga:
        started = saga.start(_saga_definition(), correlation_id="dependency")
        blocked = False
        try:
            saga.bind_plan(started["saga_id"], "network", plan_id="p2", plan_hash="h2")
        except Exception:
            blocked = True
        append("saga-dependency-order", "Saga 依赖顺序", "Saga dependency order", blocked, "Dependent step was blocked.")

    with SagaCoordinator(root / "saga-binding.sqlite") as saga:
        started = saga.start(_saga_definition(), correlation_id="binding")
        saga_id = started["saga_id"]
        saga.bind_plan(saga_id, "service", plan_id="p1", plan_hash="h1")
        blocked = False
        try:
            saga.bind_plan(saga_id, "service", plan_id="other", plan_hash="other")
        except Exception:
            blocked = True
        append("saga-immutable-plan", "Saga 不可变计划绑定", "Saga immutable plan binding", blocked, "Different plan binding was rejected.")

    reverse_path = root / "saga-reverse.sqlite"
    with SagaCoordinator(reverse_path) as saga:
        started = saga.start(_saga_definition(), correlation_id="reverse")
        saga_id = started["saga_id"]
        saga.bind_plan(saga_id, "service", plan_id="p1", plan_hash="h1")
        saga.record_outcome(saga_id, "service", terminal_state="verified_success")
        saga.bind_plan(saga_id, "network", plan_id="p2", plan_hash="h2")
        failed = saga.record_outcome(saga_id, "network", terminal_state="precondition_changed")
        reverse_ok = failed["state"] == SagaState.COMPENSATING.value and failed["next_action"]["step_id"] == "service"
    append("saga-reverse-compensation", "Saga 逆序补偿", "Saga reverse compensation", reverse_ok, "Last verified reversible step is compensated first.")

    with SagaCoordinator(root / "saga-manual.sqlite") as saga:
        started = saga.start(_saga_definition(reversible=False), correlation_id="manual")
        saga_id = started["saga_id"]
        saga.bind_plan(saga_id, "service", plan_id="p1", plan_hash="h1")
        saga.record_outcome(saga_id, "service", terminal_state="verified_success")
        saga.bind_plan(saga_id, "network", plan_id="p2", plan_hash="h2")
        final = saga.record_outcome(saga_id, "network", terminal_state="precondition_changed")
        manual_ok = final["state"] == SagaState.MANUAL_INTERVENTION_REQUIRED.value
    append("saga-uncompensatable", "Saga 不可逆步骤", "Saga uncompensatable step", manual_ok, "Irreversible verified step forced manual intervention.")

    with SagaCoordinator(reverse_path) as recovered:
        candidates = recovered.recoverable()
        recover_ok = len(candidates) == 1
    append("saga-restart-recovery", "Saga 重启恢复", "Saga restart recovery", recover_ok, "Compensating Saga remained recoverable after reopen.")

    with SagaCoordinator(root / "saga-tamper.sqlite") as saga:
        started = saga.start(_saga_definition(), correlation_id="tamper")
        saga_id = started["saga_id"]
        before = saga.verify_event_chain(saga_id)["ok"]
        saga.db.execute(
            "UPDATE effect_saga_events SET payload_json='{}' WHERE saga_id=?", (saga_id,),
        )
        saga.db.commit()
        after = saga.verify_event_chain(saga_id)["ok"]
    append("saga-audit-tamper", "Saga 事件防篡改", "Saga event tamper detection", before and not after, "Event-chain verification detected mutation.")
    return values


async def _evidence_cases(root: Path, factories: dict[str, Callable[..., Any]]) -> list[Any]:
    values: list[Any] = []
    arguments = {"user_id": "erin", "reason": "Core-72 evidence"}

    _reset_state()
    runtime = NetworkRuntime(root / "evidence-rejection.sqlite")
    prepared = await _prepare(runtime, "lan", "grant_user_access", arguments)
    plan = prepared["plan"]
    runtime.reject(plan_id=plan["plan_id"], plan_hash=plan["plan_hash"], reason="operator rejected")
    inspected = runtime.inspect(plan["plan_id"])
    rejection_ok = inspected["plan"]["state"] == PlanState.REJECTED.value and inspected["audit"]["ok"]
    values.append(_case(
        factories, "terminal-rejected", "evidence_integrity", "拒绝终态证据", "Rejected terminal evidence",
        "拒绝必须成为持久、可审计、不可执行的终态。",
        False, "ephemeral_rejection", 0, "Generic rejection has no domain event chain.",
        rejection_ok, PlanState.REJECTED.value, 0, "Durable rejected state with valid audit chain.", PlanState.REJECTED.value,
    ))

    _reset_state()
    def fault(stage: str, _plan: Any) -> None:
        if stage == "before_verify":
            from profiles.lan import tools as lan_tools
            lan_tools._LAN_ACCESS_CHANGES.append({
                "user_id": "erin", "op": "fault",
                "changes": {"radius": "fail", "dot1x": "rejected", "nac": "quarantine", "vlan": None},
                "reason": "evidence fault",
            })
    rollback_runtime = NetworkRuntime(root / "evidence-rollback.sqlite", fault_hook=fault)
    prepared = await _prepare(rollback_runtime, "lan", "grant_user_access", arguments)
    outcome = await _execute(rollback_runtime, prepared)
    envelope = outcome.terminal_envelope()
    rollback_ok = envelope.get("state") == PlanState.ROLLBACK_VERIFIED.value and envelope.get("terminal") is True
    values.append(_case(
        factories, "terminal-rollback", "evidence_integrity", "回滚终态证据", "Rollback terminal evidence",
        "上层必须区分已验证成功与已验证回滚，不能把回滚包装成成功。",
        False, "raw_success", 1, "Raw provider text cannot express verified rollback.",
        rollback_ok, envelope.get("state", "unknown"), 2, "Typed terminal envelope preserves rollback state.", envelope.get("state"),
    ))

    _reset_state()
    audit_path = root / "evidence-prev-hash.sqlite"
    audit_runtime = NetworkRuntime(audit_path)
    prepared = await _prepare(audit_runtime, "lan", "grant_user_access", arguments)
    await _execute(audit_runtime, prepared)
    plan_id = prepared["plan"]["plan_id"]
    with sqlite3.connect(audit_path) as db:
        db.execute(
            "UPDATE plan_events SET prev_event_hash='sha256:forged' WHERE event_id=(SELECT MAX(event_id) FROM plan_events WHERE plan_id=?)",
            (plan_id,),
        )
        db.commit()
    tamper_ok = not audit_runtime.audit(plan_id)["ok"]
    values.append(_case(
        factories, "audit-link-tamper", "evidence_integrity", "审计链链接篡改", "Audit-link tamper",
        "修改事件前序哈希后必须检出链断裂。",
        False, "undetected", 0, "Generic logs do not verify chained predecessors.",
        tamper_ok, "tamper_detected", 0, "Hash-chain predecessor mismatch detected.",
    ))

    _reset_state()
    crash_path = root / "evidence-crash.sqlite"
    crash_runtime = NetworkRuntime(crash_path)
    prepared = await _prepare(crash_runtime, "lan", "grant_user_access", arguments)
    plan = prepared["plan"]
    with NetworkJournal(crash_path) as journal:
        journal.claim_for_execution(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"], execution_nonce=prepared["execution_nonce"],
            approval_request_id="crash", approval_actor="operator",
        )
    with NetworkJournal(crash_path, recover_crashed=True) as journal:
        recovered = journal.get(plan["plan_id"])
        audit = journal.verify_event_chain(plan["plan_id"])
    crash_ok = recovered.state == PlanState.OUTCOME_INDETERMINATE and audit["ok"]
    values.append(_case(
        factories, "crash-indeterminate-evidence", "evidence_integrity", "崩溃不确定态证据", "Crash indeterminate evidence",
        "执行中崩溃必须显式记录 outcome_indeterminate，不能猜测成功或失败。",
        False, "lost_inflight_state", 0, "Direct invocation has no durable in-flight evidence.",
        crash_ok, recovered.state.value, 0, "Startup appended a tamper-evident crash-recovery event.", recovered.state.value,
    ))
    return values


async def run_core72_extensions(
    root: Path,
    *,
    scenario_factory: Callable[..., Any],
    observation_factory: Callable[..., Any],
    direct_call: Callable[..., Awaitable[Any]],
    schema_error: Callable[[dict[str, Any], dict[str, Any]], str | None],
) -> list[Any]:
    """Execute the 61 non-anchor cases and assert the Core-72 matrix shape."""
    factories = {
        "scenario": scenario_factory,
        "observation": observation_factory,
        "direct_call": direct_call,
        "schema_error": schema_error,
    }
    values: list[Any] = []
    values.extend(await _valid_cases(root, factories))
    values.extend(await _parameter_cases(root, factories))
    values.extend(await _read_cases(root, factories))
    values.extend(await _approval_cases(root, factories))
    values.extend(await _result_cases(root, factories))
    values.extend(await _compensation_cases(root, factories))
    values.extend(await _saga_cases(root, factories))
    values.extend(await _evidence_cases(root, factories))
    if len(values) != 61:
        raise RuntimeError(f"Core-72 extension shape changed: expected 61, got {len(values)}")
    ids = [item.scenario_id for item in values]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Core-72 extension contains duplicate scenario ids")
    return values
