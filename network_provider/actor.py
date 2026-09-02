"""Durable, capability-bound execution manager for the Network Actor MCP."""

from __future__ import annotations

import inspect
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from network_lab import ContainerlabProvider
from network_lab.containerlab import normalize_frr_config
from network_lab.tools import LabToolAdapter
from network_runtime.contracts import sha256_json
from network_runtime.evidence import config_matches, same_snapshot, typed_evidence
from network_runtime.provider_contracts import REGISTRY, ProviderCapability

from .actor_store import ActorStore, ActorStoreError


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
FaultHook = Callable[[str, str], Awaitable[None] | None]


class ActorCrashSimulated(BaseException):
    """Test-only process-crash signal intentionally not caught as an effect error."""


@dataclass(frozen=True)
class EffectContext:
    operation_id: str
    plan_hash: str
    intent_hash: str
    approved_preflight: dict[str, Any]
    effect_phase: str

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> "EffectContext":
        try:
            parsed = uuid.UUID(operation_id)
        except ValueError as error:
            raise ActorStoreError("operation_id must be a UUID") from error
        if str(parsed) != operation_id:
            raise ActorStoreError("operation_id must use canonical UUID form")
        if not _HASH.fullmatch(plan_hash) or not _HASH.fullmatch(intent_hash):
            raise ActorStoreError("plan_hash and intent_hash must be canonical sha256 values")
        if effect_phase not in {"execute", "compensate"}:
            raise ActorStoreError("effect_phase must be execute or compensate")
        if not isinstance(approved_preflight, dict):
            raise ActorStoreError("approved_preflight must be an object")
        return cls(
            operation_id, plan_hash, intent_hash,
            approved_preflight, effect_phase,
        )


class DurableNetworkActor:
    def __init__(
        self,
        provider: ContainerlabProvider,
        store: ActorStore,
        *,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self.provider = provider
        self.adapter = LabToolAdapter(provider)
        self.store = store
        self.fault_hook = fault_hook

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: EffectContext,
    ) -> dict[str, Any]:
        capability = self._actor_capability(tool_name)
        target_key = self._target_key(tool_name, arguments)
        if context.effect_phase == "compensate":
            return await self._restore(target_key, context)
        async with self.store.target_lock(target_key):
            self._validate_effect_arguments(tool_name, arguments)
            desired = self._desired_state(tool_name, arguments)
            existing = self.store.find(context.operation_id)
            if existing is not None:
                operation, _ = self.store.claim_operation(
                    operation_id=context.operation_id,
                    plan_hash=context.plan_hash,
                    intent_hash=context.intent_hash,
                    capability_id=capability.capability_id,
                    capability_version=capability.capability_version,
                    tool_name=tool_name,
                    target_key=target_key,
                    arguments=arguments,
                    approved_preflight=context.approved_preflight,
                    snapshot=existing["snapshot"],
                    desired=existing["desired"],
                )
                return await self._replay_existing(tool_name, arguments, operation)
            observed_preflight = await self._observe_preflight(tool_name, arguments)
            if not same_snapshot(context.approved_preflight, observed_preflight):
                raise ActorStoreError(
                    "Actor observed state does not match the Runtime-approved preflight"
                )
            snapshot = await self._capture_snapshot(tool_name, arguments)
            operation, replayed = self.store.claim_operation(
                operation_id=context.operation_id,
                plan_hash=context.plan_hash,
                intent_hash=context.intent_hash,
                capability_id=capability.capability_id,
                capability_version=capability.capability_version,
                tool_name=tool_name,
                target_key=target_key,
                arguments=arguments,
                approved_preflight=context.approved_preflight,
                snapshot=snapshot,
                desired=desired,
            )
            fence = int(operation["fence_token"])
            self.store.assert_fence(context.operation_id, fence)
            self.store.transition(
                context.operation_id, "executing", "effect_started",
                payload={"fence_token": fence},
            )
            try:
                payload = await self._apply(tool_name, arguments)
                await self._fault("after_effect_before_commit", context.operation_id)
            except Exception as error:
                if await self._desired_observed(tool_name, arguments, desired):
                    updated = self.store.transition(
                        context.operation_id, "applied", "effect_reconciled_after_error",
                        result={"payload": payload if "payload" in locals() else None},
                        payload={"error_type": type(error).__name__},
                    )
                    return self._result(updated, replayed=replayed, reconciled=True)
                if await self._snapshot_observed(tool_name, arguments, snapshot):
                    self.store.transition(
                        context.operation_id, "not_applied", "effect_not_observed",
                        error=f"{type(error).__name__}: {error}",
                    )
                else:
                    self.store.transition(
                        context.operation_id, "outcome_indeterminate",
                        "effect_outcome_indeterminate",
                        error=f"{type(error).__name__}: {error}",
                    )
                raise
            if not await self._desired_observed(tool_name, arguments, desired):
                self.store.transition(
                    context.operation_id, "outcome_indeterminate",
                    "effect_postcondition_not_observed",
                    result={"payload": payload},
                )
                raise ActorStoreError("Actor write returned but desired state was not observed")
            updated = self.store.transition(
                context.operation_id, "applied", "effect_applied",
                result={"payload": payload},
            )
            return self._result(updated, replayed=replayed, reconciled=False)

    async def _replay_existing(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        """Reconcile an immutable duplicate without ever replaying its write."""
        state = str(operation["state"])
        if state in {"restored", "committed", "manual_intervention"}:
            raise ActorStoreError(f"operation is already terminal in Actor state={state}")
        if state == "applied":
            if await self._desired_observed(tool_name, arguments, operation["desired"]):
                return self._result(operation, replayed=True, reconciled=True)
            self.store.transition(
                operation["operation_id"], "manual_intervention",
                "idempotent_replay_state_drift",
                error="stored applied operation no longer matches observed desired state",
            )
            raise ActorStoreError("applied Actor operation drifted before replay")
        if state in {"executing", "outcome_indeterminate"}:
            if await self._desired_observed(tool_name, arguments, operation["desired"]):
                updated = self.store.transition(
                    operation["operation_id"], "applied", "duplicate_reconciled_applied",
                )
                return self._result(updated, replayed=True, reconciled=True)
            if await self._snapshot_observed(tool_name, arguments, operation["snapshot"]):
                self.store.transition(
                    operation["operation_id"], "not_applied", "duplicate_reconciled_not_applied",
                )
                raise ActorStoreError(
                    "prior operation was not observed; create a fresh approved plan instead of replaying"
                )
            self.store.transition(
                operation["operation_id"], "manual_intervention",
                "duplicate_reconciliation_failed",
            )
            raise ActorStoreError("prior Actor outcome remains indeterminate")
        if state == "not_applied":
            raise ActorStoreError(
                "operation is durably not-applied; create a fresh approved plan"
            )
        if state != "prepared":
            raise ActorStoreError(f"unsupported Actor replay state={state}")
        # A crash after durable preparation but before effect_started is the
        # only state in which resuming the original immutable write is safe.
        return await self._resume_prepared(tool_name, arguments, operation)

    async def _resume_prepared(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        if not await self._snapshot_observed(tool_name, arguments, operation["snapshot"]):
            self.store.transition(
                operation["operation_id"], "manual_intervention",
                "prepared_snapshot_drifted",
            )
            raise ActorStoreError("prepared operation target drifted before effect")
        fence = int(operation["fence_token"])
        self.store.assert_fence(operation["operation_id"], fence)
        self.store.transition(
            operation["operation_id"], "executing", "prepared_effect_resumed",
            payload={"fence_token": fence},
        )
        try:
            payload = await self._apply(tool_name, arguments)
            await self._fault("after_effect_before_commit", operation["operation_id"])
        except Exception as error:
            if await self._desired_observed(tool_name, arguments, operation["desired"]):
                updated = self.store.transition(
                    operation["operation_id"], "applied", "resumed_effect_reconciled_after_error",
                    result={"payload": locals().get("payload")},
                    payload={"error_type": type(error).__name__},
                )
                return self._result(updated, replayed=True, reconciled=True)
            target_state = (
                "not_applied"
                if await self._snapshot_observed(tool_name, arguments, operation["snapshot"])
                else "outcome_indeterminate"
            )
            self.store.transition(
                operation["operation_id"], target_state, "resumed_effect_failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        if not await self._desired_observed(tool_name, arguments, operation["desired"]):
            self.store.transition(
                operation["operation_id"], "outcome_indeterminate",
                "resumed_effect_postcondition_not_observed", result={"payload": payload},
            )
            raise ActorStoreError("resumed Actor write did not reach desired state")
        updated = self.store.transition(
            operation["operation_id"], "applied", "resumed_effect_applied",
            result={"payload": payload},
        )
        return self._result(updated, replayed=True, reconciled=False)

    async def finalize(
        self, operation_id: str, plan_hash: str, terminal_state: str,
    ) -> dict[str, Any]:
        updated = self.store.finalize(operation_id, plan_hash, terminal_state)
        return {
            "ok": True,
            "operation_id": operation_id,
            "state": updated["state"],
            "runtime_terminal_state": terminal_state,
            "audit_chain_valid": self.store.verify_event_chain(operation_id),
        }

    async def reconcile_startup(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for operation in self.store.recoverable():
            async with self.store.target_lock(operation["target_key"]):
                tool_name = str(operation["tool_name"])
                arguments = dict(operation["arguments"])
                if operation["state"] == "restoring" and await self._snapshot_observed(
                    tool_name, arguments, operation["snapshot"],
                ):
                    state, event = "restored", "startup_reconciled_restored"
                elif await self._desired_observed(
                    tool_name, arguments, operation["desired"],
                ):
                    state, event = "applied", "startup_reconciled_applied"
                elif await self._snapshot_observed(
                    tool_name, arguments, operation["snapshot"],
                ):
                    state, event = "not_applied", "startup_reconciled_not_applied"
                else:
                    state, event = "manual_intervention", "startup_reconciliation_failed"
                updated = self.store.transition(operation["operation_id"], state, event)
                results.append({
                    "operation_id": operation["operation_id"], "state": updated["state"],
                })
        return results

    async def _restore(
        self, target_key: str, context: EffectContext,
    ) -> dict[str, Any]:
        async with self.store.target_lock(target_key):
            operation = self.store.claim_for_restore(
                operation_id=context.operation_id,
                plan_hash=context.plan_hash,
                target_key=target_key,
            )
            if operation["preflight_digest"] != sha256_json(context.approved_preflight):
                raise ActorStoreError("restore preflight does not match the durable operation")
            if operation["intent_hash"] != context.intent_hash:
                raise ActorStoreError("restore intent hash does not match the durable operation")
            if operation["state"] == "restored":
                return self._result(operation, replayed=True, reconciled=True)
            if operation["state"] == "committed":
                raise ActorStoreError("committed Actor operation cannot be compensated")
            fence = int(operation["fence_token"])
            self.store.assert_fence(context.operation_id, fence)
            self.store.transition(
                context.operation_id, "restoring", "restore_started",
                payload={"fence_token": fence},
            )
            try:
                payload = await self._apply_snapshot(
                    operation["tool_name"], operation["arguments"], operation["snapshot"],
                )
                await self._fault("after_restore_before_commit", context.operation_id)
            except Exception as error:
                if await self._snapshot_observed(
                    operation["tool_name"], operation["arguments"], operation["snapshot"],
                ):
                    updated = self.store.transition(
                        context.operation_id, "restored", "restore_reconciled_after_error",
                        result={"payload": payload if "payload" in locals() else None},
                        payload={"error_type": type(error).__name__},
                    )
                    return self._result(updated, replayed=False, reconciled=True)
                self.store.transition(
                    context.operation_id, "manual_intervention", "restore_failed",
                    error=f"{type(error).__name__}: {error}",
                )
                raise
            if not await self._snapshot_observed(
                operation["tool_name"], operation["arguments"], operation["snapshot"],
            ):
                self.store.transition(
                    context.operation_id, "manual_intervention",
                    "restore_postcondition_not_observed", result={"payload": payload},
                )
                raise ActorStoreError("durable snapshot restoration was not observed")
            updated = self.store.transition(
                context.operation_id, "restored", "restore_completed",
                result={"payload": payload},
            )
            return self._result(updated, replayed=False, reconciled=False)

    def _actor_capability(self, tool_name: str) -> ProviderCapability:
        value = REGISTRY.for_tool(tool_name)
        if value is None or value.provider_role != "actor":
            raise ActorStoreError(f"unreviewed Actor capability {tool_name!r}")
        return value

    def _validate_effect_arguments(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> None:
        if tool_name == "dc_grant_app_access":
            app_id = str(arguments["app_id"])
            app = self.provider._application(app_id)
            role = str(arguments.get("role") or app.roles[0]).strip()
            if role not in app.roles:
                raise ActorStoreError(
                    f"role {role!r} is not reviewed for application {app_id!r}"
                )

    @staticmethod
    def _target_key(tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name in {"edit_device_config", "restore_device_config"}:
            return f"device:{arguments['device_id']}:config"
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            return f"user:{arguments['user_id']}:admission"
        if tool_name in {
            "dc_grant_app_access", "dc_revoke_app_access",
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
            "network_restore_app_enforcement",
        }:
            return f"application:{arguments['user_id']}:{arguments['app_id']}:enforcement"
        if tool_name in {"fabric_set_access_vlan", "fabric_restore_access_vlan"}:
            return f"fabric:{arguments['device_id']}:{arguments['interface']}:access-vlan"
        raise ActorStoreError(f"Actor tool {tool_name!r} has no target-key contract")

    async def _observe_preflight(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name == "edit_device_config":
            rendered = await self.adapter.get_device_config(arguments)
            source = "get_device_config"
        elif tool_name in {"grant_user_access", "revoke_user_access"}:
            rendered = await self.adapter.get_user_access(arguments)
            source = "get_user_access"
        elif tool_name in {"dc_grant_app_access", "dc_revoke_app_access"}:
            rendered = await self.adapter.dc_check_user_app_access(arguments)
            source = "dc_check_user_app_access"
        elif tool_name == "fabric_set_access_vlan":
            rendered = await self.adapter.lab_get_access_vlan(arguments)
            source = "lab_get_access_vlan"
        elif tool_name in {
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        }:
            rendered = await self.adapter.network_get_app_enforcement(arguments)
            source = "network_get_app_enforcement"
        else:
            raise ActorStoreError(f"Actor tool {tool_name!r} has no preflight contract")
        return typed_evidence(source, rendered)

    async def _capture_snapshot(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "edit_device_config":
            return await self.provider.running_config(str(arguments["device_id"]))
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            return await self.provider.user_admission_snapshot(str(arguments["user_id"]))
        if tool_name in {
            "dc_grant_app_access", "dc_revoke_app_access",
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        }:
            return await self.provider.application_access_snapshot(
                str(arguments["user_id"]), str(arguments["app_id"]),
            )
        if tool_name == "fabric_set_access_vlan":
            return await self.provider.fabric_access_vlan(
                str(arguments["device_id"]), str(arguments["interface"]),
            )
        raise ActorStoreError(f"Actor tool {tool_name!r} has no snapshot contract")

    @staticmethod
    def _desired_state(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "edit_device_config":
            return {"configuration_digest": sha256_json(arguments.get("config_lines") or [])}
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            return {"admitted": tool_name == "grant_user_access"}
        if tool_name in {
            "dc_grant_app_access", "network_apply_app_enforcement",
        }:
            return {"allowed": True}
        if tool_name in {
            "dc_revoke_app_access", "network_revoke_app_enforcement",
        }:
            return {"allowed": False}
        if tool_name == "fabric_set_access_vlan":
            return {"vlan_id": int(arguments["vlan_id"])}
        raise ActorStoreError(f"Actor tool {tool_name!r} has no desired-state contract")

    async def _apply(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "edit_device_config":
            return await self.provider.apply_config(
                str(arguments["device_id"]),
                tuple(str(item) for item in arguments["config_lines"]),
            )
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            return await self.provider.set_user_admission(
                str(arguments["user_id"]), admitted=tool_name == "grant_user_access",
            )
        if tool_name in {
            "dc_grant_app_access", "network_apply_app_enforcement",
            "dc_revoke_app_access", "network_revoke_app_enforcement",
        }:
            allowed = tool_name in {"dc_grant_app_access", "network_apply_app_enforcement"}
            return await self.provider.set_application_access(
                str(arguments["user_id"]), str(arguments["app_id"]), allowed=allowed,
            )
        if tool_name == "fabric_set_access_vlan":
            return await self.provider.set_fabric_access_vlan(
                str(arguments["device_id"]), str(arguments["interface"]),
                int(arguments["vlan_id"]),
            )
        raise ActorStoreError(f"Actor tool {tool_name!r} has no effect implementation")

    async def _apply_snapshot(
        self, tool_name: str, arguments: dict[str, Any], snapshot: Any,
    ) -> Any:
        if tool_name == "edit_device_config":
            return await self.provider.restore_config_snapshot(
                str(arguments["device_id"]), str(snapshot),
            )
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            return await self.provider.restore_user_admission_snapshot(
                str(arguments["user_id"]), dict(snapshot),
            )
        if tool_name in {
            "dc_grant_app_access", "dc_revoke_app_access",
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        }:
            return await self.provider.restore_application_access_snapshot(
                str(arguments["user_id"]), str(arguments["app_id"]),
                dict(snapshot),
            )
        if tool_name == "fabric_set_access_vlan":
            return await self.provider.restore_fabric_access_vlan_snapshot(
                str(arguments["device_id"]), str(arguments["interface"]), dict(snapshot),
            )
        raise ActorStoreError(f"Actor tool {tool_name!r} has no restore implementation")

    async def _desired_observed(
        self, tool_name: str, arguments: dict[str, Any], desired: dict[str, Any],
    ) -> bool:
        if tool_name == "edit_device_config":
            current = await self.provider.running_config(
                str(arguments["device_id"]),
                str(arguments["section"]) if arguments.get("section") else None,
            )
            return config_matches(arguments, current)
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            return await self.provider.user_admitted(
                str(arguments["user_id"]),
            ) is desired["admitted"]
        if tool_name in {
            "dc_grant_app_access", "dc_revoke_app_access",
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        }:
            blocked = await self.provider.application_access_blocked(
                str(arguments["user_id"]), str(arguments["app_id"]),
            )
            return (not blocked) is desired["allowed"]
        if tool_name == "fabric_set_access_vlan":
            current = await self.provider.fabric_access_vlan(
                str(arguments["device_id"]), str(arguments["interface"]),
            )
            return current.get("ok") is True and current.get("current_vlan") == desired["vlan_id"]
        return False

    async def _snapshot_observed(
        self, tool_name: str, arguments: dict[str, Any], snapshot: Any,
    ) -> bool:
        if tool_name == "edit_device_config":
            current = await self.provider.running_config(str(arguments["device_id"]))
            return normalize_frr_config(current) == normalize_frr_config(str(snapshot))
        if tool_name in {"grant_user_access", "revoke_user_access"}:
            try:
                current = await self.provider.user_admission_snapshot(
                    str(arguments["user_id"]),
                )
            except Exception:
                return False
            return current == snapshot
        if tool_name in {
            "dc_grant_app_access", "dc_revoke_app_access",
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        }:
            try:
                current = await self.provider.application_access_snapshot(
                    str(arguments["user_id"]), str(arguments["app_id"]),
                )
            except Exception:
                return False
            return current == snapshot
        if tool_name == "fabric_set_access_vlan":
            current = await self.provider.fabric_access_vlan(
                str(arguments["device_id"]), str(arguments["interface"]),
            )
            return all(
                current.get(field) == snapshot.get(field)
                for field in ("current_vlan", "bridge", "vlans")
            )
        return False

    async def _fault(self, point: str, operation_id: str) -> None:
        if self.fault_hook is None:
            return
        result = self.fault_hook(point, operation_id)
        if inspect.isawaitable(result):
            await result

    def _result(
        self, operation: dict[str, Any], *, replayed: bool, reconciled: bool,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "code": "ok",
            "operation_id": operation["operation_id"],
            "plan_hash": operation["plan_hash"],
            "capability_id": operation["capability_id"],
            "capability_version": operation["capability_version"],
            "target_key": operation["target_key"],
            "fence_token": operation["fence_token"],
            "state": operation["state"],
            "replayed": replayed,
            "reconciled": reconciled,
            "snapshot_digest": operation["snapshot_digest"],
            "audit_chain_valid": self.store.verify_event_chain(operation["operation_id"]),
            "simulation": True,
            "payload": (operation.get("result") or {}).get("payload"),
        }
