"""Read-only reconciliation between Service desired state and Network evidence."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable


ToolCallable = Callable[[dict[str, Any]], Awaitable[str]]


METADATA: dict[str, dict[str, Any]] = {
    "reconcile_service_network_access": {
        "description": (
            "Compare Service MCP entitlement desired state, CMDB bindings, actual "
            "Containerlab enforcement, and a real HTTP probe without changing either layer."
        ),
        "parameters": {
            "user_id": "Exact enterprise user identifier",
            "app_id": "Exact enterprise application identifier",
        },
        "required": ["user_id", "app_id"],
        "returns": "Structured cross-layer consistency and drift classification",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["effect-runtime", "service", "network", "reconciliation"],
    },
}


def build(callables: dict[str, ToolCallable]) -> dict[str, ToolCallable]:
    required = {
        "access_policy_get_entitlement",
        "network_get_app_enforcement",
        "cmdb_get_endpoint_binding",
    }
    if not required.issubset(callables):
        return {}

    async def reconcile(args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        desired_raw, observed_raw, user_binding_raw, app_binding_raw = await asyncio.gather(
            callables["access_policy_get_entitlement"]({"user_id": user_id, "app_id": app_id}),
            callables["network_get_app_enforcement"]({"user_id": user_id, "app_id": app_id}),
            callables["cmdb_get_endpoint_binding"]({
                "subject_type": "user", "subject_id": user_id,
            }),
            callables["cmdb_get_endpoint_binding"]({
                "subject_type": "application", "subject_id": app_id,
            }),
        )
        try:
            desired = json.loads(desired_raw)
            observed = json.loads(observed_raw)
            user_binding = json.loads(user_binding_raw)
            app_binding = json.loads(app_binding_raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cross-layer provider returned invalid structured evidence: {error}") from error
        if not all(item.get("ok") is True for item in (desired, observed, user_binding, app_binding)):
            raise RuntimeError("cross-layer provider returned ok=false")

        probe: dict[str, Any] | None = None
        if "lab_app_probe" in callables:
            probe_raw = await callables["lab_app_probe"]({"user_id": user_id, "app_id": app_id})
            try:
                probe = json.loads(probe_raw)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"lab_app_probe returned invalid JSON: {error}") from error

        desired_allowed = desired.get("allowed") is True
        enforced_allowed = observed.get("allowed") is True
        mappings_resolved = (
            user_binding.get("network_target_resolved") is True
            and app_binding.get("network_target_resolved") is True
        )
        traffic_ok = None if probe is None else probe.get("ok") is True
        if not mappings_resolved:
            drift = "cmdb_network_binding_missing"
        elif desired_allowed != enforced_allowed:
            drift = "desired_enforcement_mismatch"
        elif desired_allowed and traffic_ok is False:
            drift = "enforcement_allows_but_data_plane_failed"
        elif not desired_allowed and traffic_ok is True:
            drift = "data_plane_bypasses_denied_policy"
        else:
            drift = "none"
        return json.dumps({
            "ok": True,
            "user_id": user_id,
            "app_id": app_id,
            "consistent": drift == "none",
            "drift_class": drift,
            "service_desired_state": desired,
            "network_observed_state": observed,
            "cmdb_bindings": {"user": user_binding, "application": app_binding},
            "traffic_evidence": probe,
            "truth_boundary": {
                "service": "MCP-backed deterministic business desired state",
                "network": "Containerlab observed enforcement and data plane",
            },
        }, ensure_ascii=False, sort_keys=True)

    return {"reconcile_service_network_access": reconcile}
