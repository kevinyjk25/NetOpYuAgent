"""Strict parameter compilation, entity resolution and deterministic risk policy."""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from typing import Any

from effect_runtime.reliability import (
    AutonomyDecision,
    Reversibility,
    RiskAssessment,
    RiskFactors,
    RiskPolicy,
)

from .contracts import RiskLevel


_INTEGER_KEYS = {
    "flows", "grace_period_s", "length", "lines", "minutes", "offset", "range_minutes",
    "route_type", "top_n", "vlan_id", "vni", "expected_revision",
}
_BOOLEAN_KEYS = {"dry_run", "force", "graceful", "rolling"}
_ARRAY_KEYS = {"config_lines", "device_ids"}
_JSON_KEYS = {"changes"}
_TARGET_KEYS = (
    "device_id", "device_ids", "node", "node_id", "tunnel", "resource_id",
    "deploy_id", "service", "user_id", "app_id", "entity_id",
)
_ENUMS: dict[str, set[str]] = {
    "environment": {"prod", "staging", "dev"},
    "to_transport": {"mpls", "broadband", "lte"},
    "severity": {"critical", "warning", "info", "error"},
    "level": {"error", "warning", "info"},
}
_SAFETY_REQUIRED: dict[str, set[str]] = {
    "edit_device_config": {"device_id", "reason"},
    "push_config": {"device_id", "config_text"},
    "restart_service": {"service", "environment"},
    "rollback_service": {"service", "version", "environment"},
    "rollback_deploy": {"deploy_id"},
    "drain_node": {"node_id"},
    "failover": {"resource_id", "target"},
    "delete_resource": {"resource_id"},
    "grant_user_access": {"user_id", "reason"},
    "revoke_user_access": {"user_id", "reason"},
    "dc_config_push": {"node", "config_lines", "reason"},
    "dc_grant_app_access": {"user_id", "app_id", "reason"},
    "dc_revoke_app_access": {"user_id", "app_id", "reason"},
    "wan_failover_path": {"tunnel", "to_transport"},
    "fabric_set_access_vlan": {"device_id", "interface", "vlan_id", "reason"},
    "access_policy_grant_entitlement": {
        "user_id", "app_id", "role", "change_id", "expected_revision", "reason",
    },
    "access_policy_revoke_entitlement": {
        "user_id", "app_id", "change_id", "expected_revision", "reason",
    },
    "platform_restart_service": {
        "service", "environment", "change_id", "expected_revision", "reason",
    },
    "platform_rollback_service": {
        "service", "environment", "version", "change_id", "expected_revision", "reason",
    },
    "network_apply_app_enforcement": {"user_id", "app_id", "change_id", "reason"},
    "network_revoke_app_enforcement": {"user_id", "app_id", "change_id", "reason"},
}
_FORBIDDEN_CONFIG = re.compile(
    r"(^|\s)(write\s+erase|erase\s+startup|format\b|reload\b|factory-reset|"
    r"delete\s+flash|no\s+aaa\b|no\s+username\b)",
    re.IGNORECASE,
)
_READ_ONLY_COMMAND = re.compile(r"^\s*(show|display)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CompilationResult:
    ok: bool
    arguments: dict[str, Any]
    provenance: dict[str, str]
    targets: tuple[str, ...]
    missing: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def _schema_type(name: str, definition: Any) -> str | None:
    if isinstance(definition, dict) and definition.get("type"):
        return str(definition["type"])
    if name in _INTEGER_KEYS:
        return "integer"
    if name in _BOOLEAN_KEYS:
        return "boolean"
    if name in _ARRAY_KEYS:
        return "array"
    if name in _JSON_KEYS:
        return "object"
    return "string"


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [item.strip() if isinstance(item, str) else item for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    return value


def _validate_type(name: str, value: Any, expected: str | None) -> str | None:
    if expected == "string" and not isinstance(value, str):
        return f"{name} must be a string"
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return f"{name} must be an integer"
    if expected == "boolean" and not isinstance(value, bool):
        return f"{name} must be a boolean"
    if expected == "array" and not isinstance(value, list):
        return f"{name} must be an array"
    if expected == "object" and not isinstance(value, dict):
        return f"{name} must be an object"
    return None


def _known_entities(profile: str, mode: str) -> dict[str, set[str]]:
    entities: dict[str, set[str]] = {}
    if mode == "pragmatic":
        from config import load

        pragmatic = load(os.environ.get("NETOPYU_CONFIG_PATH", "config.yaml")).pragmatic
        devices = {device.id for device in pragmatic.device_inventory if device.id}
        if pragmatic.lab.enabled:
            from network_lab import load_manifest

            manifest = load_manifest(pragmatic.lab.manifest)
            devices.update(manifest.devices)
            entities["user_id"] = set(manifest.users)
            entities["app_id"] = set(manifest.applications)
        entities["device_id"] = devices
        return entities
    if profile == "lan":
        from profiles.lan import tools

        entities["device_id"] = {str(item["id"]) for item in tools._DEVICE_INVENTORY}
        entities["user_id"] = {str(item["id"]) for item in tools._LAN_USERS}
    elif profile == "dc":
        from profiles.dc import tools

        entities["node"] = {str(item["id"]) for item in tools._FABRIC_NODES}
        entities["app_id"] = {str(item["id"]) for item in tools._DC_APPS}
    elif profile == "wan":
        from profiles.wan import tools

        entities["tunnel"] = {str(item["id"]) for item in tools._WAN_TUNNELS}
    return entities


def compile_parameters(
    *,
    profile: str,
    mode: str,
    tool_name: str,
    metadata: dict[str, Any],
    arguments: dict[str, Any],
) -> CompilationResult:
    if not isinstance(arguments, dict):
        return CompilationResult(False, {}, {}, (), errors=("arguments must be an object",))
    definitions = metadata.get("parameters") or {}
    if not isinstance(definitions, dict):
        return CompilationResult(False, {}, {}, (), errors=("tool parameter contract is invalid",))
    unknown = sorted(set(arguments) - set(definitions))
    errors = [f"unknown parameter: {name}" for name in unknown]
    required = set(metadata.get("required") or ())
    if bool(metadata.get("hitl")) or str(metadata.get("action_type", "read_only")) != "read_only":
        required.update(_SAFETY_REQUIRED.get(tool_name, set()))
    missing = sorted(
        name for name in required
        if name not in arguments or arguments[name] is None
        or (isinstance(arguments[name], (str, list, dict)) and not arguments[name])
    )
    compiled: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for name, raw in arguments.items():
        if name not in definitions:
            continue
        value = _normalise(raw)
        expected = _schema_type(name, definitions[name])
        type_error = _validate_type(name, value, expected)
        if type_error:
            errors.append(type_error)
            continue
        if isinstance(value, str):
            if not value and name in required:
                continue
            if "\x00" in value or any(ord(ch) < 9 for ch in value):
                errors.append(f"{name} contains forbidden control characters")
            max_length = 65_536 if name == "config_text" else 4_096
            if len(value) > max_length:
                errors.append(f"{name} exceeds {max_length} characters")
        if isinstance(value, list):
            maximum = 500 if name == "config_lines" else 200
            if len(value) > maximum:
                errors.append(f"{name} exceeds {maximum} items")
            if any(not isinstance(item, str) or not item.strip() or len(item) > 1_024 for item in value):
                errors.append(f"{name} must contain non-empty strings no longer than 1024 characters")
        if name in _ENUMS and isinstance(value, str) and value.lower() not in _ENUMS[name]:
            errors.append(f"{name} must be one of {sorted(_ENUMS[name])}")
        if name in _ENUMS and isinstance(value, str):
            value = value.lower()
        if name == "vni" and isinstance(value, int) and not 1 <= value <= 16_777_215:
            errors.append("vni must be between 1 and 16777215")
        if name == "vlan_id" and isinstance(value, int) and not 1 <= value <= 4_094:
            errors.append("vlan_id must be between 1 and 4094")
        if name == "route_type" and isinstance(value, int) and value not in {2, 3, 5}:
            errors.append("route_type must be one of 2, 3, or 5")
        if name == "grace_period_s" and isinstance(value, int) and not 0 <= value <= 3_600:
            errors.append("grace_period_s must be between 0 and 3600")
        if name == "prefix" and isinstance(value, str):
            try:
                ipaddress.ip_network(value, strict=False)
            except ValueError:
                errors.append("prefix must be a valid IPv4 or IPv6 network")
        compiled[name] = value
        provenance[name] = "user_explicit"

    config_values: list[str] = []
    if isinstance(compiled.get("config_text"), str):
        config_values.append(compiled["config_text"])
    if isinstance(compiled.get("config_lines"), list):
        config_values.extend(str(item) for item in compiled["config_lines"])
    if any(_FORBIDDEN_CONFIG.search(line) for line in config_values):
        errors.append("configuration contains a forbidden catastrophic command")
    if tool_name == "run_command" and isinstance(compiled.get("command"), str):
        if not _READ_ONLY_COMMAND.match(compiled["command"]):
            errors.append("run_command accepts only show/display commands")
    if tool_name == "edit_device_config" and not compiled.get("config_lines") and not compiled.get("changes"):
        missing.append("config_lines or changes")

    tags = {str(item) for item in (metadata.get("tags") or [])}
    # The Service MCP owns business identifiers. A user can legitimately have
    # no Containerlab endpoint; cross-layer reconciliation reports that drift
    # instead of parameter compilation rejecting the business identity.
    known = {} if {"mcp", "service"}.issubset(tags) else _known_entities(profile, mode)
    for field_name, values in known.items():
        if field_name not in compiled or not values:
            continue
        candidate = str(compiled[field_name])
        if candidate not in values:
            errors.append(
                f"{field_name}={candidate!r} does not resolve uniquely in current inventory; "
                f"known values: {', '.join(sorted(values))}"
            )
    if "device_ids" in compiled and known.get("device_id"):
        unresolved = sorted(
            str(item) for item in compiled["device_ids"]
            if str(item) not in known["device_id"]
        )
        if unresolved:
            errors.append("device_ids contain unresolved inventory values: " + ", ".join(unresolved))

    targets: list[str] = []
    for name in _TARGET_KEYS:
        value = compiled.get(name)
        if isinstance(value, list):
            targets.extend(f"{name}:{item}" for item in value)
        elif value not in (None, ""):
            targets.append(f"{name}:{value}")
    if not targets:
        targets.append(f"tool:{tool_name}")
    return CompilationResult(
        not missing and not errors,
        compiled,
        provenance,
        tuple(sorted(set(targets))),
        tuple(sorted(set(missing))),
        tuple(errors),
    )


def assess_risk_decision(
    tool_name: str,
    metadata: dict[str, Any],
    arguments: dict[str, Any],
    *,
    evidence_confidence: float = 1.0,
) -> RiskAssessment:
    """Return a versioned, explainable EXECUTE/ASK_HUMAN/REJECT decision.

    This replaces the former tool-name-only heuristic.  A reviewed L0 contract
    may still require human approval even when the generic risk policy says an
    operation is eligible for automatic execution; policy can only narrow.
    """
    action = str(metadata.get("action_type", "read_only"))
    has_compensation = bool(metadata.get("compensator") or metadata.get("rollback_tool"))
    reversibility = (
        Reversibility.STRONG
        if action == "reversible" and has_compensation
        else Reversibility.CONDITIONAL
        if action == "reversible"
        else Reversibility.IRREVERSIBLE
        if action in {"destructive", "irreversible"}
        else Reversibility.STRONG
    )
    targets = arguments.get("device_ids")
    blast_radius = len(targets) if isinstance(targets, list) else 1
    if bool(arguments.get("force")) or tool_name == "delete_resource":
        reversibility = Reversibility.IRREVERSIBLE
    service_criticality = int(metadata.get("service_criticality", 0) or 0)
    if arguments.get("environment") == "prod":
        service_criticality = max(service_criticality, 3)
    historical_success = float(metadata.get("historical_success_rate", 1.0) or 0.0)
    return RiskPolicy().evaluate(RiskFactors(
        change_scope=max(1, len(arguments)),
        blast_radius=max(1, blast_radius),
        evidence_confidence=evidence_confidence,
        reversibility=reversibility,
        historical_success=historical_success,
        service_criticality=service_criticality,
    ))


def assess_risk(
    tool_name: str,
    metadata: dict[str, Any],
    arguments: dict[str, Any],
) -> tuple[RiskLevel, tuple[str, ...]]:
    """Compatibility projection used by PreparedPlan schema v10."""
    if not bool(metadata.get("hitl")) and str(metadata.get("action_type", "read_only")) == "read_only":
        return RiskLevel.LOW, ("autonomy_decision=execute", "read-only operation")
    assessment = assess_risk_decision(tool_name, metadata, arguments)
    # Preserve the externally visible critical classification for destructive
    # or force-bypass operations.  The new autonomy decision is intentionally
    # orthogonal: a critical operation may be routed to a human instead of
    # being rejected at plan-compilation time.
    if tool_name == "delete_resource" or bool(arguments.get("force")):
        level = RiskLevel.CRITICAL
    elif assessment.decision == AutonomyDecision.REJECT:
        level = RiskLevel.CRITICAL
    elif assessment.decision == AutonomyDecision.ASK_HUMAN:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.MEDIUM
    compatibility_reasons = (
        ("irreversible deletion or force bypass",)
        if level == RiskLevel.CRITICAL
        and (tool_name == "delete_resource" or bool(arguments.get("force")))
        else ()
    )
    return level, (
        f"autonomy_decision={assessment.decision.value}",
        f"risk_score={assessment.score}",
        *compatibility_reasons,
        *assessment.reasons,
    )
