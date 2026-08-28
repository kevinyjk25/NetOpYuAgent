"""Typed evidence parsing and snapshot comparison shared by L1 adapters."""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import sha256_json


ERROR_RE = re.compile(
    r"(?:\[error(?::|\])|\[critical\]|\A[^\n]{0,100}\brequires\b|"
    r"\bunknown\s+(?:device|user|edge|tunnel|application|fabric)|"
    r"\bchange aborted\b|\boperation failed\b)",
    re.IGNORECASE,
)


def render(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def failed_output(value: str) -> bool:
    rendered = value.strip()
    if not rendered or ERROR_RE.search(rendered):
        return True
    try:
        decoded = json.loads(rendered)
    except (json.JSONDecodeError, TypeError):
        return False
    # Structured providers use ``ok`` as their protocol-level semantic
    # outcome. Treat an explicit false as failure even when transport/MCP
    # itself succeeded; prose/error-regex checks alone are insufficient.
    return isinstance(decoded, dict) and decoded.get("ok") is False


def bounded(value: str, limit: int = 4096) -> str:
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def typed_evidence(tool_name: str, rendered: str) -> dict[str, Any]:
    """Return bounded typed facts and a digest; rendered prose is never proof."""
    facts: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    if tool_name == "get_device_config":
        # Configuration preflight comparisons must ignore volatile headers but
        # must remain sensitive to every effective configuration line.
        from network_lab.containerlab import normalize_frr_config

        normalized = normalize_frr_config(rendered)
        facts = {
            "normalized_config_digest": sha256_json(normalized),
            "normalized_line_count": len(normalized.splitlines()),
        }
    elif tool_name == "get_user_access":
        fields = {
            key: match.group(1).strip()
            for key, pattern in {
                "radius": r"^\s*RADIUS auth\s*:\s*(.+)$",
                "dot1x": r"^\s*802\.1X\s*:\s*(.+)$",
                "nac": r"^\s*NAC posture\s*:\s*(.+)$",
                "vlan": r"^\s*VLAN\s*:\s*(.+)$",
            }.items()
            if (match := re.search(pattern, rendered, re.MULTILINE)) is not None
        }
        vlan = fields.get("vlan")
        if vlan is not None:
            fields["vlan"] = int(vlan) if vlan.isdigit() else None
        facts = {**fields, "admitted": "✅ ADMITTED" in rendered}
    elif tool_name == "check_nac_policy":
        facts = {"permit": bool(re.search(r"^\s*result\s*:\s*PERMIT\s*$", rendered, re.MULTILINE))}
    elif tool_name == "dc_check_user_app_access":
        roles = re.search(r"^\s*via roles\s*:\s*(.+)$", rendered, re.MULTILINE)
        facts = {
            "allowed": "✅ ALLOWED" in rendered,
            "roles": sorted(item.strip() for item in roles.group(1).split(",")) if roles else [],
        }
    elif tool_name == "wan_tunnel_status":
        tunnels: dict[str, dict[str, str]] = {}
        for match in re.finditer(
            r"^(tun-\S+)\s+(\S+)\s+(\S+)\s+(mpls|broadband|lte)\s+(.+)$",
            rendered, re.MULTILINE,
        ):
            tunnels[match.group(1)] = {
                "src": match.group(2), "dst": match.group(3),
                "transport": match.group(4), "state": match.group(5).strip(),
            }
        facts = {"tunnels": tunnels}
    elif tool_name == "service_health":
        status = re.search(r"^\s*Status\s*:\s*(\S+)", rendered, re.MULTILINE)
        replicas = re.search(r"^\s*Replicas\s*:\s*(\d+)/(\d+)", rendered, re.MULTILINE)
        facts = {
            "status": status.group(1).lower() if status else None,
            "replicas_ready": int(replicas.group(1)) if replicas else None,
            "replicas_desired": int(replicas.group(2)) if replicas else None,
        }
    elif tool_name == "dc_bgp_evpn_status":
        facts = {
            "readable": rendered.startswith("BGP EVPN status on "),
            "flapping_neighbors": len(re.findall(r"Idle \(flapping\)", rendered)),
        }
    elif tool_name in {"mock_operation_status", "dc_get_applied_config"}:
        try:
            decoded = json.loads(rendered)
            facts = decoded if isinstance(decoded, dict) else {"invalid_shape": True}
        except json.JSONDecodeError:
            facts = {"invalid_json": True}
    elif tool_name == "lab_get_access_vlan":
        try:
            decoded = json.loads(rendered)
        except json.JSONDecodeError:
            facts = {"invalid_json": True}
        else:
            if not isinstance(decoded, dict):
                facts = {"invalid_shape": True}
            else:
                # Only stable, state-defining fields participate in approval
                # drift checks and exact rollback comparison.
                facts = {
                    "ok": decoded.get("ok") is True,
                    "device_id": decoded.get("device_id"),
                    "interface": decoded.get("interface"),
                    "mode": decoded.get("mode"),
                    "current_vlan": decoded.get("current_vlan"),
                    "bridge": decoded.get("bridge"),
                    "vlans": decoded.get("vlans"),
                }
    elif tool_name in {
        "access_policy_get_entitlement", "application_check_access",
    }:
        try:
            decoded = json.loads(rendered)
        except json.JSONDecodeError:
            facts = {"ok": False, "invalid_json": True}
        else:
            facts = {
                "ok": decoded.get("ok") is True,
                "user_id": decoded.get("user_id"),
                "app_id": decoded.get("app_id"),
                "roles": sorted(decoded.get("roles") or []),
                "allowed": decoded.get("allowed") is True,
            }
            extra["concurrency_token"] = decoded.get("revision")
    elif tool_name == "platform_get_service_health":
        try:
            decoded = json.loads(rendered)
        except json.JSONDecodeError:
            facts = {"ok": False, "invalid_json": True}
        else:
            facts = {
                "ok": decoded.get("ok") is True,
                "service": decoded.get("service"),
                "environment": decoded.get("environment"),
                "status": decoded.get("status"),
                "version": decoded.get("version"),
                "replicas_ready": decoded.get("replicas_ready"),
                "replicas_desired": decoded.get("replicas_desired"),
            }
            extra["concurrency_token"] = decoded.get("revision")
    elif tool_name == "network_get_app_enforcement":
        try:
            decoded = json.loads(rendered)
        except json.JSONDecodeError:
            facts = {"ok": False, "invalid_json": True}
        else:
            facts = {
                "ok": decoded.get("ok") is True,
                "user_id": decoded.get("user_id"),
                "app_id": decoded.get("app_id"),
                "allowed": decoded.get("allowed") is True,
                "source_endpoint": decoded.get("source_endpoint"),
                "application_endpoint": decoded.get("application_endpoint"),
                "implementation": decoded.get("implementation"),
            }
    return {
        "digest": sha256_json(rendered), "bytes": len(rendered), "facts": facts, **extra,
    }


def same_snapshot(before: Any, after: Any) -> bool:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return before == after
    before_facts, after_facts = before.get("facts"), after.get("facts")
    if before_facts or after_facts:
        return before_facts == after_facts
    if "digest" in before or "digest" in after:
        return before.get("digest") == after.get("digest")
    return before == after
