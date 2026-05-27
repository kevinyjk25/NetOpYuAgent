"""
profiles/dc/skills.py — Data-center fabric SOP skills
======================================================

Business skills (SOPs) for data-center network operations. Uses the same
flat-dict schema as the LAN profile (consumed by SkillCatalogService.
register_all, which requires the `name` field).
"""
from __future__ import annotations
from typing import Any

SKILLS: dict[str, dict[str, Any]] = {
    "dc_evpn_troubleshoot": {
        "name": "DC EVPN Troubleshoot",
        "purpose": "Diagnose BGP EVPN control-plane issues in the spine/leaf fabric",
        "risk_level": "low",
        "requires_hitl": False,
        "tags": ["dc", "evpn", "bgp", "troubleshoot"],
        "description": (
            "Diagnose BGP EVPN control-plane issues (flapping neighbors, "
            "missing routes, VTEP reachability) across the fabric. Enumerate "
            "nodes, check neighbor state, verify route presence."
        ),
        "parameters": {
            "node":   "Affected leaf/spine node id (e.g. leaf-1)",
            "target": "Optional MAC/IP to look up in the EVPN control plane",
        },
        "returns": "EVPN neighbor health + route presence findings",
        "tool_deps": ["dc_list_fabric", "dc_bgp_evpn_status", "dc_evpn_route_lookup"],
        "examples": [
            {"args": {"node": "leaf-1"}, "note": "Check EVPN on leaf-1"},
        ],
    },
    "dc_path_troubleshoot": {
        "name": "DC Path Troubleshoot",
        "purpose": "Trace connectivity between two endpoints across the VXLAN fabric",
        "risk_level": "low",
        "requires_hitl": False,
        "tags": ["dc", "path", "vxlan", "troubleshoot"],
        "description": (
            "Trace and diagnose connectivity between two endpoints across the "
            "VXLAN fabric: confirm segments/VNIs, trace underlay+overlay path, "
            "check inter-VRF leaking, inspect any VIP involved."
        ),
        "parameters": {
            "src": "Source endpoint IP",
            "dst": "Destination endpoint IP",
        },
        "returns": "Hop-by-hop path with VNI/VTEP/RTT + diagnosis",
        "tool_deps": ["dc_vxlan_vni_lookup", "dc_fabric_path_trace", "dc_loadbalancer_pools"],
        "examples": [
            {"args": {"src": "10.1.0.11", "dst": "10.3.0.31"}, "note": "web->db path"},
        ],
    },
    "dc_lb_health_check": {
        "name": "DC Load Balancer Health Check",
        "purpose": "Assess load-balancer pool health and identify down/draining members",
        "risk_level": "low",
        "requires_hitl": False,
        "tags": ["dc", "loadbalancer", "health"],
        "description": (
            "Assess load-balancer pool + member health, and for any down "
            "member trace fabric reachability and confirm EVPN presence."
        ),
        "parameters": {
            "pool": "Optional pool name filter: web-prod|app-prod|api-prod",
        },
        "returns": "Pool membership health + reachability findings for down members",
        "tool_deps": ["dc_loadbalancer_pools", "dc_fabric_path_trace", "dc_evpn_route_lookup"],
        "examples": [
            {"args": {"pool": "web-prod"}, "note": "Check web-prod pool health"},
        ],
    },

    "dc_app_access_diagnose": {
        "name": "DC Application Access Diagnose",
        "purpose": (
            "Diagnose why a user cannot access a data-center APPLICATION — this "
            "is an application-layer ACCESS/PERMISSION problem (RBAC/ACL), NOT a "
            "network path / VNI / BGP-EVPN problem."
        ),
        "risk_level": "low",
        "requires_hitl": False,
        "tags": ["dc", "application", "access", "permission", "rbac", "troubleshoot"],
        "description": (
            "Use this when a user reports they cannot access / are denied / cannot "
            "reach a specific application (e.g. CRM). The cause is almost always "
            "application-layer access control, so check PERMISSIONS FIRST and only "
            "fall back to network diagnostics if access is confirmed granted. "
            "Procedure, in order: "
            "(1) dc_check_user_app_access(user_id, app_id) — does the user hold any "
            "role granting access? This is the primary check and usually reveals the "
            "root cause directly. "
            "(2) If denied, dc_get_app_acl(app_id) — inspect which roles grant access "
            "and who holds them, to decide the right role to assign. "
            "(3) If the user legitimately needs access, dc_grant_app_access(user_id, "
            "app_id, role) to grant the missing role — this is a destructive change "
            "and requires operator approval (HITL). "
            "Only if access is already granted but the app is still unreachable should "
            "you escalate to network-path diagnostics (dc_path_troubleshoot)."
        ),
        "parameters": {
            "user_id": "User reporting the access failure (e.g. alice)",
            "app_id":  "Application id (e.g. crm, wiki, payroll, grafana)",
        },
        "returns": "Access verdict (allowed/denied), roles held, root cause, remediation",
        "tool_deps": ["dc_check_user_app_access", "dc_get_app_acl", "dc_list_apps", "dc_grant_app_access"],
        "examples": [
            {"args": {"user_id": "alice", "app_id": "crm"}, "note": "Why can't alice reach CRM"},
        ],
    },
}
