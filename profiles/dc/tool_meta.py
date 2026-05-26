"""
profiles/dc/tool_meta.py — Prompt-facing metadata for DC fabric tools
======================================================================

Declares the data-center fabric tools the LLM sees. Keys MUST match the
callables in profiles/dc/tools.py.
"""
from __future__ import annotations
from typing import Any

TOOLS: dict[str, dict[str, Any]] = {
    "dc_list_fabric": {
        "description": "List data-center fabric nodes (spine / leaf / border-leaf) with model, ASN, site.",
        "parameters":  {"role": "Filter by role: spine|leaf|border-leaf (optional)"},
        "returns":     "Table of fabric node id, role, model, ASN, site",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "fabric", "inventory"],
    },
    "dc_bgp_evpn_status": {
        "description": "Show BGP EVPN neighbor + route status on a fabric node. Flags flapping neighbors.",
        "parameters":  {"node": "Fabric node id (e.g. leaf-1, spine-2)"},
        "returns":     "EVPN neighbor table: state, up/down, route count, prefixes received",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "bgp", "evpn", "control-plane"],
        "example":     {"node": "leaf-1"},
    },
    "dc_vxlan_vni_lookup": {
        "description": "Look up VXLAN VNI ↔ VLAN / segment / VRF / anycast-gateway mappings.",
        "parameters":  {"vni": "VNI number (optional)", "segment": "Segment name filter (optional)"},
        "returns":     "VNI mapping table",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "vxlan", "overlay", "vni"],
        "example":     {"vni": 10100},
    },
    "dc_loadbalancer_pools": {
        "description": "Show load-balancer pool + member health (up / down / draining).",
        "parameters":  {"pool": "Pool name filter (optional): web-prod|app-prod|api-prod"},
        "returns":     "Pool membership with per-member health",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "loadbalancer", "health"],
    },
    "dc_fabric_path_trace": {
        "description": "Trace the underlay + overlay path between two endpoints across the fabric (ECMP, VNI, VTEP, RTT).",
        "parameters":  {"src": "Source endpoint IP", "dst": "Destination endpoint IP"},
        "returns":     "Hop-by-hop path with VNI / VTEP / RTT / MTU",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "fabric", "path", "troubleshoot"],
        "example":     {"src": "10.1.0.11", "dst": "10.3.0.31"},
    },
    "dc_evpn_route_lookup": {
        "description": "Look up a MAC or IP in the BGP EVPN control plane (Type-2 / Type-5 routes, VTEP next-hop).",
        "parameters":  {"mac": "MAC address (optional)", "ip": "IP address (optional)"},
        "returns":     "EVPN route detail: type, VNI, next-hop VTEP, ESI",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "evpn", "route", "mac", "ip"],
    },
    "dc_config_push": {
        "description": "Push configuration to a fabric node. DESTRUCTIVE — takes a snapshot, HITL-gated.",
        "parameters":  {
            "node":         "Fabric node id",
            "config_lines": "List of config statements (or newline string)",
            "reason":       "Why this change is being made (for audit)",
        },
        "required":    ["node", "config_lines"],
        "returns":     "Applied config diff + snapshot/rollback id + re-convergence time",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["dc", "fabric", "config", "destructive"],
        "example":     {"node": "leaf-1", "config_lines": ["interface Ethernet1/1", "  mtu 9216"], "reason": "jumbo frames for storage VLAN"},
    },

    # ── Application access control (cross-agent HITL scenario, 2026-05) ──
    "dc_list_apps": {
        "description": "List data-center applications (id, name, VIP, owner team, tier).",
        "parameters":  {"tier": "Filter by tier: gold|silver|bronze (optional)"},
        "returns":     "Table of applications",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "application", "inventory"],
    },
    "dc_get_app_acl": {
        "description": "Show the access-control list (roles → members) for one application.",
        "parameters":  {"app_id": "Application id (e.g. crm, wiki, payroll, grafana)"},
        "required":    ["app_id"],
        "returns":     "Roles and their member users for the application",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "application", "access", "rbac"],
    },
    "dc_check_user_app_access": {
        "description": "Diagnose whether a specific user can access a specific application. READ-ONLY — the core cross-agent diagnostic when a user reports an application is unreachable.",
        "parameters":  {"user_id": "User identifier", "app_id": "Application id"},
        "required":    ["user_id", "app_id"],
        "returns":     "ALLOWED/DENIED + roles held + remediation hint",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["dc", "application", "access", "diagnose"],
        "example":     {"user_id": "alice", "app_id": "crm"},
    },
    "dc_grant_app_access": {
        "description": "Grant a user a role on an application. DESTRUCTIVE — HITL-gated; approval happens on the DC operator's console.",
        "parameters":  {"user_id": "User identifier", "app_id": "Application id", "role": "Role to grant (optional; defaults to the app's base role)", "reason": "Why (for audit)"},
        "required":    ["user_id", "app_id"],
        "returns":     "Grant confirmation with role + audit reason",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["dc", "application", "access", "destructive"],
        "example":     {"user_id": "alice", "app_id": "crm", "role": "sales-rep", "reason": "ticket #4821 — sales onboarding"},
    },
    "dc_revoke_app_access": {
        "description": "Revoke a user's access to an application. DESTRUCTIVE — HITL-gated on the DC side.",
        "parameters":  {"user_id": "User identifier", "app_id": "Application id", "reason": "Why (for audit)"},
        "required":    ["user_id", "app_id"],
        "returns":     "Revoke confirmation listing roles removed",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["dc", "application", "access", "destructive"],
        "example":     {"user_id": "alice", "app_id": "crm", "reason": "offboarding"},
    },
}
