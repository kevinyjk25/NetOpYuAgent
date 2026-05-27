"""
profiles/lan/skills.py — Enterprise LAN SOP skills
===================================================

Business skills (standard operating procedures) for enterprise LAN ops.
Migrated 2026-05 from skills/mock/registry.py (profile refactor).
"""
from __future__ import annotations
from typing import Any

SKILLS: dict[str, dict[str, Any]] = {
    "syslog_search": {
        "name":        "Syslog Search",
        "purpose":     "Search syslog entries across network devices",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["logs", "diagnostics"],
        "description": "Queries the mock syslog aggregator for matching entries.",
        "parameters":  {"host": "Device name or glob", "keyword": "Search term", "severity": "error|warning|info"},
        "returns":     "Matching syslog lines",
        "tool_deps":   ["syslog_search"],
        "examples":    [{"args": {"host": "ap-01", "severity": "error"}, "note": "Find errors on ap-01"}],
    },
    "netflow_analysis": {
        "name":        "NetFlow Analysis",
        "purpose":     "Analyse NetFlow traffic for anomalies and top talkers",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["traffic", "security"],
        "description": "Dumps and analyses NetFlow records. For large datasets, pages through stored results.",
        "parameters":  {"site": "Site name or 'all'"},
        "returns":     "Traffic summary with anomaly indicators",
        "tool_deps":   ["netflow_dump", "read_stored_result"],
        "examples":    [{"args": {"site": "all"}, "note": "Analyse all-site traffic"}],
    },
    "prometheus_query": {
        "name":        "Prometheus Query",
        "purpose":     "Query metrics from the mock Prometheus store",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["metrics", "monitoring"],
        "description": "Runs PromQL queries and returns time series data.",
        "parameters":  {"query": "PromQL expression", "duration": "Time window"},
        "returns":     "Time series table",
        "tool_deps":   ["prometheus_query"],
        "examples":    [],
    },
    "alert_summary": {
        "name":        "Alert Summary",
        "purpose":     "Summarise active monitoring alerts",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["monitoring", "alerts"],
        "description": "Retrieves and groups active alerts by severity and device.",
        "parameters":  {"severity": "Filter severity", "site": "Filter by site"},
        "returns":     "Grouped alert table",
        "tool_deps":   ["alert_summary"],
        "examples":    [],
    },
    "service_health": {
        "name":        "Service Health Check",
        "purpose":     "Check health of a named mock service",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["services", "health"],
        "description": "Checks service health across environments.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Health status with latency and pod counts",
        "tool_deps":   ["service_health"],
        "examples":    [],
    },
    "restart_service": {
        "name":        "Service Restart",
        "purpose":     "Rolling restart of a mock production service",
        "risk_level":  "high",
        "requires_hitl": True,
        "tags":        ["services", "destructive"],
        "description": "Performs a rolling restart. Always requires HITL approval.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Restart status",
        "tool_deps":   ["restart_service"],
        "examples":    [],
    },
    "rollback_service": {
        "name":        "Service Rollback",
        "purpose":     "Roll back a mock service to a previous version",
        "risk_level":  "high",
        "requires_hitl": True,
        "tags":        ["services", "destructive"],
        "description": "Rolls back to target version. Always requires HITL approval.",
        "parameters":  {"service": "Service name", "version": "Target version", "environment": "prod|staging|dev"},
        "returns":     "Rollback status",
        "tool_deps":   ["rollback_service"],
        "examples":    [],
    },

    "lan_user_access_diagnose": {
        "name":        "LAN User Access Diagnose",
        "purpose": (
            "Diagnose why a user cannot access an application. Determine whether "
            "the cause is LAN-side (identity / network admission) or "
            "application-layer (which lives on the DATA CENTER and must be "
            "delegated). Do NOT assume a network-path problem first."
        ),
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["user", "access", "permission", "identity", "nac", "troubleshoot"],
        "description": (
            "Use this when a user reports they cannot access / are denied / cannot "
            "reach an application. Access failures are usually identity or "
            "permission problems, not network-reachability problems — check those "
            "FIRST and treat fabric/path diagnostics as a last resort. "
            "Procedure, in order: "
            "(1) get_user_access(user_id) — is the user admitted on the LAN "
            "(RADIUS / 802.1X / NAC / VLAN)? "
            "(2) If blocked, check_nac_policy(user_id) explains the NAC decision; "
            "restoring admission is grant_user_access (destructive, needs approval). "
            "(3) If the user IS fully admitted on the LAN, the LAN is not the cause: "
            "the problem is application-layer access control, which is owned by the "
            "DATA CENTER. DELEGATE the application-access check to the dc agent "
            "(it owns application RBAC/ACL via dc_app_access_diagnose). Describe the "
            "task plainly as 'check whether user <id> has access permission to "
            "application <app>' — do NOT pre-frame it as a VNI / overlay / BGP-EVPN "
            "routing problem. "
            "(4) Only if BOTH LAN admission and DC application access are confirmed "
            "OK should network-path reachability be investigated."
        ),
        "parameters": {
            "user_id": "User reporting the access failure (e.g. alice)",
            "app":     "Application the user cannot reach (e.g. CRM)",
        },
        "returns": "LAN admission verdict + whether to delegate app-access check to DC",
        "tool_deps": ["list_users", "get_user_access", "check_nac_policy", "grant_user_access"],
        "examples": [
            {"args": {"user_id": "alice", "app": "CRM"}, "note": "Why can't alice reach CRM"},
        ],
    },
}
