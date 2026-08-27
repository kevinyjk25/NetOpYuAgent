"""
profiles/lan/tool_meta.py — Prompt-facing metadata for LAN tools
=================================================================

Declares description / parameters / returns / hitl / action_type / tags for
each enterprise-LAN tool. This is what the LLM sees in its tool section.
Keys MUST match the callables in profiles/lan/tools.py.

Migrated 2026-05 from tools/mock/registry.py (profile refactor).
"""
from __future__ import annotations
from typing import Any

TOOLS: dict[str, dict[str, Any]] = {
    "list_devices": {
        "description": "List all network devices. Filter by type or site tag.",
        "parameters":  {"type": "Device type: switch|router|ap|firewall (optional)", "tag": "Site tag filter (optional)"},
        "returns":     "Table of device id, model, role, site, IP",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["inventory", "discovery"],
    },
    "list_interfaces": {
        "description": "List interfaces for a specific device with status and IP.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Interface table with status, IP, speed",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["inventory", "network"],
    },
    "get_device_config": {
        "description": "Retrieve running configuration for a device or one section.",
        "parameters":  {"device_id": "Device identifier", "section": "Config section (optional): radius|ntp|vlan|interface"},
        "returns":     "Device configuration text",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["config", "read"],
    },
    "validate_device_config": {
        "description": "Validate device configuration and return a list of issues.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Validation report: issues found, severity, recommendations",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["config", "validation"],
    },
    "edit_device_config": {
        "description": "Apply a configuration change to a device. Requires HITL approval.",
        "parameters":  {"device_id": "Device identifier", "section": "Config section to change", "changes": "Change payload (object with field-value pairs OR list of IOS lines)", "config_lines": "list of IOS-style config commands (alternative to section+changes)", "reason": "Reason for change (audit log)"},
        "required":    ["device_id"],
        "returns":     "Confirmation of config push with diff",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["config", "write", "destructive"],
        "example":     {"device_id": "ap-01", "section": "radius", "changes": {"timeout": 3}, "reason": "fix RADIUS timeout"},
        "examples":    [
            {"device_id": "ap-01", "section": "radius", "changes": {"timeout": 3}, "reason": "fix RADIUS timeout"},
            {"device_id": "ap-01", "section": "ntp", "changes": {"servers": ["10.0.0.5"]}, "reason": "add NTP server"},
            {"device_id": "ap-01", "config_lines": ["radius-server timeout 3"], "reason": "alt format"},
        ],
    },
    "diff_device_config": {
        "description": "Show uncommitted configuration changes (running vs startup).",
        "parameters":  {"device_id": "Device identifier", "section": "Section to diff (optional)"},
        "returns":     "Unified diff of running vs startup config",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["config", "read"],
    },
    "device_info": {
        "description": "Get hardware facts: model, firmware, uptime, serial number.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Hardware facts table",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["inventory", "hardware"],
    },
    "syslog_search": {
        "description": "Search syslog entries across devices. Supports glob host patterns.",
        "parameters":  {"host": "Device name or glob (e.g. 'radius-*')", "keyword": "Search term", "severity": "Error level: error|warning|info", "lines": "Max lines to return (default 50)"},
        "returns":     "Matching syslog lines with timestamp, host, severity, message",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["logs", "diagnostics"],
    },
    "prometheus_query": {
        "description": "Run a PromQL query against the metrics store.",
        "parameters":  {"query": "PromQL expression", "duration": "Time range (e.g. '5m', '1h')"},
        "returns":     "Time series data as table",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["metrics", "monitoring"],
    },
    "netflow_dump": {
        "description": "Dump NetFlow traffic records for a site or all sites.",
        "parameters":  {"site": "Site name or 'all'", "top_n": "Limit to top N flows by bytes"},
        "returns":     "Stored NetFlow records [STORED:] — use read_stored_result to page",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["traffic", "security"],
    },
    "dns_lookup": {
        "description": "Resolve a hostname or reverse-lookup an IP.",
        "parameters":  {"hostname": "FQDN or IP address"},
        "returns":     "DNS records: A, PTR, CNAME",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["network", "diagnostics"],
    },
    "alert_summary": {
        "description": "Retrieve active alerts from the monitoring system.",
        "parameters":  {"severity": "Filter: critical|warning|info (optional)", "site": "Site filter (optional)"},
        "returns":     "Alert table with name, severity, duration, affected devices",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["monitoring", "alerts"],
    },
    "service_health": {
        "description": "Check health status of a named service across environments.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Health check results: status, latency, pod counts",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["services", "health"],
    },
    "restart_service": {
        "description": "Perform a rolling restart of a production service. Requires HITL approval.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Restart status with pod counts and health check",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["services", "destructive"],
    },

    "push_config": {
        "description": "Push a configuration block to a device. DESTRUCTIVE — requires HITL approval.",
        "parameters":  {"device_id": "Target device", "config_text": "Raw config to apply", "dry_run": "If true, validate only"},
        "returns":     "Push status + diff summary",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["destructive", "config"],
    },
    "rollback_deploy": {
        "description": "Roll back a previous deploy. DESTRUCTIVE — requires HITL approval.",
        "parameters":  {"deploy_id": "Deploy identifier", "scope": "Optional scope filter"},
        "returns":     "Rollback status",
        "hitl":        True,
        "action_type": "reversible",
        "tags":        ["destructive", "deploy"],
    },
    "drain_node": {
        "description": "Drain a node — evict workloads. DESTRUCTIVE — requires HITL approval.",
        "parameters":  {"node_id": "Node to drain", "grace_period_s": "Grace period seconds (default 60)"},
        "returns":     "Drain status with evicted workload count",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["destructive", "node"],
    },
    "failover": {
        "description": "Trigger failover to standby. DESTRUCTIVE — requires HITL approval.",
        "parameters":  {"resource_id": "Resource to fail over", "target": "Target replica"},
        "returns":     "Failover status + new primary",
        "hitl":        True,
        "action_type": "reversible",
        "tags":        ["destructive", "ha"],
    },
    "delete_resource": {
        "description": "Delete a resource. DESTRUCTIVE — requires HITL approval.",
        "parameters":  {"resource_id": "Resource to delete", "force": "Skip dependency check"},
        "returns":     "Deletion status",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["destructive", "delete"],
    },
    "mock_operation_status": {
        "description": "Read structured state for local destructive-operation simulation verification.",
        "parameters": {
            "operation": {"type": "string", "description": "Optional operation name"},
            "resource_id": {"type": "string", "description": "Resource identifier"},
            "deploy_id": {"type": "string", "description": "Deployment identifier"},
            "node_id": {"type": "string", "description": "Node identifier"},
        },
        "returns": "Machine-readable simulator operation state",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["simulation", "verification", "read-only"],
    },
    "rollback_service": {
        "description": "Roll back a service to a previous version. Requires HITL approval.",
        "parameters":  {"service": "Service name", "version": "Target version (e.g. '3.2.1')", "environment": "prod|staging|dev"},
        "returns":     "Rollback status with pod counts and health check",
        "hitl":        True,
        "action_type": "reversible",
        "tags":        ["services", "destructive"],
    },
    "query_radius_logs": {
        "description": (
            "Query RADIUS auth logs for a user. ASYNC HITL DEMO — pushes "
            "approval request to ops queue (3-min SLA); agent proceeds with "
            "the assumed default 'permission_ok'. Real result arrives via "
            "soft-notify or next-turn confirmed_fact. Use to demonstrate "
            "H2 (fire-and-forget) HITL semantics."
        ),
        "parameters":  {"user_id": "User to look up", "minutes": "Time window in minutes (default 60)"},
        "returns":     "Immediately returns assumed permission_ok; real RADIUS result is injected as a fact when ops responds.",
        "hitl":        False,                # not a synchronous gate
        "hitl_mode":   "async_nonblocking",  # H2 marker for the policy layer
        "action_type": "read_only",
        "tags":        ["auth", "async-hitl", "demo"],
    },

    # ── User / network-access control (cross-agent HITL scenario, 2026-05) ──
    "list_users": {
        "description": "List enterprise users (id, name, department, status).",
        "parameters":  {"dept": "Filter by department: sales|support|finance (optional)"},
        "returns":     "Table of users",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["user", "identity", "inventory"],
    },
    "get_user_access": {
        "description": "Show a user's NETWORK admission state: RADIUS auth, 802.1X, NAC posture, VLAN. READ-ONLY. The LAN half of an access diagnosis — if the user is admitted here but an APP is unreachable, the cause is application-layer (DC).",
        "parameters":  {"user_id": "User identifier"},
        "required":    ["user_id"],
        "returns":     "Network admission state + ADMITTED/BLOCKED verdict",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["user", "access", "nac", "dot1x", "diagnose"],
        "example":     {"user_id": "alice"},
    },
    "check_nac_policy": {
        "description": "Explain which NAC policy a user hits and why (PERMIT/DENY). Read-only.",
        "parameters":  {"user_id": "User identifier"},
        "required":    ["user_id"],
        "returns":     "Matched NAC policy, posture checks, authorization result",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["user", "nac", "policy", "diagnose"],
    },
    "grant_user_access": {
        "description": "Restore/grant a user's network admission (RADIUS/802.1X/NAC/VLAN). DESTRUCTIVE — HITL-gated on the LAN side.",
        "parameters":  {"user_id": "User identifier", "reason": "Why (for audit)"},
        "required":    ["user_id"],
        "returns":     "Grant confirmation with applied admission changes",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["user", "access", "destructive"],
        "example":     {"user_id": "erin", "reason": "ticket #5102 — account reinstated"},
    },
    "revoke_user_access": {
        "description": "Revoke a user's network admission (quarantine). DESTRUCTIVE — HITL-gated on the LAN side.",
        "parameters":  {"user_id": "User identifier", "reason": "Why (for audit)"},
        "required":    ["user_id"],
        "returns":     "Revoke confirmation with applied admission changes",
        "hitl":        True,
        "action_type": "destructive",
        "tags":        ["user", "access", "destructive"],
        "example":     {"user_id": "alice", "reason": "offboarding"},
    },
}
