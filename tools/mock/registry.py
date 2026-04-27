"""
tools/mock/registry.py
──────────────────────
Mock tools — available in mock mode ONLY.

Each entry declares: description, parameters, returns, hitl flag, tags.
Callables live in tools/mock_tools.py (unchanged).
This registry is the prompt-facing declaration; mock_tools.py is the implementation.

Key principle: the agent prompt is built from THIS dict at runtime.
No tool name is hardcoded in the system prompt template.
"""
from __future__ import annotations
from typing import Any

TOOLS: dict[str, dict[str, Any]] = {
    "list_devices": {
        "description": "List all network devices. Filter by type or site tag.",
        "parameters":  {"type": "Device type: switch|router|ap|firewall (optional)", "tag": "Site tag filter (optional)"},
        "returns":     "Table of device id, model, role, site, IP",
        "hitl":        False,
        "tags":        ["inventory", "discovery"],
    },
    "list_interfaces": {
        "description": "List interfaces for a specific device with status and IP.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Interface table with status, IP, speed",
        "hitl":        False,
        "tags":        ["inventory", "network"],
    },
    "get_device_config": {
        "description": "Retrieve running configuration for a device or one section.",
        "parameters":  {"device_id": "Device identifier", "section": "Config section (optional): radius|ntp|vlan|interface"},
        "returns":     "Device configuration text",
        "hitl":        False,
        "tags":        ["config", "read"],
    },
    "validate_device_config": {
        "description": "Validate device configuration and return a list of issues.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Validation report: issues found, severity, recommendations",
        "hitl":        False,
        "tags":        ["config", "validation"],
    },
    "edit_device_config": {
        "description": "Apply a configuration change to a device. Requires HITL approval.",
        "parameters":  {"device_id": "Device identifier", "section": "Config section to change", "changes": "Change payload dict", "reason": "Reason for change (audit log)"},
        "returns":     "Confirmation of config push with diff",
        "hitl":        True,
        "tags":        ["config", "write", "destructive"],
    },
    "diff_device_config": {
        "description": "Show uncommitted configuration changes (running vs startup).",
        "parameters":  {"device_id": "Device identifier", "section": "Section to diff (optional)"},
        "returns":     "Unified diff of running vs startup config",
        "hitl":        False,
        "tags":        ["config", "read"],
    },
    "device_info": {
        "description": "Get hardware facts: model, firmware, uptime, serial number.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Hardware facts table",
        "hitl":        False,
        "tags":        ["inventory", "hardware"],
    },
    "syslog_search": {
        "description": "Search syslog entries across devices. Supports glob host patterns.",
        "parameters":  {"host": "Device name or glob (e.g. 'radius-*')", "keyword": "Search term", "severity": "Error level: error|warning|info", "lines": "Max lines to return (default 50)"},
        "returns":     "Matching syslog lines with timestamp, host, severity, message",
        "hitl":        False,
        "tags":        ["logs", "diagnostics"],
    },
    "prometheus_query": {
        "description": "Run a PromQL query against the metrics store.",
        "parameters":  {"query": "PromQL expression", "duration": "Time range (e.g. '5m', '1h')"},
        "returns":     "Time series data as table",
        "hitl":        False,
        "tags":        ["metrics", "monitoring"],
    },
    "netflow_dump": {
        "description": "Dump NetFlow traffic records for a site or all sites.",
        "parameters":  {"site": "Site name or 'all'", "top_n": "Limit to top N flows by bytes"},
        "returns":     "Stored NetFlow records [STORED:] — use read_stored_result to page",
        "hitl":        False,
        "tags":        ["traffic", "security"],
    },
    "dns_lookup": {
        "description": "Resolve a hostname or reverse-lookup an IP.",
        "parameters":  {"hostname": "FQDN or IP address"},
        "returns":     "DNS records: A, PTR, CNAME",
        "hitl":        False,
        "tags":        ["network", "diagnostics"],
    },
    "alert_summary": {
        "description": "Retrieve active alerts from the monitoring system.",
        "parameters":  {"severity": "Filter: critical|warning|info (optional)", "site": "Site filter (optional)"},
        "returns":     "Alert table with name, severity, duration, affected devices",
        "hitl":        False,
        "tags":        ["monitoring", "alerts"],
    },
    "service_health": {
        "description": "Check health status of a named service across environments.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Health check results: status, latency, pod counts",
        "hitl":        False,
        "tags":        ["services", "health"],
    },
    "restart_service": {
        "description": "Perform a rolling restart of a production service. Requires HITL approval.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Restart status with pod counts and health check",
        "hitl":        True,
        "tags":        ["services", "destructive"],
    },
    "rollback_service": {
        "description": "Roll back a service to a previous version. Requires HITL approval.",
        "parameters":  {"service": "Service name", "version": "Target version (e.g. '3.2.1')", "environment": "prod|staging|dev"},
        "returns":     "Rollback status with pod counts and health check",
        "hitl":        True,
        "tags":        ["services", "destructive"],
    },
}
