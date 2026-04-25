"""
tools/pragmatic/registry.py
───────────────────────────
Real network device tools — available in pragmatic mode ONLY.

Callables live in tools/pragmatic_tools.py.
This registry is the single source of truth for what the agent knows about each tool.
The prompt is built dynamically from this dict — no tool name is hardcoded anywhere.
"""
from __future__ import annotations
from typing import Any

TOOLS: dict[str, dict[str, Any]] = {
    "list_devices": {
        "description": "List real network devices from inventory. Returns live data.",
        "parameters":  {"type": "Filter by device type (optional)", "tag": "Filter by site tag (optional)"},
        "returns":     "Device table: id, model, role, site, management IP",
        "hitl":        False,
        "tags":        ["inventory", "discovery"],
    },
    "get_device_status": {
        "description": "Get live operational status of a device: CPU, memory, uptime, interface summary.",
        "parameters":  {"device_id": "Device identifier from list_devices"},
        "returns":     "Status dict: cpu_pct, mem_pct, uptime, interface counts",
        "hitl":        False,
        "tags":        ["monitoring", "status"],
    },
    "get_device_config": {
        "description": "Retrieve running configuration from a real device via SSH/NAPALM.",
        "parameters":  {"device_id": "Device identifier", "section": "Config section (optional): radius|ntp|vlan|bgp"},
        "returns":     "Device configuration text",
        "hitl":        False,
        "tags":        ["config", "read"],
    },
    "validate_device_config": {
        "description": "Validate device configuration against compliance rules. Returns findings.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Validation report: issues, severity, recommendations",
        "hitl":        False,
        "tags":        ["config", "validation"],
    },
    "edit_device_config": {
        "description": "Push configuration change to a real device. Requires HITL approval.",
        "parameters":  {"device_id": "Device identifier", "section": "Section to change", "changes": "Change dict", "reason": "Audit reason"},
        "returns":     "Push result with diff and rollback instructions",
        "hitl":        True,
        "tags":        ["config", "write", "destructive"],
    },
    "get_syslog": {
        "description": "Retrieve recent syslog entries from a device via SSH.",
        "parameters":  {"device_id": "Device identifier", "level": "Severity filter: error|warning|info", "lines": "Max lines (default 50)"},
        "returns":     "Syslog lines with timestamp, facility, severity, message",
        "hitl":        False,
        "tags":        ["logs", "diagnostics"],
    },
    "query_interface_metrics": {
        "description": "Query interface traffic and error counters for a device.",
        "parameters":  {"device_id": "Device identifier", "interface": "Interface name (optional, default: all)"},
        "returns":     "Per-interface: rx/tx bytes, error counts, utilisation pct",
        "hitl":        False,
        "tags":        ["metrics", "network"],
    },
    "get_bgp_summary": {
        "description": "Get BGP peer summary and session state for a router.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "BGP table: peer IP, ASN, state, prefixes received/sent, uptime",
        "hitl":        False,
        "tags":        ["routing", "bgp"],
    },
    "get_device_facts": {
        "description": "Get hardware facts: model, OS version, serial number, uptime.",
        "parameters":  {"device_id": "Device identifier"},
        "returns":     "Facts dict: hostname, model, os_version, serial, uptime",
        "hitl":        False,
        "tags":        ["inventory", "hardware"],
    },
    "run_command": {
        "description": "Run an arbitrary show/display command on a device. Read-only commands only.",
        "parameters":  {"device_id": "Device identifier", "command": "CLI command string (show/display only)"},
        "returns":     "Command output as text",
        "hitl":        False,
        "tags":        ["diagnostics", "read"],
    },
    "multi_device_check": {
        "description": "Run the same check across multiple devices in parallel.",
        "parameters":  {"device_ids": "List of device identifiers", "check": "Check type: status|config|syslog|bgp"},
        "returns":     "Per-device results table",
        "hitl":        False,
        "tags":        ["inventory", "bulk"],
    },
}
