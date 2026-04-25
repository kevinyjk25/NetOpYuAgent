"""
skills/mock/registry.py
────────────────────────
Skills backed by mock tools — available in mock mode ONLY.

Format: skill_id → {name, purpose, risk_level, requires_hitl, tags,
                    description, parameters, returns, examples, tool_deps}

tool_deps: list of tool names this skill requires.
The SkillCatalogService.filter_to_registry() uses tool_deps (when present)
in addition to the skill_id match for filtering.
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
}
