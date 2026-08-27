"""
profiles/lan/__init__.py — Enterprise LAN profile
==================================================

Bundles LAN tools + metadata + skills + advertised capabilities into the
PROFILE object the framework loads when AGENT_PROFILE=lan.
"""
from __future__ import annotations

from profiles.base import Profile
from profiles.lan import tools as _tools
from profiles.lan.tool_meta import TOOLS as _TOOL_META

# Business skills are now loaded from Anthropic-standard SKILL.md folders
# under profiles/lan/skills/<name>/SKILL.md (Anthropic-standard format).
# The loader returns the internal flat-dict shape.
from skills.loader import SkillLoader as _SkillLoader
_SKILLS = _SkillLoader().profile_skill_definitions("lan")

# Map tool name → callable. The names MUST match _TOOL_META keys.
_CALLABLES = {
    "syslog_search":          _tools.syslog_search,
    "prometheus_query":       _tools.prometheus_query,
    "netflow_dump":           _tools.netflow_dump,
    "dns_lookup":             _tools.dns_lookup,
    "device_info":            _tools.device_info,
    "alert_summary":          _tools.alert_summary,
    "service_health":         _tools.service_health,
    "list_devices":           _tools.list_devices,
    "list_interfaces":        _tools.list_interfaces,
    "get_device_config":      _tools.get_device_config,
    "validate_device_config": _tools.validate_device_config,
    "edit_device_config":     _tools.edit_device_config,
    "restart_service":        _tools.restart_service,
    "rollback_service":       _tools.rollback_service,
    "diff_device_config":     _tools.diff_device_config,
    "push_config":            _tools.push_config,
    "rollback_deploy":        _tools.rollback_deploy,
    "drain_node":             _tools.drain_node,
    "failover":               _tools.failover,
    "delete_resource":        _tools.delete_resource,
    "mock_operation_status":  _tools.mock_operation_status,
    "query_radius_logs":      _tools.query_radius_logs,   # H2 async-HITL demo (2026-05)
    # User / network-access control (cross-agent HITL scenario, 2026-05)
    "list_users":             _tools.list_users,
    "get_user_access":        _tools.get_user_access,
    "check_nac_policy":       _tools.check_nac_policy,
    "grant_user_access":      _tools.grant_user_access,
    "revoke_user_access":     _tools.revoke_user_access,
}

# Capabilities advertised to peers in the AgentCard (used by Phase-2B
# delegation to decide which agent handles a cross-domain query).
_CAPABILITIES = [
    {
        "skill_id":    "lan_diagnose",
        "name":        "LAN diagnostics",
        "description": "Diagnose enterprise LAN devices: Cisco switches, "
                       "access points, internal firewalls. Syslog, interface "
                       "status, device config inspection.",
        "tags":        ["lan", "switch", "ap", "firewall", "cisco", "internal"],
    },
    {
        "skill_id":    "lan_config",
        "name":        "LAN configuration",
        "description": "Push / validate / rollback config on LAN devices "
                       "(HITL-gated for destructive changes).",
        "tags":        ["lan", "config", "destructive", "cisco"],
    },
    {
        "skill_id":    "lan_observability",
        "name":        "LAN observability",
        "description": "Prometheus metrics, NetFlow analysis, DNS lookups, "
                       "alert summaries for the campus / branch network.",
        "tags":        ["lan", "metrics", "netflow", "alerts"],
    },
]

PROFILE = Profile(
    profile_id     = "lan",
    display_name   = "Enterprise LAN Agent",
    description     = (
        "IT operations agent for the enterprise LAN: Cisco switches, access "
        "points, and internal firewalls. Alert analysis, config management, "
        "incident response on the campus / branch network."
    ),
    domain_tags    = ["lan", "switch", "ap", "firewall", "cisco", "campus", "branch"],
    tool_callables = _CALLABLES,
    tool_metadata  = _TOOL_META,
    skill_defs     = _SKILLS,
    capabilities   = _CAPABILITIES,
)
