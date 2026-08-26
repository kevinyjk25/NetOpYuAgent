"""
profiles/wan/__init__.py — Wide-area network (SD-WAN / transport) profile
==========================================================================

Bundles WAN tools + metadata + capabilities into the PROFILE the framework
loads when AGENT_PROFILE=wan.

Topology role: the WAN agent sits between the LAN edge and the DC fabric —
    LAN  ──▶  WAN  ──▶  DC
so an end-to-end branch-to-DC-app problem may hop LAN → WAN → DC via A2A
delegation. It owns SD-WAN edges, transport circuits, IPsec tunnels, WAN
routing, and path SLA.
"""
from __future__ import annotations

from profiles.base import Profile
from profiles.wan import tools as _tools
from profiles.wan.tool_meta import TOOLS as _TOOL_META

# WAN has no bespoke SKILL.md folders yet — capabilities drive delegation
# routing; skills can be added under profiles/wan/skills/<name>/SKILL.md later.
try:
    from skills.loader import SkillLoader as _SkillLoader
    _SKILLS = _SkillLoader().profile_skill_definitions("wan")
except Exception:
    _SKILLS = {}

_CALLABLES = {
    "wan_list_edges":     _tools.wan_list_edges,
    "wan_circuit_status": _tools.wan_circuit_status,
    "wan_tunnel_status":  _tools.wan_tunnel_status,
    "wan_path_sla":       _tools.wan_path_sla,
    "wan_route_lookup":   _tools.wan_route_lookup,
    "wan_failover_path":  _tools.wan_failover_path,
}

_CAPABILITIES = [
    {
        "skill_id":    "wan_transport_diagnose",
        "name":        "WAN transport diagnostics",
        "description": "Diagnose the wide-area network: SD-WAN edges, transport "
                       "circuits (MPLS/broadband/LTE), IPsec overlay tunnels, "
                       "and per-path SLA (latency/jitter/loss). Find which WAN "
                       "leg is degraded on a branch-to-DC path.",
        "tags":        ["wan", "sdwan", "transport", "circuit", "tunnel", "sla", "latency"],
    },
    {
        "skill_id":    "wan_routing",
        "name":        "WAN routing lookup",
        "description": "Look up WAN routing (BGP/OSPF over the SD-WAN overlay) "
                       "for a prefix; determine the next-hop edge and preferred "
                       "transport path toward a destination subnet.",
        "tags":        ["wan", "routing", "bgp", "ospf", "path"],
    },
    {
        "skill_id":    "wan_failover",
        "name":        "WAN path failover",
        "description": "Force a WAN tunnel onto a different transport when a "
                       "circuit degrades (HITL-gated, destructive).",
        "tags":        ["wan", "failover", "destructive"],
    },
]

PROFILE = Profile(
    profile_id     = "wan",
    display_name   = "Wide-Area Network Agent",
    description     = (
        "IT operations agent for the wide-area network: SD-WAN edges, transport "
        "circuits (MPLS / broadband / LTE), inter-site IPsec tunnels, WAN "
        "routing (BGP/OSPF overlay), and per-path SLA. Sits between LAN edge and "
        "DC fabric; handles branch-to-DC transport diagnosis and failover."
    ),
    domain_tags    = ["wan", "sdwan", "transport", "circuit", "tunnel", "sla", "mpls", "wide-area"],
    tool_callables = _CALLABLES,
    tool_metadata  = _TOOL_META,
    skill_defs     = _SKILLS,
    capabilities   = _CAPABILITIES,
)
