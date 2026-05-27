"""
profiles/dc/__init__.py — Data-center fabric profile
=====================================================

Bundles DC tools + metadata + skills + capabilities into the PROFILE the
framework loads when AGENT_PROFILE=dc.
"""
from __future__ import annotations

from profiles.base import Profile
from profiles.dc import tools as _tools
from profiles.dc.tool_meta import TOOLS as _TOOL_META
from profiles.dc.skills import SKILLS as _SKILLS

_CALLABLES = {
    "dc_list_fabric":        _tools.dc_list_fabric,
    "dc_bgp_evpn_status":    _tools.dc_bgp_evpn_status,
    "dc_vxlan_vni_lookup":   _tools.dc_vxlan_vni_lookup,
    "dc_loadbalancer_pools": _tools.dc_loadbalancer_pools,
    "dc_fabric_path_trace":  _tools.dc_fabric_path_trace,
    "dc_evpn_route_lookup":  _tools.dc_evpn_route_lookup,
    "dc_config_push":        _tools.dc_config_push,
    "dc_list_apps":             _tools.dc_list_apps,
    "dc_get_app_acl":           _tools.dc_get_app_acl,
    "dc_check_user_app_access": _tools.dc_check_user_app_access,
    "dc_grant_app_access":      _tools.dc_grant_app_access,
    "dc_revoke_app_access":     _tools.dc_revoke_app_access,
}

_CAPABILITIES = [
    {
        "skill_id":    "dc_fabric_diagnose",
        "name":        "DC fabric diagnostics",
        "description": "Diagnose data-center spine/leaf VXLAN fabric: BGP EVPN "
                       "control plane, VNI/VLAN mappings, underlay/overlay path "
                       "tracing, EVPN route lookups.",
        "tags":        ["dc", "fabric", "evpn", "vxlan", "spine-leaf", "bgp"],
    },
    {
        "skill_id":    "dc_fabric_config",
        "name":        "DC fabric configuration",
        "description": "Push / snapshot fabric config on spine/leaf nodes "
                       "(HITL-gated for destructive changes).",
        "tags":        ["dc", "fabric", "config", "destructive"],
    },
    {
        "skill_id":    "dc_loadbalancer",
        "name":        "DC load balancing",
        "description": "Inspect load-balancer pool + member health in the data center.",
        "tags":        ["dc", "loadbalancer", "health"],
    },
]

PROFILE = Profile(
    profile_id     = "dc",
    display_name   = "Data Center Network Agent",
    description     = (
        "IT operations agent for the data-center network: spine/leaf VXLAN "
        "fabric, BGP EVPN control plane, load balancers, k8s overlay. Path "
        "tracing, EVPN troubleshooting, fabric config management."
    ),
    domain_tags    = ["dc", "fabric", "spine-leaf", "vxlan", "evpn", "bgp", "datacenter"],
    tool_callables = _CALLABLES,
    tool_metadata  = _TOOL_META,
    skill_defs     = _SKILLS,
    capabilities   = _CAPABILITIES,
)
