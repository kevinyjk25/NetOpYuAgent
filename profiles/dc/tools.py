"""
profiles/dc/tools.py — Data-center fabric (spine/leaf) business tools
======================================================================

Mock tools for data-center network operations: spine/leaf VXLAN fabric,
BGP EVPN control plane, load balancers, k8s overlay networking. Each is an
async callable `async def tool(args: dict) -> str`.

These are deliberately DISJOINT from the LAN profile's tools — a DC agent
physically cannot call `list_devices` / `edit_device_config` etc. (they're not
in its registry), and a LAN agent cannot call these. Cross-domain work must go
through A2A delegation (Phase 2B). That tool isolation is the whole point of
the role split.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from tools.common_tools import _ts


# ---------------------------------------------------------------------------
# Canonical DC fabric inventory (spine/leaf)
# ---------------------------------------------------------------------------

_FABRIC_NODES = [
    {"id": "spine-1", "role": "spine", "model": "Nexus 9364C", "asn": 65000, "site": "dc-east"},
    {"id": "spine-2", "role": "spine", "model": "Nexus 9364C", "asn": 65000, "site": "dc-east"},
    {"id": "leaf-1",  "role": "leaf",  "model": "Nexus 93180YC", "asn": 65101, "site": "dc-east"},
    {"id": "leaf-2",  "role": "leaf",  "model": "Nexus 93180YC", "asn": 65102, "site": "dc-east"},
    {"id": "leaf-3",  "role": "leaf",  "model": "Nexus 93180YC", "asn": 65103, "site": "dc-east"},
    {"id": "border-1","role": "border-leaf", "model": "Nexus 9332C", "asn": 65110, "site": "dc-east"},
]

# VNI ↔ segment mapping (VXLAN overlay)
_VNI_MAP = [
    {"vni": 10100, "vlan": 100, "segment": "web-tier",   "vrf": "prod",    "anycast_gw": "10.1.0.1"},
    {"vni": 10200, "vlan": 200, "segment": "app-tier",   "vrf": "prod",    "anycast_gw": "10.2.0.1"},
    {"vni": 10300, "vlan": 300, "segment": "db-tier",    "vrf": "prod",    "anycast_gw": "10.3.0.1"},
    {"vni": 20100, "vlan": 110, "segment": "k8s-pods",   "vrf": "k8s",     "anycast_gw": "172.16.0.1"},
    {"vni": 20200, "vlan": 120, "segment": "k8s-svc",    "vrf": "k8s",     "anycast_gw": "172.17.0.1"},
]

# Mutable fabric state (config pushes mutate this, like LAN's _DEVICE_STATE)
_FABRIC_STATE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# dc_list_fabric — inventory of spine/leaf nodes
# ---------------------------------------------------------------------------

async def dc_list_fabric(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    role_filter = (args.get("role") or "").strip().lower()
    rows = [n for n in _FABRIC_NODES if not role_filter or n["role"] == role_filter]
    if not rows:
        return f"No fabric nodes match role={role_filter!r}. Roles: spine, leaf, border-leaf."
    out = ["DC Fabric Nodes:", ""]
    out.append(f"{'NODE':<10}{'ROLE':<14}{'MODEL':<16}{'ASN':<8}{'SITE'}")
    out.append("-" * 56)
    for n in rows:
        out.append(f"{n['id']:<10}{n['role']:<14}{n['model']:<16}{n['asn']:<8}{n['site']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# dc_bgp_evpn_status — BGP EVPN neighbor + route status
# ---------------------------------------------------------------------------

async def dc_bgp_evpn_status(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    node = (args.get("node") or "").strip()
    if not node:
        return ("dc_bgp_evpn_status requires a 'node' (e.g. leaf-1). "
                "Use dc_list_fabric to see nodes.")
    known = {n["id"] for n in _FABRIC_NODES}
    if node not in known:
        return f"Unknown fabric node {node!r}. Known: {', '.join(sorted(known))}."

    # Spines peer with all leaves; leaves peer with both spines.
    is_spine = node.startswith("spine")
    peers = ([f"leaf-{i}" for i in (1, 2, 3)] + ["border-1"]) if is_spine else ["spine-1", "spine-2"]
    out = [f"BGP EVPN status on {node}:", ""]
    out.append(f"{'NEIGHBOR':<12}{'STATE':<14}{'UP/DOWN':<12}{'EVPN-RT':<10}{'PFX-RCVD'}")
    out.append("-" * 60)
    for p in peers:
        flap = random.random() < 0.12
        state = "Idle (flapping)" if flap else "Established"
        updown = f"{random.randint(0,3)}h{random.randint(10,59)}m" if not flap else "00:00:42"
        evpn_rt = random.randint(40, 120)
        pfx = random.randint(200, 1200)
        out.append(f"{p:<12}{state:<14}{updown:<12}{evpn_rt:<10}{pfx}")
    flapping = [l for l in out if "flapping" in l]
    if flapping:
        out.append("")
        out.append(f"⚠ {len(flapping)} neighbor(s) flapping — check underlay reachability / MTU.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# dc_vxlan_vni_lookup — VNI ↔ VLAN/segment/VRF mapping
# ---------------------------------------------------------------------------

async def dc_vxlan_vni_lookup(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    vni = args.get("vni")
    segment = (args.get("segment") or "").strip().lower()
    rows = _VNI_MAP
    if vni is not None:
        try:
            vni_i = int(vni)
            rows = [r for r in rows if r["vni"] == vni_i]
        except (TypeError, ValueError):
            return f"Invalid vni {vni!r} — must be an integer."
    if segment:
        rows = [r for r in rows if segment in r["segment"]]
    if not rows:
        return "No VNI mappings match. Use dc_vxlan_vni_lookup with no args to list all."
    out = ["VXLAN VNI mappings:", ""]
    out.append(f"{'VNI':<8}{'VLAN':<7}{'SEGMENT':<12}{'VRF':<8}{'ANYCAST-GW'}")
    out.append("-" * 48)
    for r in rows:
        out.append(f"{r['vni']:<8}{r['vlan']:<7}{r['segment']:<12}{r['vrf']:<8}{r['anycast_gw']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# dc_loadbalancer_pools — LB pool / member health
# ---------------------------------------------------------------------------

async def dc_loadbalancer_pools(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    pool_filter = (args.get("pool") or "").strip().lower()
    pools = {
        "web-prod":  [("10.1.0.11", "up"), ("10.1.0.12", "up"), ("10.1.0.13", "down")],
        "app-prod":  [("10.2.0.21", "up"), ("10.2.0.22", "up")],
        "api-prod":  [("10.2.0.31", "up"), ("10.2.0.32", "draining"), ("10.2.0.33", "up")],
    }
    if pool_filter:
        pools = {k: v for k, v in pools.items() if pool_filter in k}
        if not pools:
            return f"No LB pool matches {pool_filter!r}. Pools: web-prod, app-prod, api-prod."
    out = ["Load balancer pools:", ""]
    for name, members in pools.items():
        up = sum(1 for _, s in members if s == "up")
        out.append(f"Pool {name}: {up}/{len(members)} members up")
        for ip, status in members:
            mark = {"up": "✓", "down": "✗", "draining": "⏳"}.get(status, "?")
            out.append(f"   {mark} {ip:<14} {status}")
        out.append("")
    return "\n".join(out).rstrip()


# ---------------------------------------------------------------------------
# dc_fabric_path_trace — underlay/overlay path between two endpoints
# ---------------------------------------------------------------------------

async def dc_fabric_path_trace(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    src = (args.get("src") or "").strip()
    dst = (args.get("dst") or "").strip()
    if not src or not dst:
        return ("dc_fabric_path_trace requires 'src' and 'dst' endpoint IPs "
                "(e.g. src=10.1.0.11 dst=10.3.0.31).")
    # Mock ECMP path through the fabric
    spine = random.choice(["spine-1", "spine-2"])
    out = [f"Fabric path {src} → {dst}:", ""]
    out.append(f"  ingress leaf : leaf-1 (VTEP 10.0.0.1)")
    out.append(f"  underlay hop : {spine} (ECMP, 2 equal-cost paths)")
    out.append(f"  egress leaf  : leaf-3 (VTEP 10.0.0.3)")
    out.append(f"  overlay VNI  : 10100 → 10300 (inter-VRF via border-1)")
    out.append("")
    out.append(f"  RTT: {random.uniform(0.05, 0.3):.3f} ms  MTU: 9000 (jumbo)  loss: 0%")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# dc_config_push — push fabric config (DESTRUCTIVE, HITL-gated)
# ---------------------------------------------------------------------------

async def dc_config_push(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    node = (args.get("node") or "").strip()
    config_lines = args.get("config_lines") or []
    reason = (args.get("reason") or "").strip()
    if not node:
        return "dc_config_push requires 'node'."
    if not config_lines:
        return "dc_config_push requires 'config_lines' (list of config statements)."
    known = {n["id"] for n in _FABRIC_NODES}
    if node not in known:
        return f"Unknown fabric node {node!r}. Known: {', '.join(sorted(known))}."

    if isinstance(config_lines, str):
        config_lines = [l for l in config_lines.splitlines() if l.strip()]

    # Record into mutable state (mock "apply")
    snap = _FABRIC_STATE.setdefault(node, {"applied": []})
    snap["applied"].extend(config_lines)

    out = [f"Config pushed to {node}:", ""]
    for line in config_lines:
        out.append(f"  + {line}")
    out.append("")
    out.append(f"  reason: {reason or '(none given)'}")
    out.append(f"  snapshot taken: {_ts()}  (rollback id: {random.randint(1000,9999)})")
    out.append(f"  status: applied, BGP EVPN re-converged in {random.uniform(0.3,1.2):.1f}s")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# dc_evpn_route_lookup — look up a MAC/IP in the EVPN control plane
# ---------------------------------------------------------------------------

async def dc_evpn_route_lookup(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    target = (args.get("mac") or args.get("ip") or "").strip()
    if not target:
        return "dc_evpn_route_lookup requires 'mac' or 'ip'."
    vtep = random.choice(["leaf-1", "leaf-2", "leaf-3"])
    vni = random.choice([10100, 10200, 10300])
    out = [f"EVPN route for {target}:", ""]
    out.append(f"  type      : {'MAC/IP (Type-2)' if ':' in target else 'IP Prefix (Type-5)'}")
    out.append(f"  VNI       : {vni}")
    out.append(f"  next-hop  : {vtep} (VTEP)")
    out.append(f"  ESI       : 00:00:00:00:00:00:00:00:00:00")
    out.append(f"  learned   : {random.randint(1,48)}h ago via BGP EVPN")
    return "\n".join(out)


# ===========================================================================
# Application access-control tools (2026-05) — for the cross-agent HITL
# fault scenario: "user X cannot access application Y". The DC owns the
# APPLICATION-LAYER access permissions (RBAC/ACL); the LAN owns network
# admission/auth. Diagnosis crosses both agents via A2A delegation.
#
# Method 甲 (approved): QUERIES are read-only (no HITL); only the GRANT/REVOKE
# (write) operations are HITL-gated, and those gates fire on the DC side —
# which is exactly what A2A Phase 3 mode B demonstrates.
# ===========================================================================

# Canonical DC application inventory + access matrix. Deliberately consistent
# with the LAN user table (profiles/lan/tools.py): user 'alice' exists and is
# admitted on the LAN, but is NOT on the CRM access list here — that absence is
# the root cause of "alice cannot access CRM".
_DC_APPS = [
    {"id": "crm",      "name": "Salesforce CRM",     "vip": "10.20.0.10", "owner": "sales-it",    "tier": "gold"},
    {"id": "wiki",     "name": "Internal Wiki",       "vip": "10.20.0.20", "owner": "platform",    "tier": "silver"},
    {"id": "payroll",  "name": "Payroll Portal",      "vip": "10.20.0.30", "owner": "finance-it",  "tier": "gold"},
    {"id": "grafana",  "name": "Observability",       "vip": "10.20.0.40", "owner": "sre",         "tier": "bronze"},
]

# app_id -> {role -> [user_ids]}. RBAC: a user can access an app if they hold
# any role listed for it. 'alice' is intentionally absent from CRM.
_DC_APP_ACL: dict[str, dict[str, list[str]]] = {
    "crm":     {"sales-rep": ["bob", "carol"], "sales-admin": ["dave"]},
    "wiki":    {"reader":    ["alice", "bob", "carol", "dave"]},
    "payroll": {"fin-user":  ["dave"], "fin-admin": ["erin"]},
    "grafana": {"viewer":    ["alice", "bob"], "editor": ["sre-oncall"]},
}

# Mutable grant log (mock "apply" target for grant/revoke).
_DC_ACCESS_CHANGES: list[dict[str, Any]] = []


def _dc_user_has_app_access(user_id: str, app_id: str) -> tuple[bool, list[str]]:
    """Return (allowed, roles_held_for_this_app)."""
    roles = _DC_APP_ACL.get(app_id, {})
    held = [role for role, members in roles.items() if user_id in members]
    return (bool(held), held)


async def dc_list_apps(args: dict[str, Any]) -> str:
    """List data-center applications (read-only)."""
    await asyncio.sleep(0)
    tier_filter = (args.get("tier") or "").strip().lower()
    apps = _DC_APPS
    if tier_filter:
        apps = [a for a in apps if a["tier"] == tier_filter]
        if not apps:
            return f"No application matches tier={tier_filter!r}. Tiers: gold, silver, bronze."
    out = ["Data-center applications:", ""]
    out.append(f"  {'ID':<10}{'NAME':<22}{'VIP':<14}{'OWNER':<12}TIER")
    for a in apps:
        out.append(f"  {a['id']:<10}{a['name']:<22}{a['vip']:<14}{a['owner']:<12}{a['tier']}")
    return "\n".join(out)


async def dc_get_app_acl(args: dict[str, Any]) -> str:
    """Show the access-control list (roles → members) for one application."""
    await asyncio.sleep(0)
    app_id = (args.get("app_id") or args.get("app") or "").strip().lower()
    if not app_id:
        return "dc_get_app_acl requires 'app_id' (e.g. crm, wiki, payroll, grafana)."
    if app_id not in _DC_APP_ACL:
        return f"Unknown application {app_id!r}. Known: {', '.join(a['id'] for a in _DC_APPS)}."
    roles = _DC_APP_ACL[app_id]
    out = [f"Access control for application '{app_id}':", ""]
    for role, members in roles.items():
        out.append(f"  role {role:<12}: {', '.join(members) if members else '(none)'}")
    # Reflect any runtime grants/revokes recorded this session.
    changes = [c for c in _DC_ACCESS_CHANGES if c["app_id"] == app_id]
    if changes:
        out.append("")
        out.append("  pending/applied changes this session:")
        for c in changes:
            out.append(f"    {c['op']} {c['user_id']} (role={c['role']}) — {c['reason'] or 'no reason'}")
    return "\n".join(out)


async def dc_check_user_app_access(args: dict[str, Any]) -> str:
    """Diagnose whether a user can access an application (READ-ONLY).

    This is the core cross-agent diagnostic: LAN delegates here to find out if
    the application layer admits the user. No HITL — querying access is not a
    sensitive operation (method 甲).
    """
    await asyncio.sleep(0)
    user_id = (args.get("user_id") or args.get("user") or "").strip().lower()
    app_id  = (args.get("app_id") or args.get("app") or "").strip().lower()
    if not user_id or not app_id:
        return "dc_check_user_app_access requires 'user_id' and 'app_id'."
    if app_id not in _DC_APP_ACL:
        return f"Unknown application {app_id!r}. Known: {', '.join(a['id'] for a in _DC_APPS)}."

    # Apply any session grants/revokes on top of the base ACL.
    allowed, held = _dc_user_has_app_access(user_id, app_id)
    for c in _DC_ACCESS_CHANGES:
        if c["app_id"] == app_id and c["user_id"] == user_id:
            if c["op"] == "grant":
                allowed, held = True, list(set(held + [c["role"]]))
            elif c["op"] == "revoke":
                allowed, held = False, []

    app = next((a for a in _DC_APPS if a["id"] == app_id), {})
    out = [f"Application access check — user '{user_id}' → app '{app_id}' ({app.get('name','?')}):", ""]
    out.append(f"  VIP        : {app.get('vip','?')}")
    out.append(f"  owner team : {app.get('owner','?')}")
    out.append(f"  result     : {'✅ ALLOWED' if allowed else '❌ DENIED'}")
    if allowed:
        out.append(f"  via roles  : {', '.join(held)}")
    else:
        out.append(f"  reason     : user holds no role granting access to '{app_id}'")
        out.append(f"  remediation: grant an appropriate role via dc_grant_app_access "
                   f"(requires operator approval on the DC side)")
    return "\n".join(out)


async def dc_grant_app_access(args: dict[str, Any]) -> str:
    """Grant a user a role on an application (DESTRUCTIVE — HITL-gated on DC).

    The HITL gate fires at the runtime-loop level because this tool is on the
    DC profile's hitl_tool_names watch-list. When the grant was requested via
    an A2A delegation, the approval card appears on the DC operator's console
    (A2A Phase 3 mode B); the result flows back and the delegating agent
    resumes.
    """
    await asyncio.sleep(0)
    user_id = (args.get("user_id") or args.get("user") or "").strip().lower()
    app_id  = (args.get("app_id") or args.get("app") or "").strip().lower()
    role    = (args.get("role") or "").strip()
    reason  = (args.get("reason") or "").strip()
    if not user_id or not app_id:
        return "dc_grant_app_access requires 'user_id' and 'app_id'."
    if app_id not in _DC_APP_ACL:
        return f"Unknown application {app_id!r}. Known: {', '.join(a['id'] for a in _DC_APPS)}."
    if not role:
        # Default to the app's most basic role if not specified.
        role = next(iter(_DC_APP_ACL[app_id].keys()), "reader")
    _DC_APP_ACL.setdefault(app_id, {}).setdefault(role, [])
    if user_id not in _DC_APP_ACL[app_id][role]:
        _DC_APP_ACL[app_id][role].append(user_id)
    _DC_ACCESS_CHANGES.append({"op": "grant", "user_id": user_id, "app_id": app_id,
                               "role": role, "reason": reason})
    out = [f"Granted application access:", ""]
    out.append(f"  user   : {user_id}")
    out.append(f"  app    : {app_id} ({next((a['name'] for a in _DC_APPS if a['id']==app_id), '?')})")
    out.append(f"  role   : {role}")
    out.append(f"  reason : {reason or '(none given)'}")
    out.append(f"  status : applied at {_ts()}")
    return "\n".join(out)


async def dc_revoke_app_access(args: dict[str, Any]) -> str:
    """Revoke a user's access to an application (DESTRUCTIVE — HITL-gated on DC)."""
    await asyncio.sleep(0)
    user_id = (args.get("user_id") or args.get("user") or "").strip().lower()
    app_id  = (args.get("app_id") or args.get("app") or "").strip().lower()
    reason  = (args.get("reason") or "").strip()
    if not user_id or not app_id:
        return "dc_revoke_app_access requires 'user_id' and 'app_id'."
    if app_id not in _DC_APP_ACL:
        return f"Unknown application {app_id!r}. Known: {', '.join(a['id'] for a in _DC_APPS)}."
    removed_from = []
    for role, members in _DC_APP_ACL.get(app_id, {}).items():
        if user_id in members:
            members.remove(user_id)
            removed_from.append(role)
    _DC_ACCESS_CHANGES.append({"op": "revoke", "user_id": user_id, "app_id": app_id,
                               "role": ",".join(removed_from) or "(none)", "reason": reason})
    out = [f"Revoked application access:", ""]
    out.append(f"  user   : {user_id}")
    out.append(f"  app    : {app_id}")
    out.append(f"  roles removed: {', '.join(removed_from) if removed_from else '(user had none)'}")
    out.append(f"  reason : {reason or '(none given)'}")
    out.append(f"  status : applied at {_ts()}")
    return "\n".join(out)
