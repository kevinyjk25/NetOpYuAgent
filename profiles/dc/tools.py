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
