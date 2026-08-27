"""
profiles/wan/tools.py — Wide-area network (SD-WAN / edge / transport) tools
============================================================================

Mock tools for wide-area network operations: SD-WAN edge routers, WAN
circuits/links (MPLS / broadband / LTE), inter-site IPsec tunnels, WAN
routing (BGP/OSPF over the overlay), and per-path SLA (latency / jitter /
loss). Each is an async callable ``async def tool(args: dict) -> str``.

These are deliberately DISJOINT from the LAN and DC profiles. A WAN agent
physically cannot call `list_devices` (LAN) or `dc_bgp_evpn_status` (DC) —
those are not in its registry. Cross-domain work goes through A2A
delegation. The three roles form a realistic topology:

    LAN edge  ──delegates──▶  WAN transport  ──delegates──▶  DC fabric

so an end-to-end "user in branch can't reach a DC app" diagnosis may hop
LAN → WAN → DC.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from tools.common_tools import _ts


# ---------------------------------------------------------------------------
# Canonical WAN inventory: SD-WAN edges + circuits + tunnels
# ---------------------------------------------------------------------------

_WAN_EDGES = [
    {"id": "edge-hq",     "site": "hq",       "role": "hub",    "model": "vEdge-2000", "region": "us-east"},
    {"id": "edge-br-sf",  "site": "branch-sf", "role": "spoke", "model": "vEdge-100",  "region": "us-west"},
    {"id": "edge-br-ny",  "site": "branch-ny", "role": "spoke", "model": "vEdge-100",  "region": "us-east"},
    {"id": "edge-dc",     "site": "dc-east",   "role": "hub",   "model": "vEdge-2000", "region": "us-east"},
]

# WAN circuits (transport links) attached to edges
_WAN_CIRCUITS = [
    {"id": "ckt-hq-mpls",   "edge": "edge-hq",    "transport": "mpls",      "bw_mbps": 500, "carrier": "AT&T"},
    {"id": "ckt-hq-inet",   "edge": "edge-hq",    "transport": "broadband", "bw_mbps": 1000, "carrier": "Comcast"},
    {"id": "ckt-sf-inet",   "edge": "edge-br-sf", "transport": "broadband", "bw_mbps": 300, "carrier": "Comcast"},
    {"id": "ckt-sf-lte",    "edge": "edge-br-sf", "transport": "lte",       "bw_mbps": 50,  "carrier": "Verizon"},
    {"id": "ckt-ny-mpls",   "edge": "edge-br-ny", "transport": "mpls",      "bw_mbps": 200, "carrier": "AT&T"},
    {"id": "ckt-dc-mpls",   "edge": "edge-dc",    "transport": "mpls",      "bw_mbps": 1000, "carrier": "AT&T"},
]

# IPsec overlay tunnels between edges (SD-WAN fabric)
_WAN_TUNNELS = [
    {"id": "tun-sf-hq",  "src": "edge-br-sf", "dst": "edge-hq", "transport": "broadband"},
    {"id": "tun-sf-dc",  "src": "edge-br-sf", "dst": "edge-dc", "transport": "broadband"},
    {"id": "tun-ny-hq",  "src": "edge-br-ny", "dst": "edge-hq", "transport": "mpls"},
    {"id": "tun-ny-dc",  "src": "edge-br-ny", "dst": "edge-dc", "transport": "mpls"},
    {"id": "tun-hq-dc",  "src": "edge-hq",    "dst": "edge-dc", "transport": "mpls"},
]

# Mutable overlay used by the local simulator.  A write is not considered
# verified merely because wan_failover_path returned a success string: the
# runtime re-reads wan_tunnel_status and observes this state independently.
_WAN_TUNNEL_STATE: dict[str, str] = {
    tunnel["id"]: tunnel["transport"] for tunnel in _WAN_TUNNELS
}


def _edge_ids() -> set[str]:
    return {e["id"] for e in _WAN_EDGES}


# ---------------------------------------------------------------------------
# wan_list_edges — SD-WAN edge inventory
# ---------------------------------------------------------------------------

async def wan_list_edges(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    role_filter = (args.get("role") or "").strip().lower()
    rows = [e for e in _WAN_EDGES if not role_filter or e["role"] == role_filter]
    if not rows:
        return f"No WAN edges match role={role_filter!r}. Roles: hub, spoke."
    out = ["SD-WAN Edges:", ""]
    out.append(f"{'EDGE':<12}{'SITE':<12}{'ROLE':<8}{'MODEL':<14}{'REGION'}")
    out.append("-" * 54)
    for e in rows:
        out.append(f"{e['id']:<12}{e['site']:<12}{e['role']:<8}{e['model']:<14}{e['region']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# wan_circuit_status — WAN transport link health
# ---------------------------------------------------------------------------

async def wan_circuit_status(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    edge = (args.get("edge") or "").strip()
    rows = [c for c in _WAN_CIRCUITS if not edge or c["edge"] == edge]
    if edge and not rows:
        return f"Unknown edge {edge!r}. Use wan_list_edges. Edges: {', '.join(sorted(_edge_ids()))}."
    out = [f"WAN circuits{f' on {edge}' if edge else ''}:", ""]
    out.append(f"{'CIRCUIT':<14}{'EDGE':<12}{'TRANSPORT':<12}{'BW':<8}{'STATE':<12}{'CARRIER'}")
    out.append("-" * 70)
    for c in rows:
        down = random.random() < 0.10
        state = "DOWN" if down else "up"
        out.append(f"{c['id']:<14}{c['edge']:<12}{c['transport']:<12}"
                   f"{str(c['bw_mbps'])+'M':<8}{state:<12}{c['carrier']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# wan_tunnel_status — IPsec overlay tunnel state
# ---------------------------------------------------------------------------

async def wan_tunnel_status(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    edge = (args.get("edge") or "").strip()
    rows = [t for t in _WAN_TUNNELS if not edge or t["src"] == edge or t["dst"] == edge]
    if edge and not rows:
        return f"No tunnels touch edge {edge!r}. Use wan_list_edges."
    out = [f"SD-WAN IPsec tunnels{f' touching {edge}' if edge else ''}:", ""]
    out.append(f"{'TUNNEL':<12}{'SRC':<12}{'DST':<12}{'TRANSPORT':<12}{'STATE'}")
    out.append("-" * 60)
    for t in rows:
        state = "up"
        transport = _WAN_TUNNEL_STATE.get(t["id"], t["transport"])
        out.append(f"{t['id']:<12}{t['src']:<12}{t['dst']:<12}{transport:<12}{state}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# wan_path_sla — per-path latency / jitter / loss SLA
# ---------------------------------------------------------------------------

async def wan_path_sla(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    src = (args.get("src") or "").strip()
    dst = (args.get("dst") or "").strip()
    if not src or not dst:
        return ("wan_path_sla requires 'src' and 'dst' edge ids "
                "(e.g. src=edge-br-sf dst=edge-dc). Use wan_list_edges.")
    known = _edge_ids()
    for e in (src, dst):
        if e not in known:
            return f"Unknown edge {e!r}. Known: {', '.join(sorted(known))}."
    out = [f"WAN path SLA {src} → {dst}:", ""]
    out.append(f"{'TRANSPORT':<12}{'LATENCY':<10}{'JITTER':<10}{'LOSS':<8}{'SLA'}")
    out.append("-" * 50)
    for transport in ("mpls", "broadband"):
        lat = random.randint(8, 45) if transport == "mpls" else random.randint(20, 90)
        jit = random.randint(1, 6) if transport == "mpls" else random.randint(3, 25)
        loss = round(random.uniform(0, 0.4), 2) if transport == "mpls" else round(random.uniform(0, 2.5), 2)
        breach = loss > 1.0 or lat > 80
        sla = "BREACH" if breach else "ok"
        out.append(f"{transport:<12}{str(lat)+'ms':<10}{str(jit)+'ms':<10}{str(loss)+'%':<8}{sla}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# wan_route_lookup — WAN routing table (BGP/OSPF over overlay)
# ---------------------------------------------------------------------------

async def wan_route_lookup(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    prefix = (args.get("prefix") or "").strip()
    if not prefix:
        return "wan_route_lookup requires a 'prefix' (e.g. 10.20.0.0/16 for DC subnet)."
    # Deterministic-ish next-hop selection
    nexthop = random.choice(["edge-hq", "edge-dc"])
    proto = random.choice(["bgp", "ospf"])
    metric = random.randint(10, 200)
    out = [f"WAN route lookup for {prefix}:", ""]
    out.append(f"{'PREFIX':<18}{'NEXT-HOP':<12}{'PROTO':<8}{'METRIC':<8}{'PREFERRED-PATH'}")
    out.append("-" * 62)
    out.append(f"{prefix:<18}{nexthop:<12}{proto:<8}{metric:<8}{'mpls (primary)'}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# wan_failover_path — DESTRUCTIVE: force a path failover (HITL-gated)
# ---------------------------------------------------------------------------

async def wan_failover_path(args: dict[str, Any]) -> str:
    await asyncio.sleep(0)
    tunnel = (args.get("tunnel") or "").strip()
    to = (args.get("to_transport") or "").strip().lower()
    if not tunnel or not to:
        return ("wan_failover_path requires 'tunnel' and 'to_transport' "
                "(mpls|broadband|lte). This is destructive — HITL-gated.")
    known = {t["id"] for t in _WAN_TUNNELS}
    if tunnel not in known:
        return f"Unknown tunnel {tunnel!r}. Use wan_tunnel_status. Known: {', '.join(sorted(known))}."
    if to not in {"mpls", "broadband", "lte"}:
        return "wan_failover_path to_transport must be one of: mpls, broadband, lte."
    _WAN_TUNNEL_STATE[tunnel] = to
    return (f"[{_ts()}] Forced failover of {tunnel} to {to} transport. "
            f"Traffic re-pinned; verify with wan_tunnel_status and wan_path_sla.")
