"""
profiles/wan/tool_meta.py — Prompt-facing metadata for WAN tools
=================================================================

Declares the wide-area-network tools the LLM sees. Keys MUST match the
callables in profiles/wan/tools.py.
"""
from __future__ import annotations
from typing import Any

TOOLS: dict[str, dict[str, Any]] = {
    "wan_list_edges": {
        "description": "List SD-WAN edge routers (hub / spoke) with site, model, region.",
        "parameters":  {"role": "Filter by role: hub|spoke (optional)"},
        "returns":     "Table of edge id, site, role, model, region",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["wan", "sdwan", "edge", "inventory"],
    },
    "wan_circuit_status": {
        "description": "Show WAN transport circuit health (MPLS / broadband / LTE) per edge, with up/down state.",
        "parameters":  {"edge": "Edge id filter (optional), e.g. edge-br-sf"},
        "returns":     "Circuit table: id, edge, transport, bandwidth, state, carrier",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["wan", "circuit", "transport", "health"],
        "example":     {"edge": "edge-br-sf"},
    },
    "wan_tunnel_status": {
        "description": "Show SD-WAN IPsec overlay tunnel state between edges. Flags rekey failures.",
        "parameters":  {"edge": "Edge id filter (optional) — shows tunnels touching that edge"},
        "returns":     "Tunnel table: id, src, dst, transport, state",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["wan", "sdwan", "ipsec", "tunnel", "overlay"],
        "example":     {"edge": "edge-br-sf"},
    },
    "wan_path_sla": {
        "description": "Show per-path SLA (latency / jitter / loss) between two edges, per transport. Flags SLA breaches.",
        "parameters":  {"src": "Source edge id", "dst": "Destination edge id"},
        "returns":     "SLA table per transport: latency, jitter, loss, SLA ok/breach",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["wan", "sla", "latency", "jitter", "loss"],
        "example":     {"src": "edge-br-sf", "dst": "edge-dc"},
    },
    "wan_route_lookup": {
        "description": "Look up the WAN routing table (BGP/OSPF over overlay) for a prefix; shows next-hop edge + preferred path.",
        "parameters":  {"prefix": "IP prefix, e.g. 10.20.0.0/16"},
        "returns":     "Route entry: prefix, next-hop, protocol, metric, preferred path",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["wan", "routing", "bgp", "ospf"],
        "example":     {"prefix": "10.20.0.0/16"},
    },
    "wan_failover_path": {
        "description": "Force a WAN tunnel to fail over to a different transport (mpls|broadband|lte). DESTRUCTIVE — re-pins live traffic.",
        "parameters":  {"tunnel": "Tunnel id (e.g. tun-sf-dc)", "to_transport": "Target transport: mpls|broadband|lte"},
        "returns":     "Failover confirmation",
        "hitl":        True,
        "action_type": "modify_state",
        "tags":        ["wan", "failover", "destructive"],
    },
}
