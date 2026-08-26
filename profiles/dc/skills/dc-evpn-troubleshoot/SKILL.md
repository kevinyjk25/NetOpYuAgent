---
name: dc-evpn-troubleshoot
description: Diagnose BGP EVPN control-plane issues in the spine/leaf fabric
metadata:
  skill_id: dc_evpn_troubleshoot
  display_name: DC EVPN Troubleshoot
  purpose: Diagnose BGP EVPN control-plane issues in the spine/leaf fabric
  risk_level: low
  requires_hitl: 'false'
  tags: dc,evpn,bgp,troubleshoot
  tool_deps: dc_list_fabric,dc_bgp_evpn_status,dc_evpn_route_lookup
  returns: EVPN neighbor health + route presence findings
---

# DC EVPN Troubleshoot

Diagnose BGP EVPN control-plane issues (flapping neighbors, missing routes, VTEP reachability) across the fabric. Enumerate nodes, check neighbor state, verify route presence.

## Parameters
- `node`: Affected leaf/spine node id (e.g. leaf-1)
- `target`: Optional MAC/IP to look up in the EVPN control plane

## Examples
- args: {'node': 'leaf-1'} — Check EVPN on leaf-1
