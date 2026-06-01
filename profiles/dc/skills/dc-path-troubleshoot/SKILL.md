---
name: dc-path-troubleshoot
description: Trace connectivity between two endpoints across the VXLAN fabric
metadata:
  skill_id: dc_path_troubleshoot
  display_name: DC Path Troubleshoot
  purpose: Trace connectivity between two endpoints across the VXLAN fabric
  risk_level: low
  requires_hitl: 'false'
  tags: dc,path,vxlan,troubleshoot
  tool_deps: dc_vxlan_vni_lookup,dc_fabric_path_trace,dc_loadbalancer_pools
  returns: Hop-by-hop path with VNI/VTEP/RTT + diagnosis
---

# DC Path Troubleshoot

Trace and diagnose connectivity between two endpoints across the VXLAN fabric: confirm segments/VNIs, trace underlay+overlay path, check inter-VRF leaking, inspect any VIP involved.

## Parameters
- `src`: Source endpoint IP
- `dst`: Destination endpoint IP

## Examples
- args: {'src': '10.1.0.11', 'dst': '10.3.0.31'} — web->db path
