---
name: dc-lb-health-check
description: Assess load-balancer pool health and identify down/draining members
metadata:
  skill_id: dc_lb_health_check
  display_name: DC Load Balancer Health Check
  purpose: Assess load-balancer pool health and identify down/draining members
  risk_level: low
  requires_hitl: 'false'
  tags: dc,loadbalancer,health
  tool_deps: dc_loadbalancer_pools,dc_fabric_path_trace,dc_evpn_route_lookup
  returns: Pool membership health + reachability findings for down members
---

# DC Load Balancer Health Check

Assess load-balancer pool + member health, and for any down member trace fabric reachability and confirm EVPN presence.

## Parameters
- `pool`: Optional pool name filter: web-prod|app-prod|api-prod

## Examples
- args: {'pool': 'web-prod'} — Check web-prod pool health
