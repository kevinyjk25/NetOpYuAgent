---
name: get-bgp-summary
description: Get BGP peer state for a router
metadata:
  skill_id: get_bgp_summary
  display_name: BGP Summary
  purpose: Get BGP peer state for a router
  risk_level: low
  requires_hitl: 'false'
  tags: routing,bgp
  tool_deps: get_bgp_summary
  returns: BGP peer table
---

# BGP Summary

Returns BGP peer table with session state and prefix counts.

## Parameters
- `device_id`: Device identifier
