---
name: list-devices
description: Enumerate all real network devices in inventory
metadata:
  skill_id: list_devices
  display_name: List Devices
  purpose: Enumerate all real network devices in inventory
  risk_level: low
  requires_hitl: 'false'
  tags: inventory
  tool_deps: list_devices
  returns: Device table from live inventory
---

# List Devices

Calls list_devices to return live inventory. Filter by type or tag.

## Parameters
- `type`: Device type filter
- `tag`: Site tag filter

## Examples
- List all devices
