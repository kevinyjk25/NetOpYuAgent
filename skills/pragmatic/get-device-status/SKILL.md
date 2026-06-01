---
name: get-device-status
description: Get live operational status for a device
metadata:
  skill_id: get_device_status
  display_name: Device Status
  purpose: Get live operational status for a device
  risk_level: low
  requires_hitl: 'false'
  tags: monitoring,status
  tool_deps: get_device_status
  returns: Status dict
---

# Device Status

Returns CPU, memory, uptime, and interface summary for one device.

## Parameters
- `device_id`: Device identifier
