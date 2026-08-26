---
name: edit-device-config
description: Push a configuration change to a real device
metadata:
  skill_id: edit_device_config
  display_name: Edit Device Config
  purpose: Push a configuration change to a real device
  risk_level: high
  requires_hitl: 'true'
  tags: config,write,destructive
  tool_deps: edit_device_config
  returns: Push result with diff
---

# Edit Device Config

Pushes config change via SSH/NAPALM. Always requires HITL approval first.

## Parameters
- `device_id`: Device identifier
- `section`: Section
- `changes`: Change dict
- `reason`: Audit reason
