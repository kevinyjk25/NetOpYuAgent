---
name: get-device-config
description: Retrieve running configuration from a real device
metadata:
  skill_id: get_device_config
  display_name: Get Device Config
  purpose: Retrieve running configuration from a real device
  risk_level: low
  requires_hitl: 'false'
  tags: config,read
  tool_deps: get_device_config
  returns: Configuration text
---

# Get Device Config

SSH/NAPALM-backed config retrieval. Can fetch one section.

## Parameters
- `device_id`: Device identifier
- `section`: Section (optional)
