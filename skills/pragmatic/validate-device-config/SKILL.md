---
name: validate-device-config
description: Check device configuration against compliance rules
metadata:
  skill_id: validate_device_config
  display_name: Validate Device Config
  purpose: Check device configuration against compliance rules
  risk_level: low
  requires_hitl: 'false'
  tags: config,validation
  tool_deps: validate_device_config
  returns: Validation report
---

# Validate Device Config

Fetches config and validates against compliance rule set. Returns issues.

## Parameters
- `device_id`: Device identifier
