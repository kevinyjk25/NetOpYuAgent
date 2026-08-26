---
name: multi-device-check
description: Run the same diagnostic across multiple real devices
metadata:
  skill_id: multi_device_check
  display_name: Multi-Device Check
  purpose: Run the same diagnostic across multiple real devices
  risk_level: low
  requires_hitl: 'false'
  tags: bulk,diagnostics
  tool_deps: multi_device_check
  returns: Per-device results
---

# Multi-Device Check

Parallel check across a device list. Returns per-device results table.

## Parameters
- `device_ids`: List of device IDs
- `check`: status|config|syslog|bgp
