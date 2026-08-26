---
name: get-syslog
description: Retrieve recent syslog entries from a real device
metadata:
  skill_id: get_syslog
  display_name: Get Device Syslog
  purpose: Retrieve recent syslog entries from a real device
  risk_level: low
  requires_hitl: 'false'
  tags: logs,diagnostics
  tool_deps: get_syslog
  returns: Syslog entries
---

# Get Device Syslog

SSH-backed syslog retrieval. Reports error if device not reachable.

## Parameters
- `device_id`: Device identifier
- `level`: Severity filter
- `lines`: Max lines
