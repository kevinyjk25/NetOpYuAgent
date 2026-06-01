---
name: syslog-search
description: Search syslog entries across network devices
metadata:
  skill_id: syslog_search
  display_name: Syslog Search
  purpose: Search syslog entries across network devices
  risk_level: low
  requires_hitl: 'false'
  tags: logs,diagnostics
  tool_deps: syslog_search
  returns: Matching syslog lines
---

# Syslog Search

Queries the mock syslog aggregator for matching entries.

## Parameters
- `host`: Device name or glob
- `keyword`: Search term
- `severity`: error|warning|info

## Examples
- args: {'host': 'ap-01', 'severity': 'error'} — Find errors on ap-01
