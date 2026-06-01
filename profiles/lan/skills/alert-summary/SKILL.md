---
name: alert-summary
description: Summarise active monitoring alerts
metadata:
  skill_id: alert_summary
  display_name: Alert Summary
  purpose: Summarise active monitoring alerts
  risk_level: low
  requires_hitl: 'false'
  tags: monitoring,alerts
  tool_deps: alert_summary
  returns: Grouped alert table
---

# Alert Summary

Retrieves and groups active alerts by severity and device.

## Parameters
- `severity`: Filter severity
- `site`: Filter by site
