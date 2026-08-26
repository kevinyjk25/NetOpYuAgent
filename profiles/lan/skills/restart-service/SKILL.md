---
name: restart-service
description: Rolling restart of a mock production service
metadata:
  skill_id: restart_service
  display_name: Service Restart
  purpose: Rolling restart of a mock production service
  risk_level: high
  requires_hitl: 'true'
  tags: services,destructive
  tool_deps: restart_service
  returns: Restart status
---

# Service Restart

Performs a rolling restart. Always requires HITL approval.

## Parameters
- `service`: Service name
- `environment`: prod|staging|dev
