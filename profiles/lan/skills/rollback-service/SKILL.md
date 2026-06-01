---
name: rollback-service
description: Roll back a mock service to a previous version
metadata:
  skill_id: rollback_service
  display_name: Service Rollback
  purpose: Roll back a mock service to a previous version
  risk_level: high
  requires_hitl: 'true'
  tags: services,destructive
  tool_deps: rollback_service
  returns: Rollback status
---

# Service Rollback

Rolls back to target version. Always requires HITL approval.

## Parameters
- `service`: Service name
- `version`: Target version
- `environment`: prod|staging|dev
