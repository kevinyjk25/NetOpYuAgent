---
name: rollback-service
description: 回滚刚才的服务配置或恢复到上一版本；roll back a mock service to a previous version
metadata:
  skill_id: rollback_service
  display_name: Service Rollback
  purpose: 回滚最近的配置变更并将服务恢复到指定或上一版本；roll back service configuration
  risk_level: high
  requires_hitl: 'true'
  tags: services,destructive,回滚配置,恢复版本
  tool_deps: rollback_service
  returns: Rollback status
---

# Service Rollback

Rolls back to target version. Always requires HITL approval.

## Parameters
- `service`: Service name
- `version`: Target version
- `environment`: prod|staging|dev
