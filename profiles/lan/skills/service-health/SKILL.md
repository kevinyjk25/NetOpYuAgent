---
name: service-health
description: 检查数据库或指定服务当前的状态和健康度；check health of a named mock service
metadata:
  skill_id: service_health
  display_name: Service Health Check
  purpose: 查看数据库、认证或其他服务现在是否健康及其运行状态；check service health and status
  risk_level: low
  requires_hitl: 'false'
  tags: services,health,database,healthy,status,服务状态,服务健康,数据库状态
  tool_deps: service_health
  returns: Health status with latency and pod counts
---

# Service Health Check

Checks service health across environments.

## Parameters
- `service`: Service name
- `environment`: prod|staging|dev
