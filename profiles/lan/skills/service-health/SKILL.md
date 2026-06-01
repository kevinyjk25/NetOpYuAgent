---
name: service-health
description: Check health of a named mock service
metadata:
  skill_id: service_health
  display_name: Service Health Check
  purpose: Check health of a named mock service
  risk_level: low
  requires_hitl: 'false'
  tags: services,health
  tool_deps: service_health
  returns: Health status with latency and pod counts
---

# Service Health Check

Checks service health across environments.

## Parameters
- `service`: Service name
- `environment`: prod|staging|dev
