---
name: prometheus-query
description: Query metrics from the mock Prometheus store
metadata:
  skill_id: prometheus_query
  display_name: Prometheus Query
  purpose: Query metrics from the mock Prometheus store
  risk_level: low
  requires_hitl: 'false'
  tags: metrics,monitoring
  tool_deps: prometheus_query
  returns: Time series table
---

# Prometheus Query

Runs PromQL queries and returns time series data.

## Parameters
- `query`: PromQL expression
- `duration`: Time window
