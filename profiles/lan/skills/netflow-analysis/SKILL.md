---
name: netflow-analysis
description: Analyse NetFlow traffic for anomalies and top talkers
metadata:
  skill_id: netflow_analysis
  display_name: NetFlow Analysis
  purpose: Analyse NetFlow traffic for anomalies and top talkers
  risk_level: low
  requires_hitl: 'false'
  tags: traffic,security
  tool_deps: netflow_dump,read_stored_result
  returns: Traffic summary with anomaly indicators
---

# NetFlow Analysis

Dumps and analyses NetFlow records. For large datasets, pages through stored results.

## Parameters
- `site`: Site name or 'all'

## Examples
- args: {'site': 'all'} — Analyse all-site traffic
