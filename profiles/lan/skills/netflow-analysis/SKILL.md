---
name: netflow-analysis
description: 分析 NetFlow 网络流量、异常流量和流量排行；analyse traffic anomalies and top talkers
metadata:
  skill_id: netflow_analysis
  display_name: NetFlow Analysis
  purpose: 检查设备或站点的网络流量是否异常并识别主要流量来源；analyse NetFlow anomalies and top talkers
  risk_level: low
  requires_hitl: 'false'
  tags: traffic,security,网络流量,异常流量,流量分析
  tool_deps: netflow_dump,read_stored_result
  returns: Traffic summary with anomaly indicators
---

# NetFlow Analysis

Dumps and analyses NetFlow records. For large datasets, pages through stored results.

## Parameters
- `site`: Site name or 'all'

## Examples
- args: {'site': 'all'} — Analyse all-site traffic
