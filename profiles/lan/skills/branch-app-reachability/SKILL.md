---
name: branch-app-reachability
description: 端到端排查分支机构用户访问数据中心应用"连不上/很慢/时通时断"的问题,跨 LAN 准入层、WAN 传输层、DC 应用层三域定位故障。LAN 侧先确认准入并判定故障层,视情况委派 WAN 排查分支到 DC 的传输路径(电路/隧道/SLA),再委派 DC 排查应用权限与 fabric 可达性。适用于"分支用户 X 访问总部/DC 应用 Y 失败"的跨域端到端诊断。
allowed-tools: get_user_access, check_nac_policy, query_radius_logs
delegates-to: wan-agent, dc-agent
degraded-capability: |
  本技能跨三域,任一对端离线时仍交付可达层的诊断,绝不臆测离线域的结论:
  - WAN 离线:完成 LAN 准入层诊断。若准入正常需排查传输,则如实告知"分支到 DC 的
    传输路径(电路/隧道/SLA)诊断需 WAN agent,当前离线,无法确认传输层是否为根因"。
    仍可尝试委派 DC(若在线)排查应用层——传输与应用是独立层,DC 可达则应用层诊断有效。
  - DC 离线:完成 LAN 准入 + (若 WAN 在线)WAN 传输诊断。应用层(RBAC/fabric 可达性)
    如实标记为"需 DC agent,当前离线,待恢复"。
  - WAN 与 DC 均离线:仅交付 LAN 准入层诊断,明确说明传输层与应用层均因对端离线无法排查,
    已排除的仅是准入层原因。
  任何情况下都给出"已排查层 / 未排查层 / 各层结论 / 下一步"的清晰边界报告。
metadata:
  skill_id: branch_app_reachability
  display_name: Branch→DC App Reachability (cross-agent, 3-domain)
  risk_level: low
  requires_hitl: 'false'
  tags: 跨域诊断,端到端,分支,应用访问,传输,准入,cross-agent,delegation,wan,dc
  returns: 分层端到端诊断报告(准入层 + 传输层 + 应用层,含各层可达性与边界说明)
---

# 分支→DC 应用端到端可达性诊断(三域跨 Agent)

排查分支机构用户访问数据中心应用失败的根因。故障可能分布在三层:
**LAN 准入层**(认证/NAC/VLAN)、**WAN 传输层**(分支到 DC 的电路/隧道/SLA)、
**DC 应用层**(RBAC 权限 / fabric 路径可达性)。本 skill 逐层定位,按需委派 WAN 与 DC。

> **对端离线是常态**:agent 环境不可控。任一对端不可达时,按 frontmatter 的
> `degraded-capability` 交付可达层诊断,如实标注未排查层,绝不臆测离线域的结论。

## Steps

1. **LAN 准入层**。查询用户的 LAN 准入状态:用 `get_user_access` 传入 user_id。
   若未准入,用 `check_nac_policy` 查原因,必要时 `query_radius_logs` 查认证日志。

2. **判定故障层(脚本,确定性)**。运行 `scripts/classify_failure_layer.py`,输入 step 1
   的准入字段,输出标签:`lan_auth` / `lan_nac` / `delegate_transport` / `unknown`。
   **不要自己判断,用脚本算。**
   - 注:本 skill 的脚本在原 app-access-troubleshoot 判定基础上,把"准入正常"细分为
     "需先查传输(delegate_transport)",因为分支场景传输层故障常见。

3. **按故障层决定路径**:
   - `lan_auth` / `lan_nac`:故障在 LAN 准入层,本地已定位,直接 step 6 汇总,不委派。
   - `delegate_transport`:准入正常,先排查传输层。**委派 WAN**:
     `[DELEGATE:wan-agent]` 检查分支边缘(如 edge-br-sf)到 DC 边缘(edge-dc)的
     电路状态、IPsec 隧道、路径 SLA(时延/抖动/丢包),判断传输层是否降级。等结果返回。
     - 若 WAN 离线(不在 AVAILABLE PEERS 或委派失败):按 degraded-capability,如实说明
       传输层无法排查,继续 step 4 尝试应用层(传输与应用独立)。
   - `unknown`:信息不足,说明还需哪些信息,不委派。

4. **DC 应用层(视情况)**。若 step 3 传输层正常、或传输不可排查但需确认应用层:
   **委派 DC**:`[DELEGATE:dc-agent]` 检查用户对该应用的 RBAC 权限、DC 内部 fabric 路径
   可达性,必要时授予角色(DC 侧审批)。等结果返回。
   - 若 DC 离线:按 degraded-capability 如实标注应用层待恢复。

5. **不要重复委派同一个 agent**。每个对端在本次请求最多委派一次。

6. **汇总分层报告**:
   - 准入层:结论(来自 step 1/2)
   - 传输层:结论(来自 WAN,或"WAN 离线未排查")
   - 应用层:结论(来自 DC,或"DC 离线未排查")
   - 根因定位 + 下一步建议
   - **边界说明**:明确哪些层已排查、哪些因对端离线未排查。

## Notes

- 故障层判定**必须**用 step 2 脚本,确定性规则见 `references/diagnosis_runbook.md`。
- 三层相互独立:传输层不可排查**不**阻断应用层诊断(反之亦然),按可达对端尽量交付。
- 跨 agent 委派遵循平台规则:同一分解任务不重复委派同一 peer;对端触发审批(HITL)时,
  等审批与执行结果都返回再汇总。
- **降级不是失败**:交付"准入正常 + 传输待查(WAN 离线)+ 应用待查(DC 离线)"这样的
  边界报告,本身就是有价值的诊断结论——它排除了准入层,缩小了范围。
