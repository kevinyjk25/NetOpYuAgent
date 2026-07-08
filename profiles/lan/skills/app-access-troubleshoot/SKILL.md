---
name: app-access-troubleshoot
description: 端到端排查用户无法访问数据中心应用的问题。LAN 侧确认网络准入并用脚本判定故障层,再委派 DC 侧排查应用权限与可达性。用于"用户 X 访问应用 Y 失败/很慢/时通时断"这类跨域诊断。
allowed-tools: get_user_access, check_nac_policy, query_radius_logs
delegates-to: dc-agent
degraded-capability: |
  DC 离线时:仍完成 LAN 准入层的完整诊断(RADIUS 认证、NAC 合规、VLAN、802.1X),
  并用 step 2 脚本给出故障层判定。若判定为 lan_auth / lan_nac,诊断本就在本地闭环,
  不受影响,正常交付。若判定为 delegate_dc(准入正常、疑似应用层),则如实告知:
  "LAN 准入层正常,应用层(RBAC/可达性)诊断需 DC agent,当前 DC 离线,无法排查;
  已排除准入层原因,建议 DC 恢复后重试应用层诊断。"绝不臆测应用层结论。
metadata:
  skill_id: app_access_troubleshoot
  display_name: App Access Troubleshoot (cross-agent)
  risk_level: low
  requires_hitl: 'false'
  tags: 跨域诊断,应用访问,准入,cross-agent,delegation,troubleshoot
  returns: 端到端诊断报告(准入层 + 应用层 + 可达性)
---

# 跨 Agent 应用访问诊断

端到端排查用户访问数据中心应用失败的根因。问题可能在 **LAN 准入层**(认证/NAC/VLAN)
或 **DC 应用层**(RBAC 权限 / fabric 路径)。本 skill 先在 LAN 侧确认准入、用脚本把
原始查询结果**确定性地**判定成一个故障层标签,再据此决定是否委派 DC。

## Steps

1. 查询用户的 LAN 准入状态。使用工具 `get_user_access`,传入 user_id。
   如果工具结果显示未准入,再用 `check_nac_policy` 查原因,必要时用 `query_radius_logs`
   查认证日志。

2. 用脚本把准入查询结果判定成故障层标签。运行 `scripts/classify_failure_layer.py`,
   输入是 step 1 拿到的准入字段(admitted / nac_compliant / vlan / auth_ok),
   输出一个确定性的 failure_layer 标签:`lan_auth` / `lan_nac` / `delegate_dc` / `unknown`。
   **不要自己判断故障层,用脚本算** —— 阈值和规则在脚本里,确定且可复现。

3. 根据 failure_layer 决定下一步:
   - `lan_auth` / `lan_nac`:问题在 LAN 准入层,本地已能定位,直接进入 step 5 汇总,
     不要委派 DC。
   - `delegate_dc`:LAN 准入正常,问题在应用层(DC 范畴)。委派给 dc-agent:
     `[DELEGATE:dc-agent]` 检查用户对该应用的访问权限(RBAC)与 DC 内部路径可达性,
     必要时授予角色(需 DC 侧审批)。等委派结果返回。
   - `unknown`:信息不足,向用户说明还需要哪些信息。

4. (仅当委派了 DC)收到 dc-agent 的结果后,合并 LAN 侧与 DC 侧的发现。
   **不要重复委派同一个 agent。**

5. 汇总最终诊断报告:准入层状态(来自 step 1/2)+ 应用层结论(来自 DC,若委派了)+
   根因 + 建议操作。如果某层未排查(因为故障定位在更前面),如实说明。

## Notes

- 故障层判定**必须**用 step 2 的脚本,不要在 prompt 里自己判 —— 这是确定性规则,
  脚本保证一字不差、可复现。
- 详细的判定规则表与各故障层的处置建议见 `references/diagnosis_runbook.md`。
- 跨 agent 委派遵循平台规则:同一分解任务不重复委派同一 peer;DC 侧若触发审批(HITL),
  等审批与执行结果都返回后再汇总。
