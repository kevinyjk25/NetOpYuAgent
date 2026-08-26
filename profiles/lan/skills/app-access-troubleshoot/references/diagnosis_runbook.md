# 应用访问诊断 Runbook(reference)

> 本文件由 `app-access-troubleshoot` skill 在需要详细判定规则/处置建议时按需引用。
> 故障层的**判定**由 `scripts/classify_failure_layer.py` 确定性完成,本文档解释规则
> 含义与各层的处置动作。

## 故障层判定规则表

脚本按以下顺序判定(先匹配先返回):

| 顺序 | 条件 | failure_layer | 含义 |
|------|------|---------------|------|
| 1 | `auth_ok == false` | `lan_auth` | 认证失败,问题在 LAN 准入认证层 |
| 2 | `nac_compliant == false` | `lan_nac` | 已认证但 NAC 不合规,问题在 LAN 准入策略层 |
| 3 | `admitted == true` 且有 VLAN 且 NAC 未明确失败 | `delegate_dc` | LAN 准入正常,问题应在 DC 应用层 |
| 4 | `admitted == false`(无更细信号) | `lan_auth` | 未准入,先按 LAN 准入层排查 |
| 5 | 以上都不满足 | `unknown` | 信息不足,需补充字段 |

## 各故障层的处置建议

### lan_auth(认证失败)
- 用 `query_radius_logs` 查认证失败日志,定位:账号密码错误 / 用户不存在 /
  Portal·802.1X 失败 / RADIUS 异常。
- 常见根因:凭据错误、账号被禁用、RADIUS 与认证源不同步。
- **不委派 DC** —— 问题在准入认证层,DC 应用权限正常与否无关。

### lan_nac(NAC 不合规)
- 用 `check_nac_policy` 查不合规项:终端合规检查(补丁/杀软/证书)未通过,被隔离到
  受限 VLAN。
- 处置:让终端达成合规要求,或调整 NAC 策略(变更类操作需审批)。
- **不委派 DC**。

### delegate_dc(应用层 / DC 范畴)
- LAN 准入已正常(认证通过、NAC 合规、分到了业务 VLAN),用户仍访问不了应用,
  说明问题在应用层或 DC 内部网络。
- 委派 dc-agent 排查:
  1. 应用 RBAC:用户是否持有访问该应用的角色(`dc_check_user_app_access` /
     `dc_get_app_acl`)。无权限则授予合适角色(`dc_grant_app_access`,需 DC 审批)。
  2. DC 内部可达性:`dc_fabric_path_trace` / `dc_loadbalancer_pools` /
     `dc_vxlan_vni_lookup` 排查 fabric 路径与后端健康。
- **只委派一次,不重复委派同一 agent。**

### unknown(信息不足)
- 准入字段不足以判定,向用户说明还需提供:用户是否能认证上网(auth_ok)、
  终端是否合规(nac_compliant)、是否拿到 IP/VLAN(admitted/vlan)。

## 跨 agent 协作约定

- 委派 DC 后原始请求 park,等 DC 的审批与执行结果回来再汇总。
- 同一分解任务不重复委派同一 peer。
- 汇总报告必须区分:哪些是 LAN 侧确认的、哪些来自 DC 委派结果;某层若未排查
  (因故障定位在更前面),如实说明而非假设正常。
