# Branch→DC 应用可达性 — 诊断 Runbook

三域故障定位的确定性规则 + 对端离线时的降级处置。

## 故障层判定规则(step 2 脚本)

| 条件(按序,首个匹配生效) | failure_layer | 含义 |
|---|---|---|
| `auth_ok = false` | `lan_auth` | 认证失败,LAN 准入认证层 |
| `nac_compliant = false` | `lan_nac` | NAC 不合规,LAN 准入策略层 |
| `admitted = true` 且 `nac_ok ≠ false` 且 `vlan` 有值 | `delegate_transport` | 准入正常,先查 WAN 传输 |
| `admitted = false`(无更细信号) | `lan_auth` | 先按准入层排查 |
| 其余 | `unknown` | 字段不足,需补充 |

## 分层处置

- **lan_auth / lan_nac**:LAN 本地闭环,不委派。给出认证/NAC 修复建议。
- **delegate_transport**:委派 WAN 查分支边缘→DC 边缘的电路/隧道/SLA。
  - WAN 报传输降级(电路 down / 隧道 rekey 失败 / SLA breach)→ 根因在传输层。
  - WAN 报传输正常 → 继续委派 DC 查应用层。

## 对端离线降级(peer-offline 是常态)

| 离线对端 | 仍可交付 | 如实标注 |
|---|---|---|
| WAN | LAN 准入层完整诊断 + (DC 在线时)应用层 | 传输层"需 WAN,离线未排查" |
| DC | LAN 准入层 + (WAN 在线时)传输层 | 应用层"需 DC,离线待恢复" |
| WAN + DC | 仅 LAN 准入层 | 传输层 + 应用层均"对端离线未排查" |

**原则**:传输层与应用层相互独立,一端不可排查不阻断另一端。降级报告必须包含
"已排查层 / 未排查层 / 各层结论 / 下一步",让运维清楚边界。降级排除了准入层、
缩小了范围,本身就是有价值的结论,不是失败。
