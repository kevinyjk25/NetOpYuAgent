# 实跑指南 — app-access-troubleshoot 跨 agent skill

这个示例 skill 把 4 种能力串在一个真实运维场景里:
**tool 调用 + script 执行 + reference 引用 + 跨 agent 委派**。

## skill 结构

```
profiles/lan/skills/app-access-troubleshoot/
  SKILL.md                              # 主流程(Anthropic 标准格式)
  scripts/classify_failure_layer.py     # 确定性判定故障层(script-as-tool)
  references/diagnosis_runbook.md        # 详细规则表 + 处置建议(按需引用)
```

四种能力分别体现在:
- **tool**:`allowed-tools: get_user_access, check_nac_policy, query_radius_logs`,
  step 1 查 LAN 准入。
- **script**:step 2 调 `scripts/classify_failure_layer.py`,把准入字段确定性判定成
  `lan_auth / lan_nac / delegate_dc / unknown` —— 这步**不让 LLM 判,用脚本算**。
- **reference**:`references/diagnosis_runbook.md` 提供完整规则表与各层处置建议。
- **跨 agent 委派**:故障层 = `delegate_dc` 时,`[DELEGATE:dc-agent]` 排查应用权限与
  DC 可达性。

## 启动两个 agent(和你之前 LAN↔DC 测试一样)

```bash
# 终端 1 — LAN agent
AGENT_PROFILE=lan AGENT_ID=lan-agent AGENT_DISPLAY_NAME="LAN Agent" \
  A2A_BASE_URL="http://localhost:8000/api/v1/a2a" \
  AGENT_PEERS="http://localhost:8001/api/v1/a2a" HITL_BACKEND=core \
  uvicorn main:app --port 8000

# 终端 2 — DC agent
AGENT_PROFILE=dc AGENT_ID=dc-agent AGENT_DISPLAY_NAME="DC Agent" \
  A2A_BASE_URL="http://localhost:8001/api/v1/a2a" \
  AGENT_PEERS="http://localhost:8000/api/v1/a2a" HITL_BACKEND=core \
  uvicorn main:app --port 8001
```

启动日志里应能看到:
- `SkillLoader[mock, profile=lan]: N skills loaded`(N 比之前 +1)
- `Skill scripts: 1 script tool(s) registered (app_access_troubleshoot__classify_failure_layer)`
- `Skill scripts: 1 tool(s) wired into ToolRouter`

## 测试查询(LAN agent 的 chat)

**场景 A — 应用层问题(会委派 DC)**
```
诊断用户 alice 访问应用 crm 失败的原因
```
预期流程(看 JOURNAL tab 的 live 事件流):
1. SKILL_LOAD: app_access_troubleshoot
2. tool: get_user_access(alice)→ 准入正常
3. tool: app_access_troubleshoot__classify_failure_layer → `delegate_dc`
4. DELEGATE: dc-agent(应用权限 + 路径)→ park → DC 审批 → 结果返回
5. 汇总报告(LAN 准入 OK + DC 应用层结论)

**场景 B — 准入层问题(不委派 DC)**
```
用户 bob 连不上网,也打不开 crm,帮我诊断
```
若 bob 准入异常,script 判定 `lan_auth`/`lan_nac` → **不委派 DC**,本地定位后汇总。
(取决于 mock 数据里 bob 的准入状态)

## 重点观察:script 是否被调用

这是验证 script 执行的关键。在 JOURNAL 的 RECENT ACTIVITY 里,应看到一条
`tool: app_access_troubleshoot__classify_failure_layer` 的调用记录 —— 说明 LLM 按
skill 指引调了脚本,脚本确定性地给出了故障层,而不是 LLM 自己拍脑袋判断。

如果 LLM 没调脚本、自己判了故障层 → 说明它偏离了 skill(软提示的固有问题)。这正是
我们之前讨论的"硬约束 vs 软提示"观察点:当前 skill 仍是软提示,脚本是否被调用取决于
模型遵循度。要强制,得上之前设计的 step gate(尚未实现)。

## 离线验证(不起服务,纯逻辑)

```bash
python3 - << 'EOF'
import asyncio
from skills.loader import SkillLoader
from skills.script_runner import build_script_tools
d = SkillLoader(mode="mock", profile="lan").skill_definitions()["app_access_troubleshoot"]
tools = build_script_tools("app_access_troubleshoot", d["skill_dir"])
fn = tools["app_access_troubleshoot__classify_failure_layer"]
# 准入正常 → 应判 delegate_dc
print(asyncio.run(fn({"admitted": True, "nac_compliant": True, "vlan": 20, "auth_ok": True})))
EOF
```
