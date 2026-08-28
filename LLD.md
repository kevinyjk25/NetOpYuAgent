# NetOpYuAgent 低层设计 / Low-Level Design

## 中文

### 1. 文档范围

本文定义 DSH/Hermes Harness Adapter、共享 Python bridge/Worker、Domain Effect Runtime、Network/Service L0 Skill、backend、MCP、A2A 和持久化组件的实现合同。这里的“必须”表示代码应 fail closed 的约束。

### 2. 仓库模块

| 路径 | 实现职责 |
|---|---|
| `dsh-plugin-netopyu/src/index.js` | DSH 组合入口、工具定义、审批卡、Memory/Capability service |
| `dsh-plugin-netopyu/src/bridge.js` | Unix Socket Worker 与短进程 fallback transport |
| `dsh-plugin-netopyu/src/hitl-store.js` | HITL、batch、grant、trajectory、continuation SQLite |
| `dsh-plugin-netopyu/src/a2a.js` | DSH remote subagent provider 与 continuation 工具 |
| `hermes-plugin-netopyu/` | Hermes `plugin.yaml` 与公开 `register(ctx)` 加载入口 |
| `hermes_adapter/plugin.py` | Hermes 工具/Skill/command 投影与精确计划审批绑定 |
| `hermes_adapter/client.py` | 同步、短连接、带 request id 的 Unix Socket client |
| `hermes_adapter/pending.py` | 进程内 plan nonce 与远端 continuation 一次性绑定 |
| `hermes_adapter/comparison.py` | DSH/Hermes Network Runtime 不变量 A/B 门禁 |
| `dsh_adapter/bridge.py` | manifest、只读调用、prepare/execute/inspect/audit/workflow API |
| `dsh_adapter/worker.py` | 持久化 JSON-lines Unix Socket 服务、请求隔离和 tracing |
| `dsh_adapter/backend.py` | mock/pragmatic backend 生命周期和公共分页工具 |
| `dsh_adapter/local_dc_peer.py` | loopback-only mock A2A peer |
| `dsh_adapter/scoped_services.py` | session/operator memory 和 profile capability retrieval |
| `effect_runtime/` | 领域中性 façade 与跨 Service/Network reconciliation |
| `network_runtime/engine.py` | Domain Effect Runtime 共享主状态机 |
| `network_runtime/contracts.py` | schema v6、Plan/Evidence/Outcome 与状态迁移 |
| `network_runtime/l0_skills.py` | L0 Skill/step/IntentSpec 注册表和 hash |
| `network_runtime/validation.py` | 参数规范化、类型、来源、实体与风险校验 |
| `network_runtime/policies.py` | 工具版本、preflight、verifier、compensator 合同 |
| `network_runtime/verifiers.py` | typed postcondition registry |
| `network_runtime/compensators.py` | 逆操作和补偿结果 registry |
| `network_runtime/workflows.py` | reviewed L1 workflow template 与阶段约束 |
| `network_runtime/journal.py` | SQLite 状态、nonce、事件哈希链和审计 |
| `network_runtime/provider_contracts.py` | Network provider capability id/version/role registry |
| `network_provider/mcp_observer.py` | identity-pinned read-only Network Observer MCP |
| `network_provider/mcp_actor.py` | identity-pinned trusted Durable Network Actor MCP |
| `network_provider/actor.py` | operation 执行、幂等重放、读回 reconciliation 与精确恢复 |
| `network_provider/actor_store.py` | SQLite/WAL operation、snapshot、lease、fence 与 Actor hash chain |
| `network_provider/models.py` | strict Observer/Actor structured result models |
| `network_lab/manifest.py` | schema-v1 lab 目标权威与路径/标识校验 |
| `network_lab/containerlab.py` | 无 shell 生命周期、FRR CLI、probe、fault 和快照恢复 |
| `network_lab/tools.py` | lab 节点到 pragmatic 工具合同的投影 |
| `service_layer/models.py` | MCP strict structured-output Pydantic 合同 |
| `service_layer/store.py` | 多进程事务、revision、幂等、审计与一次性 seed |
| `service_layer/mcp_server.py` | 六个 official-SDK MCP domain server |
| `integrations/clients/mcp_client.py` | official MCP stdio/Streamable HTTP client、identity/schema binding |
| `runtime/tool_results.py` | durable 大结果外置与引用解析 |
| `profiles/` | LAN/DC/WAN/mock callables、metadata 和 L1 Skills |
| `tools/` | 公共分页和 pragmatic network tools |
| `integrations/` | MCP、OpenAPI 和统一路由 |
| `agent_memory/` | operator/session scoped memory |
| `retrieval/` | CJK-aware BM25 与可选 hybrid retrieval |

### 3. DSH 插件启动

`apply(ctx, config)` 执行顺序：

1. 解析项目根、Python、profile、backend 和持久化路径；
2. 从 Python bridge 获取 profile manifest 与 Skill manifest；
3. 根据 `NETOPYU_DSH_ENABLE_DESTRUCTIVE` 决定是否投影写工具；
4. 初始化 HITL store，恢复中断状态；
5. 初始化 Tool Guard、Memory、Capability 和 A2A provider；
6. 通过 `ctx.provide` 注册 DSH service；
7. 注册普通工具、HITL 工具、A2A 工具和 trajectory 工具；
8. 在 Cordis effect cleanup 中关闭 SQLite。

启动失败不得留下“部分写工具已注册”的状态。manifest、Python、backend 或存储初始化失败应使插件启动失败。

#### 3.1 Hermes 插件启动

Hermes 使用官方 `plugin.yaml + register(ctx)` API：

1. `HermesAdapterConfig` 解析 profile、Worker socket、operator id 和破坏性工具开关；
2. `HermesWorkerClient.ping()` 必须成功；
3. 启用写工具时，Hermes 必须提供 `register_command`，否则在注册任何网络工具前失败；
4. 先注册 `/netopyu-approve`、`/netopyu-deny` 和 A2A 等价命令，再注册工具；
5. read handler 调用 Worker `invoke`，并以同一 task id 调用 `workflow-observe`；write handler 只调用 `runtime-prepare`；
6. write handler 从模型可见结果删除 `execution_nonce`，将 nonce 与 plan id/hash 保存到 `PendingActions`；
7. 只有 slash command handler 可原子领取绑定并调用 `runtime-execute`；
8. 插件重启后 `PendingActions` 清空，Runtime 中尚未执行的计划保持不可执行并最终过期；
9. canonical profile Skill 通过 `ctx.register_skill` 注册；`netopyu_skill_view` 或 Hermes `skill_view` hook 启动 reviewed workflow，写阶段采用 Harness-neutral 审批协议。

Hermes tool handler 按官方合同返回 JSON string；错误返回结构化 `ok=false`，不能把异常转写成成功。Hermes 内置危险命令审批不能替代 Network L0 plan 审批。

### 4. Bridge/Worker 协议

Worker 使用 Unix Domain Socket，消息为一行一个 JSON object。每个请求包含：

```json
{
  "id": "correlation-id",
  "command": "manifest|invoke|runtime-prepare|runtime-execute|runtime-inspect|runtime-audit|workflow-*",
  "profile": "lan",
  "tool": "restart_service",
  "args": {},
  "include_destructive": false,
  "allow_destructive": false,
  "l0_skill_id": "network.service.restart"
}
```

约束：

- 非 object、超长、缺字段和未知 command 返回结构化错误，不终止 Worker；
- `invoke` 只允许只读工具；任何直接写调用均拒绝；
- `execute` 必须携带 plan id/hash、execution nonce、approval request、actor 和请求级 `allow_destructive=true`；
- Worker 每个请求建立独立 backend 生命周期并在结束时关闭客户端与结果 store；
- Socket 残留只能在确认无监听者后删除。

### 5. Network/Service L0 Skill 合同

`L0SkillContract` 是 frozen、版本化的效果定义，包含：

- `skill_id`、`version`、`tool_name`、`tool_contract_id`；
- `intent_kind`、`target_fields`、`allowed_profiles`；
- 固定 `steps`；
- 由规范 JSON 计算的 `contract_hash`。

默认步骤：

| 顺序 | step | 阶段 | 失败策略 |
|---:|---|---|---|
| 1 | `validate_parameters` | prepare | clarify or reject |
| 2 | `compile_intent` | prepare | clarify or reject |
| 3 | `preflight` | prepare | reject |
| 4 | `approval` | approve | reject |
| 5 | `revalidate` | execute | abort without write |
| 6 | `execute` | execute | reconcile |
| 7 | `verify` | verify | compensate or escalate |
| 8 | `compensate` | compensate | manual intervention；仅有合同的 Skill |
| 9 | `audit` | terminal | fail closed |

每次写请求必须解析到精确的 `(skill_id, version, contract_hash)`。同一 raw tool/capability 允许注册多个语义合同；存在多个候选时，请求必须显式提供 `l0_skill_id`，Runtime 不做猜测。未知、未绑定、profile 不允许或 contract hash 不一致都在 prepare 阶段拒绝。

L0 v2 在既有执行合同之上增加 authoring/compiler 层：

- `AtomicEffect` 定义可独立执行的 S1；
- `DerivedEffect(mode=constraint)` 只允许固定值、缩小范围、增强审批，形成约束式 S11；
- `DerivedEffect(mode=extension)` 允许增加参数、前置 Observation、验证谓词和 desired-state 字段，但不能改写父语义，形成扩展式 S11；
- `CompositeEffect` 用显式 DAG、检查点和逆序补偿组合 S1+S2+…，形成 Saga；
- 编译器递归展开继承、检查单调安全规则，并生成 immutable hash；Runtime 不解释继承；
- Catalog 支持同 id 多版本和同 capability 多语义合同，`explain`、`diff`、`graph` 供代码审查使用。

当前 v2 示例是兼容式 SDK/编译原型。现有 DSH 执行路径继续使用经 Core-72 验证的 v1 合同，直到 Provider Gateway、表达式/Resolver、Catalog loader 和执行故障认证完成。

`network_runtime.l0.promotion` 实现离线 L1 → L0 流程：

1. `load_skill_source()` 使用统一 `skills.skill_format` 解析标准 `SKILL.md`；
2. `CapabilityCatalogManifest` 固定 Provider/version、capability role、profile 和输入输出 Schema；
3. `promotion_prompt()` 输出包含 L1、Catalog、L0 Schema 和禁止猜测规则的有界 JSON packet；
4. `assess_promotion()` 把 source/catalog hash 注入候选 labels，随后进行交叉检查和严格编译；
5. `package_promotion()` 仅为零 error 候选生成 `SKILL.md`、`candidate.yaml`、`compiled.json` 和 `report.json`；
6. `review_promotion()` 重算 proposal 与文件 hash，最多写入一个 approve/reject 记录；
7. 所有报告固定 `executionEligible=false`、`autoActivated=false`，不存在 Runtime loader 副作用。

### 6. IntentSpec 与参数编译

编译器输入为 profile、tool metadata、原始 arguments、argument provenance 和 L0 Skill。输出只能是成功的 `CompilationResult` 或明确错误列表。

校验顺序：

1. arguments 必须是 object；
2. 拒绝未知字段；
3. 检查 required；
4. 校验并规范化 string/integer/number/boolean/array/object；
5. 校验 enum、空值和范围；
6. 在 mock 模式校验用户、设备、服务、应用等目标实体存在；
7. 校验 provenance 值；高风险关键字段不得来自未确认猜测；
8. 根据 action type、tool 和参数计算 risk；
9. 生成 normalized arguments、targets、constraints、desired state；
10. 生成 `arguments_digest` 和 `intent_hash`。

`IntentSpec` 不包含自由文本执行指令。模型解释只能保留在 DSH 会话，不能成为 backend 命令。

### 7. PreparedPlan schema v6

核心字段：

```text
plan_id, profile, tool_name, tool_version, action_type
provider_identity, input_schema_digest, output_schema_digest
capability_id, capability_version, provider_role
arguments, argument_provenance, targets
risk_level, risk_reasons
preflight[]
verification_contract, rollback_contract
l0_skill_id, l0_skill_version, l0_contract_hash
intent_spec, intent_hash, step_contract
workflow_run_id, workflow_template_hash
created_at, expires_at, plan_hash, state, schema_version
```

`plan_hash` 对除自身和可变 state 外的规范计划内容计算 SHA-256。审批、grant、execute 和 audit 必须再次验证 hash。任何计划字段变化都产生不同 hash，旧批准不再有效。schema v6 在 provider identity 和输入/输出 schema digest 之外，把 capability id/version/role 纳入批准边界；v5 及更早 hash 仅用于读取既有 journal，不能创建新计划。

### 8. 状态机

```mermaid
stateDiagram-v2
    [*] --> plan_ready
    plan_ready --> approved
    plan_ready --> rejected
    plan_ready --> expired
    approved --> executing
    approved --> expired
    executing --> verifying
    executing --> execution_failed
    executing --> outcome_indeterminate
    executing --> precondition_changed
    execution_failed --> rolling_back
    execution_failed --> manual_intervention_required
    outcome_indeterminate --> verifying
    outcome_indeterminate --> rolling_back
    outcome_indeterminate --> manual_intervention_required
    verifying --> verified_success
    verifying --> rolling_back
    verifying --> manual_intervention_required
    rolling_back --> rollback_verified
    rolling_back --> manual_intervention_required
```

只有 `verified_success` 或 `rollback_verified` 能表示已经验证的确定状态。`precondition_changed`、`manual_intervention_required`、`rejected` 和 `expired` 是安全终态。非法迁移抛出 `StateTransitionError`。

### 9. Prepare 算法

1. `resolve_contract` 和 `L0SkillRegistry.resolve`；
2. `compile_parameters`；
3. `compile_intent`；
4. `assess_risk`；
5. 调用合同指定 preflight read；
6. 将输出转换为 typed `Evidence`；
7. reviewed workflow 存在时验证阶段和 observation 前置条件；
8. 创建具有 TTL 的 `PreparedPlan`；
9. 原子写入 plan 与 `plan_created`；
10. 返回 `plan` 或 `clarification_required/errors`，不得同时返回两者。

### 10. Approval 与 Harness Guard

DSH 插件从计划生成审批摘要。`allowed-once` 后：

- 生成随机 execution token；
- SQLite 只保存 token SHA-256 digest；
- grant 绑定 operator、request id、tool、canonical arguments、plan id/hash 和过期时间；
- conditional update 只允许 `issued -> consumed` 一次；
- 重启将未完成 grant 标记 orphaned；
- reject/timeout 会撤销或终止计划。

Runtime journal 另保存 execution nonce 的 digest。Tool Guard 和 Runtime nonce 构成两层一次性保护。

Hermes Adapter 使用不同的交互层，但不改变 Runtime 合同：

- write tool 只返回 `approval_required`、完整 plan 和 `/netopyu-approve PLAN_ID FULL_HASH`；
- execution nonce 只保存在插件进程内的 `PendingActions`，不返回模型、不写日志、不落盘；
- slash command handler 要求精确两个参数，以 constant-time hash 比较后先原子领取 pending binding；
- handler 使用配置的 operator id 生成唯一 approval request id，再调用相同 `runtime-execute`；
- hash 错误不消费 binding；正确领取、过期或重启后均不能重用；
- P0.5 的 operator id 是本地配置身份，不构成生产不可抵赖身份，P1 必须接企业身份上下文。

### 11. Execute/Verify 算法

1. 读取计划并校验 schema/hash/状态/TTL；
2. 校验请求级破坏性授权和 approval identity；
3. 原子消费 nonce；
4. 转为 `executing`；
5. 重新执行 preflight，并与原 snapshot 比较；
6. 状态漂移则 `precondition_changed`，不写；
7. 调用写 callable 一次；
8. 无论写返回是否“成功”，进入 verifier；
9. verifier 用独立 read callable 生成 typed evidence；
10. 通过则 `verified_success`；
11. 失败且有 compensator 时执行补偿并再次验证；
12. 无安全补偿或结果不确定则 `manual_intervention_required`；
13. 写入 terminal audit step。

### 12. Evidence 与 verifier

Evidence 字段为：`evidence_type`、`source`、`target`、`observed_at`、`value`、`fresh`、`passed`、`predicate`、`expected`。

Verifier 必须：

- 使用与写工具独立的读路径；
- 解析 typed facts，而不是匹配模型总结；
- 明确 expected predicate；
- 拒绝 stale 或 error marker；
- 在无法确定时返回失败/不确定，而不是乐观成功。

### 13. Workflow Runtime

Reviewed workflow template 从 canonical L1 `SKILL.md` 编译，hash 绑定模板版本。Workflow session 持久化已观察事实与完成阶段。mutating plan 必须满足：

- 当前阶段允许该 L0 Skill；
- 必需 read observation 已成功；
- plan 的 `workflow_run_id` 和 `workflow_template_hash` 匹配；
- 工具结果只能推进已定义阶段；
- final verification 不得被中间写结果替代。

### 14. A2A continuation

Remote `input-required` 结果不被当作成功。DSH 持久保存 continuation 并展示新卡片；Hermes 将结构化远端 plan hash 与 continuation 保存在插件进程，并返回用户专属 `/netopyu-a2a-approve` 命令。恢复时发送 `resume_interrupt_id + operator_decision`；两者必须成对出现。缺少结构化远端 plan hash 时 Hermes 拒绝建立可批准 continuation。

### 15. Journal 与审计

每个 plan 的事件链：

```text
event_hash = SHA256(canonical(event_without_hash) + prev_event_hash)
```

首事件使用 `GENESIS`。`runtime-audit` 重算：

- plan hash；
- 事件顺序；
- `prev_event_hash` 连续性；
- 每个 event hash；
- 终态与 record 一致性。

审计失败只报告，不修复被篡改事件。旧无 hash 记录只允许一次性迁移，不能用迁移逻辑“治愈”后续篡改。

### 16. 异常策略

| 异常 | 行为 |
|---|---|
| 参数不完整/歧义 | 返回 clarification；不建可执行计划 |
| 未知 L0 Skill/工具绑定 | 拒绝 |
| backend 未配置 | fail closed |
| 审批拒绝/过期 | `rejected`/`expired` |
| grant/hash/nonce 不匹配 | 拒绝且不写 |
| precondition 漂移 | `precondition_changed` |
| 写前 transport 失败 | `execution_failed` |
| 写是否到达不确定 | `outcome_indeterminate` 后强制验证 |
| postcondition 失败 | 补偿或人工介入 |
| rollback 验证失败 | `manual_intervention_required` |
| Worker 单请求异常 | 返回错误；Worker 保持健康 |
| A2A timeout/loop/unreachable | 当前委派失败，不伪造结果 |

### 17. 扩展规则

增加写能力必须同时提交：工具 metadata、L0 Skill、ToolContract、目标/参数规则、preflight、verifier、必要的 compensator、profile 映射、测试和双语文档更新。只增加 callable 或 Skill prompt 不构成可执行写能力。

### 18. P0.75-A Provider 细节

`LabManifest` 只接受 schema v1、manifest 目录内 topology、唯一节点名、FRR device、IP literal probe 和 `eth1+` fault target。`ContainerlabProvider` 使用 `create_subprocess_exec` argv，不调用 shell；read command 只能匹配单行 `show`，config line 必须匹配固定 FRR 白名单。

`apply_config` 在同一 backend session 保存规范化 running-config，然后发送一次 `vtysh` 命令序列。`device-config-snapshot-v1` compensator 调用 `frr-reload.py --reload`，之后通过 `get_device_config` 比较规范化摘要。带 `verification_probe_id` 的 plan 还必须得到 transmitted=received 的 manifest probe evidence。Provider snapshot 只在 execution session 内可用；进程崩溃后的不确定写禁止重放并进入人工处置。

Access manifest 额外解析 `LabUser` 与 `LabApplication`。`set_user_admission` 仅操作固定
endpoint/interface，并恢复 manifest 中的固定应用路由；`set_application_access` 仅增加或删除
应用容器内固定用户 `/32` blackhole。`application_probe` 使用 argv 形式从固定用户容器执行
`wget` 到固定 URL。LAN/DC grant 的 verifier fresh-read 同一实际状态；失败由 inverse-tool
恢复 preflight typed evidence。

### 19. P0.75-B 清单与验收细节

`LabDevice.expected_bgp_neighbors` 为可选非负整数；`verify` 只把 BGP summary 中
`State/PfxRcd` 为数字的邻居计为 Established。控制面验证最多等待 30 秒收敛，再执行
数据面探测。`LabUser.route_prefixes` 是 1–16 个归一化 CIDR，必须包含 legacy
`application_prefix`；准入恢复逐条执行 `ip route replace`，因此接口 down/up 后不会遗漏
已审核的 Internet 或业务前缀。

`small_production_lab.py` 对基线 HTTP、主出口路由和企业 BGP 聚合做额外断言。故障流程
只允许 manifest 中的 `primary-internet-uplink`，先证明 Core1 → Core2 → Edge2，再恢复
接口并证明 Core1 → Edge1 回切。

`LabLink` 为每条链路保存两个 `LabLinkMember(node, interface, address)`、relationship 和
path_role。加载器拒绝重复接口/地址、不同子网、未知节点及与 `topology.clab.yml` 不完全
相等的 wiring。`address_index` 将 hop IP 映射到 node/interface/link。

`trace_path` 只接受精确 endpoint ID，执行固定 argv 的
`traceroute -n -m 16 -w 2 -q 1`。解析器逐跳验证地址存在、当前节点与上一节点同属返回
链路、最终地址属于目标 endpoint；任何 `*`、未知 IP、断裂 adjacency 或未到达目标均返回
非成功和 `fail_closed=true`。`enforcement_path` fresh-read 用户 endpoint operstate 与应用
服务器源 `/32` route，只在两者允许时运行路径验证。四个工具均为 read-only，LAN/DC
profile 共同可见；只有带 typed links 的 manifest 才投影这些工具和对应 Skill。

### 20. P0.75-C Fabric 合同与执行细节

`LabFabric` 解析 mode、ASN、RR、VTEP、`LabFabricVlan` 与
`LabFabricAttachment`，并与 topology node/interface wiring 做精确校验。VLAN/VNI 必须
唯一；access 恰有一个 VLAN，trunk 至少两个；endpoint、device、interface 和 tenant 必须
全部来自 manifest。

只读工具直接解析 `bridge -j vlan`、`ip -j -d link type vxlan`、
`show bgp l2vpn evpn summary json`、`show evpn vni json` 与 EVPN RIB JSON。
`fabric_set_access_vlan` 只使用固定 argv，在 execution session 保存 typed bridge/PVID
快照。`fabric-access-vlan` verifier fresh-read 目标 PVID/bridge，并按需运行预声明 probe；
`fabric-access-vlan-snapshot-v1` 恢复并比较稳定 typed facts 的完全相等性。

参数编译器把 `vlan_id` 限制为 1–4094、`route_type` 限制为 2/3/5；目标仍按 inventory
唯一解析。ToolContract、L0 Skill、L1 reviewed workflow、verifier 和 compensator 必须同时
注册，否则启动或 prepare fail closed。

### 21. P0.8 Service MCP 与 Effect Runtime 实现合同

`config.service-lab.yaml` 与 `config.small-production-lab.yaml` 将每个 Service domain 声明为
独立 stdio MCP server。pragmatic backend 使用官方 SDK 完成 initialize、server info、分页
tool discovery 和 structured call；不支持 `transport=mock`。连接失败、identity/version 不符、
重复工具名、schema discovery 失败和清理异常都不能降级为本地 mock。

`MCPToolSpec` 为每个 tool 保存 server name/identity/version、input/output JSON schema、metadata
和 SHA-256 digest。受信写工具要求 `trusted_for_writes=true`、配置的期望 identity/version、
匹配的 `netopyu.contract_id` 和 `structured-content-required-v1`。`PreparedPlan` 保存这些值；
execute 重新连接 provider 后逐项比较，任一变化抛出 `PlanIntegrityError`，写不发送。MCP
transport success 但 structured payload 明确 `ok=false` 同样是 Runtime failure。

Service SQLite 使用 WAL、foreign keys、进程内 `RLock` 和跨进程 `BEGIN IMMEDIATE`。seed
仅在缺少 `store_metadata.seed_version` 时执行。entitlement/platform 写在单事务中完成：

1. 验证 approved/open change；
2. 读取并比较 `expected_revision`；
3. 检查幂等记录；只有当前状态仍等于原 after snapshot 才允许 replay；
4. 修改业务状态并单调递增 revision；
5. 同事务写 idempotency 与 audit row。

Access Policy 与 Platform 的 restore tool 标记 `internal_only`，不会进入 Harness manifest。
compensator 从 plan preflight 提取精确稳定 snapshot，以执行后 revision 调用 restore，再通过独立
read verifier 比较稳定事实。revision 是并发 token，补偿后继续单调增长，不要求回到旧数字。

`reconcile_service_network_access` 是只读 composite：并行读取 entitlement、network enforcement
和两条 CMDB binding，随后运行 manifest-bound HTTP probe。它返回
`cmdb_network_binding_missing`、`desired_enforcement_mismatch`、
`enforcement_allows_but_data_plane_failed`、`data_plane_bypasses_denied_policy` 或 `none`。
这些读取不是跨系统原子快照；所有写计划仍在 approval 后做 execution-time preflight revalidation。

### 22. P0.9/P1.0 Network Provider 合同、证据与 Actor 算法

`ProviderCapabilityRegistry` 以唯一 `capability_id` 为主键，以 tool name 为兼容索引。每条合同
固定 `capability_version`、`provider_role=observer|actor` 和 `action_type`。当前 registry 有
30 个 observer 与 12 个 actor capability。启动时，任何声明 `domain=network` 的外部 MCP tool
都必须逐字段匹配 registry；未知 tool、错误角色、错误 action 或版本漂移均中止 backend 初始化。

`network_provider.mcp_observer` 包装已审核的 `ContainerlabProvider`/`LabToolAdapter`，但只从
当前 profile callable 集合与 observer registry 的交集注册工具。它不注册写/restore callable，
也不拥有写凭据。每次调用生成严格 `NetworkEvidenceEnvelope`：

```text
ok/code/correlation_id/observed_at/simulation
provider_identity/capability_id/capability_version
payload_digest/content_type/payload
```

`payload_digest` 是 canonical JSON 的 `sha256:` 摘要。MCP client 收到
`result_contract=network-evidence-envelope-v1` 后按以下顺序失败关闭：字段完整性 → envelope
`ok=true` → 实际 initialize server identity/version 与 envelope identity 完全相等 → tool discovery
capability id/version 与 envelope 完全相等 → `observed_at` 为带时区 ISO-8601 且不早于 300 秒、不超前 30 秒 → payload digest
相等。全部通过后，兼容 content 才被替换为 payload JSON/text；原 envelope 保留在
`MCPCallResult.evidence_envelope`。payload 内部的 `ok=false` 是合法负面观测，不等于 provider
调用失败。

backend 注册 Observer 和显式 `trusted_for_writes` 的 Actor MCP；同名读写分别覆盖 backend 的
本地 callable。`ToolContract` 绑定 capability id/version、`provider_role=actor`、provider kind、
server identity/version、input/output schema digest、declared contract 与 structured result。
Actor 的 `profiles` 声明在注册时按当前 agent profile 过滤；内部参数从模型 schema 删除，仅由
`BackendSession.invoke_effect` 注入。

`ActorStore` 在 `BEGIN IMMEDIATE` 内创建 operation，并持久化 arguments/preflight/snapshot digest、
desired state、target key 和 fence token。target 的跨进程文件锁包围读写；SQLite lease 阻止另一
operation 在有效期内占用相同 target。状态机为 `prepared → executing → applied`，异常进入
`not_applied|outcome_indeterminate|manual_intervention`，补偿为 `restoring → restored`，成功终结为
`committed`。每次变化追加独立 Actor event hash chain。

同一 operation id 的 immutable 字段不一致立即拒绝。`prepared` 可在 snapshot 未漂移时恢复执行；
`executing/outcome_indeterminate` 只能读回 desired 或 snapshot，绝不重发；`applied` 重试只在
desired 仍成立时返回原结果。启动 reconciliation 使用同一规则。补偿工具忽略模型提供的旧状态，
按 operation id 加载精确 durable snapshot。Runtime 内部 finalizer 将 `verified_success` 映射为
`committed`、`rollback_verified` 映射为 `restored`，然后释放 lease。人工介入状态不释放安全
边界：target 被 durable quarantine，后续 operation 失败关闭，直到受审人工解除流程处理。

本地 fencing token 尚不能被 Containerlab/设备原生校验；当前实现依靠单主机文件锁和 SQLite
认证 crash safety。跨主机 HA 必须用远端日志/队列、leader fencing 和设备/控制器 idempotency/CAS。

### 23. P1.1 Capability SPI、Read PEP、Terminal Envelope 与 Saga 算法

`CapabilityContract.from_metadata()` 将 backend metadata 标准化为 observation/effect、domain、
provider identity/kind、schema digest、effect semantics、sensitivity、required roles、scope fields
和 freshness limit。`BackendSession` 以结构化 SPI 提供 `describe_capability()`、
`invoke_observation()`、`invoke_effect()` 和 `finalize_effect()`；Runtime 不根据传输协议名称分支。

`invoke_read()` 先编译参数，再由 `ObservationPolicy` 检查 authenticated subject、role、clearance、
purpose 和每个 `field:value` scope；拒绝发生在 Provider callable 前。DSH/Hermes Adapter 传递
operator/session/purpose，manifest 同时投影 canonical capability contract。

`ExecutionOutcome.terminal_envelope()` 删除原始 Provider result，只保留 digest、Runtime 终态、
typed evidence、error 和 compensation 状态。DSH write handler 只把该 JSON 返回模型；Hermes
slash approval 也优先返回相同封装。

`SagaCoordinator` 维护 `effect_sagas`、`effect_saga_steps` 和 `effect_saga_events`。
`SagaDefinition` 哈希固定步骤、domain、capability、依赖和 compensation capability。正向步骤只
能在依赖 `verified` 后绑定 immutable plan；失败把已验证步骤按逆序标记为
`compensation_required`。每个补偿仍绑定一个新 L0 plan；未知或不可补偿状态进入
`manual_intervention_required`。Saga 事件另有 SHA-256 链，`recoverable()` 不重放 Provider write。

### 24. Runtime A/B 基准实现

`evaluation.runtime_comparison` 实现两个执行器。参考执行器复用 DSH manifest 的 required/type/additional-properties Schema 规则和通用审批结果，然后直接调用 `BackendSession`；受控执行器调用真实 `NetworkRuntime.prepare/execute/invoke_read/audit`。每个场景返回相同结构的 `PathObservation`，并由独立 Oracle 判断是否满足安全或正确性目标。

Core-72 场景集固定为 72 项，包括 8 个有效操作和 64 个故障/风险控制，覆盖 LAN、DC、WAN 和跨域 Saga。时延 campaign 预热后交替运行两条路径，计量纯机器端到端时间并排除人工等待。`write_report()` 生成 `runtime-ab.json`、`runtime-ab.md` 与 `runtime-ab.html`；CLI 只有 Runtime 全部 Oracle 通过时才返回 0。时延不作硬门禁，防止不同主机性能造成错误失败。

---

## English

### 1. Scope

This document defines implementation contracts for the DSH and Hermes harness adapters, shared Python bridge/Worker, Domain Effect Runtime, Network/Service L0 Skills, backends, MCP, A2A, and persistence. “Must” denotes a fail-closed requirement.

The access-lab implementation parses typed `LabUser` and `LabApplication`
entities. Admission can touch only the declared endpoint/interface and route;
application access can touch only the declared user's `/32` policy. The HTTP
probe executes an argv-only `wget` from the declared user container to the
declared URL. Grant verification uses fresh state and inverse-tool compensation
must reproduce the typed preflight evidence.

### 2. Module map

| Path | Responsibility |
|---|---|
| `dsh-plugin-netopyu/src/index.js` | DSH composition, tool definitions, approval cards, scoped services |
| `dsh-plugin-netopyu/src/bridge.js` | Unix-socket Worker and process fallback transport |
| `dsh-plugin-netopyu/src/hitl-store.js` | HITL, batches, grants, trajectories, continuations |
| `dsh-plugin-netopyu/src/a2a.js` | Remote subagent provider and continuation tools |
| `hermes-plugin-netopyu/` | Official Hermes manifest and `register(ctx)` entry point |
| `hermes_adapter/plugin.py` | Tool/Skill/command projection and exact-plan approval binding |
| `hermes_adapter/client.py` | Request-bound Unix-socket Worker client |
| `hermes_adapter/pending.py` | Process-local one-shot plan nonce and remote continuation bindings |
| `hermes_adapter/comparison.py` | DSH/Hermes Runtime-invariant A/B gate |
| `dsh_adapter/bridge.py` | Manifest, read, prepare, execute, inspect, audit, workflow API |
| `dsh_adapter/worker.py` | Persistent JSON-lines Unix-socket server and request isolation |
| `dsh_adapter/backend.py` | Backend lifecycle and common paging tools |
| `effect_runtime/` | Domain-neutral façade and Service/Network reconciliation |
| `network_runtime/engine.py` | Shared deterministic effect state machine |
| `network_runtime/contracts.py` | Schema v6 plans, evidence, outcomes, transitions |
| `network_provider/` | Identity-pinned Observer MCP, durable Actor MCP/store, strict results |
| `network_runtime/l0_skills.py` | Versioned L0 and IntentSpec registry |
| `network_runtime/validation.py` | Normalization, schema, provenance, entity, and risk validation |
| `network_runtime/policies.py` | Tool, preflight, verifier, and compensator contracts |
| `network_runtime/journal.py` | State, nonces, hash-chain events, audit |
| `network_lab/manifest.py` | Strict schema-v1 lab target authority |
| `network_lab/containerlab.py` | Shell-free lifecycle, FRR CLI, probes, faults, snapshot restore |
| `network_lab/tools.py` | Pragmatic tool-contract projection for lab nodes |
| `service_layer/` | Strict official-SDK MCP domains and transactional business simulation |
| `integrations/clients/mcp_client.py` | Official stdio/Streamable HTTP client and identity/schema binding |
| `runtime/tool_results.py` | Durable oversized-result storage |
| `profiles/`, `tools/`, `integrations/` | Domain tools and mock/pragmatic adapters |

### 3. Startup and transport

DSH startup resolves configuration, loads Python manifests, projects only the active profile and allowed mutation surface, initializes persistent stores and services, registers tools, and attaches cleanup. A partial mutation surface must never survive startup failure.

Hermes uses the official `plugin.yaml + register(ctx)` surface. It requires a healthy Worker and the public slash-command API before exposing mutations. Read handlers invoke the shared Worker and record reviewed-workflow observations under the same task id. `netopyu_skill_view` and the built-in `skill_view` hook start the matching reviewed workflow. Write handlers prepare only, remove the execution nonce from model-visible JSON, and retain it in process-local `PendingActions`. Only an exact user slash command may atomically claim that binding and call `runtime-execute`. Restart discards all pending bindings safely.

The Worker accepts one JSON object per Unix-socket line. Malformed or unknown requests return structured errors without terminating the Worker. `invoke` is read-only. Mutation requires the plan-bound `execute` command with request-level authorization, nonce, approval identity, and L0 binding.

### 4. L0 and intent contracts

A frozen `L0SkillContract` binds identity/version, one tool contract, intent kind, target fields, allowed profiles, fixed steps, and a canonical contract hash. Every mutation resolves an exact `(skill_id, version, contract_hash)`. A capability may implement multiple semantic contracts; when candidates are ambiguous, Runtime requires an explicit L0 id and never guesses from the tool name.

L0 v2 adds a compatible authoring/compiler layer. `AtomicEffect` defines S1. A constraint-derived S11 can only fix or narrow parameters and strengthen approval. An extension-derived S11 can add inputs, observations, predicates, and non-conflicting desired-state fields. `CompositeEffect` binds exact child versions/hashes in a DAG with checkpoints and reverse compensation. Compilation recursively flattens inheritance and enforces monotonic safety; Runtime receives only immutable artifacts. The multi-version Catalog exposes `explain`, `diff`, and `graph` review surfaces. The bundled REST examples remain an SDK/compiler prototype; the Core-72-qualified v1 contracts stay on the DSH execution path until Provider, resolver/expression, loader, and fault-injection qualification are complete.

`network_runtime.l0.promotion` implements the offline L1 → L0 path. It parses the standard Skill through `skills.skill_format`, binds a versioned Capability Catalog, emits a bounded Agent prompt packet, injects source/catalog hashes into the candidate, cross-checks schemas and roles, compiles strictly, packages immutable review files, and permits one integrity-checked human decision. Every report remains non-executable and non-activated; no Runtime loader side effect exists.

Compilation rejects unknown/missing/type-invalid arguments, invalid entities, untrusted provenance on critical fields, and unsupported profiles. It emits normalized arguments, targets, constraints, desired state, an argument digest, and an immutable `intent_hash`. Free-form model instructions never become backend commands.

### 5. Plan and state machine

Schema-v6 `PreparedPlan` binds the complete effect description, provider identity, input/output schema digests, capability id/version/role, evidence, L0 contract, intent, workflow, TTL, and `plan_hash`. Approval and execution revalidate every binding; any change invalidates prior authorization. Older hash shapes are read compatibility only.

Only `verified_success` and `rollback_verified` represent verified outcomes. Rejection, expiry, changed preconditions, and manual intervention are safe terminal states. Illegal transitions raise `StateTransitionError`.

### 6. Prepare, approval, and execution

Prepare resolves contracts, compiles parameters and intent, assesses risk, obtains fresh preflight evidence, verifies reviewed-workflow constraints, persists the plan, and returns either a plan or clarification/errors.

A DSH allowed-once decision creates a random token whose digest is stored and atomically consumed. The grant binds the operator, request, tool, canonical arguments, plan id/hash, and expiry. Hermes instead binds the same plan to a process-local nonce hidden from the model and consumed only by `/netopyu-approve`. The Runtime independently validates and consumes the execution nonce in both paths.

Execution checks integrity and authorization, revalidates the precondition, sends the write once, and always uses a separate verifier. Verification failure invokes a registered compensator when safe; otherwise the plan escalates to manual intervention.

### 7. Evidence, workflows, and A2A

Evidence is typed, fresh, source-identified, target-bound, predicate-driven, and independently read. Model summaries are never evidence.

Reviewed workflow templates are compiled from canonical L1 Skills and bound by hash. A mutation is allowed only in the correct phase after required observations. Intermediate write output cannot replace final verification.

Remote `input-required` requires fresh exact-plan approval. DSH persists a continuation and uses a card; Hermes retains a structured remote-plan binding in process and exposes a user-only slash command. Hop limits, loop detection, one-shot claims, and timeout/error states fail closed.

### 8. Journal and failures

Each plan has an independent SHA-256 event chain starting at `GENESIS`. Audit recomputes plan integrity, event ordering, previous-hash links, event hashes, and terminal consistency. Audit reports tampering and never repairs it.

Incomplete intent, unknown contracts, missing backends, rejected approval, token mismatch, state drift, uncertain effects, failed postconditions, failed rollback, Worker exceptions, and A2A failures all have explicit non-success states. No error path may be converted to success by model prose.

### 9. Extension rule

A new mutating capability must include metadata, an L0 Skill, a ToolContract, parameter/target policy, preflight, verifier, optional compensator, profile projection, tests, and bilingual documentation. A callable or prompt alone is never an executable mutation capability.

### 10. P0.75-A provider

The schema-v1 manifest accepts only an in-directory topology, unique declared nodes, FRR devices, literal-IP probes, and non-management fault interfaces. Subprocesses use exact argv without a shell. FRR reads are single `show` commands and writes use a reviewed allowlist. `apply_config` captures a session snapshot; `device-config-snapshot-v1` uses `frr-reload.py --reload` and compares a fresh normalized configuration digest. A plan carrying `verification_probe_id` also requires lossless typed traffic evidence. A process crash never replays an uncertain write.

### 11. P0.75-B manifest and verification

Devices may declare an expected BGP-neighbor count; only summary rows with a
numeric `State/PfxRcd` count as Established. Verification waits up to 30 seconds
for OSPF/eBGP convergence before data probes. Users may declare 1–16 normalized
`route_prefixes`, including the legacy application prefix, and admission
reinstalls every reviewed route after an interface transition. The
small-production failover gate proves Core1→Core2→Edge2 forwarding and
subsequent Core1→Edge1 restoration.

Each `LabLink` contains two `LabLinkMember(node, interface, address)` values plus
relationship and path role. Loading rejects duplicate interfaces/addresses,
different subnets, unknown nodes, and any graph whose link set differs from the
Containerlab topology. The address index resolves hop IP to node/interface/link.
`trace_path` accepts exact endpoint IDs and runs the fixed argv
`traceroute -n -m 16 -w 2 -q 1`. Every address and adjacency plus the final
destination must resolve; timeouts, unknown IPs, broken adjacency, or incomplete
traces return `fail_closed=true`. `enforcement_path` fresh-reads endpoint
operstate and the application server's source `/32` route before tracing. These
read-only tools and their Skill are projected to LAN/DC only when typed links
exist.

### 12. P0.75-C fabric contracts and execution

`LabFabric` strictly validates the mode, ASN, route reflectors, VTEPs, unique
VLAN/VNI mappings, access/trunk attachments, tenants, endpoints, interfaces,
and exact topology wiring. Reads parse Linux bridge/VXLAN JSON and FRR BGP EVPN,
VNI, and RIB JSON directly.

`fabric_set_access_vlan` uses fixed argv and a provider-owned execution-session
snapshot. The `fabric-access-vlan` verifier performs a fresh bridge/PVID read
and optional declared probe. `fabric-access-vlan-snapshot-v1` restores and
compares exact stable typed facts. VLAN IDs are limited to 1–4094 and EVPN route
types to 2/3/5. Metadata, ToolContract, L0 Skill, reviewed L1 workflow, verifier,
and compensator must all be present or preparation fails closed.

### 13. P0.8 Service MCP and Effect Runtime contracts

Each Service domain is a separate official-SDK stdio MCP process. Pragmatic
mode supports real stdio or Streamable HTTP only and never an in-process mock.
Initialization discovers paged tools and records server identity/version,
input/output schemas, metadata, and schema digests. Trusted mutations require
configured identity pins, declared contract/result metadata, and structured
content. Plan execution reconnects and rechecks all bindings before sending a
write. Transport success with structured `ok=false` fails semantically.

Service SQLite enables WAL and foreign keys, protects each process connection
with an `RLock`, and serializes cross-process mutations with `BEGIN IMMEDIATE`.
Versioned seed data is installed once. Change authorization, expected revision,
safe idempotency replay, mutation, monotonic revision, and audit are one
transaction. An idempotency result is replayable only while current state still
equals its original after-snapshot; otherwise it returns a conflict.

Internal restore tools are excluded from harness manifests and callable only by
registered compensators. Compensation restores stable business facts from
preflight and then performs an independent read; revision remains monotonic.
The read-only reconciliation tool compares entitlement, two CMDB bindings,
Containerlab enforcement, and a real HTTP probe. It classifies missing mapping,
desired/enforced drift, allowed-but-broken data plane, denied-but-bypassed data
plane, or no drift. Cross-system reads are not an atomic snapshot, so every
approved write still performs execution-time preflight revalidation.

### 14. P0.9/P1.0 Network Provider contracts, evidence, and Actor algorithm

`ProviderCapabilityRegistry` keys stable semantics by unique capability id and
uses tool names only as compatibility indexes. Every entry fixes capability
version, observer/actor role, and action type. External MCP declarations for
the Network domain must match the reviewed registry exactly; unknown tools,
role/action mismatches, or version drift abort backend initialization.

`network_provider.mcp_observer` intersects the selected profile callables with
the observer registry and never registers mutation or restore functions. Every
call returns `network-evidence-envelope-v1` containing correlation id, zoned UTC
observation time, simulation marker, exact provider identity, capability
id/version, content type, canonical payload digest, and payload.

The MCP client validates required fields, semantic envelope success, initialized
server identity/version, discovered capability id/version, timezone-bearing
ISO-8601 syntax with a 300-second age and 30-second future-skew limit, and payload
digest—in that order—before exposing the unwrapped
compatibility content. The original envelope remains available on the call
result. A payload-level `ok=false` is valid negative network evidence and is
distinct from transport/provider failure.

Backend registration lets Observer reads and trusted Actor writes override
same-name local callables. Mutation contracts bind capability id/version/role,
provider kind, server identity/version, input/output schema digests, declared
contract, and structured result. Actor profile declarations are filtered for
the active agent. Runtime-only effect context is removed from model schemas and
injected by `BackendSession`.

`ActorStore` creates an immutable operation in an immediate transaction and
persists argument/preflight/snapshot digests, desired state, target key, and
fence token. A per-target process lock surrounds observation and effect, while
the SQLite lease blocks another live operation. States progress through
prepared/executing/applied, explicit uncertain states, restoring/restored, and
committed/manual terminal states; every change enters an Actor hash chain.

An immutable duplicate may resume only from prepared with an unchanged
snapshot. Executing/indeterminate duplicates reconcile desired or snapshot by
reads and never replay the write; applied duplicates return only while desired
state still holds. Startup uses the same rule. Compensation loads the exact
durable snapshot by operation id. Runtime finalization maps verified success to
committed and verified rollback to restored, then releases the lease. Manual
intervention durably quarantines the target; later operations fail closed until
a reviewed operator-resolution flow clears it.

Containerlab does not natively validate the fence token. The implementation
therefore qualifies single-host crash safety through SQLite and file locks, not
cross-host linearizability. HA requires a remote log/queue, leader fencing, and
device/controller idempotency or CAS.

### 15. P1.1 Capability SPI, read PEP, terminal envelope, and Saga

`CapabilityContract.from_metadata()` normalizes every backend into
observation/effect kind, domain, provider identity/kind, schema digests, effect
semantics, sensitivity, roles, scope fields, and freshness. `BackendSession`
structurally implements `describe_capability`, `invoke_observation`,
`invoke_effect`, and `finalize_effect`; Runtime does not branch on transport.

`invoke_read` compiles arguments and applies `ObservationPolicy` to subject,
roles, clearance, purpose, and each `field:value` scope before any Provider
call. DSH/Hermes pass operator/session purpose, and the manifest projects the
canonical contract.

`ExecutionOutcome.terminal_envelope()` replaces the raw Provider result with a
digest and exposes only Runtime terminal state, typed evidence, error, and
compensation status. Both Harness adapters return it after approval.

`SagaCoordinator` persists immutable definitions, dependency-ordered plan
bindings, terminal outcomes, reverse compensation bindings, and a separate
event hash chain. Every compensation is a fresh L0 plan. Unknown or
uncompensatable state escalates to manual intervention; recovery lists work but
never replays a Provider write.

### 16. Runtime A/B benchmark implementation

`evaluation.runtime_comparison` implements two executors. The reference executor mirrors the DSH manifest's required/type/additional-properties schema and generic approval result before invoking `BackendSession` directly. The guarded executor calls the real `NetworkRuntime.prepare/execute/invoke_read/audit` surface. Each scenario returns a common `PathObservation` evaluated by a machine oracle.

Core-72 contains 72 scenarios: eight valid operations and 64 fault/risk controls across LAN, DC, WAN, and cross-domain Saga behavior. After warm-up, latency trials alternate both paths and exclude human wait. `write_report()` emits JSON, Markdown, and HTML. The CLI exits zero only when Runtime passes every oracle; variable machine latency is reported but is not a hard gate.
