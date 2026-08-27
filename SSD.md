# NetOpYuAgent 系统规格与安全设计 / System Specification and Security Design

## 中文

### 1. 规格状态

本文件是 P0.5 的系统规格与安全基线。适用范围为本地 mock、DSH 主插件和 Hermes 可选 Adapter。P1 将在真实网络、企业身份和生产审批环境中重新认证这些控制。

### 2. 功能规格

| ID | 必须满足的规格 |
|---|---|
| F-01 | 系统不得实现自建通用 Agent runtime；DSH 为主 Harness，Hermes 只能作为可选 Adapter。 |
| F-02 | 系统必须按 `default/lan/dc/wan` profile 隔离工具与 Skill。 |
| F-03 | 默认只能注册只读工具；写工具必须显式启用。 |
| F-04 | 每个写工具必须绑定唯一、版本化的 Network L0 Skill。 |
| F-05 | 写计划必须包含规范化参数、参数来源、目标、风险、preflight、verifier 和合同 hash。 |
| F-06 | 系统必须在执行前向操作员展示精确计划并获得 allowed-once 决策。 |
| F-07 | 系统必须在写前重校验目标状态。 |
| F-08 | 系统必须使用独立读路径验证 postcondition。 |
| F-09 | 验证失败时必须按合同补偿，或进入人工介入状态。 |
| F-10 | 所有计划状态迁移必须持久化并可审计。 |
| F-11 | 大结果必须外置并通过有界分页读取。 |
| F-12 | A2A 必须支持 peer 发现、能力选择、超时、循环保护和 durable continuation。 |
| F-13 | 记忆召回必须绑定 operator 与 session，且只能显式调用。 |
| F-14 | 离线学习只能生成 proposal，不能自动启用 Skill。 |
| F-15 | DSH 与 Hermes 必须共享同一 L0 注册表、Network Runtime、verifier、compensator 和 journal 合同。 |
| F-16 | Hermes 写工具不得向模型暴露 execution nonce；只有用户 slash command 可消费进程内 plan binding。 |
| F-17 | P0.75-A lab 的设备、probe 和 fault target 必须来自版本化 manifest，模型不得构造任意容器或接口。 |
| F-18 | Lab 配置写必须使用 FRR 白名单和无 shell argv；成功必须满足配置证据及显式绑定的流量 probe。 |
| F-19 | 小型现网必须验证 OSPF/eBGP 邻居、正负路径、HTTP、双 ISP 切换/回切，并保持 L1/L0 Runtime 不变量。 |
| F-20 | 拓扑和路径回答必须来自与 Containerlab wiring 精确匹配的 typed graph 和完全解析的 observed traceroute；未知跳点或相邻关系必须失败关闭。 |

### 3. 可靠性规格

| ID | 必须满足的规格 |
|---|---|
| R-01 | 单个 malformed Worker 请求不得终止 Worker。 |
| R-02 | 未完成授权在重启后不得继续有效。 |
| R-03 | grant 与 execution nonce 必须最多消费一次。 |
| R-04 | 写是否到达不确定时不得重试同一效果，必须先验证。 |
| R-05 | 非法状态迁移必须被拒绝。 |
| R-06 | precondition 变化必须在写前终止计划。 |
| R-07 | A2A timeout、unreachable、loop 和 remote failure 不得报告为完成。 |
| R-08 | 本地可靠性门禁必须覆盖 24 请求、8 并发，p95 小于 1 秒。 |
| R-09 | Runtime event audit 必须检测断链、篡改和终态不一致。 |

### 4. 安全目标

1. **正确授权**：只有操作员批准的精确计划能执行一次。
2. **效果完整性**：模型不能修改、绕过或伪造 L0 合同。
3. **结果可信**：成功来自 fresh independent evidence。
4. **最小暴露**：profile 和部署配置只暴露必需工具。
5. **故障安全**：未知状态不自动重试或乐观成功。
6. **审计完整性**：计划和事件可检测篡改。
7. **数据最小化**：不在不必要的位置保存 prompt、敏感参数或凭据。

### 5. 信任边界

| 边界 | 不可信输入 | 强制控制 |
|---|---|---|
| 用户/模型 → Harness | 自然语言、tool selection、arguments | Tool schema、L0 prepare、clarification |
| DSH → Plugin | 工具生命周期、approval outcome | 精确 request/plan binding、Tool Guard |
| Hermes → Plugin | tool handler、用户 slash command | nonce 不回显、exact hash、进程内一次性领取 |
| Plugin → Worker | JSON 请求、环境配置 | command allowlist、大小/类型校验 |
| Runtime → Backend | 规范化参数 | versioned ToolContract、profile、request authorization |
| Backend → Runtime | 文本/JSON/transport error | typed evidence parser、freshness、predicate |
| 本 Agent → A2A peer | 委派 prompt、metadata | self-contained payload、hop/loop limits |
| A2A peer → 本 Agent | SSE event、interrupt、结果文本 | parser、timeout、continuation approval |
| Runtime → SQLite | 计划、token digest、事件 | transaction、conditional update、hash chain |

### 6. 威胁模型

#### T-01 模型幻觉工具或参数

- 风险：错误工具、缺失字段、猜测目标、虚构成功。
- 控制：strict compiler、实体校验、provenance、L0 binding、独立 verifier。
- 剩余风险：schema 合法但业务意图错误；必须由 L1 追问、reviewed workflow 和操作员审批共同降低。

#### T-02 审批后篡改参数

- 风险：批准 A，执行 B。
- 控制：canonical arguments、plan hash、intent hash、contract hash 和 grant binding。

#### T-03 重放或并发重复执行

- 风险：同一批准触发多次写。
- 控制：token digest、conditional `issued -> consumed`、Runtime nonce、plan state CAS。

#### T-04 TOCTOU 状态漂移

- 风险：preflight 后网络状态变化。
- 控制：执行前重新读取并比较 snapshot；变化后 `precondition_changed`。

#### T-05 写已到达但响应丢失

- 风险：盲目重试造成二次效果。
- 控制：`outcome_indeterminate`、强制 postcondition read、补偿/人工介入，禁止直接重试。

#### T-06 工具或 peer 伪造成功

- 风险：返回文本声称成功但状态未改变。
- 控制：独立 verifier；A2A 中间结果不能替代本域最终验证。

#### T-07 mock 污染真实环境

- 风险：生产误用模拟数据。
- 控制：默认 mock 明确 warning；pragmatic 禁止 mock MCP transport；本地 DC peer 拒绝非 loopback 和 pragmatic。

#### T-08 凭据泄漏

- 风险：凭据进入 Git、prompt、计划或日志。
- 控制：environment/secret manager 注入、auth 配置校验、trajectory 最小化、日志脱敏。

#### T-09 A2A 循环、放大或会话泄漏

- 风险：无限委派、跨域泄漏父会话。
- 控制：最大 hop、chain loop detection、自包含 prompt、不继承父历史、大小/timeout 限制。

#### T-10 审计篡改

- 风险：删除或修改失败事件。
- 控制：每计划事件哈希链、plan hash、终态一致性 audit。P1 需要外部 append-only/WORM 副本以覆盖数据库管理员威胁。

#### T-11 Hermes 把模型行为误当成人工授权

- 风险：模型生成“批准”文本、重复工具调用、调用通用危险命令审批，或 Adapter 重启后恢复旧授权。
- 控制：写 handler 永远只 prepare；模型可见 JSON 删除 nonce；只有 Hermes slash command dispatcher 可调用 approve handler；命令必须包含完整 plan id/hash；绑定进程内一次性领取；重启丢弃。
- P0.5 假设：本地 Hermes 用户、插件进程、其他已安装插件和 OS account 可信，Gateway 已配置用户 allowlist。
- 剩余风险：当前 Hermes slash command handler 不提供可验证的发送者身份，配置的 operator id 不等同于不可抵赖身份；Hermes 插件在进程内运行也不是安全沙箱。P1 必须增加企业身份上下文、独立审批服务、Worker 服务身份/进程隔离，并限制模型终端与代码执行面。

#### T-12 Lab provider 命令注入或目标扩张

- 风险：模型通过 CLI 参数执行 shell、修改管理口、访问未声明容器或把故障注入扩展到宿主机。
- 控制：严格 manifest、路径归属检查、标识符/IP/interface 校验、无 shell argv、read/config 白名单、`eth0` 禁止、预声明 probe/fault、lab 与真实 inventory 互斥、显式本地 lab 授权。
- 剩余风险：Containerlab/Docker 具有高权限，宿主 Docker daemon 与 manifest 属于可信运维边界；本 provider 不是多租户安全沙箱。

### 7. 审批规格

审批摘要至少包含：

- plan id/hash、intent hash；
- L0 Skill id/version/contract hash；
- tool 与 action type；
- 完整规范化 arguments 和 targets；
- 参数来源；
- risk level/reasons；
- preflight 摘要；
- verification/rollback contract；
- workflow binding；
- expiry。

批准只对该摘要对应 hash 有效。DSH 使用 `allowed-once` 卡片和 Tool Guard；Hermes 使用用户输入的精确 slash command，nonce 只留在插件进程。批量、异步恢复和 A2A continuation 也必须获得新的 Harness 审批，不能复用环境变量、自然语言“已批准”或历史会话状态。

### 8. 模型安全策略

模型不是信任根。每个模型必须按用途单独认证：

| 等级 | 允许用途 | 资格要求 |
|---|---|---|
| M0 | 文本总结、只读分类 | 不得触发写工具 |
| M1 | 只读诊断、候选 Skill/intent | tool-call 协议和参数提取评测通过 |
| M2 | 审批前写计划候选 | reviewed workflow、重复调用、停止条件和结果解释评测通过 |
| M3 | 生产变更候选 | M2 + 真实环境 shadow/canary + 人工审批 + SLO |

当前 `qwen3.5:27b` 仅作为本地 P0.5 默认；`qwen2.5:7b` 未通过 M2，不允许自主写流程。

### 9. 数据安全

- 密钥、密码、bearer token 和 API key 不得进入仓库。
- Tool Guard 只持久化执行 token digest。
- Hermes Adapter 不持久化 execution nonce，也不把它返回模型；进程重启使 pending plan 不可执行。
- trajectory 不保存 prompt、参数值或工具结果正文。
- A2A 只发送任务必需的自包含 prompt 和有限 provenance。
- 大结果 TTL 默认 24 小时；P1 应按数据分类配置 TTL、加密和删除策略。
- SQLite 文件权限、volume 加密、备份和 WORM audit 是部署责任；P0.5 尚未实现集中密钥管理。

### 10. 安全失败策略

| 场景 | 必须的结果 |
|---|---|
| 无法识别意图/目标 | 追问；不 prepare |
| 缺少 backend/credential | 拒绝；不回退 mock |
| approval timeout/reject | 终止计划 |
| hash/grant/nonce mismatch | 安全拒绝并记录 |
| verifier 读失败 | 不得成功；补偿或人工介入 |
| compensator 失败 | 人工介入并保留全部 evidence |
| audit 失败 | 标记完整性故障；禁止把计划当作可信成功证据 |

### 11. 验收标准

P0.5 必须同时通过：

1. Python 全套测试和子测试；
2. Harness-boundary architecture audit；
3. Node/plugin syntax 与 HITL/A2A smoke；
4. profile Skill projection；
5. retrieval Recall@3 ≥ 0.95、MRR ≥ 0.90；
6. Worker load、malformed isolation 和 restart recovery；
7. ambient destructive env 不能绕过请求级 gate；
8. plan hash、nonce、state drift、verifier、compensator 和 event audit 单测；
9. 本地 UI 中至少一次 L1 + L0 + approval + verification 实测；
10. 小模型资格失败时 Runtime 必须拦截未批准的重复效果。
11. Hermes PluginContext、slash approval、nonce hiding、restart loss、A2A continuation 和 DSH/Hermes invariant comparison。

权威命令：

```bash
scripts/netopyu-dsh retirement
```

### 12. P1 安全缺口

- 企业 SSO/RBAC 与审批人身份不可抵赖；
- Hermes Gateway sender identity 到 plan approval actor 的强绑定；
- mTLS、证书轮换、egress allowlist；
- 集中密钥、静态加密和数据库访问控制；
- WORM/远端审计副本；
- 真实设备 adapter 的命令 allowlist 和厂商差异认证；
- 变更窗口、工单、双人复核和紧急变更策略；
- HA/DR、备份恢复演练与长期 chaos；
- 生产模型/Skill 的版本发布、canary 和回退机制。

### 13. P0.75-A 验收补充

代码门禁必须验证 manifest 逃逸、shell/多行命令、未知设备、灾难性配置、exact snapshot 补偿、双重后置条件和 workflow 前置证据。实际实验门禁必须验证全部节点运行、两个 OSPF Full 邻居、双向 probe、主路径选择、主链路中断后的备路径以及主链路恢复。缺少 Docker/Containerlab 时必须返回结构化失败，不能回退 mock。

园区/IDC access 验收还必须证明：LAN/DC profile 不交叉暴露写工具；未知 user/app/role
fail closed；控制用户基线成功、目标用户基线失败；LAN 与 DC L0 分别持有独立审批和有效
hash-chain audit；最终 HTTP probe 成功；测试清理后双层拒绝状态恢复。

### 14. P0.75-B 验收补充

完整小型现网的强制门禁为：20 节点运行；所有清单 OSPF/BGP 邻居达到期望；11 条
ICMP 允许/拒绝结果匹配；Bob→CRM、Carol→Wiki、Guest→Portal 成功且 Guest/Erin→CRM
失败；主 ISP 故障后数据面经 Core2/Edge2 可用并恢复主路径；Erin L1 + LAN/DC L0 成功；
强制 HTTP 后置条件失败后为 `rollback_verified`。FRR 安全角色模拟的剩余风险必须在文档
显式披露。

拓扑/路径附加门禁为：26 条 typed links 与 topology wiring 集合相等；52 个接口地址唯一；
endpoint 不进入 device API；已知路径的每个 hop 能解析到 node/interface/link 且 adjacency
为真；注入未知 hop 时返回 `fail_closed=true`；执行点输出明确否认真实 RADIUS/802.1X、
leaf ACL/IAM 和 stateful firewall。

---

## English

### 1. Status

This document is the P0.5 system and security baseline for local mock, the primary DSH plugin, and the optional Hermes adapter. P1 must recertify these controls against real networks, enterprise identity, and production approval systems.

### 2. Functional requirements

| ID | Requirement |
|---|---|
| F-01 | The project must not implement a custom general agent runtime; DSH is primary and Hermes is optional. |
| F-02 | Tools and Skills must be isolated by profile. |
| F-03 | Only read-only tools are registered by default. |
| F-04 | Every mutation must bind to one versioned Network L0 Skill. |
| F-05 | A mutation plan must bind normalized arguments, provenance, targets, risk, preflight, verifier, and contract hashes. |
| F-06 | The exact plan must receive an allowed-once operator decision. |
| F-07 | Target state must be revalidated immediately before the write. |
| F-08 | Success must be established through an independent read path. |
| F-09 | Failed verification must compensate contractually or require manual intervention. |
| F-10 | Every state transition must be durable and auditable. |
| F-11 | Oversized results must use durable bounded paging. |
| F-12 | A2A must provide discovery, selection, timeout/loop protection, and durable continuations. |
| F-13 | Memory recall must be operator/session scoped and explicit. |
| F-14 | Offline learning may create proposals but may not activate Skills. |
| F-15 | DSH and Hermes must share one L0 registry, Network Runtime, verifier, compensator, and journal contract. |
| F-16 | Hermes must not expose execution nonces to the model; only a user slash command may consume the process-local binding. |
| F-17 | P0.75-A devices, probes, and fault targets must come from the versioned manifest; the model cannot invent containers or interfaces. |
| F-18 | Lab writes must use shell-free argv and the reviewed FRR allowlist; success requires configuration evidence and any explicitly bound traffic probe. |
| F-19 | The small-production lab must verify OSPF/eBGP, positive and negative paths, HTTP, dual-ISP failover/recovery, and unchanged L1/L0 Runtime invariants. |
| F-20 | Topology and path answers must come from a typed graph that exactly matches Containerlab wiring and a fully resolved observed traceroute; unknown hops or adjacency must fail closed. |

### 3. Security objectives

The system must provide exact authorization, effect integrity, evidence-based outcomes, least capability exposure, fail-closed uncertainty, tamper detection, and data minimization.

### 4. Threat controls

- **Hallucinated tools/arguments:** strict compilation, entity/provenance checks, L0 binding, independent verification.
- **Post-approval tampering:** canonical arguments and plan/intent/contract hashes.
- **Replay/concurrent duplication:** token digests, atomic state changes, one-shot Runtime nonces.
- **TOCTOU drift:** execution-time precondition comparison.
- **Lost write response:** indeterminate state followed by verification, never blind retry.
- **Forged success:** independent typed postconditions.
- **Mock contamination:** explicit modes, no pragmatic-to-mock fallback, loopback-only demo peer.
- **Credential leakage:** external secrets, minimized trajectories, redacted logs.
- **A2A loops/history leakage:** hop/chain controls and self-contained delegation.
- **Audit tampering:** per-plan event hash chains; P1 still requires an external append-only copy.
- **Hermes model-as-approver confusion:** prepare-only write handlers, nonce removal, exact user slash commands, process-local one-shot bindings, and safe loss on restart. P0.5 still trusts the local account and Hermes gateway allowlist; production requires authenticated sender identity and process isolation.
- **Lab command/target expansion:** strict manifests, path and identifier validation, shell-free argv, FRR read/write allowlists, management-interface exclusion, predeclared probes/faults, and process isolation from real inventory. Docker remains a trusted privileged boundary, not a multi-tenant sandbox.

### 5. Approval and model policy

Approval must display and bind the exact plan, intent, L0 contract, arguments, targets, provenance, risk, evidence, verifier/rollback, workflow, and expiry. DSH uses an allowed-once card and Tool Guard. Hermes uses an exact user slash command while retaining the nonce only in process. Batch, recovery, and A2A continuation paths need fresh harness approval.

Models are not a trust root. They are qualified by use level: summary/classification, read-only candidate generation, mutation-plan candidate generation, and production mutation candidate generation. `qwen2.5:7b` failed the mutation-plan level and is not authorized for autonomous writes. `qwen3.5:27b` remains a local P0.5 default, not a production certification.

### 6. Data security

Secrets must not enter Git, plans, prompts, or trajectories. DSH persists only token digests; Hermes keeps the pending execution nonce in process and never returns it to the model. A2A sends bounded task data. Large results expire by default. Encryption, file permissions, backups, retention, and WORM audit are deployment controls that must be completed in P1.

### 7. Acceptance

P0.5 acceptance requires the complete Python and subtest suite, harness-boundary audit, Node/HITL/A2A smoke, Skill projection, retrieval thresholds, Worker load and recovery, destructive-gate tests, Runtime integrity/verification/compensation tests, a local UI L1+L0 exercise, and Hermes PluginContext/slash/nonces/restart/A2A/parity tests proving that unapproved duplicate effects are blocked.

The authoritative command is:

```bash
scripts/netopyu-dsh retirement
```

### 8. P1 gaps

P1 must add enterprise identity and non-repudiation—including a strong Hermes gateway sender-to-approval-actor binding—mTLS and egress controls, centralized secrets and encryption, external immutable audit, real-adapter command certification, change-window/ticket/two-person policy, HA/DR and long-duration chaos, and controlled model/Skill release with canary and rollback. Hermes plugins are in-process, not a sandbox; production must also isolate the Worker and restrict model terminal/code execution authority.

### 9. P0.75-A acceptance supplement

The code gate covers manifest traversal, shell/multiline injection, unknown targets, catastrophic commands, exact snapshot compensation, dual postconditions, and reviewed workflow prerequisites. The deployed-lab gate requires all nodes, two Full OSPF adjacencies, bidirectional probes, primary-route selection, backup forwarding after a primary-link fault, and primary recovery. Missing Docker/Containerlab is a structured failure and never triggers mock fallback.

Campus/IDC access acceptance additionally proves LAN/DC write-tool isolation,
fail-closed unknown users/apps/roles, a passing control-user baseline, a denied
target-user baseline, separately approved and audited LAN/DC L0 plans, a real
successful HTTP probe, and restoration of both denial layers after cleanup.

### 10. P0.75-B acceptance supplement

The complete-lab gate requires all 20 nodes, every declared OSPF/BGP adjacency,
eleven matching positive/negative ICMP paths, reviewed HTTP allow/deny outcomes,
Core2/Edge2 ISP failover and primary recovery, successful Erin L1 plus LAN/DC
L0 execution, and `rollback_verified` after a forced HTTP postcondition failure.
Documentation must retain the limitation that FRR security roles are not
stateful-firewall or vendor emulation.

The topology/path gate additionally proves exact equality for 26 typed links,
52 unique interface addresses, endpoint/device API separation, complete
node/interface/link and adjacency resolution for a known trace, fail-closed
behavior for an injected unknown hop, and truthful enforcement labels that deny
real RADIUS/802.1X, leaf ACL/IAM, and stateful-firewall semantics.
