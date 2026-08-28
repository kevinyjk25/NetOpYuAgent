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
| F-04 | 每次 Network 写必须绑定精确的 `(skill_id, version, contract_hash)`；同一底层 Capability 可以承载多个语义 L0 Skill，但不得仅按 tool/capability 名猜测。 |
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
| F-15 | DSH 与 Hermes 必须共享同一 L0 注册表、Domain Effect Runtime、verifier、compensator 和 journal 合同。 |
| F-16 | Hermes 写工具不得向模型暴露 execution nonce；只有用户 slash command 可消费进程内 plan binding。 |
| F-17 | P0.75-A lab 的设备、probe 和 fault target 必须来自版本化 manifest，模型不得构造任意容器或接口。 |
| F-18 | Lab 配置写必须使用 FRR 白名单和无 shell argv；成功必须满足配置证据及显式绑定的流量 probe。 |
| F-19 | 小型现网必须验证 OSPF/eBGP 邻居、正负路径、HTTP、双 ISP 切换/回切，并保持 L1/L0 Runtime 不变量。 |
| F-20 | 拓扑和路径回答必须来自与 Containerlab wiring 精确匹配的 typed graph 和完全解析的 observed traceroute；未知跳点或相邻关系必须失败关闭。 |
| F-21 | Service Layer 必须通过真实 MCP stdio/Streamable HTTP 协议提供身份、应用、策略、变更、CMDB 与平台能力；pragmatic 禁止 in-process mock transport。 |
| F-22 | Service desired state、Network enforcement 和 data plane 必须是独立事实源，并可由只读 reconciliation 分类 drift。 |
| F-23 | 受信 MCP 写必须绑定 server identity/version、declared contract、structured result 与 input/output schema digest。 |
| F-24 | 每次 Service 写必须绑定精确的 Service L0 Skill 合同，并经过与 Network L0 相同的计划、审批、验证、补偿和审计内核。 |
| F-25 | 内部 restore MCP tool 不得投影给模型，只能由注册 compensator 调用。 |
| F-26 | 跨层 L1 必须把 Service 与 Network 写拆为明确计划并在步骤间重读；不得把顺序 workflow 宣称为原子分布式事务。 |
| F-27 | 外部 Network provider 必须声明唯一、版本化 capability id、observer/actor role 与 action type；不得仅按 tool name 获得权限。 |
| F-28 | Network Observer MCP 只能公开只读 capability，且结果必须使用可验证的版本化 evidence envelope。 |
| F-29 | Network evidence 必须在消费前验证 provider identity/version、capability id/version、带时区观测时间、300 秒 freshness/30 秒 future skew 和 canonical payload digest。 |
| F-30 | Network Actor MCP 必须在效果前持久化 immutable operation、approved-preflight digest、desired state 与精确 rollback snapshot。 |
| F-31 | Actor 的 operation/plan/intent hash、approved preflight 与 effect phase 必须由 Runtime 内部注入，不得由模型提供；restore/finalize 不得投影给模型。 |
| F-32 | Actor capability 必须声明 profile，backend 必须按当前 LAN/DC agent 精确投影；计划 schema 必须绑定 capability id/version/role。 |
| F-33 | Runtime 必须通过协议无关 Capability SPI 访问 observation/effect；传输协议不得成为授权语义。 |
| F-34 | Observation 必须在 Provider 调用前验证 authenticated subject、role、resource scope、purpose、clearance 与 sensitivity。 |
| F-35 | Harness/模型只能消费 Runtime terminal envelope，不得把 Actor/Provider 中间态当作执行结果。 |
| F-36 | 跨 Provider Saga 必须绑定不可变步骤定义和每步 plan id/hash；正向和补偿步骤均不得绕过 L0 审批、验证与审计。 |
| F-37 | L0 v2 的约束、扩展和组合必须在编译期展开；Runtime 只接受不可变编译产物。约束不得放宽父合同，组合步骤必须绑定子合同的精确版本和 hash。 |
| F-38 | L1 → L0 Promotion 必须把 Agent 输出视为不可信候选，绑定 L1/Capability/候选 hash，经过确定性校验和一次性人工 review；review 不得自动注册合同或授予执行权限。 |

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
| R-10 | 多 MCP 进程的 change check、revision compare、mutation、idempotency 与 audit 必须位于同一 immediate transaction。 |
| R-11 | Service seed 每个数据库只能执行一次；MCP 重启不得复活已撤销状态。 |
| R-12 | 幂等 replay 仅在当前状态仍等于原 after snapshot 时允许；否则必须返回 conflict。 |
| R-13 | MCP 初始化或身份/schema 漂移必须在写发送前关闭连接并失败，不得泄漏子进程或回退 mock。 |
| R-14 | Observer 的 transport/provider 失败与合法负面网络观测必须保持不同语义，任一方不得伪装另一方。 |
| R-15 | 同一 Actor operation 的 immutable 内容变化必须拒绝；executing/applied 重试不得盲目重发效果。 |
| R-16 | Actor 启动恢复必须只读对比 desired 与 snapshot；无法证明任一状态时必须进入人工介入。 |
| R-17 | 每个 Actor target 必须有跨进程互斥、租约和单调 fencing token；Actor 事件必须可检测断链。 |
| R-18 | Runtime 只有在独立验证或精确补偿后才能内部 finalize Actor operation 并释放 lease。 |
| R-19 | 人工介入 Actor operation 必须持久隔离 target；不得因 lease 超时自动允许新 operation。 |
| R-20 | Saga 必须在重启后列出 planned/running/compensating 操作，但不得自动重放任何 Provider write。 |
| R-21 | Saga 补偿必须按已验证步骤的逆序执行；未知或不可补偿状态必须进入人工介入。 |
| R-22 | Saga 事件必须形成独立可验证哈希链；重复绑定不同 plan hash 必须失败关闭。 |

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
| Runtime → trusted MCP | provider 声明、schema、structured result | identity/version pin、contract/schema hash、fresh verifier |
| Runtime → Network Observer MCP | capability 声明、证据封装、负面观测 | registry 精确匹配、identity/version pin、time/digest 验证、只读 role |
| Runtime → Network Actor MCP | 批准计划、效果重放、补偿上下文 | identity/schema/capability pin、内部 effect context、durable snapshot、lease/fence、Actor hash chain |
| Service MCP → SQLite | 并发业务变更 | WAL、RLock、BEGIN IMMEDIATE、revision、safe idempotency、audit |
| Service ↔ Network | 非原子跨系统状态 | 独立读取、drift 分类、步骤间重校验、新计划恢复 |

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

#### T-13 MCP 伪造、schema 漂移与共享存储竞态

- 风险：恶意/误配置 MCP 把写伪装成读、同名 server 被替换、批准后 schema 改变、两个 MCP 进程同时通过旧 revision、进程重启重放 seed 或陈旧幂等结果覆盖新状态。
- 控制：官方协议 client、受信标记、server identity/version pin、declared contract、structured result、schema digest 纳入 plan hash、执行前重新 discovery；ServiceStore 使用一次性 seed、WAL、进程内锁、跨进程 immediate transaction、revision 和状态敏感 idempotency。
- 剩余风险：本地 SQLite 与 stdio 子进程仍共享 OS account，缺少 mTLS/HSM/远端 WORM/数据库 RBAC；生产 MCP 必须独立部署和认证。

#### T-14 Network provider 越权、自证成功或崩溃丢失补偿

- 风险：observer 把写工具声明成只读；同名工具替换审核语义；伪造 freshness/digest；actor 用自己的写响应自证成功；独立 actor 在写后崩溃并丢失 rollback snapshot。
- 控制：capability registry 精确校验 role/action/version；Observer 不注册写 callable；Runtime 使用 fresh Observer read；Actor 在效果前持久化 exact snapshot，以 operation id 做幂等/补偿，使用 target lock、lease、fence、启动 reconciliation 和独立 Actor hash chain。
- 剩余风险：本地 Observer/Actor 仍共享主机、OS account、Docker daemon 与 Containerlab 真值；设备端不原生校验 fence。生产必须分离凭据/进程/故障域，并增加远端事务日志、HA leader fencing、设备/控制器 CAS 与不可变审计。

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

### 15. P0.75-C Fabric 验收补充

代码门禁必须覆盖 Fabric manifest 的唯一 VLAN/VNI、精确 attachment wiring、access/trunk
模式、`vlan_id`/EVPN route type 边界、固定 argv、未知端口/VLAN 拒绝、审批前后状态漂移、
fresh bridge/PVID 验证、流量失败补偿和 exact typed snapshot 恢复。

真实实验门禁必须证明：10 个容器运行；四台网络节点达到 OSPF/BGP EVPN 邻居期望；
VNI 10010/10020 在两台 VTEP 上存在且各有远端 VTEP；VLAN 10/20 access 与 802.1Q trunk
跨 VTEP 流量成功；租户互访失败；单条 leaf-spine 链路故障期间 L2VPN 继续转发并恢复；
L1+L0 强制后置条件失败最终为 `rollback_verified`、端口状态精确恢复、流量恢复且审计链有效。
验收报告必须同时声明 L3VPN/MPLS/vendor/RF 为未实现。

### 16. P0.8 Service MCP 验收补充

代码门禁必须覆盖 official stdio structured content、identity/schema 绑定、untrusted write
拒绝、批准后 schema drift、显式 `ok=false`、revision conflict、safe idempotency、seed 重启不
复活、验证失败 exact snapshot 补偿、internal restore 隐藏和 cross-layer workflow prerequisites。

实际本地门禁必须证明：Service revoke、Network revoke、Service grant、Network apply 四个
计划均为 `verified_success` 且 audit chain 有效；中间 desired/enforcement 均 false、真实 HTTP
失败；最终 role、enforcement 和 HTTP 语义恢复。该结果只认证本地仿真逻辑，不认证真实企业
MCP、网络设备、审批身份、分布式事务或生产可用性。

### 17. P0.9/P1.0 Network Provider 验收补充

代码门禁必须覆盖 capability 唯一性、observer/actor 分权、observer 不暴露写工具、server identity
错误、capability/digest 错误失败关闭、负面 payload 解包、同名单设备参数规范化，以及 backend
Observer 读与 Actor 写分别走 MCP、内部参数隐藏、profile 精确投影、operation immutable reuse
拒绝、crash-after-effect reconciliation、幂等不重发、精确 durable snapshot 恢复和双事件链。
完整 Python 门禁为 228 个测试和 39 个子测试。

实际本地门禁必须证明 20 节点基线全部通过，并通过 Observer MCP 读取业务/网络 reconciliation
所需事实；随后受审 Actor MCP 计划达到 `verified_success`，故意制造后置状态漂移的计划达到
`rollback_verified`，durable snapshot 与现场均恢复。该门禁认证本地 crash-safety 原型，但不
认证独立凭据、跨主机 fencing、生产遥测流、厂商设备/控制器或 HA。

### 18. P1.1 Capability/Read/Saga 验收补充

代码门禁必须证明：Capability 合同不依赖 MCP/API 名称；未认证、角色不足、clearance 不足和
resource scope 不匹配的 observation 在 Provider 调用前拒绝；DSH/Hermes 写结果只包含 Runtime
terminal envelope，Actor `applied` 不泄漏；Saga 依赖阻止乱序计划、失败后逆序补偿、重启可恢复、
不可补偿步骤进入人工介入且事件链有效。跨层本地用例必须把四个独立 L0 计划绑定到同一 Saga。
这不认证企业 PDP、多主机 Saga leader、分布式原子性或自动 bundle approval。

### 19. Runtime A/B 定量验收

基准必须固定相同工具、参数、Provider 和故障，且明确把 LLM/L1 选择排除在 Runtime 增量之外。DSH-only 参考路径必须保留基础 JSON Schema 和通用 HITL，不能构造为无保护 strawman。机器 Oracle 必须至少覆盖：有效请求、未知参数、领域安全必填、灾难命令、审批后 Provider/状态漂移、越权读取、错误后置条件与补偿、发送后不确定结果、终态信封和审计篡改。

验收要求 Runtime 路径通过全部固定 Oracle；参考路径和 Runtime 的结果均须原样报告。输出必须包含机器可读 JSON、双语 Markdown 和浏览器 HTML，并同时披露 p50/p95 绝对机器时延、样本数、人工等待排除和未测量范围。固定场景 100% 不得表述为生产成功概率。

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
| F-04 | Every Network mutation binds an exact `(skill_id, version, contract_hash)`. One underlying capability may implement multiple semantic L0 Skills, but Runtime never guesses from a tool or capability name alone. |
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
| F-15 | DSH and Hermes must share one L0 registry, Domain Effect Runtime, verifier, compensator, and journal contract. |
| F-16 | Hermes must not expose execution nonces to the model; only a user slash command may consume the process-local binding. |
| F-17 | P0.75-A devices, probes, and fault targets must come from the versioned manifest; the model cannot invent containers or interfaces. |
| F-18 | Lab writes must use shell-free argv and the reviewed FRR allowlist; success requires configuration evidence and any explicitly bound traffic probe. |
| F-19 | The small-production lab must verify OSPF/eBGP, positive and negative paths, HTTP, dual-ISP failover/recovery, and unchanged L1/L0 Runtime invariants. |
| F-20 | Topology and path answers must come from a typed graph that exactly matches Containerlab wiring and a fully resolved observed traceroute; unknown hops or adjacency must fail closed. |
| F-21 | The Service Layer must use real MCP stdio/Streamable HTTP for identity, application, policy, change, CMDB, and platform capabilities; pragmatic mode forbids in-process mock transport. |
| F-22 | Service desired state, Network enforcement, and data-plane evidence must remain independent and support read-only drift reconciliation. |
| F-23 | Trusted MCP writes must bind server identity/version, declared contract, structured result, and input/output schema digests. |
| F-24 | Every Service mutation binds an exact Service L0 contract and uses the same plan/approval/verification/compensation/audit kernel as Network L0. |
| F-25 | Internal restore MCP tools must be hidden from the model and callable only by registered compensators. |
| F-26 | Cross-layer L1 workflows must use explicit plans with reads between effects and may not claim distributed atomicity. |
| F-27 | External Network providers must declare a unique versioned capability id, observer/actor role, and action type; tool name alone grants no authority. |
| F-28 | Network Observer MCP may expose only read capabilities and must return a versioned verifiable evidence envelope. |
| F-29 | Network evidence must be checked for provider identity/version, capability id/version, zoned observation time, a 300-second freshness/30-second future-skew window, and canonical payload digest before use. |
| F-30 | Network Actor MCP must persist immutable operation data, approved-preflight digest, desired state, and the exact rollback snapshot before dispatch. |
| F-31 | Runtime must inject operation/plan/intent hashes, approved preflight, and effect phase; the model cannot supply them, and restore/finalize are hidden. |
| F-32 | Actor capabilities declare profiles, backend projection preserves LAN/DC boundaries, and plan schema binds capability id/version/role. |
| F-33 | Runtime accesses observations/effects through a transport-neutral Capability SPI; transport is not authorization semantics. |
| F-34 | Observation authorization checks authenticated subject, role, resource scope, purpose, clearance, and sensitivity before a Provider call. |
| F-35 | Harness/model consumers receive only a Runtime terminal envelope and cannot treat Actor/Provider intermediate state as an outcome. |
| F-36 | A cross-provider Saga binds an immutable step definition and per-step plan id/hash; forward and compensation steps never bypass L0 approval, verification, or audit. |
| F-37 | L0 v2 constraints, extensions, and compositions are flattened at compile time; Runtime accepts only immutable compiled artifacts. Constraints cannot weaken a parent, and composite steps bind exact child versions and hashes. |
| F-38 | L1 → L0 Promotion treats Agent output as an untrusted candidate, binds L1/Capability/candidate hashes, and requires deterministic checks plus one human review. Review cannot register a contract or grant execution authority. |

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
- **MCP spoofing/schema drift/shared-store races:** trusted flags, pinned server identity/version, declared contracts, structured results, schema/capability digests in schema-v6 plans, execution-time rediscovery, one-time seeding, WAL, process locks, immediate transactions, revisions, and state-sensitive idempotency. Local stdio and SQLite still trust the OS account; production needs authenticated independent services and database controls.
- **Network provider escalation/self-attestation/crash loss:** exact capability role/action/version matching, an observer with no mutations, fresh digest-bearing evidence, and a durable Actor operation/snapshot store with target locks, leases, fences, startup reconciliation, and a hash chain. Observer and Actor still share one host/account/Docker boundary, and devices do not validate the local fence; production needs separated credentials/failure domains, a remote log, controller CAS, HA fencing, and immutable audit.

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

### 11. P0.75-C fabric acceptance supplement

The code gate covers unique VLAN/VNI mappings, exact attachment wiring,
access/trunk semantics, VLAN and EVPN route-type bounds, fixed argv, rejection
of unknown ports/VLANs, approval-time drift, fresh bridge/PVID verification,
traffic-triggered compensation, and exact typed snapshot restoration.

The deployed gate requires all ten containers; expected OSPF and BGP EVPN
adjacencies; VNIs 10010/10020 and remote VTEPs on both leaves; successful
cross-VTEP access and tagged VLAN traffic; failed tenant crossing; surviving
L2VPN traffic during one leaf-spine fault and recovery; and an L1+L0 forced
postcondition failure ending in `rollback_verified` with exact port state,
restored traffic, and a valid audit chain. Reports must explicitly mark L3VPN,
MPLS VPNs, vendor behavior, and RF as unsupported.

### 12. P0.8 Service MCP acceptance supplement

Code gates must cover official stdio structured content, identity/schema
binding, rejection of untrusted writes, post-approval schema drift, explicit
`ok=false`, revision conflict, safe idempotency, no seed resurrection after
restart, exact-snapshot compensation after verification failure, hidden
internal restore tools, and cross-layer workflow prerequisites.

The actual local gate must prove four `verified_success` plans—Service revoke,
Network revoke, Service grant, and Network apply—with valid audit chains. The
intermediate checkpoint must have desired/enforced false and a failed real HTTP
probe; the final role, enforcement, and HTTP semantics must be restored. This
qualifies only the local simulation, not real enterprise MCP services, devices,
approval identity, distributed transactions, or production availability.

### 13. P0.9/P1.0 Network Provider acceptance supplement

Code gates cover unique capabilities, observer/actor separation, absence of
Observer writes, identity mismatch, capability/digest rejection, valid negative
payload unwrapping, single-device argument normalization, and exact backend
routing of Observer reads and Actor writes through separate MCP boundaries,
hidden Runtime context, profile projection, immutable-operation conflicts,
crash reconciliation without blind replay, exact durable restoration, and both
hash chains. The complete gate is 228 tests plus 39 subtests.

The deployed gate requires the complete 20-node baseline and cross-layer facts
read through Observer MCP. Real Actor MCP plans must reach `verified_success`;
a deliberately broken postcondition must reach `rollback_verified` and restore
the durable snapshot and live baseline. This qualifies the local crash-safety
prototype, not separated credentials, cross-host fencing, production telemetry,
vendor devices/controllers, or HA.

### 14. P1.1 capability/read/Saga acceptance supplement

Code gates prove that capability semantics do not depend on MCP/API names;
unauthenticated, under-role, under-clearance, or out-of-scope observations are
denied before Provider invocation; DSH/Hermes mutation output is a Runtime
terminal envelope with no Actor `applied` state; and Saga dependencies prevent
out-of-order plans, compensate in reverse, recover after restart, escalate
uncompensatable work, and maintain a valid event chain. The local cross-layer
case binds four independent L0 plans to one Saga. This does not certify an
enterprise PDP, multi-host Saga leader, distributed atomicity, or automatic
bundle approval.

### 15. Runtime A/B quantitative acceptance

The benchmark fixes the same tool, arguments, Provider, and fault while explicitly excluding LLM/L1 selection from the Runtime increment. The DSH-only reference retains basic JSON Schema and generic HITL and may not be reduced to an unprotected strawman. Machine oracles cover valid requests, unknown fields, domain safety requirements, catastrophic commands, post-approval Provider/state drift, unauthorized reads, failed postconditions and compensation, indeterminate writes, terminal envelopes, and audit tampering.

Runtime must pass every fixed oracle while both paths remain visible in the report. Outputs include machine-readable JSON, bilingual Markdown, and browser HTML, plus absolute p50/p95 machine latency, sample count, the exclusion of human wait, and unmeasured scope. A 100% fixed-scenario result is never a production success probability.
