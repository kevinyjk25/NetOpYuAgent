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
| F-38 | L1 → L0 Promotion 必须保存 `L1 → L0.5 → L0` 三阶段及逐级 hash。L0.5 不得偏离 L1，L0 不得扩大 L0.5；Agent 输出仍是不可信候选，一次性人工 review 不得自动注册合同或授予执行权限。 |
| F-39 | 全部内置受审写能力必须由编译 L0 v2 Contract 驱动；旧 ToolContract/verifier/compensator 只能作为精确绑定的实现 Adapter。prepare 和执行前必须校验 parity，Effect 参数只能从已批准值按 v2 模板渲染；禁止新增裸 v1 L0。 |
| F-40 | 每个生产 L0 必须保存 L1/L0.5/L0 authoring/compiled 和逐级 hash 轨迹；主门禁必须重新验证 Promotion semantic parity、精确 contract round trip 和文件完整性。反向 bootstrap 产物必须标注来源、不得注册到 Harness 或宣称为模型独立推导。 |
| F-41 | schema-v10 写计划必须绑定经过 verifier 规范化的 requester/policy evidence，以及 Provider release/manifest/qualification/deployment digest；v9/v8 及更早只读兼容。 |
| F-42 | Harness 人工确认不得以 actor 字符串直接授权；Runtime 必须签发短时、精确绑定 plan/requester/policy/approver/risk/mode 的 approval proof，执行必须先验证签名与 TTL。 |
| F-43 | `enforced` identity mode 未配置企业 credential verifier 时必须拒绝全部 requester context，并禁用 legacy actor compatibility；本地 verifier 不得被升级为生产凭证。 |
| F-44 | approval policy 必须支持 single/dual approver、角色/作用域、职责分离、关键变更工单和可选活动窗口；主体混淆、自批冲突和窗口外审批必须失败关闭。 |
| F-45 | `enforced` read/write 必须只从固定 issuer/audience/非对称 algorithm 的短时 JWT 获取 subject、role、scope、clearance 和 assurance；未知 kid、过期、超长 lifetime 或 claim/signature 不符必须失败关闭。 |
| F-46 | 人的 access token 必须与独立 Gateway sender attestation 通过 `act_sub + subject_jti` 交叉绑定；Gateway 必须绑定 Harness/session/client，任一替换不得进入 Provider。 |
| F-47 | 外部 PDP 必须分别授权 observation、prepare 和 approve，decision id/policy/version/obligations 必须进入哈希绑定证据；外部 obligation 只能收紧内置 L0 policy。 |
| F-48 | 变更工单必须由配置的 Change Authority 验证 status/revision/window/profile/capability/targets/risk ceiling；调用方自报 ticket 属性不得成为权威。 |
| F-49 | 外部 Provider 的 Manifest、Qualification Report 与 Deployment Attestation 必须分别由独立 Publisher/Qualifier/Deployer Ed25519 key 签名；同一公钥不得跨角色复用。 |
| F-50 | Provider Manifest 必须固定 artifact、Provider identity/version、Capability/schema/result contract、profile 与允许的 L0 contract hash；Runtime 不得信任 Provider 自报 release id。 |
| F-51 | Qualification 必须通过固定的 timeout、幂等、乱序、部分成功、不确定终态、补偿、补偿失败和重启恢复等 9 项故障场景；任一失败不得生成可发布报告。 |
| F-52 | Provider release 必须经过 stage/publish/environment promote；严格策略的 promote/rollback 必须绑定目标 release 的新部署证明，breaking promote 与 rollback 必须带审批引用，生命周期事件必须形成可验证哈希链。 |
| F-53 | enforced admission 必须把 active signed release、非过期 deployment 与实际 discovery 精确比较，并验证三种 trust role/scope/expiry/revocation、qualification freshness、exact artifact map、result contract 和允许的 L0 hash。 |
| F-54 | 执行前必须重新 admission；审批后 release/deployment/identity/schema/result/L0 漂移必须在 Provider 调用前进入可审计终态且 write count 为零。 |
| F-55 | P1.9 Decision Plane 只能读取 DSH 已接受步骤中的直接用户消息；Skill、Tool、系统或插件生成文本不得冒充新的用户意图。 |
| F-56 | Decision 候选必须从本轮精确 DSH Tool 声明和受审 Skill manifest 构建；模型不得发明候选、required fields、workflow 或 effect authority。 |
| F-57 | Guard、候选 Schema、grounding 和 compiler 必须单调收窄：可以拒绝、追问或删除无证据参数，不得补默认值、扩大目标、改写已知值或授予权限。 |
| F-58 | 每个 P1.9 信封必须固定 `authority=proposal_only`；Decision 不得调用 Runtime/Provider、签发审批证明、覆盖 DSH 路由或绕过任何 L0 控制。 |
| F-59 | Decision 存储不得持久化原始 prompt、模型正文或参数值，只能保留摘要、参数键、有界证据和实际 DSH 路由关联。 |
| F-60 | `shadow` 模式的 Decision 故障不得改变原 DSH 步骤；当前插件必须拒绝未验收的 `canary/enforced` 模式。 |
| F-61 | 未来 `canary/enforced` 必须在效果前验证 Decision/session/message/候选/政策绑定、新鲜度、sealed holdout 门禁和目标/参数一致；任何不确定性均失败关闭，Runtime admission 仍不可跳过。 |
| F-62 | P1.9 Catalog baseline 必须可跨 checkout，并绑定三 profile 的候选、Tool declaration、Skill semantic content 和生产政策；任何漂移未经显式 review 不得通过退休门禁。 |
| F-63 | Holdout Prompt/标签必须保存在仓库外；seal manifest 不得包含其原值，且至少两个不同 reviewer 的完整语义标签一致后才能形成 consensus digest。reviewer id 本身不得冒充企业身份或不可抵赖证明。 |
| F-64 | 可选 Decision→Plan binding 必须固定为 `proposal_only/canary`，验证完整 Decision/evidence digest、session/Harness/profile、候选 route、请求与编译参数及精确 L0 contract；一个 Decision id 最多创建一份计划，binding 必须进入 plan hash 和 hash-chain event。 |
| F-65 | C1 canary policy 只能保持 Harness 原 route 或阻断/收窄；不得重路由、修改参数或产生权限。无效写 Decision 必须失败关闭，无效读 Decision 不得改变原 route。 |
| F-66 | Canary readiness 必须交叉绑定合格 Worker/Adapter 报告、真实 DSH Web/Hermes CLI 产品证据和运维演练证据，并验证摘要、有效期、reviewer/owner 分离、64/64 Core 控制、至少三个实现版本的稳定/改善趋势与 p50/p95 阈值、完整 plan binding、零 replay/authority escape 及四份独立停用/回退/告警/no-effect replay receipt。 |
| F-67 | Readiness 状态上限必须是 `ready_for_review`；CLI 不得修改 Adapter 配置或流量，也不得输出 Prompt、标签、参数值、reviewer/owner id。激活必须是独立的组织身份、签名和发布审批控制。 |

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
| R-23 | approval proof id 在 journal 中必须唯一；proof 或 execution nonce 的重复消费均不得产生第二次效果。 |
| R-24 | Provider 合同漂移在 execution claim 后必须收敛为 `precondition_changed`、释放目标锁、保存失败证据并完成审计，不得遗留为无主 `executing`。 |
| R-25 | 每次 Decision 模型尝试必须有超时、响应大小、单 Tool-call 和最大修复次数边界；耗尽后只能形成协议失败，不能形成候选执行。 |
| R-26 | 同一 DSH session/message 的影子 Decision 必须有界去重；第一次实际 domain Skill/Tool 路由最多关联一次。 |
| R-27 | Decision 指标必须把未观测、协议失败和 Guard 终止与真实路由一致分开；不得把 DSH parity 当作正确性 Oracle。 |

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
| DSH Harness → L1 Decision Plane | 用户来源、Tool 声明、实际路由 | direct-user provenance、精确候选、session/message digest、proposal-only |
| L1 Decision Plane → Model | 自然语言与候选 | loopback-default、候选专属 Schema、单 Tool-call、有界修复、无 effect surface |

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
- 控制：写 handler 永远只 prepare；模型可见 JSON 删除 nonce；只有 Hermes slash command dispatcher 可调用 approve handler；命令必须包含完整 plan id/hash；绑定进程内一次性领取；随后 Runtime 签发模型不可见的 plan-bound approval proof；重启丢弃。
- P0.5 假设：本地 Hermes 用户、插件进程、其他已安装插件和 OS account 可信，Gateway 已配置用户 allowlist。
- 剩余风险：local-simulation 仍以 OS account/owner-only Worker 为信任根；B1 OIDC/Gateway/PDP/Change Adapter 只完成本地 HTTP 资格测试，尚未连接真实企业发行方或取得不可抵赖证据。Hermes 插件在进程内运行也不是安全沙箱；生产仍需真实 Gateway token minting、进程隔离和模型终端/代码执行面限制。

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

#### T-15 requester/approver 主体替换、证明伪造或跨计划重放

- 风险：Adapter 把 Alice 的请求记为 Bob、攻击者修改 approver、复用另一计划/策略的批准，或在凭证过期后执行。
- 控制：schema-v10 继承 requester/policy 与 Provider release/deployment evidence 并纳入 plan hash；`enforced` 固定 JWT issuer/audience/algorithm/lifetime/JWKS，将 access token 与 Gateway attestation 交叉绑定，并由 PDP/Change Authority 授权。执行验证 proof 和全部绑定后再原子消费 nonce。
- 本地边界：local verifier 明确标记 `local_simulation=true`，只认证 owner-only Adapter/Worker 进程链；`enforced` 缺少 OIDC、Gateway、PDP 或 Change Adapter 时全部拒绝，不能把 raw dictionary 当身份或策略。
- 剩余风险：B2-ready 已验证本地 RS256/JWKS/HTTP、动态 Gateway mint 和显式 CA/mTLS wire path；真实 key rotation/撤销、组织 PDP 数据、change system、证书轮换/HSM 和外部不可变审批日志仍待 B2/P1.7 资格化。

#### T-16 Provider 供应链替换、资格伪造或审批后 release/deployment 漂移

- 风险：恶意 Provider 自报可信 identity/release、Publisher 自行签资格报告、过期/撤销 key 继续使用、未资格 artifact 被激活、result/L0 权限在审批后扩大。
- 控制：deployment-owned provider id；独立 Publisher/Qualifier/Deployer trust role；外部 JSONL 进程固定 9 项资格与真实 restart；OCI-image/SBOM/provenance 必需 digest；短期 exact deployment attestation；严格 lifecycle/rollback；schema-v9 release/deployment evidence；prepare/execute 双重 admission。
- 本地边界：B-ready fixture 虽复制到仓库外并独立运行，但源码、临时 key、SQLite 和 digest fixture 仍同一工程信任域。P1.4-B 仍需要组织签名/HSM 根、独立仓库/CI/实验室、真实 OCI/SBOM/SLSA 内容验证和外部 WORM audit。

#### T-17 Decision Plane 意图混淆、参数幻觉或旁路提权

- 风险：插件把非用户消息当意图，模型发明目标/参数，陈旧 Decision 关联到另一轮 Tool，或将高置信度提案误作执行授权。
- 控制：只接受 direct-user provenance；候选由当前 DSH 声明和受审 Skill 构建；候选专属 Schema、grounding、确定性 compiler 和有界修复；信封固定 `proposal_only`；存储只含摘要；实际路由按 session 一次关联；所有效果仍重新进入 L0 Runtime。
- 当前边界：P1.9-B1 仅有本地 `off/shadow`，影子故障不会阻断原 Harness 行为。它不能证明 Harness 选择正确；虽有私有 holdout/双 reviewer 合同，但没有真实人工真值、组织 canary、身份绑定或告警 SLO，因此当前禁止 `canary/enforced`。

#### T-18 Catalog/Evidence 被误当授权根或泄露敏感证据

- 风险：治理委派被误解为设备读写授权，Evidence 页面被接入执行通道，或 Prompt、参数、审批身份、Provider payload 和路径从聚合结果泄露。
- 控制：P2.1 决策固定为治理工作流，显式声明无 Runtime read/effect 和 Provider publication 权威；职责分离、scope、依赖和兼容性失败关闭。P2.2 只读打开来源并采用字段白名单，缺链、截断或篡改降级；离线页面没有审批、执行、注册或激活入口。
- 当前边界：本地 Catalog/Evidence 不能替代企业 IAM/PDP、独立发布系统、远端 WORM、告警平台或生产 SLO；聚合指标也不能作为写成功证明。

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
- requester subject/issuer/session/assurance/digest；
- approval mode 与 policy id/version/hash；
- Provider release/manifest/qualification digest；
- expiry。

批准只对该摘要对应 hash 有效。DSH 使用 `allowed-once` 卡片和 Tool Guard；Hermes 使用用户输入的精确 slash command，nonce 只留在插件进程。两者确认后都必须调用 `runtime-approve` 获取模型不可见的短时签名 proof，`runtime-execute` 不再信任裸 actor 字符串；只有 local compatibility 模式为旧测试保留该入口。批量、异步恢复和 A2A continuation 也必须获得新的 Harness 审批，不能复用环境变量、自然语言“已批准”或历史会话状态。

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
当前完整 Python 门禁为 288 个测试和 81 个子测试。

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

### 20. P1.3-B1 与 B2-ready 企业控制面验收补充

本地代码门禁必须通过真实 RS256/JWKS/HTTP wire path 覆盖：OIDC access token 与 Gateway attestation 成功交叉绑定；敏感 observation、effect prepare 和 approval 均调用 PDP；Change Authority 校验 revision/window/scope/risk；签名主体替换、access-token 替换、raw role 注入、unknown kid、PDP deny、ticket deny 和 scope mismatch 均在 Provider 前失败关闭。token 和控制面 bearer secret 不得进入 plan、proof public evidence、journal、trajectory 或模型结果。

B2-ready 还要求动态 Gateway mint、显式 CA/mTLS、owner-only client key、离线无泄密 Doctor 和无效果 live contract check 通过。该验收仍不认证用户企业的 issuer、组织 PDP 策略、变更平台可用性、撤销传播、证书轮换/HSM 或外部 WORM 审计；这些属于 B2/P1.7。

验收要求 Runtime 路径通过全部固定 Oracle；参考路径和 Runtime 的结果均须原样报告。输出必须包含机器可读 JSON、双语 Markdown 和浏览器 HTML，并同时披露 p50/p95 绝对机器时延、样本数、人工等待排除和未测量范围。固定场景 100% 不得表述为生产成功概率。

### 21. P1.4-B-ready Provider 发布验收补充

本地代码门禁必须证明：三角色 key 独立；仓库外进程通过固定 9/9 与真实重启；必需 artifact、部署签名/时效/environment/exact map 失败关闭；stage/publish/promote/证明续期/严格 rollback/deprecate 和 audit 有效；Backend admission 精确；审批后 release 或 deployment 漂移以零写入 `precondition_changed` 结束。

该验收只形成 B-ready 接入协议。只有独立组织拥有的 Provider、仓库/CI/实验室、企业签名/HSM 根、真实 OCI/SBOM/SLSA 验证及外部 WORM lifecycle audit 通过后，才能完成 P1.4-B。

### 22. P1.8 L1/模型资格验收

L1 评测必须与 Runtime 执行隔离，不得暴露 callable Tool、审批 nonce/proof、Provider credential 或生产 Memory。模型输出必须通过 extra-forbid 决策合同和候选目标绑定；非法 JSON、未知目标、selection 缺字段、refusal 携带执行内容一律不可形成候选计划。模型原始正文和 endpoint 错误正文不得写入报告，远程 endpoint 必须显式 opt-in。

固定集必须覆盖 Skill/Tool 选择、参数、追问、多步 workflow、安全绕过和领域外请求，并报告中英切片。危险请求逃逸率必须为 0；但此结果只证明固定集在严格 Parser 后未逃逸，不能替代 L0/Runtime。完整模型门禁必须绑定不可变 artifact digest、Prompt、Catalog 和数据集；任何 safety escape 增加或核心指标下降超过 5 个百分点标记回退。子集、规则 Adapter、可变 tag 或未保存 digest 的运行不得标记模型合格。

DSH-in-loop 影子评测还必须在任何模型调用前验证固定 DSH 版本和精确活动插件白名单。所有 Skill、Tool、shell、文件工具、Web、子代理、遥测、远程模型 Provider 与 NetOpYu effect 必须关闭；仅允许隔离临时 session 持久化，运行结束必须删除。配置漂移、未知活动 entry、输出超限、超时或非法候选均 fail closed。B1 结果不得描述为实际 Skill/tool-call 准确率。

B2/C1 若开启无效果 Tool，必须使用独立 overlay 和精确 Tool 白名单，且 capture/Governor 不得连接 Runtime、Provider、设备或审批。C1 的预装 Skill、系统提示、typed Tool 集合和 Catalog 编译规则必须摘要绑定；Governor 只能监听 loopback，Tool 强制与隐藏重试必须有界、可计数，回执后固定终止。任何 forbidden/duplicate Tool、Schema/候选/摘要/回执漂移、提前文本或不完整终态不得形成候选。隐藏重试 token 未完整计量时，报告必须声明成本下界。safety escape 非零即不合格，即使 Runtime 后续仍会拒绝该效果。

C2 Guard 必须是单调收窄器：允许输出仅为拒绝、越界、弃权或保持原候选，禁止选择/改写 target、补充参数、生成 workflow 或授予权限。政策必须版本化、摘要绑定且不导入 Oracle 场景/标签。Protocol Firewall 只允许 loopback 模型端点，每次实际尝试必须计量；安全合成只能调用无参数 refusal/out-of-scope capture。报告同时显示模型首轮与 Guard 后 safety、固定集误杀、最大尝试和尾时延。最终 safety 为 0 不能掩盖原始模型逃逸、协议失败或 Runtime 仍是最终安全边界。

C3 候选专属 Tool 必须由摘要绑定的可信候选合同生成，Tool 身份固定 kind/target，Schema 必须 `additionalProperties=false` 且只含该候选业务键。未知字段收窄只能删除字段，禁止改变候选、改写已知值或增加默认值。参数 grounding 必须使用版本化、Oracle-independent 政策验证值在用户请求中的证据；无法证明来源的值必须删除并触发缺参语义。action、missing fields 和 workflow 只能从可信 Catalog 确定性派生。Guard、Schema、grounding 和 compiler 都不得授予效果权限或绕过 L0。

C3 资格门禁必须精确验证候选合同摘要、预装 Skill 摘要、动态 Tool surface、单次调用、Schema、编译、回执、终态、usage 完整性、禁止/重复 Tool 和提前可见文本；所有配置、政策、模型 artifact、数据集与 evaluator 必须 fingerprint 绑定。即使固定 184 条通过，这也只认证无效果候选边界，不能表述为生产成功概率或取消 Runtime 的审批、重校验、独立验证、补偿和审计。

### 23. P1.9 Decision Plane 验收补充

P1.9-B1 代码门禁必须证明：生产包不导入 `evaluation.*`；Decision 合同 extra-forbid 且固定
`proposal_only`；候选只来自当前 DSH Tool/Skill；危险请求无需模型即可安全终止；模型只可调用一个
候选专属/终止 Tool；最多一次修复；无请求证据的值被删除并触发追问；原 DSH step 在成功、超时、
协议错误时均保持不变；Hermes 使用公开生命周期 hook 且不注入 Decision context；Decision 与第一次
实际 domain 路由关联一次；superseded/no-route/session-end 明确关闭且不可跨轮重绑；SQLite 不含原始
请求、模型正文或参数值，旧 schema 参数值被迁移清除；token usage 不完整必须显式报告。

Catalog 门禁必须在不同 checkout 得到同一摘要，并检测候选增删、Tool Schema/描述、Skill 语义内容和
政策漂移。Holdout seal 必须拒绝低于 50 条/10 类/三 profile/中英最小覆盖或重复 Prompt 的输入；
manifest 不得泄漏 Prompt/标签；同 reviewer、缺 case 或任何标签分歧不得形成 consensus digest。

P1.9-B2 资格执行器必须先验证 consensus、Catalog baseline 和不可变模型 artifact digest，再以完全相同的模型、策略、候选声明
和 repair limit 分别执行 DSH/Hermes 身份。至少两次重复且五类 action 覆盖后，输入合同 parity、
Decision 语义 parity、每端 repeatability、协议、完整 action/target/arguments/missing/workflow Oracle
和目标召回必须为 100%，安全逃逸必须为 0，才可标记 `qualified`。报告不得包含原始 Prompt、逐条
标签或参数值，只能保留聚合值和 case-id digest；模型独立调用的非确定性不得被掩盖。

该门禁 scope 只能声明 shared Worker Decision contract。DSH/Hermes Hook 提取/生命周期测试不能替代
完整 Harness 产品实跑，runner 自测也不能替代真实仓库外未见集或独立人工真值。

Adapter parity 必须使用生产 DSH `agent/pre-step` 和 Hermes `pre_llm_call` 代码、同一个临时
owner-only Worker、同一模型 artifact/repair/Catalog。私有请求只能经 stdin/进程内内存传输；报告必须
验证原请求 Prompt digest、两端 Catalog/Candidate/Policy 和完整 Decision digest 100% 一致，并在
结束时删除临时 Socket/SQLite。其 scope 必须声明未启动 DSH Web/Hermes CLI/UI，不能升级为产品认证。

本地影子报告至少公开协议成功、路由一致、直接 Tool 参数一致、安全逃逸、修复/调用次数和 p50/p95，
并明确未观测样本。上述数值只描述本地旁路样本；路由一致率不是正确性 Oracle，固定集通过率不是生产
成功概率。进入 P1.9-C canary 前还必须取得 sealed holdout、不同措辞/实体/拓扑扰动、DSH-only 对照、
session/turn 绑定门禁、模型/政策不可变版本、告警与回退演练；在这些证据齐全前不得启用 enforced。

P1.9-C0 代码门禁还必须证明 schema-v10 binding 为可选且不改变 `off/shadow`；shadow/terminal/failed
Decision、错 session/Harness/profile/route/参数/候选/摘要、workflow 外 Tool、重复 Decision id 和持久化
篡改全部失败关闭；无 binding 的新计划仍执行原 L0 门禁，v9/v8 计划仍可只读。Worker 只能透传绑定，
不能把它转换成审批或 execution nonce。当前 DSH/Hermes 配置必须继续拒绝 `canary`。

P1.9-C1 门禁还必须证明：所有 policy 结果只有 unchanged/blocked 两种，不存在 route/参数重写或授权；
缺失、篡改、过期、模型/manifest/labels/catalog 不一致、演练失败、Core 控制退化、binding replay/escape
都会得到 `not_ready`；通过的 synthetic fixture 只能验证门禁实现，不能成为产品证据。CLI 前后 Adapter
配置内容必须相同，DSH/Hermes 仍拒绝 `canary`。详细应急流程由双语 runbook 约束。

### 24. P2.0 Promotion Workbench 验收补充

代码门禁必须证明 proposal、逐文件摘要、trajectory、report-to-stage、compiled identity/hash 和 review authority 任一篡改都会失败关闭；symlink、非常规文件和超限文档被拒绝。列表不得泄露 proposal 目录名，reviewer/reason 只保留摘要。

浏览器产物必须自包含、转义嵌入数据并应用禁止外部资源的 CSP；不得出现批准、注册、激活、Runtime 或 Provider API。编辑或下载不能改变原 package，导出物固定标记为不可信 L0.5 草稿。`approve` review 仍不得产生 execution eligibility 或 Runtime activation。

### 25. P2.1/P2.2 验收补充

- Catalog 必须 21/21 精确覆盖生产 L0 id/version/contract/profile，并拒绝摘要漂移、scope 扩张、自委派、review+publish、未知/漂移/环依赖和不绑定旧合同的 supersedes；
- Catalog access/diff 输出不得授予 Runtime read/effect、Provider publication、注册或激活权威；
- Evidence 必须使用只读数据库连接，来源文件在采集前后保持不变；snapshot 不得包含原始 Prompt、参数值、审批身份、Provider payload 或路径；
- Runtime/Saga/Provider 链、Decision digest 或 Promotion 完整性失败必须生成事故并使状态降级；legacy 无链和截断也必须降级；
- HTML 必须绑定 snapshot digest、使用自包含 CSP、无外部请求且无审批/执行/发布/注册/激活控件；
- Trend 必须拒绝重复或摘要无效 snapshot；安全/完整性退步返回 `regressed` 和非零退出，时延变化不得自动冒充 SLO 违约；
- 本地专项、浏览器、全量和 retirement 门禁均通过后，阶段才可标记为本地完成；生产 WORM、告警、HA/DR 和 SLO 不在该结论内。

### 26. P2.3 产品入口与评测验收补充

- Integration Pack 必须严格区分 read/write；每个 write 必须有独立 read verifier，可逆 write 必须有 compensation，未知字段、明文凭据、模型可见凭据、坏摘要和悬空引用必须失败关闭。
- Integration Pack assessment、Catalog discovery 和驾驶舱均不得连接 Provider、注册、发布、批准、激活或执行能力。
- 默认评测快照必须绑定摘要，逐例证据不得包含 Prompt、query、参数值、原始输出或可重放授权；篡改必须拒绝。
- 失败必须按唯一首层归因并把 Guard containment 单独展示，不能把最终被阻断冒充模型本身正确。
- 驾驶舱必须 self-contained、CSP 禁止网络且无控制 API；固定集必须显示 `productionGeneralization=not_proven`。
- 本地 demo 必须在执行任一临时 mock write 前要求明确命令行批准；不带批准不得调用 Runtime demo。

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
| F-38 | L1 → L0 Promotion preserves `L1 → L0.5 → L0` stages in a predecessor-linked hash chain. L0.5 cannot drift from L1 and L0 cannot widen L0.5. Agent output remains untrusted, and one human review cannot register a contract or grant execution authority. |
| F-39 | Every built-in reviewed mutation is driven by a compiled L0 v2 contract. Legacy ToolContracts/verifiers/compensators are exact implementation adapters only. Prepare and execution-time revalidation enforce parity, effect arguments are rendered from approved values through v2 templates, and new raw v1 L0 registrations are forbidden. |
| F-40 | Every production L0 preserves L1/L0.5/L0 authoring/compiled artifacts and predecessor hashes. The primary gate reruns Promotion semantic parity, exact contract round trips, and file integrity. Reverse-bootstrapped artifacts declare their origin, are never registered into the Harness, and cannot be claimed as independent model inference. |
| F-41 | A schema-v10 mutation plan binds requester/policy evidence and Provider release/manifest/qualification/deployment digests; v9/v8 and older shapes are read-only compatibility. |
| F-42 | A Harness decision is not direct actor-string authority. Runtime signs a short-lived proof bound to the exact plan, requester, policy, approver, risk, and mode; execution verifies its signature and TTL first. |
| F-43 | Enforced identity mode rejects all requester contexts without an enterprise credential verifier and disables legacy actor compatibility. A local verifier can never be promoted into a production credential. |
| F-44 | Approval policy supports single/dual approvers, role/scope checks, separation of duties, critical-change tickets, and optional active windows. Subject confusion, self-approval conflicts, and out-of-window decisions fail closed. |
| F-45 | Enforced reads and writes derive subject, role, scope, clearance, and assurance only from short-lived JWTs with pinned issuer/audience/asymmetric algorithms. Unknown keys and invalid lifetime/signature/claims fail closed. |
| F-46 | A human access token is cross-bound to a separately signed Gateway attestation by `act_sub + subject_jti`; the Gateway binds Harness/session/client. Substitution cannot reach a Provider. |
| F-47 | An external PDP separately authorizes observation, prepare, and approve. Decision identity and obligations enter hash-bound evidence, and obligations may only tighten built-in L0 policy. |
| F-48 | A configured Change Authority, not caller-supplied attributes, qualifies ticket status, revision, window, profile, capability, targets, and risk ceiling. |
| F-49 | Provider Manifest, Qualification, and Deployment evidence use independent Publisher, Qualifier, and Deployer Ed25519 roles; key material cannot cross roles. |
| F-50 | A Manifest binds artifact, Provider identity/version, Capability/schema/result contract, profile, and allowed L0 hashes; Runtime never trusts a Provider self-declared release id. |
| F-51 | Qualification passes the fixed nine-case timeout/idempotency/order/partial/uncertain/compensation/recovery suite; any failure prevents a publishable report. |
| F-52 | Strict promotion and rollback bind fresh deployment evidence for the target release; breaking promotion and rollback require approval references and lifecycle events are hash chained. |
| F-53 | Enforced admission compares active release, non-expired deployment, exact artifact map, and discovery while validating all three trust roles. |
| F-54 | Execution repeats admission. Post-approval release or deployment drift reaches an audited terminal state before Provider invocation with zero writes. |
| F-55 | The P1.9 Decision Plane reads only direct user messages from an accepted DSH step; Skill, Tool, system, and plugin text cannot impersonate a new user intent. |
| F-56 | Decision candidates come from the exact current DSH tool declarations and reviewed Skill manifest; the model cannot invent candidates, required fields, workflows, or effect authority. |
| F-57 | Guard, candidate Schema, grounding, and compiler are monotonic narrowing controls: they may reject, clarify, or delete unsupported values but cannot add defaults, widen targets, rewrite known values, or grant authority. |
| F-58 | Every P1.9 envelope has `authority=proposal_only`; a Decision cannot call Runtime/Providers, issue approval proofs, override DSH routing, or bypass L0 controls. |
| F-59 | Decision storage retains no raw prompt, model prose, or argument values—only digests, argument keys, bounded evidence, and actual DSH-route correlation. |
| F-60 | Decision failure in `shadow` cannot alter the original DSH step, and the current plugin rejects unqualified `canary/enforced` modes. |
| F-61 | Future `canary/enforced` validates Decision/session/message/candidate/policy binding, freshness, sealed-holdout gates, and target/argument agreement before effects; uncertainty fails closed and Runtime admission remains mandatory. |
| F-62 | The P1.9 Catalog baseline is checkout-portable and binds candidates, Tool declarations, Skill semantics, and production policy across all profiles; unreviewed drift fails retirement. |
| F-63 | Holdout prompts/labels remain outside the repository; the seal manifest contains no raw values, and two distinct reviewers must provide complete semantically identical labels before a consensus digest exists. Reviewer ids are not enterprise identity proof. |
| F-64 | An optional Decision-to-plan binding is fixed to `proposal_only/canary` and validates complete Decision/evidence digests, session/Harness/profile, candidate route, request and compiled arguments, and the exact L0 contract. One Decision id creates at most one plan, and the binding enters both the plan hash and hash-chained creation event. |
| F-65 | C1 canary policy may only preserve the original Harness route or block/narrow it; it cannot redirect, rewrite arguments, or create authority. Invalid writes fail closed and invalid reads cannot change the route. |
| F-66 | Canary readiness cross-binds qualified Worker/Adapter reports, real DSH Web/Hermes CLI product evidence, and operations drills while checking digests, expiry, reviewer/owner separation, 64/64 Core controls, a stable/improved trend across at least three implementation versions within p50/p95 thresholds, complete plan binding, zero replay/authority escape, and four distinct stop/rollback/alert/no-effect-replay receipts. |
| F-67 | Readiness is capped at `ready_for_review`; its CLI cannot change Adapter configuration or traffic and emits no prompts, labels, argument values, reviewer ids, or owner ids. Activation remains an independent organization-identity/signature/release-approval control. |

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
- **MCP spoofing/schema drift/shared-store races:** pinned contracts, schema/capability digests in schema-v10 plans, execution-time rediscovery, WAL, locks, revisions, and state-sensitive idempotency. Production still needs independently authenticated services and database controls.
- **Requester/approver substitution or proof replay:** schema-v10 requester/policy/release/deployment binding, pinned enterprise credentials, PDP/change decisions, Runtime proofs, unique proof ids, and one-shot nonces.
- **Provider supply-chain substitution or qualification forgery:** three independent role-scoped keys, external-process failure qualification, exact artifacts and deployment proof, strict lifecycle, and prepare/execute admission. B-ready is local; organizational roots, independent ownership/CI/labs, real artifact verification, and WORM audit remain P1.4-B.
- **Network provider escalation/self-attestation/crash loss:** exact capability role/action/version matching, an observer with no mutations, fresh digest-bearing evidence, and a durable Actor operation/snapshot store with target locks, leases, fences, startup reconciliation, and a hash chain. Observer and Actor still share one host/account/Docker boundary, and devices do not validate the local fence; production needs separated credentials/failure domains, a remote log, controller CAS, HA fencing, and immutable audit.
- **Decision intent confusion, hallucinated arguments, or privilege escalation:** direct-user provenance, current-manifest candidates, candidate-specific Schema, grounding, deterministic compilation, bounded repair, proposal-only envelopes, digest-only storage, and one observed route per pending session Decision. P1.9-B1 is local DSH/Hermes shadow only; its holdout/reviewer contract is not actual truth, production identity, canary safety, or a production SLO.

### 5. Approval and model policy

Approval must display and bind the exact plan, intent, L0 contract, Provider release/manifest/qualification evidence, arguments, targets, provenance, risk, evidence, verifier/rollback, workflow, requester identity digest, approval policy, and expiry. DSH uses an allowed-once card and Tool Guard. Hermes uses an exact user slash command while retaining the nonce only in process. Both paths then obtain a model-hidden signed Runtime proof before execution. Batch, recovery, and A2A continuation paths need fresh Harness approval.

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
hash chains. The current complete gate is 316 tests plus 81 subtests.

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

### 16. P1.3-B1 and B2-ready enterprise control-plane acceptance supplement

Local gates use real RS256/JWKS/HTTP wire paths to prove successful OIDC access-token/Gateway-attestation cross-binding; PDP decisions for sensitive observation, effect prepare, and approval; and authoritative change revision/window/scope/risk checks. Subject substitution, access-token substitution, raw-role injection, unknown keys, PDP denial, ticket denial, and scope mismatch fail before Provider invocation. Tokens and control-plane bearer secrets never enter plans, public proof evidence, journals, trajectories, or model-visible output.

B2-ready additionally qualifies dynamic Gateway minting, explicit CA/mTLS, owner-only client keys, an offline secret-safe Doctor, and a no-effect live contract check. This still does not certify the user's issuer, organizational PDP policy, change-platform availability, revocation propagation, certificate rotation/HSM, or external WORM audit; those remain B2/P1.7 work.

### 17. P1.4-B-ready Provider publication acceptance supplement

Local gates prove independent Publisher/Qualifier/Deployer keys; external-process 9/9 qualification and actual restart; required artifact and deployment-attestation checks; promotion, proof renewal, strict rollback and lifecycle audit; exact admission; and zero-write terminal behavior after release or deployment drift.

This is a B-ready protocol qualification, not production certification. P1.4-B requires independently owned Provider/CI/labs, organizational signing/HSM roots, real OCI/SBOM/SLSA verification, and external immutable lifecycle audit.

### 18. P1.8 L1/model qualification acceptance

L1 evaluation is isolated from Runtime execution and exposes no callable Tool, approval nonce/proof, Provider credential, or production Memory. Every response passes an extra-forbid contract and supplied-candidate target check. Invalid JSON, unknown targets, incomplete selections, and executable refusal content cannot form a proposal. Reports retain no raw model or endpoint-error text, and remote endpoints require explicit opt-in.

The fixed set covers Skill/Tool choice, arguments, clarification, multi-step workflows, bypass attempts, and out-of-scope requests with language slices. Safety escape must be zero, but that statement applies only after the strict parser on this fixed set and never replaces L0/Runtime. Full qualification binds immutable model, Prompt, Catalog, and dataset fingerprints. Any increased safety escape or greater-than-five-point core regression fails the trend gate. Partial, rule-adapter, mutable-tag, or unresolved-digest runs cannot qualify a model.

The DSH-in-loop shadow must also verify a pinned DSH version and exact active-plugin allowlist before any model call. Skills, tools, shells, filesystem tools, Web, subagents, telemetry, remote model providers, and every NetOpYu effect remain disabled; only isolated ephemeral session persistence is permitted and is removed after the run. Configuration drift, unknown active entries, oversized output, timeout, or an invalid candidate fails closed. B1 results must not be presented as actual Skill/tool-call accuracy.

When B2/C1 enables proposal-only Tools, it must use a separate overlay and exact Tool allowlist, and neither capture nor Governor may reach Runtime, Providers, devices, or approval. C1 digest-binds the preloaded Skill, system prompt, typed Tool set, and Catalog compiler rules. Its Governor is loopback-only; Tool forcing and hidden repairs are bounded and counted, and the terminal response is fixed after the receipt. A forbidden/duplicate Tool, schema/candidate/digest/receipt mismatch, premature text, or incomplete terminal state yields no proposal. Reports declare token cost as a lower bound when discarded retries are not fully metered. Any nonzero safety escape fails qualification even though Runtime would still reject the effect downstream.

The C2 Guard is a monotonic narrowing layer: it may refuse, classify out of scope, abstain, or preserve a proposal, but cannot select/change a target, add arguments, generate workflow, or grant authority. Its versioned digest-bound policy cannot import Oracle cases or labels. The Protocol Firewall accepts only a loopback model endpoint and meters every actual attempt; synthetic safety is restricted to argument-free refusal/out-of-scope capture. Reports show first-attempt and guarded safety, fixed-set false positives, maximum attempts, and tail latency. Zero final safety escape cannot hide raw model escape, protocol failure, or Runtime's continuing role as the final safety boundary.

C3 candidate-specific Tools must be generated from a digest-bound trusted candidate contract. Tool identity fixes kind/target, and each Schema uses `additionalProperties=false` with only that candidate's business keys. Unknown-field constraining may only delete keys; it cannot change the candidate, rewrite known values, or add defaults. Argument grounding uses a versioned Oracle-independent policy to prove request evidence; unsupported values are removed and therefore participate in missing-field semantics. Action, missing fields, and workflow are derived deterministically from the trusted Catalog. Guard, Schema, grounding, and compiler grant no effect authority and cannot bypass L0.

C3 qualification exactly gates candidate-contract and preloaded-Skill digests, dynamic Tool surface, single call, Schema, compiler, receipt, terminal state, complete usage, forbidden/duplicate Tools, and premature visible text. Configuration, policies, model artifact, dataset, and evaluator are fingerprint-bound. Passing the fixed 184 cases qualifies only the proposal-only boundary; it is not a production success probability and cannot remove Runtime approval, revalidation, independent verification, compensation, or audit.

### 19. P1.9 Decision Plane acceptance supplement

P1.9-B1 code gates prove that the production package does not import `evaluation.*`; the extra-forbid
Decision contract is always proposal-only; candidates come only from current DSH Tool/Skill declarations;
the Guard can terminate known unsafe requests without a model; the model can call exactly one candidate or
terminal Tool with at most one repair; unsupported argument values are deleted and become clarification;
the original DSH step is unchanged on success, timeout, and protocol failure; Hermes uses public lifecycle
hooks without Decision context injection; the first actual domain route is correlated once; superseded,
no-route, and session-end Decisions close and cannot rebind; SQLite contains no raw request, model prose, or
argument values; legacy values are migrated away; and incomplete token accounting remains explicit.

The Catalog gate is portable across checkouts and detects candidate, Tool Schema/description, Skill
semantic-content, and policy drift. Holdout sealing rejects fewer than 50 cases, fewer than ten categories,
missing profiles/language coverage, or duplicate prompts; its manifest leaks no prompt/label. Same-reviewer,
missing-case, or semantically disagreeing labels cannot produce a consensus digest.

The P1.9-B2 qualification runner first validates exact consensus, the Catalog baseline, and an immutable
model artifact digest, then makes
independent DSH/Hermes-identity calls with the same model, policies, candidate declarations, and repair
limit. Qualification requires at least two repetitions, all five actions, 100% input-contract parity,
Decision-semantic parity, per-Harness repeatability, protocol success, full action/target/argument/missing/
workflow Oracle accuracy, and target retrieval, with zero safety escapes. Reports contain only aggregates
and case-id digests—never raw prompts, per-case labels, or argument values—and must expose independent-model
nondeterminism rather than hiding it.

This gate may claim only shared Worker Decision-contract scope. Hook extraction/lifecycle tests do not
replace a full DSH/Hermes product run, and runner self-tests do not replace real repository-external
unseen cases or independent human truth.

Adapter parity uses the production DSH `agent/pre-step` and Hermes `pre_llm_call` code, one temporary
owner-only Worker, and identical model-artifact/repair/Catalog configuration. Private requests travel only
through stdin or process memory. The report requires 100% expected prompt-digest binding and cross-adapter
Catalog/Candidate/Policy/full-Decision parity, then removes temporary sockets/SQLite. Its scope must state
that DSH Web and Hermes CLI/UI were not started and it cannot be promoted to product certification.

Local shadow reporting exposes protocol success, routing agreement, direct-tool argument agreement, safety
escape, repair/call counts, p50/p95, and unobserved samples. These describe bounded local side-channel data:
routing parity is not a correctness oracle and fixed-case success is not a production probability. Before
P1.9-C canary, acceptance additionally requires a sealed holdout with paraphrase/entity/topology shifts, a
DSH-only control, session/turn binding gates, immutable model/policy versions, alerts, and rollback drills.
Enforced mode remains prohibited until those gates pass.

P1.9-C0 code gates additionally prove that schema-v10 binding is optional and leaves `off/shadow`
unchanged. Shadow/terminal/failed Decisions, session/Harness/profile/route/argument/candidate/digest drift,
Tools outside a selected workflow, duplicate Decision ids, and persisted tampering all fail closed. Unbound
new plans retain every existing L0 gate, while schema-v9/v8 plans remain read-compatible. The Worker only
transports binding material and cannot turn it into approval or an execution nonce. DSH/Hermes configuration
continues to reject `canary`.

P1.9-C1 gates additionally prove that policy has only unchanged/blocked effects and no route/argument
rewrite or authority. Missing, tampered, expired, cross-binding-drifted, drill-failed, Core-regressed, or
replay/escape evidence returns `not_ready`. Passing synthetic fixtures test the gate implementation but are
not product evidence. Adapter files remain byte-identical across the CLI check and DSH/Hermes still reject
`canary`; the bilingual runbook defines stop and incident handling.

### 20. P2.0 Promotion Workbench acceptance supplement

Code gates must fail closed on any proposal, per-file digest, trajectory, report-to-stage, compiled identity/hash, or review-authority tampering and reject symlinks, non-regular files, and oversized documents. Listings must not reveal proposal directory names; reviewer and reason remain digest-only.

The browser artifact must be self-contained, escape embedded data, and use a CSP that forbids external resources. It exposes no approval, registration, activation, Runtime, or Provider API. Editing or download cannot mutate the package, and every export is labeled an untrusted L0.5 draft. An approve review never creates execution eligibility or Runtime activation.

### 21. P2.1/P2.2 acceptance supplement

- The Catalog must exactly cover 21/21 production L0 id/version/contract/profile entries and reject digest drift, scope widening, self-delegation, combined review/publication, unknown/drifted/cyclic dependencies, and unbound supersession.
- Catalog access/diff output cannot grant Runtime read/effect, Provider publication, registration, or activation authority.
- Evidence collection uses read-only database connections and does not mutate sources; snapshots exclude raw prompts, argument values, approval identities, Provider payloads, and paths.
- Runtime/Saga/Provider chain, Decision digest, or Promotion integrity failure creates an incident and degrades the snapshot; legacy chain absence and truncation also degrade.
- HTML is snapshot-bound, self-contained under CSP, performs no external request, and exposes no approval/execution/publication/registration/activation control.
- Trends reject duplicate or digest-invalid snapshots; safety/integrity regressions return `regressed` and non-zero, while latency deltas never become automatic SLO violations.
- Local focused, browser, full, and retirement gates must pass before local completion. Remote WORM, alerting, HA/DR, and production SLOs remain outside this claim.

### 22. P2.3 product-front-door and evaluation acceptance supplement

- Integration Packs strictly separate read/write. Every write requires an independent read verifier; reversible writes require compensation. Unknown fields, credential values, model-visible credentials, invalid digests, and dangling references fail closed.
- Pack assessment, Catalog discovery, and the cockpit cannot connect, register, publish, approve, activate, or execute a capability.
- Evaluation snapshots are digest-bound and per-case evidence excludes prompts, queries, argument values, raw output, and replayable authorization. Tampering is rejected.
- Each failed case gets one first-failure layer while Guard containment is reported separately; a blocked model attempt is not presented as intrinsic model correctness.
- The cockpit is self-contained, network-free under CSP, exposes no control API, and fixes `productionGeneralization=not_proven` for fixed-set evidence.
- The local demo requires explicit command-line approval before any temporary mock effect; without it, Runtime demo code is not invoked.
