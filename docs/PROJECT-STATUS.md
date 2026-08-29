# 项目进展与路线图 / Project Status and Roadmap

> 最后核验 / Last verified: 2026-08-29  
> 当前里程碑 / Current milestone: **P1.4-B-ready 外部 Provider 协议与部署证明本地闭环 / External Provider protocol and deployment-evidence loop complete locally**

本文档是项目阶段、已完成事项和后续工作的唯一汇总入口。README 说明项目是什么；HLD、LLD、SSD 和 ARCHITECTURE 说明如何设计；本文档只回答三个问题：现在到哪里、哪些确实完成、下一步做什么。

This document is the single summary for project phase, completed work, and future work. README explains what the project is; HLD, LLD, SSD, and ARCHITECTURE explain how it is designed; this document answers where the project is now, what is actually complete, and what comes next.

---

## 中文

### 1. 当前结论

项目已经完成从旧自建 L0 Agent Framework 向 **DSH/Hermes Harness Adapter + NetOpYu Domain Effect Runtime** 的迁移，并形成了可在本地重复运行的端到端参考实现：

```text
人 / Operator
    ↓
DSH（主 Harness）或 Hermes（可选 Adapter）
    ↓  Domain L1 Skill：理解、追问、诊断、编排
Network/Service L0 Skill：确定性执行合同
    ↓
Domain Effect Runtime：校验、权限、计划、审批、执行、验证、补偿、审计
    ↓
MCP Provider：Network Observer / Network Actor / Service Providers
    ↓
Containerlab 网络仿真、Service Layer 模拟系统或后续真实外部系统
```

当前阶段可概括为：

- **框架迁移已完成**：DSH 替代旧 L0 通用智能体框架，Hermes 通过 Adapter 复用同一领域 Runtime；
- **P0.5 确定性 Runtime 原型已完成**：写操作不依赖模型直接决定安全语义；
- **本地网络/业务双层闭环已完成**：Containerlab 提供真实协议栈网络仿真，Service MCP 提供模拟业务系统；
- **21/21 内置写能力已升级到 L0 v2**：编译后的不可变合同是 Runtime 权威来源；
- **L1 → L0.5 → L0 可解释轨迹已补齐**：21/21 存量能力均有源码化、可校验、可逆向还原的三阶段材料；
- **P1.3-A 本地可信审批链已完成**：schema-v7 计划绑定 requester/policy，DSH/Hermes 人工确认后由 Runtime 签发短时签名 proof；
- **P1.3-B1 企业控制面参考 Adapter 已完成**：OIDC/JWKS access token、Gateway sender attestation、HTTP PDP 和 Change Authority 通过本地真实 HTTP 资格测试，read/write 均 fail closed；
- **P1.3-B2-ready 企业接入包已完成**：按 Harness session 动态 mint Gateway attestation、显式 CA/mTLS transport、无泄密 Doctor 和无网络效果 live contract check 均通过本地资格测试；
- **P1.4-B-ready Provider 供应链已完成**：仓库外 JSONL 独立进程、真实重启、Publisher/Qualifier/Deployer 三角色、OCI/SBOM/provenance 必需 digest、部署证明、严格 promote/rollback 与 schema-v9 binding 形成本地闭环；
- **生产环境尚未认证**：企业身份、真实审批、真实厂商设备、分布式高可用、不可变远端审计、灾备和生产 SLO 仍是后续工作。

因此，身份侧已具备 P1.3-B2 现场接入条件，Provider 侧已完成 P1.4-B-ready 接入协议。下一步是把这两个 ready 包接入真实组织系统、独立 Provider 仓库/CI、签名根和 artifact 服务，而不是扩建另一套 Agent Framework。

### 2. 阶段总表

图例：✅ 本地完成；🟡 已有原型但仍需生产资格认证；⬜ 待建设；🚫 当前明确不支持。

| 阶段 | 状态 | 已完成范围 | 完成边界 |
|---|---|---|---|
| P0 Harness 迁移 | ✅ | DSH 主路径、Hermes Adapter、共享 Python Worker、旧框架退休门禁 | 不代表 Harness 自身具备网络确定性语义 |
| P0.5 Domain Effect Runtime | ✅ | 参数与来源校验、不可变计划、风险、审批绑定、执行前重校验、验证、补偿、审计、恢复 | 本地原型完成；“100%”仅指固定 Oracle 测试，不是现实世界概率承诺 |
| P0.75-A 基础网络实验 | ✅ | Containerlab、OSPF、故障切换与恢复 | Linux/FRR 仿真，不等于厂商设备行为 |
| P0.75-B 小型现网拓扑 | ✅ | 园区、IDC、DMZ、双 ISP、OSPF/eBGP、拓扑与路径查询 | 20 节点本地实验网络 |
| P0.75-C DC Fabric | ✅ | BGP EVPN/VXLAN L2VPN 仿真与验证 | EVPN L3VPN、MPLS L2/L3VPN 尚未覆盖 |
| P0.8 Service MCP | ✅ | 身份、应用、策略、变更、CMDB、平台等模拟 Provider；stdio/HTTP MCP | 数据仍是本地模拟，不是真实企业业务系统 |
| P0.9 Observer Boundary | ✅ | 只读 Network Observer、身份与版本固定、独立读取验证 | 本地进程/凭据隔离，尚非生产隔离域 |
| P1.0 Durable Actor | ✅ | 写入 Actor、操作日志、幂等、重启恢复 | 单机持久化参考实现 |
| P1.1 Capability/Saga | ✅ | Capability SPI、读策略、终态 Envelope、跨域 Durable Saga | 本地闭环；分布式故障域仍待建设 |
| P1.2 L0 v2 与可解释轨迹 | ✅ | 21/21 编译合同、运行时绑定、L1/L0.5/L0 轨迹、Promotion 校验 | 内置能力完成；外部能力发布与签名仍待建设 |
| P1.3 企业身份与审批控制面 | 🟡 | P1.3-A 签名 proof；B1 authority Adapter；B2-ready 动态 Gateway mint、CA/mTLS、Doctor/live contract check | P1.3-B2 真实企业 issuer、Gateway、PDP、审批/变更系统和不可抵赖审计仍待现场资格化 |
| P1.4 Provider 发布与资格认证 | 🟡 | P1.4-B-ready 外部进程 9/9+restart、三角色签名、artifact digest/deployment proof、生命周期、admission 和 schema-v9 binding | 独立组织仓库/CI/实验室、企业 HSM 根、真实 OCI/SBOM/SLSA 和 WORM 仍待现场认证 |
| P1.5 真实网络/厂商集成 | ⬜ | 见后续路线图 | 生产网络写入前必需 |
| P1.6 分布式可靠性与 HA/DR | ⬜ | 见后续路线图 | 多实例生产部署前必需 |
| P1.7 安全、审计、可观测性与 SLO | ⬜ | 见后续路线图 | 生产前必需 |
| P1.8 L1/模型资格评测 | 🟡 | A/B1/B2/C1 完成 7B 160 基线；C2 完成 160+24 对抗集，最终 safety escape 0、E2E 61.96% | 协议 86.41%、workflow 50%、追问 36.67%；更强实用模型完整基线仍待完成 |

### 3. 已完成能力清单（Done）

#### 3.1 Harness 与集成边界

- DSH 是主 Harness，负责会话、模型、UI/CLI、通用 Tool/Skill 生命周期；
- Hermes 作为可选 Adapter 接入，不复制 Runtime 的安全、事务或网络效果语义；
- 两个 Harness 共用 Worker、L0 注册表、Runtime、Provider、验证器、补偿器和审计；
- A2A 具备发现、委派、循环/深度保护、持久化 continuation 和无 Peer 时的受控拒绝；
- 已提供本地 loopback DC Peer，避免 UI 测试依赖外部 A2A 环境。

#### 3.2 Domain Effect Runtime

- 严格类型、必填字段、目标存在性、枚举/范围和参数来源校验；
- 不可变 `IntentSpec`、`intent_hash`、`plan_hash`、Provider/schema/capability hash；
- 风险分级、一次性审批 nonce、主体/计划/合同绑定和过期控制；
- 执行前重新读取并校验状态，阻止 TOCTOU 漂移；
- 写入后通过独立 Observer 执行 typed postcondition 验证；
- 合同化补偿、精确回滚、人工介入终态和持久化恢复；
- SQLite 状态机、Actor Journal 和防篡改事件哈希链；
- Durable Saga 支持网络与业务 Provider 的跨域提交、验证和逆序补偿；
- 模型只能提出候选意图或编排 L1，不能绕过 L0/Runtime 安全边界。

#### 3.2.1 P1.3-A 身份与审批证明

- PreparedPlan schema v9 把 requester/policy 和 Provider release/manifest/qualification/deployment evidence 纳入不可变 plan hash；
- DSH `allowed-once` 或 Hermes 用户 slash command 之后，Adapter 调用 `runtime-approve`，而不是把 `approval_actor` 字符串直接当作授权；
- Runtime 签发短时 HMAC-SHA256 proof，精确绑定 plan、requester、policy、approver、risk 和 single/dual mode；
- execute 在消费 nonce 前验证 proof 签名、TTL 和全部绑定；proof id 在 journal 中唯一，公开证据进入事件哈希链；
- 已验证篡改签名、跨计划证明、主体/session 替换、重复消费、关键变更自批、缺少工单和窗口不合法等 fail-closed 行为；
- `local-simulation` 明确以 owner-only Worker/OS account 为模拟信任根；`enforced` 未注入企业 credential verifier 时拒绝所有主体，并禁用 legacy actor string。

#### 3.2.2 P1.3-B1 企业控制面参考 Adapter

- `network_runtime/enterprise.py` 只接受固定 issuer/audience/非对称 algorithm 的短时 JWT，并从受信配置 URL 拉取、缓存和 unknown-kid 刷新 JWKS；HTTP 仅可在显式 loopback 资格实验中启用；
- 人的 OIDC access token 与独立 Gateway sender attestation 通过 `act_sub + subject_jti` 交叉绑定；Gateway 再绑定 Harness、智能体 session 和 client id；原始 context 中伪造的 role/scope/clearance 不生效；
- 同一企业 verifier 同时覆盖敏感 observation 和 effect requester/approver；HTTP PDP 分别决策 `observation.read`、`effect.prepare`、`effect.approve`，decision id/policy/version/obligations 进入主体或 proof 证据；
- Change Authority 校验 ticket status/revision、真实维护窗口、profile/capability/targets 和 risk ceiling，返回的最小公开记录进入签名 proof 与 hash-chain journal；
- DSH/Hermes 从进程 secret 配置读取 requester/approver token、Gateway token 和 ticket，不将凭据作为模型工具参数；任何 endpoint、签名、claim、PDP 或工单异常都 fail closed；
- 本地 Threading HTTP server 使用真实 RS256/JWKS/PDP/change wire path 验证成功链、主体/令牌替换、角色注入、未知 kid、PDP deny、工单 deny 和 scope mismatch。

B1 只证明基础 Adapter 合同和安全失败语义；其后由 B2-ready 子阶段补齐本地 CA/mTLS 和动态 mint。由于仍没有真实企业系统、HSM、外部 WORM 审批日志和组织 RBAC 数据，P1.3 仍为 🟡。

#### 3.2.3 P1.3-B2-ready 企业接入包

- Gateway attestation 不再要求为每个新会话静态预置：可通过 `netopyu.gateway-mint/v1` 以 access token、Harness、session 和 purpose 动态签发，再按 B1 双 JWT 规则验证；
- JWKS、mint、PDP 和 Change Authority 使用同一个显式 TLS context；支持部署 CA 和 client certificate/key，client key 必须 owner-only，默认 `trust_env=false`；
- `scripts/netopyu-enterprise doctor` 离线检查配置、URL 策略、algorithm/AAL、证书组合和权限，只输出 digest/boolean；
- `contract-test` 实际验证 requester/approver、动态 mint、read/prepare/approve PDP 和 change record，但不创建计划、不调用 Provider、不产生网络效果；
- 本地测试同时覆盖动态 session binding、证书加载、宽权限私钥拒绝和 token/endpoint/secret 不泄漏。

这使项目可以直接进入企业现场 B2 对接，但仍缺少现场 issuer/CA、组织策略、吊销/轮换、可用性和不可抵赖审计证据，因此 P1.3 状态保持 🟡。

#### 3.2.4 P1.4-B-ready Provider 发布供应链

- `provider_release.py` 定义 Manifest、Qualification、Release Bundle、Deployment Attestation 与 scoped Trust Store；
- Publisher、Qualifier、Deployer 必须使用独立 Ed25519 key material；
- `provider_external.py` 通过受限 JSONL 独立进程覆盖固定 9 项故障语义并真实重启验证持久状态；
- SQLite registry 支持 deployment-aware promote、证明续期、严格 rollback、deprecate 和 hash-chain audit；
- `release_provider_id` 来自部署配置而不是 MCP 自报；admission 精确比较 active release 与实际 identity、Capability、schema、result contract 和 L0 contract hash；
- PreparedPlan 升级 schema v9，把 release、manifest、qualification 和 deployment digest 纳入审批 hash，审批后同 release 重部署也会在 Provider 调用前失败关闭；
- `scripts/netopyu-provider` 提供外部资格、三类证据 schema/签名/bundle/verify、compatibility 和完整 lifecycle 命令。

B-ready 测试把 fixture 复制到仓库外临时目录并以独立进程运行，但源码仍由本仓库拥有，key/SQLite/artifact digest 仍是本地 fixture。独立组织 Provider 仓库、企业 signing/HSM root、独立 CI/实验室、真实 OCI/SBOM/SLSA 内容验证和外部 WORM 属于 P1.4-B 现场认证，因此阶段仍为 🟡。完整说明见 [Provider 发布与资格认证供应链](provider-supply-chain.md)。

#### 3.3 L0 Skill SDK 与可解释性

- 支持原子 S1、约束式 S11、扩展式 S11 和 S1+S2+… Composite Saga；
- 所有派生在编译期扁平化；Runtime 只消费版本化、不可变的编译合同；
- 21/21 内置 reviewed mutation capabilities 已迁移到 L0 v2 权威合同；
- 21/21 均保存 L1 自然语言、L0.5 结构化自然语言和 L0 编译结果；
- 21/21 均通过文件覆盖、哈希链、精确 round-trip 和 Promotion-ready 校验；
- 轨迹入口：[`network_runtime/l0/production_trajectories/`](../network_runtime/l0/production_trajectories/)；
- Domain L1 Skill 入口：[`profiles/`](../profiles/)；
- L0 authoring 示例入口：[`network_runtime/l0/examples/`](../network_runtime/l0/examples/)；
- 外部候选与 Promotion 示例入口：[`network_runtime/l0/promotion_examples/`](../network_runtime/l0/promotion_examples/)。

#### 3.4 Network Layer 与 Service Layer

- 本地小型现网覆盖园区、IDC、DMZ、双 ISP、OSPF/eBGP 和路径查询；
- DC Fabric 覆盖 BGP EVPN/VXLAN L2VPN；
- Runtime 已能对 Containerlab 仿真网络执行受控配置、独立验证、故障注入和精确回滚；
- Service Layer 已拆为身份、应用、策略、变更、CMDB、平台等 MCP Provider；
- Runtime 不关心下层 Layer 之间如何交互，只依赖版本化 read/write Capability 合同；
- Network Observer 与 Network Actor 分离，读回验证不直接信任写入返回值。

#### 3.5 测试、基线与文档

最近一次本地核验结果：

| 证据 | 当前结果 | 说明 |
|---|---:|---|
| Python gate | 316 tests + 81 subtests | Runtime、Adapter、Provider、Skill、身份控制面、外部资格/三角色部署证明、审批证明、P1.8 reference/B1/B2/C1/C2 DSH shadow、恢复等 |
| Core-72：DSH only | 5/64（7.8%） | 固定风险/故障 Oracle |
| Core-72：DSH + Runtime | 64/64（100%） | 固定风险/故障 Oracle |
| 有效操作 | 8/8 vs 8/8 | 两条路径都能完成无故障请求 |
| 内置 L0 v2 绑定 | 21/21 | 编译合同是运行时权威来源 |
| 可读轨迹 | 21/21 | L1、L0.5、L0、报告和哈希链齐全 |
| 精确 round-trip | 21/21 | 可读材料精确恢复运行时合同 |
| Promotion-ready | 21/21 | 内置能力的离线资格门禁通过 |
| Retirement gate | PASS | Python、Node、静态边界、HITL/A2A、投影、检索、负载/可靠性 |

Core-72 固定了 L1 决策，因此只量化 Runtime 的确定性增量；它**不测**模型的意图识别、Skill 选择、追问质量或参数提取准确率。P1.4-B-ready 后最近 3 个不同执行代码指纹趋势为 `stable`：Runtime 仍为 64/64，三版本本机中位 p50 7.599 ms、p95 8.680 ms；P1.8-B1 后复核的 50 样本 p50 为 7.281 ms。单次机器时延不是生产 SLO。

#### 3.6 P1.8-A/B1/B2/C1/C2 L1/模型资格层

- 51 个不同语义原型、版本化 160 条语言/措辞 Oracle：28 Skill、36 Tool、32 workflow、30 clarification、20 safety refusal、14 out-of-scope；
- 中文 75、英文 51、中英混合 34，覆盖 LAN/DC/WAN，数据集与生成源码精确一致；
- 当前 Profile Tool metadata + DSH Skill manifest 形成 Catalog，126/126 有目标场景在 top-12 候选中；
- 严格 `L1Decision` 阻止 extra fields、越界 target、不完整 selection 和携带执行内容的 refusal；
- 规则基线明确 `model=none`；本地模型 Adapter 默认只允许 loopback、无重试、无 proxy、响应有界且不保存原文；
- 完整集、immutable model artifact digest、Prompt/Catalog/dataset fingerprint、绝对门槛和版本化回退比较共同决定资格；
- fingerprint-bound checkpoint 支持长模型运行显式恢复，子集不得记录或冒充完整认证；
- P1.8-A 直接使用与 DSH 相同 Ollama endpoint 和当前 Skill/Tool Catalog，但不经过 DSH 完整 session/tool loop，因此不冒充 DSH UI 端到端准确率。
- P1.8-B1 以精确 DSH `0.1.1-rc.2` 插件白名单、隔离临时 home 和无 Skill/Tool/effect overlay，真实穿过官方 Agent/Session/LLM loop；任一配置/版本漂移在模型调用前 fail closed；
- 同一 7B 的 B1 160/160 仍不合格：selection 72.92%、parameter F1 62.43%、clarification 0%、workflow 62.50%、E2E 56.87%、safety escape 0；相对 reference E2E 仅 +1.24pp；
- B1 明确只测 DSH Prompt/session composition，不冒充实际 Skill loading/tool-call；版本化摘要为 `data/l1_dsh_shadow_baselines.json`。
- P1.8-B2 已实现唯一运行时 Skill `l1-decision-capture` 和无效果 `submit_l1_decision` capture Tool；插件不连接 Runtime、Provider、shell、网络、设备或审批；
- B2 启动时精确审计 DSH `0.1.1-rc.2`、30 个活动 entry、52 个必须禁用 entry、插件路径与模型所见双 Tool；多帧 transcript 对 Skill 加载、调用顺序、重复/额外 Tool、schema、候选合同、回执和终止逐项 fail closed；
- `qwen2.5:7b` B2 完整 160 条的目录/Tool 暴露 100%，但 Skill 加载 31.25%、capture 调用 11.87%、schema/接收 5%、候选合同有效 2.50%、精确顺序 9.38%、E2E 1.87%，明确不合格；禁止 Tool 调用为 0，重复 capture 0.63%、提前文本 26.87%；
- `qwen3.6:27b` 单条功能 smoke 的全部协议门禁和候选 E2E 均通过，耗时 122.423 秒，但不可认证；
- 版本化结果为 `data/l1_dsh_tool_shadow_observations.json`；B2 构建、功能路径和 7B 完整失败基线完成，但完整强模型资格仍未完成。
- P1.8-C1 保留 B2 对照，新增摘要绑定的 L0.5 预装 Skill、五个互斥无效果类型化 Tool、Catalog 元数据编译器和 loopback 协议 Governor；模型不再负责 Skill loading、workflow/missing-field 元数据或最终终止文本；
- C1 7B 完整 160 条把 capture/schema/候选合同提升至 98.75%/80%/68.13%，选择/参数 F1/workflow/E2E 提升至 52.08%/53.02%/56.25%/36.88%；禁止 Tool 和重复 capture 为 0；
- C1 仍不合格：clarification recall 26.67%、out-of-scope 0%、safety escape 5%；17 次隐藏协议修复、2 次耗尽，transcript token 不含丢弃重试；版本化结果为 `data/l1_dsh_controlled_tool_observations.json`。
- P1.8-C2 新增版本化 Guard Policy、typed/candidate Protocol Firewall、完整实际调用计量和 24 条对抗/反误杀场景；Guard 只能拒绝、越界或弃权，不能选择 Capability 或补参数；
- C2 7B 完成 184/184：原 160 子集选择/参数 F1/E2E 为 65.62%/69.09%/58.75%，最终 safety escape 0；新增 24 条 E2E 83.33%，整体 E2E 61.96%；
- C2 仍不合格：协议有效率 86.41%、workflow 50%、clarification recall 36.67%；模型首轮 safety escape 9.38%，Guard 后才为 0。267 次模型调用、34 次修复和 121 次无效合同尝试已完整计量；版本化结果为 `data/l1_dsh_guarded_tool_observations.json`。

### 4. 后续路线图（To-do）

#### P1.3 企业身份与审批控制面（最高优先级）

目标：让“谁提出、谁审批、谁执行、凭什么有权”形成可证明的生产身份链。

- **已完成 P1.3-A**：本地主体规范化、schema-v7 requester/policy binding、Runtime 签名 proof、proof audit、single/dual/SoD/ticket/window policy，以及无企业 verifier 时的 enforced fail-closed；
- **已完成 P1.3-B1**：可配置 OIDC/JWKS、双 JWT Gateway binding、HTTP PDP/Change Authority、敏感 read/write 统一身份链与本地 HTTP 故障资格测试；
- **已完成 P1.3-B2-ready**：动态 Gateway token mint、显式 CA/mTLS、secret-safe Doctor 和 no-effect live contract check；
- **P1.3-B2 待完成**：在真实企业 issuer/SSO/JWKS、Gateway、PDP 和审批/变更系统运行接入包并留存证据，同时将 MCP service identity 纳入组织责任链；
- 用真实组织 RBAC/ABAC 数据验证资源、动作、环境、时间窗和风险上下文策略；
- 支持双人审批、职责分离、变更单/工单和维护窗口；
- 为 DSH/Hermes 建立不可冒用、不可重放的主体证明；
- 把身份与审批证据纳入计划、执行和审计哈希链。

完成判据：身份混淆、越权、审批重放、审批转移、过期窗口和职责冲突故障集全部 fail-closed，并能从审计证据重建责任链。

#### P1.4 Provider 发布与资格认证

目标：把当前内置 L0 的可靠机制扩展为外部团队也能安全使用的 Capability 供应链。

- **已完成 P1.4-B-ready 外部资格协议**：绝对 argv/cwd、最小环境、受限 JSONL、固定 9/9、真实 restart 和持久状态；
- **已完成三角色部署 admission**：required artifacts、短期 deployment attestation、严格 promote/rollback、schema-v9 四 digest binding 与部署 drift 零写入；
- **P1.4-B 现场待完成**：独立拥有的 Provider 仓库/CI/实验室、企业 signing/HSM 根、真实 OCI/SBOM/SLSA 验证和外部 WORM lifecycle audit；
- 建立真正的正向 L1 → L0.5 → L0 转换评测；当前 reverse bootstrap 证明的是可解释性和 round-trip，不等于自动正向编译准确率；
- 为人工评审提供合同 diff、依赖图、风险、验证和补偿证据。

完成判据：一个仓库外 Provider 可通过签名发布、隔离验收、灰度启用和一键退回上一合同版本，且不能绕过 Runtime 注册表。

#### P1.5 真实网络与厂商资格化

目标：从 Linux/FRR 仿真推进到真实 CLI/API/Controller 语义。

- 选择一个最小真实纵切面，例如 Huawei eNSP/测试设备或 Cisco/H3C 实验环境；
- 实现并认证 Netmiko/NAPALM/RESTCONF/NETCONF 或厂商 Controller Adapter；
- 建立命令白名单、配置快照、候选配置、commit-confirm、独立 readback 和回滚测试；
- 按厂商/型号/版本保存行为差异和资格矩阵；
- 在实验条件具备后扩展 EVPN L3VPN、MPLS L2VPN/L3VPN、状态防火墙、AAA/802.1X、无线 AP/RF 等场景。

完成判据：至少一个真实 Provider 和一组真实设备/控制器完成计划、审批、提交、独立验证、故障补偿、基线恢复和重复执行测试。

#### P1.6 分布式可靠性、HA 与 DR

目标：消除当前单机 SQLite/进程模型的生产单点。

- 远端 durable transaction store、队列和分布式 fencing/lease；
- 多实例 Actor 的幂等键、CAS、leader fencing 和未知终态协调；
- Observer/Actor 独立凭据、进程、节点和故障域；
- 备份、恢复、跨节点接管、日志重放和灾难恢复演练；
- 长时间运行、网络分区、时钟漂移、进程崩溃和存储故障测试。

完成判据：在节点/进程/网络/存储故障注入下不产生未审批重复效果，并满足定义的 RPO/RTO。

#### P1.7 安全、审计、可观测性与 SLO

目标：让系统可安全运营、可观察、可追责并可量化服务质量。

- Provider 间 mTLS、集中 secrets、静态/传输加密和网络 egress allowlist；
- 外部 append-only/WORM 审计存储与独立验证；
- metrics、trace、结构化日志、告警和每个事务的 evidence bundle；
- 定义成功率、错误预算、p95/p99、恢复时间、回滚成功率等 SLI/SLO；
- 生产规模负载、混沌、容量、泄漏和退化测试；
- 模型、Skill、合同和 Provider 的受控发布、canary 和 rollback。

完成判据：安全基线和威胁模型关闭高风险项，审计不可由 Runtime 管理员单方改写，SLO 有持续测量和告警。

#### P1.8 L1 与模型资格评测

目标：单独量化模型参与部分，不让 Runtime 的 100% 固定 Oracle 掩盖 L1 误差。

- **P1.8-A 已完成**：160 条意图/Skill/Tool/参数/追问/workflow/拒绝集、严格非执行合同、reference model Adapter、绝对门槛、报告、fingerprint、趋势和 checkpoint；
- **P1.8-A 本地基线已完成**：immutable `qwen2.5:7b` 完整 160 条明确不合格（selection 69.79%、parameter F1 62.11%、clarification 0%、workflow 62.50%、E2E 55.63%、safety escape 0）；
- **27B 仅完成 smoke**：`qwen3.5:27b` 六类各 1 条，p50 约 148 秒且 workflow 样本失败，明确不可认证；
- **P1.8-B1 已完成**：禁用 Skill/Tool/effect/shell/FS/Web/子代理的 DSH headless shadow，完成同一 immutable 7B 的 160/160 基线；
- **P1.8-B2 已完成**：受控 Skill loading + 只记录 proposal、永不调用 Runtime/Provider 的 capture Tool，已完成 7B 160/160 失败基线并以 27B 单条证明成功路径；
- **P1.8-C1 已完成**：确定性预装 L0.5 Skill + 类型化候选 Tool + 有界协议 Governor，完成同一 7B 的 160/160 对照；协议显著改善但 safety escape 5%，明确不合格；
- **P1.8-C2 已完成**：确定性安全/领域 Guard、最多三次调用的 Protocol Firewall、完整 usage 计量及 24 条对抗/反误杀集；最终 safety escape 0，但 7B 仍未通过协议和语义门槛；
- 对 27B 和云模型分别运行完整准确率、拒绝率、token/成本和时延基线；
- 已加入首批提示覆盖、Unicode 混淆、过期授权、命令注入和安全术语反误杀；仍需扩展跨域冲突、状态陈旧和长对话；
- 继续降低协议失败和尾时延，恢复 workflow；不得通过 Oracle 特判、放宽合同或让 Guard 选择 Capability 来提高成绩；
- L1 输出只能进入 L0 严格解析/校验，任何模型都不能成为审批、验证或回滚的权威；
- 对成熟 L1 建立可审计的 L0.5 候选生成，但仍需机器门禁和人工 Promotion。

当前核心判据已满足：A/B1/B2/C1 均有完整 7B 160 基线，C2 另有 184 条完整、指纹绑定基线并将固定集最终 safety escape 降为 0。P1.8 整体保持 🟡，直到至少一个可接受成本的更强模型完成全部 184 条、保持安全边界并通过协议与语义绝对门槛。

#### P2 可选增强

- L0.5 可视化编辑器、合同 diff/graph、Promotion 控制台；
- 多团队/多租户 Capability Catalog 与策略委派；
- 更完整的 evidence plane、运营分析和事故复盘；
- 更多网络/业务工作流和经业务验证的 L1 → L0 下沉；
- 生产反馈驱动的能力成熟度、弃用和兼容性治理。

### 5. 推荐实施顺序

建议下一步不要同时横向增加大量协议或 Skill，而按一个真实纵切面推进：

1. **P1.3-B2 企业现场资格化**：用已完成的 B2-ready 接入包连接真实 OIDC/JWKS/PDP、Gateway mint、部署 CA/mTLS 与变更系统；
2. **P1.4-B 现场认证**：把 B-ready 协议接到独立组织 Provider 仓库/CI/实验室、企业 HSM 根和真实 artifact registry，完成激活、证明续期、回滚与 WORM 审计；
3. **P1.5 单厂商真实纵切面**：选择一个真实设备/控制器完成端到端资格测试；
4. **P1.6–P1.7 生产平台化**：HA/DR、密钥、审计、可观测性和 SLO；
5. **P1.8 模型与 L1 优化**：在确定性底座不变的前提下提升交互与泛化能力；
6. 再依据业务需求扩展多厂商、MPLS/EVPN L3VPN、无线和安全场景。

### 6. 当前明确边界

- Containerlab/FRR 是协议和故障行为仿真，不是 Cisco、Huawei、H3C 命令及平台行为认证；
- Service MCP 使用真实 MCP 协议和 Runtime 事务，但当前业务数据是模拟的；
- 本地 SQLite、stdio 和同一 OS 账户不能替代生产进程、身份和故障域隔离；
- Core-72 的 100% 是固定 Oracle 门禁覆盖率，不是生产事故概率为零；
- Runtime 可让任意模型无法越过明确合同，但小模型仍会降低 L1 选 Skill、提参数、追问和编排质量；
- 当前不支持或未认证：EVPN L3VPN、MPLS L2VPN/L3VPN、真实无线 RF、真实状态防火墙/AAA、厂商 CLI 语义、企业 IAM 和生产规模性能。

### 7. 进展维护规则

每次实质迭代都应更新本文件，而不是只在聊天、提交信息或临时报告中记录：

1. 更新顶部“最后核验”和当前里程碑；
2. 只有满足阶段完成判据并有可复现证据时，才把状态改为 ✅；
3. 在新增/修改 Runtime 安全语义后运行以下门禁；
4. 记录一个新的、不同执行代码指纹的 A/B 样本；若为 `regressed`，不得把阶段标为完成；
5. 同步 README 状态、HLD/LLD/SSD/ARCHITECTURE 中受影响的设计事实。

```bash
scripts/netopyu-l0 runtime-validate
scripts/netopyu-l0 runtime-trajectories-validate
scripts/netopyu-dsh compare-runtime --iterations 50 --record --label <milestone>-<iteration>
scripts/netopyu-dsh retirement
```

趋势需要最近 3 个不同执行代码指纹取中位数。当前窗口已达到 3/3，状态为 `stable`。后续每次实质 Runtime 迭代仍应记录新指纹并检查 `improved`、`stable` 或 `regressed`；相同代码的重复运行不算新迭代。

---

## English

### 1. Current conclusion

The migration from the legacy general-purpose L0 agent framework to **DSH/Hermes Harness Adapters plus the NetOpYu Domain Effect Runtime** is complete as a repeatable local reference implementation.

The current milestone is **P1.4-B-ready local completion of the external Provider protocol and deployment-evidence loop**. It adds a repository-external process test, actual restart recovery, independent Publisher/Qualifier/Deployer roles, required artifact digests, strict deployment-aware lifecycle, and schema-v9 evidence. Site certification still requires independently owned systems and organizational trust roots.

This does **not** mean production certification. Enterprise identity and approval, real vendor systems, distributed HA, remote immutable audit, disaster recovery, and production SLOs remain open.

### 2. Phase summary

Legend: ✅ locally complete; 🟡 prototype requiring production qualification; ⬜ planned; 🚫 explicitly unsupported today.

| Phase | Status | Completed scope | Boundary |
|---|---|---|---|
| P0 Harness migration | ✅ | DSH primary path, Hermes Adapter, shared Worker, retirement gate | Harnesses do not own deterministic network semantics |
| P0.5 Domain Effect Runtime | ✅ | Validation, immutable plans, risk, approval binding, revalidation, verification, compensation, audit, recovery | Fixed-oracle local prototype, not a real-world probability guarantee |
| P0.75-A base network lab | ✅ | Containerlab, OSPF, failover and restoration | Linux/FRR behavior only |
| P0.75-B small production-like topology | ✅ | Campus, IDC, DMZ, dual ISP, OSPF/eBGP, topology/path query | 20-node local lab |
| P0.75-C DC fabric | ✅ | BGP EVPN/VXLAN L2VPN | EVPN L3VPN and MPLS L2/L3VPN are not covered |
| P0.8 Service MCP | ✅ | Mock identity, app, policy, change, CMDB and platform providers over MCP | Mock business data |
| P0.9 Observer boundary | ✅ | Identity/version-pinned read-only Network Observer | Local isolation only |
| P1.0 Durable Actor | ✅ | Mutation Actor, journal, idempotency and restart recovery | Single-host reference implementation |
| P1.1 Capability/Saga | ✅ | Capability SPI, read policy, terminal envelope and durable cross-domain Saga | Distributed failure domains remain open |
| P1.2 L0 v2 and readable trajectories | ✅ | 21/21 compiled contracts, runtime bindings, L1/L0.5/L0 evidence and Promotion checks | External publication/signing remains open |
| P1.3 Enterprise identity and approval | 🟡 | P1.3-A signed proof, B1 authority adapters, and B2-ready per-session Gateway minting, CA/mTLS, Doctor/live contract checks | P1.3-B2 real issuer/Gateway/PDP/change integration and non-repudiation remain open |
| P1.4 Provider publication and qualification | 🟡 | P1.4-B-ready external-process 9/9+restart, three roles, artifact/deployment evidence, lifecycle/admission, schema-v9 binding | Independent ownership/CI/lab, HSM roots, real OCI/SBOM/SLSA and WORM remain site work |
| P1.5 Real network/vendor integration | ⬜ | See roadmap | Required before production network writes |
| P1.6 Distributed reliability and HA/DR | ⬜ | See roadmap | Required for multi-instance production |
| P1.7 Security, audit, observability and SLOs | ⬜ | See roadmap | Required before production |
| P1.8 L1/model qualification | 🟡 | A/B1/B2/C1 have full 7B 160-case baselines; C2 adds 24 adversarial cases, zero final safety escape, and 61.96% overall E2E | Protocol is 86.41%, workflow 50%, clarification recall 36.67%; a stronger-model full baseline remains |

### 3. Done

- DSH and Hermes project the same domain capabilities through separate Harness Adapters.
- The Runtime enforces strict parameters and provenance, immutable intent/plan/provider/schema/capability hashes, risk, one-shot subject-bound approval, execution-time revalidation, independent typed postconditions, contractual compensation, terminal states, tamper-evident audit, and durable recovery.
- Durable Sagas coordinate network and service Providers with reverse-order compensation.
- All 21 built-in reviewed mutation capabilities use compiled L0 v2 contracts as runtime authority.
- All 21 keep source-controlled L1 prose, structured-natural-language L0.5, compiled L0, reports, exact round trips, and hash chains.
- Containerlab covers campus/IDC/DMZ/dual-ISP OSPF/eBGP, topology/path queries, and BGP EVPN/VXLAN L2VPN.
- Service capabilities are separated into MCP Providers; Network Observer and Network Actor are distinct boundaries.
- The latest local gate reports 316 tests plus 81 subtests, 21/21 L0 bindings, 21/21 readable trajectories, 21/21 exact round trips, 21/21 Promotion-ready capabilities, and a passing retirement gate.
- Schema-v9 plans bind requester/policy plus Provider release/manifest/qualification/deployment evidence. Signature tampering, replay, identity/release/deployment switching, critical self-approval, missing tickets, and invalid windows fail closed.
- P1.3-B1 verifies human OIDC access tokens and separate Gateway attestations over pinned JWKS, cross-binds them by `act_sub + subject_jti`, applies external PDP decisions to reads/prepares/approvals, and qualifies ticket revision/window/scope/risk through a Change Authority. Credentials remain model-hidden.
- The B2-ready package adds per-session Gateway minting, explicit CA/mTLS with owner-only client keys, an offline secret-safe Doctor, and a no-effect live contract qualification command.
- P1.4-B-ready adds an external JSONL qualification process, actual restart, three independent signature roles, required artifact digests, deployment-aware lifecycle/rollback, Runtime admission, and schema-v9 evidence.
- P1.8-B2 adds an exact DSH Skill plus proposal-only capture path with no Runtime/Provider authority, stable configuration fingerprints, transcript sequence/schema/receipt gates, a complete 160-case 7B failure baseline, and a fully successful one-case `qwen3.6:27b` framework smoke.
- P1.8-C1 adds a digest-bound preloaded L0.5 Skill, five typed proposal-only Tools, trusted Catalog metadata compilation, and a loopback protocol Governor. Its full 7B run improves capture/schema/contract/E2E to 98.75%/80%/68.13%/36.88%, with zero forbidden or duplicate Tool calls, but remains unqualified with 5% safety escape.
- P1.8-C2 adds a versioned safety/domain Guard, typed/candidate Protocol Firewall, complete actual-attempt metering, and 24 adversarial/false-positive cases. The full 184-case 7B run has zero final safety escape and 61.96% E2E, but only 86.41% protocol validity and remains unqualified.
- Core-72 records DSH only at 5/64 controls (7.8%) and DSH + Runtime at 64/64 (100%), while both complete 8/8 valid operations.

Core-72 deliberately fixes L1 decisions. It measures the deterministic Runtime increment, not LLM intent recognition, Skill selection, clarification, or parameter extraction. The P1.4-B-ready three-fingerprint trend is `stable`: Runtime remains 64/64, with a local median p50/p95 of 7.599/8.680 ms; the P1.8-B1 follow-up measured 7.281 ms p50. This is not a production SLO.

### 4. To-do roadmap

#### P1.3 Enterprise identity and approval control plane

P1.3-A and B1 are complete locally. The B2-ready package now adds per-Harness-session Gateway minting, shared explicit CA/mTLS transport, an offline secret-safe Doctor, and a no-effect live authority contract test. P1.3-B2 must run this package against the user's real issuer, Gateway, organizational RBAC/ABAC/PDP, approval/change system, MCP service identities, certificate/key rotation and external non-repudiation controls.

#### P1.4 Provider publication and qualification

P1.4-B-ready now supplies the external-process protocol, real restart, three-role release/deployment evidence, lifecycle, rollback, and admission reference. P1.4-B site work must use independently owned Provider/CI/labs, organizational HSM roots, real OCI/SBOM/SLSA services, and remote immutable lifecycle evidence. A genuine forward L1 → L0.5 → L0 benchmark also remains open.

#### P1.5 Real network and vendor qualification

Qualify at least one real vertical slice using a vendor lab/device/controller and Netmiko, NAPALM, NETCONF, RESTCONF, or a controller API. Add command allowlists, snapshots, commit-confirm where available, independent readback, rollback drills, and per-platform/version qualification matrices. Expand EVPN L3VPN, MPLS L2/L3VPN, firewall, AAA/802.1X, and wireless scenarios only when suitable labs exist.

#### P1.6 Distributed reliability, HA, and DR

Replace single-host durability with a remote transaction store and queue, distributed fencing/leases, idempotency/CAS across Actors, separated Observer/Actor failure domains, backup/restore, failover, replay, and partition/crash/storage-failure qualification against defined RPO/RTO.

#### P1.7 Security, audit, observability, and SLOs

Add mTLS, centralized secrets, encryption, egress policy, external append-only/WORM audit, metrics/traces/alerts/evidence bundles, SLI/SLO definitions, scale and chaos tests, and controlled model/Skill/contract/Provider canary and rollback.

#### P1.8 L1 and model qualification

P1.8-A, B1, B2, and C1 provide complete, fingerprint-bound 160-case 7B baselines. C2 adds a complete 184-case run: the comparable base reaches 58.75% E2E and zero final safety escape, while the 24-case adversarial/false-positive extension reaches 83.33%. Raw first-attempt safety escape remains 9.38%, the deterministic Guard performs the closure, protocol validity is 86.41%, workflow is 50%, and clarification recall is 36.67%. All 267 real attempts, 34 repairs, and 121 invalid-contract attempts are metered. A practical stronger-model 184-case baseline plus broader stale/cross-domain/long-conversation work remain. Model quality and Guard rules must never replace Runtime safety.

#### Optional P2 enhancements

Add an L0.5 review UI, contract diff/graph and Promotion console, multi-team/tenant catalogs, a richer evidence plane, business-driven L1-to-L0 maturation, and broader network/service workflows.

### 5. Recommended sequence

1. Close P1.3-B2 real-enterprise qualification using the completed B2-ready package.
2. Close P1.4-B by applying the B-ready protocol to an independently owned Provider/CI/lab with organizational signing/HSM, real artifact verification, activation, attestation renewal, rollback, and WORM evidence.
3. Qualify one P1.5 real vendor vertical slice end to end.
4. Build P1.6–P1.7 HA/DR, security, audit, observability, and SLO foundations.
5. Improve P1.8 models and L1 behavior without changing the deterministic safety root.
6. Expand vendors and advanced network scenarios based on actual business demand.

### 6. Maintenance rule

Update this file after every substantive iteration. A phase becomes ✅ only after its completion criteria have reproducible evidence. Run the following gates after Runtime safety changes and record one new benchmark sample with a unique execution-code fingerprint:

```bash
scripts/netopyu-l0 runtime-validate
scripts/netopyu-l0 runtime-trajectories-validate
scripts/netopyu-dsh compare-runtime --iterations 50 --record --label <milestone>-<iteration>
scripts/netopyu-dsh retirement
```

Trend status uses the median of three unique execution-code fingerprints. The current 3/3 window is `stable`. Record every later substantive Runtime iteration and review `improved`, `stable`, or `regressed`; repeated runs of identical code do not count.
