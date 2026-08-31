# 项目进展与路线图 / Project Status and Roadmap

> 最后核验 / Last verified: 2026-08-31
> 当前里程碑 / Current milestone: **P2.5 公开 210 条真实 9B 基线与 phase-typed Capability 门禁完成；当前 Runtime 重放保留 203/203 条 exact-ready，并阻断 1 条历史 false-ready，私有独立资格仍开放 / P2.5 public 210-case real-9B baseline and phase-typed Capability gate complete; current-Runtime replay preserves 203/203 exact-ready proposals and closes one historical false-ready; independent private qualification remains open**

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
L1 Decision Plane：候选收口、grounding、Guard、proposal-only 证据
    ↓
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
- **L1 → L0.5 v3 → L0 可解释轨迹已补齐**：21/21 存量能力均有源码化、可校验、可逆向还原的三阶段材料；
- **P1.3-A 本地可信审批链已完成**：schema-v7 计划绑定 requester/policy，DSH/Hermes 人工确认后由 Runtime 签发短时签名 proof；
- **P1.3-B1 企业控制面参考 Adapter 已完成**：OIDC/JWKS access token、Gateway sender attestation、HTTP PDP 和 Change Authority 通过本地真实 HTTP 资格测试，read/write 均 fail closed；
- **P1.3-B2-ready 企业接入包已完成**：按 Harness session 动态 mint Gateway attestation、显式 CA/mTLS transport、无泄密 Doctor 和无网络效果 live contract check 均通过本地资格测试；
- **P1.4-B-ready Provider 供应链已完成**：仓库外 JSONL 独立进程、真实重启、Publisher/Qualifier/Deployer 三角色、OCI/SBOM/provenance 必需 digest、部署证明、严格 promote/rollback 与 schema-v9 binding 形成本地闭环；
- **P1.9-B1 本地 shadow/证据框架已完成**：DSH/Hermes 共用严格 `proposal_only` Decision，跨轮关闭、隐私化首次路由/usage 证据、三 profile Catalog drift 和仓库外 holdout/双 reviewer 合同均已落地；默认关闭，不改变现有执行；
- **P1.9-B2 本地资格执行器已完成**：同一密封集/模型/策略分别以 DSH/Hermes 身份执行，独立计算 Oracle、重复稳定性、语义 parity、token 和 p50/p95，报告不输出 Prompt、逐条标签或参数值；真实仓库外数据与人工证据仍待取得；
- **P1.9-B2 Adapter Hook parity 已完成**：生产 DSH JavaScript `agent/pre-step` 与 Hermes Python `pre_llm_call` 通过临时 owner-only Worker 实跑并比较完整输入/Decision 摘要；完整 DSH Web/Hermes CLI/UI 与部署身份仍待产品环境认证；
- **P1.9-C0 绑定内核已完成、未启用**：PreparedPlan schema v10 可选绑定 proposal-only Decision、Harness route、请求/编译参数和 L0 合同；Journal 对 Decision id 建立唯一约束，错会话/目标/参数/摘要、重放和计划篡改均失败关闭；DSH/Hermes 配置边界继续拒绝 `canary`；
- **P1.9-C1 本地安全准备层已完成、未启用**：纯策略只能保持原 Harness route 或收窄/阻断，写路径异常失败关闭；readiness 门禁交叉绑定 Worker/Adapter/产品/运维四类外部证据，最强结论仅为 `ready_for_review`；停用、告警、回退和审计手册已落地；
- **P2.0 本地 Promotion Workbench 已完成**：不可变 proposal 校验、需求级 L1/L0.5/L0 语义覆盖/偏移门禁、轨迹/合同图、隐私最小化 review 投影和离线草稿编辑已落地；缺失/削弱/关键歧义会定位并阻断，没有批准、注册、激活或执行 API；
- **P2.1 本地 Capability Catalog 已完成**：21/21 已激活 L0 合同进入摘要绑定的 owner/steward、租户/环境、委派、依赖、消费者与兼容治理；Catalog 决策没有 Runtime read/write 或 Provider 发布权威；
- **P2.2 本地 Evidence Plane 已完成**：五类现有证据源以只读方式形成隐私最小化的统一事件链、指标、事故与离线时间线；缺链、截断或完整性失败明确降级；
- **P2.3 用户接入与收敛评测已完成**：统一 CLI、三条 Golden Path、只读 Doctor、严格 Integration Pack、源码化脱敏基线和逐层失败驾驶舱已落地；明确区分固定集控制、模型资格和未证明的生产泛化；
- **P2.4 真实 LLM Agent Golden Paths 已完成**：DSH 页面已用 `qwen3.5:9b` 实跑 Runtime L1→L0、L1→L0.5→L0 proposal 和四系统 MCP 访问三个用例；新增统一发现入口、样例 Skill、独立 MCP 演示配置和逐阶段可见证据；
- **P2.5 核心 A 公开 210 条 9B 鲁棒性基线与 phase gate 已完成**：L0.5 v3 在 21/21 轨迹保存 capability-scoped exact-intent 与三类 Observation phase；同一 `qwen3.5:9b` 历史实跑的原始/受限规范化后协议为 76.19%/99.52%、全语义 exact 96.67%、历史 Runtime 可审 204/210、safety escape 0，p50/p95 27.934/38.557 秒；0 次模型调用的当前 Runtime 重放保留 203/203 条 exact-ready，新增阻断 1 条历史错误 phase false-ready，exact-ready 回归 0；
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
| P1.8 L1/模型资格评测 | ✅ | C3.2 完成同一 7B 的 184/184：协议门禁 100%、最终 safety escape 0、E2E 91.30%，通过当前本地资格门槛 | 固定集通过不等于生产概率；跨域冲突、陈旧状态、长对话和多模型持续回归仍需扩展 |
| P1.9 L1 Decision Plane | 🟡 | B1/B2 shadow 与资格执行器；C0 plan binding；C1 单调策略、证据门禁和 runbook | 真实私有集/人工真值、完整 Harness 产品/部署证据、canary 激活/enforced 和生产 SLO 尚未完成 |
| P2.0 Promotion Workbench | ✅ | proposal 完整性/交叉绑定、需求级语义覆盖与偏移定位、L1→L0 图、Runtime 合同图、离线 L0.5 草稿编辑 | 单机静态审查工具；不批准、不注册、不激活，尚非多人生产控制台 |
| P2.1 多团队 Capability Catalog | ✅ | 21/21 L0 精确覆盖；namespace、owner/steward、tenant/environment、委派、依赖、消费者与兼容性 | 本地治理投影；不授权 Runtime read/write，不替代企业 IAM/PDP 或独立发布系统 |
| P2.2 Evidence Plane | ✅ | Runtime/Decision/Saga/Provider/Promotion 五类只读 adapter、统一事件/指标/事故和离线 HTML | 本地只读投影；无远端 WORM、告警/SLO 或跨实例 trace |
| P2.3 用户接入与收敛评测 | ✅ | Golden Path、Doctor、能力发现、proposal-only Integration Pack、Runtime/L1 统一驾驶舱与 368 条脱敏 trace | 固定集证据；没有 Provider 激活权威，也没有证明生产泛化 |
| P2.4 真实 LLM Agent 用例 | ✅ | 9B 模型实际选择 L1/调用 L0；Agent 化 Promotion；身份/应用/变更/权限四 MCP 进程联动 | 本地模拟数据与单次 UI 证据；不是模型泛化率、生产 SLO 或外部系统认证 |
| P2.5 核心 A 正向资格协议 | ✅ | 210 条/21 能力族/10 包装真实 9B 运行、L0.5 v3 exact-intent、phase-typed Observation、受限 enum 规范化、切片指标与可恢复 checkpoint | 当前门禁关闭已知错误 phase false-ready；公开反向数据与单次运行仍不能资格化模型 |

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

- 当前 PreparedPlan schema v10 继承 requester/policy 和 Provider release/manifest/qualification/deployment evidence，并加入可选 L1 Decision provenance；
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
- 当前 PreparedPlan schema v10 继承 v9 的 release、manifest、qualification 和 deployment digest 审批绑定，审批后同 release 重部署仍会在 Provider 调用前失败关闭；
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
| Python gate | 420 tests + 81 subtests | Runtime、Adapter、Provider、Skill、身份控制面、外部资格/三角色部署证明、审批证明、P1.8、P1.9、P2.0–P2.5、恢复等 |
| Core-72：DSH only | 5/64（7.8%） | 固定风险/故障 Oracle |
| Core-72：DSH + Runtime | 64/64（100%） | 固定风险/故障 Oracle |
| 有效操作 | 8/8 vs 8/8 | 两条路径都能完成无故障请求 |
| 内置 L0 v2 绑定 | 21/21 | 编译合同是运行时权威来源 |
| 可读轨迹 | 21/21 | L1、L0.5、L0、报告和哈希链齐全 |
| 精确 round-trip | 21/21 | 可读材料精确恢复运行时合同 |
| Promotion-ready | 21/21 | 内置能力的离线资格门禁通过 |
| Retirement gate | PASS | Python、Node、静态边界、HITL/A2A、投影、检索、负载/可靠性 |

Core-72 固定了 L1 决策，因此只量化 Runtime 的确定性增量；它**不测**模型的意图识别、Skill 选择、追问质量或参数提取准确率。P1.4-B-ready 后最近 3 个不同执行代码指纹趋势为 `stable`：Runtime 仍为 64/64，三版本本机中位 p50 7.599 ms、p95 8.680 ms；P1.8-B1 后复核的 50 样本 p50 为 7.281 ms。单次机器时延不是生产 SLO。

#### 3.6 P1.8-A/B1/B2/C1/C2/C3 L1/模型资格层

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
- P1.8-C3 为本次 top-12 的每个候选动态生成独立无效果 Tool；Tool 身份固定 kind/target，候选专属 Schema 固定允许参数键，模型只保留候选选择与显式值提取；确定性 compiler 派生 action、missing fields 和 workflow；
- 版本化 argument grounding 只接受请求中存在的证据，删除无来源值并执行受审别名归一化；Schema 外字段可被删除，但网关不得改变模型选择的候选或补值。Guard 仍是只收窄层，整个 C3 路径无 Runtime/Provider/设备/审批权限；
- C3.2 同一 immutable 7B 完成 184/184 并通过当前门槛：协议门禁 100%、selection 94.12%、parameter F1 93.06%、clarification precision/recall 93.55%/96.67%、missing fields 93.33%、workflow 90.62%、E2E 91.30%、新增 24 条 E2E 100%、最终 safety escape 0；
- C3.2 共 193 次模型调用、9 次修复；grounding 删除 40 个无来源字段，Schema 删除 24 个越界字段。模型首轮 safety escape 3.12%，最终 0 仍来自确定性 Guard。p50/p95 为 4.488/6.850 秒；版本化证据为 `data/l1_dsh_schema_compiler_observations.json`。
- 同口径 immutable `qwen3.6:27b` 也完成 184/184：E2E/selection/parameter F1 提升到 95.11%/96.08%/95.83%，最终 safety escape 0；但两例超过 300 秒外层协议时限，协议门禁仅 98.91%，p50/p95 为 68.193/176.923 秒，因此未通过资格门禁，也未写入合格 observation 基线。失败证据保存在 `artifacts/l1-dsh-schema-compiler/qwen3.6-27b/`；

#### 3.7 P1.9-B1/B2 正式 L1 Decision Plane 与资格执行器

- 新增独立生产包 `l1_runtime/`，不导入 `evaluation/` 场景、Oracle 或期望标签；
- DSH accepted `agent/pre-step` 与 Hermes 官方 pre-LLM/Tool/post-LLM/session hooks 都只把 direct-user 文本和当前正式 Tool/Skill surface 交给同一 Decision Plane；
- top-12 候选、候选专属 Schema、grounding、Guard 和编译器生成严格 `L1DecisionEnvelope`，`authority` 固定为 `proposal_only`；
- 默认 `off`；`shadow` 的模型或协议失败只形成脱敏错误，不替换、不拒绝原 DSH step；
- pending Decision 最多绑定一次领域路由；superseded/no-route/session-end 显式关闭，关闭后不能跨轮重绑；
- Decision store 不保存 Prompt、模型正文或实际参数值，只保存 digest、字段名、token usage 完整性、生命周期、首次路由对比和时延；旧 schema 自动迁移并清除参数原值；
- 三 profile Catalog/Tool/Skill semantic baseline 进入退休门禁；仓库外 holdout seal 要求 50+/10 类/三 profile/中英覆盖，双 reviewer 任一语义分歧均不产生 consensus digest；
- `l1_runtime.qualification` 对完全一致的私有真值执行两次以上 DSH/Hermes 独立同模型调用，分别测协议、选择、参数、追问字段、workflow、安全拒绝、候选召回、repeatability、semantic parity、token 和 p50/p95；
- `l1_runtime.adapter_qualification` 启动临时持久 Worker，以 stdin 向 DSH driver 传递私有请求，并分别触发生产 DSH/Hermes Hook；只输出 Prompt/Catalog/Candidate/Policy/Decision digest parity；
- `qualified` 是全量绝对门槛：Catalog clean、不可变模型 artifact、五类 action 覆盖、输入合同 parity、双端语义 parity、协议/完整 Oracle/候选召回 100%、安全逃逸 0；scope 仅为 shared Worker Decision contract；
- Python 契约、隐私迁移、lifecycle、Catalog/holdout、Hermes hook 及 Node → Python → fake OpenAI → DSH 路由 smoke 均已通过；
- 当前没有真实流量或人工真值基线，因此不把 smoke 的一致性写成总体准确率。

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
- **已完成 P2.5 正向转换协议、exact-intent 表达、受限 enum 规范化和 21 族 9B 复测**；下一步用仓库外私有集、双人真值和至少三次重复 Observation 取得资格证据；
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
- **P1.8-C3 已完成**：候选专属 Schema、候选身份绑定、显式参数 grounding、确定性 action/missing-field/workflow 编译和全调用计量；同一 7B 的 184/184 当前资格门槛全部通过；
- 对 27B 和云模型分别运行完整准确率、拒绝率、token/成本和时延基线；
- 已加入首批提示覆盖、Unicode 混淆、过期授权、命令注入和安全术语反误杀；仍需扩展跨域冲突、状态陈旧和长对话；
- 继续扩展未见分布、跨轮对话和 Catalog 漂移回归；不得通过 Oracle 特判、放宽合同或让 Guard 选择 Capability 来提高成绩；
- L1 输出只能进入 L0 严格解析/校验，任何模型都不能成为审批、验证或回滚的权威；
- 对成熟 L1 建立可审计的 L0.5 候选生成，但仍需机器门禁和人工 Promotion。

当前本地核心判据已满足：C3.2 让同一可接受成本的 immutable 7B 完成全部 184 条，并同时通过协议、语义和最终安全绝对门槛，P1.8 本地范围标记完成。该结论只限定于摘要绑定的模型、DSH、Catalog、政策和固定数据集；生产前仍需持续多模型/未见分布测试，且任何候选执行都必须进入 L0 Runtime。

#### P1.9 L1 Decision Plane

目标：把 P1.8 的收口原则变成 Harness 共用、可观测、可灰度、仍无效果权限的正式决策入口。

- **P1.9-A 已完成**：DSH shadow、中央 Decision/Observation store、协议/一致性/escape/时延指标和默认关闭开关；
- **P1.9-B1 已完成**：Hermes 共用入口、跨轮 Decision 生命周期、Catalog 漂移门禁、usage 完整性以及私有 holdout/双 reviewer 工具合同；
- **P1.9-B2 执行器已完成、证据待完成**：准备真实仓库外未见集，由两名独立人员完成真值并解决分歧；运行 Worker Oracle 与 Adapter Hook 两个资格命令，并在目标环境补充完整 Harness 产品/部署证据；
- **P1.9-C0 已完成、未启用**：schema v10 将 Decision/evidence digest、Harness route、请求/编译参数与 L0 contract 绑定进 plan hash；单 Decision 只能创建一份计划，v9/v8 仍可只读，Harness 启动边界仍拒绝 canary；
- **P1.9-C1 安全准备已完成、激活待证据**：已实现只保留/收窄的纯策略、四类证据交叉绑定、有效期/演练门禁、隐私最小化 readiness 报告和启停/告警/回退/审计 runbook；当前没有真实 B2/产品证据，Adapter 仍拒绝 canary；
- **P1.9-D 待完成**：在足够 shadow/canary 证据后才评估 enforced；它仍不能授予 L0 写权限。

升级判据和指标边界见 [P1.9 L1 决策面](l1-decision-plane.md)。DSH 路由一致率不是人工真值，也不能用于证明业务正确率。

#### P2.0 Promotion Workbench（本地完成）

- 校验不可变 package、逐文件摘要、四阶段前驱链、report-to-stage 和 review 权威边界；
- 以三栏联动视图展示 L1→L0.5→L0 语义关系、可复算映射置信度、语言丢失、精确修复路径、轨迹和 Runtime/Composite 依赖图；
- 导出带 CSP、无外部网络依赖的本地 HTML，并只允许下载不可信 L0.5 草稿；
- `approve` 后仍为 `approved_not_active`；页面没有批准、注册、激活、Runtime 或 Provider API；
- 设计、命令和边界见 [P2.0 Promotion Workbench](p20-promotion-workbench.md)。

#### P2.1 多团队/多租户 Capability Catalog（本地完成）

- 源码化 Catalog 精确绑定 21/21 已激活 L0 合同、schema 摘要和 profile；
- namespace、owner/steward、tenant/environment、lifecycle、consumer、dependency 与 supersedes 均为严格 schema；
- read consumer binding 与 write proposal、review、publish、deprecate 委派显式分离；自委派、review+publish、非 owner 委派、scope 扩张、依赖漂移/环均失败关闭；
- compatibility 报告识别原地合同/依赖/范围变化、删除、生命周期回退和消费者影响；只分析，不注册或激活；
- 完整命令与边界见 [P2.1/P2.2 控制面](p21-p22-control-planes.md)。

#### P2.2 Evidence Plane 与运营分析（本地完成）

- Runtime Journal、L1 Decision、Saga、Provider Release 与 Promotion package 通过五类只读 adapter 投影为统一摘要链；
- 输出终态/成功/回滚/人工介入、选择/参数/safety、Saga、发布和 Promotion 指标、失败聚类、漂移信号、跨快照趋势以及按严重度排序的事故时间线；
- SQLite 使用只读 URI 和 `query_only`；快照不含 Prompt、参数值、审批身份、Provider payload 或文件路径；
- 缺少事件链、截断或篡改时状态为 `degraded` 且 CLI 非零；页面无审批、执行、注册或激活入口；
- 生产 WORM、组织审计、告警和 SLO 仍依赖 P1.7；完整边界见 [P2.1/P2.2 控制面](p21-p22-control-planes.md)。

#### P2.3 用户接入与收敛评测（本地完成）

- `scripts/netopyu` 统一 understand/demo/integrate 三条 Golden Path、只读 Doctor 和能力发现；
- Integration Pack 强制 read/write、独立 verifier、补偿、凭据引用和 proposal-only 权威；
- 源码化基线合并 Core-72 和两个 184 条模型报告，保存 368 条无 Prompt/无参数值 trace，并为每个失败给出唯一首层；
- 驾驶舱 self-contained、只读、无外部请求和控制接口；真实私有 holdout、现场集成与生产泛化仍未完成；
- 使用与接入见 [使用与系统接入](getting-started-integration.md)，指标边界见 [LLM 收敛评测](convergence-evaluation.md)。

#### P2.5 核心 A 正向资格协议（协议与公开单次模型证据完成，私有资格开放）

- `ForwardCase` 与 `ForwardLabel` 分离；正式用例必须来自仓库外、至少 200 条、10 个能力族、LAN/DC/WAN、中英文和 5 类挑战；
- manifest 只保存覆盖统计和 digest，不包含 Prompt/Label；两名不同 reviewer 必须对密封全集完全一致；
- 同一报告只接受一个模型名和一个不可变 artifact digest，每条 case 重复次数一致且资格要求至少三次；
- 聚合测量协议完成、disposition、Capability、参数/谓词、安全合同、全语义 exact match、歧义阻断、合法 proposal yield、重复稳定性、模型调用/修复和 p50/p95；
- 风险/审批弱化、删除 preflight/独立验证/补偿或不安全处理未知写结果均计为 safety escape，资格门槛为 0；
- 210 条公开矩阵来自 21 个受审 L0 的反向轨迹，只能校准 evaluator 和多能力覆盖，明确 `qualificationEligible=false`；
- 长模型评测已支持逐 Case 原子 checkpoint 和 `--resume`；恢复前强制校验模型制品、协议、Catalog、Case/Reviewer、重复次数与 repair policy 指纹，避免中断丢失或跨运行混证；
- 真实 Agent proposal 可通过 `forward-eval-record` 从 Runtime 权威制品投影为无 Prompt Observation；完整协议见 [正向资格报告](promotion-forward-qualification.md)。

### 5. 推荐实施顺序

建议下一步不要同时横向增加大量协议或 Skill，而按一个真实纵切面推进：

1. **本地持续回归 P2.1/P2.2/P2.3**：Catalog 变更通过兼容/消费者影响门禁，Evidence adapter 随 schema 演进，Integration Pack 与驾驶舱合同保持稳定；
2. **phase-typed Capability 已收口，继续完成 P2.5 外部资格证据**：Catalog v2、L0.5 v3、模型物化边界与 Promotion 已共同限定 preflight/success-verification/compensation-verification；210 条历史 proposal 重放关闭 1 条已知 false-ready 且无 exact-ready 回归。下一步独立编写并密封 200+ 正向用例，完成双 reviewer 仲裁和同一 9B 制品至少三次运行，结构稳定后再评估 27B；
3. **外部条件具备后：P1.9-B2 人工/产品证据闭环**，使用真实 sealed holdout、双人真值和完整 Harness 产品实跑；
4. **P1.3-B2/P1.4-B/P1.5 现场资格化**，依次接入企业身份、独立供应链和真实厂商纵切面；
5. **P1.6–P1.7 生产平台化**，把本地 Catalog/Evidence 接入 HA、密钥、远端不可变审计、可观测性和 SLO；只依据真实证据推进 P1.9 canary。

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

The current milestone is **completion of the real qwen3.5:9b public robustness baseline across 210 cases, 21 families and ten bilingual/trace/safety/schema/adversarial wrappers**. It measures 76.19% raw and 99.52% bounded-normalized protocol completion, 99.05% capability exact, 96.67% parameter/full-semantic exact, 97.14% Runtime review readiness, zero safety escape, and 27.934/38.557-second p50/p95. Five failover predicates and one self-contradictory protocol fail closed; one structurally reviewable proposal still selects the wrong preflight phase capability. Public reverse data and one repetition remain deliberately ineligible for qualification.

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
| P1.8 L1/model qualification | ✅ | C3.2 completes 184/184 with the same 7B: 100% protocol gates, zero final safety escape, 91.30% E2E, and passes the current local gates | Fixed-set qualification is not a production probability; unseen, stale-state, cross-domain, long-context, and multi-model regression remain |
| P1.9 L1 Decision Plane | 🟡 | B1/B2 shadow and qualification, C0 plan binding, plus C1 monotonic policy, evidence gate, and runbook | Real private/adjudicated and full Harness product/deployment evidence, canary activation/enforced behavior, and production SLOs remain open |
| P2.0 Promotion Workbench | ✅ | Package integrity/binding, requirement-level semantic coverage/drift localization, lineage and Runtime contract graphs, offline L0.5 draft editing | Single-host static review tool; no approval, registration, or activation |
| P2.1 Multi-team Capability Catalog | ✅ | Exact 21/21 L0 coverage, ownership/stewardship, scope, delegation, dependencies, consumers, and compatibility | Local governance projection; no Runtime read/write or publication authority |
| P2.2 Evidence Plane | ✅ | Five read-only adapters, unified events/metrics/incidents, and offline HTML | Local projection; remote WORM, alerts/SLOs, and cross-instance tracing remain external |
| P2.3 Product entry and convergence | ✅ | Golden Paths, Doctor, capability discovery, proposal-only Integration Pack, unified Runtime/L1 cockpit, and 368 redacted traces | Fixed-set evidence only; no Provider activation authority and no proven production generalization |
| P2.4 Real-LLM Agent use cases | ✅ | Actual 9B L1/L0 selection and execution, agent-assisted Promotion, and four independent MCP service processes | Simulated local data and single-run UI evidence; not model generalization, production SLO, or external-system certification |
| P2.5 Core-A forward qualification | ✅ | Real 210-case/21-family/10-wrapper 9B run, L0.5 v3 exact-intent, phase-typed Observations, bounded enum normalization, slice metrics, and resumable checkpoints | The current gate closes the known phase false-ready; public reverse data and one repetition still do not qualify the model |

### 3. Done

- DSH and Hermes project the same domain capabilities through separate Harness Adapters.
- The Runtime enforces strict parameters and provenance, immutable intent/plan/provider/schema/capability hashes, risk, one-shot subject-bound approval, execution-time revalidation, independent typed postconditions, contractual compensation, terminal states, tamper-evident audit, and durable recovery.
- Durable Sagas coordinate network and service Providers with reverse-order compensation.
- All 21 built-in reviewed mutation capabilities use compiled L0 v2 contracts as runtime authority.
- All 21 keep source-controlled L1 prose, structured-natural-language L0.5, compiled L0, reports, exact round trips, and hash chains.
- Containerlab covers campus/IDC/DMZ/dual-ISP OSPF/eBGP, topology/path queries, and BGP EVPN/VXLAN L2VPN.
- Service capabilities are separated into MCP Providers; Network Observer and Network Actor are distinct boundaries.
- The latest gate reports 420 tests plus 81 subtests; retirement remains 7/7 with 21/21 L0 bindings, readable trajectories, exact round trips, and Promotion/Catalog coverage.
- P2.0 adds a fail-closed, read-only Promotion projection and self-contained offline editor. Edited documents remain untrusted drafts, and even approved reviews remain inactive.
- P2.1 adds a digest-bound governed Catalog with separated owners/stewards, scoped delegation, exact dependencies, consumer impact, and compatibility analysis. Its decisions cannot authorize Runtime reads/effects or Provider publication.
- P2.2 adds read-only Runtime/Decision/Saga/Provider/Promotion adapters, privacy-minimized digest chains, operational metrics, incidents, and a self-contained no-control timeline. Unverifiable, truncated, or invalid evidence is degraded.
- P2.3 adds one product CLI, three Golden Paths, a read-only Doctor, strict proposal-only Integration Packs, and a digest-bound cockpit over Runtime A/B plus 368 redacted L1 traces. It cannot connect or activate a Provider and always marks production generalization as unproven.
- P2.4 adds three real-LLM DSH journeys, a user-authored sample Skill, proposal-only authoring Tools with visible lineage, and a service-only configuration that talks to four independent MCP subprocesses. The records remain simulated and the single-run evidence is not a production metric.
- P2.5 adds strict Case/Label/Observation/Manifest/Adjudication/Report contracts, a 210-case public calibration matrix, L0.5 v3 exact-intent anchors, phase-typed Observations, bounded/audited enum normalization, slice metrics, resumable digest-bound checkpoints, private-set sealing, two-reviewer consensus, repeated artifact scoring, and fixed exact-match/safety/latency gates. The historical qwen3.5:9b run across 21 families and ten wrappers measures 76.19% raw and 99.52% normalized-boundary protocol completion, 99.05% capability exact, 96.67% parameter/full-semantic exact, 97.14% historical Runtime readiness, zero safety escape, and 27.934/38.557-second p50/p95. A no-model-call current-Runtime replay preserved 203/203 exact-ready proposals and closed one known phase false-ready with zero exact-ready regression. It remains a diagnostic public single-run baseline, not qualification.
- Long-running model evaluation now writes one atomic, digest-bound checkpoint per case and supports strict `--resume`; model, protocol, catalogs, cases/reviewers, repetitions, and repair-policy mismatches fail closed instead of mixing evidence.
- Schema-v9 plans bind requester/policy plus Provider release/manifest/qualification/deployment evidence. Signature tampering, replay, identity/release/deployment switching, critical self-approval, missing tickets, and invalid windows fail closed.
- P1.3-B1 verifies human OIDC access tokens and separate Gateway attestations over pinned JWKS, cross-binds them by `act_sub + subject_jti`, applies external PDP decisions to reads/prepares/approvals, and qualifies ticket revision/window/scope/risk through a Change Authority. Credentials remain model-hidden.
- The B2-ready package adds per-session Gateway minting, explicit CA/mTLS with owner-only client keys, an offline secret-safe Doctor, and a no-effect live contract qualification command.
- P1.4-B-ready adds an external JSONL qualification process, actual restart, three independent signature roles, required artifact digests, deployment-aware lifecycle/rollback, Runtime admission, and schema-v9 evidence.
- P1.8-B2 adds an exact DSH Skill plus proposal-only capture path with no Runtime/Provider authority, stable configuration fingerprints, transcript sequence/schema/receipt gates, a complete 160-case 7B failure baseline, and a fully successful one-case `qwen3.6:27b` framework smoke.
- P1.8-C1 adds a digest-bound preloaded L0.5 Skill, five typed proposal-only Tools, trusted Catalog metadata compilation, and a loopback protocol Governor. Its full 7B run improves capture/schema/contract/E2E to 98.75%/80%/68.13%/36.88%, with zero forbidden or duplicate Tool calls, but remains unqualified with 5% safety escape.
- P1.8-C2 adds a versioned safety/domain Guard, typed/candidate Protocol Firewall, complete actual-attempt metering, and 24 adversarial/false-positive cases. The full 184-case 7B run has zero final safety escape and 61.96% E2E, but only 86.41% protocol validity and remains unqualified.
- P1.8-C3 gives each retrieved candidate a distinct proposal-only Tool whose identity fixes kind/target and whose Schema fixes allowed business keys. A versioned grounding policy removes unsupported values, and a deterministic compiler derives action, missing fields, and workflow without selecting for the model or granting authority.
- C3.2 completes all 184 cases and passes the current local gates: all protocol gates 100%, selection 94.12%, argument F1 93.06%, clarification precision/recall 93.55%/96.67%, workflow 90.62%, E2E 91.30%, adversarial E2E 100%, and final safety escape zero. There were 193 model calls and nine repairs; grounding removed 40 unsupported fields and Schema constraining removed 24 unknown fields. First-attempt safety escape remains 3.12%, so the Guard and L0 boundary remain essential.
- The same immutable `qwen3.6:27b` completed 184/184 with 95.11% E2E, 96.08% selection, 95.83% argument F1, and zero final safety escape. Two cases nevertheless exceeded the 300-second outer protocol deadline, leaving protocol completeness at 98.91%; local p50/p95 was 68.193/176.923 seconds. It therefore failed qualification and was not promoted into the qualified observation baseline; the failure evidence remains under `artifacts/l1-dsh-schema-compiler/qwen3.6-27b/`.
- P1.9-B1 adds opt-in DSH/Hermes shadow over `l1_runtime/`: direct-user capture, current Tool/Skill Catalog binding, candidate-specific Schema, grounding/Guard/compiler, turn closure, immutable proposal-only evidence, reported-token completeness, first-route correlation, privacy-migrated SQLite, three-profile drift gates, and repository-external holdout/two-reviewer contracts. It is off by default and never changes the original Harness behavior.
- P1.9-C1 adds a side-effect-free monotonic policy that may preserve or block the existing Harness route but never rewrite or authorize it, plus a strict four-document Worker/Adapter/product/operations readiness gate and bilingual stop/rollback runbook. Its strongest result is `ready_for_review`; both adapters still reject canary.
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

P1.8-C3.2 completes the current local qualification scope with the same immutable 7B artifact. All 184 cases pass the protocol, semantic, and final-safety gates; overall E2E is 91.30% and the adversarial extension is 100%. Candidate identity, argument-key bounds, grounding, missing fields, and workflow are deterministic, while the model retains semantic selection and explicit-value extraction. The record is fingerprint-bound and fully metered. Broader stale-state, cross-domain-conflict, long-conversation, Catalog-drift, unseen-distribution, and alternative-model runs remain continuous hardening, not prerequisites for the now-complete local phase. Model qualification and Guard results never replace Runtime safety.

#### P1.9 L1 Decision Plane

P1.9-B1 is complete as a shared DSH/Hermes local evidence framework. B2 now has both a shared-Worker Oracle runner and a production adapter-hook runner. The latter executes DSH JavaScript `agent/pre-step` and Hermes Python `pre_llm_call` through a temporary owner-only Worker and compares privacy-minimized input/full-Decision digests. B2 evidence still requires real repository-external cases, actual independent human labels, resolved disagreements, and complete DSH Web/Hermes CLI/UI/deployment certification. P1.9-C may canary only visibility narrowing and clarification after Decision-to-plan binding, fail-closed rollback, and unchanged Runtime gates. Enforced behavior is considered only after enough shadow/canary evidence and never grants L0 authority.

P1.9-C0 implements the disabled Decision-to-plan kernel. C1 now adds monotonic route handling, strict cross-bound and expiring external evidence, privacy-minimized `not_ready|ready_for_review` reporting, and the activation/rollback runbook. Neither layer exposes an activation command or changes configuration/traffic. Both Harness adapters still reject `canary`; real B2 human truth, real DSH Web/Hermes CLI/deployment evidence, organization identity/non-repudiation, and an independently approved release change remain mandatory.

#### P2.0 Promotion Workbench (locally complete)

P2.0 validates immutable package files, predecessor hashes, report-to-stage bindings, compiled identity/hash, and review authority flags. Its self-contained page links each L1/L0.5/L0 requirement across three lanes and exposes deterministic confidence components, language-loss alerts, exact repair paths, semantic diffs, and actual contract dependencies. The editor exports only an untrusted draft; the page has no approve, register, activate, Runtime, or Provider API. See [P2.0 Promotion Workbench](p20-promotion-workbench.md).

#### P2.1 Multi-team/tenant Capability Catalog (locally complete)

The source-controlled Catalog exactly binds 21/21 activated L0 contracts and their schemas/profiles. Strict models cover ownership/stewardship, scope, lifecycle, consumers, dependencies, supersession, and separated governance actions. Compatibility and consumer-impact reports are analysis-only and never register or activate capabilities.

#### P2.2 Evidence Plane and operations analytics (locally complete)

Five read-only adapters project Runtime, Decision, Saga, Provider release, and Promotion evidence into one digest-bound model with failure clusters, drift signals, and cross-snapshot local trends, without raw prompts, argument values, identities, Provider payloads, or paths. Missing chains, truncation, and tampering degrade the snapshot. Remote immutable storage, organizational audit, alerting, and production SLOs remain P1.7 site work. See [P2.1/P2.2 control planes](p21-p22-control-planes.md).

#### P2.3 Product entry and convergence evaluation (locally complete)

`scripts/netopyu` exposes understand/demo/integrate Golden Paths. Integration Packs enforce read/write separation, independent verification, compensation, credential references, and proposal-only authority. The source-controlled convergence baseline combines Core-72 with two 184-case model runs, preserves 368 prompt-free/argument-value-free traces, and reports one first-failure layer per case. The cockpit is self-contained and read-only. Real private holdout, site integrations, and production generalization remain open.

### 5. Recommended sequence

1. Keep P2.1/P2.2 gates and the P2.3 Integration Pack/cockpit in local regression as schemas evolve.
2. Exact intent, bounded enum normalization, the real 210-case 9B run, and phase-typed preflight/success-verification/compensation-verification gates are complete. Close the remaining evidence gap with an independently authored external 200+ case set, two-reviewer consensus, and at least three runs of the immutable 9B artifact. Evaluate 27B only after the protocol remains structurally stable.
3. When external conditions exist, close P1.9-B2 with a real sealed holdout, independent adjudication, and full Harness product evidence.
4. Qualify P1.3-B2, P1.4-B, and one P1.5 vendor vertical slice against independently owned systems.
5. Build P1.6–P1.7 HA/DR, security, remote immutable audit, observability, and SLO foundations; advance P1.9 canary only from measured real evidence.

### 6. Maintenance rule

Update this file after every substantive iteration. A phase becomes ✅ only after its completion criteria have reproducible evidence. Run the following gates after Runtime safety changes and record one new benchmark sample with a unique execution-code fingerprint:

```bash
scripts/netopyu-l0 runtime-validate
scripts/netopyu-l0 runtime-trajectories-validate
scripts/netopyu-dsh compare-runtime --iterations 50 --record --label <milestone>-<iteration>
scripts/netopyu-dsh retirement
```

Trend status uses the median of three unique execution-code fingerprints. The current 3/3 window is `stable`. Record every later substantive Runtime iteration and review `improved`, `stable`, or `regressed`; repeated runs of identical code do not count.
