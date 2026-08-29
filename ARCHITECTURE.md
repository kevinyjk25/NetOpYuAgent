# NetOpYuAgent 架构说明 / Architecture

## 中文

### 1. 权威架构声明

NetOpYuAgent 是 **Harness 可适配的网络与业务运维领域插件**。DSH 是主平台，Hermes 是可选平台 Adapter；二者只能通过窄插件/Worker 协议进入同一个 Domain Effect Runtime。本仓库不实现通用 Agent Harness。

禁止重新引入以下表面：

- 自建 Agent loop、通用 planner 或模型 client；
- 独立 FastAPI/WebUI；
- 与 DSH/Hermes 并行的自建会话、模型循环或子代理框架；
- 绕过 Domain Effect Runtime 的直接写工具入口；
- 自动把轨迹/学习结果安装为可执行 Skill。

### 2. 架构分层

```text
┌──────────────────────────────────────────────────────────────┐
│ Harness Platform Layer: DSH (primary) or Hermes (optional)   │
│ Session · Model · UI/CLI · Tool lifecycle · Skill           │
└────────────────────────────┬─────────────────────────────────┘
                             │ public plugin contract
┌────────────────────────────▼─────────────────────────────────┐
│ NetOpYu Domain Control Plane                                 │
│ Harness adapters · approval binding · A2A · scoped services │
└────────────────────────────┬─────────────────────────────────┘
                             │ typed bridge commands
┌────────────────────────────▼─────────────────────────────────┐
│ Domain Effect Runtime                                       │
│ L1 workflow constraints · Network/Service L0 contracts      │
│ Provider admission · validate · execute · verify · rollback │
└────────────────────────────┬─────────────────────────────────┘
                             │ typed Tool Gateway
┌────────────────────────────▼─────────────────────────────────┐
│ Domain providers                                             │
│ Network Layer: Observer MCP · Durable Actor MCP · adapters  │
│ Service Layer: Identity/App/Policy/Change/CMDB/Platform MCP │
└──────────────────────────────────────────────────────────────┘
```

Domain L1 Skill 是“可以推理的业务编排”；Network/Service L0 Skill 是“不能自由发挥的效果合同”。Network Layer 与 Service Layer 是平行的事实和效果所有者，不得把业务授权与网络 enforcement 合并成一个 mock 状态。

### 3. 源代码边界

#### 3.1 平台集成

- `dsh-plugin-netopyu/`：DSH JavaScript Adapter；依赖 DSH/Cordis contract 和 Worker bridge。
- `hermes-plugin-netopyu/`：Hermes 官方插件加载入口；只转发到 Python `hermes_adapter`。
- `hermes_adapter/`：Hermes PluginContext 投影、Unix Socket client、进程内待审批绑定和 Adapter 对比门禁。
- `.netopyu-dsh/`：隔离的 DSH workspace 描述，不包含生成 runtime。
- `scripts/netopyu-dsh`：DSH 主平台的安装、运行、诊断和 retirement 入口。
- `scripts/netopyu-hermes`：Hermes 插件链接、Worker、doctor、运行与 Adapter 对比入口。

#### 3.2 领域控制面

- `dsh_adapter/`：把 Python 领域能力投影为 DSH 可消费的 manifest/commands。
- `effect_runtime/`：领域中性的运行时入口和跨层只读 reconciliation。
- `effect_runtime/saga.py`：跨 Provider durable Saga、计划绑定、逆序补偿和事件哈希链。
- `network_runtime/`：Effect Runtime 的计划状态机、合同、验证器、补偿器与兼容 API。
- `network_runtime/enterprise.py`：企业 OIDC/JWKS、按 session 动态 Gateway sender attestation、显式 CA/mTLS、PDP 和 Change Authority 协议 Adapter；不拥有 L0 业务语义。
- `network_runtime/enterprise_conformance.py`：离线配置 Doctor 与无效果 live contract 资格检查；不进入执行授权主路径。
- `network_runtime/provider_release.py`：Provider 发布合同、独立信任根、兼容/生命周期注册表和 Runtime admission。
- `network_runtime/provider_qualification.py`：固定故障资格套件；不得动态加载未受信 Provider 代码。
- `network_runtime/capabilities.py`：协议无关 observation/effect Capability SPI。
- `network_runtime/access.py`：只读主体、角色、scope、用途、clearance 与敏感度策略执行点。
- `network_runtime/provider_contracts.py`：稳定的 Network provider capability id/version 与角色注册表。
- `network_provider/`：身份固定的只读 Network Observer MCP、durable Network Actor MCP、版本化证据与 Actor operation store。
- `service_layer/`：官方 MCP SDK 服务、严格 structured result 和事务型业务仿真存储。
- `network_lab/`：P0.75-A manifest、Containerlab provider、FRR 命令白名单、探测与故障注入。
- `labs/p075-a-frr/`：可重建的 OSPF 主备路径实验拓扑和基线配置。
- `labs/p075-b-small-production/`：20 节点小型现网基准，覆盖园区、IDC、DMZ、双 ISP、OSPF/eBGP 与安全分区路径。
- `profiles/`：域隔离的 mock capability 与 canonical L1 Skill。
- `skills/`：common/pragmatic canonical Skill。

#### 3.3 适配与辅助

- `tools/`：公共工具与 pragmatic local adapters；
- `integrations/`：MCP/OpenAPI client 和 router；
- `registry/`：只保留 outbound A2A AgentCard discovery schema；
- `runtime/`：两个 Harness 共享的 tool-result store/tracing；
- `agent_memory/`：仅通过 scoped service 暴露；
- `retrieval/`：能力检索；
- `evaluation/`：离线、非生产执行路径的质量门禁；
- `tests/`：迁移与 P0.5 规格的可执行证据。

### 4. 依赖规则

允许的主要依赖方向：

```text
dsh-plugin-netopyu -> dsh_adapter bridge protocol
hermes-plugin-netopyu -> hermes_adapter -> Worker bridge protocol
dsh_adapter -> effect_runtime / profiles / scoped services
effect_runtime -> network_runtime safety kernel / reconciliation
network_runtime -> backend metadata/callables / tools loader
network_runtime Provider admission -> deployment-owned release registry / trust store
network_provider -> network_lab reviewed adapter / provider capability registry
service_layer -> official MCP SDK / transactional local store
profiles -> tools common helpers
pragmatic backend -> tools / integrations / schema
scoped services -> agent_memory / retrieval
evaluation -> public manifests / retrieval / test data
```

禁止的依赖方向：

- `effect_runtime`/`network_runtime` 不得依赖 DSH/Hermes UI、模型或插件 API；
- Service MCP 不得读取或修改 Containerlab；Network provider 不得冒充业务 entitlement；
- `profiles`/`tools` 不得调用审批 API；
- L1 `SKILL.md` 不得直接持有 backend credentials；
- verifier 不得复用写工具返回文本作为唯一证据；
- mock profile 不得加载 pragmatic real sources；
- `evaluation` 不得进入生产工具执行调用链；
- `agent_memory` 不得自动注入任一 Harness prompt；
- Harness Adapter 不得用环境变量、模型文本或通用危险命令审批替代 request-level plan approval；
- Hermes Adapter 不得把 execution nonce 返回给模型；只有用户 slash command handler 可以消费进程内绑定。
- Provider 自报 identity/release 或未签名 L0 授权不得成为 Runtime 信任输入。

`scripts/audit_harness_boundary.py` 和 retirement gate 负责防止历史自建 Harness 表面返回。

### 5. 核心不变量

1. **One effect path**：所有 Network/Service 写效果都经过 Domain Effect Runtime。
2. **One immutable plan**：批准、grant 和执行看到同一个 plan hash。
3. **One-shot authorization**：每个 execution token/nonce 最多消费一次。
4. **Verify, do not infer**：成功只能由独立 postcondition 产生。
5. **Fail closed**：未知 backend、状态、合同、结果或 peer 都不转成成功。
6. **Profile isolation**：未投影能力对模型不可见，也不能通过 capability search 泄漏。
7. **No automatic promotion**：学习 proposal 不进入 active Skill registry。
8. **Audit every terminal path**：成功、拒绝、过期、漂移、回滚和人工介入都有终态事件。
9. **Independent truths**：Service desired state、Network enforcement 和 data plane 分别读取并 reconciliation。
10. **Provider binding**：受信 Service 写计划绑定 MCP server identity/version 与 input/output schema digest。
11. **Capability over tool name**：Network provider 的稳定语义键是 capability id/version；tool name 只是 Harness 兼容别名。
12. **Evidence before consumption**：外部 Network observation 必须通过 identity、capability、时间和 payload digest 验证后才能进入 verifier/reconciliation。
13. **Capability, not transport**：Runtime 只依赖 observation/effect 语义，不依赖 MCP、REST、NETCONF 或 SSH 名称。
14. **Authorize every observation**：只读调用在 Provider 前验证主体、角色、scope、用途和数据敏感度。
15. **Terminal envelope only**：模型/UI 不得消费 Provider/Actor 中间态，只能消费 Runtime 终态封装。
16. **Saga never bypasses L0**：跨 Provider Saga 的每个正向和补偿步骤都是独立批准、验证的 L0 计划。
17. **Release before capability**：外部 Capability 只有在环境 active 的双签名 release 精确授权后才能进入 Runtime。
18. **Approval survives no drift**：schema-v9 绑定 release/manifest/qualification/deployment；审批后任何 Provider 部署或 L0 授权漂移均在写前终止。

### 6. 架构决策记录

#### ADR-001：DSH 主平台 + 可替换 Harness Adapter

决定：删除自建 UI、Agent loop、通用 HITL 和 inbound agent server。DSH 保持主平台；Hermes 只能作为共享领域层之前的可选 Adapter。

原因：通用 Harness 不是网络领域差异化。允许第二个 Adapter 可验证 Runtime 可移植性，但不允许第二套 Network Runtime、合同、验证或回滚语义。

#### ADR-002：保留 Python 领域 bridge

决定：DSH 插件通过持久化 Worker 调用 Python 领域代码。

原因：现有网络工具、校验器和领域测试以 Python 为主；bridge 将迁移风险限制在窄协议中，同时允许后续逐步重构。

#### ADR-003：写工具必须绑定 Network L0 Skill

决定：禁止仅凭 tool schema + DSH approval 直接执行写 callable。

原因：网络操作需要固定步骤、目标/意图 hash、preflight、revalidation、verifier 和补偿策略。

#### ADR-004：SQLite 作为 P0.5 durable store

决定：本地使用 SQLite、transaction、conditional update 和 hash chain。

原因：零外部依赖、易测试、重启可恢复。生产多实例一致性、远端不可变审计和 HA 在 P1 重新评估。

#### ADR-005：模型不是安全边界

决定：任何模型都只能提出候选计划。

原因：7B 与 27B 都可能发生阶段、工具、重复执行或总结错误；模型升级/降级不能改变 L0 安全合同。

#### ADR-006：A2A 只传自包含任务

决定：不继承父会话历史，显式记录 delegation chain。

原因：降低跨域数据泄漏、上下文污染和循环委派风险。

#### ADR-007：大结果外置

决定：超过阈值的工具输出写 SQLite，模型通过 bounded paging 读取。

原因：控制 context、避免重复传输，并在 bridge 生命周期之间保留结果。

#### ADR-008：Hermes 用户 slash command 是 Adapter 授权入口

决定：Hermes 写工具只 prepare；模型可见结果删除 execution nonce。只有操作员输入包含完整 plan id/hash 的 `/netopyu-approve` 或 A2A 等价命令才执行。待审批 nonce 只驻留插件进程，重启即失效。

原因：Hermes 的公开插件 API 可以注册工具和 slash command，但通用危险命令审批不是网络领域授权策略 API。显式用户命令保持人机边界，同时 Network Runtime 继续验证 hash、nonce、TTL、状态和证据。

### 7. Clean Code 规则

- 一个模块只保留当前 DSH/Hermes 路径实际使用的职责；
- 删除断裂 CLI、旧框架 adapter、重复文档和运行时产物，不保留“以后也许有用”的死代码；
- 公共 API 由包 `__init__` 明确导出，内部 helper 不跨层引用；
- mutation contract 必须 frozen/versioned/hashable；
- 不在 import 时启动网络、进程或写入持久化状态；
- backend 必须有显式 `close()` 生命周期；
- error message 不包含 secret value；
- 模块级文档描述当前架构，不描述已经删除的 `main.py`/legacy loop；
- 文档变更与代码合同变更同一个提交；
- `scripts/netopyu-dsh retirement` 与 `scripts/netopyu-hermes test` 是 Harness 相关变更的合并前门禁。

### 8. 新能力接入

#### 8.1 新只读工具

1. 在 profile 或 pragmatic adapter 实现 callable；
2. 添加 metadata/schema/action type=`read_only`；
3. 添加 profile projection 和测试；
4. 需要大结果时复用 `ToolResultStore`；
5. 更新 LLD/SSD 中的数据或安全说明。

#### 8.2 新写能力

除只读步骤外，还必须增加：

1. versioned ToolContract；
2. Network L0 Skill contract；
3. target/provenance/desired-state compiler；
4. independent preflight 与 verifier；
5. 安全可行时的 compensator；
6. approval card assertions；
7. hash、nonce、drift、failure、rollback 和 audit 测试；
8. model-independent deterministic demo。

#### 8.3 新 Domain L1 Skill

- 只描述诊断/业务阶段和允许调用的能力；
- 写阶段必须引用已存在的 L0 Skill；
- 必需 observation 和停止条件必须可编译为 reviewed workflow；
- L1 失败不得降低 L0 控制；
- L1 → L0 使用离线、审查门控的 Promotion Pipeline；Agent 只能生成候选，不能自动注册或执行。

### 9. 文档治理

| 文档 | 权威内容 |
|---|---|
| `README.md` | 安装、运行、状态和使用入口 |
| `ARCHITECTURE.md` | 边界、依赖、不变量和 ADR |
| `HLD.md` | 组件、部署、端到端流程和 NFR |
| `LLD.md` | 模块、数据合同、状态机和算法 |
| `SSD.md` | 规范、安全控制、威胁和验收 |

所有主文档采用同文件双语结构，中文在前、英文在后。历史迁移过程不再作为当前架构文档保留；Git 历史承担追溯职责。

### 10. P0.75-A 实验边界与决策

- **ADR-009：** Network Lab 是 pragmatic backend 的受限 provider，不是第三个 Harness 或 mock。
- lab 与真实 `device_inventory` 在同一进程中互斥，防止相同 `device_id` 指向不同信任域。
- `lab.yaml` 是设备、容器、探测和故障接口的唯一目标源；自然语言不能扩展 blast radius。
- 读写使用无 shell 参数数组；FRR 写入只接受审核白名单，`eth0` 管理口禁止变更。
- lab L0 成功要求 fresh running-config 以及计划绑定的预声明流量 probe；失败使用 provider 会话快照恢复并重新验证。
- Apple Silicon 上 Containerlab 运行在 Linux devcontainer/VM；P0.75-A 不证明厂商 CLI、ASIC、性能、硬件故障或无线 RF。
- **ADR-010：** 园区/IDC access lab 复用同一 pragmatic provider；用户、接口、地址、应用、端口、路径和角色均由 manifest 固定。LAN 与 DC 工具按 profile 隔离，只有 `network-lab` source 能取得这些 access 写合同。

### 11. P0.75-B 小型现网基准

- **ADR-011：** 完整现网实验仍是 `network-lab` pragmatic provider，不新增框架层或绕开 Network Runtime。
- Manifest 同时声明 OSPF/BGP 邻居期望、多前缀用户路由、允许/拒绝 probe 和故障目标；Runtime 只能执行这些预审对象。
- OOB Docker 管理网与业务路由分离。企业侧 OSPF、外部 eBGP、双出口回切、IDC/DMZ/访客分区和 HTTP 应用证据均在容器数据面实际运行。
- `secure-wan-edge` 是安全域路由角色。源 `/32` 黑洞表示本地 RBAC/微隔离 enforcement；不等同于状态防火墙、NAT、IPS 或厂商设备。
- **ADR-012：** 拓扑问答以 typed manifest graph 为唯一静态事实源，以 manifest-bound traceroute 为唯一运行路径证据。清单链路必须与 Containerlab wiring 精确相等；未知 hop、未证明 adjacency 或未到达目标一律 fail closed。端点不能传给设备工具，模拟准入/应用策略必须按真实实现命名。

### 12. P0.75-C EVPN/VXLAN Fabric 决策

- **ADR-013：** EVPN/VXLAN 仍属于同一 `network-lab` provider，以 typed Fabric manifest 为唯一静态事实源，以 Linux/FRR JSON 与 manifest probe 为运行证据。
- Fabric 写能力按完整扩展规则实现：metadata、`network.fabric.access-vlan.set` L0、ToolContract、workflow、verifier、compensator、测试和文档缺一不可。
- access VLAN 工具只接受声明的 VTEP/interface/VLAN 并使用固定 argv；trunk、批量端口、任意 CLI 与任意 IP 不在合同内。
- 当前执行内核不支持 NET_VRF，因此只声明 EVPN L2VPN。不得用 mock type-5/VRF 冒充 L3VPN；L3VPN 是执行环境升级后的独立资格认证。

### 13. P0.8 Service Layer 与跨层执行决策

- **ADR-014：Network Layer 与 Service Layer 平行。** Service MCP 管理业务 desired state，Containerlab/设备 adapter 管理网络 observed state；跨层一致性由 Effect Runtime reconciliation 计算，而不是让任一侧复制另一侧状态。
- **ADR-015：MCP 是协议边界，不是安全边界。** 只读 MCP 必须 identity-pinned 才能采用声明的 read-only 分类；可写 MCP 还必须显式 trusted、绑定 server name/version、declared contract、structured result 以及 input/output schema digest。
- **ADR-016：所有 Service 写继续使用 L0。** MCP tool 本身不能绕过 immutable plan、人工审批、revision preflight、fresh verifier、compensator 和 hash-chain audit。内部 restore tool 不投影给模型，只能由补偿器调用。
- **ADR-017：业务仿真状态一次初始化。** 六个 MCP 进程共享 SQLite；seed 使用版本标记只执行一次。审批检查、revision CAS、写入、幂等和审计位于同一个 `BEGIN IMMEDIATE` 事务；陈旧幂等结果不得在后续状态变化后重放。
- **ADR-018：跨层 L1 不是分布式原子事务。** `service-network-access-reconcile` 把 Service 与 Network 写拆成独立审批计划并在每步间重读；P1.1 durable Saga 固定步骤依赖、plan hash 与逆序补偿，但不合并审批，也不由模型临时拼接。

### 14. P0.9 Network Provider Boundary 决策

- **ADR-019：Runtime 拥有事务，MCP 拥有协议适配。** Domain Effect Runtime 继续拥有意图、L0 计划、风险、审批、执行状态、验证、补偿和审计；MCP server 只实现外部 Service/Network provider 合同。MCP 不因标准化传输而自动成为信任根。
- **ADR-020：Network read/write provider 分权。** `netopyu.network-observer@1.0.0` 仅公开 observer capability，并以 `network-evidence-envelope-v1` 返回 evidence；`netopyu.network-actor@1.0.0` 仅公开受审 Actor capability。Observer 不持有写入口，Actor 返回值不得替代 Observer 独立验证。
- **ADR-021：Actor MCP 必须 durable-first。** P1.0 在效果前持久化 immutable operation、审批 preflight digest、desired state 与精确 snapshot；按 target 使用进程锁、租约和单调 fencing token，启动时只读 reconciliation。重复 operation 只能恢复 prepared 状态或读回既有结果，不能盲目重放 executing/applied 写入。
- **ADR-022：效果上下文属于 Runtime，不属于模型。** operation/plan/intent hash、approved preflight 与 effect phase 由 backend 注入 MCP 的内部参数；restore/finalize 为 `internal_only`。PreparedPlan schema v9 绑定 capability、requester/policy 与 Provider release/deployment evidence。
- **ADR-023：本地 fencing 不冒充分布式一致性。** SQLite、WAL 和文件锁仅认证单主机 crash safety；Containerlab/设备端不校验 fencing token。生产部署必须增加远端事务日志或队列、设备/控制器 idempotency/CAS、独立读写身份与故障域、HA leader fencing 和远端不可变审计。
- **ADR-024：Capability SPI 高于传输协议。** Runtime 使用 `CapabilityContract`/`CapabilityProviderGateway`；协议适配留在 Provider Gateway，避免安全内核随 MCP/API 数量膨胀。
- **ADR-025：Read 也是受保护操作。** Observation 根据 sensitivity、role、scope、purpose、clearance 授权；本地隐式 system principal 只用于兼容原型。
- **ADR-026：Harness 只看 Runtime 终态。** Actor 原始结果只供 Runtime 内部处理，DSH/Hermes 对模型返回标准 terminal envelope。
- **ADR-027：跨层一致性使用 durable Saga，不假装 ACID。** Saga 持久化不可变步骤图和 plan hash，按依赖执行、逆序补偿并支持重启恢复，但每一步仍需 L0 审批和验证。
- **ADR-028：Runtime 成效必须用固定 L1 决策的 A/B Oracle 度量。** `evaluation/runtime_comparison.py` 给 DSH-only 参考路径与 Runtime 路径输入相同工具、参数、Provider 和故障，只比较 L0 增量；控制覆盖率不是生产成功概率，LLM/Skill 选择和人工等待另行评测。
- **ADR-029：L0 复用在编译期发生，执行期只认精确合同。** S1 使用 `AtomicEffect`；S11 可以是只收窄父合同的 constraint、只增加非冲突保证的 extension，或绑定 S1+S2+… 精确版本/hash 的 Composite Saga。同一 Capability 可实现多个语义 Skill，但 Runtime 不按 tool 名猜测。所有派生在编译时完全展开并重新哈希，执行期不解释继承。外部新 Capability 必须单独完成 Provider 与故障认证后才能发布。
- **ADR-030：L1 → L0 下沉是三阶段 Promotion，不是线上自修改。** 标准 `SKILL.md` 和受信 Capability Catalog 先形成可读、严格的 L0.5 StructuredNaturalLanguageSkill，再产生不可信 L0 候选。确定性检查阻止 L0.5 偏离 L1 或 L0 扩大 L0.5；Proposal 用逐级 hash 链保存 L1/L0.5/L0。人工 review 不自动激活 Catalog，也不授予执行权。未来 Runtime UI 只能复用 proposal API，禁止同一次会话生成、批准并执行新合同。
- **ADR-031：生产 L0 以编译 v2 Contract 为唯一语义权威。** 21 个内置受审写工具全部映射为 v2；旧 ToolContract、verifier 和 compensator 只作为精确 `RuntimeBinding` 下的合格实现 Adapter。prepare 与执行前重校验必须检查 Contract/Adapter parity，Provider 参数必须由 v2 模板从已批准值渲染；新裸 v1 L0 注册禁止进入生产 Registry。
- **ADR-032：每个生产 L0 必须有不参与执行的可读轨迹。** 存量合同保存 L1、L0.5、L0 authoring/compiled 和逐级 hash；主门禁同时重跑 Promotion semantic parity 与精确 compiler round trip。存量 L1/L0.5 必须标注为从受审 L0 反向 bootstrap，不能冒充历史自然语言来源、模型独立推导或执行授权；这些档案不得注册到 Harness。
- **ADR-033：审批结果必须成为 Runtime 签名证明。** schema-v9 plan 绑定 requester/policy 和 Provider release/deployment evidence；Harness 人工确认后 Runtime 才签发短时 plan-bound proof，并在消费 nonce 前验证全部绑定。
- **ADR-034：企业主体、Gateway、策略和工单是四个独立证据源。** `enforced` 只从固定 issuer/audience/非对称 algorithm 的短时 access JWT 获取 subject/role/scope/clearance/assurance；独立 Gateway JWT 绑定 Harness/session/client，并用 `act_sub + subject_jti` 绑定人的 credential。Gateway 可按每个 Harness session 动态签发；JWKS/mint/PDP/change 共用显式 CA/mTLS transport，默认不采信环境代理/CA。PDP 分别授权 observation/prepare/approve，Change Authority 验证 ticket revision/window/scope/risk。公开决策进入 requester/policy hash 或签名 proof，token/secret 不进入模型、计划或 journal。任一依赖缺失或失败均关闭 read/write；loopback HTTP 只用于本地资格实验，不构成生产信任根。
- **ADR-035：Provider 发布资格与执行事务是两个独立控制面。** Publisher、Qualifier、Deployer 使用独立 Ed25519 角色；外部 JSONL 进程通过固定 9/9 与真实 restart，Deployer 再签 exact artifact/environment。Runtime 把 release/deployment evidence 写入 schema-v9 plan并重复 admission。B-ready 的本地 fixture/key/SQLite 不冒充组织独立 Provider、HSM、真实 OCI/SBOM/SLSA 或 WORM；这些由现场 P1.4-B 提供。
- **ADR-036：L1/模型资格与 L0 安全资格必须分开。** P1.8 只把自然语言映射为无执行权候选，使用版本化 160 场景分别测候选召回、最终选择、参数、追问、workflow 与拒绝；模型看不到 callable effect、nonce/proof 或 Provider credential。只有完整数据集和不可变模型 artifact digest 可形成资格记录。模型通过不授权执行，模型失败也不能改变 Runtime 权限；Core-72 的固定 L1 结果不得冒充模型准确率。
- **ADR-037：Harness-in-loop 资格必须用无效果、配置封闭的影子路径。** P1.8-B1 只启用受审 DSH Agent/Session/LLM 基础 entry，固定版本并精确审计白名单；Skill/Tool/effect/shell/FS/Web/子代理/遥测/远程 Provider 全部禁用，临时 session 运行后删除。输出仍是无权 `L1Decision`。B1 只证明 Prompt/session composition；真实 Skill/tool-call 必须由不连接 Runtime/Provider 的 capture Tool 在独立 B2 中评测。
- **ADR-038：受控 Tool 协议只能收集候选，不能成为效果通道。** P1.8-C1 在 DSH 中预加载 digest-bound L0.5 风格 Skill，仅暴露五个互斥、强类型、无效果 proposal Tool。Loopback Protocol Governor 要求单次结构化提交，允许有限隐藏修复，并由确定性控制器校验目标、参数、workflow、Skill digest 与完整 transcript；任何缺失、冲突、越界、重复或摘要异常都不产生有效决定。该路径与 Runtime、Provider、设备、审批、凭证完全断开；固定场景通过率只用于模型资格，不等于生产成功率或执行授权。
- **ADR-039：L1 安全 Guard 只能收窄，且必须与模型成绩分账。** P1.8-C2 Guard 可要求安全拒绝、判定无领域证据的越界请求或低置信度弃权，但不得选择 Capability、补参数或生成 workflow。Loopback Protocol Firewall 校验每次 typed/candidate contract 并完整计量实际尝试；修复耗尽时只允许合成无参数 refusal/out-of-scope capture。报告必须同时保留模型首轮与 Guard 后 safety、固定集误杀、重试成本和尾时延；Guard 结果不能表述为模型准确率，也不能绕过 L0 Runtime。
- **ADR-040：候选身份与参数边界必须由 Tool/Schema 表达，模型只保留语义选择与显式值提取。** P1.8-C3 把每个检索候选编译成独立无效果 Tool，Tool 身份固定 kind/target，候选 Schema 固定允许业务键。Gateway 只能删除 Schema 外键，grounding 只能删除无请求证据的值或做受审归一化；action、missing fields 和 workflow 从可信 Catalog 确定性派生。任何层都不得替模型选择正常候选、补值或授予执行权。固定 184 条通过只认证摘要绑定的本地候选边界，所有写候选仍进入 L0 Runtime。
- 遥测、事件和大规模指标后续使用流式 evidence plane；MCP command/query 不承担高吞吐长期订阅。两条路径必须共享 correlation/target/capability schema，但不得把流事件直接当作写成功证明。

---

## English

### 1. Authoritative statement

NetOpYuAgent is a **harness-adaptable network/service-operations domain plugin**. DSH is the primary platform and Hermes is an optional adapter. Both enter the same Domain Effect Runtime through narrow public plugin and Worker contracts. This repository does not implement a general agent harness.

The architecture forbids a custom agent loop or model client, a standalone Web UI, parallel session/approval/subagent frameworks, direct mutations outside Domain Effect Runtime, and automatic activation of learned Skills.

**ADR-010:** The campus/IDC access lab reuses the pragmatic provider. Users,
interfaces, addresses, applications, ports, paths, and roles are fixed by the
manifest. LAN and DC tools are profile-isolated, and only the `network-lab`
source may receive the lab access write contracts.

### 2. Layers

1. **Harness Platform Layer:** DSH (primary) or Hermes (optional) for sessions, models, UI/CLI, tool lifecycle, and Skills.
2. **NetOpYu Domain Control Plane:** harness adapters, exact-plan approval binding, A2A, and scoped services.
3. **Domain Effect Runtime:** reviewed L1 workflow constraints, versioned Network/Service L0 contracts, and deterministic plan execution.
4. **Provider Release Control Plane:** independent publication/qualification trust, environment activation, and Runtime admission.
5. **Domain providers:** Containerlab/device adapters own network state; official-SDK MCP services own identity, application, policy, change, CMDB, and platform state.

Domain L1 Skills are model-assisted business orchestration. Network and Service L0 Skills are fixed effect contracts. Service desired state and Network enforcement are independent truths reconciled above both providers.

### 3. Source boundaries

- `dsh-plugin-netopyu/` is the DSH JavaScript adapter.
- `hermes-plugin-netopyu/` and `hermes_adapter/` implement the official Hermes plugin boundary, process-local approval binding, and Worker client.
- `dsh_adapter/` projects Python domain capabilities into typed bridge commands.
- `effect_runtime/` is the domain-neutral entry point and owns cross-layer read reconciliation.
- `effect_runtime/saga.py` owns durable cross-provider dependencies, plan bindings, reverse compensation, and its event hash chain.
- `network_runtime/` implements the shared plan kernel and remains the compatibility API.
- `network_runtime/enterprise.py` adapts enterprise OIDC/JWKS, per-session Gateway sender attestations, explicit CA/mTLS, PDP decisions, and Change Authority records without owning L0 business semantics.
- `network_runtime/enterprise_conformance.py` provides the offline Doctor and no-effect live contract qualification outside the execution authorization path.
- `network_runtime/provider_release.py` owns release contracts, independent trust roots, compatibility/lifecycle state, and Runtime admission.
- `network_runtime/provider_qualification.py` runs the fixed failure suite without dynamically importing untrusted Provider code.
- `network_runtime/capabilities.py` defines the transport-neutral observation/effect SPI, while `network_runtime/access.py` enforces read subject, role, scope, purpose, clearance, and sensitivity.
- `network_runtime/provider_contracts.py` owns stable Network provider capability ids, versions, and roles.
- `network_provider/` implements the identity-pinned read-only Network Observer MCP, durable Network Actor MCP, evidence envelope, and Actor operation store.
- `service_layer/` implements official-SDK MCP domains, strict results, and transactional local business simulation.
- `network_lab/` owns the reviewed P0.75-A manifest, Containerlab provider, FRR command allowlist, probes, and fault injection.
- `labs/p075-a-frr/` is the reproducible redundant-OSPF topology and baseline.
- `labs/p075-b-small-production/` owns the 20-node campus/IDC/DMZ/dual-ISP OSPF/eBGP reference topology.
- `profiles/` and `skills/` contain isolated capabilities and canonical L1 Skills.
- `tools/` and `integrations/` provide local, MCP, and OpenAPI adapters.
- `registry/` is outbound A2A discovery only.
- `runtime/` contains result storage and tracing shared by both adapters.
- `agent_memory/` is reachable only through scoped services.
- `evaluation/` is offline and never part of production execution.

### 4. Dependency rules

Either plugin may call the Worker bridge; the bridge may call Effect Runtime, profiles, and scoped services; Effect Runtime may call backend metadata/callables and tool loaders; pragmatic backends may call integrations and schemas; Service MCP servers own only business state; scoped services may call memory and retrieval.

Effect Runtime must not depend on DSH/Hermes UI, models, or plugin APIs. Service MCP must not modify Containerlab, and network providers must not claim business entitlement. Tools must not call approval APIs. Skills must not carry credentials. Verifiers must not trust mutation response prose. Mock must not backfill pragmatic mode. Evaluation must not enter production execution. Memory must not inject itself automatically. Environment variables, conversational consent, and generic command approval must not replace exact-plan authorization. Hermes must not expose execution nonces to the model.

Provider self-declared release identity or unsigned L0 authorization is never a trust input. Enforced admission depends only on deployment-owned Provider selection, the active registry state, scoped trust roots, and discovered contracts.

### 5. Invariants

- One mutation path through Domain Effect Runtime.
- One immutable plan shared by approval, grant, and execution.
- One-shot authorization and nonce consumption.
- Independent verification rather than inference.
- Fail closed for unknown state, backend, contract, result, or peer.
- Profile isolation for both tool projection and capability search.
- No automatic Skill promotion.
- A terminal audit event for every outcome.
- Independent Service desired state, Network enforcement, and data-plane evidence.
- Trusted MCP writes bound to provider identity/version and schema digests.
- Network providers bound by capability id/version rather than compatibility tool name.
- External observations validated for identity, capability, freshness format, and payload digest before consumption.
- External capabilities admitted only through the environment's active independently signed release.
- Schema-v9 approval binds release/manifest/qualification/deployment evidence; post-approval Provider deployment or L0 drift stops before any write.

### 6. Decisions

- **ADR-001:** DSH is the primary platform; Hermes is an optional adapter, not another domain runtime.
- **ADR-002:** a narrow persistent Python bridge preserves domain investment while containing migration risk.
- **ADR-003:** every mutation binds to a versioned Network L0 Skill.
- **ADR-004:** SQLite is the durable P0.5 store; production HA and immutable remote audit are P1 decisions.
- **ADR-005:** models are candidate generators, never a security boundary.
- **ADR-006:** A2A sends self-contained tasks and explicit delegation chains.
- **ADR-007:** oversized tool output is stored durably and paged.
- **ADR-008:** Hermes mutations prepare only; an exact user slash command consumes a process-local nonce binding.
- **ADR-009:** the local lab is a constrained pragmatic provider, never another harness or mock fallback. Its manifest is the sole target authority; lab and real inventory are isolated, writes use shell-free argv plus an FRR allowlist, and success can bind both configuration and predeclared traffic evidence.
- **ADR-010:** the campus/IDC access lab binds users, endpoints, applications, roles, and HTTP evidence to the same manifest while preserving LAN/DC tool isolation.
- **ADR-011:** the complete small-production topology remains behind the same provider and Network Runtime. It adds typed OSPF/eBGP expectations, reviewed multi-prefix endpoint routes, dual-ISP failover, DMZ/guest segmentation, and HTTP evidence without claiming stateful-firewall or vendor emulation.
- **ADR-012:** topology answers use the typed manifest graph as their only static source and manifest-bound traceroute as their only observed path evidence. Typed links must exactly equal Containerlab wiring. Unknown hops, unproved adjacency, or an unverified destination fail closed. Endpoints are never passed to device tools, and simulated enforcement is named by its actual implementation.
- **ADR-013:** EVPN/VXLAN remains inside the same `network-lab` provider. A typed fabric manifest is the only static source, while Linux/FRR JSON and declared probes are runtime evidence. The access-VLAN write has the complete metadata/L0/ToolContract/workflow/verifier/compensator/test chain and accepts only declared VTEPs, interfaces, and VLANs through fixed argv. The current kernel lacks NET_VRF, so only EVPN L2VPN is claimed; L3VPN requires a separately qualified execution environment and may not be simulated through fake type-5 or VRF evidence.
- **ADR-014:** Network and Service Layers are peers. Service MCP owns desired state; Containerlab/device adapters own observed enforcement; reconciliation compares independent facts.
- **ADR-015:** MCP is a protocol boundary, not automatically a trust boundary. Trusted writes require pinned identity/version, declared contracts, structured results, and schema digests.
- **ADR-016:** Service mutations use the same immutable-plan L0 kernel. Internal restore tools are hidden from the model and callable only by compensators.
- **ADR-017:** shared Service SQLite seeding is versioned and one-shot. Change authorization, revision comparison, mutation, idempotency, and audit are one immediate transaction; stale replays fail.
- **ADR-018:** the cross-layer L1 flow remains independently approved plans, not a distributed atomic transaction. The P1.1 durable Saga fixes dependencies, plan hashes, and reverse compensation without merging approvals or letting the model invent the workflow.
- **ADR-019:** Runtime owns end-to-end transaction semantics; MCP owns provider protocol adaptation. MCP transport does not create trust by itself.
- **ADR-020:** Network reads and writes are separated. `netopyu.network-observer@1.0.0` exposes observer capabilities and evidence only; `netopyu.network-actor@1.0.0` exposes reviewed Actor capabilities. Actor results never replace independent Observer verification.
- **ADR-021:** the Actor is durable-first. Before an effect it records immutable operation content, approved-preflight digest, desired state, and exact snapshot. Per-target locks, leases, monotonic fencing, and read-only startup reconciliation prevent blind replays after crashes.
- **ADR-022:** effect context belongs to Runtime, not the model. PreparedPlan schema v9 binds capability, requester/policy, and Provider release/deployment evidence; provider-internal effect context remains model-inaccessible.
- **ADR-023:** local fencing is not distributed linearizability. SQLite/WAL/file locks qualify one-host crash safety; production needs a remote transaction log or queue, device/controller idempotency or CAS, separated read/write identities and failure domains, HA leader fencing, and immutable remote audit.
- **ADR-024:** capability semantics sit above transport. Runtime consumes `CapabilityContract`/`CapabilityProviderGateway`; MCP/API/CLI adapters remain outside the safety kernel.
- **ADR-025:** reads are protected operations. Observation authorization binds sensitivity, role, scope, purpose, and clearance; the implicit local system principal is prototype compatibility only.
- **ADR-026:** Harnesses consume only a Runtime terminal envelope. Raw Actor states remain internal and are represented externally only by a digest.
- **ADR-027:** cross-layer consistency uses a durable Saga rather than pretending to be ACID. Immutable step/plan bindings, reverse compensation, restart recovery, and a hash chain never bypass per-step L0 approval and verification.
- **ADR-028:** Runtime value is measured with fixed-L1 A/B oracles. `evaluation/runtime_comparison.py` gives the DSH-only reference and Runtime paths the same tool, arguments, Provider, and fault. Control coverage is not a production success probability; LLM/Skill selection and human wait require separate evaluations.
- **ADR-029:** L0 reuse is compile-time only; execution consumes exact contracts. S1 uses `AtomicEffect`. S11 may be a constraint that only narrows its parent, an extension that adds non-conflicting guarantees, or a Composite Saga binding exact versions/hashes of S1+S2+…. One capability may back multiple semantic Skills, but Runtime never guesses from a tool name. Derivation is fully flattened and re-hashed before execution. Any new external capability requires separate Provider/fault qualification before publication.
- **ADR-030:** L1 → L0 is three-stage review-gated promotion, not online self-modification. A standard `SKILL.md` plus a trusted Capability Catalog first produces a strict, human-readable L0.5 StructuredNaturalLanguageSkill; the Agent then emits an untrusted L0 candidate. Deterministic checks prevent L0.5 drift from L1 and L0 widening of L0.5, while the proposal preserves L1/L0.5/L0 in a predecessor-linked hash chain. Human review neither activates the Catalog nor grants execution authority. A future Runtime UI may reuse the proposal API but cannot generate, approve, and execute a new contract in one session.
- **ADR-031:** compiled v2 contracts are the sole production L0 semantic authority. All 21 built-in reviewed mutation tools map to v2. Legacy ToolContracts, verifiers, and compensators remain only as qualified implementation adapters under exact `RuntimeBinding`s. Prepare and execution-time revalidation enforce contract/adapter parity, Provider arguments are rendered from approved values through v2 templates, and new raw v1 L0 registrations are forbidden in the production Registry.
- **ADR-032:** every production L0 requires a non-executable readable trajectory. Existing contracts preserve L1, L0.5, L0 authoring/compiled artifacts, and predecessor hashes; the primary gate reruns Promotion semantic parity plus exact compiler round trips. Existing L1/L0.5 artifacts must declare their reverse-bootstrap origin and cannot be presented as historical prose, independent model inference, or execution authority; the Harness never registers these archives.
- **ADR-033:** a Harness approval outcome becomes authority only after Runtime signs a short-lived proof bound to the exact plan, requester, policy, approver, risk, and approval mode. Execution verifies the proof before consuming the nonce. Local mode explicitly trusts only the owner OS/Worker boundary; enforced mode requires an enterprise credential verifier and rejects legacy actor strings.
- **ADR-034:** enterprise subject, Gateway, policy, and change record are independent evidence sources. Enforced mode derives subject/roles/scopes/clearance/assurance only from short-lived access JWTs with pinned issuer/audience/asymmetric algorithms. A separate Gateway JWT binds Harness/session/client and cross-binds the human credential through `act_sub + subject_jti`; it can be minted per Harness session. JWKS, mint, PDP, and change clients share explicit CA/mTLS transport and disable environment trust by default. PDP separately authorizes observation, prepare, and approve; Change Authority qualifies ticket revision/window/scope/risk. Public decisions enter requester/policy hashes or the signed proof, while tokens and secrets never enter model output, plans, or journals. Any missing or failed dependency closes reads and writes; loopback HTTP is local qualification only.
- **ADR-035:** Provider publication and effect transactions are independent control planes. Separate Publisher, Qualifier, and Deployer roots sign the Manifest, fixed external-process 9/9 report, and exact environment deployment observation. Admission binds release/deployment evidence into schema-v9 plans and repeats before execution. The B-ready local fixture is not organizational ownership, HSM/PKI, real artifact verification, or WORM audit; those remain P1.4-B site work.
- **ADR-036:** L1/model qualification is separate from L0 safety qualification. P1.8 maps natural language only to non-authoritative proposals and uses a versioned 160-case set to measure candidate recall, final choice, arguments, clarification, workflows, and refusal. The model receives no callable effect, nonce/proof, or Provider credential. Only complete runs bound to immutable model artifacts may be recorded. Passing never authorizes execution, failure never changes Runtime permissions, and Core-72's fixed-L1 result cannot be presented as model accuracy.
- **ADR-037:** Harness-in-the-loop qualification uses a no-effect, configuration-closed shadow path. P1.8-B1 enables only reviewed DSH Agent/Session/LLM infrastructure, pins the version, and audits an exact allowlist; Skills, tools, effects, shells, filesystem tools, Web, subagents, telemetry, and remote providers are disabled, and ephemeral sessions are removed. Output remains a non-authoritative `L1Decision`. B1 measures prompt/session composition only; real Skill/tool-call behavior requires a separate capture Tool with no Runtime/Provider path in B2.
- **ADR-038:** the controlled Tool protocol captures proposals only and can never become an effect channel. P1.8-C1 preloads a digest-bound L0.5-style Skill in DSH and exposes exactly five mutually exclusive, typed, no-effect proposal Tools. A loopback Protocol Governor requires one structured submission, permits only bounded hidden repair, and lets a deterministic controller validate targets, arguments, workflow, Skill digest, and the complete transcript. Missing, conflicting, out-of-scope, duplicate, or malformed output yields no valid decision. The path has no Runtime, Provider, device, approval, or credential connection; fixed-set scores are model-qualification evidence, not production success probabilities or execution authority.
- **ADR-039:** an L1 safety Guard may only narrow authority, and its outcome is accounted separately from model quality. P1.8-C2 may require safe refusal, classify a request with no domain evidence as out of scope, or abstain on low confidence; it cannot select a Capability, add arguments, or create workflow. A loopback Protocol Firewall validates every typed/candidate contract and fully meters actual attempts. Exhaustion may synthesize only an argument-free refusal/out-of-scope capture. Reports retain both first-attempt and guarded safety, fixed-set false positives, repair cost, and tail latency; Guard results are not model accuracy and never bypass L0 Runtime.
- **ADR-040:** candidate identity and argument bounds are expressed by Tools and Schemas, leaving only semantic choice and explicit-value extraction to the model. P1.8-C3 compiles each retrieved candidate into a distinct proposal-only Tool whose identity fixes kind/target and whose Schema fixes allowed business keys. The Gateway may only delete unknown keys, grounding may only remove unsupported values or apply reviewed normalization, and action/missing fields/workflow are derived deterministically from the trusted Catalog. No layer may select a normal candidate for the model, invent a value, or grant execution authority. Passing the fixed 184 cases qualifies only the digest-bound local proposal boundary; every write proposal still enters L0 Runtime.
- High-volume telemetry and event streams belong on a separate evidence plane. MCP remains the command/query protocol; both paths share target/correlation/capability schemas, and stream events never prove mutation success by themselves.

### 7. Clean-code and extension policy

Keep only responsibilities used by the DSH or Hermes paths. Remove broken CLIs, retired custom-harness surfaces, duplicate documents, and runtime artifacts. Use explicit public APIs, frozen/versioned/hashable mutation contracts, side-effect-free imports, explicit backend cleanup, secret-safe errors, current module documentation, and both adapter gates before merge.

A new read tool needs a callable, metadata/schema, profile projection, tests, and result paging when needed. A new mutation additionally needs a ToolContract, L0 contract, intent/target/provenance compiler, independent preflight/verifier, optional compensator, approval assertions, integrity/failure/rollback tests, and a deterministic demo. A new L1 Skill must reference existing L0 effects and encode observations and stop conditions in a reviewed workflow.

### 8. Documentation ownership

`README.md` owns usage; `ARCHITECTURE.md` owns boundaries and decisions; `HLD.md` owns components and deployment; `LLD.md` owns contracts and algorithms; `SSD.md` owns requirements, threats, and acceptance. Each primary document contains Chinese first and English second. Git history, not duplicate migration documents, preserves historical evolution.
