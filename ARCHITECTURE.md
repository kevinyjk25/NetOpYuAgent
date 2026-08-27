# NetOpYuAgent 架构说明 / Architecture

## 中文

### 1. 权威架构声明

NetOpYuAgent 是 **Harness 可适配的网络领域插件**。DSH 是主平台，Hermes 是可选平台 Adapter；二者只能通过窄插件/Worker 协议进入同一个 Network Runtime。本仓库不实现通用 Agent Harness。

禁止重新引入以下表面：

- 自建 Agent loop、通用 planner 或模型 client；
- 独立 FastAPI/WebUI；
- 与 DSH/Hermes 并行的自建会话、模型循环或子代理框架；
- 绕过 Network Runtime 的直接写工具入口；
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
│ Network Domain Runtime                                      │
│ L1 workflow constraints · L0 contracts · plan state machine │
│ validation · preflight · execute · verify · compensate      │
└────────────────────────────┬─────────────────────────────────┘
                             │ adapter contracts
┌────────────────────────────▼─────────────────────────────────┐
│ Backends                                                     │
│ Profile mock · pragmatic local · MCP · OpenAPI · A2A peers  │
└──────────────────────────────────────────────────────────────┘
```

Domain L1 Skill 是“可以推理的业务编排”；Network L0 Skill 是“不能自由发挥的效果合同”。二者都属于 NetOpYu Domain Layer，不应与任一 Harness 的内部层级命名混用。

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
- `network_runtime/`：所有可变更效果的唯一执行入口。
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
dsh_adapter -> network_runtime / profiles / scoped services
network_runtime -> profiles metadata/callables / tools loader
profiles -> tools common helpers
pragmatic backend -> tools / integrations / schema
scoped services -> agent_memory / retrieval
evaluation -> public manifests / retrieval / test data
```

禁止的依赖方向：

- `network_runtime` 不得依赖 DSH/Hermes UI、模型或插件 API；
- `profiles`/`tools` 不得调用审批 API；
- L1 `SKILL.md` 不得直接持有 backend credentials；
- verifier 不得复用写工具返回文本作为唯一证据；
- mock profile 不得加载 pragmatic real sources；
- `evaluation` 不得进入生产工具执行调用链；
- `agent_memory` 不得自动注入任一 Harness prompt；
- Harness Adapter 不得用环境变量、模型文本或通用危险命令审批替代 request-level plan approval；
- Hermes Adapter 不得把 execution nonce 返回给模型；只有用户 slash command handler 可以消费进程内绑定。

`scripts/audit_harness_boundary.py` 和 retirement gate 负责防止历史自建 Harness 表面返回。

### 5. 核心不变量

1. **One effect path**：所有写效果都经过 Network Runtime。
2. **One immutable plan**：批准、grant 和执行看到同一个 plan hash。
3. **One-shot authorization**：每个 execution token/nonce 最多消费一次。
4. **Verify, do not infer**：成功只能由独立 postcondition 产生。
5. **Fail closed**：未知 backend、状态、合同、结果或 peer 都不转成成功。
6. **Profile isolation**：未投影能力对模型不可见，也不能通过 capability search 泄漏。
7. **No automatic promotion**：学习 proposal 不进入 active Skill registry。
8. **Audit every terminal path**：成功、拒绝、过期、漂移、回滚和人工介入都有终态事件。

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
- L1 → L0 自动下沉仍不是 P0.5 范围。

### 9. 文档治理

| 文档 | 权威内容 |
|---|---|
| `README.md` | 安装、运行、状态和使用入口 |
| `ARCHITECTURE.md` | 边界、依赖、不变量和 ADR |
| `HLD.md` | 组件、部署、端到端流程和 NFR |
| `LLD.md` | 模块、数据合同、状态机和算法 |
| `SSD.md` | 规范、安全控制、威胁和验收 |

所有主文档采用同文件双语结构，中文在前、英文在后。历史迁移过程不再作为当前架构文档保留；Git 历史承担追溯职责。

---

## English

### 1. Authoritative statement

NetOpYuAgent is a **harness-adaptable network-domain plugin**. DSH is the primary platform and Hermes is an optional adapter. Both enter the same Network Runtime through narrow public plugin and Worker contracts. This repository does not implement a general agent harness.

The architecture forbids a custom agent loop or model client, a standalone Web UI, parallel session/approval/subagent frameworks, direct mutations outside Network Runtime, and automatic activation of learned Skills.

### 2. Layers

1. **Harness Platform Layer:** DSH (primary) or Hermes (optional) for sessions, models, UI/CLI, tool lifecycle, and Skills.
2. **NetOpYu Domain Control Plane:** harness adapters, exact-plan approval binding, A2A, and scoped services.
3. **Network Domain Runtime:** reviewed L1 workflow constraints, versioned L0 contracts, deterministic plan execution.
4. **Backends:** profile mocks, pragmatic local tools, MCP, OpenAPI, and A2A peers.

Domain L1 Skills are model-assisted business orchestration. Network L0 Skills are fixed effect contracts. Both belong to the NetOpYu domain and must not be confused with names used for the platform layer.

### 3. Source boundaries

- `dsh-plugin-netopyu/` is the DSH JavaScript adapter.
- `hermes-plugin-netopyu/` and `hermes_adapter/` implement the official Hermes plugin boundary, process-local approval binding, and Worker client.
- `dsh_adapter/` projects Python domain capabilities into typed bridge commands.
- `network_runtime/` is the only mutation execution path.
- `profiles/` and `skills/` contain isolated capabilities and canonical L1 Skills.
- `tools/` and `integrations/` provide local, MCP, and OpenAPI adapters.
- `registry/` is outbound A2A discovery only.
- `runtime/` contains result storage and tracing shared by both adapters.
- `agent_memory/` is reachable only through scoped services.
- `evaluation/` is offline and never part of production execution.

### 4. Dependency rules

Either plugin may call the Worker bridge; the bridge may call Network Runtime, profiles, and scoped services; Network Runtime may call profile metadata/callables and tool loaders; pragmatic backends may call integrations and schema; scoped services may call memory and retrieval.

Network Runtime must not depend on DSH/Hermes UI, models, or plugin APIs. Tools must not call approval APIs. Skills must not carry credentials. Verifiers must not trust mutation response prose. Mock must not backfill pragmatic mode. Evaluation must not enter production execution. Memory must not inject itself automatically. Environment variables, conversational consent, and generic command approval must not replace exact-plan authorization. Hermes must not expose execution nonces to the model.

### 5. Invariants

- One mutation path through Network Runtime.
- One immutable plan shared by approval, grant, and execution.
- One-shot authorization and nonce consumption.
- Independent verification rather than inference.
- Fail closed for unknown state, backend, contract, result, or peer.
- Profile isolation for both tool projection and capability search.
- No automatic Skill promotion.
- A terminal audit event for every outcome.

### 6. Decisions

- **ADR-001:** DSH is the primary platform; Hermes is an optional adapter, not another domain runtime.
- **ADR-002:** a narrow persistent Python bridge preserves domain investment while containing migration risk.
- **ADR-003:** every mutation binds to a versioned Network L0 Skill.
- **ADR-004:** SQLite is the durable P0.5 store; production HA and immutable remote audit are P1 decisions.
- **ADR-005:** models are candidate generators, never a security boundary.
- **ADR-006:** A2A sends self-contained tasks and explicit delegation chains.
- **ADR-007:** oversized tool output is stored durably and paged.
- **ADR-008:** Hermes mutations prepare only; an exact user slash command consumes a process-local nonce binding.

### 7. Clean-code and extension policy

Keep only responsibilities used by the DSH or Hermes paths. Remove broken CLIs, retired custom-harness surfaces, duplicate documents, and runtime artifacts. Use explicit public APIs, frozen/versioned/hashable mutation contracts, side-effect-free imports, explicit backend cleanup, secret-safe errors, current module documentation, and both adapter gates before merge.

A new read tool needs a callable, metadata/schema, profile projection, tests, and result paging when needed. A new mutation additionally needs a ToolContract, L0 contract, intent/target/provenance compiler, independent preflight/verifier, optional compensator, approval assertions, integrity/failure/rollback tests, and a deterministic demo. A new L1 Skill must reference existing L0 effects and encode observations and stop conditions in a reviewed workflow.

### 8. Documentation ownership

`README.md` owns usage; `ARCHITECTURE.md` owns boundaries and decisions; `HLD.md` owns components and deployment; `LLD.md` owns contracts and algorithms; `SSD.md` owns requirements, threats, and acceptance. Each primary document contains Chinese first and English second. Git history, not duplicate migration documents, preserves historical evolution.
