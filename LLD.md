# EnsuredSkill 低层设计 / Low-Level Design

## 中文

### 1. 实现范围

本文只描述当前 EnsuredSkill 原型执行闭环。历史企业身份、Provider 供应链、治理、Hermes/A2A 和 canary 产品化实现属于冻结扩展，不是核心 Runtime 依赖。

### 2. 核心数据结构

#### 2.1 ReliabilityContract

`effect_runtime/reliability.py` 定义领域中性合同：

```text
ReliabilityContract
  operation + version
  inputs_schema_digest
  preconditions[]
  evidence[]
  guards[]
  postconditions[]
  resources{reads,writes}
  reversibility
  idempotent
  timeout_seconds
  approval_required
  compensation_operation?
```

合同创建时验证基本闭包：操作和版本存在、Evidence id 唯一、Guard 引用有效、资源写集合无重复、可补偿性与可逆性一致。`contract_from_compiled_l0()` 将已审 L0 v2 投影到该内核，不允许模型在执行时改变合同。

#### 2.2 EvidenceRequirement / EvidenceRecord

Requirement 固定：`id`、`semantic_type`、`source_capability`、`phase`、`max_age_seconds`、`scope`、关联 Action 和谓词。

Record 固定：Evidence id/type/source、collector identity、采集时间、scope、associated action、payload、payload digest、valid 和父 Evidence id。`evaluate_evidence()` 检查：

1. 必需 Evidence 是否存在；
2. 类型和来源 Capability 是否精确匹配；
3. scope 和关联 Action 是否匹配；
4. 时间是否合法且未过期；
5. payload digest 是否仍然一致；
6. valid 是否为真。

任一失败都在 Effect 前关闭执行。

#### 2.3 Guard

Guard 是确定性谓词，不是自然语言建议：

```text
Guard{id, field, operator, expected, evidence_requirement_id?}
```

当前 L0 编译器负责静态结构和引用门禁；Runtime 负责用已验证 Evidence 求值。未知字段、未知 operator 或无法求值均失败关闭。

#### 2.4 RiskFactors / RiskAssessment

风险输入：ChangeScope、BlastRadius、EvidenceConfidence、Reversibility、HistoricalSuccess、ServiceCriticality。输出固定为：

```text
EXECUTE | ASK_HUMAN | REJECT
```

结构门禁、Evidence 和 Guard 永远先执行。受审 L0 可以比通用风险策略更保守。`delete_resource` 和 `force` 对外保持 `critical` 分类，但风险决策仍独立，避免把“Critical”错误等同于“编译阶段必须拒绝”。

#### 2.5 PreparedPlan

`network_runtime/contracts.py` 中的 PreparedPlan 绑定：

- normalized arguments 与 provenance；
- exact L0 id/version/contract hash；
- Tool/Capability contract；
- target/resource/risk；
- preflight Evidence；
- typed transaction graph；
- requester/approval binding；
- provider identity/schema binding；
- TTL、plan id 和 plan hash。

冻结扩展的 release/deployment 或 L1 Decision provenance 只在显式使用时加入；不是原型核心合同的必需字段。

### 3. Typed Execution Graph

`build_transaction_graph()` 生成固定的 phase DAG：

```text
begin
  → snapshot
  → precheck
  → awaiting_approval?
  → revalidate
  → execute
  → verify
  → commit

execute/verify
  → reconcile
  → compensate?
  → verify_recovery
  → abort | escalate
```

每个 `OperationNode` 包含 id、phase、依赖、side_effect 和 resource set。图校验拒绝重复节点、缺失依赖、环和多个不受控 Effect 节点。图摘要写入计划，审批后重编译不一致会被识别为漂移。

### 4. prepare 算法

`NetworkRuntime.prepare()`：

1. 从 Backend/Provider 获取 Tool metadata 和 CapabilityContract；
2. 严格校验参数类型、枚举、范围、未知字段、目标解析和来源；
3. 对缺参或歧义返回 `clarification_required`，不得猜测；
4. 解析 reviewed ToolContract 和 exact active L0；
5. 校验 L0→Tool/Verifier/Compensator Runtime projection；
6. 投影 ReliabilityContract 并生成 Typed Execution Graph；
7. 执行 snapshot/preflight Observation；
8. 将 Observation 绑定为有 provenance 的 EvidenceRecord；
9. 运行 Evidence、Guard 和 Risk gate；
10. 绑定本地 requester/approval policy；
11. 创建不可变 PreparedPlan 和一次性执行 nonce；
12. 写入 Journal，返回 `plan_ready` 或明确的 clarification/reject。

核心默认直接创建本地 `ApprovalControlPlane`。只有显式 `NETOPYU_IDENTITY_MODE=enforced` 时才延迟加载冻结的 enterprise adapter。

### 5. approve / execute 算法

#### 5.1 Approval

审批证明必须绑定 plan id/hash、requester、approver、policy、risk、过期时间和一次性 token。审批不能修改参数、L0、目标、Evidence 或图；需要修改时必须重新 prepare。

#### 5.2 Revalidation

执行前重新打开 Backend，并验证：

- Tool/Capability/L0/contract 仍存在且摘要一致；
- 参数和目标仍满足合同；
- typed graph 未漂移；
- plan、nonce、审批证明和 TTL 有效；
- 关键 preflight Evidence 仍新鲜且 snapshot 未变化。

失败返回 `precondition_changed`/`expired`/`rejected`，Effect 不发送。

#### 5.3 Effect and verification

Runtime 通过 `BackendSession.invoke_effect()` 发送一次效果。模型和 L1 不获得 nonce、Provider credential 或可重放句柄。写返回值只形成 receipt；Verifier 再调用独立 Observation，并将结果与 L0 postconditions 比较。仅全部通过才进入 `verified_success/COMMIT`。

#### 5.4 Uncertainty and compensation

- timeout-before-send：安全终止；
- sent/unknown：进入 reconcile Observation；
- 已达到 desired state：继续独立 verify；
- 部分或错误状态：调用精确 compensator；
- compensation 后独立 verify recovery；
- 无法证明恢复：`manual_intervention_required/ESCALATE`。

Runtime 不对非幂等写进行盲重试。

### 6. Promotion 实现

```text
L1 SKILL.md
  → L0.5 Structured Natural Language
  → L0 authoring contract
  → compiled L0
  → semantic review / human decision
  → explicit activation outside proposal directory
```

Promotion 记录逐阶段 digest、字段映射、置信度和语义丢失告警。转换模型只建议结构化内容；不可猜测的 Capability、target、precondition、Evidence、Guard、postcondition 和 compensation 必须来自受信 Catalog/显式锚点。低置信、缺引用、扩大权限、弱化 Safety 或缺 verifier/compensator 时不得进入 active Registry。

这条路径不自动从运行轨迹生成 L0。Experience Compilation 保留为未来研究：它必须基于多次真实成功与失败轨迹、聚类、参数抽象、反例验证和独立 Promotion。

### 7. Provider 接口

Runtime 只依赖协议中性的两类 Capability：

```text
Observation(arguments) -> typed evidence envelope
Effect(arguments, immutable runtime context) -> effect receipt
```

Adapter 可以使用 MCP、REST、NETCONF、SSH/CLI 或本地 callable。Protocol 不授予信任；Runtime 仍验证 capability id/version、schema、scope、freshness 和结果结构。Provider release/supply-chain admission 是延迟加载的冻结扩展。

### 8. Journal 与错误语义

Journal 使用 SQLite 保存不可变 plan 和 append-only event hash chain。当前原型要求单机 crash recovery 和可复算完整性，不宣称分布式一致性或 WORM。

主要终态：

| 终态 | 含义 |
|---|---|
| `verified_success` / COMMIT | 独立 postcondition 成立 |
| `rejected` / ABORT | 写前合同、Evidence、Guard、Risk 或审批失败 |
| `rolled_back` / ABORT | Effect 后补偿并证明恢复 |
| `manual_intervention_required` / ESCALATE | 结果或恢复无法可靠证明 |

错误对象必须区分参数错误、证据不足、前置漂移、审批错误、执行错误、结果不确定、验证失败和补偿失败；不得将它们折叠为通用 `success=false`。

### 9. 测试设计

- 单元：Contract、Evidence integrity/freshness/scope、Guard、Risk、状态迁移；
- 组件：prepare/approve/execute/verify/compensate 和 hash-chain；
- 故障注入：超时前/后、断连、部分成功、验证不一致、补偿失败；
- 集成：DSH Worker、MCP/Containerlab Provider；
- 主实验：相同条件下的 DSH L1 Control 与 DSH + EnsuredSkill Treatment；
- 消融：分别移除 Contract、Evidence、Guard、Transaction、Compensation；
- 稳定性：9B 主模型与至少一个更弱模型，Runtime 合同保持不变。

### 10. 冻结代码隔离

`enterprise.py`、`provider_release.py` 和 `proposal_binding.py` 不再由默认核心路径顶层加载：本地 Runtime 直接构造本地审批；Provider admission 仅在显式环境开关下导入；L1 binding 仅在调用者确实提供 envelope 时导入。该隔离确保未来产品扩展不会反向定义 EnsuredSkill 内核。

---

## English

### 1. Core model

`effect_runtime/reliability.py` defines the domain-neutral kernel: ReliabilityContract, EvidenceRequirement/Record, Guard, ResourceSet, RiskPolicy, OperationNode, TypedExecutionGraph, and TransactionStateMachine. `contract_from_compiled_l0()` projects a reviewed L0 v2 artifact into the kernel; the model cannot alter it at execution time.

Evidence evaluation requires exact semantic type, source capability, scope, associated action, freshness, validity, and payload integrity. Risk consumes scope, blast radius, evidence confidence, reversibility, history, and criticality, and emits only execute, ask-human, or reject. Structural gates always precede scoring.

### 2. Plan and transaction

A PreparedPlan binds normalized arguments and provenance, exact L0 and tool contracts, targets/resources/risk, preflight evidence, the typed transaction graph, approval, provider schema identity, TTL, and immutable digests.

The normal graph is begin → snapshot → precheck → optional approval → revalidate → execute → verify → commit. Failures enter reconcile, optional compensate, verify-recovery, and abort or escalate. Post-send uncertainty is observed before any retry.

### 3. Runtime algorithms

Prepare resolves the provider capability, validates and grounds parameters, refuses missing or ambiguous values, resolves the exact active L0, validates its runtime projection, builds the reliability contract and graph, gathers and validates preflight evidence, applies guards and risk, and persists an immutable plan.

Approval cannot mutate a plan. Execute reopens the provider and revalidates every mutable binding before consuming a one-shot authorization. The effect receipt is not success evidence. An independent observation proves postconditions. Failure reconciles actual state, compensates when specified, independently verifies recovery, and otherwise escalates.

### 4. Promotion and providers

L1-to-L0.5-to-L0 is an offline, review-gated authoring compilation. It records stage digests, semantic mappings, confidence, and loss warnings but cannot activate contracts automatically. Trace-based Experience Compilation remains future research.

Providers expose protocol-neutral Observation and Effect capabilities. MCP, REST, NETCONF, CLI, and local callables are adapter choices, not trust decisions.

### 5. Persistence, testing, and isolation

SQLite stores immutable plans and an append-only event hash chain for local crash recovery. It is not a distributed or WORM guarantee. Tests cover contracts, evidence, guards, risk, state transitions, full transaction paths, fault injection, DSH/Provider integration, paired control/treatment runs, five ablations, and cross-model stability.

Enterprise identity, provider supply-chain admission, and L1 canary binding are lazy optional extensions. The default prototype path does not import them, preventing frozen productization code from defining the reliability kernel.
