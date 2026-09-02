# EnsuredSkill 低层设计 / Low-Level Design

> 实现基线 / Implementation baseline: 2026-09-02。字段和状态以当前源码为准；阶段结论以[项目进展](docs/PROJECT-STATUS.md)为准。

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

#### 2.6 从 Harness 到终态的模块追踪

| 阶段 | 入口实现 | 关键输出 |
|---|---|---|
| DSH Tool 投影与一次性授权 | `dsh-plugin-netopyu/src/index.js` | read 调用或 plan-bound write grant |
| Worker 协议桥 | `dsh-plugin-netopyu/src/bridge.js`, `dsh_adapter/worker.py` | 受限 JSON command；不暴露 Provider credential |
| 参数/L0/计划编译 | `network_runtime/engine.py::prepare` | `read_ready`、`clarification_required`、`rejected` 或 `plan_ready` |
| 合同与图内核 | `effect_runtime/reliability.py`, `effect_runtime/graph_scheduler.py` | `ReliabilityContract`、`TypedExecutionGraph` 和受控分支 |
| 一次性事务 | `network_runtime/engine.py::execute` | `ExecutionOutcome` |
| 状态、事件与恢复 | `network_runtime/contracts.py`, `journal.py`, `graph_runtime.py` | immutable plan、哈希链事件、crash reconciliation |
| 解释与检查 | `engine.py::inspect`, `provenance.py` | graph summary、stage latency、provenance DAG |

DSH 写 Tool 的 `execute()` 在拿到与 PreparedPlan 绑定的一次性 token 后，先请求 Runtime 签发/验证审批证明，再调用 `runtime-execute`。最终返回给模型的是 `terminal_envelope()`，不是 Actor 的原始结果。

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

每个 `OperationNode` 包含 id、phase、依赖和 `side_effect`。图校验拒绝重复节点、缺失依赖、环和多个不受控 Effect 节点。图摘要写入计划，审批后重编译不一致会被识别为漂移。

`effect_runtime/graph_scheduler.py` 是 fail-closed 调度门禁：节点必须按已审图和分支结果推进，Effect/Compensate 都是 one-shot，Commit 只能跟随独立 Verify 成功。`network_runtime/graph_runtime.py` 从哈希链事件重建调度状态；正常、审批拒绝、写前漂移、未知 Effect、补偿、恢复验证和启动恢复都写入 `graph_node_started/finished`。崩溃时未知的写边界只记录为 `skipped/indeterminate` 并进入只读 Reconcile，不伪造重校验成功，也不重放 Effect。

`inspect()` 同时返回图执行摘要、按 snapshot/precheck/approval/revalidate/effect/verify/reconcile/compensate 拆分的 Runtime 时延，以及隐私最小化的 Evidence→Observation→Capability/Collector→Network Object DAG。Runtime 时延明确排除 Reasoning/LLM，DAG 只证明已记录的来源关系，不证明外部载荷天然真实。

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

请求级分支的等价伪代码：

```text
if capability.kind == observation:
    authorize_observation_context()
    validate_exact_arguments()
    return invoke_observation()          # never obtains an effect lease

validate_exact_arguments_or_clarify()
l0 = resolve_one_active_l0_or_reject()
validate_runtime_projection(l0, provider, verifier, compensator)
evidence = snapshot_and_precheck()
require(evidence_contract && guards)
decision = risk_policy()
require(decision != REJECT)
plan = persist_immutable_plan_and_graph()
return plan_ready(plan, one_shot_nonce)
```

模型不可将 `clarification_required` 或 `rejected` 改写成可执行请求；修改参数后必须重新调用 prepare 并产生新的 plan hash。

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

一个完整 proposal 保存以下可审制品：

```text
proposal/
  00-capability-catalog.yaml
  01-L1-SKILL.md
  02-L0.5.yaml
  03-L0-authoring.yaml
  04-L0-compiled.json
  trajectory.json
  report.json
```

`report.json` 中的 requirement-level coverage 记录 L1 原句、L0.5/L0 JSON path、`preserved/weakened/missing/ambiguous`、解释与 `fix.file/path/hint`。`trajectory.json` 绑定阶段顺序、每个文件摘要和前驱摘要；Workbench 只是该数据的只读交互投影，没有 review、publish、activate 或 execute API。

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
| `rollback_verified` / ABORT | Effect 后补偿并证明恢复；任务本身不算成功 |
| `precondition_changed` / ABORT | 审批后事实漂移，Effect 被阻断 |
| `expired` / ABORT | 不可变计划在执行前过期 |
| `manual_intervention_required` / ESCALATE | 结果或恢复无法可靠证明 |

错误对象必须区分参数错误、证据不足、前置漂移、审批错误、执行错误、结果不确定、验证失败和补偿失败；不得将它们折叠为通用 `success=false`。

Harness 获得的终态信封固定为：

```json
{
  "contract": "netopyu.effect-runtime-terminal@1.0.0",
  "terminal": true,
  "ok": true,
  "state": "verified_success",
  "plan_id": "...",
  "plan_hash": "sha256:...",
  "summary": "...",
  "evidence": [],
  "error": null,
  "compensation": {"performed": false, "verified": false},
  "provider_result_digest": "sha256:..."
}
```

只有 `state=verified_success` 时 `ok=true`。`provider_result_digest` 保留调用关联性而不把 Provider 文本当作终态事实。详细的人类阅读方法见 [Skill 与系统交互全景](docs/SKILL-SYSTEM-INTERACTION.md)。

### 9. 测试设计

- 单元：Contract、Evidence integrity/freshness/scope、Guard、Risk、状态迁移；
- 组件：prepare/approve/execute/verify/compensate 和 hash-chain；
- 故障注入：超时前/后、断连、部分成功、验证不一致、补偿失败；
- 集成：DSH Worker、MCP/Containerlab Provider；
- 主实验：相同条件下的 DSH L1 Control 与 DSH + EnsuredSkill Treatment；
- 消融：分别移除 Contract、Evidence、Guard、Transaction、Compensation；
- 稳定性：9B 主模型与至少一个更弱模型，Runtime 合同保持不变。

### 10. 冻结代码隔离

`enterprise.py`、`provider_release.py` 和 `proposal_binding.py` 不再由默认核心路径顶层加载：本地 Runtime 直接构造本地审批；Provider admission 仅在显式环境开关下导入；L1 binding 仅在调用者确实提供 envelope 时导入。DSH CLI 同样只在显式命令下导入 A2A、轨迹学习和历史 L1 shadow；能力检索 parity 已迁入 `evaluation/`，并使用内存状态而不是产品 SQLite。该隔离确保未来产品扩展或 Evaluator 不会反向定义 EnsuredSkill 内核。

---

## English

### 1. Core model

`effect_runtime/reliability.py` defines the domain-neutral kernel: ReliabilityContract, EvidenceRequirement/Record, Guard, ResourceSet, RiskPolicy, OperationNode, TypedExecutionGraph, and TransactionStateMachine. `contract_from_compiled_l0()` projects a reviewed L0 v2 artifact into the kernel; the model cannot alter it at execution time.

Evidence evaluation requires exact semantic type, source capability, scope, associated action, freshness, validity, and payload integrity. Risk consumes scope, blast radius, evidence confidence, reversibility, history, and criticality, and emits only execute, ask-human, or reject. Structural gates always precede scoring.

The concrete request trace is DSH tool projection (`dsh-plugin-netopyu`) → narrow JSON Worker bridge (`dsh_adapter`) → parameter/L0/plan compilation (`NetworkRuntime.prepare`) → domain-neutral contract and graph gates (`effect_runtime`) → one-shot transaction (`NetworkRuntime.execute`) → journal/graph/provenance inspection. The Harness receives `terminal_envelope()`, never a provider result as authoritative success.

### 2. Plan and transaction

A PreparedPlan binds normalized arguments and provenance, exact L0 and tool contracts, targets/resources/risk, preflight evidence, the typed transaction graph, approval, provider schema identity, TTL, and immutable digests.

The normal graph is begin → snapshot → precheck → optional approval → revalidate → execute → verify → commit. A pre-Effect rejection or drift reaches abort. An indeterminate Effect enters read-only reconciliation; failed verification enters optional compensation and recovery verification; unresolved state escalates. Post-send uncertainty is observed before any retry.

`effect_runtime/graph_scheduler.py` is the fail-closed schedule gate: nodes advance only under the reviewed graph and prior outcome, Effect and Compensate are one-shot, and Commit requires successful independent Verify. `network_runtime/graph_runtime.py` reconstructs this cursor from hash-chained events. Crash recovery records unknown work as skipped/indeterminate and may reconcile by reads, but cannot replay Effect or invent successful revalidation.

Runtime inspection returns graph conformance, per-stage Runtime latency, and a privacy-minimized Evidence → Observation → Capability/Collector → Network Object DAG. Runtime latency excludes Reasoning/LLM latency; recorded lineage is not itself proof that an external payload is true.

### 3. Runtime algorithms

Prepare resolves the provider capability, validates and grounds parameters, refuses missing or ambiguous values, resolves the exact active L0, validates its runtime projection, builds the reliability contract and graph, gathers and validates preflight evidence, applies guards and risk, and persists an immutable plan.

Approval cannot mutate a plan. Execute reopens the provider and revalidates every mutable binding before consuming a one-shot authorization. The effect receipt is not success evidence. An independent observation proves postconditions. Failure reconciles actual state, compensates when specified, independently verifies recovery, and otherwise escalates.

### 4. Promotion and providers

L1-to-L0.5-to-L0 is an offline, review-gated authoring compilation. It records stage digests, semantic mappings, confidence, and loss warnings but cannot activate contracts automatically. Trace-based Experience Compilation remains future research.

Each proposal preserves the trusted Catalog, original L1, L0.5, L0 authoring, compiled candidate, trajectory, and report. Requirement-level coverage links L1 statements to L0.5/L0 paths, classifies preservation or loss, and identifies the exact file/path to revise. The Workbench is a read-only projection and has no publication or execution API.

Providers expose protocol-neutral Observation and Effect capabilities. MCP, REST, NETCONF, CLI, and local callables are adapter choices, not trust decisions.

### 5. Persistence, testing, and isolation

SQLite stores immutable plans and an append-only event hash chain for local crash recovery. It is not a distributed or WORM guarantee. Tests cover contracts, evidence, guards, risk, state transitions, full transaction paths, fault injection, DSH/Provider integration, paired control/treatment runs, five ablations, and cross-model stability.

Enterprise identity, provider supply-chain admission, and L1 canary binding are lazy optional extensions. The DSH CLI also imports A2A, trajectory learning, and historical L1 shadow only for explicit commands. Retrieval parity now lives in `evaluation/` and uses memory-only state instead of product SQLite. The default prototype path therefore cannot let frozen productization or evaluator code define the reliability kernel.

The only positive terminal state is `verified_success`. `rollback_verified` proves restoration but is not task success; `precondition_changed`, `rejected`, and `expired` are safe stops; `manual_intervention_required` means the target or recovery state remains unproved. The terminal envelope carries plan identity, independent evidence, error, compensation flags, and only a digest of the provider result. See the [Skill-to-system interaction guide](docs/SKILL-SYSTEM-INTERACTION.md).
