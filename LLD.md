# EnsuredSkill 低层设计 / Low-Level Design

> 实现基线 / Implementation baseline: 2026-09-02。字段和状态以当前源码为准；阶段结论以[项目进展](docs/PROJECT-STATUS.md)为准。

## 中文

2026-09-16：[考核重置](docs/EVALUATION-RESET-20260916.md)已获授权实施，当前为 **R0 measurement partial implemented**。`bounded_budget` 使用 SQLite 原子预留、持久调用身份、未知状态停机和跨进程累计；`bounded_scoring`独立评分；`bounded_pilot`只提供 `check/prepare/inspect/score`；`bounded_execution`提供脚本式计量；`bounded_probe`产生本地测量证据。真实 DSH 全调用计量与可信预先 token 计数、物理 Provider 重置／隔离、预封存标签和 12 任务样本、真实 trace 采集未完成；自动 Effect 桥接也未实现。153 项定向通过和 0 模型探针不构成 36 项机制验收或真实 Agent 收益。旧语义阶段仍为 `paused_unmet`，正式门禁与旧失败不变；没有 `run` 命令。

当前[结构与工件实现](docs/SCHEMA-ARTIFACT-CONVERGENCE.md)：`isolated_compiler.make_request`将原Schema传入format；`kql_checks`只验证声明子集；`artifact_repair`从宿主实际交付定位失败围栏，接受一次`replacements(location,code)`。`hybrid_session`冻结`artifactRepair`策略，保留draft原件，revision先claim再解析，另存终态；未知结果禁止重试。DSH只对`revisionAllowed=true`暂停终态结束，最终仍展示宿主回执。默认策略不变。

2026-09-16 新入口细节见[隔离编译](docs/ISOLATED-COMPILER.md)：v4宿主`compilerMode=isolated`使prepare在会话锁内记录单次claim并调用`skill_authoring/isolated_compiler.py`；不向编译请求传实参/观察，运行Agent的Schema中移除submit且后端拒绝AST注入。编译成本持久化并进入评测总账；未知调用不重试，旧模式默认行为不变。

快照读取补充：v4本地宿主提供`observationModel`（宿主摘要、工具＋精确JSON参数身份、liveRefreshSupported=false）。`read()`在原状态/次数/Schema门禁后检查已完成且绑定的前段或补读记录；重复返回`read_not_reexecuted`和原观察引用，不触发Provider、不伪造新receipt、不增加completedReads，但仍消耗尝试额度。不同资源正常授权读取；严格图内部调度和实时Provider不变。[具体范围与验证](docs/SNAPSHOT-READ-SEMANTICS.md)。

任务直达补充：显式宿主v4使用`task-bound-delivery/v1`。`compile_task(null, origins)`绑定完整任务和来源摘要，拒绝模型提交替换职责；`response_schema`只定义非空answer信封，`render`逐字保留，不加章节。原生deliver与Runtime draft复用原权限/证据/静态检查/终态路径。覆盖率未知，输出不成为L0或获得操作权。[设计、对照诊断和边界](docs/TASK-FIRST-DELIVERY.md)。

宿主终态补充：显式`NETOPYU_HYBRID_TERMINAL_DELIVERY=1`启用会话/摘要绑定的终态适配器。工具体通过DSH公开`concludeTurn()`结束轮次，同轮后续混合操作（inspect除外）在bridge前阻止；新轮次独立。UI工具卡和可选headless前端显示最后有效宿主回执，不伪造assistant事件。取消、错误、后续步骤不能追认旧终态；新报告增加sessionId，旧报告不改。`semanticCoverage=not_assessed`说明结构/引用不等于语义完备。默认路由不变。[接入、实现和实际DSH机制测试](docs/HOST-TERMINAL-DELIVERY.md)。

单选协议补充：显式宿主v3使用交付合同v4，每个ID只接受类型化内容对象或未解决原因字符串，不再跨字段同步内容/状态；字符串None也保持未解决。旧v2/v3合同不自动迁移。独立`hostResult`记录拒绝、待修订或候选未验证，不能被Agent的最终文字改写；这不证明业务语义正确。[本包边界与结果](docs/DELIVERY-SINGLE-CHOICE.md)。

紧凑协议补充：显式宿主v2使用交付合同v3。`prepare`提供可逐字重建的source_ref；`submit`由宿主解析原文/偏移/摘要。候选直接填写每项类型化内容，无法提供则显式null＋同ID的unresolved说明；内部state/content/gap由已声明的编码规则生成，不修复旧请求。传输对象与规范化对象分别摘要绑定。未知引用、遗漏ID、孤立缺口和错类型仍拒绝；含义正确性不由结构证明。[协议、版本与边界](docs/DELIVERY-REFERENCE-CONTENT.md)。

交付接口补充（2026-09-15）：`submit(session_id,plan,delivery)` 将逐字来源要求编译为最多四个宿主 ID 和固定响应 Schema，并绑定会话摘要。正常 `draft(session_id)` 冻结证据、调用一次有界模型；仅执行前 fallback 使用 `submit(plan=null,delivery=...)` 和 `deliver(session_id,response_json)`，后者只检查/渲染原生 L1 文本。缺内容或显式未满足项不能计为结构完整；`semanticApproval=false`、`taskSuccess=null`。[字段语义与状态](docs/GOVERNED-SESSION.md)。

2026-09-15 收敛更新：当前可选本地主链为 DSH → `skill_authoring` 共享转译 → 原 Runtime 前置读取 → DSH/L1 受限补读 → 冻结证据 → 单次有界推理与窄范围工件检查。评测代码不进入产品依赖；通用语义自审不作为执行权限来源。转译保真、执行约束、任务质量独立验收。宿主配置的六工具和限制见[主链接入说明](docs/GOVERNED-SESSION.md)。原有写事务边界不变，本入口不授权写入。下列早期语义审阅设计是保留的研究历史，不是当前产品必经链路。

2026-09-10 后续接口：`run_hybrid(..., result_contract=...)` 可接收 `ResultContract`；`HostHybridConsent.result_contract_digest` 同时绑定任务图、参数、上下文和结果合同。`resultAssessment` 使用调度器本次节点值核验读取/字段，开放职责与草稿保留未验证。实现与错误定位见[结果合同](docs/SEMANTIC-RESULT-CONTRACT.md)。

当前可选入口是 `evaluation.semantic_closure_transfer`：构造、显式准入、原引擎读取、`hybrid_continuation_run` 有界补读、`hybrid_snapshot_review` 来源审查、`hybrid_repair_cells` 章节编辑与终审。新 `grounded_patch` 分开 `keep/write_prose` 答复编辑与独立 `source_units` 支持引用；后者在 `delivery.json` 单列，不能覆盖答复或抵消错误。原 notes 和宿主开放职责保留出处。每段最多一个范围，总计最多 8 单元；生成字段顺序按 `required` 恢复。旧 `source_patch` 留作失败诊断。有效读取前段投影保留首个坏节点及之后全部拒绝记录，不改参数、不授准入；当前可读审阅视图保留全部原文，重复导航元数据留在审计中，旧可逆文本池仅供历史诊断。详见[接口与验收](docs/SEMANTIC-CLOSURE-RUNBOOK.md)和[首批负结果](docs/SEMANTIC-CLOSURE-TRANSFER-V1.md)。

### 受控混合图实现补充（2026-09-10）

`GovernedHybridFlow` 固定原始 source/task 摘要、输入 Schema、节点依赖、输出与预算；`StrictRegion` 包含原 `StructuredFlowProposal`。`ReasoningTask` 绑定模型/配置、输入投影、输出 Schema、字节/token/时间预算；`ConditionalReasoningTask(kind=reason_if)` 增加类型化标量条件和 `otherwise` 绑定，必须来自声明依赖。条件不符时校验并保留绑定候选，记录零模型调用，不提升为观察；原 `reason` 的序列化结构不变。`CandidateAdmission` 绑定宿主独立策略；`RequiredJoin` 仅在所有必需依赖成功后汇合。代码见 [hybrid.py](network_runtime/l0/hybrid.py) 和 [hybrid_execution.py](network_runtime/l0/hybrid_execution.py)。

语义闭环编辑默认由已定位的负面意见触发；唯一一个单元确实覆盖整稿时也可承接未定位问题，但不编造语义位置。其余未定位遗漏留在 `repairPlan`，不广播编辑；只读/未调度单元用 `reason_if` 保留未批准的原稿。编辑器只看当前正文与其他章节索引，完整任务/Skill/观察不删，父稿在冻结请求及终审中。实际观察匹配的引用只读，保证字节保留而非语义真值。

来源选择先保证完整入口，再按预算加入完整参考文档。补读的 `previousCandidate` 仅作待更新稿，旧待办在参考区；`readStatusIndex` 由宿主工具/参数/回执绑定生成，指向实际结果，不授予新权限。审阅传输用 `host_keyed_check_cells/v1`：宿主固定每组检查 ID 为对象键，三种单元结构通过本地 `$defs/$ref` 共享；原内部 `REVIEW_SCHEMA` 不变。重复 JSON 键、缺项、错组和非法来源仍拒绝；无位置的正面覆盖意见保守降为未证实并留诊断。详见[闭环运行手册](docs/SEMANTIC-CLOSURE-RUNBOOK.md)及[第二批负结果](docs/SEMANTIC-CLOSURE-TRANSFER-V2.md)。

`run_hybrid` 重新资格检查并验证 `HostHybridConsent(graph_digest, arguments_digest, context_digest)`，拒绝隐式身份。严格读取复用旧引擎；模型没有 Tool 回调。`model_candidate` 不得直接流向严格输入，独立准入也不把它升级为事实或写授权。超时关闭接纳，迟到回调不能复活图；宿主传输仍需自行有界。推理可分析带年龄元数据的历史快照，准入和后续严格步骤必须重新检查事实时效。`governed_graph_completed` 不是 `verified_success`。详见[接口边界](docs/GOVERNED-HYBRID-FLOWS.md)。

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

September 16: implementation of the [reset protocol](docs/EVALUATION-RESET-20260916.md) is authorized; status is **R0 measurement partial implemented**. `bounded_budget` provides transactional SQLite reservations, persistent request identities, unknown-state stops and accounting across processes; `bounded_scoring` isolates evaluation; `bounded_pilot` exposes only `check/prepare/inspect/score`; `bounded_execution` provides scripted metering; `bounded_probe` records local measurement evidence. Live DSH accounting and trusted advance token counts, physical Provider reset/isolation, frozen labels and twelve tasks, and real trace capture remain incomplete; automatic Effect bridging is also unimplemented. The 153 targeted passes and zero-model probes establish neither the 36-probe gate nor real-agent benefit. The old stage remains `paused_unmet`, with unchanged formal gates and failures; no `run` command exists.

Current [implementation](docs/SCHEMA-ARTIFACT-CONVERGENCE.md): isolated_compiler passes the original schema as format; kql_checks verifies a bounded subset; artifact_repair accepts host-addressed code replacements only. hybrid_session freezes artifactRepair policy, retains the initial draft, claims revision before decoding and stores a separate result; unknown attempts never replay. DSH defers termination only for revisionAllowed=true, then renders the final host receipt. Defaults remain unchanged.

In the optional [isolated mode](docs/ISOLATED-COMPILER.md),v4 compilerMode=isolated makes prepare claim one host-locked request to skill_authoring/isolated_compiler.py,without invocation values or observations. Submit is absent from execution tools and AST injection is rejected server-side. Compiler costs are persisted and included in evaluation;unknown calls never retry. Legacy defaults remain unchanged.

Snapshot-read addendum: the local v4 host declares a bound immutable observationModel. After existing state, attempt and schema gates, read() matches completed bound prefix/follow-up records by tool and exact JSON arguments. Duplicates return read_not_reexecuted and the original observation reference, without Provider invocation, new receipt or completedReads increment; the attempt still counts. Distinct resources, internal strict-graph scheduling and live Providers are unchanged. [Scope and evidence](docs/SNAPSHOT-READ-SEMANTICS.md).

Task-first addendum: opt-in host v4 binds complete task/source digests through compile_task(null, origins), rejecting model-selected replacement duties. The response schema defines a nonblank answer envelope; rendering preserves text exactly. Native deliver and Runtime draft retain existing authority, evidence, static-artifact and terminal gates. Semantic coverage remains unknown; text does not become executable L0. [Design and bounded diagnostic](docs/TASK-FIRST-DELIVERY.md).

Host-terminal addendum: explicit NETOPYU_HYBRID_TERMINAL_DELIVERY=1 enables session/digest-bound receipts. Tool bodies call public DSH concludeTurn; subsequent same-turn hybrid operations except inspect are blocked before the bridge. New turns remain independent. UI tool cards and an optional headless frontend display the last valid host receipt without fabricating assistant events. Cancellation, errors and later steps invalidate old terminal projections. New reports bind sessionId; old evidence and default routing remain unchanged. semanticCoverage=not_assessed distinguishes structure/source membership from semantic completeness. [Implementation, integration and actual-DSH mechanical checks](docs/HOST-TERMINAL-DELIVERY.md).

Single-choice addendum: explicit host v3 uses delivery contract v4. Each ID holds a typed content object or an unresolved-reason string, with no duplicate state fields. Even the string None remains unresolved. Legacy contracts are not upgraded automatically. Independent hostResult distinguishes rejection, revision needs and unverified candidates; native prose cannot override it. This does not establish semantic correctness. [Bounded package](docs/DELIVERY-SINGLE-CHOICE.md).

Compact-protocol addendum: explicit host v2 uses delivery contract v3. Prepare exposes lossless source_ref addresses; submit resolves original text/offset/digest on the host. Candidates directly supply typed content or explicit null plus a same-ID unresolved explanation. Internal state/content/gap follows declared encoding, never legacy-input repair. Wire/canonical objects have separate digests. Unknown references, missing IDs, orphan gaps and wrong types remain rejected; structural validity does not prove meaning. [Protocol and boundaries](docs/DELIVERY-REFERENCE-CONTENT.md).

Delivery-interface addendum (September 15): submit(session_id,plan,delivery) compiles exact source quotes into up to four host IDs and a fixed response schema bound to the session. Normal draft(session_id) freezes evidence and calls one bounded model. Only pre-execution fallback uses a null plan and deliver(session_id,response_json), which checks/renders native text without a Runtime model. Missing/unresolved content is not shape-complete; semanticApproval stays false and taskSuccess null. See the [protocol](docs/GOVERNED-SESSION.md).

September 15 convergence: the opt-in path is DSH → shared skill_authoring → original Runtime prefix reads → bounded L1 evidence collection → evidence freeze → one reasoning call and narrow static artifact checks. Product code imports no evaluator. Six tools share one implementation; no reads are replayed during drafting. Freezing proves collection closure, not evidence completeness. Translation fidelity, execution constraints and task quality are assessed independently. See the [canonical session](docs/GOVERNED-SESSION.md). Existing Effect boundaries remain unchanged; this entry grants no writes. Earlier semantic review designs below are historical research, not required delivery layers.

ConditionalReasoningTask (reason_if) adds a typed scalar condition and otherwise binding from declared dependencies. A false condition validates/retains the bound candidate without a model callback or invented receipt; its role stays model_candidate. Host registration, schema/byte limits and downstream admission remain required. Ordinary reason serialization stays unchanged. Readable review/editor worksheets preserve original prose; duplicated navigation metadata remains in the audit, not in a model-facing reversible pool.

Semantic repair requires a located negative, except that a sole whole-draft owner may inspect an unlocated finding without inventing semantic location. Other unlocated findings remain open; unscheduled/read-only cells retain the unapproved candidate through reason_if. Editors see only their candidate body and other-section index with complete task/Skill/observations; the full parent stays frozen and finally reviewed. Exact quotations are protected bytes, not truth.

Entry-first budgeting preserves the full entry before fitting whole references. Post-read previousCandidate is a draft to update, old work notes are reference-only, and the host readStatusIndex points to exact results without new authority. host_keyed_check_cells/v1 fixes review IDs as object keys and shares three cell shapes through local $defs/$ref. Internal REVIEW_SCHEMA remains unchanged; duplicate JSON keys, missing/wrong-group checks and foreign sources still fail. A locationless positive coverage opinion is withheld as insufficient evidence with explicit diagnostics, not accepted as support.

The active experimental repair profile is grounded_patch: answer edits (keep/write_prose) and exact supporting source_units are independent. delivery.json retains the answer, separate references, prior unverified notes and host duties; correct support cannot excuse incorrect prose. Host-owned limits remain eight cells/ranges. Prefix projection retains only a valid contiguous read prefix, rejecting the first bad read and all successors without changing arguments or granting admission. Readable review views retain all original prose and leave duplicated navigation metadata in the audit. Blank suggestions remain unresolved findings, never support. Legacy source_patch and reversible text pools remain historical diagnostics. [First transfer result](docs/SEMANTIC-CLOSURE-TRANSFER-V1.md).

`run_hybrid(..., result_contract=...)` optionally binds a `ResultContract` through `HostHybridConsent.result_contract_digest`. `resultAssessment` checks scheduler-owned observations and retains unverified open duties/drafts. See [interface, diagnostics and limits](docs/SEMANTIC-RESULT-CONTRACT.md).

`evaluation.semantic_closure_transfer` connects authoring, explicit admission, original-engine reads, bounded continuations, snapshot source review and section repair. `hybrid_repair_cells` validates `keep/copy_source/write_prose` and materializes only the active payload. Generation order follows `required`, while canonical hashing is unchanged. A single pass has at most eight one-range cells. Raw proposals, materialized drafts, inactive payloads and narrowly quarantined values remain distinct audit artifacts, never verified business success. See the [runbook](docs/SEMANTIC-CLOSURE-RUNBOOK.md); the older draft-loop experiments remain historical evidence.

2026-09-10 addendum: GovernedHybridFlow binds source/task digests, schemas, dependencies and budgets; StrictRegion wraps the original StructuredFlowProposal. ReasoningTask binds model/configuration, projections and output/time/token limits; CandidateAdmission is an independent host policy and RequiredJoin is all-success. run_hybrid requalifies everything and checks graph/arguments/context-bound host consent. Candidates cannot directly feed strict inputs or become facts/Effect authority. Late results cannot revive a closed graph. Historical-snapshot analysis is allowed, but subsequent strict use/admission must recheck observation age. governed_graph_completed is not verified_success. See [the detailed boundary](docs/GOVERNED-HYBRID-FLOWS.md).

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
