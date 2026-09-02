# EnsuredSkill 规格与安全设计 / Specification and Safety Design

> 规格基线 / Specification baseline: 2026-09-02。本文规定当前研究原型的可验证行为，不构成生产安全认证。

## 中文

### 1. 规格地位

本文定义 EnsuredSkill 网络可靠执行原型的功能、安全和验收要求。它不构成生产安全认证；绝对安全、100% 准确率和 100% 可用率都不是当前可证明结论。权威研究边界见[原型准则](docs/ENSUREDSKILL-PROTOTYPE.md)。

### 2. 信任边界

```text
不受信 / 概率性
  用户自然语言
  LLM output / confidence
  L1 natural-language Skill
  Candidate Plan
  Provider success text

受约束
  reviewed L0 Contract
  Runtime parameter compiler
  typed Evidence + Guard
  immutable plan + approval binding
  transaction state machine
  independent verifier / compensator
```

LLM、Prompt、L1 Skill、普通 Tool Schema 和 Provider 自报结果都不是安全根。Runtime 代码和人工审查的 L0 也可能有缺陷，因此需要合同测试、故障注入、消融和未来的数字孪生/形式化验证。

### 3. 功能要求

| ID | 要求 |
|---|---|
| FR-01 | Reasoning Plane 只能提交 Candidate Plan，不能直接获得 Effect 句柄或凭据 |
| FR-02 | 每个可写操作必须解析到唯一、已激活、版本化的 L0 Contract |
| FR-03 | 未知、缺失、歧义、越界或来源不明的参数必须追问或拒绝 |
| FR-04 | 每个 Effect 前必须满足 Preconditions、Evidence、Guards 和 Risk Policy |
| FR-05 | Evidence 必须绑定类型、来源、采集者、时间、范围、Action 和 payload digest |
| FR-06 | 高风险操作必须输出 ask-human 或 reject；审批与 exact plan hash 绑定 |
| FR-07 | 审批后、Effect 前必须重新验证易变前置条件和合同摘要 |
| FR-08 | Effect 最多发送一次；结果不确定时先进行只读 Reconcile |
| FR-09 | Commit 必须由独立 postcondition Observation 证明 |
| FR-10 | 验证失败时按合同补偿并独立验证恢复；无法证明时升级人工 |
| FR-11 | 每条终态路径必须记录 plan、Evidence、状态和摘要链 |
| FR-12 | L1→L0.5→L0 只能生成 proposal，不能自动激活 |
| FR-13 | Authoring 必须保存 requirement-level L1→L0.5→L0 映射、语义丢失分类和可定位的修正路径 |
| FR-14 | Harness 只能把 Runtime terminal envelope 作为执行结果；Provider receipt 或模型叙述不得变成成功终态 |
| FR-15 | 操作者必须能从 plan id 追踪 immutable plan、图节点、Evidence provenance、分阶段时延和事件摘要链 |

### 4. 安全不变量

#### S-01：无绕过写路径

产品原型的所有 Effect 必须经过 Runtime。隔离 A/B evaluator 可以让原生 DSH Agent 写本地仿真 Provider，但该代码不得被产品 Adapter 导入，也不得被描述为 fallback。

#### S-02：Evidence 不等于上下文

以下内容不得单独授权执行：模型 confidence、自然语言诊断、缓存 Prompt、Tool 返回的任意文本、用户声称已审批。缺 Evidence 时结果只能是 read/clarify/propose/ask/reject。

#### S-03：失败关闭

未知合同、未知 Capability、不可解析状态、摘要不一致、过期、超时、范围漂移、无法验证结果或恢复都不得转成成功。

#### S-04：独立验证

写工具的返回值最多证明“请求被接收/调用返回”，不能证明 desired state。Verifier 应使用独立 Observation；在仿真环境中也应与 Actor 效果接口分离。

#### S-05：不确定性不触发盲重试

调用超时必须区分 before-send 与 may-have-sent。后者进入 Reconcile；除非合同证明幂等并确认当前状态，否则不得重放。

#### S-06：补偿不是魔法回滚

合同必须声明 strongly reversible、conditionally reversible 或 irreversible。补偿是新的受控操作，需要 snapshot、precondition、verification 和失败终态；不能承诺恢复所有现实影响。

#### S-07：解释不能创造权威

模型可以解释为什么选择某个 Skill，但授权和终态只能来自结构化合同、计划、Evidence、Guard、审批、图状态与摘要链。Workbench、报告、自然语言 summary 和模型 confidence 都是投影；它们不能修改 active L0、PreparedPlan 或 ExecutionOutcome。

### 5. 威胁与控制

| 威胁 | 主要控制 | 剩余风险 |
|---|---|---|
| 模型选择错误 Skill/Tool | 有界候选、exact L0 binding、参数/证据/Guard | 合法但业务意图错误，需要更强 Evidence/人审 |
| 参数幻觉或猜目标 | provenance、inventory resolution、缺参追问 | CMDB/Observation 本身可能错误 |
| Prompt 注入要求绕审批 | Candidate 无权限、plan-bound approval、Runtime 独占 Effect | 本地模拟身份不代表企业不可抵赖身份 |
| 旧 Evidence 重放 | timestamp、freshness、scope、Action、digest | 时钟和采集器本身可能失真 |
| Provider 谎报成功 | 独立 verifier、result schema、fail closed | Observer 与 Actor 共故障风险 |
| 写后断连 | reconcile、idempotency policy、no blind retry | 无法观测时需人工 |
| 部分多步骤成功 | typed dependencies、snapshot、reverse compensation | 跨设备无法实现真正 ACID |
| 合同/补偿编写错误 | static validation、round-trip、negative tests、ablation | 仍需数字孪生和独立 review |
| 自动 Promotion 扩权 | proposal-only、semantic loss alert、explicit activation | Reviewer 可能误判 |
| 模型生成事后解释掩盖失败 | 固定 terminal envelope、Provider 原文摘要化、plan/evidence/graph 可追踪 | 操作者仍需理解 `rollback_verified` 不等于任务成功 |
| 评测过拟合 | 扰动集、封存集、重复、跨模型、场景级报告 | 本地实验不能外推生产概率 |

### 6. 数据最小化

计划和 Journal 只存运行所需的规范化参数、Evidence 摘要、状态、合同摘要和终态证据。模型密钥、设备密码、Bearer token、私钥和完整企业凭据不得进入 Prompt、L0、plan 或日志。当前 SQLite 只提供本机完整性和恢复；不宣称远端不可变或组织级审计。

### 7. 原型验收场景

| 场景 | 必须结果 |
|---|---|
| 正常可逆网络变更 | 满足证据与 Guard 后执行，独立 Verify 后 Commit |
| 缺失/过期 Evidence | Effect 未发送，明确 rejected/clarification |
| 高风险或不可逆变更 | ask-human 或 reject，不由模型 confidence 放行 |
| Effect 结果不确定 | Reconcile，不盲重试；无法证明则 Escalate |
| Verification mismatch | 不得 False Commit；补偿并验证恢复 |
| 多步骤部分失败 | 已执行步骤按依赖逆序补偿，保留完整轨迹 |

### 8. 评测规范

主实验的唯一变量是是否引入 EnsuredSkill：

```text
Control: DSH + same L1 Skill + LLM native tool orchestration
Treatment: same DSH/model/L1/tools/input/provider/faults + L0 auto Runtime
```

Control 只在隔离本地仿真中运行。Treatment 的转换未达阈值时不能回退为原生写入；应报告安全停机带来的 Autonomous Coverage 变化。

必须报告场景级与聚合指标：

- Unsafe Execution Rate；
- False Commit Rate；
- Invalid Action Rate；
- Compensation Success Rate；
- Autonomous Coverage；
- Human Escalation / Rejection Rate；
- Task Completion Rate；
- p50/p95 Runtime overhead、tokens 和 tool calls。

还必须运行五项消融（去掉 Contract/Evidence/Guard/Transaction/Compensation）、至少三次主实验配对重复、扰动/封存场景，以及 9B 与更弱模型的稳定性比较。固定 Oracle 的 100% 只能表述为“该回归集通过”。

证据必须按来源分层，不允许合并成一个“准确率”：

| 层级 | 当前状态 | 允许的 claim |
|---|---|---|
| ES-P0 透明本地开发集 | 完成 | 机制和本地假设得到支持 |
| 仓库外模型合成集 | 完成 | 生成、封存、fallback 和跨类型链路可运行 |
| ES-P1-Wild-Sim 虚拟角色公开集 | 完成 | 公开 Skill + DSH + Runtime 的角色隔离原型有效 |
| ES-P1 独立人工 Private Holdout | 未完成 | 当前不得声称独立隐藏集泛化 |
| ES-P2 真实网络 | 未完成 | 当前不得声称厂商设备或生产资格 |

每份报告还必须说明失败属于 Reasoning、转换/路由、Runtime 门禁、Provider、Oracle 还是 Harness transport，避免把安全停机、模型空响应和事务失败混为一类。

### 9. 原型出场门禁

截至 2026-09-02，以下条目已经形成 **ES-P0 本地研究原型闭环**，但这只允许进入独立泛化研究，不允许直接进入生产：

1. 代码、文档和测试只有一条产品原型 Effect 路径；
2. 六类事务场景均有可复算证据；
3. 五项消融完成并可解释各机制贡献；
4. 三次以上真实 DSH 配对结果稳定；
5. 更换模型后 Runtime 安全收益稳定；
6. 明确给出 Execution Precision 与 Autonomous Coverage 的 Pareto 权衡；
7. 结论明确区分透明开发集、模型合成集、角色模拟公开集和独立人工证据，不用单次演示外推。

下一道强制门是仓库外、预注册、独立人员持有 Gold 的 ES-P1 Private Holdout。只有 ES-P1 通过后才进入 ES-P2 小范围真实网络；只有 ES-P1 和 ES-P2 同时支持核心 claim，才重新评估生产身份、供应链、治理、HA/DR、WORM 与 SLO。

交互对象、终态和定位路径见 [Skill 与系统交互全景](docs/SKILL-SYSTEM-INTERACTION.md)，阶段状态见[项目进展](docs/PROJECT-STATUS.md)。

### 10. 冻结生产安全设计

企业 OIDC/JWKS/PDP/Change Authority、Provider 签名/SBOM/供应链、多人审批治理、远端 WORM、HA/DR、mTLS/secret manager 和生产 SLO 都是未来部署安全能力。它们可以继续保留回归，但不得掩盖当前研究问题，也不得增加核心原型完成条件。

---

## English

### 1. Scope and trust

This document specifies the EnsuredSkill network reliability prototype. It is not a production security certification and does not claim absolute safety, 100% accuracy, or 100% availability.

User language, model output and confidence, L1 text, Candidate Plans, ordinary tool schemas, and provider success text are untrusted. Reviewed L0 contracts, deterministic compilers, typed evidence and guards, immutable plan/approval bindings, transaction state, independent verification, and explicit compensation form the constrained execution boundary. The Runtime and reviewed contracts can still contain bugs and require testing and independent validation.

### 2. Required behavior

The Reasoning Plane has no effect credential. Every mutation resolves to one active versioned L0 contract. Missing, ambiguous, ungrounded, or out-of-range inputs clarify or reject. Preconditions, evidence, guards, and risk must all pass before an effect. Approval binds the exact plan. Mutable facts are revalidated after approval. Effects are not blindly retried. Independent observations prove commits. Verification failure compensates and verifies recovery, otherwise escalating. Every terminal path is auditable. Promotion remains proposal-only.

Authoring also retains requirement-level L1-to-L0.5-to-L0 mappings, loss classifications, and exact revision locations. The Harness accepts only the Runtime terminal envelope as an execution outcome. Given a plan id, an operator can inspect the immutable plan, graph nodes, evidence provenance, stage latency, and event-chain integrity.

### 3. Safety properties

There is no prototype write bypass. No evidence means no action. Unknown contracts, capabilities, states, digests, expiry, drift, or recovery fail closed. A write receipt is not postcondition evidence. Post-send uncertainty enters reconciliation. Compensation is a new controlled operation rather than magical rollback and carries an explicit reversibility class.

Explanation cannot create authority. Workbench views, reports, natural-language summaries, and model confidence are projections of contracts, plans, evidence, and graph state; they cannot mutate an active L0, PreparedPlan, or ExecutionOutcome.

### 4. Evaluation

The primary experiment compares native DSH L1 orchestration with the same DSH/model/L1/tools/inputs/provider/faults plus EnsuredSkill. Native mutation is an isolated local control only. An unqualified treatment conversion stops safely instead of falling back to native mutation.

Reports must include unsafe execution, false commit, invalid action, compensation success, autonomous coverage, escalation/rejection, task completion, overhead, token, and tool-call metrics. Five mechanism ablations, at least three paired repetitions, perturbation/sealed cases, and a weaker-model comparison are required. A fixed-set 100% result means only that the fixed regression passed.

Evidence is reported separately as transparent ES-P0 development evidence, repository-external model-synthetic evidence, virtual-role public-Skill simulation, independent-human private holdout, and real-network qualification. The first three are complete at their declared scope; formal ES-P1 private holdout and ES-P2 real-network evidence remain open. Failures must be attributed to reasoning, translation/routing, Runtime gates, Provider, Oracle, or Harness transport.

### 5. Exit and frozen production security

The local ES-P0 prototype gate is complete only at its transparent local scope. The next mandatory gate is a preregistered, repository-external ES-P1 private holdout with independently owned Gold. ES-P2 then qualifies the same abstractions on a small real-network path. Production engineering is reconsidered only after independent and real evidence support the core claims.

Enterprise IAM/PDP/change systems, signed provider/SBOM supply chains, multi-party governance, remote WORM, HA/DR, mTLS/secret management, and production SLOs remain frozen future deployment security.
