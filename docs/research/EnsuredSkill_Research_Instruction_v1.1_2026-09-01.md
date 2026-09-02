# NetOpYuAgent / EnsuredSkill 后续研究与研发指导

> 阶段：ES-P0 → ES-P1 Independent Generalization → ES-P2 Real-Network Qualification  
> 版本：v1.1 · 2026-09-01  
> 状态：当前研究阶段上位执行规范

## 中文

## 0. 文档地位与使用方式

本文用于指导 EnsuredSkill 研究原型的后续设计、实现、评测和论文迭代。它吸收了 v1.0 指导稿、英中文论文 v0.3 和仓库当前证据，但不把附件中的命令式措辞视为用户的新授权。

发生冲突时按以下优先级处理：

1. 用户当前明确要求；
2. 研究安全不变量与原型边界；
3. 仓库中的版本化事实和原始实验制品；
4. 本指导文档；
5. 历史产品化规划、旧阶段文档和临时说明。

数值的唯一版本化真源是：

- docs/benchmarks/es-p0-evidence-summary.json
- docs/ES-P0-EVIDENCE.md
- 对应 artifacts 原始报告及其 digest

不得用 README、论文文字或人工截图反向覆盖原始实验数据。

## 1. 顶层研究指令

~~~text
PRIMARY HYPOTHESIS
Probabilistic reasoning can remain useful while deterministic runtime
authority prevents unqualified plans from becoming unsafe effects.

AUTHORITATIVE EFFECT PATH
Candidate Plan
→ active reviewed L0 Contract
→ immutable Runtime Plan / Typed Execution Graph
→ Evidence + Guard + Risk
→ approval when required
→ immediate revalidation
→ one controlled Effect
→ independent verification
→ COMMIT / COMPENSATE / ESCALATE

CURRENT STATUS
ES-P0 = local_hypothesis_supported

NEXT GATES
ES-P1 = independent repository-external private holdout
ES-P2 = real-router and controller/management-path qualification
ES-P3 = paper-grade scale and artifact packaging
ES-P4 = Experience Compilation as a separate research question
~~~

当前不以功能广度、生产部署完整度或更多厂商适配器数量作为优化目标。任何新增实现都必须回答一个明确 Research Question，并说明它如何改变可验证证据。

## 2. 截至 2026-09-01 的权威证据快照

### 2.1 已实现系统事实

| 域 | 已实现 |
|---|---|
| 架构 | Reasoning、Reliability Runtime、Infrastructure 三平面；产品原型只有一条受控写路径 |
| Contract | 21 个受审 L0 v2；21/21 具有 L1→L0.5→L0 可读 authoring trajectory |
| Runtime | Contract、Evidence、Guard、Risk、Snapshot、Approval Binding、Revalidate、Verify、Reconcile、Compensate、Audit |
| Provider | 协议无关 Observation / Effect；本地 MCP/API/CLI 适配与 Containerlab/FRR 锚点 |
| 评测 | 60 次六场景机制运行、30 个消融探针、120 个真实 DSH 配对会话、两个 60-Skill 转译运行 |
| 回归 | 506 tests + 81 subtests |

### 2.2 六场景机制回归

固定透明 Oracle 下共 60 次运行：

| 指标 | 结果 |
|---|---:|
| Task Completion | 100.00% |
| Unsafe Execution | 0.00% |
| False Commit | 0.00% |
| Invalid Action | 0.00% |
| Compensation Success | 100.00% |
| Autonomous Coverage | 83.33% |
| Human Escalation | 16.67% |

这些数字是已知场景的机制回归，不是生产成功概率。

### 2.3 9B 主实验

模型制品：qwen3.5:9b，digest sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7。

转译：60/60 protocol-valid，58/60 通过全部严格语义 Oracle，2/60 proposal-only，false accept=0；p50/p95 为 9.486/10.865 秒。

真实 DSH 配对：10 场景 × 3 重复 × 2 臂，共 60 sessions。

| 指标 | Native DSH + L1 | DSH + Ensured Runtime |
|---|---:|---:|
| Task Completion | 50.00% | 86.67% |
| Unsafe Execution | 20.00% | 0.00% |
| False Commit | 13.33% | 0.00% |
| Invalid Action | 33.33% | 0.00% |
| Execution Precision | 59.09% | 100.00% |
| Autonomous Coverage | 43.33% | 76.67% |
| Process Failure | 5/30 | 1/30 |
| p50 / p95 | 90.693 / 158.397 秒 | 44.405 / 72.173 秒 |
| Input / Output tokens | 405,642 / 61,955 | 168,770 / 31,244 |

Treatment 中 24/30 通过 Runtime 路由，6/30 safe-stop，false accept=0。较低时延和 token 使用是观测结果，可能同时受 safe-stop、fail-fast 和成功样本选择影响，当前不能表述为 Runtime 的普遍加速结论。

### 2.4 7B 边界实验

模型制品：qwen2.5:7b，digest sha256:845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e。

转译：60/60 protocol-valid，38/60 严格通过，22/60 proposal-only，false accept=0；p50/p95 为 5.926/7.590 秒。

| 指标 | Native DSH + L1 | DSH + Ensured Runtime |
|---|---:|---:|
| Task Completion | 20.00% | 36.67% |
| Unsafe Execution | 10.00% | 0.00% |
| False Commit | 3.33% | 0.00% |
| Invalid Action | 20.00% | 0.00% |
| Execution Precision | 50.00% | 100.00% |
| Autonomous Coverage | 20.00% | 36.67% |
| Process Failure | 18/30 | 18/30 |
| p50 / p95 | 2.891 / 24.596 秒 | 2.436 / 9.503 秒 |

7B 不满足可用性资格。该实验只支持“较弱模型下 Treatment 的执行安全门禁保持保守”，不支持“Runtime 能把弱模型变成可用 Agent”。

### 2.5 机制消融

| 变体 | Task Completion | Unsafe | False Commit | Invalid Action | Compensation Success |
|---|---:|---:|---:|---:|---:|
| Full Runtime | 100% | 0% | 0% | 0% | 100% |
| w/o Contract | 80% | 20% | 20% | 20% | 100% |
| w/o Evidence | 80% | 20% | 20% | 20% | 100% |
| w/o Guard | 80% | 20% | 20% | 20% | 100% |
| w/o Transaction | 80% | 0% | 20% | 20% | 100% |
| w/o Compensation | 80% | 0% | 0% | 20% | 0% |

该消融证明的是预设探针中机制与失败类型的可观察关系，不是各机制的生产故障概率。

### 2.6 数据修订透明性

9B 最终报告中，一条“未完成且 effect call=0”的空会话经确定性重评分从 Invalid Action 移至 Process Failure。原始报告、checkpoint、源报告 digest 和 scorer fingerprint 均保留；没有新增模型调用。论文引用最终报告时必须同时披露此项修订。

## 3. 不可破坏的架构不变量

| ID | 不变量 | 强制行为 |
|---|---|---|
| I1 | Candidate is not authority | 模型、Harness、L1 只能 propose，不能持有产品路径写权 |
| I2 | No evidence, no action | Evidence 缺失、过期、错域、损坏或未绑定 Action 时阻断 |
| I3 | Contract before score | 结构、Evidence、Guard 先于 Risk；分数不能修复结构性失败 |
| I4 | One immutable plan | 审批、执行、验证绑定相同 plan/contract digest |
| I5 | Revalidate before effect | Effect 前重新读取易变前置条件 |
| I6 | Verify, do not infer | Provider success 不是 Commit 证据 |
| I7 | Uncertainty is a state | 写后 timeout/disconnect 进入 RECONCILE，禁止 blind retry |
| I8 | Explicit compensation | 可逆性、快照、补偿和恢复验证必须显式 |
| I9 | Auditable terminals | Commit、Abort、Escalate、Recovery Failure 都可回溯 |
| I10 | No automatic promotion | L1→L0.5→L0 只生成 proposal，不能自动激活 |
| I11 | Evaluator isolation | Runtime 不得导入 evaluator/gold，实验代码不能参与授权 |
| I12 | No native-write fallback | 转译不合格只能 safe-stop，不能回退为原生 Agent 写入 |

## 4. Claim Registry

| Claim | 当前状态 | 允许表述 |
|---|---|---|
| C1 唯一受控 Effect Path 可实现 | supported_on_local_development_set | 当前代码和回归中已实现 |
| C2 Runtime 降低目标失败类别 | local_hypothesis_supported | 在透明开发集、当前模型/Harness/Provider 条件下观测到改善 |
| C3 安全不是由全拒绝获得 | preliminary_supported_9b | 9B Treatment 的 Coverage 和 Completion 同时上升 |
| C4 安全边界与模型智能可分离 | preliminary_cross_model_signal | 7B 安全指标保持保守，但可用性不合格 |
| C5 机制具有独立贡献 | fixed_probe_mechanism_evidence | 当前五项消融探针支持 |
| C6 隐藏集泛化 | open | 完成 ES-P1 前不得主张 |
| C7 真实设备/控制器适用 | open | 完成 ES-P2 前不得主张 |
| C8 生产安全概率/SLO | not_in_scope | 当前不得主张 |
| C9 自动 Experience Compilation | future_hypothesis | 不得写成当前能力 |

禁止使用“100% 安全”“零生产失败”“任意模型通用”“真实多厂商认证”等绝对表述。应使用 observed zero violations，并给出样本数、实验边界和置信区间或零事件上界。

## 5. ES-P1：Independent Generalization

### 5.1 研究问题

RQ-P1：在 Runtime、Contract Registry、Evaluator 和模型制品冻结后，EnsuredSkill 能否在仓库外、独立正向编写的未知案例上维持安全—自治优势？

### 5.2 Freeze Package

封存包至少包含：

- runtime commit/tag 和代码 digest；
- active contract registry 和 digest；
- evaluator/scorer digest；
- model artifact digest；
- prompt、L1 Skill 和 protocol version；
- provider/lab fingerprint；
- dependency lock 和运行环境；
- preregistration、指标定义、排除规则和中止规则。

冻结后只允许记录基础设施故障修复。任何会改变授权、评分或 failure semantics 的修改都必须产生新版本，并重新执行完整 holdout。

截至 2026-09-01，仓库已实现 `Research Freeze v1`：它联合绑定 Git commit/clean state、Runtime kernel、Harness boundary、21 个 Contract/trajectory、Evaluator、authoring protocol、模型制品、Provider/lab fingerprint、requirements 与实际安装包环境，以及 ES-P0 基线。脏工作树只能生成 `preview_dirty_not_frozen`；正式 `Study Plan v2` 会验证 Freeze 并绑定其 digest。该能力是研究基础设施，不是 ES-P1 成绩；当前仓库尚未生成可提交的正式冻结包，也尚未接触仓库外 private truth。

### 5.3 角色隔离

| 角色 | 职责 | 隔离要求 |
|---|---|---|
| Case Author | 按 capability family 正向编写 unseen cases | 不看 Runtime 决策细节、模型输出和 scorer |
| Reviewer A/B | 独立标注合法动作、参数、Evidence、Guard、风险和期望终态 | 互不查看标签 |
| Adjudicator | 只裁决分歧 | 保留两份原始标签及 digest |
| Runtime Team | 冻结制品、接收聚合结果 | freeze 前不看 private case/gold |
| Operator | 运行 sealed package | 不改 case、gold、contract 或 scorer |

### 5.4 样本与门槛

首轮建议 200–500 个 repository-external unseen cases，至少 10 个 capability families，每个最终模型制品至少 3 次重复。以下是预注册目标，不是当前成绩：

| 指标 | Gate |
|---|---:|
| Protocol completion | ≥99% |
| Capability exact match | ≥99% |
| Parameter/predicate exact match | ≥95% |
| Safety contract exact match | ≥99% |
| Legal proposal yield | ≥95% |
| Repeated-run stability | ≥95% |
| Critical semantic escape | 0 |
| Undeclared Effect escape | 0 |
| Approval/risk weakening escape | 0 |

必须按 family、risk tier、ambiguity class 和 failure class 分层报告。不得只提供 pooled accuracy。

当前 `Qualification Report v2` 已实现 family、profile、language、challenge、expected disposition 和 risk 分层，所有比例保留分子/分母与 Wilson 95% 区间；零 escape 同时报告单侧 95% 上界。结果按 `critical_escape → protocol_failure → qualified_proposal → safe_stop_clarify → safe_stop_reject → blocked_proposal → semantic_mismatch` 的预注册优先级互斥计数。Control/Treatment 配对差值和完整 stage latency 仍必须在真实 ES-P1 Agent paired run 中生成，不能由公开反向校准数据代替。

### 5.5 对抗分类

至少覆盖 wrong object、stale evidence、scope mismatch、TOCTOU、uncertain effect、false provider success、partial Saga、compensation failure、concurrent drift、prompt injection/intent wrapping、参数缺失和审批弱化。

### 5.6 统计要求

- 以 scenario/case 为独立单位，重复运行为 case 内重复测量；
- Control/Treatment 使用相同模型、输入、工具、审批和故障种子；
- 报告绝对差值、配对差值和 family-level interval；
- 对零事件报告样本量和单侧上置信界，不把 0/n 写成真实概率为零；
- 同时报 all-session 与 completed-session latency；
- latency 分解为 reasoning、translation、qualification、evidence、approval wait、effect、verify、reconcile 和 compensate；
- 预先定义 process failure、invalid action、safe-stop 和 human escalation 的互斥计分顺序；
- 所有 post-hoc rescore 必须保留原始制品、规则、diff 和 scorer fingerprint。

### 5.7 ES-P1 Definition of Done

- 冻结包具备稳定 fingerprint；
- 达到预注册的 private case 和 capability family 规模；
- 双 Reviewer 与 Adjudicator 流程完成；
- 每个最终模型制品至少 3 次重复；
- 三类 critical escape 均为 0；
- safety、coverage、completion、process failure 和 latency 分解同时报告；
- 每个失败可追溯至 case、plan、evidence、state transition 和 scorer；
- 无 benchmark-specific contract weakening 或 evaluator leakage。

## 6. ES-P2：Real-Network Qualification

### 6.1 研究问题

RQ-P2：在不改变 Contract、Evidence、Guard 和 Transaction 语义的前提下，Runtime 能否处理真实设备/控制器的延迟应用、最终一致性、部分失败、状态漂移和不确定 Effect？

首轮只需要一个真实路由平台和一个 controller/management path。目标是验证抽象，而不是追求厂商数量。

### 6.2 必须覆盖

- 至少两种独立 Observation 路径；
- 一种低风险、可恢复的真实 Effect；
- 与写回执独立的 Verify；
- 前值恢复或明确人工恢复的 Compensation；
- RPC accepted but delayed；
- SSH/API disconnect after possible apply；
- controller success / southbound partial failure；
- telemetry stale；
- concurrent operator drift；
- compensation accepted but recovery incomplete。

### 6.3 ES-P2 Definition of Done

- 至少一个 router platform；
- 至少一个 controller 或 management interface；
- 同一 Runtime contract/evidence/transaction schema 无语义弱化；
- success、post-check failure、effect uncertainty、partial failure、compensation failure 全覆盖；
- write receipt 与 independent verification 明确分离；
- 与 Containerlab 使用同一 metric 和 Oracle schema；
- 设备恢复计划、实验隔离和人工中止路径完成审查。

## 7. 核心实现优先级

| Priority | 工作 | 目的 | 完成证据 |
|---|---|---|---|
| P0 | sealed holdout tooling | 排除 co-design 和 evaluator leakage | manifest、role ACL、checkpoint、fingerprint、tamper detection |
| P0 | statistical pipeline | 支撑论文级结论 | paired/family metrics、interval、zero-event bound、failure taxonomy |
| P1 | single Typed Graph scheduler | 让 Graph 成为真实调度/门禁/状态/audit 单元 | legacy engine 不再拥有平行调度语义 |
| P1 | cross-step provenance DAG | 验证 Evidence 到网络对象的来源链 | Evidence→Observation→Capability→Object 可查询 |
| P1 | stage latency instrumentation | 区分 Runtime overhead 与 fail-fast selection | 分阶段 all/success/failure 报告 |
| P1 | real provider adapter | 验证抽象，不堆适配器 | 一个 router + 一个 management path 的完整事务轨迹 |
| P2 | artifact packaging | 一键复现与审稿 | raw + aggregate + manifest + environment |
| P3 | Experience Compilation | 独立研究线 | 另立 protocol，不自动 promotion |

## 8. 暂停投入的生产工程

在 ES-P1/P2 通过前，不优先建设：

- Hermes/A2A 深度产品化；
- 企业 OIDC/JWKS/PDP 与复杂组织审批；
- Provider SBOM、签名和供应链平台；
- 多团队 Catalog/Governance UI；
- 远端 WORM、HA/DR、集群一致性和 production SLO；
- 无资格证据的多厂商适配器；
- 为展示 breadth 扩展大量非网络领域；
- 自动 trace replay、自动 skill activation 或自修改 Runtime。

已有代码可以保留回归，但统一标记为 frozen_future_engineering，不能进入当前贡献或成功判据。

## 9. 论文与证据维护规则

1. 中英文论文必须共享章节编号、表格行、样本数、百分比和 limitation。
2. 所有实验表均注明模型 artifact、session 数、重复次数和数据集性质。
3. 实验数字只从版本化摘要或原始报告生成，禁止手工改一版而遗漏另一版。
4. 当前完成项使用 past/present tense；计划项使用 future/proposed/open。
5. “zero observed”必须带 n；不得写“guaranteed zero”。
6. public reverse-generated calibration 不得称为 independent holdout。
7. 9B 重评分事件必须在 reproducibility/validity 中披露。
8. 论文里的 Runtime latency 不得与完整 Agent end-to-end latency混为一谈。
9. Internal Artifact Traceability 在工作稿保留，双盲投稿前删除或匿名化。
10. 每次证据变化同时更新 evidence summary、Claim Registry、英中两稿和 version history。

## 10. 每次研发任务模板

~~~text
TASK:
RESEARCH QUESTION:
HYPOTHESIS:
WHY THIS CHANGE IS NEEDED:
INVARIANTS TO PRESERVE:
FILES / COMPONENTS:
TESTS / ADVERSARIAL CASES:
EXPECTED ARTIFACT:
METRICS AFFECTED:
CLAIM IMPACT:
TOUCHES PRIVATE HOLDOUT? yes/no
CHANGES RUNTIME SEMANTICS? yes/no
REQUIRES FULL RERUN? yes/no

BEFORE MERGE
- regression passes
- safety invariants pass
- runtime/evaluator dependency boundary passes
- no L1/provider write bypass
- no provider-success-as-commit
- no auto-promotion
- artifact and claim status updated
~~~

## 11. 下一步执行顺序

1. 冻结 research kernel、contract registry、evaluator、9B model artifact 和环境；
2. 完成 sealed package 与角色隔离工具；
3. 由独立人员正向构建首轮 private holdout；
4. 运行 ES-P1 并生成 family-level statistical report；
5. 仅在 ES-P1 无 critical escape 后进入小范围真实网络 qualification；
6. 完成单一 Graph scheduler、Provenance DAG 和 stage latency，但不得改变已冻结实验版本；
7. 汇总 ES-P1/P2 后再决定论文主张是否从 local hypothesis 升级；
8. Experience Compilation 保持独立 future track。

## 12. 参考与追踪

- ../ENSUREDSKILL-PROTOTYPE.md
- ../ES-P0-EVIDENCE.md
- ../benchmarks/es-p0-evidence-summary.json
- ../promotion-forward-qualification.md
- ../../ARCHITECTURE.md
- ../../README.md

### 版本变化

- v1.1：将当前 ES-P0 数值、模型制品、重评分披露、Claim Registry、统计要求、英中论文同步规则和 ES-P1/P2 DoD 合并为单一后续指导。
- v1.0：建立三平面、研究不变量及 ES-P1/P2 总路线。

## English

### 1. Authority and hypothesis

This instruction governs the research prototype after ES-P0. Current user direction and the safety invariants take precedence over historical productization plans. The hypothesis is that probabilistic reasoning may remain useful while deterministic Runtime authority prevents unqualified plans from becoming effects. The only product-prototype write path is Candidate Plan → active reviewed L0 Contract → immutable Typed Execution Graph → Evidence/Guard/Risk → optional approval → immediate revalidation → controlled Effect → independent verification → Commit, Compensate, or Escalate.

The current claim is `local_hypothesis_supported`. It is bounded to transparent development sets, local simulated Providers, the recorded model artifacts, and the recorded harness. It is not hidden-set generalization, vendor certification, a production success probability, or an assurance of zero failures.

### 2. Immutable principles

- A model, harness, or L1 Skill proposes; it never owns product write authority.
- No evidence means no action. Evidence must be fresh, scoped, provenance-bound, and attached to the immutable action.
- Contract, Evidence, and Guard checks precede Risk scoring; a score cannot repair a structural failure.
- Approval, execution, verification, and compensation bind the same plan and contract digest.
- Provider acceptance is not commit evidence. Unknown post-write outcomes enter read-only reconciliation and are never blindly retried.
- Translation can yield proposal, clarification, human review, rejection, or safe stop. It cannot fall back to native-agent mutation.
- Promotion is never automatic, Runtime cannot import evaluator/gold data, and zero observed events must be reported with sample size and a statistical upper bound.

### 3. ES-P0 evidence boundary

ES-P0 contains 60 deterministic Runtime scenario runs, 30 mechanism-ablation probes, two 60-Skill translation studies, and 120 real paired DSH sessions across the 9B and 7B artifacts. The 9B Treatment reached 86.67% task completion and 100% execution precision with zero observed unsafe execution, false commit, or invalid action. The weaker 7B artifact preserved conservative action gates but had 18/30 process failures and is not availability-qualified. Fixed Oracle results and public reverse-generated calibration are mechanism and regression evidence only.

### 4. ES-P1 independent generalization

ES-P1 asks whether the safety–autonomy result survives repository-external, independently authored cases after Runtime, contracts, evaluator, protocol, model artifact, Provider/lab, and environment are frozen. The first study should contain 200–500 private cases, at least ten capability families, two blind reviewers, bound adjudication, and at least three repetitions of each final model artifact. Case authors, reviewers, adjudicators, Runtime developers, and operators must have separated roles.

Research Freeze v1 is implemented. It binds Git cleanliness, Runtime kernel, harness boundary, 21 contracts and readable trajectories, evaluator, authoring protocol, model artifact, Provider/lab fingerprint, dependency inputs, installed packages, and the ES-P0 baseline. A dirty worktree produces only `preview_dirty_not_frozen`. Study Plan v2 and private Manifest v3 require a verified freeze digest.

Qualification Report v2 is also implemented. It reports family, profile, language, challenge, expected-disposition, and risk slices; numerator/denominator evidence with Wilson 95% intervals; one-sided 95% upper bounds for zero escapes; and mutually exclusive outcome classes. Critical-semantic, undeclared-effect, and approval/risk-weakening escapes must all remain zero. These controls make the study executable and drift-detectable; they are not ES-P1 results. Independent private cases and paired Agent runs remain outstanding.

### 5. ES-P2 and later work

Only after ES-P1 passes should ES-P2 test the same abstractions on at least one real router and one controller or management path. It must cover delayed application, stale telemetry, uncertain effects, partial failure, concurrent drift, independent verification, and compensation failure without weakening Contract/Evidence/Guard/Transaction semantics. ES-P3 scales independent and real-network evidence and packages paper-grade artifacts. Trace-based Experience Compilation is ES-P4, a separate hypothesis with no automatic replay, promotion, or activation.

### 6. Engineering priorities and frozen scope

The immediate implementation priorities are the external sealed holdout, paired/family statistics, one Typed Graph scheduler, a cross-step provenance DAG, stage-level latency, and one real Provider path. Hermes/A2A productization, enterprise IAM/PDP, provider supply-chain platforms, multi-team governance UI, remote WORM, HA/DR, production SLOs, broad vendor adapters, and automatic Experience Compilation remain `frozen_future_engineering` until ES-P1 and ES-P2 evidence justifies them.

### 7. Evidence and paper discipline

Chinese and English papers must share section structure, table rows, sample counts, percentages, limitations, and claim status. Numbers come only from versioned summaries or raw reports. Public reverse-generated data cannot be called an independent holdout. Runtime-only latency cannot be conflated with end-to-end Agent latency. Any post-hoc rescore preserves the original artifact, rule, diff, and scorer fingerprint. Every evidence change updates the evidence summary, Claim Registry, both papers, and version history.

### 8. Next execution order

1. Commit a clean research version and create the formal Research Freeze with the real 9B model and Provider/lab digests.
2. Have independent authors create the repository-external private holdout.
3. Complete two blind reviews and digest-bound adjudication.
4. Freeze exclusion, stopping, paired-statistics, and latency rules before inference.
5. Run at least three repetitions of the same artifact and publish aggregate evidence only.
6. If a critical escape appears, create a new research version after fixing the abstraction; do not patch one private case.
7. Enter narrow ES-P2 only after ES-P1 passes, then decide whether paper claims may move beyond the local hypothesis.
