# 文档导航 / Documentation Map

## 中文

先看[考核重置与当前实施边界](EVALUATION-RESET-20260916.md)，再看实现和使用。实施已获授权，当前仅 R0 测量部分落地；153 项定向通过，已运行的本地探针为 0 模型，不是 Agent 收益或 36 项机制门槛通过。真实 DSH／Provider／trace 接线和样本标签仍待完成，尚无 `run` 命令；旧语义阶段仍 `paused_unmet`，正式门禁不变。旧实验的“当前／下一步”是历史快照；与[原型权威准则](ENSUREDSKILL-PROTOTYPE.md)冲突的生产工程计划不生效。

| 想了解什么 | 入口 |
|---|---|
| 为什么不能收敛；新的指标、对照、R0 部分实现、总预算和硬停止方法 | [考核重置与实施边界](EVALUATION-RESET-20260916.md) / [当前进展](PROJECT-STATUS.md) |
| 当前结构修复、有界工件修订、真实任务退步及停止点 | [结构与工件验收](SCHEMA-ARTIFACT-CONVERGENCE.md) / [指标与边界](benchmarks/schema-artifact-summary.json) |
| 最新编译/执行隔离、证据获取改善及剩余查询错误 | [上下文隔离与实际结果](ISOLATED-COMPILER.md) / [本包指标](benchmarks/isolated-compiler-summary.json) |
| 最新参数协议/必需证据门禁实现、9B失败与下一设计边界 | [调用协议与证据交付](CALL-PROTOCOL-EVIDENCE.md) / [本包指标](benchmarks/call-protocol-evidence-summary.json) |
| 两轮实际结果、根因、为何停止及下一步决策 | [两轮收敛与明确指标](CONVERGENCE-CLOSURE.md) / [本批指标](benchmarks/convergence-closure-summary.json) |
| 当前快照补读边界、两个实际DSH任务与Oracle问题 | [冻结快照读取](SNAPSHOT-READ-SEMANTICS.md) / [本轮证据](benchmarks/snapshot-read-summary.json) |
| 最新任务直达设计、两次真实9B诊断及未解决问题 | [任务直达交付](TASK-FIRST-DELIVERY.md) / [逐项诊断摘要](benchmarks/task-first-delivery-summary.json) |
| 当前证据先行/交付合同主链、DSH 六工具与真实负结果 | [Agent 主链与本地使用](GOVERNED-SESSION.md) / [单选协议及停止点](DELIVERY-SINGLE-CHOICE.md) / [最新结果](benchmarks/delivery-single-choice-summary.json) |
| 宿主交付后如何停止重复调用、展示结果，哪些问题仍未解决 | [终态机制修复与接入](HOST-TERMINAL-DELIVERY.md) / [机制证据，不是语义成功率](benchmarks/host-terminal-delivery-summary.json) |
| 项目设计、能力、性能、场景和使用 | [项目 README](../README.md) |
| 当前完成什么、下一步做什么 | [项目进展](PROJECT-STATUS.md) |
| 如何区分观察、未验证草稿和未完成职责 | [双驱动结果合同](SEMANTIC-RESULT-CONTRACT.md) |
| 草稿怎样发现遗漏、修订一次、保留未解决问题 | [双向审查与有界修订](BOUNDED-DRAFT-REVIEW.md) |
| 本轮为什么反复修复、什么条件才算结束 | [语义闭环根因](SEMANTIC-CLOSURE-DIAGNOSIS.md) / [固定出口](SEMANTIC-CLOSURE-EXIT.md) |
| 9B 换 27B 能否解决语义复核问题 | [四例对照、失败与完整成本](SEMANTIC-REVIEW-MODEL-CONTRAST.md) |
| 第一阶段当前改进与未通过原因 | [证据先定位、受限备注、最新 10 次真实调用与语义反例](SEMANTIC-DUTY-CONTRACT.md) / [上轮停止点](SEMANTIC-CONTEXT-REPAIR.md) |
| 为什么增加任务/证据见证仍不能自动批准 | [9B 实验、有效检出、误报与两项修复](SEMANTIC-WITNESS-DIAGNOSTIC.md) |
| 6 Skill／12 任务首次冻结验收为什么没通过 | [结果、失败与成本](SEMANTIC-CLOSURE-TRANSFER-V1.md) / [机器摘要](benchmarks/semantic-closure-transfer-v1.json) |
| 第二批 12 任务为什么仍未通过、修复什么 | [语义闭环第二批迁移](SEMANTIC-CLOSURE-TRANSFER-V2.md) / [机器摘要](benchmarks/semantic-closure-transfer-v2.json) |
| 双驱动实际效果、失败、性能与复现 | [阶段 2 最终开发报告](STAGE-2-HYBRID-RESULTS.md) / [机器摘要](benchmarks/stage2-hybrid-summary.json) |
| 阶段 1 的结果、成本和局限 | [阶段 1 验收报告](STAGE-1-RESULTS.md) |
| 阶段 2 的 10 个公开 Skill、输入和首次验证 | [阶段 2 开发迁移验证](STAGE-2-PUBLIC-TRANSFER.md) |
| 首批失败后的通用修复与剩余缺口 | [阶段 2 表示与导航修复](STAGE-2-REPRESENTATION-REPAIR.md) |
| L1 / L0.5 / L0 与系统怎样交互、怎样定位结果 | [交互全景](SKILL-SYSTEM-INTERACTION.md) |
| 当前转译代码从哪里看、怎样运行 | [研究代码导航](../evaluation/README.md) / [语义前端](SEMANTIC-PLAN.md) |
| 自然语言推理怎样融入严格流程 | [受控混合流程](GOVERNED-HYBRID-FLOWS.md) / [阶段 2 双驱动验收](STAGE-2-HYBRID-VALIDATION.md) |
| 架构、总体与详细设计 | [ARCHITECTURE](../ARCHITECTURE.md) / [HLD](../HLD.md) / [LLD](../LLD.md) / [SSD](../SSD.md) |
| 什么证据才能证明泛化 | [正式泛化门禁](TRANSLATION-GENERALIZATION-GATE.md) / [原型准则](ENSUREDSKILL-PROTOTYPE.md) |
| 原始 Skill、引用、脚本和宿主如何保真 | [输入保真](TRANSLATION-INTAKE.md) / [任务对齐](TASK-SOURCE-ALIGNMENT.md) |
| 参数、条件与共享 Runtime 的精确接线 | [数据绑定](STRUCTURED-DATA-BINDING.md) / [流程接线](STRUCTURED-FLOW-WIRING.md) |
| 所有旧实验、冻结工程和网络 Lab 专题 | [完整历史导航](README-HISTORY-20260910.md) / [进展历史](PROJECT-STATUS-HISTORY-20260910.md) |

本地 `artifacts/` 包含原始响应、失败、源码快照、摘要和审阅结果，被 Git 忽略。文档链接在本机可查看，不等于这些制品已随 Git 发布或备份。历史验收报告是当时快照；本轮更新只改变当前文档，不覆盖旧报告、Oracle 或基线。

#### 冻结的未来工程参考

[L1 Decision Plane](l1-decision-plane.md) 和 [P1.9 Canary Runbook](p19-canary-runbook.md) 仅为冻结参考，不属于当前原型路线或已完成证据。其他冻结工程见历史导航。

## English

Start with the [evaluation reset and implementation boundary](EVALUATION-RESET-20260916.md). Implementation is authorized, with R0 measurement partially implemented: 153 targeted checks pass and executed local probes used zero model calls, establishing neither agent benefit nor the 36-probe gate. Live DSH/Provider/trace integration and frozen cases/labels remain incomplete; no `run` command exists. The old semantic stage remains `paused_unmet`. Historical current/next-step wording describes snapshots; [negative results](SCHEMA-ARTIFACT-CONVERGENCE.md), [metrics](benchmarks/schema-artifact-summary.json) and formal gates remain unchanged.

The latest [snapshot-read package](SNAPSHOT-READ-SEMANTICS.md) supplies immutable-source semantics and exact repeated-follow-up protection. Two actual DSH/9B synthetic tasks exercise recovery interpretation and distinct-resource reads; a task/Oracle mismatch is explicitly retained. [Evidence and limits](benchmarks/snapshot-read-summary.json). Earlier reports below remain historical, not updated scores.

The latest opt-in [task-first delivery](TASK-FIRST-DELIVERY.md) removes model-selected presentation kinds, not execution contracts. A two-call known-snapshot 9B diagnostic improves a specific evidence request but still produces redundant-read advice. It is not a new DSH A/B, fresh-Skill success or a closed semantic stage. [Bound review and costs](benchmarks/task-first-delivery-summary.json).

The active path is the [evidence-first governed session](GOVERNED-SESSION.md). Its opt-in [single-choice protocol](DELIVERY-SINGLE-CHOICE.md) removes duplicate state. [Latest real-model results](benchmarks/delivery-single-choice-summary.json) remain2/2 structural admissions but0/2 complete tasks. The subsequent [host-terminal repair](HOST-TERMINAL-DELIVERY.md) passes2/2 actual-DSH/scripted-model lifecycle checks; it does not fix semantic interpretation or regrade the real-model batch. Earlier failures remain unchanged; semantic self-review experiments are historical diagnostics, not required layers.

The latest [duty/predicate/local-artifact report](SEMANTIC-DUTY-CONTRACT.md) separates the September 15 ten-call evidence-first note diagnostic from prior whole-draft experiments. Reference binding and constrained transcription improve, while an explicit predicate-comparison error persists. The stage remains incomplete. Historical evidence is retained; new-source acceptance and expanded A/B remain closed.

The [task-witness/evidence-scope diagnostic](SEMANTIC-WITNESS-DIAGNOSTIC.md) retains four known 9B cases, partial detections, false positives and binding failures. Its experimental calls remain opt-in; existing review/routing gain exact delivery-witness and note-ownership checks, not semantic approval.

The [9B/27B review-only contrast](SEMANTIC-REVIEW-MODEL-CONTRAST.md) retains four known cases plus the initial timeout. One targeted omission improves, while core semantic errors remain; it does not satisfy the stage or justify a wholesale model switch.

The [second frozen transfer](SEMANTIC-CLOSURE-TRANSFER-V2.md) failed: three correct boundary responses, eight partial tasks and one failure, with no fulfilled substantive task; all 67 model calls and 24/36 met criteria remain in the [portable summary](benchmarks/semantic-closure-transfer-v2.json).

The [first frozen semantic-closure transfer](SEMANTIC-CLOSURE-TRANSFER-V1.md) failed: six Skills/twelve tasks, one fulfilled task, one correct refusal, seven partial and three failed. All 56 real 9B calls and negative results are retained in the [portable summary](benchmarks/semantic-closure-transfer-v1.json).

The current [result-contract substage](SEMANTIC-RESULT-CONTRACT.md) separates exact observation projections, unverified L1 drafts and open duties. It is not a completed semantic-generalization gate.

[Bounded draft review/revision](BOUNDED-DRAFT-REVIEW.md) adds source-located addition/omission checks and one possible revision. AI review cannot grant truth or action authority.

Current [semantic-closure diagnosis](SEMANTIC-CLOSURE-DIAGNOSIS.md) records the known failures, bounded host projection and over-withholding; [fixed exit criteria](SEMANTIC-CLOSURE-EXIT.md) require frozen new-sample transfer, not mechanical tests alone.

Start with [project capabilities and usage](../README.md), [current progress](PROJECT-STATUS.md), [Stage 1 evidence](STAGE-1-RESULTS.md), [Stage 2 public-development transfer](STAGE-2-PUBLIC-TRANSFER.md), and [Skill/system interaction](SKILL-SYSTEM-INTERACTION.md). The [code map](../evaluation/README.md) distinguishes the active opt-in semantic frontend from historical experiment and replay dependencies.

The [governed hybrid workflow](GOVERNED-HYBRID-FLOWS.md) is implemented and [Stage 2's small development loop](STAGE-2-HYBRID-RESULTS.md) is complete, with partial versus full results and failures disclosed. Use the [machine summary](benchmarks/stage2-hybrid-summary.json) and [fixed exit criteria](STAGE-2-HYBRID-VALIDATION.md), not mechanism tests as semantic accuracy. Architecture references remain [ARCHITECTURE](../ARCHITECTURE.md), [HLD](../HLD.md), [LLD](../LLD.md) and [SSD](../SSD.md). The [prototype principles](ENSUREDSKILL-PROTOTYPE.md) and [generalization gate](TRANSLATION-GENERALIZATION-GATE.md) supersede conflicting historical production plans.

Use the [historical map](README-HISTORY-20260910.md) for specialist, lab and deferred-engineering documents. Ignored local artifacts are not automatically shipped or backed up by Git. Preserve original reports, failures and baselines; updating live documentation does not rewrite frozen evidence.
