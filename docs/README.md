# 文档导航 / Documentation Map

## 中文

**当前研究入口是 [R0–R3 有限收敛方案](EVALUATION-RESET-20260916.md)。** R0 工程阶段已完成：一次获批关联重验通过 24/24 臂、54 次本地读取、0 真实推理；原失败保留。未进入 R1，不将机械结果当作 9B 成绩。详见 [R0 验收报告](R0-COMPLETION.md)和[完成机器摘要](benchmarks/r0-reacceptance-20260918-summary.json)，来源、双审与裁决见[独立材料包](../data/bounded-pilot/r0-development-20260917/README.md)。

本次[清理与测试分层](CLEANUP-20260916.md)区分当前回归、历史探索和完整回归。旧文档中的“当前／下一步”是历史快照，不是继续试跑安排；与[原型准则](ENSUREDSKILL-PROTOTYPE.md)冲突的生产工程计划不生效。

### 当前入口

| 想了解什么 | 入口 |
|---|---|
| 项目设计、能力、优势、性能、场景与使用 | [项目 README](../README.md) |
| 当前完成项、待办与停止点 | [项目进展](PROJECT-STATUS.md) |
| 新考核方案、三份成绩单、R0–R3 总预算与硬停止 | [考核重置](EVALUATION-RESET-20260916.md) |
| 当前 R0 完成清单、保留失败与独立封存输入 | [R0 验收报告](R0-COMPLETION.md) / [材料与逐案裁决](../data/bounded-pilot/r0-development-20260917/README.md) / [第三方归属](R0-THIRD-PARTY-NOTICES.md) |
| 测量与实际 DSH 接线证据，不是真实模型成绩 | [R0 测量摘要](benchmarks/bounded-pilot-r0-summary.json) / [接线摘要](benchmarks/bounded-pilot-r0-integration-summary.json) |
| 历史生命周期证据与当前离线 Token 预检 | [生命周期修复](R0-MEASUREMENT-LIFECYCLE.md) / [Token 预检现况](R0-TOKEN-PREFLIGHT.md) / [本轮摘要](benchmarks/token-preflight-20260917-summary.json) |
| 清理了什么、如何恢复、怎样选择测试 | [清理说明](CLEANUP-20260916.md) / [清理摘要](benchmarks/cleanup-20260916-summary.json) |
| 评测代码从哪里看，哪些是历史依赖 | [评测代码导航](../evaluation/README.md) |
| 架构、总体、详细与系统设计 | [ARCHITECTURE](../ARCHITECTURE.md) / [HLD](../HLD.md) / [LLD](../LLD.md) / [SSD](../SSD.md) |
| L1、L0.5、L0、宿主和 Runtime 怎样交互 | [交互全景](SKILL-SYSTEM-INTERACTION.md) / [受控会话](GOVERNED-SESSION.md) |
| 严格流程与 LLM 混合执行及权威边界 | [双驱动流程](GOVERNED-HYBRID-FLOWS.md) / [结果合同](SEMANTIC-RESULT-CONTRACT.md) |
| 当前可选编译隔离、输入、参数与流程接线 | [隔离编译](ISOLATED-COMPILER.md) / [输入保真](TRANSLATION-INTAKE.md) / [任务对齐](TASK-SOURCE-ALIGNMENT.md) / [数据绑定](STRUCTURED-DATA-BINDING.md) / [流程接线](STRUCTURED-FLOW-WIRING.md) |
| 正式泛化需要什么证据 | [正式泛化门禁](TRANSLATION-GENERALIZATION-GATE.md) / [原型准则](ENSUREDSKILL-PROTOTYPE.md) |

### 保留的历史证据

以下结果不合并成当前通过率，不因清理改分；它们解释新协议为何必要。

| 历史阶段 | 证据入口 |
|---|---|
| 最新已知 2 Skill／3 Task：结构 3/3、完整任务 0/3 | [结构与工件负结果](SCHEMA-ARTIFACT-CONVERGENCE.md) / [机器摘要](benchmarks/schema-artifact-summary.json) |
| 调用协议、证据获取、两轮收敛与快照读取 | [调用协议](CALL-PROTOCOL-EVIDENCE.md) / [两轮记录](CONVERGENCE-CLOSURE.md) / [快照读取](SNAPSHOT-READ-SEMANTICS.md) |
| 输出交付实验与宿主终态修复 | [任务直达](TASK-FIRST-DELIVERY.md) / [单选协议](DELIVERY-SINGLE-CHOICE.md) / [终态机制](HOST-TERMINAL-DELIVERY.md) |
| 职责、自审、模型对照与迁移失败 | [职责实验](SEMANTIC-DUTY-CONTRACT.md) / [见证诊断](SEMANTIC-WITNESS-DIAGNOSTIC.md) / [模型对照](SEMANTIC-REVIEW-MODEL-CONTRAST.md) / [迁移 v1](SEMANTIC-CLOSURE-TRANSFER-V1.md) / [迁移 v2](SEMANTIC-CLOSURE-TRANSFER-V2.md) |
| 早期阶段 1／2 开发结果 | [阶段 1](STAGE-1-RESULTS.md) / [阶段 2](STAGE-2-HYBRID-RESULTS.md) / [首批负结果](STAGE-2-PUBLIC-TRANSFER.md) / [表示修复](STAGE-2-REPRESENTATION-REPAIR.md) |
| 原考核与未通过出口 | [旧根因](SEMANTIC-CLOSURE-DIAGNOSIS.md) / [旧出口](SEMANTIC-CLOSURE-EXIT.md) / [有界草稿审查](BOUNDED-DRAFT-REVIEW.md) |
| 更早实验、网络 Lab、冻结生产工程 | [实验索引](FLOW-EXPERIMENTS.md) / [历史导航](README-HISTORY-20260910.md) / [进展历史](PROJECT-STATUS-HISTORY-20260910.md) |

本地 `artifacts/` 包含原始响应、失败、源码快照、摘要和审阅结果，被 Git 忽略；在本机可查看不等于已发布或备份到 GitHub。清理前本地备份位置和内容摘要见清理说明。原始报告、Oracle、基线和暴露样本属性不变。

#### 冻结的未来工程参考

[L1 Decision Plane](l1-decision-plane.md) 和 [P1.9 Canary Runbook](p19-canary-runbook.md) 仅为冻结未来工程参考，不属于当前原型路线或已完成证据。

## English

The active entry is the [bounded R0–R3 protocol](EVALUATION-RESET-20260916.md). R0 engineering is complete: one authorized linked reacceptance passes 24/24 arms and 54 local reads with zero real inference; the original failed batch remains unchanged. R1 has not started and these are not 9B results. See [R0 acceptance](R0-COMPLETION.md), the [completion summary](benchmarks/r0-reacceptance-20260918-summary.json), [standalone sources/reviews/adjudication](../data/bounded-pilot/r0-development-20260917/README.md) and [third-party attribution](R0-THIRD-PARTY-NOTICES.md).

The historical [lifecycle repair](R0-MEASUREMENT-LIFECYCLE.md) covers metering, close and late delivery. [Token preflight status](R0-TOKEN-PREFLIGHT.md) and the [current summary](benchmarks/token-preflight-20260917-summary.json) describe the authorized offline implementation and remaining boundaries, not an enabled real-model path.

Start with the [project README](../README.md), [current status](PROJECT-STATUS.md), [architecture](../ARCHITECTURE.md), [Skill/system interaction](SKILL-SYSTEM-INTERACTION.md), [governed session](GOVERNED-SESSION.md) and [hybrid boundary](GOVERNED-HYBRID-FLOWS.md). The [evaluation code map](../evaluation/README.md) separates the active measurement path from legacy dependencies. [Cleanup and test suites](CLEANUP-20260916.md) explains current/historical/all selection, recovery and the [cleanup evidence](benchmarks/cleanup-20260916-summary.json).

The [original measurement](benchmarks/bounded-pilot-r0-summary.json) and [DSH integration](benchmarks/bounded-pilot-r0-integration-summary.json) summaries retain their fingerprints. [Recent negative results](SCHEMA-ARTIFACT-CONVERGENCE.md), [Stage 1](STAGE-1-RESULTS.md), [Stage 2](STAGE-2-HYBRID-RESULTS.md), [experiment index](FLOW-EXPERIMENTS.md) and [historical navigation](README-HISTORY-20260910.md) remain evidence, not the current run plan. Their old current/next-step wording must not restart superseded repair loops.

The [prototype principles](ENSUREDSKILL-PROTOTYPE.md) and [formal generalization gate](TRANSLATION-GENERALIZATION-GATE.md) supersede conflicting production plans. Ignored local artifacts are not automatically published to GitHub. Cleanup preserves failures, labels, baselines and exposed-Skill status; archive details are in the cleanup record. Frozen identity/HA/supply-chain work is not part of current prototype completion.
