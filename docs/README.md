# 文档导航 / Documentation Map

## 中文

先看当前状态和交互流程，再按需进入专题。历史实验不再与当前主路径并列展示；与[原型权威准则](ENSUREDSKILL-PROTOTYPE.md)冲突的旧生产工程计划不生效。

| 想了解什么 | 入口 |
|---|---|
| 项目设计、能力、性能、场景和使用 | [项目 README](../README.md) |
| 当前完成什么、下一步做什么 | [项目进展](PROJECT-STATUS.md) |
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

Start with [project capabilities and usage](../README.md), [current progress](PROJECT-STATUS.md), [Stage 1 evidence](STAGE-1-RESULTS.md), [Stage 2 public-development transfer](STAGE-2-PUBLIC-TRANSFER.md), and [Skill/system interaction](SKILL-SYSTEM-INTERACTION.md). The [code map](../evaluation/README.md) distinguishes the active opt-in semantic frontend from historical experiment and replay dependencies.

The [governed hybrid workflow](GOVERNED-HYBRID-FLOWS.md) is implemented and [Stage 2's small development loop](STAGE-2-HYBRID-RESULTS.md) is complete, with partial versus full results and failures disclosed. Use the [machine summary](benchmarks/stage2-hybrid-summary.json) and [fixed exit criteria](STAGE-2-HYBRID-VALIDATION.md), not mechanism tests as semantic accuracy. Architecture references remain [ARCHITECTURE](../ARCHITECTURE.md), [HLD](../HLD.md), [LLD](../LLD.md) and [SSD](../SSD.md). The [prototype principles](ENSUREDSKILL-PROTOTYPE.md) and [generalization gate](TRANSLATION-GENERALIZATION-GATE.md) supersede conflicting historical production plans.

Use the [historical map](README-HISTORY-20260910.md) for specialist, lab and deferred-engineering documents. Ignored local artifacts are not automatically shipped or backed up by Git. Preserve original reports, failures and baselines; updating live documentation does not rewrite frozen evidence.
