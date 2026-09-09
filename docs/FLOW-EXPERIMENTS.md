# 转译实验索引 / Translation Experiment Index

## 中文

当前推荐接口见[代码导航](../evaluation/README.md)，有效下一步只以[进展](PROJECT-STATUS.md)和[计划](TRANSLATION-CORRECTION-PLAN.md)为准。下表是研究轨迹，不是多个并列的产品实现。原始回答、失败、审查和成本均保留；旧答案不修改。

| 路线 / 文档 | 定位与保留原因 |
|---|---|
| [正向辅助读取](FLOW-FORWARD-TRANSLATION.md)、[冻结开发小批](FLOW-FROZEN-DEVELOPMENT.md) | 原始结构/语义失败与辅助修订的起点 |
| [源文约束](FLOW-SOURCE-GROUNDING.md) | 引用存在不等于极性和业务含义正确 |
| [FlowTree 编译](FLOW-TREE-COMPILER.md) | 仍复用的层级表示/编译基础，不是弃用执行器 |
| [Tree 首次输出](FLOW-TREE-FORWARD-PILOT.md)、[协议探针](FLOW-TREE-PROTOCOL-CANARY.md)、[宿主约束批次](FLOW-BOUNDED-FORWARD.md) | 区分解码、表示与源语义缺口；保留失败 |
| [源账本](FLOW-SOURCE-CONSTRAINTS.md)、[完整双阶段](FLOW-TWO-PASS-TRANSLATION.md) | 源保留/映射/执行分层实验，未证明完整语义接纳 |
| [精简映射](FLOW-LEAN-MAPPING.md) | 输出节省不等于整链质量提升 |
| [职责映射](FLOW-RESPONSIBILITY-MAPPING.md)、[真实批次](FLOW-RESPONSIBILITY-PILOT.md) | 记录类型/职责冲突及合法终态表示缺口 |
| [统一节点](FLOW-CANONICAL-MAPPING.md)、[完整批次](FLOW-CANONICAL-PILOT.md) | 局部 Schema 修复与仍失败的完整映射分别保留 |
| [分层诊断](FLOW-DIAGNOSTICS.md) | 离线定位工具仍可用；不负责自动修正语义 |
| [节点证据](FLOW-NODE-EVIDENCE.md) | 显式反驳/未决是诊断信息，不是完整接纳 |
| [源义务/宿主分层](FLOW-SOURCE-DUTIES.md) | 自由 after/when 引入新错误；不采纳为默认质量升级 |
| [9B common-JSON](FLOW-9B-COMMON-JSON.md)、[跨模型设计](FLOW-MODEL-COMPARISON.md) | 未证明 common-JSON 质量提升；GPT 对照暂缓 |
| [行为修复](FLOW-BEHAVIOR-REPAIR.md) | compact/ordered/thinking 与必要条件推导的历史证据；旧 33/33 不再作为新一轮泛化依据 |
| [语义迁移](FLOW-SEMANTIC-TRANSFER.md) | 单条件补全退步、真值表/递归语法负结果、可读表达式、编译反馈和范围受限构造；首次/修订分列 |
| [无损输入](TRANSLATION-INTAKE.md)、[结构化数据绑定](STRUCTURED-DATA-BINDING.md) | 源文/宿主保真与版本化嵌套数据原语；零模型调用的机械验证，不计入语义转译成功率 |
| [结构化流程接线](STRUCTURED-FLOW-WIRING.md) | 新 Tree/图表示接入原执行器；本地嵌套读取、分支、权限/时效与候选边界，不是完整模型转译或实际设备执行 |
| [任务/源义务对齐](TASK-SOURCE-ALIGNMENT.md) | 四份固定公开材料的源锚定任务、35 项声明义务及未决问题；发现源张力和动态解码缺口，未宣称转译成功 |
| [Netdata 隔离宿主与列解码](NETDATA-ISOLATED-VALIDATION.md) | 单次 Function 宿主、通用有界列投影、权限/单位/本页输出验证；不是原厂服务、完整窗口或模型转译成绩 |

代码仍保留原文件名，避免破坏旧版本引用。当前入口已解除对 `flow_source_duty_pilot` 等历史探针的依赖。历史检查点使用对应版本回放；清理前实现快照为 `c2ebd78`，操作与验证见[收敛记录](FLOW-CONSOLIDATION.md)。

原有逐阶段 Done/To-do 移入[阶段历史](PROJECT-HISTORY.md)，原 A/B/C 计划移入[纠偏历史](TRANSLATION-CORRECTION-HISTORY.md)。这些历史文档中的“下一步”“未提交”不再描述当前状态。

## English

[Task/source alignment](TASK-SOURCE-ALIGNMENT.md) records four concrete public-source development tasks, declared obligations and unresolved host/source gaps. Source-example tensions and dynamic decoding are explicit; there is no new translation success claim.

[Structured flow wiring](STRUCTURED-FLOW-WIRING.md) connects versioned source-anchored trees and nested bindings to the shared executor with explicit local host permissions. Fixture read/branch/candidate checks do not prove whole-Skill translation or device execution.

[Lossless intake](TRANSLATION-INTAKE.md) and [structured data bindings](STRUCTURED-DATA-BINDING.md) retain source context and add versioned nested-data primitives. Their zero-model mechanical checks are not semantic translation scores.

Use the [code map](../evaluation/README.md) for the recommended research path and [current status](PROJECT-STATUS.md) / [plan](TRANSLATION-CORRECTION-PLAN.md) for active decisions. The table above indexes historical investigations, not competing production paths.

FlowTree compilation and offline diagnostics remain reusable foundations. Historical probes preserve structural, semantic and cost failures; none becomes a default improvement because its schema passes. [Current evidence](FLOW-SEMANTIC-TRANSFER.md) replaces unary guard addition with source-expression extraction and code-owned logic, retaining failed table/grammar/feedback variants. Complete Skill semantics remain unproven; GPT comparison is deferred.

Keep old module names for historical references and pinned evidence. Current authoring no longer imports historical pilots for shared infrastructure. Replay historical checkpoints with their original implementation; pre-cleanup snapshot: `c2ebd78`. See [consolidation verification](FLOW-CONSOLIDATION.md), [phase history](PROJECT-HISTORY.md) and [plan history](TRANSLATION-CORRECTION-HISTORY.md). Superseded next steps do not authorize new work.
