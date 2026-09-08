# EnsuredSkill 项目进展 / Project Status

## 中文

更新：2026-09-08。**当前为 C3h：转译原型收敛；提交后等待合入，不启动下一阶段。**

本页只记录当前事实与待办。过去各轮的实验数字、失败和当时计划均保留在[阶段历史](PROJECT-HISTORY.md)，不再逐轮向本页追加。

### Done

- Runtime 的合同、Evidence、Guard、审批、单写事务、独立验证与补偿机制原型，以及本地 C1/C2 读流程/分支接线已实现。参见[架构](../ARCHITECTURE.md)、[业务流程](L0-BUSINESS-FLOW.md)。不代表真实厂商设备或生产认证。
- 只读合同、源引用、层级 FlowTree、合同约束参数生成和必要条件推导已实现；复用现有 L0 编译/执行语义，不新增执行器。生成结果未激活，不赋予 Runtime 权限。
- 固定开发反例经显式修复后匹配 **6/6 案例、33/33 场景**：23 个可执行片段场景、10 个正确停止场景。首次生成仍为 **4/6、29/33**。参考答案由同一开发助手编写，完整源审查未完成，公开 Skill 数为 0；不是首次端到端、独立 Gold 或泛化成绩。[原始摘要](benchmarks/flow-guard-necessity-summary.json)。
- 清理前实现、测试和摘要已保存在 Git 快照 `c2ebd78`。本轮抽离重复检查点/回放逻辑，解除当前入口对旧实验的依赖，收敛 README/计划/文档导航；历史制品和失败样例保留。[范围与复验](FLOW-CONSOLIDATION.md)。

### To-do 与推进条件

| 顺序 | 工作 | 状态 / 验收 |
|---|---|---|
| 当前 | 清理回归、提交和合入 | 提交后等待用户合入；不自动推送、合并或运行新模型批次 |
| 合入后 1 | 源前提和引用审查 | 核实模型前提、否定/别名/替代条件；必要条件不等于充分性，不抹去 unknown |
| 合入后 2 | 小批量未见 Skill 首次完整转译 | 冻结实现/协议后选样，保留原始失败、辅助修订、停止与全部成本；使用 9B |
| 再后续 | 跨 Skill/仓库泛化门禁 | 至少 3 个不重叠 cohort、50 Skill、15 仓库、8 领域、600 case；详细阈值不变 |
| 门禁通过后 | 大规模 L0→Runtime / DSH 配对 | 当前不解锁；代码回归或固定 33/33 不能替代转译泛化证据 |

完整计划见[转译纠偏](TRANSLATION-CORRECTION-PLAN.md)；统计与独立性要求见[泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)。现有 100 个公开 Skill / 72 仓库 / 9 领域是已知开发库，不能重新命名为未见样本。

### 保留但不推进

- `ES-P1-Private-Human = skipped_retained_open`；AI 角色隔离不等于独立真人评测。
- GPT 对照暂缓；当前继续 `qwen3.5:9b`，不运行 7B/27B。
- 生产身份、供应链、多人治理、HA/DR、WORM、SLO 和 Hermes/A2A 为 `frozen_future_engineering`，不是原型完成条件。
- 原型权威边界由 [ENSUREDSKILL-PROTOTYPE](ENSUREDSKILL-PROTOTYPE.md) 定义。历史论文/工程报告不覆盖最新证据限制。

## English

Updated 2026-09-08. **Current phase: C3h translation-prototype consolidation. Stop after committing and wait for merge.** This page owns current status; the [history](PROJECT-HISTORY.md) preserves earlier experiments, failures and superseded plans.

### Done

- Contract/evidence/guard/approval/transaction/verification/recovery mechanisms and local C1/C2 read-flow wiring exist. These are prototypes, not real-device or production qualification.
- Source-linked FlowTree, contract-grounded argument constructors and necessary-guard synthesis reuse the existing L0 compiler/executor. Generated candidates remain inactive and unauthorized.
- Explicit re-derivation matches 6/6 known cases and 33/33 scenarios: 23 executable-fragment and 10 safe-stop scenarios. First-pass remains 4/6 and 29/33. References come from the development assistant; full-source review is incomplete and this batch contains zero public Skills. See the [unchanged summary](benchmarks/flow-guard-necessity-summary.json).
- Snapshot `c2ebd78` preserves pre-cleanup code and evidence summaries. Shared checkpoint plumbing replaces duplicate code, the current path no longer imports historical pilots, and documentation is separated into current state and history. See [consolidation verification](FLOW-CONSOLIDATION.md).

### Next, only after merge

1. Audit source premises/citations and polarity, aliases and alternatives. Preserve unknowns and the necessity/sufficiency distinction.
2. Freeze a small, genuinely unseen, source-to-translation 9B batch. Count first attempts, assisted repairs, stops and all costs separately.
3. Meet the unchanged [generalization gate](TRANSLATION-GENERALIZATION-GATE.md): at least three disjoint cohorts, 50 Skills, 15 repositories, eight domains and 600 cases, plus its quality thresholds.
4. Only then unlock large Runtime/DSH evaluation. The known 100-Skill development library cannot become unseen evidence by relabeling.

Private-human evaluation remains `skipped_retained_open`; AI role separation is not human Gold. GPT comparison is deferred. Production engineering and Hermes/A2A remain frozen under the [prototype charter](ENSUREDSKILL-PROTOTYPE.md). No next-stage model calls, automatic push or merge are part of this cleanup.
