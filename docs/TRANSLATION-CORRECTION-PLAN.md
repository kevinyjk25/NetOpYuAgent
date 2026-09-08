# 转译纠偏与闭环计划 / Translation Correction Plan

## 中文

更新：2026-09-08。当前执行顺序：**清理收敛 → 提交 → 等待合入 → 源语义审查与未见小批 → 泛化门禁 → Runtime 规模化评测**。本轮只完成合入前工作。

本页只维护有效计划；旧 A/B/C 子阶段及所有当时决策保存在[纠偏历史](TRANSLATION-CORRECTION-HISTORY.md)。阶段事实由[项目进展](PROJECT-STATUS.md)统一汇总。

### 不变原则

- 研究原型优先，不继续扩建生产控制面。Reasoning、Reliability Runtime、Infrastructure 三平面不变。
- L1 原文与真实宿主工具合同是输入；模型输出是待审提案，不是事实、Gold、授权或已执行结果。
- 复用现有 L0 合同与执行器；参数、顺序、分支、来源、缺失能力及未决条件必须可定位，不用额外执行语义掩盖转译缺口。
- 首次生成、辅助修订、可执行片段、正确停止和完整 Skill 分开计数。已用于修复的案例属于开发集。
- 不改历史源文、模型回答、Oracle、基线或代码指纹来提高成绩。脚本/第三方参考默认惰性，不在转译时运行。

### 当前路线及缺口

当前研究入口是合同约束构造 → FlowTree → 必要条件候选 → 完整源审查；模块和历史实验分类见[代码导航](../evaluation/README.md)。这是推荐继续验证的路径，不是已替换默认 DSH 的自动高准确转译器。

最新 33/33 来自六个已知开发案例的后处理，首次生成仍为 29/33；正向路径可行性未知、必要条件充分性和完整源语义均未证明。生成候选未激活。详细原始数据见[行为修复](FLOW-BEHAVIOR-REPAIR.md)。

### 合入后的验收顺序

1. **核实推导前提。** 逐项检查模型的禁止/可能/未知判断和引用位置；覆盖否定极性、字段别名、OR 与 AND、缺失前置及被废弃示例。反事实逻辑正确不能补偿错误的自然语言前提。
2. **做新的首次完整转译。** 冻结实现、源/宿主接口和协议后再选未见 Skill；从 L1 重新生成，不复用手工 L0 或已修好的候选。生成、审查、评分分离；如用 AI 审查则标记角色隔离模拟，不宣称真人 Gold。
3. **先小批，再规模化。** 不把首次小批称为正式泛化结论。若发现错误，保留测试结果并另开开发修订；不在同一密封集合调参再当独立验证。
4. **满足既有泛化门禁后再做 Runtime 对照。** ≥3 cohort、≥50 Skill、≥15 仓库、≥8 领域、≥600 case 不下调；具体安全、召回、参数与证据标准见[门禁](TRANSLATION-GENERALIZATION-GATE.md)。

每批同时报告：接受后的语义正确率、适用 Skill 召回、步骤/整 Skill 覆盖、不安全误接受、过度停止、参数错误、首次及辅助结果、调用/token 与含失败的 p50/p95；按 Skill/仓库聚类。不能用 Schema 通过率、语句数量或测试数量替代它们。

## English

Updated 2026-09-08. Active order: **consolidate → commit → wait for merge → source-semantic audit and fresh small batch → generalization gate → large Runtime evaluation**. This cleanup stops before merge. The [historical plan](TRANSLATION-CORRECTION-HISTORY.md) preserves earlier A/B/C decisions; [project status](PROJECT-STATUS.md) owns current facts.

Preserve the prototype's three planes, real source/tool contracts, inactive proposals and the existing L0 compiler/executor. Never infer authority from confidence, rewrite old answers/oracles/hashes, execute source scripts, or count safe stops as complete business Skills. Known repaired cases remain development data.

The [recommended research path](../evaluation/README.md) is contract-grounded constructors → FlowTree → necessary-guard candidates → full-source review. It has not replaced default DSH authoring. The 33/33 result is postprocessed known-development evidence, with first-pass still 29/33, retained positive-path unknowns and unproven complete semantics.

After merge:

1. Audit model premises/citations, polarity, aliases, alternatives, missing prerequisites and discarded examples. Valid inference does not establish a correct language premise.
2. Freeze the implementation/protocol before sampling unseen Skills. Run fresh L1-to-candidate generation, not prebuilt L0. Separate generation, review and scoring; AI review is not human Gold.
3. Start small. Preserve failures and use separate development revisions, never tuned reruns on a sealed set as independent evidence.
4. Meet the unchanged [generalization gate](TRANSLATION-GENERALIZATION-GATE.md) before large Runtime comparison: at least three cohorts, 50 Skills, 15 repositories, eight domains and 600 cases, including its safety/recall/parameter/evidence thresholds.

Report semantic precision, eligible-Skill recall, step/whole-Skill coverage, unsafe acceptance, excessive stops, parameter errors, first-pass/assisted outcomes and failure-inclusive calls/tokens/p50/p95, clustered by Skill/repository. Schema passes and test counts cannot replace these measures.
