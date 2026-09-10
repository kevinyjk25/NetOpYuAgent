# EnsuredSkill 项目进展 / Project Status

## 中文

更新：2026-09-10。阶段 1 已提交到本地 dev：`43a2b76`，未推送。**阶段 2 双驱动小批开发验证已完成，改动尚未提交；正式泛化门禁仍关闭。** 最终 v7 的 10 Skill 中，6 个结构候选、5 个有用读取前段；真实 9B 执行审阅为 **1 个限定请求完成、2 个局部可用、2 个草稿不接受**，不是 3 个完整成功。结果、失败、成本与可解释轨迹见[阶段 2 报告](STAGE-2-HYBRID-RESULTS.md)及[固定验收条件](STAGE-2-HYBRID-VALIDATION.md)。

### 已完成

- 双驱动：原 L0 严格片段、有界 LLM、独立候选准入、固定 DAG、串并行/all-success 汇合、过期证据和迟到结果隔离；不新增 Effect，不改变默认 DSH 路由。
- 转译入口 v7：来源/业务任务/编译说明分离；有据读取前段＋原任务直接保留的 L1；参考资料与实际观察无损分层。模型边界注释仍有错误，绝不自动视为语义或权限证明。
- 真实模型与机制分开验收：5 个本地混合执行；35/35 公开图接线检查（5 正常、30 异常，模型替身）；2630 全量测试及 81 子测试通过，39 个变更 Python 文件 Ruff 通过。完整证据与原版零调用回放保留。
- v1–v7 构造、一次失败格式诊断、v6/v7 实际执行共 84 次本地 9B 调用；537,423 输入 / 52,527 输出 token。v7 含模型图时延 p50/p95 30.79/46.36 秒，仅 5 样本且部分与 pytest 并发，不是 SLO 或因果加速证据。

### 保留的历史证据

- v65 结构修复：全量 2566 项及 81 子测试、261 项定向、35 项发布/边界/修复检查、修改代码 Ruff 通过。左右比较数据均受类型、依赖和时效限制；不是公开 Skill 成功率。
- v64/v65 静态预检暴露体积退步：预算通过由 7/9 降至 6/9（Phoenix 新增超限）。未放宽预算；原 v63 和阶段 1 的 3055 份绑定证据全部保持不变。
- 三例报告零调用回放一致，122 份新制品绑定；46,397 输入/15,309 输出 token，请求 p50/p95 为 164.26/231.01 秒。仅机制改善，语义准确率尚无提升证据；不是 Runtime 性能或独立 Gold。

- 阶段 2：10 Skill / 10 仓库 / 9 个开发者分类领域。v62 为 0/9 初始预算通过；v63 为 7/9，另 2 个预算失败、1 个纯 L1。改善的是输入准入，不是语义准确率。
- 本轮 2543 全量 + 81 子测试、22 项定向、修改代码 Ruff 通过。7 份模型报告零调用回放一致；136 份本批证据、2919 份阶段 1 证据已核对。文档分区回归失败及修复保留，未放宽测试。

- [阶段 1 验收](STAGE-1-RESULTS.md)：3 种已知流程各两次构造，6/6 受审只读区域、50/50 合成本地路径通过。20 次 9B 请求；不是 6 个独立 Skill 或生产成功率。
- 512 定向、2524 全量及 81 子测试通过；提交前再验 232 项。阶段代码 Ruff 通过，全仓仍有 224 条历史诊断。
- 源文锚定、观察作用域、原 Schema 参数槽、闭合分支、数组长度、精确 L1 交接已接入可选语义前端。默认 DSH 路由未变。
- 失败、原始模型回复、源码快照、费用与 86 份原版回放完整保留。阶段 1 证据绑定 2919 份本地制品；Git 提交不包含这些被忽略的制品。
- 历次 C3 详细结果移入[历史快照](PROJECT-STATUS-HISTORY-20260910.md)。旧文件/模块仍可引用，不以删失败记录制造提升。

### 当前工作与下一步

| 顺序 | 工作 | 出口与边界 |
|---|---|---|
| 已完成 | v65 通用结构修复及三例 9B 探针 | 0 个语义接受区域；初始预算退步单列，不扩大 Runtime A/B |
| 已完成 | 规则出处分权与受控混合流程 | 原文/调用者/模型建议分离；读取—推理—独立准入、有界并行/join；不把严格检查转给 LLM 绕过 |
| 已完成（小批开发） | 10 Skill 构造审阅、5 份真实 9B 接线 | 1 完成＋2 局部可用；不接受其余草稿、不改写失败，不等于语义泛化通过 |
| 后续，尚未开始 | 已暴露的语义与表示缺口 | 边界注释分权、事实支持校验、模板污染、同类型错绑、可验证选择/路径/查询；旧批次仅用于回归 |
| 后续，尚未开始 | 新批次迁移、跨 cohort 验证 | 未见集不用于调参；达到[正式门禁](TRANSLATION-GENERALIZATION-GATE.md)才扩大 DSH/Runtime A/B |

阶段 2 按[固定交接范围](STAGE-1-EXIT.md)执行：任务不提供期待 L0/参数答案/路由；源/任务/宿主与审阅预期先冻结，首次失败不原地重试。指标按 Skill 分母报告首次结构、语义可用区域、参数、职责保留、正确/过度停止和成本；无法评估的指标保留 unknown，而非 0 或通过。

[受控混合流程](GOVERNED-HYBRID-FLOWS.md)是可选本地原型；开放职责仍是 L1，不把整图称为确定性 L0。交接中的虚构时间/可联系性、README 中未观察到的命令仍使草稿不合格；事件的“lookup complete”过度表述、工单冗余不确定性也没有消失。自动写事务接线、动态扩图、证据刷新替换和持久恢复尚未实现。本次按要求停在阶段 2，不自动启动下一批、提交或推送。

### 保留但不推进

生产身份/多人审批、Provider 供应链、HA/DR、WORM、Hermes/A2A、真实厂商认证与大规模 Runtime 性能评测仍冻结。原型准则见[权威文档](ENSUREDSKILL-PROTOTYPE.md)。共享虚拟环境、数据库、评测证据和无关用户文件不作垃圾删除。

## English

Updated 2026-09-10. Stage 1 is committed locally at 43a2b76, not pushed. Stage 2's small mixed development loop is complete and uncommitted under [the explicit criteria](STAGE-2-HYBRID-VALIDATION.md); formal generalization stays closed. Frozen v7: 10 Skills, 6 structural candidates, 5 useful read prefixes and 5 real local 9B runs. Actual draft review: **1 fulfilled scoped request, 2 useful partial analyses, 2 rejected drafts**, not three fully successful tasks. See [all results and limits](STAGE-2-HYBRID-RESULTS.md). Full regression passes 2630 tests plus 81 subtests; 39 changed Python files pass Ruff; public graph checks pass 35/35 with model doubles. All raw failures and original-version replays remain.

The Runtime composes original strict regions with bounded model tasks, independent admission and required joins. v7 retains original L1 duties instead of rephrasing them and separates references from current observations. Across the seven authoring revisions, one failed diagnostic and two live revisions: 84 actual local 9B calls, 537,423 input / 52,527 output tokens. v7 graph p50/p95 is 30.79/46.36 seconds (n=5; some concurrent pytest load), not an SLO or causal speedup. Handoff/README errors remain; partial incident/warehouse outputs also retain explicitly documented defects. Stop here: no automatic new cohort, commit, push, DSH A/B or production rollout.

Stage 1 has three known procedures, six reviewed read regions and 50 synthetic paths, not six independent Skills or a production probability. Full regression: 2524 plus 81 subtests; stage lint passes while 224 historical diagnostics remain. The local evidence binds 2919 artifacts and preserves original-version replay; ignored artifacts are not backed up by this Git commit. Detailed C3 history remains in the [snapshot](PROJECT-STATUS-HISTORY-20260910.md).

Earlier v63 passed 2543 tests plus 81 subtests and retained seven identical replays with 136 bound artifacts. v65 and mixed-prototype evidence remain separate. Typed value comparison alone does not determine business semantics. Governed read/reason/admission and joins now exist, but durable recovery, dynamic graph expansion and automatic Effect integration do not. Unknown Oracle/coverage metrics remain unknown. Use new development batches before cross-cohort and DSH comparison gates; production engineering stays deferred.
