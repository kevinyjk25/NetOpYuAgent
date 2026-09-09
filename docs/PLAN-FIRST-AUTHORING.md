# 先规划、后绑定参数 / Plan-first Authoring

## 中文

更新：2026-09-09，C3w。**本阶段已闭合“真实 9B → 冻结计划 → 分槽参数 → 原编译器 → 合成宿主验证”的机制链。公开 Skill 的语义转译阶段尚未完成。** 默认仍为 `direct`，新 `plan_first` 是研究入口，不改变 DSH 路由、Runtime 执行器或授权机制。

### 这次具体解决了什么

之前模型同时猜流程、填参数和抄写来源，失败很难定位。现在 [source_plan](../evaluation/source_plan.py) 将它们分开：

```text
原始 Skill + 当前转译请求 + 原宿主合同
                │ 可选：显式未来业务场景、宿主对应与操作模式
                ▼
9B 规划：读取 / 条件 / 分支 / 终止 / 未决项 / 剩余职责
                ▼
结构检查 → 冻结计划摘要、源位置、工具、模式、分支与读取槽
                ▼
9B 逐槽填参 → 格式检查 → 来源检查（错误立即停止）
                ▼
原参数校验与 Tree 编译器 → 未激活候选 → 限定范围语义审阅
                ▼
宿主明确授权精确流程与请求 → 原 Runtime → 实际读取／拒绝／交回 L1
```

1. **计划不可被填参阶段改写。** 参数回复绑定计划摘要与准确读取槽；不能更改工具、模式、分支或添加调用。源包、任务、catalog、操作模式均留有摘要，未来业务场景另行保留。只允许引用输入和在当前路径上必然先执行的读取结果。
2. **不再要求模型重抄完整引文。** 源块 ID 回填准确原文；不同轮次的同名 ID 分属独立命名空间。映射工具需提供原操作名及其出现位置，但“出现”不证明该调用必要。
3. **分清转译工作和未来业务。** 可选的任务绑定 `futureScenario` 描述未来流程的业务目标，不提供预写计划、答案或执行授权。它是调用方声明，不是与原任务语义等价的证明。
4. **填参知道有哪些动态来源。** 展示输入与前序输出 Schema 的可引用路径，明确它们不是当前真实数据；不根据同名字段自动推断映射。类型、可选字段、数组元素存在性和时效仍由原链路检查。动态参数不能用编造的示例值代替。
5. **来源尽早验证。** 每个参数槽和最终降低过程复用同一个 literal 来源检查器。无来源的 `dev-001` 在第一槽就应停止，而不是继续花调用生成后续参数。出现于原文仍不保证值符合业务意图。
6. **允许有证明记录的不可达终止消除。** 两个分支都已经结束时，模型额外写出的根结束节点不可能被执行；保留原计划和规范化记录，不将它加入可达 Tree。与显式 `already_closed` 的可达 Tree 相同。开放路径不会被自动补成功，不可达读取仍拒绝，不自动去重调用。

收尾复核还修复了常量出处缓存：两个字段声明相同常量时，不能因值 Schema 相同就共用第一个字段的原始 `const` 路径；包含常量的嵌套祖先也按路径区分。v21 的[兼容性检查](../artifacts/translator-v2/source-plan-20260909/compatibility-v21/report.json)验证三份已记录接线请求及重新推导文件均未变化，未增加模型调用。它是新实现回归，不替代原版本隔离回放，也不修改历史 manifest。

计划最多 8 个读取、32 个语句、8 层嵌套。绑定来源导航最多 64 项、6 层，并报告导航截断；完整 Schema 仍保留。支持标量相等分支，不支持任意谓词、集合成员检查、循环或脚本执行。取得观测不等于验证观测内容；必要检查不能表达时，应在依赖它的操作之前交回 L1，不能跳过检查。

### 真实结果：成功和失败都保留

[摘要与 127 份绑定证据](../artifacts/translator-v2/source-plan-20260909/evidence-summary/report.json)记录 **8 次开发实验、12 次真实 qwen3.5:9b 调用**：1 个已知公开 Skill 和 1 份原有合成接线说明。新增公开 Skill 数量是 **0**。不是独立 Gold，也不是成功率评测。

| 实验 | 调用数 | 请求累计耗时 | 输入 / 输出 token | 结果 |
|---|---:|---:|---:|---|
| Netdata v14：计划分离 | 1 | 47.279 秒 | 7,775 / 713 | 4 个读取，未决项阻止继续 |
| Netdata v15：操作定位、任务分离 | 1 | 62.275 秒 | 7,969 / 949 | 仍将文档检查当调用 |
| Netdata v16：显式未来场景 | 1 | 55.792 秒 | 8,304 / 739 | 变为 discovery → query，但无充分前置判断 |
| Netdata v17：9B 推理模式 | 1 | 40.994 秒 | 8,727 / 586 | 仍停止；推理未解决问题 |
| Netdata v18：模型可读 Schema | 1 | 69.414 秒 | 10,025 / 1,190 | 仍有职责／调用混淆，未编译 |
| 合成接线 v18 | 1 | 28.889 秒 | 3,419 / 596 | 条件步骤正确，冗余终止被严格实现拒绝 |
| 合成接线 v19 | 3 | 57.785 秒 | 9,904 / 1,280 | 计划通过；编造设备／接口值，来源校验拒绝 |
| 合成接线 v20 | 3 | 42.549 秒 | 10,465 / 853 | 动态引用正确，2 个读取 + 条件编译通过 |

三版合成接线的**首次规划请求完全相同**。v19 只在可达语义不变的前提下处理冗余终止；v20 改进参数阶段的来源导航与解释，并提前检查来源。全部旧失败用各自源码快照在隔离目录零调用回放，报告逐字节一致；此前五阶段 275 份绑定文件未变。

这些累计时延不是 p50/p95。Netdata v16 增加了场景声明，v17 开启 thinking 并把输出预算从 4096 改成 8192；v18 恢复非推理并提供模型可读 Schema。不能把这些尝试合成“只换算法、同资源”的准确率对照，或将某次较快归因于系统性能提升。

### 实际运行的候选是什么

[生成 Tree](../artifacts/translator-v2/source-plan-20260909/wiring-9b-binding-sources/round-002/tree.json)来自模型，不是原手工 Tree。设备取自 `input /device/id`；条件为首个接口的 `adminUp == false`；第二个读取取同一输入设备及前序结果 `/interfaces/0/name`。接口开启则结束，只在关闭时读计数器，然后交回 L1。

[限定范围审阅](../artifacts/translator-v2/source-plan-20260909/wiring-validation/review.json)绑定 Tree 摘要 `sha256:e2470542ab4c81af9a66e642c9c2170a07e1d8d1bc290385a58afa619d78d87e`。随后走原 `run_read_flow`，不是另写执行器：[10/10 路径检查通过](../artifacts/translator-v2/source-plan-20260909/wiring-validation/report.json)。覆盖开启／关闭分支、零／非零计数、空列表、错误返回类型、未认证身份、设备／接口范围、错误授权摘要。只有合成读取；网络、源脚本、写操作均为 0。

**没有完成的部分：**空列表靠原缺值门禁停止，没有生成显式空列表分支；`errors == 0` 判断及非零变更候选没有转译，两种计数都交回 L1。`remaining.duties` 还是空数组，终止说明也未完整记载遗漏职责。故仅接受局部读取测试，拒绝“整 Skill 转译通过”的结论。公开 Netdata 候选从未进入执行；其源包装器要求、合成宿主限制、职责阶段和查询前置条件仍未收敛。

### 如何试用

原六字段输入、宿主映射和模式说明见[上一阶段](HOST-OPERATION-MODES.md)。这是离线研究 CLI，不是默认 DSH 功能：

```bash
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN \
  --inputs INPUT.json --profile plan_first
# 按需增加 --bindings BINDINGS.json --operations OPERATIONS.json --scenario SCENARIO.json
.venv/bin/python -m evaluation.source_ledger run NEW_RUN \
  --max-new-calls 4 --report-dir NEW_REPORT
# 只恢复尚未生成的下一槽；错误断点不重试。零调用回放：
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

`SCENARIO.json` 使用 `netopyu.io/authoring-scenario/v1`，包含原任务的 `taskDigest`、调用方明确提供的 `futureTask` 和 `reviewKind: caller_supplied_scenario_not_independent_gold`；可查看[本轮示例](../artifacts/translator-v2/source-plan-20260909/scenario-v1.json)。默认非推理、49152 上下文与 4096 输出预算；可显式 `--reasoning` 使用 8192 输出预算，仍须通过更严格的剩余输入预算。不会自动扩大预算或删除源文。

查错依次看：`choice.json` → `prepared-plan.json` 的读取、支配关系与 `normalizations` → 各槽 `planned-arguments.json` / `argument-origin-check.json` → `catalog-lowering.json` → `tree.json` / `compilation.json` / `plan-to-tree.json`。`compiled_region_requires_semantic_review` 不是激活许可。输出目录必须是新目录；本地 artifacts 不随 Git 分发。

下一阶段优先补齐**逐项源义务覆盖与被截断边界的准确说明**，再解决公开源的操作／前置条件归属和宿主替代范围。不能继续只靠添加提示语或扩大测试数量宣称修好。正式 ≥3 不重叠 cohort／≥50 Skill／≥15 仓库／≥8 领域／≥600 case 的泛化门禁不变，暂不扩大 Runtime A/B。

最终验收：[验证报告](../artifacts/translator-v2/source-plan-20260909/validation-summary/report.json)。新增 49 项机械测试；309 项定向通过（25.14 秒），全量 **2292 passed + 81 subtests passed（179.83 秒）**；21 个相关 Python 文件 Ruff、文档链接与空白检查通过。前一轮 2290 项全量结果也保留。修改保留在 dev，未提交、未推送；测试数量不是语义准确率。

## English

C3w closes one mechanism milestone: real 9B planning → immutable plan → separate argument slots → original compiler → reviewed synthetic-host execution. **Public-Skill semantic translation is still open.** `plan_first` is opt-in; default `direct`, DSH routing, Runtime execution and authorization are unchanged.

The planner selects reads, scalar-equality branches and explicit endings before generating arguments. A sealed plan fixes graph structure, source coordinates, tools/modes and lexical read slots. Arguments cannot change that plan or reference future/sibling results. Each response retains its own source-block namespace. Source-operation occurrence is a location witness, not proof of necessity or semantic equivalence.

A caller-supplied, task-digest-bound future scenario can distinguish the generated workflow's goal from the compiler's current authoring work. It does not supply a prewritten plan or grant permission. The exact output schema is shown to the model as well as used for decoding. Parameter requests expose bounded schema-path navigation and explain execution-time references; no actual values or automatic target mappings are invented. The same literal-origin checker runs per slot and again at final lowering. Schema compliance, occurrence and typing do not establish business correctness.

The former strict implementation rejected a redundant root terminal after both branches had already ended. The new frontend records and removes only that unreachable terminal, preserving the original proposal and every reachable trace. It does not insert success, remove calls, deduplicate observations, or close an open path. The original compiler/executor still validates the resulting Tree. Limits are eight reads, thirty-two statements and eight levels; source navigation is capped at sixty-four paths/six levels with explicit truncation. Unsupported predicates, loops and scripts remain unsupported. An observation is not itself a checked precondition.

Final review also fixes constant-origin caching: equal value schemas at different fields must retain their distinct original const locations, including through identical nested ancestors. v21 separately verifies all three recorded wiring requests and re-derived outputs remain identical, with zero new model calls. This is a new-implementation regression, not a replacement for original-version replay or a rewrite of historical manifests.

The table above includes **all eight experiments/twelve real 9B calls**, spanning one previously known public Skill and one pre-existing synthetic wiring source—zero new public Skills. Five Netdata attempts fail to compile. Future-scenario separation temporarily improves action selection, but source duties, transport requirements and required preconditions remain confused. Thinking and readable schemas alone do not resolve that failure. No public-source candidate is executed.

The synthetic control first fails on its unreachable redundant terminal. After normalization, it reaches parameter construction but invents identifiers, which provenance checks reject. After binding-source navigation/guidance, three calls take 42.549 seconds, with 10,465 input/853 output tokens, and produce two dynamically bound reads plus a conditional. The initial planning request is byte-identical across those three control versions. This is a local mechanism gain, not generalization or a latency distribution. v16 adds scenario information; v17 changes thinking/output budget; v18 restores no-thinking and exposes the schema. These are not a single-variable, same-resource accuracy experiment.

The linked digest-bound review approves only the generated conditional read region for a synthetic test. The original Runtime passes **10/10** checks: branch selection, zero/nonzero counters, empty/invalid results, identity and resource denial, and wrong consent. Network, source-script and effect calls remain zero. Empty results stop through the existing missing-value gate, not an authored empty-list branch. Both counter outcomes hand off to L1; the zero-error decision/change-candidate logic is not translated. The remaining-duty array is empty and the terminal explanation is incomplete. Whole-Skill acceptance is explicitly false.

Use the CLI above with new directories. Optional scenario input is `netopyu.io/authoring-scenario/v1`, with original `taskDigest`, caller-provided `futureTask` and the disclosed review kind. Default output is 4096 tokens; explicit thinking permits 8192 with correspondingly less input budget, not an automatic budget expansion. Recorded failures never retry or overwrite. Every experiment replays byte-identically under its original isolated source snapshot; the summary binds 127 files and verifies 275 prior bound files unchanged. Artifacts are local, not Git-distributed.

Next prioritize source-duty accounting at partial-region boundaries and correct public-source operation/precondition/host-substitution scope. Do not infer improvement from test counts or relax gates to force success. The cross-cohort ≥50 Skill/15 repository/8 domain/600 case gate remains closed to broad Runtime evaluation.

Final validation adds 49 mechanical tests: **309 targeted passes (25.14 seconds), 2292 full-suite passes plus 81 subtests (179.83 seconds)**, Ruff on 21 related Python files, document links and whitespace checks. The earlier 2290-test full run is also retained. Changes remain uncommitted/unpushed on dev; test counts are not semantic accuracy.
