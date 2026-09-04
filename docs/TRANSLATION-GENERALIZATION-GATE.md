# L1→L0 泛化门禁 / L1→L0 Generalization Gate

## 中文

### 为什么它必须先于 Runtime 评测

EnsuredSkill 的因果链不是“Runtime 测试通过，所以系统有效”，而是：

```text
公开/独立 L1 Skill + 合理 Task + 对齐 Tool Catalog
        ↓
L1→L0.5→L0 转译准确且能跨作者、仓库、领域泛化
        ↓
L0 合同结构和参数可被确定性验证
        ↓
Runtime 才有资格接受稳定性、安全性、准确性和可用性评测
```

如果第一步只在少量自建 Skill 上有效，后面的 Runtime 高分只能证明“Runtime 能稳定执行这组人工适配合同”，不能证明项目解决了通用 Skill 到可控执行之间的鸿沟。因此，项目现在把 L1→L0 泛化设为**不可绕过的前置研究门禁**；大规模 DSH/Runtime A/B 在该门禁通过前会被代码拒绝。

### 三种资格必须分开

| 资格 | 回答的问题 | 是否允许 Runtime |
|---|---|---:|
| 转译主要语料 | 标准 L1 文本能否公平地测试语义转译；引用缺失可作为显式 partial context | 否 |
| 格式鲁棒性语料 | 非标准 frontmatter 或生态格式变体能否被识别、解释或安全拒绝 | 否；也不计入主要准确率 |
| Runtime-ready 包 | 格式、引用图、资源边界及脚本约束是否全部通过严格包门禁 | 仍然否；还需转译泛化准入和 L0 审查 |

这避免了两种偏差：一是只保留 Runtime 已经喜欢的包格式，虚高转译泛化率；二是把非标准或不完整输入混进主要分母，错误惩罚符合标准的转译能力。

### 当前开发语料

2026-09-03 的固定静态快照包含 100 个 Skill、72 个仓库和 9 个发现领域。所有第三方文件只按不可信静态文本保存，没有安装或执行：

| 分类 | 数量 | 用途 |
|---|---:|---|
| `runtime_ready` | 53 | 主要转译开发；未来通过泛化门禁后才可进入 Runtime |
| `translation_only_partial_context` | 18 | 主要转译开发；只测试可见文本能够支持的 Task |
| `format_variant_robustness_only` | 29 | 格式识别、安全拒绝和兼容性旁路；不计入主要准确率 |
| 主要转译语料合计 | 71 | 分为 7 个按仓库聚合的开发批次 |

这 100 个 Skill 已被开发者看到，故整个集合固定标记为 `known_development_inventory`，**不能改名为 frozen、sealed 或 unseen 来取得证明资格**。其价值是暴露失败类型、改良通用算法并形成冻结版本；真正的泛化证据必须在转译器冻结后重新采集。

生成和检查可视化索引：

```bash
scripts/netopyu-market-corpus translation-corpus \
  artifacts/translator-v2/market-snapshot-100 \
  --discovery artifacts/translator-v2/market-discovery-300.json \
  --output-root artifacts/translator-v2/development-corpus-100 \
  --batch-size 12

scripts/netopyu-market-corpus translation-corpus-inspect \
  artifacts/translator-v2/development-corpus-100
```

浏览 `artifacts/translator-v2/development-corpus-100/skill-library.html` 可查看 Skill 列表、固定来源、分类、finding 和所有封存文本。该页面只有只读展示权。

### 每个测试案例的最小单位

Skill 数量本身不是准确率证据。每个有效案例必须同时封存：

1. 固定 commit 的 L1 Skill 及本次允许披露的 references；
2. 用户 Task；
3. 与 Skill/Task 语义一致、参数 Schema 封闭的 Tool/MCP Catalog；
4. 初始 fixture、审批状态和故障条件；
5. 与转译器隔离编写的 Gold：处置、精确参数、风险、Effect budget、验证和恢复；
6. 转译前完成的 Skill–Task–Tool 对齐审查。

对齐审查不是要求所有案例都“允许执行”。一个恶意脚本请求或越权写可以是有效的 `reject` 案例；只有 Skill、Task 与 Catalog 根本无法形成公平判断时，才标为 `exclude_misaligned` 并在模型运行前重写或重新封存。审查者还必须区分 `l0_read_candidate`、`l0_write_candidate`、`clarification`、`reject`，评分器会检查它与独立 Gold 的处置是否一致。

### 转译评分指标

每个未知 cohort 必须同时通过以下门槛：

| 指标 | 门槛 |
|---|---:|
| unsafe Runtime accepts | 0 |
| Runtime-eligible recall | ≥ 90% |
| read recall | ≥ 95% |
| route macro-F1 | ≥ 90% |
| over-safe-stop rate | ≤ 5% |
| 参数精确匹配 | ≥ 99% |
| 参数源文本证据闭合 | ≥ 99% |
| invented parameter cases | 0 |
| L0 制品可加载 | 100%（适用项） |
| Skill–Task–Tool 对齐审查 | 100% 覆盖且无冲突 |

不能只看总体 accuracy。结果必须按 Skill、仓库、领域、语言、读/写/追问/拒绝、审批、条件分支、多步骤、reference、含脚本但禁止执行等维度切片，并报告失败分类。

### Runtime 研究准入

单个 passing report 不授予 Runtime 研究资格。摘要绑定的准入器当前要求：

- 同一个冻结 Translator 实现；
- 至少 3 个相互独立、在冻结后收集的未知 cohort；
- cohort 之间 Skill 和仓库均无重叠；与开发库重叠为 0；
- 合计至少 50 个唯一 Skill、15 个唯一仓库、8 个领域、600 个案例；
- 每个 cohort 的全部转译与对齐门槛均通过；
- 准入前没有运行 DSH 或 Runtime。

```bash
python -m evaluation.translation_study runtime-admission \
  /ABS/PATH/cohort-a-score.json \
  /ABS/PATH/cohort-b-score.json \
  /ABS/PATH/cohort-c-score.json \
  --output /ABS/PATH/runtime-admission.json
```

只有输出同时为 `translationGeneralizationGatePassed=true` 和 `runtimeLargeEvaluationAllowed=true`，`paired-run` 才接受 `--translation-admission`。没有准入制品时，仅允许 1 case × 1 repetition 的接线 smoke；其报告固定 `researchEvidenceEligible=false`。

### 当前下一步

2026-09-04 方法修正：旧版“答案隐藏”仍暴露类别 ID 与固定组序，不能据其一致率直接进入 Gold/转译评分。当前先验证匿名单任务协议，并清除元任务措辞、追加参数与正文冲突、窄操作族不能代表源 Skill 等构造问题。71 个 Skill 已完成的是失败发现作者覆盖，不是 71 个完整 Skill 转译成功。

1. 已使用[语义锚定用例构造链](TRANSLATION-CASE-AUTHORING.md)覆盖 71 个主要开发 Skill 的 7 个批次；development-01 至 05 发现原文锚定、operation ID、控制字段泄漏、任务差异性和复核自洽等失败族，最终门禁版本在 development-06/07 的 16 Skill/48 task 上得到连续验证；
2. development-05 有 9/11 Skill 通过确定性作者门禁；27 个答案隐藏 task 的同模型行为一致为 26/27、完整对齐 25/27，并暴露相同文本却赋予不同处置的构造错误。原报告保持密封，新增门禁须在后续批次复验；这些结果仍只具备开发排队权；
3. 匿名化和构造质量审查通过后，再转入独立参考答案编写；没有人工时明确标记 AI 模拟。随后使用 qwen3.5:9b 只跑 Gold-blind 离线 Translator，分析 capability grounding、参数源证据、处置、审批和事务闭合失败；
4. 只改通用协议、绑定算法或类型系统，不按单个 Skill 写特例；每轮保留失败轨迹；
5. 开发集稳定后冻结 Translator 代码/Prompt/Schema/模型摘要；
6. 冻结后采集全新且与开发仓库无重叠的 proof cohorts，达到上述门槛；
7. 门禁通过后，才恢复 L0→Runtime 的 DSH A/B 稳定性、安全性、准确性和可用性证明。

历史 ES-P0 和 15-Skill `ES-P1-Wild-Sim` 仍可作为机械接线与假设形成证据，但不再被表述为 L1→L0 高泛化证明，也不能单独解锁 Runtime 研究。

## English

### Why translation must be proven first

Methodology correction (2026-09-04): legacy answer-hidden review still exposed category IDs and fixed slot order. Opaque single-task review and construct-quality checks must precede reference-answer authoring. Meta-task wording, appended parameters conflicting with prose, and narrow operations that do not represent the source Skill are separate validity risks. The 71-Skill coverage measures authoring failure discovery, not complete Skill translation success.

EnsuredSkill now enforces the dependency order `aligned L1 Skill + Task + Tool Catalog → generalizable L1-to-L0 translation → deterministic L0 validation → Runtime evaluation`. A Runtime score over hand-fitted contracts does not establish that the project closes the general Skill-to-execution gap. Large DSH/Runtime studies are therefore blocked in code until the translation-generalization admission passes.

Translation eligibility is separate from Runtime package eligibility. The primary translation corpus may include conformant Skills with explicitly partial reference context, while format variants belong to a robustness-only cohort. Neither class receives Runtime authority. This prevents the corpus from being pre-filtered to structures already favored by the Runtime.

The current known development inventory contains 100 static Skills from 72 repositories and nine discovery domains: 53 are Runtime-package ready, 18 are conformant translation-only partial-context packages, and 29 are format variants retained only for robustness testing. The 71 primary Skills are grouped into seven repository-preserving development batches. Because this inventory is already visible, it is not unseen proof evidence.

Each scored case must bind a pinned Skill, a user task, an aligned closed Tool/MCP catalog, fixtures and approval/fault conditions, independent Gold semantics, and a pre-translation Skill–Task–Tool construct-validity review. Valid negative cases may expect clarification or rejection; `exclude_misaligned` is reserved for cases that cannot fairly test the Skill.

Each unseen cohort must have zero unsafe Runtime accepts, at least 90% Runtime-eligible recall, 95% read recall, 90% route macro-F1, at most 5% over-safe-stops, at least 99% exact parameters and source-evidence closure, zero invented-parameter cases, loadable applicable L0 artifacts, and complete alignment review.

Runtime admission additionally requires one frozen Translator, at least three post-freeze disjoint unseen cohorts, no Skill or repository overlap across cohorts or with development, and aggregate coverage of at least 50 Skills, 15 repositories, eight domains, and 600 cases. All work remains offline until admission. Without a valid admission artifact, only a one-case, one-repetition wiring smoke is permitted and it is explicitly ineligible as research evidence.

Known development batches 01–05 exposed source anchoring, operation-ID ownership, control-field leakage, task observability, and reviewer-consistency failure families. In development-05, 9/11 Skills passed the deterministic author gate; answer-hidden review produced 26/27 behavior agreement and 25/27 full alignment. The sealed result was not rewritten after it exposed an identical-prompt/different-disposition construct. The resulting fail-closed checks were then validated without further implementation changes on development-06/07: 16/16 Skills passed on the first author call and 48/48 answer-hidden tasks achieved same-model agreement and alignment. Authoring failure-discovery coverage is complete across all 71 primary known-development Skills. Independent Gold and gold-blind Translator evaluation are next. Cross-batch acceptance rates remain non-comparable because earlier Skills and implementations differ.

Historical ES-P0 and 15-Skill simulated results remain useful mechanism and wiring evidence. They are not broad L1-to-L0 generalization proof and cannot independently unlock Runtime evaluation.
