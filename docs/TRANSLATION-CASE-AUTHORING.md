# 转译用例构造与语义对齐 / Translation Case Authoring and Alignment

## 中文

### 2026-09-04：盲审协议纠偏

旧版虽隐藏 `expectedBehavior`，却把含 `nominal/ambiguous/adversarial` 的 caseId 和固定三任务顺序交给模型。因此下文历史 48/48 等结果只能作为**有类别提示风险的开发诊断**，不是严格盲审证据；旧密封报告不改写，检查器会返回 `metadataBlindingVerified=false` 并取消其自动送 Gold 排队资格。

新版采用随机盐派生匿名 ID、按匿名 ID 排序，以及**每个独立请求只审查一个任务**。适配器只能收到白名单输入，不再收到原始任务/批次 ID、仓库元数据、三类任务组序或私有映射。模型返回匿名 ID 后，运行器严格校验覆盖，再映射回原始 ID 和原始评分顺序；未知/重复 ID 不猜测修复。私有绑定、实际模型输入、检查点与结果均摘要绑定，完整运行重入只读，中断恢复不重复已完成调用。

- `model-inputs.jsonl`：实际传入模型的匿名化单任务 payload；
- `case-bindings.json`：仅运行器/评分侧使用的原始 ID 映射和随机盐，不得发送给审阅模型；
- `checkpoints/*.json`：匿名输出、恢复后的结果及二者的绑定证据；
- `reviewUnit=single_task`、`latencyUnit=task`：时延不再是旧版每三任务 Skill 组时延，不能直接按百分比比较。

原始 Skill 文本、引用、工具语义和用户问题不做语义改写，因此它们自身的示例或“生成 read candidate”等任务措辞仍可能提示答案。匿名化仅消除**评测元数据提示**，不证明测试用例自然度、完整 Skill 覆盖或模型独立性；这些必须由后续构造质量审查和独立参考答案处理。

已定位的元任务替代、参数冲突和未求值字面量见[构造质量审查](TRANSLATION-CONSTRUCT-QUALITY.md)。该审查明确区分“出现了参数字符串”和“忠实理解业务请求”。

新版已用相同 9B 制品和相同 development-07 源包完成 4 Skill/12 task 复验：协议完整、元数据盲态校验通过，但行为一致仅 **7/12（58.33%）**，构造布尔对齐 **10/12（83.33%）**，因此不能进入 Gold 队列。每 task p50/p95 为 24.1/63.9 秒。自报置信度没有触发低置信度告警，却出现“解释要求追问、结构标签写 read”的矛盾；还有上传会话 Skill 无参调用与目录必填参数不一致的问题。

这不是转译准确率退步：作者不是 Gold，ID、顺序、单任务上下文及 Prompt 同时变化，不能将差异单独归因于 ID 泄漏。结果说明旧满分不足以支持语义可靠性，当前应先修订构造与建立带源证据的隔离参考答案，不继续扩大同模型自审或 Runtime。见[方法纠偏数据](benchmarks/translation-review-blinding-v2-summary.json)。

### 为什么增加这一层

旧的 15-Skill/200-case 路径把部分真实公开 Skill 配给了通用 `resource.read/apply/restore` 工具。它可以验证接线，却不能公平回答“L1→L0 是否理解了 Skill 的真实业务语义”。新链路把用例构造放在转译器之前，并严格分离四件事：

```text
固定且不执行的 L1 Skill
  → 9B 作者候选（非 Gold）
  → 确定性结构门禁与透明规范化
  → 隐藏作者答案的 Skill–Task–Tool 审查
  → 独立 Gold
  → Gold-blind Translator 评测
  → 通过泛化准入后才评测 Runtime
```

### 当前实现

新增[源证据逐项审查](TRANSLATION-SOURCE-ALIGNMENT.md)，区分每个参数的名称、类型、必填性及工具阶段的真实来源。当前 `inspect_alignment_review` 将旧布尔满分降为 `candidateSetReadyForSourceEvidenceReview`，不再直接提供 Gold 排队资格。密封的旧评分报告及历史 `review-inspect` 字段保留供重放，不具备新的准入权。

当前作者协议为 `translation-anchored-author/v3`。它**不再自动追加参数**；候选按实际参数类型与值检查，不同于 `example_value` 的合法值不能被判成缺参。正常任务中的显式冲突、未求值占位符及无源支持的评测元任务会被拦截。结果只标记为显式参数夹具；任意自然语言冲突、真实 API Schema 与完整步骤覆盖仍未证明。详见[构造质量 v3](TRANSLATION-CONSTRUCT-QUALITY.md)。历史 v1/v2 报告按原规则只读校验，不被新规则覆盖。

`evaluation/translation_case_authoring.py` 对 71 个已知开发 Skill 按 7 个仓库聚合批次工作。每个 Skill 生成一个窄操作族和三类任务：正常、缺参追问、越界/恶意拒绝。每个候选必须满足：

- 1–4 个 `SourceAnchor` 必须逐字存在于固定 Skill/reference；
- 参数名、类型和示例值进入封闭 JSON Schema，正常任务中的每个值都有显式字面证据；
- read 无 Effect；write 最多一个 Effect 且必须审批；不可逆写风险不得低于 high；
- 通用 Tool Catalog 明确 `observe/effect/verify/compensate` 角色，但固定 `executable=false`；
- 可逆写必须有唯一补偿；不可逆写不得伪造补偿；
- 失败候选不能进入对齐审查队列；第三方脚本始终不执行。

作者规范化只进行受限的机械处理并完整记录：将 `write + none` 保守解释为 `irreversible`；把不可逆写风险抬到 high；按声明的 read/write 类型闭合审批和 Effect budget；把仅有空白、换行或大小写差异且在原文中**唯一匹配**的 quote 重绑为真实原文 span。它不再追加参数，不做编辑距离或语义模糊匹配，不能修改用户问题、read/write 意图、参数集合或处置标签，也不能把 clarify/reject 改成可执行候选；非唯一或词义变化的 anchor 继续 fail-closed。原始模型候选和显式修复版本保存在 `authoringAttempts[].modelCandidate`；引用和参数字符串匹配都不是语义证明。

通用转译 Tool Catalog 与 Fixture MCP/真实 MCP 有意分离。前者回答“这个 Skill 需要什么语义能力和事务角色”；后者回答“项目现在是否已有可执行适配器”。因此一个候选可以语义上可转译，但仍因没有 Provider/Runtime adapter 而不可执行。

### 9B development-01 迭代（2026-09-03）

使用 `qwen3.5:9b` 完成 `development-01` 的 12 个 Skill 静态作者预筛：

| 项目 | 结果 |
|---|---:|
| Skill | 12 |
| 候选任务 | 36 |
| 协议结构有效 | 12/12 |
| 通过确定性作者门禁的 Skill | 6/12 |
| 输出的盲审包 | 18 |
| 模型调用 | 18（6 个 Skill 触发修复） |
| 修复挽回 | 0/6 |
| 每 Skill 作者时延 p50 / p95 | 65.7 秒 / 205.7 秒 |
| Runtime/DSH 执行 | 0 |
| 第三方代码执行 | 0 |

首轮失败并非 Translator 失败：5 个 Skill 至少有一个 source quote 不是固定原文的精确子串，3 个 read 候选仍声称 Effect budget=1；部分 Skill 同时命中两类。该 50% 是**作者候选可审率**，不是 Translator 准确率，更不是项目成功概率。

基于这两个通用失败族增加唯一原文 span 对齐和 read/write 机械闭包后，用同一模型、同一 12-Skill 批次和实现摘要重新运行：

| 项目 | v1 | v2 |
|---|---:|---:|
| 协议结构有效 | 12/12 | 12/12 |
| 通过确定性门禁 | 6/12 | 10/12 |
| 候选可审率 | 50.0% | 83.33% |
| 盲审 task | 18 | 30 |
| 修复尝试 / 挽回 | 6 / 0 | 2 / 0 |
| 模型调用 | 18 | 14 |
| 作者时延 p50 / p95 | 65.7 / 205.7 秒 | 67.7 / 193.0 秒 |

v2 剩余两个拒绝只命中 `source_anchor_not_exact`；非唯一 span、标点或词义变化没有被模糊修复。随后对 10 个通过 Skill 的 30 个 task 进行了另一个输出目录中的答案隐藏 9B 角色审查：10/10 组协议完整，行为一致 30/30，无低置信度，p50/p95 为 26.0/43.4 秒。这个 100% 只代表**同模型角色模拟认为候选可送独立 Gold 编写**；模型答案与 Gold 均未提供给审查输入，但作者和审查者仍是同一个 9B 制品，因此它不是独立证据，也不能测量 Translator 准确率。

人工式抽查仍发现下一轮需要加强的通用门禁：禁止 `l0_read_candidate`、`l0_write_candidate`、`nominal` 等泛化 operation slug；禁止把审批状态伪装成 Tool 业务参数；增加 operation mode 与 Skill 动作语义的独立校验。它们应在更多开发批次上按失败族验证，不能针对单个 Skill 写特例。

本地完整文本制品保留逐 Skill checkpoint；可提交的聚合指标见 [development-01 v1 摘要](benchmarks/translation-authoring-development-01-summary.json)和[实现绑定的 v2 摘要](benchmarks/translation-authoring-development-01-v2-summary.json)。`alignment-review.html` 展示模型候选、规范化和失败；`review-packets.jsonl` 隐藏候选 expected behavior 与 Gold。AI 审查永远标记为 `humanIndependentEvidence=false`，只能辅助排队，不能替代独立 Gold/人工证据。

### 跨批次失败驱动改良（development-02 至 05）

后续四个已知开发批次用于寻找**通用失败族**，不同批次包含不同 Skill，且实现持续变化，所以通过率不能被解释为受控 A/B 或准确率提升：

| 批次 | Skill | 确定性门禁通过 | 首轮通过 | 修复/挽回 | 调用 | p50 / p95 秒 |
|---|---:|---:|---:|---:|---:|---:|
| development-02 | 10 | 2（20.0%） | 2 | 8 / 0 | 18 | 112.4 / 160.0 |
| development-03 | 11 | 2（18.2%） | 2 | 9 / 0 | 20 | 103.4 / 149.8 |
| development-04 | 11 | 8（72.7%） | 6 | 5 / 2 | 17 | 93.3 / 191.8 |
| development-05 | 11 | 9（81.8%） | 9 | 2 / 0 | 15 | 77.7 / 259.3 |

development-02/03 证明仅靠 Prompt 软约束无法稳定消除泛化 slug、控制字段泄漏和复制 anchor 错误。随后将 operation ID 与原文 span 绑定收归编译器：模型选择语义，编译器生成稳定 ID，并通过必填 `source_span_id` 绑定披露的精确字节；read/non-write 安全 envelope 也由确定性代码闭合。development-05 不再出现泛化 slug、缺失 span ID、read Effect 或控制参数泄漏，两个拒绝均来自同内容长 Skill 的 assignment/slot/参数类型错误。

对 development-05 的 9 个通过 Skill/27 task 做答案隐藏角色复核后，行为处置一致 26/27、完整对齐 25/27，无低置信度，p50/p95 为 31.9/61.9 秒。复核发现一个重要漏检：某候选让 ambiguous 与 adversarial 使用完全相同文本，却赋予不同处置；另有一个复核布尔值与 clarification 定义自相矛盾。原密封结果保留不改，后续代码新增“规范化后挑战文本不得重复”和“clarification 参数形态判断必须自洽”门禁。它们尚需在后续开发批次复验。

逐批实现摘要、报告 digest 和完整限制见 [development-02–05 聚合摘要](benchmarks/translation-authoring-development-02-05-summary.json)。截至这里仍未编写独立 Gold、未运行 Translator，也未运行 Runtime/DSH。

新增门禁随后以同一作者/复核实现运行 development-06/07：16/16 Skill 均首轮通过，48/48 task 的答案隐藏同模型复核行为一致且完整对齐，无低置信度；作者不需要修复调用。它证明新协议可以在两个后续已知批次上稳定工作，并完成 71 个主要开发 Skill 的失败发现覆盖，但仍不能证明独立语义准确率。实现绑定明细见 [development-06–07 摘要](benchmarks/translation-authoring-development-06-07-summary.json)。下一步是独立 Gold 和 Gold-blind Translator 诊断，而不是 Runtime。

### 使用

```bash
scripts/netopyu-market-corpus anchored-author \
  artifacts/translator-v2/development-corpus-100 \
  --output-root artifacts/translator-v2/anchored-development-01 \
  --batch-id development-01 \
  --model qwen3.5:9b

scripts/netopyu-market-corpus anchored-author-inspect \
  artifacts/translator-v2/anchored-development-01 \
  artifacts/translator-v2/development-corpus-100

scripts/netopyu-market-corpus anchored-review-run \
  artifacts/translator-v2/anchored-development-01 \
  artifacts/translator-v2/development-corpus-100 \
  --output-root artifacts/translator-v2/alignment-development-01 \
  --model qwen3.5:9b

scripts/netopyu-market-corpus anchored-review-run-inspect \
  artifacts/translator-v2/alignment-development-01 \
  artifacts/translator-v2/anchored-development-01 \
  artifacts/translator-v2/development-corpus-100
```

先扩展和审查已知开发批次，只按失败类别修改通用协议/类型/链接算法。稳定后冻结 Translator，再采集仓库隔离的新 cohort。未达到[转译泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)前，不恢复大规模 Runtime A/B。

## English

### Current author protocol: v3

The [source-evidence review](TRANSLATION-SOURCE-ALIGNMENT.md) adds exhaustive field/step obligations and resolved citations. Current alignment inspection only queues source review, not Gold authoring. Sealed legacy scores and historical `review-inspect` fields remain reproducible but do not satisfy the new prerequisite.

The normalizer never appends or replaces user parameters. Typed explicit values are independent of `example_value`; alternate valid values cannot fake missing inputs. Conflicting assignments, deferred placeholders, and unsupported evaluation meta-tasks are blocked for nominal fixtures. Original model candidates and explicit repairs are retained. Reports identify explicit-parameter fixtures and narrow operation scope, not natural-language extraction accuracy or verified source API schemas. Legacy artifacts retain versioned read-only validation. See [construct quality and remaining semantic gaps](TRANSLATION-CONSTRUCT-QUALITY.md).

### 2026-09-04 blinding correction

The legacy reviewer hid `expectedBehavior` but exposed category-bearing case IDs and a fixed nominal/ambiguous/adversarial order. Historical 48/48-style results are therefore development diagnostics with known metadata cues, not strict blind-review evidence. Sealed artifacts remain unchanged; inspection flags legacy metadata blinding as unverified and disables automatic Gold-authoring queue eligibility.

The revised protocol derives opaque IDs from a stored random salt, orders calls by those IDs, and sends exactly one task per independent request. Adapters receive only allowlisted public inputs, never the private source-ID map. Output IDs, remapping, checkpoints, and actual model inputs are digest-bound and revalidated on inspection/resume. Complete runs are read-only on reentry. `model-inputs.jsonl` records model-visible payloads; `case-bindings.json` is runner/scorer-only. Latency is now per task, not per three-task Skill group. Original Skill examples and semantic cues in user prompts remain visible by design, so metadata blinding does not establish natural-task quality, whole-Skill coverage, model independence, or translation accuracy.

See the [construct-quality audit](TRANSLATION-CONSTRUCT-QUALITY.md) for concrete meta-task, conflicting-parameter, and unresolved-literal findings.

The same 9B artifact and development-07 source packets were re-reviewed under v2: all 12 single-task calls completed and metadata blinding verified, but behavior agreement was 7/12 and construct-boolean alignment 10/12. Gold-authoring queue eligibility is false. Per-task p50/p95 was 24.1/63.9 seconds. High self-reported confidence failed to reveal explanations asking for clarification while structured labels said read; another finding concerns invented required parameters for a documented no-argument upload-session call. These are unresolved development disagreements, not measured Translator errors. IDs, ordering, singleton context, and prompt changed together, so the delta is not a single-variable leakage effect. See the [methodology diagnostic summary](benchmarks/translation-review-blinding-v2-summary.json).

The former public-Skill study could validate wiring while pairing some real Skills with generic record tools that did not represent their documented operation. The new authoring layer precedes the Translator: pinned inert Skill → non-Gold 9B candidate → deterministic structural gate and recorded mechanical normalization → answer-hidden Skill–Task–Tool review → independent Gold → gold-blind Translator evaluation → Runtime only after generalization admission.

Each candidate binds exact source quotes, scalar parameter definitions, three task challenges, and a generic non-executable Tool Catalog with explicit observe/effect/verify/compensate roles. Reversible writes require one compensation; irreversible writes cannot claim a false rollback. This generic semantic catalog is intentionally distinct from a Fixture MCP or real Provider adapter, so translation capability and current execution support can be measured separately.

The first qwen3.5:9b development batch covered 12 Skills and 36 tasks. All 12 produced protocol-valid structures; six passed the deterministic author gate and produced 18 blind-review packets. After adding unique whitespace/case-only source-span rebinding and mechanical read/write safety closure, an implementation-bound rerun accepted 10/12 Skills and emitted 30 blind tasks. The two remaining failures were non-exact anchors that the non-fuzzy aligner correctly refused. Authoring p50/p95 was 67.7/193.0 seconds.

An answer-hidden same-model role review completed all 30 tasks with 30/30 behavior agreement and no low-confidence case at 26.0/43.4 seconds p50/p95. This only queues candidates for independent Gold authoring: the author and reviewer used the same 9B artifact, so the result is neither independent evidence nor Translator accuracy. Manual-style inspection also identified generic operation slugs, Runtime-control parameters leaking into Tool schemas, and subtle read/write semantics as the next general gate families. No Runtime, DSH, Tool, MCP provider, or third-party Skill code ran.

AI-role reviews remain simulated evidence (`humanIndependentEvidence=false`). They may triage candidates for later Gold authoring but cannot unlock the translation or Runtime gate.

Development batches 02–05 were then used to expose reusable failure families. Their Skills and implementation digests differ, so the observed author-gate rates—20.0%, 18.2%, 72.7%, and 81.8%—are diagnostics rather than a controlled accuracy trend. Soft prompt constraints did not reliably prevent generic route slugs, control-field leakage, or copied-anchor errors. Operation IDs and disclosed source-span binding were therefore moved into the deterministic compiler boundary, while read and non-write safety envelopes were mechanically closed.

In development-05, 9/11 Skills passed the structural gate. The answer-hidden review agreed on behavior for 26/27 tasks and marked 25/27 fully aligned. It exposed one accepted candidate whose ambiguous and adversarial prompts were textually identical despite different labels, plus one clarification judgement that contradicted the rubric's parameter-shape semantics. The sealed report remains unchanged; subsequent code rejects normalized duplicate challenge prompts and internally inconsistent clarification reviews. These post-run gates still require validation on later development batches. No independent Gold, Translator, Runtime, or DSH execution occurred. See the [implementation-bound batch summary](benchmarks/translation-authoring-development-02-05-summary.json).

The revised author and reviewer implementations then evaluated development-06/07 without further changes. All 16 Skills passed on the first author call, and all 48 answer-hidden tasks achieved behavior agreement and full same-model alignment with no low-confidence case. This validates mechanism stability across two later known batches and completes failure-discovery authoring coverage for all 71 primary development Skills; it does not establish independent semantic accuracy. See the [development-06–07 implementation-bound summary](benchmarks/translation-authoring-development-06-07-summary.json). Independent Gold and gold-blind Translator diagnostics come next; Runtime remains locked.
