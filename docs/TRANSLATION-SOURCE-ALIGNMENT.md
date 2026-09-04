# 转译源证据对齐 / Translation Source Evidence Alignment

## 中文

### 本阶段解决什么

结构合法不等于业务正确；审阅模型给出五个 `true` 也不能解释一个参数从哪里来。本阶段将整体布尔判断拆成**由代码生成、逐项引用原文的声明审查**。它位于测试候选构造之后、参考答案编写之前，不是 Translator，也不执行 L0 或第三方脚本。

具体例子：`doc-ingest-analyze` 的准备步骤明确调用 `storage_ingest_upload({})`、无参数；旧候选目录却要求 `file_type_hint` 与 `max_size_hint`。不能因为这两个值出现在任务中，就认为目录忠实反映源 API。新审查分别检查完整入参形状、参数存在性、类型和必填性，并给出引用与修改位置。

### 输入、审查与边界

1. 从密封作者制品中选择一个已有审查包。随机匿名 ID 保留，作者标签、rationale、样例答案、旧审阅结论和私有映射不进入模型。
2. 编译器从原始 Skill/reference 文本生成稳定 source span ID；任务原文另占一个 span。源码只是引用数据，不加载脚本。
3. 根据惰性 Tool Catalog 自动生成所有检查项，模型不能自行挑选容易通过的项：
   - 任务是否保留源业务目的、参数解释是否一致；
   - 每个工具的操作映射、读写性质、阶段可用性、完整输入形状；
   - 每个参数的存在性、类型与必填性。
4. 每项只能返回 `supported`、`contradicted` 或 `insufficient_evidence`，附原文 span ID 和解释；非通过项必须给出具体修改建议。
5. 确定性代码检查所有项恰好覆盖一次、引用存在、证据类型正确、输入摘要一致，再还原原文文件、字符位置及引用全文。API 声明不能仅引用用户任务，更不能拿作者目录本身作为源 API 证据。

最终状态由代码生成：

| 状态 | 含义 | 下一步 |
|---|---|---|
| `protocol_failed` | 缺项、错 ID、坏引用或调用失败 | 修复审查协议，不能补猜结果 |
| `revise_construct` | 至少一个声明被审阅模型判为与引用冲突 | 定位 `pointer`，修订任务/目录后另建版本 |
| `needs_source_evidence` | 没有已报冲突，但至少一项依据不足 | 补充可审查接口定义或保留 L1，不虚构参数/步骤 |
| `ready_for_reference_drafting_review` | 每项都有模型判定的源支持 | 可进入参考答案草稿审查，不等于 Gold 或执行授权 |

`supportedClaimFraction` 是声明支持占比，不是语义置信概率或转译准确率；`claimCoverage=1` 只指目录声明被逐项审查，**不是完整 Skill 步骤覆盖率**。同一源引用可能被错误解释：`citationBindingVerified=true` 不意味着 `semanticEntailmentProven=true`。这两个字段刻意分开；实际 API Schema、全 Skill 覆盖、独立人工证据和 Runtime 授权均不自动成立。

缺少类型声明只能报告证据不足。准备会话也可能改变状态，不能看到“prepare”就默认只读。源文未声明验证/快照/补偿时，目录拥有这些 phase 并不能证明它们存在。只审查已披露的源文本；丢失引用、外部文档和隐含语义仍是开放问题。

### 本地使用

在项目根目录运行，输出目录必须是新的：

```bash
python -m evaluation.translation_source_review prepare \
  artifacts/translator-v2/anchored-development-07-v1 \
  artifacts/translator-v2/development-corpus-100 \
  --case-id development-07-004-nominal \
  --output-root artifacts/translator-v2/source-alignment-doc-ingest-v1/input

python -m evaluation.translation_source_review run \
  artifacts/translator-v2/source-alignment-doc-ingest-v1/input \
  artifacts/translator-v2/source-alignment-doc-ingest-v1/review

python -m evaluation.translation_source_review inspect \
  artifacts/translator-v2/source-alignment-doc-ingest-v1/input \
  artifacts/translator-v2/source-alignment-doc-ingest-v1/review
```

真实调用固定本地 Ollama `qwen3.5:9b`，`think=false`，单任务新上下文，无脚本工具；只请求一次，不把参考答案或上一轮审阅回灌给模型。完整结果重入只读，不重复调用；中断的不完整单次请求保留原目录并报错，不自动重放。

- `input/model-input.json`：模型实际接收的源片段、工具目录、检查项；
- `input/private-binding.json`：只供评分侧定位原案例，模型不可见；
- `review/response.json`：原始模型输出和调用指标；
- `review/report.json`：逐项 verdict、JSON pointer、可定位引用、解释和修改建议；
- manifest/seal：输入来源、实际模型制品、实现、响应与报告的摘要绑定。历史制品不重写。

协议失败时 `report.json` 不提供已接受的逐项判断，原始结果仍保存在 `response.json`，不得替模型补引用或悄悄改变判定。

当前 `inspect_alignment_review` 把旧整体布尔满分降为 `candidateSetReadyForSourceEvidenceReview`，不再直接返回可编写人工 Gold 的资格；历史评分文件和 `review-inspect` 的旧字段仅保留用于重放，不代表新准入。源证据审查、隔离参考答案和真正的 Gold-blind Translator 评测仍是不同阶段。

### 首轮真实诊断与设计缺口

使用相同本地 9B 制品审查 doc-ingest-analyze：1 Skill、1 task、12 个检查项、1 次调用，106.2 秒。原始输出包含 7 个 `contradicted`、3 个 `supported` 和 2 个 `insufficient_evidence`；这些**不是已验证成绩**，因为前两项没有引用任务原文，整份审查为 `protocol_failed`。模型确实定位了无参调用与必填参数冲突，但将准备/生成上传 URL 判为只读，超出了源文提供的依据。结果按原样封存，不补引用、不替换标签。

定向测试 89 项、全量 676 项测试与 81 个子测试通过；制品、实现和模型摘要见[首轮诊断数据](benchmarks/translation-source-alignment-v1-summary.json)。

另一个已在代码中确认的设计限制是：旧作者 `OperationFamily.parameters` 至少要求一个参数，`_parameter_schema` 将所有参数列为必填，`_slots` 固定包含缺参追问任务。**源 API 即便无参数或参数可选，也无法被当前作者格式忠实表达。** 因此后续不能只调整提示词或归咎于小模型；需要放开参数形状、区分缺参挑战是否适用，并显式保留无法依据源文判定的 Effect/验证/恢复语义。该构造格式修订尚未完成。

## English

### Purpose and mechanism

This stage reviews source-grounded declarations after construct authoring and before reference-answer drafting. It is not a Translator and never executes L0, Runtime, or package scripts. For example, the documented no-argument `storage_ingest_upload({})` call must not acquire required hints merely because a synthetic catalog and task mention them.

A compiler derives an exhaustive checklist from one answer-hidden, opaque-ID task and its inert catalog: business-task fidelity and parameter interpretation; every tool's operation mapping, effect classification, phase availability and complete input shape; and every parameter's existence, type and requiredness. The reviewer cannot choose a convenient subset. Each claim receives `supported`, `contradicted`, or `insufficient_evidence`, source-span IDs, an explanation and an actionable revision for non-supported claims.

The evaluator rejects missing/duplicate claims, unknown citations, wrong input bindings, and inappropriate evidence types. User task text cannot establish an API schema. Resolved citations expose source paths, character offsets and exact text alongside JSON pointers into the construct. Status is derived: protocol failure, revise construct, obtain source evidence, or proceed to reference-drafting review.

Exact citation binding is not semantic entailment. Claim coverage is not whole-Skill coverage. Support fraction is not calibrated confidence or translation accuracy. API schema verification, independent-human evidence, whole-Skill coverage and Runtime authority remain false. Missing documentation, external references, implicit behavior, natural-language conflicts and model interpretation errors remain limitations. In particular, preparation may mutate state; a catalog cannot manufacture verification, snapshots or compensation.

### Local workflow and artifacts

Use the three commands above to prepare one sealed packet, run local `qwen3.5:9b` with a fresh non-thinking context, and inspect the bound result. Model-visible input excludes author answers, previous reviews and private mappings. One call is made without answer-based repair. Completed outputs are read-only on reentry; incomplete requests are preserved and rejected rather than silently replayed. Original model output, source provenance, model/implementation digests and per-field findings remain inspectable in separate artifacts. Old boolean agreement only queues source-evidence review; it no longer directly qualifies a candidate for Gold authoring.

The first real probe used one Skill/task, 12 claims and one call (106.2 seconds). Raw output reported seven contradictions, three supports and two unknowns, but omitted task citations for two claims, so the entire review failed the protocol. These counts are not validated scores. Read-only classification was also inferred beyond the source evidence. Raw output remains unchanged.

Targeted regression passed 89 tests; full regression passed 676 tests plus 81 subtests. See the [version-bound diagnostic summary](benchmarks/translation-source-alignment-v1-summary.json).

The remaining defect is not merely model prompting: the author format requires at least one parameter, marks every parameter required, and fixes a missing-parameter challenge for every operation. It therefore cannot faithfully represent zero/optional-argument APIs. Revising those constructs and declaring challenge applicability are next, alongside explicit unknown effect/verification/recovery semantics. This redesign is not yet complete.
