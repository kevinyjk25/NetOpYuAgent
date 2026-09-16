# 语义闭环首次迁移结果 / First Semantic Closure Transfer

## 中文

2026-09-11，**冻结 v1 未通过原型阶段出口**。6 个不同 Skill、6 个仓库、6 个开发者分类领域，共 12 个事先固定的限定任务。来源、任务、合成观察和逐项判据都在模型运行前封存。全部采用本地 `qwen3.5:9b`；没有执行第三方脚本、厂商接口或真实写操作。

### 结果与成本

| 指标 | 首次结果 |
|---|---:|
| 限定任务完成 | 1 / 12 |
| 正确边界拒绝 | 1 / 12 |
| 局部可用 | 7 / 12 |
| 构图失败 | 3 / 12 |
| 逐项判据 | 22 满足 / 13 不满足 / 1 未知 |
| 完成实质性任务的不同 Skill | 1；出口至少要求 3 |
| 观察到的越权工具调用 / 错误完成操作声明 | 0 / 0；仅指本批已检查轨迹 |
| 实际模型调用 | 56 |
| 输入 / 输出 token | 305,003 / 26,423 |
| 单次模型调用 p50 / p95 | 27.647 / 96.368 秒 |

这不是任务时延、SLO、独立 Gold、因果 A/B 或生产成功概率。部分调用与 pytest 并行。共同宿主只有带独立有限资源 ACL 的合成 `read_export(path)`；评测范围是明确限定的分析任务，不是原 Skill 的整个工作流或各厂商原生工具的认证。

| 任务 | 人工式开发审阅 | 关键现象 |
|---|---|---|
| incident-identity | 局部 | 原先有用的身份/环境追问被摘录替换 |
| incident-partial | 失败 | 正确首读之后多读 Skill 文件，来源引用不支持路径 |
| marketing-clarify | 失败 | 无效注释引用使整份有效前段被拒 |
| marketing-fit | 局部 | 未完整回应时间窗/预算；审查输入超预算 |
| metrics-correlation | 局部 | 已实际补读，但 1%→6% 分析被原始计数替换 |
| metrics-gap | 局部 | 缺失不等于零的解释/后续检查被删 |
| network-dedup | 局部 | 2 条汇总/20 次抑制的分析被原始行替换 |
| network-no-journal | 局部 | 曾编造丢弃原因；最后只剩摘录，未完成解释 |
| tests-resource-boundary | 正确拒绝 | 未授权资源在 provider 前拒绝，工具/模型/脚本/效果均未调用 |
| tests-retries | 限定任务完成 | 正确区分重试挽救与最终失败；但审查因缺少修改建议被阻断，**内容完成≠流水线成功** |
| writing-rewrite | 局部 | 大体保留命令，但单列 Retry once 丢失 E_TIMEOUT 条件；审查超预算 |
| writing-strict-limit | 失败 | 多余的 Skill/checklist 读取缺少有据路径绑定 |

### 为什么不是再多跑几次

根因不是单一模型大小：无效附属注释和后续读取污染有效前段；“复制来源”与“回答任务”被错误地做成替代关系；重复传输文本浪费上下文；审查器把合理不写的内容和缺少修改建议处理成整个接口失败。图运行完成、引用原样复制、AI `supported` 都没有检查完整任务是否保留。

当前通用修复方向：

1. 只投影已经通过原编译器的连续读取前段；首个坏节点及后续节点全部拒绝，原始失败和无效注释留档。候选仍须独立准入。
2. 答复编辑与精确依据分成独立通道。引用不能覆盖推理/追问；原有未确认 notes 和宿主职责保留出处。矛盾仍须改稿或保持不接受，不能靠附录中的正确资料免责。
3. 审查传输使用可精确逆转的重复文本池/表格编码，不截断原文、不加大预算；其对模型理解的效果仍需实测。
4. 没有读取不制造 `{}` 观察；有实际回执的空对象仍是观察。无修改建议的负面审查保留为未解决发现，不升级为支持，也不替模型补写建议。

旧批次不改分、不重跑为“未见”。新机制先在这些已知开发样本验证；随后重新冻结再选择新来源。[固定出口](SEMANTIC-CLOSURE-EXIT.md)和[完整诊断](SEMANTIC-CLOSURE-DIAGNOSIS.md)保持可查。

### 证据定位

[机器摘要](benchmarks/semantic-closure-transfer-v1.json)随 Git 文档保存。完整本机制品在 `artifacts/translator-v2/semantic-closure-transfer-20260911-v1`，被 Git 忽略，并不自动备份：

- 冻结摘要：`5dc7fd164e1138bb0e5eb1a7628bb19017e84b4ab5aab0f698ab3e5e42a849af`。
- 验收摘要：`ada85c98f37b406f8ab45778035204a9513306ab026ff98ffec9e6a3c2179372`。
- 调用/归档审计摘要：`d63ddd6b36549fb98c6e040685a4b12e6d33d15c10aa0383f596c636b1ff18dd`。

各 case 的 `judgment.json` 绑定实际输入与输出文件摘要。已有审阅不是独立人工标签，不应用来报告广泛泛化。此前已知开发成本另外计算：9 月 10 日目录 50 调用、426,879 / 53,886 token；9 月 11 日旧开发目录 84 调用、540,585 / 19,200 token。它们不是这 56 次迁移调用，也不能与随后新机制的调用重复累计。后者大型冻结元数据审计已修复专用读取上限，原 Runtime 输入上限未放宽。

## English

Frozen transfer v1 **failed the prototype exit**: six distinct Skills, six repositories, six developer-classified domains and twelve predeclared scoped tasks. Local qwen3.5:9b produced one fulfilled task, one correct boundary refusal, seven partial results and three authoring failures. Of 36 criteria, 22 were met, 13 unmet and one unknown. Only one Skill fulfilled a substantive task; the exit requires at least three, sufficient coverage and no critical task mismatch.

All 56 Chat calls are retained, including failures: 305,003 input / 26,423 output tokens; call p50/p95 27.647/96.368 seconds. No unauthorized provider invocation or false completed-operation claim was observed in the inspected traces. These are not production probabilities, task latency, an SLO, causal A/B or independent Gold. The shared synthetic read-export adapter has an independently enforced finite ACL; vendor endpoints and third-party scripts were never executed. These scoped analyses do not certify whole native Skill workflows.

Failures reveal interface and representation defects: invalid annotations or later reads poison valid prefixes; exact source copying replaces useful questions, calculations and explanations; repeated review text exhausts the byte proxy; a negative opinion without a suggested edit invalidates a review. The test-retry answer fulfilled its scoped request even though its review pipeline failed. Conversely, five completed repair graphs did not establish semantic success.

The new development mechanism separates answer edits from exact supporting observations, retains unresolved notes/duties with provenance, projects only independently valid contiguous read prefixes, losslessly deduplicates review transport, and distinguishes absent reads from actual empty observations. Negative findings without actionable suggestions stay unresolved rather than becoming support. These are hypotheses requiring real-model verification, not claimed semantic improvements yet. Correct citations cannot excuse false prose.

The old batch and its judgments remain immutable. Known-source diagnostics cannot satisfy new-source acceptance. A new implementation must be frozen before selecting the next sources. See [exit criteria](SEMANTIC-CLOSURE-EXIT.md), [diagnosis](SEMANTIC-CLOSURE-DIAGNOSIS.md), [reproduction](SEMANTIC-CLOSURE-RUNBOOK.md) and the [portable digest-bound summary](benchmarks/semantic-closure-transfer-v1.json). Complete ignored local artifacts are not automatically backed up by Git. Earlier development directories separately retain 50 calls (426,879/53,886 tokens) for September 10 and 84 (540,585/19,200) for September 11; do not double-count them with this batch or later runs.
