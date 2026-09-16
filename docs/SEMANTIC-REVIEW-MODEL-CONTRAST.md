# 9B / 27B 语义复核对照 / Semantic Review Model Contrast

## 中文

2026-09-14。**对照已完成，语义闭环阶段仍未完成。当前不推荐把全面切换 27B 当作主要修复方案。** 四个已知失败例中，27B 新发现了一例的目标遗漏；另外三例未显示对预定关键缺陷的实质改善。它也产生了引用缺失和不恰当的补充要求。这不是 25% 准确率，更不是转译或 Runtime 成功率。

### 固定了什么

- 保留原 9B 生成的稿件、任务、完整已供给来源、Prompt、Schema、seed、temperature=0、think=false、49,152 上下文和 4,096 输出 token 上限；模型请求只改变 model。
- 使用本地 `qwen3.6:27b`，制品摘要 `a50eda8ed977ab48a12431878896b27ffd5cef552c17af3317d9623b939a7f1e`。9B 制品摘要仍为 `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。
- 复用历史 9B 复核，不重新生成答案或改写其评分。IRQL/Mesh 使用各自原来的对象键接口，CAPA/Phoenix 使用各自原来的数组接口；每一对内部格式相同，不把跨例接口差异隐藏起来。
- 只有审阅，没有编辑、业务工具、Runtime 执行、源脚本执行、付费 API 或默认模型切换。评审是开发者 AI 对实际原文/回复的逐项检查，不是独立 Gold。

### 逐例结果

| 已知问题 | 原 9B 复核 | 27B 复核 | 本轮解释 |
|---|---|---|---|
| IRQL：正文写了半开时间窗，查询并未正确实现 | 仍支持错误查询 | 仍支持；另有 3 项肯定判断无引用，接口拒绝 | 未修复；“文字描述一致”不证明代码等价 |
| Mesh：已读指标被忽略，仍要求重复读取 | 发现指标遗漏，但认可部分陈旧缺数据表述 | 同样有该矛盾；4 个结构片段判断缺引用，接口拒绝 | 局部诊断仍不完整，没有显示实质改善 |
| CAPA：把“未批准”扩大为“未审阅或批准” | 漏判扩大后的语义 | 仍漏判；额外要求补背景信息 | 新增背景意见不等于修复目标错误 |
| Phoenix：没有提供要求的两项后续建议 | 错报已经提供 | 正确指出遗漏；仍有多余/错误的补充理由 | 一项具体检出改善，不是任务已修复 |

四份 27B 完整回复中，2 份通过接口校验、2 份未通过。接口通过不等于语义通过；接口失败的原始内容也保留供诊断。Phoenix 的 `c017` 错称 notes 没有明确写出成本/现网状态缺失，实际 note 1 已写明；这说明它仍可能引导不必要修改。

### 时间与成本

首个 IRQL 请求达到旧客户端 360 秒时限，返回 `ReadTimeout`。Ollama 日志随后确认该任务取消、槽位释放且空闲。保留这次失败后，单独冻结恢复尝试，把**客户端等待**延长至 900 秒，未增加模型 token 预算；不是两边使用同一时限的时延实验。

| 案例 | 原 9B 复核秒数 | 27B 完整回复秒数 |
|---|---:|---:|
| IRQL | 111.32 | 340.79 |
| Mesh | 113.54 | 379.64 |
| CAPA | 67.18 | 319.75 |
| Phoenix | 61.70 | 303.57 |

共 **5 次真实尝试：1 次超时＋4 次完整回复**。后四次已知用量 **21,421 输入＋8,391 输出 token**；首个超时没有返回用量，记为未知，不能说总成本只有 29,812 token。四个完整回复的请求 p50/p95 为 **330.27/373.81 秒**，含请求等待，不含前置检查。历史运行、缓存、模型加载及不同等待时限未控制，不作因果性能提升、SLO 或端到端任务时延结论。

### 本轮修复与下一步

对照结束后，已修复一个确定的通用实现缺陷：句子切分器把 `1. 正文`、`## 3. 标题` 拆出孤立编号。新 `bounded-draft-review/v8` 保留编号与同一行正文，保留所有原文、精确位置和独立否定句，不删除事实或放宽引用/权限要求。普通数值句末、空项边界、代码与小数仍需保持原语义。该修改不追溯重算本轮 9B/27B 输入或意见，也不代表所有 Markdown 分块问题已解决。

验证：76 项定向、2,901 项全量＋81 子测试（234.72 秒）、变更代码 Ruff、文档链接和 diff 检查通过。实际 Mesh 旧稿的 3 个孤立编号在新分块中消除，稿件分块由 13 变为 10；原候选、来源和全部非空白字符一致，位置精确。该零调用检查不重新生成审阅意见或改写旧语义结果。

接下来优先修机制，保持 9B 主路径：

1. 让任务职责和“实际如何满足”可定位对应，避免仅验证答案是否抄到了要求；已有三类审阅字段本身不足以保证这种对应。
2. 区分“某个索引无数据”与“整个已读证据集合无数据”，并检查原子动作/否定条件的扩大。仅有来源 ID 不构成语义支持。
3. 对代码、参数、数值和条件等可检查工件引入适用的确定性验证；开放语义继续保留有界 LLM 与未验证状态，不能由 supported 自行批准。

这些是待验证方向，不宣称已经实施完成。后续仍须已知回归后重新冻结新样本，满足[原阶段出口](SEMANTIC-CLOSURE-EXIT.md)；不扩大正式 A/B 或生产工程。

### 证据与复查

本地目录前缀为 `artifacts/translator-v2/`：

- `semantic-review-27b-20260914/`：首轮脚本、冻结、超时请求/回执。
- `semantic-review-27b-20260914-longwait/`：独立冻结、四例原始请求/回复、逐例 `judgments.json` 和 `comparison/report.json`。
- `semantic-review-27b-20260914-longwait-final-audit/`：来源归档、回执和报告完整性审计。

比较报告摘要：`7b832f5a1781f12f0d06f371bcfaa06a7deab122d1f7207747f0faad417da208`；最终审计摘要：`7d1b3ac862c542e304dd534a9b3068353bb1fee1d27a631bf00ff7edaa3a9104`。原始制品被 Git 忽略，本说明不表示它们已提交或远端备份。源码继续改动后不能重新运行冻结 driver；离线审计可以验证当时归档，不要求篡改旧摘要。

## English

The four-known-case review-only contrast completed on September 14, 2026; **the semantic-closure stage did not**. This evidence does not justify a wholesale switch to local 27B. It newly detects the missing Phoenix follow-up requirement, while the query implementation, stale mesh-data interpretation and CAPA predicate expansion remain unresolved. This is one specific diagnostic improvement, not a 25% accuracy score or a translation/Runtime success rate.

Existing 9B candidates and exact model-visible messages, sources, schema, seed, temperature, think=false and token limits were retained. Only the requested local model changed. IRQL/Mesh retain keyed schemas, while CAPA/Phoenix retain their original array schemas; each pair is internally matched. There were no writers, editors, business tools, Runtime executions, paid APIs or default model-binding changes. The content inspection is developer-AI diagnosis, not independent Gold or unseen generalization.

Two 27B responses pass binding; two fail because positive judgments omit citations. Both raw and normalized outcomes remain. IRQL still approves the incorrect query; mesh simultaneously notices missing metrics and supports stale absence claims; CAPA still expands not-approved into not-reviewed-or-approved. Phoenix correctly flags two missing instrumentation suggestions, but also offers unnecessary/incorrect omission rationales. More negative findings are not automatically better review.

The first IRQL attempt timed out after 360 seconds. Server logs confirmed cancellation and idle state. A separate frozen recovery extends only client waiting to 900 seconds, not the 4,096-output-token budget. All five attempts remain: one timeout plus four complete responses. Known usage is 21,421 input and 8,391 output tokens, with the timed-out attempt's usage unknown. The complete-response latency table above and p50/p95 of 330.27/373.81 seconds are request timings, not causal speedups, task latency or SLOs; historical runs, caching/loading and deadline differences are uncontrolled.

After the contrast ended, a generic segmentation bug was fixed in bounded-draft-review/v8: ordered-list and numbered-heading labels remain attached to same-line bodies instead of becoming isolated assertions. Original text/offsets, independent negation sentences and authority restrictions remain. This does not repair every Markdown or semantic issue, regrade the contrast or establish a model-accuracy gain.

QA passes 76 targeted tests, 2,901 full tests plus 81 subtests (234.72 seconds), changed-code lint and documentation/diff checks. Zero-call inspection of the actual mesh candidate removes three isolated labels (13 to 10 draft spans), retaining sources, candidate bytes, non-whitespace coverage and exact locations. No old semantic opinions are regenerated or regraded.

Next priorities are task-to-artifact fulfillment evidence, whole-observation scope and predicate/negation preservation, and applicable deterministic artifact validators. Open language remains bounded model output, not self-approved truth. These are pending mechanism improvements, not completed capabilities. Original [exit criteria](SEMANTIC-CLOSURE-EXIT.md) remain; no scaled A/B or production work is opened. Local evidence paths and exact comparison/audit digests appear in the Chinese section; ignored artifacts are not automatically backed up by Git.
