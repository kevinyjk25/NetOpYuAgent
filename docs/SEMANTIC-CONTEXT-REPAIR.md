# 第一阶段：任务、证据与编辑边界修复 / Stage-one Context Repair

## 中文

历史轮次说明：本报告保存上一轮四例及当时的方案停止点。用户随后已确认并实施新方案，最新状态见[职责合同与工件检查](SEMANTIC-DUTY-CONTRACT.md)；下文“尚待确认/未实施”仅指本轮封存时点，不是当前阻塞。

2026-09-14。按“完成第一阶段，直到停止或需要调整方案”的授权推进。本报告对应当前剩余阶段中的**语义闭环修复**，不是推翻此前已提交的阶段 1/2 历史记录。

**结论：工程边界已补齐一轮，第一阶段仍未通过；触发机制复盘停止条件，不启动新来源验收或扩大 A/B。** 不能用通过接口、增加测试数量或模型自评证明语义准确。

### 已实现与交互流程

1. 调用者可声明 `taskScope`：`business_request`、`execution_constraint`、`delivery_constraint`。宿主要求这些片段按顺序逐字重建原任务，原始 `task` 不变；不得删除限制、注入答案或把分类当权限。该字段进入自动构图、后续推理、补读和审阅。
2. 审阅保留每个实际读取的工具名、参数、结果位置、载荷摘要与回执摘要。`/observations/...` 是审计目录位置，**不是工具资源路径**。索引与另一个文件的观察分别绑定；历史回执不续期，也不证明其内容为真。
3. 正文和 notes 有不同所有者，但**共用一次 pass、最多 8 个编辑单元/改动范围**。notes 可保留、替换或删除；修改后的 notes 进入终审，原文始终留在审计中。超过预算的项显式延期，不能绕过宿主职责、改写观察或获得操作权限。
4. 冻结真实验证结束后修复两项工程问题：notes 的可编辑目标移至最后一条消息，并限制新备注为单段；真正拥有整个草稿的编辑器可以增加一个缺失标题，但不得删除、重排或改名旧标题。局部编辑器不获得该权限。原始引用、范围、围栏、工具权限与调用上限仍受保护。

代码入口：[任务作用域](../evaluation/hybrid_task_context.py)、[审阅来源绑定](../evaluation/hybrid_review_context.py)、[notes 单元](../evaluation/hybrid_note_cells.py)、[有界应用](../evaluation/hybrid_repair_cells.py)。这些是研究原型的可选链路，不改变默认 DSH 路由。

### 真实 9B 结果：4 个已知失败任务

冻结同一模型、代码和输入；复用旧候选及相同来源/任务/观察，**不复用旧审阅意见、不改旧分数**。本轮没有新的 Skill，没有厂商设备或生产环境实验。

| 用例 | 实际结果 | 未解决问题 |
|---|---|---|
| IRQL 查询 | 初审有效；notes 编辑被拒绝；没有新交付稿 | 错误时间表达式仍漏判；编辑器试图将整个正文放进备注，超过 2,000 字符限制 |
| Mesh 指标分析 | 修订/终审图完成；局部内容改善 | 补入错误数、p99、部署时间，但未算出 0.5%→5%，缺南区完整对照；仍要求重读已有结果，4 条陈旧 notes 保留 |
| CAPA 状态 | 修订/终审图完成；正文无实质改动 | 来源仅“未批准”，notes 的“未审阅或批准”仍被误支持；notes 编辑未触发 |
| Phoenix 追踪 | 漏项检出并生成补充建议；应用被拒绝 | 冻结版本禁止增加标题；原候选仍缺两项建议；新提案还出现未建立的“生产合规标准”措辞 |

**四个指定关键缺陷完全修复数为 0/4；这是已知失败回归，不是整个项目准确率。** Mesh 有局部事实恢复；Phoenix 有定位/提案进展，但它们不能抵消查询与否定关系的持续错误。

IRQL 判据不是模型自评：KQL 的 `between` 使用包含两端的区间，并有规定的范围语法；原候选既不符合该语法，也不满足任务的右端开放窗口。[Microsoft 官方说明](https://learn.microsoft.com/en-us/kusto/query/between-operator?view=microsoft-fabric)。本轮未执行该查询或源 Skill 的脚本。

10 次真实 `qwen3.5:9b` 调用，61,350 输入 / 13,553 输出 token，用量未知 0 次；请求时延 p50/p95 **67.00 / 149.72 秒**。这是本地模型调用统计，不是 Runtime SLO、任务时延或因果 A/B 加速。业务补读、Effect 和源脚本执行均为 0。模型摘要：`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。

### 可核查证据与工程验证

- 真实运行：`artifacts/translator-v2/semantic-context-20260914-run/`；摘要 `ab07100b74f50e3e52bd2a7eae529f1ef4a7395189c4c9b37e99f61af6cdfd65`，冻结 `ed2a4e13ca9ad242192e9c97de6ffc84e6bbd0dd5e4dd95e83d14fa0fd5b89e7`。
- 内容审阅：`artifacts/translator-v2/semantic-context-20260914-content-review/report.json`；摘要 `829a13a4c3ce2a6a78825d0d667c7bb1ef39ed5e6d0fae5752e646e936e14950`。这是披露的开发者 AI 审阅，不是独立人工 Gold。
- 原始回执、载荷、归档、失败、成本均保存。初次零调用预检的归档清单错误、离线审计将新增汇总报告误认为摘要漂移的问题均已定位；没有重复模型调用或修改旧制品。
- 后置工程修改**没有再跑 9B**。Phoenix 的原提案在新标题策略下仅通过了零调用结构应用检查；旧拒绝仍是旧拒绝，不计语义通过。notes 单段/目标分离同样只获工程验证。
- 定向 **162 项**、全量 **2,951 项＋81 子测试**通过（223.89 秒）；变更/新增项目 Python 的 Ruff、diff 及文档链接/中英顺序 **3 项**检查通过。QA 在真实模型运行结束后进行，不与时延测量并行。首次定向 3 个失败仅为错误信息的旧匹配词不兼容，修复后结果另存，未覆盖失败报告。

制品目录被 Git 忽略；本轮不提交或推送，不更改 A/B 基线，不清除历史失败。

### 为什么停止，以及需要确认的调整

已有证据排除了“没读到数据”和“Schema 没有字符串槽”作为本轮主要根因。真正的问题有三层：

- **审阅误判影响调度**：对“未审阅”等错误前提给出肯定意见，会令对应编辑器不被触发；肯定意见不是事实，但目前仍影响修订覆盖。
- **精确定位不等于语义蕴含**：引用标题或邻近真句不能证明查询正确、计算完成或限定词保留。一个调用同时检查数十个单元仍会漏项和混淆资源。
- **工件缺少独立检查器**：查询、计算、时间区间和自由文本被交给同一种文本审阅；展示保护也曾误伤合法补充。

遵守[阶段停止规则](SEMANTIC-CLOSURE-EXIT.md)：相同关键缺陷经过机制轮次仍不改善，不再继续同例提示词重跑。建议下一方案保持 9B 与“严格 L0＋有界 LLM”双驱动，**改为职责合同＋适用的本地工件验证器＋聚焦审阅**：

1. 在看候选前，从完整任务/来源生成带原文锚点的职责候选；它不是标准答案或权限证明。逐项标明可机械检查与开放语义，不给出无法证明的通过结论。
2. 对适用工件接入白名单本地检查器：结构、计算、时间边界、语言解析等；源脚本仍不执行。不支持的语言/语义显式 `unverified`，不能用字符串匹配冒充完整验证。
3. 开放语义按职责聚焦核查，避免一次输出几十个判分格；保留不确定性、原始问题和失败。LLM 负责提出候选和解释，不能单靠自己打分决定正确性或写权限。
4. **需要确认的预算变化**：先只做 2 个已知失败用例的一次机制验证；每任务最多 1 次职责提取＋7 次聚焦核查＋8 个共享编辑单元＋1 次终审，合计最多 17 次，而不是当前 10 次。未覆盖职责继续开放，不降低既定出口；不自动扩样或改用付费模型。

以上新方案尚未实施或授权开跑。本轮结束位置是**方案复盘停止点，不是第一阶段完成点**。小批新来源 6 Skill/12 任务、正式跨 cohort 泛化与 DSH A/B 均未解锁。

## English

Historical-round notice: this preserves the earlier four-case run and its design stop. The user subsequently approved the redesign; see the [new duty-contract/local-check report](SEMANTIC-DUTY-CONTRACT.md). Approval-pending/not-implemented statements below refer to this archived round, not the current blocker.

September 14, 2026. This is the current semantic-closure stage, not a reclassification of earlier committed stage-one/two work. **The stage remains incomplete and reaches a design-review stop.** Interface validity and test counts are not semantic success.

Implemented: lossless caller-declared task roles; receipt/payload-bound tool and resource context; separately owned body and note edits sharing one pass and eight cells/ranges. Original notes remain audited, revised notes reach final review, and excess cells are explicitly deferred. No classification, citation or edit grants authority. The optional research path does not change default DSH routing.

Four known historical failures were reviewed using the same drafts, sources, tasks, observations and local 9B model. No old opinions or grades were imported. IRQL still misses the incorrect time predicate and its note editor attempts to copy the entire answer; Mesh restores some metrics but omits requested calculations and retains stale notes/read requests; CAPA preserves the unsupported expansion from “not approved” to “not reviewed or approved”; Phoenix detects missing follow-ups but the frozen heading guard rejects the proposal. **None of the four designated critical defects is fully repaired.** This is a known-failure regression, not overall project accuracy.

Ten actual calls consumed 61,350 input and 13,553 output tokens with no unknown-usage calls. Request p50/p95 were 67.00/149.72 seconds, not task latency, Runtime SLO or causal A/B performance. No fresh business reads, Effects or source scripts ran. The run summary/freeze and disclosed developer-AI content-review digests are listed above; artifacts are local and Git-ignored. No commit, push, baseline change or deletion occurred.

After the frozen run, note targets were moved to the final message with single-paragraph replacements. An actual whole-draft owner may add one missing heading while preserving all existing headings; partial owners cannot. These changes have **zero additional model calls**, not demonstrated semantic gains. The retained Phoenix proposal now passes only a structural application diagnostic; its old rejection and uncertain new prose remain recorded. QA passes 162 targeted tests and 2,951 full tests plus 81 subtests (223.89 seconds), changed/new project-Python Ruff, diff and three bilingual/link checks. Three initial targeted failures were stale error-message matches and are retained in a separate report. A zero-call archive-membership mistake and the later audit's self-reference accounting mistake are preserved without model retries.

The remaining failure is not missing observations or an unrepresentable string. Fallible positive reviews suppress needed edits; exact quotations do not prove entailment; technical artifacts lack independent validators. Under the existing two-round stopping rule, more same-case prompting is not justified.

Proposed—not implemented or started—next mechanism: candidate-blind, source-anchored duty candidates; applicable allowlisted local structure/calculation/interval/language validators; and focused open-semantic review. Unsupported semantics remain unverified and never authorize writes. Keep 9B and the strict-L0/bounded-LLM split. Approval is requested for a bounded two-known-case experiment and an explicit maximum increase from 10 to 17 calls per task: one duty extraction, seven focused checks, eight shared editing cells and one final review. Do not lower the exit criteria, expand cohorts, execute source scripts or switch to paid models. New-source transfer, formal generalization and DSH A/B remain closed.
