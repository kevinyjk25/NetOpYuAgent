# 任务见证与证据作用域诊断 / Task Witness and Evidence Scope Diagnostic

## 中文

2026-09-14。**本轮有两项工程修复，但语义闭环阶段仍未通过。** 四个已知失败样例的新 9B 实验没有证明审阅整体更可靠，因此没有接入自动编辑、自动完成判断或默认 DSH 路由。不扩大新样本或正式 A/B。

### 实验改变了什么

`candidate-blind-semantic-witness/v1` 分两次独立调用：

1. **先不看答案**：只给原任务、完整 Skill 来源、实际观察和未背书 caller；整理带原任务引用的交付义务，并逐个记录观察包含什么、不包含什么。明确排除旧答案、旧 notes 和候选派生的待办。模型总结不是事实、Gold 或完整义务证明。
2. **再对照实际答案**：仍提供全部原始来源，而不是用第一步总结替换它们。逐项核对任务与交付片段，逐个观察检查冲突，单独记录其他问题。肯定完成必须定位实际答复；来源冲突需要精确的两侧原文。宿主只能验证位置、范围、摘要和绑定，不能证明蕴含关系。

入口是显式诊断工具 [semantic_witness_probe.py](../evaluation/semantic_witness_probe.py)，默认零调用预检；传入已有回执绑定的请求清单和新输出目录，才可授权有限调用。它不是 Runtime 默认的新必经步骤。实验每例增加一个来源分析调用，最多两个 4,096 输出 token 调用，不宣称同预算比较或端到端加速。

### 四例实际结果

同一批原始失败稿、任务与观察保持不变；全部使用本地 `qwen3.5:9b`、think=false。运行前记录关注缺陷，没有向模型提供判据或期望答案。以下是开发者 AI 对原文/实际回复的检查，不是独立 Gold。

| 样例 | 有效观察 | 仍存在的问题 | 是否允许据此自动修订 |
|---|---|---|---|
| IRQL 查询草稿 | 来源分析保留函数清单和半开时间窗；对照指出多余启动流程、章节过多、未证实 schema 假设 | 仍只凭正文时间窗认可查询；把“索引无清单”错误扩大成已读清单不可用 | 否 |
| Mesh 指标分析 | 对照明确指出已读指标与“无数据”冲突，并算出 0.5%→5% | 同一回复又否认读取/资源权限，部分意见自相矛盾；一条有效冲突因省略引用被暂缓 | 否 |
| CAPA 状态 | 来源分析保留负责人、时间和未批准关闭 | 仍漏掉“未批准→未审阅或批准”；反而错称已明示的修复不存在；空问题引用使接口拒绝 | 否 |
| Phoenix trace | 原始对照意见正确指出缺少两项 instrumentation 建议 | 空问题引用使接口拒绝；多个肯定引用含原文没有的省略号，复合任务仍用局部内容作见证 | 否 |

来源分析 **4/4 绑定有效**，对照 **2 份绑定有效、2 份无效**。这些不是语义准确率。Mesh 的 `other_problems` 甚至包含“该范围描述准确”的正面评价，说明**问题数组长度不是错误数量**，也不能直接触发编辑。新增分析并没有消除后续模型的错误归因。

### 根因比“9B 不够大”更具体

- **证据范围混淆**：`index contains no samples` 只描述该索引，不否认另一份已完成读取中的指标。第一步已经提取正确事实，第二步仍能丢掉作用域，因此本轮不能归因于“原文没有送进模型”。
- **交付、执行与约束混成一种任务**：审阅会要求答案重抄“不得执行”“仅用只读工具”等评测包装文字，甚至凭这些声明判断操作已遵守。执行事实应由 Runtime 回执/轨迹检查；正文应核对业务交付，两者不能互相替代。
- **关系比较不是主题匹配**：原因与修复、未批准与未审阅、正文时间窗与查询实现，分别是不同关系/工件。给出真实引用也可能支持错了对象。
- **修订地址不完整**：当前 `grounded_patch` 只拥有正文编辑单元，旧 notes 属于另一个字段。不能让正文所有者“顺便”修 notes，更不能把旧 notes 的存档保留等同于正确交付。

因此，暂不把失败审阅串联更多重试。后续机制工作顺序为：分开业务交付义务与宿主执行约束；把读取资源/回执/数据位置完整传递到审阅，限制跨资源的否定扩大；对代码、数值与边界条件使用适用的静态/确定性检查；为 notes 建立独立拥有者及有界修订。开放语义保留有界 LLM 和未知状态，不改为强行确定性编码。以上尚未全部实现，不降低[阶段出口](SEMANTIC-CLOSURE-EXIT.md)。

### 本轮实际接入的两项修复

- `bounded-draft-review/v9` / `host_keyed_check_cells/v2`：原审阅主路径的 `task_checks` 新增 `draft_span_id` 和 `artifact_quote`。肯定判断只有来源引用、没有真实答复原文时，单项降为未证实，保留原意见和其他负面发现。准确引用某个标题仍不证明任务完成；没有新增权限或自动通过条件。
- `located_negative_findings/v3`：宿主依据 claim 的原始 `/candidate/notes/N` 路径，禁止备注问题被模型回显的正文位置或“唯一整稿所有者”重定向。它保留在 `unlocatedFindings`，明确尚缺 note editor，不删除问题、不改正文、不记修复成功。**这修复了错误分派，不代表备注内容已修好。**

这两项在模型实验结束后实施，不追溯修改四例输入、意见或分数，也没有把实验的额外调用接入默认审阅路径。

验证通过：154 项定向、2,934 项全量＋81 子测试（222.66 秒）、变更 Python Ruff、文档链接及 `git diff --check`。定向首轮发现一处旧接口版本断言，更新到 v2 后保持重复 JSON 键必须拒绝的原校验。四例新接口请求仅作零调用预算检查，输入字节代理分别为 37,279／28,478／16,801／16,920，均小于不变的 40,960 上限；这不是 tokenizer 证明或新的模型结果。

### 成本、失败与可复查性

合计 **8 次真实调用，34,213 输入 / 9,851 输出 token**，本批没有未知用量。调用 p50/p95 **62.58 / 114.59 秒**，是混合来源分析/对照请求时延，不是任务时延、生产 SLO 或因果性能改善。

驱动初次在 IRQL 分析完成后出现目录重复创建错误，尚未发起对照请求。保留完整回执，修复保存路径和恢复检查，显式重用该一次成功分析；恢复过程只新发起 7 次调用。没有覆盖失败目录、重复分析或重跑历史对照。

本地 `artifacts/translator-v2/` 下：

- `semantic-witness-20260914-inputs/`：四例清单、运行前判据和离线审计脚本。
- `semantic-witness-20260914-run/`：第一次运行的冻结与唯一成功分析回执。
- `semantic-witness-20260914-recovery/`：独立冻结、明确继承回执、7 次新调用及全部原始意见。
- `semantic-witness-20260914-content-review-v2/report.json`：当前有效逐例检查及合计，摘要 `8a0e64a662022e3eb96af564eabe8248057fca45d2df8c45008071c365335282`。

开发者 AI 的首次结果文字也有两处误判：误称 IRQL 实际没有四个内容章节/没有 schema 假设。直接检查原稿后已在 v2 更正为有效审阅发现，初版报告保留并由摘要关联取代。它们不改变查询/证据范围仍错误的结论，但进一步说明 AI 审阅本身不能冒充 Gold。上述制品被 Git 忽略；文档不表示已远端备份。没有提交或推送。

## English

September 14, 2026: two implementation gaps are fixed, but **semantic closure is not complete**. The opt-in `candidate-blind-semantic-witness/v1` experiment does not justify automatic editing, acceptance or default routing changes.

The first independent 9B call sees the original task, complete sources and observations, but no candidate, notes or candidate-derived duties. It proposes task-quoted requirements and per-observation interpretations. The second sees the actual artifact AND all original sources; the plan never replaces evidence. The host checks exact quotation, ownership, source scope and lineage, not entailment or requirement completeness. Each case costs up to two 4,096-output calls, rather than claiming equal-budget gains.

Four known failed drafts were retained. All four source plans bind; two comparisons bind and two fail empty problem quotes. IRQL still approves query behavior from prose and denies an observed inventory using an empty index. It does correctly flag out-of-scope launch guidance, too many content sections and an unsupported schema assumption. Mesh identifies stale absence claims and computes 0.5%→5%, but also emits conflicting permission/resource and scope judgments. CAPA still misses not-reviewed expansion and incorrectly denies an explicitly observed fix. Phoenix detects missing two instrumentation follow-ups in its raw response, but binding fails and many positive quotations contain invented ellipses. Findings counts and binding rates are not semantic accuracy.

The evidence localizes failures beyond model size: observation-specific absence is promoted to global absence; execution constraints become demands for compliance slogans; related subjects are mistaken for entailment; and notes lack their own editing owner. Further work must separate task deliverables from execution/constraint oracles, retain read-resource/receipt context, apply suitable deterministic artifact checks, and provide bounded note editing. These remain pending, with the original stage exit unchanged. Open semantics still use bounded LLMs, not forced deterministic encoding.

After all calls ended, the existing review path gained exact task-delivery witnesses (`bounded-draft-review/v9`, keyed cells v2). A positive task grade without an exact artifact quote is withheld while raw opinions and other findings remain. A valid quote still proves no semantic fulfillment. Repair routing v3 separately prevents note findings from targeting body cells, even with an echoed body ID or a sole whole-draft owner; the unresolved note stays open. This is a dispatch fix, not repaired note content. The experimental extra calls are not wired into the default pipeline, and no old inputs or scores were rewritten.

QA passes 154 targeted tests, 2,934 full tests plus 81 subtests (222.66 seconds), changed-Python lint, documentation links and diff checks. An initial stale interface-version assertion was updated to v2 without weakening duplicate-JSON-key rejection. Zero-call new-interface input byte proxies for the four cases are 37,279/28,478/16,801/16,920, all under the unchanged 40,960 limit; these are not tokenizer attestations or new model outcomes.

All eight actual calls total 34,213 input and 9,851 output tokens; usage is known. Request p50/p95 is 62.58/114.59 seconds, not task SLO or causal performance evidence. A directory-creation error occurred after the first successful plan and before comparison HTTP. Its receipt remains; an explicit new frozen recovery reuses exactly that plan and makes seven new calls, with no duplicate analysis or hidden retries.

The canonical local content report is `semantic-witness-20260914-content-review-v2/report.json`, digest `8a0e64a662022e3eb96af564eabe8248057fca45d2df8c45008071c365335282`. It corrects two developer-AI descriptions in the retained first report: IRQL really has four content sections and an explicit schema assumption, so those model findings are valid. The critical query/evidence-scope failures remain. This correction reinforces that developer-AI inspection is not independent Gold. Artifacts are Git-ignored and not claimed as remotely backed up. No business tools, source scripts, effects, draft edits, default-model switches, commits or pushes occurred in this experiment.
