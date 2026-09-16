# 草稿双向审查与有界修订 / Bounded Draft Review and Revision

## 中文

本子阶段在原双驱动 Runtime 上补充“严格读取 → 审查旧草稿 → 一次修订 → 终审”。它针对阶段 2 已知的交接遗漏、README 无依据命令；不是新 Skill 集、独立外部评测、完整自动转译或默认 DSH 路由切换。结果合同仍见[职责与观察边界](SEMANTIC-RESULT-CONTRACT.md)。

### 为什么不能只加一个 AI 裁判

首轮真实 `qwen3.5:9b` 审查把交接稿的 18 个单元全部判为 supported，但修订后仍漏掉事件责任人。另一个 README 审查虽然指出无依据命令，却引用了不存在的来源编号，系统阻断，没有继续修订。两个失败都保留；“审查通过率”不等于结果准确率。

根因线索是审查对象和检查方向：复制在 `values` 中的原文容易被混同为已经写入交付稿；只检查草稿已有段落，也容易看不到未写出来的职责。另有来源编号生成错误。这里是根据开发样例作出的诊断，不是已验证的统计因果结论。

### 实现与边界

1. **交付稿与观察分开。** 审查只看到候选 `draft` 和 `notes`；完整候选仍有摘要绑定。复制用的 `values` 不计作交付内容，继续由原 `ResultContract` 独立核验。修订者另外收到旧 values，但这些仍非独立事实。
2. **双向审查。** 从草稿反查来源，识别无据新增；从严格读取的观察窗口反查草稿，识别任务相关的遗漏、责任、条件和限制。窗口保留原位置及完整父上下文；只是词法分段，不宣称提取了全部原子语义，也不要求无关元数据全部写进稿件。
3. **可定位而非盲信。** 复用已有 `translation_source_alignment` 审查器。每个声明单元必须恰好出现一次，引用必须属于本次来源目录，非支持项需要解释和修订建议。传输 Schema 额外限定单元编号、来源编号和输入摘要；本地仍再次检查，不能仅信模型的结构化输出。
4. **有界闭环。** 每例一个原始严格读取、最多三次新模型调用、最多一次修订。终审未解决的问题直接留下，不自动反复跑到通过。超时、超预算、缺项、非法引用均失败关闭；不截断原文，也不执行下载 Skill 的脚本。
5. **严格控制不交给裁判。** 审阅输出仍是 `model_candidate`；没有工具、动态扩图或写权限。AI 的 supported/contradicted/insufficient_evidence 是意见，不是事实证明。`OpenDuty` 不会因审阅得分自动清除；下游工具参数仍需原独立 `CandidateAdmission` 和操作权限/证据检查。

第三轮交接例还发现“知道漏了，但重写仍原样返回”。因此修订传输改为**原文锚定的局部修改提案**：模型输出原稿摘要、唯一 `expected_text`、替换文字、来源编号及解释；宿主检查摘要、唯一锚点、不重叠、不接受伪装成修复的相同文字替换，然后应用到原稿。每轮最多 8 个修改、一次修订，未改动的字节保留。空修改明确标为 `no_material_draft_change`，不包装为修复成功。

`revision-patch.json` 是模型原始提案；`patch-assessment.json` 是确定性的锚点/应用记录；`candidate.json` 是经上述投影得到的**仍未验证的候选**。宿主保留旧 values 并非模型重新提取正确，不能把这部分算成模型能力提升。锚点和引用编号合法也不证明新文字与原文语义一致。

第二轮暴露两个工程缺口，均单列而非计作语义失败或成功：Ollama 对自由草稿 Schema 返回 grammar 初始化 HTTP 400；README 反馈与重复清单叠加超过原输入预算。修正后，审阅者使用受限编号的结构化解码，修订者仍走原 JSON 模式并接受同一份本地 Schema 校验；修订上下文仅把重复的清单文字替换为可追溯位置，完整 Task、Skill、观察、草稿和反馈都保留一次。不增加 token/调用预算。新的超限预检会留下零调用诊断。

### 怎样查看、复现

实现：`evaluation/hybrid_draft_review.py` 构造双向审查输入；`evaluation/hybrid_draft_loop.py` 用宿主安装的固定图接入原 `run_hybrid`。这是开发用宿主构图适配器，目前只接一个原读取片段和一个推理节点；**不是转译器已经会从任意 Skill 自动生成完整修订图。**

从之前冻结的候选开始，保留原 Task、Skill、读取图和合成夹具。以下使用同一 9B 的独立上下文调用，不是独立模型、独立人员或外部 Gold：

```bash
.venv/bin/python -m evaluation.hybrid_draft_loop \
  artifacts/translator-v2/semantic-result-20260910-live-v4 \
  NEW_OUTPUT_DIR --max-model-calls 6
```

使用 `--max-model-calls 0` 只校验旧输入并冻结清单和源码；不调用模型、不生成完成报告。输出目录必须不存在。真实运行的每例查看：

- `freeze/inputs.json`：原候选、原报告摘要、固定新图与结果合同。
- `model/review-before/review-input.json`：草稿/观察单元、原文位置、完整来源目录。
- `model/*/candidate.json`、`review-assessment.json`：逐项意见及待修订位置；无效响应留在 `invalid-candidate.json`。
- `model/revise-draft/revision-patch.json`、`patch-assessment.json`：模型编辑提案与宿主精确应用记录。
- `model/*/request.json`、`response.json`、`receipt.json`：原始模型调用及文件摘要。
- `summary/report.json`：图状态、结果职责、实际工具/模型调用与成本；未验证草稿不等于完成任务。
- `model-preflight/*/diagnostic.json`：若输入预算不通过，记录零模型调用的预算与请求摘要。

运行根目录另存冻结实现清单与源码归档；历史轮次不覆盖。完整 `artifacts/` 在本机且被 Git 忽略，提交代码不代表已备份原始模型证据。

### 真实结果：有一处实质修复，语义闭环仍未完成

2026-09-10，4 个实现版本均保留。机器记录、原稿/修订稿、意见数量、成本及报告摘要见[完整记录](benchmarks/bounded-draft-summary.json)。本轮只重用 **2 个已知开发 Skill**，没有新增未见样本，没有原生 DSH 对照。固定图和结果映射由宿主构造，不计自动转译准确率。

| 版本 | 观察到的结果 | 判定 |
|---|---|---|
| v1，整稿审查/重写 | 交接 18 项全部 supported 却漏责任人；README 伪造引用编号 | 未解决语义问题；非法引用阻断 |
| v2，双向审查/限定编号 | 找出 Chen 遗漏；分别触发 grammar HTTP 400、修订输入超预算 | 两例均 blocked，不计修复 |
| v3，传输及重复文本修复 | 两例图完成；交接仅删除末尾换行，README 原样返回 | 无语义改善，均 partial |
| v4，锚定局部修改 | 交接应用 1 处修改；README 提出 10 处修改超过 8 处上限 | 交接局部改善；README blocked |

**交接的实质变化**：`Packet loss on uplink B` 改为 `Packet loss on uplink B (owner Chen)`，其他段落保留。但 INC-42 的责任人 Alice、开始时间 08:10 UTC 仍未进入稿件。9B 终审竟给出 **27/27 supported**，不能当作正确率。宿主结果职责保持 **2/3、partial**：Runtime 没有自动识别这两处遗漏，它只是没有让模型自报通过来清除开放职责。开发者 AI 的补充审阅也不是独立 Gold 或全部语义错误清单。

**README 没有修复成功**：10 个编辑提案在 `/edits` 的数量校验处拒绝，没有应用、没有终审。拒绝的提案还存在未闭合 Python 代码围栏，不能把它当成可用草稿。只观察目录项也不可能凭空知道正确安装命令/API；完整 README 仍需要受控补读配置与源码，不能只用更多审阅调用代替。

审查条目数量不是独立事实数量；README 的 17 条意见包含重复或过度质疑占位标题等问题，不是 17 项独立 Oracle。合法出处编号与精确应用都不证明蕴含；本轮 `semanticDraftAccuracy`、自动转译准确率均保持 null，不输出生产成功概率，也不扩大 Runtime/DSH A/B。

### 成本、验收与下一步

共 **18 次模型 Chat HTTP 尝试**，17 个响应有 token 用量，另 1 次是 HTTP 400、用量未知。已知提供方报告用量为 **153,495 输入 / 32,073 输出 token**，不含此前生成原稿的调用。v4 交接 3 次约 **364.99 秒**；README 2 次后阻断，约 **336.14 秒**。审阅增加了模型成本，不能宣称性能提升。部分与 pytest 并发，只是本机开发观测，不是隔离性能基准、SLO 或因果加速证据。

工程验收：**136 项定向、2699 项全量＋81 子测试通过**；14 个变更 Python 文件 Ruff、文档链接、diff 检查通过。完整性核对包括 4 个源码归档/316 个源文件条目、40 份报告摘要、18 份调用回执/103 个绑定文件、11 份审查报告精确离线重算、1 份已应用补丁精确重算。测试与制品完整性不等于语义正确率。

新改动未提交或推送，默认 DSH 路由、A/B 固定基线和源脚本执行策略不变。本地四轮制品和两个零调用预检清单保留。

下一步按根因收敛，而非继续增加互审轮次：

1. 改善原文职责到交付稿的定位与漏项识别，要求可核对的对应位置；优先解决角色/事件/条件的对应关系，不能只判断主题相关。
2. 改善有界编辑提案的生成可靠性与格式可用性；保留固定编辑预算、原稿绑定和首次失败，不加上限迎合样例。
3. 必要观察缺失时，补“推理提案 → 独立准入 → 受控补读/追问”，而非生成更多未知占位内容。
4. 再冻结并开展新 cohort；关键职责自动映射和同类型参数错绑仍在后续范围。不能用生产工程代替语义准确性证据。

## English

This development adapter adds one bounded read/review/revise/review graph to the existing hybrid scheduler. It reuses the known handoff and README failures, original tasks/Skills and synthetic reads. It is not unseen evaluation, independent Gold, automatic whole-Skill compilation or a default DSH route change.

The initial real 9B reviewer marked all 18 handoff units supported while the revised draft still omitted incident owners. The README review found unsupported content but invented citation IDs and was blocked before revision. These failures remain preserved, not relabelled as passes.

The revised review contract separates the delivered draft/notes from copied values, binds the complete candidate digest, and checks both draft-to-evidence additions and observation-to-draft omissions. Located lexical windows retain full parent context; they do not prove atomic semantic coverage or require irrelevant metadata. The existing source-assessment validator enforces exhaustive declared units, known citations and proposed fixes. The transport schema constrains identifiers, not semantic verdicts, and local validation remains mandatory.

Each fixed graph permits one original read, up to three actual new model calls and exactly one possible revision. Invalid review, excessive input or timeout stops the graph. There is no retry-until-passing, script execution, dynamic graph or Effect authority. Same-model separated conversations are not independent external review. AI verdicts remain candidate opinions, never clear host open duties and never substitute for downstream admission, permission or evidence checks.

Use the command and per-case paths above for reproducibility; zero calls only freezes preparation. Outputs must be new. Source archives, raw replies, failures and receipts remain separate. The host-built adapter currently requires one original strict region plus one reasoning node; arbitrary Skill-to-loop authoring is still open. Full local artifacts are ignored by Git.

The second iteration exposed a local Ollama grammar HTTP 400 and a separate over-budget revision input. Review identifiers remain constrained, while revision uses established JSON mode with unchanged local validation. Redundant checklist text is replaced with locations; all original task, Skill, observations, draft and feedback remain. Neither output/call budgets nor acceptance rules are relaxed. Subsequent budget refusals retain zero-call diagnostics.

The third handoff attempt still copied the draft despite a detected omission. Revision therefore uses anchored text patches: the model proposes an exact unique old span, replacement, source IDs and rationale, bound to the original draft digest. The host rejects stale, missing, ambiguous, overlapping and no-op edits, then materializes at most eight edits in one revision. Empty edits explicitly mean no material change. `revision-patch.json` retains raw proposals, `patch-assessment.json` records deterministic application, and `candidate.json` remains unverified. Host-preserved values are not a new model extraction success; valid anchors/citations do not establish semantic entailment.

### Actual results and remaining gaps

Four retained implementation attempts reuse only two known development Skills. See the [digest-bound record](benchmarks/bounded-draft-summary.json). v1 missed owners and invented citations; v2 encountered a grammar error and separate input-budget refusal; v3 completed both graphs but made no semantic improvement (handoff only lost its trailing newline). v4 applied one anchored edit adding Chen as uplink B's owner. Alice's incident ownership and the 08:10 start time remain omitted despite a 27/27 supported final AI review. The host retains the open duty, with 2/3 declared duties and a partial result; it did not automatically detect these omissions.

The v4 README proposal contained ten edits, exceeding eight, and was rejected without application or final review. The rejected proposal also contains an unclosed Python fence. It is not a usable revised draft. Correct project documentation additionally requires controlled configuration/source reads, not further review of directory entries alone. Reviewer item counts are not independent factual Oracles; some findings duplicate or over-reject placeholders. Developer AI inspection is not independent or exhaustive Gold.

There were 18 Chat HTTP attempts: 17 responses with provider-reported usage (153,495 input / 32,073 output tokens) and one HTTP 400 with unknown usage. Original draft generation is outside these totals. v4 handoff took three calls and approximately 364.99 seconds; documentation stopped after two calls and 336.14 seconds. Runs partly overlapped pytest; these are development observations, not an isolated speed comparison or SLO. No accuracy or generalization pass is claimed, and large DSH/Runtime A/B remains closed.

Regression passes 136 targeted tests and 2699 full tests plus 81 subtests; Ruff passes on fourteen changed Python files, with documentation and diff checks. Integrity checks cover four archives/316 source-file entries, forty report digests, eighteen receipts/103 bound files, eleven exact review re-derivations and one exact patch re-derivation. New changes are uncommitted/unpushed; defaults, the A/B baseline and inert-script policy are unchanged.

Next: better source-duty-to-draft role/entity/condition alignment; reliable bounded patch generation and formatting; independently admitted missing-observation reads/questions; then frozen new cohorts. Do not add review loops or production hardening in place of semantic evidence.
