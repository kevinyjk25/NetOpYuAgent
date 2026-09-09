# 语义迁移小批与源审查 / Semantic Transfer Probe and Source Audit

## 中文

2026-09-08。用户授权在合入前暂停之后恢复推进，直到需要大批量验证。本阶段同时关心**语义保真与有效通过率**；正确停止、首次完成、后处理完成和完整 Skill 分开，不承诺开放世界的 100%。

### 先审查旧 33/33，而不是直接认定泛化通过

审查对象是 `behavior-contract-necessity-6-20260908` 的六份原始候选，逐份阅读完整 L1、实际声明的宿主合同、候选节点与引用、有限 Oracle。审查人仍为同一开发助手；这是开发审计，**不是独立真人 Gold，也不替代已有的摘要绑定完整源审查/激活门禁**。旧候选与旧结果不修改。

| 案例 | 对原文的核对 | 发现 / 后续动作 |
|---|---|---|
| direct-read | s0002–3 要求以输入 device_id 单次读取并完成；s0005–6 是数据与权限边界 | 读取和参数正确；完成节点却引用 s0006。权限限制不能证明业务完成，引用应来自实际读取/完成指令 |
| inverted-branch | s0002–3 指定 campus→needs_l1，非 campus→使用返回 device_id 再读；边界要求无效数据阻断 | 有限分支、数据依赖和停止方向可支持；未把 planned inventory 当实时健康。真实宿主行为和开放世界解释仍不由该夹具证明 |
| missing-approval-write | s0002–3 要求先读、后审批与 VLAN 变更；缺合同处停止 | 候选保留先读及 unsupported，不伪造批准；缺能力说明重复两次，是冗余而非两项独立覆盖成绩 |
| unavailable-script-prerequisite | s0002–3 要求脚本先于读取，脚本及宿主合同缺失；源文本不能执行 | 停止在读取之前，与原文一致；重复 issue 不增加证明强度。没有执行脚本，也不把停止算业务完成 |
| backup-prerequisite | s0002–3 要求参考中的前置及可用 runner；s0004 禁止从文件名推断验证；s0005–7 为惰性代码 | 未读取、未执行代码、未虚报验证；issue 描述 runner 缺口，但未完整列出参考缺口。解释性需逐项保留，不能从一个停止结果推断整个包已解释完整 |
| access-guide | s0002 是废弃历史；s0003–4 要求存在、当前且授予读取的决定；s0005 禁止据此修改 owner | Q6 及三项条件的有限行为正确；granted 的禁止判断引用 s0003，直接依据应包含 s0004；正向分支和完成引用 s0005 也不够。原始三个正向 unknown 继续保留，不补写成 possible |

结论：**行为相同不代表引用正确，更不代表完整自然语言保真。** 源/宿主整体支持某个条件，也不能自动证明所选的单条引文支持该条件。上述问题不能靠词面正则自动“修正答案”，应进入新协议的首次生成及源审查。

### 设计调整

1. 为 `require_all` 增加对称的 `require_any`，只在所有候选条件都不满足时停止。它编译成原有 `if_equal`，共享后续读取，不复制操作或增加执行器；否定仍通过与 false 比较表达。AND/OR 混合保留现有嵌套分支。
2. 明确操作、分支和完成的引文需要支持各自语义。数据/权限免责声明不是完成证据。该提示降低歧义，但不构成自动语义验证器。
3. 必要条件推导仍不是充分性证明。单变量反事实可以给出必要条件，但无法从每个变量“都不单独必要”恢复整个 OR。联合关系必须由完整候选表示并验证，不能删除后改称通过。
4. 加入源文专用 `flow_contract_authoring author`，复用公共一次调用/检查点逻辑，明确隔离 Oracle。新 `flow_semantic_probe` 只允许 1–12 个显式案例，冻结后不得改源、协议或答案；大批运行需要另行决策。

### 本轮小批边界

六份新建、可直接阅读的 Skill 包见 [examples/semantic-transfer](../examples/semantic-transfer/README.md)：发票 OR、发布负向条件、调度 AND＋OR、联系人参考规则、资产返回值引用、归档脚本缺 runner。共 **42 个有限场景**；其中 5 个可执行片段案例、1 个缺能力停止案例。带引用和脚本的文件完整留存，脚本不执行。

它们由开发助手构造，宿主为明确标记的惰性测试合同；是新的 9B 首次输入，但**不是未见公开 Skill/独立评测**。一次首次构造，存在布尔依赖时再做一次源条件判断；不按 Oracle 自动重试、改图或换模型。首次与条件推导后的结果分列，未决条件、失败和所有成本均保留。

冻结清单：`artifacts/translator-v2/semantic-transfer-6-20260908-v2/manifest.json`，摘要 `sha256:af4d3db66ec284586b53c67986316f9108fa792792942bb7e31b315f33f27a00`。模型固定 `qwen3.5:9b`，无思考输出模式。以下是已完成的结果，旧回答未重写。

### 真实结果与设计纠偏

| 实验 | 结果 | 结论 / 是否采纳 |
|---|---|---|
| 首批六例构造 | 4/6 案例，38/42 场景 | OR 漏项、复合条件丢失仍存在 |
| 同批追加单变量必要条件 | 1/6 完整匹配；已执行 20/24 场景，另 18 场景无候选 | 会反转否定、误判 OR 并阻断原本正确的路径；退出推荐流程 |
| 同源五例直接填联合真值表 | 22/26 组合正确；混合 AND/OR 出现 4 个错误放行判断 | 仅作非修改型诊断，不能让它自动覆盖原候选 |
| 递归 JSON 条件 v1 | 0/5 指定条件，5 份报未解决 | 模型可读请求未附输出语法；失败保留 |
| 附语法后的 v2 | 3/5 指定条件，2 份报未解决 | 9B 仍误解复杂表达式 Schema；不称适用性不足 |
| 可读表达式 v3 | 5/5 条件可用；连同原停止例，经显式修订匹配 6/6、42/42 | 模型提取逻辑，代码计算组合；这是已知案例修订，不是新一轮泛化 |
| 第二批新输入六例 | 构造 2/6、30/46；条件阶段后 5/6、35/46 | 一份结构失败导致 11 场景未运行；不是 46/46，也不是整 Skill 接纳 |
| 引文约束修复后的工单 canary | 仍结构失败：分支重复别名、终止后的重复操作 | 旧失败不回填；不删除步骤来凑通过 |
| 同工单一次编译错误反馈 | 仍生成相同结构错误 | 显式 opt-in 修订能力保留，但不是默认重试或已证实的质量提升 |
| 紧凑两阶段入口 v1，同源五例 | 0/5；一个命名空间不匹配，四个未解决 | 模型误把“编译器负责的条件执行”当成底层工具缺失能力 |
| 补充编译能力上下文与 facts 字段命名空间，v2 | 4/5，34/45 场景；另 11 场景无候选 | 修好了工单，副本例却错误报不支持；只能作为范围受限的可选入口，不能按 Oracle 拼出全通过 |

所有这些数字都基于同一开发助手的有限 Oracle。第二批的“新”仅指此前未交给 9B 的输入；共 **12 个独特 Skill 包、0 个公开未见 Skill**，不是十个独立 cohort。四/五个可用区域不等于四/五个已完成完整源审查的 Skill。

### 当前如何收敛 L1→L0.5→L0

1. **语义提取层**：L1 原文 + 实际宿主合同 + 明确的编译器能力 → 工具/参数/引文和可读条件，如 `confirmed and (not sensitive or delegated)`。
2. **结构与逻辑层**：白名单解析为 `all/any/not/match`，最多 4 个布尔事实、16 个组合；确定性生成原有 FlowTree，不执行 Python、源脚本或外部工具。`facts.field` 只指一个声明的返回字段，不进行对象属性访问。
3. **诊断层**：原提案、表达式、联合赋值、Tree/L0 位置和引文分别保留。未知、冲突、全停止判断、范围外和参数/引文未审事项不能变成执行授权。
4. **编译结构由代码负责**：紧凑入口固定结果名与控制流接线，模型不必输出边/别名/结束节点。它只适用于真实符合两阶段读取的区域；更复杂 Skill 不能强行压缩进模板。
5. **完整源审查与激活仍独立**：本轮 `wholeSkillTranslations=0`、`semanticAccuracy=null`、`runtimeAuthorityGranted=false`。没有把这些候选装入生产 L0 或默认 DSH。

例子的原文/格式位于 [Skill 索引](../examples/semantic-transfer/README.md)，操作命令见[代码导航](../evaluation/README.md)。

### 未解决的问题必须看得见

- 第二批 dataset 的条件正确，但条件阶段把读取引文选为 s0006（事实读取），真正目标读取在 s0008；停止引文选到 s0009（错误处理边界），业务停止条件在 s0007。紧凑入口修正了目标读取引文，但仍保留停止引文错位。
- storage 的表达式能在原编译器上运行并匹配有限场景；紧凑入口却报“无法表达条件控制流”。这是**模型对已提供能力的误判**，不是 Runtime 真不支持这个表达式。
- 引文结构合法、逻辑真值正确和执行行为匹配，是三个不同判据；不能互相替代。完整源/参数/解释审查仍 required_not_run。
- 当前没有预先冻结的自动路由规则来在两种构造入口间选择，也没有校准过的语义概率。不得根据本轮 Oracle 选取各路径的最好结果，宣称总体 100%。

### 成本、证据与复验

[机器可读摘要](benchmarks/semantic-transfer-summary.json)包含各阶段原始文件/检查点摘要、全部失败和调用成本。**本轮总计 53 次 9B 请求，输入 141,027 token，输出 14,995 token；POST 总时长约 18.93 分钟**。这是请求时间合计，不是任务墙钟时长或 Runtime 性能。

第二批新输入的 10 次请求 p50/p95 为 **20.91 / 35.82 秒**；紧凑 v2 的 5 次请求为 **14.90 / 17.02 秒**。它们的协议/工作量不同，不能据此宣称受控时延提升。模型固定摘要为 `sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。

主要原始报告（本地 artifacts 不提交为源码）：

- [首批构造/旧 Guard](../artifacts/translator-v2/semantic-transfer-6-20260908-v2/report.json)
- [第二批首次结果](../artifacts/translator-v2/semantic-followup-6-20260908-v3/report.json)
- [工单结构 canary](../artifacts/translator-v2/semantic-helpdesk-repair-20260908/report.json)
- [紧凑 v2](../artifacts/translator-v2/compact-read-region-development-5-20260908-v2/report.json)

旧代码回放使用 Git 基线 `14baa0a269fdffdac8443cb41ec7f22fbd2d44ad` **加上**该轮保存的 `source-snapshot.tar.gz` 覆盖层，而不是修改指纹。快照不单独包含全部未修改依赖；基线提供 effect_runtime 等依赖。`condition-expression-development-5-20260908-v1` 快照包含其 69 个指纹文件；之后的 v2、semantic-followup、semantic-helpdesk-repair、compact-v1 也保存了变更前源代码。必须在隔离临时目录还原，使用原环境依赖、传入**绝对的原始输出目录**零调用回放，不覆盖工作树。第二批 report 已如此重建并用 cmp 验证逐字节一致。当前源码拒绝旧 fingerprint 是预期行为。53 个调用检查点的 receipt 均逐一核对。

定向回归 **117 passed**；全量 **1792 passed + 81 subtests passed**（154.14 秒）。修改文件 Ruff 和 git diff --check 通过；全仓仍有原先的 224 项 lint 问题。测试数量是代码回归，不是新增 Skill 证据。当前进入**批量验证决策点**：冻结协议后独立抽样公开 Skill，验证范围识别、源引用与语义精度/召回。还未通过泛化门禁，不解锁大规模 Runtime/DSH 对照。

## English

The user authorized resuming after consolidation, up to the bulk-validation decision point. This stage targets both semantic fidelity and useful acceptance, without promising open-world 100% accuracy. First attempts, assisted derivations, safe stops and complete Skills remain distinct.

The development assistant audited all six original L1 texts, declared host contracts, candidates/citations and finite oracles behind the old 33/33. Direct-read completion cites a restriction rather than completion; access-guide cites inadequate evidence for granted/positive completion; missing-capability issues contain redundancy and incomplete explanation. The inverted branch and guarded stopping behaviors remain source-supported within the fixture scope. Original outputs and positive unknowns are unchanged. This audit is not independent human Gold or a substitute for existing digest-bound source review/activation.

The v2 constructor adds symmetric `require_any` lowering to existing branches with a shared continuation, retains `require_all` and negative comparisons, and clarifies operation-specific citations. Unary necessity cannot reconstruct a joint OR or prove sufficiency. No new executor or authority is introduced. Source-only `author` reuses one-attempt checkpoints; the small probe caps explicit cases at twelve and excludes oracles from generation APIs.

Two batches contain twelve [assistant-authored packages](../examples/semantic-transfer/README.md), not public holdouts or independent Gold. First-batch construction matches 4/6 cases and 38/42 scenarios; unary guard addition regresses to one fully matching case, with 20/24 evaluated scenarios and 18 unrun. Direct model truth tables misclassify four mixed-logic assignments. Recursive expression v1 yields five unresolved outputs; providing its schema improves to three specified outputs. Readable expressions yield five specified conditions, with explicit known-case revisions matching the six original finite fixtures (42/42), not fresh independent evidence.

The new six-input / 46-scenario batch improves from **2/6, 30/46** after construction to **5/6, 35/46** after conditions. Eleven scenarios remain unrun after a structural failure. Quote-constraint correction and one explicit compiler-feedback attempt do not fix the subsequent alias/unreachable-operation failure. A compact two-read front end gives code ownership of aliases/edges/terminals. Its v1 fails all five known-source inputs; binding compiler capabilities and a safe facts.field namespace yields **4/5, 34/45**, with eleven unrun. It fixes helpdesk but falsely rejects storage. These paths must not be cherry-picked with the oracle to claim overall perfection.

The model extracts symbolic logic, while an allowlisted parser and existing FlowTree compiler perform bounded computation. No eval/exec, source scripts, provider operations, activation or new executor are introduced. The helper covers a two-stage read region with at most four Boolean fields, not arbitrary workflows. Unknowns and source disagreements remain explicit. Dataset read/stop citation errors and model capability misconceptions remain open; finite behavior cannot establish citation entailment or complete source semantics. Whole-Skill translations remain zero, semantic accuracy unestablished and Runtime authority false.

The [digest-bound summary](benchmarks/semantic-transfer-summary.json) records all **53 requests**, **141,027 input / 14,995 output tokens**, and **18.93 minutes total POST time**, including failed development variants. New-input request p50/p95 is **20.91/35.82 seconds**; compact v2 is **14.90/17.02 seconds**. Different workloads/protocols make this descriptive cost data, not a controlled latency A/B or Runtime performance result. All 53 receipts were verified. Replay uses base Git commit 14baa0a plus the per-run source overlay in an isolated directory, not the overlay alone or edited hashes. The second-batch report was replayed without model calls and compared byte-for-byte successfully.

Targeted regression: **117 passed**; full suite: **1792 passed + 81 subtests**, 154.14 seconds. These are code regressions, not more independent Skills. Next freeze a protocol and validate separate public Skill batches, including applicability, citations and semantic precision/recall. The formal generalization gate and large Runtime/DSH evaluation remain locked.
