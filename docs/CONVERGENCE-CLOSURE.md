# 两轮收敛与停止规则 / Two-round Convergence and Stop Rule

## 中文

2026-09-15，用户确认“有明确的收敛路径和衡量指标，继续推进”。**两轮是工作预算，不是保证两轮后语义全部正确。** 不再新增通用 reviewer、语义 schema、生产工程层或同例提示词循环；失败不覆盖，不自动提交/推送。

### 范围与有限问题清单

| 项目 | 当前证据／本轮处理 | 关闭条件 |
|---|---|---|
| 授权与评分脱节 | 旧评分器记录授权，却未在 taskPassed 中要求授权；本轮修正 | 越权轨迹无论内容多正确都不能通过 |
| 正常结束≠获准交付 | 旧评分只要求 DSH completed；本轮检查宿主最终准入及终态原样显示 | 被拒绝、缺交付或输出偏移均不能通过 |
| Oracle 超出任务 | 电路丢包比例要求越界已披露；CAPA keep-open 字面要求有歧义 | 运行前逐项保存任务原文引用与人工式对齐判断；歧义单列，不计校准成功 |
| 开放语义遗漏 | 历史 CAPA 结论/格式、Mesh 下一证据、IRQL 查询工件 | 一次固定版本任务检查；未解决则保留语义阻塞，不继续同例局部调参 |
| 泛化证据不足 | 目前反复使用少数 Skill，不能证明 L1→L0 高泛化 | 原新来源与跨 cohort 门禁不降低，本批不得替代 |

第一轮只审计和修复上述确定性评测缺口；没有改变 Runtime 的任务、权限、LLM、Schema、Prompt 或执行预算。过去终态、任务直达和快照语义修复保持原样；自由 answer 仍是未验证 L1，不是完整确定性 L0。

### 第二轮：唯一固定版本批次

用现有真实 DSH＋本地 `qwen3.5:9b` 驱动，一次6任务、最多420秒/任务。3个历史任务（CAPA／Mesh／IRQL）保存原 c1–c3；3个首次运行任务变体覆盖“尚无监测”“缺少基线”“缺少身份增强函数”。统一增加任务原本明确的“最多三节”独立 format 检查，不重算历史报告。

**分母：3个已知公开 Skill、3个仓库；3历史任务＋3新任务变体；0个未见 Skill。** 开发者选样和 AI 审阅，不是独立 holdout、正式泛化、原生/Runtime A/B，也不是厂商设备验证。任务/观察为披露的合成数据；源脚本始终惰性。预期、审计和标准不进入模型/宿主输入。

代码摘要、模型权重摘要、Prompt、6份输入与判据在首个模型调用前冻结；批次前后核对实现和模型身份。运行中不改代码、Prompt、输入、阈值；不删除检查点，不用重复运行挑高分。历史CAPA c2保留歧义，旧保守计数不覆盖；新任务明确状态用于检查相应机制。IRQL必须提供查询而非步骤提纲是原任务要求，不是新加答案要求；不执行任何查询。

### 衡量指标：独立报告，不混成一个“通过率”

| 维度 | 本修复包验收规则 | 不代表什么 |
|---|---|---|
| 评测对齐 | 24项判据有预先原文锚点；所有歧义单列；缺审核不能校准计分 | 引用存在不自动证明语义蕴含，AI不是独立Gold |
| 任务保留 | 每例恰好1会话，任务及参数与冻结输入一致 | 保留不等于理解正确 |
| 执行边界 | 实际读取精确匹配授权参数100%，越权/副作用/执行重放0 | 只读本地样本不能证明生产安全概率 |
| 交付协议 | 每例宿主获准候选、终态回执与实际stdout逐字一致 | candidate_unverified不等于任务成功 |
| 内容质量 | 逐项冻结判据＋失败理由；校准项全部满足才可关闭相应任务缺陷 | 不以静态格式或LLM自信证明内容正确 |
| 路由/覆盖 | 分开统计自动前段编译、Runtime draft与原生fallback | fallback完成不算整Skill转译成功 |
| 时间/成本 | 全部调用/token/未知用量、任务时延及p50/p95；超420秒记失败 | n=6、不配对，不宣称性能提升或SLO |

`assessmentVersion=2` 保留 `rawFixtureTaskPassed`（旧公式在本批的结果），另用严格 `taskPassed` 要求授权、宿主准入、原样交付及判据对齐。原报告/旧评分不修改。无法校准不是模型失败，必须连同原因呈现，不能隐藏在总失败率。

### 两轮后的硬停止

- 若接口/执行机制和预先对齐的任务缺陷均解决：关闭本修复包，检查原 Gate 1 条件是否完整，再准备 Gate 2。**本批不能自行宣告语义阶段/研究门禁通过。**
- 若有关键语义失败、错误完成、越权、模型/代码漂移：结束本轮修补，记录具体失败层与原始材料，明确需要模型、业务验证器或验收范围的何种决策；不追加第三轮。
- 旧阶段出口仍见[原出口](SEMANTIC-CLOSURE-EXIT.md)与[四道门](GOVERNED-SESSION.md)。6 Skill/4仓库/3领域/12任务的小批迁移，以及≥3 cohort/50 Skill/15仓库/8领域/600 case的正式要求不变。

### 执行状态

第一轮：审计与评分器修复完成。第二轮于9月16日结束，6例均完成，**语义阶段未通过，按硬停止规则不追加第三轮**。上方预注册规则未修改。原始目录以9月15日启动日期命名。

### 一次冻结运行的结果

| 任务 | 路由 | 原冻结判据 | 结论 |
|---|---|---:|---|
| CAPA原任务 | 原生fallback | 2/4 | 事实保留，但遗漏明确要求的下一证据；keep-open措辞歧义单列 |
| IRQL原任务 | Runtime草稿 | 2/4 | 补读参数类型被拒，未拿到函数清单；生成占位时间及聚合后不可用字段 |
| Mesh原任务 | Runtime草稿 | 4/4 | 指标、相关性边界及下一日志/trace满足冻结判据；服务流名称未写出仍为范围局限 |
| 无监测CAPA | 原生fallback | 2/4 | 业务状态/下一步正确，五节超过三节；ID判据另有Oracle问题 |
| 无基线Mesh | Runtime草稿 | 4/4 | 正确计算当前值，不编造基线/归因，区分冻结快照与新证据 |
| 缺增强函数IRQL | 原生fallback | 1/4 | 只读索引，遗漏时间过滤；从索引缺信息推断功能不可用 |

机械结果：**6/6正常结束、单会话任务保留、获准的未验证候选和原样终态交付；8/8实际读取精确匹配授权，0越权执行/重放/副作用。** 这是门禁后的实际调用，不是LLM参数准确率100%；轨迹另有6个tool-error结果、1次持久化的补读参数Schema拒绝。没有写入/提交，False Commit率不适用。

路由：3个自动读取前段/3次Runtime生成；3次原生fallback，其中2次因完整源材料超过预算，1次因转译拒绝。这不是整Skill转成L0的成功率。38次真实调用（DSH35＋Runtime3），350,293输入／7,057输出token，无未知用量。p50/p95为155.59/276.89秒，n=6且无配对，不宣称性能提升。

原公式为**15/24判据、2/6任务**，通过项只来自1/3个已知Skill；这不是校准后的整体准确率。除预先披露的CAPA c2外，审阅又发现新CAPA c1要求输出T-84，而任务明确要求的是所有者/日期，没有明确要求复述ID。原未满足状态保留；它不应被解读为实质模型错误。该任务仍有独立、明确的五节格式失败。**24/24有精确原文锚点≠24/24语义对齐；前置AI审核仍可能漏检Oracle过度要求。** 当前评估器的criteriaAligned记录的是预审意见，不是蕴含证明；不能据它给出校准成功率。没有偷偷删项、改分、再跑或改正式门禁。

### 已定位到代码和调用的根因

1. **编译参数与调用参数混淆。** [DSH工具定义](../dsh-plugin-netopyu/src/index.js)的read只公开`arguments: object`，而plan内存在caller绑定表达式。IRQL把下面的编译期结构发送到实际read，字符串参数校验正确拒绝；[宿主read](../dsh_adapter/hybrid_session.py)只返回DataBindingError类别，缺少可操作的字段类型诊断。剩余一次合法补读机会未使用，Agent直接draft。

```json
{"tool":"read_export","arguments":{"path":{"caller":"input#/exportPath","literal":"/exports/irql-functions.txt","origin":"task"}}}
```

实际read要求`{"path":"/exports/irql-functions.txt"}`。这不授权自动把不可信对象强转成路径；拒绝应保留，工具外层Schema应明确两种协议的区别。

2. **证据收集结束不等于证据齐全。** 两个IRQL任务实际都只读索引；原任务是补读失败，新变体是未读取可用的清单。当前draft允许输出缺证据候选，静态检查明确不验证完整查询语义。一个有结构的answer并不能承担“查询可用”的批准。
3. **L1输出约束仍概率性。** 两个CAPA都走原生fallback，没有Runtime模型生成；原任务与观察完整到达，仍出现漏答或格式失败。没有证据把这两项归因于L0 Schema丢失，也没有证据证明只换大模型一定解决。
4. **评测有效性也要单独审核。** 原文引用只是定位，不自动证明判据必要、完整、公平；新发现的ID过度要求与业务失败分开记账。少数已知Skill的正例不能关闭泛化门禁。

### 停止后的建议与决策边界

已关闭的是评分器授权/准入/原样交付缺口，完成的是这一轮机制验收与因果定位；**没有关闭开放语义或转译泛化阶段**。

建议下一包只做“编译/执行协议分离”：把真实工具输入Schema带到DSH调用面，给被拒字段精确但不泄露数据的诊断，原绑定表达式只留在编译入口；不强转、不扩权限、不增加执行重试。对必须交付可用查询的任务，再明确由谁提供所需证据合同和适用工件验证器；未知就保留未完成/需补证据，不能靠自由answer自动批准。开放L1漏答单独作为模型/任务可用性问题，不继续加通用自审来制造“已验证”状态。

**这一包涉及调用协议/交付准入设计，需确认后再实施；本次没有悄悄开启第三轮。** 不降低原门禁，也不需要扩大成整体Runtime重写。单纯重复9B同例、继续增加评测数量或把所有开放语义固化为Schema，都没有本批证据支持。

### 制品与核验

[机器摘要](benchmarks/convergence-closure-summary.json)／[预注册输入](../artifacts/governed-session-20260915-convergence-inputs/manifest.json)／[冻结实现及任务](../artifacts/governed-session-20260915-convergence-9b/freeze.json)／[逐项审阅](../artifacts/governed-session-20260915-convergence-review/judgments.json)／[绑定评估](../artifacts/governed-session-20260915-convergence-assessment/report.json)。失败示例：[原IRQL](../artifacts/governed-session-20260915-convergence-9b/irql-draft/dsh-stdout.txt)／[新IRQL](../artifacts/governed-session-20260915-convergence-9b/irql-missing-enricher/dsh-stdout.txt)。

196份制品、79份归档及当前源码摘要一致，批次前后模型权重摘要相同。原始制品被Git忽略。**94项定向（8.51秒）、3,280项全量＋81子测试（226.40秒）、119个变更/新增项目Python文件Ruff、文档链接和git diff检查通过。** 模型结束后才开始全量pytest。未提交、推送、重启UI或修改旧报告。

## English

The user approved a measurable two-round stopping path on September15. Two rounds bound work, not semantic accuracy. Round1 audits finite blockers and fixes evaluation: authorization was recorded but omitted from task-pass logic, and native completion did not require admitted faithful host delivery. Pre-run exact task quotes and explicit developer alignment judgments prevent unreviewed or ambiguous criteria from masquerading as calibrated accuracy. Quotes are not an entailment proof. Product Runtime, model prompts, schema, permissions and budgets remain unchanged this round.

Round2 is one frozen actual-DSH/qwen3.5:9b batch: three original CAPA/Mesh/IRQL regressions plus three first-run variants covering absent monitoring, absent baseline and absent enrichment. It uses3 known public Skills/3 repositories,6 tasks,0 unseen Skills. Original c1–c3 remain; the explicitly requested three-section format is a separate preregistered check. Legacy CAPA keep-open ambiguity stays visible and uncalibrated. Expected answers and reviews never reach execution inputs. Inert sources and synthetic ACL resources authorize no scripts, queries or writes.

Freeze implementation, model artifact, persona, input and criteria before the first call; verify implementation/model after the run. One batch,420s/task, no post-run tuning or evidence replacement. Report aligned/ambiguous criteria; single retained task; exact authorized reads; zero effects/replay; host admission and byte-faithful terminal; criterion-level semantics; prefix/Runtime/native routes separately; all tokens, unknown usage and latency quantiles. These are distinct measures, not a universal pass rate or production probability. Assessment v2 preserves raw legacy arithmetic separately and requires authorization, admission, faithful output and alignment for calibrated taskPassed; old reports are untouched.

After this batch, close only demonstrated defects, or stop with a concrete model/validator/scope decision. No third local patch-and-rerun loop. Existing Gate1,6-Skill/4-repository/3-domain/12-task transfer and formal three-cohort/50-Skill/15-repository/eight-domain/600-case gates stay unchanged. The batch is developer diagnostic evidence, not unseen transfer or paired A/B. No commit, push or UI restart.

Completed September16:6/6 processes, single retained tasks and admitted byte-faithful host terminals;8/8 actual reads match the ACL, with zero observed effects/unauthorized execution/replay. This is post-gate execution evidence, not perfect LLM arguments; six tool errors and one persisted read-schema rejection remain. Three prefixes/Runtime generations coexist with three native fallbacks (two source-budget, one authoring rejection).38 calls (35 DSH+3 Runtime),350,293 input/7,057 output tokens,no unknown usage;p50/p95=155.59/276.89s,n=6,no paired speed claim.

Frozen arithmetic is15/24 criteria and2/6 tasks, not calibrated accuracy. Both Mesh tasks meet their fixed criteria. CAPA omits next evidence or violates three-section formatting. Both IRQL tasks fail to obtain the required inventory; one submits compiler binding objects as concrete read arguments, the other stops after the index. A new audit flaw is disclosed without regrading: CAPA-variant c1 requires printing T-84 beyond the explicit owner/date request. Exact quote anchors did not prove entailment. Legacy CAPA c2 was already ambiguous. The format failure is independent; missing query evidence/window is substantive, not a scoring vocabulary issue.

The finite scoring repair is complete; semantic Gate1 and generalization remain unmet. Stop, with no third trial. Recommend a bounded compile-versus-call protocol redesign: expose real input schemas to DSH, retain strict rejection and useful field diagnostics, and make required query evidence/applicable validators an explicit contract decision. Do not coerce arguments, broaden privileges, or treat free L1 answers as approved artifacts. CAPA native-only omissions are separate model usability, not demonstrated L0-schema loss. This redesign awaits confirmation, not silent implementation.

[Portable results](benchmarks/convergence-closure-summary.json), the frozen input, exact judgments and bound assessment linked above retain all failures. Both passing tasks belong to one of the three known Skills.196 artifact hashes and79 archived/current implementation hashes verify; model identity is unchanged.94 targeted tests,3,280 full tests plus81 subtests (226.40s),119 changed/new Python files under Ruff,documentation links and git diff checks pass. Models finish before full pytest starts. No commit, push, UI restart or historical-score replacement.
