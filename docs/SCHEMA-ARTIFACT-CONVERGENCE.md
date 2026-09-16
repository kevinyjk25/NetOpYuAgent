# 结构约束与有界工件修订 / Schema-Constrained Compilation and Bounded Artifact Revision

## 中文

2026-09-16，承接[隔离编译的失败结果](ISOLATED-COMPILER.md)。本包不改原任务、数据、评分或阶段出口，不宣称任意自然语言语义均可验证。修复前两个查询虽均获取清单，完整任务仍为0/2。

### 实现范围

- 隔离编译将原author Schema同时用于Ollama `format`及独立JSON Schema准入，不再仅请求JSON对象；不删字段、填答案或放宽原编译器。原始响应、结构拒绝路径和成本全部留存。默认非隔离模式不变。
- 静态检查只支持明确子集：单管道、字面UTC时间交集（包括between加右开补充条件）、简单命名聚合后的列存在性。字符串/注释不是操作；未知函数、复杂表达式、歧义窗口保持未验证。函数自然语言说明不会被自动当作已证明的输出Schema。
- v4宿主显式设置`artifactRepair=true`后，只有明确失败且已闭合的代码块可触发**一次原生Agent修订**。Agent通过现有deliver工具提交`replacements(location,code)`，不是重生成全文。宿主保留正文、围栏、其他代码块、原任务和已冻结证据，重新检查完整结果。
- 无新的工具、观察、效果、写权、Runtime模型重跑或LLM自审。修订请求在解析前占用预算；失败、无改善、未知结果不能重试。原始交付保留，最终修订另存revision目录。显式无法修复可关闭为未完成。
- 检查通过仍是`candidate_unverified`，不是业务正确或执行许可。静态检查不能阻止所有语义偏移；例如替换代码体可能删掉检查器不认识的业务逻辑，仍须独立任务判据审阅。

结构解码依据[Ollama官方说明](https://docs.ollama.com/capabilities/structured-outputs)；局部语言规则依据[Kusto between](https://learn.microsoft.com/en-us/kusto/query/between-operator?view=microsoft-fabric)及[summarize](https://learn.microsoft.com/en-us/kusto/query/summarize-operator?view=microsoft-fabric)。这些规则不等于完整Kusto编译器或数据库验证。

### 运行前固定验证计划

1. 先通过结构、修订预算、双路由、未知结果、终态、权限及支持/不支持语言子集的定向测试和全量回归。
2. 冻结实现/Schema/Prompt/模型，然后仅跑一次真实DSH＋qwen3.5:9b批次：原`irql-draft`、`irql-missing-enricher`、`mesh-incident`。沿用原输入及c1–c3/format；2个已知Skill/仓库，3任务，0未见Skill。旧样本用于诊断，不冒充泛化。
3. 每任务420秒、编译180秒、一次编译、一次Runtime生成（若路由进入），检查失败最多一次原生代码体修订；全部调用计入总成本。相较前包增加有条件修订预算，不能声称相同调用预算或无成本收益。
4. 报告初稿与修订分别是否通过、前段编译、清单获取、独立完整任务判据、实际读取/权限、未知结果和全部模型调用/token/p50/p95。保留首次失败，不修改原报告或评分。
5. 若完整任务仍无改善，停止同例提示词/重跑，记录阻塞层；不以新增测试数或安全拒绝代替完成。只有原Gate满足，才进行冻结后新Skill迁移；本包不解锁正式规模化A/B或生产工程。

以上计划在真实模型运行前登记。唯一批次已结束；没有修改执行代码、重跑模型或改动判据。

### 实际结果：结构改善，完整任务退步

| 同样的3个任务 | 前包 | 本包 |
|---|---:|---:|
| 编译结构/前段准入 | 2/3 | 3/3 |
| 查询清单实际取得 | 2/2 | 1/2 |
| 完整任务满足冻结判据 | 1/3 | **0/3** |
| 冻结判据满足 | 9/12 | 7/12 |
| 原生代码修订 | 未开放 | 1次，无实际改动 |
| 全部模型调用 | 15 | 15 |
| 输入/输出token | 110,926 / 3,245 | 116,059 / 3,444 |
| p50/p95端到端 | 183.48 / 190.02秒 | 176.19 / 278.83秒 |

1. **irql-draft，2/4，290.23秒**：取得真实清单；初稿仍使用双闭时间及聚合后丢失的EnvTime。宿主检出时间错误，开放一次修订；9B原样提交，最终正文逐字未变，仍为`needs_revision`。未知invoke后的列血缘仍未验证；独立审阅依据实际函数说明确认该列错误。
2. **irql-missing-enricher，2/4，176.19秒**：结构通过，但Agent直接从索引进入生成，没有读索引已给出的获准清单路径，误把未读到的时间窗口说成不存在。查询因此无时间条件；没有实际窗口证据时检查器只能保持未知，不能凭评测期望补答案。c3对“明确声明未执行”的判定较保守；即便此项改判为通过，c1仍失败，完整任务计数仍为0。
3. **mesh-incident，3/4，165.48秒**：指标正确，部署原因仍按原宽松判据视为假设；但唯一后续动作编造了`/exports/mesh-stack.txt`及其内容。该资源不在真实观察/宿主清单中，c3失败。此前通过的哨兵退步，不能换成更容易的例子。

3/3正常结束、单会话保留任务、忠实宿主终态；5/5实际读取在ACL内，0观察到的越权/重放/写入。**这不等于语义成功或生产安全概率。** 总15次=DSH9＋编译3＋Runtime3；修订所在任务增加调用，而漏读任务减少调用，汇总相同掩盖了工作量变化。p50降低不算提速，p95反而增加。n=3、单次、已知样本没有证明总体因果退步或普遍准确率。

离线重放前包两份失败查询时，新检查均能明确检出时间边界错误；没有模型调用，没有重评分或覆盖旧报告。这证明局部错误检出，不证明能自主修正。当前修订策略默认关闭，不能作为已证实的质量提升对外宣传。

### 更深的缺口与停止点

- **当前author profile并非完整业务逻辑编译器。** `prefix.lower`主要生成获准read前段，再把完整任务交给一个LLM reason节点。时间、字段、动态必读义务等仍未成为可执行的业务合同。JSON结构通过不能证明这些职责已被确定性收敛。
- **编译意见与事实仍有残余耦合。** 三个提案都包含错误的missing_host解释，将“编译时没有观察”误解为“运行时不可获得”。`compiler.compile_proposal`仍把authoring_boundaries送入Runtime reason输入。它不具备授权效力，却可能影响后续文字；这不是已验证的因果归因，尤其不能据此直接解释运行Agent为什么提前结束补读。
- **自由文本修订不保证有效。** 本包提供了准确的局部诊断和受控编辑位置，9B仍作无效改动。没有同输入强模型对照，不能把根因全部归咎于9B，也不能承诺继续叠提示词即可修好。
- **静态检查有真实盲区。** 未取得的证据、任意文本断言、未知函数行为不在其证明范围。保留正文的代码修订也保留了初稿中的不当示例集群假设；不能把部分检查当作完整结果验证。

按停止规则，本批停止追加同例模型试跑，不降低[原阶段出口](SEMANTIC-CLOSURE-EXIT.md)。后续应改的是**合同覆盖与证据状态表达**：把编译意见与宿主事实分开，显式表达动态证据义务，支持的可验证工件约束进入合同及确定性lowering；未支持职责如实保留L1/追问，而非手工为每个Skill写答案。需同时度量语义合同覆盖、来源绑定、未知职责、错误接受、自治完成与成本。此设计尚未实施；不是宣告泛化通过、开放规模化Runtime测试或更换9B模型。

### 证据与工程验证

- [运行冻结](../artifacts/governed-session-20260916-schema-artifact-9b/freeze.json)、[原始观测](../artifacts/governed-session-20260916-schema-artifact-9b/summary/report.json)、[逐项审阅](../artifacts/governed-session-20260916-schema-artifact-review/judgments.json)、[摘要绑定评估](../artifacts/governed-session-20260916-schema-artifact-assessment/report.json)。原始制品被Git忽略，远端需另行备份。
- [可入Git指标及局限](benchmarks/schema-artifact-summary.json)：130份制品及82份归档/当前源码摘要一致，模型摘要前后不变。3份实际decoder schema均等于Prompt中的author schema，未注入requiredReads或人工AST。
- 148项定向、8项文档/边界检查、124个变更/新增Python文件Ruff通过；全量**3,324项＋81子测试通过，251.44秒**。模型与pytest未重叠。工程通过不抵消语义失败。
- 实验开关保持可选，默认UI不变；未提交、推送或重启。

## English

September16 successor to the [isolated-compiler failures](ISOLATED-COMPILER.md). Original tasks, observations, rubrics and stage gates remain unchanged. Both query inventories were previously acquired, but zero of two complete query tasks passed.

The isolated compiler now uses the unchanged author schema for decoder format and independent local admission; no field stripping or answer injection. Legacy nonisolated behavior remains. Raw responses, rejection paths and costs persist.

Partial inert checks recognize literal UTC interval conjunctions and a small post-aggregation column subset. Between plus a strict upper predicate is not forbidden. Unknown functions, complex expressions and ambiguous windows remain unverified; natural-language function descriptions do not become certified schemas.

Optional v4 operator policy `artifactRepair=true` offers one native Agent code-body patch only after a definite static failure. Existing deliver accepts host-addressed replacements, preserving prose, fences, other artifacts and frozen evidence. No new tool, read, effect, Runtime generation or semantic-judge call is allowed. Claim precedes decoding; rejected, unchanged or unknown attempts consume the budget. Initial and revised evidence remain separate. Passing partial checks still yields an unverified candidate; unsupported business-logic deletion can only be caught by independent task assessment, not these limited validators.

Preregistered plan: targeted and full tests, then one frozen actualDSH/9B run of unchanged irql-draft, irql-missing-enricher and mesh-incident. Two known Skills/repositories, three tasks, zero unseen Skills. Same c1–c3/format and420s/task; one compiler call, at most one Runtime draft and one diagnostic-triggered native patch. This adds a conditional model-turn budget and must be charged. Report initial/final results separately, prefix/inventory yield, complete task criteria, actual access/unknowns and total calls/tokens/p50/p95. No same-case tuning after the run, no changed historical scores, no stage/generalization/production claims. Results pending at preregistration.

### Observed outcome and stopping point

The one frozen run completed, without execution-code changes or model reruns. Compiler admission2/3→3/3, but actual query inventory acquisition2/2→1/2 and complete tasks1/3→0/3; frozen criteria9/12→7/12. Allthree processes retain one task and faithful host terminals;5/5 actual reads are authorized withzero observed effects/replay. These are not semantic successes or production probabilities.

IRQL draft still violates time bounds and projects a removed column. The host diagnoses the time error, but the sole native patch is byte-identical and ends needs_revision. Missing-enricher skips the linked inventory and falsely calls its unobserved window absent; unknown evidence cannot be supplied by the checker. Its conservative c3 judgment does not affect task failure under any more permissive interpretation because c1 fails. Mesh metrics remain correct and causal wording remains a hypothesis under the original rubric, but its sole next source is an invented file with invented contents: the previously passing sentinel regresses.

Total15 calls=9DSH+3compiler+3Runtime,116,059/3,444 input/output tokens,zero unknown usage. p50/p95=176.19/278.83s versus183.48/190.02s. Extra correction work on one case and skipped evidence on another explain why equal total calls do not mean equal work; a lower median is not a speedup. Known n=3 and one run establish no population-level causal regression. Offline old-artifact replay detects both time defects,without model calls or regrading; it establishes detection,not correction.

The core remaining gap is contract coverage: this author profile lowers primitive reads plus one original-task reason node,not all business constraints. Unverified missing_host annotations still enter Runtime reason context; their causal role has not been isolated,and they do not by themselves explain the execution Agent's skipped read. A correct diagnostic did not induce an effective9B patch. No stronger-model control proves model size is the sole root cause. Static checks cannot establish missing evidence, unknown function behavior or arbitrary prose truth; code-only editing preserves unsupported prose too.

Stop same-case model reruns. Keep schema/check improvements,but do not market optional artifactRepair as a proven quality gain. Next design work must represent dynamic evidence obligations and supported verifiable constraints, separate compiler opinions from host facts,and measure semantic contract coverage/unknown duties/false acceptance/autonomous completion together. Do not replace this with per-Skill handwritten answers or silently lower the original gate. This next design is not yet implemented; no model change, scaledAB or production qualification is unlocked.

Evidence links above and the [portable summary](benchmarks/schema-artifact-summary.json) bind130 artifacts and82 archived/current sources. Model identity is unchanged.148 targeted checks,8 documentation/authority checks,Ruff on124 changed/new Python files and full3,324 tests plus81 subtests pass (251.44s),without model/pytest overlap. Raw artifacts are Git-ignored. No commit,push,defaultUI change or restart.
