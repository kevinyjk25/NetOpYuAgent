# 原文义务与宿主绑定分层 / Source Duties and Host Binding

## 中文

### 为什么调整

[上一轮 9B 诊断](FLOW-9B-COMMON-JSON.md)发现：模型同时面对原文、流程、宿主规则和目的标签时，会补造检查义务、丢失目的，或把“候选是否忠实”误判为“操作是否已经执行”。本轮仍使用 `qwen3.5:9b`，拆分任务，不扩权限、不另造执行器，也不改写冻结失败。

### 已实现的接口

| 阶段 | 输入与输出 | 确定性检查 | 不能据此保证的事情 |
|---|---|---|---|
| 原文义务候选 | 显式提供的文档 → 逐行多个 requirement/context/unknown 声明 | 原文/偏移保留，每个非空正文行必须交代，来源 ID 有效，禁止额外字段；代码不解释 | 语义完整、归类正确、否定/条件忠实 |
| 原文审查 | 原文 + 候选 → 逐声明和逐原文行的审查 | 重新生成清单、摘要绑定、必需引用、缺行/旧审查拒绝 | AI 审查不是独立 Gold 或准确率 |
| 业务骨架 | 审查支持的候选 + 完整原文 + 实际宿主合同 → 现有 FlowTree 请求 | 复用原有宿主约束 Schema 与图编译器，不替换原文 | Schema 合格不等于完整转译 |
| 固定图绑定 | 不可变义务 + 实际节点/宿主规则 → 仅选择绑定目标 | 禁止增义务/改图；每节点需义务；context/unknown 不得支持操作 | 名称相似、目标有效不证明保障真正落实 |
| 整体审查 | 全文、义务、绑定、完整图、宿主上下文 → 完整审查结果 | 原文覆盖、节点语义、参数/顺序/极性/前置、审查摘要重新绑定 | 暂不接激活入口，不授予 Runtime 权限 |

主要入口：[原文提取](../evaluation/flow_source_duties.py)、[业务骨架/绑定/完整审查](../evaluation/flow_duty_binding.py)、[冻结 9B 探针](../evaluation/flow_source_duty_pilot.py)。它们是原型评测接口，并非已替换 DSH 的默认转译链。

### 与旧协议的实质区别

第一步没有工具目录、宿主规则、执行图或 `objective` 标记。它只从提供的文档提出义务。例如“读取一次；出错就停止”，不能因为宿主具有权限检查，额外声称原文要求了权限检查。

每项候选包含三项：

```json
{
  "kind": "requirement",
  "statement": "游标缺失时停止并追问，不能解释为没有更多对象。",
  "evidence_ids": ["d001-l0003"]
}
```

以上是格式示例，不是模型测试答案。`d001-l0003` 由代码按文件/行位置生成，后续义务 ID 也由代码生成。每行允许多个义务以表达复合语义；附加引用可跨行、跨显式提供的文档。引用只是完整行证据，不声称得到语义原子级精确跨度；“引用正确但删掉否定”必须由语义审查发现。

`context` 仅用于真实描述性信息，不能把禁止、限制或前置条件降级以规避绑定。`unknown` 保留不能解析的含义。完整正文始终留在后续输入中；缺失引用不自动下载，指定文档路径仅作证据标签，绝不据它读取磁盘。标题可能有规范性要求，不能一律删除。

代码块和显式 opaque 文件由代码保留为未解释材料，模型不能将其行为当作已验证事实。本原型只处理显式提供的文本；没有实现完整 Anthropic Skill 资源发现/动态脚本理解。行数、声明数或审查引用数超限就报错，不能截断换通过率。

### 绑定与执行边界

宿主本来具有的检查放在独立 `hostBaseline`，它不增加原文义务。绑定步骤只能选择实际节点、实际规则、`retained` 或 `unresolved`，不能改业务操作、参数、图边、源候选或父问题。旧候选的来源别名不作为第二步目标语义，但原始树仍保留在编译制品和完整审查中。

`retained` 只表示文字保留，`unresolved` 表示映射缺口，两者不能与执行目标混填，借部分落实冒充全部落实。缺少宿主能力的正确停止，可以语义上忠实，却仍不可完成业务。审查是否支持与 `admissionBlockers` 分开；所有结果都为 `runtimeReady=false`、`runtimeAuthorityGranted=false`。下一步接入现有激活/Runtime 前仍要通过原门禁，不新增快捷授权路径。

### 本地探针与证据口径

[四个开发输入](../examples/source-duty-development/cases.json)使用存储分页、工单审批、备份脚本前置、传感器分支/引用，覆盖中英文本、禁止式标题、循环需求、反向条件、外部审批、缺失引用和惰性代码。它们由同一开发助手构造并已可见，不是市场 Skill、未知集或独立 Gold。

```bash
.venv/bin/python -m evaluation.flow_source_duty_pilot freeze artifacts/translator-v2/source-duty-4-20260908 --inputs examples/source-duty-development/cases.json
.venv/bin/python -m evaluation.flow_source_duty_pilot run artifacts/translator-v2/source-duty-4-20260908 --max-new-calls 4
.venv/bin/python -m evaluation.flow_source_duty_pilot report artifacts/translator-v2/source-duty-4-20260908 --output /tmp/source-duty-report.json
```

冻结目录只能新建。重入已完成目录只重放，不重发；不完整收据则停止，不猜测请求是否执行。原始失败、模型制品、完整请求、Schema、实现指纹和成本全部留存。已有目录不可用第一条命令覆盖；新实验须另选目录并显式授权预算。

本轮真实调用范围仅第一步原文提取。绑定和整链审查有离线反例验证，但尚不是新协议的真实 9B 完整双阶段成绩。模型结构合格后才审查完整清单，不能从结构比例计算语义准确率，也不与旧四例端到端结果混合分母。最新实测结果与回归见 [PROJECT-STATUS](PROJECT-STATUS.md)。

### 2026-09-08 首轮实测：原文阶段可定位，但质量仍不完整

四次真实 9B 首次调用，无修复重试：**4/4 结构合格，4/4 完成源文清单审查，1/4 在源候选范围内获支持**。共 50 个相互重叠的声明，44 supported、4 contradicted、2 insufficient；不可把 44/50 当语义准确率。审阅者是同一开发助手，不是独立人工或模型盲评。

| 开发输入 | 源候选审查 | 具体发现 |
|---|---|---|
| 存储分页 | blocked | 标题明确“只读/不得删除”，陈述文字保留了限制，但标为 context；分类错误会使后续不能把它作为义务绑定 |
| 工单审批 | blocked | 动作、审批条件、验证失败分支均保留，但“读取 T-42 后再请求该工单审批”的依赖/作用域未明确进入审批声明；独立义务数组没有声明时间控制语义 |
| 备份前置 | 源候选获支持 | 保留前置引用、缺依赖时先停再解释、脚本名不证明验证；代码仍 opaque，引用内容/宿主能力并未验证，不能执行 |
| 传感器与引用 | blocked | 反向分支、单次读取及引用限制保留，但 Observe only 标题被缩成描述性 context，缺少规范性边界 |

`context` 判错与原文删除是不同问题；原文仍完整传递。工单判断为“候选未充分交代依赖”，不是断言已经生成了错误执行图。本次尚未对这些新源运行宿主图生成/绑定，不能报告端到端改善。

输入/输出 **4,587 / 1,992 token**，总 POST **127.89 秒**，单请求 p50/p95 **31.19 / 36.34 秒**。这是四个开发输入的第一步成本，不与旧两步协议的时延作因果比较。全部正常 stop，无超时/截断/业务执行。

制品：`artifacts/translator-v2/source-duty-4-20260908`；审查后报告摘要 `sha256:462c38bf81513a08419659ded92d7d5403211db9bbe196194316cfbce5b66685`。Git 中保存[原文、原始候选、显式审阅决策及成本摘要](benchmarks/flow-source-duty-4-summary.json)，原始 HTTP/检查点保持本地。

新增 **49 项回归**；模型完成后全量 **1592 tests + 81 subtests 通过（203.78 秒）**。测试覆盖多文档/标题/代码惰性、原文与审查漂移、伪造目标/来源、缺义务、图不变、context 不得绑定操作、拒绝重试和原始失败保留。不是模型准确率或 Runtime 性能。

新批原始/审查后报告及零调用重入均重放一致；追溯摘要通过校验，前三份历史报告保持不变。定向 Ruff 与 `git diff --check` 通过；不声称全仓既有 Ruff 告警已清理。

### 首轮之后的计划（历史，后续关系字段探针见下）

继续在第一阶段验证**规范性范围与显式依赖**，而不是继续增加宿主规则菜单。首先用异质措辞反例说明每项声明应保留什么：禁止式标题、例子与事实、先后/条件/作用域、引用缺口。规范性分类和依赖都必须经来源审查；不能检测到标题就强制 context→requirement，也不能把数组位置自动当控制流。

再另冻新版做首次输出验证，不修当前四份答案。只有源候选质量更稳定后，才把实际已支持源与真实宿主合同接入已经实现的骨架/绑定接口，运行完整 9B 链；缺循环、脚本、审批等能力仍明确保留为缺口，不将安全停止当业务完成。本轮没有修改默认 DSH 路由、解锁 C4–C6 或提交/推送 Git。

### 显式作用域、条件与先后关系（新增原型接口）

`source-duty-guarded-meaning/v1` 在旧源候选旁增加独立版本：[实现](../evaluation/flow_source_guards.py)、[冻结批入口](../evaluation/flow_source_guard_pilot.py)。旧协议/制品不变，不通过重新解释旧答案更新成绩。

每项候选按 `statement → scope → when → after → kind → evidence_ids` 输出。先表达含义和来源关系，再判断规范性；Markdown 标题只是格式，不自动决定 kind。`scope` 指源文作用对象/范围，`when` 指适用条件，`after` 指必须先发生的事项。三者都是待审查自然语言，**还不是可执行谓词、图边或权限**；空值仅表示没有明确表达，不能解释成没有前置条件或无条件执行。

完整来源审查同时检查这几项是否遗漏、伪造、指代不清或极性反转，也检查 context 是否掩盖强制限制。引用仍只是完整行证据，不证明其含义正确。描述性/历史/引用示例不能变成实际请求；同一行分拆出的义务仍需保留共享条件和作用域。

对旧绑定接口使用无损文字投影：把每个非空关系字段带标签保留在候选陈述中，不静默丢字段，也不借此自动生成控制流。`binding_input()` 仅对新协议审查支持的候选导出 sidecar；投影及后续完整绑定仍需新的摘要绑定审查，不能拿新协议审查直接授权旧接口。默认 DSH 转译及 Runtime 未切换。

六例探针包含原样四例回归与[两个新增开发反例](../examples/source-duty-development/guard-extra-cases.json)：发布预览中的只读边界/被拒绝指令，以及访问指南中的描述性标题/历史规则/所有者决定前置。来源内容未改写，回归组、新开发组分开统计。两组都由同一助手构造并可见，不是未知集或市场 Skill。原始运行目录 `artifacts/translator-v2/source-guard-6-20260908`，只执行源提取阶段，不执行源脚本或业务工具。

```bash
.venv/bin/python -m evaluation.flow_source_guard_pilot report artifacts/translator-v2/source-guard-6-20260908 --output /tmp/source-guard-report.json
```

未运行的请求显示 not_run；原始结构通过不隐含源审查。既有完成检查点重入只重放；若请求不确定或模型身份/传输异常则停止，不重试换答案。

### 六例实际结果与方案取舍：不采用自由文本关系字段作为默认升级

2026-09-08 已完成六次真实 9B 首次调用，并对所有结构合格候选完成同一开发助手的源审查。旧代码、提示词、四例来源及旧审查结果未改写。

| 分组 | 结构合格 | 完整源审查 | 源候选获支持 | 输入/输出 token | POST 总计 | 请求 p50/p95 |
|---|---|---|---|---|---|---|
| 原样四例回归 | 4/4 | 4/4 | 0/4 | 5,103 / 3,265 | 236.89 秒 | 54.70 / 78.15 秒 |
| 两例新增开发 | 2/2 | 2/2 | 0/2 | 2,298 / 1,257 | 89.88 秒 | 40.34 / 49.53 秒 |

合计 11,923 token、326.76 秒 POST。全部正常 stop，没有重试、截断、超时、脚本或业务执行。72 个重叠声明中，45 supported、19 contradicted、8 insufficient；既不是独立 Gold，也不是准确率分母。相比上轮同四个输入，token 观察增加 27.2%、POST 增加 85.2%；运行负载、协议与审查字段不同，不作单因素速度或模型能力推断。原先 1/4 获支持、现在 0/4，也不能直接当作泛化准确率差值。

实质发现：

- **局部改善但未闭合**：工单 `when` 明确了读取 T-42 后再申请审批；`after` 却仍用包含多个动作的来源行 ID，无法唯一指明前置动作。
- **真实新错误**：备份缺 runner 时的停止被加上“先尝试运行前置”的要求；工单验证被写成可跟在审批拒绝路径后；传感器禁止重复读取被推迟到请求人工解释之后。这不是单纯字段格式问题。
- **范围漂移**：存储缺游标的停止/解释限制被加上 truncated=true 条件；预览的发布禁止范围也未完整解决。
- **规范性仍错**：三个实际有限制意义的标题仍标成 context；被拒绝/历史引用在新增样本中没有被转成执行命令。不要因此把所有标题统一升级为 requirement。
- **表达歧义**：None/unknown/行 ID 被写进本应表达关系的自由文本。审查没有一概把这些字符串算作整例失败，而是逐项定位具体范围/前置错误；它们也不能被下游自动解释为无条件、无依赖或图边。

因此**不将本版自由文本关系字段设为默认转译方案**。保留代码和失败制品用于诊断，原文/宿主分层原则不变。下一步应把关系改为**可定位的原文证据候选**，将“是否存在关系/具体源片段”和“该片段表示什么条件、对象或前置”分开审查；不继续堆叠任意描述字段。规范性范围也仍需单独判断，精确引用不自动证明语义。此纠偏尚未实现/获得新模型成绩，默认 DSH 与 Runtime 没有切换。

本轮新增 **36 项回归**；模型完成后全量 **1628 tests + 81 subtests 通过（204.34 秒）**。新原始/审查后报告、零调用重入、追溯摘要与四份历史报告重放校验通过；定向 Ruff/diff 通过。完整审查后报告摘要 `sha256:9801c65c26e747c0dac1f77cedb0e95b734528edf0a669b192242e6f924693d4`；[Git 内证据摘要](benchmarks/flow-source-guard-6-summary.json)保留原文、原始候选和 72 项显式判断。未提交/推送，不解锁 C4–C6/规模化 Runtime。

## English

### Purpose and implemented scope

The [previous 9B diagnostic](FLOW-9B-COMMON-JSON.md) exposed source/host confusion, lost objectives and candidate/execution confusion. Continue with the same local 9B model but separate source-only semantic candidates, source review, business skeleton generation, fixed-graph binding and full-chain review. This is an experimental interface, not yet the default DSH translation route.

The extractor receives only explicitly supplied inert documents. It has no host rules, tools, graph or objective flags. Each nonblank prose line must have requirement/context/unknown candidates with compiler-bound line IDs; composite duties may have separate statements and cross-line/document citations. Exact full-line citations are not atomic semantic spans or proof of entailment. Headings remain visible; restrictions cannot be downgraded to context. Fenced code and explicitly opaque files are archived without interpretation. Missing references are not fetched, and paths are evidence labels rather than filesystem access requests. Bounded inputs exceeding limits fail without truncation.

Source review regenerates every candidate claim and every source-line coverage claim, binding judgments to original content and requiring exact evidence. Even all-supported AI judgments are not independent Gold. The business-skeleton request preserves original text and actual host contracts with reviewed duty candidates; it uses the existing FlowTree schema/compiler, not a new executor.

Binding targets come only from the actual graph and host rules. Compiler-owned duties cannot be expanded and graph nodes cannot be edited. Every node needs source-duty evidence; context/unknown cannot justify operations. HostBaseline is separate system context, not invented user requirements. Retained/unresolved are not combined with implementation targets. Full review receives original documents, candidates, bindings, actual graph and host context; it checks omissions, parameters, order, polarity, prerequisites and outcomes. Representation support and admission blockers are distinct. No activation is wired: runtimeReady and runtimeAuthorityGranted remain false, even for a supported representation.

### Usage and evidence boundaries

See the Python interfaces and shell commands above. The [four known-development inputs](../examples/source-duty-development/cases.json) cover storage pagination, ticket approval, backup prerequisites/inert scripts and sensor branches/supplied references. They are same-developer examples, not public Skills or hidden Gold. Only source extraction is called on real 9B in this batch; binding/full-review behavior is exercised offline. Do not pool these first-stage results with historical whole-chain quality denominators.

Freeze into a new directory; completed reentry replays, uncertain checkpoints stop without retry. Original requests/responses, model digest, schema, implementation and failures are preserved. Document/claim/citation limits reject rather than truncate. Full-source review is required beyond structural qualification, and no result grants Runtime authority or establishes calibrated accuracy. Latest measurements and regression counts appear in [Project Status](PROJECT-STATUS.md).

### First measured source-only batch, 2026-09-08

Four real first attempts: **4/4 structural qualification, 4/4 complete source-only checklist review, 1/4 source candidates review-supported**. Fifty overlapping claims contain 44 supported, four contradicted and two insufficient judgments; these are not an accuracy denominator. The same development assistant authored and reviewed these visible inputs, not independent Gold.

Storage and sensor cases incorrectly classify normative headings as context. Storage still preserves the restriction in its statement, but the binding role is wrong; sensor reduces Observe only to a title. Ticket actions and approval/verification conditions remain, but the explicit read-then-approval dependency and ticket scope are insufficiently stated in the approval duty. Array order has no declared temporal semantics for independent duties. The backup candidate is supported only as source representation: absent reference contents and opaque script behavior are not known or executable. Source text was not deleted. No fresh host skeleton/binding generation has yet run for these four cases; no whole-chain quality gain is claimed.

Input/output **4,587/1,992 tokens**, total POST **127.89 s**, per-request p50/p95 **31.19/36.34 s**. All ended normally, no retry, truncation, timeout or business execution. First-stage costs cannot be causally compared with old two-stage probes. The [tracked summary](benchmarks/flow-source-duty-4-summary.json) contains sources, raw candidates, explicit decisions and costs; raw HTTP checkpoints stay local. Reviewed-report digest: `sha256:462c38bf81513a08419659ded92d7d5403211db9bbe196194316cfbce5b66685`.

**49 new regressions; 1592 tests + 81 subtests passed in 203.78 s**, after model completion. Next refine normative scope and explicit dependencies using heterogeneous development counterexamples, then freeze new first-attempt evidence. Never automatically promote headings or infer flow edges from list order. Once source quality stabilizes, test the existing skeleton/binding interfaces against actual host contracts with real 9B generation. Missing loops/scripts/approval stay explicit; safe stopping is not completion. No default DSH routing change, C4–C6 unlock or Git commit/push.

New raw/reviewed reports and zero-call reentry replay identically; the summary digest verifies and all three preceding historical reports remain unchanged. Targeted Ruff and `git diff --check` pass; existing repository-wide Ruff warnings are not claimed fixed.

### Explicit scope, condition and precedence prototype

New independent protocol `source-duty-guarded-meaning/v1` adds statement/scope/when/after before kind/evidence selection; previous code and evidence stay frozen. See the [implementation](../evaluation/flow_source_guards.py) and [probe](../evaluation/flow_source_guard_pilot.py). Relations remain source prose, not executable predicates, edges or authorization. Empty means not explicitly represented, never unconditional permission or proof of no prerequisite. Full source review checks omitted/invented relations, scope, polarity and normative roles; headings are not automatically promoted and rejected/historical quotes are not live commands.

Lossless labeled-text projection preserves every nonempty relation in the old inactive binding interface. `binding_input()` requires supported guarded-source review, but still requires new projection/full-binding review; no review is reused as authorization and no runtime activation or default-route change occurs.

The six-case source-only batch uses four unchanged regression inputs plus [two new visible development counterexamples](../examples/source-duty-development/guard-extra-cases.json). Cohorts are scored separately and neither is hidden/public-Skill evidence. It checks release-preview boundaries/rejected commands and access-guide descriptive headings/historical rules/owner-decision prerequisites. Original sources, first responses and failures stay intact. Completed reentry only replays; uncertain requests or transport/model-envelope faults stop without retries. Read-only report command and output directory appear above; current measured results are in [Project Status](PROJECT-STATUS.md).

### Completed six-case results and decision

Six real first attempts: unchanged regression group **4/4 structural, 4/4 source-reviewed, 0/4 supported**; new-development group **2/2 structural, 2/2 reviewed, 0/2 supported**. Same-assistant review has 72 overlapping claims (45 supported, 19 contradicted, eight insufficient), not independent Gold or accuracy. Regression inputs/outputs: **5,103/3,265 tokens, 236.89 s POST, p50/p95 54.70/78.15 s**. New group: **2,298/1,257 tokens, 89.88 s, p50/p95 40.34/49.53 s**. Total 11,923 tokens and 326.76 s; normal stops, no retry/truncation/timeout/business execution. Observed regression cost rises 27.2% in tokens and 85.2% in POST versus the previous four-source probe. Protocol/review/load differences preclude single-factor timing, capacity or generalization conclusions.

Ticket read-then-approval becomes partly explicit, but ambiguous multi-action source IDs remain as predecessors. New defects include requiring a prerequisite run before stopping for a missing runner, allowing verification after approval refusal, delaying the no-extra-read restriction, and narrowing missing-cursor handling to truncated=true. Normative headings still become context, while quoted rejected/historical instructions remain non-executable in the new examples. None/unknown fillers are not automatically scored as whole-case failure or treated as authorization; judgments identify concrete defects.

**Do not promote this free-text relation schema as the default upgrade.** Retain its diagnostic evidence and the source/host separation principle. Next use locatable source-evidence candidates for relationships and separately assess their existence and meaning; do not add more unrestricted descriptive fields. Exact quotes do not themselves prove semantics, and normative scope still requires review. This correction is not yet implemented or newly measured. Default DSH/Runtime routing stays unchanged.

**36 new tests; 1628 tests + 81 subtests passed in 204.34 s** after model completion. New raw/reviewed reports, zero-call reentry, summary digest and four historical reports verify; targeted Ruff/diff pass. Reviewed-report digest: `sha256:9801c65c26e747c0dac1f77cedb0e95b734528edf0a669b192242e6f924693d4`. [Tracked evidence](benchmarks/flow-source-guard-6-summary.json) includes sources, original candidates and every explicit judgment. No Git commit/push or C4–C6/large Runtime unlock.
