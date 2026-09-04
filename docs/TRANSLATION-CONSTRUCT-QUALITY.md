# 转译测试构造质量 / Translation Construct Quality

## 中文

### 当前判断

2026-09-04 的项目内人工式抽查发现：候选结构合法、同模型复核一致，仍不足以证明用例真实代表源 Skill。以下是**项目团队的开发发现**，不是独立 Gold，也不是 Translator 的错误率。原密封制品不改写；修订后的 Task/Catalog 必须另建版本。

| 现有候选 | 可直接核查的问题 | 对评测的影响 |
|---|---|---|
| development-06-001 / object-storage | 操作摘要涉及上传，问题却要求“Generate a read candidate” | 可以测试规划/候选生成，不能据此证明真实上传流程已被转译 |
| development-06-005 / runner-modal-images | 正文指定 `my-runner-job`，追加参数写 `image_name=my-runner` | 参数出现不等于参数无冲突，当前字面闭合检查可能掩盖歧义 |
| development-06-011 / code-delivery-review | 任务变成读取 Skill 定义并生成三个开发候选 | 测到了评测元任务，而不是原始代码交付审查业务 |
| development-06-004 / runner-modal-deploy | 参数样例包含字面字符串 `$(openssl rand -hex 32)` | 该字符串从未执行；但未求值的表达式不是已解析的真实凭证值，必须明确 fixture 语义 |

来源是 `artifacts/translator-v2/anchored-development-06-v1/candidates.jsonl`；其密封作者报告 digest 为 `sha256:4a8ba898dcf8b77ae0e33e8e4ee62d40d1fb52ebc76cab1168bd1cb86dcb3569`。

### 进入参考答案与 Translator 评分前

1. **先定义业务任务，再编写测试目录。** 原始 Skill 的操作、引用、步骤和安全边界应支持 Task；不能把不熟悉的 Skill 统一降为“读取 Skill 文本”。只覆盖一个窄操作族时必须报告覆盖范围，不能计作完整 Skill 转译成功。
2. **保持输入自然且无答案提示。** 移除由评测器加入的“生成 read/write candidate”等元任务措辞；匿名 ID、随机调用顺序和单任务模型会话已经实现，但不会自动修复原始问题的语义。
3. **不靠追加参数掩盖语义缺陷。** 正文、示例值、引用和 Schema 之间的冲突应单独记录。开发夹具可使用显式参数，但必须与自然语言任务分层统计，不能由作者补齐后再声称模型准确提取了缺失值。
4. **允许不可转译。** 开放式创作、诊断或缺少工具定义的部分可以保留 L1；分开报告总 Skill 数、适用 Skill 数、被覆盖操作/步骤数和正确转译数。
5. **答案独立于作者与 Translator 输出。** 参考答案角色只看固定 Skill、Task 和 Tool Catalog，不能看到作者处置、Translator 输出或前一个审阅者答案。无人工时标记 AI 模拟及模型相关性，不升级为人工独立证据。
6. **脚本只按不可信文本解释。** 区分能静态理解脚本声明与能够安全执行脚本。本阶段不执行第三方 scripts/hooks/installers，也不恢复 DSH/Runtime 大规模评测。

### 已实现：构造检查 v3（2026-09-04）

- 作者规范化不再追加、补齐或替换用户参数，逐次保留原始模型候选和修复版本；缺参交给显式作者修订，不在编译器中掩盖。
- `name=value` / `name is value` 的有限语法校验按实际类型取值，而不是要求等于 `example_value`；字符串大小写有区别，另一个合法值不能冒充缺参。
- 显式重复赋值冲突保留原文和字符位置；正常任务冲突会被拦截。缺参/冲突类任务可保留为待审追问候选。
- 已知“生成 read/write candidate”等评测元任务若缺少源任务支持，将被拦截；未求值表达式和占位符不能当作已闭合的正常任务输入。第三方文本永不执行。
- 新制品标记 `explicit_parameter_fixture`、`one_narrow_operation_family_not_whole_skill`、`sourceApiSchemaVerified=false`；Schema 是作者提出的惰性夹具，不是已核实的真实 API。
- 旧 v1/v2 制品按原版规则核验封存完整性；新规则审计必须写到新路径，不能覆盖原报告或改写 Gold。

对 development-06/07 的 16 个旧已接受候选进行静态复查，9 个通过新机械规则、7 个被拦截。这是**缺陷发现计数**，不是转译成功率或泛化证据。审计还发现 persistence-ports 与 customer-crm-lookup 的重复赋值冲突。仅通过机械检查的 9 个也不代表语义正确。

真实 `qwen3.5:9b` 在新协议下重新构造 object-storage（1 Skill/3 task），初次及一次修复后仍把上传改为“生成 read candidate”，并未提供要求的显式参数输入；两次调用合计 242.5 秒，0 个候选通过。新门禁拒绝了该错误候选、未自动补参；这验证了缺陷拦截，不证明作者生成或转译准确率提升。运行与本地全量测试并发，单样本含修复的时延不作性能对比。原始输出及修复版本均保留。

本轮定向测试 71 项、全量测试 658 项及 81 个子测试通过。实现摘要、制品路径与 digest 见[构造检查 v3 数据](benchmarks/translation-construct-v3-summary.json)。

剩余明确边界：例如原文写“named 'my-runner-job'”而参数为 `image_name=my-runner`，任意自然语言中的共指、否定和隐含冲突仍未被可靠识别。检测有限的同名赋值冲突不能声称解决全部语义冲突。无参 API 被作者补成必填参数、虚构验证/补偿能力、把完整写流程缩为读取等问题，仍须逐参数/逐步骤的源证据审查。

复查命令（输出路径必须是新的；不执行模型和第三方代码）：

```bash
python -m evaluation.translation_construct_audit \
  artifacts/translator-v2/development-corpus-100 \
  artifacts/translator-v2/anchored-development-06-v1 \
  artifacts/translator-v2/anchored-development-07-v1 \
  --output artifacts/translator-v2/construct-audit-v3/report.json
```

### 后续交付

[逐参数/步骤源证据审查](TRANSLATION-SOURCE-ALIGNMENT.md)已接通：检查项由代码生成，引用按原文定位，冲突/不足/协议错误均不进入后续参考答案步骤。首个真实 9B 案例识别了无参调用与两个必填参数的冲突，但因任务引用不完整而被整体拒绝；另有只读性质推断过强的问题，不能当作有效审查或转译成绩。

接下来修正构造与审阅中的通用语义缺陷，形成有效、带源证据的审查及隔离参考答案，再做 Gold-blind Translator 基线。评估至少分开显示：构造有效率、语义保真、适用覆盖、路由/参数准确率、引用/条件/审批/验证/恢复语义保留和时延。冻结后未知 cohort 才能提供泛化证据。

## English

### Current finding

A project-team audit on 2026-09-04 found that structurally valid candidates and same-model agreement can still misrepresent the source Skill. These are development findings, not independent Gold or measured Translator errors. Existing sealed artifacts remain unchanged; revised tasks/catalogs require new versions.

- `development-06-001` asks to generate a read candidate for an upload-related Skill. That can test planning, but not translation of the upload workflow.
- `development-06-005` names `my-runner-job` in prose while the appended `image_name` is `my-runner`. Literal presence does not establish conflict-free grounding.
- `development-06-011` asks to read Skill definitions and generate development candidates, replacing the code-delivery business task with an evaluation meta-task.
- `development-06-004` contains the inert literal `$(openssl rand -hex 32)` as a parameter example. It was never executed; its fixture meaning must not be confused with a resolved credential.

The source is the sealed development-06-v1 authoring candidate artifact; its report digest is recorded above. Before reference-answer authoring and Translator scoring, tasks must be grounded in actual Skill operations; natural inputs must be separated from explicit-parameter fixtures; contradictions must not be hidden by appending values; and narrow operation coverage must not count as whole-Skill conversion. Open-ended or unsupported semantics may remain L1, with applicability and correct coverage reported separately.

Reference authors must not see author labels, Translator outputs, or previous reviewer answers. AI simulation remains disclosed as correlated model evidence, never human-independent truth. Third-party scripts stay inert. The next work is to revise invalid development constructs, add conflict/meta-task checks, and then obtain isolated reference answers and a gold-blind Translator baseline. Unseen generalization requires post-freeze disjoint cohorts.

### Implemented: construct checks v3 (2026-09-04)

Normalization no longer appends or replaces user parameters; original model candidates and explicit repair attempts are retained. A bounded `name=value` / `name is value` parser validates actual typed values independently of example values, preserves case-sensitive conflicts with source offsets, and flags deferred expressions/placeholders. Unsupported evaluation meta-tasks are blocked. New artifacts explicitly identify parameter fixtures and narrow operation coverage; source API schema fidelity remains unverified. Legacy v1/v2 sealed artifacts retain their original validation rules; new audits use new files without relabeling originals.

A static re-audit of 16 previously accepted development-06/07 constructs passed nine and blocked seven under the new mechanical checks. These are defect-discovery counts, not translation accuracy or generalization. Arbitrary prose conflicts (including “named 'my-runner-job'” versus `image_name=my-runner`), negation, source API requiredness, and genuine verification/compensation availability still require source-grounded semantic review. Mechanical passes do not establish correctness. The next gate remains per-parameter/per-step evidence review and isolated references before Translator evaluation; Runtime remains locked.

A fresh 9B object-storage probe (one Skill, three tasks, two calls including repair) still substituted upload with a read-candidate meta-task. It was rejected without parameter insertion. This demonstrates defect interception, not improved author/Translator accuracy. Its 242.5-second latency includes repair and concurrent local testing, so it is not a performance comparison. Targeted tests: 71; full regression: 658 tests plus 81 subtests. See the [implementation-bound summary](benchmarks/translation-construct-v3-summary.json).

The [source-evidence reviewer](TRANSLATION-SOURCE-ALIGNMENT.md) now derives per-parameter/per-step obligations and resolves citations to source locations. Its first real 9B probe located the no-argument/schema contradiction but failed the citation-type gate; an overconfident read-only inference also remains. This is diagnostic evidence, not a valid review or Translator score. Valid source-cited reviews and isolated reference answers remain prerequisites.
