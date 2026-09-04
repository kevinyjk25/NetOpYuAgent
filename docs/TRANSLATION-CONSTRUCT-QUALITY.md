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

### 下一步交付

先修订上述有问题的开发案例，并增加“不添加参数”“正文/显式参数冲突”“业务流程被降为元任务”的负向检查，再形成隔离参考答案和 Gold-blind Translator 基线。评估至少分开显示：构造有效率、语义保真、适用覆盖、路由/参数准确率、引用/条件/审批/验证/恢复语义保留和时延。冻结后未知 cohort 才能提供泛化证据。

## English

### Current finding

A project-team audit on 2026-09-04 found that structurally valid candidates and same-model agreement can still misrepresent the source Skill. These are development findings, not independent Gold or measured Translator errors. Existing sealed artifacts remain unchanged; revised tasks/catalogs require new versions.

- `development-06-001` asks to generate a read candidate for an upload-related Skill. That can test planning, but not translation of the upload workflow.
- `development-06-005` names `my-runner-job` in prose while the appended `image_name` is `my-runner`. Literal presence does not establish conflict-free grounding.
- `development-06-011` asks to read Skill definitions and generate development candidates, replacing the code-delivery business task with an evaluation meta-task.
- `development-06-004` contains the inert literal `$(openssl rand -hex 32)` as a parameter example. It was never executed; its fixture meaning must not be confused with a resolved credential.

The source is the sealed development-06-v1 authoring candidate artifact; its report digest is recorded above. Before reference-answer authoring and Translator scoring, tasks must be grounded in actual Skill operations; natural inputs must be separated from explicit-parameter fixtures; contradictions must not be hidden by appending values; and narrow operation coverage must not count as whole-Skill conversion. Open-ended or unsupported semantics may remain L1, with applicability and correct coverage reported separately.

Reference authors must not see author labels, Translator outputs, or previous reviewer answers. AI simulation remains disclosed as correlated model evidence, never human-independent truth. Third-party scripts stay inert. The next work is to revise invalid development constructs, add conflict/meta-task checks, and then obtain isolated reference answers and a gold-blind Translator baseline. Unseen generalization requires post-freeze disjoint cohorts.
