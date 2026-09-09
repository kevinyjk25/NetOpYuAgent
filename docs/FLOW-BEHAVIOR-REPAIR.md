# 转译行为闭环修复 / Translation Behavior Repair

## 中文

> 本文保留各轮实验事实及当时的验证数字；“本轮/下一阶段/未提交”是实验发生时的记录。当前入口见[代码导航](../evaluation/README.md)，实际推进状态见[项目进展](PROJECT-STATUS.md)。旧检查点须以清理前快照 `c2ebd78` 回放，不能用新实现改写旧 manifest；见[收敛说明](FLOW-CONSOLIDATION.md)。

### 修复目标与实际边界

本轮停止把“新增 Schema 字段、引用清单、拒绝测试”当成转译提升。改用原有 `FlowTree → FlowProposal → run_read_flow`：先证明人工参考表示在该执行结构中可行，再让模型从完整原文和宿主合同独立生成，最后检查真实执行器的内存调用轨迹。没有新增执行器，也没有扩大写权限。

本轮修复的是**实验转译入口与行为判定闭环**，不是自动上线 DSH 的新授权入口。源语义审阅、原文解释性限制、生产身份与真实设备安全仍不由有限行为测试证明。

### 根因与改动

| 原问题 | 本轮改动 | 仍不能声称 |
|---|---|---|
| 自由文本 statement/when/after 多次解释同一含义 | 新入口直接生成既有有序 AST；不经逐行义务改写，引用和原文仍保留 | 任意自然语言已可自动形式化 |
| 目的/引用标签抢先于操作生成，业务原文被长 Schema 隔开 | 校正版先生成步骤、后生成目的引用，并将完整原文放到输入末尾；保留语法说明、完整 Schema 和宿主合同 | 单个位置变化的因果贡献已独立测定 |
| 测试预先提供 contradicted，仅证明门禁拒绝 | 运行实际 read-flow 执行器；比较操作顺序、参数、次数、终态，自动输出反例 | Oracle 本身具有独立人工权威 |
| 注释歧义、能力缺口、业务错误合为“失败” | 分别报告 representation、behavior、scope/capabilityGaps、fullSourceReview；终态不匹配也单列 | 正确停止就是完成业务 |
| 无能力时反复要求模型生成完整流程 | 先跑人工可表达性 witness；缺失能力只验证停止边界 | 人工 witness 是模型生成成功 |

两个程序接口分别只接收它们需要的数据：

```python
wire = compact_request(sources)              # 原文 + 宿主；没有参考树/Oracle
result = check_behavior(sources, tree, suite) # 模型输出 + 私有行为用例
```

`compact_request` 复用既有 FlowTree，不增加自由文本语义字段。v1 删除重复 Schema 后出现错误引用和重复读取截断；v2 恢复语法说明、调整生成顺序后仍有错误；v3 仅对初始请求开启同一 9B 的思考开关，第一例达到 4096 token 上限。**三个实验变体均不采纳为默认升级**，新运行必须显式选择版本。`suite` 是测试数据，不是 L0 或模型生成的 Schema。逐行源审查旧协议保留为历史诊断证据，不再作为本实验生成业务树的必经前置步骤；这不意味着绕过产品激活门禁。

### 行为检验如何工作

每个场景提供调用输入、独立内存返回数据、预期调用和预期终态。模型看不到这些数据或人工参考树。返回数据按工具及参数查找，不按正确调用顺序喂答案；错误或未声明调用会先留痕再阻断。没有网络、CLI、插件加载或源脚本执行。

例如访问控制列举 available/current/granted 的全部八种布尔组合：只有全部为真才允许读取保护属性。另测缺返回字段、类型错误和 Provider 错误。原返回 Schema 校验仍由 Runtime 执行，不由模型或 Oracle 代替。

本轮机械反例已验证：

- 反转分支：可以通过类型编译，但实际调用和终态不符合要求。
- 复读使用原始输入而不是上次返回 ID：通过类型编译，第二次调用参数检查失败。
- 所有路径都停止：不会被计为业务成功。
- 在缺前置依赖时先读取元数据：即便最终阻断，多出来的调用仍使行为失败。
- 只检查 granted、漏掉 available/current：布尔组合提供具体越界读取反例。
- 仅修改合法变量别名：行为相同则通过，不使用参考 JSON 完全相等作为语义标准。

报告包含原文要求、输入与返回、期望/实际调用、首个差异、终态，以及关联的 `treePointer`、`l0Pointer`。终态错误、错误操作和无效引用应分别解读：例如缺依赖时返回 needs_l1 而不是 unsupported 是路由协议差异，不能声称已发生网络误操作。

### 范围与数据

共六个已经可见的开发源输入，人工参考表示通过 33 个有限行为场景：

| 输入 | 行为场景 | 当前检验范围 |
|---|---:|---|
| direct-read | 6 | 单次读取、参数、异常与鉴权阻断 |
| inverted-branch | 6 | 条件极性、输入/返回值区别、复读参数 |
| missing-approval-write | 6 | 首读后在审批/写合同缺失处停止；不完成变更 |
| unavailable-script-prerequisite | 3 | 缺脚本时停止在首次读取之前 |
| backup-prerequisite | 1 | 原失败样例原文不变；缺引用/runner 时不读取元数据 |
| access-guide | 11 | 所有者决定组合、固定 Q6 参数、保护属性读取、异常 |

前三个可检查业务片段共 23 个场景；三个缺能力停止案例共 10 个场景，两个分母始终分开。访问/备份工具是显式本地测试合同，内存 Provider 不是企业系统或真实网络设备。

仍无法靠本路径完整表达的例子：存储分页的无界循环；单个终态 Effect 后继续写后验证的完整工单流程；缺少实际引用和运行合同的脚本前置。这些必须保留能力缺口，接既有其他能力或回退 L1，不能静默降低为单次读取后宣称整 Skill 成功。

### 一个重要的历史纠正

对旧 `flow-node-evidence-4-20260908` 四个原始 9B 树做新的事后行为检查，原文与宿主合同逐项比对相同，原检查点和旧源审查未改：**21/21 有限行为场景匹配**，其中两例业务片段、两例缺能力停止。

因此，旧批次完整源审查全阻断，并不意味着其四个业务树都无法正确运行。新的行为 Oracle 是本轮开发助手事后构造，不能反向篡改历史成绩，不能从中推算新方案的语义准确率或泛化提升。这说明必须把业务行为与剩余解释/审阅缺口分开，而不是取消完整源审查。

### 本地使用与定位

源码：[行为验证](../evaluation/flow_behavior.py)、[可见场景与人工参考](../evaluation/flow_behavior_examples.py)、[基准探针](../evaluation/flow_behavior_probe.py)、[步骤优先转译请求](../evaluation/flow_behavior_compact.py)。

```bash
# 从项目根目录执行。冻结到新目录，不覆盖已有实验。
.venv/bin/python -m evaluation.flow_behavior_probe freeze artifacts/my-behavior-parent

# 基准请求：6 次首次调用，仍为 qwen3.5:9b/no-think。
.venv/bin/python -m evaluation.flow_behavior_probe run artifacts/my-behavior-parent --max-new-calls 6

.venv/bin/python -m evaluation.flow_behavior_probe report artifacts/my-behavior-parent \
  --output artifacts/my-behavior-report.json
```

每例目录中查看 `request.json`、`response.json`、`candidate.json`、`behavior.json`。结构失败时先看 `structuralError`；编译成功后查看 `scenarios[].differences` 与关联轨迹。原始响应、正常 stop/截断/传输失败和成本均保留。完成检查点重入只重放；不完整检查点或传输错误不自动重试。生成器源文件作为惰性快照留存，重放仍要求原始请求精确重建，以及原 Runtime、编译器、源文和 Oracle 指纹不变。

### 第一轮实测：检验闭环已修复，生成准确性未修复成功

所有比较使用相同六例、相同人工参考和私有行为 Oracle，不把它们送入模型。输入改变是在同一已知开发集上调试，不能作为未见集或单因素因果结论；只有思考对照的请求确实只改变 `think` 开关。

| 请求 | 实际新调用 | 结果 | 输入/输出 token | POST 合计 |
|---|---:|---|---:|---:|
| 初始直接 AST、无思考 | 6 | 可执行片段 2/3、缺能力停止 2/3 匹配；21/33 场景匹配 | 16,508 / 1,203 | 131.37 秒 |
| v1 精简请求 | 3 | 前两例非法 input 引用；第三例 4096 token 截断；余三例未运行 | 2,645 / 4,722 | 272.50 秒 |
| v2 步骤/源文顺序调整 | 6 | 同样 4/6 匹配；备份新增越过依赖读取，访问例因虚构输入字段编译失败 | 16,970 / 1,442 | 126.55 秒 |
| v3 原请求仅开启思考 | 1 | 第一例达到 4096 token 上限；余五例未运行 | 2,809 / 4,096 | 225.35 秒 |

本轮共 **16 次真实新调用、50,395 token、755.76 秒 POST**，包含截断失败，不自动补跑。初始六例请求 p50/p95 **20.12/34.18 秒**；v2 **18.62/36.45 秒**，质量未提升，不宣称时延收益。思考模式在当前预算下没有完成，不等于证明更长预算或 9B 整体能力无效。

初始备份例只产生 `needs_l1`，没有元数据调用，是终态协议差异；v2 却尝试读取元数据，随后因没有对应内存返回而阻断，属于真实操作反例。初始访问例直接宣称完成；v2 开始生成动作但引用了不存在的 caller `request_id`，且候选文本漏掉 available/current。不能把动作数增加当提升，或者因最终被阻断忽略之前的错误调用。

因此，**本轮完成的是可执行性证据、行为反例检测、源/AST/L0 定位以及失败分类，不是生成器准确率修复**。默认 DSH/Runtime 和既有门禁未切换；不增加测试数量或继续无界改提示词来宣布进步。后续应解决参数来源绑定和前置条件保真生成，依据同一闭环验证，再做独立源审查和未知样本验证。

完整可追溯数字见 [Git 内实测摘要](benchmarks/flow-behavior-repair-summary.json)；本地原始目录为 `artifacts/translator-v2/behavior-{repair,compact,ordered,thinking}-6-20260908`。新增 21 项测试；全量 **1649 tests + 81 subtests 通过（204.17 秒）**，定向 Ruff/diff 通过。未提交/推送，原型仍需完整源审阅、未见 Skill 验证；有限匹配比例不是生产成功概率。

### 后续修复：由编译器收紧参数和前置条件构造

新增 [契约化生成器](../evaluation/flow_contract_authoring.py) 与 [固定 Oracle 探针](../evaluation/flow_contract_probe.py)。这次不是再加一层自由文本 `after/when`，而是改变实际允许生成的操作：

- 每个工具使用自己的参数 Schema：必填、额外字段、标量类型都进入生成约束，并在模型返回后再验证。空 caller Schema 无法凭空获得 `input.request_id`。
- 输入引用只能选择声明的兼容字段；返回引用必须使用 `r0`、`r1` 等结果别名，实际前序可用性和该结果的字段/类型仍由原编译器检查。相同类型不代表语义正确，不自动猜参数。
- 参数常量从原文的词法候选中选择，保留源行及字符位置。没有硬编码 Q6、工具名或开发用例答案；同样不把“原文出现过”当成“适合该参数”。组合字符串、计算值、复杂对象等不在此有限标量前端的能力范围内，不能猜值冒充支持。
- `require_all` 把多个必须满足的条件集中表达，再确定性展开成原有 `if_equal`：任何一项不满足即停止，全部满足才继续。保留原有 `if_equal` 多路径能力；宏不新建执行器，也不会自动推断所有布尔字段都必须为真。
- `unavailable` 明确表示缺少必需能力，编译为 `unsupported` 终态并保留缺能力问题。若仍在其后放置读取，原编译器拒绝不可达步骤，不静默删掉或修复模型答案。

例如，访问控制人工解释为：读取决定 → 同时检查 available/current/granted → 读取属性。模型负责从源文选出这些含义，编译器只负责忠实展开已选择的条件。**模型仍可能漏选条件、选错具有词法依据的值或生成无动作终态**，所以行为 Oracle、完整源审查与激活门禁都不能取消。

新结果保留 `proposal.json → lowering.json → candidate.json → behavior.json`：`constructorOrigins` 连接构造器与原 AST；`argumentBindings[].lexicalWitnesses` 定位参数的原文字面依据；原 `compilation.origins` 继续连接 AST 与 L0。这些证据不会授予执行权限。

```bash
# 同一冻结源文、宿主合同、参考表示和 33 个 Oracle，不把答案送给模型。
.venv/bin/python -m evaluation.flow_contract_probe run artifacts/my-contract-probe \
  --parent artifacts/translator-v2/behavior-repair-6-20260908 --max-new-calls 6
.venv/bin/python -m evaluation.flow_contract_probe report artifacts/my-contract-probe \
  --parent artifacts/translator-v2/behavior-repair-6-20260908 \
  --output artifacts/my-contract-report.json

# 独立使用前端：FlowSources 是原文与已编译的真实宿主合同，不包含 Oracle。
.venv/bin/python -m evaluation.flow_contract_authoring request sources.json --output request.json
.venv/bin/python -m evaluation.flow_contract_authoring compile sources.json \
  --proposal proposal.json --output lowering.json
```

探针使用 qwen3.5:9b、no-think、同一 seed/上下文/输出预算；每例仅一次生成，不根据隐藏答案修图或重试。本文上一节的失败数据保持不变。新构造器仍为实验 authoring 入口，不是默认 DSH/Runtime 升级。

### 后续实测：参数有改善，完整前置条件保真仍未完成

同一六例、原文/宿主合同/Oracle 未改，完成六次新 9B 生成：

| 指标 | 第一轮基准 | 契约化首次生成 |
|---|---:|---:|
| 整例有限行为匹配 | 4/6 | 4/6 |
| 可执行片段 | 2/3，12/23 场景匹配 | 2/3，20/23 场景匹配 |
| 缺能力停止 | 2/3，9/10 场景匹配 | 2/3，9/10 场景匹配 |
| 全部固定场景匹配 | 21/33 | 29/33；实际运行 32，结构失败 1 |
| 输入 / 输出 token | 16,508 / 1,203 | 25,831 / 1,617 |
| POST 合计；p50 / p95 | 131.37 秒；20.12 / 34.18 秒 | 198.72 秒；28.67 / 43.48 秒 |

访问例正确把 `request_id` 绑定到原文常量 `Q6`，不再虚构 caller 字段，但仍只检查 granted：八种布尔组合中的 `001/011/101` 出现提前读取保护属性的反例。**29/33 不是“已经修好”**；这是已知输入上的局部改善，完整匹配例数未变，时延和 token 成本上升。备份例选择了缺依赖停止，但重复输出 `unavailable → end(unsupported)`，仍被旧严格编译器拒绝；原候选不改。

针对这两种具体错误，另做可追溯的后处理，而不是抹掉首次失败：

1. [重复终态规范化](../evaluation/flow_guard_binding.py)只折叠相邻且同为 unsupported 的停止声明，保存两处源引用、原语句、规则和指针。不是删除任何操作，也不允许忽略停止后的读取/写候选或不同终态。备份例由此恢复正确停止，0 新模型调用。
2. 同文件的 `slots_for → guard_request → author_guards → bind_guards` 为每个后续读取枚举**已经可用的布尔事实**，要求每槽明确判断 require_true、require_false、not_individually_required 或 unresolved，并给出原文引用。必需条件确定性编译到目标读取之前；不默认所有布尔量为真。任一 unresolved 时不产生新业务候选。输入仅原文、宿主和原候选，绝不包含 Oracle、参考树或行为判定。

本批只有访问例存在这样的依赖槽，因此只增加 **1 次 9B 调用，1,327/124 token，15.87 秒**。结果：available → require_true；granted → require_true；**current → unresolved**，定位在原文 s0003、读取 `/steps/1/when_equal/0` 之前的 `r0.current`。这是把一个静默遗漏暴露成可定位的未决项，**不是自动推导成功，也不是正确的业务停止**。没有替 9B 填上 require_true 来计算自动通过。

后处理后为 **5/6 有限用例匹配、1/6 待解疑**：其中两例可执行片段、三例缺能力停止；实际执行 22 个场景且匹配，访问例 11 个场景明确为 `not_run`。不能用 22/22 掩盖未决的 11 个场景，也不能与首次生成 29/33 混成同一种准确率。两阶段合计 **7 次新调用、28,899 token、214.59 秒 POST**；没有额外重试或人工答案修订。

限制：布尔槽位只覆盖候选中真实存在、已可用的读取结果；不能发现被整体遗漏的生产者/动作，也不自动解决 OR、数值范围、循环、权限真伪或原文歧义。新增约束是转译辅助，不是通用语义证明或激活权限。下一阶段应针对 current 这类源语义与字段含义的解疑进行诊断，再验证完整源审查与未见样本；本轮不扩大 Skill 库，不切默认 DSH/Runtime，不开启大规模 Runtime A/B。

见 [Git 内逐例摘要与反例](benchmarks/flow-contract-guard-repair-summary.json)。原始两批分别保存在 `artifacts/translator-v2/behavior-contract-6-20260908` 和 `behavior-contract-guard-6-20260908`；前者可零调用重放，后者保留逐项规范化、一次源判断及 unresolved 结果。所有源脚本和第三方 Provider 均未执行。

工程验证：新增 20 项定向回归，全量 **1669 tests + 81 subtests 通过（208.78 秒）**；相关定向回归 81 项、新增文件 Ruff、`git diff --check` 通过。测试验证机械约束与检查点行为，不是模型准确率。未提交/推送 Git。

### 最新修复：必要条件与正向路径可行性分离

核对原文和真实宿主声明，`current` 没有丢失：原文 s0003 要求 current owner decision，宿主说明也明确 current 表示该决定是否 current。此前没有证据足以把未知归因于模型大小或上下文漏传。

本轮增加一次 [反事实源判断](../evaluation/flow_guard_counterfactual.py)：针对每个已可用布尔事实，分别询问“为假时，在原文允许的其他条件下，这次读取是否可能执行”与“为真时是否可能”。问题只使用原文、宿主及固定候选；不包含测试输入/返回值、Oracle、参考树或原先的判定。字段说明原样放到对应槽旁，不补写语义。只判断指定的读取时点，不凭空增加新读取或替代证据。

9B 的真实回答对 available/current/granted 均为：**假时 forbidden，真时 unknown**。原先要求一侧 forbidden、另一侧 possible 才生成 Guard 的推导规则，继续把访问例留在 unresolved；该失败报告原样保留。

这里发现了代码中可修复的逻辑过强约束：

```text
“F 为假时禁止执行”  ⇒  “允许执行必须满足 F 为真”
                      ≠ “F 为真就允许执行”
```

[必要条件推导器](../evaluation/flow_guard_necessity.py)现在只从明确禁止的一侧构造必要 Guard；另一侧未知仍作为 `retainedUncertainty` 保存。负向条件对称处理；两侧都 forbidden 时不生成存活路径；两侧都 possible 不强加单项条件；没有足够禁止证据时仍 unresolved。不是把 unknown 填成 possible，也不是增加 `current=true` 的特判。

结果继续显式记录 `pathFeasibilityProven=false`、`sufficiencyProven=false`、`fullSourceReview=required_not_run`、`activationEligibility=not_established`。前提本身来自模型对原文的判断，尚不是被独立证明的事实。**这次允许生成未激活的必要条件候选，不允许绕过源审查或扩大执行权限。** 单项必要条件也不能证明 OR/联合条件、整段遗漏或完整 Skill 的充分性。

#### 同一组数据的分阶段结果

| 阶段 | 整例有限行为匹配 | 场景 | 新模型调用 |
|---|---:|---|---:|
| 上轮契约化首次生成，保持历史成绩 | 4/6 | 29/33 匹配 | 本轮 0；复用原六次 |
| 新反事实回答，旧严格推导 | 5/6 | 22 匹配、11 未运行 | 1 |
| 同一回答，分离必要条件/可行性后 | **6/6** | **33/33 执行且匹配** | 0 |

最后一行包括 **3/3 可执行片段、23/23 场景**，以及 **3/3 缺能力停止、10/10 场景**；不是六个完整业务 Skill。访问例所有八种布尔组合及三个异常场景均匹配。三项正向 unknown、旧首次失败、旧严格推导失败仍保留。源文、宿主、参考表示与 Oracle 摘要全部未变，没有人工替模型选择 Guard，没有为通过而重试。

本轮只新增 **1 次 qwen3.5:9b/no-think 调用，1,600 输入＋256 输出 token，23.14 秒 POST**；离线新推导新增模型调用为 0。连同上轮六次生成和一次失败的直接 Guard 判断，累计探索为 **8 次调用、30,755 token、237.74 秒 POST**。这不是新跑一遍端到端批次，也不表示降低了延迟；原首次生成的 p50/p95 和失败成本没有改写。

可复用入口如下。`author` 完成模型提问与必要条件推导；`bind` 可对保存的原始回答离线推导。产物始终未激活，已完成检查点可用零调用预算重放：

```bash
.venv/bin/python -m evaluation.flow_guard_necessity author sources.json tree.json \
  --output artifacts/my-necessary-guards --max-new-calls 1
.venv/bin/python -m evaluation.flow_guard_necessity bind sources.json tree.json \
  --answers answers.json --output necessity.json
```

查看 `derivations` 中的原始正反回答、原文引用、推导规则、目标读取位置和生成条件，再看 `retainedUncertainty`；不能只取出 Guard 而宣称其充分性已证明。实测见 [Git 内摘要](benchmarks/flow-guard-necessity-summary.json)。本地证据在 `artifacts/translator-v2/behavior-contract-counterfactual-6-20260908` 与 `behavior-contract-necessity-6-20260908`。

**本轮完成必要条件生成的一处算法修复，不宣称完整自动转译成功。** 下一阶段优先审查模型的源引用和条件前提，再对否定/别名/替代条件做小范围抗词面依赖验证；暂不扩大 Skill 库或解锁大规模 Runtime A/B。固定开发集匹配不是未见泛化或生产成功概率。

新增 23 项回归覆盖全部九种正反判断组合、负向必要条件、未知保留、矛盾停止、原始行为检验与零调用重放。全量 **1692 tests + 81 subtests 通过（209.41 秒）**；定向 Ruff 与 diff 检查通过。它们验证实现，不等于模型准确率。未提交/推送 Git。

## English

This document preserves historical experiments and their then-current test counts/next steps. Use the [code map](../evaluation/README.md) and [project status](PROJECT-STATUS.md) for active work. Replay old checkpoints with snapshot `c2ebd78`, never by changing old manifests; see [consolidation guidance](FLOW-CONSOLIDATION.md).

This repair separates executable behavior from explanatory annotations, representation failures and missing host capabilities. It reuses the existing FlowTree compiler and real read-flow executor, with inert in-memory providers only. The translator receives source text and host contracts, never the manual reference graph, observations or expected answers. No new executable IR, script runner or activation permission is introduced.

Six visible development sources have manual feasibility witnesses across 33 scenarios: 23 executable-fragment checks and 10 safe-partial-stop checks. Behavioral mutants demonstrate that reversed branches, wrong data dependencies, missing owner-decision guards, premature reads and stop-everywhere proposals produce actual trace counterexamples without a supplied AI verdict. Alpha-renaming does not fail an otherwise equivalent graph.

An additional post-hoc check of four archived 9B trees matches 21/21 finite scenarios. This does not rewrite their original blocked source reviews: it shows why behavioral correctness and unresolved explanatory/source-review issues must be measured separately. The oracles were authored by the same development assistant and are not independent Gold.

The original direct-AST/no-think probe matches two of three executable fragments and two of three partial-stop cases (21/33 scenarios). Three request variations failed to demonstrate improvement: v1 produced invalid aliases and truncated repetitive reads; v2 still matched 4/6 and introduced a premature backup read plus an invalid access input; v3 changed only the thinking switch and exhausted 4096 tokens on its first case. All failed outputs remain; no variant becomes the default. Across these batches, 16 actual calls consumed 50,395 tokens and 755.76 seconds of POST time. Unrun cases and truncations are not silently retried or dropped.

The verification/diagnostic loop is repaired, but generator quality is **not** repaired. The remaining work is faithful argument-source binding and prerequisite generation, followed by independent source review and unseen validation. See the commands above and the linked summary; inspect `behavior.json` for expected/actual calls, outcomes, counterexamples and AST/L0 pointers.

No production adapter or default DSH activation path is replaced. Full-source interpretation review, real-world host authority and unseen-Skill generalization remain open; finite behavior matching is not whole-Skill semantic accuracy or production reliability.

The follow-up contract-grounded front end specializes each tool's required arguments and typed input/result references. Argument constants must have exact lexical source witnesses. A `require_all` constructor lowers conjunctive prerequisites to existing branches; `unavailable` lowers an explicit missing-capability stop and issue. Existing multi-path branches, source review, graph/type validation and Runtime authority remain intact. These constructors do not infer the intended meaning, silently repair model output, or prove that a source-present literal is the correct parameter. Computed/composite values are outside this bounded scalar front end.

See the commands above for the standalone request/compiler interfaces and the comparison probe. `proposal.json`, `lowering.json`, `candidate.json` and `behavior.json` retain constructor/AST/L0 mappings and source offsets. The probe reuses unchanged sources, host contracts and private oracles; no reference tree or expected behavior enters model input. Historical failures are preserved, and this experimental entry does not replace default DSH/Runtime admission.

The follow-up measured six fresh 9B calls: complete finite-case matching remains 4/6; scenario matching changes from 21/33 to 29/33 (32 executed, one structural failure). The access parameter now uses the correct source literal Q6, but three Boolean combinations still cause premature protected reads because available/current were omitted. Input/output tokens rise to 25,831/1,617; POST time is 198.72 s, p50/p95 28.67/43.48 s. This is a local argument-binding improvement, not overall semantic success or a performance gain.

A separately recorded narrow normalization collapses only an adjacent unavailable/unsupported duplicate stop, preserving both citations; it never erases operations or different outcomes. A mandatory Boolean-dependency decision pass uses source, host and candidate only. One additional 9B call (1,327/124 tokens, 15.87 s) requires available/granted but marks current unresolved. No answer is filled in manually. The followed-up batch therefore has five matched finite cases and one unresolved case: 22 scenarios execute and match; 11 are explicitly not run. This is not a 22/22 success claim or first-pass accuracy. Total incremental cost: seven calls, 28,899 tokens, 214.59 s POST.

The dependency pass cannot discover entirely omitted producers/actions or prove arbitrary disjunctions, ranges, loops, authority or full-source meaning. Automatic semantic availability remains unresolved. See the [versioned summary](benchmarks/flow-contract-guard-repair-summary.json); no default DSH/Runtime path or admission gate is changed.

Engineering verification: 20 added regression tests; the full suite passes **1669 tests + 81 subtests in 208.78 s**. All 81 related targeted tests, targeted Ruff and diff checks pass. These counts are not model accuracy. No Git commit or push was made.

The latest optimization separates necessary conditions from positive-path feasibility. One new source-only counterfactual call reports false forbidden / true unknown for all three owner-decision facts. The old inference requires forbidden/possible and stays unresolved. The corrected inference uses `forbidden when false ⇒ execution requires true` without claiming `true ⇒ permitted execution`. Positive unknowns remain unchanged in `retainedUncertainty`; both-forbidden, unknown-only and negative-polarity cases are handled explicitly. The model premises, full-source semantics, joint sufficiency and activation remain unproven.

On the same archived first-pass trees and new unchanged model answers, corrected offline derivation matches **6/6 finite cases and 33/33 inert scenarios**: three executable fragments (23 checks) and three missing-capability stops (10 checks). This is not a fresh first-pass batch, six fully translated business Skills or unseen generalization. First-pass performance remains 4/6 cases and 29/33 scenarios. The strict-counterfactual failure and all positive unknowns are retained; no answers or oracles are edited, and no model call is retried to obtain a pass.

Incremental cost is one qwen3.5:9b/no-think call, 1,600/256 input/output tokens and 23.14 s POST; offline re-derivation adds zero calls. Including prior first-pass calls and the failed direct guard pass, cumulative exploration costs eight calls, 30,755 tokens and 237.74 s POST. Use `evaluation.flow_guard_necessity author` or `bind` as shown above; candidates remain inactive with source review required and no activation eligibility. See the [versioned evidence](benchmarks/flow-guard-necessity-summary.json). Next review source premises/citations and run narrow polarity/alias/alternative-condition checks, not a larger Skill corpus or Runtime A/B.

Verification adds 23 regressions for all nine answer pairs, negative guards, retained unknowns, contradictions, old behavior checks and checkpoint replay. The full suite passes **1692 tests + 81 subtests in 209.41 s**; targeted Ruff/diff checks pass. These are implementation checks, not model accuracy. No Git commit or push.
