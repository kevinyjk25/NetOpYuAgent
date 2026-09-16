# 收敛后的 Agent 运行主链 / Canonical Governed Agent Session

## 中文

2026-09-16 可选v4扩展：[编译/执行隔离](ISOLATED-COMPILER.md)。宿主配置`compilerMode=isolated`后prepare自动进行一次独立编译、原校验和获准只读前段；执行Agent使用五工具，不再看到编译Schema或调用submit，后端同样拒绝替换AST。缺省仍是下文六工具流程。与[显式必读门禁](CALL-PROTOCOL-EVIDENCE.md)兼容，无新增权限；机制通过不代表任务语义通过。

当前v4宿主已补齐[冻结快照语义和重复补读保护](SNAPSHOT-READ-SEMANTICS.md)。模型获得宿主绑定的observationModel；精确重复补读返回原观察引用，不产生新读取/回执；不同资源仍走原网关，尝试上限不变。这是固定本地数据的语义，不是实时设备规则。两例真实DSH＋9B结果及一项Oracle问题见专题；旧分与正式出口不变。

最新设计调整：[任务直达交付](TASK-FIRST-DELIVERY.md)。显式宿主v4使用`delivery=null`，完整任务由宿主绑定；开放输出为`{"answer":"..."}`，不经过模型挑选的职责类型/片段。严格读取、有限补读、一次生成、工件检查与终态控制不变。两次9B快照诊断有局部改善、仍有重复读建议；旧DSH评分不变，Gate 1仍开放。下方为此前协议和历史结果。

2026-09-15。根据本轮复盘，暂停继续叠加通用语义审阅/修订层。旧语义闭环批次保留为**未通过**，不改分、不改称成功。当前实施重心改为一条可使用、可分别测量的原型主链；原有独立泛化门禁和写操作边界不降低。

最新真实结果见[单选交付协议](DELIVERY-SINGLE-CHOICE.md)：显式宿主v3／交付v4每项为内容对象或未解决原因字符串，宿主状态与Agent文字独立。冻结DSH＋9B验收结构准入2/2、原判据4/6、完整任务0/2；Gate 1未通过，停止同例重跑。后续[宿主终态修复](HOST-TERMINAL-DELIVERY.md)已可选接入DSH公开生命周期并原样显示回执，实际DSH＋脚本化模型机制2/2通过；模型所选kind只作呈现提案，语义误配与遗漏没有解决。下方紧凑v3、wire-state和分入口实验为保留历史；旧配置不自动迁移。[真实模型机器摘要](benchmarks/delivery-single-choice-summary.json)。

### 使用链路

```text
用户任务 + 宿主配置的原始 Skill/工具合同
  → DSH 原生 LLM：prepare 获取原文，提出 read_prefix + 交付要求
  → submit：检查逐字任务/Skill 引用及来源/参数/图，冻结交付合同
  → 原 Runtime 仅执行严格读取，返回观察，尚不生成草稿
  → DSH/L1 检查索引并经同一网关补读必要证据（最多两次）
  → draft：冻结证据 → 一个有界 9B 节点处理原始任务
  → 类型化交付 → 确定性渲染 + 零模型调用工件检查 → 未验证候选
```

没有手工编写的每 Skill 执行图，也不要求先生成一份“AI 审阅批准文件”。**宿主允许读取，不等于宿主批准语义。** 输出 Schema、引用和图通过不自动授予业务成功状态。

产品 fallback 只能只读诊断、追问、提案或人工交接，不能直接写 Provider；L1 提议的补读也要经过原 `execute_host_read` 和精确资源 ACL，不是原地扩图。预算按尝试计数，拒绝也占用次数。原始提交只能执行一次；超时/未知状态只能 inspect，不能静默重试。

**编译拒绝不等于执行失败。** 当前最多接收两份提案：只有第一次在任何读取/模型调用前被确定性编译拒绝、并持久保存拒绝记录时，才返回 `translation_needs_correction` 允许改正一次。修订不放松来源、参数或权限合同，也不自动猜正确参数；已接纳的读取前段最多执行一次。中断导致提案状态未知时不允许修订/重放。第二次仍失败则转原生只读 L1；不可用的是 Runtime 模型生成路径，不是 L1 生成用户所需文本草稿的能力。fallback 使用独立 deliver 工具的文本检查入口，见下文。

`submit` 现在只执行编译图的严格前段，状态为 `collecting_evidence`，不返回模型草稿。新增 `draft(session_id)` 冻结前段及补读回执，宿主将实际历史观察投影到原始保留任务节点，以新的只推理图运行；不重放读取、不刷新证据有效期、不授予行动权限。空前段明确记录 `no_prefix_reads`，不伪装读取完成。

补读与冻结共享进程间锁；未知读取结果不能被跳过，冻结后禁止晚到补读或重复生成。模型不能在 draft 中提交证据、评分、路径或审批。**冻结只证明“这一批已收集的数据不再变化”，不证明“已收集到所有需要的数据”**；是否补读仍是 L1 的开放判断，当前最多两次，复杂探索可能需交接。

| 当前状态 | 允许继续做什么 | 不允许做什么 |
|---|---|---|
| 等待提案 | submit 提出读取前段及有原文锚点的交付要求 | 提前写草稿并声称已读取 |
| 首次确定性编译拒绝 | 同一会话修正一次提案 | 伪造引用、扩大权限、创建新会话逃避次数限制 |
| collecting_evidence | inspect、最多两次 read、最后 draft | 将图编译/读索引成功等同任务成功 |
| draft 已冻结 | inspect；接收一次候选及其静态检查 | 晚到补读、重复生成、改证据 |
| 来源预算或两次编译失败的 fallback | 先 submit(null plan) 绑定交付合同，再受限读；原生 L1 提交类型化文本给 deliver 检查/渲染 | 调 Runtime 模型、直接写 Provider、绕过合同先读 |
| 执行已停止／状态未知 | inspect、基于已知信息解释或交接 | 补读重放、静默重试 |

`skill_authoring/artifact_checks.py` 对候选执行不运行源码的围栏、JSON/Python 语法、字面比例及窄 KQL 日期/半开区间检查。时间依据只来自原任务和真实读取载荷，不取 Skill 示例或模型自述。报告给出候选摘要、观察摘要、逐项 pass/fail/unverified；明确错误标记 `candidate_needs_revision`，没有自动重写或重试。未支持的表达式不冒充正确；这不是完整 KQL 解析器、函数签名校验或数据库执行，也不能判断一般自然语言蕴含。DSH 随后自行改写的最终文字不在原候选摘要的校验范围内。

### 三类结果，不再用一个“通过”混在一起

| 结果 | 本轮如何记录 | 不代表什么 |
|---|---|---|
| 转译 | 原始包摘要、Agent 提交的任务文字、参数来源、严格读取节点、保留的 L1、编译拒绝 | Schema 通过不代表语义保真；prepare 保留提交文本，不证明 DSH 没有改写用户意图 |
| 执行 | 原读取回执、资源准入/拒绝、模型节点状态、写调用数、异常和耗时 | 图完成不代表用户问题解决 |
| 任务 | `candidate_unverified`，成功字段默认 `null`；评测在模型调用之外逐条对照预先固定判据 | 模型自评 supported、引用摘录和工程测试数均不能升级成功 |

安全控制可以接受突变/反例测试；开放语义质量则必须用外部任务判据评估。二者分别报告，不能以安全停止冒充可用性，也不能让正确的回答绕过权限。

### 本地接入

六个工具：`netopyu_hybrid_prepare`、`netopyu_hybrid_submit`、`netopyu_hybrid_inspect`、`netopyu_hybrid_read`、`netopyu_hybrid_draft`、`netopyu_hybrid_deliver`。DSH 插件和 Python CLI/常驻 Worker 调用相同实现。

宿主设置 `NETOPYU_HYBRID_HOST_PROFILE` 指向配置文件，`NETOPYU_HYBRID_SESSIONS_DIR` 指向输出目录。未设置时默认关闭；这些路径不能由工具参数提供。已有 DSH/Worker 进程需要带新环境重启，本轮不自动重启用户的页面实例。

配置结构为 `apiVersion`、`enabled: true`、`packet`、`resources`。`netopyu.io/local-hybrid-host/v1`保留旧引用/包装协议；显式选择`netopyu.io/local-hybrid-host/v2`启用[引用ID与紧凑内容协议](DELIVERY-REFERENCE-CONTENT.md)，两版不能混投，不自动迁移运行中会话。`packet` 包含原始 Skill 包、任务、输入 Schema、工具目录和来源绑定的 Read 合同；`resources` 是 `[精确参数, 模拟观察]` 清单。目前这是单机操作者授权的本地参考宿主，**不是生产身份、真实设备适配器或任意 API 配置器**。

可读样例见 [Skill](../examples/governed-session/SKILL.md)及其引用资料；完整可运行宿主见 [host.json](../examples/governed-session/host.json)。首次可用下面的任务测试：

在项目根目录、启动DSH和Worker之前设置（不修改正在运行的实例）：

```sh
export NETOPYU_HYBRID_HOST_PROFILE="$PWD/examples/governed-session/host.json"
export NETOPYU_HYBRID_SESSIONS_DIR="$PWD/data/hybrid-sessions"
export NETOPYU_OLLAMA_MODEL=qwen3.5:9b
export NETOPYU_OLLAMA_FAST_MODEL=qwen3.5:9b
```

之后沿用本项目既有启动方式。启动检查已将共享转译代码和启用的宿主配置加入指纹，避免仍展示旧工具Schema；本文不要求自动停止现有会话。

> 请按配置的 Skill 分析园区链路快照，区分观察与推测，给出一个后续检查建议，不下配置。输入参数：`{"exportPath":"/exports/campus/index.txt"}`。先 prepare，再按返回 Schema 提交计划；检查索引，用受限 read 补齐必要快照，最后 draft。

独立入口 `dsh-plugin-netopyu/src/hybrid-local.js` 仅注册这六个工具；普通 NetOpYu 插件在配置宿主时也注册它们。源脚本不会运行。CLI 可用 `python -m dsh_adapter.cli hybrid-prepare` 从标准输入接收 `task`、`arguments`；submit 接收 `session_id`、`plan`、`delivery`。正常 draft 只接收 `session_id`；仅执行前 fallback 的 deliver 接收 `session_id,response_json`（明确的一次JSON文本传输），检查原生 L1 文本且不调用 Runtime 模型。工具不接收 confidence、审批、输出路径或策略。完整来源超预算不能记为整 Skill 转译成功；其交付合同只能引用实际提供的任务，不能引用未提供的 Skill 页。

### 源文锚定的交付物合同

`skill_authoring/delivery.py` 把交付类型与执行权限分开。L1 从实际用户任务/可见 Skill 提出最多四项要求，宿主检查逐字引用、来源摘要和位置，分配稳定 ID，并在任何读取/生成前绑定会话。不是人工给每个用例写 L0，也不读取评测答案。

| 类型 | 固定内容槽 | 可以检查 | 不能证明 |
|---|---|---|---|
| artifact | body、语言 | 缺失内容、部分明显提纲形态；支持语言的独立静态检查 | 完整业务语义、可执行正确性 |
| analysis | text | 内容存在、结构合法 | 解释真实、充分 |
| decision | conclusion、basis | 显式结论及依据是否填写 | 推论成立 |
| next_steps | items | 后续项非空 | 行动适当、证据确实缺失 |

每项可标记 `unresolved`，此时必须留空内容并给出缺口；不能靠编造事实填满 Schema。当前v2使用 `unrepresented: [{origin, quote, reason}]` 保留这些类型无法表达的固定交付要求，逐字引用必须真实存在。它不是读取待办，不能用自由文本“还没读”替代原文要求。原生fallback与Runtime候选使用同一响应Schema、渲染器和静态检查；fallback不增加内层模型调用。合同、候选、冻结证据、检查结果分别有摘要，跨会话替换合同被拒绝。

**引用存在不等于类型选对，更不等于要求已全覆盖。** `shapeComplete` 只表示已声明结构填满且无unrepresented；`declaredCoverageComplete` 也仅指没有声明无法表达的要求，不是自然语言覆盖率。模型仍可能漏选decision/next_steps，或对一段真实引用作出错误解释。`semanticApproval=false`、`taskSuccess=null` 始终保留。DSH被要求原样交付 `task.delivery.rendered`，但仍可改写最终回复；候选摘要不覆盖改写后的文字。旧v1合同及其陈旧uncovered只读保留，不自动迁移、清空或重评分。

### 当前v2：传输、固定要求与证据状态分离

| 对象 | 谁生成 | 表达什么 | 不表达什么 |
|---|---|---|---|
| requirements / unrepresented | L1提出、宿主检查原文锚点并冻结 | 固定的交付类型与无法表达的源文要求 | 暂时没读到数据、不完整观察 |
| evidenceState | 宿主根据实际前段报告和补读回执生成 | 读取完成/拒绝/结果未知、精确工具参数、记录摘要及收集是否冻结 | 数据充分、事实为真、业务成功 |
| response.uncertainties / 每项gap | LLM看完观察后提出 | 当前仍不确定的内容 | 更改读取回执、清除固定要求或获得权限 |

`read`、`inspect` 返回宿主读取状态；冻结后该状态与证据一起绑定，并传给同一Runtime推理节点。记录“已读索引”不等于“已取得全部目标数据”；不存在读取记录也不证明底层数据不存在。完成读取不会自动清除unrepresented，模型的待办文字也不能把已完成读取改成未读取。

原生文本入口现为 **`deliver(session_id,response_json)`**。`response_json` 是明确声明的JSON字符串，宿主只解码一次，再使用与Runtime相同的严格内容Schema。不兼容旧的隐式 `response` 对象/字符串混用，不递归解码、不补括号、不执行代码，不转换数值/布尔/文本类型。拒绝重复键、非有限数、非对象根、Markdown围栏和超限内容；上限131,072 UTF-8字节。保留原始wire及摘要，失败给出 `diagnostic.code/pointer/detail`；JSON语法错误含行列位置，字段错误含JSON Pointer，均不回显拒绝载荷。

旧CAPA失败的零模型诊断进一步发现：字符串本身也在第1行第845列JSON损坏。新解码器仍拒绝，**不是“改用字符串就能修好旧内容”**。诊断制品 `artifacts/governed-session-20260915-wire-state-diagnostic/report.json`，摘要 `sha256:eaddc49a3d69af0673ff63a66c6cc77a34ec45ef2c9dd84a2f4edf8adb311b3d`；没有改旧候选或旧评分。新版本真实验证单独保存在 `...-wire-state/`。

### 历史真实验证：wire-state，未通过

同一DSH、9B、CAPA/Mesh原任务和原判据，另目录冻结运行，不改历史结果。**0/2任务、2/6内容判据**；相较上次同两例1/2、5/6是本批观测退步，不能以JSON传输改善宣称整体提升，也不能用两例判定统计显著因果。

| 环节 | CAPA | Mesh |
|---|---|---|
| 转译/路由 | 完整来源超预算，原生fallback | 两次非逐字unrepresented引用被拒，随后fallback绑定合同 |
| 实际读取 | 1份记录 | 索引及窗口，共2份 |
| JSON传输 | 首次候选成功解析，无修复 | 首次候选成功解析，无修复 |
| 内容结构 | 缺 `/delivery/d1/gap`，拒绝 | 缺 `/delivery/d1/state`，state错放到父对象，拒绝 |
| 用户结果 | 2/3内容判据；仍无明确开放状态，夹杂对接口失败的错误解释 | 420秒超时，只留下过程文字，0/3；不把被拒工具输入当最终答复 |

两例均为原生fallback，没有Runtime模型生成。因此本批不能证明内层推理的状态分离效果。CAPA将真实“三小节”格式要求保留为unrepresented是可解释的，读取不会清除它；Mesh仍把模板没有实例数据解释成固定交付缺口，即使引用最终逐字匹配，**语义误解依旧存在**。Mesh未接纳的候选还建议回滚实验，与本任务不改网络的范围不符；没有实际执行或授予权限。

19份完整DSH回复，249,800输入／8,249输出token；Mesh第11步只有开始记录、没有完整回复和用量，**总费用未知，上述数量是下界**。统计器的callsWithUnknownUsage=0仅覆盖已完成回复。端到端p50/p95 311.20／409.15秒，包含超时，n=2；两例与此前三例批次不混算。开发者AI按原判据审阅，不是独立Gold，也不是新来源或正式A/B。

绑定报告 `artifacts/governed-session-20260915-wire-state-assessment/report.json`，摘要 `sha256:c0d71c8536eb5e7ea2550af8076deca80241207ae802ee14db66066242499aac`；[可随仓库查看的机器摘要](benchmarks/governed-session-wire-state-summary.json)。63份运行制品和75个归档源码文件摘要一致。

**运行后修复，尚未真实模型复验**：公共结构校验器把缺字段诊断从父对象精确到缺失成员，不自动补默认值、移动字段或回显拒绝数据；draft/deliver明确返回retryAllowed=false和终止指引，说明“保留候选”不等于“候选校验通过”。两份原始失败候选零调用复查仍拒绝，但准确定位gap/state；旧报告不覆盖。同类失败没有实质改善，本轮停止追加同例提示词/模型重跑。

本轮最终工程验证：174项定向、**3,166全量测试＋81子测试**（243.02秒）、105个变更/新增项目Python文件Ruff、3项文档检查、实际六工具Schema和git diff检查通过。全量pytest在真实模型批次结束后运行。新诊断最初导致1项旧错误文案断言失败，更新为精确缺字段语义并保留拒绝/不填默认值断言后通过；没有放宽原Schema或任务评分。既有未修改文件的lint问题不在本轮清理范围，不能宣称全仓lint clean。

### 当前门禁与下一步

| 顺序 | 当前状态 | 要解决的问题/出口 |
|---|---|---|
| Gate 1 主链可用性 | 进行中，未通过 | 先解决合同类型解释、固定要求与读状态的语义混淆、候选完整提交及最终交付；本轮只完成传输/诊断机制修复 |
| Gate 2 小批冻结迁移 | 待Gate 1满足后开始 | 冻结实现后至少6 Skill／4仓库／3领域／12任务；旧已知例不得计新来源 |
| Gate 3 跨批泛化 | 未开始 | 至少3个不相交cohort、合计50 Skill／15仓库／8领域／600 case，按既定门禁逐项验收 |
| Gate 4 原生DSH对照 | 未开始 | 同模型/Skill/工具，唯一变量为auto Runtime；分别报告路由、任务、安全、成本和时延 |

引用寻址、紧凑内容和Schema约束解码的后续实现/失败已在[新协议报告](DELIVERY-REFERENCE-CONTENT.md)记录。下一包优先互斥状态表示、职责类型解释和最终交付；是否加入有界、仅文本的修正，仍需独立定义次数、冻结证据与成本，不能混成工具重试。现有重试权限不变，不叠加通用AI自审，生产工程后置。

### 历史：交付合同v1真实结果与接口修复

首批交付合同版本 `artifacts/governed-session-20260915-delivery-contract-local/` 使用不变的三份 Skill、任务、输入及原判据，**0/3 完整任务、3/9 判据**，相较前版 1/3、5/9 是退步。不能用合同上线或工程检查数宣称效果提升。

| 位置 | 实际失败 | 本轮处理 |
|---|---|---|
| 意图→合同 | IRQL 被选成 analysis＋next_steps，没有 artifact；逐字引用仍能通过 | 留作类型解释失败，不用查询关键词硬编码答案或改评分 |
| 读取计划 | 同一参数混合 caller 和 literal 两种互斥来源，第二次提交原样重复 | 两次均拒绝；实际 fallback 读取两份数据，不获得转译成功记分 |
| 合同→候选 | 三例均把读取前的暂时未知写成固定 uncovered；CAPA无明确开放结论 | 缺口保留，不能声称合同完整或语义正确；后续需把固定要求与证据状态分开 |
| 候选→最终答复 | CAPA/IRQL均被DSH改写；IRQL只给“已交付”的摘要 | 两份候选与最终文本精确一致 0/2；不把不一致自动等同语义错误，但候选检查不覆盖改写 |
| 会话模式 | Mesh已接纳读取图，却7次提交仅fallback允许的 response，420秒超时 | 拆分 draft（Runtime生成）与 deliver（原生文本）；返回实际状态和精确下一入口；null plan不得切换已接纳路由 |

CAPA 2/3（下一证据恢复、仍缺明确开放结论）；IRQL 1/3（实际读清单，但没有完整查询，且把“未捕获结果”扩成“没有实际事件”）；Mesh 0/3（不把未接纳工具参数中的分析算用户交付）。开发者AI按原判据审阅，不是独立 Gold；CAPA继续使用既有的保守显式状态口径。

首批1个自动图、5次授权读取、2份原生文本候选、0次Runtime模型生成。26份已完成DSH模型回复记录347,906输入／9,509输出token；Mesh还有1个已开始但无完整回复/用量的步骤，**总成本未知，已知数只是下界**。旧汇总器的 `callsWithUnknownUsage=0` 只统计完整回复，不含这个中断步骤；[机器摘要](benchmarks/governed-session-delivery-summary.json)明确补充该限制。含超时样本的端到端p50/p95为287.96／406.83秒，非SLO或因果性能对照。

绑定报告摘要 `sha256:b6fded1b53dc6ec5ebfc66457698a693a7c29d482e7a8092cd76835d20db0988`；另有最初沙箱TCP绑定失败目录 `...-delivery-contract/`，模型调用0，原封保留。后续分入口复验单独放在 `...-delivery-split/`，不与首批三例混算，不重评分IRQL，不扩大新来源或正式A/B。

历史分入口版本的 draft **只接收 session_id**；当时的 deliver **接收 session_id,response**，仅执行前fallback可用；当前v2改为上文的response_json协议。错误模式不触发隐式模型调用、不自动切换路由。补读回执会返回 `deliveryAction`；重复 submit 明确 `submissionIgnored=true` 和 `existingSessionRoute`，避免将“已提交”误读成“已转fallback”。这修复接口歧义，不等于解决类型解释、证据状态和语义质量。

分入口定向复验已结束：**Mesh原3项内容判据满足，CAPA 2/3**，按旧评分器为1/2、合计5/6判据；这不是完整三例重跑，也不是所有任务要求都被证明。Mesh在196.51秒内完成一次Runtime生成，之后一次多余deliver只返回原候选，无替换/重复生成，未再循环超时。最终仍有四个小节，超过原任务三个小节的格式要求；候选还残留已读待办、陈旧uncovered及不受观察支持的证书告警建议，不能把内容判据满足当作全语义正确。

CAPA在79.50秒内产生原生最终文字，但提交的response是JSON编码字符串而非对象，`DataBindingError` 拒绝，`task.delivery=null`、`task.status=not_completed`。它没有已校验渲染物；最终文字也仍缺明确开放结论。两例共1图、3读取、1Runtime候选、0份准入的原生文本候选；不能说两条路径都通过。可比较的候选/最终文本精确一致0/1，另一例没有有效候选可比。

复验13次模型调用（DSH12＋Runtime1），130,104输入／3,563输出token，本批完整回复均有用量。端到端p50/p95 138.01／190.66秒，n=2，不用于证明性能提升。绑定摘要 `sha256:18a89771f3e27b18e065d697c70d76ef646125a62da2ddfb0294514ce55556f2`；[两批分列的机器摘要](benchmarks/governed-session-delivery-summary.json)。没有修改旧成绩或用例答案，也没有执行源脚本、设备写入或查询。

**本轮出口：交付合同工程和模式循环修复完成，语义阶段仍未通过。** 剩余改进优先级是原生提交协议的清晰类型与诊断、固定交付要求/临时证据状态分离、交付类型与最终答复偏移定位；不靠多轮自审或在同一批上无限调提示词。新来源门禁、大规模Runtime A/B继续关闭；没有提交/推送或重启用户UI。

本轮工程验证：91项定向、全量 **3,138测试＋81子测试**（227.18秒）、3项文档检查、103个变更/新增项目Python文件Ruff、实际导出的六工具Schema及 `git diff --check` 通过；两批153份绑定制品摘要逐项一致。全量pytest在所有真实模型运行结束后执行。全库另有未改动文件的历史Ruff问题，不宣称全仓lint clean。工程检查数不能替代本轮未通过的语义验收。

### 代码与旧实验的关系

- `skill_authoring/`：原转译器、参数绑定、原文分页、来源校验与模型输入的共享实现，不导入评测代码。
- `dsh_adapter/hybrid_session.py`：宿主配置、单次会话、受限 fallback 和分离状态。
- `network_runtime/l0/`：继续使用原混合调度器和严格读取网关，没有另建执行引擎。
- `evaluation/`：预设判据、冻结实验、报告。旧转译模块仅兼容导入共享实现；旧语义自审实验不进入新主链。

### 历史：证据先行 v2 验收与当时根因

`artifacts/governed-session-20260915-evidence-first-v2/` 三例真实原生DSH＋9B已结束。相同Skill、任务、输入、预先判据；没有手写L0、提供期待查询或换模型。结果为 **1/3完整任务、5/9判据**，不是阶段语义通过。

| 任务 | 实际改善/保留行为 | 仍未满足 |
|---|---|---|
| CAPA | 原生fallback读取记录，保留25/25实施、5/30监测及未批准 | 未明确开放状态；此次还缺用户要求的下一份待补证据，1/3判据 |
| IRQL | 首次伪原文常量被拒，9B在执行前改正为caller绑定；读索引、补读清单、冻结后才生成 | 提供查询步骤/片段而非完整查询，缺EnvTime上的实际半开区间谓词，1/3判据；片段有局部价值，但不计完整交付 |
| Mesh | 实际窗口补读先于生成，数值、相关性/因果边界及日志下一步正确 | 本例3/3，但不能外推其他任务 |

2个自动混合图、5次实际授权读取；两次Runtime生成均使用先补读后冻结的证据，未重放读取。16次模型调用（DSH外层14＋Runtime内层2），136,754输入／3,982输出token，用量未知0。端到端p50/p95为138.67／172.44秒，模型运行未与本轮全量pytest并发；仍只有3例，不是SLO或因果性能对照。对比首批证据先行版本，任务率未变、判据6/9→5/9，不能以调用17→16宣称整体改善。

两份Runtime候选的静态报告都是 `no_supported_artifact`：一个是普通分析，一个是查询步骤列表，都没有受支持的独立代码工件。因此没有“查询校验通过”的证据，检查器也没有返回整体成功。IRQL的内层候选仍含不必要的重新验证/重读待办；这些未验证字段保留在原报告中，不应被描述为语义已修好。

评分为披露的开发者AI审阅，不是独立Gold；开放状态和完整查询采用保守显式口径，理由与候选/最终输出摘要可复查。[两版本逐项机器摘要](benchmarks/governed-session-evidence-first-summary.json)。最新绑定报告 `...-evidence-first-v2-assessment/report.json` 的摘要为 `sha256:1ef062de1557d68596079c2b356d2cfe8cd3aeff05505ea2a2298b38eb6d59ad`，首批失败不覆盖。无写操作、查询执行、来源脚本或真实设备操作。

**结论：证据次序和编译恢复的机制改动完成，整体语义验收未通过。** 剩余核心不是再加通用AI自审，而是当前自由文本 `draft` 缺少明确交付物合同，允许“查询说明”替代“查询工件”，也允许状态回答遗漏下一步。下一阶段应从真实用户任务/L1输出要求提取可审查的交付类型、必需结构和未满足项，分别验证代码工件与开放分析；不能将评测答案写入宿主配置或为每例手写L0。先完成这项通用接口修订，再进行有界开发验收；不继续在当前三例上堆提示词轮次，也不启动大规模Runtime A/B。未见来源门禁仍保留。

工程收尾：最终版本68项定向检查、全量 **3,115测试＋81子测试**（224.78秒）、双语/文档链接3项、变更路径Ruff、JS语法及 `git diff --check` 通过。扩展到全库的Ruff扫描另发现34条旧问题，全部位于未改动文件，本轮未自动清理；不宣称全库lint clean。未提交/推送，也没有重启用户当前UI；需要以新代码重启配置了宿主的DSH/Worker，才能加载五工具协议。

### 历史：先草稿后补读版本的原生 DSH + 9B 结果

使用三份已有公开 Skill、三个原始任务及原判据，没有向模型提供手写 L0、预期答案或审阅结果。首次运行因入口问题无完整任务成功，原制品保留；修正工具 Schema 和 fallback 后另目录运行。两批都是已知开发数据，不能冒充未见验证。

| 任务 | 实际链路 | 按原判据审阅 |
|---|---|---|
| CAPA 状态 | 完整来源超预算，真实 DSH 经 L1 fallback + 原读取网关取回记录 | 2/3 判据；人员、25/25实施、5/30监测及未批准均保留，但未明确给出记录保持开放的结论，保守不计完整成功 |
| IRQL 查询 | DSH 自动转译成功，原 Runtime 读索引并调用9B | 1/3 判据；DSH未补读清单而复述候选，缺时间窗并新增 `FailedCount > 5`，不合格 |
| Mesh 分析 | 自动转译 → Runtime读索引/推理 → DSH经同网关补读窗口 → 最终答复 | 3/3 判据；北区0.5%→5%、p99 400→900ms、南区对照正确，保留根因未知并列出下一证据来源 |

**总体：1/3 完整任务，2/3 形成自动混合图；4 次实际读取全部匹配宿主资源清单。** 三例用户任务和参数均逐字/逐值保留。12 次真实9B调用（DSH外层10次、Runtime内层2次），80,505输入／3,502输出token，用量未知0。观察到的端到端p50/p95为119.51／172.92秒；部分CPU测试与模型运行重叠，不是SLO或因果性能对照。

评分来自披露的开发者AI内容审阅，不是独立Gold。CAPA采用保守的显式状态口径，具体理由可复查；不可将这个小样本比率解释为总体准确率。没有配置写入、源脚本或真实设备操作。IRQL说明**未验证候选仍可能被DSH直接复述**；权限收口并没有自动解决答案正确性。

[可移植的逐项指标与判据](benchmarks/governed-session-local-summary.json)。本机原始制品为 `artifacts/governed-session-20260915-live/`（首次失败）、`...-live-v2/`（修正后），内容审阅为 `...-review-input/judgments.json`，绑定证据为 `...-assessment/report.json`，摘要 `sha256:d615e99421c20960010d37c87e1d3907ea0c428b2fab630c8314240249b60113`。没有覆盖历史记录。全量3,101测试及81子测试通过；工程通过不升级语义成绩。

这批完成的是**单一原型入口的收敛、接入和已知开发验证**，不是完整语义泛化阶段。以上负结果促成当前证据先行版本。后续新来源冻结验收仍至少6 Skill／4仓库／3领域／12任务；三例接线结果不能替代它。若验收失败，输出失败分类与路线决策，不自动无限追加审阅层。

### 证据先行首批：负结果也保留

`artifacts/governed-session-20260915-evidence-first/` 使用不变的三个任务/判据。结果仍是 **1/3 完整任务、6/9 判据**。Mesh 完成“前段读取→补读→冻结→一次生成”，前段不会再提前生成草稿；CAPA仍缺显式开放状态。IRQL把调用参数误写成带伪原文引用的常量，编译正确拒绝；fallback确实读取两份资料，但把“Runtime draft工具不可用”误解为无法提供查询草稿，最终只输出摘要。它不是查询校验成功。

这批只有1份自动混合图，5次实际读取，17次真实9B调用，146,784输入／3,442输出token；端到端p50/p95为120.11／127.51秒。任务率没有提升，调用数及输入成本反而增加；样本太少且流程/模型回复不同，不能宣称速度改善。绑定报告 `...-evidence-first-assessment/report.json` 的摘要为 `sha256:21609be222903bb223897d2020c2cd65c68e828594ed3f52d795f13cc22d5243`。开发者AI按原判据审阅，不是独立Gold。

上述问题促成“执行前一次有界提案改正”和明确fallback文本输出权限；旧结果不重评分。它们是接口修订，不新增通用AI审阅层。

## English

Optional v4 [compiler/executor isolation](ISOLATED-COMPILER.md) uses operator compilerMode=isolated: prepare performs one separate author request and original admission/read prefix. The execution Agent has five tools,no author schema/submit;backend injection is also denied. Default remains the six-tool flow below. Existing [explicit evidence gates](CALL-PROTOCOL-EVIDENCE.md) remain compatible;no new authority or implied semantic approval.

The v4 local host now declares [immutable snapshot semantics](SNAPSHOT-READ-SEMANTICS.md). Exact repeated follow-ups reference the original observation without new reads/receipts; distinct resources retain the same gate and attempt limits. This is not a live-device rule. Two actual DSH/9B task results and an Oracle mismatch are documented without regrading old results or changing research exits.

Latest opt-in design: [task-first delivery](TASK-FIRST-DELIVERY.md), host v4 with delivery=null and a host-bound complete task. Open output is one unverified answer, not model-selected typed duties or executable L0. Read, evidence, one-generation, artifact and terminal controls remain. Two known-snapshot 9B calls show a local improvement while redundant-read advice persists; old DSH scores and gates are unchanged. Earlier protocols/results follow.

Latest real-model results: [single-choice delivery](DELIVERY-SINGLE-CHOICE.md), explicit host v3 / delivery v4, yields2/2 structural admissions,4/6 criteria and0/2 tasks. Same-case reruns stop; Gate1 remains unmet. The later opt-in [host-terminal repair](HOST-TERMINAL-DELIVERY.md) passes2/2 actual-DSH/scripted-model lifecycle checks and displays host receipts unchanged. Model-selected kinds remain presentation proposals; semantic errors and omissions are not fixed. Compact v3, wire-state and earlier experiments below remain historical; no configuration is automatically upgraded. [Portable real-model results](benchmarks/delivery-single-choice-summary.json).

### Latest wire-state result and exit

The frozen two-known-case DSH/9B run finishes at **0/2 tasks,2/6 original content criteria**, versus the preceding selected run's1/2 and5/6. This observed regression is retained; two samples establish neither statistical causality nor performance improvement. Both first native JSON submissions decode without repair, but neither satisfies the typed response schema. CAPA lacks /delivery/d1/gap; Mesh lacks /delivery/d1/state with state misplaced at the parent. CAPA's final prose preserves two criteria but no explicit open conclusion and wrongly calls the rejected structure correct. Mesh reads the actual index/window but times out at420 seconds, leaving only diagnostic prose. Rejected tool arguments are not final delivery.

Both paths are fallback, with three authorized synthetic reads and no Runtime model generation. CAPA's real three-section output restriction is retained; Mesh still treats missing template data as an output requirement despite exact quote binding. Its rejected text suggests a rollback experiment outside the no-change request; nothing executes. Nineteen complete DSH replies account for249,800 input/8,249 output tokens, plus one started Mesh step with unknown usage. End-to-end p50/p95 is311.20/409.15 seconds including timeout, n=2, not SLO or causal A/B. Developer-AI review is not independent Gold. Bound assessment:c0d71c8536eb5e7ea2550af8076deca80241207ae802ee14db66066242499aac;63 run artifacts and75 archived source files verify. [Portable summary](benchmarks/governed-session-wire-state-summary.json).

After this frozen run, missing-required-field diagnostics now name the exact schema field without defaults, relocation, coercion or payload echo. Draft/deliver return retryAllowed=false and explain that retaining a candidate does not mean validation passed. Both originals still fail zero-call rechecks at the correct pointers; **these post-run repairs have no real-model retest** and do not regrade the failed run. Stop repeat prompting on these cases. Gate1 usability is still open; subsequent gates are frozen6-Skill/4-repository/3-domain/12-task transfer, at least3 disjoint cohorts totaling50 Skills/15 repositories/8 domains/600 cases, then same-DSH native versus auto Runtime A/B. Next reduce redundant structure/source transcription and clarify kind/final-delivery semantics. Any future bounded text-only correction must have distinct attempt/evidence/cost rules, never imply execution replay. Production engineering stays deferred.

Final engineering QA:174 targeted checks,3,166 full tests plus81 subtests in243.02 seconds, Ruff on105 changed/new project Python files, three documentation checks, actual six-tool schema and diff checks pass. Full pytest ran after the live batch. The new required-field wording initially failed one old exact-message assertion; its expectation now verifies the precise missing field and unchanged rejection/no-default semantics. Original schema and task criteria are not weakened. Existing lint findings in untouched files are out of scope, not a repository-wide clean claim.

### Current protocol and historical results

Current v2 separates the declared transport, fixed output requirements and dynamic evidence state. Native deliver(session_id,response_json) takes an explicit JSON-text envelope, decoded once with duplicate-key/nonfinite/non-object/fence/byte-budget rejection, then validated against the SAME typed response schema as Runtime output. There is no implicit field coercion, recursive decoding, bracket repair or code execution. Raw wire/digests are retained; safe diagnostics expose JSON line/column or field pointer. The maximum is131,072 UTF-8 bytes. A zero-model diagnostic of the old CAPA input reveals malformed JSON at line1,column845, in addition to the earlier string/object mismatch; it is still rejected, not repaired/regraded (digest:eaddc49a3d69af0673ff63a66c6cc77a34ec45ef2c9dd84a2f4edf8adb311b3d).

The v2 contract replaces free-text uncovered with source-anchored unrepresented entries {origin,quote,reason} for fixed output requirements the kinds cannot express. Host-generated evidenceState records actual read completion/rejection/unknown outcomes, exact tools/arguments, source report digests and collection closure. It is returned during collection and frozen into the reasoning input and final candidate report. It does not prove payload sufficiency or truth; reading an index is not acquiring every required fact. Current uncertainties/gaps remain model claims after reading and cannot rewrite receipts or erase fixed requirements. Shape/declared coverage flags do not prove semantic coverage. Old v1 contracts retain their historical annotations unchanged. Final text may still be rewritten by DSH and needs separate evaluation. The new native check is stored as wire-state; older results below remain historical.

The first source-anchored delivery-contract batch regressed to **0/3 complete tasks, 3/9 unchanged criteria**. CAPA restores next evidence but lacks explicit open status (2/3 under the same conservative reading). IRQL chooses analysis/next_steps rather than artifact, repeats a mixed caller/literal binding, reads both exports in native fallback, but delivers no query and makes an unsupported no-events assertion (1/3). Mesh completes authorized reads but repeatedly submits fallback-only response arguments on the admitted Runtime path; seven rejections precede the fixed 420-second timeout (0/3). Unaccepted tool-call arguments are not final answers. Developer-AI review, not independent Gold.

One compiled graph, five authorized reads, two native text candidates, no Runtime model drafts. Twenty-six completed DSH replies record347,906 input/9,509 output tokens, plus one started harness step without a completed reply/usage: costs are lower bounds, not complete totals. The legacy assessment's zero unknown usage refers only to completed replies; the [portable summary](benchmarks/governed-session-delivery-summary.json) explicitly discloses the outstanding step. End-to-end p50/p95 including timeout:287.96/406.83 seconds; no SLO/causal claim. Bound report:b6fded1b53dc6ec5ebfc66457698a693a7c29d482e7a8092cd76835d20db0988. An earlier localhost-bind startup failure has zero model calls and remains preserved.

The historical split-tool interface separated Runtime draft(session_id) from native-fallback deliver(session_id,response); current v2 uses response_json instead. Wrong-mode calls never silently switch routes or invoke a model. Read receipts return a state-specific deliveryAction; repeated submissions explicitly report submissionIgnored and existingSessionRoute. A null plan cannot convert an admitted Runtime session into fallback. The selected repair check is stored separately as delivery-split, never pooled into the failed three-case score. No IRQL retuning/regrading or large A/B. Kind interpretation, transient pre-read gaps wrongly frozen as uncovered, and rewritten final answers remain open issues. Exact candidate/final text matches are0/2, not a semantic-equivalence score.

The split-tool repair probe is complete: Mesh meets its three frozen content criteria; CAPA meets2/3 (legacy score1/2 tasks,5/6 criteria). Mesh completes one Runtime draft in196.51 seconds, with no loop/timeout; a later unnecessary deliver returns the existing candidate, without replacement or another model call. The final still uses four subsections despite a three-section instruction. Its candidate contains already-read next actions, stale uncovered items and an unsupported certificate-alert suggestion. Criterion completion is NOT proof of every task requirement.

CAPA produces native final prose in79.50 seconds, but passes JSON encoded as a string instead of an object: DataBindingError, task.delivery=null and status=not_completed. No validated native rendering exists, and explicit open status remains missing. The two cases yield one graph, three reads, one Runtime candidate and zero accepted native text candidates—not two passed paths. Exact candidate/final matches:0/1, with the other case unavailable for comparison. Thirteen model calls (12 outer DSH, one Runtime),130,104 input/3,563 output tokens; complete replies have usage, p50/p95 138.01/190.66 seconds (n=2, no causal performance claim). Bound digest:18a89771f3e27b18e065d697c70d76ef646125a62da2ddfb0294514ce55556f2. [Separate batch summaries](benchmarks/governed-session-delivery-summary.json).

This closes the delivery-contract implementation and the observed mode-loop repair, NOT semantic acceptance. Next priorities are clear native-response types/diagnostics, fixed requirements versus transient evidence state, and delivery-kind/final-answer drift. Do not add universal self-review loops, retune indefinitely on this batch or open large Runtime A/B. No commits/pushes, user-UI restart, source-script execution, device writes or queries.

Historical split-tool QA:91 targeted checks;3,138 full tests plus81 subtests in227.18 seconds; three documentation checks, Ruff on103 changed/new project Python paths, actual exported six-tool schemas and git diff checks pass. All153 bound artifact digests across both batches match. Full pytest ran after all live model work finished. Pre-existing Ruff findings in unchanged files remain; no repository-wide lint-clean or semantic-pass claim.

September 15: stop accumulating general semantic review/repair layers. Historical semantic-closure batches remain failed and unchanged. Focus on one usable prototype path without lowering independent generalization or Effect boundaries.

Historical evidence-first v2 acceptance: still **1/3 complete tasks, 5/9 criteria**, versus 6/9 in the preceding run. CAPA preserves facts but omits explicit open status and next evidence. IRQL corrects one invalid caller binding before any read, gathers the inventory before drafting, but delivers only pipeline fragments rather than a complete query with an actual half-open time predicate. Mesh fulfills all three criteria. Two compiled graphs, five authorized reads and two evidence-frozen Runtime drafts; no read replay. Sixteen real model calls (14 outer DSH, two inner Runtime), 136,754 input/3,982 output tokens, zero unknown usage; end-to-end p50/p95 138.67/172.44 seconds, without concurrent full pytest in this run. Three known cases are not a causal performance comparison or SLO.

Final QA:68 targeted checks;3,115 full tests plus81 subtests in224.78 seconds; three bilingual/link tests, changed-path Ruff, JS syntax and diff checks pass. A wider Ruff scan reports34 pre-existing findings exclusively in unchanged files, not fixed here; no repository-wide lint-clean claim. No commit/push or restart of the user's UI. Existing configured DSH/Worker processes need a restart to load the five-tool protocol.

Both Runtime candidates receive no_supported_artifact, not query validation success. IRQL's inner candidate also retains unnecessary re-read/revalidation notes; source/semantic correctness is not solved. Developer-AI review and conservative explicit-status/complete-query criteria are disclosed in the [portable comparison](benchmarks/governed-session-evidence-first-summary.json); bound report digest:1ef062de1557d68596079c2b356d2cfe8cd3aeff05505ea2a2298b38eb6d59ad. No writes, executed queries/scripts or real devices. Mechanism changes are complete, semantic acceptance fails. Next is a general deliverable contract grounded in actual task/L1 requirements—not evaluator answers or per-case hand-written L0—so code artifacts cannot silently become narrative outlines. Do not add more prompt/self-review rounds on these cases or open large Runtime A/B; fresh-source gates remain separate.

Native DSH obtains the original Skill/task/host contracts with prepare, proposes a read_prefix, and submits it once. Submission executes ONLY the strict prefix. L1 inspects the observations and gathers required evidence using up to two additional reads through the same gateway BEFORE calling draft once. Draft freezes sealed observations and projects them into the original retained-task reasoning node, executed as a new reason-only graph. Reads are not replayed; snapshots never regain freshness or action authority. Collection closure does not prove evidence completeness. A cross-process lock prevents collection/freeze races. Unknown operations are not retried; late reads are rejected.

One deterministic pre-execution compilation rejection permits one corrected proposal (two proposals maximum). No read/model operation may have occurred; both proposals/rejections are retained. Unknown proposal/execution state never permits correction or replay. At most one admitted prefix executes. Native fallback may still deliver requested textual drafts via the text-only deliver entry; only Runtime model generation is unavailable. This is bounded interface recovery, not semantic self-review or weakened admission.

Shared static artifact checks make no model calls and execute no source code. They cover fences, JSON/Python syntax, literal ratios and a narrow KQL date/half-open interval subset, with anchors from the actual task/read payloads, not Skill examples or model assertions. Explicit failures mark candidate_needs_revision; unsupported expressions stay unverified. They do not validate full KQL syntax, function signatures, source truth, natural-language entailment or final prose independently rewritten by DSH. No semantic self-review/automatic repair loop was added.

Three independent axes remain visible: translation structure/provenance (not semantic fidelity), execution contracts/receipts (not task completion), and task quality assessed outside the model against predeclared criteria. Task success defaults to null. Preserving prepare's submitted text does not prove that DSH preserved the original user's intent. AI confidence is neither evidence nor authority.

Configure NETOPYU_HYBRID_HOST_PROFILE and NETOPYU_HYBRID_SESSIONS_DIR on the host; no tool argument can select them. The feature is disabled without a profile. Restart an existing DSH/Worker with those variables to load it; this change does not restart the user's UI. The tracked host.json and Skill example above are runnable synthetic local inputs, not real devices or production identity. The isolated hybrid-local.js plugin exposes prepare/submit/inspect/read/draft/deliver; the normal plugin also exposes them when configured. CLI and Worker share the same implementation. Source-budget/compilation fallback retains native read-only L1 and earns no whole-Skill translation credit; text-only checking does not invoke Runtime reasoning.

The current source-anchored delivery contract is proposed from the actual task/visible Skill before observations or generation. Up to four host-assigned IDs select artifact (body/language), analysis (text), decision (conclusion/basis) or next_steps (items). Exact quote membership, source digest/offset and session binding are checked; coverage and type interpretation are explicitly NOT proven. No evaluation answers or per-Skill hand-written L0 are supplied. Unresolved items require empty content plus an explicit gap; uncovered requirements remain visible. Shape completion never changes semanticApproval=false or taskSuccess=null.

Submit now takes session_id, plan and delivery. Normal draft(session_id) freezes receipts and calls one bounded Runtime model using the typed response schema. Pre-execution fallback first binds a null-plan contract, then gathers authorized reads, composes the typed response in native L1 and calls deliver(session_id,response); that entry validates/renders text with zero Runtime model calls and no artifact execution. Overflow can quote only the supplied task, not unseen Skill pages. Contract swaps across sessions, native override of admitted Runtime drafts, late reads and repeated generation are rejected. Deterministic rendering narrows silent format substitution but does not establish semantics. DSH is instructed to deliver task.delivery.rendered faithfully; final rewriting is still possible and must be assessed separately. Model-proposed uncovered items may also incorrectly capture transient pre-read gaps. No general AI reviewer or automatic repair loop was added.

skill_authoring owns shared compilation/source validation without evaluator imports. dsh_adapter owns host sessions. The existing Runtime owns execution. evaluation owns frozen tasks, expectations and reporting; old authoring modules are compatibility aliases, not competing compilers. Old review loops are off this path. The current known-case native DSH check is neither unseen generalization nor paired A/B. A fresh frozen 6-Skill/4-repository/3-domain/12-task check remains separate; failed acceptance must end with a diagnosis/decision, not an unbounded new reviewer loop.

Live results: the first three-case native run failed at entry; its evidence remains. After the interface/fallback repair, CAPA used native L1 plus the original read gateway (2/3 criteria, conservatively incomplete because open status was not explicit). IRQL compiled/executed but failed its task (1/3: no follow-up inventory read, missing time window and invented threshold). Mesh completed the scoped task (3/3) through automatic compilation, Runtime read/reason, native DSH follow-up and final analysis. Overall 1/3 tasks, 2/3 compiled graphs, four actual reads within the host inventory. Original tasks/arguments were retained in all three cases. Developer-AI review is disclosed, not independent Gold; the conservative CAPA judgment and all reasons are portable in the linked JSON.

First evidence-first run (preserved): still 1/3 tasks, 6/9 criteria. Mesh completed collection before its only bounded draft; CAPA remained partial. IRQL fabricated a source quote for a caller value, was correctly rejected, read both exports in fallback, but confused unavailable Runtime drafting with unavailable native textual drafting and delivered only a summary. One compiled graph, five reads, 17 real model calls, 146,784 input/3,442 output tokens; observed p50/p95 120.11/127.51 seconds. No task-rate improvement and greater call/input cost; not a causal speedup or generalization claim. The bound report digest is 21609be222903bb223897d2020c2cd65c68e828594ed3f52d795f13cc22d5243. This motivated bounded pre-execution proposal correction and explicit fallback output permission; old scores stay unchanged.

There were 12 real 9B calls (10 outer DSH, two inner Runtime), 80,505 input and 3,502 output tokens, no unknown usage. Observed end-to-end p50/p95:119.51/172.92 seconds, partly overlapping CPU regression work; not SLO or causal speedup. No writes, scripts or real devices. Full regression:3,101 tests plus81 subtests. The integration milestone is complete, not semantic generalization: IRQL demonstrates that an unverified candidate can still be repeated by the harness. Next work should place bounded evidence gathering before delivery and evaluate suitable independent artifact checks, not add a universal AI self-review stack.

That earlier next step is now implemented as the evidence-first protocol above. Execution-stop fallback also forbids additional reads to avoid replaying a possibly uncertain prefix. The full semantic/generalization gate is not thereby passed.
