# 语义规划前端 / Semantic Planning Frontend

## 中文

本前端处理 **L1 原文 → 原文核对/清单 → 携带原文 ID 的闭合控制树 → 机械保留来源 → 分步填参 → L0 读取图**。它是可选研究路径，不改变 DSH 默认路由，不自动激活合同。阶段 1 的 v62 已提交；阶段 2 保留 [v63 首批负结果](STAGE-2-PUBLIC-TRANSFER.md)，当前 [v65 修订](STAGE-2-REPRESENTATION-REPAIR.md)支持动态右值比较、input/observation 路径分区、完整行优先分页和缺失诊断触发的惰性来源检索。完整源文与预算保留；闭合控制树现为 v3，不另用模型重新绑定出处。见[当前进展](PROJECT-STATUS.md)；下文各旧版本是历史诊断，不代表当前能力或新的成功率。

### 为什么调整

阶段 1 的历史验收：同版 v62 / 9B 的 6/6 只读区域通过原文/参数审阅，50/50 本地路径通过。详见[完整结果与边界](STAGE-1-RESULTS.md)；三种已知开发流程不是公开泛化证据。阶段 2 已启动，不把预算预检通过或 Schema 合法计作语义通过。

原来的后置义务对账能够指出错误，但不能替模型构造正确的程序。新的前端把模型需要同时处理的问题拆开：

1. **先写带出处的步骤清单，再决定业务流程。** `source_scan` 保留全部原文片段的角色与解释；`procedure` 保留业务行动、两侧条件结果与剩余输出职责，宿主要求单独进入 `execution_requirements`。然后生成读取、条件、终止和待交接职责，参数稍后构造。清单不是独立 Gold，也不自动证明图覆盖了原文；它让“原文理解错误”和“清单到图遗漏”可以分开审查。
2. **观察结果使用固定命名空间。** 模型为每个读取选择不同的 `obs0`–`obs7` 观察槽，后续只引用 `input` 或已定义的观察槽，并用精确 JSON Pointer 选择字段；原文 ID、工具名和字段名不能当数据变量。编译器建立符号表并解析为 Tree 路径，再检查先后和分支作用域。不允许引用未来或兄弟分支结果，也不允许重复名称遮蔽输入。语法允许某个槽，不代表它已经定义或选对了业务依赖。
3. **提供真实类型导航。** `planningValuePaths` 来自调用方输入和原工具输出 Schema；不包含运行结果、答案或自动目标映射。它限制拼错的字段路径，但不能证明选对字段。
4. **补齐确定性数组长度。** `array_length` 只接受声明为数组且实际值符合整个源 Schema 的数据，结果为整数。它经过原数据绑定器和共享 Flow 引擎，不增加执行器。含该表达式的绑定制品使用 v2；原表达式仍生成 v1。
5. **职责放在准确的终止路径上。** `needs_l1` / `unsupported` 必须有出处、适用条件和具体职责；完成终止不能同时欠交接职责。全局任务外职责仍保存在 `remaining`。不能删除带有职责的不可达终止来制造通过。
6. **未来执行要求单独保留。** `execution_requirements` 记录源授权、资源范围和时效要求，始终 `satisfied=false`。离线构造无需实时凭据，但源文中特定业务审批条件仍必须进入业务流程；不得挪成泛用权限声明。

7. **来源随节点传递，不事后重建。** 每个读取、条件、终止及单项交接职责在生成时携带 `source_id`，选择原文块或原始行的 ID。[内联来源前端](../evaluation/source_inline_program.py)只按该选择保留准确引文、位置和完整父块，不能改业务逻辑。程序重排同步保留新旧节点位置；未知或漏填的 ID 被拒绝。片段枚举不排序、不猜业务对应；真实出处仍可能不支持模型逻辑，必须审阅。
8. **短引用主动纳入上下文。** 可选前端在预算内装入入口当前页直接链接的最多两个短、无歧义、已封存的原文参考文件。只加入完整页；大文件、缺失、歧义或脚本仍显式留在导航中，不下载或执行。装入不等于审阅，原始源包不裁剪。
9. **后续阶段不丢掉已有的语义上下文。** 填参保留完整冻结计划和原文；清单仍是未验证解释，不是 Gold，入口链接不代替被引用文件中实际定义条件的段落。填参使用同一冻结观察名称，而非让模型重新拼深层 Tree 路径。代码只对当前读取的支配来源做名称→路径解析，保留原回复与逐项映射；不会解释模板字符串或放宽字面值出处检查。原默认前端仍使用原路径表示。

模型的 `program` 是**闭合控制树**，不是程序字符串或运行时 Tree。每个节点有 `op` 和 `source_id`；交接中的每项 `duties` / `restrictions` 也各有自己的 `source_id`。`read` 有一个 `next` 节点；`if_equal` 有 `when_equal` / `otherwise` 两个子节点；`complete` / `handoff` 没有后继。因此语法本身不允许在全部路径终止后追加工作，不能生成空分支或隐式成功。比较统一使用 `value.kind=field`（标量）或 `length`（数组长度）；原 Schema 限制相应指针和值类型，实际业务依赖、比较值和两侧流程仍由模型选择。读取只选原 Catalog 工具和观察名称，参数稍后填写。

[闭合语法转换器](../evaluation/source_closed_program.py)逐节点转换到原类型化语句，保留原节点→语句位置的 `controlSyntaxLowering`，不删除节点、不插入条件、不推断成功。整个程序仍限制 32 个节点、8 层条件及 8 个读取，实际来源、类型和作用域继续由原编译器校验。内部语句列表与等价 `if_length_equal` 保留供旧语法验证，不是当前模型格式。语法闭合不证明原文覆盖完整或语义正确。

[类型化语句渲染器](../evaluation/source_program_lines.py) 将每行机械转为受限 Python 风格文本，再由原白名单解析器检查作用域、类型、控制流和预算。它不修改、补齐或优化模型选择的业务流程。以下只是渲染语法示意，不是预置答案或可执行 Python：

```python
observation = read("原工具名")
if field(observation, "/原输出字段") == True:
    end("read_path_completed", "源文定义的读取流程完成")
else:
    end("needs_l1", "还有解释工作未完成",
        [("当前分支条件", "需要 L1 完成的具体职责和限制")])
```

[source_program](../evaluation/source_program.py) 只用 `ast.parse` 构建并检查白名单语法树，**不调用 Python 的 `compile/eval/exec`，不导入或执行源脚本**。读取参数仍由后续单独阶段构造；规划代码中不能调用真实工具。条件的分支、顺序与终止再机械转换为原有计划块和 FlowTree。分支自然落入后续语句是控制流延续，不是推断成功；未结束的根路径被拒绝。没有开放任意 Python、循环、函数、算术、属性访问或写操作。

早期直接生成程序字符串会混入工具参数；v35 的字符串正则实验又触发本地结构化解码的 JSON 转义错误，已停止并保留失败记录。v36 扁平语句层级数字、后续分支列表仍可能生成无效延续；v60 用闭合树约束后继，普通封闭 JSON 对象不再依赖代码字符串正则。该变更只约束合法表示，不能证明正确选择了源语义。

v59 六份构造已经完成：审批、引用各两份通过原文/参数审阅，合计 32/32 本地路径通过；接口两份仍被拒绝，原因是终止后追加宿主授权工作，且错误计数读取的出处指向结果处理行。固定六份门禁仍失败。v60 不修改这些旧结果，改用闭合控制树，并在选择读取适配器前声明其原文出处；待新一版实测，不预报语义成功。

v60 首轮接口、审批分别通过原文审阅与 9/9、8/8 本地路径；引用流程虽编译，却将原文 `ready=false` 翻成等价的 `ready=true`，并把条件出处误选成“否则读取”句，因此拒绝。全量 2522 项与 81 子测试通过不能改变该结论；第二轮已发出的接口请求留存中断，不重试。v61 要求保留原文显式比较值与分支顺序；“否则”所继承的条件必须引用定义句，不能只引用动作句。没有给出任何样例专属答案，也不自动修正旧图。

v61 的提示词调整未修好出处，并出现“可用→整体健康”的终点扩写，因此只跑完引用首份后停止该版本。v62 让所有节点先选 `source_id`，再构造类型化操作数/动作；闭合控制树 v2 不接受终点自由文本 `explanation`。编译器只生成固定的“读取路径到达边界，未断言更广业务结果/未执行剩余职责”状态标签，原文中必须完成的解释、诊断和建议仍是带来源的 L1 `duties`，不能以标签代替。`controlSyntaxLowering.terminalLabels` 公开这些固定模板；不把模型草稿或流程状态升级为业务事实。旧自由文本候选不被覆盖或清洗。全仓 Ruff 另发现 224 条历史问题，均在与 HEAD 相同的 58 个文件内；本阶段修改文件检查通过，不宣称全仓 clean。

v37 还暴露了**支配关系与词法嵌套不一致**：`if 条件: 结束; else: 读取 B` 后使用 B，B 实际出现在所有能够到达后续步骤的路径上，却被旧前端一律拒绝。当前只对“两臂中恰有一臂能继续”做结构化延续转换，将后续语句移入唯一继续臂。另一臂必须由显式终止证明已关闭；不猜条件、不复制/删除调用、不删除不可达工作，也不允许两臂都继续时泄漏局部值。`controlFlowNormalization` 留下原语句、移动规则及新旧位置映射；原词法检查和共享 Runtime 支配校验继续生效。它不是按业务答案修复计划。

v42 对齐旧前端已有的冗余终止规则：所有进入路径已显式终止后，允许留痕移除**单个、无职责的末尾完成节点**。原语句和位置存入 `controlFlowNormalization.redundantTerminals`，不是覆盖模型回复。不可达读取、交接职责、其他工作或多个末尾节点仍拒绝；不能把 `needs_l1` 改成完成，也不补出任何业务条件。

内部/人工语法仍允许 `part = field(observation, "/object")`；它是惰性引用别名，不是新增读取或即时 Python 运算。编译器组合指针、检查类型和来源并保留 `fieldBindings`。模型端不暴露该便利语法，因为真实 9B 曾把标量别名误当原观察根，生成 `/approved/approved` 一类重复路径；使用观察根和完整路径能表达同样的读取/条件逻辑。未来权限只进入 `execution_requirements`；真正缺失的业务规则进入 `business_gaps` 并在填参前阻断；当前任务以外职责进入 `outside_task_duties`。不会靠删除缺失业务规则换取通过。

局部重复参数在此可选前端显式记录：非推理为 `presence_penalty=0, repeat_penalty=1`；显式 `--reasoning` 为 `presence_penalty=1.5, repeat_penalty=1`。后者恢复此前观察到的模型 presence 默认值：v39 在 0 惩罚的推理模式下耗尽 8192 token，没有最终回复，因此另立配置验证。早先非推理模式的惩罚参数单独对照并未解决语义问题；不能把配置变化直接描述为准确性提升的主因。默认 `direct` 保留原调用配置。

v42 的审批流程通过构造及 8/8 本地路径检查，但引用多步流程的推理请求在 360 秒超时；因此该配置没有通过退出门禁。v43 保留同一类型化前端，重新验证已有的非推理配置。历史请求和失败不覆盖、不重试。Ollama 仅报告所保存的输入/输出计数，没有独立 reasoning token 分解；这些计数不能被描述为经过核验的全部推理成本。

v43 的剩余失败集中在表示选择：将字段访问变成新增读取、把深层字段路径截短、把动态实参写成模板字符串。v44 不修改源文、字段答案或接受规则，而是清理输入界面：规划指令只解释一种 JSON 语句语言，不混入 Python 风格调用示例；填参只展示当前目标工具及所有支配观察的名称、原工具、原文和完整类型导航，不再让参数构造重新解释已冻结的带注释程序。完整原计划仍留存在制品中，原文页和当前调用的源依据继续提供。参数类型相同不代表业务角色相同，模板也不会被当作引用执行。

上述 v44 填参上下文缩减**未被保留为当前方案**：它仍生成对象到标量的错误绑定和虚构字面值，并让审批参数退步。v45 恢复完整冻结计划上下文，保留简洁的 JSON 规划语言；增加显式 `--argument-reasoning`，仅在填参阶段为同一个 9B 启用推理和 8192 输出预算，规划/出处绑定仍为非推理 4096。它与全阶段 `--reasoning` 互斥，限 `--semantic-plan` 使用；选项、实际请求和每阶段成本全部留痕。不能把该配置调整单独归因于算法准确性提升。L1 交接不得额外发明审批或确认前置条件。

v45 的单独填参推理仍耗尽 8192 token（352.816 秒），没有最终候选，因此不能算可用。v46 在固定、全必填、封闭对象形状上加入[参数槽前端](../evaluation/source_argument_slots.py)：代码从原 API Schema 生成参数路径和对象结构，模型只给每个标量槽选择类型兼容的来源 ID，如 `input#/request/id`，或声明带出处的真实常量。参数名称只是 Schema 路径，不是业务答案；同类型的多个来源全部保留，不做字段名猜配。

该路线仍保留完整冻结计划/原文上下文。每项选择机械还原为原绑定表达式，并依次经过类型、作用域、字面值出处和原编译器；不解释模板。合法但未列出的类型化路径/数组索引可通过显式 reference 对象表达，继续执行原 presence/type 检查。可选键、开放对象、数组参数或超出导航预算等复杂形状**返回原通用参数构造路径**，不裁剪字段或丢弃功能。参数槽路线最多 64 个标量槽、6 层目标形状；不构成任意 Schema 的新编译器。

这不是语义保证：合法图仍可能重复调用、遗漏判断、反转条件或错误解释原文。候选保持 `compiled_region_requires_semantic_review`；必须审阅原文、参数来源和路径行为。模型解释、引用命中、类型正确均不授予 Runtime 权限。

v46 的参数构造能完成三个首份流程，但引用流程遗漏了父文档输出限制，且把条件统一绑定到入口段落，仍未通过语义审阅。v47 将 `handoff.duties`（待做工作）与 `handoff.restrictions`（适用限制）分开填写；各最多 8 项，渲染后作为同一路径的最多 16 项原交接元数据，不执行任何限制或工作。该变化补回了引用流程的保密限制，但“抄引文 + 填块 ID”又暴露改写和坐标不一致。

v48 改为**选择出处，不生成出处**：`frozenProgram.evidenceChoices` 列出当前原文块及逐行片段，`sources.sNNN` 只选一个 ID。代码按原偏移保留精确引文、所选片段和完整块；重复原句可以靠不同位置区分，多行语义仍可选完整块。最多 512 个选择，超限明确要求缩小窗口，不静默丢源。模型仍决定哪个片段支持哪个角色；检查仅证明文本真实，绝不自动证明语义蕴含。失败批次和已中断请求全部保留，当前版本是否通过以阶段验收结果为准。

v48 的接口候选仍遗漏源文定义的空列表结束路径；编译通过被语义审阅否决。v49 在同一次规划响应中增加 `source_scan`：对 `sourceScanFragments` 中每个原文行/块填写角色和含义，然后再写完整 `procedure` 和 `program`。角色、解释和程序均由模型生成，代码只检查已提供的片段有没有漏答，并在 `program-draft.json.sourceScan` 绑定原引文。原始完整页仍提供，短行和上下文不从源包删除。此核对不是独立评审，更不证明每条规则已进入图；目的在于避免“简短摘要”界面诱发遗漏，并让遗漏位置可审查。

v49 的核对与清单都写对了空列表要求，程序仍漏掉，因而明确定位为**清单→程序**偏移。v50 将数组长度比较单独暴露为 `if_length_equal`，避免与标量字段选择混在一个 `kind` 中。它是原数组长度能力的语法前端，不是新 Runtime 能力；原内部 `if_equal + length` 继续支持，两种表示生成相同的受限程序。不能把“图缺检查时会被 Runtime 阻断”算成源语义已实现。

v50 出现数组与字符串 `"[]"` 比较以及 `"false"` / `"0"` 类型混淆。v51 将**原 Schema 类型约束前移到模型语法**：从所有调用方/工具 Schema 枚举标量指针及对应 JSON 比较类型，数组指针只进入长度比较。所有类型兼容候选保留；不自动选择观察来源、业务字段、比较值、条件极性或分支。重名指针跨不同输出的歧义仍由原编译器按实际观察来源校验，类型正确仍不是业务正确。

v51 消除了上述类型混淆，但仍遗漏空列表分支。v52 保留所有类型约束，把模型比较统一回 `if_equal` 的 `field/length` 两种类型化操作数，减少不同分支语句的表示差异。没有添加源特定分支模板或预置比较值；原参数/源数据、原编译器、固定退出门禁不变。

v52 仍遗漏已写入清单的分支。v53 试验了一次固定同模型原文对齐复核，但首份接口候选的前后程序相同，仍缺少空列表判断；新增调用没有修正该错误。此复核**未保留为当前流程**，失败、额外成本及 `source-refinement.json` 留在 v53 原版本快照和制品中，不重写为成功，也不称为独立审阅。

v54 修复了实际上下文缺口：原 `schema_location` 已给出路径是否保证存在，但规划导航没有传递此标记。现在每条导航包含 `sourcePathGuaranteedPresent`，数组另保留原 `minItems` / `maxItems`。数组元素内部字段必填，不代表元素本身存在；允许空数组与必填数组属性也不矛盾。模型仍须从原文选择是否判断长度、比较值和两侧流程；代码不推断或补入业务分支。该元信息修复是否带来完整构造改善，以冻结批次的原文审阅和本地路径结果为准。

v54 首份接口候选正确保留空列表及后续判断，原引擎 9/9 合成路径通过；但引用候选仍有两处来源对齐错误（读取绑定到入口，禁止脚本要求绑定到子文档），不能通过同版六份验收。v55 在源槽输出中保留 `basis` + `evidence_id`，并在每个 Schema 属性旁呈现原槽角色/程序内容，避免输出只剩不透明的 `sNNN → bNNNN` 映射。全部原出处选择保留，依据不能修改程序、授予权限或代替语义审阅；未知或缺依据仍失败。改善是否成立须继续实测，不能以解释文字存在代替正确性。

v55 仍把三项父文档限制绑定到子文档，且对应解释中出现条件极性错误；全量 2491 项回归通过不能使这个候选成为语义成功。v56 改为**构造时携带来源**：每条程序语句及单项交接职责选择原文 ID，机械生成带来源的受限程序及 `inlineSourceMap`，直接进入原计划/参数编译。旧的事后模型出处绑定只保留为历史实验接口，不再出现在新语义构造链中。

全部原文块/行选择保持可选；Schema 共享 `ProgramSourceId` 定义，正文仅在 `sourceBlocks` / `sourceScanFragments` 中出现，选择表复用相同 ID，避免重复拷贝耗尽预算。没有提高上下文或输出上限。机械检查不判断所选原文是否蕴含业务逻辑；选错的真实出处仍须被审阅否决。`source_id` 不是权限、运行时事实或模型置信度。

v56 的父文档限制和读取出处得到保留，但条件节点把出处选到了分支动作行；其中一条终止解释还把 available 扩大成 healthy。因此 v57 将条件 `source_id` 放在操作数/比较值之后、两个分支之前，明确其指向谓词定义，而非子节点动作。终止解释只描述读取路径完成，不扩大成全局健康或外部工作已验证。该变化不提供正确来源答案；历史候选不改写。

v57 首份引用候选通过原文审阅与 8/8 本地路径检查，但接口输入在 freeze 阶段因重复位置说明超预算而停止，未调用模型。v58 的逐行扫描视图仅保留原始 `id`、`block_id` 和完整 `text`；文件/父块位置已在 `sourceBlocks` 中，精确行坐标仍在本地源映射和审阅制品中。三类输入分别为 40,809 / 37,610 / 31,953 字节调度代理量，均在原 40,960 上限内；这不是 tokenizer 认证。正文、片段选择、条件、输出预算和验收标准没有改变。

v58 完成六份首次构造：引用两份来源/参数正确，各通过 8/8 本地路径；接口两份把原文 ID 当数据变量并添加不可达授权任务，审批两份把字段名当观察变量，均被拒绝。v59 将模型数据变量限定在八个观察槽及 input，仍由原作用域检查拒绝未定义或错误分支的引用；不替模型选择正确依赖。未来要求、真实缺口和任务外职责先于程序单独声明，避免把宿主授权追加成业务步骤。共享 Schema 定义抵消重复说明，三类输入仍保持原预算；是否提高语义通过需实际新批次确认。

### 如何使用和定位

使用已对齐的源包、任务、输入 Schema、工具 Catalog 和原始读取合同组成输入，详见[先规划后填参](PLAN-FIRST-AUTHORING.md)。输出目录必须新建：

```bash
python -m evaluation.source_ledger freeze /tmp/my-semantic-run \
  --inputs /absolute/path/input.json --profile plan_first --semantic-plan
python -m evaluation.source_ledger run /tmp/my-semantic-run \
  --max-new-calls 6 --report-dir /tmp/my-semantic-report
```

可显式增加 `--reasoning` 使用同一个 9B 的推理模式及 8192 输出预算；默认非推理为 4096。必须报告该资源变化，不能把两者耗时当成同资源对照。未完成槽位可在同一目录继续；失败检查点不能重试或覆盖。`--semantic-plan` 与后置 `--account-duties` 互斥，避免两条未经验证的语义解释自动混用。

若只需要为参数绑定分配推理预算，可改用 `--argument-reasoning`，不要与 `--reasoning` 同时提供。此选项仍是研究配置，是否可进入下一阶段以实际冻结验收报告为准。

| 要检查的问题 | 制品 |
|---|---|
| 模型实际看了什么、生成了什么 | 每轮 `request.json`、`choice.json`、`receipt.json` |
| 历史 v53 固定复核为何未保留 | v53 制品的 `source-refinement.json`：前后程序相同，空列表判断仍遗漏；原始草稿及成本不覆盖 |
| 语句如何变成可读程序、来源是否准确传递 | `program-draft.json` 的 `choice.program`、`controlFlowNormalization`、`renderedProgram`、`inlineSourceMap`；`program-source-bindings.json` 绑定源映射和零次额外模型重绑定 |
| 读取是否必要、分支是否正确 | `prepared-plan.json` 的原始 `choice.program`、`procedure`、解析后的 `tree` 和 `observationPaths` |
| 参数是否来自正确输入或前置读取 | `planned-arguments.json`、`argument-origin-check.json` |
| 观察名称如何变成精确参数来源 | `argument-reference-lowering.json` 保留原参数、解析后的参数与名称→读取路径映射 |
| 模型是否只选择参数来源、对象结构从何而来 | `parameter-slot-packet.json`、`parameter-slot-lowering.json` 保留原 Schema、全部可选来源、原回复及还原后的表达式 |
| L0 如何对应源文和计划 | `tree.json`、`plan-to-tree.json`、`compilation.json` 的 `origins` |
| 哪个路径仍需要 L1 | `handoffs.json`、`remaining.json` |
| 哪些未来执行要求没有满足 | `execution-requirements.json` |

### 边界

- 这是有界只读区域，不是任意 Anthropic Skill 的完整编译器。上限为 8 个读取、32 条语句、8 层规划嵌套；规划字段导航最多 128 项 / 6 层，超限明确停止，不静默截断。数组导航仅示例索引 0，不证明元素存在；参数阶段原导航上限为 64 项并显式标记截断。
- 支持标量相等、类型明确的数组长度相等和嵌套分支；不新增范围比较、循环、分页、任意脚本或写操作。不能表达的必要检查，必须在依赖操作之前交给 L1，不能省略。
- 引用分页和带出处的笔记沿用原账本；“已经显示原文”不等于“模型已保留全部语义”。
- 交接职责是待审说明，不是已经执行的 L1、输出脱敏器或审批系统。实际读取仍经过宿主授权、资源和证据时效检查。
- `evaluation.stage1_validation` 只在开发者明确给定已审 Tree 摘要后，用合成返回值验证原共享引擎；它会重新编译原 Tree 检查制品一致性。期望行为不进入模型输入；此测试不能解锁大规模 Runtime A/B。

## English

Current implementation is v65 / closed syntax v3. See the [representation and navigation repair](STAGE-2-REPRESENTATION-REPAIR.md) for dynamic RHS references, caller/output path partitioning, lossless line-first paging and provisional-gap retrieval. The following v62/v63 descriptions retain historical experimental context, not the current version or a new success claim.

Stage 1's historical v62/9B baseline yields six reviewed read regions and 50 local paths, now committed at 43a2b76. [Stage 2](STAGE-2-PUBLIC-TRANSFER.md) uses a separately frozen v63 revision with shared fragment schemas and 2,048-byte pages; full originals, unread pages and unchanged budgets remain. See [results and limits](STAGE-1-RESULTS.md); three known developer procedures are not public generalization evidence.

This opt-in frontend generates an original-source scan, business outline and closed control tree with per-node source IDs, mechanically preserves those origins, fills arguments and compiles through the existing Flow engine. Provenance preservation cannot prove entailment. Default DSH routing and activation are unchanged. See [Stage 1 exit criteria](STAGE-1-EXIT.md). Historical trials remain below; closed syntax v2 and inline provenance still avoid a second model call to reconstruct sources. Budget admission and schema validity are not semantic acceptance.

An ordered, source-anchored `procedure` interpretation precedes graph construction, separating source misunderstanding from outline-to-graph loss during review. It is not independent Gold or an automatic completeness proof. Named observations replace model-authored nested Tree addresses during planning. The compiler resolves names and enforces uniqueness, lexical scope and dominance. Typed schema navigation supplies exact JSON Pointers but neither runtime values nor correct semantic mappings. `array_length` accepts only schema-valid arrays and runs through the existing binding/flow implementation; its binding artifact is versioned v2 while legacy expressions retain v1.

The model's `program` is a closed control tree, not a string or runtime Tree. A read has one next node; if_equal has two child nodes, when_equal/otherwise; complete/handoff have no successor. Empty branches, implicit completion and terminal suffixes are syntactically forbidden. Original-schema field/length types constrain syntax, but the model still selects the business source, value, polarity and both branches. The closed lowerer maps each node exactly once to the existing statement representation, with original/normalized positions in controlSyntaxLowering. It does not discard work or insert guards. Existing source, scope and type checks remain; closed syntax is not semantic completeness. Internal statement arrays and if_length_equal remain available for legacy mechanics, not the current model wire. The source_program parser only uses ast.parse and a whitelist, never Python compile/eval/exec or source scripts. No arbitrary calls, imports, loops or effects are added.

Direct strings previously mixed arguments with planning; v35 regex decoding produced malformed JSON. Flat indentation and later branch arrays still allowed invalid continuations, so v60 uses a closed tree with ordinary JSON object constraints, not correct-source templates. Inspect original choice, controlSyntaxLowering, renderedProgram and inlineSourceMap in program-draft.json. Limits remain 32 nodes, eight reads and eight conditional levels.

v59 completed all six constructions: approval and reference each passed source/argument review twice and 32/32 local paths overall. Both wiring candidates were rejected for unreachable host-authorization work and a counter-read origin bound to the result-processing line. The fixed gate still failed. v60 preserves those negatives, separates host requirements from the business outline, and anchors read provenance before adapter selection. New real-model validation is required; no success is assumed.

v60's first wiring/approval constructions passed source review and 9/9, 8/8 local paths. Reference compiled with equivalent Boolean behavior but inverted ready=false into ready=true and cited the otherwise action rather than the defining predicate, so it was rejected. The 2522 passing tests plus 81 subtests do not override that rejection. The in-progress second wiring request remains interrupted, not retried. v61 asks for original explicit comparison values/order and the defining predicate behind otherwise. No fixture-specific answer or automatic old-graph repair is supplied.

v61's prompt-only adjustment did not fix provenance and additionally expanded available into globally healthy, so only its first reference construction was completed. v62 anchors source_id before every node's operands/action. Closed syntax v2 disallows free-text terminal explanation; the compiler supplies fixed control-boundary labels that assert neither broader business outcomes nor completion of remaining work. Source-required explanation, diagnosis and recommendations remain sourced L1 duties, never replaced by status labels. controlSyntaxLowering.terminalLabels exposes the templates. Old candidates are not rewritten. A separate whole-repository Ruff check found 224 diagnostics in 58 files identical to HEAD; changed files pass, not a whole-repository clean claim.

v37 exposed a dominance/lexical-scope mismatch: if one arm explicitly terminates and the other observes B, B dominates subsequent reachable work. A bounded continuation-passing normalization now moves subsequent statements only into the unique continuing arm. It never infers a business condition, duplicates/removes calls, drops unreachable work or leaks branch-local values when both arms continue. `controlFlowNormalization` records moves and original/normalized statement pointers. Original lexical checks and the shared Runtime dominator qualification still apply; this is control-flow lowering, not answer-driven plan repair.

v42 aligns with the older frontend's redundant-terminal rule: a single trailing completion without duties may be removed only after every incoming path already explicitly terminates. The original statement and location remain in `controlFlowNormalization.redundantTerminals`; the raw response is unchanged. Unreachable reads, handoffs, other work or multiple trailing nodes still reject. No pending work is converted into completion and no business condition is invented.

Source binding retains the prior checklist with original quotations as unverified navigation, not Gold; a parent's link is not the direct source of a referenced predicate. Semantic argument construction reuses frozen observation names. Only dominating names are admitted and mechanically lowered to exact read pointers, with original/lowered expressions and mappings in `argument-reference-lowering.json`. Template strings remain inert literals and must pass original literal-origin checks. Legacy frontends retain path-based bindings.

Named field aliases are immutable lazy references, not additional observations or eager Python computation. Their composed pointers, source types and lexical scope are checked and retained. Business gaps remain blocking; future host policy belongs only in unsatisfied execution requirements. Planning and source-binding blocks have separate namespaces. Up to two directly linked, unambiguous, already sealed short prose references are initially disclosed as whole pages within budget. Large/missing/ambiguous/script references remain explicit navigation, never downloaded or executed. This frontend explicitly records presence/repeat penalties: 0/1 without reasoning and 1.5/1 with explicit reasoning. The latter restores the observed model presence default after v39 exhausted 8192 tokens on repeated reasoning without final content. A penalty-only non-reasoning comparison did not fix semantics; configuration changes alone do not prove accuracy improvements. Default direct retains its original configuration.

v42's approval region compiled and passed 8/8 local path checks, but the referenced multi-step planning request timed out after 360 seconds. That configuration did not pass the exit gate. v43 retains the typed frontend and tests the existing non-reasoning configuration anew; old failures/requests are neither overwritten nor retried. Ollama's saved input/output counts lack a separate reasoning-token breakdown and are not a verified total reasoning-cost measurement.

v43 still confused field selection with another observation, shortened nested field paths and emitted a dynamic argument as a template literal. v44 changes neither source semantics nor acceptance: its planning prompt uses only the JSON statement language, without mixed Python examples. Parameter prompts contain the current target tool and all dominating observations' names, tools, original sources and full type navigation, rather than asking the model to reinterpret the already frozen annotated program. The complete plan remains in artifacts; original source pages and the current read's source remain available. Equal parameter types do not prove matching business roles; templates are never executed as references.

The v44 parameter-context reduction was **not retained**: it still produced object-to-scalar references/invented literals and regressed approval binding. v45 restores complete frozen-plan context while retaining concise JSON-only planning. Explicit `--argument-reasoning` enables the same 9B's thinking/8192 output budget only for argument slots; planning/source anchoring remain non-thinking/4096. It requires semantic-plan mode and is mutually exclusive with all-phase `--reasoning`. Options, actual requests and costs are recorded; configuration changes alone cannot establish algorithmic causality. Handoffs must not invent approval/confirmation prerequisites.

v45 binding-only reasoning still exhausted 8192 tokens in 352.816 seconds without a final candidate. v46 adds [schema-derived parameter slots](../evaluation/source_argument_slots.py) for fixed, all-required closed-object shapes. Code derives scalar target paths and object wrappers from the original API schema; the model selects type-compatible reference IDs or explicit source-grounded constants. All same-type sources remain available; there is no name-based business mapping or supplied answer. Complete frozen-plan and original-source context remain.

Selections lower into the original binding expressions and retain original type/scope/literal-origin/compiler checks. Templates stay literals, never executable references. An explicit reference object preserves valid unlisted typed pointers and arbitrary legal array indices, with original presence/type checks. Optional keys, open objects, array parameters and over-budget navigation use the original generic authoring path, without silently dropping fields or capability. The slot route has at most 64 scalar targets/six shape levels; it is not a new arbitrary-Schema compiler. Inspect `parameter-slot-packet.json` and `parameter-slot-lowering.json` for the schema, all alternatives, raw choice and reconstruction.

v46 compiled all three initial families but reference semantics still failed: parent output restrictions were missing and predicates were all assigned to the entry block. v47 separates handoff positive `duties` and applicable `restrictions`, each capped at eight, mechanically retaining up to sixteen existing path-bound metadata duties. No remaining work or restriction is thereby executed. Exact-quote transcription plus a separate block ID then introduced paraphrase/coordinate errors.

v48 selects rather than generates provenance. `frozenProgram.evidenceChoices` enumerates current original blocks and lines without ranking or semantic matching; each source slot chooses one ID. Exact quotations/coordinates and the complete original block are retained mechanically. Duplicate lines retain distinct offsets; full blocks allow multi-line support. More than 512 choices explicitly requires a smaller window, not silent omission. Selecting real text does not prove it entails the program. Failures and interrupted requests remain immutable; readiness requires actual source review and exit evidence.

v48 wiring still omitted the source-defined empty-list completion and was rejected despite compilation. v49 adds `source_scan` before procedure/program in the same planning response: the model supplies roles and meanings for every presented original line/block in `sourceScanFragments`. Code enforces response-key coverage and retains original quotes in `program-draft.json.sourceScan`; it does not supply roles, repairs or business answers. Complete original pages remain visible, including short lines/context. This avoids a lossy short-summary interface and improves localization, but is neither independent review nor proof that each rule reached the graph.

v49 correctly stated the empty-list rule in both scan and procedure but omitted it in program, localizing drift to checklist-to-program. v50 exposes array cardinality as a distinct `if_length_equal` rather than hiding it within a scalar operand kind. It is syntax for the existing length primitive, not a new Runtime feature; the old internal equality-plus-length representation remains equivalent. Runtime blocking on a missing array element cannot be counted as implementing source-defined completion.

v50 compared an array with string `"[]"` and confused booleans/numbers with strings. v51 moves original Schema constraints into model predicate syntax: all original scalar pointers retain corresponding JSON constant types, while array pointers appear only in cardinality comparisons. It supplies no observation choice, correct business field/value/polarity or branch. Same-pointer ambiguity across different outputs remains subject to the original compiler's actual-source/type checks. Type validity is not business correctness.

v51 removed those type mismatches but still omitted the empty-list branch. v52 retains type constraints and unifies model comparisons under if_equal with field/length operands. No source-specific branch template or correct comparison value is supplied. Source/input contracts, the original compiler and fixed exit gate are unchanged.

v52 still omitted a checklist rule. v53 tested one fixed same-model source-alignment pass, but its first wiring program remained identical and still omitted the empty-list check. The extra call did not fix this error. This refinement is **not retained in the current pipeline**. Its original-version snapshot, drafts, source-refinement.json and additional costs remain unchanged; it is neither successful repair nor independent review.

v54 repairs an actual context omission: schema_location already returned whether a path was guaranteed present, but planning navigation dropped that flag. Navigation now includes sourcePathGuaranteedPresent and original array minItems/maxItems. Required item fields do not prove that an element exists; a required array property may still be empty. The model still chooses source-supported checks, values and both branches; code does not insert business guards. Improvement requires source review and local behavior validation of the frozen batch.

The first v54 wiring candidate retained the empty-list and later checks and passed 9/9 synthetic engine paths. The reference candidate still misbound two origins, so the six-run gate failed. v55 preserves a short unverified basis before evidence_id for each source slot, with the original role/program text beside its Schema property. This makes mapping decisions reviewable rather than only opaque slot-to-block IDs. All original alternatives remain; explanations cannot alter the program, grant authority or prove entailment. Actual review, not the presence of an explanation, determines improvement.

v55 still assigned three parent restrictions to child text and misstated one predicate in its rationale; 2491 passing regression tests do not turn it into semantic success. v56 instead carries source_id on every constructed statement and individual duty/restriction. The [inline frontend](../evaluation/source_inline_program.py) mechanically emits annotated inert syntax and inlineSourceMap, preserving exact fragments, parent blocks and original/normalized node locations. It enters original planning/binding directly; post-hoc model source binding remains only a historical experimental interface.

All original choices remain. A shared ProgramSourceId Schema definition and shared sourceBlocks/sourceScanFragments text avoid duplication without increasing context/output limits or removing original evidence. Mechanical provenance still cannot establish entailment; a real but wrong source must be rejected by review. A source_id is not permission, observed fact or confidence.

v56 retained parent restrictions and read origins but cited branch actions for predicate nodes; one terminal explanation also broadened available into healthy. v57 puts predicate source_id immediately after operand/comparison and before child branches, explicitly denoting the predicate definition rather than a child action. Terminal explanations may describe only read-path completion, not stronger global health or verified external work. This supplies no correct source answer and rewrites no old candidate.

The first v57 reference candidate passed source review and 8/8 local paths, but wiring freeze stopped before any model call because redundant location metadata exceeded the unchanged input budget. v58 presents each scan fragment as original id, block_id and complete text, sharing parent location metadata with sourceBlocks; exact line coordinates remain in local provenance artifacts. Wiring/reference/approval preflights use 40,809/37,610/31,953 bytes of the unchanged 40,960 scheduling proxy, not an attested tokenizer count. Original text/choices, business conditions, output budget and exit criteria are unchanged.

v58 completed six constructions: both reference candidates were source/argument-correct and separately passed 8/8 local paths; wiring confused source IDs with observations and added unreachable authorization work, while approval used field names as observations. All four were rejected. v59 provides eight distinct obs0–obs7 slots plus input instead of open-ended model variable names. The original scope checker still rejects undefined/sibling references; no correct dependency is selected by code. Future requirements, true gaps and outside duties precede program construction. Shared schema definitions keep original budgets. Actual new-batch review, not these restrictions alone, determines improvement.

Each non-completed terminal carries source-bound, conditional remaining duties. Source-specific business prerequisites stay in the graph; future authorization/consent/scope/age requirements remain separate and explicitly unsatisfied. Unreachable terminals with duties are rejected rather than discarded. Successful compilation still requires semantic review: it cannot rule out duplicate calls, missing predicates or incorrect interpretations.

Use the commands above with fresh directories. Optional `--reasoning` uses the same 9B model with an explicit 8192-token output budget instead of the default 4096; disclose the resource difference. Resume only unfinished slots, never failed checkpoints. Inline semantic planning and post-hoc duty accounting cannot be combined.

Alternatively, opt into `--argument-reasoning` for binding-only reasoning, never together with `--reasoning`. This is a research configuration; readiness depends on the frozen exit evidence, not the existence of the option.

Inspect raw requests/choices, the immutable plan and symbol table, per-read origins, compiled Tree/source mappings, path-bound handoffs and unsatisfied execution requirements. Limits remain eight reads, thirty-two statements and eight planning levels; navigation is bounded and array index zero is not an existence proof. General thresholds, loops, pagination, scripts and effects are not added. Unsupported necessary checks must stop before dependent calls. Handoff text does not execute L1 or enforce output redaction.

The developer-only behavior validator requires an explicit reviewed Tree digest, recompiles it against original contracts and uses synthetic observations with the existing engine. Its expected traces never enter model input. Passing these checks is not unseen generalization, production accuracy or admission to large Runtime A/B.
