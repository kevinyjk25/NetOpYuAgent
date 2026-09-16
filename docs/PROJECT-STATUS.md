# EnsuredSkill 项目进展 / Project Status

## 中文

### 2026-09-16 当前：已授权实施，R0 测量部分完成，0 模型调用

用户已授权实施[考核重置与有限收敛方案](EVALUATION-RESET-20260916.md)，当前为 **R0 measurement partial implemented**。`bounded_budget/scoring/pilot/execution/probe`已提供持久预算、独立评分、输入准备、脚本式计量和本地探针；仍分开转译保真、Runtime 约束和真实 Agent 整体收益。旧公开 A/B 的预转译与漏计尾部成本不能作为冷启动收益证据；旧数值不改，原语义阶段仍为 `paused_unmet`。

153 项定向测试通过；[可入 Git 测量摘要](benchmarks/bounded-pilot-r0-summary.json)记录 12 类／24 个评分 fixtures、10 项预算检查、3 项既有模拟网关机制检查，模型调用为 0。评分回执／语义判断／用量为合成 fixtures；模拟网关检查复用既有机制。这不是完整 36 项机制门槛、自动 Effect 桥接、真实 DSH 冷启动实验或 Agent 收益证明。全量首轮 3,434 通过＋81 子测试，另有 18 项受本地 socket／Docker 访问权限影响；授权环境复核这 18 项全部通过，原报告保留，不冒充单次全量全绿。新增 9 个 Python 文件 Ruff 与 diff 检查通过。

剩余 R0：真实 DSH 全调用计量与可信预先 token 计数、物理 Provider 重置／隔离接入、预封存独立标签及 12 任务样本、真实 trace 采集。CLI 仅 `check/prepare/inspect/score`，没有 `run` 命令；R0 未完成，不进入 R1／R2。保留 9B、双驱动、总预算、正式泛化门禁和写边界；无模型调用、提交或推送。下文“下一步”均为历史记录，不自动续开修复包。

### 历史：结构机制完成，任务验收退步，停止同例模型试跑

[结构约束/有界工件修订](SCHEMA-ARTIFACT-CONVERGENCE.md)完成：完整author Schema解码及独立校验；窄范围时间/列检查；宿主可选一次代码体修订，无新读取/效果或Runtime重跑。**真实DSH＋9B同3任务结构准入2/3→3/3，但查询清单2/2→1/2、完整任务1/3→0/3。** 9B原样提交唯一修订；另一查询漏读清单；Mesh编造后续资源。不能宣告收敛或以单元测试抵消负结果。

15次模型调用，116,059/3,444输入/输出token，p50/p95=176.19/278.83秒；5/5实际读取获准，无观察到的效果/重放。148项定向、8项文档/边界、124文件Ruff、**3,324项全量＋81子测试**通过，源码/模型全程冻结。详见[可入Git摘要](benchmarks/schema-artifact-summary.json)。没有覆盖旧结果、改判据、添加expected reads或追加模型重试。

当前停在机制设计边界：现有author profile主要为read前段＋一个保留原任务的LLM节点，未把动态证据和可验证业务约束充分落实为合同。编译模型的虚假missing_host仍进入Runtime推理但非事实/授权；其因果影响未作消融。下一步需明确证据义务/合同覆盖与来源绑定，保留未知职责，不能再只加提示或通用自审。原阶段出口和新来源门禁不降低；实验修订默认关闭，不扩展正式A/B，不提交/推送或重启UI。

### 历史：上下文隔离完成，证据获取改善，语义阶段仍开放

[隔离编译实现及结果](ISOLATED-COMPILER.md)：v4可选`compilerMode=isolated`，独立编译请求只接收原Skill/任务/输入Schema/工具合同；运行Agent没有submit工具，后端拒绝AST注入。原校验器、ACL、证据门禁、补读预算和终态不变，不加写权。编译确定拒绝进入原生受限fallback；未知结果停止，无纠错循环。

唯一冻结DSH＋9B批次是2个已知Skill/仓库、3个原任务，0未见Skill。3/3正常保真交付，6/6实际读取授权，0观察到的越权/重放/写入；前段2/3通过、原判据9/12、完整任务1/3。两个IRQL的实际清单获取从0/2变为2/2，但仍0/2完整查询：两例都使用含上界的between，第一例另投影聚合后不存在的EnvTime。第二例caller附带非法origin/quote被拒、改走native。Mesh原判据通过，条件式重读静态快照和缺少服务流名称仍列为范围局限，不改评分。

15次真实调用＝DSH10＋编译3＋Runtime2；110,926输入／3,245输出token，p50/p95=183.48/190.02秒。仅对相同两个IRQL对比，调用13→10、输入150,818→79,874，但中位时延175.34→183.33秒；不声称整体提速或因果改进。122份制品、80份归档/当前源码及模型摘要一致，运行中/结束后没有修改源码或重跑模型。100项定向、**3,312项全量＋81子测试（237.77秒）**、121个Python文件Ruff和文档/边界/diff检查通过，完整成本见[机器摘要](benchmarks/isolated-compiler-summary.json)。

本包隔离机制完成，Gate 1仍开放。下一项只针对完整编译Schema约束和可验证工件义务，不扩展通用自审/生产架构，不靠同例提示词循环刷分；这两项尚未实施。未重测的CAPA问题不算关闭，新来源/跨cohort门禁不降低。不提交/推送、重启UI或扩大A/B。以下为历史记录。

### 历史：参数/证据机制包完成，唯一9B回归仍未通过

[有限重构实现及结果](CALL-PROTOCOL-EVIDENCE.md)：v4使用真实工具参数Schema，宿主可声明原ACL内的必需读取；缺证据时不生成/接纳答案，支持主动未完成退出。实际DSH脚本探针2/2通过，0真实模型调用。没有自动从自由文本推导必需清单，也没有新增权限或补读额度。

仅对原两份IRQL任务做一次9B回归，未添加requiredReads：2/2正常原样交付，**0/2完整任务、0/2读取前段编译成功**。两例在编译计划中重复填入实参字符串，原生fallback后只读索引；函数清单获取0/2。第二例一次过早读取被生命周期拦截。2次实际读取均授权，0观察到的写入/越权执行/重放；不是参数准确率或生产安全概率。4/8原判据含局部查询评分解释，不给校准准确率。

13次真实调用均为DSH，Runtime生成0；150,818输入／2,992输出token，p50/p95=175.34/203.95秒，n=2非性能对照。模型后修正了一处fallback禁止draft的旧提示与主动退出的冲突，并做配置拒绝/状态标签修正；不修改冻结数据或结果，不再跑9B。最终源码与模型归档有一处已披露差异，最终门禁接线探针2/2通过。85项定向、**3,295项全量＋81子测试**、119个Python文件Ruff及文档/diff检查通过；首次测试中的源码变更保护失败与固定代码后的重跑均见[机器摘要](benchmarks/call-protocol-evidence-summary.json)。

**本包机制已完成，Gate 1仍开放。** 下一项应先设计编译上下文与具体调用Agent的职责隔离，并分别衡量绑定错误、实际证据获取和完整任务完成，不增加通用自审器或用更多同例测试刷分。未实施该新编排，不扩大A/B，不提交/推送或重启UI。以下为历史记录。

### 历史：两轮结束，机制缺口关闭，语义未通过并停止试跑

[两轮结果与停止决策](CONVERGENCE-CLOSURE.md)：评分器授权、有效交付与原样终态条件已修复，未改Runtime/Prompt。固定DSH＋9B的6任务（3历史＋3首次变体，3已知Skill/仓库，0未见Skill）均正常结束，8/8实际读取授权匹配，0副作用/越权执行/重放。3自动前段＋Runtime生成，3原生fallback。38真实调用（DSH35＋Runtime3），350,293输入／7,057输出token；p50/p95=155.59/276.89秒，无性能提升推断。

原公式15/24判据、2/6任务；两个Mesh满足冻结判据，CAPA漏答/格式失败，两个IRQL未成功获取必需清单。CAPA旧c2有预先歧义，新CAPA c1又暴露ID要求超出显式任务；精确原文锚点不是对齐证明，不给校准总体准确率。旧报告不改，原门禁仍开放。

**本包已硬停止，不自动第三轮。** 关键根因已定位：编译caller绑定对象混入实际read字符串参数，外层read仅公开object；Schema拒绝只给错误类别；缺证据候选仍可交付；原生L1输出仍会漏答。建议下一包只重构编译/调用协议和必要的证据/工件准入，需确认后再实施，不扩大Runtime整体架构或通用自审。

196份制品、79份归档/当前源码及模型摘要核验一致；94项定向、**3,280项全量＋81子测试（226.40秒）**、119个变更/新增Python文件Ruff及diff检查通过，见[机器摘要](benchmarks/convergence-closure-summary.json)。模型与pytest不重叠。没有提交/推送、重启UI或正式A/B。下方为历史状态。

### 历史：快照语义和重复补读机制修复完成

[本轮报告](SNAPSHOT-READ-SEMANTICS.md)：v4本地宿主明确数据是不可刷新的会话快照，精确重复补读不再进Provider或生成新证据，不影响不同资源，不放宽次数/权限/冻结限制。该保护不扩展到实时设备或严格图内部调度。

新增1个synthetic Skill／2个任务，真实DSH＋9B均正常结束：恢复任务正确识别旧快照不能证明恢复；库存任务读取3个不同资源并完成18→13的比较。4次实际读取、10次模型调用（DSH8＋Runtime2），52,874输入／1,068输出token，无未知用量。端到端45.53/65.17秒，n=2不是性能对照。

保留原冻结计数5/6判据、1/2 fixture taskPassed：电路c1额外要求任务未明确要求的丢包百分比，已标记Oracle/任务不一致，不能据此称模型失败率50%。未来生成器v2已明确这项要求，未跑新版批次，旧输入/输出/评分不改。库存满足所有判据；本轮未重跑Mesh/CAPA/IRQL，也不证明广泛泛化。

9项新增回归、**3,263项全量＋81子测试（244.33秒）**通过；116个变更/新增Python文件Ruff通过。75份制品和78份归档/当前源码一致。未提交/推送、重启UI或扩大A/B。[机器摘要](benchmarks/snapshot-read-summary.json)。下方为历史状态。

### 历史：移除交付语义压缩，单点改善但仍未收敛

[任务直达模式](TASK-FIRST-DELIVERY.md)已接入：显式宿主v4保留完整任务，submit使用delivery=null；Runtime draft与native deliver都输出同一非空answer信封，不让模型挑选的kind/source_ref固化业务职责，不自动加标题。权限、精确资源读取、证据冻结、一次生成、静态工件和终态控制不变；answer始终是未验证L1，不是L0。

一次冻结Mesh快照两臂诊断已结束，不再重跑：真实9B旧结构回复与历史逐字一致；任务直达新增具体trace请求，但仍建议重读旧文件，业务流也未完整写出。两调用共14,107输入／668输出token；单调用40.76/33.81秒，不是总体性能改善。没有重新执行DSH/Runtime，旧0/2任务评分不变；CAPA/IRQL未重测。实际DSH＋脚本化模型的两条终态接线2/2通过，分别计量。[机器摘要](benchmarks/task-first-delivery-summary.json)。

25项新增回归、206项定向（含文档）、**3,254项全量＋81子测试（245.49秒）**、114个变更/新增项目Python文件Ruff和diff检查通过。10份模型诊断制品/71份归档源码、66份接线制品/76份归档源码核验通过。语义阶段和正式泛化门禁仍开放；不扩大A/B、不降低旧判据、不新增自审、不提交或重启UI。下方为历史进展。

### 历史：终态机制缺陷关闭，语义问题仍开放

[宿主终态修复及接入](HOST-TERMINAL-DELIVERY.md)已完成：可选模式通过真实DSH公开`concludeTurn()`结束轮次，同轮重复混合操作在bridge前拒绝；UI工具卡/专用headless前端显示宿主原文。实际DSH＋脚本化模型两条路径2/2通过，每条6次协议请求、1次实际读取、无第七次请求，stdout与宿主结果逐字一致。首次前端接线失败0/2完整保留。**本轮0次真实模型调用，不是9B语义提升。**

新增`semanticCoverage=not_assessed`明确模型所选kind只是呈现提案，填满结构不证明完整理解。这是边界澄清而非语义修复。最新真实9B结果仍为结构2/2、业务判据4/6、完整任务0/2；CAPA结论、Mesh下一证据、未重测的IRQL问题仍开放。下一项只聚焦原任务职责、证据使用与输出遗漏的因果修复；不加自审层、不用同例反复生成刷分、不改变四道研究门禁。

本包验收：24项Node、178项定向Python、**3,229项全量＋81子测试（231.28秒）**、111个变更/新增项目Python文件Ruff通过。两次探针各66份制品、76份归档源码核验通过，后次与当前实现一致。[机器摘要](benchmarks/host-terminal-delivery-summary.json)。默认UI未修改/重启，未提交或推送；下面所有“下一步”均为历史记录。

### 历史：单选协议修复包结束，停止同例重跑

[唯一冻结验收与具体停止点](DELIVERY-SINGLE-CHOICE.md)：显式宿主v3／交付v4消除双重状态表示，新增独立hostResult及宿主/Agent并列诊断视图。真实DSH＋9B结构准入2/2、原业务判据4/6、完整任务0/2。CAPA为2/3（缺明确保持开放结论），Mesh为2/3（下一证据建议不合格）；业务来源类型误配仍存在。历史IRQL未重测，仍开放。**接口修复包完成≠Gate 1通过。**

14份完整模型回复，176,566输入／3,183输出token；1图、3授权读取，无操作重放或写入。p50/p95 241.80/366.68秒，观察上慢于前批，n=2不能作因果归因。Mesh在Runtime准入后重复调用一次deliver，宿主阻止执行重放，但没有约束DSH终态选工具。这是明确生命周期缺口，不再靠提醒文字和同例试跑处理。[机器摘要](benchmarks/delivery-single-choice-summary.json)。

已履行“一个修复包＋一次冻结验收”的停止条件，不自动追加同例模型批次。建议下一设计分开处理：宿主终态直接交付、模型所选职责只作提案/呈现而非语义完备证明。尚未实施DSH生命周期接管、尚未修改研究出口；需要改变出口时明确决策。后续四道门依然包含当前Gate 1，然后新来源小批、跨cohort泛化、同DSH原生/auto Runtime A/B。未提交/推送、扩样或重启UI。以下记录均为历史。

### 历史：引用/紧凑协议修复，Gate 1未通过

[本轮协议与结果](DELIVERY-REFERENCE-CONTENT.md)：宿主提供可恢复原文的短引用，候选直接填内容，减少状态包装；本地Runtime使用输出Schema约束解码且独立校验。首批暴露静态目录枚举过大（CAPA 1,475引用、37,132字节Schema），0/2任务；随后静态Schema缩为802字节、来源成员校验不变，CAPA在72.50秒恢复原生候选准入。旧失败/超时不覆盖，既有宿主配置不自动迁移。

最新批仍0/2完整任务、3/6原判据：CAPA 1/3，缺明确开放结论和下一证据；Mesh 2/3，基础Schema通过但提供内容与unresolved字符串None冲突，宿主拒绝，原生最终文字仍错误称完成。1图、3授权读取、13模型调用，154,176输入／3,690输出token；p50/p95 166.26/250.65秒，n=2非因果性能。前批17份完整回复及1个未知用量步骤单列。[机器摘要](benchmarks/delivery-reference-content-summary.json)。

下一包：把“已提供/无法提供”做成单一互斥表示，避免跨字段状态同步；修复输出职责/来源类型误配及宿主状态与最终文字偏移。不清除None字符串刷通过，不追加同例提示词循环。仍依次通过Gate 1主链可用性、Gate 2冻结新来源、Gate 3跨cohort、Gate 4原生DSH/auto Runtime A/B；本轮不扩样、不提交/推送、不重启UI。工程与取证结果见本轮报告。以下为历史记录。

### 历史：wire-state主链修复

本轮完成显式JSON文本入口、固定源文要求与宿主读取状态分离、实际回执绑定。真实DSH＋9B两例：JSON解析2/2，类型化候选准入0/2，完整任务0/2、原内容判据2/6；不是整体提升。CAPA缺gap字段和明确开放结论；Mesh转译拒绝后fallback，state层级错误，420秒超时。3次授权读取，无Runtime模型生成或写入。19份完整回复、249,800输入／8,249输出token，另1个中断步骤用量未知；p50/p95 311.20／409.15秒，n=2含超时，不用于性能归因。[完整报告](GOVERNED-SESSION.md)／[机器摘要](benchmarks/governed-session-wire-state-summary.json)。

运行后修复缺字段精确诊断及retryAllowed=false终态指引；不自动补字段、移动值、重试或修改旧分。两份原始候选零调用复查仍拒绝，分别定位d1/gap和d1/state。174项定向、3,166项全量＋81子测试、105个变更/新增Python文件Ruff、文档/工具Schema/diff检查通过；63份运行制品及75个归档源文件摘要一致。未提交/推送或重启用户UI。

后续仍有四道门（包含当前）：**Gate 1主链可用性 → Gate 2小批冻结新来源 → Gate 3跨批泛化 → Gate 4同DSH原生/auto Runtime对照**。当前不能跳到批量测试。下一包收敛模型重复抄写结构/引用的协议负担、合同类型解释及候选到最终交付；纯文本结构修订与执行重试必须分离，现有权限和次数没有放宽。停止同例提示词重跑，不增加通用AI自审层。下列条目均是保留的历史版本进展。

### 历史：交付合同与模式分离

已实现真实任务/可见Skill逐字引用的交付合同：artifact、analysis、decision、next_steps 类型化槽，显式未满足项、会话摘要绑定、确定性渲染；不导入评测答案、不执行来源脚本、不赋予写权限。六工具将 Runtime 的 `draft(session_id)` 与原生fallback的 `deliver(session_id,response)` 分开，返回实际状态和 `deliveryAction`；已接纳会话不能用null plan切换路由。

真实9B首批为0/3任务、3/9原判据，包含Mesh反复错误模式调用的420秒超时。保留26份有用量回复及1个中断未完整记量步骤，不声称成本完整。拆分后两例定向复验：Mesh的循环修复，原3项内容判据满足；CAPA为2/3且类型化提交被拒（对象写成字符串），最终文字不能冒充合同准入。13次模型调用，130,104输入／3,563输出；p50/p95 138.01／190.66秒（n=2，非性能A/B）。[可复查结果](GOVERNED-SESSION.md)／[机器摘要](benchmarks/governed-session-delivery-summary.json)。不混算两个不同分母的批次，不改旧分。

| 状态 | 范围 |
|---|---|
| 已实现 | 源文锚定交付合同、四种类型、原生/Runtime共用检查和渲染、分入口及权限/重放边界 |
| 局部修复有真实证据 | Mesh模式循环消除，恢复一次Runtime生成；不是所有任务准确或所有接口通过 |
| 待修复 | 原生提交对象/字符串协议易用性；交付类型误解；固定要求与临时证据缺口混淆；候选到最终回复的内容改写 |
| 仍关闭 | 新来源泛化通过声明、大规模Runtime A/B、生产工程扩展 |

下一步优先收敛上述输入/输出及状态语义，不增加通用AI自审层、不通过放宽JSON类型或任务判据制造通过。先给失败可定位的诊断，再验证独立来源；已知小批修复不能代替至少6 Skill／4仓库／3领域／12任务的冻结门禁。以下“当前/下一步”按各轮历史时间理解。

工程收尾：91定向、3,138全量＋81子测试、103个变更/新增Python文件Ruff、文档/工具Schema/diff检查通过；153份绑定制品摘要一致。未提交/推送，未重启用户UI；历史未改动文件的lint问题不在本轮清理范围。

### 2026-09-15：证据收集先于草稿生成

当前实施改为 [prepare → submit 前段读取 → 受限补读 → draft 冻结/生成](GOVERNED-SESSION.md)。同一原 Runtime 分别执行严格前段与保留 L1 的推理图，不重放读取。补读和冻结共享锁；模型不能提交证据/审批，未知状态不自动重试。适用工件增加零模型调用静态检查，不增加通用语义自审层。确定性编译拒绝且尚未执行时，可改正一份提案；最多两份提案、一次接纳执行，权限合同不放松。原生 fallback 可自行写文本草稿，不可绕过 Runtime 执行操作。

证据先行两批均已结束：完整任务仍1/3，判据6/9→5/9，不能称总体提升。修订版恢复了IRQL的错误参数绑定并在草稿前实际补读，但仍只交付查询步骤说明；CAPA缺明确开放状态和下一证据建议，Mesh完成。最新2自动图、5读取、16真实9B调用，136,754输入／3,982输出，端到端p50/p95 138.67／172.44秒。两份Runtime候选的静态检查均为no_supported_artifact，没有查询校验通过。[逐项证据与剩余根因](GOVERNED-SESSION.md)／[机器摘要](benchmarks/governed-session-evidence-first-summary.json)。旧失败不覆盖，无写入/脚本/真实设备。

本轮机制改动完成，语义阶段未通过。下一步是来源可审查的通用交付物合同，区分完整工件、分析说明、状态和未满足项，避免自由文本替代要求的交付；不注入评测答案、不增加通用AI审阅层、不在本轮继续调同三例提示词、不启动大规模A/B。最终68定向、3,115全量＋81子测试（224.78秒）、3文档检查、变更路径Ruff/JS语法/diff通过；全库Ruff另有34条未改动文件的旧问题，未冒称全部clean。未提交或推送，未重启当前UI。

### 2026-09-15：停止局部自审堆叠，收敛端到端主链

按用户确认的复盘方案，当前主线改为[共享转译＋原 Runtime＋DSH 原生 fallback](GOVERNED-SESSION.md)。`skill_authoring/` 抽出原转译/来源/参数公共实现，产品不导入评测；旧入口兼容引用同一编译器。新增宿主配置的 prepare/submit/inspect/read 四工具，单次提交、同网关限额补读，模型不能提交权限或任务成功评分。转译、执行、任务结果分别记录。

首次原生 DSH 三例已知任务没有完整成功：大 Skill 的入口预算 fallback 没有会话，两个计划把原始路径写进要求绑定表达式的参数槽。已修复真实 fallback 会话及一等工具参数 Schema；保留首次输出，新版本另目录验证。共享层提取产生的三项分页测试兼容回归已修复，不放宽分页/UTF-8 判据。此处不以工程测试数声明语义完成。

旧语义闭环和冻结新来源门禁仍未通过；不重评分，不扩大正式 A/B，不提交/推送。当前只完成新主链的接入/已知开发验收，下一次新来源冻结验收仍须独立进行。

收敛版本验收补记：原生DSH＋9B三例全部结束，1/3完整任务、2/3自动混合图、4次实际读取；12次模型调用，80,505输入／3,502输出。Mesh完成补读后的分析；CAPA保守记局部；IRQL未补读便复述错误草稿。完整逐项理由和证据见[主链报告](GOVERNED-SESSION.md)。全量3,101＋81子测试通过。不启动第三轮同例调参；本轮入口集成完成，泛化阶段没有宣告通过。

### 历史记录（下列“下一步”以各轮当时为准）

2026-09-15 当前：[证据先定位＋只引用观察的备注](SEMANTIC-DUTY-CONTRACT.md)完成两个已知样例的限定真实验证。复用现有驱动新增显式模式：源文逐字保留、引用 ID 先定位、逐谓词/片段判断、备注仅 keep 或摘录观察，不自由扩写。10 调用全部绑定，7 谓词／14 比较可重建，4 条摘录，26,053 输入／1,543 输出，请求 p50/p95 11.19／25.17 秒；不是性能 A/B。严格取证通过，无摘要兼容例外。**CAPA 未批准→未审阅仍被模型误判为支持，第一阶段未通过**。正文与参数均不变，不宣称查询已修好或摘录完成任务。下一步改进源／目标命题独立表示与关系对齐；不靠增加自审次数或降低门禁。不扩样、提交、推送。下方为历史。

本轮 QA：92 定向、3,083 全量＋81 子测试、变更/新增 Python Ruff、文档和 diff 检查通过；22 项新增回归。工程与语义结论分别记录。

2026-09-14 后续：[宿主职责原文＋分步谓词审阅](SEMANTIC-DUTY-CONTRACT.md)完成限定真实检查。取消 LLM 合同重写，保留带类型的逐字任务锚点；谓词先提取再审阅，宿主汇总未知；独立编辑不互相阻断。18 调用（IRQL 8／CAPA 10），95,154 输入／6,240 输出，请求 p50/p95 16.80／54.45 秒，不是可比性能提升。CAPA 去掉无依据“未审阅”，另一备注的整稿误投被拒绝；查询编辑损坏围栏，整稿未发布。审阅仍有无证据肯定、角色解释漂移，**第一阶段未通过**。运行后修复单元结构准入及检查点摘要键冲突；旧坏摘要不改，新增取证视图明确原校验未通过。不重跑、扩样、提交或推送。下方是历史记录，不能混算提升。

本轮最终 QA：158 定向、3,061 全量＋81 子测试，变更/新增项目 Python Ruff、文档与 diff 检查通过。后置结构/摘要修复是工程验证，不改写原始 18 次模型调用。

2026-09-14 当前：[职责合同与工件检查](SEMANTIC-DUTY-CONTRACT.md)已实施并完成限定真实验证。首轮完整流程 0/2；随后仅接续独立未尝试节点，不重跑失败调用。CAPA 去掉“未审阅”无依据断言；IRQL 的区间操作符由 9B 修订、两个日期字面量由独立零调用类型化渲染修正。累计 17 调用（12＋5），126,955 输入／10,192 输出，请求 p50/p95 31.32／77.44 秒；不是 Runtime 性能结论。职责合同仍有来源类型/解释漂移，审阅仍有自相矛盾，**第一阶段未通过，不开新来源或大 A/B**。QA：224 定向、3,013 全量＋81 子测试、变更 Python Ruff、文档和 diff 检查通过。未提交/推送。下方均为历史记录。

2026-09-14 当前：[第一阶段上下文/编辑边界修复](SEMANTIC-CONTEXT-REPAIR.md)已完成本轮实现及 4 例真实 9B 回归，**关键缺陷完全修复 0/4，阶段未通过，停止重复调提示词并等待验证方案确认**。10 调用、61,350 输入／13,553 输出，请求 p50/p95 67.00／149.72 秒；不是 Runtime 性能改善。已实现任务作用域、实际读回执来源、正文/notes 共用 8 单元修订。运行后再修复 notes 整稿误投与全稿标题追加限制，仅有工程/零调用结构验证，未改旧分。162 定向、2,951 全量＋81 子测试及变更 Python Ruff 通过。不扩样、不提交、不推送。下方同日和此前记录均为历史状态。

2026-09-14 最新：[任务见证与证据作用域诊断](SEMANTIC-WITNESS-DIAGNOSTIC.md)结束，**阶段仍未完成**。四例 9B 共 8 调用（34,213 输入／9,851 输出），来源分析均绑定有效，对照 2 有效／2 无效；有 Mesh/Phoenix 局部检出，但查询、证据范围和否定关系仍有错误，不启用自动编辑、不扩样。本轮修复：任务肯定判断需精确交付引用，备注问题不得误投正文；原 notes 内容修订仍待实现。154 项定向、全量 2,934 项＋81 子测试（222.66 秒）、变更 Python Ruff、文档与 diff 检查通过；四例新接口零调用预算检查均在原上限内。实验驱动一次目录保存失败及开发者 AI 两处初评文字更正均保留。以下为历史进展，不是新的阶段通过声明。

2026-09-14：[9B/27B 四例仅审阅对照](SEMANTIC-REVIEW-MODEL-CONTRAST.md)已结束，**本阶段仍未完成**。27B 新发现 Phoenix 的两项建议遗漏，但查询错误、否定扩大和 Mesh 的陈旧缺数据判断仍在；2 份回复通过接口、2 份因缺引用拒绝。共 5 次尝试（首个超时＋4 个完整回复），已知 21,421 输入／8,391 输出 token，超时用量未知。未改 Runtime 默认模型、旧评分或新来源计数。之后已修复编号列表/标题被误拆成独立断言的问题；76 定向、全量 2,901 项＋81 子测试、变更文件 Ruff、文档和 diff 检查通过。实际 Mesh 输入的 3 个孤立编号已消除，正文/来源与所有非空白字符完整保留，零模型调用，不重评分。下一步优先任务/证据作用域与工件验证机制，不以全面换模型替代修复。以下 9 月 11 日的“待授权”文字为历史状态。

当前修复进度（2026-09-11）：第二批冻结验收已经失败封存，**本阶段尚未完成**。新版本已实现入口优先的来源预算、补读状态索引与上一版候选保留、单一整稿编辑区承接未定位问题、无位置支持意见降为未证实；已知失败回归已结束，仍有语义缺陷，等待下一步对照方案确认，不能计为新来源或重评分。原权限闸门、只读引用、章节/范围保护及无新观察保留原稿继续生效。机制、限制与复现见[闭环手册](SEMANTIC-CLOSURE-RUNBOOK.md)和[根因记录](SEMANTIC-CLOSURE-DIAGNOSIS.md)。

最新回归已结束：v9/v10 共 20 次新增 9B 调用，补回部分职责/计数并修复审阅 ID 绑定，但查询误支持、Mesh 忽略已读数据仍未解决。2,891 全量＋81 子测试、77 定向、51 个变更 Python 文件 Ruff 通过，不能替代语义通过。下一步建议仅对 4 个已知失败例用本地 27B 作审阅对照，**待用户同意，尚未运行**；保持 9B 生成、旧判据和全部失败记录，不扩大测试或提交代码。

最新结果：transfer v2 的 6 Skill／6 仓库／6 个开发分类领域／12 任务为 **0 个完整实质任务、3 个正确边界、8 个部分完成、1 个失败**；24/36 判据满足，67 次调用、342,581 输入／52,968 输出 token。没有越权工具调用不代表回答正确；源码供给漏页、已读仍列待办、遗漏无法定位、审阅误支持均有实际证据。[第二批逐例结果](SEMANTIC-CLOSURE-TRANSFER-V2.md)。以下为旧里程碑，不能视为本轮完成声明。

最新结论（2026-09-11）：首次冻结 transfer v1 已结束并封存，**未通过语义闭环出口**。6 Skill／6 仓库／6 个开发分类领域／12 任务，1 完成＋1 正确拒绝＋7 局部＋3 失败；36 判据为 22 满足／13 不满足／1 未知。56 次真实 9B 调用及失败全部保留。[完整结果与成本](SEMANTIC-CLOSURE-TRANSFER-V1.md)。新版本已知开发验证正在检查连续有效前段投影、答复/依据双通道、无损审阅传输及未解决职责保留。旧批次不改分，重新冻结新来源前不能宣布阶段完成。以下为此前里程碑，不是本轮通过声明。

2026-09-11：当前持续修复范围为[三个修复包与一次冻结验收](SEMANTIC-CLOSURE-EXIT.md)，**尚未完成**。最新固定版本的两例已知开发复核保住交接责任/时间、README 实际 API 与安装示例；README 未确立的许可证由宿主窄范围暂缓，不能说成 9B 自行修正。随后冻结 transfer v1，再固定 **6 Skill／6 仓库／6 个开发分类领域／12 任务**；全部输入与判据在模型运行前封存。使用共同的合成只读 export adapter，不是完整厂商接口或整 Skill 工作流评测。首次构图已暴露多余读取来源文档、引用未供给页的失败；不修改该批代码或预期刷通过。总成本明确包含来源审查 1 次、最多 8 个编辑单元及终审 1 次，不宣称同预算改善。[根因轨迹](SEMANTIC-CLOSURE-DIAGNOSIS.md) / [复现与逐项审阅](SEMANTIC-CLOSURE-RUNBOOK.md)。未启动正式 A/B 或提交/推送。

更新：2026-09-10。**阶段 2 双驱动小批开发验证已提交本地 `dev`：`25b08c0`，未推送；正式泛化门禁仍关闭。** 后续[结果合同](SEMANTIC-RESULT-CONTRACT.md)及[草稿审查/有界修订](BOUNDED-DRAFT-REVIEW.md)已完成本轮工程验证，但语义修复尚未完成。阶段 2 冻结 v7 的 10 Skill 中，6 个结构候选、5 个有用读取前段；真实 9B 执行审阅为 **1 个限定请求完成、2 个局部可用、2 个草稿不接受**，不是 3 个完整成功。旧结果、失败、成本与可解释轨迹见[阶段 2 报告](STAGE-2-HYBRID-RESULTS.md)及[固定验收条件](STAGE-2-HYBRID-VALIDATION.md)。

### 已完成

- 草稿审查/有界修订：同图内双向来源审查、一次锚定修改和终审；确定性应用不授予语义或操作权限。4 轮、2 个已知 Skill、18 次 Chat 尝试（17 有用量、1 个 HTTP 400）。最终只成功补入交接稿的 Chen 责任人；Alice/事件开始时间仍漏，9B 27/27 supported 也不代表正确。README 的 10 个编辑超过 8 个上限，拒绝且未终审。[完整结果和失败](BOUNDED-DRAFT-REVIEW.md) / [机器记录](benchmarks/bounded-draft-summary.json)。
- 本轮验收：136 定向、全量 **2699 项＋81 子测试**、14 个变更 Python 文件 Ruff、文档与 diff 检查通过；4 份源码归档、18 份模型回执、11 份审查及 1 份补丁离线精确重算通过。新改动未提交/推送，没有放宽预算、修改默认 DSH 路由或 A/B 基线；不是新增样本或泛化通过。

- 后续结果合同：同一 Runtime 分开图状态、精确观察字段、未验证草稿和未满足职责；模型不能自行删除合同或宣布职责完成。3 个已知 Skill 各两次真实 9B，共 6 调用；最终 Notion 声明职责 2/2，交接 2/3，README 2/4，后两者保持 partial。**这不是 3 个任务成功；README 草稿仍不合格。** 结果映射为宿主手写审查，转译泛化未评分。[原始边界与失败](SEMANTIC-RESULT-CONTRACT.md)。
- 修复 authoring v8 与混合图 qualification 的 JSON 对象顺序敏感；旧 v7 材料保留，新版本完整比对 Flow 后重新批准。两个零模型失败预检保留，不绕过摘要检查。
- 此前结果合同子阶段验收：81 项定向、全量 2664 项＋81 子测试、10 个变更 Python 文件 Ruff、文档链接、Git diff 和 200 项制品完整性检查通过。首轮 1 个缺失摘要链接失败已修复，两个全量报告均保留；不改 A/B 固定基线。

- 双驱动：原 L0 严格片段、有界 LLM、独立候选准入、固定 DAG、串并行/all-success 汇合、过期证据和迟到结果隔离；不新增 Effect，不改变默认 DSH 路由。
- 转译入口 v7：来源/业务任务/编译说明分离；有据读取前段＋原任务直接保留的 L1；参考资料与实际观察无损分层。模型边界注释仍有错误，绝不自动视为语义或权限证明。
- 真实模型与机制分开验收：5 个本地混合执行；35/35 公开图接线检查（5 正常、30 异常，模型替身）；2630 全量测试及 81 子测试通过，39 个变更 Python 文件 Ruff 通过。完整证据与原版零调用回放保留。
- v1–v7 构造、一次失败格式诊断、v6/v7 实际执行共 84 次本地 9B 调用；537,423 输入 / 52,527 输出 token。v7 含模型图时延 p50/p95 30.79/46.36 秒，仅 5 样本且部分与 pytest 并发，不是 SLO 或因果加速证据。

### 保留的历史证据

- v65 结构修复：全量 2566 项及 81 子测试、261 项定向、35 项发布/边界/修复检查、修改代码 Ruff 通过。左右比较数据均受类型、依赖和时效限制；不是公开 Skill 成功率。
- v64/v65 静态预检暴露体积退步：预算通过由 7/9 降至 6/9（Phoenix 新增超限）。未放宽预算；原 v63 和阶段 1 的 3055 份绑定证据全部保持不变。
- 三例报告零调用回放一致，122 份新制品绑定；46,397 输入/15,309 输出 token，请求 p50/p95 为 164.26/231.01 秒。仅机制改善，语义准确率尚无提升证据；不是 Runtime 性能或独立 Gold。

- 阶段 2：10 Skill / 10 仓库 / 9 个开发者分类领域。v62 为 0/9 初始预算通过；v63 为 7/9，另 2 个预算失败、1 个纯 L1。改善的是输入准入，不是语义准确率。
- 本轮 2543 全量 + 81 子测试、22 项定向、修改代码 Ruff 通过。7 份模型报告零调用回放一致；136 份本批证据、2919 份阶段 1 证据已核对。文档分区回归失败及修复保留，未放宽测试。

- [阶段 1 验收](STAGE-1-RESULTS.md)：3 种已知流程各两次构造，6/6 受审只读区域、50/50 合成本地路径通过。20 次 9B 请求；不是 6 个独立 Skill 或生产成功率。
- 512 定向、2524 全量及 81 子测试通过；提交前再验 232 项。阶段代码 Ruff 通过，全仓仍有 224 条历史诊断。
- 源文锚定、观察作用域、原 Schema 参数槽、闭合分支、数组长度、精确 L1 交接已接入可选语义前端。默认 DSH 路由未变。
- 失败、原始模型回复、源码快照、费用与 86 份原版回放完整保留。阶段 1 证据绑定 2919 份本地制品；Git 提交不包含这些被忽略的制品。
- 历次 C3 详细结果移入[历史快照](PROJECT-STATUS-HISTORY-20260910.md)。旧文件/模块仍可引用，不以删失败记录制造提升。

### 当前工作与下一步

阶段 2 已于本地 `dev` 提交为 `25b08c0`，未推送。后续进入[结果合同与语义闭环修复](SEMANTIC-RESULT-CONTRACT.md)：宿主绑定的职责、实际读取/字段校验和未验证草稿分离；不是完整语义修复结束。新的结果映射仍由开发者 AI 审查安装，不计自动转译准确率。

| 顺序 | 工作 | 出口与边界 |
|---|---|---|
| 已完成 | v65 通用结构修复及三例 9B 探针 | 0 个语义接受区域；初始预算退步单列，不扩大 Runtime A/B |
| 已完成 | 规则出处分权与受控混合流程 | 原文/调用者/模型建议分离；读取—推理—独立准入、有界并行/join；不把严格检查转给 LLM 绕过 |
| 已完成（小批开发） | 10 Skill 构造审阅、5 份真实 9B 接线 | 1 完成＋2 局部可用；不接受其余草稿、不改写失败，不等于语义泛化通过 |
| 进行中 | 已暴露的语义与表示缺口 | 已接入结果职责、双向草稿审查和锚定修订；仍需解决审查漏判、有界编辑生成、受控补读、职责自动映射及同类型错绑 |
| 首批失败，后续待验 | 小批新来源迁移、跨 cohort 验证 | 6 Skill／12 任务首批失败已封存；修复后重新冻结新来源，正式门禁仍关闭 |

阶段 2 按[固定交接范围](STAGE-1-EXIT.md)执行：任务不提供期待 L0/参数答案/路由；源/任务/宿主与审阅预期先冻结，首次失败不原地重试。指标按 Skill 分母报告首次结构、语义可用区域、参数、职责保留、正确/过度停止和成本；无法评估的指标保留 unknown，而非 0 或通过。

[受控混合流程](GOVERNED-HYBRID-FLOWS.md)是可选本地原型；开放职责仍是 L1，不把整图称为确定性 L0。阶段 2 草稿中的虚构内容与过度完成表述继续作为失败回归，不因结果合同上线而改写旧评分。自动写事务接线、动态扩图、证据刷新替换和持久恢复尚未实现。按用户后续授权，仅推进已知开发集上的局部机制与语义修复，不启动正式新 cohort、扩大 DSH A/B 或推送。

### 保留但不推进

生产身份/多人审批、Provider 供应链、HA/DR、WORM、Hermes/A2A、真实厂商认证与大规模 Runtime 性能评测仍冻结。原型准则见[权威文档](ENSUREDSKILL-PROTOTYPE.md)。共享虚拟环境、数据库、评测证据和无关用户文件不作垃圾删除。

## English

Current September 16: implementation of the [evaluation reset](EVALUATION-RESET-20260916.md) is authorized; **R0 measurement is partially implemented** in the bounded budget/scoring/pilot/execution/probe modules. They provide persistent budgets, isolated scoring, input preparation, scripted metering and local probes. Translation, enforcement and real-agent benefit remain distinct; old pretranslated/partial-latency A/B cannot establish cold-start benefit. The old semantic stage remains `paused_unmet`.

153 targeted tests pass. The [portable measurement summary](benchmarks/bounded-pilot-r0-summary.json) records 24 scorer fixtures across twelve families, ten budget checks and three existing simulated-gateway mechanism checks, with zero model calls. Scorer receipts, semantic judgments and usage are synthetic; gateway checks reuse existing mechanisms. This establishes neither the full 36-probe gate, automatic Effect bridging, a live DSH cold-start experiment nor agent benefit. The first full suite had 3,434 passes plus 81 subtests, with eighteen failures/errors caused by blocked local sockets/Docker access. All eighteen passed an approved environment recheck; the first report remains, and no single all-pass full run is claimed. Ruff on nine new Python files and diff checks pass.

Remaining R0 work: live DSH full-call accounting and trusted advance token counts, physical Provider reset/isolation, frozen independent reference labels and twelve tasks, and real trace capture. The CLI has only `check/prepare/inspect/score`, no `run`; R0 is incomplete, so R1/R2 have not begun. Retain 9B, dual-drive execution, finite budgets, formal gates and product authority. No model calls, commit or push. All next-step wording below is historical, not an automatic new repair cycle.

Historical [schema/artifact implementation](SCHEMA-ARTIFACT-CONVERGENCE.md) passes engineering verification but regresses actual tasks. On the same threeDSH/9B cases:compiler admission2/3→3/3,query inventory acquisition2/2→1/2,complete tasks1/3→0/3. The single native code patch is unchanged,one query skips inventory,andMesh fabricates a next source.15 model calls,116,059/3,444 tokens,p50/p95=176.19/278.83s;5 authorized reads,no observed effects/replay.148 targeted,8 docs/authority checks,Ruff124 files and full3,324 tests plus81 subtests pass. [Bound evidence](benchmarks/schema-artifact-summary.json). Execution source/model remain frozen;no reruns or changed expectations.

Stop same-case model runs. The current author profile mainly compiles reads plus an original-task reason node;dynamic evidence obligations and verifiable business constraints remain insufficiently contracted. Compiler opinions also enter Runtime reason context,but causal attribution is unproven. Next design work must address these representations rather than more prompts or general reviewers. Original gates remain open;optional correction is not a proven quality improvement or enabled by default. No scaledAB,commit,push or UI restart. Below are historical records.

Current September16:[isolated compilation](ISOLATED-COMPILER.md)is implemented as an optional v4 mode. The tool-free author context receives symbolic inputs;the execution Agent has five tools/no AST submit,with backend enforcement and unchanged read/evidence/authority limits. The sole frozen run covers2 known Skills/3 unchanged tasks:3 faithful deliveries,6 authorized reads,2 admitted prefixes and1 frozen task pass. BothIRQL inventories are now read,but both query tasks still fail inclusive time bounds;the first also projects a removed column. One compiler proposal adds invalid origin/quote to caller and falls back. Mesh meets its old rubric,with snapshot-reread/service-flow caveats preserved.15 calls=10DSH+3compiler+2Runtime,110,926/3,245 tokens;p50/p95=183.48/190.02s. The identical two-query subset uses fewer tokens but has a slower median;no causal speed claim.122 artifacts/80 archived-current sources and model identity verify,no tuning/rerun. [Portable QA and results](benchmarks/isolated-compiler-summary.json). Gate1 stays open;complete compiler-schema output constraints and independently verifiable artifact obligations are next,not implemented. No new authority,broaderAB,commit,push or UI restart. Following entries are historical.

Historical September16: the [bounded call-protocol/evidence package](CALL-PROTOCOL-EVIDENCE.md) added concrete read schemas,explicit operator prerequisites and incomplete exit. Actual-DSH scripted checks passed2/2;its sole9B regression had2 faithful deliveries,0 compiled prefixes,0 inventory acquisitions and0 complete tasks. All13 calls wereDSH,150,818/2,992 tokens;p50/p95=175.34/203.95s. Post-run guidance/config-error/status fixes and archive differences remain disclosed in its [summary](benchmarks/call-protocol-evidence-summary.json). The subsequent context-isolation package is recorded above;old scores are not replaced.

Historical September16: the [two-round contract](CONVERGENCE-CLOSURE.md) finished with6/6 normal faithful deliveries,8/8 authorized reads,zero observed effects/replay;3 Runtime drafts and3 native fallbacks.38 calls,350,293 input/7,057 output tokens;p50/p95=155.59/276.89s,no speed inference. Frozen15/24 criteria and2/6 tasks are not calibrated accuracy: two Oracle ambiguities remain,both queries miss required inventory,and CAPA omits content or violates format. Its bounded successor is recorded above;old results remain unchanged.196 artifacts/79 sources verified at that checkpoint;QA is in the [historical summary](benchmarks/convergence-closure-summary.json).

Current September 15: the [snapshot-read mechanism](SNAPSHOT-READ-SEMANTICS.md) is fixed within the v4 local immutable host. Exact duplicate follow-ups do not invoke Providers or fabricate evidence; distinct resources and original attempt/authority gates remain. Two actual DSH/9B synthetic tasks (one Skill) finish: correct snapshot/recovery interpretation and a three-resource inventory comparison. Four reads,10 model calls,52,874/1,068 tokens;45.53/65.17s, not performance A/B. Raw scoring stays5/6 criteria,1/2 fixture tasks, with circuit c1's extra loss-percentage requirement disclosed as an Oracle/task mismatch rather than a calibrated50% task-failure rate. Future fixture v2 fixes the visible request but was not model-run. Nine new tests,3,263 full tests plus81 subtests in244.33s,and Ruff on116 changed/new project Python files pass;75 artifact and78 archived/current source hashes verify. No old regrading, Mesh/CAPA/IRQL rerun, broad generalization claim, commit or UI restart. All following states are historical.

Current September 15: [task-first delivery](TASK-FIRST-DELIVERY.md) removes model-selected output duties under an explicit v4 host. Complete tasks and execution controls remain; answer is unverified L1, not executable L0. One frozen two-call Mesh/9B diagnostic restores a concrete trace request, but redundant reads and omitted business-flow naming persist. Control exactly reproduces the historical reply. Total14,107 input/668 output tokens;40.76/33.81s per call, not a population performance result. No DSH/Runtime rerun or old score change; CAPA/IRQL not retested. Separate actual-DSH/scripted-model terminal checks pass2/2.25 new tests and Ruff on114 changed/new project Python files pass; final full-suite status is in the [summary](benchmarks/task-first-delivery-summary.json). Evidence10/71 and66/76 artifact/source sets verify. Semantic gates remain open; no self-judge, broader A/B, commit or UI restart. All following states are historical.

Current September 15: the opt-in [host-terminal mechanism](HOST-TERMINAL-DELIVERY.md) is repaired using public DSH concludeTurn, a same-turn operation guard and host-receipt presentation. Actual DSH with scripted model transport passes 2/2 after frontend wiring correction; the first 0/2 empty-output attempt is preserved. Six protocol requests and one actual read per case, no seventh request, exact host stdout. **Zero real model calls; no new semantic accuracy claim.** Model-selected presentation kinds do not prove complete task interpretation; semanticCoverage is not_assessed. Latest real 9B remains 2/2 structure, 4/6 criteria, 0/2 tasks. CAPA/Mesh omissions and the IRQL issue, not retested in this package, remain open. Next work must address task-duty/evidence/output interpretation causally, not add reviewers or alter stage exits. 24 Node tests, 178 targeted Python tests, 3,229 full tests plus 81 subtests in 231.28s, and 111 changed/new project Python Ruff files pass. Both 66-file evidence sets and 76-file source archives verify. [Portable mechanism evidence](benchmarks/host-terminal-delivery-summary.json). No default UI change/restart, commit or push. All following entries describe historical states.

Historical single-choice package: the [single-choice package](DELIVERY-SINGLE-CHOICE.md) is implemented and its sole frozen DSH/9B run is finished. Structural admissions2/2, original criteria4/6, complete tasks0/2. CAPA still omits the explicit keep-open conclusion; Mesh fails actionable next-evidence guidance. The IRQL gap is not retested.14 complete replies,176,566 input/3,183 output tokens,one graph,three authorized reads,no writes/replays. p50/p95 241.80/366.68s is slower than the prior batch, n=2 not causal A/B. [Portable results](benchmarks/delivery-single-choice-summary.json).

The bounded package is closed, not Gate1. Same-case model reruns stop. The host terminal guard prevents replay but not a redundant DSH deliver call; semantic source-kind selection remains unqualified. Proposed next decisions separate real harness terminal delivery from model-proposed presentation/semantic obligations. Neither lifecycle takeover nor a changed research exit has been implemented. Fresh-source, cross-cohort and paired A/B gates remain closed; no commit/push or UI restart. All entries below are historical.

Latest September15: [source-addressed compact delivery](DELIVERY-REFERENCE-CONTENT.md) implements host-owned references/metadata and constrained Runtime output with independent validation. The first metadata-heavy run remains0/2 tasks; the revised static schema is802 instead of37,132 bytes without relaxed membership. CAPA now admits a native candidate in72.50 seconds but misses explicit open status/next evidence (1/3). Mesh passes base schema yet fails content/unresolved consistency; final native text wrongly claims completion (2/3 content criteria). Overall0/2 tasks,3/6 criteria,one graph,three reads,13 calls,154,176 input/3,690 output tokens, p50/p95 166.26/250.65 seconds, n=2 not causal A/B. The earlier17 complete replies plus one unknown-usage step remain separate. [Machine evidence](benchmarks/delivery-reference-content-summary.json).

Next use one mutually exclusive content/inability representation, then fix requirement/source interpretation and host-status/final-answer divergence. Do not reinterpret literal None, weaken checks or repeat prompt tuning. Gate1 remains open; fresh-source, cross-cohort and same-DSH A/B follow only after their exits. No commit/push, new cohort or UI restart. QA is recorded in the current report. All following entries are historical.

Latest September15: **Gate1 end-to-end usability remains unmet**. Explicit JSON-text transport and separate fixed requirements/host read state are implemented. Both native9B candidates parse; neither passes the content schema. Results are0/2 tasks,2/6 unchanged criteria,3 authorized reads,zero Runtime model calls. CAPA lacks gap and an explicit open conclusion; Mesh misplaces state and times out at420 seconds after compiler fallback. Nineteen complete replies record249,800 input/8,249 output tokens; one interrupted step has unknown usage. p50/p95 is311.20/409.15 seconds, n=2 including timeout, not causal performance evidence. [Report](GOVERNED-SESSION.md) / [portable summary](benchmarks/governed-session-wire-state-summary.json).

Post-run diagnostics identify the exact missing field; explicit no-retry guidance distinguishes retained candidates from validated renderings. Zero-call checks keep both original candidates rejected. QA passes174 targeted checks,3,166 full tests plus81 subtests, Ruff on105 changed/new Python files, documentation/tool-schema/diff checks. Sixty-three run artifacts and75 archived source files verify. No regrading, model rerun, commit/push or UI restart. Four gates remain including the current one: usability, frozen fresh-source transfer, cross-cohort generalization, then same-DSH native/auto Runtime A/B. Next reduce structural/source-copying burden and clarify output interpretation/final delivery, without universal self-review or broader authority. Any future bounded text correction requires explicit attempt/evidence/cost semantics and must not become execution retry. The entries below are historical.

### Historical delivery-contract and split-tool version

Current September15: source-anchored delivery contracts and six separate-mode tools are implemented; the semantic stage is NOT passed. Four typed output kinds, exact task/visible-Skill quotes, session binding, explicit gaps and deterministic rendering grant no authority or semantic approval. Runtime draft(session_id) and native-fallback deliver(session_id,response) no longer overload one call; actual state/deliveryAction is explicit and null-plan resubmission cannot switch an admitted route.

The first real9B batch regressed to0/3 tasks,3/9 criteria, including a420-second wrong-mode loop;26 replies have usage and one interrupted step remains unaccounted. A separate two-case repair recovers Mesh's single Runtime generation and3/3 content criteria; CAPA remains2/3 and its native candidate fails type validation (string instead of object). Thirteen calls,130,104 input/3,563 output tokens; p50/p95 138.01/190.66 seconds (n=2, not performance A/B). [Evidence](GOVERNED-SESSION.md) / [portable summary](benchmarks/governed-session-delivery-summary.json). Do not pool denominators or regrade history. Remaining work: native response transport, kind interpretation, fixed requirements versus transient evidence gaps, and checked-candidate/final-answer fidelity. No large A/B or production expansion; old next-step statements below are historical.

Current QA:91 targeted checks,3,138 full tests plus81 subtests, Ruff on103 changed/new Python paths, docs/tool-schema/diff checks pass;153 bound artifacts match. No commit/push or user-UI restart. Unrelated historical lint findings remain out of scope.

Current evidence-first v2: mechanism changes complete, semantic acceptance not passed. Two known-case runs both achieve 1/3 tasks; criteria decline 6/9→5/9. One IRQL parameter binding is corrected before execution and evidence is gathered before drafting, but the output is only a query outline. CAPA omits explicit open status and next evidence; Mesh completes. Two compiled graphs, five reads, 16 real9B calls, 136,754 input/3,982 output tokens, p50/p95 138.67/172.44 seconds. Both Runtime drafts have no_supported_artifact, not successful query validation. [Evidence and next scope](GOVERNED-SESSION.md): a general source-grounded deliverable contract, not more self-review/prompt rounds, per-case L0 or evaluator answers. No large A/B, commits/pushes, writes, scripts or real devices. Earlier next-step statements below are historical.

September 15: the active path is [shared compilation + original Runtime + native DSH fallback](GOVERNED-SESSION.md). Product imports no evaluator. Four host-configured tools separate translation, execution and task status. The first native three-known-case run failed before useful completion: source overflow lost its fallback session; two plans used raw paths where typed parameter bindings were required. A new version repairs the usable fallback and exposes the exact tool schema, while retaining the failed run. Three paging compatibility regressions from extraction are fixed without weakening checks. These changes do not pass historical semantic or fresh-source gates, trigger large A/B, or commit/push. Older next-step statements below are historical.

Current September 15: the [evidence-first, observation-quote-only note mode](SEMANTIC-DUTY-CONTRACT.md) completes two known cases. It reuses the driver with lossless source presentation, host-ID location, pairwise predicate comparison and keep/quote-only notes. Ten calls bind; seven predicates/14 comparisons reconstruct; four excerpt notes; 26,053 input/1,543 output tokens, request p50/p95 11.19/25.17 seconds, not performance A/B. Strict auditing passes without a digest exception. **CAPA nonapproval is still misjudged as evidence of nonreview; the stage remains incomplete.** Bodies/parameters remain unchanged; no claim of query repair or whole-task completion. Next work is independent source/target predicate representation and alignment, not more self-review or weaker gates. No expansion, commit or push. Historical entries follow.

QA: 92 targeted tests, 3,083 full tests plus 81 subtests, changed/new Python lint, docs and diff checks pass; 22 regressions are new. Engineering and semantic conclusions remain separate.

Later September 14: the [host-text/predicate diagnostic](SEMANTIC-DUTY-CONTRACT.md) completes its bounded live checks. Host-owned exact task anchors replace LLM contract rewrites; note decomposition precedes keyed evidence review and host aggregation; independent edits isolate failures. Eighteen calls (IRQL eight/CAPA ten), 95,154 input/6,240 output tokens, request p50/p95 16.80/54.45 seconds, not comparable performance gains. CAPA removes an unsupported not-reviewed predicate despite a rejected sibling note. Query assembly rejects a broken code fence and publishes no candidate. Unsupported positive reviews and role-interpretation drift remain; **the stage is incomplete**. Post-run fixes address per-cell structure admission and checkpoint digest-key collision. Original invalid metadata stays untouched, with explicit derivative forensic evidence. No model retries, expansion, commit or push. Entries below are historical and not combined as improvement.

Final QA: 158 targeted tests, 3,061 full tests plus 81 subtests, changed/new project-Python lint, docs and diff checks pass. Post-run structural/digest fixes are engineering validation, not revised grades for the 18 original model calls.

Current September 14: [duty contracts and local artifact checks](SEMANTIC-DUTY-CONTRACT.md) are implemented and the bounded live experiment is complete. First-pass completion is 0/2. Only independent unattempted siblings are subsequently continued; failed calls are not retried. CAPA loses the unsupported not-reviewed assertion. 9B changes IRQL interval operators, while a separate zero-call typed renderer fixes two datetime literals. Total 17 calls (12+5), 126,955 input/10,192 output tokens, request p50/p95 31.32/77.44 seconds, not Runtime performance. Duty source/interpretation drift and contradictory review remain; **the stage is incomplete, new-source and large A/B gates stay closed**. QA passes 224 targeted tests, 3,013 full tests plus 81 subtests, changed-Python lint, docs and diff checks. No commit or push. Entries below are historical.

Current September 14: [stage-one context/owned-edit repair](SEMANTIC-CONTEXT-REPAIR.md) completes this implementation round and four real 9B regressions, but fully repairs none of their designated critical defects. The stage stops for validation-design review, not another same-case prompt run. Ten calls use 61,350 input/13,553 output tokens; request p50/p95 are 67.00/149.72 seconds, not Runtime performance improvement. Lossless task roles, read provenance and shared eight-cell body/note repair are implemented. Post-run note-target and heading fixes have engineering/zero-call structural checks only. QA passes 162 targeted tests, 2,951 full tests plus 81 subtests and changed-Python Ruff. No cohort expansion, commit or push. Entries below are historical.

Latest September 14: the [task-witness/evidence-scope diagnostic](SEMANTIC-WITNESS-DIAGNOSTIC.md) completes eight 9B calls (34,213 input/9,851 output). Four source plans bind; two comparisons bind and two fail. Partial mesh/Phoenix detections coexist with core errors, so no automatic editor integration or new cohort is enabled. Exact task-delivery witnesses and note/body ownership guards are implemented; actual note repair remains pending. QA passes 154 targeted tests, 2,934 full tests plus 81 subtests (222.66 seconds), changed-Python lint, docs and diff checks. All four new-interface zero-call budget checks stay within the unchanged limit. The driver interruption and two corrected developer-AI review descriptions remain in the audit. The stage is incomplete; entries below are historical.

2026-09-14: the [four-known-case 9B/27B review contrast](SEMANTIC-REVIEW-MODEL-CONTRAST.md) finished, but the stage remains incomplete. 27B newly detects missing Phoenix suggestions; query/predicate and stale-data errors remain. Two responses pass binding and two fail missing citations. Five attempts include the first timeout and four complete responses: 21,421 known input / 8,391 output tokens, plus unknown timed-out usage. Default Runtime models, old scores and new-source counts are unchanged. The subsequent numbered-list/heading fix passes 76 targeted checks, 2,901 full tests plus 81 subtests, changed-file lint, documentation and diff checks. A zero-call check of the actual mesh input removes three isolated labels while preserving source/candidate data and all non-whitespace characters, without regrading it. Next priorities are task/evidence scope and artifact validation, not a wholesale model replacement. September 11 approval-pending entries below are historical.

Current repair status (2026-09-11): frozen v2 failed and the stage is incomplete. Entry-first source budgeting, completed-read indexing plus previous-candidate continuity, sole-whole-draft omission assignment and conservative withholding of unlocated positive opinions are implemented. Known-only regression has finished with unresolved semantic defects; the proposed comparison awaits user approval. Read-only quotations, section guards, no-read retention and original authority gates remain. No new-source credit or old-batch regrading is allowed.

Known v9/v10 now finished with 20 additional 9B calls: specific owner/count omissions and check-ID binding improve, but incorrect query support and stale mesh evidence use remain. QA passes 2,891 full tests plus 81 subtests, 77 targeted tests and 51 changed Python files lint. A four-known-case local 27B review-only comparison awaits user approval; it has not run. No larger batch, code submission or semantic-pass claim is made.

Latest v2 result: six Skills/six repositories/six developer domains/twelve tasks, zero fulfilled substantive tasks, three correct boundary responses, eight partial and one failed; 24/36 criteria met, 67 calls and 342,581 input / 52,968 output tokens. Contained tool execution does not establish correct prose. [Actual cases and costs](SEMANTIC-CLOSURE-TRANSFER-V2.md). Entries below are earlier milestones, not a new completion claim.

Latest result (2026-09-11): frozen v1 finished and failed the semantic-closure exit. Six Skills/six repositories/six developer domains/twelve tasks: one fulfilled, one correct refusal, seven partial and three failed; 22/13/1 met/unmet/unknown criteria. All 56 real 9B calls remain. [Results and cost](SEMANTIC-CLOSURE-TRANSFER-V1.md). New known-development verification tests valid-prefix projection, separate answer/support channels, lossless review transport and retained unresolved duties/findings. Old judgments are immutable; the stage remains incomplete. The following entries are earlier milestones, not a new acceptance claim.

As of 2026-09-11, [three repair packages and frozen transfer acceptance](SEMANTIC-CLOSURE-EXIT.md) remain **incomplete**. The latest fixed-version known checks retain handoff responsibilities/times and the actual README API/install example. Unsupported licensing is narrowly quarantined by the host, not semantically corrected by 9B. Transfer v1 was then frozen before selecting six Skills/six repositories/six developer-classified domains/twelve tasks; all inputs and criteria were sealed before model calls. This uses a shared synthetic read-export adapter, not complete native tools or full Skill workflows. Initial construction exposes source-document/data confusion and unseen-page citation failures; this batch is not tuned or relabeled. Review/edit/final-review costs are explicit. [Diagnosis](SEMANTIC-CLOSURE-DIAGNOSIS.md) and [reproduction/judgment](SEMANTIC-CLOSURE-RUNBOOK.md) preserve failures. No formal A/B, commit or push is claimed.

Latest bounded-review substage: four retained attempts over two known Skills, eighteen Chat requests (seventeen with usage and one HTTP 400). An anchored edit adds Chen's responsibility, but Alice's incident ownership/start time remain missing despite all-supported AI review. Documentation's ten edits exceed eight and are rejected without final review. [Evidence and remaining gaps](BOUNDED-DRAFT-REVIEW.md) / [portable record](benchmarks/bounded-draft-summary.json). Regression passes 136 targeted and 2699 full tests plus 81 subtests, fourteen changed Python files lint, documentation/diff checks, four archives, eighteen receipts, eleven exact review and one exact patch re-derivation. New work is uncommitted/unpushed; default DSH and A/B settings are unchanged. This is local repair, not new sample coverage or semantic generalization.

Updated 2026-09-10. Stage 2 is checkpointed locally at 25b08c0, not pushed, under [the explicit criteria](STAGE-2-HYBRID-VALIDATION.md); formal generalization stays closed. Its frozen v7 has 10 Skills, 6 structural candidates, 5 useful read prefixes and 5 real local 9B runs: **1 fulfilled scoped request, 2 useful partial analyses, 2 rejected drafts**. [Old evidence](STAGE-2-HYBRID-RESULTS.md) is unchanged. Subsequent [result-boundary work](SEMANTIC-RESULT-CONTRACT.md) adds host-bound observation/duty checks and unverified drafts, with six real calls over three known Skills. Final Notion duties are 2/2; handoff 2/3 and documentation 2/4 remain partial. These are not task accuracy scores; the README draft still fails developer review. The mappings are host-authored, not automatically translated. Object-order replay faults are fixed in new canonical compilation; two zero-call failures are retained.

The Runtime composes original strict regions with bounded model tasks, independent admission and required joins. Stage 2 v1–v7, its diagnostic and live revisions retain 84 actual 9B calls, 537,423 input / 52,527 output tokens; v7 p50/p95 30.79/46.36 seconds is not an SLO or causal speedup. Subsequent local result-boundary probes are counted separately. Current work remains known-development repair, not a new formal cohort, push, scaled DSH A/B or production rollout.

Stage 1 has three known procedures, six reviewed read regions and 50 synthetic paths, not six independent Skills or a production probability. Full regression: 2524 plus 81 subtests; stage lint passes while 224 historical diagnostics remain. The local evidence binds 2919 artifacts and preserves original-version replay; ignored artifacts are not backed up by this Git commit. Detailed C3 history remains in the [snapshot](PROJECT-STATUS-HISTORY-20260910.md).

Earlier v63 passed 2543 tests plus 81 subtests and retained seven identical replays with 136 bound artifacts. v65 and mixed-prototype evidence remain separate. Typed value comparison alone does not determine business semantics. Governed read/reason/admission and joins now exist, but durable recovery, dynamic graph expansion and automatic Effect integration do not. Unknown Oracle/coverage metrics remain unknown. Use new development batches before cross-cohort and DSH comparison gates; production engineering stays deferred.
