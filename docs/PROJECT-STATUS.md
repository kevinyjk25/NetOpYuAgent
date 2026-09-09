# EnsuredSkill 项目进展 / Project Status

## 中文

更新：2026-09-09。**C3w：真实 9B 的独立规划／分槽填参／原编译器／合成局部执行机制链已闭合；公开源语义转译仍未通过。** C3i–C3p 已本地提交 dev `f0499ec`；C3q–C3w 尚未提交。本轮未推送或合入 master；下方各阶段“未提交”是当时的历史状态。默认 DSH 路由未改变，研究候选未自动激活。

### Done

- C3w 可选 `plan_first`：先冻结源锚定操作骨架与分支，再逐槽构造参数；动态来源导航、原词法作用域及每槽来源校验阻止编造输入。无可达路径的冗余终止可留痕消除，不删调用、不补成功；原执行器及授权机制不变。[设计、使用、全部结果](PLAN-FIRST-AUTHORING.md)。
- C3w **8 次实验／12 次真实 9B 调用**。同一已知 Netdata 的 5 次仍未编译，thinking／可读 Schema 没有解决源义务和操作混淆。另用原有合成接线说明隔离验证：v18 被不可达结束拒绝；v19 计划通过但编造参数；v20 正确使用输入／前序结果引用，3 次调用、42.549 秒、10,465/853 token，生成 2 个读取与条件。三版首次规划请求相同，没有预写 L0 输入。
- C3w 经绑定摘要的开发者助手审阅后，生成的局部读取流程在原 Runtime 上 **10/10 本地路径检查通过**；仅合成读取，无网络／脚本／写操作。`errors == 0` 决策及变更候选未转译，剩余职责列表不完整，不接受整 Skill 成功。新增公开 Skill 数为 0，不扩大 Runtime A/B。
- C3w 最终新增 49 项机械测试；**309 项定向通过（25.14 秒），全量 2292 passed + 81 subtests passed（179.83 秒）**；21 个相关 Python 文件 Ruff、文档链接与空白检查通过。[验收记录](../artifacts/translator-v2/source-plan-20260909/validation-summary/report.json)保留前一轮 2290 项全量结果和常量出处缓存修复。8 份报告按原源码隔离零调用回放一致；[摘要](../artifacts/translator-v2/source-plan-20260909/evidence-summary/report.json)绑定 127 个文件、此前 275 份证据不变。当前阶段完成的是机制接线，不是公开 Skill 高准确率门禁。
- C3v 增加可选 mode_bound：宿主模式与原 catalog/合同/Schema 摘要绑定；只表达显式对象的必填/允许键与互斥性，逐叶构造并阻止动态引用隐藏受约束形状。行动依据复用源块 ID，保留准确原文和操作轨迹；不代表行动必要性、语义通过或权限。[范围、用法和证据](HOST-OPERATION-MODES.md)。
- C3v **2 次新 9B 调用，0 编译通过、0 底层/脚本执行**。v12 为 2 次相同 discovery 读取，101.794 秒、8,115/1,165 token；精确命令引文抄写失败被保留。v13 回填原文块后又生成 8 次相同读取，215.441 秒、8,113/3,228 token，14 个出处错误、7 未决项、无终止。两版模式混用均为 0，但**整体可用性未提升**；不择优拼接或修改旧结果。下一步应先区分原行动、替代路径、示例和义务阶段，再冻结骨架并填参。
- C3v 新增 40 项机械测试；跨模块 300 项定向通过（24.96 秒），全量 **2243 passed + 81 subtests passed（255.63 秒）**；19 个相关 Python 文件 Ruff、已跟踪/25 个未跟踪源码文档空白及文档链接检查通过。两份源码快照隔离零调用回放逐字节一致，[摘要](../artifacts/translator-v2/source-modes-20260909/evidence-summary/report.json)绑定 39 文件、此前 236 份绑定文件不变。测试数量不是语义准确率。
- C3u 增加可选 catalog_bound：按原宿主 Schema 生成参数、字面值来源留痕、路径引用转为代码分配的唯一别名；不改变原编译/执行门禁。独立报告未决义务、来源、类型、作用域、缺少/越过终止，重复实参只提示不自动去重。[能力与真实结果](CATALOG-DIRECTED-AUTHORING.md)。
- C3u **1 次新 9B 调用 / 214.388 秒 / 7,787 输入、3,452 输出 token**。同一已知源和对齐任务上，宿主 Schema 校验消息 **56→0**、额外重复别名 **7→0**；仍为 8 次相同读取、无终止、7 个未决项、8 个来源问题，**0 编译通过 / 0 底层执行**。发现宿主代码禁止 discovery/query 混用，但其简单 Schema 未声明该约束；本轮只静态核查，未执行候选或改宿主。不能称为 Skill 成功或整体性能提升。
- C3u 新增 38 项机械测试，跨模块 333 项定向通过；全量 **2203 passed + 81 subtests passed（178.36 秒）**，17 个相关 Python 文件 Ruff、diff 和文档链接检查通过。原模型报告隔离零调用回放逐字节一致，最终静态定位独立保存；[证据摘要](../artifacts/translator-v2/source-catalog-20260909/evidence-summary/report.json)绑定 21 份文件，此前三阶段 215 份绑定证据不变。后续优先公开操作模式约束、构造操作流程骨架与逐叶参数来源，不扩 Runtime A/B。
- C3t 增加可选源义务审阅：源规则与任务/宿主 prose 隔离，多阶段分类、解释来源和未满足门禁分开；未证明审阅收益，CLI/API 默认 direct。加入本地字面缺口检索和生成前表达式语法；Schema 注解压缩不改变约束/字面数据。[当前流程和失败定位](SOURCE-OBLIGATION-AUTHORING.md)。
- C3t 同一已知 Skill 的 **7 次开发实验 / 12 次真实 9B 调用**：旧任务仍停于缺口；新对齐任务明确只做合成宿主离线局部构造，不能与旧 Cloud 业务任务合并计算提升。最后直接路径 **1 次 / 151.381 秒 / 8,099 输入、3,380 输出 token**，完整候选存在但未决义务阻止编译；8 次重复别名、字符串误生对象、空筛选数组仍错误。全阶段 **0 编译通过区域 / 0 底层与脚本执行**。
- C3t **295 项定向通过（14.27 秒）；全量 2165 passed + 81 subtests passed（177.20 秒）**。15 个相关 Python 文件 Ruff 通过；7 份历史实现报告隔离零调用回放逐字节一致，[摘要](../artifacts/translator-v2/source-obligations-20260909/evidence-summary/report.json)绑定 112 份文件。首轮收尾的提前发布链接失败已保留并修正；前两阶段 103 份证据不变。前置审阅仅保留为实验，后续优先宿主类型引导的参数、值来源和稳定别名，不扩大 Runtime A/B。
- C3s（历史）分离已交付原文、当前精确可见区间和语义未审核；重复回读转入原文联合窗口候选/缺口决策。新增源操作→宿主参数名的版本化、合同/Schema 摘要绑定声明；明确不等于语义等价或授权。[使用、边界和实测](SOURCE-DECISION-AUTHORING.md)。
- C3s 两次同源开发修订：v4 **1 次 / 44.280 秒 / 7,868 输入、335 输出 token**，4 条引用重复一个缺凭据判断；v5 明确离线构造与现有执行门禁后 **1 次 / 27.635 秒 / 7,765 输入、83 输出 token**，仍首轮停止。都为 **0 候选、0 编译读取、0 底层调用**；引用不能证明宿主凭据缺失。联合回读分支只有机械验证，不能把更早停止称为转译提升。
- C3s 最终 **273 项定向；全量 2132 passed + 81 subtests passed（176.33 秒）**；9 个相关 Python 文件 Ruff、diff、文档链接及证据检查通过。两版隔离零调用回放逐字节一致；旧失败/Oracle/基线保留，未调用旧 manifest 不计模型调用。下一步是义务所属阶段与停止依据的类型化诊断，不扩大 Runtime A/B。

- C3r 实现 UTF-8 有界源页、窗口切换、带来源的 note 与依赖请求记录、原句回填。块 ID 取代模型抄写引文；同文件区间并集合并重复原文，不删除 note、不合并不同解释。[流程与报告](SOURCE-LEDGER-AUTHORING.md)。
- 四份已知公开材料无损分页验证；三次协议修订对同一 Netdata 开发请求做 10 次真实 9B 调用，失败全部保留。v1 因引文换行变化停止；v2 来源校验通过但重复回填超预算；v3 同一待发状态代理从 **50,403→36,971（−26.6%）**，14 条 note 与覆盖相等。新增实际 v3 运行 6 次/535.562 秒，未再触发来源或输入预算错误，但根文/指南往返回读，**0 候选/0 底层执行**。不能计算转译准确率或解锁 Runtime 大评测。
- C3r 最终验证：**246 项定向；全量 2105 passed + 81 subtests passed（278.90 秒）**。相关 6 个 Python 文件 Ruff、302 个本地文档链接、摘要与 diff 检查通过。三版报告分别从原源码快照离线回放，逐字节一致；[开发审阅和 72 份绑定证据](../artifacts/translator-v2/source-ledger-20260909/evidence-summary/report.json)保留循环轨迹、未决状态与宿主对应关系问题，不是独立 Gold。

- C3q 冻结原始源包、任务/宿主、源码/环境/模型摘要和显式调用预算，9B 只请求保留的惰性文本页，候选送入原 Tree 编译器；精确引文、来源支配、结构失败和断点漂移均保留，不自动重试或执行。[完整处理和首次结果](PROGRESSIVE-STRUCTURED-AUTHORING.md)。
- Netdata 首次实际调用 2 次，共 **44.426 秒、13,870 输入 / 259 输出 token**；根文→单设备指南→请求 Cloud 协议。第三次所需请求 45,057 字节超过 36,000 上限，未调用；28 页只提交 2 页，0 候选，语义准确率为空。检索理由出现 category/severity 用词混淆；不是参数执行错误的证明，也不能把预算停止归因于 Schema 或 9B 语义能力。原结果保留，不扩预算覆盖重跑。
- C3q 新增 29 项机械测试、跨模块 211 项定向通过；全量 **2070 passed + 81 subtests passed（161.34 秒）**。两份新增 Python 文件 Ruff、279 个本地文档链接、14 份绑定证据和 diff 检查通过；基于 `f0499ec` + 源码快照的隔离离线回放逐字节一致。C3q 新工作尚未提交。

- C3p 增加通用 `column_rows`：保留原字段名，按当前 metadata 解码全部有界行；重复索引、越界、缺列和非法值拒绝，不猜测/跳行。接入原绑定、Tree/Flow，来源支配与回执时效继续约束候选。[用法](NETDATA-ISOLATED-VALIDATION.md)。
- 一个明确的进程内合成 Function 宿主，通过原读取网关实际调用 info/query；检查身份/范围、有效参数、listener、秒/微秒和返回筛选条件。输出 3 行本页计数（crit 1、warning 2），不保存原始回执；不是原厂服务、完整窗口或 9B 成绩。全任务 `taskCompleted=false`，默认路由未变。[证据](benchmarks/netdata-isolated-summary.json)。
- C3p 最终验证：**190 项定向；全量 2041 passed + 81 subtests passed（161.07 秒）**。36 个变更 Python 文件 Ruff、468 个本地文档链接、diff 及源码摘要通过。隔离源码重建的 6 份新版输出逐字节一致，旧接线 9 份输出不变；历史 C3n 证据由原快照核验，不改旧 Oracle。Git 未提交。

- C3o 新增离线 `task_alignment`：每文件审查区间与原句绑定，区分候选读取/Effect、L1、约束、条件及任务外义务；全量源分页与开发审阅结论分离，不泄露判定给未来模型输入。[用法与四例](TASK-SOURCE-ALIGNMENT.md)。
- 四个开发请求记录 **35 项声明义务、11 个问题、12 条未绑定宿主需求（9 工具／2 上下文／1 策略）**；不是穷尽覆盖或已实现工具，不虚构认证/脱敏 API 补数。Netdata 同 commit 新补一份协议，达到 17 文件／202,930 字符／28 页。保留原始日志落盘与隐私要求的张力、示例列索引不一致、动态解码与时间单位语义。源脚本未执行，9B/Provider 调用为 0，转译指标为空。[绑定证据与回归](benchmarks/task-alignment-summary.json)。
- C3o 最终验证：**122 项定向；全量 1975 passed + 81 subtests passed（161.82 秒）**。31 个变更 Python 文件 Ruff、439 个本地文档链接、20 份隔离重建输出逐字节一致及 diff 检查通过。分类修正不改变四例原始源包、任务或模型输入；旧版档案及快照保留，C3n 证据未变。不是语义准确率；Git 未提交。
- C3n 将嵌套输入/输出、JSON Pointer、分支和候选生成接入原 `qualify_flow` / `run_read_flow` / `execute_host_read`，不新建执行器。精确引用和 Tree 摘要绑定、支配关系、嵌套资源权限、控制条件回执时效、数据复制及异常脱敏保留。[完整处理过程与复现](STRUCTURED-FLOW-WIRING.md)。
- 一份开发 fixture 实际执行 **2 次本地只读回调、0 写入、0 模型调用**，停在 Effect 候选；另存空数组/越权/未认证及两个正常分支。原七个开发图与基线 v1 资格包/摘要一致，主演示回放逐字节一致。不是公开 Skill 准确率，也不认证设备采样时间；[回归与证据摘要](benchmarks/structured-flow-summary.json)。
- C3n 最终验证：**227 项定向；全量 1949 passed + 81 subtests passed（226.55 秒）**。29 个变更 Python 文件 Ruff、418 个本地文档链接、隔离源码重建的 9 份逐字节回放及 diff 检查通过；前两轮绑定证据未变。保留初始测试构造失败记录，未把通过数计入语义指标。Git 未提交。
- C3m 新增版本化结构化 Schema 和数据绑定原语：嵌套对象/数组/null、原键名、约束与有界局部引用；显式 JSON Pointer、常量、对象/数组构建，源值与最终目标每次验证。旧扁平合同和历史证据未改变。[用法、支持范围与证据](STRUCTURED-DATA-BINDING.md)。
- 沿用上轮合成 catalog：同两份 Schema 的可表达性由旧入口 0/2 到新入口 2/2；真实离线 CLI 验证嵌套参数、输出固定索引及非法枚举/空数组/catalog 漂移阻断。不是 Skill 语义准确率，LLM/Provider 调用均为 0，未增加执行器或激活合同。
- C3m 最终验证：**94 项定向测试；全量 1908 passed + 81 subtests passed（213.70 秒）**。23 个变更 Python 文件 Ruff、390 个文档本地链接、21 份本轮绑定证据通过；上轮 40 份证据未变。首次全量因并行编辑文档触发 `sourceState` 漂移，负结果保留；静止工作区复验通过，未修改冻结门禁。Git 未提交。
- C3l 新增无损、惰性的转译输入包和分页，保留引用位置、模板/正文/代码角色、路径歧义；接入 `netopyu-market-corpus translation-intake`，不改变旧执行包门禁。[实现、用法和证据](TRANSLATION-INTAKE.md)。
- 同四份固定源文验证：OpenMontage 65,778 字符完整保存；Netdata 同 commit 补取 8/8 引用，扩为 16 文件／146,532 字符／23 页。Git 大小列解析失败和修复后的成功链均保留，不增加采样 Skill 数；新引用尚未全部闭合。
- 原始宿主 Schema 无损保存并给出 JSON Pointer 缺口诊断；复杂结构尚未接入旧 L0 执行。完整阅读 Netdata wrapper 后确认认证、凭据缓存、通用 HTTP 方法及条件参数边界，不能把 query 名称当只读证明。本轮新增 9B 调用为 0，语义指标为空。
- C3l 最终验证：**81 项定向测试；全量 1851 passed + 81 subtests passed（156.60 秒）**。18 个变更 Python 文件 Ruff、文档链接、40 份绑定证据及 diff 检查通过；真实 CLI 的离线源包/宿主诊断复现和拒绝覆盖检查通过。未提交 Git，未运行第三方脚本；这些是机械回归，不是 1851 个 Skill 的语义测试。
- 新采集保存 12 份查询、合并 237 个 URL 候选；固定抽样 60 个／44 仓库，处理 60/60，保存 **53 Skill／38 仓库**，7 个排除不替换。29 个格式合格、24 个格式变体；这是入库资格，不是转译准确率。[全部候选与原文](../artifacts/translator-v2/public-source-20260908-round2/report/skill-library.html)。
- 新汇总器绑定抽样、四批静态快照与原文索引，区分全部处理完和 importer accepted 上限；额外已暴露仓库显式排除，源码不执行，失败不丢失。
- 四份固定根源文初审定位：嵌套元数据、长源文、真正包外引用与模板占位误报、结构化数据、条件写入及 L1 泛化职责。[诊断与边界](PUBLIC-TRANSLATION-BATCH.md)。未固定完整任务/宿主/Gold，本轮没有新增 9B 成绩。
- C3k 最终验证：定向 29 passed；全量 **1814 passed + 81 subtests passed**；16 个变更 Python 文件 Ruff、309 个文档本地链接、摘要和 diff 校验通过。Git 未提交；页面自动化被 file URL 策略阻止，不标交互验收通过。
- Runtime 的合同、Evidence、Guard、审批、事务、验证/补偿及本地 C1/C2 分支原型已实现，生产/真实设备资格未完成。
- 清理提交 c2ebd78、14baa0a；重复检查点已收敛，原始失败保留在[历史](PROJECT-HISTORY.md)。
- 逐份审查旧 33/33 的六份源文，发现完成引文错位和必要性/充分性混淆，旧结果未修改。
- 新实测证明旧单条件补全会降低可用性；已退出推荐流程。新增 require_any、可读表达式、联合分歧定位及显式未激活编译修订，不新增执行器。
- 新六例/46 场景：构造 **2/6、30/46**；条件阶段后 **5/6、35/46**。其中一份结构失败、11 场景未运行；四个可执行片段案例及一个正确停止匹配。辅助修订另报，不回填首次结果。[证据](FLOW-SEMANTIC-TRANSFER.md)。
- 构造 Schema 现在对齐既有 Quote 约束，排除结构无效的 `---` 等引用。结构有效仍不等于语义支持。
- 可选紧凑读取入口在同源开发修订中通过 4/5、34/45；修复工单但误拒绝副本案例，不能按 Oracle 与其他路径择优拼接。全量 1792 测试及 81 子测试通过；第二批报告已从隔离源码快照零调用回放，结果逐字节一致。
- 公开语料新增 `inert-text`、脚本原文索引、历史仓库排除与固定种子抽样。旧 269 个尝试中 35 个因脚本表面被排除，已全量列为开发补充；新元数据接口遇 HTTP 429。排除 199 个已尝试仓库后，旧缓存仅余 22 Skill／15 仓库且同属 finance analysis，不能冒充跨领域批次。[细节](PUBLIC-TRANSLATION-BATCH.md)。
- 脚本补充实际保存 **33/35 Skill、24 仓库、321 份隔离文本**，另 1 份二进制只留摘要。20 份进入转译研究语料，13 份作格式鲁棒性；新增 Runtime-ready 与模型调用均为 0。和旧库共 133 package ID／93 仓库，不是 133 个通过成绩。[可点击原文的索引](../artifacts/translator-v2/script-recovery-20260908/library-v2/skill-library.html)。
- C3j 最终回归 **1805 passed + 81 subtests**，定向 20 passed；修改代码 Ruff、diff、文档链接和摘要校验通过。未提交 Git；历史模型证据未修改。

### To-do 与边界

| 顺序 | 下一步 | 验收 / 限制 |
|---|---|---|
| 已收口 | 回归、源引用缺口、可回放报告 | 修改文件 Ruff / diff 检查通过；全仓仍有原先 224 项 lint，不混入本轮清理 |
| 入库完成 | 60 个冻结候选、旧脚本补充库 | 新库 53/60 保存；失败保留，采集限流已解除；搜索类别/静态包检查不能证明领域独立性/语义成功 |
| 入口完成 | 源文、分页、显式引用补取、宿主诊断 | 原文可完整拼回，旧证据不覆盖；不是跨页语义编译或完整引用闭合 |
| 绑定/接线完成 | 版本化嵌套数据、源锚定 Tree、共享执行器 | 有离线绑定 CLI、显式宿主只读 smoke 和 9B 候选入口；完整模型转译及语义准入未完成，不代表整 Skill 编译完成 |
| 任务档案完成 | 四份固定材料的具体任务、义务/问题与宿主需求 | 有源锚定开发审阅，不是独立 Gold 或全部源义务闭合；未读附件保留 |
| 局部实现完成 | 本地 Netdata 隔离适配器、info/query fixture、有界列解码 | 本页计数与异常检查已跑通；无原厂互操作、全窗口/保留期证明，不把枚举输出当生产隐私证明 |
| 已接入、首次停止 | 渐进 9B 源文请求与结构化 Tree 构造入口 | 2 次真实调用停于上下文预算，未生成候选；不能标记转译通过或直接进入 Runtime |
| 已修复输入组织 | 区分字节/token 指标；UTF-8 分页、源块选择、依赖记录、原文回填和精确去重 | 资源代理不是 tokenizer 认证；不会抄错引文不等于选对语义依据。四源仅静态分页，最新模型仍只验证一个已知 Skill |
| 边界机制完成 | 分开文本交付/语义审核、源操作/宿主声明、离线构造/执行门禁；联合窗口候选或缺口报告 | 真实 9B 新两版仍首轮拒绝，未实测触发联合回读恢复；更早停止不是可用性提升 |
| 诊断/语法机制完成 | 可选源义务审阅、输入隔离、字面缺口检索、表达式生成 Schema、独立任务对齐 | 前置审阅收益未证实；不默认启用；完整候选有参数/别名缺陷，仍不可执行 |
| 参数/定位机制完成 | 宿主 Schema 引导参数 + 显式值来源 + 代码分配稳定别名 + 独立错误定位 | 一个已知源的形状/别名错误消除；模型仍重复调用、来源与操作模式错误，不是语义成功 |
| 模式机制完成 | 显式宿主互斥键合同 + 逐叶参数来源 + 源块行动轨迹 | 只证明已声明组合约束；动态对象不隐式放行；新增宿主信息必须独立标识，模型整 Skill 仍失败 |
| 规划／绑定机制完成 | 不可变计划、分槽参数来源、动态引用导航、冗余不可达终止规范化 | 真实 9B 合成局部流程通过 10 个执行路径检查；不是公开 Skill 或整 Skill 成功 |
| 下一步语义收敛 | 逐项义务覆盖、截断边界与 L1 交接定位；公开源操作／前置条件／宿主替代范围 | 当前 remaining 仍遗漏职责；Netdata 不得执行；保留失败，不能靠删除未决项或叠加提示语宣称通过 |
| 对齐完成后 | 冻结任务/私有 Oracle，先小批后批量转译 | 9B；固定主候选路径、无自动重试；不把同批调参称泛化 |
| 正式转译门禁 | ≥3 不重叠 cohort、≥50 Skill、≥15 仓库、≥8 领域、≥600 case | 数量只是必要条件，质量阈值不下调 |
| 门禁通过后 | 大规模 L0→Runtime / DSH 配对 | **仍未解锁**，有限布尔区域通过不代表整 Skill 高准确转译 |

源引用仍有角色错位，完整源审查未完成。两批 12 份 Skill 均由同一开发助手构造，不是公开未见集或真人 Gold。现有 100-Skill 开发库也不能改称未见集。见[代码边界](../evaluation/README.md)、[纠偏计划](TRANSLATION-CORRECTION-PLAN.md)。

### 保留但不推进

ES-P1-Private-Human = skipped_retained_open；GPT 对照暂缓，使用 qwen3.5:9b。生产身份、供应链、治理、HA/DR、WORM、SLO、Hermes/A2A 为 frozen_future_engineering。权威原则见 [ENSUREDSKILL-PROTOTYPE](ENSUREDSKILL-PROTOTYPE.md)。

## English

Updated 2026-09-09. **C3w closes the real-9B plan/fill/original-compiler/synthetic-execution mechanism; public-source semantic translation remains open.** C3i–C3p were locally committed as `f0499ec`; C3q–C3w are uncommitted. No push/master merge or default routing/activation changes.

C3w preserves all eight experiments/twelve real 9B calls. Five known-Netdata attempts remain blocked; thinking and readable schemas do not solve source-duty/action confusion. The pre-existing synthetic wiring control progresses from redundant-terminal rejection, through invented-argument rejection, to a two-read conditional compiled from three model calls (42.549 seconds, 10,465 input/853 output tokens). Its initial planning request is identical across three versions. Dynamic-reference guidance, early origin checks and recorded dead-terminal normalization do not alter Runtime authorization. After a digest-bound developer-agent review, ten local branch/denial/abnormal-stop checks pass with zero network/script/effect calls. Counter-result decisions are delegated to L1, and remaining-duty accounting is incomplete; this is not whole-Skill acceptance or a public-Skill accuracy score. Final validation adds 49 mechanical tests: **309 targeted passes (25.14 seconds), 2292 full-suite passes plus 81 subtests (179.83 seconds)**, lint on 21 related files, links and whitespace checks. The earlier 2290-test run is preserved; equal-constant origin caching is fixed without changing the three recorded wiring requests or derived outputs. All eight original-version isolated replays are byte-identical; 127 new files are bound and 275 prior files remain unchanged. Next improve source-duty accounting and precise partial-region handoffs before public-source generalization or broad Runtime A/B. See [current design and complete results](PLAN-FIRST-AUTHORING.md).

C3v adds digest-bound, closed-key mode constraints, leaf origins and block-grounded action traces without a new executor. Two new 9B calls both generate candidates but neither compiles or invokes providers/scripts. v12 has two identical discovery reads and failed exact action quotations (101.794 seconds, 8,115/1,165 tokens). v13 rehydrates source blocks but again emits eight identical reads, fourteen invalid origins, seven unresolved entries and no terminal (215.441 seconds, 8,113/3,228 tokens). Mode mixing stays zero; **overall availability has not improved**. Forty new mechanical tests and 300 targeted checks pass in 24.96 seconds; **2243 full-suite tests plus 81 subtests pass in 255.63 seconds**, with Ruff on nineteen related Python files, tracked/untracked whitespace and document-link checks. Both isolated source-snapshot replays are byte-identical with zero calls; 39 files are bound and 236 older bound files are unchanged. See [current implementation, results and next work](HOST-OPERATION-MODES.md).

C3u makes one fresh 9B call in 214.388 seconds (7,787/3,452 input/output tokens). Same-source/task host-schema error messages drop 56→0 and duplicate aliases 7→0, but eight identical reads, seven unresolved entries, eight invalid origins and no terminal remain. Zero regions compile and no provider executes. Static host inspection reveals a discovery/query exclusivity rule missing from its simple schema. This is not Skill success or an overall performance improvement. There are 38 new mechanical tests and 333 targeted passes; **2203 full-suite tests plus 81 subtests pass in 178.36 seconds**, with lint on seventeen related Python files, diff and document-link checks. The original model report replays byte-identically with zero calls; final diagnostics are separately saved. The summary binds 21 files and 215 preceding bound files are unchanged. See [current evidence](CATALOG-DIRECTED-AUTHORING.md). Next expose operation modes and construct a source-grounded operation skeleton with leaf-level origins; broad Runtime evaluation stays locked.

C3t records seven development variants and twelve actual calls on one known Skill, not unseen generalization. The separately aligned inactive-authoring task must not be pooled as same-task improvement over the old Cloud task. Final direct generation takes 151.381 seconds (8,099/3,380 input/output tokens) and produces an unresolved candidate with eight repeated aliases, object-for-string arguments and empty required filters. All variants have zero compiled regions and zero provider/source-script execution. **295 targeted tests (14.27 seconds); 2165 full-suite tests plus 81 subtests (177.20 seconds)** pass. Fifteen related Python files pass lint; seven isolated zero-call replays are byte-identical. The summary binds 112 files; an initial premature-document-link failure is retained and repaired, and 103 preceding evidence files remain unchanged. See [current evidence and localized failures](SOURCE-OBLIGATION-AUTHORING.md). Next prioritize catalog-directed arguments, value origins and deterministic aliases before new-source testing; broad Runtime evaluation remains gated.

C3s v4/v5 make one call each (44.280/27.635 seconds; input/output 7,868/335 and 7,765/83 tokens). Both produce unverified gaps, not candidates or provider calls. Source instructions do not prove live credential absence; missing execution credentials alone should not prevent offline candidate authoring. Joint-window recovery is mechanically tested but not exercised by these early-stopping model runs. Final **273 targeted tests; 2132 full-suite tests plus 81 subtests (176.33 seconds)** pass, with lint, links, digests and byte-identical isolated replays. See [mechanisms, evidence and next work](SOURCE-DECISION-AUTHORING.md). Next type obligations by phase/evidence origin and examine premature abstention across varied sources; larger Runtime evaluation remains locked.

C3r reconstructs four known bundles losslessly and records three same-Netdata development revisions (ten actual model calls). V1 preserves a quote line-wrap failure; v2 preserves a duplicate-context budget failure. V3 retains all fourteen notes and exact source coverage while reducing the pending byte proxy 50,403→36,971. Its fresh six-call/535.562-second run passes citation and resource checks but alternates between root and recipe, producing zero candidates/provider executions. This is not a semantic accuracy or Runtime score. Final validation: **246 targeted tests; 2105 full-suite tests plus 81 subtests (278.90 seconds)**; all three isolated zero-call replays are byte-identical. See [workflow and evidence](SOURCE-LEDGER-AUTHORING.md).

The C3r next-step plan (retrieval/review separation, explicit host mappings and bounded convergence) was subsequently implemented in C3s/C3t above. Do not fake semantic approval or force unsafe candidates to stop a loop. The current argument-construction plan above supersedes historical next steps; broad Runtime evaluation remains gated.

C3q preserves frozen inputs, implementation/environment/model bindings, exact citations, first failures and no-retry checkpoints. The model requested the linked recipe and then a Cloud protocol page. Two calls cost 44.426 seconds and 13,870 input/259 output tokens. The next 45,057-byte request exceeded the experiment's 36,000-byte limit and was not sent; this is not proof of exhausting the model's token context. Two of twenty-eight pages were submitted, zero candidates were produced, and no Runtime/provider/source script executed. A category/severity wording error in retrieval is recorded without inferring query correctness. Twenty-nine new mechanical tests and 211 targeted tests pass; the full suite passes **2070 tests plus 81 subtests in 161.34 seconds**. Lint on both added Python files, 279 local links, fourteen bound artifacts, diff and byte-identical isolated replay pass. See [results and scope](PROGRESSIVE-STRUCTURED-AUTHORING.md).

C3q's then-next input-organization work is now implemented and measured in C3r above. Summaries remain navigation, not authority; both first failures and later revisions are retained. Candidate construction/semantic closure remains open, and C3q/C3r changes remain uncommitted.

C3p preserves original field names, bounded all-row decoding, dominance/freshness and the candidate-only Effect boundary. The host performs two actual local callbacks with explicit identity/scope and domain checks. Its three-row page yields crit 1/warning 2; no raw receipts are persisted. This is developer wiring, not vendor interoperability, full-window coverage or a 9B score. See [scope and use](NETDATA-ISOLATED-VALIDATION.md) and [evidence](benchmarks/netdata-isolated-summary.json).

Final C3p validation: **190 targeted tests; 2041 full-suite tests plus 81 subtests in 161.07 seconds**. Lint on 36 changed Python files, 468 local links, diff and source digests pass. Six isolated replay files are byte-identical, nine legacy wiring files are unchanged, and historical C3n bindings verify against their archived source. No historical Oracle changes; Git is uncommitted.

C3o records 35 declared obligations, eleven findings and twelve unbound host requirements (nine tools, two contexts, one policy) across four developer-authored requests, not exhaustive coverage or implemented tools. Exact review ranges/citations are checked; full source pages and future model inputs exclude developer review decisions. One pinned Netdata protocol supplement brings its bundle to seventeen files/202,930 characters/twenty-eight pages. Source tensions, inconsistent illustrative indices, dynamic decoding and time units remain explicit. No model/provider/script execution or semantic score. See [task dossiers](TASK-SOURCE-ALIGNMENT.md) and [evidence](benchmarks/task-alignment-summary.json).

Final C3o validation: **122 targeted tests; 1975 full-suite tests plus 81 subtests in 161.82 seconds**. Lint on 31 changed Python files, 439 local documentation links, twenty byte-identical isolated replay files and diff checks pass. Requirement-kind separation leaves all four source bundles, tasks and model inputs unchanged; prior dossiers/overlays and C3n evidence are retained. This is not semantic accuracy. Git remains uncommitted.

C3n reuses the existing qualifier, runner and read gateway for nested data, explicit pointers, branches and candidates. It preserves exact citations/Tree-digest binding, dominance, nested resource scopes, control-evidence age checks, copy isolation and redacted provider errors. One developer fixture performs **two local read callbacks, zero writes and zero model calls**; additional variants cover safe completion and refusal. Seven original developer graphs retain v1 packets/digests, and the positive demo replays byte-identically. These are wiring/compatibility checks, not public-Skill accuracy or device timestamp authentication. See [the full process](STRUCTURED-FLOW-WIRING.md) and [validation evidence](benchmarks/structured-flow-summary.json).

The full execution demo is still developer-wired, not a model answer. The new candidate entry does not close page/retention/semantic gaps; large Runtime evaluation stays locked.

Final C3n validation: **227 targeted tests; 1949 full-suite tests + 81 subtests passed in 226.55 seconds**. Lint on 29 changed Python files, 418 local documentation links, nine byte-identical artifacts from an isolated source-overlay reconstruction and diff checks pass. Prior bound evidence is unchanged. Initial fixture-construction failures remain recorded; mechanical pass counts do not become semantic metrics. Git is uncommitted.

C3m adds versioned nested schemas and literal/reference/object/array bindings, retaining keys, constraints and bounded local references. Sources and final arguments are validated on every materialization. The previous synthetic catalog's two schemas move from 0/2 old-flat compatibility to 2/2 new-profile compatibility; offline CLI checks nested arguments, explicit output indexing, enum failures, empty arrays and catalog drift. This is not semantic Skill accuracy, a new executor or contract activation. Model/provider calls remain zero; see [scope and evidence](STRUCTURED-DATA-BINDING.md).

Final C3m validation: **94 targeted tests; 1908 full-suite tests + 81 subtests passed in 213.70 seconds**. Lint on twenty-three changed Python files, 390 local document links and twenty-one bound evidence files pass; forty prior evidence files are unchanged. The initial full suite detected sourceState drift during concurrent document edits; that negative result remains recorded, followed by a passing stationary-worktree rerun. The freeze gate was not modified. Git is uncommitted.

The new corpus CLI produces inert source bundles, lossless pages, reference-role/path diagnostics and optional raw-host-schema diagnostics. Four fixed sources were exercised: OpenMontage retains 65,778 characters; Netdata recovered eight explicit same-commit files, reaching sixteen files/146,532 characters/twenty-three pages. A padded Git size-column parsing failure and its successful retry are both retained; supplemental files are not additional sampled Skills. Reading the full Netdata wrapper exposes authentication/cache effects, general HTTP methods and conditional arguments. Complete reference closure, structured L0 execution and semantic compilation remain pending. No new model calls or semantic score; see [intake evidence and next steps](TRANSLATION-INTAKE.md).

Final C3l validation: **81 targeted tests; 1851 full-suite tests + 81 subtests passed in 156.60 seconds**. Lint for eighteen changed Python files, document links, forty bound evidence files and diff checks passed. Offline CLI source/host-diagnostic replay and overwrite rejection were verified. Changes are uncommitted; no third-party source was executed. These are mechanical regressions, not 1851 translated Skills.

Twelve saved queries yielded 237 unique source URLs. All 60 frozen candidates/44 repositories were processed: **53 Skills/38 repositories** saved, seven exclusions retained. The parser classifies 29 format-qualified inputs and 24 variants, not semantic successes. A bound report keeps all outcomes and distinguishes processing completion from an importer acceptance target. Four root-entry reviews locate source, schema, template and whole-Skill scope problems; references and complete task/host alignment are still pending. No new 9B translation score. See [results and next steps](PUBLIC-TRANSLATION-BATCH.md).

Final C3k validation: 29 targeted tests; **1814 tests + 81 subtests passed**. Changed-code lint, 309 local documentation links, evidence hashes and diff checks passed. Changes are uncommitted. Browser automation was blocked by the file URL policy, so interactive acceptance is not claimed.

The old six sources were audited; citation and necessity/sufficiency gaps remain. New measurements show unary guard addition can damage correct paths. The recommended path now uses readable Boolean extraction, deterministic evaluation and explicit inactive revisions. Compiler quote constraints are mirrored before generation, without proving entailment.

The new six-package / 46-scenario batch improves from **2/6 cases, 30/46 scenarios** to **5/6, 35/46** after condition extraction. A structural failure leaves eleven scenarios unrun. The optional compact front end subsequently matches **4/5, 34/45**, fixing helpdesk but falsely rejecting storage; do not cherry-pick paths using the oracle. Repairs are separate in the [report](FLOW-SEMANTIC-TRANSFER.md). Both batches are assistant-authored, not public holdouts or independent Gold; complete source/citation review remains open. Full regression passes 1792 tests and 81 subtests; isolated zero-call replay is byte-identical. Changed-file lint passes; 224 pre-existing repository lint issues remain.

The prior C3j tools retain inert script evidence and exclude attempted repositories. All 35 old script exclusions entered development recovery. Its HTTP 429 and single-query reserve were historical limitations, now superseded by the new acquisition above. Source/task/host alignment, private oracles and new translation results remain pending. The unchanged gate requires at least three cohorts, 50 Skills, 15 repositories, eight domains and 600 cases plus quality thresholds. Large Runtime/DSH evaluation stays locked; production engineering and human-review deferral remain unchanged.

Recovery saved 33/35 Skills from 24 repositories with 321 isolated text sources and one binary hash-only source. Twenty are translation-research inputs and thirteen format-robustness inputs. Combined known inventory: 133 package IDs/93 repositories, not successful translations. No new model calls or Runtime-ready packages.

Final C3j regression: 1805 tests and 81 subtests passed; twenty targeted tests, changed-code lint, diff, documentation links and evidence hashes passed. Git changes remain uncommitted and historical model evidence unchanged.
