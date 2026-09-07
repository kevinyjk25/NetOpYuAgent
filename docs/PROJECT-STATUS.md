# EnsuredSkill 项目进展 / Project Status

## 中文

### 当前阶段

**当前主阶段：L1→L0 转译泛化门禁。**

#### 2026-09-07 Git 阶段检查点

C3c–C3h 代码、测试及开发依赖已提交到本地 `dev`：`1c5fc7e`。本节及配套文档/评测摘要随独立文档提交保存；未推送、未合入 master。以下历史条目的“未提交”描述保留其当时状态，以本节为当前准。已有验证为 **1288 tests + 81 subtests 通过**，不代表语义质量验收完成；新版职责协议仍待真实 9B 验证。

冻结的 `evaluation/flow_lean_pilot.py` 末尾一行空行保留，以避免改变已记录实验指纹；暂存检查除此格式例外通过。原始实验制品仍在 Git 忽略的本地 `artifacts/`，本次仅提交可追溯摘要，不包含完整原始证据包。

#### 2026-09-07 C3h 职责映射：离线修复完成，新模型批次待验证

- [x] 新独立协议把源句拆成精确引文子要求，明确操作/分支、输入/返回结构、读权限、错误传播、业务前置、解释/权限限制及缺能力停止。按真实规则/节点职责机械拒绝错配；条件节点遗漏提供精确指针，不猜补。
- [x] 保留旧完整源审查，追加逐句分解完整性与逐子要求类型/目标审查。错分类、漏拆分、否定词截断仍可能发生，不能把兼容表当作语义证明。代码块不执行、未解决和父问题保留、审查不授权。
- [x] 新增 **75 项开发回归**；含原精简映射/批次共 **98 项定向回归通过**，不是新增 Skill 或模型成功率。提供离线 request/compile/assess CLI，拒绝覆盖已有输出；不修改冻结协议/模型回答/旧报告，不切换产品或执行器。
- [x] 全量 **1288 tests + 81 subtests 通过（131.58 秒）**；Ruff/diff、旧精简批次与旧完整双阶段含审查报告重放一致。全量回归包含原有隔离执行测试，其耗时不是模型或 Runtime 性能指标。
- [ ] 冻结独立新版真实 9B 完整双阶段小批，分别测职责错配/漏分类/完整原意/正确停止/误接受及两步全部成本。当前 **0 次新模型调用**，没有新版准确率或时延结论，不能用这些回归更新前轮 3/4、语义全阻断结果。

见[设计、离线使用与边界](FLOW-RESPONSIBILITY-MAPPING.md)。仍在 C3h；C4–C6 保持门禁，未提交/推送。历史“下一步”不覆盖本节。

#### 2026-09-07 C3h 精简版：生成负担下降，语义质量门禁仍未通过

- [x] 编译器拥有已知锚点、节点来源汇总及候选解释，模型仅选择真实目录目标与必要额外来源。禁止自由说明/改业务，保持完整源、节点/条件/参数、片段/规则审查；缺覆盖、错误引用、未解决事项和旧审查失效仍阻断。
- [x] 新冻结完整两步批次：**4 已知流程 / 1 工具 / 0 公开 Skill / 8 次真实 9B 调用**，无历史树替换、修图或重试。第一步 4/4 结构合格，第二步 3/4；反向分支遗漏 `/steps/1` 条件节点来源，未自动猜补。
- [x] 3 项完整同一助手审查 **128 声明：103 supported / 25 insufficient**，0 审查支持的未激活流程，全部 blocked。仍有数据背景误选操作、脚本前置误选读权限、形状检查冒充数据性质、复合句覆盖不全及检查/错误传播混用；这些是候选映射缺口，不是实际网络故障或 Runtime 执行器不安全的证明。
- [x] 对照上轮的观察成本：第二步输出 **9,919→2,619 token（−73.6%）**，输入 **19,520→25,172（+29.0%）**；两步总 token **41,436→39,788（仅 −4.0%）**。第二步 POST 811.20→413.99 秒，两步 932.05→577.64 秒。负载/载荷/预算不同且部分并行回归，不作因果性能或准确率推断。
- [x] **23 项新增回归**；全量 **1213 tests + 81 subtests** 通过（165.05 秒）；Ruff/diff、新报告重放、完整检查点重入及旧 C3h 完整报告一致性通过。模型批次 Runtime/业务工具/脚本/写执行为 0，原始回答/失败/基线保留。
- [ ] **继续 C3h 语义职责匹配，不进入 C4**：优先区分业务前置、读取授权、结构检查、错误传播、数据解释及缺能力停止，补条件节点/复合句完整覆盖的独立反例；分开业务图错误与细粒度来源标注缺口，不能自动放行。目录/Schema 重复输入为次级优化，不得为省 token 删源义务或审查。

生成成本改善不等于完整转译成功。C4–C6 及大规模 Runtime 仍受门禁约束；同一助手不是独立 Gold，声明数不是准确率。C3c–C3h 改动尚未提交或推送。见[设计、正反结果与复现](FLOW-LEAN-MAPPING.md)、[可追溯摘要](benchmarks/flow-lean-c3h-summary.json)。历史“下一步”不覆盖本节。

#### 2026-09-07 C3h：完整双阶段实现及首轮评测完成，质量验收未通过

- [x] 新增精确偏移的标点分段、逐片段必填账本与多来源节点映射；完整段落投影仅作原文保留，不替代模型片段判断。代码保持惰性，旧的完整源/操作/条件/参数审查继续保留。
- [x] 冻结源→新 9B 流程→编译器节点目录→新 9B 映射的完整链路。两阶段分别保存请求/回答/状态/成本；失败不自动修图或重试，完整检查点重入不重复调用。父流程、问题和执行语义不允许第二步修改。
- [x] **4 已知流程 / 1 工具 / 0 公开 Skill，8 次真实调用**：第一步 4/4 结构合格；第二步 2/4 合格。反向分支 44 记录中 31 条缺自身锚点，审批 30 记录中 15 条缺自身锚点，均结构阻断。
- [x] 两个合格映射同一助手完整审查 **68 声明：53 supported / 15 insufficient**。直接读混淆检查/错误传播、缩窄无授权范围；脚本案把缺能力停止映射为输入/权限/返回校验，并有跨来源缺引用和未知脚本只读断言。**0 个审查支持的未激活流程，全部 blocked**；结构失败项不伪造语义评分。
- [x] 全部 POST **932.05 秒、30,536 / 10,900 token**，第二步占 811.20 秒。失败/等待计入，预检/审查/测试不计；部分回归并行，不作因果性能比较。不是未知集泛化、C3g 控制变量准确率对比或可用性提升证明。
- [x] **29 项新增回归**、全量 **1190 tests + 81 subtests** 通过（170.75 秒）；Ruff/diff、新报告及旧 C3f/C3g 重放通过。模型批次 Runtime/业务工具/脚本/写执行为 0；没有改动旧回答或基线。
- [ ] **继续 C3h 精简，不进入 C4**：让编译器绑定条目已有锚点，减少模型重复引用及自由扩写；仅生成必要的额外来源/处理类型/目标，保留所有源蕴含审查。先离线验证反例，再另冻完整新批同时检查质量和成本，不能补改当前输出刷通过率。

链路实现完成不等于 C3h 质量完成；C4 异质公开 Skill、C5 未知集与 C6 Runtime 价值评测仍按顺序受门禁约束。原始模型制品在本地，Git 摘要不是完整原始证据包。C3c–C3h 未提交或推送。见[设计、结果与复现](FLOW-TWO-PASS-TRANSLATION.md)、[摘要与精确失败位置](benchmarks/flow-two-pass-c3h-summary.json)。历史“下一步”不覆盖本节。

#### 2026-09-07 C3g：源约束账本与受限映射修复完成，完整转译质量门禁未通过

- [x] 分离完整源存档、业务目的、约束记录与执行保障；每个非标题源段必须有账本项，操作/文档/宿主规则/流程引用/未解决采用互斥结构。源和映射均进入逐项审查，不以文字保留冒充规则已执行。
- [x] 可选约束首次 9B 批 **1/4 结构合格**：直接读取仍遗漏约束，其余混用文档与执行引用。强制账本新批 **1/4 合格**：3 项使用工具名代替节点路径；缺脚本例 18 声明支持但仍因真实缺能力 blocked。原始回答均保留，不修复重计首次成功。
- [x] 新增受限第二步：复用 C3f 已封存的真实 9B 树，编译器枚举真实节点，模型只选来源与约束映射，禁止改工具/参数/条件/顺序/终态/原问题。父提案、执行投影和元数据摘要绑定审查，修改说明或别名也不能复用旧审查。
- [x] 第二步 **4/4 结构合格**；同一助手审查 **98 声明：87 supported / 11 insufficient**。直接读存在保障过度陈述，分支例存在引用错配和新增报告语义；审批/脚本两案的停止及限制映射获支持，但缺能力事项原样保留。**0 个审查支持的未激活流程，全部 blocked**。此为辅助元数据结果，不是新的完整首次转译。
- [x] 三批共 **12 次 9B 调用、651.79 秒、46,625 / 5,572 token**；仅 4 个已知流程、1 工具、0 公开 Skill。第二步自身 228.08 秒，父 C3f 生成成本另计 156.61 秒；不以不同协议/声明比例宣称准确率或因果性能改善。
- [x] **68 项新增定向回归通过**；最终全量 **1161 tests + 81 subtests** 通过（108.08 秒，模型批次完成后）；Ruff、diff、新三批与 C3f 证据重放一致。本轮模型批次 Runtime/业务工具/脚本/写执行为 0。
- [ ] 下一步 C3h：补逐子句/多来源绑定及工具说明扩写用户步骤的开发反例；再冻结从源文到流程到映射的完整新批，报告两步成本、完整保真、正确停止与误接受。不得修改本次冻结回答刷通过率。

这是**映射协议的可校验性与定位能力修复**，不是泛化证明。逐行源段、长度/节点上限和同一助手审查的局限仍在；12 个公开 Skill 整流程及大规模 Runtime A/B 不解禁，不进入生产工程扩展。C3c–C3g 改动尚未提交或推送 Git。见[设计、真实失败、成本与复现](FLOW-SOURCE-CONSTRAINTS.md)、[第二步摘要](benchmarks/flow-mapping-c3g-summary.json)。以下历史“下一步”不覆盖本节。

#### 2026-09-07 C3f：新版无答案正向小批完成，结构改善但完整源审查未通过

- [x] 源/合同沿用 C3d 的 4 个已知流程；按 C3e 宿主能力约束，冻结新版请求、代码、依赖版本和模型后，各首次调用一次 9B。未提供目标 JSON、参考树或修图重试。
- [x] **4/4 完整结构/合同合格**（C3d 同源批 0/4）；直接读取、反向条件的极性/引用/终态正确；缺审批和脚本例保留正确停止位置及真实事项。这个变化不等于公开 Skill 泛化或成功概率，也不能唯一归因于正文 Schema 或词表约束。
- [x] 同一助手逐项审查 **62 声明：49 supported、0 contradicted、13 insufficient**，4 项仍 blocked。每例共享数据/权限限制缺少显式映射，审批首读还引用标题；源双语覆盖与 purpose 声明重叠，不能据声明比例计算准确率。
- [x] **156.61 秒、11,016 / 981 token**；模型阶段同时有回归负载，不作因果时延比较。0 批次 Runtime/业务工具/脚本/写执行，0 辅助修订，旧 C3d/C3e 证据不变。
- [x] **14 项新增回归**及最终全量 **1093 tests + 81 subtests** 通过（93.06 秒，模型完成后运行）；Ruff/diff/报告重放通过。逐项原始提案、完整审查和摘要均保留。
- [ ] 下一步 C3g：将业务目的、约束保留和操作映射分开，显式区分“已保存原文”“有映射”“可确定性校验”“仍不支持”；不得用文本携带冒充已执行约束。先补跨段/标题/共享限制回归，再冻无答案批；不删源要求、放宽审查或重跑当前批。

见[结果、定位与复现](FLOW-BOUNDED-FORWARD.md)、[可重算摘要](benchmarks/flow-bounded-forward-summary.json)。这是 4 个已知流程、1 工具、0 公开 Skill；12 个公开 Skill 整流程与规模化 Runtime A/B 仍受门禁限制。当前 C3c–C3f 改动尚未提交 Git。历史条目的旧“下一步”不覆盖本节。

#### 2026-09-07 C3e：协议探针完成，宿主能力收口实现；新版正向转译待验证

- [x] 独立冻结 4 个明确给答案的构造探针 × 3 组接口/消息条件，12 次首次 9B 调用，不计入 Skill/转译成功数。Schema-only 与 Schema-visible 各 4/4 精确复制，普通 JSON-visible 2/4；两个失败抄入输入封套，其中一个截断，均未修复重试。
- [x] 确认该环境能表达本次基本终态/条件/空分支，不支持“解码器完全不支持 end”的解释；不能据此认定正文未附 Schema 是 C3d 唯一根因，或证明所有递归组合兼容。
- [x] 生成 Schema 按实际宿主删除不可用读/Effect 构造器，有能力时枚举真实工具/目标 ID；新请求正文与接口共用 Schema，输出仍须通过独立 Schema 检查和旧编译/合同门禁。不依据用例名修图，不授予写权限。
- [x] 成本 **268.97 秒、14,438 / 2,735 token**（包括失败回答，测试并行，非因果性能比较）。业务工具/Runtime/脚本/写执行均为 0。旧 C3c/C3d 证据保持一致。
- [x] **23 项新增回归**；全量 **1079 tests + 81 subtests** 通过（147.50 秒），新增合法 Effect 断言后 23 项再次通过；Ruff/diff/报告重放通过。`jsonschema` 只加入开发依赖。
- [ ] 下一步 C3f：另冻不给答案的宿主收口正向小批，完整保留首次输出，分别评估合法结构、语义、真实缺能力停止和误接受。当前新请求尚无真实转译结果；12 个公开 Skill 整流程及大规模 Runtime A/B 不解禁。

见[诊断、能力边界与用法](FLOW-TREE-PROTOCOL-CANARY.md)、[探针摘要](benchmarks/flow-tree-canary-summary.json)。当前 C3c–C3e 改动尚未提交 Git；以下历史阶段不覆盖本节。

#### 2026-09-07 C3d 正向小批：已完成，生成质量门禁未通过

- [x] 新增层级树 authoring 请求与 freeze/run/report 命令；原文/宿主合同/Schema 输入，不提供手工树答案。保持旧 C3b/C3c 协议和原始证据不变。
- [x] 同一 4 个已知开发流程各首次调用 9B 一次，3 个树可解析、**0/4 完整结构/合同合格**。终态被写成 Effect 目标；审批案重复至 2,200 token 截断；脚本案先读后停并有不可达代码。反向条件的极性/引用局部正确不等于整个提案可用。
- [x] 冻结摘要、原始响应、失败状态与文件 receipt；不修复/重试失败输出。无合格树，因此逐项合格提案语义审查数量为 0，不伪造支持率；当前助手对失败的诊断不是独立评测。
- [x] 请求共 **261.69 秒、4,652 / 3,584 输入/输出 token**。0 Runtime/业务工具/脚本/写执行；测试与模型并行，不用耗时作因果性能比较。
- [x] **13 项新增协议回归、73 项定向回归**通过；全量 **1056 tests + 81 subtests** 通过（230.10 秒），Ruff/diff、新报告与旧 C3c/离线树示例重放通过。
- [ ] 下一步 C3e：先独立冻结最小 Schema/结构化解码兼容性 canary，再按宿主合同收窄可生成构造器/目标 ID；明确区分协议接线测试与源文转译，禁止答案规则修图。通过后另冻开发批，不能重跑本批刷通过率。

见[结果、失败定位与复现](FLOW-TREE-FORWARD-PILOT.md)、[摘要](benchmarks/flow-tree-9b-summary.json)。这是 4 个已知流程、0 公开 Skill，不是新增泛化证据；尚不能将失败唯一归因于模型或解码器。12 个公开 Skill 整流程和规模化 Runtime A/B 继续暂停。C3c/C3d 阶段改动尚未提交 Git；以下旧“下一步”保留作历史，不覆盖本节。

#### 2026-09-07 C3d：层级流程编译及离线验证完成，9B 正向验证待开展

- [x] 新增有界顺序/if_equal/显式终态 authoring 树；作者提供操作、条件和读结果别名，编译器生成节点 ID、顺序/真假/汇合边与必经前序步骤。不新增执行器，不替模型修正业务极性。
- [x] 复用既有源引用、宿主合同、类型、DAG 与 read Runtime。禁止自引用、未完成读取、越作用域结果、死代码及隐式成功；分支结果不向外自动合并，单 Effect 只作终态候选。
- [x] 树→图位置 origins 及整树进入审查摘要；仅修改别名但图不变也会令旧审查失效。新增手工 JSON 示例和 schema/example/compile/assess 命令，输出均未授权。
- [x] **60 项新增回归**通过；7 类手工流程与 20 个固定种子树在两种库存数据上，对照独立测试解释器和实际本地文件读取的旧 Runtime，检查结果/读取顺序/文件不变；权限失败在 Provider 调用前阻断。它们不是新增 Skill 或语义 Gold。
- [x] 全量 **1043 tests + 81 subtests** 通过（86.60 秒），Ruff/diff 校验通过；C3b/C3c 和新编译示例重放一致。本阶段新 LLM 调用为 0，写操作为 0；已有 C3c 未提交改动完整保留。
- [ ] 下一步 C3d 正向小批：只给原文、宿主合同和层级 Schema，让 9B 生成树，不提供本轮手工树答案；另行冻结协议，分别计结构接线、业务保真、字段事实和辅助成本。不能把离线编译正确性当作模型准确率提升。

见[设计、限制与用法](FLOW-TREE-COMPILER.md)、[手工示例](../examples/read-flow/flow-tree.json)、[可重算编译摘要](benchmarks/flow-tree-example-summary.json)。网络业务事实含义混用仍待解决；12 个公开 Skill 整流程和大规模 Runtime A/B 继续受门禁约束。当前 C3c/C3d 改动尚未提交 Git。

#### 2026-09-07 C3c：源文约束实现与两协议诊断完成，质量门禁未通过

- [x] 只改实验 authoring/review 层：传入既有宿主合同业务说明，用途/终态采用源文投影，分支分别绑定真假来源；声明前置依赖须在所有路径成立，映射变更使旧审查失效。未新增执行器、未改 Runtime 写门禁。
- [x] 摘录协议 4 次首次 9B 调用：1/4 引用及结构合格；缺脚本案 10 项声明获同一助手支持，因真实缺能力事项仍 blocked。工具文档与目标源文混淆、漏分支引用及旧提示字段继承问题保留为诊断；不再作为默认入口。
- [x] 另冻原文编号协议，同一 4 流程各 1 次：2/4 合格；单读案 11 supported / 2 insufficient，审批案 4 supported / 13 contradicted / 4 insufficient，均 blocked。编号减少引用抄写，但不能证明源段蕴含操作。
- [x] 两批总计 306.33 秒请求时间（不含预检/审查/测试）、9,972 / 2,986 输入/输出 token；0 个获审查支持的未激活流程、0 写/脚本/Runtime 执行。反向条件、非读节点引用、库存状态冒充审批、缺失前置依赖等错误仍存在，不能宣称准确率提升。
- [x] 新增 **22 项回归**；全量 **983 tests + 81 subtests** 通过（125.48 秒），Ruff/diff、原 C3b 及新两批证据重放通过。所有既有密封源、输出和报告保持不变。
- [ ] 下一步 C3d：先离线验证层级顺序/if/otherwise 到既有图的机械编译，减少模型直接手写 ID/边/汇合的错误；字段业务事实含义仍需真实合同和源审查。通过后才另冻小批，不继续在当前反例上堆提示词。

这仍是 **4 个已知开发流程、两种不同协议、同一助手参与作者/审查**，不是 8 个 Skill、独立评测或泛化提升。正确安全停止、证据不足与业务逻辑错误分别报告。完整用法、失败定位和后续方向见 [C3c 报告](FLOW-SOURCE-GROUNDING.md)、[可重算摘要](benchmarks/flow-source-grounding-c3c-summary.json)。12 个公开 Skill 整流程与大规模 Runtime A/B 继续保持门禁，不进入生产工程扩展。

#### 2026-09-07 C3b：冻结协议、8 个异质开发流程与源审查完成

- [x] 用户授权后，C2/C3a 全部项目改动已提交本地 `dev`：`2e4f762`，未推送远端。
- [x] 冻结源文/宿主合同、模型、请求与实现摘要；同一序号协议逐项只调一次 9B，无答案图输入。完整 checkpoint 重入只读验证，不重发；中断不明或漂移拒绝恢复。
- [x] **8 个手工已知流程 / 1 领域 / 1 宿主工具 / 0 公开 Skill**，覆盖顺序、数据引用、反向/嵌套/汇合分支及缺审批、脚本、无界循环。没有扩充为“8 个公开 Skill”。
- [x] 真实 9B 首次结果：**5/8 结构通过，3/8 结构阻断**。5 项通过结构检查的提案经当前助手完整声明审查后均 blocked：89 项声明为 63 supported、15 contradicted、11 insufficient_evidence；0 个审查支持的未激活流程、0 写/脚本/Runtime 执行、0 修订重试。
- [x] 发现业务目的元任务化、伪未解决问题、条件极性反转、库存状态冒充审批事实、跳过脚本前置依赖和不可达停止节点。不能只修说明文字，更不能删问题列表刷通过率。
- [x] 新增 **17 项回归**；相关 51 项通过，最终全量 **961 tests + 81 subtests** 通过（87.38 秒），Ruff/diff 校验与冻结批次重入/报告重算通过。
- [ ] 下一步 C3c：通用源语义到目的/限制、条件路径、依赖及字段事实含义的映射与问题表示。修正后另冻新批，不在本批重跑；12 个公开 Skill 整流程仍 not_run，规模化 Runtime A/B 仍暂停。

8 次请求共 360.18 秒客户端计时（含每项模型预检与等待，非纯推理耗时），6,268 / 3,596 输入/输出 token，不含审查和测试时间。当前助手参与了用例编写和审查，不是独立泛化证据；声明计数互有重叠，不能算准确率。见[流程清单、结果与用法](FLOW-FROZEN-DEVELOPMENT.md)、[源审查汇总](benchmarks/flow-development-8-evidence-summary.json)。以下保留历史阶段，旧“下一步”不覆盖本节。

#### 2026-09-07 C3a：真实 9B 正向流程与辅助语义修订闭环

- [x] 输入源文＋宿主工具合同，不提供手工参考图；9B 生成步骤、条件、参数引用及终态。编译器只绑定工具合同和宿主规则，不替模型补步骤。
- [x] 三次开发调用：命名节点协议两次结构阻断；序号协议由编译器生成 ID 后结构通过，但用途元任务化、needs_l1/blocked 解释混淆被审查拦下。原始回答不改写，三个原始版本均未执行。
- [x] 双向检查覆盖源要求和输出节点/条件/参数。第三次 32 项审查中 24 supported、5 contradicted、3 insufficient_evidence；8 项发现交叉覆盖两个解释性根因，不是准确率。
- [x] 独立 sidecar 保存父提案/源摘要、两处说明旧值/新值及引文，禁止更改图、参数、分支和问题。当前助手重新审查后，辅助版本实际完成 3 次本地文件读取：园区 read_path_completed，IDC needs_l1；0 写入，未接 DSH Agent 循环。
- [x] 新增 **34 项回归**；全量 **944 tests + 81 subtests** 通过（86.91 秒），定向 Ruff 与 diff 校验通过。两次结构失败、原始语义阻断及修订结果分别保存。
- [ ] C3b：冻结通用序号协议，扩充异质已知开发流程及源合同；区分原始/辅助结果、语义错误、环境缺失和 unsupported。12 个公开 Skill 的完整流程转译仍 not_run，规模化 Runtime A/B 不恢复。

这是 **1 个已知本地流程、3 个不同开发协议、同一助手修订/审查**，不是独立人工或泛化证据；合计模型响应 148.80 秒、2,535/1,348 输入/输出 token，不把辅助通过算成无辅助成功率。见[实验与用法](FLOW-FORWARD-TRANSLATION.md)、[可重算摘要](benchmarks/flow-forward-9b-summary.json)。

提交核查：本次按用户要求将 `acf0afb` 后的 C2 新模块、回归与报告和 C3a 正向转译闭环一并纳入本地 `dev` 阶段提交；不推送远端，不修改无关文件。原始大体积实验制品保留在本地 `artifacts/`，Git 保存带摘要的报告，二者不应混淆为完整可移植原始证据包。

#### 2026-09-07 C2：本地分支证据绑定与单写事务闭环完成

- [x] 在用户提交 `955bc92` 后继续实现；未修改旧基线或提交记录。新增 HostFlowGate，实际读取求值，不信任外部提交的“分支通过”报告。
- [x] 带流程门禁的计划采用 schema 11：流程/合同、读取上下文、路径/事实/来源与目标参数纳入 planHash。旧无门禁计划保留 schema 10 兼容；启用门禁的实例拒绝省略 gate 及旧无绑定计划。
- [x] 审批后重新读取，原预检之后、发送之前再读一次；变更/缺失/越权/超时均阻断。重启需恢复同一宿主 gate；不恢复则 fail closed。继续复用原审批、一次性 nonce、Verify 与补偿。
- [x] 本地三用例：verified_success；分支变化 precondition_changed（0 次 mock 变更）；注入验证失败后 rollback_verified。读取实际临时库存，写入为 LAN mock，0 次 LLM/真实网络设备调用。
- [x] 新增 **25 项回归**；全量 **910 tests + 81 subtests** 通过（86.03 秒），定向 Ruff、diff 校验通过。
- [ ] 下一步 C3：宿主上下文驱动的 9B 整流程提案与逐节点保真审查；12 个公开 Skill 的完整转译仍 not_run。当前接线是手工规则，不计为转译准确率。

这是本地原型单写门禁闭环，不证明源数据实时性、跨设备原子性或生产成功概率。完整投影值/Provider 证据比较可能保守拒绝；最后重读与写入间仍有 TOCTOU 窗口。多写、循环、并行和 DSH UI 全链未纳入此次成果。见[设计与复现](L0-BUSINESS-FLOW.md)、[最终报告](benchmarks/flow-effect-local-v2-summary.json)。

#### 2026-09-07 C1：确定性只读业务流程接线完成

- [x] 用户确认宿主上下文＋整 Skill 混合提案路线，同时要求补齐最小业务流程执行；不得只做提案即结项。
- [x] 增加封闭标量类型的数据引用、确定性相等分支、无环/可达性与全路径引用校验。权限、字段缺失、过期与工具错误均阻断，不能当成 false 分支。
- [x] 宿主分别绑定精确流程/请求和读取合同；实际读取继续走原 `execute_host_read`。新增可复现本地命令：园区路径读取两次、IDC 路径读取一次后 needs_l1，库存文件不变。没有模型调用或写入，非转译准确率。
- [x] 新增 **33 项回归**；全量 **885 tests + 81 subtests** 通过（86.49 秒），定向 Ruff、报告摘要和 diff 校验通过。
- [ ] C2：将分支事实/流程摘要绑定入 PreparedPlan，并在写前重新校验，再接通原审批、Effect、验证和补偿。当前写叶仅 awaiting_effect_admission，禁止直接执行；多写/并行/循环尚不支持。
- [ ] C3：9B 正向整流程提案、逐节点源保真审查与异质开发小批，首次与辅助结果分开。原 12 个公开 Skill 的完整转译仍 not_run。

见[最小业务流程与安全边界](L0-BUSINESS-FLOW.md)、[本地运行报告](benchmarks/read-flow-local-summary.json)。C1 可独立作为阶段提交检查点，但不是完整 C 阶段验收或生产就绪。

#### 2026-09-07 C：异质开发小批选样与源绑定盘点完成

- [x] 从封存已知开发库选择 **12 Skill / 11 仓库 / 10 类结构**；沿用 8 个源领域标签，不把标签数量宣称为跨域泛化。
- [x] 每项保存环境依赖、单读范围缺口、原文引句/位置/摘要；本轮当前助手非独立标注。授权与人工审批区分，第三方脚本保持惰性。
- [x] 新增源绑定 intake 命令和 **18 项回归**；全量 **852 tests + 81 subtests** 通过（85.87 秒），定向 Ruff / diff 校验通过。
- [ ] C 完整转译/执行未开始：本次未提交完整宿主工具环境，结果为 `not_run`，**不是 0% 转译准确率或 Skill 无效**。本轮模型调用、Runtime 执行、第三方代码执行均为 0。
- [ ] 设计检查点：建议“宿主工具上下文绑定 + 整 Skill 混合流程提案”，区分可编译步骤、L1 推理、缺环境和不支持，不以子操作成功代替整个 Skill。确认后再扩展单操作 authoring；不新造并行执行器或恢复大规模 Runtime A/B。

见[小批报告、12 项清单与下一步边界](TRANSLATION-BOUNDARY-PILOT.md)和[可重算报告](benchmarks/translation-boundary-pilot.json)。下面保留历史阶段记录；历史“下一步”不覆盖本节当前状态。

#### 2026-09-07 B2：问题解析留痕与单操作辅助闭环完成

- [x] 新增独立解析侧文件：固定父提案摘要、问题原文/序号、答案和源引文位置；父提案不变，子提案只处理未解决问题，不修改工具、参数、输出或权限。
- [x] 新审查要求基础声明与答案充分性全部覆盖，答案引用必须对应具体证据；旧审查、删题、重复问题、错误引文和不支持的答案均不能通过。当前助手解疑/重审仍标为非独立 AI 模拟。
- [x] 23 项声明/答案经本轮重审获支持后，宿主授权实际读取 `campus-sw1`，返回 campus / planned-lab 库存快照，文件内容保持不变。父提案两个问题与原始回答仍完整保留；未追加 9B 调用，也未全局激活合同。
- [x] 实际负向检查：缺少宿主授权、复用旧审查、越权设备、否定命令均拒绝，Provider 调用数均为 0。制品绑定和报告重算一致。
- [x] 新增 **16 项回归**；相关 **62 项**、全量 **834 tests + 81 subtests** 通过（86.33 秒），定向 Ruff / diff 校验通过。
- [ ] 下一步 C：8–12 个异质已知开发 Skill 的小批闭环，分别报告原始生成、辅助修订、危险误接受和覆盖率。当前只是 1 个新建单操作样例，未证明全自动转译、完整 Skill/DSH 循环或未知集泛化；规模化 Runtime A/B 仍暂停。

查看[解析与运行说明](L0-READ-CONTRACTS.md)和[本轮证据摘要](benchmarks/read-local-resolution-summary.json)。以下条目保留各自历史阶段，原 blocked 结果没有被改写为成功。

#### 2026-09-07 B2：正向 9B 与宿主授权本地读取接线

- [x] 新建库存 L1 Skill 和可审查的本地文件读取适配器，工具名在 Skill 中明确声明，实际代码只读取宿主固定文件并投影输出字段。没有补改旧健康检查样例。
- [x] 宿主读取入口复用 `ObservationPolicy`，校验精确合同/能力/Schema/敏感级别、显式身份、角色和能力/设备范围；拒绝隐式系统身份、通配权限、额外路径参数和不合格结果。仅本地实验信任边界，不是生产身份认证。
- [x] 真实 9B 正向提出用途/工具/只读分类/问题，源 Schema 由代码组装进 L0.5。两次 HTTP 400 失败目录保留；诊断为解码 grammar 失败，调整解码提示且保留完整输出校验后，第三次生成成功（10.80 秒、547/115 输入/输出 token）。首次没有留存错误正文，不能声称所有失败证据完整。
- [x] 模型选择和用途正确，但有 2 个未解决问题；当前助手模拟审查支持 21 项声明，仍判 blocked。执行前检查确认模型提案调用 0 次，未删改原回答以获通过。接线回归单独读取实际实验库存，不计为模型执行成功。
- [x] 新增 **32 项回归**；相关 **86 项**、全量 **818 tests + 81 subtests** 通过（84.87 秒），定向 Ruff / diff 校验通过。
- [ ] 下一步：基于既有 Schema/适配器代码解析问题，保存原始→证据→修订提案轨迹，再重新审查并验证本地读取；完整自然语言请求、DSH Agent 循环和未知 Skill 泛化仍未证明。

入口和限制见[只读合同使用](L0-READ-CONTRACTS.md)，结果见[正向实验摘要](benchmarks/read-local-forward-summary.json)。

#### 2026-09-07 B2b：只读 L0.5 映射与审查子链

- [x] 增加可读 `ReadL05Proposal`，用途和操作字段确定性映射到现有只读合同；未解决问题单独保留，不补造执行语义。
- [x] 自动列全源证据检查项，绑定 L0.5/L0 位置、源原文/位置/摘要，复用已有审查协议；缺项、错引用、矛盾和证据不足不能晋级，全部 supported 仍未授权。
- [x] 提供 scaffold / review-input / assess 本地命令，拒绝覆盖已有输出；scaffold 明确是从手工 L0 反向构造的编辑模板，不计为 L1 正向转译成果。
- [x] 当前助手模拟审查 12 项：10 项声明 supported、2 项 insufficient_evidence，另有 1 个未解决待办；最终 blocked、0 个晋级合同、0 次工具执行。问题定位到 Skill–工具名映射和缺少适配器只读实现证据；不是独立人员测试或转译准确率。
- [x] 新增 **14 项回归**；相关 **70 项**、全量 **786 tests + 81 subtests** 通过（84.86 秒），定向 Ruff / diff 校验通过。
- [ ] B2 剩余：可审查的真实/本地适配器证据、正向 L1→L0.5 提案、请求语义审查和授权读取。本轮没有推进到执行，未恢复大规模 Runtime A/B。

命令与定位说明见[只读 L0 合同](L0-READ-CONTRACTS.md)，数据见[诊断摘要](benchmarks/read-l05-b2b-summary.json)。

#### 2026-09-07 B2a：未激活只读合同与独立请求实例化

- [x] 在现有 L0 v2 类型/编译器/目录/CLI 增加 `AtomicRead` / `CompiledAtomicRead`；支持无参、可选参数与有参读取，不更改写操作的审批、预检和验证规则。
- [x] 固定 Skill/tool/adapter 原文及摘要，对齐工具名、读写声明、输入/输出 Schema 与权限范围声明；摘要不等于来源认证，声明一致不等于语义正确。
- [x] `instantiate_read` 单独生成结构化参数请求草案；复核合同摘要，不固化测试参数、不授权执行。`validate_read_result_shape` 只校验结构，不宣称业务成功。
- [x] 写能力查询不返回 read，Saga 拒绝 read，旧 effect L0.5 promotion 明确阻断 read；无第三方脚本、真实设备或模型调用。
- [x] 新增 **40 项回归**；相关 **76 项**、全量 **772 tests + 81 subtests** 通过（85.20 秒），定向 Ruff / diff 校验通过。合成示例的 CLI validate/explain/schema 和 Python 草案/返回形状接口已本地验证。
- [ ] B2b：只读 L0.5 的逐项来源映射和语义审查，真实工具/适配器证据，参数提取与授权后受限读取。B2 尚未整体完成，未恢复规模化 Runtime A/B。

使用和限制见[只读 L0 合同](L0-READ-CONTRACTS.md)。新增示例是独立的手工合成夹具，不修改 B1 的密封制品，不计为 LLM 转译或泛化成绩。

#### 2026-09-07 A/B1 复查与 B2 边界修正

- [x] 修复转义字符串被错误还原，以及数值溢出为 Infinity 仍被接受的问题；解码后占位符、非法 JSON 转义、非有限夹具值和源 JSON 常量/指数溢出均拒绝。
- [x] 新增 20 项字面量边界回归，相关 **63 项**通过；全量 **732 tests + 81 subtests** 通过（81.96 秒），定向 Ruff 和 diff 校验通过。
- [x] B1 原 9B 制品只读复核 `verified=true`、`implementationDrift=true`：原报告/检查点仍完整，当前实现已变化。仍有 1 个待语义审查任务，没有改写历史结果，也未新增模型调用。
- [x] 确认既有写合同编译器无法直接表达 B1 无参只读例子；没有伪造目标/预检/验证，也没有放松写审批门禁。
- [ ] B2 先在既有 L0 类型/编译体系补充只读合同，再做源证据绑定与独立请求实例化；具体验收顺序见[纠偏计划](TRANSLATION-CORRECTION-PLAN.md)。本轮未完成 B2、未运行 Runtime，字面量测试不代表语义准确率或泛化率。

#### 2026-09-07 B1：合同优先的任务构造

- [x] 新接口从带摘要的工具源文本解析 Schema，经源声明审查后确定适用槽位；无参/全可选只生成 1 个正常槽位，k 个必填项生成 k 个缺参槽位。
- [x] 9B 只允许输出 `user_prompt`，不能改合同、N/A、槽位或参考答案；源证据不支持、读写未知、Schema 不支持时 0 次调用。
- [x] 参数检查复用 Translator 的 schema binder，保存实际模型输入、逐调用 checkpoint 和摘要；完成后只读重入，中断不自动覆盖/重跑。
- [x] 新增 16 个定向回归；相关 43 项通过，全量 **712 tests + 81 subtests** 通过，Ruff/diff 校验通过。
- [x] 真实 9B + 显式合成合同/审查夹具：1 次调用、5.93 秒、1 个适用任务，缺参族 N/A 由代码确定。模型只返回 `health_snapshot`，参数检查后仍是 `needs_task_semantic_review`，不是理想的业务请求，未认定为合格用例或 Gold。
- [ ] B2：真实源合同/适配器证据、任务语义审查，L0.5→现有可复用 L0 编译器→请求实例化。

接口与命令见[合同优先任务构造](CONTRACT-FIRST-TASKS.md)。该入口适用于已有工具合同的场景；旧 v4 源 Skill 作者仍只是非权威候选发现，不能冒充已审查合同。Runtime 大规模评测继续暂停。

#### 2026-09-07 基础纠偏 A

- [x] Translator v2.1 移除模型输入的 caseId/challenge/language/catalog assignmentId，并增加评分元数据变化不影响 Prompt 的回归；原始业务请求不改写。
- [x] 字段归属、同名冲突和无效值联合校验；复现的跨字段错误引用、冲突值采信被拦截，可选参数不再强制补齐。
- [x] 作者 v4 / Catalog v2 支持无参、可选参数和缺参槽位不适用；只生成候选主操作，不再虚构预检、验证、补偿接口。
- [x] 116 项定向测试通过；补齐开发目录指向已有本地环境的 `.venv` 链接后，全量 **696 tests + 81 subtests** 通过。首次全量的 19 个环境失败保留为诊断，不算代码测试成功。
- [x] 真实 9B 单 Skill 验证：doc-ingest-analyze，2 次调用，234.1 秒。模型识别了 `parameters=[]`，但仍输出缺参任务且把 N/A 标记放错槽位，最终拒绝，0 个审查包，未生成可执行 L0。这不是泛化提升成绩。
- [x] 新制品完整性校验通过；旧 development-04/06/07 与 construct-v3-object-storage 重检通过。旧 01-v2/02-v1/03-v1/05-v1 的同版本规则漂移在修改前 HEAD 也可复现，未改写原件或标签。
- [ ] B：工具合同/源证据先于任务生成，任务适用性由已审查合同确定；贯通 L0.5→现有可复用 L0 编译器→请求参数实例化。
- [ ] C/D：异质小批语义闭环、隔离参考答案和冻结后未知集合；规模化 Runtime A/B 继续暂停，生产工程仍冻结。

实施边界见[纠偏与闭环计划](TRANSLATION-CORRECTION-PLAN.md)，证据见[本轮摘要](benchmarks/translation-correction-v4-summary.json)。以下日期条目保留历史状态，不能代替当前结论。

权威命题是：概率性 Reasoning 只提出 Candidate Plan；Contract、Evidence、Guard、Risk 和 Transaction 决定是否允许 Effect；不合格转换安全停机，不能回退原生写。

ES-P0 的 Runtime 机械原型和小样本接线结论保留为 `local_hypothesis_supported`，但不再被解释为 L1→L0 高泛化证明。只有转译器先在冻结后采集的跨 Skill/仓库/领域未知集合上通过门禁，L0→Runtime 的规模化安全、稳定和准确性评测才具有研究意义。

#### 2026-09-04 评测可信性修正

- [x] 发现并修复旧盲审中类别后缀 ID 与固定三任务组序的提示泄漏；旧 48/48 结果保留但降级为有元数据提示风险的开发诊断；
- [x] 新增随机盐匿名 ID、单任务独立请求、模型输入白名单、私有映射与结果重绑定、可核查的实际输入及检查点完整性校验；
- [x] 用相同真实 9B 与 development-07 源包完成匿名化复验：4 Skill/12 task 协议完整、元数据盲态校验通过，行为一致 7/12、构造对齐 10/12；Gold 排队资格为 false，原满分不再被用来支持语义可靠性；
- [x] 全量回归 621 tests + 81 subtests 通过；定向 Ruff、JSON 和 diff 校验通过；未执行 Translator、Runtime 或第三方脚本；
- [ ] 清理用例中“生成某类候选”等元任务措辞，审查窄操作族是否忠实代表源 Skill；
- [ ] 构造与模型输出隔离的参考答案，再进行真正的 Gold-blind Translator 评测；没有人工时 AI 角色结果始终标记模拟证据。

#### 2026-09-04 构造检查 v3

- [x] 删除自动追加正常任务参数的规范化；保存原始模型候选及显式修复版本，不再掩盖缺参；
- [x] 显式参数使用类型化证据而非预设样例值匹配，保留冲突值原文/位置，拦截正常任务的无效字面量、未求值占位符和无源支持的评测元任务；
- [x] 新增只读历史审计：development-06/07 原先 16 个已接受 Skill 中，9 个通过新机械规则、7 个被拦截；旧密封制品和标签未修改；这不是 Translator 准确率；
- [x] 新报告明确区分显式参数夹具、窄操作族、未验证的源 API Schema 与完整自然语言/Skill 语义，历史 v1/v2 保持原规则检查；
- [x] 真实 9B 新协议复验 object-storage：1 Skill/3 task，2 次调用后仍生成错误元任务，门禁拒绝，0 个通过；验证了拦截，未证明生成质量提升；定向 71、全量 658 tests + 81 subtests 通过；
- [ ] 逐参数/逐步骤源证据审查，解决任意正文冲突、API 必填项虚构、完整写流程被降为读，以及验证/补偿能力是否真实存在；
- [ ] 隔离参考答案与 Gold-blind Translator。Runtime 规模化评测仍锁定。

实现与限制见[构造质量说明](TRANSLATION-CONSTRUCT-QUALITY.md)。

#### 2026-09-04 源证据对齐审查 v1

- [x] 按任务、工具操作/读写/阶段/入参形状，以及每个参数的存在性/类型/必填性自动生成完整检查项；禁止模型自选审查子集；
- [x] 源引用绑定到原文件和字符位置，缺项、错引用、任务文本冒充 API 证据均 fail-closed；报告支持比例与语义正确性分开，不能自动产生 Gold 或 Runtime 权限；
- [x] 单任务匿名 9B 审查器、原始响应留存、结果摘要绑定、完成后只读重入；旧整体布尔审查仅能排队源证据审查；
- [x] doc-ingest-analyze 真实 9B 单次审查返回 12 个声明结果，指出无参 API 与必填参数冲突；但漏引任务原文，整体 `protocol_failed`，且只读性质推断缺少充分依据；未修补模型答案；
- [x] 定向 89 项测试、全量 676 tests + 81 subtests 通过，定向 Ruff 和制品完整性检查通过；
- [ ] 修正源证据引用协议与读写语义推断，获得有效的逐项审查；重新设计无参/可选参数及不适用缺参测试，不能强制每个操作编造必填项；
- [ ] 完成隔离参考答案、Gold-blind Translator 与未知 Skill 泛化验证；Runtime 规模化评测继续锁定。

接口、用法和证据边界见[转译源证据对齐](TRANSLATION-SOURCE-ALIGNMENT.md)。

#### 2026-09-03 转译优先重置

- [x] 新建 Gold-blind Translator v2：模型只输出语义意图和参数源证据，Capability 选择、参数绑定、事务闭合与 L0 制品加载由确定性代码完成；
- [x] 独立 Skill–Task–Tool 对齐审查区分 read/write/clarification/reject 与 construct-invalid，评分器要求审查全覆盖并与 Gold 处置一致；
- [x] 100 个静态公开 Skill 已建立可搜索索引：72 仓库、9 领域、第三方执行 0；53 `runtime_ready`、18 `translation_only_partial_context`、29 `format_variant_robustness_only`；
- [x] 主要转译开发语料为 71 Skill，按仓库聚合为 7 个批次；整个已知库固定 `proofCohortEligible=false`；
- [x] Runtime 大规模评测加入硬门禁：没有有效 admission 时只允许 1 case × 1 repetition 接线 smoke，且 `researchEvidenceEligible=false`；
- [x] 准入器要求同一冻结 Translator、至少 3 个互不重叠的未知 cohort，以及合计 ≥50 Skill、≥15 仓库、≥8 领域、≥600 case；
- [x] 建成语义锚定作者链：精确原文锚点、参数字面证据、通用不可执行 Tool Catalog、确定性结构门禁、透明保守规范化与答案隐藏的 AI 审查格式；
- [x] 完成 `development-01` 的实现绑定 12-Skill/36-task 9B 作者迭代：唯一非模糊 span 对齐与 read/write 机械闭包把门禁通过从 6/12 提升到 10/12，盲审 task 从 18 增至 30，模型调用从 18 降到 14；2 个剩余失败均为非精确 anchor；
- [x] 建成独立密封输出的答案隐藏 AI 角色审查器：10 个通过 Skill/30 task 全部协议完整、行为一致且无低置信度，p50/p95 为 26.0/43.4 秒；作者与审查者同为 `qwen3.5:9b`，因此只具备开发排队权，`humanIndependentEvidence=false`、`semanticAlignmentProven=false`；
- [x] development-02 至 05 已完成 43 个已知 Skill 的失败驱动作者实验；编译器接管 operation ID 和精确 source-span 绑定，禁止 Runtime 控制字段进入业务参数，并机械闭合 read/non-write envelope；development-05 为 9/11 通过，两个失败来自长上下文 Skill 的 assignment/slot/参数类型错误；
- [x] development-05 的 9 个通过 Skill/27 task 完成答案隐藏复核：行为一致 26/27、完整对齐 25/27，无低置信度；复核暴露相同任务文本被赋予不同处置及 clarification 布尔自相矛盾，原密封报告未改写，代码已新增对应 fail-closed 门禁；
- [x] 最终门禁版本在 development-06/07 连续运行：16/16 Skill 首轮通过、0 修复，48/48 task 的同模型答案隐藏复核行为一致且完整对齐；71 个主要已知开发 Skill 的失败发现作者覆盖完成；
- [ ] 为合格候选完成独立 Gold 并运行 Gold-blind Translator；当前不得把作者门禁率、同模型自审率或跨不同批次的表面变化称为 Translator 准确率；
- [ ] 使用 qwen3.5:9b 分批只跑离线转译，基于失败类别改良通用算法，禁止按单 Skill 打补丁；
- [ ] 冻结稳定 Translator 后再收集全新 proof cohorts；门禁未通过前不恢复规模化 Runtime A/B。

详细规则见 [L1→L0 泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)。

### Done

#### 架构与执行边界

- [x] Reasoning / Reliability Runtime / Infrastructure 三平面成为唯一权威架构；
- [x] README、ARCHITECTURE、HLD、LLD、SSD 与统一 Skill→系统交互说明对齐，明确 authoring/在线执行双生命周期、只读 fallback、写入 safe-stop、终态和证据定位；
- [x] 产品原型唯一写路径为 Candidate → active L0 → Runtime → Provider；
- [x] 原生 Agent mutation 只存在于隔离本地 A/B Control；
- [x] 不合格 L1→L0 转换只能 read / clarify / proposal / ask-human / reject；
- [x] 企业控制面、供应链、Hermes/A2A、HA/DR、WORM 和生产 SLO 冻结为未来工程。

#### Contract-Governed Skill

- [x] 21 个受审 L0 v2 网络合同和 21/21 L1→L0.5→L0 可读轨迹；
- [x] `ReliabilityContract` 与 `TypedExecutionGraph` 固定 inputs、Evidence、Guard、Risk、资源、后置条件和 Compensation；
- [x] Promotion 是 proposal-only，模型产物不能自动进入 active Registry；
- [x] 60 个跨域 Anthropic Skill 开发集支持 references、scripts、审批、分支、多步和组合。

#### Reliability Runtime

- [x] Evidence provenance、freshness/scope/action/integrity gate；
- [x] 确定性 Guard 与 revision/maintenance-window 写前重校验；
- [x] `execute / ask_human / reject` Risk Policy；
- [x] Snapshot、Precheck、Approval、Revalidate、Execute、Verify、Commit；
- [x] 写后不确定只读 Reconcile，不盲重试；
- [x] Compensate、Verify Recovery、多步骤 Saga 逆序恢复；
- [x] 参数 exact binding、不可变计划、SQLite journal/hash-chain 和终态审计。

#### ES-P0 证据

- [x] 六类场景 × 10 次，共 60 个真实 Runtime 运行；
- [x] Full + 去除 Contract/Evidence/Guard/Transaction/Compensation 的 30 个消融探针；
- [x] `qwen3.5:9b` 60-Skill 转译：58/60 严格通过，误接受 0；
- [x] `qwen2.5:7b` 60-Skill 转译：38/60 严格通过，误接受 0；
- [x] 两模型各 10 场景 × 3 次 × 2 臂，共 120 个真实 DSH 会话；
- [x] 9B Treatment：Task Completion 86.67%，Execution Precision 100%，Unsafe/False Commit/Invalid Action 0；
- [x] 7B Treatment：三项 Action 风险为 0，但 18/30 Process Failure，明确判定可用性不合格；
- [x] Execution Precision / Autonomous Coverage 曲线、场景级 Wilson 95% 区间和代码/模型/输入指纹；
- [x] 最终全量回归：`506 passed, 81 subtests passed`；
- [x] 聚合证据状态：`local_hypothesis_supported`。

详细数字、制品摘要与复现命令见 [ES-P0 本地证据报告](ES-P0-EVIDENCE.md)。

### 未完成，但不阻塞本地 ES-P0 结论

| 项目 | 当前状态 | 后续验证 |
|---|---|---|
| 独立泛化 | 透明开发集 | 仓库外、预注册、封存/私有用例和独立 Reviewer |
| 真实网络资格 | 本地仿真 Provider 与 Containerlab/FRR | Cisco/Huawei/H3C 或控制器实验床 |
| Typed Graph | journal-backed scheduler 已门禁正常、拒绝、漂移、未知结果、补偿与崩溃恢复分支 | 后续把 legacy PlanState/L0 兼容事件收敛为图的派生视图 |
| Provenance | 已生成 Evidence→Observation→Capability/Collector→Network Object 跨步骤 DAG | 在 ES-P2 检验真实 Provider collector/object 绑定 |
| Stage latency | 已按图节点拆分 Runtime active 与 approval wait，并声明排除 Reasoning/LLM | 在真实 ES-P1 paired run 预注册 all/completed 与分层统计 |
| Experience Compilation | authoring compilation 已完成 | 基于长期成功轨迹的独立研究协议 |
| 生产工程 | 冻结 | 完成独立泛化和真实设备证据后再决策 |

### 当前活动阶段与后续计划

**当前活动阶段：先完成 L1→L0 跨 Skill 泛化；Runtime 规模化评测已被硬门禁暂停。**

#### ES-P1 已完成的研究基础设施

- [x] `Research Freeze v1` 联合绑定 Git commit/clean state、Runtime kernel、Harness boundary、21 个 Contract/trajectory、Evaluator、authoring protocol、模型制品、依赖环境与 ES-P0 基线；脏工作树只能生成不可注册的 preview；
- [x] `Study Plan v2` 与 `Private Manifest v3` 强制绑定已验证 `freezeDigest`，正式 CLI 拒绝 preview、漂移或篡改的 Freeze；
- [x] 仓库外工作区把 Research Freeze、Case Author、两个 Reviewer、Adjudicator、密封数据和模型运行分成独立 Gate；Doctor 不输出 Prompt、Label、语义合同或参数值；
- [x] `Qualification Report v2` 增加 family/profile/language/challenge/expected-disposition/risk 分层，Wilson 95% 区间、零事件单侧 95% 上界和互斥结果分类；
- [x] safety escape 拆为 critical semantic、undeclared effect、approval/risk weakening 三类，三类均为零才可能通过资格门；
- [x] Freeze、预注册、密封、盲审/仲裁、重复评分和篡改检查的相关回归为 `29 passed`。
- [x] 完成 I11 权威边界清理：DSH 产品 Adapter 不再导入 Evaluator/Golden Set，能力检索 parity 移入 `evaluation/` 且只使用内存状态；A2A、轨迹学习与历史 L1 shadow 改为命令触发时延迟加载；
- [x] 活跃文档导航移除 P1.9 Decision Plane/Canary 产品化路线，真实 private holdout 与“只有协议工具”的边界已更正；冻结实现继续作为 fail-closed 回归，不计入当前能力；
- [x] Runtime 语义收敛原型：Typed Graph 已成为 journal-backed 分支门禁；执行前安全中止不再依赖可补偿性；崩溃边界显式记录 `skipped/indeterminate` 并只读对账；`inspect()` 输出图一致性、跨步骤 Provenance DAG 与 stage latency；
- [x] 当前仓库全量回归为 `599 passed, 81 subtests passed`。ES-P0 证据报告中的 506 是当时冻结制品的历史计数，不回写成新的实验结果。
- [x] 建立仓库外 synthetic evidence 通道：240 条模型生成候选经 Reviewer A/B 独立 Prompt 盲审、按需裁决和 Skill/Case/Role 摘要封存；覆盖 6 个 Skill 特征族、10 个事务/故障模式、6 个 MCP 域和 3 种语言组；受控 Loader 强制其 `officialEsP1QualificationEligible=false`；
- [x] 完成 v3 synthetic evidence：240/240 Skill 包零 finding，9B 转译协议有效 240/240、可信 Oracle 合格 235/240、fallback 5、false accept 0；10 场景×3 次真实 DSH 配对经 effect-budget v3 无模型重评分后，Treatment Task Completion 93.33% 对 Control 76.67%，unsafe 0 对 4，invalid 1 对 5，p50 64.3 秒对 103.6 秒；17/17 Runtime 审计有效。该结果是模型合成证据，不是独立泛化或生产概率。
- [x] 完成 `ES-P1-Wild` 静态导入 pilot：SkillsMP 发现 100 个候选，处理 60 个后以许可证、无脚本、固定 commit 和摘要门禁接纳 20 个/13 仓库；第三方执行与可执行文件物化均为 0；严格 Runtime 包门禁 15 passed、5 blocked。该静态导入阶段当时尚无任务、Gold/Oracle 或 DSH paired run；后续另行完成的角色模拟结果仍不是 ES-P1 资格结果。
- [x] 从 15 个通过门禁的公开 Skill 导出独立 author kit：45 个任务槽位、固定来源包、Task/Gold/Tool Catalog Schema 和全文件摘要；工作区不含 Runtime/Evaluator、模型输出或自动 Gold，避免基础设施伪造独立真值。
- [x] 增加显式非权威的 9B 草案辅助通道：15 个 assignment 中 14 个通过协议与安全形状校验，覆盖 42/45 槽位；12 个需要修复调用，p50/p95 为 59.2/97.2 秒；1 个 Effect budget 矛盾被持续拒绝。草案不含 Gold/Oracle、不能进入 Runtime，仍需独立人工审阅。
- [x] 建成测试 Skill 索引库：可搜索/筛选 15 个实际测试 Skill，点击查看 22 个封存文本文件、固定来源、许可证、45 个任务槽位和 42 个草案；页面离线只读、内容以纯文本展示、第三方执行为 0，并提供可提交的元数据索引。
- [x] 建成模型辅助但人工拥有的 Case Author Review Kit：只暴露 42 个问题候选并扣除全部模型语义标签；45 个槽位必须人工接受/修改/从零编写/拒绝，Gold/Oracle 注入和源篡改均 fail-closed。真实工作区当前 45 pending，不能自动导出 Gold Author Kit。
- [x] Review Kit v3 在封存材料门上增加 Tool Catalog v2/fixture-state 校验；盲态 Gold Author Kit 只在 Case Review 完成后导出空白 Gold/Oracle，并拒绝未封存文件、非盲态作者与自动资格升级。真实 45-pending 工作区的导出负测正确失败且未创建输出。
- [x] 增加 Public Paired Study Kit：人工 Gold 完成后才可封存 Agent/Scoring/Evidence 物理分区；预注册 9B、三次重复和唯一 Treatment 变量，拒绝 Gold 泄漏、能力缺失、包漂移与资格伪造。Study Kit 自身只准备输入，不冒充已执行的 paired 结果。
- [x] 增加通用声明式 fixture MCP：六种受审操作、封闭参数 Schema、独立 SQLite 状态、调用摘要审计、审批及四类故障注入；native/Runtime/safe-stop 权限边界和官方 MCP stdio 已通过本地测试。
- [x] 完成公开 Skill 的 Gold-blind 9B 转译与绑定：保存 L1→L0.5→L0/fallback 轨迹；confidence 不授予权限；Capability、唯一 Effect、Catalog 参数物化、审批、预检、验证、补偿与脚本禁用全部确定校验；纯读降级原生 L1，不合格写禁止原生 fallback。
- [x] 完成真实 DSH Public paired runner：两臂暴露相同 Skill/Tool/Task/fixture/审批/故障，Treatment 只改变合格 Effect 后端；Gold 在全部 Agent 运行结束后才解析。9B 技术 smoke 的合法只读和名义写场景两臂均 1/1 通过，Treatment 写路径完成一次 Effect 与独立验证；补偿路径通过确定性测试。单例只证明接线，不是 ES-P1-Wild 指标。
- [x] 完成 `ES-P1-Wild-Sim` 角色隔离本地协议：15 个公开 Skill、45 case、3 repetitions、135 组成对观察和 270 次真实本地 DSH 实验臂全部摘要绑定；Control/Treatment 为 82.22%/97.78%，L0 路由为 21/42→42/42，p95 为 109.3→56.1 秒，unsafe/false commit 两臂均 0，三轮结果完全一致。转译后验路由一致 43/45、unsafe Runtime 误接纳 0。虚拟 Case/Gold 角色固定 `humanIndependent=false`，因此只完成模拟协议，不构成正式 ES-P1 资格。
- [x] 后续 ES-P1 新模型运行固定为 `qwen3.5:9b`；7B 仅保留为冻结历史证据，不再新增实验。

这些完成项证明“实验可以按预注册方式运行且不会悄悄漂移”，并新增模型合成泛化信号；仍不证明独立隐藏集泛化。ES-P1 的核心未完成项仍是由 Runtime 团队之外的人在仓库外编写和审阅真实 private holdout。合成数据不能替代该 Gate。

| 顺序 | 阶段 | 要回答的问题 | 主要交付 | 进入下一阶段的门槛 |
|---:|---|---|---|---|
| 1 | ES-P1 Freeze | 结果能否脱离作者已知场景 | Runtime/Contract/Evaluator/9B 模型/环境封存包与 digest | 版本、指标、排除和中止规则全部预注册 |
| 2 | ES-P1 Private Holdout | 是否存在 benchmark co-design/overfitting | 仓库外 200–500 unseen cases、≥10 families、独立双审和 adjudication | 三类 critical escape=0；family-level 统计完成 |
| 3 | Runtime 语义收敛 | 执行图和证据链是否真正在控制运行 | 单一 Typed Graph scheduler、跨步骤 Provenance DAG、stage latency | 不改变冻结实验语义；不引入旁路 |
| 4 | ES-P2 Real Network | 抽象能否处理真实异步和部分失败 | 一个 router + 一个 controller/management path 的完整事务矩阵 | 同一 Contract/Evidence/Guard/Transaction 无弱化 |
| 5 | ES-P3 Paper-grade | preliminary signal 能否成为可投稿证据 | 2k–5k 分层 executions、interval、failure taxonomy、artifact package | 主要 claim 有 independent + real evidence |
| 6 | ES-P4 Experience Compilation | 可靠轨迹能否下沉成候选自动化 | 独立 compiler/qualification/promotion protocol | 不与当前 safety paper 混合，不自动激活 |

近期执行顺序：

1. 从干净提交生成正式 Research Freeze，并保存 qwen3.5:9b 的真实模型制品 digest；
2. 在已完成的 `ES-P1-Wild` 静态 pilot 上预注册正式采样，扩充语言和来源，并冻结 50–100 个公开 Skill；全部继续按不可信数据隔离，普通评测禁止执行 scripts/hooks/installers；
3. 由独立 Case Author 在 Review Kit 中完成 45 个首轮槽位决定，再由隔离的 Gold Author/Reviewer 补 fixture、Tool Catalog、Gold 与 Oracle；扩大到 200–500 条公开生态 paired cases。模型只提供问题候选，不能充当独立真值；其结果只支持兼容性和外部有效性，不替代 private Gate；
4. 由 Runtime 团队之外的 Case Author 正向构造 200–500 条 private holdout；
5. 两名独立 Reviewer 盲审，Adjudicator 只处理摘要绑定的分歧；
6. 在模型运行前冻结排除/中止规则、paired statistics、effect-call budget 和 latency breakdown；
7. 对同一制品至少重复三次并生成只含聚合值的 Qualification Report v2；
8. 专门修复 synthetic 暴露的两个 L1 边界：safe-stop 会话可用性，以及调用 Runtime 前把一般条件分支误判为事实；通过新版本合成/人工集验证，不对单 case 打补丁；
9. 将恶意 Skill、脚本和提示注入放入独立 `ES-P1-Sec` 强隔离安全集，不在普通 Agent/Runtime 环境中试运行；
10. 若出现 critical escape，先定位抽象缺陷并创建新研究版本；
11. ES-P1-Private 通过后进入小范围 ES-P2，不追求厂商 breadth；
12. 只有 ES-P1/ES-P2 证据支持后，才重新评估身份、供应链、HA/DR、WORM 和生产 SLO。

公开市场语料的证据分层、采样规则与零执行边界见 [ES-P1 公开 Skill 市场语料](ES-P1-PUBLIC-SKILL-CORPUS.md)。

详细原则、指标 Gate、角色边界和任务模板见[后续研究与研发指导 v1.1](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md)。

## English

**2026-09-07 Git checkpoint:** C3c–C3h code/tests/development dependency were committed locally on `dev` as `1c5fc7e`; documentation and evidence summaries are saved in a separate documentation commit. No push or master merge. Historical “uncommitted” entries describe their earlier state and are superseded here. Prior validation: **1288 tests + 81 subtests passed**, not semantic acceptance; fresh 9B responsibility-protocol evaluation remains pending. One trailing blank line in frozen `flow_lean_pilot.py` is intentionally retained to preserve recorded implementation fingerprints; other staged whitespace checks passed. Ignored local `artifacts/` retain raw evidence, not included in the committed summaries.

Latest verification: **1288 tests + 81 subtests passed in 131.58 seconds**, including existing isolated execution fixtures; Ruff/diff and both previous lean/full two-pass report replays passed unchanged. Test duration is not model or Runtime performance.

**2026-09-07 C3h offline responsibility mapping:** a separate exact-quote requirement protocol rejects incompatible declared target duties and reports omitted condition-node pointers. Full-source review remains, with complete-clause decomposition and atomic type/target review added. Seventy-five new development tests and ninety-eight targeted tests passed; these are not Skills or model accuracy. Offline request/compile/assess refuses existing outputs; no frozen answer/protocol/report, product default or executor changed. Zero new model calls: previous 3/4 mapping qualification and blocked semantic results remain the latest evidence. New 9B fresh paired quality/cost validation is pending, especially misclassification that evades compatibility. See [design and boundaries](FLOW-RESPONSIBILITY-MAPPING.md). C3h continues, C4–C6 remain gated; uncommitted/unpushed.

**2026-09-07 C3h lean generation improvement, semantic gate still unmet:** compiler-owned anchors/node citations/candidate explanations replace repeated model copies and prose, while exhaustive source/operation/constraint review remains. Eight fresh 9B calls on four known flows/one tool/zero public Skills: flow qualification **4/4**, mapping **3/4**; inverted branch omitted condition-node `/steps/1` evidence. Three complete same-assistant reviews: **103 supported / 25 insufficient across 128 overlapping claims**, zero review-supported inactive flows, all blocked. Wrong guarantee roles, compound-clause gaps and interpretation-as-operation remain candidate defects, not executed network errors or proof of unsafe Runtime execution. Mapping output **9,919→2,619 (−73.6%)**, input **19,520→25,172 (+29.0%)**; whole-chain total tokens **41,436→39,788 (only −4.0%)**. Mapping POST 811.20→413.99 s; whole POST 932.05→577.64 s. Different payloads/budgets/load and partial concurrent tests prevent causal timing or accuracy claims. Twenty-three new regressions; **1213 tests + 81 subtests passed in 165.05 s**, plus Ruff/diff, new-report replay/checkpoint re-entry and unchanged previous C3h report. Zero batch Runtime/tools/scripts/writes, retries or answer repair. Continue C3h requirement/guarantee-role alignment and complete condition/compound-clause coverage; distinguish semantic-flow errors from annotation gaps without removing review. Schema/catalog input duplication is secondary. C4–C6 remain gated; C3c–C3h changes are uncommitted/unpushed. See [report](FLOW-LEAN-MAPPING.md) and [summary](benchmarks/flow-lean-c3h-summary.json).

**2026-09-07 C3h implementation and first complete evaluation finished; quality gate unmet.** Exact-offset fragments and multi-source mappings now connect fresh source→9B flow→compiler nodes→9B metadata, with no historical-tree substitution or retries. Eight real calls on four known flows/one tool/zero public Skills: **4/4 flow qualification, 2/4 mapping qualification**. Inverted branch had 31 missing own anchors among 44 records; approval had 15 among 30. Complete same-assistant reviews of the two qualified mappings: **53 supported / 15 insufficient across 68 overlapping claims**, zero review-supported inactive flows, all blocked. Errors include validation/error-propagation conflation, narrowed authority, unsupported stops treated as checks, incomplete cross-source evidence and an unproven whole-operation read-only assertion. No fabricated semantic scores for structural failures. Costs **932.05 s POST, 30,536/10,900 tokens**, with 811.20 s in mapping; failures/waiting included, preflight/review/tests excluded, partial test concurrency prevents causal timing comparisons. This is not unseen evidence, a controlled C3g accuracy comparison or demonstrated usability gain. Twenty-nine new regressions and **1190 tests + 81 subtests** passed (170.75 s); Ruff/diff and new/old evidence replay passed. Zero batch Runtime/business/script/write executions. **Continue simplifying C3h before C4**: compiler-owned known anchors, fewer redundant model references/free explanations, with all semantic review obligations preserved, then a new complete frozen batch. C3c–C3h changes remain uncommitted/unpushed. See [report](FLOW-TWO-PASS-TRANSLATION.md) and [summary](benchmarks/flow-two-pass-c3h-summary.json).

**2026-09-07 C3g mapping repair complete, full translation gate unmet:** source archives, objectives, mandatory paragraph ledgers and enforcement references are separate. Optional-constraint and mandatory-ledger first-attempt 9B batches each qualified 1/4; omissions, contradictory reference roles and tool-name-as-node failures remain preserved. A restricted metadata pass over immutable **C3f real-9B parent trees** qualified 4/4 using compiler-enumerated nodes, with no business edits. Full same-assistant review of 98 claims found **87 supported / 11 insufficient**: direct-read guarantees were overstated; branch metadata miscited restrictions and added reporting semantics. Approval/script mappings faithfully retained missing-capability stops, but their issues remain. **Zero review-supported inactive flows; all blocked.** This is auxiliary mapping evidence, not fresh end-to-end translation or independent Gold. Three new batches: **12 calls, four known flows, one tool, zero public Skills; 651.79 s, 46,625/5,572 tokens**. Mapping-only cost is 228.08 s; C3f parent generation additionally cost 156.61 s. No causal timing or claim-ratio accuracy conclusion. Sixty-eight new focused tests and final **1161 tests + 81 subtests** passed (108.08 s after model completion), with Ruff/diff and three new/C3f replay checks. Zero batch Runtime/business-tool/script/write executions. Next C3h adds clause/source and tool-description-leakage regressions before a fresh full source→flow→mapping batch. Whole-public-Skill, large Runtime and production work remain gated. C3c–C3g changes remain uncommitted/unpushed. See [design and evidence](FLOW-SOURCE-CONSTRAINTS.md).

**2026-09-07 C3f answer-free pilot complete:** four first-attempt 9B proposals on the same known C3d sources now qualify structurally and against host contracts (**4/4 versus 0/4**). Direct/inverted paths and missing-approval/script stops are supported, with no invented Effect or reference answers. Complete same-assistant review of 62 claims yields **49 supported, zero contradicted, 13 insufficient**, and all four remain blocked: shared restrictions lack explicit mapping; the approval read cites a title. Counts overlap and are not accuracy or independent evidence. Costs: **156.61 s, 11,016/981 tokens**, concurrent regression, no causal latency comparison. Zero batch Runtime/tools/scripts/writes or assisted repairs. Fourteen new tests and final **1093 tests + 81 subtests** passed (93.06 s after model calls), plus Ruff/diff and old/new replay. Next C3g separates objective, retained constraints and enforceable mappings before another frozen forward batch; text retention is not enforcement. Public whole-Skill and large Runtime work remain gated. C3c–C3f changes are uncommitted. See [results](FLOW-BOUNDED-FORWARD.md) and [summary](benchmarks/flow-bounded-forward-summary.json).

**2026-09-07 C3e protocol milestone:** 12 first-attempt explicit-answer 9B probes: format-only and format-visible each copied 4/4 targets exactly; JSON-visible copied 2/4, with envelope copying/truncation preserved. Basic constructor expressibility is demonstrated in this environment, not every recursive combination or source fidelity; absent visible Schema is not an established sole root cause. Host-derived generation now removes unavailable constructors, enumerates actual tool/Effect IDs and validates output again before existing compilation; no answer-based repair or authority. Costs including failures: **268.97 s, 14,438/2,735 tokens**, concurrent tests, no causal timing claim. Twenty-three new tests; **1079 tests + 81 subtests** passed (147.50 s), then 23 focused tests after an extra valid-Effect assertion; Ruff/diff and old/new replay passed. The next C3f batch must evaluate the new request without answers before claiming translation improvement. Zero business/Runtime/script/write executions; public whole-Skill and large Runtime studies remain gated. C3c–C3e changes are uncommitted. See [diagnostics](FLOW-TREE-PROTOCOL-CANARY.md) and [summary](benchmarks/flow-tree-canary-summary.json).

**2026-09-07 C3d forward pilot complete, quality gate unmet:** a frozen hierarchical authoring protocol processed the same four known flows once with 9B, without reference trees. Three trees parsed; **0/4 fully qualified**. End outcomes became Effect IDs, approval output repeated to the 2,200-token limit, and the missing-script flow read before its prerequisite and appended terminals. Correct local polarity/reference wiring does not establish whole-flow fidelity. Raw responses and receipts are preserved without repairs/retries; zero qualified-flow source reviews, Runtime, tools, scripts or writes. Total POST time **261.69 s**, **4,652/3,584 tokens**; regression ran concurrently, so timing is not causal performance evidence. Thirteen new and 73 focused tests passed; **1056 tests + 81 subtests** passed (230.10 s), plus Ruff/diff and old/new replay. Next C3e separates minimal Schema/decoder canaries from translation evidence, then narrows constructors/target IDs using actual host capabilities before a new frozen batch. Failure attribution remains open. Public whole-Skill work and large Runtime A/B remain gated; C3c/C3d changes are uncommitted. See [pilot](FLOW-TREE-FORWARD-PILOT.md) and [summary](benchmarks/flow-tree-9b-summary.json).

**2026-09-07 C3d offline compiler milestone:** a bounded sequence/if/otherwise authoring tree compiles mechanically to the existing graph. The compiler owns IDs, edges, joins, lexical read bindings and dominating predecessors, not business polarity or authorization. Existing source/contract/type/DAG/Runtime gates remain. Tree origins and review digests preserve provenance; no branch-value merging, implicit success, dead-code dropping or post-Effect continuation. Sixty new regressions passed; seven hand-authored patterns and twenty seeded trees were compared on two sites with a test-only interpreter and real temporary-file reads through the existing Runtime. **1043 tests + 81 subtests** passed (86.60 s), plus Ruff/diff and old/new evidence replay. Zero new LLM calls or operational writes. Next is a separately frozen 9B forward-tree pilot without reference answers; business-fact semantics and public-Skill/Runtime generalization gates remain unresolved. See [compiler](FLOW-TREE-COMPILER.md) and [example summary](benchmarks/flow-tree-example-summary.json). C3c/C3d changes remain uncommitted.

**2026-09-07 C3c diagnostic milestone, quality gate still unmet:** added host semantic context, cited/ID-selected source projection, mandatory polarity citations, prerequisite dominance and mapping-bound review, without changing Runtime execution. Two protocols each processed four known flows once with 9B: excerpt qualification 1/4, source-ID qualification 2/4. Three complete same-assistant source reviews all ended blocked; a faithful missing-script stop retained a real capability issue, direct read lacked two mappings, and approval still confused inventory state with authorization. Total 306.33 seconds and 9,972/2,986 tokens; zero review-supported inactive flows or tool/Runtime/write/script execution. Twenty-two new regressions; **983 tests + 81 subtests** passed (125.48 s), plus Ruff/diff and old/new evidence replay. No claim of accuracy/generalization improvement. Next C3d first validates a hierarchical sequence/if/otherwise-to-existing-graph compiler offline, while business facts remain bound to real host semantics and review. See [report](FLOW-SOURCE-GROUNDING.md) and [evidence](benchmarks/flow-source-grounding-c3c-summary.json).

**2026-09-07 C3b complete:** C2/C3a committed locally as `2e4f762`, no remote push. A frozen indexed protocol processed eight known hand-authored flows, one domain/tool and zero public Skills, with one 9B call each. Five structurally qualified; three were rejected. Complete same-assistant claim review blocked all five qualified drafts (89 overlapping claims: 63 supported, 15 contradicted, 11 insufficient evidence). No review-supported inactive flow, edits, retries, Runtime, script or write execution. Findings include reversed polarity, inventory-as-approval fabrication, missing prerequisites and unreachable unsupported nodes, beyond explanatory text defects. Seventeen new regressions; 51 focused tests and **961 tests + 81 subtests** passed (87.38 s), plus Ruff/diff and frozen replay. Client timing totalled 360.18 seconds, including per-case preflight and waiting, not pure inference; 6,268/3,596 tokens. Next C3c improves general source-to-purpose/conditions/dependencies/fact semantics and question representation in a new frozen batch. Public whole-Skill work and large Runtime A/B remain gated. See [scope/results](FLOW-FROZEN-DEVELOPMENT.md) and [evidence](benchmarks/flow-development-8-evidence-summary.json).

**2026-09-07 C3a assisted forward-flow closure:** real 9B received source prose/host contracts, not an answer graph. Two named-node development calls failed structure; compiler-owned indexed identities enabled structural qualification on call three, but source review blocked a meta-task purpose and needs_l1/blocked explanation mismatch. All original answers remain unexecuted. The 32-claim review recorded 24 supported, five contradicted and three insufficient-evidence judgments, overlapping two explanation-level roots. A parent/source-bound sidecar revised only two text leaves; fresh same-assistant review enabled three real local reads (campus read_path_completed; IDC needs_l1), with zero writes or DSH loop. Thirty-four new regressions and **944 tests + 81 subtests** passed (86.91 s), plus targeted Ruff/diff checks. This is one known flow and three different development protocols (148.80 s, 2,535/1,348 tokens), not autonomous accuracy or independent generalization evidence. C3b must freeze the protocol and expand heterogeneous development examples; twelve public Skills remain not_run and large Runtime A/B remains paused. See [experiment](FLOW-FORWARD-TRANSLATION.md) and [summary](benchmarks/flow-forward-9b-summary.json). C2 dependencies and C3a were subsequently committed locally on dev as `2e4f762`; no remote push. Raw experimental artifacts remain local and ignored, while Git stores digest-bound summaries.

**2026-09-07 C2 local single-Effect closure:** implemented after user commit `955bc92`. HostFlowGate reruns reads rather than trusting submitted reports. Flow-bound schema-11 plans hash contracts/context/path/facts/provenance/target arguments; standalone schema-10 compatibility remains. Configured instances reject omitted gates and old unbound plans. Restart requires the same host gate. Reads are repeated after approval and again at final dispatch; drift, missing context, access denial and expired read budgets prevent Effect. Existing approval, nonce, verification and recovery are retained. Three local cases reached verified_success, zero-mutation precondition_changed and rollback_verified. Reads use actual temporary files; writes use the LAN mock; no LLM or real-device calls. Twenty-five new regressions and **910 tests + 81 subtests** passed (86.03 s), with targeted Ruff/diff checks. C3 whole-flow 9B generation/source review remains open; twelve public Skills remain not_run. This is a hand-authored local prototype, not source-freshness/atomicity/production proof. Full-payload comparison may over-stop and a final read/write TOCTOU gap remains. See [design](L0-BUSINESS-FLOW.md) and [final report](benchmarks/flow-effect-local-v2-summary.json).

**2026-09-07 C1 read-path wiring complete:** the approved scope requires actual minimal business-flow execution as well as proposals. Added scalar data references, deterministic equality branches, acyclic/reachability/dominance checks and distinct unknown/error stops. Exact host flow/request consent and existing read-contract access checks govern real local reads. Campus performed two reads; IDC performed one then stopped at needs_l1; inventory bytes unchanged, zero model calls/writes. Thirty-three new regressions and **885 tests + 81 subtests** passed (86.49 s), with targeted Ruff/diff and report-digest checks. C2 still needs branch evidence/flow identity bound to PreparedPlan and revalidated before original Effect/approval/verification/compensation. C3 still needs 9B whole-flow generation and source fidelity review; the twelve public Skills remain not_run for whole translation. C1 is a stage-commit checkpoint, not complete phase C or production readiness. See [flow scope](L0-BUSINESS-FLOW.md) and [local report](benchmarks/read-flow-local-summary.json).

**2026-09-07 C intake complete, translation still open:** selected 12 known-development Skills from 11 repositories covering 10 annotated structures. Eight domain labels are inherited, not independently validated generalization evidence. Each entry retains source quotes/offsets/digests, environment dependencies and single-read limits; reviewer is the current non-independent assistant. Eighteen new regressions and **852 tests + 81 subtests** passed (85.87 s), plus targeted Ruff/diff checks. No complete host tool-environment bundle was supplied, so all outcomes are `not_run`, not zero translation accuracy or invalid Skills. No model, Runtime or third-party-code execution occurred. The design checkpoint recommends host tool-context binding plus whole-Skill mixed-flow proposals, preserving L0/L1/missing/unsupported step boundaries and reusing existing compilers. Large Runtime A/B remains paused. See [pilot and proposed next scope](TRANSLATION-BOUNDARY-PILOT.md) and [report](benchmarks/translation-boundary-pilot.json). Entries below retain their historical next steps rather than superseding this current status.

**2026-09-07 B2 assisted single-operation closure:** added parent-bound question-resolution sidecars with exact question/quote offsets and renewed full review. The child only resolves questions; operation/schema/access fields and the original model answer remain unchanged. After the current assistant's non-independent review of 23 base/answer claims, an explicit host authorization performed one real local inventory read (campus-sw1/campus/planned-lab), with unchanged file content and no new 9B call/global activation. Missing host approval, old-review reuse, another device's scope and a negated command were denied with zero provider calls. Sixteen new regressions, 62 related tests and **834 tests + 81 subtests** passed (86.33 s), plus Ruff/diff and evidence replay checks. Next is an 8–12-Skill heterogeneous known-development pilot, reporting raw vs assisted outcomes separately. Autonomous translation, whole-Skill/DSH loops and unseen generalization remain unproven. See [usage](L0-READ-CONTRACTS.md) and [evidence summary](benchmarks/read-local-resolution-summary.json); historical blocked results remain intact.

**2026-09-07 B2 forward/local wiring:** added a new inventory L1 Skill, auditable fixed-file reader and explicit host-bound read execution reusing `ObservationPolicy`. Contract/schema/capability/sensitivity and role/capability/object scopes are checked; implicit-system/wildcard shortcuts are rejected. No production identity or global activation claim. Two 9B HTTP 400 attempts were retained (first error body missing); grammar diagnostics led to simpler decoder hints with unchanged output validation. The third request generated in 10.80 s (547/115 tokens) with correct tool/purpose but two unresolved questions. The current assistant supported 21 source declarations; the proposal remains blocked with zero executions. Separate local wiring tests read actual experimental inventory and verify denials/filtering; they are not model execution evidence. Thirty-two new tests, 86 related tests and **818 tests + 81 subtests** passed (84.87 s), with Ruff/diff checks. Next: evidence-backed question resolution, explicit revision lineage and fresh review; arbitrary-prose intent, DSH loops and unseen-Skill generalization remain open. See [usage](L0-READ-CONTRACTS.md) and [summary](benchmarks/read-local-forward-summary.json).

**2026-09-07 B2b review subchain implemented:** `ReadL05Proposal` maps purpose/operation fields into existing inactive read contracts. An exhaustive source checklist binds L0.5/L0 pointers and source spans/digests using the existing assessment protocol. Missing/incorrect citations, contradictions, insufficient evidence and unresolved questions block promotion; supported candidates remain unauthorized. Local scaffold/review-input/assess commands refuse overwrites. The scaffold is explicitly a reverse editing template, not forward translation evidence. The current assistant's visible-context AI simulation found 10 supported declarations and 2 evidence gaps among 12 claims, plus one unresolved question; outcome blocked, zero promoted contracts/executions. Fourteen new regressions, 70 related tests and **786 tests + 81 subtests** passed (84.86 s), with targeted Ruff/diff checks. Genuine/auditable adapter evidence, forward L1→L0.5 proposals, request semantics and authorized reads remain open. See [read contracts](L0-READ-CONTRACTS.md) and the [diagnostic summary](benchmarks/read-l05-b2b-summary.json).

**2026-09-07 B2a implemented:** the existing L0 v2 models/compiler/catalog/CLI now support inactive `AtomicRead` / `CompiledAtomicRead` with zero, optional or required scalar inputs. Source text/digests and tool/adapter/schema/access declarations are cross-checked, without claiming authenticated origin or semantic correctness. Typed request drafts are instantiated separately; result checks validate shape only. Write gates remain unchanged; effect lookup, Saga and effect L0.5 promotion do not admit reads. Forty new regressions, 76 related tests and the full **772 tests + 81 subtests** passed (85.20 s), with targeted Ruff/diff and local CLI/Python checks. The example is a new hand-authored synthetic fixture, not LLM or generalization evidence; prior sealed artifacts remain unchanged. B2b read-L0.5 source/semantic review, genuine interface evidence and authorized execution remain open. See [read contracts](L0-READ-CONTRACTS.md).

**2026-09-07 A/B1 review:** corrected escaped-string reconstruction and acceptance of numeric overflow as Infinity. Added 20 literal-boundary regressions covering invalid escapes, decoded placeholders, non-finite fixtures and source JSON constants/exponent overflow. All 63 related tests and **732 tests + 81 subtests** passed (81.96 s); targeted Ruff and diff checks passed. The original B1 9B artifact verifies read-only with implementation drift explicitly reported; its one task still needs semantic review. No historical evidence was rewritten and no model/Runtime run was added. B2 remains open: extend the existing L0 type/compiler system for explicit read contracts before source-bound reusable compilation and request instantiation. Existing effect contracts cannot honestly represent the zero-input read fixture; write approval/observation gates were not weakened. See the [correction plan](TRANSLATION-CORRECTION-PLAN.md). Literal integrity is not semantic accuracy or generalization evidence.

### Current phase

**2026-09-07 B1 implemented:** pinned source-tool Schema review now precedes deterministic slot applicability, while 9B outputs only task prose. Unknown/unsupported contracts make zero model calls. Actual inputs and per-call checkpoints are sealed; completed reentry is read-only. Sixteen new regressions were added; 43 related tests and the full **712 tests + 81 subtests** passed. A real 9B call on an explicitly synthetic contract/review fixture took 5.93 seconds and returned only `health_snapshot`; it remains `needs_task_semantic_review`, not an accepted business task, Gold or generalization evidence. B2 reusable L0 compilation/request instantiation and real interface evidence remain open. See [contract-first authoring](CONTRACT-FIRST-TASKS.md).

**2026-09-07: correction A implemented.** Translator v2.1 excludes scoring metadata and requires named ownership/conflict checks. Author v4 supports zero/optional inputs and explicit N/A slots; Catalog v2 no longer invents transaction tools. Targeted regression passed 116 tests; after restoring the local `.venv` link, the full suite passed **696 tests + 81 subtests** (the first run had 19 missing-interpreter environment failures).

A real 9B doc-ingest probe used two calls / 234.1 seconds. It identified zero arguments but still generated a missing-input task and attached N/A to the wrong slot; it was rejected, producing zero review packets and no executable L0. Artifact integrity verified. Historical development-04/06/07 and construct-v3-object-storage revalidate; four older roots have pre-existing rule drift also reproduced at the unmodified HEAD. No historical labels/artifacts were rewritten. Next: source-backed contracts before task generation, deterministic applicability, reusable-contract compilation and separate request instantiation. See the [plan](TRANSLATION-CORRECTION-PLAN.md) and [diagnostic summary](benchmarks/translation-correction-v4-summary.json). Dated entries below are historical, not current acceptance claims.

**Active phase: L1-to-L0 translation generalization gate.** ES-P0 Runtime mechanics and small-sample wiring remain `local_hypothesis_supported`, but they are not broad translation evidence. No scaled Runtime study is meaningful until the Translator passes post-freeze, cross-Skill, cross-repository, and cross-domain unseen cohorts.

On 2026-09-04, an audit found that legacy answer-hidden reviews still exposed category-bearing IDs and fixed task order. Those results remain sealed but are downgraded to metadata-cued development diagnostics. A new opaque-ID, independently called single-task protocol records model-visible inputs, keeps mappings scorer-side, and validates resume/output bindings. Revalidation on the same 9B and development-07 packets completed 12/12 calls, but reached only 7/12 behavior agreement and 10/12 construct alignment, with Gold-queue eligibility false. Natural-task construct quality and evidence-grounded isolated reference answers now precede Translator evaluation; AI role simulation never becomes human-independent evidence.

Construct checks v3 stop parameter insertion, distinguish actual typed values from author examples, retain conflicting literal offsets, and block nominal placeholders and unsupported evaluation meta-tasks. A read-only re-audit blocked seven of 16 previously accepted development constructs; all legacy artifacts and labels remain unchanged. A fresh 9B object-storage probe still produced the wrong meta-task after repair and was rejected (one Skill, three tasks, two calls). This demonstrates interception, not improved generation accuracy. Targeted regression passed 71 tests; full regression passed 658 tests plus 81 subtests. Arbitrary prose conflicts and source API/step fidelity still require evidence-grounded review. See [construct quality](TRANSLATION-CONSTRUCT-QUALITY.md).

Source-evidence review v1 now derives exhaustive field/step claims and resolves exact citations. Its real 9B doc-ingest probe returned 12 claims and located the no-argument/schema contradiction, but omitted task citations and therefore failed the protocol. Read-only classification was also inferred too strongly. Raw responses remain unchanged; valid source review, zero/optional-argument construct support, isolated references and Translator generalization are still open. See [source evidence alignment](TRANSLATION-SOURCE-ALIGNMENT.md).

The 2026-09-03 reset adds a Gold-blind Translator v2, mandatory pre-run Skill–Task–Tool construct review, and a digest-bound admission gate. The current known development inventory contains 100 static public Skills from 72 repositories and nine domains, with zero third-party execution. Fifty-three are Runtime-package ready, 18 are conformant translation-only partial-context inputs, and 29 format variants are robustness-only; the 71 primary inputs form seven repository-preserving development batches. This entire visible inventory has `proofCohortEligible=false`.

The semantic case-authoring lane is now implemented. It binds Skill quotes, explicit parameter evidence, a generic non-executable Tool Catalog, conservative recorded normalization, deterministic structural rejection, and answer-hidden review. Development batches 02–05 exposed reusable failure families while the implementation evolved. Compiler-owned operation IDs, required exact source-span IDs, control-field exclusion, mechanical read/non-write closure, distinct challenge text, and reviewer consistency became fail-closed gates. Without further changes, the final implementation accepted 16/16 Skills on the first call in development-06/07, while all 48 answer-hidden tasks achieved same-model behavior agreement and alignment. Failure-discovery authoring coverage now spans all 71 primary known-development Skills. These are development diagnostics, not independent Gold, Translator accuracy, unseen generalization, or Runtime qualification.

Runtime evaluation now requires the same frozen Translator to pass at least three disjoint post-freeze unseen cohorts totaling at least 50 unique Skills, 15 repositories, eight domains, and 600 cases. Every cohort must pass the safety, recall, macro-F1, exact-parameter, source-evidence, artifact-loadability, and alignment gates. Without a valid admission artifact, the DSH runner permits only a 1-case × 1-repetition wiring smoke and marks it ineligible as research evidence. See the [L1-to-L0 generalization gate](TRANSLATION-GENERALIZATION-GATE.md).

Completed evidence includes 60 deterministic Runtime scenario runs, 30 mechanism-ablation probes, two 60-Skill translation studies, 120 real paired DSH sessions across 9B and 7B models, precision/coverage curves, scenario-level Wilson intervals, and a final regression of 506 tests plus 81 subtests.

For 9B, treatment reached 86.67% task completion and 100% execution precision with zero unsafe execution, false commits, or invalid actions. The weaker 7B treatment also kept those action-safety metrics at zero, but 18/30 sessions failed at the DSH/model availability layer and the model is therefore not availability-qualified.

This is local transparent-development evidence—not production probability, hidden-set generalization, or vendor-device certification.

The ES-P1 research infrastructure is now ready, but independent-generalization evidence has not yet been collected. Research Freeze v1 jointly binds Git cleanliness, Runtime, harness boundary, 21 contracts and readable trajectories, evaluator, authoring protocol, model artifact, environment, and the ES-P0 baseline. Study Plan v2 and private Manifest v3 require that verified freeze digest. Qualification Report v2 adds family/profile/risk/disposition slices, Wilson intervals, zero-event upper bounds, mutually exclusive outcomes, and separate critical-semantic, undeclared-effect, and approval/risk-weakening escapes. The focused freeze/qualification regression is 29/29 passing.

These controls prove that an external study can be preregistered and drift-checked; they do not prove hidden-set generalization. The remaining ES-P1 work must be performed outside the Runtime team: independently author 200–500 private cases, obtain two blind reviews and bound adjudication, freeze exclusions/stopping rules, run the same 9B artifact at least three times, and publish only aggregate Qualification Report v2 evidence. The local Graph gate, provenance DAG, and Runtime-stage timing milestone is now implemented; it does not replace the required private paired evidence.

The I11 authority cleanup is also complete. Product DSH adapter code no longer imports evaluator or golden-set modules; retrieval parity now lives in `evaluation/` and uses memory-only state. Frozen A2A, trajectory-learning, and historical L1-shadow extensions are loaded only when their explicit commands are invoked. Active documentation no longer presents P1.9 Decision Plane/Canary productization or holdout tooling as current evidence. README, ARCHITECTURE, HLD, LLD, SSD, and the unified Skill-to-system interaction guide now use one lifecycle vocabulary: offline authoring versus online execution, read-only fallback versus write safe-stop, and evidence-backed terminal outcomes.

The local Runtime-convergence prototype now uses a journal-backed Typed Graph scheduler to gate normal execution, rejection, precondition drift, indeterminate outcomes, compensation, and crash-boundary recovery. Unknown crash work is recorded as skipped/indeterminate and only reconciled by reads; Effect is never replayed. Runtime inspection exposes graph conformance, a privacy-minimized cross-step Evidence provenance DAG, and stage latency separated into Runtime-active and approval-wait time with an explicit Reasoning/LLM exclusion. The current repository regression is `599 passed, 81 subtests passed`. The 506-test number in frozen ES-P0 evidence remains historical and is not rewritten.

A separate repository-external synthetic evidence path is now operational. It sealed 240 model-authored cases after two blind model-review prompts and digest-bound packaging, covering six Skill feature families, ten transaction/fault patterns, six MCP domains, and three language groups. The loader structurally fixes `officialEsP1QualificationEligible` to false. qwen3.5:9b produced 240/240 schema-valid proposals; 235 passed every trusted Oracle, five remained fallback-only, and no rejected proposal received Runtime authority. Across ten stratified scenarios and three real-DSH repetitions, Treatment improved task completion from 76.67% to 93.33%, reduced unsafe executions from four to zero, and reduced p50 latency from 103.6 to 64.3 seconds. All 17 applicable Runtime audits were valid. Residual Treatment failures expose pre-Runtime L1 factual-decision and safe-stop availability limits. This is synthetic evidence, not independent generalization or a production probability.

The earlier `ES-P1-Wild` 15-Skill authoring path remains a historical pilot. The expanded static development inventory now has 100 accepted Skills from 72 repositories. It is used to discover translation failure modes, not to claim unseen generalization. Marketplace packages remain untrusted and `static_only`; a separate disposable `ES-P1-Sec` sandbox will cover malicious-package behavior. Public-market evidence cannot replace the formal private holdout gate. See [ES-P1 public Skill-market corpus](ES-P1-PUBLIC-SKILL-CORPUS.md).

An explicitly non-authoritative qwen3.5:9b draft-assistance lane has also been exercised on the 15-package author kit. Fourteen assignments and 42/45 slots passed protocol and safety-shape validation; 12 assignments needed repair calls, p50/p95 latency was 59.2/97.2 seconds, and one persistent proposal/effect-budget contradiction remained rejected. The artifact passed binding and digest inspection. It contains no trusted Gold or execution authority and only reduces blank-page work for independent authors.

A digest-bound tested-Skill library now makes the corpus inspectable: users can search 15 Skills and click through 22 inert text files, exact provenance, licenses, 45 task slots, and 42 model drafts. The offline page has no installation or execution path and reports zero third-party execution. Git retains a metadata-only pinned index while complete third-party bodies remain in local generated artifacts.

A separate Case Author Review Kit exposes only the 42 candidate user prompts while withholding all model semantic labels. Its 45 slots require explicit human accept/edit/from-scratch/reject decisions, rationale, attribution, and independence disclosure; Gold/Oracle fields and sealed-source drift fail closed. Review Kit v3 adds Tool Catalog v2 and fixture-state validation. A generic declarative fixture MCP now provides six reviewed operations, independent SQLite state, approval/fault inputs, call-digest auditing, native/Runtime/safe-stop boundaries, and a tested official stdio transport without executing package scripts. A blind Gold Author Kit can export only after the human gate completes, and the Public Paired Study Kit physically separates Agent inputs from scoring Gold. The Gold-blind qwen3.5:9b translator, sealed binder, declarative transactional Runtime, and real-DSH paired runner are now implemented. Controlled one-case read and nominal-write smokes passed in both arms; Treatment's write issued one Effect and independently verified it, while deterministic tests cover compensation. The human workspace remains 45/45 pending, so no human-authored Gold or formal public paired qualification exists. All new ES-P1 model runs use qwen3.5:9b; 7B is frozen historical evidence only.

The downgraded `ES-P1-Wild-Sim` local protocol is complete. Fifteen public Skills and 45 cases ran for three repetitions, producing 135 paired observations and 270 real local DSH arm executions. Control completed 82.22% and Treatment 97.78%; the L0 route improved from 21/42 to 42/42, p95 fell from 109.3 to 56.1 seconds, and both arms recorded zero unsafe executions and zero false commits. Every repetition was 37/45 versus 44/45. Post-run translation-route agreement was 43/45 with zero unsafe Runtime accepts. Virtual Case/Gold provenance fixes `humanIndependent=false`, so this completes only the role-separated simulation protocol and does not alter the open independent-human ES-P1 gate. See [ES-P1-Wild role-separated simulation results](ES-P1-WILD-SIMULATED-RESULTS.md).

Only after ES-P1 passes will ES-P2 qualify the same abstractions on at least one real router and one controller or management path. ES-P3 scales the evidence and packages a paper-grade artifact. Trace-based Experience Compilation is ES-P4, a separate research line with no automatic activation. Production identity, supply chain, governance, HA/DR, WORM audit, and SLO engineering remain frozen until independent and real-network evidence justifies them. See the [v1.1 research instruction](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md).
