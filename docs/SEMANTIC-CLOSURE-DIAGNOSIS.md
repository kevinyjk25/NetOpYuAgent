# 语义闭环修复根因记录 / Semantic Closure Repair Diagnosis

## 中文

2026-09-14 当前：[职责合同与确定工件检查实测](SEMANTIC-DUTY-CONTRACT.md)已完成。关键新发现：自由文本任务引用槽会混入观察、职责解释会增强原约束、notes 审阅仍会混入正文并自相矛盾；独立备注失败还会阻断原串联编辑图。显式接续和窄范围类型化渲染改善具体候选，不证明这些转译/审阅问题已解决。下方方案待确认文字属于历史记录。

2026-09-14 最新：[上下文/编辑边界机制验证](SEMANTIC-CONTEXT-REPAIR.md)结束，触发方案复盘停止点，**不是阶段完成**。原始任务与读回数据已逐字/逐回执保留，任务角色和资源参数已到达实际 9B 审阅请求；但查询/否定误判继续，肯定意见还会阻止必要的编辑调度。Mesh 补回部分指标，Phoenix 检出建议遗漏，却未形成完整有效交付。10 次调用、4 个已知关键缺陷完全修复 0 个。新增的 notes 所有权和全稿标题追加修复不自动证明自由文本语义正确。当前需要确认的是职责合同、适用工件验证器和聚焦核查方案；不再直接扩大测试或累加同例 Prompt。

2026-09-14 续：[候选不可见的来源分析＋任务见证诊断](SEMANTIC-WITNESS-DIAGNOSTIC.md)已跑四个保留样例。来源分析能提取已读事实，后续审阅仍会混淆索引与数据、执行约束与正文交付、相邻但不同的谓词；不能靠增加 schema 字段宣称语义通过。实验保持隔离，另已修复原审阅的任务无交付引用、备注错投正文两个实现缺口。实际 notes 修订和适用的工件验证仍待完成，下一轮不直接扩样或重复同类提示词试验。

### 2026-09-14：跨模型对照与确定分块修复

[四例 9B/27B 对照报告](SEMANTIC-REVIEW-MODEL-CONTRAST.md)已经封存。27B 新检出 Phoenix 的建议遗漏，但仍漏判查询实现错误和“未批准→未审阅”扩大；Mesh 的局部遗漏意见仍与陈旧缺数据判断并存。两个模型都有盲点，不能直接归因为“9B 太小”，也不能据四例断言所有大模型都无效。27B 有 2 份接口有效、2 份缺引用拒绝；5 次实际尝试含 1 次超时，未知用量保留。没有编辑原稿或执行 Runtime，没有重评分。

对照完成后才改代码：`_blocks` 原来在数字后句点切句，把 `1. 正文`／`## 3. 标题` 拆出孤立编号。`bounded-draft-review/v8` 将编号与同一行内容合并为导航单元，完整文本、位置、否定和权限边界不变；它不证明语义、不放宽缺引用校验，也不解决所有标题/缩写切分。76 定向检查通过；原四例冻结 payload 和结果不重建。后续核心修复仍是任务实际满足、证据集合的作用域和适用的工件验证，尚待实施与新样本验收。以下记录均为此前当时状态。

### 模型切换前的实际输入核对（2026-09-11，零新增调用）

已检查 v10 保存的实际 `wireRequest.messages`，不只检查上游业务请求。两类缺陷的必要输入已送达，不能以“补读没有返回”解释这两个具体结果：

- **查询审阅**：用户消息内 `reviewInput.sourceSpans` 的 `s012` 含 `[2026-09-10T09:00:00Z,2026-09-10T10:00:00Z)`；`statement_checks` 的 `c007` 是完整查询代码。回复却以“s012 说明未执行查询/无结果”作为 `c007=grounded` 的依据。`c023` 只对应到正文中的时间窗口声明 `d003`，没有证明代码实现该边界。固定编号解决了结构绑定，未解决理由与被审对象的语义对应。
- **Mesh 修订**：第二条消息的 `readOnlyContext.observedData` 中，`s008` 含两组 2,000 请求、10/100 次 5xx、400/900 ms p99、南区对照和未知事项。`locatedRepairFindings.c021/c022/c027` 已要求补指标和计算。实际新稿仍声称缺样本，只修正索引路径。因此本例不是编辑器没收到指标或没有修订要求，而是未按已有证据完成修订。
- 这排除了上述具体输入缺失解释，**没有排除长上下文、旧稿干扰、接口表达或模型能力的影响**。没有运行模型、增加权限、修改旧评分或新增未见样本；仍等待是否允许本地 27B 小批复核对照的决定，不把重复同配置调用当作诊断进展。

复查目录：`artifacts/translator-v2/semantic-closure-known-v10-20260911/`。查询 `review-before/model/review-before/request.json` 文件 SHA-256 为 `979e5de99806b12d2f0d40f3aa28c42b9541bdc8f2f93e140fc9a9eaf65c8ef5`，对应原始意见为 `fe734a0f8e1dd9cb5c403a25fffa8b4a2e31f8fd1224e1a238a7aac372c36c30`；Mesh `repair/model/e00/request.json` 为 `a362e4c96a747ab0b11b9232ad888552a3945e6c7ad7dabcf8cfcfaff80fdb7c`，实际 `repair/materialized/candidate.json` 为 `b6090d9a05aac5a75777e28b618a2f7769b797aceffe9b1b439e3e790581f7b6`。这些是已封存制品的文件摘要，不是语义评分。

### 最新机制复盘（2026-09-11，仍未达阶段出口）

已知 v9/v10 已结束，仍未通过阶段。v9：15 次调用，83,600 输入／12,172 输出 token；v10：5 次，37,732／9,285。v9 诊断摘要 `71035359b1cfb5714599da4cad02620fd73b1c8444c6f04278179c27dd045111`；v10 为 `fca943a63ff6e879a7cdb2eaa03a5be59476858e1f83c628c145a3884dce26c9`。各自的 `content-review/report.json` 绑定实际稿件，不用模型 supported 数打分。

- v9 补回 CAPA 的 Omar、Phoenix 的 120＋30＝150 token；但旧备注和有用后续建议仍有问题。Mesh 受旧稿影响忽略实际新指标；IRQL 恢复代码形态但查询仍不正确。两份审阅出现重复/缺失 ID，正确阻断，但未完成任务。
- v10 改为宿主固定对象键的检查单元。两次初审不再因检查 ID 冲突失败；未增加上下文，使用本地 `$defs/$ref` 共享结构。初版逐项展开 Schema 的零调用预算失败也保留。**IRQL 初审/终审仍全部支持错误查询；Mesh 发现漏项但编辑只修了路径，未补计算；终审缺来源引用而停止。** 接口修复不等于语义修复。
- 当前工程验证为 **77 定向、全量 2,891 项＋81 子测试、51 个变更 Python 文件 Ruff、文档链接与 diff 检查通过**。阶段出口仍关闭，不提交/推送，不扩大 A/B。

同类开放语义问题仍未稳定改善，下一步不继续靠同一 9B 的 supported 自证。建议保持 9B 生成，仅用本地已有的 `qwen3.6:27b` 对 4 个已知失败例作一次复核对照，以区分模型能力与表示/流程问题；**已请求用户同意，尚未切换模型或执行该对照**。这不是承诺 27B 一定修复，也不是降低既有判据或改写负结果。

[第二批冻结验收](SEMANTIC-CLOSURE-TRANSFER-V2.md)已结束且未通过：6 Skill／12 任务，0 个完整实质任务、3 个正确边界、8 个局部、1 个失败；67 次 9B，342,581 输入／52,968 输出 token。安全执行与正确回答必须分开：未观察到越权调用，但多个回答存在语义丢失或不正确操作建议；模型终审仍有误支持。

本次机制根因不是笼统归因于“9B 太小”或“Schema 不够宽”：

1. **来源调度错误**：所有参考文档超预算时直接退到第一页，实际完整 CAPA/Phoenix 入口都能放入既有预算。改为完整入口优先、再加入完整参考文档；源摘要和实际已供给比例分别报告。
2. **已执行状态没有成为清晰输入**：实际补读已成功，却以匿名观察字典和旧 LLM 待办一起交给 writer。新增宿主生成的工具/参数/结果位置索引；历史和本次读取完成状态分列，旧建议不再混入当前证据区。它不刷新权限、时效或证明返回内容为真。
3. **增量处理缺失候选本体**：补读 writer 不见上一稿，只见其待办，容易丢掉原查询/工件形态。现在保留上一稿为待更新候选，原任务与实际观察仍优先，旧稿不升级为证据。
4. **缺失信息没有现成文字位置**：单一整稿编辑所有者可接收未定位负面问题；多章节不广播，不能通过仅选择一个章节伪装成整稿。改的是编辑归属，不是自动语义批准。
5. **审阅局部格式失误丢弃全体问题**：一个 `preserved` 缺位置导致整份审阅失败。现在只把该项降为未证实并保留诊断，其他已合法定位/引用的问题可继续使用。未知来源、重复/错组 ID 和 Schema 错误仍拒绝。

后续 `semantic-closure-known-v9-20260911` 只做已知失败机制回归，不改 v2 分数，不算新来源。首个 CAPA 修订确实补回 Omar，原职责/日期/否定保留，但未证实的旧备注仍在；不能把一个漏项修复叫作完整语义任务成功。更深的隐含因果、技术工件正确性和自由文本安全建议仍须实际审阅，不能以工程测试数量替代。

v4/v5/v6 均为已知样本诊断，不能改称新来源。v4：3 次 9B／18,331 输入／4,742 输出；v5：18 次／102,946／9,168；v6：6 次／35,583／2,928。原回执与失败分别保留在 `artifacts/translator-v2/semantic-closure-known-v{4,5,6}-20260911`。v6 诊断摘要 `ec4e6665c9a8f4cce4b86ad5e87be4b096e34b7ffb1042bbec6a5ec9591f86a7`。

范围约束阻止了营销稿的跨章节复制、交接稿的重复标题；精确引用锁保住 README API，但编辑器再次从 Skill 模板猜测 MIT License。**语法/范围保护没有解决自由文本的语义判断。** v5 还删除了已观察到的交接链接。任何终审 supported 都不抵消这些实际退步。

连续失败后不再继续调同一编辑 Prompt，而修正触发机制：此前即使没有定位负面问题也让全部章节重新生成，扩大了无谓变动。当前默认只处理一个宿主章节内已定位的负面意见；无定位则保留原稿、显式保留未解决项。词面差异不再等价于必须编辑，正面审查也不等价于正确。局部编辑仅可见当前正文和其他章节索引；完整原任务/Skill/观察保留，父稿留冻结与终审。引用锁保障精确保留，不判断引用是否适用。已知回归及后续冻结验收必须分别计费与报告，不把少调用、不改稿或安全停止计为新增语义成功。

v7 的两次终审共 12,313 输入／4,349 输出，零编辑调用且原稿保留；但驱动进程运行时修改了交付元数据及冻结声明，存在已加载模块与后续磁盘源码记录不同的风险。因此完整保留该轮，**不用于固定实现认证**；v8 新进程增加前后完整指纹核对。此为执行纪律问题，不是语义提升或应该隐去的重试成本。

最新续报：已知 v3 四任务已封存，**2 完成、1 局部、1 失败**，另含一次未修好的网络审阅探针；共 17 次新 9B，110,053 输入／21,130 输出 token（不重复计入复用的 v2 初稿调用）。无补读保稿分支保住网络解释和写作条件；营销编辑却删掉两项活动提案。摘要 `0af08df95ead7f00b77e10aa47900655f9603f57f5a9d156eda6f03104b09b69`，见 `artifacts/translator-v2/semantic-closure-known-v3-20260911/diagnosis/report.json`。这不是迁移验收通过。

随后 v4 将“事实断言／观察覆盖／任务检查”改为不同响应类型，保留原始 `role-review.json`，由宿主核对组别、ID、定位后映射到统一审阅格式；依据和理由先于结论。任务检查不再要求原稿自称合规，但**隐含错误因果仍漏判**，不继续靠同例提示词重跑声称已解决。编辑边界新增 ATX、Setext 和独立非整句粗体标签识别；它成功拒绝了将整篇塞入开头段的提案，没有生成损坏稿，但仍不是可用修复。

v5 当前验证“只读周边上下文 → 本次精确可编辑片段”的分离输入，完整原稿可由 before＋target＋after 原样重建，范围/标题检查不放宽。同时对历史交接、README 稿做新审阅/修订回归：旧回执、来源归档、应用范围和稿件摘要先核对，**仅导入数据，不导入或重算旧 AI 意见，不产生新来源计数、业务读取或权限**。新样本仍须另行冻结后选择。

最新状态（2026-09-11）：[迁移 v1](SEMANTIC-CLOSURE-TRANSFER-V1.md)失败后，已知开发 v2 已封存：6 个已知任务，2 个限定任务完成、4 个局部可用，36 次真实 9B，218,870 输入／20,228 输出 token；不是新样本或通过出口。答复/支持观察双通道保住了指标计算和测试重试解释，但空补读分支仍损坏身份追问，无新增观察的重复生成改坏网络解释和条件句，编辑器仍有预算阻塞。证据位于 `artifacts/translator-v2/semantic-closure-known-v2-20260911/{evidence,diagnosis,judgments}`；证据摘要 `c2163c7d2ec7caaa68427006750700bebc707e80f4e75d32983df4f3d1ade3c7`。2,827＋81 子测试是 v2 工程回归，不是语义成功率。

v3 正在验证三项通用机制：①宿主 `reason_if` 根据严格读取结果决定是否调用模型；没有新观察则原样保留旧候选，追问另存，既不重写答案也不伪造回执。②选择器传输 Schema 按 decision 分支，追问/回答只允许空 requests；真正的读取仍需原严格验证和独立 Gate。③审阅/编辑输入改为可读工作表，保留全部原任务、源文、观察和候选原文，减少重复导航元数据；旧可逆文本池保留为历史工具，不能把“可反解”当作“小模型容易理解”。原预算不增加，超过容量仍明确停止。

v3 第一次真实审阅诊断**仍然漏判**：旧网络稿里的“是否在因关闭 journal 而丢弃之前收到 payload”包含未经支持的丢弃因果前提，9B 对独立句子和 notes 都报 supported。完整请求/回复保存在 `artifacts/translator-v2/semantic-closure-known-v3-20260911/network-review`。细分句子不是语义原子，不宣称审阅器已经可靠；不能继续把正面模型评分当验收。当前验证预防性保稿分支是否避免这次无证据改写，最终任务质量仍需逐项核对真实稿件。该失败不会被新链路结果覆盖。

已知开发 v2 的首个真实执行还发现独立问题：选择器生成正确的 `decision=clarify` 和身份/环境追问，却同时给 `requests.read_export={}`。接口要求工具对象必须有 path，因此本次无读取分支被拒绝；provider 未调用，原稿和失败均保留。后续应在模型传输层用带判别的选择 Schema，明确无读取分支只有空 requests；实际执行仍通过原严格 Schema 和独立 Gate，不能通过猜 path 或静默批准无效读取“修复”。本轮冻结实现不改动，先完成已选样本再接入下一版本。

历史审计另有实现缺陷：1.06 MB 冻结输入超过业务 JSON 读取上限，导致成本统计中断。现使用独立有界 16 MiB 审计读取器，完整校验归档/摘要，不跳过文件、不增加 Runtime 输入上限。9 月 11 日旧开发目录已审计 84 调用、540,585 输入／19,200 输出 token，摘要 `f0aca7e7040dfa0861fb97b862352fa8b21c7285181a4264dab938aaf1c3b631`。下文保留当时的失败与调整历史，不把旧不足包装成已经解决。

2026-09-10。两个 `located-slots` 开发尝试没有修好已知交接缺陷，按[阶段规则](SEMANTIC-CLOSURE-EXIT.md)暂停同链反复重跑，改做机制定位。原报告与模型回复保存在 `artifacts/translator-v2/semantic-closure-20260910/`；不是新样本评测。

### 可复查的根因

- **主题相似被误当成细节完整**：9B 对 31 个交接文本单元全部报 supported；其中原文 `owner Alice;` 的草稿引用没有 Alice，理由却是“上下文暗含归属”。原文事件开始时间 `08:10` 也未出现于其引用。原句定位能暴露矛盾，但不能令模型意见自动正确。
- **错误审查意见污染修订**：新增词项差异已定位 Alice、Chen、08:10 和未确认交接等信息，修订器仍重复“全部有依据”，生成原文不变/仅格式变化的替换。问题不是 JSON 缺少字符串字段，也不是多堆测试就能解决。总 verdict 不能作为下游事实或验收依据。
- **编辑接口切断了 Markdown 结构**：按空段划槽会把带空行的 fenced code block 切开；模型补上自认为完整的 fence 后，整稿可能反而失配。应由宿主提供完整安全块，不让模型靠字符偏移重建上下文。
- **缺观察无法靠改稿填补**：README 原先只有目录条目，却包含项目安装命令/API 示例。第二次修订仍用“请再确认”的提示包住猜测内容；这不是补足事实。必须实际受控读取配置/代码，或保持局部结果。

### 历史调整方向（2026-09-10，后续修订见下）

1. 保留双向审查作定位辅助，不把正面 AI 评分转给修订器充当事实。新增“原文子句—实际草稿引用—未对应词项—编辑槽”的局部修复视图，先用一次有限定位诊断验证；差异并不直接等于语义错误。
2. 多槽方式仍发生跨槽拷贝，改为“先给每块声明证据状态，再提出完整修订稿”，宿主计算精确差异，最多 8 个改动区间、一次修订、原 2,048 输出预算均不增加。空改动不代表修复。全部原文、旧方案和失败保留。
3. 原自动提案续接有限只读混合图：模型选读/追问，宿主依据当前资源权限独立准入，再由原 L0 执行器读取；历史快照只做分析参考，不能续期为操作依据。候选草稿仍未经语义批准。
4. 不更换 9B，不扩大正式 A/B，不清除失败，不宣称阶段完成。先解决两例实质缺陷，再冻结后迁移验收。

### 首次定位结果与续接缺口

`handoff-focused-diagnostic-v1` 单次 9B（8,816 输入 / 789 输出 token，59.71 秒）在移除正面审查意见、仅提供原文与局部差异后，实际补回了 Alice 事件归属、08:10 开始时间、Chen 调查责任和 Bob 尚未确认交接。修订通过摘要/编辑范围/格式校验；它是一次已知样例的局部定位诊断，不是 Runtime 验收或独立泛化。

README 续接第一轮暴露了历史观察投影缺失“工具＋参数”的问题，重复列目录后停止；第二轮恢复合同绑定的工具/参数/结果三元组后实际读到 `pyproject.toml`，但模型提前选择 answer、仍猜测 `main()`，不能接受其草稿。当前将前轮待办以非权威进度信息传入，并保留实际读过资源的禁止重复检查。两批原结果完整保留，不能把 `bounded_continuation_completed` 解释为业务完成。

### 后续根因与改造（不覆盖上述失败）

- `handoff-integrated-v4` 完整三步链补回 Alice、Chen、08:10 和 Bob 未确认状态，严格图完成但 ResultContract 仍为 partial，AI 的全部 supported 不清除职责。
- `documentation-continuation-v3` 经独立准入实际补读配置与代码：6 次 9B、2 次只读，生成真实 `summarize_ms` 示例，但无依据写了 MIT。API 缺观察问题已缓解，整稿仍不接受。
- 自由抄写 `draft_quote` 曾把源码当成交付稿；改由宿主提供 `d000...` 精确位置。随后又发现模型将“许可证无依据”的判断挂在参数表编号上，因此宿主不使用模型重新指定的段落作为事实。
- 修订器会把旧草稿的免责声明理解为保留猜测的理由。当前只将完整原文和实际观察作为修订依据，旧 AI notes 保留在审计，不作为保留事实的指令。单独调整此项仍未修好，负结果保留。
- **诊断正确不等于动作已完成**：`documentation-evidence-first-diagnostic-v2` 明确标记许可证无依据，却仍输出 MIT 并声称已经移除。程序拒绝了原样矛盾输出；这不是字段不够丰富，而是模型判断与文本操作脱节。
- 当前宿主对模型明确标为无依据却仍保留的唯一匹配块，执行带提示的暂缓展示；引用原文、判断、被暂缓内容和原回复全部留档。位置歧义、格式损坏、超过差异预算仍拒绝。Markdown 末尾换行差异也有专门回归。**暂缓不等于证明该内容为假**：9B 同时错误地怀疑有效的安装命令，因此必须记录过度暂缓，不能只报“消除幻觉”。
- 暂缓后的用户提示只描述宿主实际做了什么，不把模型“项目缺元数据”等错误理由提升为事实；完整模型理由仍在 evidence-report 内。终审仍为有界 AI 意见，不能自我批准。

### 2026-09-11：重复失败后的机制审查

旧整稿方案没有稳定收敛：后续交接再次漏掉 Alice 的事件责任；README 去掉 MIT 的同时错误暂缓了有效安装说明。因此之前单轮改善不等于两例已稳定解决。

- 隔离编辑减少跨段落错配，但仍有“指出遗漏却没有改进文本”、无变化补丁、重复补充和超长自我解释。完整 `handoff-range-pass-v1` 在第 5 次调用失败；v2 完成 8 次调用却有 3 个 no-op、9 个提议范围，未合稿。两轮不算语义成功。
- 一个受控传输诊断中，相同请求仅切换 `format`，JSON 模式补回了关键关系，Schema 约束模式没有；JSON 回复又漏了冗余回显 ID。宿主已接管候选身份与位置绑定，实际补丁仍做本地严格验证。**一次差异不是 JSON 模式普遍更准确的证明。** 同一 9B 的单次 native-thinking 诊断也没有修好已知遗漏，不据此更换模型或隐去成本。
- 新定位保留完整观察句的主语、分号关联责任与条件，不依赖审查器正面结论。词面分数仅建议位置，平分不强行定位，全部原始上下文保留。它是导航，不是语义 Oracle。后续整稿仍漏掉负责人，说明只优化定位不够。
- 引入**引用式修订**：模型可选择已有观察的精确片段 ID，宿主核验父来源、偏移、引用 ID 后原样渲染到实际草稿，不再要求模型重抄确定事实。解释、未知说明仍允许生成。来源选择和适用性仍是模型判断，复制准确不代表语义正确，更不能获得执行权限。
- 每个编辑单元最多一个范围，最多 8 单元、一次修订 pass；no-op 记为未修改，不计修复。**总调用预算已改变**：来源审查 1 次＋最多 8 个编辑调用＋终审 1 次，编辑单次仍 2,048 输出 token。这是成本增加的机制替换，不是保持总预算的性能改善。
- README 暴露章节标题与正文被分到两单元的问题。新增完整章节切分，不截断 fence，也不在只有相邻标题但无正文的位置切分；事实修订不得改变标题序列。旧段落模式供历史重放，不回写旧摘要。
- **生成字段依赖与摘要规范化冲突**：图资格检查排序对象键；真实 `documentation-tagged-pass-v1` 按 `content → end → operation → start` 生成，先写正文，后选 `copy_source`。生成端现在按 Schema 原有 `required` 数组恢复操作优先，本地 Schema 语义和规范化摘要不变，顺序策略纳入配置摘要。`documentation-ordered-pass-v1` 已确认顺序生效，但仍把引用内容写为正文，说明顺序不是唯一根因。
- 进一步分离**引用 ID 的枚举槽**和**自由文字槽**，由操作标签选择唯一活动分支；未选中的载荷完整留审计但不应用。引用须在当前来源目录内，宿主从原回执片段渲染；不再用一个自由字符串同时表示 ID 和正文，也不扩展 Runtime 尚未支持的 `anyOf`。复制精确仍不代表选择正确；待审问题和模式选择失败照常计入。
- `documentation-typed-pass-v1` 能无损复制，却用配置覆盖安装命令、把引用塞进代码例子、用无关构建信息充当许可证；它是 **7 次调用后的可用性失败**，不是成功。因此引用目录改为仅含已定位到本章节的片段，带 fence 的章节不允许引用替换。后续 `documentation-scoped-pass-v1` 保住安装和示例，但仍保留 MIT。
- 负面疑点曾被过度过滤。现在按实际正文与负面 rationale 的词项交集检索疑点，不信任可能错位的 claim 编号，也不传递“继续假设”等审查建议。单独恢复疑点后 9B 仍原样保留 MIT，因此这一项也不是语义成功。
- 新增**窄范围未确立值暂缓**：仅对原稿中独立一行的短原子标识值（当前原型为拉丁字母起始、字母/数字/`_.+-`、2–64 字符）、被负面疑点明确提及、没有观察词项依据、且修订仍独立原样保留的情况暂缓展示。排除代码 fence、不同段落和已经改为未知的说明。它不支持所有语言或一般事实判断，不证明值为假；只是显式风险策略，误报与原值继续留档。不得把这种暂缓称为 9B 理解改善或百分之百语义准确。

所有这些是已知开发样本的机制修复；冻结后的 6 Skill/12 任务迁移验收尚未完成。性能需要连同审查、修订和失败调用一起计算，不能只取最快的最终一轮。

### 冻结 transfer v1 首次结果中的新缺口（进行中）

本批先冻结实现，再保存六个来源和十二项任务。前三个 authoring 失败不是执行异常：两项把 Skill/参考文档路径当作业务 `read_export` 参数，所引标题并不含路径；一项在边界注释中引用了尚未供给的页，导致整个提案被 Schema 拒绝。**原先有据的第一步读取也一并丢失**。这暴露“有效前段、未接受后缀、非权威注释”没有独立处理的问题。冻结结果仍为失败；后续离线诊断从原回复保留了合法第一步，拒绝原错误后缀/注释，不改常量、不补造绑定、不执行任何操作。此诊断不进入当前冻结链或重算本批成功数。

身份追问与指标分析的首次修订又暴露了同一类任务丢失：原草稿能提出追问/计算 1%→6%，引用式替换却把完整回答压成观察摘录。**来源复制准确，但任务推理/开放职责丢失**；不能把这叫语义准确改善。修复方向是把证据摘录与任务答复分成两个输出通道，不能让引用操作无条件覆盖答复；开放待办/追问必须带原始出处保留，不因没有形成 L0 字段就消失。事实矛盾仍需真正改正或明确保持不接受，不能只在旁边贴一份正确材料。

营销任务的初审在调用前超过保守上下文代理（46,957 > 40,960 bytes），不是 9B 已审查后通过/失败。当前离线诊断用可精确反解的重复文本池和同构行表去掉传输重复，未删源文/观察/任务/判据、未增上下文限额；代理降至 40,915，尚待新版本真实模型验证。首两次不够省空间的诊断保留。此改进针对序列化，不证明模型读懂了压缩关系。

旧 day11 证据审计也发现收集器误读同目录的大 `inputs.json`（约 1.06 MB），在找到实际归档 manifest 前触发通用 JSON 预算。失败已保存；待本批结束后修复 manifest 选择，不绕过冻结指纹或改写历史证据。

## English

Current September 14: the [duty-contract/artifact experiment](SEMANTIC-DUTY-CONTRACT.md) is complete. Task-quote fields can receive observations, duty descriptions strengthen constraints, note checks can target body text and contradict themselves, and one note failure blocks the original serial edit graph. Explicit independent continuation and narrow typed rendering improve particular candidates, not translator/reviewer reliability. Approval-pending statements below are historical.

September 14 follow-up: the [candidate-blind task-witness diagnostic](SEMANTIC-WITNESS-DIAGNOSTIC.md) retains four known cases. Correct source extraction does not prevent later index/data scope, execution/deliverable and predicate confusion. Schema fields alone are not semantic proof. The experiment stays isolated; existing code now checks exact task-delivery witnesses and prevents note findings from targeting body edits. Note repair and suitable artifact verification remain pending; no immediate scaled cohort or repeated prompt-only trial is justified.

### September 14: model contrast and concrete segmentation repair

The [four-case 9B/27B contrast](SEMANTIC-REVIEW-MODEL-CONTRAST.md) is sealed. 27B newly detects missing Phoenix suggestions, but still misses query implementation and not-approved/not-reviewed expansion; mesh omission findings coexist with stale absence judgments. This does not support blaming only 9B size or declaring all larger models ineffective. Two responses pass binding and two fail citations; five attempts retain one timeout with unknown usage. No drafts, Runtime operations or old grades changed.

Only after the run ended, bounded-draft-review/v8 fixed a concrete boundary defect: `_blocks` no longer separates ordered-list/numbered-heading prefixes from same-line bodies. Text, offsets, negation and authority remain; this is not semantic proof or a complete Markdown repair. Seventy-six targeted checks pass. Frozen payloads/results remain untouched. Task fulfillment, observation-set scope and applicable artifact validation still need implementation and new-sample acceptance. Entries below are historical states.

### Pre-switch wire-input audit (2026-09-11, zero additional calls)

The retained v10 wire messages, not merely upstream requests, contain the necessary evidence. IRQL source `s012` includes the half-open interval and check `c007` contains the complete query; the model grounds that code using the unrelated fact that no query was executed. Coverage `c023` points to the prose window declaration, not its implementation. Host-owned IDs do not establish semantic alignment. Mesh editor context contains both full measurement windows, the comparison region and explicit metric/calculation findings; its output still claims data are missing and changes only the index path. These cases are not explained by missing delivery of those inputs. Context length, stale-candidate interference, interface representation and model capacity remain possible influences. The Chinese section records exact artifact hashes. No model call, permission expansion, old-result regrading or new-source credit occurred; the local 27B comparison still awaits approval.

Known v9/v10 finished without satisfying the stage. V9 costs 15 calls, 83,600 input / 12,172 output tokens; v10 costs five calls, 37,732 / 9,285. Their diagnostic digests appear above, and content-review/report.json binds actually inspected candidates. V9 restores the CAPA owner and Phoenix token accounting, but loses useful follow-up content and regresses mesh evidence use; IRQL regains code form, not a correct query. Duplicate/missing review IDs block two tasks.

V10 gives each check a host-owned object key, sharing cell schemas through local $defs/$ref. Both initial reviews avoid the old identity conflict. The initial expanded-schema zero-call budget failure remains. Nevertheless both IRQL reviews endorse the wrong query; the mesh editor changes a path without adding required calculations, and final review blocks on a positive judgment without citations. Interface validity is not semantic success.

Current engineering QA: 77 targeted tests, 2,891 full tests plus 81 subtests, lint on 51 changed Python files, documentation links and diff checks pass. The stage stays open, with no push or scaled A/B. A four-known-case local qwen3.6:27b review-only comparison has been proposed for user approval while keeping 9B generation. It has not run; no stronger-model success, lowered criterion or regraded old outcome is claimed.

Frozen v2 failed: six Skills/twelve tasks, zero fulfilled substantive tasks, three correct boundaries, eight partial and one failure; 67 calls, 342,581 input / 52,968 output tokens. No observed unauthorized call does not establish correct prose. Positive review opinions missed several defects.

New mechanism corrections address concrete host-side faults: preserve the full entry before budget-filling references; supply a host read-status index with exact tool/argument/payload pointers; retain the previous artifact as an unverified candidate during post-read updates; route unlocated negatives only to a sole whole-draft owner; and withhold a locationless positive coverage row without discarding other findings. Permissions, freshness gates, source immutability and other invalid-ID/schema rejection remain unchanged. These are not a broader semantic schema or a stronger-model substitution.

Known v9 is a separate diagnostic, never v2 regrading or unseen credit. Its first CAPA edit restores Omar while preserving existing dates/negation, but an unsupported historical note remains. Open causal/technical/procedural judgments still need actual content review; engineering test counts cannot replace it.

Latest mechanism review (2026-09-11): known v4/v5/v6 cost 3/18/6 real 9B calls respectively, with 18,331/102,946/35,583 input and 4,742/9,168/2,928 output tokens. V6 diagnosis digest: `ec4e6665c9a8f4cce4b86ad5e87be4b096e34b7ffb1042bbec6a5ec9591f86a7`. Scope guards reject foreign sections/repeated headings and quote locks retain API text, yet documentation reintroduces unsupported MIT licensing; v5 loses an observed handoff URL. These remain failures, regardless of positive AI review.

The repeated failure changes repair scheduling, not another same-prompt retry: all-section regeneration ran even without a located negative. The default now edits only a negative finding within one host-owned section; absent location retains the unverified draft and explicitly open findings. Lexical difference is not an edit command, positive opinion is not correctness, and no-op is not repair credit. Editors see only the current body plus other-section labels, retaining original task/Skill/observations; the full draft remains frozen and finally reviewed. Exact quotes are protected bytes, not proven relevance. Known regression and a subsequent frozen transfer require separate accounting; stage exit is still pending.

V7 costs two final reviews, 12,313 input / 4,349 output tokens and zero editor calls. Its original drafts remain intact, but delivery metadata/freeze declarations changed while the driver still had earlier modules loaded. Preserve it but exclude it from fixed-implementation certification. V8 uses a new process with before/after full fingerprints; this is an execution-discipline correction, not semantic uplift or hidden retry cost.

Known v3 is sealed: two fulfilled scoped tasks, one partial and one failed, plus a separately failed network-review probe; 17 new 9B calls, 110,053 input / 21,130 output tokens, excluding reused v2 initial calls. No-read retention preserves network interpretation and writing conditions; marketing repair deletes both required campaigns. Diagnosis digest: `0af08df95ead7f00b77e10aa47900655f9603f57f5a9d156eda6f03104b09b69`. This is not transfer acceptance.

V4 separates statement judgments, observation coverage and task-level checks, retaining raw role-review.json and validating group/ID/location before deterministic normalization. Evidence/reasons precede decisions. Task checks no longer demand self-declared compliance, but the false implicit causal premise remains missed. Structural protection now includes ATX, Setext and standalone non-sentence strong labels; it rejects whole-document insertion into the preamble before damage, without claiming useful repair. V5 tests separate immutable surroundings and the exact editable target. The full draft reconstructs as before+target+after. Historical handoff/README checks import verified edit-snapshot data only, never old judgments, new source credit, fresh business reads or authority. Fresh-source selection still requires a separate implementation freeze.

Known-development v2 is sealed after the [failed v1 transfer](SEMANTIC-CLOSURE-TRANSFER-V1.md): two scoped tasks fulfilled, four partial, 36 real 9B calls, 218,870 input / 20,228 output tokens. It supplies no unseen-source credit. Answer/support separation preserves metric calculations and retry-aware interpretation, but empty read requests break clarification; unnecessary no-read rewriting introduces false causal premises and loses conditions. Editor inputs still exceed budget. Evidence digest: `c2163c7d2ec7caaa68427006750700bebc707e80f4e75d32983df4f3d1ade3c7`. The 2,827 tests plus 81 subtests are v2 engineering results only.

V3 tests host-controlled conditional model invocation: retain the exact prior candidate when no new observation exists, with a separate unverified selection notice and no invented model receipt. Decision-specific generation schemas leave original Runtime validation and independent admission intact. Readable worksheets retain all original prose while removing duplicated navigation metadata; the reversible pool remains historical, not the active model representation. Budgets stay unchanged. The first actual v3 review still misses the false causal premise inside a network uncertainty question/notes. Located sentences are not semantic atoms, and positive model opinions cannot serve as acceptance. The failed review remains separately recorded while the preventive no-read retention path is tested. Earlier diagnoses below remain historical evidence, not retroactive successes.

Frozen transfer v1 exposes additional failures. Three author responses fail before execution: two confuse Skill/reference paths with business export resources and cite headings that do not contain the proposed literals; one cites an unseen page in non-authoritative boundary metadata. A valid initial read is discarded with the rest. Offline diagnostics preserve only the already-valid prefix, quarantine invalid annotations and reject the entire invalid suffix without inventing bindings, running tools or regrading the frozen batch.

The first identity/metric revisions retain exact observations but lose useful clarification and derived rates from the answer. Lossless copying is not task preservation. Evidence excerpts and the requested answer need distinct channels; open questions/duties must retain provenance. Existing contradictions cannot be excused by attaching a correct source. Marketing review is blocked before a model call by the unchanged conservative byte proxy (46,957 > 40,960). A reversible text-pool/table diagnostic reduces it to 40,915 without dropping text or increasing limits, but has not yet been model-validated. The first two insufficient packing attempts remain recorded. A separate historical-audit reader failure on a 1.06 MB neighboring input file awaits bounded manifest-discovery repair after this batch; no frozen hash check is bypassed.

Additional mechanism findings: paragraph buckets split section headings from their bodies, causing duplicated/misplaced text; the current editor uses complete sections and preserves heading sequences. Canonical graph key sorting also ordered grammar payload fields before the operation selector. The transport now restores the declared required-field order without changing schema meaning or hash canonicalization. The first ordered probe confirms that order works but still confuses prose with source IDs, so it is not a complete causal explanation. A typed, tagged payload now separates enumerated observation IDs from free prose. Only the selected branch is materialized; inactive payload remains audited, never executed or displayed as fact. This does not broaden the Runtime schema profile or prove semantic selection correct.

Update, 2026-09-11: the earlier whole-draft mechanism did not stabilize. Later handoff drafts again omit ownership, while README withholding removes both fabricated licensing and useful installation advice. Full isolated-cell passes retain no-ops, duplication, excessive ranges and explanation-budget failures. A lossless sentence-to-line navigation index retains semicolon-linked relations independently of reviewer opinions, but navigation alone still does not fix generated omissions.

The new source-reference repair option lets the model select observation slice IDs; the host validates provenance, exact offsets and citations, then renders the exact source into the actual deliverable. Free prose remains possible for explanations and unknowns. Selection/entailment remain unverified; exact copying never authorizes actions. One repair pass now contains at most eight one-range editing calls, plus initial and final reviews. Each editing call retains 2,048 output tokens, but the total call budget has explicitly increased. This is a costed mechanism change, not a same-budget speed improvement. No-op proposals receive no repair credit. All old alternatives and failures remain evidence; new-sample acceptance is still pending. The following paragraphs describe historical attempts, not the latest architecture.

Two known-development `located-slots` attempts failed to repair the handoff. Repetition of the same review/revision chain is paused under the stage exit rule. Exact evidence shows topic overlap mistaken for detail preservation, false-positive reviewer opinions carried into revision, unsafe paragraph splitting across Markdown fences, and README guesses that cannot be repaired without actual further observations.

The next diagnostic separates source-to-draft repair locations from positive AI verdicts, preserves complete editing blocks, and uses independently admitted bounded read continuations through the original strict engine. Lexical differences are inspection leads, not proven semantic loss. Historical snapshots never become refreshed action authority. The 9B model, retained failures, fixed budgets and formal generalization boundaries remain unchanged. This is not stage completion or new-sample accuracy evidence.

The first focused diagnostic (one 9B call, 8,816 input / 789 output tokens, 59.71 seconds) restored both incident/investigation owners, the start time and the unacknowledged handoff. This is a known-case localization improvement only. README continuation v1 repeated a directory read because history omitted its exact tool/arguments; v2 restored contract-bound request/result association and read the config, but stopped prematurely with a guessed API. Its graph completion is not task success. Both failures remain recorded.

Subsequent integrated handoff v4 restores the known details in the three-step graph while open duties remain partial. README continuation v3 actually reads configuration and source (six model calls/two primitive reads) and uses the real function, but invents MIT licensing. Review quote copying, wrong claim-to-paragraph association, disclaimer contamination, cross-slot duplication and “removed” claims without actual removal are separately recorded failures.

The current editor first declares block evidence status, then proposes one coherent snapshot. The host computes exact differences under the unchanged eight-change/one-revision/2,048-output bounds. A uniquely located unchanged block marked unsupported is withheld with an explicit notice; ambiguity and invalid structure fail closed. This is a conservative projection, not a factual verdict. The model also over-withholds valid installation advice, so that loss must be reported. Raw notes remain in audit rather than becoming host facts. Final review is still an AI opinion, never an approval. Frozen new-sample acceptance remains pending; include every failure and review call in cost accounting.
