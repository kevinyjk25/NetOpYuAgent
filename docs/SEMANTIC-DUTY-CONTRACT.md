# 职责合同与本地工件检查 / Duty Contracts and Local Artifact Checks

## 中文

### 当前增量（2026-09-15）：先定位证据，备注不再自由扩写

**限定机制验证完成，语义闭环阶段仍未通过。** 本轮只验证两个已知任务的备注，不重跑正文/查询修订，不扩样、不切模型。新增显式入口：

```sh
.venv/bin/python -m evaluation.semantic_typed_duty_probe MANIFEST.json NEW_OUTPUT --evidence-first-notes
# 默认零调用预检；真正调用 9B 须使用新目录并加 --run。
```

原清单格式不变。三个阶段是：仅看备注提取谓词 → 在宿主目录中选择证据 ID → 对每个“谓词／所选片段”分别判断。随后每条备注只允许 `keep` 或 `quote_observations`；没有自由文本 replacement、删除、正文编辑或新职责字段。宿主从原观察中取出原文，生成带“未验证观察摘录，不是已验证结论”标签的备注，保留原备注和宿主未完成职责。

- **引用由宿主生成**：目录只含 Skill 指导和原始观察，明确区分类型；任务、调用方参数不能成为可选事实证据。备注渲染进一步只接受观察 ID，不接受指导文档作为实例事实。
- **先定位再比较**：比较节点必须处理已选的每个证据 ID，不能漏项、换引用或再重抄一遍可能出错的 quote。没有所选证据的谓词直接保留 unknown，不能由模型补一个空引用的肯定结论。
- **不以省上下文为由丢原文**：定位视图只展示一次完整来源，窗口间空白也可逐字重建。比较节点拿到所选片段及其完整父来源，但不能由此宣称搜索穷尽；未选证据可能改变结论。
- **限制新增断言，不声称完成推理**：摘录中的字符来自观察；但引用可能不相关或不完整。原备注中的问题不会因换成摘录而自动被解决，`sourceProjectionIsNotReasoningSuccess=true`。

同一驱动新增模式，而不是新增一套 Harness。现有完整诊断模式和默认 DSH 路由保持不变。新模式上限为 3 次提取/定位/比较＋最多 8 次备注选择，最多 11 次/任务，仍在原 17 次授权内；本轮每例有两条备注，实际各 5 次。沿用 Runtime reason 节点、单次回执、源码冻结及失败不重试规则；源脚本、新业务读、Effect 均为 0。未重新执行已有失败的查询编辑。

### 本轮实测：引用问题收敛，命题蕴含仍有失败

| 指标 | 本轮结果 | 正确解释 |
|---|---:|---|
| 实际模型调用 | 10/10 节点绑定成功 | 仅说明有界协议完成，不是准确率 |
| 谓词／片段比较 | 7 个谓词、14 项比较，引用均能重建 | 不是 14 个语义判断正确 |
| 备注输出 | 4 条观察摘录、0 条模型自由扩写 | 引用选择仍需评估，不能计作 4 个业务任务完成 |
| 正文与映射参数 | 两例均不变 | 查询原有缺陷未修复，也未验证查询执行 |
| 模型意见 | CAPA 四项 supported；IRQL 三项 unknown | 意见不是实际事实或审批 |

**明确保留的反例：** CAPA 的目标谓词是“Pia 尚未审阅”，所选观察是“Pia 尚未批准”；9B 仍判 `direct_support`。引用本身正确，谓词不同，不能得出该肯定结论。此例已经把故障定位到比较判断，而不是证据缺失、引用抄错或工具输出未接入。监测周期“5/30 天已观察→周期尚未完成”还涉及推导与直接陈述的标签区分，不能把所有 supported 当作严格直接证明。

最终 CAPA 摘录只包含原观察中的“未批准”，不再新增“未审阅”；这是受限输出形式控制了断言，不证明比较器理解正确。IRQL 摘录列出实际 enricher 字段、目标窗口和“未运行查询”，没有新编部署假设或宿主职责；但这些摘录**没有回答数据保留策略是否兼容**。所有原备注及未知职责仍在审计里；完整任务通过数不在本轮评估范围。

总计 **26,053 输入 / 1,543 输出 token**，用量未知 0；请求 p50/p95 **11.19 / 25.17 秒**。由于这次只跑备注流程，不能与上一轮整稿诊断直接计算速度提升，更不是 Runtime SLO。预检曾用每备注 8 个占位谓词检查预算：IRQL 的该占位组合超出原字节预算，未提高上限；实际 9B 只提取 3 个谓词，真实请求在上限内。将来真实输入超限仍停止，不截断或保证任意 Schema 最大值均能同时容纳。

证据位置：

- 原始运行：`artifacts/translator-v2/evidence-first-notes-20260915-run/`；总摘要 `4d89238dcef5a7fbb7644cb445344a3fbd17ba43a3aacfca823dabee466f22f3`。`before/catalog.json` 是引用目录，`locate/locate/model/locate/` 是选择结果，`compare/compare/model/compare/` 是逐项判断，`project/<备注ID>/model/<备注ID>/` 是只引用的修订选择。
- 严格零调用重建：`artifacts/translator-v2/evidence-first-notes-20260915-audit/`；摘要 `8dd2434645b1f30a577124e8f25393ce4392250b41a4e71a5ca1b8c7eecdff73`。50 个回执绑定文件和源码快照验证通过；所有原报告自摘要有效，**不使用历史摘要重建例外**。
- 披露的开发者 AI 内容审阅：`artifacts/translator-v2/evidence-first-notes-20260915-content-review/`；摘要 `21cd0527ddb3e96dd91842bc07dedbe76f8860a19e009493b05ce2c85eadc4ba`。明确保存上述否定谓词反例，不是独立人工 Gold、自动语义 Oracle 或总体准确率。

下一步应针对**来源命题和候选命题的独立表示／对齐**，并继续区分可确定的结构与开放语义；不能再仅增加“是否支持”的自审次数。源／目标的主体、动作、否定、时间和对象需要可定位的差异，无法证明等价时保留 L1 未验证状态，而不是降低门禁或把摘录数量当作业务成效。当前不启动新来源验收或扩大 Runtime A/B。

本轮 QA：新增 22 项回归，57 项相关接口检查、92 项定向、**3,083 项全量＋81 子测试**通过（全量 219.44 秒）；变更/新增项目 Python Ruff、文档中英顺序/链接 3 项、diff 检查通过。测试涵盖定位目录类型、所有原文字符保留、无引用不准肯定、遗漏/替换比较项拒绝、来源漂移、相冲证据、备注无自由文本通道、指导不可冒充观察、长度超限拒绝及依赖失败不重试。QA 在真实调用完成后运行；制品在 `artifacts/translator-v2/evidence-first-notes-20260915-qa/`。工程通过不抵消上述语义反例。

### 2026-09-14：宿主职责原文与分步谓词审阅（历史）

2026-09-14 同日后续。按“继续”授权，不更换 9B、不扩样、不提高单任务 17 调用上限。**阶段仍未通过；本次也没有两个任务的完整首次成功。** 下面先说明当前实现，之后保留上一轮的原始结果。

| 现在的机制 | 解决的具体问题 | 尚不能证明 |
|---|---|---|
| 宿主直接生成 `task:NNN` 锚点：原始文字、偏移、角色、来源类型；按调用方角色分组 | 不再由 LLM 重抄任务或改写“最多／不执行”；观察不能替代宿主任务合同，省去一次合同生成调用 | 角色可能分错，模型仍可能误解；无损保留文字不等于完整理解。无 `taskScope` 时保留整个任务为未分类，不猜测角色 |
| 仅看备注的谓词提取 → 冻结每个谓词 ID → 对照完整来源 | 拆分和真假判断分开；不把正文误抄进备注作为合法锚点 | 提取是否完整、解释是否忠实、证据是否蕴含仍是模型判断 |
| 宿主根据逐项证据关系汇总，不接受模型的整体 supported 覆盖未知项 | 强肯定缺精确证据、解释不确定、仅推导或其他谓词证据均保留 unknown | 引用存在也不证明推理正确；全 supported 也不批准任务 |
| 独立编辑单元使用独立单节点 Runtime 图 | 一个备注失败不阻断没有依赖关系的另一个编辑；失败节点不重试、不使用其输出 | 不是通用生产 DAG 恢复；组合后的候选仍需结构与语义检查 |
| 有限类型化字面量渲染在新诊断入口中显式接线 | 符合已有白名单的零模型步骤扣除共享变更预算，之后才终审 | 不是完整查询验证；旧入口、默认 DSH 路由不变 |

当前上限为 **16 次/任务，低于原授权 17 次**：最多 3 个任务分组检查＋备注前后各 2 次＋最多 8 次共享正文/备注编辑＋1 次最终任务检查。多图组合不提高单图上限；所有节点仍没有业务工具、脚本执行或 Effect 权限。未知项触发的是检查，不是自动删除或批准。

实现入口：[宿主职责](../evaluation/hybrid_typed_duties.py)、[分步谓词审阅](../evaluation/hybrid_predicate_review.py)、[诊断驱动](../evaluation/semantic_typed_duty_probe.py)。原来的 LLM 合同和接续入口保留为历史机制，不能将其结果算成新流程成功。

```sh
.venv/bin/python -m evaluation.semantic_typed_duty_probe MANIFEST.json NEW_OUTPUT
# 默认零调用预检；限定真实验证使用另一个新目录并加 --run。
```

清单格式与下文相同。`before/contract.json` 可查看完整任务锚点；`notes-before/extract/model/extract/` 查看原句→谓词；`notes-before/evidence/model/evidence/` 查看逐项证据意见和宿主汇总；`edit/<单元>/model/<单元>/` 查看真实编辑；`notes-after/` 使用新候选重新提取审阅。失败未生成 `materialized/candidate.json` 时，不存在可宣称已修好的最终稿。

### 本次真实结果与修复边界

- **CAPA**：成功拆出“未审阅”和“未批准”，但第一次证据审阅把四个谓词均判 direct_support 且引用为空，宿主全部降为 unknown。第一个备注编辑误交整篇正文，被拒绝；第二个独立备注完成，把无依据的“未审阅”删去，保留“未批准”。原正文和首个备注不变。后审仍缺引用；最终任务审阅也有位置错误和“最多三节→恰好三节”的解释偏移。**局部编辑有效，不能证明审阅器准确或整任务通过。**
- **IRQL**：本地检查再次检出区间错误，两个备注产生了过度扩展的解释；正文编辑虽改了操作符，却把关闭代码围栏的行包含在替换范围里而未补回。组装拒绝，**没有发布修订候选，不能把未应用的查询或备注算成修复**。该任务仅运行到第 8 次调用；后续检查未运行。
- **结构修复已补入代码，但未重跑模型**：`validate_cell_proposal` 现在在单元绑定前验证与原父稿拼接后的围栏，并检查所有者原文；坏编辑在独立单元处被拒绝，不留到最终拼装才阻断其他候选。这是安全隔离改进，不是“模型已经会正确生成围栏”。

本次 **18 次真实调用**：IRQL 8 次、CAPA 10 次；95,154 输入 / 6,240 输出 token，用量未知 0。请求 p50/p95 **16.80 / 54.45 秒**。调用组成与上轮不同，不可由此宣称吞吐提升、因果 A/B 改善或 Runtime SLO。17 个回复通过节点绑定、1 个回复被拒绝；其中已绑定的查询仍在后续组装失败，**17/18 不是语义准确率**。两个任务均未达到全节点无错误且内容可接受的阶段出口。

### 原始汇总故障与可审计取证

原运行在最后证据汇总时发现检查点摘要冲突：把子报告摘要也命名为 `reportDigest`，再自封装时覆盖了该字段。已改为独立的 `caseReportDigest`。**原错误检查点不修改，原运行没有最终总摘要**；不能声称所有原报告校验通过。

取证器默认仍严格拒绝任意摘要错误。显式诊断选项只接受这一种已知计算：有效子报告必须能重现原检查点的错误 hash，状态和失败计数还必须吻合；随后在新审计制品中保存原摘要、原因和重建视图，明确 `allOriginalReportSealsValid=false`。错误、未知和旧成绩不改写，不把一般篡改当兼容格式忽略。

- 原始运行：`artifacts/translator-v2/semantic-typed-duty-20260914-run/`，保留 18 次请求/回复及 90 个回执绑定文件、一份执行源码快照。
- 新的零调用审计：`artifacts/translator-v2/semantic-typed-duty-20260914-audit/`。报告摘要 `284cb4d07a4ae00f8c42031b4e2505292c0ea99b9d3db738d1eb06d520cf9292`，证据摘要 `ef61e23527774b6dbe8c8dd17c7f0907766a3bf99cfec13b4e9eca5fdeae23cd`。重建实际候选或复现其组装失败，不采用期待答案，也不重放模型。
- 内容结论来自披露的开发者 AI 对已知样例的审阅，不是独立人工 Gold；旧运行、基线、分数和源码快照保留，不提交/推送。

最终 QA：**158 项定向、3,061 项全量＋81 子测试**通过（全量 234.53 秒）；相对上一轮增加 48 项回归。变更/新增项目 Python Ruff、文档中英顺序与链接 3 项、`git diff --check` 通过。包含错误围栏提前拒绝、合法代码编辑、独立失败不重试、原始摘要严格拒绝、精确碰撞取证及任意篡改拒绝。制品：`artifacts/translator-v2/semantic-typed-duty-20260914-qa/`。这些结果验证工程合同，不证明 9B 的语义准确性；围栏/摘要修复后的源码没有新的模型实测。

下一步的机制重点是：**证据先定位再判断，限制备注修订扩张新断言，宿主保护确定的工件边界**。不能仅靠增加审阅次数解决证据误判，也不能用更严格的拒绝率冒充更好的任务完成率。新的类型化职责解决了“引用来自哪里”，没有解决所有“这句话意味着什么”。下面保留上一轮过程和成本，不合并计算准确率。

### 上一轮：LLM 职责合同与独立接续（历史）

2026-09-14。用户已确认[上一轮复盘方案](SEMANTIC-CONTEXT-REPAIR.md)。本轮实现“候选无关的职责提取 → 本地工件检查 → 逐职责/独立备注审阅 → 受控编辑 → 重新检查”，先验证查询与否定关系两个已知失败任务，不扩样、不修改既定出口。

### 机制与边界

| 环节 | 谁负责 | 能证明什么，不能证明什么 |
|---|---|---|
| 职责候选 | 9B 从完整原任务、来源、角色片段中提取，最多 6 项 | 宿主验证逐字任务锚点；不能证明职责解释准确或覆盖完整。超量必须披露，不能自动变成权限 |
| 本地工件检查 | 白名单纯本地解析/计算函数 | 只证明具名检查子集；未知语言、完整业务语义和依赖仍为 `unverified` |
| 聚焦审阅 | 每职责一调用，最多 6 次；第 7 次专门审阅所有备注 | 正文肯定意见不跳过备注；备注拆分独立断言。引用和位置可验证，蕴含关系仍是模型意见 |
| 修订 | Runtime 中独立正文/备注 reason 节点，总计最多 8 次 | 宿主限定所有者、行范围、引用和标题；保留原备注与未完成职责。不能写业务系统或自签审批 |
| 终审 | 最多 1 次，接收新候选、原任务与完整来源，不接收旧肯定意见 | 重跑本地检查并重新审阅；图完成、全 `met` 均不等于语义成功或准入 |

总预算最多 **17 次/任务**。它由多个已有 Runtime 有界图组成，**没有提高单图最多 8 次调用的限制**。任何调用失败不自动重试，输入超过原预算不截断。超过编辑预算的单元明确延期，原始错误不会因此变成通过。流程仍是可选研究入口，不修改默认 DSH 路由或生产权限设计。

本地检查器当前支持：Markdown 围栏闭合；JSON 严格解析（拒绝重复键、NaN 等）；Python `ast.parse`（不导入或执行）；显式十进制比例等式；KQL `between` 的有限形状检查，以及单一来源右端开放时间窗口下两个字面比较谓词的有限检查。它**不是完整 Kusto 解析器**，不证明字段选择、函数签名、后续管道语义或查询能在设备上运行。KQL 区间语义依据 [Microsoft 文档](https://learn.microsoft.com/en-us/kusto/query/between-operator?view=microsoft-fabric)。未调用数据库或执行 Skill 自带脚本。

### 如何查看和复现

- [职责合同、逐职责核查、备注断言绑定](../evaluation/hybrid_duty_contract.py)
- [白名单工件检查器](../evaluation/hybrid_artifact_checks.py)
- [真实 Runtime 驱动](../evaluation/semantic_duty_probe.py)

从项目根目录运行：

```sh
.venv/bin/python -m evaluation.semantic_duty_probe MANIFEST.json NEW_OUTPUT
# 检查输入后，明确使用新的输出目录与预算授权才加 --run。
```

清单最多两项，每项仅接受 `source`（已完成历史执行）、`request`（原始候选审阅请求）、`taskScope`（无损角色片段）。原回执、任务、实际观察和候选必须匹配；不允许期待答案进入清单。默认零调用预检。不得覆盖已有目录或重置失败检查点。

输出中，`plan/model/plan/bound.json` 展示职责与任务引用；`focused/model/` 展示每项对照和备注拆分；`before/local-checks.json` 给出检查范围与代码位置；`editing-plan/` 列出编辑/延期单元；`materialized/` 保存修订稿及前后摘要；`final/` 保存终审；`summary/` 汇总实际调用和边界。每个模型节点都有原始请求、回复、用量及回执，源码快照在 `freeze/`。制品被 Git 忽略，需要单独备份。

### 实际结果：有具体修复，但阶段未完成

固定 `qwen3.5:9b`，仍是两个历史失败任务，不是新 Skill、独立 Gold、生产成功概率或大样本泛化。**首轮完整流程完成 0/2；后续独立接续和本地渲染形成两项窄范围改善，不改写首次失败，也不等于两项任务完整通过。**

| 用例 | 首次运行 | 后续真实变化 | 仍未解决 |
|---|---|---|---|
| CAPA 否定关系 | 职责提取将观察句填进任务引用字段，来源校验拒绝，合同整体隔离 | 在不依赖该合同的备注支路，9B 去掉了“未审阅”，保留“未批准”；正文、负责人、时间与状态原样保留 | 业务职责合同仍失败，未补成或部分晋升；审阅器曾给复合句 supported，同时又承认其中“未审阅”无依据 |
| IRQL 查询 | 聚焦检查定位区间错误；备注编辑越界，编辑图停止，正文未执行 | 接续未尝试的正文节点，9B 改为 `>=` / `<`；另一个零调用本地渲染步骤将两个 `datetime'…'` 改为合法字面量形式，有限区间检查通过 | 9B 自身仍生成错误字面量；失败备注保留，额外章节和引用解释风险仍在；未验证完整 Kusto 语义或执行结果 |

局部查询变化是：

```kql
| where EnvTime >= datetime(2026-09-10T09:00:00Z) and EnvTime < datetime(2026-09-10T10:00:00Z)
```

此处**不声称执行了查询或整个查询正确**。本地渲染只接受受限 where 比较式、源中已有且可 round-trip 的 UTC 时间值，保留列、操作符、时间值、其他正文、notes 和 values；忽略未知表达式、数据字符串、其他语言和逐字源代码。变更上限仍从共享 8 次预算扣除，超过则原子拒绝。原模型稿、本地修复稿和两个精确补丁分别保存；后者没有新的 LLM 终审。规则依据 [Kusto datetime 字面量文档](https://learn.microsoft.com/en-us/kusto/query/scalar-data-types/datetime?view=microsoft-fabric)，不是为这些时间值硬编码答案。代码：[类型化渲染](../evaluation/hybrid_artifact_lowering.py)。该步骤是显式可选 API，本轮由离线审计调用，**尚未默认接入原始 probe 或 DSH**。

### 独立接续，不重试失败节点

新增[接续入口](../evaluation/semantic_duty_resume.py)：验证已停止运行的回执和源码归档，在新图、新目录中接续未尝试的独立工作。失败的职责合同不供给业务检查；原始来源仍可供独立备注检查。已失败的备注不重试、不清除，成功但尚未应用的候选经原绑定重新验证后可作为候选保留。新输入不从失败节点取结果。

每项任务把父运行和接续调用合计限制在 17 次以内；同一父运行创建一次性旁路认领记录，不能换输出目录重置预算。原目录不变，失败和未决项明确展示。该入口是研究诊断的显式补救，**不是已完成的通用持久恢复或生产调度机制**；原主 probe 的串联编辑图仍会按原合同停止，不能把接续结果说成其首次成功。

### 成本、证据和验收

总计 **17 次真实调用**：查询 12 次、CAPA 5 次；126,955 输入 / 10,192 输出 token，用量未知 0 次。两次有明确版本的诊断合计请求 p50/p95 **31.32 / 77.44 秒**，不是任务时延、Runtime SLO 或因果 A/B 性能改善。源脚本、业务新读取、Effect 均为 0；类型化渲染没有新增模型调用。

- 首轮：`artifacts/translator-v2/semantic-duty-20260914-run/`，摘要 `dc5b7aea9d2bfb10d1c40c066047fda1e595e56288836161ac53573e800949c4`；11 调用，92,213 / 6,711 token。
- 接续：`artifacts/translator-v2/semantic-duty-20260914-resume/`，摘要 `dcb77ee6faac8bbe4079cc58f1aa21929577d3c53cc43f6fe0366015da6cc190`；6 新调用，34,742 / 3,481 token。
- 内容审阅、原候选重建、类型化补丁：`artifacts/translator-v2/semantic-duty-20260914-audit/`，摘要 `f5901b95bc8b0b5200adbb72572f8ee9d35b4e6a07673eaa2c6bf9a387522bed`。这是披露的开发者 AI 已知失败审阅，不是独立人工或可复用的字符串语义 Oracle。
- 所有运行保留模型原始请求/回复、实际用量、回执、源码快照。未修改旧分数、A/B 基线、默认模型或 DSH 路由；不提交或推送。

最终 QA：62 项新增、224 项相关定向、**3,013 项全量＋81 子测试**通过；最终全量 222.60 秒，变更/新增项目 Python 的 Ruff、文档中英顺序/链接 3 项、diff 检查通过。QA 在模型验证完成后执行。初次测试暴露并修正了图时限超限、测试桩传输封装和目录重复创建错误，都在对应真实调用前修复，没有放宽运行限制。真实首轮还揭示本地形状检查需要区分“不支持”和“错误”，已增加合法多行/额外条件、字符串/注释、错误日期及受限渲染的反例测试。最后补入父输入必须匹配实际模型请求的接续校验，两例原输入均零调用核对通过；随后全量复核，没有重跑 17 个模型请求或改写旧摘要。QA 制品在 `artifacts/translator-v2/semantic-duty-20260914-qa/`。

### 下一步：不能用局部修复掩盖转译问题

1. **职责来源类型收口**：以宿主拥有的任务/观察锚点引用代替让 LLM 重抄长句；观察事实可以支持任务，但不能冒充用户职责。第一份职责候选还把“不执行”改述成“只输出查询”、把“最多三节”改述成“恰好三节”，这种解释偏移不能靠位置正确掩盖。
2. **审阅拆分仍需改良**：本轮 notes 在部分检查中仍混入正文、复合断言仍有自相矛盾判定；错误候选能被编辑器修好不代表审阅器会稳定检出。未知或错误定位不能给语义通过率充数。
3. **把适用的确定工件交给类型化生成与检查**，开放语义仍保留 LLM 和未验证状态；不能通过无限补关键词/提示词或擅自把自由文本全部宣称为确定 L0 来“通过”。

在这些问题收敛前，[固定阶段出口](SEMANTIC-CLOSURE-EXIT.md)未通过，不启动新来源 6 Skill/12 任务验收或更大 DSH A/B。该轮结束于**具体机制修复与结果交接点，不是第一阶段完成**。

## English

### Current increment (September 15): evidence first, no free-text note expansion

The bounded note mechanism check is complete; the semantic stage remains incomplete. The explicit `--evidence-first-notes` mode above uses the same two known tasks, 9B, Runtime reason nodes, immutable receipts and no failed-call retries. It does not rerun query/body editing, expand the cohort, change default DSH routing, run source scripts, issue fresh business reads or grant Effects.

The model first extracts note propositions without seeing sources, then selects host evidence IDs from complete source text, then compares each selected proposition/excerpt pair. Task/caller text is not selectable fact evidence. Guidance and observations are typed separately. Missing selection remains unknown; comparison cannot replace or omit selected IDs. A reversible presentation shows each original source character once, including whitespace; comparison receives selected excerpts and complete parents, without claiming exhaustive selection.

Note editing permits only keep or exact observation quotations, not new prose, deletion, body edits or duties. The host labels rendered notes as unverified source excerpts, retaining original notes and unresolved host duties. Guidance cannot be promoted into an observed-instance quotation. Exact transcription proves neither relevance nor semantic coverage. The cap is three evidence phases plus eight note selections (11 calls, within the prior 17-call authorization); these two-note cases each used five calls.

**Results:** all 10 model nodes bind; seven predicates have 14 exactly bound comparison pairs; four source-excerpt notes contain no freely authored replacements. Both bodies and parameter maps remain unchanged. These are protocol/projection results, not semantic accuracy or whole-task success.

The key counterexample remains: CAPA's claim says not reviewed, while its selected observation says not approved. 9B labels that pair directly supported. Retrieval and binding work, but the semantic comparison is wrong. The final constrained quotation retains only the observed nonapproval: this controls generated assertions without proving the comparator understood them. IRQL's three opinions remain unknown. Its new excerpts show actual enricher fields, the window and no-query execution, not invented deployment assumptions or host duties; they do not resolve retention-policy compatibility. Original uncertainties remain in the audit and do not count as completed tasks.

Actual cost is **26,053 input / 1,543 output tokens**, no unknown usage, request p50/p95 **11.19/25.17 seconds**. This note-only workload is not comparable to prior whole-draft timings or a Runtime SLO. An eight-placeholder-predicates-per-note budget preview exceeded the unchanged IRQL byte bound; the actual model extracted three predicates and actual requests fit. Future oversize requests still stop without truncation; schema maxima are not simultaneous capacity guarantees.

The run, strict audit and disclosed developer-AI content-review paths/digests above retain 50 receipt-bound files and the execution archive. All original report self-digests validate without the historical collision exception. The negative predicate judgment is preserved, not regraded or used as model input. This is not independent gold or a general semantic oracle. Next work should independently represent and align source/candidate predicates, distinguishing subject, action, negation, time and object. Unproved equivalence remains unverified L1, not an automatic Runtime admission. New-source acceptance and larger A/B stay closed.

QA adds 22 regressions and passes 57 related interface checks, 92 targeted tests, **3,083 full tests plus 81 subtests** (219.44 seconds), changed/new project-Python Ruff, three bilingual/link checks and diff checks. Coverage includes typed catalogs, lossless source display, absent/changed references, conflicting evidence, source drift, quote-only edits, guidance/observation separation, size rejection and dependency stops without retries. QA runs after live measurement, with artifacts under the path above. Passing mechanics does not override the semantic counterexample.

### September 14: host-owned task text and two-pass predicate review (historical)

The later September 14 round retains 9B, the same two known failures and the authorized 17-call ceiling. The stage remains incomplete. Host-owned task IDs preserve exact text, offsets and caller roles without an LLM-written requirement paraphrase. Missing role metadata yields one unclassified whole-task anchor. Roles are navigation, not semantic authority; original constraints remain even if misclassified.

Note extraction sees only notes, not sources/body. A second call judges each frozen predicate against original notes and complete sources. The host computes aggregate states: missing exact evidence, uncertain interpretation, inference-only or other-predicate evidence cannot establish support. Predicate coverage and entailment are still unproved; there is no global model verdict that overrides unknown entries. Independent edits run in separate one-node Runtime graphs; failed siblings are not retried or consumed.

The new opt-in CLI above permits at most 16 calls: three task checks, two note passes before and after, eight shared editors, and one final task check. Existing graph limits and all no-action boundaries remain. Narrow typed-literal rendering is explicitly wired into this diagnostic before final review, not default DSH or historical probes.

**Actual results:** CAPA splits not-reviewed from not-approved, but its reviewer labels all four initial predicates directly supported with empty evidence. The host retains unknowns. One note editor returns the whole answer and is rejected; its independent sibling removes the unsupported not-reviewed clause. The body and first note remain unchanged. Later reviews still lack valid locations/evidence and strengthen at-most-three to exactly-three. This is useful local editing, not reliable semantic review or task approval.

IRQL's editor changes interval operators but consumes the closing Markdown fence. Assembly rejects the candidate; no revised draft or note is published. Expanded assumptions in its unapplied notes remain negative evidence. A post-run fix now validates assembled-parent fences at individual edit binding, isolating malformed candidates earlier. It has regression coverage, not a new live-model success.

**18 actual calls** (IRQL eight, CAPA ten), 95,154 input / 6,240 output tokens, no unknown usage; request p50/p95 16.80/54.45 seconds. Different call composition prevents causal performance comparisons. Seventeen bound replies and one rejected reply are not 17/18 semantic accuracy: the bound query subsequently fails assembly. Neither case satisfies the stage exit.

The original final aggregation also fails: a linked case digest was overwritten by the checkpoint's self-digest. Future rows use `caseReportDigest`. The original invalid file remains untouched and no original final summary exists. Default collection still rejects all drift. Explicit forensic reconstruction recognizes only the exact old hash computation, requiring a separately valid linked case report and matching state/counts. The derived evidence explicitly states `allOriginalReportSealsValid=false`; it neither normalizes arbitrary corruption nor changes historical grades.

The original run and new zero-call audit paths/digests above retain 18 model receipts, 90 bound files, the execution archive and actual candidate reconstruction or reproduced assembly failure. Content findings are disclosed developer-AI known-case review, not independent gold. No new source scripts, reads, Effects, baseline edits, commits or pushes occur. Next mechanism priorities are evidence localization before judgment, nonexpansive note editing and host-owned artifact structure; rejection alone must not be counted as improved usefulness. Historical results below remain separate.

Final QA passes **158 targeted tests, 3,061 full tests plus 81 subtests** (234.53 seconds), 48 more regressions than the prior round. Changed/new project-Python Ruff, three bilingual/link checks and diff checks pass. Tests cover early fence rejection, valid code edits, independent failure isolation, strict original-digest rejection, exact collision reconstruction and arbitrary-drift rejection. Artifacts use the QA path listed above. These verify engineering contracts, not model semantic accuracy; the post-run fence/digest fixes have no new live-model measurement.

### Previous round: LLM duty contracts and explicit sibling continuation

September 14, 2026. The user approved the prior redesign. This optional diagnostic implements candidate-blind duty proposals, applicable local artifact checks, focused duty and separate note review, bounded editing, and a fresh final review. It does not change default DSH routing, production permissions, old scores or fixed exit criteria.

The cap is 17 model calls per task: one proposal, up to six duty checks plus one mandatory note audit, up to eight shared body/note editors, and one final review. Existing Runtime graphs still permit at most eight calls each. Failures are retained without retries; excess work remains open and oversized input is not truncated. Source anchors prove text locations, not correct interpretation or completeness. Neither positive reviews nor graph completion authorize writes or semantic acceptance.

Pure local checks cover fence closure, strict JSON parsing, Python AST parsing without execution, literal decimal ratios, and a narrow KQL time-predicate subset. Full language semantics, field/function validity, source truth and open meaning remain unverified. The checker is not a complete Kusto parser and runs no source scripts or database queries. Microsoft’s interval documentation is linked above.

The code links and CLI above identify the entry points. A manifest contains at most two historical `source`/`request`/lossless `taskScope` entries, no expected answers. Receipts and actual input provenance are checked. Default invocation is zero-call preflight; `--run` requires a new output directory and explicit budget. Outputs retain the contract, focused opinions, local findings, edit ownership, revised candidate, final review, costs, receipts and frozen source archive. Local artifacts are Git-ignored.

The first real run completes zero of two end-to-end tasks. IRQL identifies the interval defect but a note edit exceeds its scope and blocks the original edit graph. CAPA puts an observation in the task-quote field; the entire contract is rejected. These failures remain unchanged.

An explicit new recovery path runs only unattempted, source-independent siblings, never failed calls. CAPA's unsupported “not reviewed” predicate is removed while nonapproval and original body facts remain. Its failed business-duty contract is still quarantined. IRQL's unattempted editor changes the operators to >= and < but leaves invalid datetime literals. A separate zero-model-call, source-bound typed renderer fixes exactly two literal spellings without changing timestamps or operators. The narrow interval check now passes, NOT full Kusto validity, query execution or unassisted model repair. The typed output has no new LLM final review. Original note failure, extra sections and citation/explanation risks remain. The renderer is an opt-in API invoked by this offline audit, not enabled by default in the original probe or DSH.

The recovery validates parent receipts/archives, rechecks retained successful candidates, never consumes failed node output, preserves open items, and uses a one-shot sibling claim so renaming output cannot reset the per-task 17-call cap. It is a research recovery diagnostic, not general production durable scheduling. The original probe still stops its serial edit graph under the original contract; recovery is not a first-pass success.

Across two versioned runs: **17 real calls**, 12 for IRQL and five for CAPA, with 126,955 input and 10,192 output tokens, no unknown usage. Request p50/p95 are 31.32/77.44 seconds, not task latency, Runtime SLO or causal A/B gains. Initial and recovery costs, artifact paths and digests appear above. The disclosed developer-AI audit reconstructs actual candidates and saves typed patches; it is neither independent gold nor a general string-matching semantic oracle. No source scripts, fresh business reads, Effects, old-score changes, commits or pushes occur.

Final QA passes 62 new tests, 224 related targeted tests, **3,013 full tests plus 81 subtests** (222.60 seconds), changed/new project-Python Ruff, three bilingual/link checks and diff checks. QA runs after model measurement. Graph-timeout, test-envelope and duplicate-directory mistakes were corrected before their corresponding live calls. Local checks gained negative tests for valid multiline/extra predicates, strings/comments, invalid dates and unsupported rendering contexts. Limits were not relaxed. A final recovery guard requires parent inputs to match the actual model request; both historical inputs pass zero-call verification followed by full regression. None of the 17 model calls or historical summaries was rerun/rewritten. QA artifacts are under the path listed above.

The stage remains incomplete. Next priorities are host-owned, typed task versus observation anchors; reliable note/atomic-predicate targeting and contradiction handling; and typed generation/checking of applicable deterministic artifacts. Model interpretations still strengthen “at most” to “exactly” and “do not execute” to “only output query text”; one reviewer marks a composite assertion supported while admitting an unsupported predicate. Successful editing does not validate the reviewer. New-source transfer and expanded DSH A/B remain closed under the unchanged exit criteria.
