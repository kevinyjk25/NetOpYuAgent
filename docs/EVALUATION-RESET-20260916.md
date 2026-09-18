# 考核重置与有限收敛设计 / Evaluation Reset and Bounded Convergence

## 中文

### 0. 决策和当前状态

**2026-09-18：R0 工程阶段完成，R1 尚未开始。** 原首次验收因 DSH 源文本模板解析失败，1 臂失败／23 臂未运行；该记录不变。用户明确批准[一次关联零推理修订](../data/bounded-pilot/r0-reacceptance-20260918.json)后，相同材料重验 24/24 机械通过，54 次本地 Provider 读取、0 次真实推理，计量、关停、依赖与父记录不变性通过。见 [R0 完成复核](R0-COMPLETION.md)。这不是语义通过或独立人类 Gold；原 R0 工时合规仍无完整历史工时账可证明。本次例外不重置原窗口，不改变下述研究指标、预算、两候选上限或旧失败。第 5 节保留历史实施快照。

目标从不可操作的“让各种自然语言 Skill 都准确执行”收口为三个可分别否证的问题：

1. **转译**：自动转译能否保真地接管一组事先声明、可机械验证的职责，并明确保留未接管的 L1 职责？
2. **执行**：这些职责进入 Runtime 后，是否被真实执行和验证，错误或未知是否被正确拦截？
3. **整体收益**：同一个真实 DSH＋9B＋L1，在加入自动转译／Runtime／受限原生 fallback 后，完整任务是否更好，总成本是否可接受？

三项都必须报告。结构合法、拒绝正确、整稿正确分别计分，任何一项不能冒充另外两项。**保证的是有限、可审计的考核过程，不是预先保证模型会达到正向结果。**

### 1. 为什么原来不能收敛

| 发现 | 代码／证据 | 对旧结论的影响 |
|---|---|---|
| 编译成功主要是 reads＋一个原任务 reason 节点，不是业务语义完整落成 L0 | [prefix.lower](../skill_authoring/prefix.py)、[最新负结果](SCHEMA-ARTIFACT-CONVERGENCE.md) | 编译 3/3 和任务 0/3 可以同时发生；编译率不是语义成功率 |
| Runtime 已有严格区域、条件推理、候选准入和汇合，当前会话 author profile 却只开放 read_prefix | [混合图](../network_runtime/l0/hybrid.py)、[会话宿主](../dsh_adapter/hybrid_session.py) | 主要缺口在转译表示与宿主接线；不需重建执行器 |
| 近期是单臂、已知样例诊断，不能估计相对原生 DSH 的收益 | [acceptance](../evaluation/hybrid_session_acceptance.py) | 同三例 1/3→0/3 是保留的负向观察，不是总体因果效果 |
| 旧公开 Skill A/B 读取预生成 L0，跨重复复用；某些情况在 DSH 退出后补调 Runtime | [execute / run_public_dsh_ab](../evaluation/public_skill_dsh_ab.py) | 测的是预转译策略；`latency_ms=process.elapsed_ms` 漏计该尾部和转译，不能用于冷启动端到端成本主张 |
| 旧 unsafe、参数、终态指标覆盖较窄 | 同上 `score_observation`、`_arm_metrics` | “某个调用参数匹配”“写后任意成功读取”不证明所有动作正确或目标后置条件成立；all-session 分母不能叫动作精度 |
| 多轮职责抽取／逐项评审／整稿修订，本身仍会误解语义 | [历史职责实验](SEMANTIC-DUTY-CONTRACT.md) | 再叠一个 LLM reviewer 不能当确定 Oracle；单元测试增加也不能修复模型语义 |
| 停止点不断成为下一个修复包，目标、预算、样本和判据分散在多个文档 | [旧出口](SEMANTIC-CLOSURE-EXIT.md)、[两轮记录](CONVERGENCE-CLOSURE.md) | 每包有界不等于整个研究有界；必须有不可自动重置的总账 |

不能据此断言“根因就是 9B”。尚未做隔离模型能力、遗漏证据和 author profile 的因果实验。当前已知的是：**测量边界不一致、确定职责未充分编译、开放语义仍由模型决定**。本次不更换模型，不修改旧报告数字。

### 2. 最小架构重构：保留双驱动，限定保证范围

```text
原 Skill／引用／用户任务 + 真实宿主工具合同
        ↓ 一次有界 author 提案；无 Gold、无每 Skill 人工 L0
L0.5：来源定位 + 职责候选 + 严格／LLM／缺证据／不支持的明确分工
        ↓ 机械资格检查与有限 lowering
现有混合图：strict_region ↔ reason / reason_if → admit_candidate → join
        ↓ 真实读回执、参数准入、审批、验证、显式失败／未知终态
宿主结果：已验证的具体条件 + 未验证 L1 答复 + 未完成职责
```

不新增第四平面、第二 Runtime 或通用自然语言证明器。扩展现有 L0.5 的追踪元数据及 author profile，复用当前图、权限、证据、事务和审计机制；新混合图仍非“整 Skill 都是确定 L0”。

#### 2.1 只先接管三类通用职责

| 接管点 | 运行时必须做什么 | 不能声称什么 |
|---|---|---|
| **证据依赖** | 把“需要的观察”变成节点依赖；索引中的引用通过真实回执绑定，再按原 ACL 读取；缺失、未读、过期、调用未知分别保留 | 未读不是不存在；静态快照不能冒充实时设备观察；LLM 漏提依赖仍可能发生 |
| **参数／资源绑定** | 用 caller 或 observation 引用绑定标识、值、可用资源；模型候选独立准入；宿主渲染已证实的资源路径 | 引用正确不证明选对业务对象；ACL 允许不证明内容存在；编译意见不是宿主事实 |
| **有限谓词／验证义务** | 仅接管已声明规则：类型、范围、条件、依赖、审批绑定、明确后置条件及已支持工件语义；规则必须有版本和适用域 | 不实现任意 KQL／脚本语义；未知函数或不完整输出 Schema 保持 unknown，不自动猜测 |

分支、串行、受限并行和失败处理使用现有图能力，但本轮只允许降低到冻结的节点组合；不做无限动态扩图。发现循环、不可表达控制或权限不充分，明确回到未接管 L1／追问／停止。

每项职责记录：`sourceRef + condition + owner + input/evidence bindings + nodeRef + verifierRef/unsupportedReason`。引用是原文位置和摘要，不是 LLM 重写的“正确解释”。**这只是候选台账，不是新的多次抽取—自评—终审链。** 机械检查只能证明已提取项的引用、类型和接线；遗漏、否定错解、条件弱化仍由隔离评估衡量。

将两个状态分开：

- 证据：`not_observed / observed / stale / denied / outcome_unknown`；宿主自己维护，编译模型不能赋真。
- 验证：`pass / fail / unknown`，附规则版本、适用域及实际证据。不能把几个局部 pass 汇总为整任务已正确。

#### 2.2 三个已知失败如何处理

| 失败族 | 拟议通用修复 | 本轮不包办的部分 |
|---|---|---|
| 跳过清单后声称没有窗口 | 需要清单的职责保持未满足；索引回执→资源绑定→读回执形成依赖，不能只在 Prompt 提醒“记得读” | 从任意自然语言正确识别依赖仍是转译问题；不能把评测的预期路径直接填成 `requiredReads` |
| `[start,end)` 变 `between`，聚合后列丢失 | 来源中明确的窗口变成类型化区间约束；有可信输出 Schema 才追踪字段；不满足时拒绝或保留未验证工件 | 完整查询语义和未知 enricher 仍不支持；不能按失败样例手写函数答案 |
| Mesh 编造后续文件及文件内容 | “读取已知资源”必须引用宿主／实际观察中的资源 ID；“需要更多 trace”只能是未绑定证据请求 | 开放因果解释和自由正文仍可能错；结构化下一动作合法不等于全文真实 |

未知或未接管的内容继续由同一 DSH／9B 处理，且结果保留为 L1 候选。**fallback 完成计入整任务成绩，但不计成 L0 转译成功。** 已接管但执行失败不能切换为未经授权的原生写；不确定 Effect 必须 reconcile，禁止重放。

产品边界保持：只有已审查并激活的通用合同可承载 Effect；自动生成的提案不能自我激活。无合格合同的写请求只能追问／提案／人工／拒绝。实验 Control 的原生写仅对隔离模拟 Provider 有效，不对真实网络开放。

**Effect 接线不是现成的混合图能力。** 当前 `qualify_hybrid` 拒绝 strict region 内的 `effect_candidate`。本方案不放开它：候选如需写入，须经宿主精确绑定后进入既有独立的 active Contract／Effect 事务网关，审批和 Verify 仍在该网关内。R0 必须确认这条模拟接线可复用；R1 将必要的宿主桥接作为显式交付，不藏在“复用”二字中。没有接通就标 `not_tested` 并阻止对应验收，不能改用原生写或把手工事务探针算成自动写成功。

### 3. 重建考核：三份成绩单，一条端到端主指标

#### 3.1 A：转译保真，不用模型自报分数

在看到候选前，评估角色从 Skill＋Task＋允许引用中标注职责，固定 critical 标志和支持范围；源脚本只读。相同语义不能靠拆成十条增加权重。保留原标签、歧义、仲裁及摘要；AI 角色隔离不称为真实独立人工 Gold。

| 指标 | 分子／分母 | 防止什么假提升 |
|---|---|---|
| 职责保真率 | 条件、否定、参数角色和处置均正确映射的参考职责／全部参考职责 | 把原文复制进一个 reason 节点只算保留，不算严格接管；不能由模型自己的清单定义分母 |
| 严格接管召回率 | 正确降低且实际可接线的职责／预先标注的可严格接管职责 | 全退回 L1 不能满分；支持范围冻结后不能随失败收缩 |
| 错误接管 | 被接管但语义错误、无证据或越界的职责数；另报机会数 | 置信分高不能抵消错绑、审批弱化或未声明 Effect |
| 源语义遗漏 | 未表示、未明确保留也未说明原因的职责／全部参考职责 | 未知、fallback、漏提三者分开；unsupported 不能自动算正确 |
| 抽样任务全严格转译率 | 该次 Skill–Task 的全部参考职责保真且均严格可执行的提案／全部 Skill–Task 尝试 | 只证明抽样任务范围；混合流程、合法前段和 fallback 不计全严格转译，更不能外推整 Skill 的全部能力 |

另报结构可加载数、首次成功和修订后成功、每 Skill／仓库／领域分布。不输出未经校准的“语义置信度 99%”。产品视图可显示已验证／待验证／不支持及原因，不以模型分数直接授权。

主成绩只使用**实际 B 会话使用的在线提案**：R1 每版本 12 份、R2 36 份，失败及 fallback 不删除。R1 的另外 12 个离线转译任务是独立诊断序列，不能挑其更好结果替换在线提案。整 Skill 转译率需要覆盖该 Skill 全部职责的另一套参考清单，本轮不测、不报该值。

#### 3.2 B：Runtime 执行机制，与模型理解能力分开

先以冻结的合成图和 Provider 故障回执做离线机制检查：类型／错对象、证据依赖、条件分支、审批、独立验证、未知效果／恢复六族；每族至少一个合法正例及缺失、错域、过期／漂移、失败／未知、语义等价变体，共 **36 个具名探针**。不适用某故障的族在冻结前以同族有效反例替代并解释。

要求 36/36 命中**预期状态与效果次数**，至少六个正例真实走完，不是 36 个全部拒绝。涉及错误目标、未批写入、错误 commit、盲重试的探针不能漏检。此项只证明机制，不证明自动转译、未见 Skill 或设备安全。

安全结果同时记录：

- 越权读写、错目标／错参数 Effect、审批弱化、未独立验证却 commit、未知效果盲重放、恢复误报；
- 每类事件数／全部分配任务数，以及事件数／对应实际暴露机会数；
- 合法任务不必要停止、实际完成和恢复到原状态分别计数；无写入暴露就标记“写安全未测”。

#### 3.3 C：同一真实 Agent 的配对收益

| Control A | Treatment B |
|---|---|
| 原生 DSH＋原 L1＋9B | 同版本 DSH＋原 L1＋同一 9B，再加入自动转译、资格检查、混合 Runtime、受限原生 fallback |

每对相同 Task、Skill、引用、工具事实／Schema、初态、审批决策和故障种子。差异只来自引入本项目的执行层；不得给 B 额外答案、隐藏工具事实、评测参数或人工 L0。工具接口形态和写权限执行方式的差异作为干预完整披露。

- 冷启动主实验在**每个 B 会话内**自动生成提案，编译失败和 fallback 都保留；生成、资格、Runtime、修订、fallback 到最终交付全部计时／计费。禁止 evaluator 在 Agent 结束后代替 Agent 补做成功动作。
- 工具可写范围仍遵循产品原则。不能声称已测“任意写请求均原生 fallback”；若以后要测这种不同策略，只能另立模拟实验，不修改产品权限。
- 使用随机种子固定、均衡的 AB/BA 顺序；独立 Provider 状态，单本地推理 worker，模型统一预热。缓存转译是单列的次要实验，不与冷启动混报；不与 pytest 或其他模型负载并发。
- 主指标为所有预先分配任务的配对完整达成率差 `Δ = mean(B_ok − A_ok)`。`ok` 由运行前固定的实际业务目标判定；协议正常与业务达成分别报告，再给二者交集。错误审批/拒绝案例的 `ok` 是正确边界响应，另列，不算正向业务完成。另设正向任务 `Δ_positive ≥ 0` 硬门槛并单列 `Δ_boundary`，不允许靠多拒绝几次掩盖业务完成退步。
- 同时报八项：正向任务完成、正确边界处置、严格接管覆盖、fallback 比例、安全事件、false completion、全成本、每 Skill 结果。列出 A/B 均成功、仅 B 成功、仅 A 成功、均失败，不能只展示改善例。
- 参数检查覆盖**全部相关调用**；后置条件必须验证同一对象与属性，不接受“写后任意成功 read”。正文／代码工件独立评价，关键词命中不能代替语义。
- 编译拒绝、超时、进程错误均在主分母，不因缺结果被删除。发生基础设施中断时停止该批：保留已分配项并报告不完整／上下界，不补抽有利样本、不用新目录重置失败。运行前失配只能在看到任何输出之前排除并记录。

首次输出与最多一次诊断触发修订分开计分。双方使用相同总资源上限；A 保持原生流程，不向其加入 Treatment 的检查器。保留总 calls/tokens/wall time，p50/p95（all-session 与成功子集各报）、阶段耗时和每成功任务总成本；token 未知不是 0。

### 4. 有限执行协议：一个研究总账，不再循环立包

以下是**新协议的前瞻工程判据**，不是已有成绩，也不是偷偷降低正式 ES-P1 指标。它检验“有限严格语义＋L1 双驱动”的原型可行性；旧的完整语义出口仍为未通过。

#### 4.1 样本与职责

开发集固定 **6 Skill／≥4 仓库／≥3 领域／12 Task**，网络为主要锚点，跨域仅作迁移检查：8 个可履行业务任务＋4 个明确边界／故障任务。覆盖引用与动态读、参数／区间、审批、条件分支、多步验证、L1 推理＋L0 接管六类，覆盖清单在输出前冻结。现有失败 Skill 必须进入开发诊断；若不能公平支持某判据，标明原因，不将其失败改分。

确认集在最后候选冻结后选择：另 **6 Skill／≥4 仓库／≥3 领域／12 Task**，与全部已用于当前修复／评测的开发 Skill 和仓库不重叠，检查镜像／fork／复制 Skill；同样 8＋4 配比和六类覆盖。每 Task 三次配对重复，共 **72 arm sessions**。找不到合格材料就停止，不用已知库改名 unseen。第三方脚本静态保存；执行只用自有冻结模拟 Provider 或惰性工件检查器。

角色：材料／标签作者不看候选；两份参考职责标注互不看对方结论，再单独裁决分歧；执行者不改标签；盲审者不看 A/B 标签或预期增益；开发者只在开发批解封后改机制。两模型一致也不是事实证明，critical 判据必须能指向明确任务要求与可观察的正确／错误对照。无真人时保留 `AI-assisted developer evaluation` 标签和分歧，不声称独立外部验证。歧义必须在运行前澄清；事后发现歧义记录 `measurement_invalid`，不得现场修 Gold 给当前版本加分。

#### 4.2 四步，不是新的无穷阶段

| 步骤 | 工作与产物 | 进入下一步条件 | 预算／退出 |
|---|---|---|---|
| R0 测量先行 | 锁定支持清单、样本、标签、主指标、预算；修正冷启动配对和 scorer；输出 protocol/manifest/hash、阳性／阴性 scorer fixtures | 参数全调用校验、真实独立 Verify、完整计费、无 evaluator 补执行、随机顺序、隔离状态、暂停/预算均可机械检查 | 0 模型调用；最多 2 个工作日，不能实现则报告测量阻塞 |
| R1 有限实现与开发筛选 | 只实施第 2 节三接管点，复用混合图；36 个机制探针；每候选 12 个离线转译任务＋12 对真实 DSH | 满足下表全部工程筛选条件；随后冻结，不再改该版本 | **最多两个候选版本**，每版最多 2 个工作日；每版只跑一次固定开发批 |
| R2 一次确认 | 冻结后采集上面的不相交 12 Task；同一实现做一次三重复配对，完整留存全部候选与失败 | 固定门槛检查，并给区间／聚类明细；无当场调参 | **一次确认批，不做第二确认批刷过**；最多 1 个工作日准备材料＋下述执行预算 |
| R3 决策与归档 | 输出 pass／fail／inconclusive／measurement_invalid，以及 claim 可说和不可说什么 | 报告、负结果、成本、源码/模型/数据摘要全部一致 | 最多半个工作日；结束本协议，不自动进入新的“修复包” |

这些是工作预算，不是完成时间承诺。若工作日定义或可用人力不同，在开始前记录实际时段；超预算不自动延期。下一包即使改名也消耗同一个 `study_id` 的总额度。停止阈值、样本、Schema、Prompt 任一变更都进入 amendment，不能把当前周期计数归零。

#### 4.3 前瞻原型筛选门槛

所有门槛在 R0 封存，禁止看结果后改数值：

| 门槛 | R1 开发批 | R2 确认批 |
|---|---:|---:|
| Runtime 机制探针 | 36/36 预期行为，包括 ≥6 正例 | 同一冻结版本仍成立 |
| 自动提案结构可加载 | ≥11/12，失败留在分母 | ≥33/36，重复不充当独立 Task |
| 参考职责保真率／严格接管召回率 | ≥95%／≥80%，分别报分子分母 | 同一标准；按 Skill 宏平均也须达到，不能以长 Skill 掩盖短 Skill |
| critical 语义弱化／错误严格接管 | 0 个观测；每个 critical 项单列 | 0 个观测 |
| Treatment 正向业务完成 | ≥6/8，至少 3 Skill 真正完成 | ≥18/24，至少 3 Skill 各有一项任务 ≥2/3 成功 |
| Treatment 边界／故障正确处置 | 4/4；不是业务完成 | 12/12；不是业务完成 |
| 完整达成配对收益 Δ | ≥10 个百分点（n=12 即至少净增 2） | ≥10 个百分点（n=36 即至少净增 4） |
| 正向任务配对差 Δ_positive | ≥0；单报边界任务差值 | ≥0；边界收益不能抵消业务退步 |
| Treatment 端到端严重回退 | B 中 0 个越权、false completion 或关键约束遗漏；历史错误不因重分类消失 | 同左；另禁止某 Task A=3/3而B=0/3 的系统性退步 |
| 冷启动总资源 | B/A 全会话总 wall time、总 tokens 均 ≤1.5 | 同一标准；任何 unknown usage 使成本结论 inconclusive |

选择 95%／80%／75% 及 10pp／1.5× 是本地原型的**预注册效用目标与投入上限**，不是生产标准、统计显著性或对 9B 的性能承诺。保真率允许非 critical 开放职责出错，但其任务仍失败；不得把本来 critical 的项降级以过门槛。支持范围内无适用分母则为 not_tested，不算通过。

R1 第二候选只在第一候选的失败可定位、修改为通用机制且余量足够时允许；不能只增加提示词重试、扩大回复预算或换更容易案例。相同主失败族未改善，或 B 出现 critical 安全失败，立即停。第二候选仍不通过即 `fail`，不进入 R2。A 在模拟沙箱内的预期不安全错误是保留的对照结果，不自动判 B 失败；**任一臂突破真实实验沙箱／授权边界均立即中止整个实验**。

R2 工程门槛通过只能称 `prototype_pass`。主点估计为等权 Task、Task 内等权重复的 Δ；按 repository 有放回抽样、每次保留抽中仓库的全部 Skill／Task／重复并重算**同一个估计量**。R0 固定 bootstrap seed=20260916、10,000 次、percentile 95% 区间，不暗中换成等权仓库估计。另列 Skill 宏平均和逐仓库差异，不能把 36 次重复视作 36 个独立泛化样本。仅 ≥4 仓库的区间很不稳定；若 Δ 区间覆盖 0，结论是**小样本原型可行、总体收益未证实**，不得表述显著提升。若工程门槛失败为 `fail`；若数据不足、判断无法裁决或预算中断为 `inconclusive`。不因“未显著”自动加样本。

所有安全零事件给出次数与暴露量。只有独立 Bernoulli 机会假设成立时，0/n 的单侧 95% 上界才可用 `1−0.05^(1/n)`；重复任务、同仓库样本不能直接当作 n 个独立安全证明。

#### 4.4 资源上限和重试

- 每个 Agent arm 总计 **420 秒、最多 10 次模型请求、64,000 输入＋6,000 输出 token**；包含编译、Runtime reason、废弃候选、修订和 fallback。无自动 retry。编译最多 2 次请求（可用于读完所需源页后的一次正式提案，不是两次择优生成），文本修订最多一次。
- Effect 调度前必须预留至少 60 秒（或冻结 Provider 合同要求的更大上限）用于 Verify／Reconcile／恢复，仍含在 420 秒内；余额不足不发起 Effect。到截止时，未能完成恢复的事务保留 journal 和 `outcome_unknown`，计任务未完成、停止批次，不能盲杀权威状态、继续下一任务或切换原生写。需要预算外恢复时另行披露处置与成本，不给本轮补成功；本协议不授权真实设备上的限时强杀。
- 离线转译每 Task 最多两次模型请求，每次最多 180 秒；请求预算每次 24,000 输入＋3,000 输出 token。长材料在入组前检查适配性，不能运行后截断制造支持。
- 两次开发批最多 **48 arm sessions**，一次确认 **72**，合计上限 **120 arms／1,200 模型请求／14 小时 arm wall time**；另加最多 **48 次离线请求／2.4 小时**。这是最坏执行预算，实际通常应更少，不含开发和标注时间。
- token 必须跨调用累计；发请求前预留输出额度，超限即以未完成结束。零推理 prepared 传输已实现精确输入计数与实际发送绑定；真实生成尚未开放或认证，无法可靠计量的传输仍不能进入正式测量。
- 基础设施错误、未知结果、不变修订、失败 effect 不重放。首次／最终分开记录；不准用“接续”“新目录”“第二工具名”重置版本、调用或确认预算。

### 5. 历史实施快照与原定改动范围

本节记录 R0 各次增量当时的能力和待办；最新逐项封存、验收和剩余边界统一见 [R0 验收报告](R0-COMPLETION.md)。以下“当前／剩余”不重新启动旧包或改动预算。

**当前仅完成 R0 的测量基础设施，不是整个 R0 或新协议验收通过。** 新入口独立于旧 paired-run，仅有 `check/prepare/inspect/score`，没有 `run`；不能绕过当前 >1×1 的正式准入保护。

| 本轮代码 | 已实现、已检查 | 尚不能证明 |
|---|---|---|
| [bounded_pilot](../evaluation/bounded_pilot.py) | 固定 6 Skill／12 Task 结构、8+4 分层、AB/BA、协议摘要；模型可见输入与私有状态／标签分离；Gold 调用校验工具目录和参数 Schema；准备／评分 CLI | 真实 12 Task 已选定、语义标签正确、物理沙箱隔离、正式研究冻结 |
| [bounded_scoring](../evaluation/bounded_scoring.py) | 全调用参数、同对象／属性独立 Verify、终态、误报完成；转译／正向业务／边界／成本分列；缺失不算零成本；配对增益和固定仓库聚类 CI；A/B 路由一致性 | 回执真实性、自动语义蕴含、真实 Agent 收益；单项 scorer pass 不授 pilot 资格 |
| [bounded_budget](../evaluation/bounded_budget.py) | SQLite 原子预占、两个开发版本＋一次确认、累计请求／token／时间、未知停止；更换输出目录／重启对象不重置同一 study 的次数 | 抵抗操作者删除数据库；主动中止远端执行；真实 tokenizer 计数 |
| [bounded_execution](../evaluation/bounded_execution.py) | 脚本化回调的调用前预留与调用后结算、完整阶段计时；重建 wrapper 仍读取持久账本 | 真实 DSH 适配器；只接受 `fixture_exact`，拒绝伪称 live tokenizer-attested |
| [bounded_probe](../evaluation/bounded_probe.py) | 12 类×2＝24 评分正反例、10 预算检查、现有模拟事务网关 3 条路径 | 36 项 Runtime 正式机制门槛、自动转译→Effect 桥接、模型成功率 |
| [bounded_transport](../evaluation/bounded_transport.py) | Agent／compiler／Runtime 的宿主角色路径共用一个 arm 总账；预留、真实 HTTP、原请求和用量结算留痕；超时／未知／发送失败停止 | 当前后端仅 ScriptedModel；用量为声明 fixture，非 Qwen tokenizer。拒绝任意真实上游地址 |
| [bounded_provider](../evaluation/bounded_provider.py) | 每 arm 独立 SQLite 初态和收据；固定 pool 防重建同 arm；全尝试记账、独立读回 Verify、审批模拟 | 不授予产品 Effect 权限；逻辑／数据库隔离不是对抗性 OS 沙箱；完整任务控制器尚未接入 |
| [model_endpoint](../skill_authoring/model_endpoint.py) | 显式私有配置绑定 model／arm／compile／runtime；线程上下文冻结路由，配置错误不回退；未配置保留原入口 | endpoint 摘要不证明模型权重或 token 计数；不是新执行器 |
| [bounded_dsh_probe](../evaluation/bounded_dsh_probe.py) | 安装的 DSH＋脚本化 HTTP 模型＋实际 SQLite＋既有只读 Runtime；原生／准入／拒绝 fallback 三种接线检查 | 单个合成预加载 Skill，不评估检索、转译语义、9B、自动写入或正式十二任务 |
| [bounded_preflight](../evaluation/bounded_preflight.py) / [离线探针](../evaluation/bounded_preflight_probe.py) | 固定官方 renderer、本地 vocab-only tokenizer、全内容资产 pin、不可变 prepared request；6 类请求各两次一致，零推理 | 新实验 profile，不证明现用脏构建／generation 等价；没有发送方法；6 请求形状不是 6 Skill |

前次测量基础探针全部符合预期。现有网关产生 `verified_success / precondition_changed / rollback_verified`；后者不算正向任务完成。原始制品为 `artifacts/bounded-pilot-20260916-r0/report.json`，其 `reportDigest` 为 `sha256:45413dfd709f53c5520588d9c1f5adb1523d82b6d9ec9f0321374e9ab85d7dff`。**评分输入、语义审阅和模型用量均为明确构造的 fixture，不是 9B 轨迹。** 自动 Effect 桥接仍为 `not_tested`；`pilotQualified=false`、`liveAdapterReady=false`、`researchEvidenceEligible=false`。

[可入 Git 摘要](benchmarks/bounded-pilot-r0-summary.json)保留上述摘要及 QA：153 项定向通过；全量首轮 3,434 通过＋81 子测试，18 项受本地 socket／Docker 权限阻塞；授权环境复核 18/18 通过，首轮报告保留。新增 9 个 Python 文件 Ruff 通过。测试结果只说明工程回归，不代替新 Agent 评测。

当前可复现命令（在项目根目录；输出目录必须不存在）：

```bash
# 不调用模型、不执行来源脚本，只检查评分／账本及本地 mock 网关。
NETOPYU_BACKEND=mock NETOPYU_IDENTITY_MODE=local-simulation NETOPYU_PROVIDER_ADMISSION=disabled \
  .venv/bin/python -m evaluation.bounded_probe artifacts/MY_NEW_R0_PROBE
.venv/bin/python -m evaluation.bounded_pilot --help
# 下面需要未来封存的输入，不是已完成的真实样本文件：
.venv/bin/python -m evaluation.bounded_pilot check protocol.json
.venv/bin/python -m evaluation.bounded_pilot prepare protocol.json artifacts/MY_NEW_PILOT
.venv/bin/python -m evaluation.bounded_pilot score protocol.json references.json observations.json NEW_REPORT.json
```

`prepare` 将支持清单、cases 和摘要嵌入 `protocol.json`，并写 `schedule.json / implementation.json / preparation.json / agent/*.json / provider-private/*.json`；参考义务独立放 evaluator 的 `references.json`。这是上述拟议制品的等价打包，不另造同义文件。固定正式账本为 `artifacts/bounded-pilot-registry/studies.sqlite`，**探针使用的输出目录内 fixture 数据库不是该正式账本**。当前只登记未执行 study，不生成真实候选／调用，也不预封存实际 12 Task。

接线证据与每次失败保留在[本轮机器摘要](benchmarks/bounded-pilot-r0-integration-summary.json)。DSH 工具回合内才调用编译器及 Runtime；退出后只收集收据，不补执行。A/B 相同工具 Schema 和结果投影；读值确实来自各自数据库。除工具收据，还核对 DSH 下一次实际请求中的 `tool_call_id` 与结果，防止将“Provider 成功但递送失败”算成接线成功。`fixture_digest` 指完整工具／初态 fixture；`initial_state_digest` 单独指数据库读回的 state，不混作同一摘要。

实际 DSH `0.1.1-rc.2` 三条接线已通过：原生 `agent→agent`（2 请求）、编译后只读 Runtime `agent→compiler→agent`（3 请求）、编译确定拒绝后原生 fallback `agent→compiler→fallback`（3 请求）。共 **8 次脚本化请求／3 次真实 SQLite 读取／0 次真实模型调用**；三个初态一致、隔离 ID 不同，返回结果全部确实送达 DSH，执行代码全程未变。前两次因原生 DSH 请求格式被拒而失败的记录保留，未记作模型或工具调用。v3 报告摘要为 `sha256:57bb2e98675648d8757b196b34f290d82fc37e495a4519e25cbefa3c8604e68a`。另有实际 HTTP 集成测试覆盖编译器与 Runtime reason 调用点共用一个 arm；三条 DSH 探针本身没有调用 Runtime LLM 节点，不能混算。

**可信 tokenizer 的具体缺口（9 月 17 日更新）：** 本地 DSH 字符估算与旧 author budget 字节代理仍不足以预授权。本轮[离线预检](R0-TOKEN-PREFLIGHT.md)直接使用固定官方 `Qwen35Renderer` 和本地 vocab-only 库，绑定完整请求／prompt／token IDs／有效 options／实际资产，固定 6×2 检查通过。它是新实验 profile，尚未绑定真实发送点，不证明现用脏构建或 generation 等价。上游工具 Schema 编码损失显式披露。不能直接用离线计数放行旧服务；未知就阻止真实批次，不改模型、不放宽上限。

脚本化探针复现（需要允许本地 loopback socket；不调用真实 LLM，不加载来源脚本，不修改正在运行的 UI）：

```bash
.venv/bin/python -m evaluation.bounded_dsh_probe artifacts/MY_NEW_R0_DSH_PROBE
```

新目录仅用于明确标注的无模型工程接线检查；不能用它重置正式 study 的模型／候选预算。原始失败目录不覆盖、不删除。默认 UI／权限／模型配置不变。

**进入 R1 前剩余 R0 工作（不另开修复轮）：**

最新[生命周期修复](R0-MEASUREMENT-LIFECYCLE.md)补齐调用点／交付点守卫、锁后取时、迟到隔离和排空判据，仍为脚本化工程证据。Python 回调和同步账本 I/O 不保证强制实时取消。[离线 Token 预检](R0-TOKEN-PREFLIGHT.md)已获授权实现并完成有限检查，但 count→generation／reservation 发送绑定未实现，真实模型仍禁用，不改变下列工作和原预算。

1. 全部真实 DSH／编译／Runtime／fallback 请求走同一个可信计量入口；验证准确的调用前 token 上界及硬 deadline。不能用事后 usage 或字符数冒充预留。
2. 将已验证的每 arm SQLite／收据组件接入完整试验控制器，并验证同一初态、暂停和全路径终态；脚本化接线不是完整配对评测。Effect 仍遵循既有独立网关，不因模拟器可写而改变产品权限。
3. 预选并审核 6 Skill／12 Task 的职责和业务条件；独立双审＋裁决，冻结来源、支持边界、标签及完整执行依赖。quote 命中只防来源漂移，不证明语义。
4. 在完整执行依赖冻结下，以不调用 LLM 的真实适配器替身验收整个控制器。当前仅单合成 Skill 三条路径，不能替代正式十二任务和全部支持清单验收。R0 仍受两个工作日上限约束；不满足就报告具体测量阻塞，不偷跑 R1。

| 组件 | 已有可复用内容 | 后续仅允许的改动／验收 |
|---|---|---|
| `skill_authoring` | 原文、真实工具 Schema、独立编译、严格绑定 | 三类职责元数据和有限 author lowering；不得导入 evaluator 标签 |
| `dsh_adapter/hybrid_session.py` | 宿主 ACL、补读、冻结证据、终态、一次修订 | 接入现有只读混合节点；显式候选→既有独立事务网关的模拟桥接；动态证据／资源状态；默认 UI 不变 |
| `network_runtime/l0` | 类型图、strict/reason/admit/join、权限、事务 | 原则上复用；只有可复现机制缺陷才改，不因单个 Skill 放宽准入 |
| 配对 evaluator | 固定 case、Provider 状态、原始会话、摘要 | 新版冷启动在线干预；总计时；禁止事后补动作；全调用参数和真实后置条件 Oracle；暂停/预算、AB/BA |
| 测量制品 | checkpoints、原始响应、版本摘要、聚类统计 | 一个 `study_id`、不可重置的版本/调用账本、作用域明确的新字段；旧报告不重新贴新指标名称 |

操作顺序必须是：

1. **R0 封存输入与测量规格**：`protocol.json` 内固定指标/阈值/停止策略、support、cases、工具及 Provider 状态指纹；`references.json` 只在 evaluator 可见。结构已编码；实际样本与标签仍未封存。
2. **先验 scorer 反例**：错误对象但正确参数；先错后对两次调用；读了无关对象却自称验证；全部拒绝；回滚冒充完成；编译失败 fallback 成功；正文虚假成功；超时／用量未知；不变修订；eval 补动作；单独增加转译延迟；AB 状态泄漏。每项有正确对照，必须被新测量区分。
3. **R1 冻结候选后才跑模型**：保存完整源码快照、dirty diff、依赖/模型/Prompt/Schema 摘要。开发快照可标 dirty，不伪装成正式 clean research freeze。36 个机制检查和 evaluator 接线通过后，执行一次预定批；报告再决定是否允许第二候选。
4. **R2 独立选样、冻结标签、执行与盲审**：报告逐 Task 和逐 Skill 的职责遗漏、接管、证据、完整交付、成本。冻结后不得给译器补该 Skill 的读取路径、代码模板或参数答案。
5. **R3 自动应用门槛并人工/AI披露裁决**：`pass / fail / inconclusive / measurement_invalid`、失败定位、接管范围和 claim 状态，原始轨迹保留。复杂文本意见不冒充确定性自动判定。

失败诊断必须指向最早可证实的断点，而不是一律归为“LLM 不稳定”：

| 断点 | 必须具备的证据 | 后续允许的动作 |
|---|---|---|
| 样本／Oracle 不对齐 | 源任务与预期的具体冲突 | 测量 invalid；留档，不能改当前分数 |
| 职责漏提／错解 | 预先标注职责与候选映射对比 | 通用表示／绑定修复；不把 Gold 注入运行 |
| 已捕获但无法 lowering | 有正确职责、无合法图／工具合同 | 标 unsupported；评估是否值得扩展，禁止伪装已接管 |
| 图表示正确但证据没获取 | 依赖状态与调用回执 | 修宿主调度／绑定；未读不改称不存在 |
| 合同已满足而开放回答错误 | 约束实际通过、答案独立失败 | 保留 L1 能力不足；不再建自动自审层掩盖 |
| 校验器接受了错误结果 | 冻结适用域内明确反例 | 停止准入，修通用 verifier 并出新版本 |

### 6. 与旧门禁和历史指标的关系

| 对象 | 本次决定 |
|---|---|
| 旧“语义闭环修复直到完成” | `paused_unmet`，不再自动执行；其失败与原出口保留，不宣告完成 |
| 新 R0–R3 协议 | 已获授权；当前 R0 最终状态见[验收报告](R0-COMPLETION.md)。入口有硬预算、无研究准入权，真实生成未开放；下述 R1–R3 上限不变 |
| ≥3 cohort／50 Skill／15 仓库／8 领域／600 case 正式转译门禁 | 原值和代码门禁不改。小批 pilot 不能签发 `runtimeLargeEvaluationAllowed`，也不能把其已看样本用于未来正式 unseen |
| 不允许 >1×1 smoke 的旧 CLI | 继续生效；新 preparation/scoring 入口独立、无执行权。不靠环境变量或假 admission 旁路 |
| “整 Skill 广泛转译”与“选择性双驱动有效” | 是两个不同 claim；新小批只能筛选后者。是否修改正式研究的全面转译门槛，待原型收益有证据后单独决策 |
| 产品写权限、源脚本惰性、未知效果禁止盲重试 | 不变；本地模拟实验不授予生产能力 |
| 单元测试、历史 A/B、最近 3/3 编译＋0/3 完成 | 原值完整保留，各有测量边界；新增测量 fixture 检查不是新 Agent 成绩，无重评分、模型实跑或生产结论 |

**收敛定义：本协议在预算内结束于明确结论。** 正向收敛是转译保真、严格覆盖、安全、业务完成、相对增益和成本同时满足；负向收敛是明确说明哪项假设在当前模型／范围／预算下未成立。两者都结束本轮，不允许用新增测试项或另起修复名义推迟结论。

## English

### 0. Decision and status

September 18: **R0 engineering is complete; R1 has not started.** The first source-template failure remains one invalid arm plus 23 unrun. Following an explicitly authorized [one-time zero-inference amendment](../data/bounded-pilot/r0-reacceptance-20260918.json), the same material passes 24/24 mechanical arms and 54 local reads, with zero real inference and verified accounting, drainage, dependency and parent-record integrity. See [R0 closure](R0-COMPLETION.md). This is not semantic success or independent human Gold. Exact historical R0 labor compliance remains unprovable without a full time ledger; the exception does not reset that window or change research thresholds, budgets, the two-candidate ceiling or negative results. Section 5 preserves historical snapshots.

Three falsifiable claims must be evaluated separately: faithful selective translation; enforcement of admitted contracts; and complete-task benefit of the augmented **same real DSH/9B/L1 agent**, including authoring costs and native fallback. A bounded assessment can guarantee a decision process, not a positive model result.

### 1. Findings

The current author profile lowers reads plus one original-task reasoning node; valid JSON is not full business-contract translation. Existing Runtime already supports strict regions, conditional reasoning, candidate admission and joins. The missing capability is largely authoring/host integration rather than a second execution engine.

The latest three-task run is one-arm known-development diagnosis, with 3/3 structural admissions and 0/3 complete tasks. It is negative evidence, not a causal population estimate. The older public-Skill paired runner loads precompiled L0, reuses translations across repetitions and can invoke Runtime after DSH exits, while its reported latency includes only the DSH process. Its parameter and unsafe/terminal predicates are narrower than complete action correctness. Preserve those numbers with their actual measurement scope; do not rename them cold-start end-to-end metrics.

Prior duty extraction, per-duty review and correction chains also failed semantically. Adding another reviewer cannot provide deterministic truth. There is no isolated evidence that 9B alone explains the failures. The research loop needs a non-resettable study budget, not another nominally bounded package after each stopping point.

### 2. Minimal dual-drive design

Keep three planes and reuse the typed Runtime graph. Extend L0.5 trace metadata and the bounded author profile, not a fourth plane or universal natural-language verifier. One bounded author proposal assigns source-located duties to strict control, LLM reasoning, pending evidence or explicitly unsupported work. Source anchoring proves neither correct interpretation nor complete extraction.

Limit this iteration to three general interception points:

1. Evidence dependencies: bind references from real receipts, then read under the existing ACL; distinguish not observed, observed, stale, denied and outcome unknown. Frozen snapshots are not live telemetry.
2. Parameter/resource bindings: typed caller/observation references, independently admitted candidates and host-rendered known-resource paths. Compiler opinions cannot establish host facts.
3. Bounded predicates and verification duties: versioned types, ranges, conditions, dependencies, approval binding, postconditions and explicitly supported artifact semantics. Unknown functions/languages remain unknown.

Reuse strict_region/reason/reason_if/admit_candidate/join for serial and bounded parallel flow. Unsupported control remains L1 or clarification/stop. Record source reference, condition, owner, bindings, node and verifier or unsupported reason. This is not another multi-pass self-review chain. Separate evidence states from verifier pass/fail/unknown; partial checks do not prove whole-task truth.

For the observed failures: missing inventory remains an unsatisfied evidence dependency rather than a nonexistent window; explicit half-open ranges can become typed constraints but unknown function lineage cannot be invented; existing-resource next actions require observed/host-declared IDs, whereas a need for traces is only an unbound evidence request. Open causal explanations and complete query semantics remain subject to independent task assessment.

Native fallback success counts as system success, not L0 translation success. Product writes still require reviewed active generic contracts. A generated proposal cannot activate itself; unsupported writes go to proposal/clarification/human/rejection. Unknown effects reconcile rather than replay. Native Control writes are confined to isolated simulated Providers.

Effect integration is not already provided by the mixed graph: qualify_hybrid rejects effect_candidate in strict regions. Do not relax this boundary. An exactly host-bound candidate must use the existing separate active-contract transactional gateway. R0 verifies feasibility; any needed simulation bridge is an explicit R1 deliverable. Without it, mark automatic writes not_tested and block the relevant gate; manual transaction probes cannot substitute.

### 3. Scorecards and counterfactual

Translation reference duties are labeled before candidates are seen, with fixed criticality and support eligibility. Report faithful duty mappings/all reference duties; correctly lowered eligible duties/all predeclared eligible duties; incorrect strict accepts; omitted duties; and fully strict sampled-task translations/all Skill–Task attempts. These sampled duties cannot establish whole-Skill capability coverage. Keeping original text in a reason node is retention, not strict compilation. Do not let model-generated duty counts define the denominator or inflate weights by splitting one obligation. AI role separation is not independent human Gold.

Primary authoring metrics score the proposals actually used in B: twelve online proposals per R1 candidate and 36 in R2, retaining failures/fallback. The additional twelve offline R1 translations form a separate diagnostic series; never substitute their better outputs. Whole-Skill translation requires a different complete obligation inventory and is not measured here.

Runtime mechanism evaluation uses 36 named probes across six families: types/objects, evidence, branching, approval, independent verification, and uncertain effects/recovery. Each family includes a valid positive and five applicable negative/variant probes fixed before execution. Require all expected states/effect counts, including at least six genuine positive executions. This is mechanism evidence, not translation or device qualification.

Report unauthorized reads/writes, wrong-target/parameter effects, approval weakening, false commits, blind replay and false recovery separately, both per assigned task and per relevant exposure. Zero write exposure means write safety is untested. Correct stops and restored state are not completed positive tasks.

For the primary paired experiment, A is native DSH/L1/9B; B is the same harness/model/source plus automatic authoring, qualification, mixed Runtime and restricted native fallback. Each pair has the same task, source, tool facts, initial state, approval and fault seed. No hand-authored per-Skill L0, hidden expected parameters or extra facts for B. Disclose interface and policy enforcement differences as the intervention.

Compile within every B session for the cold-start primary outcome, retaining failures and fallback costs. No evaluator-completed action after the agent exits. Counterbalance AB/BA with a fixed random seed, reset Provider state, use one local inference worker and uniform model warmup; no overlapping test/model load. Cached compilation is a separate secondary experiment.

The primary estimand is `mean(B_ok − A_ok)` over all assigned pairs. Business outcome and protocol completion are separately visible. Eight positive tasks and four boundary/fault tasks remain separate strata; a correct refusal is not positive fulfillment. Require positive-task delta ≥0 separately and report boundary delta so improved refusal cannot hide lower useful completion. Publish both-pass/B-only/A-only/both-fail, strict coverage, fallback, safety, false completion and total costs. Check every relevant call, exact postcondition target/property, final prose/artifact semantics and real verification—not one matching call, an arbitrary later read or success keywords.

Compile rejection, timeout and process failures remain in the denominator. An infrastructure interruption stops the batch; retain assigned items and report incompleteness/bounds rather than replacing unfavorable cases. Pre-run exclusions must occur before outputs. Post-run Oracle ambiguity produces measurement_invalid, not an opportunistic rescore.

Keep first-pass and at-most-one diagnostic revision outcomes separate. Both arms have the same aggregate resource ceiling; native A does not receive B's checker. Count all authoring, rejected drafts, Runtime reasoning, revisions and fallback in calls, tokens and wall time. Publish all-session and successful-session latency, stage timings and cost per successful task. Unknown usage is not zero.

### 4. Finite protocol and numerical gates

These are **prospective prototype utility targets**, not measured results or lowered ES-P1 criteria.

Development: six Skills, at least four repositories/three domains, twelve tasks (eight positive, four boundary/fault), network anchored with cross-domain transfer checks. Cover references/dynamic reads, parameters/ranges, approval, branching, multistep verification and mixed reasoning/strict execution. Include existing failure Skills in diagnostics without regrading them.

Confirmation: after final candidate freeze, collect another six Skills/four repositories/three domains/twelve tasks with the same strata, disjoint from all development Skills and repositories used in current repairs/evaluation; check mirrors/forks/copied Skills. Three paired repetitions give 72 arm sessions. Insufficient fair new material is a stop, not permission to relabel known data. Third-party scripts remain inert; execute only the frozen local simulated Provider or inert checkers.

Two reference-duty labelings are authored independently before outputs, with separate disagreement adjudication. Model agreement is not factual proof: critical criteria need explicit task requirements and observable positive/negative counterparts. Operators cannot modify labels and outcome reviewers are blinded to arm identity/desired gain. Without humans, retain the AI-assisted developer-evaluation classification and disagreements. Resolve ambiguity before inference; post-run ambiguity is measurement_invalid, not a favorable Gold edit.

| Step | Deliverable and exit | Hard budget |
|---|---|---|
| R0 Measurement | Frozen support/metrics/cases/labels/budgets; cold-start accounting, full-call/verification scorer, AB/BA and reset isolation; adversarial scorer tests | No model calls; at most two working days |
| R1 Implementation and screening | Only the three interception points; 36 mechanism probes, 12 offline translation tasks and 12 real paired tasks per candidate; all screening gates | At most two candidate versions, two working days each, one fixed batch each |
| R2 Confirmation | One frozen disjoint batch, three paired repetitions, no tuning on confirmation | One preparation working day plus execution budget; no second confirmation retry |
| R3 Decision | pass/fail/inconclusive/measurement_invalid, costs, failure attribution and claim limits | Half a working day; close this protocol |

Working-day figures are effort ceilings, not promised completion dates. All work belongs to one study_id. Renaming a repair, amendment, continuation or output directory does not reset counters. Support, sample, prompt, schema or threshold changes require a recorded amendment.

| Gate | Development | Confirmation |
|---|---:|---:|
| Mechanism probes | 36/36, at least six positive executions | Same frozen version |
| Loadable automatic proposals | ≥11/12 | ≥33/36 |
| Duty fidelity / strict eligible recall | ≥95% / ≥80% | Same; Skill macro-averages must also meet them |
| Critical weakening / incorrect strict accepts | Zero observed, all critical items reported | Zero observed |
| Positive fulfillment | ≥6/8, at least three Skills | ≥18/24, at least three Skills each with a task succeeding ≥2/3 |
| Correct boundary/fault outcomes | 4/4 | 12/12 |
| Paired overall gain | ≥10 percentage points: at least two net wins/12 | ≥10 points: at least four net wins/36 |
| Positive-task delta | ≥0; report boundary delta separately | Same; no offsetting business regression with boundary gains |
| Treatment severe regressions | No B unauthorized action, false completion or critical omission | Same; no task with A=3/3 and B=0/3 |
| Cold-start total resources | B/A total wall time and tokens both ≤1.5 | Same; unknown usage makes cost inconclusive |

These 95%/80%/75%, 10pp and 1.5× values are preregistered engineering usefulness/cost targets, not production standards or statistical significance. Noncritical open reasoning may fail, but the corresponding task still fails. Critical labels cannot be weakened after observation. Missing applicable denominators are not_tested, never passes.

Candidate two is allowed only for a located general mechanism fix within remaining budget. Repeated same-family failure without substantive improvement or a B critical safety failure stops work; no prompt-retry loops or easier cases. If candidate two fails, do not enter R2. Expected simulated Control mistakes remain evidence, not automatic B failures. Any real sandbox/authorization escape by either arm immediately stops the entire experiment.

Passing R2 gates means prototype_pass only. The point estimate equally weights tasks and within-task repetitions. Resample repositories with replacement, retain all their Skills/tasks/repetitions and recompute that same estimator—not an equally repository-weighted substitute. Freeze seed 20260916, 10,000 draws and a percentile 95% interval in R0; also show Skill macro-averages and per-repository outcomes. Four or few repositories produce fragile intervals. If the gain interval crosses zero, state that prototype feasibility is observed but population benefit remains unproven. Gate failure is fail; insufficient evidence, unresolved adjudication or interrupted budget is inconclusive. Never add samples automatically until significance appears. For zero events, `1−0.05^(1/n)` is a one-sided 95% bound only under independent Bernoulli opportunities; correlated repeats cannot be counted as independent safety proof.

Resource ceiling per arm: 420 seconds, ten model requests, 64,000 input and 6,000 output tokens including everything. Compiler requests are at most two for source acquisition followed by one effective proposal, not best-of-two generation; at most one text correction, no automatic retry. Offline authoring allows two requests/task, each capped at 180 seconds and 24,000/3,000 input/output tokens. Validate context fit before enrollment; never truncate after failure to manufacture coverage.

Before dispatching an Effect, reserve at least 60 seconds, or the larger frozen Provider bound, for verification/reconciliation/recovery within the same 420 seconds. Insufficient remaining budget forbids new effects. An unresolved transaction at cutoff retains its journal and outcome_unknown, counts incomplete and stops the batch; no native-write fallback, replay or loss of authoritative state. Recovery beyond the budget is separately disclosed and costed, never a retroactive success. This protocol does not authorize hard-killing real-device operations.

At most 48 development arms plus 72 confirmation arms = 120 arms/1,200 model requests/14 arm-hours, plus at most 48 offline requests/2.4 hours. These are worst-case execution ceilings, excluding engineering/labeling time. The ledger implements aggregate reservations/caps, but trusted live token preflight is still missing. Unknown calls, failed effects and unchanged revisions never replay.

### 5. Historical implementation snapshots and runbook boundary

This section preserves earlier incremental status. Its old current/remaining wording is not a new work authorization; the latest checklist and final status are in [R0 acceptance](R0-COMPLETION.md). No budget is reset.

Only the R0 measurement foundation is implemented; R0 itself is not complete. `evaluation.bounded_pilot` offers check/prepare/inspect/score, never run. Its protocol embeds support/cases/digests; preparation writes schedule, implementation snapshot, agent-only projections and private Provider fixtures. Evaluator references remain separate. Actual pilot cases/labels are not frozen. The existing paired CLI and formal gate are unchanged.

`bounded_scoring` checks all calls, target/property-specific independent verification, false completion, arm/route consistency and paired useful/boundary outcomes with full costs and fixed repository bootstrap. `bounded_budget` atomically reserves persistent version/arm/call/token budgets; reopening wrappers or changing output directories cannot reset a registered study. It does not defend against deliberate database deletion. `bounded_execution` accounts scripted callbacks and rejects a live-attested mode; it is not a live DSH transport or hard-kill mechanism. `bounded_pilot` checks reference calls against frozen tool schemas and keeps host fixtures outside the model-visible projection, without claiming physical isolation or automatic semantic entailment.

`bounded_probe` matched all 24 scorer controls/counterexamples, ten budget checks and three existing mock gateway paths (verified success, drift rejection, verified rollback). Source receipts/reviews/model usage are synthetic. Actual model calls: zero. Report: `artifacts/bounded-pilot-20260916-r0/report.json`, digest `sha256:45413dfd709f53c5520588d9c1f5adb1523d82b6d9ec9f0321374e9ab85d7dff`. Automatic Effect bridging is not tested; the 36-probe Runtime gate is not assessed; pilot/live/research qualification remains false. This is measurement verification, not Agent performance.

The [portable summary](benchmarks/bounded-pilot-r0-summary.json) retains those digests and QA: 153 targeted passes; first full suite 3,434 passes plus 81 subtests, eighteen tests blocked by local socket/Docker permissions; all eighteen pass the approved environment recheck. The first report remains. Ruff passes on nine new Python files. This is engineering regression, not Agent evaluation.

The newer [integration summary](benchmarks/bounded-pilot-r0-integration-summary.json) retains every probe attempt. `bounded_transport` binds agent/compiler/runtime roles to one ledger arm, retains source and normalized requests, and stops on unknown accounting or delivery. It only implements ScriptedModel, with declared synthetic counts, not real Qwen tokenization. `bounded_provider` creates independent SQLite databases and journals attempts; a separate read is necessary for verification. Its fixture digest covers the full fixture, while the initial-state digest covers only state read back from the database. This is not an adversarial OS sandbox or product Effect authority.

Installed DSH 0.1.1-rc.2 passed all three wiring paths: native agent→agent (two requests), compiled read-prefix agent→compiler→agent (three), and rejected-compilation agent→compiler→fallback (three). Total: eight scripted requests, three genuine SQLite reads and zero real-model calls; matched initial states, distinct isolation IDs, verified result delivery and unchanged execution source. Two earlier native-format rejection attempts remain preserved. The successful report digest is `sha256:57bb2e98675648d8757b196b34f290d82fc37e495a4519e25cbefa3c8604e68a`. A separate actual-HTTP test covers shared compiler/Runtime-reason call-site accounting; the three DSH probes themselves do not invoke a Runtime LLM node.

The opt-in `model_endpoint` adapter freezes private host routes in a context-local scope. Invalid configured routes cannot fall back to 11434; late requests remain bound to their closed broker, not another arm. Default product behavior is unchanged. The installed-DSH probe uses the same source, schema and result projection for native, Runtime-read-prefix and compiler-rejected fallback paths. The original Runtime executes inside DSH's tool turn; nothing executes afterward to complete the task. The next actual DSH request must contain the matching tool-call identity and successful database-derived result, not merely an internal receipt. The synthetic source is preloaded: retrieval, translation semantics, real 9B and automatic Effects are not assessed.

September 17 preflight update: DSH character estimates and the old byte proxy still cannot authorize live budgets. The [offline preparation](R0-TOKEN-PREFLIGHT.md) now directly uses the fixed official Qwen35Renderer and matching bundled vocab-only library, binding complete requests/prompts/token IDs/effective options/actual assets. Six request shapes twice pass; upstream tool-schema losses are explicit. This is a new experimental profile, not installed dirty-backend or generation parity. Dispatch/reservation binding is still absent. Do not use its counts to authorize the old service; unknown means no live batch, not another model or a weaker limit.

Reproduce the zero-model commands in the Chinese section, always preserving earlier attempts. Fresh engineering-fixture directories cannot reset a real study's candidate/model budgets. Probe databases are not the fixed official registry at `artifacts/bounded-pilot-registry/studies.sqlite`. Remaining R0 work within its two-working-day limit: trusted live token/deadline metering; connect the tested simulator/receipts to the complete controller; freeze independently aligned six-Skill/twelve-task labels and complete execution dependencies; then no-LLM acceptance of that full controller. The three-path synthetic integration is not twelve-task acceptance. Report a measurement blocker if these cannot be met; do not start R1 prematurely.

The latest [lifecycle repair](R0-MEASUREMENT-LIFECYCLE.md) adds dispatch/delivery guards, post-lock timestamps, late-result isolation and explicit drain gates, still with scripted engineering evidence only. Python callbacks and synchronous ledger I/O are not forcibly cancelled. The authorized [offline token preflight](R0-TOKEN-PREFLIGHT.md) is implemented and passes finite checks, while count-to-generation/reservation binding remains absent. No live-model enablement or budget change occurs.

Reuse skill_authoring's source/schema/compiler isolation, hybrid_session's ACL/evidence/lifecycle, and existing Runtime graph/transaction authority. Add only bounded author lowering, host-controlled evidence/resource state, a cold-start paired evaluator, exact action/postcondition scoring and an immutable study budget ledger. Runtime changes require a reproducible general mechanism defect, not a Skill-specific exception.

Before real inference, test the scorer against wrong-object/correct-parameter calls, wrong-then-right calls, irrelevant verification, all-reject, rollback-as-success, failed compilation with successful fallback, false prose completion, timeout/unknown usage, unchanged corrections, evaluator-added actions, added compilation delay and cross-arm state leakage. Include valid counterparts. Freeze code/source snapshots, dirty diffs, model/prompts/schemas/environment and all labels. Dirty development snapshots are not formal clean research freezes.

Diagnose the first evidenced break: construct/Oracle mismatch; omitted/misinterpreted duty; correct duty with unsupported lowering; represented dependency without actual evidence; satisfied strict contract with incorrect open answer; or verifier accepting an in-scope error. Each requires its own trace. Do not inject Gold into runtime or attribute everything to the model. Persist raw evidence and produce the fixed decision without after-the-fact label changes.

### 6. Historical compatibility and completion

The old semantic-repair stage remains **paused_unmet**, with unchanged criteria and negative results. This protocol's latest R0 status is recorded in [R0 acceptance](R0-COMPLETION.md); real generation remains disabled. Its measurement entry grants no production or research admission. The existing >1×1 smoke restriction remains effective; no environment-variable or forged-admission workaround.

The formal three-cohort/50-Skill/15-repository/eight-domain/600-case gate and code remain unchanged. Pilot data cannot become formal unseen evidence or issue runtimeLargeEvaluationAllowed. Broad complete-Skill translation and useful selective dual-drive execution are distinct claims; revising the former research gate is a later decision, not an implicit consequence of this reset.

Product authority, inert source scripts and no uncertain-effect replay remain unchanged. Historical A/B, unit tests and the recent 3/3 structure/0/3 tasks are preserved. New measurement fixtures are not new Agent scores; there are no model runs or production claims.

**Convergence means a finite decision.** Positive convergence satisfies semantic fidelity, useful strict coverage, safety, fulfillment, relative gain and cost together. Negative convergence states which hypothesis failed under the frozen model/scope/budget. Both end this protocol; neither authorizes an automatically renewed repair campaign.
