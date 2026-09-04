# EnsuredSkill 项目进展 / Project Status

## 中文

### 当前阶段

**当前主阶段：L1→L0 转译泛化门禁。**

权威命题是：概率性 Reasoning 只提出 Candidate Plan；Contract、Evidence、Guard、Risk 和 Transaction 决定是否允许 Effect；不合格转换安全停机，不能回退原生写。

ES-P0 的 Runtime 机械原型和小样本接线结论保留为 `local_hypothesis_supported`，但不再被解释为 L1→L0 高泛化证明。只有转译器先在冻结后采集的跨 Skill/仓库/领域未知集合上通过门禁，L0→Runtime 的规模化安全、稳定和准确性评测才具有研究意义。

#### 2026-09-04 评测可信性修正

- [x] 发现并修复旧盲审中类别后缀 ID 与固定三任务组序的提示泄漏；旧 48/48 结果保留但降级为有元数据提示风险的开发诊断；
- [x] 新增随机盐匿名 ID、单任务独立请求、模型输入白名单、私有映射与结果重绑定、可核查的实际输入及检查点完整性校验；
- [x] 用相同真实 9B 与 development-07 源包完成匿名化复验：4 Skill/12 task 协议完整、元数据盲态校验通过，行为一致 7/12、构造对齐 10/12；Gold 排队资格为 false，原满分不再被用来支持语义可靠性；
- [x] 全量回归 621 tests + 81 subtests 通过；定向 Ruff、JSON 和 diff 校验通过；未执行 Translator、Runtime 或第三方脚本；
- [ ] 清理用例中“生成某类候选”等元任务措辞，审查窄操作族是否忠实代表源 Skill；
- [ ] 构造与模型输出隔离的参考答案，再进行真正的 Gold-blind Translator 评测；没有人工时 AI 角色结果始终标记模拟证据。

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

### Current phase

**Active phase: L1-to-L0 translation generalization gate.** ES-P0 Runtime mechanics and small-sample wiring remain `local_hypothesis_supported`, but they are not broad translation evidence. No scaled Runtime study is meaningful until the Translator passes post-freeze, cross-Skill, cross-repository, and cross-domain unseen cohorts.

On 2026-09-04, an audit found that legacy answer-hidden reviews still exposed category-bearing IDs and fixed task order. Those results remain sealed but are downgraded to metadata-cued development diagnostics. A new opaque-ID, independently called single-task protocol records model-visible inputs, keeps mappings scorer-side, and validates resume/output bindings. Revalidation on the same 9B and development-07 packets completed 12/12 calls, but reached only 7/12 behavior agreement and 10/12 construct alignment, with Gold-queue eligibility false. Natural-task construct quality and evidence-grounded isolated reference answers now precede Translator evaluation; AI role simulation never becomes human-independent evidence.

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
