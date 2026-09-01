# EnsuredSkill 项目进展 / Project Status

## 中文

### 当前阶段

**ES-P0：本地研究原型证据闭环已完成。**

权威命题是：概率性 Reasoning 只提出 Candidate Plan；Contract、Evidence、Guard、Risk 和 Transaction 决定是否允许 Effect；不合格转换安全停机，不能回退原生写。

聚合结论为 `local_hypothesis_supported`。它只适用于透明开发集和本地仿真 Provider，不是生产成功概率、隐藏集泛化或真实厂商设备资格。

### Done

#### 架构与执行边界

- [x] Reasoning / Reliability Runtime / Infrastructure 三平面成为唯一权威架构；
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
| Typed Graph | 已生成、校验并绑定计划 | legacy engine 全部迁移到单一图调度器 |
| Provenance | 单证据与计划绑定完成 | Evidence→Observation→Tool→Object 跨步骤 DAG |
| Experience Compilation | authoring compilation 已完成 | 基于长期成功轨迹的独立研究协议 |
| 生产工程 | 冻结 | 完成独立泛化和真实设备证据后再决策 |

### 当前活动阶段与后续计划

**当前活动阶段：ES-P1 Independent Generalization 基础设施已就绪，等待仓库外独立数据。**

#### ES-P1 已完成的研究基础设施

- [x] `Research Freeze v1` 联合绑定 Git commit/clean state、Runtime kernel、Harness boundary、21 个 Contract/trajectory、Evaluator、authoring protocol、模型制品、依赖环境与 ES-P0 基线；脏工作树只能生成不可注册的 preview；
- [x] `Study Plan v2` 与 `Private Manifest v3` 强制绑定已验证 `freezeDigest`，正式 CLI 拒绝 preview、漂移或篡改的 Freeze；
- [x] 仓库外工作区把 Research Freeze、Case Author、两个 Reviewer、Adjudicator、密封数据和模型运行分成独立 Gate；Doctor 不输出 Prompt、Label、语义合同或参数值；
- [x] `Qualification Report v2` 增加 family/profile/language/challenge/expected-disposition/risk 分层，Wilson 95% 区间、零事件单侧 95% 上界和互斥结果分类；
- [x] safety escape 拆为 critical semantic、undeclared effect、approval/risk weakening 三类，三类均为零才可能通过资格门；
- [x] Freeze、预注册、密封、盲审/仲裁、重复评分和篡改检查的相关回归为 `29 passed`。

这些完成项证明“实验可以按预注册方式运行且不会悄悄漂移”，不证明隐藏集泛化。ES-P1 的核心未完成项仍是由 Runtime 团队之外的人在仓库外编写和审阅真实 private holdout。

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
2. 由 Runtime 团队之外的 Case Author 正向构造 200–500 条 private holdout；
3. 两名独立 Reviewer 盲审，Adjudicator 只处理摘要绑定的分歧；
4. 在模型运行前冻结排除/中止规则、paired statistics 和 latency breakdown；
5. 对同一制品至少重复三次并生成只含聚合值的 Qualification Report v2；
6. 若出现 critical escape，先定位抽象缺陷并创建新研究版本，不针对单 case 打补丁；
7. ES-P1 通过后进入小范围 ES-P2，不追求厂商 breadth；
8. 只有 ES-P1/ES-P2 证据支持后，才重新评估身份、供应链、HA/DR、WORM 和生产 SLO。

详细原则、指标 Gate、角色边界和任务模板见[后续研究与研发指导 v1.1](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md)。

## English

### Current phase

**ES-P0 local research-prototype evidence is complete.** The aggregate result is `local_hypothesis_supported`: reasoning proposes candidates, while Contract, Evidence, Guard, Risk, and Transaction control effects; an unqualified translation stops safely and never regains native write authority.

Completed evidence includes 60 deterministic Runtime scenario runs, 30 mechanism-ablation probes, two 60-Skill translation studies, 120 real paired DSH sessions across 9B and 7B models, precision/coverage curves, scenario-level Wilson intervals, and a final regression of 506 tests plus 81 subtests.

For 9B, treatment reached 86.67% task completion and 100% execution precision with zero unsafe execution, false commits, or invalid actions. The weaker 7B treatment also kept those action-safety metrics at zero, but 18/30 sessions failed at the DSH/model availability layer and the model is therefore not availability-qualified.

This is local transparent-development evidence—not production probability, hidden-set generalization, or vendor-device certification.

The ES-P1 research infrastructure is now ready, but independent-generalization evidence has not yet been collected. Research Freeze v1 jointly binds Git cleanliness, Runtime, harness boundary, 21 contracts and readable trajectories, evaluator, authoring protocol, model artifact, environment, and the ES-P0 baseline. Study Plan v2 and private Manifest v3 require that verified freeze digest. Qualification Report v2 adds family/profile/risk/disposition slices, Wilson intervals, zero-event upper bounds, mutually exclusive outcomes, and separate critical-semantic, undeclared-effect, and approval/risk-weakening escapes. The focused freeze/qualification regression is 29/29 passing.

These controls prove that an external study can be preregistered and drift-checked; they do not prove hidden-set generalization. The remaining ES-P1 work must be performed outside the Runtime team: independently author 200–500 private cases, obtain two blind reviews and bound adjudication, freeze exclusions/stopping rules, run the same 9B artifact at least three times, and publish only aggregate Qualification Report v2 evidence. The next semantic work is a single Typed Graph scheduler, a cross-step provenance DAG, and stage-level latency instrumentation without changing a frozen experimental version.

Only after ES-P1 passes will ES-P2 qualify the same abstractions on at least one real router and one controller or management path. ES-P3 scales the evidence and packages a paper-grade artifact. Trace-based Experience Compilation is ES-P4, a separate research line with no automatic activation. Production identity, supply chain, governance, HA/DR, WORM audit, and SLO engineering remain frozen until independent and real-network evidence justifies them. See the [v1.1 research instruction](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md).
