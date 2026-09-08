# 文档导航 / Documentation Map

## 中文

### 阅读路径

首先阅读 [EnsuredSkill 原型权威准则](ENSUREDSKILL-PROTOTYPE.md)。它覆盖与其冲突的历史阶段计划和生产工程设计。

| 你想回答的问题 | 首选文档 |
|---|---|
| 当前项目以什么理念、边界和完成判据为准 | [EnsuredSkill 原型权威准则](ENSUREDSKILL-PROTOTYPE.md) |
| 为什么必须先证明 L1→L0，门禁和指标是什么 | [L1→L0 泛化门禁](TRANSLATION-GENERALIZATION-GATE.md) |
| 项目是什么、有什么能力、如何开始 | [README](../README.md) |
| L1、L0.5、L0、Runtime 怎样交互，结果怎样解释和定位 | [Skill 与系统交互全景](SKILL-SYSTEM-INTERACTION.md) |
| 当前做到哪一步、还缺什么 | [项目进展与路线图](PROJECT-STATUS.md) |
| ES-P0 后遵循什么研究原则、按什么顺序推进 | [后续研究与研发指导 v1.1](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md) |
| 当前论文如何陈述系统与证据 | [中文论文 v0.4](research/EnsuredSkill_Paper_Draft_CN_v0.4_2026-09-01.md) / [English paper v0.4](research/EnsuredSkill_Paper_Draft_v0.4_2026-09-01.md) |
| 分层、依赖规则和架构决策 | [ARCHITECTURE](../ARCHITECTURE.md) |
| 系统组件、部署和端到端数据流 | [HLD](../HLD.md) |
| 接口、合同、状态机和异常处理 | [LLD](../LLD.md) |
| 威胁模型、安全控制和验收门禁 | [SSD](../SSD.md) |
| 两项核心能力到底达到什么效果 | [ES-P0 本地证据报告](ES-P0-EVIDENCE.md) |
| 公开 Skill 的角色隔离模拟效果与边界 | [ES-P1-Wild 角色隔离模拟结果](ES-P1-WILD-SIMULATED-RESULTS.md) |
| 没有独立团队时如何自动生成、封存并接入合成用例 | [仓库外合成 Holdout](SYNTHETIC-HOLDOUT.md) |
| 如何把 SkillsMP/GitHub 公开 Skill 安全纳入外部评测 | [ES-P1 公开 Skill 市场语料](ES-P1-PUBLIC-SKILL-CORPUS.md) |
| 如何构造与审查语义对齐的 Skill–Task–Tool 用例 | [转译用例构造与语义对齐](TRANSLATION-CASE-AUTHORING.md) |
| 为什么结构通过仍不代表转译准确，当前用例有哪些问题 | [转译测试构造质量](TRANSLATION-CONSTRUCT-QUALITY.md) |
| 参数、类型、必填性和验证/回滚步骤有什么源证据 | [转译源证据对齐](TRANSLATION-SOURCE-ALIGNMENT.md) |
| 本轮纠偏修改了什么，距离真正转译闭环还差什么 | [转译纠偏与闭环计划](TRANSLATION-CORRECTION-PLAN.md) |
| 有源工具合同后，如何让 9B 只生成适用任务文本 | [合同优先任务构造](CONTRACT-FIRST-TASKS.md) |
| 无参/可选参数的只读操作如何编译为未激活 L0 | [只读 L0 合同](L0-READ-CONTRACTS.md) |
| 公开 Skill 小批缺什么，为什么不能直接套单次读取 | [异质 Skill 边界小批](TRANSLATION-BOUNDARY-PILOT.md) |
| 条件分支与步骤输出引用如何实际执行，写路径还缺什么 | [最小业务流程](L0-BUSINESS-FLOW.md) |
| 9B 能否正向生成整个流程，原始失败与辅助修订如何区分 | [整流程正向转译实验](FLOW-FORWARD-TRANSLATION.md) |
| 固定协议后，顺序/分支/依赖缺失有哪些真实失败 | [C3b 冻结开发小批](FLOW-FROZEN-DEVELOPMENT.md) |
| 原文引用和编号能否解决语义错误，下一步如何改 | [C3c 源文约束与保真诊断](FLOW-SOURCE-GROUNDING.md) |
| 如何用层级流程表达代替模型手写节点和连线 | [C3d 层级流程编译](FLOW-TREE-COMPILER.md) |
| 9B 不看答案能否生成层级流程，失败在哪里 | [C3d 正向小批与原始失败](FLOW-TREE-FORWARD-PILOT.md) |
| 如何区分格式协议错误与业务转译错误 | [C3e 解码探针与宿主能力收口](FLOW-TREE-PROTOCOL-CANARY.md) |
| 新版不看答案的转译是否改善，为什么仍被阻断 | [C3f 正向结果与语义缺口](FLOW-BOUNDED-FORWARD.md) |
| 如何避免遗漏限制和猜节点，映射正确是否代表已经执行 | [C3g 源账本与受限第二步映射](FLOW-SOURCE-CONSTRAINTS.md) |
| 两次生成如何完整接起来，如何追溯到具体片段及统计全部成本 | [C3h 完整双阶段转译](FLOW-TWO-PASS-TRANSLATION.md) |
| 精简模型输出后节省多少，哪些语义错误仍存在 | [C3h 精简映射实测](FLOW-LEAN-MAPPING.md) |
| 如何区分业务前置、权限与校验，定位复合要求遗漏 | [C3h 源要求与保障职责](FLOW-RESPONSIBILITY-MAPPING.md) |
| 职责类型加入真实 9B 后效果如何，模型和协议各有什么问题 | [C3h 职责映射真实验证](FLOW-RESPONSIBILITY-PILOT.md) |
| 如何消除节点别名歧义、补齐终态，协议探针证明了什么 | [C3h 统一节点与受约束映射](FLOW-CANONICAL-MAPPING.md) |
| 统一节点进入完整双阶段后，局部修复是否带来闭环 | [C3h 统一节点完整双阶段验证](FLOW-CANONICAL-PILOT.md) |
| 转译失败是表达不足、语义错误还是证据遗漏，如何离线定位 | [分层诊断与命令](FLOW-DIAGNOSTICS.md) / [四例可读报告](benchmarks/flow-diagnostics-c3h-report.md) |
| 每节点必填是否会强迫伪造支持，新一轮结果如何 | [节点证据/反驳、真实验证与方案纠偏](FLOW-NODE-EVIDENCE.md) |
| 如何分开原文要求与宿主规则，新分层完成到哪里 | [原文义务、绑定接口与四例 9B 实测](FLOW-SOURCE-DUTIES.md) |
| 参数/必要条件怎样修复，33/33 行为匹配为什么不等于完整语义通过 | [转译行为闭环、必要条件与可行性分离](FLOW-BEHAVIOR-REPAIR.md) |
| 继续 9B 后实际结果如何，下一步如何调整 | [9B 单模型诊断、成本与阶段职责](FLOW-9B-COMMON-JSON.md) |
| 后续如何进行 GPT/9B 对照，哪些因素不能混算 | [暂缓的强模型对照设计、运行方式与边界](FLOW-MODEL-COMPARISON.md) |
| 如何本地演示或接入自己的系统 | [使用与系统接入](getting-started-integration.md) |

### 按主题查找

#### Agent、L1 与模型

- [Skill 与系统交互全景](SKILL-SYSTEM-INTERACTION.md)：离线 authoring、在线执行、路由分支、终态和证据定位的统一入口。
- [真实 LLM Agent 用例](AGENTIZED-USE-CASES.md)：DSH 页面 Prompt、Tool 链和外部 MCP 交互。
- [L1 模型资格](l1-model-qualification.md)：固定评测集、模型门槛和解释边界。
- [LLM 收敛评测](convergence-evaluation.md)：已解决、部分解决和未解决的问题。

#### L0 与 Promotion

- [L0 v2 设计](l0-v2-design.md)：原子、约束、扩展、组合合同。
- [L0 v2 Runtime 迁移](l0-v2-runtime-migration.md)：21 个激活合同和兼容边界。
- [L1 → L0 Promotion](l1-to-l0-promotion.md)：L1、L0.5、L0 三阶段编译与留痕。
- [L1→L0 泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)：当前优先级、100-Skill 开发库、独立对齐审查、未知 cohort 指标和 Runtime 硬准入。
- [通用渐进式确定化](progressive-determinization.md)：跨域边界、Anthropic Skill 包和风险路由。
- [真实 Harness 自动 Runtime A/B](general-effect-ab.md)：原生 Agent 仅作为隔离 Control；Treatment 的不合格转换安全停机；旧单次工具基线已降级。
- [ES-P0 本地证据报告](ES-P0-EVIDENCE.md)：六场景、消融、9B/7B 三次配对、性能和外推边界。
- [仓库外合成 Holdout](SYNTHETIC-HOLDOUT.md)：240 条模型合成 Skill、双盲模型审阅、摘要封存、受控导入和正式 ES-P1 边界。
- [ES-P1 公开 Skill 市场语料](ES-P1-PUBLIC-SKILL-CORPUS.md)：SkillsMP/GitHub 公开生态的采样、封存、零执行隔离和证据边界。
- [转译用例构造与语义对齐](TRANSLATION-CASE-AUTHORING.md)：精确原文锚定、通用不可执行 Tool Catalog、确定性候选门禁和答案隐藏的审查队列。
- [ES-P1-Wild 角色隔离模拟结果](ES-P1-WILD-SIMULATED-RESULTS.md)：15 Skill、45 case、三重复真实 DSH 配对、分层指标、残余失败与非真人证据边界。
- [ES-P1-Wild pilot 摘要](benchmarks/es-p1-wild-pilot-summary.json)：20 个静态接纳包、15 包 author kit，以及 9B 非权威草案辅助的通过、修复、失败和时延结果。
- [ES-P1-Wild 测试 Skill 索引](benchmarks/es-p1-wild-skill-index.json)：15 个实际测试 Skill 的固定来源、commit、许可证、文件清单、任务槽位与草案状态。
- [真实 Harness 冒烟摘要](benchmarks/real-harness-smoke-summary.json)：历史 fallback evaluator 的 DSH 配对轨迹，仅作前期探索证据。
- [Promotion Workbench](p20-promotion-workbench.md)：并排语义审查、告警和离线编辑。
- [正向资格协议](promotion-forward-qualification.md)：Research Freeze、公开回归、私有正向用例、双盲审、重复运行和分层统计。
- [存量 L0 轨迹索引](../network_runtime/l0/production_trajectories/INDEX.md)：21/21 可读三阶段制品；目录名是历史兼容名称，不代表生产认证。

#### Runtime 与评测

- [Runtime 组件基线](benchmarks/runtime-ab-baseline.md)：Core-72 事务控制回归；不是原生 DSH/Hermes Agent 对比。
- [版本化 ES-P0 摘要](benchmarks/es-p0-evidence-summary.json)：可提交的最终指标与制品摘要。
- [后续研究与研发指导 v1.1](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md)：Claim Registry、ES-P1/P2 Gate、统计规范和冻结范围。
- [中文论文 v0.4](research/EnsuredSkill_Paper_Draft_CN_v0.4_2026-09-01.md) / [English paper v0.4](research/EnsuredSkill_Paper_Draft_v0.4_2026-09-01.md)：与最终 ES-P0 摘要对齐的双语研究稿。
- [历史双核心功能报告](core-capability-evaluation-report.md)：材料重构前的工程报告，仅作历史参考。

#### 冻结的未来工程参考

- [企业控制面](enterprise-control-plane.md)：OIDC、PDP、Change Authority 和 mTLS；`frozen_future_engineering`。
- [Provider 供应链](provider-supply-chain.md)：release、qualification、deployment 和 admission；`frozen_future_engineering`。
- [Capability Catalog 与 Evidence Plane](p21-p22-control-planes.md)：治理投影和统一证据；`frozen_future_engineering`。
- [L1 Decision Plane](l1-decision-plane.md)：历史 shadow/proposal-only 产品化实验；当前仅保留无执行权边界与回归参考。
- [P1.9 Canary 手册](p19-canary-runbook.md)：历史产品化准备；`frozen_future_engineering`。

#### 本地网络实验

- [FRR/OSPF 基础实验](../labs/p075-a-frr/README.md)
- [园区与 IDC 实验](../labs/p075-a-campus-idc/README.md)
- [典型小型现网](../labs/p075-b-small-production/README.md)
- [EVPN/VXLAN Fabric](../labs/p075-c-evpn-vxlan/README.md)

### 文档权威边界

- `README.md` 只维护项目设计、能力、优势、性能、场景和入口，不记录完整阶段历史。
- `SKILL-SYSTEM-INTERACTION.md` 是 Skill 层级、authoring/执行双生命周期、用户可见结果和解释路径的统一说明；其他设计文档引用而不另造口径。
- `docs/PROJECT-STATUS.md` 是 Done、To-do 和阶段边界的唯一汇总来源。
- `ARCHITECTURE.md`、`HLD.md`、`LLD.md`、`SSD.md` 分别拥有架构、组件、实现和安全设计事实；同一细节不在 README 重复展开。
- 自动生成的评测报告只由对应命令刷新，不能手工把固定集结果改写成生产概率。
- `artifacts/` 是运行证据和本地报告，不是源码设计文档；`data/` 中的基线与测试数据不能作为清理临时文件处理。
- 所有项目级文档采用中文在前、英文在后的同文档双语结构；面向独立投稿和逐段校对的中英文论文稿作为例外，保持两个结构一致的文件。

---

## English

[Translation behavior repair](FLOW-BEHAVIOR-REPAIR.md) is the current diagnosis/repair entry. Necessary-guard inference now separates forbidden counterfactuals from positive-path feasibility: the fixed candidate batch matches 33/33 inert scenarios after explicit postprocessing, while positive unknowns and full-source review remain open. First-pass, assisted results and costs remain separate; no complete-Skill/generalization or default Runtime admission claim is made.

[Source duties and host binding](FLOW-SOURCE-DUTIES.md) documents implemented source-only extraction, review and graph-binding interfaces, the initial four-case probe, and the subsequent six-case explicit-relation negative result. Free-text relations are not adopted as the default upgrade: regression/new-development source support is 0/4 and 0/2, with new scope/predecessor defects. Next investigate source-bound relation evidence; there is no fresh whole-chain model result.

[9B-only diagnostics](FLOW-9B-COMMON-JSON.md) record twelve completed calls, separate fixed-tree/end-to-end results, first-stage source reviews and costs. Both modes compile 0/4 mappings; common JSON is not adopted as a quality improvement. Next separate source duties from host binding while retaining constrained schemas. The [GPT/9B comparison design](FLOW-MODEL-COMPARISON.md) remains available but GPT is deferred by user choice, not a blocker for current work.

[Node evidence and residual source duties](FLOW-NODE-EVIDENCE.md) preserves explicit disagreement rather than forcing positive support. The fresh paired batch compiles 1/4 mappings but accepts none after applicable source review; it motivates source-first duty extraction separate from host guarantees.

[Layered translation diagnostics](FLOW-DIAGNOSTICS.md) explains aggregate mechanical checks, first-pass/complete source reviews, L0 projection comparison and separate assisted witnesses. The [four-case report](benchmarks/flow-diagnostics-c3h-report.md) links missing evidence to source and L0.5/L0 locations without rescoring the frozen batch or claiming semantic accuracy.

[Canonical-node fresh paired validation](FLOW-CANONICAL-PILOT.md) separates local Schema gains from still-unqualified complete mappings, with objective/node-coverage diagnostics and failure-inclusive costs.

[Canonical nodes and constrained mapping](FLOW-CANONICAL-MAPPING.md) repairs Schema compatibility and terminal representation, with three real 9B decoder probes distinguished from whole-Skill validation.

[Real responsibility-mapping validation](FLOW-RESPONSIBILITY-PILOT.md) records the fresh 9B negative result, protocol/model failures, cost changes and unchanged evidence boundaries.

[Source requirements and guarantee responsibilities](FLOW-RESPONSIBILITY-MAPPING.md) explains atomic source quotes, necessary type/target compatibility, complete semantic review, offline usage and the still-unmeasured model benefit.

[C3h lean mapping results](FLOW-LEAN-MAPPING.md) separate output savings from increased input burden, preserve complete semantic failures, and explain compiler-owned anchors without removing review obligations.

[C3h fresh two-pass translation](FLOW-TWO-PASS-TRANSLATION.md) connects source-to-flow and exact-offset clause mapping, with per-phase checkpoints, multi-source review and costs that include failures. This is not a product activation or generalization claim.

[C3g source ledgers and restricted mapping](FLOW-SOURCE-CONSTRAINTS.md) separate retention, mapping and enforcement; preserve three frozen 9B batches, semantic failures and auxiliary costs; and explain why qualified metadata or faithful missing-capability stops remain blocked.

[C3f answer-free forward results](FLOW-BOUNDED-FORWARD.md) distinguish observed structural gains on four known flows from still-incomplete restriction/citation fidelity; no Runtime execution or generalization claim.

[C3e protocol canaries](FLOW-TREE-PROTOCOL-CANARY.md) distinguish explicit-answer constructor checks from source translation and explain host-derived generation constraints. New answer-free validation remains pending.

[C3d forward 9B pilot](FLOW-TREE-FORWARD-PILOT.md) preserves all four first attempts, qualification failures, costs and the next protocol-compatibility checks. Zero qualified flows is not a Runtime performance result.

[C3d hierarchical flow compilation](FLOW-TREE-COMPILER.md) covers tree syntax, compiler-owned wiring, lexical reference limits and offline read-Runtime verification.

[C3c source-grounding diagnostics](FLOW-SOURCE-GROUNDING.md) explain source IDs, dependency/polarity review, actual 9B failures and the remaining quality gap.

For the eight known development flows, frozen input/protocol boundaries and preserved 9B failures, see [C3b frozen flow development](FLOW-FROZEN-DEVELOPMENT.md). These are not public-Skill generalization results.

### Reading path

| Question | Primary document |
|---|---|
| What is the project and how do I start? | [README](../README.md) |
| Why must L1-to-L0 be proven first, and what unlocks Runtime evaluation? | [L1-to-L0 generalization gate](TRANSLATION-GENERALIZATION-GATE.md) |
| How do L1, L0.5, L0, and the Runtime interact, and how are outcomes explained? | [Skill-to-system interaction](SKILL-SYSTEM-INTERACTION.md) |
| What is complete and what remains? | [Project status and roadmap](PROJECT-STATUS.md) |
| What governs post-ES-P0 research? | [Research instruction v1.1](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md) |
| How does the current paper state the system and evidence? | [English paper v0.4](research/EnsuredSkill_Paper_Draft_v0.4_2026-09-01.md) / [Chinese paper v0.4](research/EnsuredSkill_Paper_Draft_CN_v0.4_2026-09-01.md) |
| What are the layers, dependency rules, and ADRs? | [ARCHITECTURE](../ARCHITECTURE.md) |
| What are the components, deployments, and end-to-end flows? | [HLD](../HLD.md) |
| What are the interfaces, contracts, states, and failure paths? | [LLD](../LLD.md) |
| What is the threat model and acceptance gate? | [SSD](../SSD.md) |
| What evidence supports the two core capabilities? | [ES-P0 local evidence](ES-P0-EVIDENCE.md) |
| What did the role-separated public-Skill simulation show? | [ES-P1-Wild simulation results](ES-P1-WILD-SIMULATED-RESULTS.md) |
| How can I generate and seal synthetic cases before an independent ES-P1 study? | [Repository-external synthetic holdout](SYNTHETIC-HOLDOUT.md) |
| How can public SkillsMP/GitHub Skills be evaluated safely? | [ES-P1 public Skill-market corpus](ES-P1-PUBLIC-SKILL-CORPUS.md) |
| How are semantically aligned Skill–Task–Tool cases authored and reviewed? | [Translation case authoring and alignment](TRANSLATION-CASE-AUTHORING.md) |
| Why can a valid structure still misrepresent a Skill? | [Translation construct quality](TRANSLATION-CONSTRUCT-QUALITY.md) |
| What source evidence supports each parameter and execution step? | [Source evidence alignment](TRANSLATION-SOURCE-ALIGNMENT.md) |
| What changed and what remains before translation closure? | [Translation correction plan](TRANSLATION-CORRECTION-PLAN.md) |
| How does a source-backed contract constrain 9B task authoring? | [Contract-first task authoring](CONTRACT-FIRST-TASKS.md) |
| How are zero/optional-input reads compiled into inactive L0 contracts? | [Read-only L0 contracts](L0-READ-CONTRACTS.md) |
| What prevents heterogeneous public Skills from using a single-read path? | [Skill boundary pilot](TRANSLATION-BOUNDARY-PILOT.md) |
| How do branches and step-output references run, and what remains for writes? | [Minimal business flow](L0-BUSINESS-FLOW.md) |
| What did 9B whole-flow generation achieve unaided versus with revision? | [Forward flow experiment](FLOW-FORWARD-TRANSLATION.md) |
| How do I run a demo or integrate my systems? | [Usage and integration](getting-started-integration.md) |

### Topic index

- Agent and L1: [Skill-to-system interaction](SKILL-SYSTEM-INTERACTION.md), [Agent use cases](AGENTIZED-USE-CASES.md), [model qualification](l1-model-qualification.md), and [convergence evaluation](convergence-evaluation.md).
- L0 and Promotion: [L1-to-L0 generalization gate](TRANSLATION-GENERALIZATION-GATE.md), [translation case authoring and alignment](TRANSLATION-CASE-AUTHORING.md), [general progressive determinization](progressive-determinization.md), [general Effect Runtime A/B](general-effect-ab.md), [L0 v2 design](l0-v2-design.md), [Runtime migration](l0-v2-runtime-migration.md), [L1-to-L0 Promotion](l1-to-l0-promotion.md), [Workbench](p20-promotion-workbench.md), [Research Freeze and forward qualification](promotion-forward-qualification.md), and the [production trajectory index](../network_runtime/l0/production_trajectories/INDEX.md).
- Runtime evaluation: [ES-P0 local evidence](ES-P0-EVIDENCE.md), [ES-P1-Wild role-separated simulation](ES-P1-WILD-SIMULATED-RESULTS.md), the [synthetic holdout workflow](SYNTHETIC-HOLDOUT.md), its [versioned summary](benchmarks/es-p0-evidence-summary.json), the [research instruction](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md), the [English paper](research/EnsuredSkill_Paper_Draft_v0.4_2026-09-01.md), the [Chinese paper](research/EnsuredSkill_Paper_Draft_CN_v0.4_2026-09-01.md), the [paired protocol](general-effect-ab.md), and the component-only [Runtime A/B](benchmarks/runtime-ab-baseline.md).
- Frozen future-engineering reference: [enterprise control plane](enterprise-control-plane.md), [provider supply chain](provider-supply-chain.md), [Capability Catalog/Evidence Plane](p21-p22-control-planes.md), the historical [L1 Decision Plane](l1-decision-plane.md), and the [P1.9 canary runbook](p19-canary-runbook.md).
- Local labs: [FRR/OSPF](../labs/p075-a-frr/README.md), [campus/IDC](../labs/p075-a-campus-idc/README.md), [small production network](../labs/p075-b-small-production/README.md), and [EVPN/VXLAN](../labs/p075-c-evpn-vxlan/README.md).

### Source-of-truth rules

`README.md` is the concise product entry. `SKILL-SYSTEM-INTERACTION.md` is the single lifecycle and outcome-explanation guide. `docs/PROJECT-STATUS.md` is the only phase summary. The architecture, HLD, LLD, and SSD own their respective design facts. Generated evaluation reports must be refreshed by their commands and must not convert fixed-set evidence into a production probability. `artifacts/` contains run evidence, while versioned `data/` files are baselines or test inputs rather than disposable output. Project-level documents keep Chinese before English; the structurally matched Chinese and English submission drafts remain separate for independent editing and review.
