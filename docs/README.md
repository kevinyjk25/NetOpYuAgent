# 文档导航 / Documentation Map

## 中文

### 阅读路径

首先阅读 [EnsuredSkill 原型权威准则](ENSUREDSKILL-PROTOTYPE.md)。它覆盖与其冲突的历史阶段计划和生产工程设计。

| 你想回答的问题 | 首选文档 |
|---|---|
| 当前项目以什么理念、边界和完成判据为准 | [EnsuredSkill 原型权威准则](ENSUREDSKILL-PROTOTYPE.md) |
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
- [通用渐进式确定化](progressive-determinization.md)：跨域边界、Anthropic Skill 包和风险路由。
- [真实 Harness 自动 Runtime A/B](general-effect-ab.md)：原生 Agent 仅作为隔离 Control；Treatment 的不合格转换安全停机；旧单次工具基线已降级。
- [ES-P0 本地证据报告](ES-P0-EVIDENCE.md)：六场景、消融、9B/7B 三次配对、性能和外推边界。
- [仓库外合成 Holdout](SYNTHETIC-HOLDOUT.md)：240 条模型合成 Skill、双盲模型审阅、摘要封存、受控导入和正式 ES-P1 边界。
- [ES-P1 公开 Skill 市场语料](ES-P1-PUBLIC-SKILL-CORPUS.md)：SkillsMP/GitHub 公开生态的采样、封存、零执行隔离和证据边界。
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

### Reading path

| Question | Primary document |
|---|---|
| What is the project and how do I start? | [README](../README.md) |
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
| How do I run a demo or integrate my systems? | [Usage and integration](getting-started-integration.md) |

### Topic index

- Agent and L1: [Skill-to-system interaction](SKILL-SYSTEM-INTERACTION.md), [Agent use cases](AGENTIZED-USE-CASES.md), [model qualification](l1-model-qualification.md), and [convergence evaluation](convergence-evaluation.md).
- L0 and Promotion: [general progressive determinization](progressive-determinization.md), [general Effect Runtime A/B](general-effect-ab.md), [L0 v2 design](l0-v2-design.md), [Runtime migration](l0-v2-runtime-migration.md), [L1-to-L0 Promotion](l1-to-l0-promotion.md), [Workbench](p20-promotion-workbench.md), [Research Freeze and forward qualification](promotion-forward-qualification.md), and the [production trajectory index](../network_runtime/l0/production_trajectories/INDEX.md).
- Runtime evaluation: [ES-P0 local evidence](ES-P0-EVIDENCE.md), [ES-P1-Wild role-separated simulation](ES-P1-WILD-SIMULATED-RESULTS.md), the [synthetic holdout workflow](SYNTHETIC-HOLDOUT.md), its [versioned summary](benchmarks/es-p0-evidence-summary.json), the [research instruction](research/EnsuredSkill_Research_Instruction_v1.1_2026-09-01.md), the [English paper](research/EnsuredSkill_Paper_Draft_v0.4_2026-09-01.md), the [Chinese paper](research/EnsuredSkill_Paper_Draft_CN_v0.4_2026-09-01.md), the [paired protocol](general-effect-ab.md), and the component-only [Runtime A/B](benchmarks/runtime-ab-baseline.md).
- Frozen future-engineering reference: [enterprise control plane](enterprise-control-plane.md), [provider supply chain](provider-supply-chain.md), [Capability Catalog/Evidence Plane](p21-p22-control-planes.md), the historical [L1 Decision Plane](l1-decision-plane.md), and the [P1.9 canary runbook](p19-canary-runbook.md).
- Local labs: [FRR/OSPF](../labs/p075-a-frr/README.md), [campus/IDC](../labs/p075-a-campus-idc/README.md), [small production network](../labs/p075-b-small-production/README.md), and [EVPN/VXLAN](../labs/p075-c-evpn-vxlan/README.md).

### Source-of-truth rules

`README.md` is the concise product entry. `SKILL-SYSTEM-INTERACTION.md` is the single lifecycle and outcome-explanation guide. `docs/PROJECT-STATUS.md` is the only phase summary. The architecture, HLD, LLD, and SSD own their respective design facts. Generated evaluation reports must be refreshed by their commands and must not convert fixed-set evidence into a production probability. `artifacts/` contains run evidence, while versioned `data/` files are baselines or test inputs rather than disposable output. Project-level documents keep Chinese before English; the structurally matched Chinese and English submission drafts remain separate for independent editing and review.
