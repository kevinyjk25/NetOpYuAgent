# 文档导航 / Documentation Map

## 中文

### 阅读路径

首先阅读 [EnsuredSkill 原型权威准则](ENSUREDSKILL-PROTOTYPE.md)。它覆盖与其冲突的历史阶段计划和生产工程设计。

| 你想回答的问题 | 首选文档 |
|---|---|
| 当前项目以什么理念、边界和完成判据为准 | [EnsuredSkill 原型权威准则](ENSUREDSKILL-PROTOTYPE.md) |
| 项目是什么、有什么能力、如何开始 | [README](../README.md) |
| 当前做到哪一步、还缺什么 | [项目进展与路线图](PROJECT-STATUS.md) |
| 分层、依赖规则和架构决策 | [ARCHITECTURE](../ARCHITECTURE.md) |
| 系统组件、部署和端到端数据流 | [HLD](../HLD.md) |
| 接口、合同、状态机和异常处理 | [LLD](../LLD.md) |
| 威胁模型、安全控制和验收门禁 | [SSD](../SSD.md) |
| 两项核心能力到底达到什么效果 | [ES-P0 本地证据报告](ES-P0-EVIDENCE.md) |
| 如何本地演示或接入自己的系统 | [使用与系统接入](getting-started-integration.md) |

### 按主题查找

#### Agent、L1 与模型

- [真实 LLM Agent 用例](AGENTIZED-USE-CASES.md)：DSH 页面 Prompt、Tool 链和外部 MCP 交互。
- [L1 模型资格](l1-model-qualification.md)：固定评测集、模型门槛和解释边界。
- [L1 Decision Plane](l1-decision-plane.md)：shadow、proposal-only、绑定与证据。
- [P1.9 Canary 手册](p19-canary-runbook.md)：证据门禁、停用和回退。
- [LLM 收敛评测](convergence-evaluation.md)：已解决、部分解决和未解决的问题。

#### L0 与 Promotion

- [L0 v2 设计](l0-v2-design.md)：原子、约束、扩展、组合合同。
- [L0 v2 Runtime 迁移](l0-v2-runtime-migration.md)：21 个激活合同和兼容边界。
- [L1 → L0 Promotion](l1-to-l0-promotion.md)：L1、L0.5、L0 三阶段编译与留痕。
- [通用渐进式确定化](progressive-determinization.md)：跨域边界、Anthropic Skill 包和风险路由。
- [真实 Harness 自动 Runtime A/B](general-effect-ab.md)：原生 Agent 仅作为隔离 Control；Treatment 的不合格转换安全停机；旧单次工具基线已降级。
- [ES-P0 本地证据报告](ES-P0-EVIDENCE.md)：六场景、消融、9B/7B 三次配对、性能和外推边界。
- [真实 Harness 冒烟摘要](benchmarks/real-harness-smoke-summary.json)：历史 fallback evaluator 的 DSH 配对轨迹，仅作前期探索证据。
- [Promotion Workbench](p20-promotion-workbench.md)：并排语义审查、告警和离线编辑。
- [正向资格协议](promotion-forward-qualification.md)：公开回归、私有正向用例、双盲审和重复运行。
- [生产 L0 轨迹索引](../network_runtime/l0/production_trajectories/INDEX.md)：21/21 可读三阶段制品。

#### Runtime 与评测

- [Runtime 组件基线](benchmarks/runtime-ab-baseline.md)：Core-72 事务控制回归；不是原生 DSH/Hermes Agent 对比。
- [版本化 ES-P0 摘要](benchmarks/es-p0-evidence-summary.json)：可提交的最终指标与制品摘要。
- [历史双核心功能报告](core-capability-evaluation-report.md)：材料重构前的工程报告，仅作历史参考。

#### 冻结的未来工程参考

- [企业控制面](enterprise-control-plane.md)：OIDC、PDP、Change Authority 和 mTLS；`frozen_future_engineering`。
- [Provider 供应链](provider-supply-chain.md)：release、qualification、deployment 和 admission；`frozen_future_engineering`。
- [Capability Catalog 与 Evidence Plane](p21-p22-control-planes.md)：治理投影和统一证据；`frozen_future_engineering`。
- [P1.9 Canary 手册](p19-canary-runbook.md)：历史产品化准备；`frozen_future_engineering`。

#### 本地网络实验

- [FRR/OSPF 基础实验](../labs/p075-a-frr/README.md)
- [园区与 IDC 实验](../labs/p075-a-campus-idc/README.md)
- [典型小型现网](../labs/p075-b-small-production/README.md)
- [EVPN/VXLAN Fabric](../labs/p075-c-evpn-vxlan/README.md)

### 文档权威边界

- `README.md` 只维护项目设计、能力、优势、性能、场景和入口，不记录完整阶段历史。
- `docs/PROJECT-STATUS.md` 是 Done、To-do 和阶段边界的唯一汇总来源。
- `ARCHITECTURE.md`、`HLD.md`、`LLD.md`、`SSD.md` 分别拥有架构、组件、实现和安全设计事实；同一细节不在 README 重复展开。
- 自动生成的评测报告只由对应命令刷新，不能手工把固定集结果改写成生产概率。
- `artifacts/` 是运行证据和本地报告，不是源码设计文档；`data/` 中的基线与测试数据不能作为清理临时文件处理。
- 所有项目级文档采用中文在前、英文在后的同文档双语结构。

---

## English

### Reading path

| Question | Primary document |
|---|---|
| What is the project and how do I start? | [README](../README.md) |
| What is complete and what remains? | [Project status and roadmap](PROJECT-STATUS.md) |
| What are the layers, dependency rules, and ADRs? | [ARCHITECTURE](../ARCHITECTURE.md) |
| What are the components, deployments, and end-to-end flows? | [HLD](../HLD.md) |
| What are the interfaces, contracts, states, and failure paths? | [LLD](../LLD.md) |
| What is the threat model and acceptance gate? | [SSD](../SSD.md) |
| What evidence supports the two core capabilities? | [ES-P0 local evidence](ES-P0-EVIDENCE.md) |
| How do I run a demo or integrate my systems? | [Usage and integration](getting-started-integration.md) |

### Topic index

- Agent and L1: [Agent use cases](AGENTIZED-USE-CASES.md), [model qualification](l1-model-qualification.md), [Decision Plane](l1-decision-plane.md), [canary runbook](p19-canary-runbook.md), and [convergence evaluation](convergence-evaluation.md).
- L0 and Promotion: [general progressive determinization](progressive-determinization.md), [general Effect Runtime A/B](general-effect-ab.md), [L0 v2 design](l0-v2-design.md), [Runtime migration](l0-v2-runtime-migration.md), [L1-to-L0 Promotion](l1-to-l0-promotion.md), [Workbench](p20-promotion-workbench.md), [forward qualification](promotion-forward-qualification.md), and the [production trajectory index](../network_runtime/l0/production_trajectories/INDEX.md).
- Runtime evaluation: [ES-P0 local evidence](ES-P0-EVIDENCE.md), its [versioned summary](benchmarks/es-p0-evidence-summary.json), the [paired protocol](general-effect-ab.md), and the component-only [Runtime A/B](benchmarks/runtime-ab-baseline.md).
- Frozen future-engineering reference: [enterprise control plane](enterprise-control-plane.md), [provider supply chain](provider-supply-chain.md), [Capability Catalog/Evidence Plane](p21-p22-control-planes.md), and the [P1.9 canary runbook](p19-canary-runbook.md).
- Local labs: [FRR/OSPF](../labs/p075-a-frr/README.md), [campus/IDC](../labs/p075-a-campus-idc/README.md), [small production network](../labs/p075-b-small-production/README.md), and [EVPN/VXLAN](../labs/p075-c-evpn-vxlan/README.md).

### Source-of-truth rules

`README.md` is the concise product entry. `docs/PROJECT-STATUS.md` is the only phase summary. The architecture, HLD, LLD, and SSD own their respective design facts. Generated evaluation reports must be refreshed by their commands and must not convert fixed-set evidence into a production probability. `artifacts/` contains run evidence, while versioned `data/` files are baselines or test inputs rather than disposable output. Project-level documents keep Chinese before English.
