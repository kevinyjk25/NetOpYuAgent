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

### 下一决策门

当前不应继续无边界扩展生产工程。若继续研究，优先级是：

1. 建立仓库外封存测试集和独立评分角色，验证是否过拟合；
2. 在可获得的真实设备/控制器实验床上复用同一 Oracle；
3. 仅在上述证据支持后，决定是否恢复身份、供应链、HA/DR 和生产 SLO；
4. Typed Graph 单一调度器与 Experience Compilation 可作为独立研究支线，不改变当前安全边界。

## English

### Current phase

**ES-P0 local research-prototype evidence is complete.** The aggregate result is `local_hypothesis_supported`: reasoning proposes candidates, while Contract, Evidence, Guard, Risk, and Transaction control effects; an unqualified translation stops safely and never regains native write authority.

Completed evidence includes 60 deterministic Runtime scenario runs, 30 mechanism-ablation probes, two 60-Skill translation studies, 120 real paired DSH sessions across 9B and 7B models, precision/coverage curves, scenario-level Wilson intervals, and a final regression of 506 tests plus 81 subtests.

For 9B, treatment reached 86.67% task completion and 100% execution precision with zero unsafe execution, false commits, or invalid actions. The weaker 7B treatment also kept those action-safety metrics at zero, but 18/30 sessions failed at the DSH/model availability layer and the model is therefore not availability-qualified.

This is local transparent-development evidence—not production probability, hidden-set generalization, or vendor-device certification. Open work includes an external sealed/private set, real-device/controller qualification, complete graph-driven scheduling, cross-step provenance DAGs, and trace-based Experience Compilation. Production identity, supply chain, governance, HA/DR, WORM audit, and SLO engineering remain frozen until stronger evidence justifies them.
