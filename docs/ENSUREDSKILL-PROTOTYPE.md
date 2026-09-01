# EnsuredSkill 原型权威准则 / Authoritative Prototype Charter

## 中文

### 1. 文档地位

本文档依据《EnsuredSkill：面向自治网络智能体的可靠执行运行时》确定当前项目的权威目标。若历史阶段文档、生产工程设计、评测口径或代码注释与本文冲突，以本文为准。

项目当前只建设和验证 **EnsuredSkill 网络可靠执行原型**。真实企业身份、Provider 供应链、多人治理、远端不可变审计、HA/DR、生产 SLO 和厂商设备认证全部后置；已有实现保留为冻结实验材料，不进入当前成功判据。

### 2. 核心命题

```text
Reasoning correctness does not imply execution safety.
LLM decides what to attempt; Runtime decides what is allowed to happen.
No evidence, no action.
```

项目必须将概率性推理和确定性执行分离。LLM、DSH 和 L1 Skill 只能产生诊断假设、任务意图和 Candidate Plan；它们不能获得基础设施直接写权限。

### 3. 三平面架构

```text
Reasoning Plane
  DSH + LLM + L1 Skill
  understand / diagnose / plan / propose
                    │ Candidate Plan
                    ▼
Reliability Runtime
  Contract / Typed Graph / Evidence / Guard / Risk
  Snapshot / Precheck / Execute / Verify / Compensate
                    │ Validated Operation
                    ▼
Infrastructure Plane
  Network and supporting service Providers
  MCP / API / CLI / NETCONF / Containerlab
```

所有现有模块必须映射到这三个平面。Decision Plane、Promotion、Catalog、Provider Adapter 和审批 Adapter 是内部组件或未来扩展，不是新的平级权威层。

### 4. 唯一效果边界

产品原型中，所有副作用必须经过已激活的 Contract-Governed L0 和 Reliability Runtime。L1 转换失败或置信度不足时，只允许：

- 只读诊断；
- 精确追问；
- 生成 proposal；
- 请求人工处理；
- 拒绝。

不得回退为 L1 直接写 Provider。原生 Agent 写路径只允许存在于隔离的 A/B evaluator 中，且只能连接本地仿真 Provider；它是实验 Control，不是产品 fallback。

### 5. Contract-Governed Skill

可执行 Contract 至少绑定：

- 输入类型和来源；
- 前置条件；
- Evidence Requirement 与 Provenance；
- Guard；
- 读写资源集；
- 时间和幂等约束；
- 风险策略；
- 后置条件；
- 可逆性等级和补偿操作；
- Typed Execution Graph；
- Contract 和 Plan 摘要。

L1 自然语言、L0.5 结构化自然语言和 L0 执行合同是可解释的 authoring compilation 轨迹。它不等同于 Experience Compilation，也不能自动激活新 L0。

### 6. Evidence-Gated Execution

Evidence 必须至少包含语义类型、来源 Capability、采集主体、时间、范围、有效性、payload 摘要以及关联的 Action。普通 Prompt 上下文、模型 confidence 和 Provider 成功文本不是 Evidence。

```text
Execute(a) iff Preconditions(a)
              and Evidence(a)
              and Guards(a)
              and RiskPolicy(a)
```

### 7. 事务和自治

主路径：

```text
BEGIN → SNAPSHOT → PRECHECK → APPROVAL? → REVALIDATE
      → EXECUTE → VERIFY → COMMIT
```

失败路径：

```text
EXECUTE / VERIFY → RECONCILE → COMPENSATE
                 → VERIFY_RECOVERY → ABORT / ESCALATE
```

Risk Policy 只能输出 `EXECUTE / ASK_HUMAN / REJECT`。结构门禁和 Evidence 先于风险分数；分数不能授予权限。当前受审 L0 可以比通用风险策略更保守。

### 8. 原型范围

当前原型必须优先完成：

1. Contract、Evidence、Guard、Risk 和事务图闭环；
2. 正常可逆变更；
3. 缺少或过期 Evidence 阻断；
4. 高风险审批或拒绝；
5. 写后断连只读对账；
6. 验证失败后的补偿和恢复验证；
7. 多步骤部分失败和逆序补偿；
8. DSH + L1 与 DSH + EnsuredSkill 的真实配对评测；
9. Contract/Evidence/Guard/Transaction/Compensation 消融；
10. 至少一个弱模型与主模型的 Runtime 稳定性对比。

网络是唯一优先验证锚点。跨域通用性只保留接口和少量回归，不继续扩展场景数量。

### 9. 生产工程冻结区

以下能力不是当前原型完成条件，也不得反向增加核心 Runtime 复杂度：

- Hermes 深度产品化和 A2A；
- 企业 OIDC/JWKS/PDP/Change Authority；
- Provider 发布、SBOM、签名、供应链资格；
- 多团队 Catalog 和治理工作台；
- 远端 WORM、HA/DR、集群一致性和生产 SLO；
- 真实厂商设备认证；
- 更多非网络业务域。

已有代码可继续通过回归测试，但状态统一为 `frozen_future_engineering`，不得称为当前架构核心或原型价值证据。

### 10. 评测口径

主指标为 Unsafe Execution Rate、False Commit Rate、Invalid Action Rate、Compensation Success Rate、Autonomous Coverage、Human Escalation Rate、Task Completion Rate 和 Runtime Overhead。

固定 Oracle 的 100% 只证明机制在已知场景按预期工作，不是生产成功概率。公开反向构造的 L1/L0 数据只作为编译器回归；核心结论必须来自真实 Harness 配对、故障注入、重复运行、扰动集和消融实验。

### 11. 原型完成判据

只有同时满足以下条件，项目才可从研究原型转向生产工程：

- 三平面和唯一写路径在代码、文档与测试中一致；
- 六类关键网络事务场景有完整 Evidence/状态/补偿轨迹；
- 五项核心机制完成消融；
- 主实验至少三次配对重复，并给出场景级统计；
- 弱模型与主模型下 Runtime 安全收益稳定；
- 明确给出 Execution Precision 与 Autonomous Coverage 的权衡；
- 不再依赖反向构造数据或单次演示支撑核心结论。

---

## English

### 1. Authority

This charter applies the EnsuredSkill design to the current repository. It supersedes conflicting historical stage plans, production-engineering designs, evaluation claims, and code comments.

The current objective is a network-first EnsuredSkill reliability prototype. Enterprise identity, provider supply chain, multi-team governance, remote immutable audit, HA/DR, production SLOs, and vendor certification are deferred. Existing implementations are frozen experimental material and are not prototype exit criteria.

### 2. Thesis and architecture

Reasoning correctness does not imply execution safety. The LLM decides what to attempt; the Runtime decides what is allowed to happen. No evidence means no action.

The authoritative architecture has three planes: a Reasoning Plane that proposes candidate plans, a Reliability Runtime that enforces contracts and transactions, and an Infrastructure Plane that owns observations and effects. Every existing component must map into one of these planes.

### 3. Effect boundary

Every product-prototype side effect requires an active Contract-Governed L0 and the Reliability Runtime. An unqualified translation may only read, clarify, produce a proposal, ask a human, or reject. Native-Agent mutation is allowed only inside the isolated local A/B evaluator as the experimental control; it is never a product fallback.

### 4. Contract, evidence, transaction, and risk

Executable contracts bind typed inputs, preconditions, evidence requirements and provenance, guards, read/write resources, temporal and idempotency constraints, risk, postconditions, reversibility, compensation, a typed execution graph, and immutable digests.

Execution follows BEGIN, SNAPSHOT, PRECHECK, optional approval, REVALIDATE, EXECUTE, VERIFY, and COMMIT. Failures reconcile, compensate, verify recovery, and then abort or escalate. Risk policy emits only EXECUTE, ASK_HUMAN, or REJECT. Scores never override a structural or evidence failure.

### 5. Prototype and deferred engineering

The prototype prioritizes six representative network transaction failures, real DSH paired evaluation, mechanism ablation, repeated runs, and model-independence evidence. Cross-domain breadth is not a current priority.

Hermes productization, A2A, enterprise control planes, provider supply-chain qualification, multi-team governance, WORM/HA/DR/SLOs, real vendor certification, and additional non-network domains are frozen future engineering.

### 6. Evaluation and exit

Primary metrics are unsafe execution, false commit, invalid action, compensation success, autonomous coverage, human escalation, task completion, and Runtime overhead. Fixed-set 100% results are mechanism evidence, never a production probability. Production engineering resumes only after the unique effect path, six transaction scenarios, five ablations, at least three paired repetitions, cross-model stability, and the precision-versus-coverage trade-off are demonstrated.
