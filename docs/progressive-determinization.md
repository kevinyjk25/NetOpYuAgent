# 通用渐进式确定化 / General Progressive Determinization

## 中文

### 1. 结论

NetOpYuAgent 的核心现在定义为**通用 Effect Runtime**，不再把网络当作框架边界。网络仍是第一个、也是当前最完整的参考 Profile，因为它天然要求参数精确、变更审批、独立验证、未知结果对账与补偿；IAM、云资源、工单、数据和其他外部系统可以复用同一控制面，只需提供自己的 Capability Catalog、Provider 和 L0 合同。

```text
用户 / 外部事件
  → Harness（DSH 主路径，Hermes 可选）
  → L1 Skill：理解、追问、诊断、开放式编排
  → Promotion：L1 → L0.5 → L0 proposal
  → 硬门禁 + 风险分级证据决策
      ├─ active L0 + 通过门槛 → Effect Runtime
      ├─ 只允许调用已有 active L0 → L1/L0 hybrid
      ├─ 只读低置信 → L1 read-only
      ├─ 语义缺失/歧义 → clarification
      └─ 写入低置信/未激活 → proposal-only
  → Provider（MCP / REST / CLI / NETCONF / Controller）
  → 任意外部 Layer
```

### 2. 哪些控制是确定性的

置信度之前先执行硬门禁：

1. `SKILL.md` 必须满足格式约束；符号链接、越界引用、缺失资源和超限包失败关闭。
2. `scripts/`、`references/`、`assets/` 与引用图全部计算摘要。Promotion 只披露从 `SKILL.md` 可达的资源。
3. 包内脚本只作为不可信文本检查，Promotion 绝不导入或执行脚本；可能产生副作用的脚本必须显式绑定受审 Capability。
4. L1、L0.5 和编译后 L0 的参数、意图、Capability、风险、审批、验证和补偿必须通过现有 Promotion 语义门禁。
5. 高风险或破坏性效果还要求人工审过的激活制品和可用审批控制。任何分数都不能自动激活或自动执行合同。

只有硬门禁全部通过后，策略才把语义映射、L1→L0.5、L0.5→L0、包引用覆盖、重复稳定性、仿真通过率组合为风险分级证据分数。未经校准或未绑定模型/校准制品摘要的模型置信度不参与路由。该分数是路由证据，不是生产成功概率。

### 3. L1 与 L0 如何混用

| 边 | 规则 |
|---|---|
| L1 → L1 | 允许开放式分析和编排，但没有直接写权限 |
| L1 → L0 | 只允许调用 active、版本化、摘要绑定的 L0 |
| L0 → L0 | 允许使用精确依赖和 Saga，禁止运行时猜测 |
| L0 → L1 | 事务内禁止；L0 必须先停止并返回 replan 结果，再由 Harness 开启新一轮 L1 |

低置信度不会“回退到原始 L1 直接写 MCP”。安全回退只有只读诊断、追问、proposal-only，或 L1 编排已有的可信 L0。

### 4. Anthropic Skill 结构验证集

当前 fixture suite 有 10 个包，覆盖 8 个领域与 20 种结构特征：只读、参考资料、渐进披露、审批、条件分支、多步骤、脚本、资产、独立验证、补偿、回滚、L1/L0 组合、不可逆效果、缺失引用和路径穿越等。10/10 结构 Oracle 已通过；这只证明包检查器按预期接受或阻断，不等于 LLM 转译准确率。

```bash
python -m evaluation.progressive_skill_suite \
  --output artifacts/progressive-determinization/skill-package-fixtures.json

scripts/netopyu-effect inspect-package \
  --skill evaluation/fixtures/progressive-skills/scripted-network-change \
  --bound-script scripts/apply.py=network.interface.description.set \
  --bound-script scripts/rollback.py=network.interface.restore
```

用例清单在 [`evaluation/fixtures/progressive-skills/cases.yaml`](../evaluation/fixtures/progressive-skills/cases.yaml)，版本化机器报告在 [`docs/benchmarks/progressive-skill-fixtures.json`](benchmarks/progressive-skill-fixtures.json)。

### 5. 当前 9B 验证

本轮使用固定 `qwen3.5:9b` 制品新跑了 3 条真实 model→semantic proposal→受信 materialization→Promotion/Runtime smoke：3/3 达到 `ready_for_review`，原始协议、完整语义、Runtime 可审均为 100%，safety escape、repair 和 transport failure 为 0；本机 p50/p95 为 35.629/38.326 秒。样本只有 3 条且来自公开反向校准数据，所以只能证明新接线未断，不能证明跨领域泛化或模型资格。

网络 URL1 锚点同时通过包门禁和语义门禁，风险分级证据分数为 90.64，已激活合同的路由结果是 `l0_runtime`。该固定用例结果也不是生产成功概率。

机器汇总见 [`docs/benchmarks/progressive-determinization-baseline.json`](benchmarks/progressive-determinization-baseline.json)。

### 6. P2.6-B 本地控制变量 A/B

本地开发集已经按控制变量方案落地：

- A：同一 DSH、9B、MCP、Anthropic Skill、输入和故障，原始 L1 Skill 直接使用工具；
- B-auto：同一输入先自动转成 L0.5/L0；硬门禁和风险阈值通过时走 Effect Runtime，未通过时只读/追问/proposal；不允许人工修 candidate；
- B-reviewed：人工修订后的生产上限，单独报告，不能混入 B-auto；
- 实际集：六域 24 个 MCP Capability 和 60 个 Skill，六类结构各 10 条，包含正常、缺参、越界参数、审批拒绝、revision 冲突、发送后未知、验证失败和补偿失败。

历史固定决策组件微基准为 direct 36/60、Runtime 60/60，但它没有真实 Harness/LLM 多轮执行，不能作为 Agent 产品对照。当前主实验比较原生 Harness+L1 Agent 与可信转换自动 Runtime；转换失败在 Treatment 中安全停机，原生写只保留为隔离 Control。完整 9B 转译达到 raw 60/60、严格语义 58/60（96.67%）、0 误接受和 2 个不合格转换。协议和限制见[真实 Harness 自动 Runtime A/B](general-effect-ab.md)。

## English

### 1. Conclusion

NetOpYuAgent now defines its core as a **general Effect Runtime**. Network is the first and most complete reference profile, not the framework boundary. IAM, cloud, service-desk, data, and other external systems can reuse the same control plane by supplying domain Capability Catalogs, Providers, and L0 contracts.

The flow is Harness → L1 Skill → Promotion → deterministic gates and risk-tier evidence routing → active L0 Effect Runtime → protocol adapter → external layer.

### 2. Deterministic controls

Hard gates precede confidence. The system validates the Skill format, package paths, resource graph and digests; discloses only reachable `scripts/`, `references/`, and `assets/`; never imports or runs bundled scripts during authoring; requires potential effects to be Capability-bound; and applies the existing L1/L0.5/compiled-L0 semantic checks. Scores cannot activate or execute a contract.

After those gates pass, a weighted geometric evidence score combines semantic mapping, both translation transitions, package traceability, repeat stability, and simulation results. An uncalibrated or non-digest-bound model judge is excluded. The score controls routing and is not a production-success probability.

### 3. L1/L0 composition

L1 may call L1 or an active version-and-digest-bound L0. L0 may compose exact active L0 dependencies. L0 cannot call L1 inside an effect transaction; it must stop and return a replan outcome before the Harness starts a new L1 turn. Low-confidence writes never fall back to direct L1 MCP mutation.

### 4. Heterogeneous Skill fixtures

The current ten-package suite spans eight domains and twenty structural features, including references, progressive disclosure, approvals, branches, multi-step workflows, scripts, assets, verification, compensation, L1/L0 composition, irreversible effects, missing resources, and path traversal. All ten structural Oracles pass. This proves inspector behavior only, not translation accuracy.

### 5. Current 9B evidence

A fresh three-case `qwen3.5:9b` smoke traversed real model proposal, trusted materialization, Promotion, and Runtime-review gates. All three reached `ready_for_review`; raw protocol, full semantics, and Runtime readiness were 100%, with zero safety escape, repair, or transport failure. Local p50/p95 was 35.629/38.326 seconds. The tiny public reverse-bootstrap sample is wiring evidence, not cross-domain qualification.

The URL1 network anchor passes package and semantic gates. Its evidence score is 90.64 and an already active contract routes to `l0_runtime`. See the [versioned machine report](benchmarks/progressive-determinization-baseline.json).

### 6. P2.6-B local controlled A/B

The 24-capability, 60-Skill development corpus retains the historical 36/60-versus-60/60 one-shot component regression, but that result is no longer treated as a Harness Agent comparison. The current protocol compares native Harness+L1 execution with qualified auto Runtime; an unqualified treatment conversion stops safely, while native mutation remains an isolated control only. The complete 9B translation run records 60/60 raw protocol validity, 58/60 strict semantics (96.67%), zero false accept, and two unqualified translations. See the [real Harness auto-Runtime A/B](general-effect-ab.md).
