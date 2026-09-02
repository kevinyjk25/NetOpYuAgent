# LLM 到可控执行的收敛评测 / LLM-to-Controlled-Execution Evaluation

## 中文

### 1. 根本问题的当前答案

项目不是把 LLM 变成“100% 正确”，而是把问题拆成可以分别证明的三段：

| 段 | 当前结论 | 证据 | 不能推出什么 |
|---|---|---|---|
| L1 语义理解 | 部分收敛、仍为概率性 | 7B/27B 的 184 条 Skill/Tool/参数/追问/workflow 评测 | 未见请求或生产准确率 |
| L1→L0 协议/Guard | 固定合同内可确定性拒绝、删值、派生和限次修复 | Candidate Schema、grounding、Guard、compiler 和逐例 trace | 模型天然安全或语义永不出错 |
| L0→Effect Runtime | Core-72 固定 Oracle 中 64/64 控制通过 | plan/approval/TOCTOU/verifier/compensation/audit Oracle | 真实设备、企业权限、分布式 SLO 已认证 |

因此更准确的回答是：**“LLM 直接获得执行权”这个根问题已经在架构上解决；“LLM 是否选对 Skill、提对参数和编对流程”只在固定集上达到可量化水平，尚未证明生产泛化。** 即使 L1 选错，模型也不能跳过 L0 的参数、权限、审批、验证和补偿边界；但合法范围内选错业务操作仍可能造成错误提案，所以 L1 资格、追问和人工审批依然重要。

### 2. 如何查看

```bash
scripts/netopyu evaluate
open artifacts/convergence/cockpit.html       # macOS
# xdg-open artifacts/convergence/cockpit.html # Linux
```

默认读取源码化的 `data/convergence_baseline.json`，克隆后无需重新跑数小时模型评测。该快照绑定摘要，包含 72 条 Runtime Oracle、两种模型汇总和 368 条隐私最小化案例，只保留 Skill/Tool id、布尔门禁、失败首层、Guard 介入计数和时延；不含 Prompt 或参数值。

使用最新原始报告重新生成：

```bash
scripts/netopyu evaluate \
  --runtime-report artifacts/runtime-ab/runtime-ab.json \
  --l1-report artifacts/l1-dsh-schema-compiler/qwen2.5-7b/l1-dsh-schema-compiler.json \
  --l1-report artifacts/l1-dsh-schema-compiler/qwen3.6-27b/l1-dsh-schema-compiler.json \
  --output-dir artifacts/convergence/current
```

驾驶舱是 self-contained、CSP 限制、无网络请求的只读页面，没有审批、激活、注册或执行入口。

### 3. 当前显性数据

| 数据 | 结果 | 解释 |
|---|---:|---|
| DSH-only 控制 Oracle | 5/64 | 通用 Tool Schema + 一次 HITL 无法承担事务语义 |
| DSH + Runtime 控制 Oracle | 64/64 | 固定 Core-72 中所有风险/故障控制通过 |
| 7B 全门禁案例 | 168/184 | 8 个语义选择、8 个参数 grounding 首层失败 |
| 7B E2E / p50 / p95 | 91.30% / 4.488s / 6.850s | 当前本地固定集合格默认 |
| 27B 全门禁案例 | 175/184 | 3 个语义选择、4 个参数 grounding、2 个协议失败 |
| 27B E2E / p50 / p95 | 95.11% / 68.193s / 176.923s | 语义更高但因两次超时不合格 |

失败采用“首层归因”，避免一个案例在多项指标里重复算作多个根因：`retrieval → protocol → semantic_selection → clarification/workflow → parameter_grounding → unattributed`。Guard containment 单独计数，不会把“被 Guard 挡住”伪装成模型本身正确。

### 4. 是否对固定用例过拟合

存在风险，不能仅凭公开固定集排除。当前缓解包括：

- Core-72 固定 L1，只测 Runtime，不用语义调参解释事务提升；
- 基线跨语言、六类场景和 24 条对抗/反误杀集；
- 模型 artifact、dataset、Catalog、Guard、Schema 和 evaluator 都绑定 digest；
- 已提供仓库外密封 holdout、双 reviewer、重复运行和 Harness parity 的**协议、Schema 与离线工具**；仓库内没有真实私有用例、人工真值或独立资格结果。

但仓库还没有真实人工标注的私有 holdout，所以 `productionGeneralization` 必须是 `not_proven`。下一步有效投入不是继续公开集调参，而是收集未见、陈旧状态、长对话、跨域冲突和真实失败样本；在不向实现团队暴露标签的情况下运行重复资格和分层回归。

### 5. 回归判定

每次改变 L1 prompt/Schema/Guard/retrieval/model 或 Runtime 合同后：

1. 运行完整测试与 retirement；
2. 运行 Runtime A/B 并记录新的实现指纹；
3. 对目标模型跑完整 184 条或外部 holdout；
4. 在驾驶舱比较失败首层、资格状态、p50/p95 和 Guard 介入，而不是只看总 E2E；
5. Runtime 任一 Oracle 退步、最终 safety escape 非零、协议完整率不足 100%，或时延超过既定抗噪/资格阈值，都视为阻断性回归。

允许的表述是“固定 Core-72 为 64/64”或“该 184 条集合 E2E 91.30%”；禁止表述为“生产成功率 100%”或“7B 可以保证 100% 意图准确”。

---

## English

### 1. Current answer

The project does not make an LLM 100% correct. It separates semantic intent, deterministic protocol/Guard controls, and effect execution. The architecture has solved the direct-execution-authority problem: model output is proposal-only and cannot bypass L0 parameters, authority, approval, verification, compensation, or audit. Semantic Skill selection, extraction, clarification, and workflow composition remain probabilistic and are only quantified on fixed datasets.

Core-72 verifies 64/64 Runtime controls versus 5/64 for DSH-only while fixing L1. The 7B fixed set has 168/184 all-gate passes; the remaining first failures are eight semantic-selection and eight parameter-grounding cases. The 27B set has 175/184 all-gate passes, but two protocol timeouts make it unqualified despite better semantic metrics. None of these figures is a production success probability.

### 2. Use the cockpit

Run `scripts/netopyu evaluate` and open `artifacts/convergence/cockpit.html`. The source-controlled digest-bound baseline contains 72 Runtime Oracles and 368 redacted model case projections without prompts or argument values. Supply new Runtime and repeated `--l1-report` files to analyze a fresh run.

The cockpit is self-contained, CSP-restricted, network-free, and read-only. It has no approval, registration, activation, or execution path.

### 3. Overfitting and regression

Public fixed-set overfitting remains possible. Digests, multiple languages/categories, adversarial cases, fixed-L1 Runtime isolation, and the repository-external holdout/dual-reviewer protocol reduce confounding but do not prove unseen generalization. Real private holdout evidence is still absent, so the report fixes `productionGeneralization` to `not_proven`.

After changing retrieval, prompts, Schema, Guard, models, or Runtime contracts, compare first-failure layers, qualification status, p50/p95, Guard intervention, and Runtime Oracles—not E2E alone. Any Runtime Oracle regression, non-zero final safety escape, incomplete protocol, or threshold-breaking latency is blocking.
