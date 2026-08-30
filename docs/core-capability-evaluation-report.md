# NetOpYu 双核心功能与性能评估 / Core Capability Evaluation

> 自动生成于 `2026-08-30T10:59:41.165944+00:00`；摘要 `sha256:8327e7f686fb88004170991ab232c0200606e2dbd3395943d82fe00c3dd8b49c`。这是工程证据报告，不构成专利可授权性或生产 SLA 结论。

## 中文

### 1. 项目要证明的两件事

```text
用户意图 / L1 Skill
        │
        ▼
[核心 A] L1 → L0.5 → L0：把开放语言逐步编译为可审查、可执行的语义合约
        │ reviewed L0 contract
        ▼
[核心 B] Network Runtime：把合约收敛为审批绑定、验证、恢复和审计的确定性事务
        │
        ▼
Service / Network Provider + 独立 Verifier
```

| 核心 | 已实现功能 | 当前量化结论 | 尚未证明 |
|---|---|---|---|
| A：分阶段语义合约编译 | 三阶段留痕、双段映射、语义丢失告警、安全门禁、防篡改审查 | 固定 URL1 样例 gate 通过；22/24 preserved，2 项需人工判断 | 未见自然语言的正向模型准确率、转换成功率、p50/p95 与 HA |
| B：确定性 Effect 事务 Runtime | 不可变计划、审批绑定、执行前重校验、独立验证、对账、补偿、Saga、审计 | Core-72：有效请求 8/8；风险/故障 Oracle 64/64 | 真实厂商设备、人工审批时延、并发长稳、分布式 HA 与生产 SLO |

项目当前已经分别回答了“语义如何被约束”和“确定操作如何安全落地”，但还不能声称任意自然语言或真实生产环境达到 100% 准确、稳定或可用。

### 2. 核心 A：L1 → L0.5 → L0

#### 2.1 功能

1. 保存自然语言 L1、结构化自然语言 L0.5 和机器执行 L0 三份独立制品。
2. 分别建立 L1→L0.5 与 L0.5→L0 requirement 映射，支持按风险和用户关注点展开。
3. 计算可追溯证据分、机器约束覆盖和语义表示覆盖；对低置信、缺失、弱化和歧义项告警。
4. 参数删除、作用域扩大、风险/审批弱化、未知 Effect、不独立验证和不安全重试失败关闭。
5. 阶段及前驱 SHA-256 绑定；模型只有 proposal 权限，不能自行注册、激活或执行。

#### 2.2 固定正向样例指标

| 指标 | 结果 |
|---|---:|
| 状态 / semantic gate | `ready_for_review` / `passed` |
| Requirement | 24 |
| Preserved | 22 |
| Non-machine-verifiable | 2 |
| Blocking | 0 |
| L1 → L0.5 证据分 | 97.50% |
| L0.5 → L0 证据分 | 90.00% |
| 端到端映射证据分 | 91.29% |
| 机器执行约束覆盖 | 91.67% |
| 语义表示覆盖 | 100.00% |

这些是**可追溯证据分，不是 LLM 准确率**。`non_machine_verifiable` 表示语言仍可见但没有确定性 L0 谓词，必须人工审查。

另有 21 个存量合同轨迹通过 Promotion 与精确 round-trip，但方向是受审 L0 反向生成 L1/L0.5 基线，只证明结构闭环和编译一致性，不证明模型正向泛化。

#### 2.3 当前性能与资格缺口

- Agent 正向 authoring 目前只开放 `lan-user-access` 一个可信 Catalog；真实 DSH/LLM 只有单次 Golden Path，不是统计基准。
- 尚未测量正向转换准确率、合法输入 proposal yield、歧义阻断率、修复耗尽率、模型调用数、转换 p50/p95、并发、长稳和 HA。
- 下一步应使用至少 200 个独立人工标注用例、双人仲裁、私有 holdout、同义改写/中英文/冲突/恶意扩权切片；固定安全集关键语义、Effect 和审批弱化逃逸必须为 0。

### 3. 核心 B：Network Runtime 确定性执行

#### 3.1 功能

1. 将已校验参数编译成不可变执行计划。
2. 审批绑定身份、能力、版本、参数、计划哈希、nonce 和执行前状态。
3. 执行前重新校验并阻断审批后漂移、重放和越权读取。
4. 使用独立 Verifier 判断真实目标状态，不信任 Provider 的成功文本。
5. 对断连或不确定结果先对账，再补偿/回滚；跨域操作执行 Saga 逆序补偿。
6. 终态和事件链防篡改审计，不能把未知结果伪装成成功。

#### 3.2 Core-72 功能对比

| 指标 | DSH only | DSH + Runtime | 增量 |
|---|---:|---:|---:|
| 有效请求完成率 | 100.0%（8/8） | 100.0%（8/8） | +0.0 pp |
| 参数与意图收口率 | 16.7%（2/12） | 100.0%（12/12） | +83.3 pp |
| 越权读取阻断率 | 25.0%（2/8） | 100.0%（8/8） | +75.0 pp |
| 审批后漂移阻断率 | 8.3%（1/12） | 100.0%（12/12） | +91.7 pp |
| 结果判定与恢复率 | 0.0%（0/12） | 100.0%（12/12） | +100.0 pp |
| 补偿与回滚正确率 | 0.0%（0/8） | 100.0%（8/8） | +100.0 pp |
| 跨域 Saga 控制率 | 0.0%（0/6） | 100.0%（6/6） | +100.0 pp |
| 终态与审计完整率 | 0.0%（0/6） | 100.0%（6/6） | +100.0 pp |
| 故障/风险控制有效率 | 7.8%（5/64） | 100.0%（64/64） | +92.2 pp |

两条路径使用相同 Tool、参数、Provider 和注入故障，固定 L1 决策并排除模型选择影响。`100%（64/64）`只表示当前 Runtime 通过全部固定本地风险/故障 Oracle，不是生产成功概率。

#### 3.3 本地机器时延与趋势

| 路径 | p50 | p95 | 样本 |
|---|---:|---:|---:|
| DSH only | 0.313 ms | 0.393 ms | 50 |
| DSH + Runtime | 7.704 ms | 8.681 ms | 50 |

Runtime p50/p95 绝对增量为 7.391/8.288 ms；人工审批等待不计入。最近 3 个不同实现指纹趋势为 `stable`，Runtime p50/p95 中位数为 7.599/8.680 ms。mock 直接路径接近零成本，不能用相对倍数外推生产性能。

### 4. 双核心组合后的真实边界

- 核心 A 限制“LLM 想做什么、遗漏了什么、哪些语义没有进入机器约束”；核心 B 限制“获准的确定意图如何执行、验证、失败恢复和留证”。
- A 的 proposal 即使 gate 通过也不能绕过人工 review/publish；B 只接受激活的 L0 合约，不能替模型修复错误业务意图。
- 当前最强证据是**固定语义样例可追溯 + 固定 Runtime Oracle 全覆盖**。最大证据缺口是**独立正向语义基准 + 真实 Provider/设备资格化 + 生产 SLO**。

### 5. 复算

```bash
# 先刷新核心 B 的本地证据
scripts/netopyu-dsh compare-runtime --iterations 50

# 再生成本双核心报告
scripts/netopyu-l0 core-eval-report

# 回归门禁
.venv/bin/python -m pytest -q
```

机器快照：[`artifacts/core-capability-evaluation/current.json`](../artifacts/core-capability-evaluation/current.json)。详细设计见 [L1 → L0 Promotion](l1-to-l0-promotion.md)、[Promotion Workbench](p20-promotion-workbench.md)、[Runtime A/B 基线](benchmarks/runtime-ab-baseline.md) 和 [架构](../ARCHITECTURE.md)。

---

## English

### 1. Two core capability families

Capability A compiles an open-ended L1 Skill through a reviewable L0.5 representation into an enforceable L0 contract. Capability B executes an activated L0 contract as a deterministic transaction with approval binding, revalidation, independent verification, recovery, compensation and tamper-evident audit.

For Capability A, the fixed URL1 sample passes its semantic gate with 22/24 requirements preserved, 2 explicitly non-machine-verifiable, and 0 blocking. Evidence scores are 97.50% for L1→L0.5, 90.00% for L0.5→L0 and 91.29% end to end. They measure deterministic traceability—not model accuracy. The 21 reverse-bootstrapped trajectories prove compiler closure, not forward generalization.

For Capability B, the Core-72 campaign preserves 8/8 valid completions and raises fixed fault/risk Oracle coverage from 5/64 (7.8%) to 64/64 (100.0%). Runtime p50/p95 are 7.704/8.681 ms in the local mock campaign; human approval wait is excluded.

The project therefore has concrete evidence for semantic traceability gates and deterministic execution controls. It does not yet have statistical forward-language accuracy, conversion availability, real-vendor qualification, distributed HA, or a production SLO. Fixed-set 100% must not be presented as a production success probability.

### 2. Reproduce

```bash
scripts/netopyu-dsh compare-runtime --iterations 50
scripts/netopyu-l0 core-eval-report
.venv/bin/python -m pytest -q
```

See [L1 → L0 Promotion](l1-to-l0-promotion.md), [Promotion Workbench](p20-promotion-workbench.md), [Runtime A/B baseline](benchmarks/runtime-ab-baseline.md), and [Architecture](../ARCHITECTURE.md).
