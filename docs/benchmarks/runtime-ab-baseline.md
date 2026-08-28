# DSH only vs DSH + Network Runtime / 定量基线

## 中文

### 1. 比较目的

本基线显式量化 Domain Effect Runtime 在 DSH 之上的增量，而不是比较不同模型。两个路径接收相同工具、参数、Provider 和故障：

- **DSH only**：工具 JSON Schema、通用一次性 HITL、直接 Provider 调用；
- **DSH + Runtime**：相同能力再经过领域参数编译、L0 Skill、不可变计划、审批绑定、执行前重校验、独立验证、补偿和防篡改审计。

模型意图提取和 L1 Skill 选择被固定并排除。人工审批等待不计入机器时延。

### 2. 当前本地结果

测量时间：2026-08-28。环境：本地 macOS、Python 3.11、`profile-mock` Provider。控制集包含 11 个固定场景；时延包含 50 次预热后的有效写入样本。

| 指标 | DSH only | DSH + Runtime | Runtime 增量 |
|---|---:|---:|---:|
| 有效请求完成率 | 100.0%（1/1） | 100.0%（1/1） | +0.0 pp |
| 基础 Schema 阻断率 | 100.0%（1/1） | 100.0%（1/1） | +0.0 pp |
| 领域危险输入阻断率 | 0.0%（0/2） | 100.0%（2/2） | +100.0 pp |
| 审批后漂移阻断率 | 0.0%（0/2） | 100.0%（2/2） | +100.0 pp |
| 越权读取阻断率 | 0.0%（0/1） | 100.0%（1/1） | +100.0 pp |
| 结果判定与恢复率 | 0.0%（0/2） | 100.0%（2/2） | +100.0 pp |
| 终态与审计完整率 | 0.0%（0/2） | 100.0%（2/2） | +100.0 pp |
| **故障/风险控制有效率** | **10.0%（1/10）** | **100.0%（10/10）** | **+90.0 pp** |

| 路径 | p50 | p95 | 样本 |
|---|---:|---:|---:|
| DSH only | 0.325 ms | 0.413 ms | 50 |
| DSH + Runtime | 7.933 ms | 8.976 ms | 50 |

Runtime 本地 p50 绝对增量为 7.608 ms。由于 mock 直接调用接近零成本，不应使用 24 倍这一相对数推断生产性能；真实网络 RTT 和人工审批通常远大于本地编排开销，重复运行的时延也会随主机负载变化。

### 3. 机器判定场景

| 场景 | DSH only | DSH + Runtime | Oracle |
|---|---|---|---|
| 有效变更完成 | PASS | PASS | 正确请求执行后，目标状态必须证明成功 |
| 基础 JSON Schema | PASS | PASS | 未知字段在 Provider 调用前拒绝 |
| 缺失审计原因 | FAIL | PASS | 必须追问且写调用数为零 |
| 灾难命令阻断 | FAIL | PASS | `reload` 在 Provider 调用前拒绝 |
| 审批后 Provider 漂移 | FAIL | PASS | Provider 身份/合同改变后写调用数为零 |
| 审批窗口状态漂移 | FAIL | PASS | 当前状态与审批快照不同后写调用数为零 |
| 结果错误与自动补偿 | FAIL | PASS | 独立读回检出错误并证明恢复审批前状态 |
| 受限读取授权 | FAIL | PASS | 角色不匹配时读取 Provider 调用数为零 |
| 发送后连接中断 | FAIL | PASS | 只发送一次写，通过只读对账得到确定终态 |
| 标准终态结果 | FAIL | PASS | 上层只收到终态信封、类型化证据与摘要 |
| 审计防篡改 | FAIL | PASS | 篡改事件后哈希链验证失败并定位损坏 |

### 4. 复现

```bash
cd /Users/steven/NetOpYuAgent
scripts/netopyu-dsh compare-runtime --iterations 50
```

输出位于 `artifacts/runtime-ab/`：

- `runtime-ab.json`：机器可读原始证据；
- `runtime-ab.md`：中英双语表格报告；
- `runtime-ab.html`：浏览器可视化。

退出码非零表示 Runtime 未通过全部固定 Oracle，因此该命令可以作为 CI 回归门禁。

每个实质性 Runtime 版本完成后增加 `--record --label VERSION`。系统只记录不同的执行代码指纹，最近 3 个版本取中位数，并输出 `trend.status`：

- `regressed`：功能 Oracle 下降，或时延同时恶化超过 25% 和 3 ms；
- `improved`：保持全部功能 Oracle，同时显著降低时延或增加通过的控制场景；
- `stable`：功能保持且时延处于抗噪范围；
- `collecting`：尚不足 3 个不同实现版本。

阈值的权威来源是 `data/runtime_ab_baseline.json`，历史保存在 `artifacts/runtime-ab/history.jsonl`。

### 5. 限制

“100%（10/10）”只表示当前代码通过全部固定故障场景，不是“生产环境 100% 正确”。以下项目未测量：

- LLM 意图提取和 L1 Skill 选择准确率；
- 不同模型（7B/27B/云模型）的端到端成功率；
- 厂商真实设备和控制器兼容性；
- 人工审批耗时；
- 分布式 HA、远端审计和生产 SLO。

## English

This baseline quantifies only the deterministic controls added by the Domain Effect Runtime. The L1 decision is fixed: both paths receive the same tool, arguments, Provider and fault. DSH only retains JSON Schema and generic one-shot HITL before direct Provider invocation; the Runtime path adds domain compilation, immutable plans, approval binding, revalidation, independent verification, compensation and tamper-evident audit.

The current local campaign passes 1/10 fault/risk controls for the reference path and 10/10 for the Runtime path. Valid requests complete on both. Median machine overhead is 7.608 ms with human approval wait excluded. These are fixed-oracle coverage results, not production correctness probabilities or SLOs.

Run `scripts/netopyu-dsh compare-runtime --iterations 50` to regenerate JSON, Markdown and HTML reports under `artifacts/runtime-ab/`. Add `--record --label VERSION` after each substantive Runtime iteration. Three unique execution-code fingerprints are required before trend status becomes `improved`, `stable`, or `regressed`.
