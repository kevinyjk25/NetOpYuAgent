# DSH only vs DSH + Network Runtime / 定量基线

> **Retired product claim / 产品口径已降级**：此处 “DSH only” 实际是固定意图后的 Schema/HITL/Provider 单次调用，不经过真实 DSH/Hermes Agent、Skill 加载或 LLM 多轮工具编排。数据继续用于事务控制组件回归，但不能再用于对比原生 Harness。真实对照见[Harness auto Runtime A/B](../general-effect-ab.md)。

## 中文

### 1. 比较目的

本基线显式量化 Domain Effect Runtime 在 DSH 之上的增量，而不是比较不同模型。两个路径接收相同工具、参数、Provider 和故障：

- **DSH only**：工具 JSON Schema、通用一次性 HITL、直接 Provider 调用；
- **DSH + Runtime**：相同能力再经过领域参数编译、L0 Skill、不可变计划、审批绑定、执行前重校验、独立验证、补偿和防篡改审计。

模型意图提取和 L1 Skill 选择被固定并排除。人工审批等待不计入机器时延。

### 2. 当前本地结果

测量时间：2026-08-28。环境：本地 macOS、Python 3.11、`profile-mock` Provider。Core-72 控制集包含 72 个固定场景（8 个有效操作、64 个风险/故障控制）；时延包含 50 次预热后的有效写入样本。

| 指标 | DSH only | DSH + Runtime | Runtime 增量 |
|---|---:|---:|---:|
| 有效请求完成率 | 100.0%（8/8） | 100.0%（8/8） | +0.0 pp |
| 参数与意图收口率 | 16.7%（2/12） | 100.0%（12/12） | +83.3 pp |
| 读取权限控制率 | 25.0%（2/8） | 100.0%（8/8） | +75.0 pp |
| 审批绑定控制率 | 8.3%（1/12） | 100.0%（12/12） | +91.7 pp |
| 结果判定与恢复率 | 0.0%（0/12） | 100.0%（12/12） | +100.0 pp |
| 补偿与回滚正确率 | 0.0%（0/8） | 100.0%（8/8） | +100.0 pp |
| 跨域 Saga 控制率 | 0.0%（0/6） | 100.0%（6/6） | +100.0 pp |
| 终态与审计完整率 | 0.0%（0/6） | 100.0%（6/6） | +100.0 pp |
| **故障/风险控制有效率** | **7.8%（5/64）** | **100.0%（64/64）** | **+92.2 pp** |

| 路径 | p50 | p95 | 样本 |
|---|---:|---:|---:|
| DSH only | 0.314 ms | 0.381 ms | 50 |
| DSH + Runtime | 7.893 ms | 8.871 ms | 50 |

Runtime 本地 p50 绝对增量为 7.579 ms。由于 mock 直接调用接近零成本，不应使用约 25 倍这一相对数推断生产性能；真实网络 RTT 和人工审批通常远大于本地编排开销，重复运行的时延也会随主机负载变化。

### 3. Core-72 场景设计

| 场景族 | 数量 | 变化维度与机器 Oracle |
|---|---:|---|
| 有效操作 | 8 | LAN 授权/撤权/设备配置/服务重启，DC 授权/撤权/Fabric 配置，WAN 隧道切换；独立读回必须为 `verified_success` |
| 参数与意图 | 12 | JSON 类型、未知字段、缺失安全原因、空配置、未知实体、环境枚举、灾难命令、控制字符、超长字段、非法传输；写调用数必须为零 |
| 读取权限 | 8 | 未认证、错误角色、密级不足、缺失目的、错误/部分 scope，以及精确 scope/系统主体合法读取；同时验证拒绝与不过度拦截 |
| 审批绑定 | 12 | Provider 身份、输入/输出 Schema、capability id/version/role、计划哈希、nonce、重放、L0 入口、前置状态和显式拒绝 |
| 结果恢复 | 12 | LAN/DC/WAN/Fabric 假成功和写后断连；必须独立验证、零重试对账并得到确定终态 |
| 补偿回滚 | 8 | 逆向补偿、精确快照、无补偿契约、补偿异常/no-op、崩溃恢复；只有精确恢复才能 `rollback_verified` |
| 跨域 Saga | 6 | 依赖顺序、不可变计划、逆序补偿、不可逆步骤、重启恢复、事件篡改 |
| 终态与审计 | 6 | verified/rejected/rollback/indeterminate 终态及 payload/前序哈希篡改检测 |

场景不是通过复制同一请求改名扩充：每个场景有唯一 ID，并至少改变操作、目标、权限决策、审批绑定字段、故障阶段或恢复 Oracle 之一。11 个锚点在 `evaluation/runtime_comparison.py`，其余 61 个可执行场景在 `evaluation/runtime_core72.py`。逐场景输入、Provider 调用数、终态和 Oracle 结果由 `artifacts/runtime-ab/runtime-ab.json` 与 HTML 报告生成。

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

“100%（64/64）”只表示当前代码通过全部固定故障场景，不是“生产环境 100% 正确”。以下项目未测量：

- LLM 意图提取和 L1 Skill 选择准确率；
- 不同模型（7B/27B/云模型）的端到端成功率；
- 厂商真实设备和控制器兼容性；
- 人工审批耗时；
- 分布式 HA、远端审计和生产 SLO。

## English

This baseline quantifies only the deterministic controls added by the Domain Effect Runtime. The L1 decision is fixed: both paths receive the same tool, arguments, Provider and fault. DSH only retains JSON Schema and generic one-shot HITL before direct Provider invocation; the Runtime path adds domain compilation, immutable plans, approval binding, revalidation, independent verification, compensation and tamper-evident audit.

The Core-72 local campaign contains eight valid operations and 64 fault/risk controls spanning parameter/intent closure, read policy, approval binding, outcome recovery, compensation, cross-domain Saga, and evidence integrity. The reference path passes 5/64 controls (7.8%); the Runtime path passes 64/64 (100%). Both complete 8/8 valid operations. Median machine overhead is 7.579 ms with human approval wait excluded. These are fixed-oracle coverage results, not production correctness probabilities or SLOs.

Run `scripts/netopyu-dsh compare-runtime --iterations 50` to regenerate JSON, Markdown and HTML reports under `artifacts/runtime-ab/`. Add `--record --label VERSION` after each substantive Runtime iteration. Three unique execution-code fingerprints are required before trend status becomes `improved`, `stable`, or `regressed`.
