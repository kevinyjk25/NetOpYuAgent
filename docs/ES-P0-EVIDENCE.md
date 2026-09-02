# ES-P0 本地证据报告 / Local Evidence Report

## 中文

### 结论

截至 2026-09-01，EnsuredSkill 的 **ES-P0 本地研究原型证据闭环完成**，聚合状态为 `local_hypothesis_supported`：在透明开发集和本地仿真 Provider 上，加入 L0 资格门禁与 Reliability Runtime 后，9B 与较弱 7B 两种模型的 Unsafe Execution、False Commit 和 Invalid Action 均降为 0；不合格转换全部安全停机，误接受为 0。

这不是生产成功概率、隐藏集泛化证明或真实厂商设备认证。封存/私有测试集和真实设备资格仍未完成。

### 实验矩阵

| 证据 | 规模 | 目的 |
|---|---:|---|
| 六场景 Runtime | 6 场景 × 10 次 = 60 | 验证 Contract、Evidence、Guard、事务、对账、补偿和 Saga 逆序恢复 |
| 五机制消融 | 5 探针 × 6 变体 = 30 | 分别移除 Contract、Evidence、Guard、Transaction、Compensation |
| L1→L0 转译 | 60 Skills × 2 模型 | 测量严格语义通过、误接受、safe-stop 和转译时延 |
| 真实 DSH 配对 | 10 场景 × 3 次 × 2 臂 × 2 模型 = 120 会话 | 比较原生 L1 Agent 与只新增 auto Runtime 的 Treatment |
| 全量代码回归 | 506 tests + 81 subtests | 覆盖 Runtime、Promotion、DSH、MCP、Docker/Containerlab 和文档 |

### 真实 DSH 主结果

#### qwen3.5:9b

| 指标 | DSH + L1 原生 | DSH + L0 auto Runtime |
|---|---:|---:|
| Task Completion | 50.00% | **86.67%** |
| Unsafe Execution | 20.00% | **0.00%** |
| False Commit | 13.33% | **0.00%** |
| Invalid Action | 33.33% | **0.00%** |
| Execution Precision | 59.09% | **100.00%** |
| Autonomous Coverage | 43.33% | **76.67%** |
| Process Failure | 5/30 | **1/30** |
| p50 / p95 | 90.693 / 158.397 秒 | **44.405 / 72.173 秒** |

60 条转译样本中 58 条通过全部 Oracle，2 条 proposal-only；严格通过率 96.67%，误接受 0，p50/p95 为 9.486/10.865 秒。十场景实验按三次重复计 24 次 Runtime 路由和 6 次 safe-stop。

#### qwen2.5:7b

| 指标 | DSH + L1 原生 | DSH + L0 auto Runtime |
|---|---:|---:|
| Task Completion | 20.00% | **36.67%** |
| Unsafe Execution | 10.00% | **0.00%** |
| False Commit | 3.33% | **0.00%** |
| Invalid Action | 20.00% | **0.00%** |
| Execution Precision | 50.00% | **100.00%** |
| Autonomous Coverage | 20.00% | **36.67%** |
| Process Failure | 18/30 | 18/30 |
| p50 / p95 | 2.891 / 24.596 秒 | **2.436 / 9.503 秒** |

60 条转译样本中 38 条通过全部 Oracle，22 条 proposal-only；严格通过率 63.33%，误接受 0，p50/p95 为 5.926/7.590 秒。大量 DSH 会话因 qwen2.5:7b 返回 `EMPTY_RESPONSE` 失败，因此该模型不能作为当前可用性合格模型；结果只支持“安全边界没有随模型变弱而退化”。

### 机制证据

完整 Runtime 在 60 个固定场景运行中 Task Completion 100%、Unsafe/False Commit/Invalid Action 0%、补偿成功 100%。这是固定 Oracle 的机制回归，不是生产概率。

| 消融 | 直接影响 |
|---|---|
| 无 Contract | Unsafe、False Commit、Invalid Action 各 20% |
| 无 Evidence | Unsafe、False Commit、Invalid Action 各 20% |
| 无 Guard | Unsafe、False Commit、Invalid Action 各 20% |
| 无 Transaction | False Commit、Invalid Action 各 20% |
| 无 Compensation | 补偿成功率 100%→0，Task Completion 100%→80% |

### 可复现入口

```bash
scripts/netopyu-ensuredskill --output-root artifacts/ensuredskill-es-p0 --iterations 10
scripts/netopyu-ensuredskill-ablation --output-root artifacts/ensuredskill-ablation

python -m evaluation.general_effect_model \
  --model qwen3.5:9b --output-root artifacts/es-p0-9b-translation
scripts/netopyu-harness-ab \
  --model qwen3.5:9b \
  --translation-report artifacts/es-p0-9b-translation/model-translation.json \
  --output-root artifacts/es-p0-dsh-9b \
  --stratified-patterns --repetitions 3

scripts/netopyu-es-p0-report \
  --scenario-report artifacts/ensuredskill-es-p0/report.json \
  --ablation-report artifacts/ensuredskill-ablation/report.json \
  --main-harness-report artifacts/es-p0-dsh-9b/real-harness-ab.json \
  --weak-harness-report artifacts/es-p0-dsh-7b/real-harness-ab.json
```

本次本机制品位于 `artifacts/ensuredskill-es-p0`、`artifacts/ensuredskill-ablation`、`artifacts/es-p0-dsh-9b-final-v4`、`artifacts/es-p0-dsh-7b-final-v2` 和 `artifacts/es-p0-evidence-final`。`artifacts/` 不入 Git；版本化摘要见 [es-p0-evidence-summary.json](benchmarks/es-p0-evidence-summary.json)。9B 报告的一条“空会话、零效果”通过确定性重评分从 Invalid Action 移入 Process Failure；原始报告、checkpoint、源报告 digest 和 scorer fingerprint 均保留，未增加模型调用。

### 尚未证明

- 没有仓库外封存/私有泛化集；
- 没有 Cisco/Huawei/H3C 等真实设备认证；
- 没有生产网络故障率、HA/DR、远端不可变审计或 SLO 结论；
- 当前 Typed Graph 已绑定计划，但 legacy engine 尚未完全由单一图调度器驱动；
- Agent-to-Automation Experience Compilation 仍是后续研究，不属于本轮 authoring compilation。

## English

As of 2026-09-01, the local ES-P0 research-prototype evidence loop is complete and its aggregate status is `local_hypothesis_supported`. Across the transparent development set, the 9B treatment improved task completion from 50.00% to 86.67% while reducing unsafe execution, false commits, and invalid actions to zero. The weaker 7B treatment also kept all three action-safety metrics at zero, but 18/30 sessions in each arm failed at the DSH/model availability layer, so that model is not availability-qualified.

The evidence comprises 60 deterministic six-scenario runs, 30 mechanism-ablation probes, 120 real paired DSH sessions across two models, two 60-Skill translation runs, and a final regression of 506 tests plus 81 subtests. Translation false accepts were zero. These results are local development evidence—not a production probability, hidden-set generalization result, or vendor-device certification. The sealed/private set and real-device qualification remain open.
