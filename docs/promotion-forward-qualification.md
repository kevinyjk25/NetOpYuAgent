# L1 → L0.5 → L0 正向资格协议 / Forward Qualification

> 生成于 `2026-08-31T02:16:56.950805+00:00`。当前仓库数据是公开反向校准集，不是模型正向准确率证据。

## 中文

### 当前完成

- 已建立独立 Case、双 Reviewer Label、模型 Observation、密封 Manifest、Adjudication 和聚合 Report 六类协议。
- 公开校准矩阵包含 **210 条**、**21 个能力族**、每族 **10 个**提示/语言/安全包装变体。
- 校准来源是 21 个已受审 L0 合同反向生成的 L1/L0.5 轨迹，只用于验证评分器、语义投影和覆盖矩阵。
- 报告禁止输出 Prompt 和 Label，只保留聚合指标与 case-id digest。

### 真实 qwen3.5:9b 公开包装鲁棒性基线

| 指标 | 结果 |
|---|---:|
| 用例 / 能力族 / 变体 / 重复 | 210 / 21 / 10 / 1 |
| 原始协议 / 受限规范化后协议 | 76.19% / 99.52% |
| Capability exact | 99.05% |
| 参数/谓词 / Safety exact | 96.67% / 99.52% |
| Intent / 全语义 exact | 99.52% / 96.67% |
| Runtime ready_for_review / safety escape | 97.14% / 0.00% |
| 受限 enum 规范化 | 50 条 / 150 个值 |
| 本机 p50 / p95 | 27.934 / 38.557 s |

| 包装变体 | 协议完成 | 全语义 exact | Runtime 可审 |
|---|---:|---:|---:|
| adversarial-en | 100.00% | 100.00% | 100.00% |
| adversarial-zh | 100.00% | 95.24% | 100.00% |
| direct-en | 100.00% | 95.24% | 95.24% |
| direct-zh | 100.00% | 95.24% | 95.24% |
| safety-en | 100.00% | 95.24% | 95.24% |
| safety-zh | 100.00% | 100.00% | 100.00% |
| schema-en | 100.00% | 95.24% | 95.24% |
| schema-zh | 95.24% | 90.48% | 90.48% |
| trace-en | 100.00% | 100.00% | 100.00% |
| trace-zh | 100.00% | 100.00% | 100.00% |

这是同一 9B 制品在 21 个能力族、10 个中英文/追踪/安全/Schema/对抗包装上的真实模型调用，仅一次重复。该历史运行使用 L1/L0.5 v2 显式意图锚点，intent exact 为 99.52%；受限边界只将精确的 `{"value": primitive}` enum 包装还原为 primitive，并逐路径留证，不改变 L0 Schema。历史失败分布为 `model_protocol=1, promotion_assessment=5`；未通过的候选被 Runtime 失败关闭。历史 `ready_for_review` 只证明当时的结构与 Catalog 自洽，不等于人工真值 exact：本轮存在一个可审但 phase capability 选择偏移的候选。当前 Catalog v2/L0.5 v3 已加入 phase-typed 门禁，并在不调用模型的重放中阻断该候选；重放结果见双核心评估。原始协议率与规范化后协议率同时保留，因此不能把兼容处理伪装成模型原始正确。该公开反向单次结果仍是诊断基线，不是私有资格或生产成功概率。


### 为什么当前不能宣称模型通过

| 门槛 | 当前状态 |
|---|---|
| 至少 200 条 | 已满足（210） |
| 至少 10 个能力族 | 已满足（21） |
| 独立正向人工编写 | **未满足** |
| 仓库外私有 holdout | **未满足** |
| 两名独立 reviewer 一致 | **未满足** |
| 同一模型制品至少三次运行 | **未运行** |

因此当前状态是 `protocol_ready_model_not_qualified`，不能把 evaluator self-check 或反向 round-trip 表述为 LLM 准确率。

### 正式门槛

- Protocol completion ≥99%；Capability exact match ≥99%；
- 参数/谓词 exact match ≥95%；Safety contract exact match ≥99%；
- 歧义阻断和合法 proposal yield ≥95%；重复稳定性 ≥95%；
- 关键语义、未声明 Effect、审批/风险弱化逃逸必须为 0。

### 命令

```bash
# 重建公开校准矩阵和本报告
scripts/netopyu-l0 forward-eval-calibrate

# 真实运行本地 9B：21 个能力族各一个直接英文变体
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 21

# 运行完整 210 条公开反向校准；每条完成即写入指纹绑定 checkpoint
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \
  --output-root artifacts/promotion-forward-model/qwen3.5-9b-public-210

# 中断后以完全相同的模型、数据和策略恢复；任一指纹不一致都会拒绝
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \
  --output-root artifacts/promotion-forward-model/qwen3.5-9b-public-210 --resume

# 查看仓库外 Case、Label、Observation 的严格 JSON Schema
scripts/netopyu-l0 forward-eval-schema

# 密封仓库外的独立正向用例
scripts/netopyu-l0 forward-eval-seal CASES.jsonl \
  --dataset-id private-forward --version v1 --provenance independent_forward \
  --output MANIFEST.json

# 双人一致性检查
scripts/netopyu-l0 forward-eval-adjudicate CASES.jsonl MANIFEST.json \
  REVIEWER-A.jsonl REVIEWER-B.jsonl --output ADJUDICATION.json

# 把一次真实 Agent proposal 标准化成无 Prompt Observation
scripts/netopyu-l0 forward-eval-record \
  --case-id CASE-ID --repetition 1 --model MODEL \
  --model-artifact-digest sha256:... --authoring-protocol-digest sha256:... \
  --catalog-snapshot-digest sha256:... \
  --disposition proposal \
  --proposal /path/to/proposal --catalog-id CATALOG-ID \
  --latency-ms 1234 --model-calls 1

# 对一个不可变模型制品的重复 Observation 评分
scripts/netopyu-l0 forward-eval-score CASES.jsonl MANIFEST.json \
  REVIEWER-A.jsonl REVIEWER-B.jsonl OBSERVATIONS.jsonl --output REPORT.json
```

## English

The repository now contains a sealed forward-qualification protocol and a 210-case public calibration matrix across 21 reviewed contract families. The matrix is reverse-bootstrapped and public, so it can validate evaluator closure but cannot qualify model accuracy. Qualification requires an external independent 200+ case private holdout, two-reviewer consensus, one immutable model artifact, at least three repetitions, zero safety escapes, and all fixed thresholds.
