# L1 → L0.5 → L0 正向资格协议 / Forward Qualification

> 生成于 `2026-08-30T17:53:16.349515+00:00`。当前仓库数据是公开反向校准集，不是模型正向准确率证据。

## 中文

### 当前完成

- 已建立独立 Case、双 Reviewer Label、模型 Observation、密封 Manifest、Adjudication 和聚合 Report 六类协议。
- 公开校准矩阵包含 **210 条**、**21 个能力族**、每族 **10 个**提示/语言/安全包装变体。
- 校准来源是 21 个已受审 L0 合同反向生成的 L1/L0.5 轨迹，只用于验证评分器、语义投影和覆盖矩阵。
- 报告禁止输出 Prompt 和 Label，只保留聚合指标与 case-id digest。

### 真实 qwen3.5:9b 单次宽度基线

| 指标 | 结果 |
|---|---:|
| 用例 / 能力族 / 重复 | 21 / 21 / 1 |
| 原始协议 / 受限规范化后协议 | 76.19% / 100.00% |
| Capability exact | 100.00% |
| 参数/谓词 / Safety exact | 95.24% / 100.00% |
| Intent / 全语义 exact | 100.00% / 95.24% |
| Runtime ready_for_review / safety escape | 95.24% / 0.00% |
| 受限 enum 规范化 | 5 条 / 15 个值 |
| 本机 p50 / p95 | 36.815 / 44.449 s |

这是同一 9B 制品的真实模型调用，但只跑了公开反向矩阵中每族一个直接英文变体且仅一次重复。L1/L0.5 v2 显式意图锚点使 intent exact 达到 100.00%；受限边界只将精确的 `{"value": primitive}` enum 包装还原为 primitive，并逐路径留证，不改变 L0 Schema。当前失败分布为 `promotion_assessment=1`；未通过的候选被 Runtime 失败关闭。原始协议率与规范化后协议率同时保留，因此不能把兼容处理伪装成模型原始正确。该结果仍是诊断基线，不是资格结论。


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
