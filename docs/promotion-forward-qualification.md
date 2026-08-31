# L1 → L0.5 → L0 正向资格协议 / Forward Qualification

> 生成于 `2026-08-31T11:12:51.736913+00:00`。当前仓库数据是公开反向校准集，不是模型正向准确率证据。

## 中文

### 当前完成

- 已建立独立 Case、预注册 Study Plan、v2 密封 Manifest、双 Reviewer Label、摘要绑定 Resolution、模型 Observation、Adjudication 和聚合 Report 协议。
- 已提供 reviewer 专属乱序盲审包、只含分歧的仲裁包，以及支持 checkpoint/resume 的私有 9B 三次运行入口；原始 reviewer 文件不需要也不允许因仲裁而改写。
- 公开校准矩阵包含 **210 条**、**21 个能力族**、每族 **10 个**提示/语言/安全包装变体。
- 校准来源是 21 个已受审 L0 合同反向生成的 L1/L0.5 轨迹，只用于验证评分器、语义投影和覆盖矩阵。
- 报告禁止输出 Prompt 和 Label，只保留聚合指标与 case-id digest。
- Catalog v3 为每个 Observation phase 声明受信最低 `phasePredicates`；候选可以附加更强约束，但不能删除或改写最低证明。
- v7 逐案 Catalog guide/validator 收口 capability/phase/output/proof；v8 将等价 guide 封装为指纹绑定的紧凑稳定 JSON packet，并在连续 transport 故障时先 checkpoint 再暂停。
- P2.5-E 提供仓库外角色隔离工作区和只读 Doctor；它只生成 Schema、占位模板、目录与阶段门禁，不生成独立用例、Reviewer 真值或资格结论。

### 最终 v8 qwen3.5:9b 公开包装鲁棒性基线

| 指标 | 结果 |
|---|---:|
| 用例 / 能力族 / 变体 / 重复 | 210 / 21 / 10 / 1 |
| 原始协议 / 受限规范化后协议 | 76.19% / 100.00% |
| Capability exact | 100.00% |
| 参数/谓词 / Safety exact | 100.00% / 100.00% |
| Intent / 全语义 exact | 100.00% / 100.00% |
| Runtime ready_for_review / safety escape | 100.00% / 0.00% |
| 成功返回 proposal / exact-ready | 210 / 210 |
| 模型协议 / transport / Promotion 失败 | 0 / 0 / 0 |
| 受限 enum 规范化 | 50 条 / 150 个值 |
| 输入 / 输出 token | 555,845 / 113,136 |
| Prompt 表示字节 / 相对 v7 | 1,672,293 / -18.98% |
| 本机 p50 / p95 | 27.179 / 37.285 s |

| 包装变体 | 协议完成 | 全语义 exact | Runtime 可审 |
|---|---:|---:|---:|
| adversarial-en | 100.00% | 100.00% | 100.00% |
| adversarial-zh | 100.00% | 100.00% | 100.00% |
| direct-en | 100.00% | 100.00% | 100.00% |
| direct-zh | 100.00% | 100.00% | 100.00% |
| safety-en | 100.00% | 100.00% | 100.00% |
| safety-zh | 100.00% | 100.00% | 100.00% |
| schema-en | 100.00% | 100.00% | 100.00% |
| schema-zh | 100.00% | 100.00% | 100.00% |
| trace-en | 100.00% | 100.00% | 100.00% |
| trace-zh | 100.00% | 100.00% | 100.00% |

这是同一 9B 制品在 21 个能力族、10 个中英文/追踪/安全/Schema/对抗包装上的最终 v8 真实模型调用，仅一次重复。Catalog v3 把 phase-scoped 最低证明纳入 Provider-owner 受信合同；v8 以指纹绑定的紧凑 JSON packet 传输逐案 guide，并在物化前收口 capability/phase/output/proof。成功返回的 210 个 proposal 均达到全语义 exact 和 Runtime-ready；失败分布为 `none`。原始协议率与规范化后协议率同时保留，因此不能把受限兼容处理伪装成模型原始正确。该公开反向单次结果仍是诊断基线，不是私有资格或生产成功概率。


### P2.5-D 服务韧性与 Prompt 成本

最终 v8 已完整运行 210 条：210/210 全语义 exact/current-Runtime-ready，0 repair、0 模型协议/transport/Promotion/物化失败。相对最终 v7，输入 token 下降 18.89%，p50/p95 下降 13.79%/53.03%，全语义 exact 与 Runtime-ready 均提高 0.95 个百分点，transport 故障从 2 降为 0；相对更早的历史 210 基线，输入 token 仅增加 1.31%，输出 token 下降 3.96%，p50/p95 下降 2.70%/3.30%，全语义 exact 提高 3.33 个百分点。完整 210 条 Prompt 表示字节相对 v7 等价格式下降 18.98%。每次 start/resume 保存只证明注册表可达/模型已注册的 preflight；连续 transport 故障达到阈值后，触发故障先进入不可变 checkpoint，运行再暂停。恢复跳过旧失败而不静默重试。这些公开单次对照是重构回归证据，不是模型资格或生产成功概率。

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

### 私有资格工作流

正式资格必须先冻结 Study Plan，再密封 Case。Plan 将模型制品、authoring protocol、Catalog snapshot、evaluator fingerprint、重复次数，以及 case author、两名 reviewer、adjudicator 的互斥角色绑定在一起。两个 reviewer 得到不同排序且不含 gold/model output 的任务包；有分歧时生成单独仲裁包，Resolution 同时绑定两份原标签 digest。旧 v1 manifest 仍可读取和诊断，但不能通过 `preregistered_study` 门禁。

### 命令

```bash
# 重建公开校准矩阵和本报告
scripts/netopyu-l0 forward-eval-calibrate

# 真实运行本地 9B：21 个能力族各一个直接英文变体
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 21

# 运行完整 210 条公开反向校准；每条完成即写入指纹绑定 checkpoint
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \
  --output-root artifacts/promotion-forward-model/my-public-run \
  --transport-failure-limit 2

# 中断后以完全相同的模型、数据和策略恢复；任一指纹不一致都会拒绝
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \
  --output-root artifacts/promotion-forward-model/my-public-run \
  --transport-failure-limit 2 --resume

# 查看仓库外 Case、Label、Observation 的严格 JSON Schema
scripts/netopyu-l0 forward-eval-schema

# 在仓库外创建角色隔离的资格工作区；已有非空目录会失败关闭
scripts/netopyu-l0 forward-eval-study-kit --output-root /private/forward-study

# 每一步后只读检查完整性、覆盖、预注册、密封、盲审/仲裁和运行状态
scripts/netopyu-l0 forward-eval-study-doctor --root /private/forward-study

# 0 次推理：解析计划需要冻结的模型/协议/Catalog/evaluator digest
scripts/netopyu-l0 forward-eval-study-inputs CASES.jsonl --model qwen3.5:9b

# 在运行模型和 reviewer 互看前预注册计划；三类角色必须互斥
scripts/netopyu-l0 forward-eval-study-init \
  --dataset-id private-forward --version v2 --case-author-id author-team \
  --reviewer-id reviewer-a --reviewer-id reviewer-b \
  --adjudicator-id adjudicator-c --model qwen3.5:9b \
  --model-artifact-digest sha256:... --authoring-protocol-digest sha256:... \
  --catalog-snapshot-digest sha256:... --repetitions 3 --output STUDY.json

# 生成 v2 manifest，并为两名 reviewer 生成不同顺序、无 gold 的私有盲审包
scripts/netopyu-l0 forward-eval-study-seal CASES.jsonl STUDY.json --output MANIFEST.json
scripts/netopyu-l0 forward-eval-review-pack CASES.jsonl MANIFEST.json STUDY.json \
  --reviewer-id reviewer-a --output-root REVIEW-A
scripts/netopyu-l0 forward-eval-review-pack CASES.jsonl MANIFEST.json STUDY.json \
  --reviewer-id reviewer-b --output-root REVIEW-B

# 检查一致性；若有分歧，只向 adjudicator 输出分歧和两份摘要绑定标签
scripts/netopyu-l0 forward-eval-adjudicate CASES.jsonl MANIFEST.json \
  REVIEWER-A.jsonl REVIEWER-B.jsonl --study-plan STUDY.json \
  --output ADJUDICATION.json
scripts/netopyu-l0 forward-eval-resolution-pack CASES.jsonl MANIFEST.json STUDY.json \
  REVIEWER-A.jsonl REVIEWER-B.jsonl --adjudicator-id adjudicator-c \
  --output-root RESOLUTION

# 对同一预注册 9B 制品运行完整私有集三次；中断后追加 --resume
scripts/netopyu-l0 forward-eval-run-private \
  CASES.jsonl MANIFEST.json STUDY.json REVIEWER-A.jsonl REVIEWER-B.jsonl \
  --resolutions RESOLUTIONS.jsonl --model qwen3.5:9b --repetitions 3 \
  --output-root /private/qwen3.5-9b-run

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
  REVIEWER-A.jsonl REVIEWER-B.jsonl OBSERVATIONS.jsonl \
  --study-plan STUDY.json --resolutions RESOLUTIONS.jsonl --output REPORT.json
```

## English

The repository contains a pre-registered forward-qualification workflow and a 210-case public calibration matrix across 21 reviewed contract families. Catalog v3 binds phase-scoped minimum proof predicates; protocol v8 transports the equivalent per-case guide in a compact, stable, fingerprint-bound JSON packet. The final same-artifact v8 run completed all 210 wrappers with 210/210 full-semantic exact/current-Runtime-ready outcomes and zero repair or failure. Versus final v7, input tokens fell 18.89%, p50/p95 fell 13.79%/53.03%, exact/readiness rose 0.95 percentage points, and transport faults fell from two to zero. P2.5-E adds a repository-external, role-separated workspace and a read-only staged Doctor; it creates schemas and workflow controls but never manufactures independent cases, reviewer truth, identity proof, or qualification. Registry preflight has a narrow claim, and a consecutive transport-fault streak is checkpointed before the run pauses; resume never retries or rewrites old fault evidence. A v2 private study still freezes the model artifact, protocol, Catalog, evaluator, repetitions, and disjoint roles before execution. This public reverse-bootstrap, single-run result is regression evidence—not model qualification or a production success probability.
