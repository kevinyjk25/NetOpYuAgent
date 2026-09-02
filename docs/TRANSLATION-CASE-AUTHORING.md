# 转译用例构造与语义对齐 / Translation Case Authoring and Alignment

## 中文

### 为什么增加这一层

旧的 15-Skill/200-case 路径把部分真实公开 Skill 配给了通用 `resource.read/apply/restore` 工具。它可以验证接线，却不能公平回答“L1→L0 是否理解了 Skill 的真实业务语义”。新链路把用例构造放在转译器之前，并严格分离四件事：

```text
固定且不执行的 L1 Skill
  → 9B 作者候选（非 Gold）
  → 确定性结构门禁与透明规范化
  → 隐藏作者答案的 Skill–Task–Tool 审查
  → 独立 Gold
  → Gold-blind Translator 评测
  → 通过泛化准入后才评测 Runtime
```

### 当前实现

`evaluation/translation_case_authoring.py` 对 71 个已知开发 Skill 按 7 个仓库聚合批次工作。每个 Skill 生成一个窄操作族和三类任务：正常、缺参追问、越界/恶意拒绝。每个候选必须满足：

- 1–4 个 `SourceAnchor` 必须逐字存在于固定 Skill/reference；
- 参数名、类型和示例值进入封闭 JSON Schema，正常任务中的每个值都有显式字面证据；
- read 无 Effect；write 最多一个 Effect 且必须审批；不可逆写风险不得低于 high；
- 通用 Tool Catalog 明确 `observe/effect/verify/compensate` 角色，但固定 `executable=false`；
- 可逆写必须有唯一补偿；不可逆写不得伪造补偿；
- 失败候选不能进入对齐审查队列；第三方脚本始终不执行。

作者规范化只处理可证明等价的机械表示，并完整记录：给正常任务追加显式参数夹具；将 `write + none` 保守解释为 `irreversible`；把不可逆写风险抬到 high；按 read/write 类型闭合审批和 Effect budget；把仅有空白、换行或大小写差异且在原文中**唯一匹配**的 quote 重绑为真实原文 span。它不做编辑距离或语义模糊匹配，不能修改 read/write 意图、标点/词义、参数集合或处置标签，也不能把 clarify/reject 改成可执行候选；非唯一或词义变化的 anchor 继续 fail-closed。

通用转译 Tool Catalog 与 Fixture MCP/真实 MCP 有意分离。前者回答“这个 Skill 需要什么语义能力和事务角色”；后者回答“项目现在是否已有可执行适配器”。因此一个候选可以语义上可转译，但仍因没有 Provider/Runtime adapter 而不可执行。

### 9B development-01 迭代（2026-09-03）

使用 `qwen3.5:9b` 完成 `development-01` 的 12 个 Skill 静态作者预筛：

| 项目 | 结果 |
|---|---:|
| Skill | 12 |
| 候选任务 | 36 |
| 协议结构有效 | 12/12 |
| 通过确定性作者门禁的 Skill | 6/12 |
| 输出的盲审包 | 18 |
| 模型调用 | 18（6 个 Skill 触发修复） |
| 修复挽回 | 0/6 |
| 每 Skill 作者时延 p50 / p95 | 65.7 秒 / 205.7 秒 |
| Runtime/DSH 执行 | 0 |
| 第三方代码执行 | 0 |

首轮失败并非 Translator 失败：5 个 Skill 至少有一个 source quote 不是固定原文的精确子串，3 个 read 候选仍声称 Effect budget=1；部分 Skill 同时命中两类。该 50% 是**作者候选可审率**，不是 Translator 准确率，更不是项目成功概率。

基于这两个通用失败族增加唯一原文 span 对齐和 read/write 机械闭包后，用同一模型、同一 12-Skill 批次和实现摘要重新运行：

| 项目 | v1 | v2 |
|---|---:|---:|
| 协议结构有效 | 12/12 | 12/12 |
| 通过确定性门禁 | 6/12 | 10/12 |
| 候选可审率 | 50.0% | 83.33% |
| 盲审 task | 18 | 30 |
| 修复尝试 / 挽回 | 6 / 0 | 2 / 0 |
| 模型调用 | 18 | 14 |
| 作者时延 p50 / p95 | 65.7 / 205.7 秒 | 67.7 / 193.0 秒 |

v2 剩余两个拒绝只命中 `source_anchor_not_exact`；非唯一 span、标点或词义变化没有被模糊修复。随后对 10 个通过 Skill 的 30 个 task 进行了另一个输出目录中的答案隐藏 9B 角色审查：10/10 组协议完整，行为一致 30/30，无低置信度，p50/p95 为 26.0/43.4 秒。这个 100% 只代表**同模型角色模拟认为候选可送独立 Gold 编写**；模型答案与 Gold 均未提供给审查输入，但作者和审查者仍是同一个 9B 制品，因此它不是独立证据，也不能测量 Translator 准确率。

人工式抽查仍发现下一轮需要加强的通用门禁：禁止 `l0_read_candidate`、`l0_write_candidate`、`nominal` 等泛化 operation slug；禁止把审批状态伪装成 Tool 业务参数；增加 operation mode 与 Skill 动作语义的独立校验。它们应在更多开发批次上按失败族验证，不能针对单个 Skill 写特例。

本地完整文本制品保留逐 Skill checkpoint；可提交的聚合指标见 [development-01 v1 摘要](benchmarks/translation-authoring-development-01-summary.json)和[实现绑定的 v2 摘要](benchmarks/translation-authoring-development-01-v2-summary.json)。`alignment-review.html` 展示模型候选、规范化和失败；`review-packets.jsonl` 隐藏候选 expected behavior 与 Gold。AI 审查永远标记为 `humanIndependentEvidence=false`，只能辅助排队，不能替代独立 Gold/人工证据。

### 使用

```bash
scripts/netopyu-market-corpus anchored-author \
  artifacts/translator-v2/development-corpus-100 \
  --output-root artifacts/translator-v2/anchored-development-01 \
  --batch-id development-01 \
  --model qwen3.5:9b

scripts/netopyu-market-corpus anchored-author-inspect \
  artifacts/translator-v2/anchored-development-01 \
  artifacts/translator-v2/development-corpus-100

scripts/netopyu-market-corpus anchored-review-run \
  artifacts/translator-v2/anchored-development-01 \
  artifacts/translator-v2/development-corpus-100 \
  --output-root artifacts/translator-v2/alignment-development-01 \
  --model qwen3.5:9b

scripts/netopyu-market-corpus anchored-review-run-inspect \
  artifacts/translator-v2/alignment-development-01 \
  artifacts/translator-v2/anchored-development-01 \
  artifacts/translator-v2/development-corpus-100
```

先扩展和审查已知开发批次，只按失败类别修改通用协议/类型/链接算法。稳定后冻结 Translator，再采集仓库隔离的新 cohort。未达到[转译泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)前，不恢复大规模 Runtime A/B。

## English

The former public-Skill study could validate wiring while pairing some real Skills with generic record tools that did not represent their documented operation. The new authoring layer precedes the Translator: pinned inert Skill → non-Gold 9B candidate → deterministic structural gate and recorded mechanical normalization → answer-hidden Skill–Task–Tool review → independent Gold → gold-blind Translator evaluation → Runtime only after generalization admission.

Each candidate binds exact source quotes, scalar parameter definitions, three task challenges, and a generic non-executable Tool Catalog with explicit observe/effect/verify/compensate roles. Reversible writes require one compensation; irreversible writes cannot claim a false rollback. This generic semantic catalog is intentionally distinct from a Fixture MCP or real Provider adapter, so translation capability and current execution support can be measured separately.

The first qwen3.5:9b development batch covered 12 Skills and 36 tasks. All 12 produced protocol-valid structures; six passed the deterministic author gate and produced 18 blind-review packets. After adding unique whitespace/case-only source-span rebinding and mechanical read/write safety closure, an implementation-bound rerun accepted 10/12 Skills and emitted 30 blind tasks. The two remaining failures were non-exact anchors that the non-fuzzy aligner correctly refused. Authoring p50/p95 was 67.7/193.0 seconds.

An answer-hidden same-model role review completed all 30 tasks with 30/30 behavior agreement and no low-confidence case at 26.0/43.4 seconds p50/p95. This only queues candidates for independent Gold authoring: the author and reviewer used the same 9B artifact, so the result is neither independent evidence nor Translator accuracy. Manual-style inspection also identified generic operation slugs, Runtime-control parameters leaking into Tool schemas, and subtle read/write semantics as the next general gate families. No Runtime, DSH, Tool, MCP provider, or third-party Skill code ran.

AI-role reviews remain simulated evidence (`humanIndependentEvidence=false`). They may triage candidates for later Gold authoring but cannot unlock the translation or Runtime gate.
