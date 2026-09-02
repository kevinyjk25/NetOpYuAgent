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

作者规范化只处理机械表示，并完整记录：给正常任务追加显式参数夹具；将 `write + none` 保守解释为 `irreversible`；把不可逆写风险抬到 high。它不能修改 read/write 意图、修复原文锚点、增加参数，或把 clarify/reject 改成可执行候选。

通用转译 Tool Catalog 与 Fixture MCP/真实 MCP 有意分离。前者回答“这个 Skill 需要什么语义能力和事务角色”；后者回答“项目现在是否已有可执行适配器”。因此一个候选可以语义上可转译，但仍因没有 Provider/Runtime adapter 而不可执行。

### 首个 9B 开发批次（2026-09-03）

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

失败并非 Translator 失败：5 个 Skill 至少有一个 source quote 不是固定原文的精确子串，3 个 read 候选仍声称 Effect budget=1；部分 Skill 同时命中两类。该 50% 是**作者候选可审率**，不是 Translator 准确率，更不是项目成功概率。0/6 修复挽回说明当前 9B 作者修复提示没有实际收益，下一轮应优先减少作者字段自由度、使用确定性原文 span 选择并补强 read/write 机械闭包，而不是把不洁案例交给 Translator。

本地完整文本制品按前三个和后九个检查点保存；可提交的聚合指标见 [development-01 摘要](benchmarks/translation-authoring-development-01-summary.json)。`alignment-review.html` 展示模型候选、规范化和失败；`alignment-review/review-packets.jsonl` 隐藏候选 expected behavior 与 Gold；`review-schema.json` 规定 AI 角色审查格式。AI 审查永远标记为 `humanIndependentEvidence=false`，只能辅助排队，不能替代独立人工证据。本轮运行尚未写入作者实现摘要，因此明确 `researchEvidenceEligible=false`；后续冻结 cohort 必须补齐实现绑定。

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
```

先扩展和审查已知开发批次，只按失败类别修改通用协议/类型/链接算法。稳定后冻结 Translator，再采集仓库隔离的新 cohort。未达到[转译泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)前，不恢复大规模 Runtime A/B。

## English

The former public-Skill study could validate wiring while pairing some real Skills with generic record tools that did not represent their documented operation. The new authoring layer precedes the Translator: pinned inert Skill → non-Gold 9B candidate → deterministic structural gate and recorded mechanical normalization → answer-hidden Skill–Task–Tool review → independent Gold → gold-blind Translator evaluation → Runtime only after generalization admission.

Each candidate binds exact source quotes, scalar parameter definitions, three task challenges, and a generic non-executable Tool Catalog with explicit observe/effect/verify/compensate roles. Reversible writes require one compensation; irreversible writes cannot claim a false rollback. This generic semantic catalog is intentionally distinct from a Fixture MCP or real Provider adapter, so translation capability and current execution support can be measured separately.

The first qwen3.5:9b development batch covered 12 Skills and 36 tasks. All 12 produced protocol-valid structures; six passed the deterministic author gate and produced 18 blind-review packets. Six Skills entered repair and none was salvaged. Five rejected Skills contained at least one non-exact source quote, while three read candidates incorrectly retained a positive Effect budget. Per-Skill authoring latency was 65.7 seconds p50 and 205.7 seconds p95. No Runtime, DSH, Tool, or third-party code ran. The 50% result is candidate reviewability—not Translator accuracy, independent generalization, or production probability.

AI-role reviews remain simulated evidence (`humanIndependentEvidence=false`). They may triage candidates for later Gold authoring but cannot unlock the translation or Runtime gate.
