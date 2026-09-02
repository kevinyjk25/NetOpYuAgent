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

### 首个 9B 开发 pilot（2026-09-03）

使用 `qwen3.5:9b` 对 `development-01` 的前三个 Skill 做了静态作者 pilot：

| 项目 | 结果 |
|---|---:|
| Skill | 3 |
| 候选任务 | 9 |
| 通过确定性作者门禁的 Skill | 2/3 |
| 输出的盲审包 | 6 |
| Runtime/DSH 执行 | 0 |
| 第三方代码执行 | 0 |

`agent-bridge` 和 `brand-voice` 进入盲审队列；`stacks-commerce` 因两个 source quote 不是固定原文的精确子串而拒绝。该 66.67% 是**作者候选可审率**，不是 Translator 准确率，更不是项目成功概率。它说明门禁可以保留精确失败并阻止不洁用例污染后续评测。

本地制品位于 `artifacts/translator-v2/anchored-development-01-pilot-3/`，其中 `alignment-review.html` 展示模型候选、规范化和失败；`alignment-review/review-packets.jsonl` 隐藏候选 expected behavior 与 Gold；`review-schema.json` 规定 AI 角色审查格式。AI 审查永远标记为 `humanIndependentEvidence=false`，只能辅助排队，不能替代独立人工证据。

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

The first qwen3.5:9b development pilot covered three Skills and nine tasks. Two Skills passed the deterministic author gate and produced six blind-review packets; one failed because two source quotes were not exact substrings. No Runtime, DSH, Tool, or third-party code ran. The 66.67% result is candidate reviewability—not Translator accuracy, independent generalization, or production probability.

AI-role reviews remain simulated evidence (`humanIndependentEvidence=false`). They may triage candidates for later Gold authoring but cannot unlock the translation or Runtime gate.
