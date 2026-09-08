# 转译分层诊断 / Layered Translation Diagnostics

## 中文

### 本轮完成什么

2026-09-08，在冻结转译器旁增加**离线诊断器 v1**，不修改 Schema、提示词、编译门禁或模型原始回答。它回答“观测到的问题在哪层、哪些信息仍不知道”，不是自动修图器或新的执行引擎。

- [直接阅读四例诊断视图](benchmarks/flow-diagnostics-c3h-report.md)
- [完整机器报告：引文、偏移、节点、解释和建议](benchmarks/flow-diagnostics-c3h-report.json)
- [诊断实现](../evaluation/flow_diagnostics.py)、[可读视图实现](../evaluation/flow_diagnostic_report.py)

原始真实 9B 批次仍为 **4/4 流程结构合格、0/4 映射合格、0 完整语义审查**。诊断从验证过的冻结检查点重放，自动复现 **8 个目的/自身分类冲突、4 个节点证据缺口、32 项 unresolved**，另外列出 **2 项模型报告的宿主缺能力**。这些是不同类别的机械观察/声明计数，不相加计算错误概率。本轮没有新模型调用、业务工具、脚本、Runtime 或网络写执行。

| 已知开发流程 | 自动定位 | 不能由此推导 |
|---|---|---|
| direct-read | 1 个目的冲突；完成节点 `/steps/1` 已存在，缺映射 | L0 不支持完成，或整个完成语义已经丢失 |
| inverted-branch | 4 个目的冲突；条件 `/steps/1` 已存在并下沉到 `/nodes/1`，缺映射 | Runtime 不支持分支 |
| missing-approval-write | 2 个目的冲突；读取 `/steps/0` 缺映射；声明缺审批/写能力 | 映射缺口和宿主能力缺口是同一个原因 |
| unavailable-script-prerequisite | 1 个目的冲突；已有停止 `/steps/0` 缺映射；声明缺脚本 | 为了提高通过率可以忽略前置脚本或继续执行 |

范围仅 **4 个已知开发流程 / 1 工具 / 0 公开 Skill**。人工之前发现的“读取被误分成参数校验”等语义问题，不能只靠类型兼容规则自动定罪：诊断器报告可证实的内部不一致/证据缺失，语义归因需要源文审阅。

### 分层与可解释轨迹

```text
源文完整行/精确片段 → 候选语义要求 → L0.5 节点/约束 → 候选 L0 节点
                           │               │                    │
                     类型/引文检查      第一阶段源审阅       下沉前后投影检查
                           └────── 完整源语义审阅 ───────────────┘
宿主声明缺口单列；执行与结果验证在本诊断器内始终 not_evaluated
```

每份报告包含源文/宿主上下文、树、映射和诊断实现摘要；批次还绑定原 manifest/report 摘要并验证原始响应推导与文件收据。源文行含标题和空白，机械片段沿用原切分规则；标题留存不是已经实施其语义。

`candidateRequirements` 保留生成条目的原指针、候选 ID、精确来源跨度 ID、类别、目标、机械资格。`nodeTrace` 包含原 L0.5 节点、父来源 ID、实际 L0 节点和双方指针、所有声明引用与合格引用。父来源仍是不可信候选，不能自动为原流程背书。`sourceClauses` 反向列出每个源片段关联哪些候选。

**这还不是独立语义义务总表**：模型未提取的要求，不会因已经给每个片段填一行就被证明不存在。`independentObligationCount`、`semanticAccuracy`、`semanticLossRate` 和 `confidenceProbability` 均为 `null`，不显示误导性百分比。

### 如何定位和处置

| 检查层 | 当前机制 | 应修改哪里 |
|---|---|---|
| 原文/输入 | 检查源和宿主上下文结构；保留精确全文与偏移 | 澄清原文或补充真实上下文，不能推测补齐 |
| 表达/合同 | 使用原 FlowTree 编译和宿主合同校验；记录拒绝原因 | 先提供源文审阅后的独立 witness，区分表达不足与生成错误 |
| 第一阶段语义 | 可导入 `--tree-review`；即使第二阶段被阻断也可验证其摘要并显示反例 | 修改源理解、步骤、分支极性、参数或前置依赖 |
| 映射 | 全量 Schema 错误，加精确引文、重复条目、目标兼容、目的一致性和节点覆盖检查 | 修改第二阶段生成组织或具体证据，不机械补图 |
| L0 一致性 | 比较父流程和映射后 L0 的可执行投影 | 若参数、边、终态、合同或其他执行字段变化，检查下沉实现 |
| 完整语义 | 编译后可导入原 `--review`，验证摘要并定位 contradicted/insufficient 判断 | 根据精确来源修改分类、缺失要求或无依据新增 |
| 宿主能力 | 单列原模型声明及出处，标为待独立核实 | 核对真实宿主合同/系统；不能以映射修订冒充接入完成 |
| Runtime | 本工具不执行，始终未评估 | 只有前置门禁通过后再做独立执行评测 |

Schema 和映射的独立检查尽量一次列全；旧编译器仍保持遇错拒绝。如果后续旧编译/结构检查只返回首个错误，诊断不会编造其他根因。`earliestFailedStage` 是**最早观测到的失败层**，不是自动确定的唯一因果根因。模型报告的 unresolved/缺能力是待核实警告，不混入机械错误数。

L0 投影比较仅排除顶层 `purpose` 和终态 `explanation` 两类来源说明；输入合同、参数、绑定、边、终态、有效期等字段都参与比较。这能发现映射改变执行图，**不能发现两个结果共享的编译器缺陷，也不能证明二者忠实于原文**。源文审阅和独立反例仍必要。

### 本地使用

在项目根目录执行；输出必须不存在，重复运行请换路径。诊断输出禁止放入所检查的冻结批次目录。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_diagnostics batch \
  artifacts/translator-v2/flow-canonical-4-20260907 \
  --output /tmp/flow-diagnostics-local.json

.venv/bin/python -m evaluation.flow_diagnostic_report \
  /tmp/flow-diagnostics-local.json --output /tmp/flow-diagnostics-local.md
```

单例需要准备 `FlowSources`、`FlowTree`、`CanonicalMapping` 三份 JSON：

```bash
.venv/bin/python -m evaluation.flow_diagnostics case \
  sources.json tree.json mapping.json --output diagnosis.json
```

可选 `--tree-review first-pass-review.json` 和 `--review complete-review.json` 使用项目既有 `ReadL05Review` 格式，必须绑定当前相应 reviewInput 摘要。旧审阅、不同源文、不同树或不同映射不能套用。测试夹具和 AI 角色模拟都明确保留其身份，不成为独立 Gold。

`--witness alternative.json` 接受一个另存的辅助候选，包含 `sourcesDigest`（完整 FlowSources 经 `sha256_json` 的摘要）、`tree`、`mapping`，可附 `tree_review` 和 `review`。它只回答同一源/宿主上下文下另一候选是否可构造、是否维持相同执行树、是否有审阅。即使原树无法解析也可诊断 witness；不修订原结果、不算首次成功。只有独立审阅后的忠实 witness 才能支持“该语义在现机制可表达”的判断。

### 验证与仍需完成的工作

新增反例覆盖：多错误并列、已有分支漏证据、错误类型/目标、捏造/重复引文、无效 JSON 结构、否定被截掉但 Schema 仍合法、缺能力与缺证据分开、第一阶段审阅不依赖映射成功、审阅摘要漂移、分支下沉反转、witness 不替代原结果、不执行源脚本、输出防覆盖、报告内容转义及摘要验证。

本轮 **34 项新增回归**通过；全量 **1478 tests + 81 subtests 通过（125.37 秒）**，Ruff 和 `git diff --check` 通过。原报告摘要仍为 `sha256:e856360c9fa44bd7b4bc143dac8175cf991123182a260813df71bdc5ac2d5607`。回归耗时不是 Runtime 性能指标。

现阶段完成的是**可重放的定位工具及已知故障回归**，不是完整自动语义诊断或转译质量提升。下一步按诊断证据调整每节点必填来源与目的关联，先做离线反例，再冻结新批；不能修改本次失败答案刷分。独立源语义义务提取/审阅、跨 Skill 能力覆盖矩阵、最小反例自动缩减和置信度校准仍待建设。C4–C6 与规模化 Runtime 评测保持门禁。

## English

### Scope and observed results

The 2026-09-08 offline diagnostic layer sits beside frozen translators. It changes no generation Schema, prompt, admission guard or original response. [Readable report](benchmarks/flow-diagnostics-c3h-report.md) and [full evidence](benchmarks/flow-diagnostics-c3h-report.json) preserve the original **4/4 structurally qualified flows, 0/4 mappings and zero complete source reviews**. Validated checkpoint replay automatically reproduces **eight objective/self-classification conflicts, four missing-node evidence links and thirty-two unresolved candidates**, separately from two model-declared missing capabilities. No new model, Runtime, provider, script or network-write execution occurs.

These are four known development flows, one tool and zero public Skills. Missing evidence for an existing lowered completion, condition, read or stop is not a missing node. A compatible label does not prove source entailment; the previously observed semantic misclassification of reading as input validation still requires source review.

### Mechanism and use

Input and implementation digests bind exact source lines/clauses, candidate requirements, source offsets, L0.5 nodes and actual candidate L0 nodes. Reverse clause-to-candidate and node-to-evidence indexes distinguish declared from mechanically eligible links. Candidate requirements are not an independent obligation ledger; semantic accuracy, semantic-loss rate, confidence probability and the independent denominator remain null.

The stages are source structure, representation/host-contract qualification, optional first-pass semantic review, mapping diagnostics, executable-projection comparison, complete semantic review and unevaluated execution. The optional first-pass review works even when mapping is blocked. Mapping diagnostics aggregate independent errors instead of hiding missing evidence behind the first objective exception. Existing compiler exceptions remain fail-closed; earliest observed failure is not a definitive causal root cause. Declared missing capabilities and unresolved rows remain unverified warnings.

The projection check excludes only top-level purpose and terminal explanation, preserving executable arguments, references, edges, outcomes, input contracts and other fields. It detects drift between parent and mapped L0, not shared compiler defects or source fidelity. Exact-source review remains required.

Use the commands above to diagnose the preserved batch and render Markdown. The single-case command accepts FlowSources, FlowTree and CanonicalMapping JSON, with optional digest-bound `--tree-review` and `--review`. `--witness` accepts a separate source/host-bound alternative tree/mapping and optional reviews. Its constructibility is assisted diagnostic evidence, never a replacement answer or first-pass success. A faithful independently reviewed witness is needed to argue semantic representability. Outputs refuse overwrite, and batch diagnostics cannot write into the frozen input directory. Raw source scripts are inert; Markdown text is escaped and report digests are validated.

### Remaining work

Regression fixtures exercise aggregated faults, missing branch links, type/quote defects, malformed structures, negation loss not detectable by Schema alone, host-gap separation, independent first-pass review, digest mismatch, injected lowering polarity drift, witness isolation, inert scripts and output integrity. They are not independent semantic Gold or translator success cases.

All **34 new regressions** passed, as did the full **1478 tests + 81 subtests in 125.37 s**, Ruff and `git diff --check`. The original report digest remains unchanged. Regression duration is not a Runtime performance metric.

Next improve mandatory node evidence and objective associations based on diagnosed defects, first with offline counterexamples and then a separately frozen batch. Do not repair/rescore this batch. Independent source-obligation extraction/review, cross-Skill representability matrices, automatic minimal counterexample reduction and calibrated confidence remain open. C4–C6 and large Runtime evaluations stay gated; production engineering is outside this milestone.
