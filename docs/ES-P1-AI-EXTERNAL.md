# ES-P1-AI-External 角色隔离评测 / Role-Isolated Evaluation

## 中文

### 结论

`ES-P1-AI-External` 协议已经完整执行，但结果是一个明确的**负向机制信号**，不是 EnsuredSkill 效果提升证据：在 15 个公开 Skill、8 个仓库、200 个 case、3 次重复和 1200 次真实本地 DSH 实验臂执行中，Treatment 保持了零 unsafe execution 和零 false commit，但因 L1→L0 转译门大面积过度安全停止，Task Completion 从 Control 的 65.67% 降至 46.83%。

本轮是角色隔离 GPT 模拟。`humanIndependent=false`、`officialEsP1QualificationEligible=false`；它不是独立人工 Private Holdout、真实系统证据、正式 ES-P1 资格或生产成功概率。`ES-P1-Private-Human` 保留为 `skipped_retained_open`。

### 实验设计

| 项目 | 固定值 |
|---|---|
| Control | DSH + 原始 L1 Skill + qwen3.5:9b 原生 Tool 编排 |
| Treatment | 相同 DSH/Skill/Tool/Task/fixture；唯一增量为 Gold-blind L1→L0 转译门、L0 auto Runtime 和安全 fallback |
| Skill / 仓库 | 15 / 8 |
| Case | 200；nominal 45、fault injection 45、read-only 45、ambiguous/missing 35、adversarial 30 |
| 重复与执行 | 3 次；600 个配对观察、1200 个实验臂 |
| 语言 | 中文 75、英文 75、中英混合 50 |
| 模型 | `qwen3.5:9b`，native chat，`think=false`，`num_ctx=32768` |

第三方 Skill 只作为不可信静态文本读取；评测没有执行其 scripts、hooks 或 installers。Gold 在 Agent 执行完成后才加载评分。

### 核心结果

| 指标 | Control | Treatment | 差值 |
|---|---:|---:|---:|
| Task Completion | 65.67% | 46.83% | -18.84 pp |
| Oracle pass | 70.00% | 50.83% | -19.17 pp |
| Parameter binding | 88.17% | 87.50% | -0.67 pp |
| Execution precision | 88.17% | 87.50% | -0.67 pp |
| Unsafe executions | 0 | 0 | 0 |
| False commits | 0 | 0 | 0 |
| p50 latency | 69.2 s | 55.1 s | -14.1 s |
| p95 latency | 177.2 s | 98.5 s | -78.7 s |

配对结果为 Treatment win 4、Control win 117、both pass 277、both fail 202。时延下降主要来自大量请求提前 safe-stop，不能解释为相同任务成功条件下 Runtime 更快。

以 Skill 为聚类单元，完成率差值的 95% bootstrap CI 为 `[-20.11, -17.48]` pp；以仓库为聚类单元为 `[-20.21, -17.72]` pp。区间整体低于零，说明退化不是由少数重复 case 单独造成，但 15 Skill / 8 仓库仍只达到 `mechanism_feasibility_15_skills`，不具备广泛泛化说服力。

### 转译路由诊断

预期路由是 L0 Runtime 90、L1 原生只读 45、safe-stop 65；实际路由是 1、4、195：

| 预期 → 实际 | L0 Runtime | L1 原生只读 | safe-stop |
|---|---:|---:|---:|
| L0 Runtime | 1 | 0 | 89 |
| L1 原生只读 | 0 | 4 | 41 |
| safe-stop | 0 | 0 | 65 |

- 路由一致率：70/200，35.0%。
- Runtime-eligible recall：1.11%。
- Unsafe Runtime accepts：0。
- Over-safe-stops：130。
- 195 个 safe-stop 中，189 个失败于 `primary_catalog_bound`，132 个同时失败于 `proposal_disposition`。

安全门禁完成了“无证据不授权”，却没有完成“合格请求稳定进入 Runtime”。本轮还暴露了 `apply` 词面锚点过拟合：模型更容易把字面含有 apply 的任务映射到 Effect capability，却把 `set`、`update`、`change` 等语义等价写请求解释为缺少可绑定主能力。这个问题应通过新的、预注册的转译版本和新样本修复验证，不能在当前密封结果上调参后重跑并覆盖。

实际唯一 L0 case `ai-ext-0029` 在三次 Treatment 中均进入 `l0_runtime` 并得到 `rollback_verified`。报告字段 `runtimeAutoInvocations=0` 只表示 Agent 已先调用受 Runtime 管理的 Effect，因此没有触发 runner 的事后 auto-invocation 补救；它不等于该 case 未经过 L0 Runtime。后续报告应把“L0 模式执行次数”和“runner 补救调用次数”拆开命名，避免误解。

### 外部审阅发现与修订轨迹

所有早期审阅结果均原样保留，没有覆盖 reviewer finding：

1. v1 Reviewer B 拒绝 45 个 verification-mismatch case，共 90 个 finding。原因是旧字段 `effectCalls` 同时被解释为向前变更预算和 apply+restore 总数。v2 合同拆分为 `forwardEffectCalls`、`compensationEffectCalls` 和 `totalStateChangingCalls`。
2. v2 Reviewer A 拒绝 22 个 safe-stop case。原因是 raw author 的 `approvalRequired` 表示“所请求操作类别是否需审批”，canonical Gold 则表示“预期执行路径是否需审批”。v3 通过语义桥分别保存 `requestedOperationApprovalRequired` 与 `executionPathApprovalRequired`，canonical `approvalRequired` 只作为后者别名。
3. 最终 v3 两名隔离 Reviewer 均审阅 200/200 并全部 accept；这证明最终注释合同内部闭合，不把 GPT 审阅等同于独立人工真值。

### 下一步

1. 将本轮结果和 digest 冻结为回归输入，不覆盖、不在其上调阈值。
2. 修复转译器的 capability grounding：从词面动词匹配改为受 Tool Catalog 约束的操作语义分类，并增加同义表达、否定、引用与领域术语的变形测试。
3. 将 L0 模式、Runtime Effect 执行和 runner 补救调用拆成三个观测指标。
4. 使用全新的预注册集合验证修复，最低目标为至少 50 Skill、15 仓库、8–10 个领域和 600–800 case。
5. `ES-P1-Private-Human` 继续保留；只有独立人员 private holdout 才能回答正式 ES-P1 泛化问题。

### 证据与复现

本轮可移植摘要绑定以下 digest：

- Evidence report：`sha256:5937c8d7852c9669ee15918f38f6587405a9119565759c9d2f5e654233b23cb2`
- DSH report：`sha256:138b7796a8576c8060f0b13873385ffd727ed91a315c15883ee02c43d4e07b95`
- Translation report：`sha256:359e36255e25e844d123afd1e626cc213896fb2c8580cabee73a48dc5887c9f2`
- Bound study：`sha256:4bd6c62f141412c325143d5875bd8ea6d2e208f60b903a87e4b0b1d76e06444d`

本地证据目录：

```text
/Users/steven/Documents/Codex/2026-08-26/wo/es-p1-ai-external-20260902/
├── translation/
├── bound-study/
├── dsh-paired-run-w4/report.json
└── evidence-report/{summary.json,report.md,report.html,manifest.json}
```

验证 digest-bound 报告：

```bash
.venv/bin/python -m evaluation.ai_external_report inspect \
  /Users/steven/Documents/Codex/2026-08-26/wo/es-p1-ai-external-20260902/evidence-report
```

完成后的代码验证结果：定向 AI-External/Public-Skill 回归 `12 passed`；全量回归在允许本地回环端口、Unix socket 和 Docker 的本机权限下为 `572 passed, 81 subtests passed`；相关评测模块 Ruff 通过；`git diff --check` 通过。全仓库 Ruff 仍报告 224 个既有 lint finding，集中在历史 `agent_memory`、profiles、scripts 和旧测试代码，因此不能把项目级 lint 声明为通过；该债务与本轮只改文档的结果封存分开记录。

## English

### Conclusion

The `ES-P1-AI-External` protocol completed, but it produced a clear **negative mechanism signal**, not evidence of an EnsuredSkill improvement. Across 15 public Skills, eight repositories, 200 cases, three repetitions, and 1,200 real local DSH arm executions, Treatment retained zero unsafe executions and zero false commits, while task completion fell from 65.67% to 46.83% because the L1-to-L0 gate stopped too many eligible requests.

This is a role-isolated GPT simulation with `humanIndependent=false` and `officialEsP1QualificationEligible=false`. It is not an independent-human private holdout, real-system evidence, formal ES-P1 qualification, or a production success probability. `ES-P1-Private-Human` remains `skipped_retained_open`.

Control used DSH, the original L1 Skill, qwen3.5:9b, and native tool orchestration. Treatment held the Skill, tools, tasks, fixtures, approvals, faults, and model fixed; its only functional increment was the Gold-blind translation gate, L0 auto Runtime, and safe fallback.

### Results

Treatment changed task completion by -18.84 percentage points, Oracle pass rate by -19.17 points, and parameter binding by -0.67 points. Control/Treatment p50 latency was 69.2/55.1 seconds and p95 was 177.2/98.5 seconds, but the apparent latency gain is confounded by early safe stops. Paired outcomes were four Treatment wins, 117 Control wins, 277 both-pass, and 202 both-fail. Skill-clustered and repository-clustered 95% bootstrap intervals were `[-20.11, -17.48]` and `[-20.21, -17.72]` percentage points.

The expected route distribution was 90 L0 Runtime, 45 native read, and 65 safe-stop cases. The observed distribution was 1, 4, and 195. Route agreement was 35.0%, Runtime-eligible recall was 1.11%, unsafe Runtime accepts remained zero, and over-safe-stops reached 130. Of the 195 safe stops, 189 failed `primary_catalog_bound` and 132 failed `proposal_disposition`. The evidence therefore separates safety from availability: the gate failed closed, but did not reliably admit eligible work.

The run also exposed lexical overfitting to the word `apply`; semantically equivalent write verbs such as `set`, `update`, and `change` often failed capability grounding. This sealed result must not be overwritten by retuning and rerunning on the same cases. A new translator version needs semantic operation classification constrained by the Tool Catalog and validation on a newly preregistered corpus.

Two preserved review rounds improved the annotation contract before the final seal. Reviewer B first rejected 45 verification-mismatch cases with 90 findings because Effect count semantics conflated forward and compensation calls. Reviewer A then rejected 22 safe-stop cases because operation-level and execution-path approval semantics were conflated. The final contract keeps separate forward/compensation/total Effect counts and separate requested-operation/execution-path approval fields. Both final isolated reviewers accepted all 200 cases; that establishes internal annotation closure, not human-independent truth.

The next persuasive public-ecosystem run should cover at least 50 Skills, 15 repositories, 8–10 domains, and 600–800 cases. `ES-P1-Private-Human` remains open and cannot be replaced by this GPT simulation.

Post-run verification completed with 12 focused tests passing and the full local regression passing with `572 passed, 81 subtests passed` when loopback ports, Unix sockets, and Docker were available. Ruff passed for the relevant evaluation modules and `git diff --check` passed. Repository-wide Ruff still reports 224 pre-existing findings concentrated in historical modules and tests; project-wide lint therefore is not claimed as passing.
