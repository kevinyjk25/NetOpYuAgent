# 冻结协议的异质流程开发批次 / Frozen-protocol flow development batch

## 中文

### 目的与范围

C3a 只有一个辅助修订后可运行的流程。C3b 先检查：保持同一生成协议时，9B 能否表达不同的顺序、引用和控制流，以及能否保留不支持的操作。**这不是公开 Skill 泛化评测，也没有启动 Runtime 性能 A/B。**

[用例源文清单](../data/flow_translation_development.json)含 8 个手工已知开发流程，全部使用既有本地库存读取合同；1 个领域、1 个宿主工具，公开 Skill 数为 0。JSON 中的 `source` 加 `sharedBoundary` 是模型看到的完整源文；`feature`、用例 ID 和评测标签不进入请求。第二项的宿主输入包含两个设备标识，其余一项。没有给模型参考答案图。

| 流程 | 主要检查内容 |
|---|---|
| direct-read | 单次读取后结束 |
| two-input-sequence | 两个不同输入按顺序绑定到相同工具参数 |
| inverted-branch | 真假方向与先前读取结果引用 |
| nested-branch | 两层条件，同一读结果中的不同字段 |
| branch-merge | 两个分支读取后的共同继续步骤 |
| missing-approval-write | 审批和写合同缺失，不能伪造审批/配置成功 |
| unavailable-script-prerequisite | 脚本前置依赖缺失，不能跳过依赖先读 |
| unbounded-refresh | 无界循环不支持，不能改成单读完成 |

后 3 项可以产生结构合法的停止提案，但不能被记为完成了业务任务。“环境缺失”“运行时不支持”“模型转译错误”必须分开分析。

### 2026-09-07 真实结果

8 次首次 9B 调用全部返回 HTTP 200；模型响应合计 **360.18 秒，6,268 输入 / 3,596 输出 token**，不含源审查和测试时间。**5/8 结构合格，3/8 结构阻断；5 项合格提案经当前助手逐项源审查后均 blocked，0 个获审查支持的未激活流程，0 次 Runtime/写入/脚本执行。** 没有修订或重试。

计时口径：`latencyMs` 为客户端从每项 author 开始到响应落盘前的时间，包含模型预检和等待，不是 Ollama 纯推理时间；不用于与 C3a 不同调用协议作时延改善结论。最终 **961 tests + 81 subtests** 通过（87.38 秒），本阶段新增 17 项回归，相关 51 项通过，Ruff/diff 与冻结报告重算通过。

| 流程 | 结构 | 原始结果中的主要发现 |
|---|---|---|
| direct-read | 通过 | 读取和输入接线正确；业务目的写成翻译任务，问题列表含 `None: ...` |
| two-input-sequence | 通过 | 两输入顺序及参数来源正确；同样出现元任务目的及伪未解决问题 |
| inverted-branch | 阻断 | 两个终态说明为空；原始图还把真假分支写反，不能只修文字 |
| nested-branch | 通过 | 两层比较及路径一致；业务目的及非实时健康限制未保真 |
| branch-merge | 通过 | 两分支及共同读取正确；元任务目的及 `None: ...` 未解决问题仍在 |
| missing-approval-write | 通过 | 虚构 `inventory.status == approved`，将库存字段误当审批事实来源 |
| unavailable-script-prerequisite | 阻断 | null 比较与不可用引用；原始图还绕过缺失脚本前置条件，存在单读完成出口 |
| unbounded-refresh | 阻断 | 图中自循环/回边，unsupported 节点不可达，文字却声称已路由到停止节点 |

5 个结构合格提案共 **89 项声明审查：63 supported、15 contradicted、11 insufficient_evidence**。这些判断含重叠源要求和节点，不是 63/89 转译准确率；同一助手参与用例与审查，不是独立评审。其余 3 项在结构层拒绝，仅审阅原始反例，没有伪造完整语义合格率。

[结构快照](benchmarks/flow-development-8-structural-summary.json)只描述生成结束时的待审状态，故语义值仍为 null；后续以[源审查证据汇总](benchmarks/flow-development-8-evidence-summary.json)为准，包含输入/审阅/结果摘要、具体指针和拒绝解释。原始制品目录：`artifacts/translator-v2/flow-development-8-20260907`；审查目录：`artifacts/translator-v2/flow-development-8-review-20260907`。

该结果表明：序号协议可以承载嵌套及汇合，但 **9B 的整流程语义保真仍不足**。停止机制保留了错误，不等于证明业务成功率；“0 个准入”也不是总体模型准确率为 0。

### 冻结、留痕与中断行为

1. `freeze` 在第一次调用之前固定所有源文、宿主合同、模型制品摘要、每项完整请求摘要，以及生成/降级编译/审查实现摘要。
2. 每项只调用一次 `qwen3.5:9b`，`think=false`；固定参数与 C3a 序号协议一致。构造请求的重构已核对与 C3a 最后一次保存的请求相同。
3. 每项记录源、完整请求、原始响应、诊断、提案、审查包和状态，再保存文件摘要 receipt。失败也保留，不改成成功。
4. 重入只跳过已完整保存并验证的 checkpoint；实现或请求漂移直接拒绝。已有目录却无 receipt 时暂停检查，不自动重发可能已完成的模型调用。
5. 批次只生成提案，不调用 Provider、脚本、DSH 循环或写工具。`awaiting_source_review` 不是可执行资格；没有完整语义审查时 `semanticAccepted=null`，不能写成 0% 或 100%。

摘要用于发现意外变更，不是签名、来源认证或第三方独立证据。大体积原始制品在本地忽略目录；Git 中的小报告不能代替完整实验包。

### 使用

在仓库根目录，以尚不存在的目录启动一个新批次：

```bash
.venv/bin/python -m evaluation.flow_translation_batch freeze artifacts/translator-v2/my-flow-batch
.venv/bin/python -m evaluation.flow_translation_batch run artifacts/translator-v2/my-flow-batch --output artifacts/translator-v2/my-flow-batch/structural-report.json
.venv/bin/python -m evaluation.flow_translation_batch report artifacts/translator-v2/my-flow-batch
```

`freeze` 需要本地 Ollama 已安装 9B。原始 `run` 输出目录已有完成报告时，重入请省略 `--output` 或指定新的不存在路径；旧报告不会覆盖。代码变化后不能继续冻结旧批，需另建开发批次并明确列出协议变更。

完整语义审查沿用 [C3a 的 `assess` 协议](FLOW-FORWARD-TRANSLATION.md)，从原始响应重建清单；审阅与修订文件放在批次外的独立目录，避免改变已封存的 case receipt。当前助手审查只能标为非独立 AI 开发审查。

本轮完整摘要可重算到新的输出文件：

```bash
.venv/bin/python -m evaluation.flow_batch_evidence artifacts/translator-v2/flow-development-8-20260907 artifacts/translator-v2/flow-development-8-review-20260907 --output /tmp/c3b-evidence-new.json
```

### 后续判据

先定位原始输出的业务目的、条件/引用、缺失依赖、终态和未解决问题；不在本批上调提示词重跑。后续新批次才应用通用修正。不能靠不断换写同一库存例子取代原 12 个公开 Skill 的宿主合同盘点，也不能把局部子流程通过率计为整 Skill 成功率。

下一阶段 C3c 优先：将源业务目的/限制、条件真值路径、步骤前置依赖和字段的事实含义显式映射到提案审查；区分真正缺事实的问题与“无问题”陈述。需基于宿主已有合同补齐可用语义，不能新造审批工具或按用例名称写特判。修改后另冻新批，并报告新增辅助成本；Runtime 大规模 A/B 和公开 Skill 完整转译仍未解锁。

## English

C3b freezes the indexed proposal protocol and evaluates eight hand-authored known development flows. They cover sequencing, polarity, data dependencies, nested conditions, reconvergence and unavailable approval/write/script/loop behavior. All use one existing inventory tool in one domain: **zero public Skills, no held-out generalization claim**.

Before inference, the manifest binds source/context snapshots, exact request digests, model identity and generation/lowering/review code fingerprints. Each case gets one Qwen3.5:9b no-think call. Original responses and failures are preserved. Completed receipts are verified and skipped; ambiguous interrupted directories and implementation drift are rejected rather than retried.

The batch never executes providers, scripts, writes or a DSH agent loop. Structural qualification means only `awaiting_source_review`; pending semantic acceptance is null, not an accuracy score. Valid unsupported-stop proposals do not mean business completion. Reviews belong in a separate directory so frozen case files remain unchanged. Same-assistant review is not independent evidence.

The completed batch made eight first-attempt calls: five structurally qualified and three rejected. All five qualified drafts were then blocked by source review (89 claims: 63 supported, 15 contradicted, 11 insufficient evidence). Zero review-supported inactive flows, Runtime executions, writes, scripts or assisted edits. Responses totalled 360.18 seconds and 6,268/3,596 input/output tokens, excluding review/testing. Findings include meta-task purposes, nonempty “None” questions, reversed branch polarity, invented inventory-as-approval semantics, skipped prerequisites and loops contradicting the textual stop explanation. See the [review evidence](benchmarks/flow-development-8-evidence-summary.json); overlapping claim counts are not accuracy.

Next, C3c should improve explicit source-to-purpose/condition/dependency/fact mappings and real-question representation in a new frozen batch, without per-case patches, fabricated host tools or reopening large Runtime A/B.

Timing includes each client's model preflight and waiting, not pure inference and not directly comparable to C3a. Final verification: 961 tests and 81 subtests passed in 87.38 seconds; 17 new regression cases, 51 focused tests, Ruff/diff and frozen report replay passed.

Use `freeze`, `run` and `report` as shown above. Preserve old evidence, apply any general corrections only in a new declared development batch, and keep public-Skill environment gaps separate from model errors and Runtime limitations.
