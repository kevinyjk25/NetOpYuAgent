# C3f：宿主能力收口后的正向转译 / Host-Bounded Forward Translation

## 中文

### 结论

**本轮第一次出现四项完整结构/合同合格；完整源语义门禁仍未通过。** 使用 C3e 的宿主能力约束和正文 Schema，不提供参考树、目标 JSON 或答案。C3d 同一 4 个已知开发流程为 0/4 完整资格合格，本轮为 **4/4**。这是真实 9B 首次生成结果，不是复制探针或人工修订后的结果。

当前助手对 4 项完整提案逐声明审查，共 **62 项：49 supported、0 contradicted、13 insufficient_evidence**。4 项最终仍为 blocked，获全部审查支持的未激活流程为 0。声明跨中英文和图/来源映射重复覆盖，不能用 49/62 算作转译准确率。`0 contradicted` 也不证明不存在未发现的错误。

这仍是 **4 个手工已知流程 / 1 个库存工具 / 0 个公开 Skill**，不是新的泛化证据或 100% 成功率。旧失败批不改动，不在原目录重跑；当前助手参与实现与审查，非独立、非盲测。

### 各流程发生了什么

| 流程 | 首次生成的操作路径 | 源语义剩余问题 | 审查计数 supported / insufficient |
| --- | --- | --- | --- |
| 单次读取 | input.device_id → read → read_path_completed | 没有显式保存共享数据/权限限制 | 10 / 3 |
| 反向条件 | 先读；site=campus → needs_l1；否则以第一次返回的 device_id 再读 → 完成 | 同上；本次真假极性、数据引用和终态正确 | 22 / 3 |
| 缺审批/写能力 | 先读 → unsupported，事项保留缺审批及 VLAN 写合同 | 共享限制遗漏；首读引用标题而非支持具体操作的正文 | 10 / 4 |
| 缺脚本前置条件 | 入口即 unsupported，事项保留缺脚本/合同；不先读 | 共享限制没有显式映射 | 7 / 3 |

以上为**静态提案路径审查**，不是已执行这些操作。审批/脚本两项正确表达了停止位置，不是业务完成。真实缺能力事项即使其他审查都获支持，也仍阻断激活；不能删事项来提高可用率。

本批没有再生成不存在的 Effect 目标，没有将库存状态当成审批事实，没有输出截断，也没有跳过脚本前置操作。只能报告这些已知样例上的变化，不能据此声称所有复杂流程问题已解决。正文 Schema 和宿主词表约束同时发生变化，不能把改善唯一归因于其中一项。

### 为什么仍被阻断，用户该改哪里

共同遗漏是源文中的“本地计划库存，不是实时设备健康”等数据解释限制。宿主执行层依然检查权限、参数与返回值，但**执行层有门禁，不等于转译文档完整保留了语义**。本批 `business_source_ids` 全部选了标题和操作段，`purpose` / 终态说明没有保存共享限制，也未明确标记为未解决映射。

- `/source` 对中英文共享限制的完整覆盖各产生一项 insufficient；这是同一个缺口的双语检查，不是两个独立业务错误。
- `/purpose` 的 `purpose_and_limits` 仍缺限制映射，与上述发现重叠。
- 审批例额外的 `/steps/0`：`source_id=s0001` 指向标题。正文能够支持读取，但当前所选标题本身不能证明参数、工具与前置关系。该映射与“图上读取动作是否合理”分开评价。

定位时查看每项 `tree.json` 的 `business_source_ids` / `source_id`，再查看 `compilation.json` 的 `origins` 和 `review-input.json`，最后查看对应审查文件的逐声明 rationale。原始提案不在本批内改写。

生成协议将业务目的和限制合并到最多两个来源 ID 中，标题容易占一个位置；更复杂源文中这个表达上限也可能不足。下一步应改进约束承载与来源覆盖模型，而不是仅重复提示“不要遗漏”。自动保存原文仍只是文本保留，不得升级成“约束已经执行或已被正确理解”的证明。

### 成本、证据与复现

模型 `qwen3.5:9b`，think=false，temperature=0，seed=20260907，context=12,288，输出上限=2,200。每项首次一次，无修图、重试、脚本或业务 Provider 调用。源文/合同沿用 C3d，新增请求、代码和 pydantic/jsonschema/httpx 版本在调用前冻结。

四项 POST 耗时 **33.99 / 57.55 / 37.71 / 27.35 秒**，共 **156.61 秒**；输入/输出合计 **11,016 / 981 token**。计时含等待，不含预检、审查与测试；全量回归同时运行，输入长度也改变，不能用它宣称 Runtime 时延或因果性能提升。

manifest：`sha256:5ce0633ee1aa9449cf875c11fcfd7fad26454385a14a11772cf6a59fa53d2308`。

- [可重算摘要](benchmarks/flow-bounded-forward-summary.json)：绑定原始请求/响应/派生文件与完整审查摘要，包含计数和缺口定位。
- 本地首次输出：`artifacts/translator-v2/flow-bounded-4-20260907/`。
- 同一助手审查：`artifacts/translator-v2/flow-bounded-4-review-20260907/`。
- 入口：[`evaluation/flow_tree_bounded_pilot.py`](../evaluation/flow_tree_bounded_pilot.py)。`artifacts/` 不入 Git，仓库摘要不能替代完整可移植原始证据包。

```sh
# 项目根目录；新批用新的目录，不读取旧批答案来生成。
.venv/bin/python -m evaluation.flow_tree_bounded_pilot freeze artifacts/translator-v2/flow-tree-4-20260907 --output /tmp/bounded-forward-new
.venv/bin/python -m evaluation.flow_tree_bounded_pilot run /tmp/bounded-forward-new
.venv/bin/python -m evaluation.flow_tree_bounded_pilot report /tmp/bounded-forward-new --output /tmp/bounded-structural-new.json

# 完成本轮审查结果的离线重放，不会请求模型或执行业务。
.venv/bin/python -m evaluation.flow_tree_bounded_pilot report artifacts/translator-v2/flow-bounded-4-20260907 --reviews artifacts/translator-v2/flow-bounded-4-review-20260907 --output /tmp/bounded-reviewed-new.json
```

完整 checkpoint 重入只验证，不重复请求；未知中断或漂移拒绝自动恢复。审查目录中每个合格 case 必须有同名 JSON，整树摘要不同或缺任意声明都不能通过。部分失败的 token 仍计入成本，不只统计通过项。

新增 **14 项协议/审查回归**通过：无答案请求、模型/依赖漂移、未知目标、截断留存、不重复调用、伪造派生文件、完整审查覆盖及真实事项不能被忽略。最终全量 **1093 tests + 81 subtests** 通过（93.06 秒）；Ruff、git diff --check、新审查报告和旧 C3d/C3e 证据重放通过。此最终全量运行在模型调用完成后进行，不能混入前述模型计时。

### 下一步

1. 把业务目的、约束保留和操作编译分开表示，明确区分“原文已保留”“有操作/合同映射”“可确定性检查”“仍不支持”，不得靠文本携带就宣称约束已执行。
2. 为标题引用、共享限制、多段约束和无法执行的业务约束加入异质离线回归。保留本批反例，不删源要求或降低审查标准刷通过。
3. 再冻结不给答案的小批；通过后扩大跨工具/复杂流程及公开 Skill。12 个公开 Skill 整流程与大规模 Runtime A/B 仍保持门禁，不开展生产工程扩展。

## English

**Four first-attempt trees now qualify structurally and against host contracts; complete source review still blocks all four.** This is answer-free 9B generation using the C3e host-bounded request, not an explicit-answer canary or repaired output. C3d qualified 0/4 of the same known flows; C3f qualifies 4/4. Both visible Schema and host vocabulary changed, so no single-cause attribution is justified.

The same assistant reviewed all 62 claims: **49 supported, zero contradicted and 13 insufficient-evidence judgments**, with zero fully review-supported inactive flows. Bilingual/source/graph claims overlap; the ratio is not accuracy and zero contradictions is not proof of no errors. These remain four known hand-authored inventory flows, one tool and zero public Skills, without independent review or holdout evidence.

Direct read now ends correctly. The inverted branch preserves polarity and first-result references. The missing-approval flow reads then stops unsupported with real missing capabilities; the missing-script flow stops before any read. These are static proposals, not actual tool executions. Correct unsupported stops are not business completion. No fabricated Effects, approval-from-inventory substitution, truncation or skipped script prerequisite appeared in this batch.

All proposals omit explicit preservation of shared data/authority limitations, notably planned inventory versus live health. Existing Runtime authorization checks do not establish complete documentary fidelity. Purpose selection uses the title and an operation paragraph, omitting shared restrictions. The approval read additionally cites a title instead of the supporting operational paragraph. Source-coverage and purpose findings overlap; they should not be misrepresented as unrelated business failures.

Inspect `tree.json`, compiled `origins`, `review-input.json` and the per-case review JSON. The current two-ID objective/limit carrier may be too restrictive for richer sources. Next separate objective, source retention and enforcement mappings; preserving opaque text must not be counted as understanding or deterministic enforcement. Keep the frozen failures intact and test a new version before broader Skill studies.

Four POSTs took **156.61 seconds** with **11,016 input / 981 output tokens**, including waits but excluding preflight/review/tests. Regression ran concurrently; timings are not Runtime quantiles or causal performance evidence. Model/settings were fixed to 9B, no thinking, temperature zero, seed 20260907, context 12,288 and output limit 2,200. Code, dependency versions, sources and requests were frozen before calls.

See the [digest-bound summary](benchmarks/flow-bounded-forward-summary.json) and commands above. Raw outputs/reviews stay locally in `artifacts/translator-v2/flow-bounded-4-20260907/` and `flow-bounded-4-review-20260907/`; Git summaries are not a full portable evidence bundle. Completed checkpoints are not retried and stale/incomplete reviews fail closed. No Runtime, business provider, script or write was executed. Public whole-Skill and large Runtime studies remain gated.

Final verification: 14 new protocol/review tests and **1093 tests + 81 subtests** passed (93.06 seconds), plus Ruff/diff and old/new report replay. The final full-suite run occurred after model calls; its duration is not part of model latency. Test fixtures are not translation evidence.
