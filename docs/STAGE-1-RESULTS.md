# 阶段 1 转译修复验收 / Stage 1 Translation Repair Results

## 中文

状态：2026-09-10，**阶段 1 开发验收完成，已停在阶段 2 之前**。冻结协议为 `windowed-source-ledger/v62`，模型为本地 `qwen3.5:9b`。同一实现/配置下三种已知流程各两份首次构造，**6/6 只读区域通过源文与参数审阅，50/50 合成本地路径通过**。[固定退出条件](STAGE-1-EXIT.md)未放宽；这不是公开 Skill 泛化通过。

[可随代码保存的指标摘要](benchmarks/stage1-translation-summary.json)；[本地完整证据](../artifacts/translator-v2/stage1-20260909/evidence-summary/report.json)；[逐份源文审阅](../artifacts/translator-v2/stage1-20260909/developer-review-v62.json)。证据报告摘要为 `sha256:076184068488daa3019d34473dfc2ad408918bf27f9443e76ec92555c5a0ccd4`。

### 验证了什么

输入是完整原文、引用、调用任务及原宿主 Schema，不是手写 L0 答案。9B 构造带出处的控制树，再逐槽生成参数；原编译器生成只读区域，开发者 AI 对照原文审阅后，交给原共享 Runtime 做合成本地路径检查。

| 已知开发流程 | 首次构造 A | 重复构造 B | 两份本地路径 |
|---|---|---|---|
| 接口检查：空列表、首接口状态、错误计数、仅提变更候选 | 通过 | 通过 | 18/18 |
| 变更审批记录 → 返回设备健康 → 解释/调查交接 | 通过 | 通过 | 16/16 |
| 服务 → 引用中的条件检查 → 健康/告警 → 受限解释交接 | 通过 | 通过 | 16/16 |

每份只有一次规划，再按读取逐槽填参，并不是一次模型请求完成全部构造。最终 **20 次真实模型调用，88,024 输入 / 13,850 输出 token**，请求耗时合计 992.70 秒。每次请求 p50/p95 为 **24.07/132.34 秒**；其中规划 6 次为 114.63/134.83 秒，填参 14 次为 21.93/28.11 秒。这是本地模型构造耗时，不是 Runtime 时延；部分调用与回归并行，不作为隔离性能基准。

已完成的工程检查：512 项定向；2524 项全量及 81 项子测试。阶段修改的 26 个 Python 文件 Ruff 通过。额外全仓检查有 **224 条历史问题，涉及 58 个与 HEAD 完全相同的文件**，因此不是“全仓 clean”。运行期间出现的无关根目录文件未修改，不纳入阶段 lint 范围。

### 修复点及权威边界

| 问题 | 当前机制 | 仍不能据此推断什么 |
|---|---|---|
| 文档 ID、字段名被当成数据变量 | `input` / `obs0`–`obs7` 命名空间，原作用域与类型检查 | 合法变量不代表选对业务来源 |
| 所有分支结束后又追加授权工作 | 闭合控制树；未来宿主要求单独保留且未满足 | 图合法不代表实际获得权限 |
| 条件/动作出处错位 | 先选原文 ID，再构造操作数；原文与节点位置机械保留 | 引文存在不证明蕴含关系，仍需审阅 |
| 终点把“可用”扩写为“整体健康” | 固定流程状态标签；业务解释另留为 L1 职责 | 流程终点不是业务事实或完整任务成功 |
| 参数错名、丢嵌套、编造 ID | 原 Schema 的参数槽与所有兼容来源供模型选择，再由原绑定器校验 | 类型兼容不证明业务角色匹配 |

当前流程仍是有界只读树：最多 8 个读取、32 个节点、8 层条件，精确标量相等和数组长度；不提供任意循环、共享 DAG 汇合、分页聚合、脚本或写操作。自然语言解释/建议保留在 L1 交接，并未在这轮执行。[受控混合流程](GOVERNED-HYBRID-FLOWS.md)中的图内 LLM、恢复与并行调度仍是后续设计，不作为通过此门禁的替代。

### 怎样检查一次转换

以引用流程 A 为例，按顺序查看：

1. [原 L1 Skill](../evaluation/fixtures/stage1/reference/SKILL.md)和[原引用](../evaluation/fixtures/stage1/reference/references/checks.md)。
2. [模型原始选择](../artifacts/translator-v2/stage1-20260909/reference-v62-a/round-000/choice.json)：原文角色、步骤、未来要求及控制树。
3. [来源与转换轨迹](../artifacts/translator-v2/stage1-20260909/reference-v62-a/round-000/program-draft.json)：`inlineSourceMap`、`controlSyntaxLowering`、固定状态标签及受限语法。
4. [实际生成的只读 Tree](../artifacts/translator-v2/stage1-20260909/reference-v62-a/round-003/tree.json)：调用者 serviceId → 观察返回 device.id → 健康返回 alarmId。
5. [本地执行检查](../artifacts/translator-v2/stage1-20260909/behavior-reference-v62-a/report.json)：每条路径的预期/实际调用与参数，包括非法返回和无权限情况。

`artifacts/` 为本地产物，未承诺随 Git 发布。阶段历史保留 **109 个运行目录、245 次有结果的模型调用、23 个无回执请求**；另有 2 份回执缺少完整 token 计数，未知不等于零费用。**86 份完整报告**均用各自原版代码零调用回放且逐字节一致；最终证据绑定 2919 个制品。旧的 33+127 份历史制品摘要也再次核对不变。没有覆盖失败、重试中断或拼接不同版本的好结果。

### 不等于哪些结论

这是经过许多次修订、反复用于开发的 **3 种流程**，存在明显的开发集适配偏差，不能据此估计总体准确率。同一固定 seed 的重复用于检查可复现性，不是独立统计样本。接受的单位是只读区域，不是完整 Skill；没有执行 L1 输出生成/脱敏、源脚本、真实网络写入或 DSH-only 对照。有限路径全过不代表生产成功概率。

现已停止在阶段 2 前，未提交或推送 Git，默认 DSH 路由与自动激活策略未变。下一步预选 10 个已知公开开发 Skill，至少 6 个仓库、3 类领域，在模型运行前固定来源、任务、宿主与审查要求；检验向新材料迁移，而不是继续在这三例上调参。正式跨 cohort 门禁另行推进，不直接扩大 Runtime A/B。

## English

Status: 2026-09-10, **Stage 1 development exit criteria passed; stopped before Stage 2**. Frozen windowed-source-ledger/v62 and local qwen3.5:9b constructed each of three known procedures twice: 6/6 source/argument-reviewed read regions and 50/50 synthetic paths. The [fixed criteria](STAGE-1-EXIT.md) were not relaxed. See the [portable metrics](benchmarks/stage1-translation-summary.json), [local bound evidence](../artifacts/translator-v2/stage1-20260909/evidence-summary/report.json) and [individual review](../artifacts/translator-v2/stage1-20260909/developer-review-v62.json).

Original source/reference text, tasks and host schemas—not expected L0 answers—go to 9B. It proposes a sourced control tree and per-read parameters. Original compiler/Runtime checks pass 18/18 wiring, 16/16 approval and 16/16 reference paths across both constructions. Each has one planning call then per-read binding calls: 20 real calls, 88,024 input/13,850 output tokens, summed request time 992.70 seconds. Per-request p50/p95: 24.07/132.34 seconds; planning: 114.63/134.83; binding: 21.93/28.11. These are local model construction costs, not Runtime latency or isolated performance; some calls overlap regression. Targeted tests: 512; full: 2524 plus 81 subtests. All 26 stage Python files pass Ruff. Repository-wide lint retains 224 diagnostics across 58 files identical to HEAD; unrelated concurrent root files are untouched, not a whole-repository clean claim.

Repairs separate document/data namespaces, close control-flow syntax, retain unsatisfied future host requirements, anchor source before operands, replace free-form terminal success prose with fixed control labels, and use original-schema parameter slots. These constrain representation and authority, not semantic truth. Source review is still necessary. Limits remain eight reads, 32 nodes and eight conditional levels, with exact scalar/array-length checks; no arbitrary loops, shared DAG joins, pagination, scripts or effects. Source-required explanation and recommendations remain L1 duties, not executed output. In-graph LLM tasks/resume/parallel scheduling remain [design](GOVERNED-HYBRID-FLOWS.md).

The Chinese section links original source, raw model choice, provenance/lowering, actual generated Tree and per-path behavior. Local artifacts are not promised Git contents. History retains 109 runs, 245 receipted model calls and 23 unreceipted requests, with two receipts missing complete token counts; unknown is not zero cost. All 86 complete reports replay byte-identically under their original code without model calls. Final evidence binds 2919 artifacts; 33+127 earlier immutable artifacts also match. No failed/interrupted checkpoint is overwritten/retried or successes combined across implementations.

Many revisions on these three developer procedures create substantial development-set adaptation bias. Fixed-seed repeats test reproducibility, not independent population accuracy. Read-region acceptance is not whole-Skill success, unseen generalization or production probability. No L1 output/redaction, source script, network write or DSH-only comparison ran. No Git commit/push, default-route change or automatic activation occurred. Stopped before Stage 2's preselected ten known public development Skills from at least six repositories and three domains, to test transfer rather than tune these same examples. Formal cross-cohort and large Runtime A/B gates remain separate.
