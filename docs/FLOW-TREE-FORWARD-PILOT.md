# C3d：9B 层级流程正向实验 / Forward Tree Pilot

## 中文

### 结论与范围

**本批完成，但质量门禁未通过。** 层级编译器离线回归正确，不意味着小模型能正确生成层级提案。本次 4 个首次回答中，3 个能解析成树，**0 个通过完整结构/合同资格检查**；没有合格提案进入逐项语义审查，更没有执行 Runtime。不能把“全部拦截”表述为转译成功或生产安全概率。

这是与 C3c 相同的 **4 个手工已知开发流程 / 1 个库存工具 / 0 个公开 Skill**，不增加独立 Skill 数，不是 holdout、独立人工评测或泛化成功证据。源文与合同不变；authoring 提示及 Schema 改为层级协议，因此也不是单变量性能实验。

入口为 [`evaluation/flow_tree_authoring.py`](../evaluation/flow_tree_authoring.py)。只给目标原文编号、真实宿主合同说明、输入/输出 Schema 和层级 JSON Schema，**未提供手工树、节点图或标准答案**。宿主目标清单为空时，通用 Schema 仍能表达 Effect 候选，但编译门禁会拒绝不存在的目标；这暴露了生成侧能力收口不足。

### 首次结果与定位

| 已知流程 | 完整资格结果 | 原始回答中可见的问题 |
| --- | --- | --- |
| 单次读取 `direct-read` | blocked：宿主 Effect 目标不存在 | 先正确读取；随后在 `/steps/1` 把读结果别名 `inventory_result` 当作 Effect 目标，没有生成只读完成终态 |
| 反向条件 `inverted-branch` | blocked：宿主 Effect 目标不存在 | 相等/不等路径、`site=campus` 和再次读取的数据引用保留正确；但 `/steps/1/when_equal/0` 把 `needs_l1` 当作 Effect 目标，另一分支同样把 `read_path_completed` 当作目标 |
| 缺审批/写能力 `missing-approval-write` | blocked：JSON 截断 | 初始读取后反复生成相同 Effect 候选，达到 2,200 输出 token 上限；`done_reason=length`，未产生完整树；不补齐 JSON、不删重复项重试 |
| 缺脚本前置条件 `unavailable-script-prerequisite` | blocked：终态后不可达代码 | `/steps/0` 已提前读取，随后连续生成 Effect 候选；虽在 `issues` 记录真实脚本缺失，但不能抵消跳过前置依赖的问题 |

上述定位是**当前助手对失败原始回答的诊断**，不是完整合格树的源保真审查。结构合格数为 0，所以 `sourceReviewed=0`；这不是遗漏了 4 项本应进行的合格提案审查，也不能将未执行的语义测量记为 0% 准确率。

共同现象是模型输出把终态混入 Effect。当前证据尚不能区分小模型语义错误、提示/Schema 表达负担和结构化解码器兼容性。不能仅凭这四项断言根因在模型，或断言层级编译设计无效。反向分支的局部接线改善不抵消整体资格失败。

### 协议、成本与证据

- 模型 `qwen3.5:9b`，`think=false`，temperature 0，seed 20260907，context 12,288，输出上限 2,200；模型制品摘要记录在 manifest。
- 每项只请求一次，协议、源/合同、请求和实现摘要在请求前冻结；未重试、未修订原始输出、未激活 L0、未调用业务工具或第三方脚本。
- 四项请求耗时依次 **20.96 / 37.23 / 149.21 / 54.29 秒**，合计 **261.69 秒**；输入/输出 token 合计 **4,652 / 3,584**。
- 计时为客户端 POST 含等待、不含预检/审查/测试。全量回归与模型调用并行，存在资源竞争；不与旧协议计算因果性能提升，也不将四项时延视作 Runtime p50/p95。
- manifest：`sha256:78f95ddaabc771e4b54df3c2fbc2897daebeb01a87c6e29dd7faaa79578af02f`。
- [可重算摘要](benchmarks/flow-tree-9b-summary.json)绑定请求、原始响应、状态和文件摘要。完整原始证据在本地 `artifacts/translator-v2/flow-tree-4-20260907/`；该目录被 Git 忽略，文档摘要本身不是完整可移植证据包。

### 使用与复现

在项目根目录执行；新实验必须用全新的输出目录：

```sh
# 从既有冻结开发清单取得源文/合同；不读取原批模型答案。
.venv/bin/python -m evaluation.flow_tree_authoring freeze artifacts/translator-v2/flow-selected-4-20260907 --output /tmp/tree-pilot-new
.venv/bin/python -m evaluation.flow_tree_authoring run /tmp/tree-pilot-new
.venv/bin/python -m evaluation.flow_tree_authoring report /tmp/tree-pilot-new --output /tmp/tree-pilot-report-new.json

# 已完成本批仅离线重算，不请求模型。
.venv/bin/python -m evaluation.flow_tree_authoring report artifacts/translator-v2/flow-tree-4-20260907 --output /tmp/tree-pilot-replay-new.json
```

完整 checkpoint 重入只验证摘要，不重复请求。中断不明或文件被改动则拒绝自动重试。`report --reviews REVIEW_DIR` 可以对合格树加载 `CASE_ID.json` 完整逐项审查，再按整树摘要验证；始终不授予运行权限。本批没有合格树，因此没有生成虚假的“全部支持”审查文件。

新增 **13 项协议回归**，与树编译回归合计 **73 项**通过；全量 **1056 tests + 81 subtests** 通过（230.10 秒，和模型并行）。Ruff、diff 校验、新报告及旧 C3c/离线树示例重放通过。测试使用手工树作为测试夹具不计入真实 9B 成果，也没有将其传给本批模型。

### 下一步：先验证协议可表达性，再改善生成

后续 [C3e 协议探针与宿主能力收口](FLOW-TREE-PROTOCOL-CANARY.md)已完成下述前两项的实现和最小诊断；新收口版本不给答案的正向小批仍待开展。以下保留本次失败后制定的顺序，不代表将复制探针视作转译验收。

1. 独立冻结最小 Schema/解码兼容性 canary：检查 `end`、`if_equal`、空分支和缺能力停止能否被正确输出。它是协议接线诊断，允许显式指定目标构造器，不计为源文转译准确率；不修改或重跑本批。
2. 基于宿主合同机械收窄可生成能力：无 Effect 目标时不提供 Effect 构造器，有目标时只允许精确宿主 ID；区分读结果别名与目标 ID。不能用用例名或答案规则补业务逻辑，运行前既有合同校验仍保留。
3. 兼容性和离线约束测试通过后，再另冻正向开发批，分别测引用/结构、语义、缺能力停止、误接受与成本。之后才扩大到跨工具/跨领域的完整 Skill；12 个公开 Skill 整流程和规模化 Runtime A/B 仍未获准开展。

不在此阶段扩展生产工程，或靠放松安全门禁、提高 token 上限和反复重试刷通过率。

## English

The C3d forward pilot is complete, **but the quality gate remains unmet**. Four first-attempt responses produced three parseable trees and zero fully qualified flows. No qualified tree entered claim-by-claim source review (`sourceReviewed=0`), and no Runtime, business tool, script or write was executed. Rejection is not translation success or a production safety probability.

These are the same four known hand-authored inventory flows used in C3c, not four new public Skills or a holdout. The model received target source spans, actual host contracts and a hierarchical Schema, never reference trees or answer graphs. Both the prompt and output protocol changed; this is development diagnosis, not a single-variable causal comparison.

The direct read used its read alias as an Effect target. The inverted branch preserved polarity and read-result wiring, but emitted `needs_l1` and `read_path_completed` as Effect IDs instead of end outcomes. The approval case repeated Effect candidates until the 2,200-token limit and returned incomplete JSON. The missing-script case named the actual missing capability yet read before that prerequisite and appended multiple terminals. Local wiring improvements do not establish whole-flow fidelity.

These raw-output findings are same-assistant diagnostics, not completed semantic reviews. The repeated end/Effect confusion does not by itself distinguish model reasoning, prompt/Schema burden or structured-decoder compatibility. The generic generation Schema still exposes Effect despite an empty host target inventory; the existing compiler correctly rejects unavailable targets.

The frozen protocol uses `qwen3.5:9b`, no thinking, temperature 0, seed 20260907, context 12,288 and a 2,200-output-token limit. Four POST durations were 20.96, 37.23, 149.21 and 54.29 seconds: **261.69 seconds total**, with **4,652 input / 3,584 output tokens**. Concurrent regression tests may compete for resources. These are not Runtime latency quantiles or evidence of performance improvement.

See the [digest-bound replay summary](benchmarks/flow-tree-9b-summary.json). Raw evidence stays locally under `artifacts/translator-v2/flow-tree-4-20260907/`; Git stores the summary, not a complete portable raw bundle. The commands above freeze new inputs, run once and replay offline. Completed receipts prevent duplicate calls; ambiguous or changed checkpoints require investigation, never automatic retry. Optional full reviews bind the entire tree and confer no execution authority.

Thirteen new protocol regressions and all 73 focused tests passed. Full regression: **1056 tests + 81 subtests**, 230.10 seconds alongside model calls; Ruff/diff and old/new evidence replay passed. Test-only reference trees were not model inputs.

Next: separate minimal Schema/decoder canaries from translation evaluation; narrow generation constructors and target IDs mechanically from host capabilities; then freeze a new forward batch without answer-dependent repairs. Canary success is not semantic accuracy. Public whole-Skill generalization and large Runtime A/B remain gated; no production expansion, relaxed gates or retries on this frozen batch.
