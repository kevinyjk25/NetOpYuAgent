# 源要求与保障职责 / Source Requirements and Guarantee Responsibilities

## 中文

> 后续真实批次已完成：流程 4/4、映射 0/4 合格，未证明可用性提升。新增类型暴露了协议表达缺口和模型分类问题，详见[真实结果与纠偏方向](FLOW-RESPONSIBILITY-PILOT.md)。以下“待验证”是本协议离线里程碑当时的状态。

### 当前结果：离线修复完成，模型收益待验证

2026-09-07，C3h 新增独立实验协议 `typed-requirement-responsibility/v1`。它在[精简映射](FLOW-LEAN-MAPPING.md)之上，把一句话内的不同要求显式拆开，再检查各要求声明的类型与目标职责是否兼容。**没有修改任何旧协议、模型回答或冻结报告，没有切换产品默认入口或改变 Runtime 执行器。**

本轮新增 **75 项开发回归**，与旧精简映射/批次回归合计 **98 项通过**。这些包含参数化正反例，不是 75 个 Skill、独立 Gold 或转译准确率。新协议未调用模型、业务 Provider、脚本或写操作；全量回归仍包含已有的隔离执行测试。上一批 9B 的结构合格 3/4、语义全阻断仍是最新模型实测结果。新版 9B 质量、输出长度和总成本尚未测量。

全量 **1288 tests + 81 subtests 通过（131.58 秒）**；Ruff、`git diff --check` 通过。旧精简批次及完整双阶段批次的完整报告（含审查）重放一致，未发生新模型调用或修改冻结证据。开发回归耗时不是 Runtime/LLM 性能指标。

### 为什么需要这一层

原先要求模型从目录里选择目标，能防止虚构目标，却不能防止职责错配。现在区分三件事：

1. **原文到底要求什么**：模型提出带精确引文的子要求和类型，仍需审查。
2. **所选目标是否可能承担该类型的职责**：编译器按实际节点/规则目录做必要条件检查。
3. **它是否真正实现了完整原意**：继续审查完整原文、子要求、参数、条件极性、前置条件和目标；必要条件通过不是充分证明。

| 声明的要求类型 | 本实验允许的候选目标 | 不允许冒充它的目标 |
|---|---|---|
| 操作 | 实际 read、effect candidate 或 read-path-completed 节点 | 数据背景、unsupported 终态 |
| 条件分支 | 实际 if_equal 节点 | 仅引用分支内部叶节点 |
| 输入形状 | read_input_shape | read_access、错误传播 |
| 读权限 | read_access | 业务审批或配置事实 |
| 返回形状 | read_result_shape | 数据实时性、来源可信或健康事实 |
| 错误传播 | observation_error_blocks | 发现错误所需的输入/返回校验 |
| 业务前置 | 实际条件节点，且谓词/顺序必须另审 | 读权限规则、文档说明 |
| 数据解释 / 权限声明边界 | documentation，仅保留解释 | 自动执行、授权或事实证明 |
| 缺能力停止 | 实际 unsupported 终态 | 正常成功终态、输入/权限规则 |
| 无法分类 | unresolved | 任意可执行目标 |

所有类型均可显式选择 `unresolved`，但它阻断接受；与其他目标同时选择也不能被抵消。此表是**当前有限图协议的能力边界**，不是声称所有业务前置都只需 if_equal。不能用现有谓词、步骤顺序或合同完整表达的要求必须 unresolved；不在本轮新造审批/脚本执行能力。

### 复合要求如何保留与定位

对“输入结构无效或返回结构无效时停止”，模型需要保留不同职责，而不是只选一个泛化的“安全检查”：

```json
{
  "exact_quote": "输入结构无效",
  "kind": "input_shape",
  "targets": ["rule:read_input_shape"],
  "extra_sources": []
}
```

这只是子要求格式示意，不是评测参考答案。该句还需返回形状和错误传播子要求。可重叠引用，但每段引文必须在所属原文片段中**精确且唯一出现**；编译器计算 Unicode 字符偏移，不猜偏移或模糊修正。重复短词需改为能唯一定位的较完整引文，不自动选择第一次出现。

审查中保留两层独立问题：

- **整句分解是否完整**：有没有漏掉返回检查、否定词、条件或后续动作？引文覆盖率不是语义覆盖率。
- **每个子要求的类型和目标是否正确**：即使“输入校验→documentation”通过错误的模型类型声明绕过兼容表，也必须审查原文是否支持该分类。

每项记录 `/selections/clause-XXXX/N`、所属原文、精确引文及字符位置、目标 ID 和规则证据。旧的完整源/操作/参数/前置/极性审查只增加、不删除。修改引文而不改执行图，也会使旧审查摘要失效。每个条件节点必须显式有来源；遗漏时错误信息直接给出缺少的节点指针，不自动替模型补齐。

这轮没有新增浏览器视图；结果是可定位的 JSON 审查包。原有视图是否展示新增子要求还需要后续适配，不能把数据结构完成称为 UI 已完成。

### 实现和本地使用

- [协议、请求、投影、编译和审查](../evaluation/flow_responsibility_mapping.py)
- [开发正反例和 CLI 回归](../tests/test_flow_responsibility_mapping.py)

以下路径是输入文件占位，需要提供实际 `FlowSources`、`FlowTree`、模型提案及审查 JSON。三个命令**均离线**，`request` 只生成待发送的 9B 请求，不调用模型；输出目录须已存在，目标文件须不存在。

```bash
.venv/bin/python -m evaluation.flow_responsibility_mapping request \
  sources.json tree.json --output request.json
.venv/bin/python -m evaluation.flow_responsibility_mapping compile \
  sources.json tree.json proposal.json --output compilation.json
.venv/bin/python -m evaluation.flow_responsibility_mapping assess \
  sources.json tree.json proposal.json review.json --output assessment.json
```

输出请求使用 `qwen3.5:9b`。Schema 复用共享 Requirement 定义，不再在用户正文重复整份 Schema；**新协议引文和类型也增加生成负担，不能据此预宣称更快或更省 token**。现有完整两步 runner 保持冻结，本轮不把新请求塞入旧检查点或复用旧回答当作新输出。

上限：每片段最多 6 个子要求，每个最多 8 个目标、3 个额外来源；合并后仍受旧协议每片段 8 个目标/3 个额外来源、每节点 6 个来源和完整审查 256 声明等限制。超限拒绝，不截断。代码块只能 unclassified/unresolved，仍可引用但不会执行。未解决事项及父流程真实缺能力问题始终保留；全部支持也只得到未激活候选，不授予 Runtime 权限。

### 下一步与验收

仍在 C3h，不进入 C4 或大规模 Runtime A/B。下一步为独立冻结的**新版真实 9B 完整双阶段小批**，同时记录：

- 结构/合同资格、职责错配拒绝、节点来源缺失；
- 完整原意审查、漏分解、错分类、错误目标、正确缺能力停止和误接受；
- 两步各自 token/时延以及失败成本，不只展示结构合格率；
- 开发集/独立未知集的证据边界，不把新增规则回归算作模型收益。

尤其要检查是否只是把错误从“选错目标”转移成“先选错类型”。这轮证明的是已声明职责错配可被机械拒绝，以及新的审查链路完整；**尚未证明模型分解、分类或泛化提升**。不通过就继续分析抽象缺口，不能改旧答案刷通过率，也不能通过删除审查标准放行。

## English

> Subsequent real-model validation is complete: 4/4 flows but 0/4 mappings qualified, with no usability gain demonstrated. See [results and protocol/model diagnosis](FLOW-RESPONSIBILITY-PILOT.md). Pending-model statements below describe the earlier offline milestone.

### Offline milestone, not a new model result

The separate `typed-requirement-responsibility/v1` experiment adds exact-quote atomic requirements and a necessary type-to-target compatibility check over an immutable proposed flow. Seventy-five new development regressions, ninety-eight including the existing lean mapping/pilot tests, passed. Parameterized tests are not Skills, independent Gold or translation accuracy. The new protocol makes no LLM, business-provider, script or write calls; the full regression suite retains existing isolated execution fixtures. The previous real 9B batch remains the latest model evidence: 3/4 mapping qualification, no complete semantic acceptance.

Full regression: **1288 tests + 81 subtests passed in 131.58 seconds**, plus Ruff and diff checks. Both previous lean and complete two-pass reports, including their reviews, replay unchanged without new model calls. Regression duration is not Runtime/LLM performance.

Frozen protocols, answers and reports remain unchanged. No product-default switch or Runtime executor change. Request generation, compilation and assessment are offline CLI operations; the commands above require actual source/tree/proposal/review files and refuse existing output files. Generating a request does not call 9B. The old paired runner is not repurposed for the new protocol.

### Responsibilities and evidence

Input shape, returned shape, read authorization and error propagation name different implementation-bound rules. Business approval/script prerequisites are not read authorization; data freshness/provenance is not returned shape. Requested branches must cite the actual condition node, not just its leaves. Missing-capability stops reference unsupported terminals, never successful completion. Business prerequisites may reference an actual condition only when its predicate and ordering truly support the source; unsupported combinations remain unresolved. This is a bounded experiment, not a claim that all prerequisites reduce to equality.

The model proposes exact uniquely occurring substrings and requirement kinds. The compiler owns Unicode offsets and source anchors, rejects nonexact/ambiguous quotes and incompatible declared responsibilities, and never repairs answers. Overlapping source fragments are allowed for distinct duties; overlap or text retention does not prove completeness. Every clause receives a full decomposition claim, and every atom receives an exact-source/context-bound classification/target claim. Existing full-source, objective, operation, parameter, polarity and prerequisite review remains.

Wrong classification can still evade the compatibility table: a mandatory check mislabeled as documentation remains subject to source review. Dropped duties, cherry-picked negations and wrong predicates remain semantic errors even when types pass. Atom-only changes invalidate old reviews despite identical execution graphs. Unresolved selections and original issues block acceptance; complete support yields an inactive candidate with no Runtime authority.

### Bounds, verification and next step

See the linked implementation/tests for Unicode offsets, compound duties, responsibility cross-products, branch-node coverage, unresolved/issue persistence, review tampering, inert code and no-overwrite CLI checks. Maximum six atoms per clause, eight targets and three extra sources per atom; inherited merged-target/source/node and 256-claim bounds also apply without truncation. Opaque code is unclassified/unresolved. JSON review pointers are available; a new browser presentation has not been implemented.

Shared Schema definitions and removing a duplicated user-message Schema reduce duplication, but quotes/types can increase output burden: no new latency or token-saving claim is justified. Next is a separately frozen real 9B fresh two-pass development batch, reporting failure-inclusive per-phase costs, structural qualification, decomposition/classification/target fidelity, missing-node evidence, correct capability stops and false accepts. C4–C6 remain gated. These tests show mechanical rejection and review plumbing, not improved model understanding or generalization.
