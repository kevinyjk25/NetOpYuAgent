# 完整双阶段转译 / Fresh Two-Pass Translation

## 中文

### 范围与设计（C3h，2026-09-07）

本阶段连接两次真实生成：**L1 源文与宿主合同 → 9B 流程树 → 编译器真实节点目录 → 9B 逐片段映射 → 编译及完整源审查**。第一步不提供历史树或参考答案；失败就不进入第二步。第二步不能修改业务流程，映射失败也不触发自动修图或重试。

这是原型研究 authoring 链，不是 DSH 产品入口升级，也不授予 Runtime 执行权。当前只使用已知开发源，公开 Skill、未知集泛化和 Runtime A/B 仍有独立门禁。

### 完整批次结果：链路实现完成，质量门禁未通过

冻结后对同一 4 个已知开发源各执行两次真实 `qwen3.5:9b` 调用；**8 次调用，无重试、无历史树替换、无手工修图**。4/4 第一步完整结构合格，2/4 第二步结构合格。两个合格映射接受同一开发助手完整审查，合计 **53 supported / 15 insufficient / 0 contradicted**；0 个审查支持的未激活流程，全部仍 blocked。声明重叠、不同层次不能算准确率；其余两个结构失败项不伪造语义分数。

| 已知流程 | 第一步耗时 | 第二步耗时 | 最终结果与定位 |
|---|---:|---:|---|
| direct-read | 26.94 秒 | 89.77 秒 | 结构合格；27 supported / 6 insufficient。输入/结果检查被错误传播规则替代；无授权范围被缩窄；纯数据限制混入业务目的 |
| inverted-branch | 40.09 秒 | 304.41 秒 | 13 片段生成 44 记录，31 条没有引用自己的锚点；结构阻断。4,016 输出 token，正常 stop，不能称为截断 |
| missing-approval-write | 26.58 秒 | 233.34 秒 | 15 片段生成 30 记录，15 条缺自身锚点；结构阻断。父提案保留缺审批/写合同事项 |
| unavailable-script-prerequisite | 27.23 秒 | 183.68 秒 | 结构合格；26 supported / 9 insufficient。把缺脚本的 unsupported 节点冒充非法输入/访问/返回检查；部分跨来源解释缺引用，并出现未经证明的“整个操作严格只读”说法 |

客户端 POST 总计 **932.05 秒（约 15 分 32 秒）、30,536 / 10,900 输入/输出 token**。其中第一步共 120.84 秒、11,016 / 981 token；第二步共 **811.20 秒、19,520 / 9,919 token**。包括失败和等待，不含预检/审查/测试；回归与部分生成并行，不能作因果时延比较。第二步成本高且未获得可接收完整转译，不能因引用变细宣称效率或可用性提升。

这是 **4 已知流程 / 1 工具 / 0 公开 Skill**，不是扩大语料或独立未知集。第一步使用既有协议和固定种子，重新调用不代表新增独立实现或新增 Skill。模型阶段 Runtime、业务工具、脚本和写入均为 0。与 C3g 辅助映射的样本链路、声明和输出预算不同，不用 4/4→2/4 推断严格控制变量的准确率下降；当前协议的失败与绝对成本本身已足以阻止解禁。

见[摘要、逐项发现及 46 个缺锚点位置](benchmarks/flow-two-pass-c3h-summary.json)。完整报告绑定源/请求/响应/编译/审查，保存在本地 `artifacts/translator-v2/flow-two-pass-4-20260907-report.json`；摘要中的 `summaryOfReportDigest` 指向该完整报告摘要。

### 相比 C3g 的改动

- 将长段落按标点机械切为带字符偏移的片段。中英文和重复文字各自保留位置；代码围栏及行内代码避免内部标点分割。完整原文仍存档，代码始终是惰性数据。
- 每个片段必须有处理记录，每条记录必须引用自己的片段；若解释涉及其他片段，可加多个来源。每个真实流程节点也可以引用多个片段。
- 操作、文档限制、宿主检查、流程约束和未解决项仍使用互斥类型。工具说明只提供能力/类型依据，不能替用户增加“报告全部字段”等操作。
- 新声明逐条审查片段含义、处理类型和多来源节点支持；原来的整源、目的、节点、参数、极性与前置依赖审查没有移除。
- 为兼容旧编译产物，段落投影明确标记 `compiler_retention_only_not_model_enforcement`。它只是原文载体，**实际模型解释在 `clauseMappings`/`clauseProposal` 中**，不能用投影完整来判定约束已实现。
- 请求、模型、实现、依赖和原始结果冻结；两阶段独立检查点保留失败及成本。部分检查点拒绝恢复，完整检查点只重放，不重复调用。运输失败也保留记录，未知 token 数显式标为不完整。

主要文件：[逐片段映射](../evaluation/flow_clause_mapping.py)、[双阶段 runner](../evaluation/flow_two_pass_pilot.py)。旧 C3f/C3g 文件及证据保持不变。

### 查看和复现

本地批次根目录：`artifacts/translator-v2/flow-two-pass-4-20260907/`。

```text
manifest.json                 源、第一步请求、模型和实现冻结
<case>/flow/response.json      第一次真实回答（不是历史父树）
<case>/flow/tree.json          可解析的业务流程提案
<case>/mapping/request.json    由实际树生成的节点目录和片段 Schema
<case>/mapping/mapping.json    模型片段分类、解释和多来源映射
<case>/mapping/review-input.json  全部声明、来源位置和摘要
```

结构失败时后续文件可以不存在；这不是缺数据应补造的成功。两个阶段均保留 `status.json`、`response.json`、`receipt.json`。审查文件应放在独立目录，完整覆盖所有声明并绑定本次 `inputDigest`。

```bash
# 重算，不调用模型；输出路径必须不存在。
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_two_pass_pilot report \
  artifacts/translator-v2/flow-two-pass-4-20260907 \
  --output /tmp/flow-two-pass-replay.json
```

若已存在完整审查，可加 `--reviews <审查目录>`。无审查时 `sourceReviewed=0`，不能把结构合格当作语义支持。原始大文件位于被 Git 忽略的 `artifacts/`；Git 摘要不是完整可移植原始证据包。

本批完整审查目录为 `artifacts/translator-v2/flow-two-pass-4-review-20260907`。已完成带审查报告重放；29 项新增定向回归、全量 **1190 tests + 81 subtests** 通过（170.75 秒，与模型部分并行），Ruff/diff 及旧 C3f/C3g 证据重放通过。

### C3h 后续纠偏，不进入 C4

目前证据指向两个不同问题：一是协议让模型重复填写已知锚点，并在同一片段下反复扩写别的片段，增加出错面和成本；二是模型仍混淆错误传播、真正校验、缺能力停止和权限含义。不能把两者都归因为模型大小，更不能通过自动补引用或放宽审查隐藏失败。

下一轮先做离线精简：由编译器绑定条目所属来源，模型仅选择必要的额外来源、处理类型与实际目标；减少自由解释和重复双语映射，解释优先由真实引用与映射投影形成。仍须审查类型/目标是否被来源蕴含，不能通过删除解释字段删除安全审查。完成离线反例验证后，另冻新批检查完整保真与生成成本，本批原始结果保持不变。**C3h 实现及首轮评测完成，质量验收仍未完成；C4 不解锁。**

### 验证限制

分段是标点定位，不是语义理解：一句仍可能含多个检查，指代、跨段关系、复杂 Markdown 和工具文档泄漏仍需审查。最多 96 个片段、每片段 4 条记录、每条最多 6 个来源；既有 64 节点、16 层、256 声明、4,000 字符 purpose 上限继续有效，超限拒绝，不截断。精确引用并不证明蕴含，多来源也不能洗白错误操作。

本阶段定向回归已覆盖 29 项：中英文/重复源偏移、代码惰性、空源/超限拒绝、遗漏片段、错误锚点、混合类型、非法节点、旧审查失效、缺能力保留、工具说明扩写、两阶段成本/恢复/运输失败与防重复调用。测试中的全支持审查只验证接线，不是语义 Gold。批次实测及最终回归状态以[项目进展](PROJECT-STATUS.md)为准。

## English

### Scope and changes

The complete frozen batch made **eight real 9B calls on four known flows**, with no retries, historical-tree substitution or manual repair. First-pass qualification: **4/4**; second-pass qualification: **2/4**. Complete same-development-assistant reviews of the two qualified mappings yielded **53 supported / 15 insufficient**, zero review-supported inactive flows, and all outcomes remain blocked. Overlapping claims are not accuracy; structurally failed proposals have no fabricated semantic score.

Direct read: 27 supported / 6 insufficient (incomplete validation-rule mapping, narrowed no-authority language, and data limits classified as objectives). Inverted branch: 44 records for 13 fragments, 31 missing own anchors, 304.41 s mapping and 4,016 output tokens with normal stop, not truncation. Approval/write: 30 records for 15 fragments, 15 missing anchors, 233.34 s mapping. Script prerequisite: 26 supported / 9 insufficient; a correct unsupported stop was misrepresented as input/access/result checking, with unsupported cross-source explanations and a whole-operation read-only claim.

Total POST cost including failures/waiting: **932.05 s, 30,536/10,900 tokens**. Flow generation cost **120.84 s, 11,016/981**; mapping cost **811.20 s, 19,520/9,919**. Preflight/review/tests are excluded; some regression ran concurrently, preventing causal latency comparisons. Four known flows, one tool, zero public Skills and zero batch Runtime/tools/scripts/writes. C3g auxiliary mapping and this fresh two-pass batch differ in chain, claims and token budget, so qualification ratios are not a controlled accuracy comparison. High absolute cost and current failures already preclude advancement. See the [digest-bound summary and findings](benchmarks/flow-two-pass-c3h-summary.json).

C3h connects **source and host contracts → fresh 9B flow → compiler node catalog → fresh 9B clause metadata → compilation and exhaustive source review**. It does not inject historical parent trees into the first call. Failed first passes skip mapping; second passes cannot edit business operations or trigger automatic repair/retry. This is research authoring, not a DSH product entry change or Runtime authority.

Source lines are mechanically segmented at punctuation with exact character offsets. Repeated/bilingual text retains separate locations; fenced and inline code remain inert. Each fragment requires a disposition citing its own anchor, with additional fragments available for cross-source explanations. Every actual node also accepts multiple fragment citations. Tool documentation supplies capability/type information, never new user intent.

Existing full-source, purpose, node, parameter, polarity and prerequisite reviews remain. Added claims cover fragment fidelity, handling and multi-source node support. The legacy paragraph projection is explicitly `compiler_retention_only_not_model_enforcement`; actual model interpretations live in `clauseMappings`/`clauseProposal`. Retention is not implementation or an execution receipt.

Both phases freeze requests, model identity, implementation, dependencies and raw responses. Complete checkpoints replay without new calls; partial checkpoints fail closed. Transport failures and incomplete token counts are preserved. The command above replays without a model call and refuses an existing output path; optional `--reviews` supplies separately stored digest-bound complete reviews. No review means `sourceReviewed=0`, not semantic acceptance. Ignored local artifacts contain raw evidence; tracked summaries alone are not a portable full evidence bundle.

### Limits and checks

Punctuation segmentation is not semantic parsing: mixed requirements, coreference, complex Markdown and host-description leakage still need review. Limits are 96 fragments, four records per fragment, six sources per record, plus existing 64-node/16-depth/256-claim/4,000-character limits. Oversized inputs fail rather than truncate. Exact or multiple citations cannot prove entailment or justify a wrong operation.

Twenty-nine focused regressions cover exact offsets, inert code, limits, omitted anchors, exclusive types, illegal references, stale reviews, unresolved capabilities, invented reporting, two-pass costs/checkpoints and transport failures. Test-only supported-review fixtures are not semantic Gold. Known-development findings do not unlock public whole-Skill, unseen generalization or large Runtime studies. See [Project Status](PROJECT-STATUS.md) for measured batch and final suite status.

Final regression: **1190 tests + 81 subtests passed in 170.75 s**, partially concurrent with model work; Ruff/diff, complete new report and old C3f/C3g replays passed. Reviews reside in the separate local `flow-two-pass-4-review-20260907` directory.

**C3h implementation and first complete evaluation are finished, but quality acceptance is not; C4 remains gated.** Next simplify redundant model-owned anchor copying and free explanations offline: compiler-owned item anchors, only necessary additional sources/type/targets, and explanations projected from actual evidence. Preserve semantic review obligations, especially validation versus error propagation, unsupported stops and authority. Do not silently repair these frozen outputs or weaken review to raise acceptance. Validate the simplified protocol on a new batch before any generalization claim.
