# 源约束账本与受限映射 / Source Ledger and Restricted Mapping

## 中文

### 结论与边界（2026-09-07，C3g）

已完成一个**可校验、可定位的映射协议修复**，尚未证明转译成功率或泛化能力提高。业务流程先固定，编译器提供真实节点位置，9B 再生成来源与约束映射；第二步不能修改工具、参数、条件、顺序、终态或缺能力事项。

本轮只复用 **4 个已知开发流程、1 个库存工具、0 个公开 Skill**。三个独立冻结的协议批次共 12 次本地 `qwen3.5:9b` 调用，无人工修图后重计首次成功。所有结构失败、语义缺口及原始回答保留。同一开发助手做声明审查，**不是独立 Gold 或泛化证据**。

| 协议批次 | 完整结构合格 | 完整声明审查 | 审查支持的未激活流程 | 主要发现 |
|---|---:|---:|---:|---|
| 可选约束：`source-constraint-flow/v1` | 1/4 | 1 项；10 supported / 3 insufficient | 0 | 直接读省略所有约束；其余输出混用文档类型与执行引用 |
| 强制逐段账本：`total-source-ledger-flow/v1` | 1/4 | 1 项；18 supported / 0 insufficient | 0 | 3 项把工具名填为节点路径；脚本案忠实保留缺依赖停止 |
| 固定流程后的受限映射：`immutable-flow-mapping-pass/v1` | 4/4 | 4 项；87 supported / 11 insufficient | 0 | 路径均合法；2 项仍有语义映射问题，另 2 项忠实保留缺能力停止 |

最后一批使用 **C3f 已封存的真实 9B 流程**，不是重新从源文端到端生成的四次首次成功。它只测第二步辅助映射；不能把表中 1/4→4/4 当作可控实验的转译成功率提升。不同协议的声明集合和检查负担不同，不能比较声明比例来算准确率。重叠声明也不是独立样本。

全部提案仍 blocked，Runtime/业务工具/第三方脚本/写入执行均为 **0**。结构合法、来源支持、缺能力正确停止、业务完成是四种不同结果。

### 修复了什么

1. **原文、映射、执行分离。** `sourceArchive` 保存完整源文、位置和摘要；`source_dispositions` 解释每个非标题源段；编译产物另列宿主规则引用及节点映射。`allSourceTextRetained=true` 不代表 `allRequirementsImplemented=true`。
2. **防止静默省略。** 所有非空非标题行都有必填账本项；操作、文档、宿主规则、流程约束、未解决五种类型互斥，混合段可拆多条。未知来源、缺段、空项、文档携带执行引用等被独立 Schema/编译检查拒绝。
3. **由编译器提供真实映射目标。** 第二步只接收已生成的树和节点目录，Schema 枚举真实路径。模型不再需要在生成流程前猜未来节点名。父流程不是参考答案：若无法忠实对应，必须留下未解决项。
4. **来源修订不能改业务。** 可把错误标题引用改到真实操作段；不可改工具、参数、比较值、真假极性、顺序、终态、原始缺能力事项。父树、元数据及执行投影摘要绑定审查；即使最终图未变，修改别名或说明也会使旧审查失效。
5. **保留完整语义审查。** 源要求覆盖、目的与限制、节点/条件/参数/前置依赖，以及每条约束与账本解释均须审查。引用存在不代表支持；宿主规则要引用自己的规则证据。未解决或证据不足继续阻断，全支持也不授予运行权限。

主要实现：[语义载体](../evaluation/flow_semantics.py)、[逐段账本](../evaluation/flow_source_ledger.py)、[受限第二步](../evaluation/flow_mapping.py)。这些是研究 authoring/review 协议，尚未替换产品 DSH 的默认交互入口，也未新增执行器。

### 真实失败如何定位

| 第二步用例 | 审查结果 | 定位与解释 |
|---|---|---|
| direct-read | 17 supported / 6 insufficient | `/constraints/0` 将 documentation 描述为 Enforce；宿主规则组合没有明确覆盖返回形状，并把禁止写/上层调用混入读规则保障。对应账本项也有同一缺口，共 6 个重叠声明，不是 6 个独立错误 |
| inverted-branch | 34 supported / 5 insufficient | `/constraints/1`、`/constraints/3` 将共享限制引用到操作段，账本重复该问题；`/source_dispositions/s0002/0` 说明新增 Report 返回字段，但源操作和所指读取节点不含报告步骤 |
| missing-approval-write | 20 supported / 0 insufficient，仍 blocked | 首读的标题引用已改为真实操作段；审批/写工具缺失事项不变。忠实表达缺口不是获批或配置成功 |
| unavailable-script-prerequisite | 16 supported / 0 insufficient，仍 blocked | 保留缺脚本前置依赖，在读取之前停止；没有运行或补造脚本 |

每项发现都有 `claimId`、`l05Pointer`、`l0Pointer`、来源引用、理由及建议修订。第二步的文档/约束保真不等于规则已经执行；`read_result_shape` 也只能说明返回结构校验，不证明实时设备健康或业务真实性。这里没有关键词自动打分，也没有把模型自评当作准入依据。

### 成本、证据与复现

| 新批次 | 4 次 POST 总耗时 | 输入 / 输出 token |
|---|---:|---:|
| 可选约束 | 172.66 秒 | 14,432 / 1,495 |
| 强制账本 | 251.05 秒 | 18,300 / 2,027 |
| 受限映射（仅第二步） | 228.08 秒 | 13,893 / 2,050 |
| 本阶段新增合计 | 651.79 秒 | 46,625 / 5,572 |

失败成本全部计入；上述是客户端 POST 时间，含等待、不含预检/审查/测试，部分时段并行跑回归，不构成因果性能比较。映射所用 C3f 父流程生成成本另为 **156.61 秒、11,016 / 981 token**；不能把第二步成本称为完整转译成本。这两个历史批次拼合约 384.69 秒，仅为成本记账，不是新端到端基准。

逐项摘要：[可选约束](benchmarks/flow-semantic-c3g-summary.json)、[强制账本](benchmarks/flow-ledger-c3g-summary.json)、[受限映射](benchmarks/flow-mapping-c3g-summary.json)。摘要绑定原始请求、响应、编译结果和审查摘要；完整原始制品在本地 `artifacts/translator-v2/`，该目录被 Git 忽略。仅提交摘要不是完整可移植证据包。

在项目根目录重算已有证据，不会调用模型：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_mapping_pilot report \
  artifacts/translator-v2/flow-mapping-4-20260907 \
  --reviews artifacts/translator-v2/flow-mapping-4-review-20260907 \
  --output /tmp/flow-mapping-c3g-replay.json
```

输出路径必须不存在；源、请求、依赖、实现或完成文件漂移会拒绝重放。`run` 对完整检查点不重复调用，部分/不明检查点拒绝继续。未来新批次须另建目录并冻结，不能重写本次回答。

新增 **68 项定向回归通过**，覆盖全段账本、互斥类型、精确路径、错误来源、规则引用、父流程不变、旧审查失效、未解决项阻断和单次调用/重放防篡改。全量回归结果见[项目进展](PROJECT-STATUS.md)。测试夹具中的全支持审查只测接线，不是语义证据。

### 下一步和剩余限制

下一阶段 C3h：先使用开发集补充逐子句、多来源映射及“工具说明不得扩写用户步骤”的反例回归，再冻结**从源文生成流程到第二步映射的完整新批次**，单独报告两步成本、整体源保真、正确停止与误接受。不再不断修补当前四个封存回答以追求表面通过。

目前源段按非空行划分，不是语义子句解析器；混合段可能导致过宽说明与引用错配。最多 64 节点、16 层嵌套、256 声明和 4,000 字符 purpose；超限拒绝而非截断。规则目录是实现摘要绑定，不是形式化证明或完整传递依赖证明。审阅可信度、真实宿主环境和独立未知集仍未解决。12 个公开 Skill 整流程、大规模 Runtime A/B 与生产工程扩展继续受门禁约束。

## English

### C3g outcome and scope

The milestone is a **checkable, locatable mapping-protocol repair**, not demonstrated translation accuracy or generalization. A compiler enumerates actual nodes from an immutable model-generated flow; a second 9B call supplies source and constraint metadata only. It cannot change tools, arguments, comparisons, polarity, order, outcomes or unresolved issues. Citation changes remain review-bound and grant no execution authority.

Three frozen batches reused four known development flows, one inventory tool and zero public Skills. Optional constraints qualified **1/4** (10 supported / 3 insufficient across the one reviewed proposal). Mandatory per-paragraph ledgers qualified **1/4** (18 supported claims for a correctly stopped missing-script proposal). Restricted mapping qualified **4/4** (87 supported / 11 insufficient across four reviewed proposals). **Zero review-supported inactive flows and zero Runtime/business-tool/script/write executions.**

The last batch maps preserved **C3f real-9B parent proposals**; it is auxiliary second-pass evidence, not four fresh end-to-end successes. Protocols and claim sets differ. Qualification changes and overlapping claim counts are not calibrated accuracy. The current development assistant reviewed all claims; this is neither independent Gold nor unseen generalization evidence.

### Design and findings

The source archive retains full text and digests. A mandatory ledger accounts for every non-heading body line using exclusive operation, documentation, host-rule-reference, flow-reference or unresolved dispositions. Mixed paragraphs can have several entries. Existing full-source, node, parameter, polarity and prerequisite review remains mandatory; metadata explanations receive additional exact-source review.

The compiler supplies real node keys rather than asking the model to invent future paths. Parent-tree, mapping and execution-projection digests invalidate stale reviews even when the compiled graph is unchanged. Correct citations cannot launder incorrect operations. Missing host capability remains a blocker even when its description is fully supported.

Actual second-pass findings:

- Direct read: **17 supported / 6 insufficient**. Documentation claimed enforcement; host references did not explicitly account for result-shape checks and overextended read rules to write/upper-layer restrictions. Duplicate claim views overlap.
- Inverted branch: **34 / 5**. Shared constraints cited operation paragraphs, and one explanation added a reporting operation absent from the selected source/node.
- Missing approval/write: **20 / 0**, still blocked. The title citation was corrected without changing the read or missing-capability stop.
- Missing prerequisite script: **16 / 0**, still blocked. The unavailable prerequisite remains visible and no read/script is executed.

Each finding exposes claim and L0.5/L0 pointers, source evidence, rationale and suggested revision in the [mapping summary](benchmarks/flow-mapping-c3g-summary.json). Retention, mapping, implementation and execution receipts are distinct. Rule references are not proof of live health or business truth.

### Costs, verification and next step

Four-call totals: optional constraints **172.66 s, 14,432/1,495 tokens**; mandatory ledger **251.05 s, 18,300/2,027**; mapping-only **228.08 s, 13,893/2,050**. This phase added **12 calls, 651.79 s, 46,625/5,572 tokens**, including failures. POST time includes waiting and excludes preflight/review/tests; concurrent regression prevents causal timing comparisons. C3f parent generation additionally cost **156.61 s, 11,016/981 tokens**. Adding the two historical costs gives approximately 384.69 s, not a freshly measured end-to-end benchmark.

The command above replays existing evidence without model calls and refuses an existing output path. Complete checkpoints are not retried; drift or partial checkpoints fail closed. Git tracks summaries, while ignored local artifacts hold raw evidence: summaries alone are not portable complete evidence bundles. **68 new focused tests passed**; final suite status is recorded in [Project Status](PROJECT-STATUS.md). Test-only supported-review fixtures are not semantic judgments.

C3h should first add development regressions for clause-level/multi-source mappings and tool-description leakage, then freeze a fresh complete source→flow→mapping batch. Preserve present failures rather than tuning/retrying them as first successes. Line-based segmentation, 64-node/16-depth/256-claim/4,000-character limits, review dependence and incomplete transitive implementation evidence remain explicit limitations. Public whole-Skill validation, large Runtime A/B and production engineering remain gated.
