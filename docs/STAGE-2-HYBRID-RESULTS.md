# 阶段 2：双驱动公开 Skill 验证 / Governed hybrid public-Skill validation

## 中文

状态：2026-09-10，**阶段 2 小批开发验证闭环完成，语义失败仍保留开放**。按[预先固定条件](STAGE-2-HYBRID-VALIDATION.md)，达到了三个有用受限混合示例：**1 个限定请求完成，2 个局部分析可用**，不是三个完整任务成功。[正式泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)仍关闭。

仓内可审查的[机器摘要、逐例意见与草稿摘录](benchmarks/stage2-hybrid-summary.json)随代码保存；完整原始证据仅在本地。

### 最终结果（冻结 v7）

| 指标 | 本轮事实 | 不能推导的结论 |
|---|---|---|
| 固定 Skill / 仓库 / 开发领域 | 10 / 10 / 9 | 不属于未见集 |
| 首次结构通过 | 6/10：5 个读取前段＋1 个纯 L1 | 不是 6 个完整 L0 |
| 有用读取前段获本地试验准入 | 5/10；仅 1 个整请求计划获接受 | 不自动授予执行/语义权限 |
| 真实 9B 执行 | 5 个不同 Skill，各 1 次；图完成 5/5 | 图完成不等于草稿准确 |
| 草稿审阅 | **1 完成 / 2 局部可用 / 2 不接受** | 不得写成 3/5 完整成功或准确率 |
| 实际读取参数 | 5/5 组符合预先冻结的本地隔离夹具 | 不包括被拒候选，不是总体参数准确率 |
| 公开生成图接线检查 | 35/35：5 正常＋30 异常路径，模型替身 | 不是 35 次真实 LLM 成功 |
| 全量回归 / 变更代码 lint | 2630 项＋81 子测试；39 个 Python 文件 Ruff 通过 | 不证明语义泛化或生产安全 |

| Skill | 构造 / 执行结论 | 关键限制与失败 |
|---|---|---|
| Notion | 正确读取＋限定请求完成 | 原文 Markdown 和未批准状态保留；多余环境建议仍有噪声 |
| on-call handoff | 前段准入；真实草稿不接受 | 虽正确保留活动事件，仍编造“昨天”和 Slack 可联系时间，并漏掉升级规则 |
| code-documentation | 前段准入；真实草稿不接受 | 未读配置/代码，却给出未经支持的安装、dev extra、pytest 命令 |
| incident-alert-tickets | 前段准入；**局部比对可用** | 正确保留 AND 条件、不套用未证实 Fix；但没读详情/评论，“lookup complete”表述过度，不算整任务 |
| warehouse-sync tickets | 前段准入；**局部核验/升级建议可用** | 地域、租户、身份均保持未验证，不查数据库；冗余不确定性和地域核验措辞仍需改善 |
| simple-english | 结构上是纯 L1，审阅不接受，未执行 | 错把未来入参/无工具当成无法改写；不计 L0 成功 |
| SNMP | 构造失败，未执行 | 参数绑定格式、相对时间和过滤含义仍有问题 |
| Playwright results | 构造失败，未执行 | 合成 SQL 未经准入，且存在字段/分组问题 |
| Phoenix tracing | 构造失败，未执行 | 字面量不在引用中；不能伪装为有依据参数 |
| Tetragon | 构造失败，未执行 | 使用未提供的脚本页作为依据；脚本没有执行 |

“局部可用”只说明特定读取结果和部分分析有用，**保留措辞缺陷与未完成职责，不足以直接发布为完整答案**。源文保留、同类型参数、Schema 通过、模型自述都不构成语义正确证明。

### 成本与时延

- v7 构造：10 次 9B 调用，67,742 输入 / 3,733 输出 token；请求 p50/p95 为 **12.64 / 23.30 秒**。
- v7 实际混合执行：5 次 9B 调用，29,320 输入 / 2,113 输出 token；含模型的图执行 p50/p95 为 **30.79 / 46.36 秒**，仅 5 个样本。
- v1–v7 构造、一次失败格式诊断、v6/v7 实际执行合计 **84 次本地 9B 调用，537,423 输入 / 52,527 输出 token**；包含全部失败。v63/v65 的更早成本另列，不混算。
- v6 真实草稿审阅为 1/5 可用；v7 为 1 完成＋2 局部可用。五个严格读取片段不变，但提示、输入顺序和 JSON 解码一起变化，不是单因素消融。v7 构造和部分执行还与全量 pytest 并发，缓存/负载未控制；**不能据轮间时延宣称加速收益或 SLO**。

正式未见语义准确率、总体正确/过度停止率、完整职责保留率、完整 L0 转换率和生产成功概率均为**未评估**，不是 0，也不是 100%。

### 测试的是哪种能力

固定的 10 个公开 Skill，来自 10 个仓库、9 个开发者分类领域。原始业务任务、Skill 快照及宿主工具合同不变；公开脚本一律作为惰性文本，不安装、不执行。此批已经参与诊断和多轮修复，是**已知开发集**，不是独立 Gold 或未见集。

```text
原始 Skill / 引用 + 用户任务 + 宿主合同
  → 9B 构造解释性混合提案
  → 本地 Schema / 来源 / 依赖 / 候选权限检查
  → 开发者 AI 审查参数含义、职责、部分边界
  → 宿主按图 / 参数 / 上下文摘要批准本地试验
  → 原 L0 严格读取 → 真实 9B 推理 → 未验证草稿
```

最后一步是 `model_candidate`，不是已验证事实或变更成功。混合图完成只证明调度与合同约束；实际草稿还需单独审查。当前模型节点不能调用工具、改图、扩大权限或执行 Effect。详见[实现接口和限制](GOVERNED-HYBRID-FLOWS.md)。

### 修复轨迹与根因

- **表示能力**：原来的纯严格图不足以表达开放诊断、解释、撰写等职责。新增有界推理、独立候选准入和固定依赖汇合；严格片段仍交给原 Flow 引擎。没有用 LLM 替代必须精确执行的检查。
- **阶段混淆**：编译包装说明混进业务任务，使模型错误地要求“现在”提供未来调用参数。v4 开始精确分离旧包装和独立冻结的业务任务，保存 `task-provenance.json` 的原字符串、摘要和去除的包装；不是按含糊关键词删限制。业务任务未变，author-input 字节及协议版本改变。
- **输入与输出协议**：v1–v3 的输出合同只给了解码器，没有完整展示给模型；v4 修复可见性。v5 去掉重复的大解码 Schema，只保留普通 JSON 生成与模型可见合同，生成后仍用同一严格 Schema、本地编译器检查。所有版本使用原保守字节预算，未把预算失败改写为成功。
- **语义绑定**：`pageId → page_id` 可以正确绑定，但 `environment → label` 等同类型错绑仍会通过类型检查。引用存在也不证明它支持该参数含义。结构正确不能自动准入。
- **开放计算与动态选择**：从列表挑条目、构造路径/SQL、跨节点条件等，不都在当前紧凑 author surface 范围内。不能靠编造 JSON Pointer 或把合成 SQL 伪装成原文常量来填补；需明确保留边界，或下一版本加入可验证的表示能力。
- **开放职责重复转述**：v6 改为窄 `read_prefix` 入口，原始任务直接保留到末端 L1，不让模型重新“翻译”自然语言职责。原提案与固定降低规则分别放在 `plan` / `loweredPlan`；全部已提供原文和未完成边界都保留。当前不是任意混合控制流的自动构造器。
- **参考材料污染现场结论**：v6 的 5 份真实执行草稿只有 Notion 通过本轮质量审阅。交接复制示例且截断；文档猜测版本/API；事件把部分匹配当成修复依据；工单给出越界、过多的动作。v7 把参考资料和本次观察无损分层、实际证据置后，加入通用事实/权限/简洁输出要求并保留本地 Schema 校验。五个严格片段和原始模型转译选择未改变；不能宣称靠 Schema 检出了这些语义错误。

每版首次失败、原始回复、费用、源码快照和检查点保留。v63/v65 的旧负结果见[首批](STAGE-2-PUBLIC-TRANSFER.md)、[表示修复](STAGE-2-REPRESENTATION-REPAIR.md)。本轮多项协议与输入包装变化并存，不能把轮间差值解释成单因素因果收益。

### 如何看证据与复现

本地证据根目录为 `artifacts/translator-v2/stage2-20260910/`，不随 Git 提交。各 `hybrid-v*-preparation` 保存来源、任务、合同和版本快照；`hybrid-v*-run` 保存请求、原始模型回复、回执和编译图；`hybrid-v*-review` 为开发者 AI 的逐份意见。摘要绑定文件内容，不能原地编辑旧版本或在原失败目录重试。

| 用户想查看什么 | 本地文件（相对证据根目录） |
|---|---|
| 原 L1、原始任务、参数和工具合同 | `hybrid-v7-preparation/CASE/author-input.json` |
| L0.5 模型提案、意图与边界 | `hybrid-v7-run/CASE/round-00/choice.json`；失败看 `raw-choice.json` |
| 严格 L0 片段与受控 L1 的组合、固定降低规则、来源映射 | 同目录 `compilation.json` 的 `flow` / `plan` / `loweredPlan` / `sourceTaskMappings` |
| 实际读取、模型成本、返回草稿或阻断节点 | `hybrid-v7-live/CASE/summary/report.json` |
| 草稿为什么可用/不可用 | `hybrid-v7-live-review/CASE/report.json` |

下列命令只对已存在、已审阅的本地证据运行；`NEW_OUTPUT_DIR` 必须换成全新目录，不能复用正式试验目录。重新执行会产生新的模型调用与费用，不属于零调用回放。

```bash
.venv/bin/python -m evaluation.hybrid_live_demo \
  artifacts/translator-v2/stage2-20260910/hybrid-v7-preparation/notion/author-input.json \
  artifacts/translator-v2/stage2-20260910/hybrid-v7-run/notion/round-00/compilation.json \
  artifacts/translator-v2/stage2-20260910/hybrid-v7-local-admission/notion/review.json \
  NEW_OUTPUT_DIR --case notion --max-model-calls 1
```

实际运行入口是 `evaluation.hybrid_live_demo`。它要求原始 packet、完整 compilation 和绑定 `compilationDigest` 的本地审阅文件；不会因为编译通过就自动执行。宿主固定注册模型/配置/回调，以及每个只读工具的权限和隔离资源。默认模型调用预算为 0，调用者须显式设置预算，输出目录必须全新。

`evaluation.hybrid_public_checks` 是公开生成图上的参数、权限、Provider 和模型身份负向接线检查，使用模型替身；它**不计真实 LLM 成功或语言准确率**。`tests/test_governed_hybrid.py` 检查原 Flow 分支、串并行、候选准入、证据时效、必需分支失败和迟到结果。

### 不代表什么

这不是 10 个 Skill 都转为完整确定性 L0，不是原始 Agent 与 Runtime 的新 A/B，不是生产认证。只读原型没有新增 Effect 接线；不能把“未发生写入”当作一次真实写事务安全性试验。开发者 AI 审阅不是外部人工，少量本地推理输出也不提供生产成功概率或稳定的时延 SLO。正式跨 cohort 泛化与大规模 Runtime 对照仍须另行验收。

## English

Status: Stage 2's small development loop is complete on 2026-09-10, with semantic failures explicitly open. The [predeclared criteria](STAGE-2-HYBRID-VALIDATION.md) require three useful limited mixed demonstrations, met as **one fulfilled scoped request plus two useful partial analyses**, not three wholly successful tasks. The [portable summary, reviews and draft excerpts](benchmarks/stage2-hybrid-summary.json) are versioned; raw local artifacts are not.

Frozen v7: ten Skills/ten repositories/nine developer domains; six structural proposals (five read prefixes, one rejected pure L1); five prefixes admitted to local trials; five real model calls and five graph completions. Draft review accepts Notion's scoped request and only partial incident comparison / warehouse triage; handoff and README drafts are rejected for unsupported facts or commands. Incident's “lookup complete” overstatement and warehouse's noisy uncertainties remain defects; neither is a complete answer. All five executed argument sets match the predeclared synthetic fixtures. Public generated-graph checks pass 35/35 (five normal, thirty negative), using model doubles. Full regression passes 2630 tests plus 81 subtests; all 39 changed Python files pass Ruff.

v7 authoring: ten calls, 67,742 input / 3,733 output tokens; request p50/p95 12.64/23.30 seconds. Five actual hybrid runs: 29,320 input / 2,113 output tokens; model-inclusive graph p50/p95 30.79/46.36 seconds (n=5). Across v1–v7 authoring, one failed diagnostic and v6/v7 live trials: 84 actual local 9B calls, 537,423 input / 52,527 output tokens, failures included. Earlier v63/v65 are separate. Some v7 trials overlapped full pytest and caches/load were uncontrolled: no causal speedup or SLO claim. Unseen accuracy, whole-Skill conversion, population stop rates and production probabilities remain unassessed.

The fixed cohort has ten public Skills from ten repositories and nine developer-classified domains. Source snapshots, business tasks and declared host contracts are retained. Public scripts remain inert. This repeatedly inspected cohort is development evidence, not independent Gold or an unseen test.

The pipeline is original Skill/references + caller task + host contracts → real 9B proposal → local schema/provenance/dependency checks → developer-AI semantic review → host digest-bound admission → original strict read engine → real 9B candidate draft. Graph completion is distinct from draft quality and from verified business success. The model receives no tool, graph mutation, scope expansion or Effect authority.

Repairs address missing open reasoning representation, compile-time versus invocation-time confusion, model-invisible output contracts, duplicated schema/context overhead, and source/argument provenance. v4 explicitly separates only the exact legacy authoring wrapper from the independently frozen business task and records its provenance. Packet bytes change; business tasks do not. v5 uses JSON syntax generation with the complete schema shown to the model and strict local validation retained. All original failures, versions, requests and costs survive. Multiple changes coexist, so revision deltas are not a single-variable causal estimate.

v6 retains the exact original task in a final bounded L1 node after a grounded read prefix. The public graph API remains richer than this narrow author surface. Of five actual v6 drafts, only Notion passed developer review: the others copied template facts, invented unsupported requirements/causes, or recommended inappropriate actions. v7 separates reference material from actual task/observations without losing input values, places current evidence last and adds generic factuality/scope/brevity guidance. The five strict regions and original author choices remain unchanged. Schema validation did not detect those semantic errors.

Same-type business misbindings and unsupported dynamic selection/path/SQL synthesis remain meaningful failures. Citations and types do not prove semantics. Local artifacts under `artifacts/translator-v2/stage2-20260910/` are ignored by Git. The live demo requires the original packet, compilation and a digest-bound developer review, an explicit model budget and a new output directory. Synthetic public graph counterfactuals and unit callbacks are never counted as real LLM accuracy.

Inspect the input packet for L1, choice.json for the explanatory L0.5 proposal, compilation.json for original read regions / retained L1 / lowering and source mappings, and live summary plus draft review for actual outputs and their limitations. The command above runs a new local-model trial with compute/token cost against synthetic data, not an offline replay; preserve the original output directories.

This is neither whole-Skill deterministic conversion, a new native-Agent A/B, production certification, nor a real Effect safety trial. Independent human review, unseen generalization and stable latency SLOs remain unproven. Large Runtime comparison and production engineering stay deferred.
