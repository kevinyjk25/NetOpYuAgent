# 转译分层诊断 / Layered Translation Diagnostics

## 中文

本视图由冻结输入的离线诊断生成，不修复答案、不重新评分、不调用模型或执行工具。

**判读边界：已有节点缺证据 ≠ 节点不存在；引文精确 ≠ 原意完整；模型报告缺能力 ≠ 已独立验证；未评估 ≠ 0% 成功。**

这里的要求条目由原模型生成，不是独立审阅的语义义务总表，因此准确率、语义丢失率和置信概率均不提供。

诊断摘要：`sha256:03d420eab1dae852c0b24737c89ea9019ddff320c9e33ddf10128548e21147c0`

原报告摘要：`sha256:e856360c9fa44bd7b4bc143dac8175cf991123182a260813df71bdc5ac2d5607`；原资格保持 **流程 4/4、映射 0/4**。

| 独立诊断类别 | 数量（不是错误概率） |
|---|---:|
| 模型报告宿主缺能力（待核实） | 2 |
| 已有节点缺来源证据 | 4 |
| 目的与自身分类冲突 | 8 |
| 模型标为未解决的要求 | 32 |

### direct-read

最早观测到的失败层：**分类与来源映射**。

| 阶段 | 检查状态 |
|---|---|
| 原文与输入 | mechanically\_valid |
| 流程表达/合同检查 | structurally\_qualified |
| 第一阶段源文审查 | not\_evaluated |
| 分类与来源映射 | failed |
| 映射前后 L0 一致性 | not\_evaluated |
| 完整语义审查 | not\_evaluated |
| Runtime 执行 | not\_evaluated |

| 阻断项 | 定位 | 关联来源 |
|---|---|---|
| 目的与自身分类冲突 | /objective | clause-0006 |
| 已有节点缺来源证据 | /steps/1 | clause-0002 |

下列节点**已经存在并下沉**，缺的是可用来源映射；父节点引用仅作定位线索，不能作为正确性证明。

| L1 原文线索 | L0.5 节点 | L0 位置 | 应检查什么 |
|---|---|---|---|
| Read input device\_id once with read\_inventory\_device, then finish the read path.<br> | /steps/1 (end) | /nodes/1 | 原文是否支持此节点；补忠实证据或标记节点无依据，不自动补图 |

未解决候选：0；声明事项：0。完整逐条解释、建议和来源偏移见同名 JSON。

### inverted-branch

最早观测到的失败层：**分类与来源映射**。

| 阶段 | 检查状态 |
|---|---|
| 原文与输入 | mechanically\_valid |
| 流程表达/合同检查 | structurally\_qualified |
| 第一阶段源文审查 | not\_evaluated |
| 分类与来源映射 | failed |
| 映射前后 L0 一致性 | not\_evaluated |
| 完整语义审查 | not\_evaluated |
| Runtime 执行 | not\_evaluated |

| 阻断项 | 定位 | 关联来源 |
|---|---|---|
| 目的与自身分类冲突 | /objective | clause-0007 |
| 目的与自身分类冲突 | /objective | clause-0008 |
| 目的与自身分类冲突 | /objective | clause-0010 |
| 目的与自身分类冲突 | /objective | clause-0012 |
| 已有节点缺来源证据 | /steps/1 | clause-0001, clause-0002, clause-0003 |

下列节点**已经存在并下沉**，缺的是可用来源映射；父节点引用仅作定位线索，不能作为正确性证明。

| L1 原文线索 | L0.5 节点 | L0 位置 | 应检查什么 |
|---|---|---|---|
| 以输入 device\_id 读取库存。返回 site 等于 campus 时返回 needs\_l1 等待推理；不相等时以第一次返回的 device\_id 再读一次，然后完成只读路径。<br> | /steps/1 (if\_equal) | /nodes/1 | 原文是否支持此节点；补忠实证据或标记节点无依据，不自动补图 |

未解决候选：7；声明事项：0。完整逐条解释、建议和来源偏移见同名 JSON。

### missing-approval-write

最早观测到的失败层：**分类与来源映射**。

| 阶段 | 检查状态 |
|---|---|
| 原文与输入 | mechanically\_valid |
| 流程表达/合同检查 | structurally\_qualified |
| 第一阶段源文审查 | not\_evaluated |
| 分类与来源映射 | failed |
| 映射前后 L0 一致性 | not\_evaluated |
| 完整语义审查 | not\_evaluated |
| Runtime 执行 | not\_evaluated |

| 阻断项 | 定位 | 关联来源 |
|---|---|---|
| 目的与自身分类冲突 | /objective | clause-0005 |
| 目的与自身分类冲突 | /objective | clause-0012 |
| 已有节点缺来源证据 | /steps/0 |  |

下列节点**已经存在并下沉**，缺的是可用来源映射；父节点引用仅作定位线索，不能作为正确性证明。

| L1 原文线索 | L0.5 节点 | L0 位置 | 应检查什么 |
|---|---|---|---|
| \# 库存后的审批变更 / Approval and change after inventory<br> | /steps/0 (read) | /nodes/0 | 原文是否支持此节点；补忠实证据或标记节点无依据，不自动补图 |

未解决候选：10；声明事项：1。完整逐条解释、建议和来源偏移见同名 JSON。

### unavailable-script-prerequisite

最早观测到的失败层：**分类与来源映射**。

| 阶段 | 检查状态 |
|---|---|
| 原文与输入 | mechanically\_valid |
| 流程表达/合同检查 | structurally\_qualified |
| 第一阶段源文审查 | not\_evaluated |
| 分类与来源映射 | failed |
| 映射前后 L0 一致性 | not\_evaluated |
| 完整语义审查 | not\_evaluated |
| Runtime 执行 | not\_evaluated |

| 阻断项 | 定位 | 关联来源 |
|---|---|---|
| 目的与自身分类冲突 | /objective | clause-0012 |
| 已有节点缺来源证据 | /steps/0 | clause-0005, clause-0006, clause-0007, clause-0008 |

下列节点**已经存在并下沉**，缺的是可用来源映射；父节点引用仅作定位线索，不能作为正确性证明。

| L1 原文线索 | L0.5 节点 | L0 位置 | 应检查什么 |
|---|---|---|---|
| Running scripts/check\_inventory.py is a prerequisite before reading input device\_id. The script and host contract are absent. Stop with the missing prerequisite visible; do not skip it or execute this source text.<br> | /steps/0 (end) | /nodes/0 | 原文是否支持此节点；补忠实证据或标记节点无依据，不自动补图 |

未解决候选：15；声明事项：1。完整逐条解释、建议和来源偏移见同名 JSON。

## English

This is an offline diagnostic view, not a rescore. No model, Runtime, provider or script execution occurs. A missing node citation is not a missing node. Exact text matching is not source entailment. Declared capability gaps and unresolved candidates require independent source/host verification. Downstream not_evaluated is not zero-percent success.

Source/host, tree and mapping digests bind the input. Findings aggregate independent mechanical defects with exact source spans and L0.5/L0 pointers. Optional digest-bound first-pass reviews remain usable even when mapping is blocked. Complete mapping reviews require successful compilation. An assisted witness tests bounded constructibility without replacing or rescoring the original. Projection equality checks mapped versus parent L0, not arbitrary compiler correctness or source fidelity.

No calibrated semantic confidence, semantic-loss rate, independent obligation denominator or automatic admission is claimed. Candidate requirement counts and development fixtures are not independent Gold, public-Skill generalization, or production success probabilities.
