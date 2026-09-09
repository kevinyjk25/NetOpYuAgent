# 精简映射协议 / Lean Mapping Protocol

## 中文

### 本轮结论（2026-09-07）

**精简实现与新批次验证已完成，生成负担下降；完整转译质量门禁仍未通过。** 同一 4 个已知开发源重新生成流程及精简映射，共 8 次真实 `qwen3.5:9b` 调用，未修图或重试。第一步 4/4 结构合格，第二步 3/4 合格；3 项完整同一助手审查合计 **103 supported / 25 insufficient / 0 contradicted**，0 个审查支持的未激活流程。全部仍 blocked。

| 观察指标 | 上轮双阶段 | 本轮精简双阶段 |
|---|---:|---:|
| 映射结构合格 | 2/4 | 3/4 |
| 映射输出 token | 9,919 | **2,619（减少 73.6%）** |
| 映射输入 token | 19,520 | **25,172（增加 29.0%）** |
| 映射 POST 总耗时 | 811.20 秒 | 413.99 秒 |
| 完整两步 POST 总耗时 | 932.05 秒 | 577.64 秒 |
| 完整两步输入 / 输出 token | 30,536 / 10,900 | 36,188 / 3,600 |
| 完整两步输入＋输出 token | 41,436 | 39,788（仅减少 4.0%） |
| 审查支持的未激活流程 | 0 | 0 |

减少自由输出同时增加了枚举 Schema/目标目录的输入负担，不能仅展示输出下降就声称总体 token 节省 73.6%。耗时含失败和等待、不含预检/审查/回归；部分回归并行，机器负载、载荷和输出预算不同，**这是观察到的成本变化，不是严格因果性能或泛化准确率证明**。本轮第一步共 163.65 秒、11,016 / 981 token；第二步 413.99 秒。

本轮仅 **4 已知流程 / 1 工具 / 0 公开 Skill**，同一开发助手非独立审查。声明重叠且三项与上轮两项审查集合不同，不能把声明比例算成准确率。0 Runtime/业务工具/脚本/写执行；所有问题都是候选转译及映射问题，不是已经发生的网络故障，也不能据此认定 Runtime 执行器不安全。

### 真实问题及位置

| 用例 | 本轮结果 | 剩余缺口 |
|---|---|---|
| direct-read | 29 supported / 7 insufficient | 目的混入限制；片段 6 数据背景误选读操作/流程目标；片段 4/7 的检查组合漏返回形状验证 |
| inverted-branch | 结构阻断，语义未计分 | 已选真/假分支内部节点，却遗漏条件节点 `/steps/1` 的来源；编译器拒绝猜补 |
| missing-approval-write | 38 supported / 9 insufficient，仍有真实缺能力事项 | 把“读取/推理不是审批或配置事实”映射到读权限规则；英文复合要求仅映射读取部分；具体检查与错误传播混用 |
| unavailable-script-prerequisite | 36 supported / 9 insufficient，仍有真实缺脚本事项 | 把脚本前置/合同缺失映射到 read_access；把计划库存/非实时健康映射到 read_result_shape；未解决条目保留但不能洗白错误规则选择 |

精确 claim、源引用、L0.5/L0 位置和修订建议见[本轮摘要](benchmarks/flow-lean-c3h-summary.json)。原始完整报告位于 `artifacts/translator-v2/flow-lean-4-20260907-report.json`，摘要以 `summaryOfReportDigest` 绑定它。

### C3h 内部纠偏：减少生成负担，不减少审查义务

上一轮[完整双阶段实验](FLOW-TWO-PASS-TRANSLATION.md)第二步耗时 811.20 秒，生成 9,919 token；重复填锚点、自由扩写和约束语义混淆同时存在。本轮不修改旧批次、不补写旧回答，新增独立协议 `compiler-owned-anchor-mapping/v1`。

模型仅输出：

```json
{
  "objective": ["clause-0001"],
  "selections": {
    "clause-0001": {
      "targets": ["operation:/steps/0", "operation:/steps/1"],
      "extra_sources": []
    }
  }
}
```

这是格式示意，不是批次参考答案；真实请求要求全部源片段都有记录，目标集合由实际宿主规则和第一步模型流程产生。

| 信息 | 由谁负责 | 保留的检查 |
|---|---|---|
| 条目所属来源与字符位置 | 编译器，来自账本键及原文 | 不要求模型重复抄自身锚点；原文摘要和位置仍绑定 |
| 额外跨片段来源 | 模型选择 | 必须存在、无重复、不能再写自身；最多 3 个 |
| 操作/流程/宿主规则/文档/未解决目标 | 模型选择 | 只能选真实目录 ID；仍审查所选类型和目标是否符合完整源要求 |
| 解释文字 | 编译器投影 | 只描述候选映射，不添加报告、权限、只读属性或“已校验”结论 |
| 节点来源 | 编译器由选择汇总 | 每个真实节点必须被映射；超过既有来源上限拒绝，不截断或猜测 |
| 语义支持及运行权限 | 完整源审查及独立宿主门禁 | 机械通过不是语义正确；任何未解决项、证据不足或父问题仍阻断；无自动运行授权 |

同类节点/规则选择机械合并为旧结构字段，不能丢弃目标。精简对象、展开对象、父流程、规则目录和源片段全部绑定审查摘要。修改额外来源，即使业务图不变，旧审查也失效。

模型不能再通过自由说明新增“报告全部字段”等业务含义，但仍可能**选错目标**：例如将输入/返回检查映射为 `documentation`、`flow:<unsupported 节点>` 或仅 `read_access`。这些错误仍有逐片段声明，回归验证它们能被证据不足审查阻断。移除文字字段不等于证明模型已正确理解。

### 实现与复现

- [精简模型与编译](../evaluation/flow_lean_mapping.py)：复用旧片段、树/合同编译和完整审查，不改执行器。
- [新双阶段批次入口](../evaluation/flow_lean_pilot.py)：第一步重新生成树，失败跳过第二步；原始回答、成本与完整/部分检查点行为继续留痕。
- [映射回归](../tests/test_flow_lean_mapping.py)、[批次回归](../tests/test_flow_lean_pilot.py)：23 项新增回归，不是新增 Skill 或语义 Gold。

本地批次根目录为 `artifacts/translator-v2/flow-lean-4-20260907`。读取已完成报告，无模型调用：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_lean_pilot report \
  artifacts/translator-v2/flow-lean-4-20260907 \
  --output /tmp/flow-lean-replay.json
```

输出路径必须不存在；完整审查以 `--reviews <目录>` 传入。没有审查时不得把 `mappingQualified` 当作语义接受。原始制品在 Git 忽略的本地 `artifacts/`，Git 摘要不是完整证据包。

本轮审查目录为 `artifacts/translator-v2/flow-lean-4-review-20260907`。完整报告重放、完整检查点重入和旧 C3h 报告一致性验证通过，未重复调用模型。23 项新增回归及全量 **1213 tests + 81 subtests** 通过（165.05 秒，部分与模型并行）；Ruff/diff 校验通过。

### 下一步仍在 C3h

不再把主要工作放在压缩 JSON 上。下一轮优先处理**源要求类型与实际保障职责的匹配**：区分业务前置依赖、读取授权、结构合法性、错误传播、数据解释和缺能力停止；补完整条件节点/复合句覆盖的独立反例，同时区分“业务流程本身错误”与“细粒度来源标注不完整”。不能把后者直接称为已执行业务失败，也不能跳过完整源审查自动放行。

目录/Schema 重复造成的输入增长作为次级问题，需在保持同样校验义务下优化；不得为降低 token 删除源要求、规则边界或失败证据。完成离线验证后另冻新批，当前所有输出和审查不改写。C4 公开 Skill 整流程、C5 未知集和 C6 Runtime 价值评测继续受质量门禁约束。

### 边界

这是同一已知开发源的协议纠偏，不是公开 Skill 扩充、未知集泛化或独立 Gold。首次与第二步成本分别计，失败计入；模型种子、机器负载和输出预算影响时延，不能把观察到的耗时变化归为严格因果提升。完整源审查的结论、未解决事项和错误接收必须与 token/结构指标分开。

依然存在标点分段不等于语义分段、混合限制、多来源引用上限、模型误分类、对宿主规则解释错误以及审阅非独立等限制。最多 8 个目标/片段，展开仍受旧协议条目/节点/审查/长度上限约束；不支持的复杂组合必须显式阻断。当前不替换产品默认入口，不解锁大规模 Runtime A/B。

## English

### Measured outcome

**Lean implementation and a fresh batch are complete: generation burden decreased, but semantic acceptance remains unmet.** Eight new real 9B calls on four known flows, one tool and zero public Skills: flow qualification 4/4; mapping qualification 3/4; three complete same-developer-assistant reviews yield **103 supported / 25 insufficient**, zero review-supported inactive flows, all blocked. Overlapping claims and different review sets are not accuracy or independent Gold.

Mapping output fell **9,919→2,619 tokens (−73.6%)**, while input grew **19,520→25,172 (+29.0%)**. Mapping POST time was **811.20→413.99 s**; whole-chain POST **932.05→577.64 s**. Whole-chain input/output changed **30,536/10,900→36,188/3,600**, so total tokens decreased only **41,436→39,788 (−4.0%)**. Do not present output savings as total-token savings. Costs include failures/waiting, exclude preflight/review/tests; differing payloads/budgets and partial concurrent regression prevent causal timing attribution. First-pass generation cost 163.65 s and 11,016/981 tokens. Zero batch Runtime/business-tool/script/write execution.

Direct read retained data-description-as-operation and incomplete rule mappings (29/7). Inverted branch missed the condition node `/steps/1`, despite citing branch leaves, and failed structure. Approval retained real missing capabilities but conflated read authorization with approval/configuration facts and incompletely mapped a compound clause (38/9). Script prerequisite retained the correct stop but selected read-access for script dependency and result-shape checks for inventory provenance (36/9). These are candidate-mapping defects, not observed network failures or proof that the Runtime executor is unsafe. See [exact findings and bound costs](benchmarks/flow-lean-c3h-summary.json).

### Reduce generation burden without removing review obligations

The preceding two-pass batch spent 811.20 s and 9,919 output tokens on mapping. This separate protocol leaves its raw answers and implementation unchanged. The model now selects only business-objective fragments, actual target IDs and necessary additional sources; the object key already supplies its compiler-owned anchor.

The compiler renders candidate-only explanations, merges same-kind explicit references without discarding targets, and derives node citations. Every actual node must have source coverage. Unknown/duplicate sources or targets, repeated own anchors, missing clauses/nodes and excess limits fail closed. No generated prose can add reporting, authorization or read-only guarantees.

Selections can still be semantically wrong. Choosing documentation, an unsupported terminal or a read-access rule for required input/result validation remains a reviewable error. Existing full-source, purpose, node, parameter, polarity, prerequisite and clause claims remain. Parent issues, unresolved selections and insufficient evidence still block; complete support grants no Runtime authority. The lean object, expansion, rules, source and parent all bind the review digest, including metadata-only changes.

### Implementation, verification and limits

See [mapping implementation](../evaluation/flow_lean_mapping.py), [fresh two-pass runner](../evaluation/flow_lean_pilot.py), and the 23 new [mapping](../tests/test_flow_lean_mapping.py)/[runner](../tests/test_flow_lean_pilot.py) regressions. Synthetic supported-review fixtures test plumbing, not semantic Gold. The command above replays completed evidence without model calls; optional `--reviews` supplies a separate digest-bound full review. Existing outputs are not overwritten. Ignored local artifacts contain raw evidence; tracked summaries are not a portable complete evidence bundle.

This is known-development protocol correction, not public-Skill expansion or unseen/independent evidence. Costs include failed phases; different budgets, seeds and machine load prevent strict causal timing attribution. Token reduction, qualification, semantic acceptance, unresolved capabilities and unsafe acceptance require separate reporting. Punctuation segmentation, mixed requirements, source/claim limits and model/reviewer errors remain. Eight targets per fragment and inherited expansion limits are enforced without truncation. No product-default switch or large Runtime A/B is authorized by these mechanics.

Full new-report replay, completed-checkpoint re-entry without calls and unchanged previous C3h replay passed. **23 new regressions; 1213 tests + 81 subtests passed in 165.05 s**, partly concurrent with generation, plus Ruff/diff checks. Reviews are in local `flow-lean-4-review-20260907`.

Next remains inside C3h: prioritize source-requirement types versus actual guarantee responsibilities (business prerequisites, read authorization, shape validation, error propagation, interpretation and missing-capability stops), plus complete condition/compound-clause coverage. Distinguish wrong business flow from incomplete fine-grained annotations without dropping source review or automatically accepting either. Optimize duplicated Schema/catalog input secondarily without deleting obligations. Preserve this batch; freeze a new one only after offline checks. C4–C6 remain gated.
