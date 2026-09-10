# 转译纠偏与闭环计划 / Translation Correction Plan

## 中文

更新：2026-09-10。用户授权完成阶段 1，具备进入阶段 2 条件后停止。有效顺序：**源审查 → 新输入小批 → 机制纠偏与诊断 → 跨 Skill 泛化验证 → Runtime 对照**。不从授权推断合入状态。见[进展](PROJECT-STATUS.md)、[旧计划](TRANSLATION-CORRECTION-HISTORY.md)。

### 不变原则

- 原型优先，生产控制面冻结，Reasoning / Reliability Runtime / Infrastructure 三平面不变。
- L1 原文及真实宿主合同是输入；模型产物是待审解释，不是事实、Gold、授权或执行结果。
- 易用性/通过性不能靠降低安全标准换取；缺能力、未知、源冲突显式保留，源脚本惰性。
- 首次、修订、结构通过、有效片段、正确停止、完整 Skill 分开计数；不改旧失败/Oracle/指纹。

### 已调整设计

1. **退出单条件自动补全。** 旧反事实判断会误解 OR/否定并阻断正确图；保留作诊断。
2. **模型提取语义，代码计算逻辑。** 可读 and/or/not → 白名单结构 → 联合条件 → 原 FlowTree 编译，禁止 eval/exec，不新增运行时。
3. **分歧先定位，不按高分自动择优。** 保留原候选、表达式、具体赋值、引文和节点；修订另存且未激活，未知/范围外不算接受。
4. **前端对齐既有编译约束。** 无效长度/重复引文提前排除，完整原文保留；不据此声称引用语义正确。

目前辅助模块只验证最多四个布尔事实的两阶段读取区域，不覆盖任意多步/循环/数值谓词或完整 Skill。参数、引用角色、宿主真实性及完整源审查仍需独立验证。[实际结果与限制](FLOW-SEMANTIC-TRANSFER.md)。

### 接下来评估什么

**阶段 1 已完成并停止：**[验收报告](STAGE-1-RESULTS.md)记录同版 v62 / 9B 三类已知流程各两次首构，6/6 受审只读区域、50/50 本地路径。下一步才按[固定交接范围](STAGE-1-EXIT.md)预选 10 个公开开发 Skill；当前未执行，不扩大 Runtime A/B。[语义前端](SEMANTIC-PLAN.md)分离原文核对、业务步骤、未来宿主要求、闭合树、来源和填参；[逐项义务对账](SOURCE-DUTY-ACCOUNTING.md)及 C3 阶段保留为历史诊断。已知开发集经过反复调试，这个结果不是完整 Skill 或总体语义准确率。

C3p 已实现[隔离宿主与动态列解码](NETDATA-ISOLATED-VALIDATION.md)，两次原网关读取后仅输出合成本页计数。接下来将明确的可编译段、L1 职责、未支持的分页/完整性区分开，把当前开发者接线映射成可审查候选并接入小批 9B 首次构造；不能继续只增加脚本检查，也不能把手写 demo 算作模型转译通过。原始 Skill 不裁剪，适配和发布策略变化显式保留。完整任务/语义门禁未通过前，不解锁规模 Runtime 比较。

C3o 的[四份任务对齐档案](TASK-SOURCE-ALIGNMENT.md)保留 35 项开发者声明义务、11 个问题与当时 12 条未绑定宿主需求。C3p 针对其中 Netdata 的一部分建立隔离宿主、列解码和单位/页级输出检查，不回填 C3o 成绩，也未解决全部隐私张力及完整任务语义。明确声明的本地评测 adapter 不需真实设备；但不得冒充原厂合同、运行来源脚本或把聚合/推理隐藏成一个工具。源审阅与编译结果始终分开。

此前 C3l 已落实[无损输入与引用补取](TRANSLATION-INTAKE.md)，C3m 已实现[版本化结构化参数绑定](STRUCTURED-DATA-BINDING.md)，C3n 将源锚定 Tree/嵌套参数接入[原共享流程执行器](STRUCTURED-FLOW-WIRING.md)，保持支配关系、资源权限、本地回执时效与候选边界。C3o 在这些基础上绑定了具体任务与未决源义务，但并未完成全部引用/宿主/语义闭合。数据形状不等同集合循环语义，范围外、参数含义、分支极性及 L1 职责继续保留。这四轮没有模型调用或新的语义评分。

分批采集现已处理 60 个冻结候选，保存 53 个 Skill／38 仓库；失败保留，不以旧缓存补数。四份根源文初审发现包外引用、模板占位误报、长源文与结构化读取限制。先做完整引用/任务/宿主对齐，再固定候选版本做小批 9B 转译；不把问题全部归因模型，也不虚构汇总工具隐藏缺口。命令、证据和边界见[公开批次准备](PUBLIC-TRANSLATION-BATCH.md)。本轮不改 Translator 或既有 Oracle 来迎合新库。

- 冻结实现/协议，独立抽样不同 Skill/仓库/领域；分批完成转译资格和源审查，再决定 Runtime/DSH 成本。
- 以 Skill 级适用召回、接受后的源语义正确率、引用蕴含、参数角色正确率为主；报告不安全误接受、过度停止、范围外比例。真值/Schema 通过不能替代它们。
- 分开记录构造、条件提取、编译、审查四类失败及所有调用/token/含失败 p50/p95。AI 审查只是模拟角色，不称真人 Gold。
- 正式门禁仍为 ≥3 cohort、≥50 Skill、≥15 仓库、≥8 领域、≥600 case 及既有质量阈值；通过前不扩大 Runtime A/B。

## English

Stage 1 passed and stopped: the [report](STAGE-1-RESULTS.md) records frozen v62/9B, 6/6 reviewed read regions across three known procedures repeated twice and 50/50 local paths. Stage 2's preselected ten public development Skills have not run; follow the [handoff](STAGE-1-EXIT.md), not large Runtime A/B. The [semantic frontend](SEMANTIC-PLAN.md) separates source review, business outline, future requirements, closed control syntax, provenance and arguments. Earlier C3 accounting remains historical diagnosis. Repeated development-set tuning is not whole-Skill or population accuracy.

C3p now implements an [isolated host and column decoding](NETDATA-ISOLATED-VALIDATION.md), producing synthetic page-only counts through two original-gateway reads. Next separate compilable regions, L1 duties and unsupported pagination/completeness, map developer wiring into reviewable candidates, and connect small 9B first construction. More script checks alone are not the goal; the handwritten demo cannot count as translation success. Keep full Skills and explicit adaptation/publication changes. Large Runtime evaluation remains locked.

C3o's [four task-alignment dossiers](TASK-SOURCE-ALIGNMENT.md) retain 35 declared obligations, eleven findings and twelve then-unbound host needs. C3p addresses part of Netdata with an isolated host, column decoding and unit/page checks, without revising historical scores or closing all privacy/whole-task semantics. A disclosed local evaluation adapter needs no live devices but cannot impersonate vendor interfaces, execute source scripts or hide aggregation/reasoning in tools. Source review and compilation remain distinct.

The user authorized progress up to the bulk-validation decision point. The route is source audit, fresh-input small probes, mechanism correction, cross-Skill validation, then Runtime comparison. Prototype-first scope, inert sources, inactive proposals and immutable evidence remain mandatory.

Unary necessary-guard addition is no longer recommended. The model extracts readable Boolean conditions; code evaluates and compiles them using existing FlowTree semantics. Disagreements and unknowns remain explicit; revisions are separate and inactive. Citation constraints mirror compiler structure, not semantic entailment.

The helper covers up to four Boolean facts in a two-stage read region, not arbitrary workflows or complete Skills. Citation roles, parameter meanings and full-source review remain open. [Evidence](FLOW-SEMANTIC-TRANSFER.md) preserves failures and scope.

After the current task/authoring closure, freeze a separate public-Skill evaluation protocol. Measure eligible-Skill recall, semantic precision, citation entailment, parameter accuracy, unsafe acceptance, over-stops and unsupported scope; separate all four failure layers and costs. The unchanged [gate](TRANSLATION-GENERALIZATION-GATE.md) requires three cohorts, 50 Skills, 15 repositories, eight domains and 600 cases plus quality thresholds before large Runtime comparison.

[Public acquisition](PUBLIC-TRANSLATION-BATCH.md) processed 60 frozen candidates and saved 53 Skills/38 repositories, retaining failures. C3l added [lossless intake](TRANSLATION-INTAKE.md), C3m implemented [structured bindings](STRUCTURED-DATA-BINDING.md), and C3n connected nested data to the [shared executor](STRUCTURED-FLOW-WIRING.md). C3o now binds concrete tasks and unresolved source duties, but does not complete reference/host/semantic closure. Parameter meanings, branch polarity, L1 duties and unsupported scope remain explicit. Collection loops are separate; none of these four rounds added model calls or semantic scores. Synthetic catalogs must not masquerade as public-Skill hosts. Historical oracles and failures are unchanged.
