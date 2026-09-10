# 阶段 2：表示与导航修复 / Representation and Navigation Repair

## 中文

更新：2026-09-10。当前作者版本 **v65**，基于 dev / `43a2b76`，尚未提交。本轮是[第一批负结果](STAGE-2-PUBLIC-TRANSFER.md)后的已知问题回归，不是新增独立 Skill、未见集或生产验证。**阶段 2 泛化门禁仍未通过。**

### 修复了什么

| 已确认限制 | 本轮修改 | 仍然不能证明什么 |
|---|---|---|
| 条件右侧只能写常量，无法比较返回值和用户输入 | 类型化语法 → 惰性解析 → Tree → 同一个 Runtime，全链支持右侧字段/数组长度引用 | 两个类型兼容的字段不一定是业务上应该比较的字段 |
| 输入和返回值的路径混在同一个选择集合 | 解码 Schema 区分 input 路径与工具输出路径；解析器继续按具体工具检查 observation | 多个工具的 observation 归属仍由生成后的词法/类型检查确认，不是解码器自动选择 |
| UTF-8 定长切页可能切断一句话或引用名 | 先合并相邻旧片段，优先完整行，超长行退到空白/UTF-8 字符边界；字节、偏移完整保留 | 超长单行仍可能分段；读到某页不等于理解或保留其全部职责 |
| operation_plan 的 business_gaps 不进入原有来源检索 | 在程序准入前检查缺失诊断，按引用的字面文本/相对路径查找未读惰性源页，并将检索原因带给下一轮 | 找到文件不是解决缺失、业务批准或语义通过；真实宿主缺口仍保留 |

以下仅展示比较节点的两个操作数字段（省略出处和闭合子树），是**手工语法示意**，不是模型测评结果：

```json
{
  "value": {"kind": "field", "source": "obs0", "pointer": "/title"},
  "equals": {"kind": "field", "source": "input", "pointer": "/requestedTitle"}
}
```

这表达 `obs0.title == input.requestedTitle`，不是 `obs0.title == "requestedTitle"`。编译后两侧进入原数据绑定器；右侧改为内部 `reference` 表示，来源与 JSON Pointer 不变。这里展示操作数关系，整个片段不是可执行合同。

Runtime 对两侧都检查：Schema、支配关系、实际数据类型/存在性和证据时效；下游依赖这次分支时，右侧观察证据同样不能过期。比较区分 JSON 布尔与数字（`false` 不等于 `0`），不做隐式字符串转换、eval 或脚本执行。常量分支和 v1 路径保留。

### 冻结与回归范围

- 保留 v63 全批原始报告：7 Skill / 8 次 9B / 1 个结构候选 / 0 个接受区域，不能改分数。
- v64 是修复后的静态预检版本，没有模型调用；v65 补充把检索原因保留到模型请求。两份准备记录及作者差异分别保留。
- 完整十例源包、任务和宿主声明不变；原 40,960 字节代理门槛、4,096 输出 token 和 9B 模型配置不变。
- **资源退步：初始通过从 7/9 降至 6/9。** 新类型约束增加 Schema 体积，Phoenix 从 40,717 增至 41,591 字节而超限；SNMP、Tetragon 仍超限。未放宽预算或删除原文。
- 定向 9B 探针预先选定 Handoff、Playwright、Incident，每例最多 4 次调用。预算用尽属于不完整探针，不当作完整首构失败、正确停止或通过；后续只允许保留检查点继续，禁止失败请求原地重试。
- 源脚本、业务 Provider 和设备不执行；本轮 Runtime 检查只使用手工合成单元回调，不能作为公开 Skill 的运行成功率。时延受并行本地回归影响，不是隔离性能。

本机证据：[v64 静态准备](../artifacts/translator-v2/stage2-20260910/preparation-v64/freeze/manifest.json)、[v65 静态准备](../artifacts/translator-v2/stage2-20260910/preparation-v65/freeze/manifest.json)、[三例预选与调用上限](../artifacts/translator-v2/stage2-20260910/repair-probe-v65/freeze/selection.json)。`artifacts/` 被 Git 忽略，不等于已发布备份。

### 结果与后续边界

本轮三例探针已结束，共 **6 次真实 9B 调用，1 个结构候选，0 个可接受区域**。本机三个报告零调用回放全部一致；无失败重试、无未回执请求、无激活或业务工具调用。

| 已知开发例 | 观察到的变化 | 最终结果 |
|---|---|---|
| Handoff | 缺失引用触发真实原文检索；不再立即停止 | 第二轮把模板条目展开为重复读取，耗尽 4096 输出 token，无完整候选 |
| Playwright | 不再把 databasePath 当工具输出；补取原文 | 出现 `databasePath == databasePath` 无意义条件；混入无关示例参数，仍有未解决要求 |
| Incident | 真实生成的动态 RHS 经原引擎成功编译 | `label` 错绑 environment；以证据全文对第一张工单描述的相等替代匹配与分类；语义拒绝 |

Incident 还省略监控 ID/标题/环境匹配、工单评论和部分匹配的处理；直接访问允许为空的 `issues[0]`。即便 Runtime 会阻止空值访问，也不能把这种停止算成正确的“无工单”业务分类。原候选没有裁掉坏分支后计为成功。

成本为 **46,397 输入 / 15,309 输出 token**，请求累计 920.59 秒，单请求 p50/p95 **164.26/231.01 秒**。这是定向三例、不同调用阶段且部分重叠本地回归的请求开销，不是整体性能 A/B 或 Runtime 时延；不与 v63 七例的时延直接比较。

[可随 Git 保存的指标](benchmarks/stage2-representation-repair-summary.json)；[122 份制品绑定的证据](../artifacts/translator-v2/stage2-20260910/evidence-v65/report.json)；[逐项审阅与定位](../artifacts/translator-v2/stage2-20260910/repair-probe-v65/developer-review.json)。审阅由同一开发者 AI 完成，不是独立 Gold；完整参数/职责/正确停止 Oracle 尚缺，指标保留 null。

全量回归 **2566 passed + 81 subtests**；261 项定向检查、35 项文档/边界/修复检查及修改代码 Ruff 通过。阶段 1 的 2919 份和 v63 的 136 份证据摘要全部未变。结论是**机制限制部分修复，语义接受率尚无提升证据**，不是阶段 2 完成。

原文规则、用户任务约束和模型建议仍需要分权表示及语义审阅；本轮**没有**宣称通过引用即可自动证明规则。例如“从 notesPath 读取笔记”来自本次任务，而 Skill 正文中的 `references/details.md` 是编写时的文档引用。二者都可能需要读文本，但不是同一操作，也不能互相充当出处。当前只能给规划节点绑定原文证据，存在把任务要求硬挂到无关 Skill 句子的风险。该诊断不是 9B 唯一原因的证明。

开放式分析/撰写职责不能被“读完”替代；图内受控 LLM、恢复、并行/join 仍是[未来设计](GOVERNED-HYBRID-FLOWS.md)。

下一步按实际诊断调整来源覆盖、约束来源和类型 Schema 的体积；在新的开发批次检查迁移后，再进入跨 cohort 和大规模 Runtime A/B，不把本批反复调试计作泛化提升。

## English

Author v65 is an uncommitted development revision on 43a2b76. This repairs known failures from the first Stage 2 batch; generalization admission remains closed. Dynamic RHS field/array-length references now survive typed syntax, inert parsing, Tree lowering and the same Runtime binder. Both operands retain type, presence, lexical dominance and freshness checks, including downstream control evidence. JSON booleans do not compare equal to numbers. No new executor or execution authority is introduced.

The decoder separates caller-input paths from tool-output paths. Observation-to-specific-tool pairing remains a post-generation lexical/type check, not a semantic choice made by the decoder. Lossless paging rejoins old chunks, prefers complete lines and falls back to whitespace/UTF-8 boundaries for oversized lines. Unverified business gaps can trigger bounded literal/path retrieval of retained unread source before program admission; retrieval is not resolution, approval or acceptance.

The v63 negative evidence remains unchanged. v64 is static preflight only; v65 additionally preserves retrieval context in subsequent requests. The same ten source/task/host packets, 40,960-byte proxy and 4,096-token output budget remain. **Initial admission regresses from 7/9 to 6/9**: Phoenix rises from 40,717 to 41,591 bytes; SNMP and Tetragon still exceed the cap. No source was removed and no limit raised.

The three preselected known-failure probes have finished: six qwen3.5:9b calls, one structurally compiled candidate, zero accepted regions. Handoff retrieves the retained reference but subsequently exhausts output on repeated reads. Playwright avoids the old input/output path error but generates a tautological path comparison and irrelevant example requirements. Incident uses a real dynamic RHS through the existing compiler, yet binds environment as the label and replaces monitor/cause matching with evidence-to-first-description equality. It omits empty-result and positive L1 duties; the candidate is rejected intact, not salvaged into success.

Cost: 46,397 input/15,309 output tokens; summed requests 920.59 seconds; request p50/p95 164.26/231.01 seconds. This selected, partly regression-overlapping probe is not a comparable performance A/B or Runtime latency measure. Three reports replay identically with zero calls; 122 artifacts are bound. See [portable metrics](benchmarks/stage2-representation-repair-summary.json) and [developer review](../artifacts/translator-v2/stage2-20260910/repair-probe-v65/developer-review.json). Missing complete Oracles remain null; this is not independent Gold.

Full regression passes 2566 tests and 81 subtests; 261 targeted checks, 35 publication/boundary/repair checks and changed-code lint pass. All 2919 Stage 1 and 136 v63 bound artifacts remain unchanged. Public scripts and business providers are inert. Synthetic Runtime unit callbacks are separate from public-Skill success. Mechanical limitations are partly repaired; semantic acceptance improvement is not demonstrated and Stage 2 is not complete.

Original rules, caller constraints and model suggestions still need authority-aware representation and review. Open L1 duties must survive read boundaries. In-graph governed reasoning/resume/joins remain design, not implemented features. Preserve immutable evidence, improve mechanisms from diagnostics, then use a new development batch before generalization and large Runtime comparison.
