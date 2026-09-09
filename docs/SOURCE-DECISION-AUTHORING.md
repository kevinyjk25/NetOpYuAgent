# 检索、构造与执行的边界 / Retrieval, Authoring and Execution Boundaries

## 中文

历史阶段说明：本页保留 C3s 原始结果；当前实现、任务边界修订和后续 9B 失败见 [C3t](SOURCE-OBLIGATION-AUTHORING.md)。本页“下一步”为当时计划，不覆盖当前进展。

更新：2026-09-09，C3s。**机制修复完成，转译有效性仍未通过。** 已分离文本获取与语义审核、增加显式宿主映射和有界回读决策，并展示离线构造与执行权限的区别。但两版真实 9B 都在首轮以缺少凭据停止，均未生成候选。这不是准确率提升；大规模 Runtime 评测仍未解锁。

### 本阶段实际改了什么

| 边界 | 现在如何表示 | 不能推导的结论 |
|---|---|---|
| 获取原文 / 理解原文 | 页索引分别记录历史提交、当前 full/partial/absent、精确区间；请求记录单独列 textDelivery 和实际 deliveryRound | 原文已提交不表示已读懂、语义已审核或依赖闭合 |
| 回读 / 构造或报告缺口 | 再次请求同一页集合时，下一轮同时放入当前页与请求页，进入 candidate/gap_report 决策；最后一轮也必须决策 | 不能强迫生成候选；联合窗口仍受预算限制，超限明确停止 |
| 原文工具 / 宿主工具 | 独立、版本化声明绑定原文操作、原文位置、宿主工具、合同/Schema 摘要、参数名对应和限制 | 参数名对应不是业务等价、参数值、查询构造器或权限 |
| 离线构造 / 实际执行 | authoringBoundary 从已有读取合同展示 access/resourceScopes 和现有执行检查，明确当前不执行且没有权限满足证明 | 不需要真实凭据才能草拟候选，但未知领域前置条件不能自动交给权限检查兜底 |
| 缺口位置 / 缺口正确性 | gap_report 留存原始块、偏移、类别、缺少什么、下一步；同时记录原始条数和完全相同诊断数 | 引用有效不证明缺口成立；“给了停止理由”不算转译成功 |

新增 [source_retrieval](../evaluation/source_retrieval.py) 和 [source_host_binding](../evaluation/source_host_binding.py)，在原 [source_ledger](../evaluation/source_ledger.py) 中接线；没有新执行器、权限放行路径或默认 DSH 路由变化。

`sourceIndex.document` 通过 `sourceDocumentPaths` 恢复完整路径，full 的当前区间就是 `[start,end]`，absent 为空，partial 单列区间。引用 Schema 使用共享 `$ref`。这些只减少重复的元数据/声明，原文、路径、页、note 和允许引用的块不删减。

### 对接一个已有工具

原来的六字段输入包保持不变：`bundle/task/taskOrigin/inputSchema/catalog/reads`。宿主对应声明是可选的独立文件，不能由模型猜测或偷偷混入原始 Skill。结构示意如下；摘要和偏移必须来自实际文件/合同，示意值不能直接通过校验。

```json
{
  "apiVersion": "netopyu.io/source-host-bindings/v1",
  "bindings": [{
    "id": "adapter-operation-v1",
    "sourceBundleDigest": "sha256:实际源包摘要",
    "source": {"path": "SKILL.md", "start": 100, "end": 160, "quote": "实际操作原文"},
    "sourceOperation": "原文操作名",
    "hostTool": "catalog 中的实际工具名",
    "contractHash": "sha256:实际读取合同摘要",
    "inputSchemaDigest": "sha256:实际输入 Schema 摘要",
    "outputSchemaDigest": "sha256:实际输出 Schema 摘要",
    "parameterMap": [{"sourceArgument": "原文参数名", "hostParameter": "宿主顶层参数名"}],
    "scope": "明确支持的有限操作范围",
    "limitations": ["不支持或不声称等价的行为"],
    "reviewKind": "developer_reviewed_adapter_declaration_not_independent_gold"
  }]
}
```

校验原文精确区间、操作/参数字面存在、合同和输入/输出摘要、宿主参数存在、映射不歧义，以及声明来源类型。**字面存在不等于语义等价**，开发者声明也不是独立 Gold；映射从不被执行。没有映射就明确为空，不按名称补猜。

Netdata 本地声明仅把原文 `agents_call_function` 的 `--node/--function/--body` 与既有 `fixture_netdata_function` 的 `node/function/body` 对应；没有改变任务、原宿主 Schema 或读取合同。认证、Cloud 传输、缓存、分页、脱敏和聚合均不由该声明提供；原任务要求 Cloud 而本地只有合成宿主的差异保留为真实限制。没有提供正确 Tree、查询值或审阅答案。

```bash
# 仓库根目录，必须使用新目录；此步骤只检查本地模型身份。
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN \
  --inputs INPUT.json --bindings BINDINGS.json
# 显式最多六次新生成；保留失败，不重试/覆盖已完成检查点。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN \
  --max-new-calls 6 --report-dir NEW_REPORT
# 当前版本的已完成运行：零模型调用回放。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

不传 `--bindings` 仍可用原六字段输入。绑定文件必须是上面的版本化 JSON 对象，不是裸数组。构造入口不收凭据、不加载脚本、不执行 provider。实际执行仍由原 `execute_host_read` 检查精确合同绑定、显式认证身份、最小权限 scope、资源访问策略、参数及结果形状。领域特定的前置条件并不因此自动被覆盖。

### 新的 9B 结果：失败仍然保留

以下是同一份已知 Netdata Skill、相同六字段源包与任务/宿主的开发修订，不是未见集、独立审阅或单变量消融。源包仍为 17 文件、202,930 字符、36 页。模型为本地 `qwen3.5:9b`，`think=false`，温度 0、固定种子；两次均只提交根文 p012。

| 版本 | 实际调用 / 耗时 | 输入 / 输出 token | 输出 | 候选 / 编译读取 / 底层执行 |
|---|---|---|---|---|
| v4：检索状态、宿主映射和缺口出口 | 1 / 44.280 秒 | 7,868 / 335 | 4 条缺口，实际重复同一个“没有 Cloud 凭据”的判断 | 0 / 0 / 0 |
| v5：再明确离线构造/执行边界，整理提示和声明 | 1 / 27.635 秒 | 7,765 / 83 | 1 条 `live identity/credentials` 缺口，仍提前停止 | 0 / 0 / 0 |

[v4 原始报告](../artifacts/translator-v2/source-decision-20260909/report/report.json)、[v4 审阅](../artifacts/translator-v2/source-decision-20260909/review-v4/report.json)、[v5 原始报告](../artifacts/translator-v2/source-decision-20260909/report-v5/report.json)、[v5 审阅](../artifacts/translator-v2/source-decision-20260909/review-v5/report.json)。这些本地 artifact 不随 Git 分发。

v4 的第 2、3 个引用分别涉及节点范围、示例变量，不能证明“凭据不存在”。v5 的引用是源规则，也不是实际宿主状态证明。缺少真实凭据确实阻止实际 Cloud 查询，但**凭据缺失本身不应成为离线生成未激活候选的充分拒绝理由**；完整 Cloud 等价或所有源语义可表达也仍未得到证明。

两次都在回读前停止，所以**真实模型尚未验证“联合窗口成功恢复循环”**；这条分支目前只有机械测试。调用减少和重复缺口减少不能记为语义准确率、完成率或性能提升。准确率保持 `null`，whole-Skill 和 Runtime authority 保持 false。当前仍只有一个模型验证的已知 Skill，不是 2 个新 Skill。

### 回归与复现

- 最终 **273 项定向测试通过；全量 2132 passed + 81 subtests passed（176.33 秒）**；9 个相关 Python 文件 Ruff、diff 和本地文档链接检查通过。
- 覆盖联合窗口原文可拼回、当前/历史/尚未实际交付状态、残缺检查点不重试、假引用不接纳、缺口不是成功、合同/原文/Schema/映射漂移、CLI 版本化文件、离线构造不授予权限。原宿主权限拒绝先于回调的测试继续通过。
- v4、v5 分别用 Git `f0499ec` 加原源码覆盖包隔离回放，零模型调用，报告逐字节一致。v4 覆盖包为 `source-snapshot-v3.tar.gz`（文件修订编号，不是协议 v3），v5 为 `source-snapshot-v5.tar.gz`，都在本阶段 artifact 目录。
- 开发预检曾发现裸数组 CLI 错误，以及联合页/重复元数据预算问题，均在模型调用前修订。未调用的旧 manifest、源码快照和失败说明保留；只有 `netdata-9b-decision-first`、`netdata-9b-authoring-boundary` 两个目录有新模型调用。
- 对之前同一 14-note 联合窗口状态，最终 v5 字节代理为 **40,027 / 40,960**；不丢原文、不提高预算。这是静态资源检查，不是转译成功或精确 tokenizer 认证，更大的窗口仍可能停止。

完整摘要和绑定证据见[阶段证据](../artifacts/translator-v2/source-decision-20260909/evidence-summary/report.json)。源码仍在 dev 未提交；未推送、未合并 master，旧报告/Oracle/基线均未改写。

### 下一步不是继续堆测试数

当前确定的失败是**构造前条件与执行时条件混淆，以及停止依据不足**。可能还受任务中“无权限停止”的表述、长上下文，以及可立即选择的缺口出口影响；尚未做因果隔离，不能断言仅由 9B 或 L0 Schema 引起。

下一步先做“源义务 → 所属阶段 → 证据来源 → 宿主/Schema 能否承载 → 区域候选或有依据的停止”的可审阅类型化决策。对齐源规则、宿主声明和实际状态，不再只追加提示词或要求候选数量。要在多种源材料上检查拒绝是否必要、是否过早停止；不能删除拒绝机制、伪造权限，或把 gap 算通过。之后才扩大公开源转译验证。既有 ≥3 cohort / ≥50 Skill / ≥15 仓库 / ≥8 领域 / ≥600 case 的门禁不变，Runtime 大评测继续关闭。

## English

Historical C3s evidence is retained below. See [C3t](SOURCE-OBLIGATION-AUTHORING.md) for current implementation, task-boundary corrections and subsequent negative 9B results. Historical next steps are not the current plan.

C3s implements source-delivery facts independent of semantic review, explicit digest-bound source/host declarations, bounded joint-window decisions, and a clear inactive-authoring/execution boundary. It changes neither the execution gateway nor default DSH routing. Mechanisms pass regression; **translation effectiveness remains unproven**.

The index distinguishes historical submission, current full/partial/absent visibility and exact intervals. Document-path dictionaries and shared citation-schema references remove duplicate metadata without removing pages, original text, notes or allowed IDs. Repeated page requests join current and requested originals before a candidate-or-gap decision. The unchanged resource cap still applies. A gap is an unverified diagnosis, never a successful translation.

Keep the original six-field packet. The optional versioned `--bindings` object above declares source locations, operations, parameter-name correspondence, exact host contract/schema digests, scope and limitations. It supplies neither parameter values nor a reference candidate. Literal source matches and reviewed declarations do not prove semantic equivalence or grant authority. The Netdata declaration is only a synthetic Function-call correspondence, not Cloud transport, authentication, aggregation, redaction or full-wrapper equivalence.

`authoringBoundary` exposes access/resource declarations from existing read contracts. Missing live credentials prohibit execution but do not by themselves prevent drafting an inactive candidate. Unknown domain-specific prerequisites cannot automatically be delegated to identity/scope gates. The existing `execute_host_read` still enforces explicit identity, exact binding, least privilege and resource policy before callbacks. No provider or source script runs during authoring.

Two fresh **known-source development revisions**, both with local qwen3.5:9b, retain negative results. V4 makes one call in **44.280 seconds**, using **7,868/335 input/output tokens**, and reports four citations repeating one missing-credentials diagnosis. V5 clarifies the authoring boundary and makes one call in **27.635 seconds**, using **7,765/83 tokens**, but still stops for live identity/credentials. Each generates **zero candidates, compiled reads or provider calls**. Source rules and example variables do not establish live credential absence. These are not independent cohorts or a single-variable ablation.

Both runs stop before retrieval, so the joint-window cycle-recovery branch is mechanically tested, **not demonstrated by these model runs**. Lower call counts/duplicate diagnostics are not improved accuracy or efficiency. Semantic accuracy stays null; full-Skill translation and activation remain unproven. This is one known Skill, not two new Skills.

Final validation: **273 targeted tests; 2132 full-suite tests plus 81 subtests in 176.33 seconds**. Changed-code lint, local document links and diff checks pass. Separate Git `f0499ec` plus archived overlays reconstruct both implementations; zero-call isolated replays are byte-identical. Pre-model CLI/metadata-budget defects, unused manifests and earlier failures remain preserved. The final same-state fourteen-note joint-window byte proxy is **40,027/40,960**, a static resource observation, not certified token accounting or translation success. Linked artifacts above are local and are not distributed in Git; changes remain uncommitted and unpushed.

Next use explicit obligation phases and evidence origins before candidate/abstention admission, then test whether early stops are necessary across varied sources. Do not merely add prompts, force candidates, remove refusals or count gaps as success. The larger semantic-generalization gate and Runtime evaluation remain locked.
