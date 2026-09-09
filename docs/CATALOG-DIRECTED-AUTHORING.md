# 宿主参数约束与可定位构造 / Catalog-Directed, Traceable Authoring

## 中文

更新：2026-09-09，C3u。**在同一个已知 Skill 上，真实 9B 的参数形状错误和别名重复已消除，但完整候选仍不可用。** 流程重复、参数语义来源、操作模式与未决职责仍有错误。本阶段只增加可选研究入口 `catalog_bound`，不改变默认 DSH 路由、原 Runtime 执行器或任何授权门禁；CLI/API 默认仍是 `direct`。

本文保留 C3u 历史结果；后续补齐模式声明及新的失败，见 [C3v 操作模式与行动来源](HOST-OPERATION-MODES.md)。

[阶段证据](../artifacts/translator-v2/source-catalog-20260909/evidence-summary/report.json)保存同任务对照和失败；[实际调用报告](../artifacts/translator-v2/source-catalog-20260909/report-v11/report.json)、[模型原始候选](../artifacts/translator-v2/source-catalog-20260909/netdata-9b-catalog-bound/round-000/choice.json)、[最终静态定位](../artifacts/translator-v2/source-catalog-20260909/reanalysis-final/catalog-lowering.json)均可直接检查。这些 artifact 为本地证据，不随 Git 分发。

### 改了什么

新增 [source_catalog](../evaluation/source_catalog.py)，接入 [source_ledger](../evaluation/source_ledger.py)；未增加一轮模型审阅。

| 层次 | 新机制 | 不等于 |
|---|---|---|
| 参数生成 | 逐工具使用原输入 Schema 约束表达式中的值、类型、必填字段、数组长度、额外字段；本地引用、可空类型、开放属性仍保留 | 所有设备领域约束已表达、业务参数一定正确 |
| 值的出处 | 字面参数带 task 原句、source 块及原句、或宿主 Schema const/enum 指针；保留精确坐标和摘要 | 句中出现了值就证明语义、极性、使用场合正确 |
| 结果引用 | 模型使用读取语句路径，如 `/steps/0`；代码分配 `read_000` 等唯一别名，转换到原编译器 | 自动删除重复读取、允许引用未来/自己/另一分支 |
| 独立诊断 | 参数来源、宿主常量约束、引用作用域、未决义务、缺少终止、终止后语句分别定位；相同读取参数成组提示 | 自动修补参数、删除未决义务、推断成功或给重复调用定罪 |
| 原执行边界 | 成功构造也只送入原 Tree/Flow 编译器，再等待语义审查；动态值仍须逐次验证 | 候选自动激活、审批/权限已满足、源码脚本可以运行 |

表达式仍是原来的 `literal/reference/object/array/column_rows`。代码剥离来源注释并留在独立轨迹里，不把注释作为 API 参数传出。对全常量参数，再用完整原始宿主 Schema 验证组装后的 JSON，防止 `uniqueItems` 等集合约束因不同来源包装被绕过；动态引用仍由原绑定/执行检查校验，不声称完整子类型证明。

来源核验只接受精确可见片段。不把“最近 24 小时”自动变成 `-86400`，也不把 `-1`、`10`、`true` 当成 `1`；计算/语义推导若无显式支持仍是缺口。宿主的 default/examples 不被当作 const/enum 授权值。即使来源位置成立，开发测试也保留“原文某句话能通过字符串类型，却不是合理设备 ID”的反例，明确说明语义门禁仍不可省略。

### 真实 9B 对照

原文、对齐后的离线局部构造任务、宿主目录/合同、显式映射保持完全相同：仍是 **1 个已知 Netdata Skill，17 文件、36 页**。模型 `qwen3.5:9b`，不思考模式、温度 0、同一固定种子；本轮新增 **1 次调用**。下表比较上一阶段 direct 与本轮 catalog_bound，不混入较早的真实 Cloud 业务任务。

| 观察项 | C3t direct | C3u catalog_bound |
|---|---:|---:|
| 完整候选产生 | 1 | 1 |
| 原输入 Schema 校验错误消息 | 56 | 0 |
| 额外重复别名 | 7 | 0 |
| 读取语句 / 不同实参请求 | 8 / 1 | 8 / 1 |
| 编译通过区域 | 0 | 0 |
| 模型请求耗时 | 151.381 秒 | 214.388 秒 |
| 输入 / 输出 token | 8,099 / 3,380 | 7,787 / 3,452 |
| provider / 源脚本执行 | 0 / 0 | 0 / 0 |

56 是同一批重复请求的 Schema 校验消息数量（每份 7 条），不是 56 个独立用例。别名唯一是确定性构造结果，不是模型规划改善。调用更慢，不能声称整体性能提升；单次开发对照也不能估计时延分布。生成 Schema、值来源表示、别名表示和系统提示一起改变，不能据此证明单个机制的因果收益或未见泛化。

**可证明的局部效果：** 此次生成没有再把 node/function 字符串写成对象，4 组筛选数组也不再为空。原工具 Schema、原输入任务没有为本轮改宽或补入参考答案。

### 为什么仍被拒绝

1. **8 次读取的实参完全相同。** 模型仍把源文段落/职责机械转成多次读取；代码只保证引用正确，不擅自去重可能有意义的重复观测。
2. **8 个参数来源失败。** 模型使用整对象 literal，再给出整段任务或描述性原文作为一个出处。这些片段并不是那个 JSON 对象，也不能证明每个字段如何组合而来。后续要改为逐叶参数来源，而非宽泛整段关联。
3. **7 个未决项混淆角色。** 包括执行门禁名称和 `liveCredentialsRequiredForAuthoring` 等字段标签；原输入中这些布尔状态并不表示“当前条件缺失”。不能自动删除它们后编译来伪造通过。
4. **缺少显式结束节点。** 最终诊断独立指出这件事，不再让未决项遮挡结构问题。
5. **Schema 还缺少条件操作约束。** 本轮每个 body 同时带 `info:true` 和 `selections`，并未表达查询时间窗。现有 [宿主 observe 实现](../evaluation/netdata_fixture.py)只允许独立 discovery 或完整 query，明确禁止混用；简单输入 Schema 却接受这个对象。这是静态代码核对，不是本轮已执行设备测试。

原调用时保留 15 项诊断（7 个未决项、8 个来源问题）；最终对同一响应增加缺少终止和重复实参定位，得到 16 项阻塞诊断与 1 组重复提示。该补充是 **零模型调用的静态再分析**，原调用报告不覆盖、不重试。语义准确率保持 `null`，整 Skill 成功和 Runtime authority 均为 false。

### 使用与复现

```bash
# 新目录，不覆盖既有证据。仍用原六字段输入和独立版本化宿主映射。
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN \
  --inputs INPUT.json --bindings BINDINGS.json --profile catalog_bound
.venv/bin/python -m evaluation.source_ledger run NEW_RUN \
  --max-new-calls 6 --report-dir NEW_REPORT
# 同一实现零调用回放；源码漂移需使用原快照，不绕过检查。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

示意参数（不是可直接运行的完整候选）：

```json
{
  "kind": "object",
  "fields": {
    "device_id": {
      "kind": "literal",
      "value": "edge-42",
      "origin": {"kind": "task", "quote": "edge-42"}
    },
    "interface": {"kind": "reference", "source": "/steps/0", "pointer": "/selected_interface"}
  }
}
```

`device_id` 是否存在、值的类型以及 `/selected_interface` 是否真实可引用，仍取决于实际宿主合同和前序输出 Schema。示意值不补进模型实验，引用也不能跨出其支配分支。

首个静态生成 Schema 曾超预算，未发送模型。通过复用相同定义、共享列投影规则、删除不可达 Schema 定义，最终输入字节代理 **39,744 / 40,960**；完整源文、有效约束、const/enum 字面数据均保留，没有增大模型预算。这不是精确 tokenizer 认证，后续回读窗口仍可能超限。

代码与证据：38 项新增机械测试；跨模块 **333 项定向通过（15.40 秒）**，最后的数值出处严格比较另经 38 项定向验证。全量 **2203 passed + 81 subtests passed（178.36 秒）**，17 个相关 Python 文件 Ruff、diff 和文档链接检查通过。原模型实现由 Git `f0499ec` 加 v11 源码覆盖包隔离回放，报告逐字节一致、零模型调用。最终静态诊断源码另存快照，摘要绑定 21 份文件；此前三个阶段 215 份绑定证据不变。这些测试数量不作为语义准确率。

### 下一步

先将宿主 **操作模式/条件参数约束** 作为通用、可审查的接口合同公开，再构造源锚定的 **操作流程骨架 + 逐叶参数来源**。不要再按段落生成调用，也不要把接口实现里的隐藏约束当作转译器已经知道的事实。

不能用写死 Netdata 正确查询、放宽来源、删除必要守卫来换取通过。已知源的流程构造闭合后再换源验证，≥3 不重叠 cohort / ≥50 Skill / ≥15 仓库 / ≥8 领域 / ≥600 case 门禁不变，Runtime 大批量验证仍未解锁。改动保留在 dev，未提交、未推送。

## English

This page retains C3u evidence. See the [C3v operation-mode follow-up](HOST-OPERATION-MODES.md) for subsequent implementation and failures.

C3u adds an opt-in **catalog_bound** research profile. It uses each original host input schema to constrain expression generation, records literal origins, and replaces model-chosen result aliases with compiler-owned names derived from lexical statement paths. It introduces no new executor, permission path, source-script execution or default DSH routing change; the CLI/API default remains direct.

Task/source origins must identify exact visible fragments supporting the literal representation. Host origins may cite const/enum, not default/examples. Location and type validity do not prove semantic suitability. References cannot target future/self/sibling reads. Fully constant arguments are revalidated as assembled JSON, including aggregate constraints; dynamic values still require existing per-instance validation. Independent diagnostics preserve unresolved issues, provenance/type/scope failures and missing/unreachable terminals. Repeated argument groups are warnings, not automatic deletion of observations.

One fresh qwen3.5:9b call uses the **same known Netdata Skill, aligned inactive-authoring task, original host contracts and mappings** as the preceding direct run. The table above records the result: 56 host-schema error messages become zero, and seven duplicate aliases become zero. Both candidates still contain eight identical requests, have zero compiled regions and make zero provider/script calls. The new request takes **214.388 seconds versus 151.381 seconds**, with 7,787/3,452 input/output tokens. This is not an overall performance gain, a latency distribution, independent Gold or unseen generalization. Multiple representation/prompt components change together.

The candidate remains blocked: eight aggregate-literal origins cannot establish leaf-value provenance; seven unresolved entries copy execution-gate/environment labels; no explicit terminal exists; and all requests mix info discovery with query selections while omitting the query time window. Existing host code explicitly disallows that combination, but the simple published JSON Schema accepts it. This is a **declaration/conditional-operation gap found by static inspection**, not an executed host test. Schema validity is not operational validity.

The original run retains fifteen diagnostic issues. A separately saved zero-model-call reanalysis adds the missing terminal and a repeated-arguments group; it does not overwrite the original checkpoint, clear unresolved entries or retry generation. Semantic accuracy remains null and whole-Skill translation/activation remain false.

Use the CLI above with a new directory and explicit --profile catalog_bound. The initial static schema exceeded budget and was never sent. Shared definitions and removal of unreachable schema definitions bring the byte proxy to **39,744/40,960** without removing source text, changing assertions/literal data or increasing the budget. This is not tokenizer certification.

Validation: 38 new mechanical tests; **333 targeted tests (15.40 seconds)**, followed by 38 targeted checks for final strict value-origin comparison. The full suite passes **2203 tests plus 81 subtests (178.36 seconds)**; seventeen related Python files pass Ruff, with diff and document-link checks. The original model implementation replays byte-for-byte from Git f0499ec plus its archived overlay with zero model calls. The final diagnostic source is separately archived; the summary binds 21 files and 215 preceding evidence files remain unchanged. Next expose generic operation-mode constraints and construct a source-anchored operation skeleton with leaf-level parameter provenance, then fresh-source testing. The broad semantic-generalization gate and Runtime evaluation remain locked. No commit or push is performed.
