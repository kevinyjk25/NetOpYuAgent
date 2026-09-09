# 源义务、候选生成与失败定位 / Source Obligations, Candidate Generation and Failure Diagnosis

## 中文

历史阶段说明：本页保留 C3t 原始结果；后续参数构造修复与新的操作模式缺口见 [C3u](CATALOG-DIRECTED-AUTHORING.md)。本页下一步是历史计划。

更新：2026-09-09，C3t。**修复了输入污染、生成表达式语法和实验任务边界；尚未得到一个通过准入的真实 9B 候选。** 前置义务审阅没有证明收益，保留为显式实验选项，CLI 与 Python API 均默认 `direct`。默认 DSH 路由、Runtime 执行器和权限门禁未改变。

本阶段是同一个已知 Netdata Skill 的 7 次开发实验、12 次真实模型调用，不是 7 个新 Skill。没有独立 Gold，语义准确率仍为 `null`，大规模 Runtime 对比继续关闭。[绑定证据摘要](../artifacts/translator-v2/source-obligations-20260909/evidence-summary/report.json)与下方原始报告是本地 artifact，不随 Git 分发。

### 1. 实际修复与边界

| 环节 | 已实现 | 仍不能保证 |
|---|---|---|
| 源义务审阅（可选） | 单独一轮输出原文块、义务、类别、所属阶段、承载方式、证据来源；允许跨阶段 | 模型可能漏项、错分类；结构校验不是独立语义审核 |
| 审阅输入隔离 | 不向审阅器提供用户任务、输入 Schema、宿主目录及映射说明；只给原文、导航、未满足的执行门禁元数据 | 不能证明所有间接影响消失，或模型已读全原文 |
| 构造输入 | 构造器仍收到未改写的源包/任务/宿主声明；审阅条目只作带出处的辅助信息，解释理由留盘、不重复塞入提示 | 错误辅助信息仍会干扰构造，尚未证明该路径优于直接构造 |
| 缺口回读 | 可选审阅路径中，从缺口的引号/反引号字面提取词，在未提交的本地惰性源页中检索；任务原词优先，最多选 1 页 | 词面命中不是语义补全；没有命中不等于原文没有答案；新窗口仍可能超预算 |
| 生成表达式 | 生成前声明 `literal/reference/object/array/column_rows` 递归语法；拒绝裸参数字典和未定义表达式 | 尚未把每个宿主参数的类型、必填字段、别名作用域完整前移到生成约束 |
| 生成预算 | 每个顶层/分支块最多 8 条；去掉 Schema `title` 注解，保留验证约束、真实属性名及 const/enum 内容 | 不是整棵 Tree 最多 8 条；Runtime 原限制仍为 64；不是语义裁剪许可或必然避免截断 |

代码：[源义务](../evaluation/source_obligations.py)、[本地缺口检索](../evaluation/source_gap_search.py)、[生成 Schema](../evaluation/source_candidate_schema.py)、[入口](../evaluation/source_ledger.py)。旧 [C3s 边界设计](SOURCE-DECISION-AUTHORING.md)仍适用，但其中“下一步”已成为本阶段的历史计划。

`host_gate` 只允许原文授权义务、仅 `execution` 阶段和已声明的精确门禁 ID 对应。所有门禁仍为 `satisfied=false`，不存在由模型判为“已有权限”的路径。源规则、模型解释、宿主声明、实际运行观测是不同证据；本阶段没有实际运行观测。

总模型预算仍为 6 次（含审阅），上下文 49,152、输出 4,096 token；消息加 format 的 UTF-8 字节代理上限 40,960。字节代理不是精确 tokenizer 认证。删除 Schema 注解使同一待构造请求 **41,153→40,717**，没有提高预算或丢源文。更大的回读仍会停止。

### 2. 修正了一个评测设计问题

旧任务要求真实 Cloud 查询和脱敏业务结果，宿主却只是一个合成 Function 读取工具。**Cloud 传输、完整脱敏/聚合缺失是实际宿主能力差异，不能全部归咎于转译器。** 缺少当前凭据不应单独阻止离线草拟，但草拟也不等于这些能力已经存在。

因此新增了一个明确不同的任务：在同一源文和合成宿主上生成未激活的局部候选，真实执行、Cloud 等价和其余业务职责仍未完成。仅 `task` 字段改变，源包、宿主目录、合同、Schema 与显式映射均未更改。[任务差异记录](../artifacts/translator-v2/source-obligations-20260909/aligned-task/task-change.json)。**新旧任务不得合并计算同任务提升；旧失败不覆盖、不改判成功。**

源包仍为 17 文件、202,930 字符、36 页。所有源码/脚本仅作惰性文本，未执行。模型均为本地 `qwen3.5:9b`，`think=false`、温度 0、固定种子。

### 3. 真实结果：失败保留，不用测试数量掩盖

| 开发版本 / 任务 | 调用 | 总请求耗时 | 输入 / 输出 token | 结果 |
|---|---:|---:|---:|---|
| [v6 / 旧任务](../artifacts/translator-v2/source-obligations-20260909/report-v6/report.json) | 2 | 184.980 秒 | 17,362 / 2,544 | 审阅把 12 项全列为 authoring；任务 IP 混入源解释；5 条缺口、无候选 |
| [v7 / 旧任务](../artifacts/translator-v2/source-obligations-20260909/report-v7/report.json) | 2 | 157.849 秒 | 15,542 / 2,297 | 隔离输入后 11 项 execution、1 项 authoring，仍有错分类；3 条缺口、无候选 |
| [v8 / 旧任务](../artifacts/translator-v2/source-obligations-20260909/report-v8/report.json) | 2 | 115.280 秒 | 15,339 / 2,060 | 2 条宿主缺口；缺口字面检索未命中，无候选 |
| [v8 / 新对齐任务](../artifacts/translator-v2/source-obligations-20260909/report-aligned/report.json) | 2 | 241.262 秒 | 15,474 / 5,815 | 输出以 candidate 开始，但构造调用耗尽 4,096 token，截断；不是完整候选 |
| [v9 / 新对齐任务](../artifacts/translator-v2/source-obligations-20260909/report-v9/report.json) | 1 | 58.798 秒 | 5,973 / 1,719 | 加入表达式语法后，构造请求字节代理超限，未调用构造器 |
| [v10 审阅路径 / 新对齐任务](../artifacts/translator-v2/source-obligations-20260909/report-v10/report.json) | 2 | 102.688 秒 | 15,564 / 2,193 | 去掉 Schema 注解后可调用；返回 4 条缺口，选中回读页后下一窗口超预算，未交付该页 |
| [v10 直接路径 / 新对齐任务](../artifacts/translator-v2/source-obligations-20260909/report-direct/report.json) | 1 | 151.381 秒 | 8,099 / 3,380 | 生成完整候选，但未决义务阻止编译；参数和别名也有明显错误 |

各版均 **0 编译通过的区域、0 provider 调用、0 源脚本执行**。v8 截断原响应保留，没有修 JSON、补正确参数或自动重试。v10 的前置路径/直接路径对照保持任务、源包、模型、宿主和生成语法相同，但也改变了辅助 notes、上下文长度、调用数及缺口回读行为；只是一份已知材料的一次可选路径对照，不能据此估计因果准确率或可靠的时延分布。

### 4. 一个可定位的失败候选

[原始 Tree](../artifacts/translator-v2/source-obligations-20260909/netdata-9b-direct-ablation/round-000/tree.json)和[逐参数诊断](../artifacts/translator-v2/source-obligations-20260909/review-direct/report.json)可直接查看。诊断只静态检查 JSON 常量，不执行候选，也不删除 `unresolved` 后再宣称通过。

| 位置 / 现象 | 原因证据 | 该修什么 |
|---|---|---|
| `/unresolved`，4 项 | bundle digest 实际匹配；准入报告里的合并错误消息包含 digest mismatch，但本例首先阻塞在未决义务 | 拆分失败类别；区分区域内部问题、未来执行门禁、外部剩余职责 |
| `/steps/*/bind`，8 次同名读取 | 模型把宿主对应声明 ID 反复作为输出别名；原文段落/义务被机械复制成多次调用 | 编译器负责稳定别名；原文依据和运行步骤分开，不一段一调用 |
| `/steps/*/arguments` 的 node/function | 原宿主要求字符串，模型分别生成 `{id:…}`、`{name:…}` 对象 | 由宿主 Schema 引导值生成，不只约束表达式外壳 |
| selections 的 4 组数组 | 均为空，违反宿主 `minItems=1`；固定业务条件未正确进入查询 | 每个参数携带来源并校验类型/必填/范围；未知就显式保留，不填空占位 |
| 剩余义务解释 | 直接抄入宿主限制中“unchanged user task mentions Cloud”的旧任务措辞 | 宿主声明只表示能力边界，不能冒充当前任务事实；后续声明需避免任务相关描述 |

上述证据说明，**既有 Schema 能拒绝部分坏候选，但生成端还没有稳定遵守这些已有能力**。不能仅归因于 Schema 不够丰富，也不能仅归因于 9B。前置审阅还引入辅助解释错误和预算开销；目前不默认开启它。编译/拒绝机制通过回归，不代表转译质量已解决。

### 5. 使用与复现

```bash
# 六字段原始输入包不变；可选映射沿用 C3s 的版本化文件。
# NEW_RUN 必须不存在；freeze 只做本地模型身份预检，不生成、不执行工具。
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN \
  --inputs INPUT.json --bindings BINDINGS.json --profile direct
# 仅在需要研究该路径时显式使用 --profile obligation_first。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN \
  --max-new-calls 6 --report-dir NEW_REPORT
# 同一实现的检查点零调用回放；默认不会重新生成。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

CLI 和 Python `freeze` 均默认 `direct`。所有实验固定实际 profile，不因后续默认值调整而改变。历史报告用 Git `f0499ec` 加各版本源码覆盖包隔离回放；7 份报告均零模型调用、逐字节一致。源码改变后不能强行用当前实现接续旧 manifest。v6–v10 快照、原始请求/响应、截断/超预算和任务差异全部保留。

验证：**295 项定向测试通过（14.27 秒）；全量 2165 passed + 81 subtests passed（177.20 秒）**。15 个相关 Python 文件 Ruff 通过。第一次收尾回归为 2164 passed、1 failed：文档过早引用尚未生成的汇总文件；该失败已保留，修正文档顺序后全量通过，证据落盘后另验最终链接。摘要绑定 112 份文件，前两阶段 103 份绑定证据内容不变。这些是机械回归和证据文件数量，不是测试 Skill 数量或语义准确率。

### 6. 下一步优先级

1. 先做 **宿主 Schema 引导的参数构造 + 显式值来源 + 稳定别名分配**，把已知结构约束前移；不直接塞入 Netdata 正确答案。
2. 在不放行候选的前提下，汇总独立的类型、别名、未决义务、源语义缺口，避免第一个报错遮挡其他问题。前置审阅保持可选，只有对照证明有益才考虑推广。
3. 已知失败修复后换源/换任务、小批验证，再进入 ≥3 不重叠 cohort / ≥50 Skill / ≥15 仓库 / ≥8 领域 / ≥600 case。不能把本轮反复修订当未见泛化；Runtime 大评测仍未解锁。

## English

This page retains historical C3t results. See [C3u](CATALOG-DIRECTED-AUTHORING.md) for catalog-directed repairs and newly localized operation-mode gaps; historical next steps do not supersede that plan.

C3t repairs source-inspection contamination, exposes the existing expression grammar before generation, and corrects a task/host mismatch. **No real 9B candidate passes admission yet.** Optional front-review has not demonstrated a benefit, so both the CLI and Python API default to `direct`; `obligation_first` is explicitly opt-in. Default DSH routing, execution and authorization gates are unchanged.

The optional inspector receives original source/navigation and unsatisfied execution-gate metadata, not the user task, input schema, host catalog or mapping prose. It retains source-bound multi-phase obligations as unverified model interpretations. Gate correspondence neither establishes semantic equivalence nor satisfies permission. The constructor still receives the original task, source and declarations. Classification remains imperfect; twelve source notes do not constitute complete source review.

Provisional-gap retrieval searches quoted literals in retained inert unread pages, prioritizing literals from the task. It selects at most one page and does not prove absence, resolution or semantic coverage. Explicit `literal/reference/object/array/column_rows` schemas reject plain argument maps but do not yet enforce all tool-specific types or alias scope during generation. Each authoring block is bounded to eight statements; the Runtime limit remains 64. Removing schema `title` annotations preserves validation and literal payloads, reducing the same pending byte proxy 41,153→40,717 without increasing the budget. Larger windows can still be blocked. The byte proxy is not tokenizer certification.

The old task asks for real Cloud execution and sanitized business output, while the host is a synthetic Function primitive. Missing Cloud/full-output support is a genuine host mismatch, not solely a translation defect. A separately labeled new task asks for an inactive local region; only `task` changed. Source, contracts, schemas and mapping are unchanged. **Old failures remain failures, and the two tasks must not be pooled as same-task improvement.**

The table above records seven known-source development variants and twelve actual qwen3.5:9b calls on **one Skill** (17 files, 202,930 characters, 36 pages), not seven new Skills or independent cohorts. V6 exposes task-value contamination and phase errors; v7 reduces the observed contamination but still stops. V8 on the aligned task starts a candidate but truncates at 4,096 output tokens. V9 blocks before construction for resource budget; v10 removes schema annotations but still produces gaps, then blocks a pending retrieval window. The direct v10 path produces a complete candidate in 151.381 seconds, but it is unresolved and invalid. All variants have **zero compiled regions, provider calls and source-script executions**. Semantic accuracy remains null.

The direct candidate has a matching source digest but four unresolved issues, eight repeated read aliases, object literals where the host requires strings, and four empty selection arrays per read despite `minItems=1`. It also copies stale task-specific wording from host limitations. The linked static diagnostic does not execute or modify the candidate. Expression grammar compliance is not contract validity, source entailment or whole-Skill completion.

Front-review versus direct also changes notes, context length, call count and gap retrieval. This is an exploratory optional-path comparison, not an isolated causal accuracy estimate or robust latency distribution. Front-review is therefore not promoted by default. All seven archived implementations replay their recorded reports byte-for-byte without model calls; failed and truncated checkpoints are never retried or replaced.

Validation: **295 targeted tests in 14.27 seconds; 2165 full-suite tests plus 81 subtests in 177.20 seconds**. Fifteen related Python files pass Ruff. The first finalization run retained one failure (2164 passed): documentation linked to a summary before it existed. Correcting the publication order restored the full suite; final links are checked again after evidence generation. The summary binds 112 files; 103 bound files from the preceding two stages are unchanged. Mechanical tests are not Skill counts or production probabilities. Next prioritize catalog-directed arguments, explicit value origins, deterministic alias ownership and independent error localization, then fresh-source validation. The larger semantic-generalization gate and Runtime evaluation remain locked. No commit or push is performed in this stage.
