# 转译输入保真与宿主诊断 / Lossless Translation Intake

## 中文

2026-09-08，C3l。**本轮修复的是转译前的源文装载、引用定位和宿主 Schema 诊断，不是宣布整 Skill 转译完成。** 新模块在 `evaluation/`，不改变默认 DSH、L0 执行器或执行授权门禁。

后续进展：C3m 已提供独立的[结构化数据绑定原语](STRUCTURED-DATA-BINDING.md)。本文中的旧合同诊断与本轮结果保留，不回填成新版本成绩。

### 为什么需要这一层

公开 Skill 不一定是一个短 Markdown 加一个扁平工具：它可能引用其他 Skill、脚本、模板和复杂 API。丢失这些输入后再让模型重试，无法区分源文缺失、宿主表达限制和模型语义错误。

现在由 [translation_intake.py](../evaluation/translation_intake.py) 保留原始信息，分别输出：

| 制品 | 内容与边界 |
|---|---|
| `bundle.json` | 每份文件的仓库路径、commit 绑定、字节摘要、原始文本和引用位置；格式变体不被改写，脚本只作数据 |
| `review-pages.json` | 默认每页最多 12,000 字符，记录文件、精确偏移、行号、源摘要和引用 ID；按文件拼回可恢复原文，不静默截断 |
| `host-diagnostic.json`（可选） | 原样保留 MCP 风格 `tools`、输入/输出 Schema、所有约束；以 JSON Pointer 定位现有读取合同不支持的部分 |
| `report.json` | 缺失/歧义引用、宿主是否提供、制品绑定；状态保持 `pending_source_task_host_alignment`，没有转译成绩或执行许可 |

长文本分页解决的是**无损保存和可分段审阅**。当前 `FlowSources` 的 64,000 字符限制与单次模型预算没有被悄悄提高；跨页整 Skill 语义汇总、覆盖审查和编译仍待完成，不能把分页数当作成功转译数。

### 引用不再只有“找到 / 找不到”

- Markdown 模板中的链接保留为 `template_example_candidate`，不混同为确定缺失的业务附件。
- 正文链接与代码中的资源路径分别保留；没有简单地忽略全部代码块。
- 代码中的 `scripts/a.sh` 可能相对仓库根目录，也可能相对 Skill。保留候选路径和歧义，不根据文件名自动猜测。
- 目录引用、普通文件、符号链接目标文字、二进制摘要分开记录；不跟随符号链接。
- 外部 URL、越界路径只记录，不自动访问；引用识别只是词法线索，不是完整 Markdown/代码依赖解析，也不证明所有语义义务都已闭合。

原有可执行包检查器保持保守；不会因为研究入口认为某个链接像模板占位符，就给包恢复执行资格。

### 实际公开材料验证

仍使用 C3k 每批第一个候选的四份源文，不按适配性换样；这四份已经是开发诊断材料，不冒充新未见 Gold。

| 来源 | 这次处理结果 |
|---|---|
| OpenMontage `agents` | 六份文件、65,778 字符，完整保存为八页；没有丢弃嵌套元数据或超限尾部 |
| Netdata `query-snmp-traps` | 原八份文件、31,967 字符；按同一 commit 明确补取八份引用，扩至十六份文件、146,532 字符、二十三页 |
| DeerFlow `code-documentation` | 四个模板链接候选与一个正文链接分开；原严格包门禁未放松 |
| `browser-testing-with-devtools` | 原文完整保存；没有因静态包检查通过而虚构已安装的宿主工具合同 |

Netdata 只补取已列明的文件，不递归爬取整个仓库，不增加采样 Skill 数。第一次补取全部因 Git `ls-tree -l` 大小列的空格解析失败；修复公共解析器后，在同一 commit 重试 **8/8 保存**。失败链和原包均保留：`original-v2 → supplement → supplement-v2`。补入的文档又产生新的引用，因此**没有宣称完整传递引用闭合**。

读完补入的 `_lib.sh`（546 行）后，发现不能将这些 wrapper 笼统映射为只读：

| 源位置 | 实际代码边界 | 对后续合同的要求 |
|---|---|---|
| 62–81 行 | 环境装载通过 shell `source` 执行 `.env` | 不把环境文件当普通 JSON；评测不执行这段逻辑 |
| 201–212、246–300 行 | 申请 bearer，并创建/更新本地凭据缓存 | 业务读取与认证/文件系统效果分别声明；读取名称不等于纯只读 |
| 307–332 行 | 通用 wrapper 接收 HTTP method/path/body，包括写方法 | 不能给通用 HTTP 能力整体贴只读标签 |
| 384–429 行 | 按 cloud/agent 分支切换，agent 分支还要求 host/machine-guid | 条件参数与前置条件必须保留；body 默认值和必填约束不能由模型任意补充 |

以上是开发助手对已固定源码的检查，不是执行实测、独立人工 Gold 或对源码全部安全性的认证。来源中的自测同样未运行。

### 宿主诊断现在做什么、还不做什么

支持直接读取 MCP 风格的 `{"tools": [...]}`。原始嵌套对象、数组、字段大小写、枚举、required 和其他 JSON Schema 约束均保留。不修改字段，不把对象 JSON 字符串化，不补造缺失的 outputSchema，不联网解析 `$ref`。重复工具名、空 catalog 和不完整 Schema 显式报错/报告。

`currentReadSchemaCompatible` **只表示现有 `ReadObjectSchema` 的数据形状兼容**，不验证提供者真实性、只读性、权限或语义。即使为 true，执行权仍为 false。复杂 Schema 目前可无损接收并定位缺口，**还不能因此由旧 L0 读取执行器执行**。

接口示例在 [mcp-catalog.json](../examples/translation-intake/mcp-catalog.json)。这是明确标记的合成接口示例，不是四个公开 Skill 的真实宿主或 Gold，不能拿它替代缺失能力来刷通过率。嵌套 `selections`、数组 `interfaces` 和 `minLength` 等限制会得到具体字段诊断。

### 如何使用

只读已有快照，生成新的输入包（无需 LLM、网络或执行脚本）：

```bash
scripts/netopyu-market-corpus translation-intake \
  artifacts/translator-v2/public-source-20260908-round2/batches/public-02/snapshot \
  netdata-netdata-docs-netdata-ai-skills-query-snmp-traps-skill-md \
  --output-root artifacts/new-netdata-intake
```

提供自己的宿主 catalog：追加 `--host-catalog /path/to/captured-tools.json`。工具名与 Skill 的实际对齐仍需审查，不会因上传 Schema 自动激活合同。

也可使用 `.venv/bin/python -m evaluation.translation_intake SNAPSHOT CANDIDATE_ID --output NEW_DIRECTORY`；两个入口共用实现，输出目录已存在或宿主 JSON 含重复字段时，在可能的网络补取前拒绝。

明确补取同仓库、同 commit 的文件：追加可重复的 `--supplement-path docs/netdata-ai/skills/query-netdata-agents/scripts/_lib.sh`，并使用新的输出目录。只进行无 checkout 的 Git blob 读取，不执行源码或加载真实 `.env`。API `supplement_bundle` 可接受此前的 bundle，保留父摘要和失败链；不会更换已有文件。每次输出必须是新目录，不自动覆盖或重试已完成制品。

本地证据位于 [Netdata 补充包报告](../artifacts/translator-v2/intake-20260908/public-02/supplement-v2/report.json)、[完整原文包](../artifacts/translator-v2/intake-20260908/public-02/supplement-v2/bundle.json)、[分页源文](../artifacts/translator-v2/intake-20260908/public-02/supplement-v2/review-pages.json)；[机器摘要](benchmarks/translation-intake-summary.json)绑定原始/补充源和实现。`original` 是本轮入口开发中的初稿，`original-v2` 是带路径歧义诊断的当前版本；初稿未用于模型评测。

机械复核覆盖四份原始输入的 16 份文本／20 页，按文件拼回与固定快照逐字节一致；补充包的父链、文本与分页摘要通过检查。现有 corpus CLI 在临时新目录离线复现，三个输出文件逐字节一致。源码留存在 [20 文件实现快照](../artifacts/translator-v2/intake-20260908/source-snapshot.tar.gz)，以摘要中的 Git 基线和覆盖清单解释，不包括第三方源码执行许可。制品摘要用于内容一致性检测，不是独立语义证明或提供者身份认证。

### 下一步的实际边界

本轮验收：81 项定向测试通过；全量 **1851 passed + 81 subtests passed，156.60 秒**。18 个变更 Python 文件 Ruff、文档链接、40 份绑定证据与 diff 检查通过。额外使用现有 CLI 读取合成 catalog，确认七个精确缺口、无执行许可和拒绝覆盖。以上是输入/安全边界的机械验证，不是公开 Skill 的语义准确率。

1. 按明确用户任务审查引用角色、宿主原始 Schema 与业务/认证/文件效果，不把整个公开仓库都当必需输入。
2. 为结构化参数建立新版本的有界类型绑定。**数据结构支持**与**集合循环语义支持**分开实现；保留所有约束，不把缺口塞进假汇总工具。旧合同/旧证据不改写。
3. 固定任务、未覆盖义务、Oracle 和候选版本后，执行小批 9B 转译。分开报告片段、整 Skill、正确停止和真实缺口。

本轮没有新 9B 调用、转译成功率或 Runtime 性能成绩；[正式泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)未解锁。

## English

C3l fixes **pre-translation source intake and host-schema diagnosis**, not whole-Skill compilation. The module lives in `evaluation/`; default DSH, Runtime executors and authority gates are unchanged.

Subsequent C3m work provides separate [structured-data binding primitives](STRUCTURED-DATA-BINDING.md). This document's old-contract diagnostics and results remain historical, not retroactively upgraded scores.

`bundle.json` retains file paths, pinned commits, exact bytes/digests, raw text and reference offsets. Scripts are inert, symlinks are not followed, binaries remain hash-only. `review-pages.json` partitions text into at most 12,000-character pages with exact file/offset/line bindings; concatenation reconstructs each original without truncation. Pagination does not remove FlowSources' existing 64,000-character bound or prove cross-page semantic coverage. Optional `host-diagnostic.json` retains raw MCP-like tools and schemas, reporting precise current-read-contract incompatibilities. Reports remain pending alignment, with no execution authority or translation scores.

Reference candidates distinguish template examples, prose, code, directories, external URLs and ambiguous path bases. These are lexical hints, not complete dependency parsing or semantic obligations. Code fences are not universally ignored; a template hint never relaxes the executable-package gate. External URLs and repository escapes are not fetched. Supplementation is an explicit same-repository, exact-commit blob allowlist, with no checkout, source execution or automatic recursion. Parent evidence and failures remain intact.

Four predetermined source entries were exercised. OpenMontage preserves six files/65,778 characters in eight pages. Netdata grew from eight files/31,967 characters to sixteen files/146,532 characters in twenty-three pages after eight explicit references were recovered. A Git padded-size parsing bug initially blocked all eight; after fixing the parser, the same paths at the same commit succeeded 8/8. The failed child and successful revision are separate. New references appear in supplements, so complete transitive closure is **not** claimed. Supplemented documents do not increase the sampled-Skill count. DeerFlow's four template-link candidates are distinct from its prose link; the browser Skill is not assigned an invented host catalog.

Reading the complete 546-line Netdata wrapper reveals effects hidden by the original package boundary: shell-based environment loading (62–81), bearer issuance and filesystem caches (201–212, 246–300), general HTTP methods (307–332), and conditional cloud/agent arguments (384–429). These require explicit adapter/effect declarations. Query names and read-only hints are not purity or permission proofs. This is developer source inspection, not execution evidence or independent-human Gold; even source self-tests were not run.

Host intake preserves nested objects, arrays, key case, enums, required fields and constraints. It does not stringify objects, invent output schemas or fetch remote `$ref`s. Empty catalogs and duplicate names fail closed. `currentReadSchemaCompatible` refers only to the current ReadObjectSchema subset, never authentication, permission or semantic alignment. Nested data is now preserved and diagnosed, **not automatically executable through the old L0 reader**. The [synthetic interface example](../examples/translation-intake/mcp-catalog.json) is not a public Skill host or benchmark Gold.

Use the CLI above with a new output directory; add `--host-catalog` for a captured catalog or repeated `--supplement-path` options for explicitly selected pinned files. The programmatic supplement API can preserve the previous bundle's failure chain. Inspect the [report](../artifacts/translator-v2/intake-20260908/public-02/supplement-v2/report.json), [bundle](../artifacts/translator-v2/intake-20260908/public-02/supplement-v2/bundle.json), [pages](../artifacts/translator-v2/intake-20260908/public-02/supplement-v2/review-pages.json) and [bound summary](benchmarks/translation-intake-summary.json). The initial `original` intake drafts remain separate from `original-v2`; they were not model-evaluation inputs.

Mechanical checks reconstruct sixteen original text files/twenty pages from the four inputs byte-for-byte and verify supplemental texts, pages and parent digests. The existing corpus CLI reproduces all three files offline in a new temporary directory. A [twenty-file source overlay](../artifacts/translator-v2/intake-20260908/source-snapshot.tar.gz) is bound to the Git base and manifest; it grants no third-party execution rights. Digests establish content consistency, not independent semantic truth or provider authentication.

Next align a concrete task with source/reference roles and host/effect declarations, then add versioned bounded structured-type bindings without changing historical contracts. Data-shape support and collection-loop semantics are separate requirements. Freeze tasks, uncovered obligations and oracles before a small 9B batch. This stage adds no model calls, semantic success rate or Runtime latency result; the formal generalization gate remains locked.

Validation: 81 targeted tests; **1851 full-suite tests plus 81 subtests passed in 156.60 seconds**. Eighteen changed Python files pass lint; document links, forty bound evidence files and diff checks pass. An additional offline CLI smoke with the synthetic catalog confirms seven precise diagnostics, no authority and overwrite rejection. These are mechanical intake/boundary checks, not public-Skill semantic accuracy measurements. Git remains uncommitted.
