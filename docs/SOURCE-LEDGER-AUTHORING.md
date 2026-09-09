# 原文窗口、依赖记录与块引用 / Source Windows, Ledger and Block Citations

## 中文

更新：2026-09-09，C3r。当前主线是 **L1→L0 的来源保真和语义泛化**，不是继续堆叠生产工程。本阶段实现源文窗口、原句回填和块 ID 引用；不新增执行器，不改变 DSH 默认路由，不授予候选权限。

本页保留 C3r 历史协议和原始结果。后续检索状态、宿主映射与构造/执行边界的当前实现及两次 9B 失败见 [C3s](SOURCE-DECISION-AUTHORING.md)；旧 v1–v3 回放仍使用各自源码快照。

### 为什么修改

[上轮](PROGRESSIVE-STRUCTURED-AUTHORING.md)把每次请求的整页累计进输入，触及人为设定的 36,000 字节 wire 上限；这不等于耗尽模型的 token 上下文。本轮将问题拆开：

| 问题 | 当前实现 | 不保证什么 |
|---|---|---|
| 长源包无法全部驻留 | 原包完整保存，只切换当前原文窗口；全量页索引、未读根文页和已提交状态可见 | “提交过”不代表读懂或义务覆盖完整 |
| 切页会忘记约束 | 记录带来源的模型 notes，并回填原句及前后各最多 160 字符；可重新请求原页 | notes 可能遗漏、错解；不是权威摘要 |
| 重复回填又撑大上下文 | 合并同文件的重叠/相邻原文区间，已在窗口中的内容复用；保留全部 note 和对应锚点 | 不合并不同解释，不压缩原文，不证明义务充分 |
| 模型抄引文时改换行 | 模型选本请求的 `block_id`，系统提取原始文本和精确偏移 | 选对存在的块不等于选对语义依据 |
| 字节/token 混淆 | 分开记录消息字节、格式声明字节、wire 字节、服务端报告的输入 token 和输出预留 | 字节代理不是精确 tokenizer 或服务器未截断证明 |

源码：[source_ledger](../evaluation/source_ledger.py)、[source_blocks](../evaluation/source_blocks.py)。沿用原源包验证、冻结检查点、9B 传输、结构化 Tree 编译器和资格检查；编译阶段不调用底层工具。

### 完整交互与数据格式

```text
原始 L1 源包（包括脚本惰性文本）+ 开发请求 + 原宿主/读取合同
  → 冻结源码、输入、环境、模型摘要与资源策略
  → 当前 sourcePages 元数据 + sourceBlocks 原文 + 全量索引
  → 9B：request_pages + 来源锚定 notes
  → 校验块 ID，保留原文偏移与依赖请求，切换窗口
  → 回填 notes 对应的原句/上下文，再由 9B 请求源页或提出 Tree
  → 原编译器检查 Schema、引用、支配关系和合同
  → 未激活候选及审阅材料，或保留明确失败
```

模型不再输入 `quote` 字符串，而选择当前请求中实际提供的原文块：

```json
{
  "source": {"block_id": "b0007"},
  "kind": "constraint",
  "interpretation": "模型对这段原文的解释；仅供导航和后续审阅"
}
```

系统保存 `path/start/end/quote/documentDigest`，再从完整原文重新取回上下文，绝不从解释反造“原文”。块按原始空行和最多 1,200 字符划分，连同空白可拼回当前页；重复原句用不同位置区分。少于 8 字符的块仍展示但不能独立作引文。块 ID **只在对应请求内有效**，旧轮 ID 不得跨轮复用。

`dependencyRequests` 记录从哪页请求哪页、模型理由；`semanticDependencyResolved=false` 不被自动改成 true。它是可追踪的依赖请求记录，不是完整依赖分析。尚未保留的义务不会因为窗口切换而自动被判成无关。完整原文仍在输入快照内，移出窗口不等于删除。

候选结果单独列出 `candidateProduced`、`compiled`、`compiledReadNodes` 和 `terminalOutcomes`；纯 `needs_l1/unsupported` 停止图即使可编译，也不是可执行任务成功。`sourceCoverageProven`、`semanticDependencyClosureProven`、`wholeSkillTranslationProven`、`runtimeAuthorityGranted` 继续为 false。

### 验证与真实失败

四份已有公开材料的全量文本已按 UTF-8 字节分页并无损拼回：更新 agent 名称 9 页、浏览器诊断 3 页、README 生成 3 页、Netdata 36 页。Netdata 仍是同一份 17 文件/202,930 字符源包；从 28 到 36 页只是字节分页粒度改变，**不是新增 Skill**。见[静态分页审计](../artifacts/translator-v2/source-ledger-20260909/source-paging-audit.json)。另外有长中文、emoji、CRLF 和重复原句机械测试。这不是四份材料的转译成功率。

本阶段先验证了引文字符串版 `windowed-source-ledger/v1`：2 次真实 9B 请求，合计 **120.173 秒、12,307 输入 / 961 输出 token**。第二轮消息字节代理由第一轮的 28,548 降为 24,339；窗口未触及预算。但第三条 note 的原句换行被模型改动，精确匹配失败，整个记录未准入，没有生成候选。[原始失败报告](../artifacts/translator-v2/source-ledger-20260909/report/report.json)保留；不是靠模糊匹配、放宽校验或自动重试变成通过。

随后针对这个接口问题增加原文块 ID，使用新协议 `windowed-source-ledger/v2`、新输出目录，保留前一版源码快照。属于**同源开发修订，不是未见集或独立消融**：输入窗口、预算表达、分页和引用接口均有变化，不能将耗时或结果差异全部归因于一个因素。

原文块版 v2 真实调用 **2 次、149.218 秒、13,511 输入 / 1,387 输出 token**。两轮引用均通过，累计保留 14 条 note，没有因改换行而停止；但根文重访时重复回填重叠区间，第三轮字节代理达到 50,403，超过 40,960，未发送。该版仍未生成候选，不能计为转译成功。[原始 v2 报告](../artifacts/translator-v2/source-ledger-20260909/report-v2/report.json)。

为此增加 v3 的**精确区间并集**：以同一份已冻结的 14 条 note 和同一待发请求作离线资源重建，原文副本由 17,110 字符降到 9,879 个唯一原文字符，全部 note/覆盖区间保留；字节代理由 **50,403 降为 36,971（−26.6%）**，未提高上限。见[逐区间/逐 note 验证](../artifacts/translator-v2/source-ledger-20260909/interval-union-verified/report.json)。这是确定性空间改善，不是模型成功率或运行时性能；该审计未调用模型。

v3 实际完成 **6 次新调用、535.562 秒、47,332 输入 / 3,117 输出 token**。六轮来源 ID/输入预算均通过，实际字节代理范围为 28,476–38,962；服务端报告每次输入为 6,511–9,379 token，不据此假定未来数据的 token 比率。最终 14 条 note，**0 候选、0 编译读取节点、0 Provider/脚本执行**，终态 `source_window_round_budget_exhausted`。不是 Runtime 或语义准入成功。[v3 原始报告](../artifacts/translator-v2/source-ledger-20260909/report-v3/report.json)。

其轨迹是 `根文 → 指南 → 根文 → 指南 → 根文 → 指南`。回读时目标页仅有部分原文片段在当前上下文，因此不能简单说“整页明明已在当前上下文，模型还要重读”；但此前均完整提交过，且模型没有把读取结果推进到局部候选。见[逐轮覆盖与定位诊断](../artifacts/translator-v2/source-ledger-20260909/evidence-summary/report.json)。这份报告绑定 72 份本地文件，由同一开发助手审阅，不是独立 Gold。

三版累计 10 次新模型调用，都是同一份已知开发 Skill 的修订，不是 10 个 Skill。所有模型请求只访问本地 `qwen3.5:9b`；任务仍是固定输入、合成宿主，不是可重用的完整 Skill 或 Cloud 互操作认证。不能把通过引文/预算检查当成转译准确率。

v2 定向验证 244 项，全量 **2103 passed + 81 subtests passed（245.29 秒）**。v3 追加区间并集的来源集合相等、note 不丢失与原窗口复用测试；最终定向 **246 项通过**，全量 **2105 passed + 81 subtests passed（278.90 秒）**。三版均由各自源码快照完成隔离零调用回放，报告逐字节一致。覆盖旧页伪引用、虚构块、模型引文替换、原文重复、预算中断、残缺检查点、两次 preflight 模型漂移及零调用回放。测试数不是 Skill 数。

### 使用和复现

输入 JSON 仍严格只有 `bundle/task/taskOrigin/inputSchema/catalog/reads`，没有审阅答案、义务 Gold 或 query builder。每个原 catalog 工具要有完全一致的结构化读取合同。本入口不开放 Effect targets。

```bash
# 在新目录冻结输入和当前实现；只查询本地模型身份，不生成。
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN --inputs INPUT.json
# 最多六次新调用；不会重试已有无效候选或覆盖失败。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 6 --report-dir NEW_REPORT
# 已完成检查点离线回放；零新调用。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

单原页最多 12,000 UTF-8 字节；一次请求 1–2 个原页；最多 32 条去重 notes、6 次调用。超限则保留停止原因，不静默丢弃 notes。输出预留 4,096、模板预留 4,096、上下文配置 49,152 token；以消息和格式声明的 UTF-8 字节和不超过 40,960 作为保守调度代理，wire 另限 131,072 字节。该代理**没有 tokenizer 级认证**；报告另存服务端实际 `prompt_eval_count`，不按经验 token/字节比声称保证。

旧 v1 回放必须使用 Git `f0499ec` 加本阶段 `source-snapshot.tar.gz`；v2/v3 用同基线分别加 `source-snapshot-v2.tar.gz` / `source-snapshot-v3.tar.gz`。位于本地 `artifacts/translator-v2/source-ledger-20260909/`，不随 Git 分发，不可跳过源码指纹检查来重放旧证据。

### 仍待解决

源块引用解决的是抄写和定位，不是自然语言蕴含。9B 首轮 notes 仍偏向复述任务，之后虽提取出部分前置条件，仍未形成可构造片段。隐私、认证、结果完整性和原文冲突需要继续审查。根据本轮轨迹，优先修正：

1. **检索状态与语义审核状态分开。** 当前请求日志一直保留 `semanticDependencyResolved=false`，模型反复以“语义依赖未解决”为理由回读。应另列原文是否已取得、当前提供范围、具体尚缺哪一段；不能靠把语义审核标 true 来结束循环。这是有证据支持的改进假设，还不是已完成因果隔离。
2. **显式声明宿主对应关系。** 原文用 `agents_call_function`/Cloud wrappers，本地 catalog 是 `fixture_netdata_function`。要检查并提供经过审查的操作映射及适配边界，区分原文知识与宿主参数权威，不能让模型猜测等价，也不能宣称完整 Cloud 行为等价。
3. **有界检索后进入片段构造或带证据的缺口报告。** 回源需要说明缺少的具体义务/上下文，而不是仅重复任务；不得强迫产出不安全候选来改善成功数。再用不同源材料验证，保持本轮失败不变。

义务记录应区分**用户参数、源文约束、宿主声明和未知项**。不要用“引用存在/可编译”代替语义准入，也不能在来源不完整时自动回退执行危险动作。跨 Skill 泛化门禁和大规模 Runtime 评测保持未解锁。C3q/C3r 新改动尚未提交 Git；上次 dev 提交仍为 `f0499ec`，本轮未推送。

## English

This page retains historical C3r protocols/results. See [C3s](SOURCE-DECISION-AUTHORING.md) for current retrieval/host/authoring boundaries and the two new negative 9B runs. Historical versions require their archived source overlays.

C3r adds a replaceable source window, an anchored dependency-request ledger and request-bound block citations. Originals remain immutable; prior pages can be requested again. Notes are untrusted navigation and are rehydrated from original text with local context, not converted back from model summaries. Default DSH routing, provider execution and activation authority are unchanged.

The model selects `source.block_id` instead of copying quotations. Deterministic blocks retain exact text/character offsets, including whitespace and duplicates; they reconstruct the current page. IDs are valid only in their recorded request. V3 unions overlapping/adjacent intervals of the same original file, reusing full-page content while preserving every note and source position. Interpretations are not merged or summarized. Exact source location does not establish semantic entailment. Missing obligations, unresolved dependencies and whole-Skill coverage remain explicitly unproven.

Four known-source bundles reconstruct losslessly: agent-name update 9 pages, browser diagnosis 3, README generation 3 and Netdata 36. The last is the unchanged seventeen-file/202,930-character bundle; page count is not Skill count or translation accuracy. Mechanical tests include Unicode, emoji, CRLF, duplicate text, fabricated/stale citations, resource limits, model drift and offline replay.

The first ledger protocol used quote strings. Two real local 9B calls cost 120.173 seconds, 12,307 input and 961 output tokens. The second request's byte proxy decreased from 28,548 to 24,339, but the model changed line wrapping in its third note's quote. Exact matching rejected that update before any candidate. The [failure report](../artifacts/translator-v2/source-ledger-20260909/report/report.json) and source snapshot remain unchanged. V2 block citations passed both real calls (149.218 seconds; 13,511 input/1,387 output tokens) and retained fourteen notes, but overlapping source copies pushed the next byte proxy to 50,403. No candidate was generated; see the [v2 report](../artifacts/translator-v2/source-ledger-20260909/report-v2/report.json).

V3's offline reconstruction of that same frozen state preserves all fourteen notes and exact original coverage, while reducing duplicate source copies from 17,110 to 9,879 unique characters. The input byte proxy decreases **50,403 → 36,971 (−26.6%)**, below the unchanged 40,960 limit. This is deterministic resource improvement, not semantic accuracy or model outcome. See the [verified zero-model-call resource audit](../artifacts/translator-v2/source-ledger-20260909/interval-union-verified/report.json).

The fresh v3 run made **six calls in 535.562 seconds, using 47,332 input/3,117 output tokens**. All citation/resource checks passed; request byte proxies were 28,476–38,962 and reported input counts 6,511–9,379 tokens. Nevertheless it alternated between root and recipe, ending at the six-round cap with fourteen notes and **zero candidates/compiled reads/provider/script executions**. See the [v3 report](../artifacts/translator-v2/source-ledger-20260909/report-v3/report.json) and [source-coverage diagnosis](../artifacts/translator-v2/source-ledger-20260909/evidence-summary/report.json), which binds 72 files. Revisited targets were only partially rehydrated, not wholly present in the current frame. These ten calls across three versions concern one known Skill, not ten Skills, independent Gold or a single-variable model ablation.

The CLI above requires fresh directories and an explicit zero-to-six new-call budget. Default zero means offline replay, not retry. Input packets contain original sources/task/schema/catalog/read declarations, not reviewer answers or query builders. Current limits are 12,000 UTF-8 bytes per source page, one or two requested pages, 32 retained notes and six model calls. Message/format bytes, wire bytes, output reserves and server-reported input tokens are distinct. The scheduling byte proxy is conservative policy, not tokenizer-level proof of complete processing.

V2 passes 244 targeted tests and 2103 full-suite tests plus 81 subtests (245.29 seconds). Final v3 validation passes **246 targeted tests and 2105 full-suite tests plus 81 subtests (278.90 seconds)**. All three versions replay byte-identically with zero model calls from their respective source reconstructions. Git `f0499ec` plus the archived source overlays reconstructs each implementation; local artifacts are not distributed in Git.

Next separate source retrieval completion/current availability from semantic approval, and explicitly review source-operation-to-host mappings. The model cites the unresolved semantic dependency flag as a reason to reread; this is a plausible contributor, not a causally isolated explanation. Catalog and source-wrapper roles also need explicit separation, without claiming Cloud equivalence. A bounded reading phase should lead to a regional candidate or an evidence-bound gap report, never forced unsafe completion. Distinguish user parameters, source constraints, host declarations and unknowns. Compilation or valid citation IDs do not unlock broad Runtime evaluation. C3q/C3r changes remain uncommitted; no push was performed.
