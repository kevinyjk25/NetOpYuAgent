# 阶段 2 公开 Skill 迁移验证 / Stage 2 Public-Skill Transfer

## 中文

本页是 **v63 首批结果快照**。后续版本的能力和结果单列在[表示与导航修复](STAGE-2-REPRESENTATION-REPAIR.md)，不反改本批分数。

2026-09-10。**本批首次构造与开发者 AI 审阅已结束，阶段 2 的可用性闭环未通过。** 预选 10 Skill / 10 仓库 / 9 类任务领域；7 个进入真实 9B，8 次调用，1 个编译通过但语义审阅不通过，最终没有可接受的只读区域。不把预检改善称为转译准确率提升，也不扩大 Runtime 对照。

[可随 Git 保存的指标摘要](benchmarks/stage2-public-transfer-summary.json)；[完整证据](../artifacts/translator-v2/stage2-20260910/evidence-v63/report.json)；[逐例源审阅](../artifacts/translator-v2/stage2-20260910/developer-review-v63.json)。报告摘要为 sha256:cd90787293a09149484e79094bb57b6af9bc9428f62f2548b064bcbce3bf291d，绑定 136 份制品。

| 指标 | 结果 / 分母 |
|---|---|
| 样本 | 10 Skill / 10 仓库；9 个工具型，1 个纯 L1 |
| 初始输入预算通过 | v62 为 0/9；v63 为 7/9，另 2 个预算停止 |
| 真实构造 | 7 Skill，8 次 9B 调用；没有重试或中途修提示 |
| 结构编译 | 1/7；4 个计划未决，2 个字段/来源校验失败 |
| 已编译候选的语义审阅 | 0/1 接受；7 个模型用例最终无可接受区域 |
| 参数 Oracle、职责完整率、正确/过度停止率 | 尚无完整 Oracle/分母，不给数值 |
| 成本 | 60,558 输入 / 20,746 输出 token；无缺回执/未知 token 用量 |
| 模型请求耗时 | 合计 978.65 秒；p50/p95 126.09/167.79 秒 |
| Runtime 效果 / 时延 | 未执行，不给成功率或时延 |

模型耗时与本地回归重叠，不是隔离性能基准。7 份完整报告按本版代码零调用回放一致；阶段 1 的 2919 份制品不变。22 项定向检查通过；全量 2543 passed + 81 subtests，阶段代码 Ruff 与文档检查通过。第一次全量因清理时遗漏冻结文档分区标识失败，已恢复标识并复验，未放宽测试；失败记录保留。

### 冻结样本与用途

| ID | 仓库 | 任务 / 必须保留的边界 |
|---|---|---|
| snmp | netdata/netdata | 查询 SNMP trap；源选择、隐私、凭据 wrapper 与未完成 how-to |
| notion | openclaw/openclaw | 读取页面；不执行安装/登录/写入 |
| handoff | wshobson/agents | 异步值班交接草稿；引用模板、人类确认与报警验证 |
| playwright | microsoft/playwright | CI 跨运行 flaky 分析；重试粒度、SQL 语义与未执行代码 |
| simple-english | moeru-ai/airi | 纯语言改写；保留代码/错误文字、自检，不虚构外部操作 |
| documentation | bytedance/deer-flow | 读取代码后提 README；模板不是实际路径，写作仍为 L1 |
| incident | langfuse/langfuse | 查询 incident-alert；精确监控身份、部分匹配、人类写回边界 |
| phoenix | github/awesome-copilot | Python tracing 提案；引用/SDK、代码生成和遥测不执行 |
| warehouse | posthog/posthog | 工单诊断；身份与租户核对，禁止把客户自述当授权 |
| tetragon | mukul975/anthropic-cybersecurity-skills | 已安装状态审查；2 份实际脚本附件保持惰性，不安装/部署/执行 |

[预选清单](../data/stage2_public_selection.json)；[任务与本地宿主声明代码](../evaluation/stage2_cases.py)；[冻结与运行入口](../evaluation/stage2_batch.py)。域类别是开发者分类，不是独立认证。完整源版本/原始路径/文档/脚本摘要在各 source-bundle.json，不对同名 Skill 作成功去重。

### 已发现的真实问题

- 冻结 Git 43a2b76 / v62 首次预检：9 个工具型用例全部因输入预算停止；纯语言用例为当前只读作者不适用。此时模型调用为 0，不能叫 9B 语义失败。
- 原因之一：每个源片段都内联一份相同 source_scan Schema，而且 Schema 同时提供给模型和结构约束解码。原始文字页 12,000 字节的上限没有计入逐片段结构开销。
- v63 通用修订：用共享 $defs 保留全部必填片段/角色/含义约束，将页粒度降到 2,048 UTF-8 字节。全文、偏移、所有未读页仍保留；预算仍是 40,960 的字节代理，输出仍 4,096 token，窗口轮次和严格语义门禁未放宽。
- 修订后 7/9 工具型用例首次预检可运行；snmp 和 tetragon 仍超预算。分页可能增加导航成本或撞到轮次上限，不把初始预检通过当成完整构造通过。

两版独立记录：[v62 预检](../artifacts/translator-v2/stage2-20260910/preparation-v62/freeze/manifest.json)、[v63 冻结](../artifacts/translator-v2/stage2-20260910/preparation-v63/freeze/manifest.json)。这是已知开发样本上的诊断修订，不是未见泛化提升。

### 模型输入、审阅和执行分离

每个用例目录含 source-bundle.json、task.json、author-input.json（纯 L1 除外）、review-requirements.json 和 initial-request.json。模型只接收原始源文、任务、Schema/合同和其自身窗口状态；审阅要求不作为生成答案提供。当前宿主多数是明确披露的本地原子操作声明，**不是真实 API/MCP 抓取，也没有执行或证明厂商兼容**；不把匹配、诊断、SQL 生成或权限判断藏入聚合工具。

审阅要求只声明任务相关源段定位，不宣称已审核全部传递引用。需在结果阶段逐份对照实际可见原文、参数来源、条件、L1 职责及缺口；Schema 合法不是语义合格。无法判定的指标为 unknown，不填零。纯语言适用性判断是开发者预检，不计为模型成功停止。

### 根因与下一步修复顺序

1. **源文导航与业务执行仍混在一起。** handoff 把被分页切开的 “Incoming en” 和库中已有的引用当作缺失来源；phoenix 未读取已保留的定义页就声称没有定义。下一步应按完整行/段和依赖定位构造窗口，明确“未读”不等于“不存在”，并在规划业务图前完成必要来源读取。原文始终保留；不能以缩小页面后模型更早停止宣称成功。
2. **数据引用存在表示和构造两类问题。** Playwright/Phoenix 将 input 的 databasePath/applicationPath 错挂到 obs 返回值上，原编译器已拒绝。Incident 把 monitorTitle 当成字面字符串；当前分支 RHS 只允许字面值，无法表达两个动态值直接比较。下一步补双侧有类型操作数、作用域约束和来源说明，不让模型用字段名字符串伪装动态绑定。
3. **原文规则、调用者约束与模型建议缺乏明确区分。** Notion 正确绑定了 pageId，却发明 markdown == "null" 分支，引用的原文只是在解释 Markdown 输出。需要明确条件/约束的来源类别与权威；模型补充的处理策略只能作为候选，不伪装成原文中的严格规则。
4. **开放职责仍需混合流程承接。** 多个用例在读取后遗漏写作/诊断/比较职责；documentation 又把执行时才能获得的内容当离线缺参。明确保留这些 L1 任务及结果准入，不能以“读取结束”替代完成。Warehouse 的 SQL/成员身份工具确实缺失，须保留为宿主缺口，不靠模型或虚构聚合工具补齐。

例如：[Notion 原始候选](../artifacts/translator-v2/stage2-20260910/first-construction-v63/notion/round-000/choice.json)中的 /program/next，与[原文/偏移映射](../artifacts/translator-v2/stage2-20260910/first-construction-v63/notion/round-000/program-draft.json)直接对照，即可定位“有引用但不蕴含”的错误。该候选没有被裁掉坏分支后计为成功。

后续修复必须另建版本；本批只作开发回归，另选新的开发批次检查迁移，不能把这 10 例调整后全过当成未见泛化。当前结果不能单独归因于 9B，也没有证据证明换大模型就能修复 Schema 或导航缺陷。

```bash
# 准备必须使用新目录；先冻结全部用例，再接触模型。
.venv/bin/python -m evaluation.stage2_batch prepare data/stage2_public_selection.json NEW_PREPARATION
.venv/bin/python -m evaluation.stage2_batch run NEW_PREPARATION NEW_RUN --max-new-calls 98
```

98 是此批最多 7 个可运行用例 × 原单次构造上限 14 次的显式预算，不是预期调用数；首次失败/不完整回执不会重试。运行目录必须全新，旧结果不覆盖。这个命令不启动 DSH，不执行工具、网络设备或公开脚本。

### 清理记录

当前进展/文档导航/研究代码入口已缩短，旧内容留在同级历史快照，原有链接及回放依赖保留。三个可再生缓存目录已移到 /private/tmp/netopyu-cache-cleanup-20260910.UsuUyr，可恢复但不作为长期备份。两个无关 MLP 文件、共享 .venv、数据库与 243 MB 评测制品均未删除。没有删除仍被引用的“旧”源码，也未顺手处理全仓 224 条 lint 技术债。

阶段 1 文档绑定记录是其发布时快照；旧文档可由 Git 43a2b76 恢复。本轮更新当前文档须以新版本记录，不修改旧 publication/report.json 的摘要或宣称它绑定当前文档。

## English

This page preserves the v63 first-batch result. Later implementation changes are reported separately in [representation and navigation repair](STAGE-2-REPRESENTATION-REPAIR.md), without rescoring this batch.

Predeclared known-development batch: ten Skills, ten repositories and nine developer-classified task domains; not unseen holdout or independent Gold. The table lists network, knowledge, incident, testing, language, documentation, tracing, warehouse and security sources. Two actual Tetragon sidecar scripts remain inert.

Frozen v62 preflight blocks all nine tool-bearing tasks before any model call. Repeated per-fragment scan schemas and large raw-source pages exceed the unchanged byte proxy. v63 shares the identical fragment schema via $defs and uses 2,048-byte pages while retaining full originals, offsets and unread-page accounting. The 40,960-byte proxy, 4,096-token output allowance and round/semantic limits remain. Seven initial requests fit; SNMP and Tetragon still do not. Smaller pages may increase navigation or hit the round limit; this is not construction or semantic success.

The first batch and developer-AI review have finished; Stage 2 usability has not passed. Seven Skills made eight real 9B calls: one compiled candidate fails semantic review, four plans retain unresolved issues and two fail schema/source validation. No region is accepted. Cost: 60,558 input and 20,746 output tokens; summed request time 978.65 seconds, p50/p95 126.09/167.79 seconds, overlapping local regression rather than isolated performance. Seven reports replay identically without calls; 136 artifacts are bound. Full regression: 2543 plus 81 subtests; 22 targeted checks and changed-code lint pass. A missing frozen-document heading caused the first regression failure and was restored without weakening its test.

Tasks use disclosed primitive local host declarations, not captured vendor APIs or live integrations. Separate reviewer expectations never enter author inputs. Developer precheck classifies tool-free language work as L1-only, not a model safe-stop success. Missing complete parameter/duty/stop Oracles remain null. No source script, provider, network-device operation, Runtime behavior evaluation or DSH experiment ran. See the [portable metrics](benchmarks/stage2-public-transfer-summary.json) and [bound review](../artifacts/translator-v2/stage2-20260910/developer-review-v63.json).

Next fix source-navigation versus business-operation confusion, intact semantic window boundaries, input versus observation typing, and true dynamic-to-dynamic comparisons: the current RHS is literal-only, so variable-name strings are not a valid substitute. Separate original rules, caller constraints and model-suggested guards; the Notion candidate invented a null-string branch unsupported by its cited prose. Preserve open L1 work rather than terminating after reads, and retain genuine warehouse host/identity gaps. Changes require a new version and a new development transfer batch; same-batch tuning is not unseen evidence, nor do these results isolate 9B as the sole cause.

Housekeeping preserves historical navigation, source/replay dependencies, negative evidence, databases, shared environment and unrelated MLP files. Only three reproducible caches were moved to a recoverable temporary directory. Live document edits supersede stale status but do not rewrite Stage 1's historical publication digest; its original documents remain recoverable from 43a2b76.
