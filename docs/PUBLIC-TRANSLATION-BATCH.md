# 公开 Skill 分批验证准备 / Public Skill Batch Preparation

## 中文

2026-09-08。本文记录 **C3k：新公开来源采集完成与源文初审**；后续 C3l 的无损入口、引用补取与宿主诊断见[新报告](TRANSLATION-INTAKE.md)。这是入库和诊断报告，不是新一轮转译成功率报告。研究顺序和[正式门禁](TRANSLATION-GENERALIZATION-GATE.md)不变；默认 DSH/Runtime 未改动。

### C3k：本次推进结果

SkillsMP 接口恢复后，分别保存 12 个查询响应，合并为 **237 个不同来源 URL 候选**。固定种子抽取 **60 个候选／44 个仓库／12 个搜索类别**，排除所有已知尝试仓库、检索时已暴露的仓库，以及已在本项目使用/研究的 DSH、Hermes、Anthropic 官方来源。额外暴露清单显式保存，不能声称穷尽了历史接触；URL 去重也不是语义去重。`sampling` 初稿保留，补充这三类框架来源排除后、下载前生成的 `sampling-v2` 为当前清单，没有根据转译结果选样。

| 入库阶段 | 结果 |
|---|---:|
| 已处理候选 / 冻结候选 | 60 / 60 |
| 已保存 Skill / 来源仓库 | 53 / 38 |
| 排除候选 | 7：6 个未被当前许可证检测器识别，1 个当前路径无根 SKILL.md |
| 当前解析器下格式合格 / 格式变体 | 29 / 24 |
| 静态包检查通过 / 格式合格但资源不完整 | 12 / 17 |
| 隔离源文件 | 38 |
| 新 9B 调用 / 新转译成绩 / 第三方代码执行 | 0 / 无 / 0 |

**12 个静态包通过，不等于 12 个 L0 可用或安全可执行；29 个格式合格，也不是 29 个语义转译成功。** 许可证未识别不等于项目没有许可证。全部 7 个失败原样保留，没有补换。为避免同仓库跨批，四个采集批为 **19、19、20、2**；这些不是四个已合格未见评测 cohort。前两批 importer 的 accepted 上限设为 20，而实际只选了 19 个；机器报告单列 `allCandidatesProcessed` 与 `importerAcceptedTargetMet`，不能把 `complete=false` 误读为仍有任务在后台运行。

打开[60 候选清单与 53 个 Skill 原文](../artifacts/translator-v2/public-source-20260908-round2/report/skill-library.html)：默认折叠，逐项查看结果、commit、排除原因、正文与隔离脚本。全部内容作为纯文本，没有脚本执行权。[机器摘要](benchmarks/public-source-round2-summary.json)绑定抽样、四份快照/索引和逐项审阅笔记；旧库、旧模型检查点和旧成绩不覆盖。

### 四个真实源文揭示的下一步

固定选择每个采集批的第一个候选，不按“容易转译”筛选；读完四份根 SKILL.md，另读了 SNMP 的 how-to 索引。**尚未读完全部附属引用，也没有固定用户任务、闭合宿主合同或独立 Gold。** [逐项笔记](benchmarks/public-source-alignment-review.json)记录源文件摘要、具体行号、已读/待读文件和修订方向。

| Skill | 已定位的问题 | 先改哪里 |
|---|---|---|
| OpenMontage `agents` | 嵌套元数据被当前格式解析器拒绝；包含安装、远程创建和外呼等不同效果；六份文本合计 65,778 字符，超过 `FlowSources.source_text` 的 64,000 上限 | 研究入口保留格式变体；按文件维护源证据和引用，不静默截断；绑定明确任务与真实/惰性宿主合同 |
| Netdata `query-snmp-traps` | 依赖包外令牌安全 wrapper；数组/嵌套选择；按节点循环；新问题还要求写 how-to；包含隐私约束 | 先补同 commit 的包外静态证据，再做宿主结构化绑定和有界集合处理；保留条件写入与 L1 分析职责 |
| DeerFlow `code-documentation` | 模板中的占位 `link` 被误报成缺失引用；项目分析和文档编写本来就是 L1 泛化任务 | 将模板示例与真正必需引用分开审阅，不直接放松执行门禁；确定性文件操作/验证只算片段 |
| `browser-testing-with-devtools` | 静态包可通过，但正文只有工具能力描述，没有固定工具 Schema；还有隔离、秘密、导航和审批要求 | 固定宿主 catalog 和任务后审查权限约束；不能把“只读”标签当安全实现 |

本地直接探测现有 `ReadObjectSchema`：扁平标量可接受；嵌套对象、数组和大写字段被拒绝，**不经过 LLM 也会失败**。这只定位读取合同子集的表达限制，不表示整个 Runtime 不支持分支或所有 Effect 合同都不支持结构化参数。FlowTree 有分支，但当前是无环图，没有原生集合循环。不能靠虚构“汇总工具”将缺口藏进一个 API。

因此下一步从“继续堆 9B 次数”调整为：**源文/引用角色诊断 → 源保真的宿主绑定与结构表达 → 冻结完整任务、未覆盖义务和 Oracle → 小批 9B 转译**。真正需要语义判断的片段继续由 L1 负责；不能为了整 Skill 通过率而强行确定化。后续代码修改后另行固定实现版本，新结果不回填本轮准备清单。这四例只提供问题定位，不代表 53 个 Skill 的失败率分布。

可重复生成汇总（仅读取已有制品，无网络、无模型）：

```bash
.venv/bin/python -m evaluation.translation_corpus sample-report \
  artifacts/translator-v2/public-source-20260908-round2/sampling-v2 \
  --batches-root artifacts/translator-v2/public-source-20260908-round2/batches \
  --output-root artifacts/new-public-sample-report
```

多查询合并使用 `scripts/netopyu-market-corpus merge-discoveries QUERY_JSON... --output NEW_JSON`；采样时额外暴露仓库通过可重复的 `--prior-repository owner/repo` 显式排除。合并、采样和报告都不覆盖已有输出。

本轮验证：**29 项定向测试，全量 1814 passed + 81 subtests passed（156.86 秒）**；16 个变更 Python 文件 Ruff、diff、309 个文档本地链接与证据摘要校验通过。去重保留 Git 分支/路径大小写，已检查本轮原始查询没有大小写碰撞，冻结样本不变。浏览器标签页已创建，但自动化读取被 `file://` 策略拒绝，未声称交互验收通过。未提交 Git，未执行第三方包，未改模型/Runtime 默认激活。

以下 C3j 记录保持为前一轮历史，不是当前接口仍然限流。

### C3j：此前发现的偏差

旧 `market-snapshot-100` 尝试了 269 个候选：100 个入库、108 个许可证未声明、35 个因为含可执行表面被排除、26 个获取错误。**35 是明确记录的脚本排除数，不代表所有失败都已经排除了脚本因素。** 因此旧 100-Skill 库不能用来证明带脚本 Skill 的适配性。

旧发现清单还有顺序截断：12 个搜索词不代表 12 个实际领域。把既往 100-Skill 快照与两个早期市场快照合并排除，共覆盖 199 个曾尝试的仓库；旧 300 候选仅剩 22 个 Skill／15 仓库，全部来自 `finance analysis` 搜索类别。它们不是合格的跨领域批次。搜索类别本身也不是人工确认的业务领域。

本次按新协议获取 SkillsMP 元数据时收到 HTTP 429，未连续重试或改称独立采集成功。原清单和成绩不变；先把那 35 个脚本排除项**全部**列入开发库恢复清单，不按脚本内容、可编译性或模型结果挑选。旧候选的新 Git 获取会固定当前返回的 commit，不伪称与旧快照版本一致；失败保留，不自动替换。

### 本轮已完成的入库结果

| 项目 | 结果 |
|---|---:|
| 原脚本排除候选 / 成功保存 | 35 / 33 |
| 成功保存的来源仓库 | 24 |
| 可开展转译研究的格式合格文本（仍缺 Runtime 资源） | 20 |
| 当前格式解析器下的鲁棒性样本 | 13 |
| 隔离源文件 | 322：321 份 UTF-8 文本，1 份仅二进制摘要 |
| 本轮 Runtime-ready / 新模型调用 / 新转译成绩 | 0 / 0 / 0 |

两条 `rka-project/rka-writer` 历史候选路径在此次固定版本下没有根 SKILL.md，获取失败已保留。另一个 `charts` 的 SKILL.md 带执行位：初版索引将隔离后的入口判成源文缺失，修正后的索引只读取隔离文本并检查格式，不恢复执行资源；该文本仍不满足当前格式解析器，所以归入鲁棒性，而不是把它补成通过。原快照及索引 v1 保留，新结果写到 `library-v2`。

和原库合计为 **133 个不同 package ID／93 个仓库**，其中 91 个仅具有主要转译语料资格；不是 133 个已测试通过的 Skill，也没有核实复制/近重复来源的独立性。新增材料属于已知开发库恢复，不计正式未见 cohort。

打开[33-Skill 补充库](../artifacts/translator-v2/script-recovery-20260908/library-v2/skill-library.html)，点击条目后查看 `[quarantine]` 文件原文。版本化的[机器摘要](benchmarks/public-source-preparation-summary.json)绑定候选、快照、逐条失败、两个索引和准备协议；[原始静态报告](../artifacts/translator-v2/script-recovery-20260908/report/public-skill-pilot-report.md)不含模型成绩。旧 [100-Skill 库](../artifacts/translator-v2/development-corpus-100/skill-library.html)未覆盖。

### 修复了什么

`scripts/netopyu-market-corpus snapshot --script-policy inert-text` 现在将来源分开保存：

| 来源 | 保存与使用边界 |
|---|---|
| 普通 SKILL.md / 参考文档 | 固定 commit，保持原文，仅作研究输入 |
| 脚本、可执行位文件、执行目录及配置 | 独立 `quarantine/<package>/<hash>.txt`，去执行权限，保留原路径、字节摘要和文本；不放进 Skill 资源目录 |
| 符号链接 | 只保存链接目标文字，不创建/跟随链接 |
| 二进制、子模块 | 二进制只留摘要，子模块仅记缺口；不导入、不安装、不执行 |

隔离文本可在生成的 `skill-library.html` 中点击查看，带 `[quarantine]` 标记。缺失执行资源会保持 `runtimeReady=false`，也不会进入旧 Runtime-ready author kit。检查器验证摘要、额外文件、符号链接与执行权限；后续扩库能复用已验证快照。

这只是**研究数据存储隔离**，不是允许执行任意脚本的操作系统沙箱。即使脚本原文可见，也不代表其依赖、效果、输入输出和回滚已建模。

### 下一轮固定规则

机器协议：[public_translation_protocol.json](../data/public_translation_protocol.json)。

1. 目标 60 个 Skill、每仓库最多 2 个、每批最多 20 个，同仓库不得跨批。按固定种子与搜索类别轮转抽样；排除所有已知尝试仓库，而不只排除成功入库者。
2. 抽样只看元数据；完整获取、格式资格、任务适用性分别报告。下载失败不替换；格式变体不混入主要准确率。全部采样数仍作为入库/覆盖分母。
3. 转译前先审查完整源文与未覆盖义务，固定用户任务、真实或明确声明的惰性宿主合同、参数来源和私有 Oracle。缺上下文、缺宿主、L1 推理区、可编译片段、完整 Skill 分开，不把市场 Skill 强行改写成两次读取。
4. 主候选路径固定为合同构造 v3，再在结构适用时使用条件表达式 v3；最多两次 9B 调用。紧凑入口不参加主路径；禁止按 Oracle 挑选最好候选，不自动重试或同批调参。
5. 同时报首次编译率、完整 Skill/片段覆盖、语义精度和适用召回、参数正确及来源闭合、引用角色、误拒绝、未覆盖义务、含失败的 token 与 p50/p95。源语义审查不能被真值表通过代替。

抽样制品绑定协议正文/摘要及 evaluation、network_runtime、effect_runtime、l1_runtime、tools 源文件。**这是准备协议，尚无完整任务 Gold 或新的转译结果；不能用采样摘要代替正式转译/Runtime 准入。** AI 辅助审阅不冒充独立真人 Gold，公共样本也不自动成为私有未见集。

### 命令与证据

在项目根目录运行；输出目录必须是新的，不覆盖旧证据：

```bash
scripts/netopyu-market-corpus recover-scripts \
  artifacts/translator-v2/market-discovery-300.json \
  artifacts/translator-v2/market-snapshot-100 \
  --output artifacts/new-script-recovery/discovery.json

scripts/netopyu-market-corpus snapshot artifacts/new-script-recovery/discovery.json \
  --output-root artifacts/new-script-recovery/snapshot --limit 35 \
  --script-policy inert-text --license-policy known --source-backend git

scripts/netopyu-market-corpus translation-corpus artifacts/new-script-recovery/snapshot \
  --discovery artifacts/new-script-recovery/discovery.json \
  --output-root artifacts/new-script-recovery/library
```

正式新抽样使用 `sample NEW_DISCOVERY --prior-snapshot OLD_SNAPSHOT`（可重复提供），并传入 `--seed ensuredskill-public-source-20260908-v1 --protocol data/public_translation_protocol.json --output-root NEW_ROOT`。样本短缺输出 `complete=false`，不能补上旧仓库来伪造达标。

本轮验证：定向 20 passed；最终全量 **1805 passed + 81 subtests passed**（156.54 秒）。修改代码 Ruff、git diff --check、232 个文档本地链接和摘要绑定校验通过；旧 53 个模型检查点 receipt 仍有效。未提交 Git、未改默认激活；全仓原有 lint 债务不在本轮修改范围内。

## English

This document records the C3k acquisition and entry-source review. The subsequent C3l lossless intake, pinned supplements and host-schema diagnosis are recorded in the [intake report](TRANSLATION-INTAKE.md); neither stage is a new semantic success-rate result.

**Current C3k:** discovery recovered. Twelve saved query responses merge to 237 unique source URLs. Metadata-only sampling selected 60 candidates from 44 repositories and twelve query strata, excluding attempted repositories, search-exposed sources and framework repositories already known to the project. The pre-download v2 selection preserves the first plan; no translator outcome informed this correction. Exposure tracking is not exhaustive historical independence, and URL deduplication is not semantic deduplication.

All 60 candidates were processed: **53 Skills from 38 repositories** saved, six license-detector exclusions and one missing root SKILL.md. The current parser classifies 29 as format-qualified and 24 as robustness variants; twelve pass static package inspection and seventeen have resource gaps. Thirty-eight source files remain quarantined. New 9B calls, translation results and third-party execution: **zero, unavailable, zero**. Static inspection grants no execution authority or semantic score. Failures remain visible without replacement. Repository grouping yields batches of 19/19/20/2; importer acceptance ceilings are distinct from processing completion. These are source batches, not qualified unseen cohorts. Inspect the [complete list and source text](../artifacts/translator-v2/public-source-20260908-round2/report/skill-library.html) and [bound summary](benchmarks/public-source-round2-summary.json).

Entry-level review used the first candidate of each frozen batch, retaining unsuitable/format-variant entries. All four root SKILL.md files and the SNMP how-to index were read, **not every transitive reference**. No task/host catalog or independent Gold is frozen. [Line-bound notes](benchmarks/public-source-alignment-review.json) locate: nested metadata and a 65,778-character source bundle in OpenMontage; missing sibling wrappers, arrays, iteration and conditional documentation writes in Netdata; a README template placeholder incorrectly classified as a missing reference in DeerFlow; and missing closed DevTools contracts despite passing static inspection. ReadObjectSchema directly rejects nested objects, arrays and uppercase property names without an LLM call. This is a read-subset limit, not a claim that the whole Runtime lacks branching or structured effects.

Next: distinguish source/reference roles, preserve host data structures and bounded collection semantics, then freeze explicit tasks, uncovered obligations and private oracles before a small 9B batch. Keep genuinely generative work in L1; do not hide omissions behind invented aggregate tools, truncate references, equate fragments with whole Skills or weaken the Runtime gate. These four diagnostic entries are not the failure distribution of all 53 Skills. New implementation revisions require a separate freeze. The report command above rebuilds from existing evidence without network or model calls. C3j history follows; its prior rate limit is resolved.

Validation: **29 targeted tests; 1814 tests + 81 subtests in 156.86 seconds**. Lint passed on sixteen changed Python files; diff, 309 local documentation links and evidence digests passed. Source URL deduplication preserves case-sensitive Git refs/paths; saved responses contain no such collisions, so the frozen sample is unchanged. The report tab exists, but browser automation is blocked by the `file://` policy; interactive UI acceptance is not claimed. No commit, third-party package execution or default activation change.

This is C3j preparation, not a new semantic-accuracy result. The previous importer attempted 269 candidates: 100 accepted, 108 license exclusions, 35 executable-surface exclusions and 26 acquisition failures. Excluding scripted packages biases applicability claims. The three prior inventories cover 199 attempted repositories; the cached reserve has only 22 Skills from 15 repositories, all in the finance-analysis query stratum. Query labels are not independently verified domains. Fresh SkillsMP discovery returned HTTP 429; no repeated retry or successful unseen-cohort claim was made.

The new `inert-text` policy retains executable source as non-executable, hash-bound `.txt` evidence outside the Skill package. Original paths and content remain inspectable in the static library. Symlinks are never created or followed, binary files remain hash-only and submodules remain unavailable. Withheld resources block Runtime-ready classification and legacy author-kit export. This is inert evidence storage, not an execution sandbox or script correctness proof. All 35 old script exclusions are recovered as development candidates without content-based selection; new Git commits and failed downloads are disclosed rather than replacing historical evidence.

The [preparation protocol](../data/public_translation_protocol.json) targets 60 Skills, at most two per repository and 20 per batch, with repository grouping and metadata-only query-balanced ordering. Missing downloads stay visible; format variants are separate from primary accuracy. Full-source scope, task/host alignment, parameter provenance and private oracles must be frozen before model calls. The primary path is constructor v3 plus expression v3 where structurally applicable, at most two 9B calls, no automatic retry, compact-path cherry-picking or same-cohort tuning. Whole Skills, fragments, missing capabilities/context, L1 reasoning and unresolved scope remain separate. Measure citation support and eligible recall as well as finite behavior and cost.

Sampling pins the protocol and source files, but is not a completed translator freeze, independent Gold, semantic score or Runtime admission artifact. The [formal gate](TRANSLATION-GENERALIZATION-GATE.md), prototype scope and frozen production-engineering boundaries remain unchanged.

The completed recovery saved **33/35 packages from 24 repositories**: twenty format-qualified translation-research inputs and thirteen robustness inputs under the current parser. There are 322 quarantined sources: 321 UTF-8 text files and one binary hash-only record. Two historical rka-writer paths lack a root SKILL.md. An executable-mode Markdown entry is now read from quarantine for format classification without restoring Runtime resources; it remains a format variant. The original snapshot and v1 index are retained separately from v2. Combined known inventories contain 133 package IDs/93 repositories, not 133 successful translations or independently deduplicated Skills. New Runtime-ready packages, model calls and translation results are all zero. Inspect the [library](../artifacts/translator-v2/script-recovery-20260908/library-v2/skill-library.html) and [digest-bound summary](benchmarks/public-source-preparation-summary.json).

Validation: twenty targeted tests; final full suite **1805 passed + 81 subtests**, 156.54 seconds. Changed-code lint, diff checks, 232 local documentation links and bound evidence hashes passed; all 53 historical model receipts remain valid. No Git commit or default activation change was made; pre-existing repository lint debt is outside this change.
