# EnsuredSkill 项目进展 / Project Status

## 中文

更新：2026-09-09。**C3p：通用有界列投影与本地合成 Netdata 宿主已实现；原读取网关完成 info/query 和页级结果验证，完整模型转译、全窗口覆盖及语义准入仍未完成。** 默认 DSH 路由未改变，研究候选未激活。不从继续授权推断 master 合入。

### Done

- C3p 增加通用 `column_rows`：保留原字段名，按当前 metadata 解码全部有界行；重复索引、越界、缺列和非法值拒绝，不猜测/跳行。接入原绑定、Tree/Flow，来源支配与回执时效继续约束候选。[用法](NETDATA-ISOLATED-VALIDATION.md)。
- 一个明确的进程内合成 Function 宿主，通过原读取网关实际调用 info/query；检查身份/范围、有效参数、listener、秒/微秒和返回筛选条件。输出 3 行本页计数（crit 1、warning 2），不保存原始回执；不是原厂服务、完整窗口或 9B 成绩。全任务 `taskCompleted=false`，默认路由未变。[证据](benchmarks/netdata-isolated-summary.json)。
- C3p 最终验证：**190 项定向；全量 2041 passed + 81 subtests passed（161.07 秒）**。36 个变更 Python 文件 Ruff、468 个本地文档链接、diff 及源码摘要通过。隔离源码重建的 6 份新版输出逐字节一致，旧接线 9 份输出不变；历史 C3n 证据由原快照核验，不改旧 Oracle。Git 未提交。

- C3o 新增离线 `task_alignment`：每文件审查区间与原句绑定，区分候选读取/Effect、L1、约束、条件及任务外义务；全量源分页与开发审阅结论分离，不泄露判定给未来模型输入。[用法与四例](TASK-SOURCE-ALIGNMENT.md)。
- 四个开发请求记录 **35 项声明义务、11 个问题、12 条未绑定宿主需求（9 工具／2 上下文／1 策略）**；不是穷尽覆盖或已实现工具，不虚构认证/脱敏 API 补数。Netdata 同 commit 新补一份协议，达到 17 文件／202,930 字符／28 页。保留原始日志落盘与隐私要求的张力、示例列索引不一致、动态解码与时间单位语义。源脚本未执行，9B/Provider 调用为 0，转译指标为空。[绑定证据与回归](benchmarks/task-alignment-summary.json)。
- C3o 最终验证：**122 项定向；全量 1975 passed + 81 subtests passed（161.82 秒）**。31 个变更 Python 文件 Ruff、439 个本地文档链接、20 份隔离重建输出逐字节一致及 diff 检查通过。分类修正不改变四例原始源包、任务或模型输入；旧版档案及快照保留，C3n 证据未变。不是语义准确率；Git 未提交。
- C3n 将嵌套输入/输出、JSON Pointer、分支和候选生成接入原 `qualify_flow` / `run_read_flow` / `execute_host_read`，不新建执行器。精确引用和 Tree 摘要绑定、支配关系、嵌套资源权限、控制条件回执时效、数据复制及异常脱敏保留。[完整处理过程与复现](STRUCTURED-FLOW-WIRING.md)。
- 一份开发 fixture 实际执行 **2 次本地只读回调、0 写入、0 模型调用**，停在 Effect 候选；另存空数组/越权/未认证及两个正常分支。原七个开发图与基线 v1 资格包/摘要一致，主演示回放逐字节一致。不是公开 Skill 准确率，也不认证设备采样时间；[回归与证据摘要](benchmarks/structured-flow-summary.json)。
- C3n 最终验证：**227 项定向；全量 1949 passed + 81 subtests passed（226.55 秒）**。29 个变更 Python 文件 Ruff、418 个本地文档链接、隔离源码重建的 9 份逐字节回放及 diff 检查通过；前两轮绑定证据未变。保留初始测试构造失败记录，未把通过数计入语义指标。Git 未提交。
- C3m 新增版本化结构化 Schema 和数据绑定原语：嵌套对象/数组/null、原键名、约束与有界局部引用；显式 JSON Pointer、常量、对象/数组构建，源值与最终目标每次验证。旧扁平合同和历史证据未改变。[用法、支持范围与证据](STRUCTURED-DATA-BINDING.md)。
- 沿用上轮合成 catalog：同两份 Schema 的可表达性由旧入口 0/2 到新入口 2/2；真实离线 CLI 验证嵌套参数、输出固定索引及非法枚举/空数组/catalog 漂移阻断。不是 Skill 语义准确率，LLM/Provider 调用均为 0，未增加执行器或激活合同。
- C3m 最终验证：**94 项定向测试；全量 1908 passed + 81 subtests passed（213.70 秒）**。23 个变更 Python 文件 Ruff、390 个文档本地链接、21 份本轮绑定证据通过；上轮 40 份证据未变。首次全量因并行编辑文档触发 `sourceState` 漂移，负结果保留；静止工作区复验通过，未修改冻结门禁。Git 未提交。
- C3l 新增无损、惰性的转译输入包和分页，保留引用位置、模板/正文/代码角色、路径歧义；接入 `netopyu-market-corpus translation-intake`，不改变旧执行包门禁。[实现、用法和证据](TRANSLATION-INTAKE.md)。
- 同四份固定源文验证：OpenMontage 65,778 字符完整保存；Netdata 同 commit 补取 8/8 引用，扩为 16 文件／146,532 字符／23 页。Git 大小列解析失败和修复后的成功链均保留，不增加采样 Skill 数；新引用尚未全部闭合。
- 原始宿主 Schema 无损保存并给出 JSON Pointer 缺口诊断；复杂结构尚未接入旧 L0 执行。完整阅读 Netdata wrapper 后确认认证、凭据缓存、通用 HTTP 方法及条件参数边界，不能把 query 名称当只读证明。本轮新增 9B 调用为 0，语义指标为空。
- C3l 最终验证：**81 项定向测试；全量 1851 passed + 81 subtests passed（156.60 秒）**。18 个变更 Python 文件 Ruff、文档链接、40 份绑定证据及 diff 检查通过；真实 CLI 的离线源包/宿主诊断复现和拒绝覆盖检查通过。未提交 Git，未运行第三方脚本；这些是机械回归，不是 1851 个 Skill 的语义测试。
- 新采集保存 12 份查询、合并 237 个 URL 候选；固定抽样 60 个／44 仓库，处理 60/60，保存 **53 Skill／38 仓库**，7 个排除不替换。29 个格式合格、24 个格式变体；这是入库资格，不是转译准确率。[全部候选与原文](../artifacts/translator-v2/public-source-20260908-round2/report/skill-library.html)。
- 新汇总器绑定抽样、四批静态快照与原文索引，区分全部处理完和 importer accepted 上限；额外已暴露仓库显式排除，源码不执行，失败不丢失。
- 四份固定根源文初审定位：嵌套元数据、长源文、真正包外引用与模板占位误报、结构化数据、条件写入及 L1 泛化职责。[诊断与边界](PUBLIC-TRANSLATION-BATCH.md)。未固定完整任务/宿主/Gold，本轮没有新增 9B 成绩。
- C3k 最终验证：定向 29 passed；全量 **1814 passed + 81 subtests passed**；16 个变更 Python 文件 Ruff、309 个文档本地链接、摘要和 diff 校验通过。Git 未提交；页面自动化被 file URL 策略阻止，不标交互验收通过。
- Runtime 的合同、Evidence、Guard、审批、事务、验证/补偿及本地 C1/C2 分支原型已实现，生产/真实设备资格未完成。
- 清理提交 c2ebd78、14baa0a；重复检查点已收敛，原始失败保留在[历史](PROJECT-HISTORY.md)。
- 逐份审查旧 33/33 的六份源文，发现完成引文错位和必要性/充分性混淆，旧结果未修改。
- 新实测证明旧单条件补全会降低可用性；已退出推荐流程。新增 require_any、可读表达式、联合分歧定位及显式未激活编译修订，不新增执行器。
- 新六例/46 场景：构造 **2/6、30/46**；条件阶段后 **5/6、35/46**。其中一份结构失败、11 场景未运行；四个可执行片段案例及一个正确停止匹配。辅助修订另报，不回填首次结果。[证据](FLOW-SEMANTIC-TRANSFER.md)。
- 构造 Schema 现在对齐既有 Quote 约束，排除结构无效的 `---` 等引用。结构有效仍不等于语义支持。
- 可选紧凑读取入口在同源开发修订中通过 4/5、34/45；修复工单但误拒绝副本案例，不能按 Oracle 与其他路径择优拼接。全量 1792 测试及 81 子测试通过；第二批报告已从隔离源码快照零调用回放，结果逐字节一致。
- 公开语料新增 `inert-text`、脚本原文索引、历史仓库排除与固定种子抽样。旧 269 个尝试中 35 个因脚本表面被排除，已全量列为开发补充；新元数据接口遇 HTTP 429。排除 199 个已尝试仓库后，旧缓存仅余 22 Skill／15 仓库且同属 finance analysis，不能冒充跨领域批次。[细节](PUBLIC-TRANSLATION-BATCH.md)。
- 脚本补充实际保存 **33/35 Skill、24 仓库、321 份隔离文本**，另 1 份二进制只留摘要。20 份进入转译研究语料，13 份作格式鲁棒性；新增 Runtime-ready 与模型调用均为 0。和旧库共 133 package ID／93 仓库，不是 133 个通过成绩。[可点击原文的索引](../artifacts/translator-v2/script-recovery-20260908/library-v2/skill-library.html)。
- C3j 最终回归 **1805 passed + 81 subtests**，定向 20 passed；修改代码 Ruff、diff、文档链接和摘要校验通过。未提交 Git；历史模型证据未修改。

### To-do 与边界

| 顺序 | 下一步 | 验收 / 限制 |
|---|---|---|
| 已收口 | 回归、源引用缺口、可回放报告 | 修改文件 Ruff / diff 检查通过；全仓仍有原先 224 项 lint，不混入本轮清理 |
| 入库完成 | 60 个冻结候选、旧脚本补充库 | 新库 53/60 保存；失败保留，采集限流已解除；搜索类别/静态包检查不能证明领域独立性/语义成功 |
| 入口完成 | 源文、分页、显式引用补取、宿主诊断 | 原文可完整拼回，旧证据不覆盖；不是跨页语义编译或完整引用闭合 |
| 绑定/接线完成 | 版本化嵌套数据、源锚定 Tree、共享执行器 | 有离线绑定 CLI 及显式宿主只读 smoke；完整 9B authoring/语义审查未接入新 Tree，不代表整 Skill 编译完成 |
| 任务档案完成 | 四份固定材料的具体任务、义务/问题与宿主需求 | 有源锚定开发审阅，不是独立 Gold 或全部源义务闭合；未读附件保留 |
| 局部实现完成 | 本地 Netdata 隔离适配器、info/query fixture、有界列解码 | 本页计数与异常检查已跑通；无原厂互操作、全窗口/保留期证明，不把枚举输出当生产隐私证明 |
| 下一步候选 authoring | 明确可编译段、L1 职责、未支持语义，接入当前结构化 Tree 的小批 9B 构造 | 当前完整 demo 仍手写；需保留源解释、首次失败和分层结果，不能只换外壳把脚本称为模型答案 |
| 对齐完成后 | 冻结任务/私有 Oracle，先小批后批量转译 | 9B；固定主候选路径、无自动重试；不把同批调参称泛化 |
| 正式转译门禁 | ≥3 不重叠 cohort、≥50 Skill、≥15 仓库、≥8 领域、≥600 case | 数量只是必要条件，质量阈值不下调 |
| 门禁通过后 | 大规模 L0→Runtime / DSH 配对 | **仍未解锁**，有限布尔区域通过不代表整 Skill 高准确转译 |

源引用仍有角色错位，完整源审查未完成。两批 12 份 Skill 均由同一开发助手构造，不是公开未见集或真人 Gold。现有 100-Skill 开发库也不能改称未见集。见[代码边界](../evaluation/README.md)、[纠偏计划](TRANSLATION-CORRECTION-PLAN.md)。

### 保留但不推进

ES-P1-Private-Human = skipped_retained_open；GPT 对照暂缓，使用 qwen3.5:9b。生产身份、供应链、治理、HA/DR、WORM、SLO、Hermes/A2A 为 frozen_future_engineering。权威原则见 [ENSUREDSKILL-PROTOTYPE](ENSUREDSKILL-PROTOTYPE.md)。

## English

Updated 2026-09-09. **C3p: bounded generic column projection and an isolated synthetic Netdata-shaped host run discovery/query and page checks through the original read gateway. Full model translation, window coverage and semantic admission remain open.** Default DSH routing/activation are unchanged; resumption does not prove a master merge.

C3p preserves original field names, bounded all-row decoding, dominance/freshness and the candidate-only Effect boundary. The host performs two actual local callbacks with explicit identity/scope and domain checks. Its three-row page yields crit 1/warning 2; no raw receipts are persisted. This is developer wiring, not vendor interoperability, full-window coverage or a 9B score. See [scope and use](NETDATA-ISOLATED-VALIDATION.md) and [evidence](benchmarks/netdata-isolated-summary.json).

Final C3p validation: **190 targeted tests; 2041 full-suite tests plus 81 subtests in 161.07 seconds**. Lint on 36 changed Python files, 468 local links, diff and source digests pass. Six isolated replay files are byte-identical, nine legacy wiring files are unchanged, and historical C3n bindings verify against their archived source. No historical Oracle changes; Git is uncommitted.

C3o records 35 declared obligations, eleven findings and twelve unbound host requirements (nine tools, two contexts, one policy) across four developer-authored requests, not exhaustive coverage or implemented tools. Exact review ranges/citations are checked; full source pages and future model inputs exclude developer review decisions. One pinned Netdata protocol supplement brings its bundle to seventeen files/202,930 characters/twenty-eight pages. Source tensions, inconsistent illustrative indices, dynamic decoding and time units remain explicit. No model/provider/script execution or semantic score. See [task dossiers](TASK-SOURCE-ALIGNMENT.md) and [evidence](benchmarks/task-alignment-summary.json).

Final C3o validation: **122 targeted tests; 1975 full-suite tests plus 81 subtests in 161.82 seconds**. Lint on 31 changed Python files, 439 local documentation links, twenty byte-identical isolated replay files and diff checks pass. Requirement-kind separation leaves all four source bundles, tasks and model inputs unchanged; prior dossiers/overlays and C3n evidence are retained. This is not semantic accuracy. Git remains uncommitted.

C3n reuses the existing qualifier, runner and read gateway for nested data, explicit pointers, branches and candidates. It preserves exact citations/Tree-digest binding, dominance, nested resource scopes, control-evidence age checks, copy isolation and redacted provider errors. One developer fixture performs **two local read callbacks, zero writes and zero model calls**; additional variants cover safe completion and refusal. Seven original developer graphs retain v1 packets/digests, and the positive demo replays byte-identically. These are wiring/compatibility checks, not public-Skill accuracy or device timestamp authentication. See [the full process](STRUCTURED-FLOW-WIRING.md) and [validation evidence](benchmarks/structured-flow-summary.json).

Next explicitly separate compilable regions, L1 duties and unsupported semantics, then connect small 9B construction to the current structured Tree with source review and first-attempt accounting. The full demo is still developer-wired, not a model answer. Page/retention/semantic closure remains open; large Runtime evaluation stays locked.

Final C3n validation: **227 targeted tests; 1949 full-suite tests + 81 subtests passed in 226.55 seconds**. Lint on 29 changed Python files, 418 local documentation links, nine byte-identical artifacts from an isolated source-overlay reconstruction and diff checks pass. Prior bound evidence is unchanged. Initial fixture-construction failures remain recorded; mechanical pass counts do not become semantic metrics. Git is uncommitted.

C3m adds versioned nested schemas and literal/reference/object/array bindings, retaining keys, constraints and bounded local references. Sources and final arguments are validated on every materialization. The previous synthetic catalog's two schemas move from 0/2 old-flat compatibility to 2/2 new-profile compatibility; offline CLI checks nested arguments, explicit output indexing, enum failures, empty arrays and catalog drift. This is not semantic Skill accuracy, a new executor or contract activation. Model/provider calls remain zero; see [scope and evidence](STRUCTURED-DATA-BINDING.md).

Final C3m validation: **94 targeted tests; 1908 full-suite tests + 81 subtests passed in 213.70 seconds**. Lint on twenty-three changed Python files, 390 local document links and twenty-one bound evidence files pass; forty prior evidence files are unchanged. The initial full suite detected sourceState drift during concurrent document edits; that negative result remains recorded, followed by a passing stationary-worktree rerun. The freeze gate was not modified. Git is uncommitted.

The new corpus CLI produces inert source bundles, lossless pages, reference-role/path diagnostics and optional raw-host-schema diagnostics. Four fixed sources were exercised: OpenMontage retains 65,778 characters; Netdata recovered eight explicit same-commit files, reaching sixteen files/146,532 characters/twenty-three pages. A padded Git size-column parsing failure and its successful retry are both retained; supplemental files are not additional sampled Skills. Reading the full Netdata wrapper exposes authentication/cache effects, general HTTP methods and conditional arguments. Complete reference closure, structured L0 execution and semantic compilation remain pending. No new model calls or semantic score; see [intake evidence and next steps](TRANSLATION-INTAKE.md).

Final C3l validation: **81 targeted tests; 1851 full-suite tests + 81 subtests passed in 156.60 seconds**. Lint for eighteen changed Python files, document links, forty bound evidence files and diff checks passed. Offline CLI source/host-diagnostic replay and overwrite rejection were verified. Changes are uncommitted; no third-party source was executed. These are mechanical regressions, not 1851 translated Skills.

Twelve saved queries yielded 237 unique source URLs. All 60 frozen candidates/44 repositories were processed: **53 Skills/38 repositories** saved, seven exclusions retained. The parser classifies 29 format-qualified inputs and 24 variants, not semantic successes. A bound report keeps all outcomes and distinguishes processing completion from an importer acceptance target. Four root-entry reviews locate source, schema, template and whole-Skill scope problems; references and complete task/host alignment are still pending. No new 9B translation score. See [results and next steps](PUBLIC-TRANSLATION-BATCH.md).

Final C3k validation: 29 targeted tests; **1814 tests + 81 subtests passed**. Changed-code lint, 309 local documentation links, evidence hashes and diff checks passed. Changes are uncommitted. Browser automation was blocked by the file URL policy, so interactive acceptance is not claimed.

The old six sources were audited; citation and necessity/sufficiency gaps remain. New measurements show unary guard addition can damage correct paths. The recommended path now uses readable Boolean extraction, deterministic evaluation and explicit inactive revisions. Compiler quote constraints are mirrored before generation, without proving entailment.

The new six-package / 46-scenario batch improves from **2/6 cases, 30/46 scenarios** to **5/6, 35/46** after condition extraction. A structural failure leaves eleven scenarios unrun. The optional compact front end subsequently matches **4/5, 34/45**, fixing helpdesk but falsely rejecting storage; do not cherry-pick paths using the oracle. Repairs are separate in the [report](FLOW-SEMANTIC-TRANSFER.md). Both batches are assistant-authored, not public holdouts or independent Gold; complete source/citation review remains open. Full regression passes 1792 tests and 81 subtests; isolated zero-call replay is byte-identical. Changed-file lint passes; 224 pre-existing repository lint issues remain.

The prior C3j tools retain inert script evidence and exclude attempted repositories. All 35 old script exclusions entered development recovery. Its HTTP 429 and single-query reserve were historical limitations, now superseded by the new acquisition above. Source/task/host alignment, private oracles and new translation results remain pending. The unchanged gate requires at least three cohorts, 50 Skills, 15 repositories, eight domains and 600 cases plus quality thresholds. Large Runtime/DSH evaluation stays locked; production engineering and human-review deferral remain unchanged.

Recovery saved 33/35 Skills from 24 repositories with 321 isolated text sources and one binary hash-only source. Twenty are translation-research inputs and thirteen format-robustness inputs. Combined known inventory: 133 package IDs/93 repositories, not successful translations. No new model calls or Runtime-ready packages.

Final C3j regression: 1805 tests and 81 subtests passed; twenty targeted tests, changed-code lint, diff, documentation links and evidence hashes passed. Git changes remain uncommitted and historical model evidence unchanged.
