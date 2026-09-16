# 编译与执行上下文隔离 / Isolated Compilation and Execution

## 中文

历史冻结报告：下述成绩不改分。后续[结构约束与工件修订](SCHEMA-ARTIFACT-CONVERGENCE.md)已实现本文末尾的部分结构/检查工作，但真实任务出现退步；当前状态请以后续报告为准，不把本页Mesh通过外推为最新结果。

2026-09-16，承接[参数协议与证据门禁修复](CALL-PROTOCOL-EVIDENCE.md)。上一批真实9B仍在编译AST中填入实参，两个任务均未获取函数清单。本包只改变职责分配，不扩大Runtime Schema、不增加语义自审、不引入生产工程化。

### 实现与使用边界

在现有显式启用的`netopyu.io/local-hybrid-host/v4`配置中增加`"compilerMode":"isolated"`，默认模式不变。它不是用户/模型可设置的参数。

1. 运行Agent使用原任务和实参调用prepare。宿主建立会话、冻结源码/配置/参数，在已有锁内记录唯一编译尝试。
2. 独立9B请求只收到原Skill、原任务、输入Schema及只读工具合同；**没有实参字典、观察、对话历史、调用工具或评测标准**。复用原author请求/Schema/模型配置，只产生待校验AST，不持执行权。任务原文本身可能含具体值，不把这称为数据脱敏。
3. 原编译器和Runtime校验AST；通过后在原精确资源ACL内执行只读前段。确定性拒绝直接进入原生受限fallback，无自动修订调用；超时/未知结果只能inspect或交接，不重试。
4. 执行Agent只看到原始Skill/任务、实际参数、已取得观察和交付动作，不看到authoring Schema/AST。公开工具从六个变为五个，移除submit；Python入口也拒绝注入/替换AST，不能仅靠前端隐藏。
5. 前段与补读均走原网关；原两次补读、快照去重、requiredReads、原样宿主终态不变。开放草稿始终是未验证L1，不是全Skill确定性L0，也没有写权。

source超预算时不截断、不调用编译模型；明确告知执行Agent未提供完整Skill，绑定task-only原生fallback。模型格式/Schema拒绝与未知传输分开记录；重复调用不能覆盖已有编译请求/成本。全部编译调用计入总模型次数、token及端到端时延，不隐去这项额外成本。

### 运行前固定验证计划

- 定向机制检查：两种上下文的数据隔离、移除AST入口、原编译器拒绝非法字符串/未知工具、fallback、未知结果不重试、成本留存、源超预算和旧模式兼容。脚本化/单元检查不是语义成功证据。
- **仅一次真实DSH＋qwen3.5:9b批次**：沿用`governed-session-20260915-convergence-inputs`的`irql-draft`、`irql-missing-enricher`和`mesh-incident`。两个已知失败＋一个已知通过的回归哨兵；2个已知Skill/仓库、3任务、0未见Skill。
- 不修改任务、观察或c1–c3/format，不添加requiredReads或人工正确AST。输入参数仍由运行Agent提供；编译模型不读取外部judgments。每任务最多420秒，编译HTTP最多180秒；每会话一次编译调用，无纠错循环。冻结源码/模型/Prompt后不改代码，不并行跑pytest，不追加同例9B批次。
- 分开记录：编译提案错误/前段通过数、IRQL清单实际获取数、完整任务通过数、各路由、获准终态、授权/重放，以及所有DSH＋编译＋Runtime调用/token和p50/p95。Mesh冻结判据没有要求完整服务流名称，这个既有评分范围局限继续保留，不补加事后标准。

诊断目标是两个IRQL都获取清单并完成任务、Mesh不退步；这不是保证，也不是替代原Gate 1。若AST错误减少但证据获取或任务仍失败，明确记录剩余层，不通过同例提示词试跑宣称收敛。n=3/已知来源/开发AI审阅不是原生对照、独立金标准、跨Skill泛化或生产概率；额外编译调用可能增加成本，不能预设性能提升。

### 结果：读取链路有改善，完整查询仍未通过

唯一批次已完成：3/3正常结束、单会话保留任务、获准未验证候选及原样终态；6/6实际读取在原ACL内，0观察到的越权/重放/写入。**编译前段2/3通过，完整任务1/3满足冻结判据**，不能当作总体准确率。两个查询取得函数清单，但仍0/2完整任务。Mesh满足原判据，仍有下述范围局限。

| 任务 | 路由／原判据 | 实际结果 | 端到端 |
|---|---|---|---:|
| irql-draft | Runtime，2/4 | caller绑定正确，确实读到清单；查询使用双闭区间，聚合后又投影已消失的EnvTime | 190.74秒 |
| irql-missing-enricher | native fallback，3/4 | caller多带origin/quote被拒；fallback读到两份导出，正确解释无富化；仍把右开时间写成双闭区间 | 175.91秒 |
| mesh-incident | Runtime，4/4 | 指标/相关性边界及请求部署日志符合冻结判据；条件式重读快照建议不适用于当前宿主 | 183.48秒 |

两查询都在文字中写出了正确`[start,end)`，却在工件中用了包含上界的`between`；第一例`EnvTime`又不在聚合结果或已声明富化输出中。这是业务工件错误，不是“来源没有送到模型”。语言规则依据：[Kusto between](https://learn.microsoft.com/en-us/kusto/query/between-operator?view=microsoft-fabric)和[summarize](https://learn.microsoft.com/en-us/kusto/query/summarize-operator?view=azure-monitor)。没有执行查询或连接数据库来复现报错。

现有工件检查实际把右开时间/完整语言标成`unverified`，**没有证明正确**；宿主的`candidate_unverified`只是形状准入。编译第一例还提出了错误的missing_host边界；边界是模型的未验证解释，不可当作宿主事实。Mesh c3按“请求相关日志”计分，未要求日志/trace/饱和度三项全列；重读静态快照和遗漏checkout→payments名称分别保留为局限，不偷偷新增或删除原判据。因此9/12只是冻结计数，开发AI审阅不是独立Gold。

### 同一两个查询任务的历史对照

不把本轮3任务与前一批2任务直接比较；下表仅取两份完全相同的IRQL输入/判据。没有同时原生对照，且上下文、提案预算和调用分配改变，故只是诊断性历史对照。

| 指标 | 前包 | 当前隔离模式 |
|---|---:|---:|
| 前段编译通过 | 0/2 | 1/2 |
| 实际获取函数清单 | 0/2 | 2/2 |
| 完整查询任务 | 0/2 | 0/2 |
| 全部真实模型调用 | 13 | 10 |
| 输入／输出token | 150,818／2,992 | 79,874／2,323 |
| p50／p95 | 175.34／203.95秒 | 183.33／190.00秒 |

**观察到token减少，但中位时延增加，不能说整体提速。** 全3例共15次调用＝DSH10＋编译3＋Runtime2，110,926输入／3,245输出token，未知用量0；p50/p95=183.48/190.02秒。只读合成环境没有测试写事务，不能推断生产安全概率。

### 可追溯证据与复现

- [冻结记录](../artifacts/governed-session-20260916-isolated-compiler-9b/freeze.json)摘要`sha256:0a42018399a49348801eb70fb11c1e31fed13d39cf7fe55a77e4e37888433bcb`；80份归档/当前源码一致，运行前后模型摘要一致。
- [原始观测](../artifacts/governed-session-20260916-isolated-compiler-9b/summary/report.json)、[逐项审阅](../artifacts/governed-session-20260916-isolated-compiler-review/judgments.json)、[摘要绑定报告](../artifacts/governed-session-20260916-isolated-compiler-assessment/report.json)。122份制品哈希通过；评估摘要`sha256:ae96a860b97590a0b4f42f5913670d5b0186d8eee6bee8bf0645f189472ea607`。
- 3份实际编译请求均与原symbolic author请求逐字结构一致，没有实际导出路径、实参字典或观察注入；不是人工正确AST。运行中/结束后未调参或改执行源码，未重跑同例模型。
- [可入Git摘要与QA](benchmarks/isolated-compiler-summary.json)。原始artifacts被Git忽略，以上原始链接仅在保留制品的本机可用；远端需另行备份。实现通过100项定向、8项文档/产品边界、121个变更/新增Python文件Ruff和diff检查；**3,312项全量＋81子测试通过（237.77秒）**，模型与pytest无重叠，全程固定执行源码。

复现需安装本地DSH与`qwen3.5:9b`，并保留原输入制品；使用不存在的新输出目录：

```bash
.venv/bin/python -m evaluation.hybrid_session_acceptance \
  artifacts/governed-session-20260915-convergence-inputs \
  /tmp/ensuredskill-isolated-new \
  --cases irql-draft irql-missing-enricher mesh-incident \
  --task-delivery --isolated-compiler --run
```

省略`--run`只做配置预检。本包已经执行唯一真实模型批次，不自动再运行上述命令。

### 收敛判断与下一项

隔离机制已实现；两个查询的证据获取从0/2到2/2是本批观察到的局部改善，不是语义阶段完成。剩余两项已分层定位：

1. **编译表示合法性**：目前输出约束只是JSON对象，生成时并没有完整author Schema约束；额外origin/quote导致拒绝。下一项应评估完整结构约束解码，保留独立编译校验，不删字段刷通过、不放宽绑定/权限、不增加自由自审。
2. **工件业务正确性**：自然语言正确描述了窗口，但生成工件不满足它。下一项应为支持的工件明确时间边界、列血缘等可验证义务，采用独立确定性验证；不支持/未验证保持未知，不因非空文本或有来源而算完成。这不是再改L0的通用执行图Schema。

上述下一项尚未实施。原Gate 1和后续新来源小批/跨cohort/正式原生对照门禁均不降低；CAPA等未重测的问题也未宣布关闭。默认UI不变，未提交、推送或重启。

## English

Historical frozen report, with unchanged scores. The subsequent [schema/artifact package](SCHEMA-ARTIFACT-CONVERGENCE.md) implements structural/partial-check work but regresses actual tasks. Its report is current;this page's passing Mesh result is not the latest outcome.

Bounded successor to the [call-protocol/evidence package](CALL-PROTOCOL-EVIDENCE.md), September16. Optional operator-only compilerMode=isolated on the existing enabled v4 host separates symbolic compilation from execution-agent calls. No Runtime schema expansion,semantic self-review stack or production engineering is introduced.

Prepare freezes a session and claims one author attempt. A tool-free9B request receives the unchanged Skill/task,input schema and read contracts,not invocation arguments,observations,conversation or evaluator criteria. The original task may itself contain values; this is context separation,not anonymization. The same compiler/runtime qualifies the untrusted AST and permits only existing ACL-bound reads. Known rejection binds native read-only fallback without model repair;uncertain transport stops without retry. Source overflow is explicit task-only fallback with no truncation or author call.

The execution Agent receives original context,concrete inputs,observations and delivery actions,not author schemas/AST. Five tools are exposed;submit is removed and the backend independently denies AST injection. Existing read budget,snapshot identity,evidence prerequisites and faithful terminal remain. Open answers stay unverified L1;no whole-Skill deterministic conversion or write authority is claimed. All compiler calls/tokens/latency,including unknown usage,are counted.

Preregistered verification: mechanism tests then one actualDSH/qwen3.5:9b batch using unchanged irql-draft,irql-missing-enricher and mesh-incident from the prior convergence inputs. Two known failures and a known passing sentinel:2 known Skills/repositories,3 tasks,zero unseen Skills. No new prerequisites,expected answers or handcrafted ASTs;unchanged c1–c3/format.420s/task,180s compiler HTTP,one author call/session,no repair loop. Freeze code/model/persona before running;no code changes,pytest overlap or same-case model reruns. Mesh's existing rubric does not require complete service-flow naming;retain that limitation without retroactive criteria.

Report compiler errors/prefix yield,actual inventory acquisition,full-task criteria,route/admission/ACL/replay and allDSH+compiler+Runtime costs/latency separately. Diagnostic target:bothIRQL inventories/tasks and noMesh regression,not a guaranteed result or substitute for Gate1. Reduced AST errors with continued evidence/task failures must be reported as remaining issues,not semantic convergence. This is developer diagnostic evidence,not independent gold,paired native control,generalization or production probability;additional author calls may increase cost.

### Results

The sole frozen batch finished:3 normal retained-task,faithful unverified deliveries;6 authorized reads andzero observed writes/unauthorized execution/replay. Two of three prefixes compile;one of three tasks meets frozen criteria. BothIRQL inventories are acquired but both complete query tasks still fail. The original task compiles a proper caller binding but produces an inclusive time range and projects EnvTime after aggregation removes it. The second proposal adds forbidden origin/quote to caller and falls back;native execution reads both exports,correctly omits absent enrichment,but still includes the upper time bound. Mesh meets its existing metrics/correlation/relevant-log-request rubric,with conditional same-snapshot reread and missing service-flow name retained as limitations.

Kusto [between includes both bounds](https://learn.microsoft.com/en-us/kusto/query/between-operator?view=microsoft-fabric);[summarize outputs group/aggregate columns](https://learn.microsoft.com/en-us/kusto/query/summarize-operator?view=azure-monitor). Correct interval prose does not fix the query. No query was executed. Existing artifact checks mark these dimensions unverified;candidate_unverified is shape admission,not business validation. The first compiler's incorrect missing-host explanation is also an unverified claim.9/12 criteria and1/3 tasks are developer-AI scoring,not calibrated accuracy or independent gold.

For the same twoIRQL tasks only:prefix yield0/2→1/2,inventory acquisition0/2→2/2,complete tasks0/2→0/2;13→10 total calls,150,818/2,992→79,874/2,323 input/output tokens;p50/p95=175.34/203.95→183.33/190.00s. Lower observed tokens coexist with a slower median: no overall speed claim or causal pairedAB. Across allthree tasks:15 calls=10DSH+3compiler+2Runtime,110,926/3,245 tokens,zero unknown usage,p50/p95=183.48/190.02s.

All122 artifacts and80 archived/current source files verify,with unchanged model/code. Allthree real compiler wires equal the original symbolic request and contain no invocation export path. No injected prerequisites,handwritten ASTs,post-run tuning or same-case model retries. Exact traces/review/report are linked above;[portable summary and QA](benchmarks/isolated-compiler-summary.json) retains metrics and limitations. Raw artifacts are Git-ignored and need separate backup.100 targeted checks,8 documentation/authority checks,Ruff on121 changed/new Python files and diff checks pass. Final full suite:3,312 tests plus81 subtests in237.77s,unchanged execution source,no model/pytest overlap.

Remaining layers:constrain compiler output to the complete author schema rather than merely JSON;independently verify supported artifact obligations such as exact time bounds and column lineage. Do not coerce rejected bindings,relax authorization or treat unsupported verification as completion. These next changes are not implemented. Gate1 and subsequent fresh-source/cross-cohort/native-comparison gates remain open;unretested CAPA defects are not closed. No defaultUI change,commit,push or restart.
