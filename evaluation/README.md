# 转译研究代码导航 / Translation Research Code Map

## 中文

这是研究 authoring/评测，不是默认 DSH 的生产转译器。**推荐：合同约束构造 → 可读源条件 → 确定性编译 → 完整源审查。** 单变量反事实补全会损害正确路径，已退出推荐流程，原文件保留作历史回放。

| 模块 | 职责与边界 |
|---|---|
| [source_plan](source_plan.py) 的 `--semantic-plan` | 带出处的步骤清单、结果名称与精确字段路径、类型化数组长度、路径交接职责及未来执行要求分离；原编译器/读取引擎不变。[设计、使用和边界](../docs/SEMANTIC-PLAN.md) |
| [source_closed_program](source_closed_program.py) / [source_inline_program](source_inline_program.py) | 闭合控制树限定 read 后继、双分支与无后继终点；每个节点/职责保留原文 ID 和新旧位置。只做无损语法转换，不修复或认证模型语义。 |
| [stage1_cases](stage1_cases.py) / [stage1_validation](stage1_validation.py) | 已知合成开发输入与原共享引擎的行为检查；模型输入没有手写 L0/Oracle。显式受审摘要、重新编译、记录精确调用；不是公开 Skill 或大规模 Runtime 证据。[验收标准](../docs/STAGE-1-EXIT.md) |
| [source_duty_accounting](source_duty_accounting.py) | 可选 `plan_first --account-duties`：逐项源义务与冻结计划对账、前置分支路径检查、准确 L1 交接。已提取行覆盖不等于完整源覆盖；真实 9B 仍失败。[结果与定位](../docs/SOURCE-DUTY-ACCOUNTING.md) |
| [netdata_fixture](netdata_fixture.py) / [netdata_task_demo](netdata_task_demo.py) | 一个合成进程内 Function 宿主，经原读取网关执行 info/query；解码与页级计数在工具外进行，不是 9B 转译或真实 Netdata。见[演示和边界](../docs/NETDATA-ISOLATED-VALIDATION.md) |
| [structured_authoring](structured_authoring.py) | 冻结原源包/任务/宿主和显式 9B 预算，按需请求原文页，再进入现有 Tree 编译；不执行候选，首轮上下文停止保留。见[真实首次结果](../docs/PROGRESSIVE-STRUCTURED-AUTHORING.md) |
| [source_ledger](source_ledger.py) / [source_blocks](source_blocks.py) | 当前窗口 + 来源锚定 notes + 原文块 ID + 精确区间并集；保留历史失败，不将引用存在/编译通过计作语义成功。[范围和验证](../docs/SOURCE-LEDGER-AUTHORING.md) |
| [source_retrieval](source_retrieval.py) / [source_host_binding](source_host_binding.py) | 区分文本交付和语义审核，联合回读决策；显式宿主映射与现有执行前提。没有新权限路径。[新 9B 失败与边界](../docs/SOURCE-DECISION-AUTHORING.md) |
| [source_obligations](source_obligations.py) / [source_gap_search](source_gap_search.py) / [source_candidate_schema](source_candidate_schema.py) | 可选的源义务阶段审阅、本地字面缺口检索、生成前表达式语法。CLI/API 默认 direct，审阅未证明收益。[C3t 历史结果与诊断](../docs/SOURCE-OBLIGATION-AUTHORING.md)；新构造路径见下行。 |
| [source_catalog](source_catalog.py) | 可选 catalog_bound：宿主类型引导、参数值出处、代码分配别名、独立问题定位。已知源的参数形状错误消除，但流程/来源/操作模式仍阻塞。[当前 9B 对照与边界](../docs/CATALOG-DIRECTED-AUTHORING.md) |
| [source_modes](source_modes.py) | 可选 mode_bound：摘要绑定的互斥对象键合同、模式引导逐叶构造、动态形状阻断和源块行动轨迹。参数模式已修复，整 Skill 仍失败。[两次 9B 结果与边界](../docs/HOST-OPERATION-MODES.md) |
| [source_plan](source_plan.py) | 可选 plan_first：冻结骨架、逐槽填参及来源检查、词法来源导航、不可达冗余终止规范化，复用原编译器。真实 9B 合成局部流程跑通，公开源仍失败。[全部实验与边界](../docs/PLAN-FIRST-AUTHORING.md) |
| [public_skill_corpus](public_skill_corpus.py) / [translation_corpus](translation_corpus.py) | 多查询合并、固定抽样、显式暴露仓库排除、脚本隔离文本、原文索引及保留全部失败的 sample-report；入库不等于转译通过。见[新批次结果与源文诊断](../docs/PUBLIC-TRANSLATION-BATCH.md) |
| [translation_intake](translation_intake.py) | 转译前的无损原文/分页、显式同 commit 引用补取、原始宿主 Schema 及 JSON Pointer 诊断；不做跨页语义编译，不给旧 L0 增加执行权限。见[用法和边界](../docs/TRANSLATION-INTAKE.md) |
| [task_alignment](task_alignment.py) | 完整保留原始分页，核对任务引用/已读区间/义务与宿主需求；审阅档案与未来模型输入分离，不是自动语义准入。见[四份实际材料](../docs/TASK-SOURCE-ALIGNMENT.md) |
| [structured_binding_probe](structured_binding_probe.py) | 新结构化数据绑定原语的离线 compile/materialize/demo；验证嵌套参数与输出，不调用模型/Provider。尚未接入下方旧 FlowSources。见[实际例子](../docs/STRUCTURED-DATA-BINDING.md) |
| [structured_flow_tree](structured_flow_tree.py) / [structured_flow_demo](structured_flow_demo.py) | 源偏移 + 嵌套数据 Tree → FlowProposal v2 → 原共享执行器；本地 fixture 读取/分支/候选演示，不是 9B authoring 或完整语义审核。见[使用与边界](../docs/STRUCTURED-FLOW-WIRING.md) |
| [flow_contract_authoring](flow_contract_authoring.py) | 原文/宿主合同 → 受约束提案；参数词表、AND/OR 及引文结构约束，不证明语义 |
| [flow_read_region](flow_read_region.py) | 可选的两阶段读取入口：源文直接生成工具/参数/条件，编译器负责别名/边/结束节点；不是通用替代品 |
| [flow_condition_expression](flow_condition_expression.py) | 9B 提取 `and/or/not` → 白名单结构；代码计算真值，不使用 eval/exec |
| [flow_joint_conditions](flow_joint_conditions.py) | 非修改型核对，定位具体赋值、源引用、Tree/L0 节点；直接让模型填真值表仅作诊断 |
| [flow_joint_lowering](flow_joint_lowering.py) | 显式生成另一份未激活修订；保留原图、数据绑定、未知项和来源 |
| [flow_semantic_probe](flow_semantic_probe.py) | 冻结 1–12 案例，分列首次构造/条件阶段/惰性评分；预算、一次尝试、零调用回放 |
| [flow_tree](flow_tree.py) / [flow_checkpoint](flow_checkpoint.py) | 原有编译/源审查；共用摘要、原始响应及严格检查点 |
| [flow_behavior](flow_behavior.py) | 有限私有 Oracle + 原执行器/惰性 Provider，不证明完整 Skill 或独立语义 |

### 使用

公开包首先用 `scripts/netopyu-market-corpus translation-intake SNAPSHOT CANDIDATE_ID --output-root NEW_DIRECTORY` 检查输入。可提供 `--host-catalog`；原始 Schema 能被保留不等于能转换为以下旧 FlowSources。缺口不得通过截断原文、重命名参数或虚构工具消除。

从项目根目录执行。`sources.json` 是完整 FlowSources（原文、宿主输入、读合同和效果目标），不是仅 Markdown 路径。成功构造后的 `output/construction/candidate.json` 就是下条命令使用的 FlowTree；失败时该文件不存在，不应编造。

```bash
# 一次 9B 构造。已有完整目录可去掉预算，零调用回放。
.venv/bin/python -m evaluation.flow_contract_authoring author sources.json --output output/construction --max-new-calls 1

# 条件解释、结构表达式、联合比较及未激活修订。
.venv/bin/python -m evaluation.flow_condition_expression author sources.json output/construction/candidate.json --output output/condition --max-new-calls 1

# 离线重建保存的表达式，不调用模型或系统。
.venv/bin/python -m evaluation.flow_condition_expression derive sources.json output/construction/candidate.json --proposal output/condition/expression.json --output derivation.json

# 可选：只有符合两阶段读取范围时使用，不按 Oracle 自动择优。
.venv/bin/python -m evaluation.flow_read_region author sources.json --output output/read-region --max-new-calls 1

# 显式小批：私有 Oracle 不传入生成接口。
.venv/bin/python -m evaluation.flow_semantic_probe freeze output/probe --cases cases.json
.venv/bin/python -m evaluation.flow_semantic_probe run output/probe --max-new-calls 12
.venv/bin/python -m evaluation.flow_semantic_probe report output/probe --output output/probe/report.json
```

已知案例修订必须在 freeze 添加 `--evidence-role known_case_development_revision`。原文见[示例目录](../examples/semantic-transfer/README.md)，完整结果见[本轮报告](../docs/FLOW-SEMANTIC-TRANSFER.md)。源脚本不执行。

`flow_contract_authoring author --proposal rejected.json` 是显式编译报错修订：只接受真实编译失败的合法形状提案，在**另一输出目录**保存一次新尝试。它不接收 Oracle，也不会自动循环重试；本轮工单实验未观察到这条路径的改善。

### 能力与权威边界

- 辅助条件模块覆盖一次事实读取及后续目标读取，允许分支上等价目标/参数；最多 4 个布尔事实、16 个组合。当前表达式只支持标识符字段和布尔操作，不支持数值计算或特殊符号键名的显式索引。范围外是该模块未验证，**不是底层 FlowTree 不支持多分支**。
- 未知不变成通过；不把任意流程压成两步。全停止判断不自动生成“成功”修订。参数语义、引用蕴含及完整源审查仍独立存在。
- finite_agreement 是代码与模型解释一致，不是解释正确。查看 `expression.json`、`structured-expression.json`、`derivation.json` 的 comparison.rows、expressionOrigins、revision.tableOrigins 和 preservedUnverified，定位 L1→L0.5→L0。
- 候选永不因测试或 confidence 获得执行权限，默认 DSH 激活不变。

### 回放

原始失败、协议和成本见[实验索引](../docs/FLOW-EXPERIMENTS.md)。旧指纹在新代码下拒绝回放是预期行为，不得改 manifest 绕过。本轮回放以 Git 基线 `14baa0a` 为底，再覆盖该轮 source-snapshot.tar.gz，其他未改依赖由基线提供；位置及复验见[报告](../docs/FLOW-SEMANTIC-TRANSFER.md)。清理前历史快照仍为 c2ebd78。

## English

Current opt-in [semantic planning](../docs/SEMANTIC-PLAN.md) retains a source outline, resolves named observations, supports typed array length, and separates terminal duties from unsatisfied future execution requirements. `stage1_cases` supplies only known synthetic source/task/contracts; `stage1_validation` checks explicitly reviewed generated Trees using the existing engine. These do not constitute public-Skill generalization or large Runtime A/B.

`source_closed_program` constrains read successors, both decision paths and successor-free terminals. `source_inline_program` carries each node/duty's original source ID and original/compiled positions. These are lossless syntactic transformations, not semantic repair or certification.

`source_duty_accounting` adds opt-in pre-argument inventory/plan accounting and precise handoff artifacts. It reuses the original source inspector/compiler; structural path checks are not semantic entailment or authority. See [real 9B failures and scope](../docs/SOURCE-DUTY-ACCOUNTING.md).

`netdata_fixture` provides one synthetic in-process Function-call primitive. `netdata_task_demo` uses the original read gateway, bounded column binding and explicit page/domain/egress checks. It is developer wiring, not model translation or live Netdata; see [scope and reproduction](../docs/NETDATA-ISOLATED-VALIDATION.md).

`structured_authoring` adds frozen, explicitly budgeted 9B source-page requests and candidate lowering through the existing structured Tree compiler. No candidate or provider executes; the first two-call context stop is preserved in the [authoring report](../docs/PROGRESSIVE-STRUCTURED-AUTHORING.md). This is known-source development, not a semantic accuracy or independent holdout score.

`source_ledger`/`source_blocks` retain exact windows and citations; `source_retrieval`/`source_host_binding` separate delivery, review, declarations and authority. Optional obligation inspection, literal retrieval, catalog-directed shapes and host modes remain available; default is `direct`. `source_plan` now freezes the graph before per-slot arguments, exposes lexical sources, checks origins early and records semantics-preserving dead-terminal removal. A real 9B synthetic region compiles and passes ten local execution checks; public-source semantics remain open. See [current plan-first results](../docs/PLAN-FIRST-AUTHORING.md), [host modes](../docs/HOST-OPERATION-MODES.md), [catalog construction](../docs/CATALOG-DIRECTED-AUTHORING.md), [task alignment](../docs/SOURCE-OBLIGATION-AUTHORING.md), [boundaries](../docs/SOURCE-DECISION-AUTHORING.md) and [source windows](../docs/SOURCE-LEDGER-AUTHORING.md). These are not whole-Skill accuracy scores.

`task_alignment` prepares bound task/source/host dossiers while keeping the original source pages and future model inputs separate from developer review decisions. It records missing host schemas, source tensions and unsupported semantics without certifying acceptance. See [four concrete source cases](../docs/TASK-SOURCE-ALIGNMENT.md).

`structured_flow_tree` adds source-anchored nested-data trees lowering into FlowProposal v2 and the existing execution gateways. `structured_flow_demo` performs explicitly host-bound local fixture reads, branches and candidate generation, with zero writes/models. This is wiring, not complete semantic or model authoring. See [reproduction and boundaries](../docs/STRUCTURED-FLOW-WIRING.md).

`structured_binding_probe` exposes the versioned data-binding primitive through offline compile/materialize/demo commands. Nested schemas and explicit paths do not automatically integrate with legacy FlowSources or acquire authority. See [usage and limits](../docs/STRUCTURED-DATA-BINDING.md).

Start public packages with `netopyu-market-corpus translation-intake`: lossless inert source pages, explicit same-commit supplements and raw host-schema diagnostics. This does not compile cross-page semantics or widen the old L0 executor. Do not truncate source, rename parameters or invent tools to fit FlowSources. See [intake usage and evidence](../docs/TRANSLATION-INTAKE.md).

Public corpus tools now support repository-disjoint sampling, recovery of known script exclusions and `inert-text` quarantine. The static index displays original script evidence without making it a Runtime resource. See [batch preparation](../docs/PUBLIC-TRANSLATION-BATCH.md); this adds no semantic pass claims.

Recommended research path: **contract-grounded construction → readable source condition → deterministic compilation → full-source review**. Default DSH/Runtime is unchanged. Unary guard addition is historical diagnostics after measured regressions.

The model emits symbolic and/or/not expressions; an allowlisted parser and existing compiler handle logic without code execution. Joint checks locate disagreements; explicit revisions retain original candidates, data bindings, unknowns and provenance. Shared checkpoints enforce one attempt and strict replay. Use the commands above; known-case repairs require the explicit evidence-role flag.

The optional read-region author takes source/host contracts directly and lets code own aliases/edges/terminals. It is scope-limited, not an automatic replacement or oracle-selected fallback. Explicit compiler feedback is opt-in in a new checkpoint, with no observed improvement in this round. The expression language currently excludes numeric computation and indexed special-character field names.

This auxiliary checker covers one fact read and an equivalent downstream read across branches, at most four Boolean fields / sixteen assignments. Unsupported scope is not a limitation of the underlying FlowTree. Unknowns, all-stop judgments and malformed proposals do not grant authority. Parameters, citation entailment and whole-Skill semantics require separate review. Inspect raw/structured expressions, comparison rows and origins, not just a pass count.

See [results and immutable snapshots](../docs/FLOW-SEMANTIC-TRANSFER.md), [examples](../examples/semantic-transfer/README.md) and [history](../docs/FLOW-EXPERIMENTS.md). Replay with base commit 14baa0a plus the per-run source overlay; never bypass a fingerprint with edited manifests.
