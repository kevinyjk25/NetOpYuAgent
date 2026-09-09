# 宿主操作模式与行动来源 / Host Operation Modes and Action Witnesses

## 中文

更新：2026-09-09，C3v。新增研究入口 `mode_bound`，补齐上一轮发现的“参数 Schema 合格，但参数组合被宿主拒绝”的声明缺口。**这是参数组合的机械保证，不是整 Skill 转译准确率。** 默认仍为 `direct`，未修改 Runtime 执行器、授权门禁或 DSH 路由。旧 [C3u 结果](CATALOG-DIRECTED-AUTHORING.md)保留。

### 声明什么，如何检查

[source_modes](../evaluation/source_modes.py)接收独立版本化侧包 `netopyu.io/host-operation-modes/v1`。每个工具声明绑定完整 catalog、读取合同、输入/输出 Schema 摘要；原六字段转译输入不被替换。声明由宿主集成人员审核，既不是独立 Gold，也不是经过身份认证的远端宿主证明。

每个模式描述指定 JSON 对象的 `requiredKeys` 和 `allowedKeys`。例如一个通用 API 的 `/request`：`describe` 模式只接受 `inspect`，`search` 模式要求且只接受 `filter`、`limit`。这是示意规则，不自动适配任何 API。

| 检查层 | 实现 | 不承诺 |
|---|---|---|
| 声明 | 摘要一致；工具/模式/对象规则唯一；字段确实存在；不得禁止原必填字段 | 声明内容真实、完整或代表权限 |
| 互斥 | 模式必须有可证明互斥的必填/允许字段集合 | 任意 JSON Schema 条件求解 |
| 生成 | 选定模式后按收窄 Schema 构造；普通对象/数组拆开，字面叶子逐项给出处 | 选择正确模式、值符合用户业务意图 |
| 降低到原 Tree | 从显式对象构造或常量证明受约束对象的键集合；动态引用不能隐藏组合 | 仅靠引用类型重叠就证明完整子类型关系 |
| 原执行链 | 原类型/值校验、身份、范围、Evidence、时效等仍适用 | 候选自动激活或业务成功 |

只支持“对象存在、必填键、允许键”的有限合同。需要复杂值条件、循环、跨结果谓词或数组元素模式时，必须显式扩展设计，不能把未支持规则丢掉。对象路径必须是显式不可空的 object 类型；共享 `$defs` 按被修改路径独立展开，不污染其他字段。若受约束对象是完整的动态引用，本阶段直接阻止降级放行；普通叶值仍可动态绑定，并由原 Runtime 逐次校验。

新增声明改变了转译器获得的信息。即使原宿主代码、任务和源包不变，也不能称为“相同信息、只换算法”的对照。

### 如何定位“为什么生成这一步”

每个读取包含 `operationMode`（若该工具有模式声明），并沿用 `source.block_id` 选择行动依据。代码按 ID 回填**完整原文块**，不用模型重新抄写。输出 `operation-plan.json` 将它与 Tree 路径、所选工具/模式、原文路径/坐标、键集合证明关联起来。参数来源仍保存在 `catalog-lowering.json`。

`blockResolved=true` 只证明块能定位，不证明它要求当前调用。示例、规则、参数说明仍可能被模型误读为额外步骤。因此报告保留 `sourceActionEntailmentProven=false`、`sourceOrderCorrectnessProven=false`；重复实参成组展示，不自动去重。本轮是**可诊断的行动轨迹**，尚不是一个经过语义审查的独立“先规划、后填参”编译阶段。

本轮第一次实现要求模型输出精确 `actionQuote`，结果模型把多行命令压成单行，重现历史上已解决的引文抄写问题。该 v12 失败完整保留、原实现零调用回放一致；v13 撤销重复抄写，复用源块机制，不做模糊匹配或修改原模型答案。

### 使用

```bash
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN \
  --inputs INPUT.json --bindings BINDINGS.json \
  --operations OPERATIONS.json --profile mode_bound
.venv/bin/python -m evaluation.source_ledger run NEW_RUN \
  --max-new-calls 2 --report-dir NEW_REPORT
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

所有路径必须使用新目录。manifest 封存原输入、模式声明、源码/环境/模型版本；历史断点不能用新实现覆盖或自动重试。模式声明只参与离线构造，不运行源脚本、不调用底层宿主或获取凭据。

### 本轮证据与下一步

同一已知 Netdata 源包、离线任务、原宿主合同和参数名映射，新增模式声明。**两次新的 qwen3.5:9b 调用均产生候选，均未编译通过；底层宿主和源脚本调用均为 0。** [绑定证据](../artifacts/translator-v2/source-modes-20260909/evidence-summary/report.json)保留完整结果，[v12](../artifacts/translator-v2/source-modes-20260909/report-v12/report.json)与[v13](../artifacts/translator-v2/source-modes-20260909/report-v13/report.json)分别封存，不覆盖失败。这些 artifact 是本地证据，不随 Git 分发。

| 观察项 | C3u | C3v v12 精确引文 | C3v v13 源块回填 |
|---|---:|---:|---:|
| 读取语句 / 不同实参 | 8 / 1 | 2 / 1 | 8 / 1 |
| discovery/query 混用的读取 | 8 | 0 | 0 |
| 原宿主 Schema 错误消息 | 0 | 0 | 0 |
| 来源核验通过 / 所检查的 literal 出处 | 0 / 8（整对象） | 3 / 6（逐叶） | 10 / 24（逐叶） |
| 精确行动引文错误 | 不适用 | 2 | 不再要求抄写 |
| 未决项 / 显式终止 | 7 / 0 | 4 / 0 | 7 / 0 |
| 编译通过区域 | 0 | 0 | 0 |
| 请求耗时 | 214.388 秒 | 101.794 秒 | 215.441 秒 |
| 输入 / 输出 token | 7,787 / 3,452 | 8,115 / 1,165 | 8,113 / 3,228 |

模式混用缺陷已在这两次生成中消除，但**整体可用性没有改善**。v13 再次生成 8 个相同 discovery 读取；14 个逐叶出处失败、7 个未决项、1 个缺少终止，共 22 项阻塞。有效来源中包含同一节点参数的重复引用，不能算 10 个独立成功用例。来源粒度改变，不把前后分母直接当准确率比较；单次时延不是 p50/p95 或稳定提升。

具体定位：模型在命令使用块里引用 `"snmp:traps"`，而该块只有 `$SNMP_TRAPS_FUNCTION`，定义在另一块；有的块仅是标题、规则或字段表，却被映射成一次调用。未决项仍混入四个执行门禁名称。宿主旧映射 prose 中还残留早期 Cloud 任务描述，本轮不静默改动封存输入，下一步需分离宿主限制与当前任务要求。

首个静态请求字节代理 43,091 超限，未调用模型。共享 Schema 定义、压缩本地 Schema 标识符及不重复传输已封存的合同摘要/prose 后，v13 为 **40,140 / 40,960**；完整源文、模式约束和原宿主 Schema 均保留，模型预算未增加。不是精确 tokenizer 认证。

两版均通过各自源码快照隔离回放，零新模型调用、报告逐字节一致。证据摘要绑定 39 个文件；此前四阶段的 236 份绑定文件未变。新增 40 项机械测试；300 项跨模块定向通过（24.96 秒），全量 **2243 passed + 81 subtests passed（255.63 秒）**，19 个相关 Python 文件 Ruff、25 个未跟踪源码/文档及已跟踪变更的空白检查、文档链接检查通过。不据测试数量推断语义准确率，不扩展 Runtime A/B。修改保留在 dev，未提交、未推送。

下一阶段必须由“边生成参数边猜流程”改为独立操作规划：

1. 分开源行动、示例、替代路径、前置条件和任务外职责；运行时门禁是未来执行要求，不能凭名称判断当前缺失。
2. 先冻结可审查的操作骨架、分支与显式终止，再填参；不靠删除重复调用或自动补成功节点换取通过。
3. 参数声明/使用位置分开追踪，原变量定义、使用和实参来源相互关联；不执行脚本，不推断未知变量或凭据。

先闭合已知源，再做换源验证。正式 ≥3 不重叠 cohort／≥50 Skill／≥15 仓库／≥8 领域／≥600 case 的泛化门禁不变。

## English

C3v adds the opt-in `mode_bound` authoring profile. A digest-bound, developer-reviewed host supplement declares required and allowed keys at explicit object paths. It refines the original schema rather than replacing it. Modes must be structurally exclusive; drift, conflicting required keys, undeclared properties and unsupported refinements fail closed. This is neither authenticated host attestation nor independent semantic Gold.

Generation uses mode-specific argument shapes and leaf-level literal origins. Before lowering to the existing Tree compiler, the constrained object's key set must be established from construction syntax or constants. A dynamic reference cannot conceal that shape: type overlap is not a subtyping proof. Normal dynamic leaves still require original per-instance validation. No Runtime executor, permissions, routing defaults or activation policy changes.

Each read reuses `source.block_id`; code rehydrates the entire exact block in the operation report alongside coordinates, tool, mode and Tree location. Block resolution is not action entailment; examples and constraints can still be mistaken for operations. Duplicate requests are preserved and reported, not automatically deleted. This is an auditable action trace, **not yet a separate semantically reviewed plan-before-arguments stage**. The first v12 attempt required an exact action quotation, reintroducing flattened-command transcription failures. That failed run is retained and replays under its original source snapshot; v13 restores block rehydration instead of fuzzy-matching or repairing a recorded answer.

Two new qwen3.5:9b calls on the same known Skill/task generate candidates, but **neither compiles**. v12 takes 101.794 seconds (8,115/1,165 input/output tokens); v13 takes 215.441 seconds (8,113/3,228). Both eliminate mixed discovery/query arguments while retaining original-schema compliance. v12 contains two identical discovery reads; v13 again contains eight. The final candidate has fourteen invalid leaf origins, seven unresolved entries and no terminal: twenty-two blocking issues. Provider/source-script calls remain zero. No whole-Skill availability gain or semantic accuracy is established.

The table above reports all attempts. Ten verified literal occurrences in v13 include repeated node origins, not ten successful cases. Aggregate-origin versus leaf-origin denominators are not comparable accuracy rates. One-request timings are not latency distributions. Mode information and representation change together, preventing unchanged-information or single-mechanism causal claims.

Both versions replay byte-identically from isolated original source snapshots, with zero new calls. The summary binds 39 files; 236 previously bound files remain unchanged. Forty new mechanical tests, 300 targeted checks (24.96 seconds), and **2243 full-suite tests plus 81 subtests (255.63 seconds)** pass, along with Ruff on nineteen related Python files, tracked/untracked whitespace checks and document links. These counts are not semantic accuracy. Changes remain uncommitted/unpushed on dev. Static preflight initially exceeds budget at 43,091 bytes and is not sent. Shared definitions, schema-identifier compaction and omission of repeated frozen declaration metadata bring v13 to 40,140/40,960 without deleting source text/constraints or increasing model limits; this is not tokenizer attestation.

Use the commands above with new directories; failures are never overwritten or retried. The next step is an independent, source-grounded operation plan before argument construction: distinguish alternatives from sequences, preserve prerequisites/explicit exits, and link variable definitions to uses. Stale task-related prose in the host binding must be separated in future inputs, not silently repaired in sealed experiments. Broad Runtime evaluation stays gated by cross-Skill semantic generalization.
