# 任务直达交付：简化中间层及限定诊断 / Task-first delivery

## 中文

历史快照：本文记录任务直达包当时的固定实现及两臂结果。后续[快照读取语义](SNAPSHOT-READ-SEMANTICS.md)补齐数据不可刷新声明及补读去重；本文旧样例没有重测，分数和制品不变。

2026-09-15：可选任务直达模式已实现，**有局部内容改善，语义阶段仍未完成**。本轮没有修改原任务、业务判据、历史评分或正式门禁，也没有新增自审器。

### 诊断与调整

复查原始 CAPA、Mesh 制品后发现：原任务和观察未在传输中丢失；但模型先选的 kind/source_ref 被固定为交付结构，选择却可能引用“先读导出”“最多三节”，而不是对应的业务交付义务。引用检查只能证明文本存在，不能证明选对。旧 renderer 还自动加 Analysis/Next steps 标题，可能破坏原任务格式。

这些证据不能说明所有业务遗漏都由 Schema 导致。因此调整的是一个可分离机制：**开放推理直接回答完整任务，不再先把任务压缩成模型选择的类型化交付槽。** 原类型化接口保留为历史兼容；本轮不自动升级现有会话。

| 仍由 Runtime 严格控制 | 留给 L1/LLM，并明确未获证明 |
|---|---|
| 原始任务及来源摘要绑定 | 开放任务的理解、分析、结论、建议 |
| read_prefix 编译、工具参数、精确资源 ACL | 回答是否覆盖全部业务要求 |
| 有限补读、证据冻结、一次生成、不重放 | 引用内容是否支持自然语言推断 |
| 工件静态检查、宿主终态交付 | 文本整体正确性、完整查询可执行性 |
| 原受审 L0 的权限、验证和 Effect 边界不变 | 此处的 answer 绝不是自动激活的 L0 |

### 接入和交互

操作者对**新的隔离会话**选择宿主 `apiVersion: netopyu.io/local-hybrid-host/v4`。其他 host 字段、packet、resources 和原只读隔离设置不变。`v1/v2/v3` 不自动迁移；不要修改运行中会话的宿主配置或源码。

```text
prepare(task, arguments)
  → 完整任务/Skill/工具及计划 Schema
submit(session_id, plan=read_prefix, delivery=null)
  → 严格读取前段 → 至多两次同网关补读
draft(session_id)
  → 冻结证据 → 一个受限 LLM 节点 → {"answer":"完整原任务的回答"}
  → 原静态检查 → 宿主终态，不能自行批准业务成功
```

未执行前发生原生 fallback 时仍先 `submit(plan=null, delivery=null)` 绑定宿主任务，再经同网关读取；原生 Agent 用 `deliver(session_id, response_json)` 提交同一 answer 对象。不会调用 Runtime 模型或执行文本。两条路径共用检查和渲染，不增加第三个执行器。

`task-bound-delivery/v1` 记录完整原任务和来源摘要，没有模型选出的 requirements；模型必须传 `delivery=null`，不能提交替换任务、片段或类型。返回对象只要求非空 answer，宿主逐字保留，不插标题或自动补结论。不能提供时在 answer 说明限制。

`shapeComplete=true` 只表示非空回答信封；`declaredCoverageComplete=null`、`semanticCoverage=not_assessed`、`semanticApproval=false`、`taskSuccess=null`。代码片段仍接受原有非执行式静态检查；缺查询的纯说明仍是未验证回答，**不能计为查询完成或 L0 转译成功**。降低开放回答的序列化负担没有降低任何执行权限门槛。

代码入口：[交付绑定/渲染](../skill_authoring/delivery.py)、[会话与两条路径](../dsh_adapter/hybrid_session.py)、[DSH六工具](../dsh-plugin-netopyu/src/index.js)。需原样终态展示时同时按[终态接入说明](HOST-TERMINAL-DELIVERY.md)开启宿主结束模式；本轮没有修改或重启用户 UI。

### 一次冻结两臂诊断：不是新的 DSH 或 Runtime A/B

使用上一批 Mesh 已封存的单个模型请求：任务、Skill、观察、authoring 注释、模型制品、seed、temperature、上下文和输出 token 上限完全相同。只有交付 contract、相应指导后缀和输出 Schema 一起作为一个呈现机制改变。它们不是三个独立归因变量。

先冻结源码、两份请求、运行顺序和审查维度，再按 choice→task 各调用一次本机 `qwen3.5:9b`；不重试、不调参续跑、不访问工具/设备。旧结构臂的原始回复与历史回复**逐字相同**，重现了冗余重读和无用建议。新臂不是重新执行原图，没有实际 DSH 会话，也没有新增 Skill 分母。

| 开发者 AI 审阅维度 | 类型化交付 | 任务直达 |
|---|---|---|
| 南北区域错误率与 p99 数值 | 保留 | 保留 |
| 区分部署相关性与已证明根因 | 保留 | 保留 |
| 提出具体的新 trace/log/saturation 证据 | 没有具体请求 | 请求北区 09:05–09:10 trace spans |
| 不重复建议读取已有快照 | **不满足** | **仍不满足**：首选建议重读同一文件 |
| 明确 checkout→payments 业务流 | 未完整写出 | 未完整写出 |
| 原任务格式 | 本例仍在三节内，宿主有附加标题 | 三段，无宿主附加章节；不宣称本例格式得分提升 |

新臂恢复了有用下一证据的表达，但仍含重复读建议；**不宣布任务完整成功，不把内容审查冒充原判据中“实际读取”的执行验证，不改旧 DSH 的 0/2。** 审查者是本项目开发者 AI，不是独立人工 Gold；五个审查维度在调用前冻结，上表对 facts_and_scope 展开显示数值与业务流的不同结果。

| 本次单请求观测 | 类型化交付 | 任务直达 |
|---|---:|---:|
| 输入 token | 7,250 | 6,857 |
| 输出 token | 408 | 260 |
| 调用耗时 | 40.76秒 | 33.81秒 |

共两次真实模型调用，14,107 输入／668 输出 token。这是一个已知快照，顺序和缓存会影响时延，期间有小型单元/脚本化接线测试；没有与全量 pytest 并发。不能宣称总体准确率、p50/p95 改善、DSH 端到端加速或跨 Skill 泛化。

### 接线与回归

扩展现有终态探针支持 `--task-bound`，不新增独立测试框架。实际安装 DSH＋脚本化模型对候选、拒绝两条路径 **2/2** 通过：每条六次协议请求、一次实际只读调用、正常终态输出；与上面的真实模型内容诊断分别计量。

```bash
# 零真实模型的实际 DSH 接线检查，输出必须是不存在的新目录
.venv/bin/python -m evaluation.hybrid_terminal_probe /tmp/task-direct-terminal-new --task-bound
```

21 项新增任务绑定/会话回归覆盖全文保留、无法替换任务、逐字呈现、旧协议不降级、坏信封、静态工件检查、原生/Runtime 两条路径、冻结后拒读和不重放；另 4 项对照输入冻结检查。工程测试不是语义样本。

- [可携带摘要与逐项审查](benchmarks/task-first-delivery-summary.json)
- [两臂冻结输入](../artifacts/governed-session-20260915-task-direct-pair/freeze.json)／[原始调用结果](../artifacts/governed-session-20260915-task-direct-pair/summary/report.json)
- [旧结构臂原文](../artifacts/governed-session-20260915-task-direct-pair/choice/delivery/report.json)／[任务直达原文](../artifacts/governed-session-20260915-task-direct-pair/task/delivery/report.json)
- [实际DSH接线](../artifacts/governed-session-20260915-task-direct-terminal/summary/report.json)

10 份两臂制品/71 份归档源码、66 份接线制品/76 份归档源码全部核对，与当前相应实现一致。原报告不改；`artifacts/`不随 Git 自动发布。206项定向（含文档检查）、**3,254项全量＋81子测试（245.49秒）**、114个变更/新增项目Python文件Ruff和diff检查通过；不是全库历史lint-clean声明。没有提交、推送、运行来源脚本或授权写入。

### 收敛结论

消除了一个不可靠的语义选择中间层及宿主格式干扰；一次真实诊断出现局部改善，重复读建议仍未解决。CAPA 的明确结论和 IRQL 查询没有重测。不能继续声称只是工程进展，但也远未达到项目核心验证完成。

下一步应固定任务直达机制，围绕“已有证据为何被当成下一步、业务对象为何遗漏”做独立新来源的限定诊断；诊断与正式门禁分别记录。不要在同一个 Mesh 快照继续调模板或加审阅层。若要改变原阶段出口或放宽执行准入，须单独决策；本包没有作这种改变。

## English

Historical snapshot: the later [snapshot-read package](SNAPSHOT-READ-SEMANTICS.md) adds immutable-data semantics and repeated-follow-up protection. This report's original pair was not rerun or regraded; source comparisons refer to its fixed version.

The opt-in v4 local host implements task-first delivery: bind the complete original task and source identities, require `delivery=null` at submit, then return one `answer` from the existing bounded Runtime reason node or native fallback. Model-selected kinds/source fragments no longer define the open answer's shape. The host preserves answer text without additional headings. Legacy profiles remain unchanged.

This removes an unreliable presentation intermediate, **not** the active reviewed L0 or any authorization/parameter/evidence/Effect gate. Exact-resource reads, limited follow-ups, frozen evidence, one generation, static artifact checks and terminal/no-replay controls remain. An open L1 answer is not executable L0; nonblank shape acceptance keeps semantic coverage unknown, semantic approval false and task success null. An explanation without a requested query is not a completed query.

One pre-frozen diagnostic used the exact historical Mesh task, Skill, observations, authoring annotations and local 9B settings. Only the delivery contract, corresponding instruction suffix and output schema changed together. Two calls, no retry or tuning; no new DSH session, Runtime graph execution, tools or fresh Skills. The control reply exactly reproduced the historical raw reply. Task-first retained regional metrics and causal uncertainty, and added a concrete north-region trace request. **It still recommended rereading the same snapshot**, and neither answer fully named checkout→payments. No complete-task or generalization success is claimed; old DSH scores remain unchanged.

Control: 7,250 input / 408 output tokens, 40.76s. Task-first: 6,857 / 260 tokens, 33.81s. Total: two real calls,14,107/668 tokens. One known snapshot, fixed order, caching and concurrent small mechanical tests preclude population or performance claims; full pytest started only after these calls. Review is developer AI, not independent human Gold. The five review dimensions were frozen before generation.

Separately, the existing actual-DSH/scripted-model terminal probe supports `--task-bound`; candidate/rejection both pass, six protocol requests and one read each.25 new unit cases check task preservation, no model replacement, exact rendering, existing artifact checks, both routes, replay boundaries and frozen comparison inputs.206 targeted checks including documentation,3,254 full tests plus81 subtests in245.49s,Ruff on114 changed/new project Python files,and diff checks pass. This does not claim all historical files are lint-clean. Evidence hashes verify:10 files/71 archived sources for the model pair;66/76 for terminal integration. [Portable review](benchmarks/task-first-delivery-summary.json) retains both positive and negative observations.

The mechanism has a local improvement signal, not a closed semantic stage. CAPA and IRQL were not retested. Do not tune the same snapshot or add self-judges; next diagnosis must address evidence reuse and omitted business entities under frozen inputs. Research exits and execution gates are unchanged. No default UI change/restart, commit, push, source scripts or writes.
