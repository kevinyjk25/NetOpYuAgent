# 受控混合流程 / Governed Hybrid Flows

## 中文

状态：2026-09-10，依据用户补充确认的架构原则。**这是下一步实现约束，不表示节点已经可执行。** 当前 `network_runtime/l0/flow.py` 只有 read、branch、effect_candidate 和 end；`needs_l1` 是终止交接，不是同一图内可恢复的 LLM 节点。本文不放宽[阶段 1 原有验收](STAGE-1-EXIT.md)，不解锁阶段 2 或规模 Runtime A/B。

### 固定权力和流程，保留语义自由度

Runtime 管理可审查的混合任务图：确定性节点与受控推理任务按依赖串行，独立任务可以并行。LLM 仍作为 Reasoning Plane 提供推理服务，Runtime 负责调度、准入和记录。调度位置不改变执行权限，三平面和唯一 Effect 边界保持不变。

| 类型 | 可以负责 | 不得获得的权限 |
|---|---|---|
| 确定性节点 | 已声明 Capability、参数绑定、精确谓词、格式转换、验证 | 不能把 Provider 成功文字当成已验证事实 |
| 推理任务 | 理解、诊断、解释、抽取候选参数、提出方案 | 不能修改主图、扩大 Scope、执行工具、批准自己或提交成功终态 |
| 候选准入 | 输出 Schema、引用、资源、业务 Guard、Evidence 和风险检查 | 类型正确不等于语义正确，confidence 不授予权限 |
| 分支/汇合 | 已声明的依赖、条件、合并和失败方向 | LLM 不能跳过必需依赖或宣布并行工作已完成 |
| Effect 事务 | 复用原激活合同、审批、重校验、执行、独立验证、补偿 | 模型输出不能直接形成写权限或伪造执行/恢复证据 |

整体称为 **Governed Hybrid Skill（受控混合 Skill）**：严格片段仍是 L0，开放语义部分仍是 L1。含推理节点的整图不能称为“确定性的 L0”。

### 依赖与并行

例：资产读取与告警读取并行 → 固定 join 汇合 → LLM 生成诊断/变更候选 → 确定性参数、事实和资源校验 → 原审批/执行/验证/补偿链。

- 依赖同时包括数据和控制前置条件，先检查无环、来源可达、类型兼容。读取数据不自动代表业务前置条件满足。
- 独立读取/推理可并行，但并发、调用次数、时间预算与可见输入范围预先限定。
- join 必须声明 all-success 等规则；必需分支失败不能当成功跳过。可选分支、部分结果和冲突合并不能临场交给 LLM 决定。
- 冲突资源操作不得因为两个分支都有模型建议而并行执行；Effect 复用现有事务引擎，不新增平行写执行语义。
- 超时、取消、迟到回复分别记录。停止接纳结果不等于证明外部请求已取消；迟到结果不能恢复或授权下游。

### 推理节点的最低合同

绑定主图/节点摘要、原始自然语言指令及引用、输入投影和来源、模型/配置、输出 Schema、允许候选范围、预算、后继与失败方向。模型不得修改这些绑定。

输出只形成 `model_candidate`，不得自行升级为 `observed_evidence`、审批、验证成功或补偿完成。确定性准入独立于生成提示；无法机械核实的业务判断保留不确定性，必要时追问或人工处理。工具使用意图只能提出子计划，不能获得原生工具执行通道；动态扩图须另建摘要与资格，不能原地改批准图。源 Skill 脚本和引用继续作为惰性输入。

### 转译和验收

转译保留三类结果：严格片段、受控推理任务、追问/人工/不支持边界。不得为了增加静态比例丢掉自然语言职责，也不得把可精确表达的必要检查移到 LLM 来掩盖转译缺陷。

当前空列表判断已能用原数组长度能力表达，失败发生在“原文/清单 → 程序”的遗漏；这项确定性验收不改变。历史模型候选、失败、未完成请求与费用全部保留。

下一步实施顺序：

1. 定义混合图和候选/事实类型边界，接入串行读取—推理—准入，复用原合同/网关/事务。
2. 补依赖调度、并发预算、确定性 join 和失败/迟到结果处理；不恢复生产工程建设。
3. 转译器显式保留严格片段、原文推理节点和交接边界，展示源文到混合图的对应。
4. 分别验证 LLM 任务质量和 Runtime 约束：非法参数、越权、输出注入、缺失/过期事实、并行失败、资源冲突和伪造成功。模拟接口测试与真实 9B 结果分开披露。

指标分别报告严格片段覆盖、混合任务完成、LLM 语义错误/调用数、非法候选阻断、错误停止、未经授权的 Effect、验证/补偿，以及纯 Runtime 和含 LLM 端到端 p50/p95。混合完成率不是静态 L0 转译率；零观测到越权不是生产零概率；符合图约束不证明诊断永远正确。

## English

Status: user-confirmed design principle on 2026-09-10, **not implemented capability**. Current read Flow has read, branch, effect_candidate and end; needs_l1 terminates rather than resuming an in-graph model task. Original Stage 1 criteria remain; Stage 2 and large Runtime A/B are not unlocked.

The Runtime should govern a heterogeneous dependency graph of deterministic operations and bounded reasoning tasks, with sequential dependencies and declared parallelism. Reasoning remains a Reasoning Plane service. Scheduling it inside the workflow does not grant execution authority or collapse the three planes. Call the whole artifact a Governed Hybrid Skill: deterministic L0 regions and open-semantic L1 tasks retain distinct claims.

Model tasks may interpret, diagnose, summarize and propose parameters/plans. They may not mutate approved graphs, widen scopes, execute tools, approve themselves or declare verified success. Bind graph/node digests, original instructions/references, visible input projections/provenance, model/configuration, output schema, candidate scope, budgets, successors and failure routes. Outputs remain model_candidate, not observed_evidence or approval/execution proof. Schema validity is not semantic truth; confidence is not permission. Unverifiable judgments remain uncertain and may require clarification/human review. Dynamic subplans need new qualification rather than in-place graph mutation. Source scripts remain inert.

Example: parallel inventory/alarm reads, fixed join, bounded diagnosis, deterministic candidate/evidence admission, then the existing approval/revalidation/effect/verify/compensate transaction. Dependencies cover control and data. Bound concurrency/cost and define joins, optional branches, conflicts and failures beforehand. Required branch failure is not success. Conflicting resource effects cannot run concurrently merely because both were suggested. Reuse the original effect engine, not a parallel writer. Timeouts/cancellation do not prove external work stopped; late results cannot authorize or revive downstream work.

Translation retains strict regions, governed reasoning tasks and unresolved boundaries. Do not discard source duties to inflate static coverage or move expressible mandatory checks into LLMs to hide translator defects. The current empty-list omission is a checklist-to-program defect; the Runtime already supports length. Its deterministic test and all historical failures remain.

Implement narrow graph/authority types and serial read-reason-admission first; then bounded scheduling/joins/failure handling; source-to-hybrid-graph authoring; and negative tests for invalid arguments, scope escape, output injection, missing/stale facts, parallel failure, conflict and false success. Disclose simulations separately from real 9B runs. Production engineering stays deferred.

Report strict-region coverage, mixed task completion, model errors/calls, rejected invalid candidates, over-stops, unauthorized effects, verify/compensate outcomes and separate Runtime/model-inclusive latency. Mixed completion is not static L0 translation rate; zero observed escapes is not a production zero probability or universal diagnostic correctness.
