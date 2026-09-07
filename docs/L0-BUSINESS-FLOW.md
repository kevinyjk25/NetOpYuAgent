# 最小业务流程 / Minimal business flow

## 中文

2026-09-07，按已确认的“宿主工具上下文＋整 Skill 混合流程＋最小 Runtime 补齐”路线实施。当前完成 **C1 只读接线＋C2 本地单写事务门禁**，不是整个 C 阶段完成。既有 Effect/Saga/事务图不被替换。C1 候选本身仍不可执行；C2 通过宿主配置进入原写事务。

### 已能运行的内容

`network_runtime/l0/flow.py` 提供三个入口：

- `FlowProposal`：最多 64 个无环节点，包含 read、branch、end、effect_candidate。输入和读取结果使用现有封闭标量 Schema；不执行字符串表达式或脚本。
- `qualify_flow(proposal, reads, effects)`：宿主提供精确只读合同和写候选目标；检查图、字段、类型、源引用、全路径可用性，输出包含源摘要和工具合同的 flowDigest。结构通过不证明源语义保真。
- `run_read_flow(...)`：宿主另行绑定 flowDigest 和本次参数摘要。每次读取复用 `execute_host_read` 的权限与结果校验；条件和数据依赖由确定性代码处理。不是面向不可信请求的身份/令牌 API。

宿主上下文与 LLM 提案是不同参数：模型不能上传一个 Python callable 或 flowDigest 就取得权限。当前 `reads` 带源 Skill/tool/adapter 声明及合同，`bindings` 由宿主提供真实读取实现。`effects` 只是候选路由信息，不会加载 Provider 或授予写权限。

| 节点 | 行为 | 终态/边界 |
|---|---|---|
| read | 解析参数→权限检查→实际读取→结果结构检查 | 错误即 blocked；结果不是实时真值证明 |
| branch | 类型兼容的标量相等比较 | 明确 true/false；缺失、错误、过期不得默认为 false |
| end | 结束当前路径 | read_path_completed / needs_l1 / unsupported 分开计数 |
| effect_candidate | 解析并校验写候选参数 | awaiting_effect_admission；执行次数为 0 |

例如引用 `{"kind":"reference","source":"lookup","field":"device_id"}` 表示使用 `lookup` 的结构化输出，不是拼接字符串。引用节点必须出现在**所有**到达当前节点的路径上；不能从另一条未执行分支取值。可选字段允许在 Schema 声明，但运行时缺失则阻断，不能猜默认值。常量不接受数组、对象、NaN 或 Infinity；布尔值不冒充数值。

读取结果使用独立快照，记录原读取 receipt；分支轨迹记录源引用、判定和所选路径。权限失败或 Provider 异常不触发另一条业务分支。报告只保留异常类型，不直接返回可能含凭据的 Provider 异常正文。

### 本地复现与观测

输出路径必须不存在：

```sh
.venv/bin/python -m evaluation.flow_local_demo \
  --allow-local-read --output /tmp/read-flow-local-review.json
```

当前[本地报告](benchmarks/read-flow-local-summary.json)记录：

- `campus-sw1`：读取→site 等于 campus→引用第一次结果的设备 ID 再读→read_path_completed。
- `idc-sw1`：读取→site 不等于 campus→needs_l1。没有调用模型，更没有把 L1 待处理算成成功。
- 两个请求合计 **3 次实际本地文件读取，0 次写入，0 次模型调用**；库存文件内容保持不变。

[源流程说明](../examples/read-flow/flow-source.md)与[演示生成器](../evaluation/flow_local_demo.py)均为手工开发夹具。重复读取用于验证数据引用，不是业务最佳实践。此报告不计入公开 Skill 数量、9B 转译准确率或 Runtime 性能对比。

### 未闭合的安全边界与下一步

1. **C2 已接通本地单写入口**：`HostFlowGate` 每次从宿主上下文重新求值，不接受外部提供的“通过报告”。带门禁的 PreparedPlan 使用 schema 11，哈希包含流程/源合同摘要、请求摘要、读取身份与权限摘要、整条已选路径的读取事实/来源、目标 L0 和精确参数。原无门禁计划仍用 schema 10，保持旧哈希格式。宿主配置该目标门禁后，省略门禁及旧无绑定计划都不能通过该实例；这不是所有独立 Runtime 实例自动共享的全局策略。
2. **写前重校验**：审批证明绑定上述 planHash；执行重启后必须恢复相同宿主 gate。先重新读取并对比，再在原预检之后、发送之前再次读取。当前比较整条路径的完整已投影读值和 Provider 证据，可能因无关值变化而保守拒绝；尚未只挑与条件有关的最小字段。路线/参数/事实/权限/配置改变或读取失败都阻断，已签审批不豁免检查。返回变化是 PRECONDITION_CHANGED、未发送写入，释放锁，不补偿未发生的写入。
3. **时效边界**：C1 检查本进程读取完成后的经过时间；C2 从整条重读开始计时，并在发送前检查预算。它们不证明源数据本身新鲜，也不构成跨设备原子快照。最终重读到外部写入之间仍有 TOCTOU 窗口；生产级条件原子性需要 Provider CAS/版本条件写等，不得声称本实现 100% 稳定或实时设备事实已证明。
4. **C3 正向转译/审查**：从不可改写的 Skill 包和宿主合同生成整流程提案，逐节点审查遗漏、条件和读写归属。当前源摘要与结构门禁不等于这一能力已实现。公开 12 Skill 仍未开始完整转译实验。
5. 多写分支、并行、循环、嵌套组合、自动 L1 回入和任意脚本暂不支持。effect_candidate 是终端节点，禁止 next；没有跨多写的隐含事务保证。DSH 页面与任意外部 MCP 的自动注册不在本次接线成果中。

### C2 调用与实际结果

宿主构造 `HostFlowGate`（只读合同、实现、权限、写候选与精确流程/请求 consent），传入 `NetworkRuntime(..., flow_gates={"local": gate})`。调用 `prepare(..., flow_gate_id="local")` 后必须继续原 approve/execute API。它不会自动签发审批，不能由模型提交一个 gate 对象取代宿主配置。

```sh
.venv/bin/python -m evaluation.flow_effect_demo \
  --approve-local-simulation --output /tmp/flow-effect-review.json
```

[最终本地报告](benchmarks/flow-effect-local-v2-summary.json)包含：正常路径 verified_success；分支变化 precondition_changed 且 0 次 mock 变更；注入写后验证失败后 rollback_verified。读取实际临时库存文件，写入使用既有 LAN mock Provider；临时状态/环境变量运行后恢复，没有真实设备写入、没有 LLM 调用。园区库存触发 erin 授权是专门的接线夹具，不是经过源 Skill 转译的企业业务规则。[初次接线报告](benchmarks/flow-effect-local-summary.json)保留在最终发送前加固之前，不能替代 v2 结果。

整阶段验收仍要求“正向保真转译＋最小读写条件流程＋原事务验证/补偿”，不能用 C1 的只读结果代替。本地手工控制流先作为确定性回归，之后再分别统计 9B 首次生成和辅助修订结果。

## English

September 7, 2026: **C1 read-path wiring and C2 local single-Effect admission are implemented**, not the complete phase. Effect, Saga and transaction graphs are reused; C1 candidates alone remain unauthorized.

`FlowProposal` is a bounded acyclic graph (at most 64 nodes). `qualify_flow` binds source/tool contracts and validates field types, reachable nodes, dependencies and dominance: a referenced read must precede the consumer on every incoming path. Qualification is structural, not semantic alignment. `run_read_flow` requires separate trusted host consent for the exact flow/request and reuses `execute_host_read` for each read's access and result checks. Model text cannot supply execution callbacks or grant authority.

Nodes are read, typed equality branch, terminal outcome and non-executable effect candidate. Missing optional data, stale elapsed-read evidence, provider errors and access denial block instead of taking the false branch. Result snapshots and branch traces are retained; raw exception text is withheld. `needs_l1`, `unsupported`, `blocked` and `read_path_completed` remain distinct. The host effect-target dictionary is a candidate declaration, not a verified active binding or permission.

The command above performs three actual local file reads across campus and IDC requests, with zero writes/model calls and unchanged inventory bytes. Campus takes the second-read path using a prior result's device ID; IDC stops at `needs_l1`. See the [report](benchmarks/read-flow-local-summary.json), [source fixture](../examples/read-flow/flow-source.md) and [demo code](../evaluation/flow_local_demo.py). This is hand-authored wiring, not public-Skill translation evidence, live telemetry or a performance benchmark.

**C2:** a host-configured `HostFlowGate` reruns reads rather than trusting a supplied report. Flow-bound plans use schema 11 and bind flow/source contracts, request, read context, observed path/facts/provenance and exact target/arguments into the approval hash. Standalone plans retain schema 10/hash compatibility. A configured target cannot omit its gate or execute an older unbound plan in that Runtime instance. This is host-local policy, not automatic global enforcement across separate instances.

Execution requires the same restored host gate after restart. It reruns and compares reads, then reads again at the final dispatch boundary after original preflight. Changes to route, facts, context or arguments, and missing/failed reads, abort without Effect and release locks. Existing approval, one-shot execution, independent verification and compensation remain in force. Full projected payloads/provider evidence are compared conservatively; unrelated changes may also stop execution. The total re-read time is bounded but does not attest source freshness. A TOCTOU gap remains between the last read and external write; atomic Provider conditions/CAS are not implemented.

The second command above exercises the original mock Effect engine with actual temporary local-file reads. The [v2 report](benchmarks/flow-effect-local-v2-summary.json) records verified_success, zero-write precondition_changed and rollback_verified after injected verification failure. The campus-to-erin rule is a synthetic wiring fixture, not translated business policy. No model or real-device write occurs. The [first report](benchmarks/flow-effect-local-summary.json) predates final-dispatch hardening and is retained separately.

Remaining **C3**: source-backed whole-flow 9B generation and per-node semantic review. The twelve public Skills remain untested for whole-Skill translation. DSH UI wiring and arbitrary external MCP auto-registration are not claimed by this local API integration.

Multiple effects, concurrency, loops, nesting, automatic L1 reentry and arbitrary scripts remain unsupported. Effect candidates cannot have successors. Full-stage acceptance still requires forward fidelity plus conditional read/write execution and original transaction verification/recovery, not just this read-path smoke.
