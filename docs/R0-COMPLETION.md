# R0 测量验收与完成复核 / R0 Measurement Acceptance and Closure

## 中文

### 当前结论

2026-09-18：**R0 工程阶段完成，状态为 `r0_engineering_complete`；未进入 R1。** 用户明确批准一次关联零推理重验后，同一封存的 6 Skill／12 Task 完成 12 对／24 臂机械验收。它证明测量与接线可用，**不证明 9B 的语义准确率、转译泛化或生产安全概率**。

[完成复核机器摘要](benchmarks/r0-reacceptance-20260918-summary.json)记录清单、逐臂成本和原始报告摘要；[授权修订](../data/bounded-pilot/r0-reacceptance-20260918.json)记录本次例外，原失败与原账本不重置。

### 两次记录分别保留

| 记录 | 分配臂数 | 实际启动 | 机械完成 | 未运行 | 真实实验模型调用 |
|---|---:|---:|---:|---:|---:|
| 9 月 17 日首次验收 | 24 | 1 | 0 | 23 | 0 |
| 9 月 18 日获批重验 | 24 | 24 | 24 | 0 | 0 |

累计是 **48 个分配臂、25 个实际启动、24 个机械完成、1 个原测量失败、23 个原未运行**，仍只有 12 个不同任务，不是 48 个独立样本。原失败未改判；不能把跨批结果写成“从未失败的 24/24”。

原失败原因：DSH 把 Mesh Skill 的 `{{destination_service_name}}` 误当 `deployment:persona` 的宿主模板变量，尚未调用模型或工具就停止。修复通过宿主注册的不透明变量插入完整原文，不删除、不改写 Skill 字节。原生 renderer／插件的 10 类逐字节保真检查与本次实际 DSH 回合均已通过。

原 v1 报告误把 `len(rows)=1` 命名为 `completed_arms`，并把未创建的 receiver 当成清理失败；实际为 1 个尝试、0 个完成、1 个已创建且已排空的 receiver。[原失败摘要](benchmarks/r0-closure-20260917-summary.json)披露修正含义；原始 JSON 保持不变。

### R0 出口清单

| 项目 | 本次证据 |
|---|---|
| 封存输入与参考 | 6 Skill／6 仓库／12 Task，8 正向＋4 边界；双角色 AI 原标注、补充、裁决、源码与模拟初态均按字节绑定；材料摘要保持不变 |
| 完整控制器 | `claimed=24 / protocol_completed=24 / mechanics_passed=24 / unrun=0`；固定均衡 AB/BA、相同输入、不同 Provider 数据库且初态读回一致 |
| 调用与递送 | 54 次实际本地 Provider 读取：原生 27、Runtime 22、fallback 5；工具结果与 call ID 在 DSH 后续真实请求中匹配，无退出后补执行 |
| 精确计量与预算 | 100 次替身传输请求全部 settled：agent 73、compiler 22、fallback 5；无未知用量／悬挂请求；每臂均在原 420 秒、10 请求、64,000 输入＋6,000 输出限额内 |
| 生命周期 | 24 个 receiver、各臂宿主与 broker 全部关闭排空；所有臂终态 completed |
| 不变性与一次性 | 声明执行依赖前后相同；原 halted study 的持久行保持原样；同一工程库中的固定子记录已领取并消费，不能换目录重跑 |
| 工程回归 | 完整 `pytest --test-suite=all`：**4,030 项测试＋81 项子测试通过，281.31 秒**；变更范围 Ruff、Node／shell 语法、文档与权限边界、diff 检查通过 |

100 次是模拟模型端点的实际协议请求，**不是 100 次真实推理**。输入合计 390,953 token，由固定离线 tokenizer 计数；输出合计 100 是明确的 fixture 用量，不是模型生成 token。累计 arm wall time 为 1,233.09 秒，包含重复资产哈希；最长臂 74.53 秒、单臂最多 6 请求／33,510 输入 token。这些是本次工程成本，不是 9B 性能对照。

既有评分器 24 个正反例、预算 10 个探针和模拟事务网关三条路径保留为独立机制证据，不充当本批自动 Effect 桥接或 36 项 R1 门禁。旧 exact-send smoke 的 arm 曾保留 active，只可证明请求绑定；本次完整批另行提供终态与排空证据。

### 证据边界

- 来源与职责详见[独立材料包](../data/bounded-pilot/r0-development-20260917/README.md)。它是已知开发材料，标记为 **AI-assisted developer evaluation，不是真人人工 Gold**，不是未见确认集。
- 完整原文、引用、任务与工具合同进入运行侧；参考答案和机械探针脚本不进入 Agent 输入。固定替身根据预声明步骤返回响应，不是自主选工具或整 Skill 自动转译成绩。语义评分保持 unknown。
- B 在实际 DSH 工具回合内最多编译两个只读前段；剩余读取走原生 fallback，共用原 arm 预算。没有执行下载的 Skill 脚本、外部业务写入或真实 LLM。
- 全部 YAML/KQL 正确性、因果推理、语义脱敏不因局部严格校验而变为确定性证明。自动提取错误仍属于后续固定分母，不能用“不支持”剔除。
- 资产、解释器、已声明项目／Python／DSH 依赖前后冻结；OS/kernel 仍是明示信任边界，不是整机供应链证明。同步 I/O 与 Python 回调不保证强制硬实时取消。
- 原两工作日上限不重新起算；没有完整历史工时账，不能补称已证明历史工时合规。此次只执行用户批准的一次关联重验，没有增加 R1/R2 的候选、调用或样本预算。

材料摘要：`sha256:8d7d66ac5f07049f4e84bf65f7d5397a48f4a55eb2ddd805b11a85e6a69af079`。本次依赖摘要：`sha256:09d6a639f11beac2428f32c62abbd07beb39133b3737f54a5932c22642da5b15`。实际机械报告摘要：`sha256:022d97e56fd661dcfdfcbe7ff9eb878e27ca3bf081665edc875562292c6ecc76`。

原始机械报告仍保留 `r0Complete=false` 和 `r0_completion_requires_separate_full_checklist=true`：它只判机械批次，不能单独宣布整个阶段完成。**本文件的出口复核与完成机器摘要**才给出 R0 工程完成决定，不追改原始报告。

### 使用、提交与下一步

材料可独立校验，不调用模型：

```bash
.venv/bin/python -m evaluation.bounded_material verify data/bounded-pilot/r0-development-20260917
```

本次实际命令是 `python -B -m evaluation.bounded_acceptance` 加四个材料／资产／codec／输出位置参数及 `--reacceptance data/bounded-pilot/r0-reacceptance-20260918.json`，并使用全新空的绝对 `PYTHONPYCACHEPREFIX`。它已消费一次性许可，**不要直接再次执行**。固定工程库是 `artifacts/bounded-r0-engineering/budget.sqlite3`；子 study 的 active 注册状态不表示还有运行中的臂，也不授予新的重验额度。

本轮完成后只做用户要求的本地提交，不自动推送或进入 R1。第三方许可、归属和 tokenizer 来源 pin 见[包外记录](R0-THIRD-PARTY-NOTICES.md)；公开发布需单独核对，不悄悄改写封存输入。

下一步 R1 仍按[原有限协议](EVALUATION-RESET-20260916.md)：三个通用接管点、36 个机制探针、有限真实 DSH＋9B 开发筛选。**真实生成适配／计量等价性、自动 Effect 桥接和盲化结果审阅仍待验收**；当前可选 reviewer 接收含 arm/route 的观察，不能冒称已验证盲审。最多两个候选，失败按协议停止；R2 一次不相交确认，R3 给出明确结论。R0 完成不放宽正式泛化门禁或生产写权限。

## English

### Decision and preserved attempts

September 18, 2026: **R0 engineering is complete; R1 has not started.** Following explicit authorization for one linked zero-inference reacceptance, the same six Skills and twelve development tasks completed all twenty-four mechanical arms. This establishes measurement and integration, not 9B semantic accuracy, translation generalization or production safety probability.

See the [completion review](benchmarks/r0-reacceptance-20260918-summary.json), [explicit amendment](../data/bounded-pilot/r0-reacceptance-20260918.json) and [preserved original failure](benchmarks/r0-closure-20260917-summary.json). The first run assigned 24 arms, attempted one, completed zero and left 23 unrun. The approved repeat assigned/attempted/completed 24 with none unrun. Across both records: **48 assignments, 25 starts, 24 completions, one retained measurement failure and 23 retained unrun assignments**, still only twelve distinct tasks. Do not describe this as a failure-free original 24/24 or forty-eight independent samples.

The original failure occurred before model/tool calls: DSH interpreted literal Skill `{{destination_service_name}}` as a persona-template variable. Opaque host-variable substitution preserves the complete source bytes. Ten native-renderer/plugin fixtures pass byte-for-byte, and the actual DSH batch now passes. Original v1 attempted/completed and receiver-cleanup mislabels are disclosed without modifying the JSON.

### Closure evidence

- Fixed balanced AB/BA schedule: 24 claimed, protocol-completed and mechanically-passed arms; zero unrun. Paired inputs match, with separate Provider databases and matching initial-state readback.
- Fifty-four actual local Provider reads: 27 native, 22 Runtime and five fallback. Results/call IDs appear in subsequent actual DSH requests; no evaluator execution after DSH exit.
- One hundred stand-in transport requests all settled: 73 agent, 22 compiler and five fallback. Zero real experimental inference, unknown usage or pending requests. All arms satisfy the unchanged 420-second/10-request/64,000-input/6,000-output limits.
- All twenty-four receivers and per-arm hosts/brokers close and drain. Declared dependencies remain identical; original halted parent rows remain unchanged. The single linked child claim is consumed.
- Full regression: **4,030 tests and 81 subtests pass in 281.31 seconds**. Scoped Ruff, Node/shell syntax, final documentation/authority and diff checks pass. These are engineering tests, not 4,030 agent tasks.

The 390,953 input tokens are actual offline tokenizer counts. The 100 output tokens are synthetic fixture usage, not generated model output. Arm time totals 1,233.09 seconds, including repeated asset hashing; maximum arm time is 74.53 seconds, with at most six requests and 33,510 input tokens per arm. This is not a 9B performance comparison.

Twenty-four scorer fixtures, ten budget probes and three existing simulated gateway paths remain separate evidence. They do not establish automatic Effect integration or the thirty-six-probe R1 gate. The old exact-send smoke retained an active arm and supports binding only; this completed batch supplies lifecycle evidence.

### Scope, authority and next step

The [standalone package](../data/bounded-pilot/r0-development-20260917/README.md) binds sources, eight positive/four boundary tasks, synthetic facts and two AI annotation roles with adjudication. It is **AI-assisted developer evaluation, not independent human Gold or unseen confirmation data**. Complete source and public contracts reach the agent; evaluator references and mechanical probe plans do not. Scripted responses are not autonomous tool selection or whole-Skill translation, and semantic scores remain unknown.

Treatment compiles at most two read prefixes within actual DSH turns and retains native fallback under the same arm budget. No downloaded Skill scripts, external business writes or real LLM are executed. Partial strict predicates do not prove complete YAML/KQL, causality or semantic redaction. Extraction errors remain in future fixed denominators.

Execution dependencies are declared/content-bound, with an explicit OS/kernel trust boundary, not full-machine attestation. Synchronous I/O and Python callbacks are not forcibly hard-real-time cancellable. The original two-working-day window is not restarted; exact historical labor compliance cannot be established without a complete time ledger. This explicit exception does not expand R1/R2 budgets.

The immutable batch report intentionally retains `r0Complete=false`: a mechanical wrapper cannot approve the entire stage. This separate checklist and completion-review summary provide that engineering closure. The active registry label on the child is not an active arm or permission to rerun; the fixed one-use claim is consumed. Preserve both attempts.

Verify the material using the command above. The actual acceptance used isolated no-write bytecode, the pinned assets/parser, and the explicit amendment flag; do not replay it. Only the requested local commit follows, with no push or automatic R1 start. See [third-party attribution and tokenizer pins](R0-THIRD-PARTY-NOTICES.md) for redistribution and host-asset boundaries.

R1 retains the original three takeover points, thirty-six probes and bounded real DSH/9B screening, at most two candidates. Live generation/measurement parity, automatic Effect integration and blinded outcome-review projection still need acceptance; the current optional reviewer sees arm/route data and is not an attested blind-review interface. R2 is one disjoint confirmation and R3 a bounded decision. Formal generalization gates and production authority remain unchanged.
