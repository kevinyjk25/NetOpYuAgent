# 冻结快照读取：语义与重复请求 / Immutable snapshot reads

## 中文

2026-09-15，本轮关闭的范围是：**本地冻结快照的补读不能被解释为实时刷新，也不应重复调用 Provider。** 这是对[任务直达模式](TASK-FIRST-DELIVERY.md)的补充，不是任意网络读取的全局缓存，不证明自然语言分析都正确。

### 已定位问题与修复

宿主原来只提供“已读”状态，而实际 resources 是会话绑定的固定本地数据，期间不允许变更。模型不知道同一路径不能取得下一时间窗口；同时 `read` 确实可以再次进入 Provider。这与“实时设备轮询”不同，不能仅用提示词“不要重复读”代替对数据语义的说明。

显式 v4 本地宿主现在通过 prepare、evidenceState 和最终生成输入提供 `observationModel`：

```json
{
  "kind": "immutable_session_snapshot",
  "snapshotId": "sha256:<operator-host-digest>",
  "identity": "tool_and_exact_validated_arguments",
  "liveRefreshSupported": false,
  "sameIdentityRead": "no_new_observation_repeat_not_executed"
}
```

模型不能提供这个声明，source 的“我已经读过”也不能生成已读记录。当前声明来自此宿主确实冻结 resources 的实现，不从 readOnlyHint 推断幂等性或不变性。

- 补读先检查同一宿主/实现、会话锁、未知尝试、证据冻结、次数和参数 Schema，再匹配已完成且摘要绑定的记录。
- 只有同一会话快照、同一工具、精确 JSON 参数相同才命中；对象键顺序不影响匹配，页码、作用域、工具或类型不同不能等同。
- 命中返回 `read_not_reexecuted`、`existingObservation` 和 `newObservation=false`，不调用 Provider，不伪造新 receipt、不刷新时效、不增加 completedReads。
- 重复请求仍占原有两次补读额度；不能用缓存路径绕过次数、未知结果或终态限制。不同资源仍经原权限/参数/资源网关读取。
- 冻结交付保留重复尝试的审计摘要，但不把它追加成新的事实。命中依据包括原严格前段及成功补读；本包**没有改写严格图内部的调度或给所有 Provider 加去重**。
- 旧 v1–v3 路径和真实实时数据源没有自动启用本规则。将来实时刷新需要独立声明的采集能力、时间/版本身份和权限，不得冒用此快照声明。

实现集中在[原会话模块](../dsh_adapter/hybrid_session.py)；没有新增 reviewer、LLM 自审或第二个读取执行器。原可执行 L0、写权限、审批和 Effect 门槛不变。

### 两个新的实际 DSH＋9B 开发任务

一次性构造 **1 个 synthetic Skill／2 个任务**，沿现有驱动执行 prepare→模型提出 read_prefix→原 Runtime 读取→受限补读→一次 9B 生成→宿主终态。输入与判据在运行前冻结。不是两个公开 Skill，不是独立 holdout，也不是原生/Runtime 配对 A/B；没有再运行 Mesh、CAPA 或 IRQL。

| 场景 | 实际过程 | 可观察结果 |
|---|---|---|
| 电路恢复判断 | 1个模型提出的严格前段、1次读取，45.53秒 | 识别 BR-7、Alder→Iris、10:00–10:05；明确重读冻结记录不能证明恢复，需要新采集的后续证据 |
| 库存差异 | 1个前段读取索引＋2次不同资源补读，65.17秒 | 正常读取全部3个资源；报告 connector-K/W4，08:00的18件→12:00的13件，减少5件，并指出缺少流水证据 |

两例均正常结束、只有一个会话，没有重试/重放/写入或虚构已执行操作。库存任务满足全部冻结判据。真实运行均没有尝试重复读，因此**重复拦截分支是单元测试证据，不能说这两例实际触发了拦截**；不同资源未被误拦截则有真实轨迹证据。

### 必须披露的评测设计问题

本轮由开发者 AI 预设的电路 c1 还要求报告“50%丢包”，但可见任务仅明确要求电路、时间窗口、重读能否证明恢复及所需证据，没有明确要求丢包量化。模型没给该比例。

保留原判据、不重跑、不改分：原严格计数是 **5/6 判据、1/2 fixture taskPassed**。但这一额外要求存在 Oracle/任务不一致，因此**1/2 不能解释为校准后的用户任务成功率或模型退步**。恢复任务的核心快照理解正确，不据此补记整例通过。完整原文和逐项解释均保留。

后续用例冻结前，critical 判据应能追溯到可见任务或适用 Skill 的明确要求；额外期望需单列，不能悄悄加入任务成功条件。生成器已修正为fixtureVersion=2，未来新任务明确要求量化丢包，并增加回归；**没有生成或运行新版模型批次**。本轮已冻结的v1输入、原文和分数不改，不追溯修改既有出口和数据。

### 成本、证据与使用

真实模型共 **10次：8次 DSH＋2次 Runtime**，52,874 输入／1,068 输出 token，无未知用量；4次实际授权读取。端到端 p50/p95 为55.35/64.19秒（n=2），仅描述这两个小任务，不能对比历史复杂任务宣称性能提升。模型结束后才启动全量 pytest。

未来新验证可使用已有驱动和新的空输出目录；下列生成器产生修正后的v2任务，不是本轮v1输入的精确重放。本轮原始输入以冻结文件为准：

```bash
.venv/bin/python -m evaluation.snapshot_session_cases /tmp/new-snapshot-inputs
.venv/bin/python -m evaluation.hybrid_session_acceptance \
  /tmp/new-snapshot-inputs /tmp/new-snapshot-run \
  --cases circuit-recovery stock-difference --task-delivery --run
```

第二条会调用本地9B。现有[宿主终态前端](HOST-TERMINAL-DELIVERY.md)用于输出；这些命令不改变默认 UI 配置。本文记录一次冻结运行，不授权自动反复生成挑选通过结果。

- [可携带摘要与判据问题](benchmarks/snapshot-read-summary.json)
- [冻结任务、判据及源码清单](../artifacts/governed-session-20260915-snapshot-9b/freeze.json)
- [逐项审阅](../artifacts/governed-session-20260915-snapshot-review/judgments.json)／[摘要绑定评估](../artifacts/governed-session-20260915-snapshot-assessment/report.json)
- [恢复判断原文](../artifacts/governed-session-20260915-snapshot-9b/circuit-recovery/dsh-stdout.txt)／[库存比较原文](../artifacts/governed-session-20260915-snapshot-9b/stock-difference/dsh-stdout.txt)

75份运行制品、78份归档/当前源码核对一致。原始制品被Git忽略，不随代码自动备份。7项新快照边界回归及2项fixture隔离检查通过；全量结果见机器摘要。没有提交、推送、切换或重启当前UI。

**收敛结论：**快照语义和补读重复调用缺口已在限定范围内修复；任务直达的实际 DSH 链路可以处理这些小任务。仍不能证明大范围转译、完整业务语义或生产可靠率。旧 Mesh 的重复建议、CAPA 结论和 IRQL 工件问题没有重测，旧分保留。

## English

This package fixes a scoped defect: the local v4 host serves immutable session snapshots, but previously advertised only completed reads and allowed duplicate follow-up Provider calls. It now supplies host-owned observationModel metadata to prepare, read state and generation. This is a property of its frozen inert resources, not an inference from read-only tools or a global cache for live networks.

After existing host/version, lock, unknown-result, freeze, attempt and schema checks, an exact same-tool/JSON-arguments match against a completed bound record returns read_not_reexecuted. It references the original observation, invokes no Provider and creates no fresh receipt, timestamp or evidence. The attempt still consumes budget. Different pages/scopes/tools remain distinct and use the original access/resource gate. Internal strict-graph scheduling, legacy profiles and live providers are unchanged. No self-judge or alternate executor was added.

One frozen run used actual DSH and qwen3.5:9b on **one developer-authored synthetic Skill and two tasks**. Circuit recovery used one read in45.53s and correctly explained why rereading cannot establish recovery; later acquired evidence is required. Inventory used an index plus two distinct reads in65.17s, correctly reporting connector-K/W4,18→13 units and missing transaction evidence. Both had one session and no duplicate attempts, replay, effects or fabricated execution. Duplicate interception itself has unit coverage; different-resource admission has real-model trace evidence.

Raw frozen scoring is5/6 criteria and1/2 fixture tasks. A disclosed evaluator defect prevents interpreting1/2 as calibrated user-task accuracy: circuit c1 additionally demanded50% loss although the visible task did not explicitly request loss quantification. The answer omitted it. Preserve the unmet original criterion and annotate the mismatch; do not regrade, rerun, silently exclude or count the case as a complete success. The generator now emits fixtureVersion=2 with an explicit loss-quantification request and a regression check. No v2 model batch was generated or run; commands above generate future v2 inputs, not the historical v1 input. Future critical criteria must be traceable to visible requirements before freezing.

Ten real calls (eight DSH,two Runtime),52,874 input/1,068 output tokens,four actual reads,zero unknown usage. p50/p95 are55.35/64.19s over two small tasks, not a comparable performance gain. No model/full-pytest overlap. [Portable evidence](benchmarks/snapshot-read-summary.json), [bound assessment](../artifacts/governed-session-20260915-snapshot-assessment/report.json) and both stdout artifacts preserve successes and evaluation flaws.75 artifact hashes and78 archived/current sources verify. Nine new tests cover snapshot/fixture boundaries; final QA is in the summary.

The scoped snapshot mechanism is closed; broad semantic/generalization/production gates are not. Mesh,CAPA and IRQL were not rerun or regraded. No default UI change, restart, commit or push. Local artifacts are Git-ignored and require separate backup.
