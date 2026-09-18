# R0 计量生命周期修复 / Measurement Lifecycle Repair

## 中文

2026-09-16，用户确认清理提交已推送后继续。**本包只修复可复现的计量／交付边界，不继续旧语义调参，也不启动 R1。** 当前仍为 `r0_measurement_partial`，旧 `paused_unmet`、原始失败、Oracle、预算和正式准入条件不变。验证数据见[机器摘要](benchmarks/bounded-lifecycle-summary.json)。

本页保留为生命周期修复的历史报告；后续已获授权的离线预备实现及已完成的有限请求形状联检见[Token 预检现况](R0-TOKEN-PREFLIGHT.md)与[本轮摘要](benchmarks/token-preflight-20260917-summary.json)，不改变本页历史数据。

### 修复与验证对象

| 缺口 | 实现及验收方式 |
|---|---|
| 等待 SQLite 写锁后仍使用锁前时间，可能让过期请求获准 | 在取得事务锁后取时；真实锁等待跨过 deadline 的正反例。边界统一为 `now >= deadline` 拒绝 |
| 预留到真正调用之间可能暂停／关闭 | 精确 `arm_id + request_id` 的 `guard_reserved_call` 在调用点再检查；仅允许自己的 pending reservation，不授新能力、不重置预算 |
| 关闭／暂停后的回调可能仍被当成功；普通异常误当确定拒绝 | 结算与交付分离；交付前再检查；未知异常暂停。只有 qualification 的显式 `QualificationRejected` 可表示已知拒绝；异步返回不能逃出同步计量边界 |
| 阻塞脚本回调及排队请求无限等待 | 回环 broker 按绝对截止时间停止等待，保守结算未知成本；迟到 worker 只能保存实际返回证据，不能再次结算或交付 |
| 慢速 HTTP 分段传输突破 idle timeout | 请求行／headers／body 和响应编码／headers／body／flush 分别有绝对时间检查；真实 socket 慢速输入与背压测试，不只 mock timeout |
| close 遇到锁繁忙漏掉清理，或未排空也算通过 | 先撤销能力，再独立调度清理；调用者可有界返回 `drained=false`，无需第二次 close 才开始清理。DSH 探针显式要求 `brokerClose.drained is True` |

源代码： [账本](../evaluation/bounded_budget.py)、[同步计量](../evaluation/bounded_execution.py)、[脚本模型传输](../evaluation/bounded_transport.py)、[DSH 接线探针](../evaluation/bounded_dsh_probe.py)。新增生命周期回归为 [同步回调测试](../tests/test_bounded_execution_lifecycle.py) 与 [传输测试](../tests/test_bounded_transport_lifecycle.py)。

### 已有证据与解释边界

最终定向检查 **213 项通过**（含 73 项传输检查）。安装版 DSH `0.1.1-rc.2` 在新目录重验三条路径全部通过：原生读取、在线脚本编译后既有 Runtime 只读执行、确定编译拒绝后原生 fallback；共 **8 次脚本协议请求、3 次真实 SQLite 读取、0 次真实模型调用**。各路径都检查原样工具结果递送、相同初态、隔离数据库、宿主与 broker 排空，以及执行源码不漂移。265 份执行源码指纹与最终探针一致，报告摘要核验通过，但不是完整依赖快照。模型输出和 token 成本仍是 fixture，不是 9B 性能或语义成绩；自动 Effect 桥接仍为 `not_tested`，本探针不调用 Runtime LLM 节点。

首轮传输检查的 3 项失败保留：旧断言期望过期返回 BudgetError、关闭时继续 pending 并采纳迟到成本；修复后改为明确 timeout、未知结算不重写，并加强断言。首次全量另有 2 项正常关闭时序失败：客户端读完 body 不等于服务端已完成最终交付检查，立即 close 会撤销在途交付。仅让正路径测试先有界等待服务端排空，仍检查无错误、账本 active/completed；关闭中途的拒绝规则和负测试未变。没有删除测试、改正式评分或覆盖旧报告。完整回归和 lint 结果以机器摘要为准。

**不声称强制取消线程或远端模型。** Python 回调、编码和本地数据库／文件 I/O 属于协作式宿主边界；操作系统调度、锁等待或 I/O 可能延迟清理。socket watchdog 能中断本地传输，不能撤回已发送字节，部分交付必须停止后续调用。`drained=false` 是尚未退出的明确状态，不是成功。真实上游仍未实现，不能把本包当作真实推理的硬取消证明。

最终完整 `all` 回归 **3,772 项＋81 子测试通过**（260.57 秒）；文档／权限边界 51 项通过，10 个变更 Python 文件 Ruff 及 diff 检查通过。旧文件的 31 条既有 lint 问题未在本包处理，不宣称全库零 lint。本包修复验收完成，R0 总体仍部分完成。

### 明确下一步，不新增无限修复轮

报告后的进展是：用户已授权隔离方案，离线共享 renderer、vocab-only tokenizer 与不可变 prepared request 已实现；六种请求形状各两次的离线联检全部通过，重复计数和制品绑定一致。现用 Ollama 的 debug-render 可能先预热，仍不能冒充零推理计数。live 计量／dispatch 绑定未完成，generation 禁止，本轮 0 次真实模型调用，不替换现用服务；这不是六份 Skill 的语义评测。

R0 还缺完整控制器／执行依赖冻结、6 Skill／12 Task 和独立职责标签。仍受[原协议](EVALUATION-RESET-20260916.md)两个工作日与零模型上限约束；计量方案不成立就报告阻塞，不放宽预算、不改模型、不追加同例 9B 试跑。本包不提交／推送 Git，不重启 UI，不动无关用户文件。局部工程修复完成不等于 R0 或研究原型完成。

## English

After the user's push confirmation on September 16, this package fixes reproducible metering/delivery lifecycle defects only. It does not resume semantic prompt tuning or start R1. Status remains `r0_measurement_partial`; historical `paused_unmet`, failures, labels, budgets and admission gates remain intact. See the [portable evidence summary](benchmarks/bounded-lifecycle-summary.json).

This page remains the historical lifecycle-repair report. See [token preflight status](R0-TOKEN-PREFLIGHT.md) and the [current summary](benchmarks/token-preflight-20260917-summary.json) for the subsequently authorized offline implementation and completed finite request-shape checks; the historical data below is unchanged.

Time is sampled after acquiring the SQLite transaction lock, with half-open deadlines. An exact reserved-request guard rechecks dispatch without renewing time or granting authority. Usage settlement does not authorize delivery after close/pause. Unknown local failures halt; only explicit qualification rejection remains deterministic. Deferred asynchronous results cannot escape the synchronous metering API.

The scripted broker bounds callback/queue waiting and conservatively settles unknown outcomes. Late workers retain evidence but cannot resettle or deliver. Absolute ingress/egress watchdogs cover slow HTTP input and socket backpressure. Close schedules eventual cleanup even when its caller times out on a busy lifecycle lock; undrained or failed cleanup cannot pass the DSH probe.

Final targeted regression passes 213 tests, including 73 transport checks. Three fresh installed-DSH paths pass: native read, scripted online compilation plus the existing read Runtime, and native fallback after deterministic rejection. They produce eight scripted requests, three actual SQLite reads and zero real-model calls, checking result delivery, equal initial state, isolation, source stability and drain. All 265 source fingerprints match the final probe and report digests verify; this is not a complete dependency snapshot. Tokens and responses remain synthetic. Automatic Effect bridging is `not_tested`; no Runtime LLM node runs in this probe. The first three assertion failures remain recorded; assertions were strengthened for timeout/unknown semantics. Two first-full-suite failures exposed positive tests closing after client body receipt but before server delivery bookkeeping finished. Those tests now wait boundedly for idle before close, preserving error/ledger assertions and all negative-close guards. Full-suite/lint results are in the summary.

This is **not forced thread or remote-model cancellation**. Python callbacks, encoding, database/file I/O and scheduling remain cooperative. A watchdog cannot retract already transmitted bytes; partial delivery halts subsequent calls. `drained=false` explicitly discloses unresolved work. No live upstream is enabled.

Final full `all` regression passes **3,772 tests plus 81 subtests** in 260.57 seconds. Documentation/authority checks pass 51 tests; Ruff passes on all ten changed Python files and diff whitespace checks pass. The 31 previously documented legacy lint findings are untouched; no repository-wide lint-clean claim is made. This lifecycle package is verified; R0 as a whole remains incomplete.

Subsequently, the user authorized the isolated approach. The offline shared renderer, vocab-only tokenizer and immutable prepared request are implemented; the six-request-shape, two-repeat offline check passes with stable counts and bindings. The existing service's debug-render may still warm up the model and is not a zero-inference counter. Live metering/dispatch binding remains unfinished, generation is disabled, real-model calls remain zero, and the current service is untouched. This is not a six-Skill semantic evaluation. The full controller/dependency freeze and independently labelled six-Skill/twelve-task corpus remain unfinished. The original two-working-day R0 limit and zero-model budget remain binding. This lifecycle package makes no commit/push, UI restart or unrelated file changes, and does not establish semantic improvement or prototype completion.
