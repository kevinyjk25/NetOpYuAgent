# 宿主终态交付：机制收敛报告 / Host-terminal delivery: convergence report

## 中文

历史快照说明：本报告记录终态修复包结束时的源码核验和机制结果。后续[任务直达交付](TASK-FIRST-DELIVERY.md)扩展了交付模式；本报告制品和原始评分不改，“当前源码”指本包当时的固定版本。

本包关闭一个确定性缺陷：**宿主已经冻结并给出交付结果后，DSH 不应再让模型重抄一次 deliver 或自行改写完成状态。** 使用已安装 DSH 的公开生命周期接口，不修改其依赖源码，不增加模型自审器，不改变原业务判据。

**机制修复已通过，语义阶段未完成。** 最新真实 9B 证据仍是[单选协议冻结验收](DELIVERY-SINGLE-CHOICE.md)的结构 2/2、业务判据 4/6、完整任务 0/2。本包的 2/2 是脚本化模型替身下的终态接线检查，不是两个真实智能体任务成功，不与前者混算。本包没有调用 9B、27B 或 GPT，没有性能 A/B。

### 已收敛与仍开放的问题

| 问题 | 当前状态 | 可复查边界 |
|---|---|---|
| 交付后继续选工具、重抄结果 | 已修复，可选启用 | 成功的宿主终态回执调用 DSH `concludeTurn()`；同一轮后续六工具操作在进入 bridge 前阻止，仅允许 inspect |
| headless 停了但没有输出 | 已修复，可选前端 | 禁用原 headless runner 并插入宿主回执前端；stdout 与最后有效宿主结果逐字一致 |
| Agent 最终文字覆盖宿主状态 | 已隔离 | UI 工具卡及 headless 展示宿主候选/拒绝，明确 `taskSuccess=null`、`semanticApproval=false` |
| 所选职责填满被误认为任务覆盖 | 标记语义边界，非算法修复 | `semanticCoverage.status=not_assessed`，原任务仍具权威；模型所选 kind 只作呈现提案 |
| 来源类型误配、结论或下一证据遗漏 | **未解决** | CAPA、Mesh 原失败和未重测的 IRQL 保留；不得用结构通过、更名职责或候选展示改分 |

后续只应围绕最后一项做有界方案修复：区分原任务的交付义务与模型所选呈现结构，检查遗漏发生在职责提案、证据使用还是最终生成。先解释通用机制为何能改变已定位的因果环节，再冻结验证；不堆叠同例提示词、自审器或测试数量。没有证据承诺“再几轮必过”。Gate 1、小批新来源、跨 cohort 泛化和同 DSH A/B 的出口均未改变。

### 实现及权威边界

```text
DSH 六工具 → 原 Python 宿主 / 只读 Runtime 或 native fallback
           → 摘要绑定的终态 report + sessionId
           → 工具体校验 → concludeTurn + 本轮工具关闭
           → DSH 完成该轮 → 宿主工具卡 / headless 原样显示
```

- [终态适配器](../dsh-plugin-netopyu/src/hybrid-terminal.js)只识别 draft、deliver、inspect 返回的当前会话宿主终态。要求证据冻结、不可重试、无写权限、摘要关联，以及拒绝／待修订／候选未验证状态。不是模型可传入的 stop 或 success 字段。
- [六工具定义](../dsh-plugin-netopyu/src/index.js)在工具体调用公开 `concludeTurn()`。DSH 在成功工具结果完成后应用终态；异常、取消和结果策略拒绝不会被伪装成成功。缺少该 API 或作用域时在调用宿主前失败，不能静默降级。
- DSH 仍会排空同一步已提交的工具批次，因此增加**按 Agent、turn/start 序号**的守卫：终态后拒绝新的 prepare/submit/read/draft/deliver，不触达 bridge；inspect 只读保留。新用户轮次、其他 Agent 不被关闭。守卫不是进程持久状态；跨重启防重放仍由原宿主 journal/冻结状态负责。
- 新的 steering 不被丢弃，也不能继承前一步旧终态输出；DSH 可以继续处理其上下文。同轮新的混合操作仍受上述关闭规则限制，需要新的用户轮次。没有声称终态会清空所有待处理输入。
- UI 使用 DSH `presentResult` 工具卡。headless 的[可选前端](../dsh-plugin-netopyu/src/hybrid-headless.js)读取成功终轮的工具回执及持久化 metadata，绑定 call、step、session 和报告摘要；不伪造 assistant 事件。不完整／错误／后续步骤使旧终态无效，保留原生输出规则。
- 原生 DSH headless 默认只打印最终 assistant 文字，不能只调用 concludeTurn 而不改输出前端。进程 exit 0 只表示轮次正常结束；候选仍未获语义批准，拒绝也不等于业务成功。
- [宿主](../dsh_adapter/hybrid_session.py)在新终态报告中绑定 sessionId。旧封存报告没有该字段则不会被新前端追认。原权限、预算、读取次数、执行重试和旧协议含义不变。
- [交付渲染](../skill_authoring/delivery.py)保留兼容字段 `declaredCoverageComplete`，新增解释：它只表示没有模型声明的未表示项，不证明真实任务完整。`semanticCoverage` 明确为 not_assessed；这不是新增语义评分器。

### 本地接入与复查

默认 UI/启动命令没有被修改或重启。仅限已经配置好[窄六工具入口与宿主配置](GOVERNED-SESSION.md)的只读环境，由操作者显式设置：

```bash
export NETOPYU_HYBRID_TERMINAL_DELIVERY=1
# headless 另外指定已安装 DSH 官方 headless 的 lib/index.js 绝对路径：
export NETOPYU_DSH_HEADLESS_ENTRY='/absolute/installed/node_modules/@deepseek-ai/dsh-headless/lib/index.js'
```

headless overlay 必须**禁用官方 runner 后插入新前端**，不能在已有 id 下写 name 期待替换。以下片段合并到既有只读隔离 overlay；它本身不负责关闭 shell、文件系统、写工具或 A2A，不能单独当安全配置使用：

```yaml
- id: headless-runner
  disabled: true
- insert:
    - id: netopyu-hybrid-headless
      name: !!js process.env.NETOPYU_ROOT + '/dsh-plugin-netopyu/src/hybrid-headless.js'
      inject:
        - headlessStartup
      config:
        task: !!js ctx.headlessStartup.task
```

保留已安装的 `netopyu-hybrid-local` 六工具插件。运行前用 `dsh --profile headless --patch <overlay> --dump-config` 检查官方 runner 已禁用、新前端已启用，以及原隔离工具均禁用。未设置终态开关的旧路径保持原行为。这里只验证了本机 DSH 0.1.1-rc.2；升级需重做接口检查。

可独立运行无真实模型的完整接线复查，必须使用尚不存在的新输出目录：

```bash
.venv/bin/python -m evaluation.hybrid_terminal_probe /tmp/netopyu-terminal-probe-new
```

[探针源码](../evaluation/hybrid_terminal_probe.py)建立隔离 DSH home、loopback 脚本化模型服务和实际 Python 宿主。固定六次模型协议响应：prepare、两次不支持计划的拒绝、绑定 native fallback、精确 ACL read、deliver。没有手写可执行 L0 计作转译成功，也不执行来源 Skill 的脚本。第七次请求会失败。正常候选与坏 JSON 拒绝分别检查；这是 transport/lifecycle 测试，不是能力基准。

### 证据与结果

| 保留的尝试 | 模型替身协议请求 | 实际只读调用 | 终态后新步骤 | stdout 逐字一致 | 机制通过 |
|---|---:|---:|---:|---:|---:|
| 首次接线，两条路径 | 6＋6 | 1＋1 | 0 | 0/2 | 0/2 |
| 修正 overlay，两条路径 | 6＋6 | 1＋1 | 0 | 2/2 | 2/2 |

首轮已经结束循环，但 overlay 没有替换官方 headless，stdout 只有换行；该失败不删除、不改分。修正为 disable＋insert 后，正常候选和拒绝均原样输出，未出现第七次请求。本包共四条脚本化运行、24 次协议请求、4 次实际只读调用、**0 次真实模型调用**。未测量真实模型 token、p50/p95 或 UI 视觉效果；Runtime draft 的模型生成终态由单元测试覆盖，本轮实际 DSH 接线走的是 native fallback。

- [可携带摘要](benchmarks/host-terminal-delivery-summary.json)
- [保留的首次失败](../artifacts/governed-session-20260915-host-terminal-probe/summary/report.json)
- [修正后摘要](../artifacts/governed-session-20260915-host-terminal-probe-wired/summary/report.json)／[冻结清单](../artifacts/governed-session-20260915-host-terminal-probe-wired/freeze.json)
- [候选终端原文](../artifacts/governed-session-20260915-host-terminal-probe-wired/candidate/dsh-stdout.txt)／[拒绝终端原文](../artifacts/governed-session-20260915-host-terminal-probe-wired/rejected/dsh-stdout.txt)

两版各 66 份制品摘要、76 份归档源码均匹配；后版 76 份也与当前实现一致。前版当前源码仅探针 overlay 接线不同，旧归档未改。额外离线核对 stdout 与宿主正文逐字相同、最终 tool/result 后没有新 step/start。`artifacts/` 被 Git 忽略，不会随代码自动备份。

回归：24 项 Node 生命周期／展示测试、178 项定向 Python 测试、**3,229 项全量＋81 子测试（231.28秒）**、111 个变更/新增项目 Python 文件 Ruff 通过。工程回归不计入 Skill 或语义成功分母；不声称全库历史文件 lint-clean。未提交、推送、重启 UI、连接真实设备或追加真实模型批次。

## English

Historical snapshot: this report's source comparisons refer to the terminal package's fixed version. The later [task-first mode](TASK-FIRST-DELIVERY.md) extends delivery; these original artifacts and scores remain unchanged.

This bounded package closes a deterministic defect: once the host has frozen and delivered its result, the harness should not ask the model to duplicate deliver or reinterpret completion. It uses installed DSH public APIs without changing DSH dependencies, adding a self-judge or weakening existing business criteria.

The opt-in adapter validates a session-bound, digest-linked, read-only terminal receipt and calls `concludeTurn()` inside the tool body. Successful finalized tool results may terminate the turn; errors, cancellation and blocked results are not fabricated as success. DSH still drains an already-submitted tool batch, so an agent/turn-scoped gate rejects further hybrid operations before the bridge, except read-only inspect. New user turns and other agents remain independent. Steering is not discarded; later steps invalidate older terminal output, while same-turn hybrid operations remain closed. The in-memory gate does not replace persistent host replay protection.

UI tool cards show the host candidate/status. The optional headless frontend binds the final successful tool receipt to its call, step, session and replayable metadata, printing it unchanged without fabricating assistant events. The official headless frontend prints assistant prose only: explicitly disable `headless-runner` and insert the new frontend in the existing isolated six-tool overlay. Enable `NETOPYU_HYBRID_TERMINAL_DELIVERY=1`; headless additionally requires the operator-supplied absolute `NETOPYU_DSH_HEADLESS_ENTRY` for the installed official package. Do not treat the configuration fragment as tool isolation. Defaults are unchanged. The actual tested DSH version is 0.1.1-rc.2.

Source membership and filled presentation slots do not prove task coverage. New metadata states `semanticCoverage.status=not_assessed`; the original task remains authoritative. This clarifies the boundary, **not a repair of semantic interpretation**. Candidates keep `semanticApproval=false` and `taskSuccess=null`; process exit 0 is not business success. Legacy frozen reports are not retroactively terminalized.

Two actual-DSH scripted-model probes were preserved: the first pair stopped after six protocol requests and one host read each, but failed because the official frontend still printed only a newline. After correcting disable/insert wiring, the second pair passed: six requests, one read each, no later model step, and byte-exact host stdout for candidate and rejected deliveries. Total: four scripted runs,24 protocol requests,four reads,**zero real model calls**. These are lifecycle checks, not semantic tasks or latency evidence. Runtime draft termination, UI presentation contracts, cancellation, blocked receipts, steering and agent scope have unit coverage; this actual DSH probe uses native fallback. No visual UI verification or real-model rerun occurred.

Latest real 9B results remain **2/2 structural admissions,4/6 criteria,0/2 complete tasks** in the [unchanged frozen assessment](DELIVERY-SINGLE-CHOICE.md). CAPA/Mesh semantic omissions and the untested IRQL issue remain open. Gate1 and all later fresh-source/generalization/A/B requirements remain unchanged. Do not convert lifecycle2/2 into semantic success or promise a fixed number of model-tuning rounds. The remaining work must explain and address the specific task-duty/evidence/generation failure, not stack reviewers or add more tests without a causal fix.

[Portable summary](benchmarks/host-terminal-delivery-summary.json) retains both attempts. Each has66 artifact hashes and76 archived implementation files; all verify. The corrected run also matches current source. Only the probe wiring differs from the first source archive.24 Node tests,178 targeted Python tests,**3,229 full tests plus81 subtests in231.28s**, and Ruff on111 changed/new project Python files pass; this does not claim every historical file is lint-clean. Local artifacts are ignored by Git. No commit, push, UI restart, real device access or new model batch was performed.
