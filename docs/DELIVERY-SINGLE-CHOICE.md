# 单一交付状态：有限修复与停止条件 / Single-choice delivery: bounded repair

## 中文

历史快照说明：本文保留单选协议的原始9B结果和当时停止点。后续[宿主终态机制](HOST-TERMINAL-DELIVERY.md)已修复并完成无真实模型接线检查；不改动本文任务评分，也不代表语义缺陷已解决。

本包不是继续增加语义自审器。目标只有两个：消除内容与状态在不同字段重复表达造成的矛盾；把宿主的交付回执和 Agent 的最终文字分开显示。

显式宿主 `netopyu.io/local-hybrid-host/v3` 使用交付协议 `source-anchored-delivery/v4`。每个固定 ID 只能选择：类型化内容对象，或表示无法提供的原因字符串。没有独立的 `unresolved` 映射；字符串 `None` 也只能是未解决，不能当成成功哨兵清除。旧 v2/v3 合同、旧失败按原规则保留，不自动迁移。

```json
{"delivery":{"d0":{"text":"实际分析内容"},"d1":"缺少必要观察，不能给出该项结论"},"uncertainties":[]}
```

这不是对业务内容的判断规则。若 `d1` 未解决，结构虽可解析，整体 `shapeComplete` 仍为 false。宿主单独记录 `hostResult`：`rejected`、`needs_revision` 或 `candidate_unverified`；没有“语义已批准”状态，`taskSuccess` 始终未知。最终自然语言不具有改变宿主回执的权限。独立只读视图并排展示两者，并可展开原任务和被选职责，便于人工定位错误类型或遗漏；不冒充自动语义审阅器。

### 预先确定的本轮验收边界

1. 单元/链路检查：四种交付形状、互斥状态、原生与 Runtime 两条路径、旧协议不自动修复、授权/回执/不重放边界。
2. **只进行一个冻结的真实 DSH＋qwen3.5:9b 批次**：CAPA、Mesh 原任务、原事实和原判据完全不变，每例上限 420 秒。此前已知样例，不算泛化数据。
3. 格式准入、实际读取、最终业务内容独立记录；对保留但被拒的候选不算最终回答，不以过程完成冒充业务成功。不把两个样本的 p50/p95 当性能证明。
4. 无论结果如何，本轮不再调参重跑同两例。若关键业务判据仍失败，则结束该轮优化，记录明确机制缺口；不宣布 Gate 1 通过，不转大规模 A/B。若通过，也不能据此声明新来源泛化，且历史 IRQL 缺口仍需单独交代。

### 业务遗漏的已知诊断

上一批 CAPA 把“先读导出”映射为 artifact，而不是把原任务要求的状态/下一证据映射为适当交付职责；Mesh 把输出格式限制选作 next_steps 的锚点。引用成员检查只能证明这段文本来自输入，不能证明“这是一项交付要求”或“这些选择覆盖任务”。原任务确实保留到推理输入，并非源文被确定性丢弃；语义选择和最终内容仍会遗漏。因此不得将四种形状及 `declaredCoverageComplete` 当作任务覆盖证明。本包不添加案例答案或词面特判去掩盖这一缺口。

### 唯一冻结验收结果：修复包结束，Gate 1 仍未通过

| 维度 | 上一批紧凑协议 | 本批单选协议 |
|---|---:|---:|
| 结构交付准入 | 1/2 | 2/2 |
| 原业务判据 | 3/6 | 4/6 |
| 完整任务 | 0/2 | 0/2 |
| 最终文字错误声称完成 | 1例 | 本批未发现 |
| 端到端 p50 / p95 | 166.26 / 250.65秒 | 241.80 / 366.68秒 |

这些是两个已知样例的不同冻结实现，不是因果A/B或统计泛化。**结构改善，整体时延观察值反而更差，不能宣传整体性能提升。** 本批14次完整模型回复（原生DSH 13＋Runtime 1），176,566输入／3,183输出token；13个宿主步骤均有完整回复，没有超时未知用量。唯一Runtime调用为56.72秒、7,250输入／408输出。1个自动读取图、3次授权读取，没有写入、来源脚本或操作重放。

- **CAPA：103.04秒，2/3判据。** 读了实际数据；保留责任人、时间、25/25与5/30差异和未批准状态，补出下一证据。仍没有明确“记录保持开放”的结论。按历史保守口径保留失败：这是结论显式性遗漏，不是声称已批准。另有四个实质小节违反最多三节的原约束，单列披露而不追加评分项。整Skill超预算仍走任务限定原生fallback，不算整Skill转译成功。
- **Mesh：380.56秒，2/3判据。** 补读窗口并正确报告南北指标，保留相关性而非根因/未确认恢复的说明；但下一步没有请求规定的trace/log/saturation证据，只在不确定性中说它们缺失，还重复建议已读取的数据、把索引说成变更日志。上一批通过的c3本批退步，c1则恢复，不能只报恢复项。单选候选在Runtime内准入后，DSH仍额外调用deliver重抄内容；终态保护阻止重放，但没有阻止模型浪费生成。最后正常结束，没有放宽420秒上限。

业务要求的来源误配也未消失：CAPA的next_steps仍引用“先读导出”，Mesh仍引用“三节以内”。本批不再补提示词或重跑，也不将4/6改称阶段完成。历史IRQL未完成查询问题本批没有重测，仍开放。

### 具体停止点与后续建议（未执行）

本包的**单一状态协议、宿主独立结果、诊断视图已经实现**；“所有语义关键问题修复”未实现。不能承诺再两轮一定通过。下一步不应继续增加同例测试数量或语义自审层，而应把两个问题分开决策：

1. **宿主交付生命周期**：候选终态后应由真实Harness生命周期直接展示宿主制品、结束可调用工具阶段，避免模型再生成一个deliver调用或重新解释完成状态。当前只有禁止重放的宿主保护和提示，没有接入DSH终态调度控制；本批没有实现该能力，也没有重启现有UI。
2. **输出职责的权威边界**：原始用户任务和Skill仍是语义参照；模型所选kind/source_ref只能是提案/呈现提示，不能成为“业务要求已经完整形式化”的证明。自由分析仍属于L1概率推理；真正可执行产物才需经独立可判定的合同。保持现有任务评分，不能通过更名L1或降低判据把旧失败变成功。若下一设计改变研究出口，须先明确决策。

这一分离符合“Runtime管允许发生什么，不证明LLM所有想法正确”的原型原则。当前资料不足以断定剩余错误只由9B导致，也没有GPT同条件对照。本轮未换模型、未扩大样本或A/B。

### 查看与复查

- [可携带机器摘要](benchmarks/delivery-single-choice-summary.json)
- [冻结运行](../artifacts/governed-session-20260915-single-choice/freeze.json) / [原判据逐项审阅](../artifacts/governed-session-20260915-single-choice-review/judgments.json) / [摘要绑定的完整报告](../artifacts/governed-session-20260915-single-choice-assessment/report.json)
- [原任务、所选职责、宿主结果与Agent文字并列视图](../artifacts/governed-session-20260915-single-choice-view/host-result.html)
- [CAPA最终原文](../artifacts/governed-session-20260915-single-choice/capa-status/dsh-stdout.txt) / [Mesh最终原文](../artifacts/governed-session-20260915-single-choice/mesh-incident/dsh-stdout.txt)

`artifacts/`不随Git发布，机器摘要和本文可携带，完整本地证据需要另行保留。视图不加载脚本或远端资源，所有文本转义；3项视图回归通过。浏览器策略阻止file URL，未绕过，因此**未完成实际浏览器视觉核验**。

176项定向检查、**3,227项全量＋81项子测试（242.34秒）**、109个变更/新增Python文件Ruff、3项文档检查、六工具声明和git diff检查通过。67份运行制品、76份归档及当前源文件、11份视图来源摘要一致。没有模型运行与全量pytest并发，不宣称仓库全部历史代码lint-clean。本轮未提交/推送。

## English

Historical snapshot: the later [host-terminal mechanism repair](HOST-TERMINAL-DELIVERY.md) has scripted-model integration evidence. It does not change this package's original real9B scores or close its semantic gaps.

The explicit v3 host uses delivery protocol v4. Each item is either its typed content object or a string explaining inability. There is no second unresolved map. Any string, including `None`, remains unresolved; it is never repaired into success. Legacy protocols and failures retain their original meaning.

The host records rejected / needs_revision / candidate_unverified separately from native Agent prose. Neither source membership nor schema acceptance proves semantic coverage. A read-only view places the host receipt beside native final prose and exposes the original task and selected requirements for inspection; it is not an automated semantic judge.

This package permits exactly one frozen DSH + qwen3.5:9b run of the unchanged known CAPA and Mesh cases, capped at 420 seconds each. No same-case prompt-tuning rerun follows. Critical failures leave Gate 1 open; even success would not prove unseen-source generalization or close the separately outstanding IRQL gap. Previous source-kind mistakes demonstrate that exact references can still encode the wrong interpretation. No expected answers or case-specific rules are added.

The single frozen run is finished: **2/2 structurally admitted deliveries, 4/6 original criteria, 0/2 complete tasks**. CAPA preserves facts and adds next evidence but omits the explicit keep-open conclusion under the unchanged conservative interpretation; four substantive subsections also exceed the original format limit. Mesh restores north/south metrics and causal uncertainty but fails the actionable trace/log/saturation criterion, recommends already-read data and mislabels the index. The native harness redundantly calls deliver after the Runtime candidate; no operation is replayed. Final prose contains no false completion assertion in this run.

Fourteen complete replies (13 native, one Runtime) record 176,566 input / 3,183 output tokens. All started harness steps have replies; neither case times out. CAPA takes103.04s and Mesh380.56s; end-to-end p50/p95 are241.80/366.68s, **slower than the prior known-case batch**, not causal performance evidence. The one Runtime generation takes56.72s with7,250/408 tokens. One automatic graph and three authorized reads occur; no effects, commits or source scripts.

The implementation package is closed; same-case model reruns stop here. Gate1 and the separately outstanding IRQL gap remain open. Proposed next work, not implemented: host-controlled terminal delivery that removes redundant model orchestration, and a clear distinction between model-selected presentation kinds and genuinely qualified executable output contracts. Do not claim a model-selected schema proves complete task interpretation, add a universal self-judge, or lower existing scores. Changes to research exit criteria require an explicit decision. No evidence establishes 9B as the sole cause.

[Portable metrics](benchmarks/delivery-single-choice-summary.json), [bound assessment](../artifacts/governed-session-20260915-single-choice-assessment/report.json) and [side-by-side view](../artifacts/governed-session-20260915-single-choice-view/host-result.html) preserve host outcomes separately from native prose. Local artifacts are not Git-published. Browser file-URL policy prevented visual verification; no workaround was attempted.176 targeted checks,3,227 full tests plus81 subtests in242.34s,109 changed/new Python Ruff files,three documentation checks, six-tool declarations,git diff,67 artifact digests,76 archived/current source digests and11 view-source digests pass. No model/full-suite overlap, commit, push or UI restart occurred.
