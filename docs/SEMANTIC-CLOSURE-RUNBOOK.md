# 语义闭环本地验收 / Local Semantic Closure Runbook

## 中文

2026-09-15：同一 `evaluation.semantic_typed_duty_probe` 新增 `--evidence-first-notes` 模式，只做提取→引用定位→逐项比较→只引用观察的备注，不修改正文。默认仍零调用预检；10 次实测和未解决的语义反例见[报告](SEMANTIC-DUTY-CONTRACT.md)。本轮严格取证通过，无须历史摘要兼容；不要混算成整稿通过或重跑旧失败。

同日后续：新的显式入口是 `evaluation.semantic_typed_duty_probe`，保留宿主任务原文，以两步谓词审阅替代整体备注判定，独立单元失败不重试。18 次真实结果、查询组装失败和检查点摘要冲突均见[同一报告](SEMANTIC-DUTY-CONTRACT.md)。旧运行不能覆盖或改分；取证默认严格，已知摘要键冲突只能经显式、可重现的派生审计披露，不是忽略损坏文件。

2026-09-14 最新可选入口：[职责合同/逐职责核查/本地检查](SEMANTIC-DUTY-CONTRACT.md)及 `evaluation.semantic_duty_resume`。原运行和独立接续的已知两例验证均已结束；未重试失败调用。类型化字面量渲染为单独零调用 API/制品，不计成原模型输出或完整查询执行验证。后文“新批次尚未启动”是上一轮历史状态。

2026-09-14 当前增量：可选 `taskScope` 必须逐字覆盖原任务；审阅 `read_context` 仅由已验证执行/回执生成。notes 负面问题现在可由独立单元处理，与正文共用 8 个单元，超过预算显式延期；原 notes 和宿主职责保留。当前完整稿所有者可增加 1 个标题，旧标题不删改，局部单元不越界。真实四例结果及后置工程修改的区别见[阶段停止报告](SEMANTIC-CONTEXT-REPAIR.md)。不要把新策略对旧提案的零调用应用检查算成新 9B 成功；新的真实批次尚未启动。

2026-09-14：当前审阅接口为 `bounded-draft-review/v9` / `host_keyed_check_cells/v2`，在此前编号分块修复上增加任务交付精确引用。`task_checks.satisfied` 没有真实 `draft_span_id`＋`artifact_quote` 时单项暂缓；引用正确也不是语义通过。`located_negative_findings/v3` 将 notes 问题留在未解决队列，不交给正文所有者。新的双调用来源/任务见证仅作[隔离诊断](SEMANTIC-WITNESS-DIAGNOSTIC.md)，没有增加默认流程模型调用。下面的旧命令/版本属于相应冻结历史；不要用当前源码改写旧回执。

这是可选的本地研究入口，不改变 DSH 默认路由。使用真实 `qwen3.5:9b` 推理，但业务工具是明确公开的内存数据夹具，不访问生产系统，不执行 Skill 自带脚本。完成条件见[阶段出口](SEMANTIC-CLOSURE-EXIT.md)，负结果见[根因记录](SEMANTIC-CLOSURE-DIAGNOSIS.md)。

### 分权处理链

```text
原始 Skill + 实际任务 + 低层工具合同
  → 9B 自动构造只读前段 / 保留原 L1 任务
  → 明确的本地审阅准入（不是模型自动批准）
  → 原 L0 读取 + 有界 9B 草稿
  → 有待办时：有限选读/追问 → 独立资源准入 → 原 L0
      有新增观察：保留待更新候选＋宿主已读索引，再生成修订稿；无新增观察：保留原稿，另存追问
  → 来源审查 → 定位修订计划 → 一次有界编辑 pass（最多 8 单元）→ 终审 → 待开发者/人工判断
```

补读不是宿主写死业务参数：模型从原任务、原 Skill、真实前次工具＋参数＋回执中选择下一步；宿主仅维护可用资源清单与权限。原快照可用于分析，不能更新为当前操作权限或有效期。

来源预算优先保留完整入口，再按目录顺序加入能整体放入的惰性 Markdown 参考文档；入口本身过大时保留连续分页并走显式补页。预算不增加，未供给页仍在目录内，不标为已读。补读写作节点保留 `previousCandidate` 作为待更新稿，而非事实；旧 `previousRemainingActions` 位于参考区。宿主 `readStatusIndex` 逐项指向精确工具、参数与实际结果位置，区分历史已读与本次已读，不重新授予权限/时效，也不虚构 tool message。

### 可复现入口

先运行 `python -m evaluation.semantic_closure_transfer freeze <新批次目录>`。冻结包含执行源码、模型文件摘要和配置；再选新样本，禁止看到该批结果后修改实现再称同批为未见验收。

每个任务提供独立 JSON specification，包含：

- `snapshot`、`candidateId`：已隔离、固定仓库版本的公开 Skill 源材料；不是执行目录。
- `task`、`inputSchema`、`tools`：真实限定请求和低层只读工具合同，不能写预期流程图或参考答案。
- `fixture: {arguments, resources}`：当前用户输入和独立宿主的有限资源/结果清单。数据应披露为合成。
- `domain`、`expectations`：运行前的验收要求，单独保存，**不进入任何模型请求或 Runtime 参数**。

顺序调用：

```bash
python -m evaluation.semantic_closure_transfer prepare <批次> --case <任务ID> --specification <文件>
python -m evaluation.semantic_closure_transfer author <批次> --case <任务ID>
```

检查 `cases/<ID>/inputs/packet.json` 和 `author-summary/compilation.json` 中的原文、供给页、候选与实际图。若可在本地只读执行，显式写入该任务的 `admission.json`：`case`、`compilationDigest`、`decision: admit_local_read_reason_only`、`reviewKind: developer_ai_not_independent_gold` 和真实审阅 `rationale`。这个准入不能由“Schema 通过”或预期答案自动生成；当前开发者 AI 审阅也不是独立 Gold。

```bash
python -m evaluation.semantic_closure_transfer execute <批次> --case <任务ID>
python -m evaluation.semantic_closure_evidence <批次> <新的审计输出目录>
```

工具输出、模型原回复、被拒绝的候选、独立准入、源码包和摘要均保留。已开始的尝试不自动重试；变更冻结代码、模型或预期必须使用新批次。模型调用包含构造、补读选择、草稿、审查和失败，不能只算最终成功一次。

### 怎样读结果

| 查看内容 | 文件/字段 | 能说明什么 |
|---|---|---|
| 自动构造 | `author/call-*/request.json`、`compilation.json` | 从哪些原文形成哪些边界；不是语义批准 |
| 真实工具执行 | `initial/summary/report.json`、`continuation/round-*/summary/report.json` | 精确参数、合同、宿主准入与实际只读回执 |
| 来源审查 | `review-before/model/review-before/review-input.json`、`review-assessment.json` | 原始交付稿、实际观察和双向定位；正面评分不是事实 |
| 原始编辑提案 | `repair/model/e*/candidate.json`、`host-validation/report.json` | 模型是否提出了可实际应用的修订；坏补丁在继续下一单元前拒绝 |
| 实际修订稿 | `repair/materialized/candidate.json`、`application.json` | 宿主实际应用后的完整交付稿及精确变更；不能用模型自述代替 |
| 答复、支持依据与未解决上下文 | `repair/materialized/delivery.json` | 新双通道模式的答复、独立引用、原 notes 出处和宿主职责；引用不代替回答 |
| 来源引用渲染 | `repair/materialized/source-resolution.json` | 观察片段的父来源、精确位置、未修改单元；引用选择仍待语义审阅 |
| 剩余语义问题 | `repair/final/model/review-after/review-assessment.json` | 定位和解释线索，不能用 supported 计数代替准确率 |
| 成本及证据完整性 | 审计报告 `calls`、`archives`、`boundArtifactDigests` | 保留全部成本/失败，检查回执与源包未漂移 |

模型可提出来源选择、解释和候选，不能自我批准。当前 `grounded_patch` 模式将 `keep/write_prose` 答复编辑与 `source_units` 引用分成独立通道；来源支持单列在 `delivery.json`，**不能把分析/追问替换成摘录，也不能靠正确附录抵消错误正文**。原 notes 按原候选出处保留，宿主职责未清除。所有位置、来源、实际修改在宿主校验；来源选择与解释仍需语义审阅，不以安全停止冒充完成。

2026-09-11 的 `artifacts/translator-v2/semantic-closure-transfer-20260911-v1` 使用旧 `source_patch`，首次冻结[结果未通过](SEMANTIC-CLOSURE-TRANSFER-V1.md)，不改分。新机制验证先用 `freeze --known-development` 明确标识已知来源，验收器禁止把它计入未见出口。正式新来源小批仍须先冻结再选择材料。两种模式都保留完整章节切分、标题/范围校验、操作优先的生成顺序及规范化摘要。

当前审查传输使用 `hybrid_review_views` 的可读工作表：完整原任务、源文、观察、草稿及 notes 逐字保留，重复导航文本/偏移仍在原 `review-input.json` 留档，不称整份审计 JSON 可逆编码。编辑器采取下面的局部稿件投影，不能说它看到了完整稿件。旧 `textPool` 仅供历史诊断。无效注释及坏读取后缀会出现在 author 的 `prefix-projection.json`，原 `raw-choice.json` 不覆盖。有效前段候选仍须原有显式准入。负面审阅缺少修改建议时列为 `nonActionableFindings`，不能升级为 supported。

2026-09-14：`bounded-draft-review/v8` 不再把 `1. 正文` 或 `## 3. 标题` 的编号误拆为独立断言；编号与同一行内容合并，原文和精确位置保留，普通数值句末/否定句仍分别检查。只改变导航分块，不删除内容、自动认定标题无事实或放宽引用要求。冻结历史 payload 不重建为新版本。[9B/27B 对照](SEMANTIC-REVIEW-MODEL-CONTRAST.md)已结束，未改变默认 9B。

无补读时 `deliver` 使用 `reason_if` 保留前轮候选；trace 明确 `modelInvoked=false`，不会创建虚构的 `model/deliver` 回执。`hybrid_snapshot_review` 核对条件、原摘要、历史候选及绑定值后才能审阅；另存的 `noReadSelection` 是未经验证的追问/说明，不是新观察、已满足职责或行动授权。实际存在新观察才调用 writer；选读请求依然接受原合同、有限 ACL 和独立 Gate 检查。

当前只把 clarify 分支的原追问传入未解决上下文；answer 分支的自由解释只留在原选择器轨迹和摘要中，不再进入编辑器作为新答复依据。审阅响应分为 `statement_checks`、`coverage_checks`、`task_checks`；原回复保留在 `role-review.json`，统一格式的 `candidate.json` 只是确定性映射。位置可验证，语义仍可能漏判。

编辑器保留完整任务、Skill 与实际观察，但仅看到本单元正文＋其他章节索引；完整父稿在冻结请求与终审中，不在局部编辑 Prompt 中。ATX／Setext／独立非整句粗体标签（含列表包裹）受保护，普通整句加粗不是标题。与实际观察精确匹配的原有 blockquote 在本 pass 内只读；这保障引用字节保留，不证明引用适用性、真实性或权限。历史 `materialized` 稿可用 `hybrid_snapshot_review <历史repair目录> <新目录> --max-model-calls 1` 新审阅：须核对旧冻结/应用/稿件与回执归档，保留 `historicalImportReceipt`；不导入旧模型意见，不计未见验收。

默认 `located_findings` 修订触发器仅对定位到一个宿主章节的负面意见启用编辑；词面差异与正面意见不单独触发重新生成。无法定位的遗漏放在 `repairPlan.unlocatedFindings`，需要后续审阅/定位，不能广播到所有章节。没有定位问题或全部只读的单元由 `reason_if` 原样保留，零编辑调用、零修复记功，**不是已正确**。`--repair-trigger all_cells_diagnostic` 仅供旧机制诊断，不是默认运行。终审仍独立留档，不作为批准。原型暂不能可靠自动定位所有语义遗漏或隐含错误前提。

v2 触发策略增加一个严格条件：**只有一个单元且其范围覆盖整个原稿**时，未定位负面问题可交给该唯一所有者检查；只选中多章节中的一个单元不满足此条件。记录 `locationBasis=unique_whole_draft_owner_not_semantic_location`，不编造 draft ID、语义匹配或批准。审阅 `preserved` 缺少位置时不再丢弃所有其他意见，而将该行降为 `insufficient_evidence` 并保留 `review-binding-issues.json`；原回复不改，跨组/重复 ID、未知来源、Schema 错误等仍拒绝。

新的 `host_keyed_check_cells/v1` 传输由宿主把每个检查 ID 固定为对象键，模型只填判断内容；不再自由生成 ID 数组。内部 `candidate.json` 仍规范化为原 `REVIEW_SCHEMA`，未扩展 Runtime 语义或权限。三组单元用本地 `$defs/$ref` 共享 Schema，避免逐项展开超限；不放宽上下文。原始重复 JSON 键也拒绝，不能通过 JSON 后值覆盖来隐藏重复意见。旧数组回复仅供原版回放/离线诊断，当前 live reviewer 不自动兼容回退。

`hostQuarantine` 单列窄范围风险处理：被明确质疑且没有观察依据、仍独立原样保留的短原子值暂缓展示，不扩展为整段/代码的自动删除。原型支持范围与误报风险见[根因记录](SEMANTIC-CLOSURE-DIAGNOSIS.md)。原值、AI 疑点和宿主实际处置均保留；暂缓不证明值为假，也不意味着任务已完成。

此前整稿/宿主暂缓方案和所有失败作为历史证据保留。最多 8 个编辑调用是显式增加的总成本，不能只比较某个最终调用的时延。

### 逐项验收而非自动打分

`expectations` 的每项要求必须提前具有唯一 `id`、`statement` 和布尔 `critical`。全部任务材料先保存，再启动模型。运行后审阅实际稿件、工具轨迹和失败回执，为每项给出 `met / not_met / unknown`、解释和实际证据路径；原预期不改变。

每个任务的 `judgment.json` 采用 `evaluation.structured_authoring.seal` 封存，包含 `case`、`reviewKind: developer_ai_not_independent_gold`、`reviewedArtifacts`（相对任务目录的路径到文件 SHA-256）、`criteria`（`id/result/explanation/evidence`）、`outcome`、`substantive`、`unsafeCallObserved`、`falseCompletionObserved`、`criticalTaskMismatch`。至少绑定 intake 与实际 author/execute 摘要；不能从模型 supported 数自动生成判断。`scoped_task_fulfilled` 不允许存在未满足或未知判据。

```bash
python -m evaluation.semantic_closure_acceptance <批次> <新的验收报告目录>
```

收集器仅校验封存与验收算术：不同 Skill 按原始入口文本摘要去重，不以任务数、改名或仓库元数据膨胀分母。达到样本规模、至少三个有信息 Skill 完成且没有关键错误，才会输出 `prototypeExitMetByDeveloperJudgment`。它不是独立人工 Gold、自动语义 Oracle 或正式泛化门禁。

## English

September 15: the same `evaluation.semantic_typed_duty_probe` adds `--evidence-first-notes`, limited to extraction, reference location, pairwise comparison and observation-only note quotations; no body edits. Default remains zero-call preflight. The [report](SEMANTIC-DUTY-CONTRACT.md) records ten actual calls and the unresolved semantic counterexample. Strict auditing passes without legacy reconstruction; do not conflate this with whole-draft success or rerun old failures.

Later the same day: `evaluation.semantic_typed_duty_probe` is the explicit host-text/two-pass-predicate diagnostic, with independent owned edits and no failed-node retries. Its 18-call results, query assembly rejection and checkpoint-digest collision are disclosed in the [same report](SEMANTIC-DUTY-CONTRACT.md). Never overwrite or regrade old runs. Evidence collection remains strict by default; exact known digest collisions require disclosed reproducible derivative auditing, not skipping corrupted files.

Latest September 14 opt-in entry points: [duty/focused/local checks](SEMANTIC-DUTY-CONTRACT.md) and `evaluation.semantic_duty_resume`. Both the two-known-case original and independent continuation runs are complete without failed-call retries. Typed literal rendering is a separate zero-call API/artifact, not the original model's output or complete query-execution validation. Statements below about not starting a new batch are historical.

Current September 14 additions: optional lossless taskScope and verified read_context; separately owned negative-note edits sharing the body's eight-cell cap, with explicit deferral and original notes/host duties retained. A genuine whole-draft owner may add one heading while preserving existing headings; partial owners cannot. See the [design-stop report](SEMANTIC-CONTEXT-REPAIR.md) for frozen live results versus subsequent engineering-only fixes. New-policy structural application of an old proposal is not a new successful 9B run.

September 14: current review v9/keyed cells v2 adds exact task-delivery witnesses on top of the prior segmentation fix. Positive task opinions without an actual draft ID and exact quote are withheld; valid quotation is not semantic proof. Repair routing v3 leaves note findings unresolved rather than assigning them to body owners. The new two-call source/witness path is an [isolated diagnostic](SEMANTIC-WITNESS-DIAGNOSTIC.md), not extra default model calls. Historical versioned commands below belong to their frozen evidence; do not rewrite old receipts with current code.

Bounded-draft-review/v8 keeps ordered-list and numbered ATX-heading labels with their same-line bodies. It preserves original text/offsets and ordinary numerical sentence endings; this navigation fix neither drops claims nor approves titles or weakens citations. Do not rebuild historical frozen payloads as v8. The [four-case model contrast](SEMANTIC-REVIEW-MODEL-CONTRAST.md) is complete without changing the default 9B.

Source supply now prioritizes the complete entry, then whole inert Markdown reference documents that fit the unchanged budget. Oversized entries retain a contiguous paged prefix with explicit retrieval; unsupplied pages are not marked read. After a new read, previousCandidate remains a draft to update, never evidence; old remaining-actions move to the reference section. The host readStatusIndex points to exact tool/arguments/payloads and distinguishes historical from current completed reads without renewing authority/freshness or inventing tool messages.

The v2 repair trigger additionally accepts an unlocated negative only when one owned cell covers the entire original draft. Selecting one of several sections does not qualify. Its locationBasis explicitly denotes ownership, not proven semantic location. A preserved coverage opinion without a draft location is conservatively withheld as insufficient_evidence, with review-binding-issues.json and the unchanged raw response retained; other findings survive. Invalid schemas, duplicate/wrong-group IDs and invented citations still fail closed.

host_keyed_check_cells/v1 fixes each check ID as a host-owned object key rather than a model-selected array identity. The internal REVIEW_SCHEMA is unchanged. Local $defs/$ref share three cell shapes without increasing the context budget. Raw duplicate JSON keys are rejected before normalization. Legacy arrays remain historical/offline data, not an automatic live transport fallback. This fixes worksheet identity handling, not semantic judgment reliability.

Current grounded_patch separates keep/write_prose answer edits from exact supporting source_units; delivery.json retains both channels, prior unverified notes and host duties. Support is never a replacement answer or an excuse for false prose. Legacy source_patch is frozen in the [failed v1 batch](SEMANTIC-CLOSURE-TRANSFER-V1.md). Use freeze --known-development for known-source diagnostics; the acceptance collector rejects them as new-source evidence. Review worksheets retain all original prose; editor views intentionally expose only the owned candidate body and other-section index, with complete task/Skill/observations. The full parent remains in the frozen request and final review. Neither view is reversible encoding of the full audit JSON. Invalid annotations/read suffixes remain archived and require explicit admission; non-actionable negatives never become supported.

When no new observation is produced, reason_if retains the exact prior candidate with modelInvoked=false and no fake deliver model receipt. Snapshot review verifies the original candidate, digests, condition and bindings. A separate noReadSelection remains an unverified question/notice, never an observation, discharged duty or action permission. A new observation can invoke the bounded writer; proposed reads still face original contracts, finite resource ACLs and independent admission.

Only clarify questions enter unresolved context; answer-branch prose stays in the original selector trace/digest. Purpose-specific statement_checks, coverage_checks and task_checks remain in role-review.json before checked normalization. ATX, Setext and standalone strong labels, including list-wrapped labels, are protected. Exact observation-matching blockquotes are read-only within this pass; preservation is not truth or relevance. Historical edited drafts permit a fresh review after verifying freeze/application/candidate bindings and archives; old opinions and unseen-source credit are not imported.

Default located_findings schedules only a negative finding assigned to one host-owned section. Lexical differences and positive opinions do not trigger regeneration. Unlocated omissions remain explicitly open in repairPlan.unlocatedFindings instead of broadcasting edits. Fully read-only or unscheduled cells use reason_if to retain the unverified candidate without editor calls or repair credit. all_cells_diagnostic is an explicit historical diagnostic option. Final review is still not approval; general omission localization and implicit-premise detection remain unreliable.

This opt-in research workflow uses real local `qwen3.5:9b` with disclosed in-memory business fixtures. It neither changes default DSH routing nor executes source scripts or production operations. Freeze implementation/model/configuration before selecting transfer samples. Prepare each original Skill, scoped task, primitive tool catalog and fixture separately from predeclared expectations; expected answers never enter model or Runtime inputs.

Use `evaluation.semantic_closure_transfer` in `freeze → prepare → author → explicit admission → execute` order. Admission binds the reviewed compilation digest and is not generated from structural success. A developer-AI reviewer is not independent Gold. The original strict engine handles reads; optional finite continuations select reads/questions with independent resource admission. Historical snapshots cannot renew action authority. Review, one bounded revision and final review remain non-authoritative.

Inspect `review-before/model/review-before`, `repair/model/e*`, `repair/materialized` and `repair/final/model/review-after`. Source-reference mode renders exact host-validated observation slices in the deliverable; selecting the correct evidence and interpreting it remain open judgments. Per-cell validation protects ranges and section headings before downstream calls. Natural-language explanations remain model candidates, not approval. The former whole-draft/withholding alternatives remain historical failure evidence. One repair pass now allows up to eight editing calls: a disclosed cost increase, not a same-budget gain. The 2026-09-11 transfer v1 is frozen; six sources and twelve predeclared tasks follow the freeze, without result-driven replacement or tuning.

`evaluation.semantic_closure_evidence` audits receipts/source archives and counts every model attempt, including failure and unknown usage. Call latency is not task latency, SLO or causal A/B uplift. Existing attempts are not overwritten or retried; changed code/model/expectations require a new batch.

Predeclare unique criterion IDs and criticality before any result. Explicit developer review seals `judgment.json` with actual artifact hashes, criterion outcomes/explanations, task outcome and safety flags. `evaluation.semantic_closure_acceptance` checks bindings and arithmetic, not semantics: original entry-content digests define the Skill denominator; a fulfilled task cannot contain unmet/unknown criteria. Minimum coverage and three substantive fulfilled Skills do not excuse critical failures. A small developer-judged exit is neither independent Gold nor formal generalization admission.
