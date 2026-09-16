# 引用寻址与紧凑交付 / Addressed Sources and Compact Delivery

## 中文

本轮收敛交付协议，不新增语义自审、工具权限或执行重试。两批真实DSH＋9B已结束，**最新完整任务仍0/2、原内容判据3/6，Gate 1未通过**。原生候选准入有所恢复，但不能称整体语义问题已修好。机制测试不等于语义准确率。

### 解决什么

此前原生候选反复漏填 `gap`、将 `state` 放错层级，复制原文时也会改变字符。模型同时承担“理解内容”和“抄写宿主已知结构”的负担。新版本将地址、摘要和机械状态交给宿主，保留模型对交付类型、内容、缺口的判断责任。

| 位置 | 新协议 | 仍需验证 |
|---|---|---|
| 来源→要求 | 模型选 `source_ref`，宿主按精确来源/偏移恢复文本与摘要 | 选中的文本是否真的表达该要求，类型是否选对 |
| 内容→结构 | 模型直接填每项内容；宿主构造内部状态，不要求重复包装 | 内容真实、充分、回答了原问题 |
| 无法完成 | 对应ID显式填null，并在unresolved同ID给原因 | 缺口判断是否正确、是否遗漏本可完成内容 |
| 候选→用户 | 保留原始候选与规范化摘要，确定性渲染 | DSH最终回复是否忠实；目前仍需单独评价 |

### 两种版本，不能混用

宿主 `apiVersion: netopyu.io/local-hybrid-host/v2` 启用交付合同 `source-anchored-delivery/v3`。既有宿主v1仍用原交付合同v2；老请求和失败报告不自动升级，不迁移或重评分。版本只能由操作者设置，Agent不能提交版本、策略或权限。其余宿主字段与[本地主链接入](GOVERNED-SESSION.md)相同。

不要直接修改运行中会话的宿主配置：会话绑定配置/代码摘要，漂移会拒绝。新版本应使用独立配置和输出目录，旧页面不会被自动重启。

### 引用不是模型生成的原文

`prepare` 提供 `deliverySourceReferences`，例如：

```json
{"task:0":{"origin":"task","text":"解释观察到的状态。 "},
 "task:10":{"origin":"task","text":"指出下一项所需证据。"}}
```

此处ID仅为格式示意，实际ID由宿主按原文偏移生成，不能照抄示意值。分隔符、空白、Unicode和重复文本位置保留；逐片拼接可恢复原文。机械片段最长900字符，不做语义分段或摘要，也不意味着片段就是一个完整要求。提案只提交对应ID：

```json
{"requirements":[{"kind":"analysis","language":"","source_ref":"task:0"}],
 "unrepresented":[]}
```

宿主不接受模型附加quote、offset或digest，也不从未知ID猜近似匹配。来源超预算时只提供实际任务引用；不可引用未供给的Skill页。要求类别仍有artifact、analysis、decision、next_steps；固定但无法表达的要求使用source_ref和reason保留。未读取事实仍不是固定交付缺口。

### 内容只写一次

假设宿主固定d0为analysis、d1为next_steps，原生deliver的response_json只编码一次下列对象；Runtime内部生成也使用同一Schema：

```json
{"delivery":{"d0":{"text":"观察说明，不是审批。"},
             "d1":{"items":["获取尚缺的授权观察。"]}},
 "unresolved":{},"uncertainties":["尚未独立验证原因。"]}
```

没有state/content/gap包装；宿主根据明确编码生成内部provided状态，但不产生语义批准。若d1无法提供：

```json
{"delivery":{"d0":{"text":"现有观察只能支持局部说明。"},"d1":null},
 "unresolved":{"d1":"缺少确定下一项检查所需的范围信息。"},"uncertainties":[]}
```

每个delivery ID必须出现。null必须有同ID非空原因；已提供内容不能同时标为unresolved。空文本不自动变为有效交付，类型不转换，孤立缺口/陌生ID/旧包装结构均拒绝。提供内容但仍有保留意见时放入uncertainties，不会丢弃。宿主只将显式null映射为内部未完成及对应空内容，这不是给错误的旧请求自动补默认值。

原生外层仍是 `deliver(session_id,response_json)`，严格解码一次，不修复JSON或执行工件。原始candidateDigest与canonicalCandidateDigest分别记录传输对象和规范化对象；渲染不改变模型所填文本。结构检查失败仍保存诊断，不能使一次Runtime生成变成可重放操作。

### 工程验证与证据边界

静态工具声明只公布引用字段形状，不在Agent尚未prepare时枚举全Skill的ID。实际可见引用由prepare提供，编译器按会话来源逐一校验成员关系。这是减少重复上下文，不是允许任意ID。CAPA的静态交付Schema从37,132字节降至802字节；不以字节减少宣称任务准确率或时延提升。

Runtime的本地Ollama请求现在将宿主绑定的outputSchema放入format，而非仅设置format=json；同时在模型上下文保留Schema。[Ollama官方结构化输出说明](https://docs.ollama.com/capabilities/structured-outputs)。宿主仍做独立类型/约束校验，并检查null与unresolved的跨字段关系。解码约束不是权限、证据真实性或语义证明，服务拒绝/生成失败不自动降级为自由JSON或重试。原生DSH外层并未获得同样的逐候选解码保证，不能将内层机制外推到所有tool call。

测试覆盖来源逐字可恢复、重复文本不同偏移、未知/未见引用拒绝、四种内容类型、显式null、孤立缺口、错类型、旧协议误投、回执绑定、单次Runtime和原生fallback。也保留反例：错选analysis替代完整代码时，结构可以完整而语义仍错误。不会把这种情况标为语义批准。

### 真实运行结果与下一步

同两份已知Skill/原任务/原数据/原判据，两次分别冻结实现，不合并为四份Skill或未见证据，也不覆盖失败。

| 批次 | CAPA | Mesh | 汇总 |
|---|---|---|---|
| reference-content | 入口字段反复错填，420秒超时，无有效会话，0/3 | 自动读取图＋两份实际读取＋一次Runtime；空unresolved字符串未通过Schema，0/3 | 0/2任务，0/6判据 |
| reference-content-bounded | 72.50秒完成原生候选准入/渲染，1/3；仍缺明确开放结论、下一证据 | 260.03秒结束；基础Schema通过，但“已有内容＋unresolved为None字符串”矛盾被拒；最终文字局部满足2/3 | 0/2任务，3/6判据 |

第二批CAPA保留工件正文与不确定性，但输出类型引用的是“先读取export”指令，不证明类型选择正确。Mesh还遗漏南区0.5%和390ms，建议重读已读窗口、构造未提供的导出路径；原生最终文字却称“交付/任务已完成”。**宿主task.delivery=null、not_completed没有被改成成功；但宿主状态不能自动约束DSH的最终文字。** 该错误完成声明单独保留，没有真实网络Effect/commit。

首批17份完整模型回复，280,724输入／5,224输出token；另有1个CAPA中断步骤用量未知，已知数是下界；代理在客户端超时后记录BrokenPipeError。首批p50/p95为411.87/419.23秒（含超时）。第二批13次模型调用（DSH12＋Runtime1），154,176输入／3,690输出token，完整回复均有用量；p50/p95为166.26/250.65秒。每批n=2，仅已知开发验证；多个机制同时修复，不能归因为单项因果性能提升、SLO或生产成功概率。

[机器摘要与逐项限制](benchmarks/delivery-reference-content-summary.json)。两份评测报告摘要分别为`sha256:f69d2506165082db7174596513ce9048ec4f211f9c42d30115e90410704bab45`、`sha256:24c29717948d00deec4cf2f20347b0b4dfe7da0495fd053700a6a41ce221a6f4`；两批115份运行制品和各75个归档源文件摘要均核对。开发者AI按原判据审阅，不是独立Gold。

本轮停止模型重跑。下一包应把“已提供”和“无法提供”编码为同一个位置上的互斥选择，减少跨字段同步；不把None字符串自动清除、不降低检查。还需解决职责类型/原文的错误关联与最终交付偏移；不再泛化增加审阅层。新来源验收、大规模A/B、生产工程继续关闭。未提交/推送或重启用户UI。

工程收尾：151项定向、**3,202项全量测试＋81子测试**（241.52秒）、106个变更/新增项目Python文件Ruff、文档/工具Schema/diff检查通过。全量pytest在两批模型运行结束后执行；未修改文件的既有lint问题不在本轮清理范围，不宣称全仓lint clean。工程通过不抵消上述语义与交付失败。

## English

This version reduces mechanical transcription, not semantic uncertainty. Operator host profile `netopyu.io/local-hybrid-host/v2` enables delivery contract v3; existing host v1 keeps the legacy contract/protocol. Do not change an active session's profile: host/source fingerprints remain binding. No automatic UI restart, migration, regrading or broader authority.

The host addresses exact original source spans. Agents select source_ref IDs instead of copying quotes/offsets/digests. Separators, Unicode and repeated-text positions are preserved; concatenation reconstructs the original, with a900-character mechanical span bound. Spans are not semantic requirement units. Unknown or unseen IDs are refused, never fuzzy-matched. Source overflow exposes task-only references. Interpretation/kind selection and completeness remain unproven.

The compact response has delivery, unresolved and uncertainties. Every host delivery ID directly contains its typed content or explicit null. Null requires its own nonempty unresolved explanation; supplied content cannot also be unresolved. Unknown/missing IDs, orphan explanations, wrong types and legacy wrappers are rejected without coercion. Host-owned internal provided/unresolved metadata follows this declared encoding, not guessed meaning or repaired legacy input. Nonempty-but-wrong content can still pass shape checks; semanticApproval stays false and taskSuccess null.

Both native fallback and Runtime use the same schema/rendering. Native deliver still takes one strict JSON-text envelope. Raw candidate and canonical representation have separate digests; supplied text is preserved. Static validation failure remains recorded and never enables execution replay. Final text rewritten by DSH requires separate evaluation. Tests cover lossless addressing, strict rejection, four content kinds, null handling, evidence binding and no-replay behavior; these are not model accuracy or generalization scores.

Static tool definitions now advertise reference shape without preloading the entire Skill ID inventory. Prepare supplies visible references; compilation still enforces exact membership. CAPA's advertised delivery schema drops from37,132 to802 UTF-8 JSON bytes, not proof of task/speed improvement. Runtime sends the host-bound outputSchema in Ollama's format field, retaining independent validation and null/reason cross-field checks. [Official structured-output API](https://docs.ollama.com/capabilities/structured-outputs). Decoder structure is not truth, authority or semantic success; failures do not downgrade to loose JSON or retry. Native DSH outer calls do not automatically gain the same per-candidate decoder constraint.

Two separately frozen runs of the same known tasks finish; they are not four independent Skills. The first scores0/2 tasks,0/6 criteria: CAPA never prepares a valid session before420 seconds; Mesh reads both exports but fails base schema on empty unresolved strings. It has17 complete model replies,280,724 input/5,224 output tokens plus one interrupted step with unknown usage, and a proxy BrokenPipeError after client timeout. p50/p95 is411.87/419.23 seconds including timeout.

The bounded-metadata/schema-decoder run scores0/2 tasks,3/6 criteria. CAPA admits a native candidate in72.50 seconds but lacks the explicit open conclusion and next-evidence note (1/3). Mesh passes base schema, then fails cross-field consistency because supplied content also has unresolved values of the literal string None. Its native final prose has partial value (2/3) but omits south metrics, suggests an already-read and an invented export, and falsely claims delivery/task completion despite host not_completed and null delivery. No network Effect/commit occurs. This run has13 model calls (12 outer,one inner),154,176 input/3,690 output tokens with complete usage, p50/p95 166.26/250.65 seconds. n=2 per batch, multiple changes, not causal performance/SLO evidence. Source IDs also still anchor wrong output-kind interpretations.

[Portable evidence and bound digests](benchmarks/delivery-reference-content-summary.json):115 run artifacts and75 source files per archive verify. Developer-AI review is not independent Gold. Gate1 stays open; no fresh-source acceptance, large A/B, commit/push or UI restart. Stop model reruns this turn. Next encode content versus inability as one mutually exclusive choice, without deleting literal None values or relaxing validation, and address requirement interpretation and host-result/final-prose divergence.

QA passes151 targeted checks,3,202 full tests plus81 subtests in241.52 seconds, Ruff on106 changed/new project Python files, documentation/tool-schema/diff checks. Full pytest runs after both live batches. Untouched historical lint findings remain out of scope, not a repository-wide clean claim. Engineering checks do not cancel failed semantic/task results.
