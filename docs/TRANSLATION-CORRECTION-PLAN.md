# 转译纠偏与闭环计划 / Translation correction and closure plan

## 中文

2026-09-07。保留 Reasoning / Reliability Runtime / Infrastructure 三平面，以及 active L0、Evidence、审批和事务边界。当前优先修正转译研究链，不扩展生产工程。本文细化实施顺序；唯一阶段总表仍是 [PROJECT-STATUS](PROJECT-STATUS.md)。

### A：基础纠偏（已实现，验证结果见进展表）

- Translator v2.1 的模型输入不再包含 caseId、challenge、language 或 catalog assignmentId；这些评分元数据留在制品侧。用户原文和真正的工具标识保持不变，不通过改写请求提高成绩。
- 参数必须绑定到自己的显式字段和值。其他字段的真实字面量不是该参数的证据；所有同名赋值都参与冲突/类型检查，模型不能绕过它们。可选参数不提供时保持缺省，不猜默认值。
- 作者协议 v4 支持 0–6 个标量参数及逐项 required。没有必填参数时，缺参任务必须记为 `no_required_parameters`，而不是制造假参数。有效任务数和不适用槽位分别计数。
- 不可执行 Tool Catalog v2 只生成候选主操作，不再按“可逆”标签补造 preflight/verify/restore。源接口真实存在、读写属性和事务闭合仍需独立证据；构造通过不授予执行权限。
- 历史源文件、标签、基线和密封结果不改写。v1/v2/v3 检查路径保留；历史同一版本号内已有的规则漂移需另行定位，不能宣称所有历史重检均通过。

绑定当前仅覆盖 `name=value`、`name: value`、`name is value` 和 `name value` 的标量语法，不证明任意自然语言中的否定、指代或修正关系。没有字段归属证据时不进入 L0；只读可留给受约束的 L1，写入安全停止。该限制应体现在召回/覆盖率里，不能隐藏为“参数 100% 准确”。嵌套对象、数组、条件必填与完整 Skill 的自动编译仍未完成。

### B：贯通合同编译（单操作辅助闭环已验证，整体泛化仍待验证）

最新进展：证据解疑、修订留痕与重新审查已接通；单个新建库存 Skill 的 9B 提案经当前助手辅助审查后完成宿主授权读取。原始提案及失败记录不变。该闭环依赖助手解疑/审查、固定源合同和字面命令，不等于全自动 Skill 转译。见[证据摘要](benchmarks/read-local-resolution-summary.json)。

B1 已将源工具 Schema 审查、适用性推导和 9B 任务文字生成拆开；接口与例子见[合同优先任务构造](CONTRACT-FIRST-TASKS.md)。通过仅表示可送任务语义审查，不是 Gold 或可执行 L0。B2a 已补充只读合同编译与结构化请求草案；B2b 已接通只读 L0.5 来源映射和审查门禁。后续库存样例已接通显式宿主授权，范围仍仅限单操作实验。

目标链路：

```text
固定源 Skill + 有来源的工具合同
  → 逐声明源证据与缺失项
  → L0.5 可读语义提案（参数、条件、步骤、数据依赖、未知部分）
  → 现有 L0 合同编译器（可复用、尚未激活）
  → 本次用户请求的参数实例化
  → Runtime 接线 smoke（不是规模化研究证明）
```

必须区分“可复用 Skill 合同”与“单次执行计划”。当前公共 Skill evaluator 的单主操作计划不等于整个 Skill 编译成功。优先复用 `network_runtime/l0/compiler.py`，不增加第三套 L0 执行语义。

2026-09-07 复查发现 B2 的具体架构缺口：现有 `IntentSpec` 要求非空目标字段，`AtomicEffectSpec` 要求独立预检、效果验证及审批；它们是写操作合同，不能直接表示 B1 的无参只读操作。不能制造目标、预检或写操作来让样例编译通过，也不能放松写合同的门禁。B2 按以下顺序实施：

1. 在现有 L0 类型/编译体系中补充显式只读合同，支持无参和可选参数；读权限、敏感数据范围与结果 Schema 仍需真实源证据。不把只读等同于无风险。
2. 将源 Skill、工具 Schema、适配器读写属性与 L0.5 的逐项来源绑定到未激活的可复用合同。缺输出 Schema 或无法确认只读时保持 unresolved，不猜测。
3. 单独实例化本次请求；拒绝缺参、冲突、无效字面量与未解析模板。不同请求共享同一合同，不将测试输入常量烘焙进合同。
4. 通过“无参读、可选参数读、有参读、受控写、证据不足拒绝”的编译/实例化回归后，才做本地只读接线 smoke。写合同继续要求既有审批、独立观察及异常处置；多步骤/条件分支的整 Skill 转译不能由单操作结果代替。

B2a 已实现未激活的 `AtomicRead`，绑定源文本及输入/输出/适配器声明，并单独实例化参数、校验返回形状。B2b 的历史样例因 Skill 未绑定工具名及缺少实际实现证据而阻断；该结果不变。后续新建库存样例已完成正向提案、辅助解析与授权读取。接口和边界见[只读 L0 合同](L0-READ-CONTRACTS.md)。任意自然语言请求、整 Skill 流程与跨样本泛化仍未闭合；原 B1 密封样例没有被补造证据、编译或执行。

本轮 9B 暴露了一个顺序问题：即使选对无参操作，模型仍可同时生成缺参任务和错误 N/A。下一步先审查操作及真实工具合同，再由代码推导任务槽位适用性，最后才让模型写对应业务请求；不要继续让同一次模型输出同时决定参数 Schema、适用任务、答案和审查依据。

源证据允许区分 Skill 说明、实际 API/MCP schema、适配器映射和人工/AI 审查；不能拿作者自己生成的 Catalog 循环证明自身正确。缺验证或补偿接口就报告缺失，不补造。有歧义的声明保持 unresolved，模型置信度不授予权限。

### C：异质开发小批闭环（选样完成，转译尚未开始）

用户已确认先补最小业务流程执行，不能只产出提案。目前 [C1/C2](L0-BUSINESS-FLOW.md) 已完成只读条件/数据依赖，以及分支依据绑定 PreparedPlan、写前重读和原单写事务验证/补偿的本地接线。C3 仍需整流程 9B 正向生成和保真审查。原 Effect/Saga 保留；C1 候选本身不可执行，只有显式配置的宿主 gate 能进入原审批/执行入口。

已完成 12 Skill / 11 仓库 / 10 类结构的源绑定盘点，未提交各 Skill 的完整宿主工具环境，因此本轮模型/Runtime 调用均为 0，结果是 `not_run` 而非 0% 转译准确率。详见[边界小批报告与下一步设计收口](TRANSLATION-BOUNDARY-PILOT.md)。下一步建议将整流程提案与逐步可编译范围显式化，保留 L1 推理和未支持项，不能用 12 个专用适配器或一次读取来代替泛化。

B2 已经在独立修订记录中解析两个问题、重新审查并验证读取；[原实验](benchmarks/read-local-forward-summary.json)仍保持阻断历史。C 可以基于这个辅助闭环推进，但需同时统计原始首次生成率和经辅助修订后的完成率，不能抹去解疑成本，也不能把未支持流程悄悄降为单次读取。

从现有已知库选择 8–12 个不同结构的 Skill，覆盖无参、可选参数、引用、多步骤、条件分支、脚本依赖及不可支持的流程。先明确操作/步骤覆盖范围，再生成适用任务；不强制每个 Skill 相同任务数，不执行第三方脚本。

参考答案与 Translator 输出隔离。9B 作为转译器；GPT 可模拟外部审查，但保留来源、分歧与证据不足，标记 AI 模拟而非独立真人 Gold。失败样本和不适用项保留在总库中。

### D：冻结与未知集合（待实施）

小批闭环稳定后冻结实现，重新采集未知集合；原有 ≥3 cohort、≥50 Skill、≥15 仓库、≥8 领域、≥600 case 的[泛化门禁](TRANSLATION-GENERALIZATION-GATE.md)不下调。网络仍是应用锚点；跨域语料用于检验转译泛化，不扩展生产 Provider 产品线。

同时报告：接受后的语义正确率、适用 Skill 召回、整个 Skill/步骤覆盖、危险误接受、过度拒绝、参数错误，以及包含失败/修复成本的 p50/p95、模型调用和 token。按 Skill/仓库聚类统计。构造通过率、引用协议通过率、转译正确率、Runtime 执行正确率分别统计，不相互替代。

## English

2026-09-07. Preserve the three planes and active-L0, evidence, approval and transaction boundaries. Prioritize translation validity over production engineering. [PROJECT-STATUS](PROJECT-STATUS.md) remains the single phase summary.

### A: foundational corrections (implemented; validation in the status document)

Translator v2.1 removes scoring IDs/categories/language and catalog assignment IDs from model context without rewriting real user text or capability identities. Binding requires named ownership and checks every conflicting/invalid assignment; exact text elsewhere is insufficient. Omitted optional inputs stay absent.

Author protocol v4 supports zero to six scalar parameters with explicit requiredness. With no required inputs, the missing-input slot is recorded as `no_required_parameters`, not fabricated. Inert Catalog v2 materializes only the proposed primary operation; it no longer invents observation, verification or compensation interfaces. Neither schema conformance nor a reversible label establishes transaction closure or execution authority.

Historical sealed bytes, labels and baselines remain unchanged. Historical validation paths are retained; pre-existing intra-version rule drift needs separate investigation. Do not claim universal historical revalidation success.

The binder supports a bounded named-scalar grammar, not arbitrary prose entailment, negation or coreference. This limitation must appear in recall/coverage metrics. Nested/array/conditional schemas and whole-Skill automatic compilation remain open.

### B–D: remaining implementation

The approved [C1/C2](L0-BUSINESS-FLOW.md) local chain now supports read branches/data references, plan-bound business evidence, pre-write rereads and the original single-Effect verification/recovery. C3 whole-flow 9B generation and fidelity review remain open. Effect/Saga are reused; candidates alone remain unauthorized and only explicit host gates enter existing approval/execution.

The latest B2 experiment closes one assisted local chain using auditable inventory code, an original 9B proposal, evidence-backed resolution, fresh current-assistant review and explicit host authorization. Original blocked artifacts remain unchanged; the new [resolution evidence](benchmarks/read-local-resolution-summary.json) is separate. C can proceed as a heterogeneous known-development pilot, counting raw first-pass and assisted outcomes/costs separately. This does not establish autonomous whole-Skill translation or unseen generalization.

B1 separates source-contract review, deterministic slot applicability and 9B task prose; see [contract-first authoring](CONTRACT-FIRST-TASKS.md). B2a adds inactive read-contract compilation and typed request drafts; B2b adds source mapping and per-claim review. The later inventory example supplies auditable implementation evidence and closes assisted forward authoring and host-authorized reading. Arbitrary request semantics and whole-Skill generalization remain open. B1 candidates confer no Gold/Runtime authority.

The September 7 review identified a concrete B2 gap: existing effect contracts require nonempty targets, independent preflight/verification and approval, so the zero-input read fixture cannot use them honestly. B2a extends the existing compiler with inactive `AtomicRead`, pinned source/adapter/schema declarations, separate typed request drafts and offline result-shape checks; B2b adds a source-linked review gate. Its historical sample remains blocked for missing Skill/tool mapping and implementation evidence; the later inventory experiment does not rewrite it. No write gate was weakened and no fixture arguments were baked into contracts. Unknown semantics or missing result schemas stay unresolved. The old sealed B1 fixture was neither supplemented nor executed. See [read contracts](L0-READ-CONTRACTS.md).

1. Connect source-backed tool contracts and per-claim evidence to readable L0.5 and the existing reusable L0 compiler. Instantiate per-request parameters separately; the current single-operation evaluator plan is not whole-Skill compilation. Keep missing semantics unresolved and all generated contracts inactive. The 9B probe motivates contract-first authoring: derive slot applicability deterministically from reviewed inputs before asking for task prose, rather than asking one model response to invent schema, tasks, labels and evidence together.
2. Close an 8–12-Skill known-development pilot covering zero/optional inputs, references, multi-step/conditional flows, scripts and unsupported workflows. Intake now covers 12 Skills, 11 repositories and 10 structures, but translation has not run: complete host tool environments were not supplied. See [intake evidence and proposed scope](TRANSLATION-BOUNDARY-PILOT.md). Keep third-party files inert, unsupported steps visible and reference drafting separate from Translator outputs; GPT reviewers remain AI simulations, not independent human Gold.
3. Freeze the implementation, then collect unseen cohorts under the unchanged [generalization gate](TRANSLATION-GENERALIZATION-GATE.md). Cross-domain samples test research generalization; network remains the deployment anchor.

Report semantic precision among accepted translations, eligible Skill recall, whole-Skill/step coverage, unsafe accepts, over-stops, parameter errors, and p50/p95/calls/tokens including failed attempts and repairs. Preserve rejected and not-applicable denominators; use Skill/repository clustering. Construct, citation-protocol, translation and Runtime scores are distinct.
