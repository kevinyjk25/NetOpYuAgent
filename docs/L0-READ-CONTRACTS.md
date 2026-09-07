# 只读 L0 合同 / Read-only L0 contracts

## 中文

2026-09-07，B2 开发原型。**单操作辅助闭环已验证：9B 原始提案→带证据的问题解析→当前助手重新审查→宿主授权本地读取。原始提案仍保留阻断状态；这是新建样例的辅助开发验证，不是全自动转译或泛化证明。**

### 解决什么问题

写操作的 `AtomicEffect` 必须有目标、预检、审批和独立验证，不能为无参查询编造这些字段。本轮在同一套 `network_runtime/l0` 类型、编译器、目录及 CLI 中增加 `AtomicRead` / `CompiledAtomicRead`，不修改已有写合同要求。两者沿用现有 v2 包版本，由 `kind` 区分。

```text
固定 Skill 文本 + 工具 Schema 文本 + 适配器声明文本
  → 摘要 / 工具映射 / 输入输出结构 / 读写与权限声明一致性检查
  → 未激活的 CompiledAtomicRead（可跨请求复用）
  → 本次结构化参数校验
  → ReadRequestDraft（等待语义审查和授权，不执行）
```

### 合同内容与边界

| 字段或接口 | 已实现 | 尚不能证明 |
|---|---|---|
| `sources` | 必须各有一份 Skill、tool、adapter 原文与摘要；解析 JSON 时拒绝重复键和非有限数字 | 源真实可信、Skill 与工具语义一致、接口确实只读 |
| `inputSchema` / `outputSchema` | 与 tool 源声明比对；封闭对象、最多 64 个标量字段；支持无参、必填、可选 | 嵌套对象、数组、enum、条件必填；不支持时拒绝，不扁平化 |
| `access` | 必须声明非空权限范围与数据分类，并与 adapter 原文一致 | 当前用户是否有权限、对象级访问限制、数据脱敏与输出释放策略 |
| `instantiate_read` | 复核合同摘要，拒绝未知/缺失/错误类型/非有限值/未解析模板；省略可选项不补默认值 | 从任意自然语言正确提参、意图无偏移、审批或授权通过 |
| `validate_read_result_shape` | 校验返回对象结构并给出摘要 | 业务结果正确、数据新鲜、来源真实；不是运行成功判定 |

工具的 `readOnlyHint` 不是安全证明；若它明确为非 true，本实现拒绝与只读声明冲突的候选。即使 Schema 和所有声明一致，`runtimeAuthorityGranted=false`、`semanticAlignmentProven=false` 也不能改为 true。本模块没有 provider、网络或脚本调用。

输入是已结构化的参数对象，不是用户自然语言。不得把这里的类型校验成绩算作 L1 参数提取准确率。源摘要用于完整性复核，不是签名或授权令牌；攻击者若同时伪造全部源声明，机械一致性检查不能替代独立审查。

现有写能力查询 `for_capability` 不会返回只读合同；写 Saga 投影拒绝只读合同；现有 effect L0.5 promotion 也会明确阻断 `AtomicRead`，直到只读语义审查路径实现。没有新增隐式读写混编或原生写回退。

### 本地使用

示例 [health.yaml](../examples/read-contracts/health.yaml) 是**手工合成夹具**，没有真实服务、模型调用或公网 Skill 的转译成绩。它独立于旧 B1 密封样例，不补写旧样例缺少的输出/适配器证据。

在项目根目录运行：

```bash
.venv/bin/python -m network_runtime.l0.cli --source examples/read-contracts/health.yaml validate
.venv/bin/python -m network_runtime.l0.cli --source examples/read-contracts/health.yaml explain fixture.health.read
.venv/bin/python -m network_runtime.l0.cli schema --kind read
```

```python
from network_runtime.l0 import L0Catalog, instantiate_read, validate_read_result_shape

contract = L0Catalog.from_path("examples/read-contracts/health.yaml").require("fixture.health.read")
draft = instantiate_read(contract, {})  # 无参数；只生成草案
assert draft["runtimeAuthorityGranted"] is False

# 离线构造返回值：只是形状检查，不是调用 health_snapshot。
shape = validate_read_result_shape(contract, {"healthy": True})
assert shape["shapeValid"] is True
assert shape["businessCorrectnessProven"] is False
```

带参数的合同中，每次请求单独传入参数；合同摘要不变，请求摘要随参数变化。合同里不存测试输入常量。暂不支持 read 的派生/组合及执行加载，避免用写合同语义隐式解释只读操作。

### B2b：L0.5 来源映射与审查

新增 `ReadL05Proposal`：`purpose` 为自然语言用途，`scope` 固定为单次读取操作，`operation` 使用既有只读合同声明，`unresolvedQuestions` 保留未解决问题。用途与编译后 `metadata.description` 必须一致；操作字段映射到 `spec`，不重新定义第三套执行流程。

`build_read_review_input` 自动生成完整检查项，而非让模型挑选审查范围：用途、Skill–工具映射、适配器映射、只读性质、完整输入/输出形状、每字段存在性/类型/必填性、权限范围和数据分类。每项带 `l05Pointer` / `l0Pointer`，源引用带原文、位置、来源及摘要；scope 不进入执行字段，`l0Pointer=null`，避免伪造对应关系。

`assess_read_l05` 复用已有源证据审查协议。缺项、重复项、错误来源种类、未知引用或摘要不一致不能通过；`contradicted` / `insufficient_evidence` 必须给出修改建议。非 supported 项或未解决问题会使 `compiledContract=null`。全部 supported 也仅得到 `review_supported_inactive_candidate`；不产生执行权限或 Gold。

当前每份审查最多 256 项，超过时明确拒绝，不能截断检查项。源引用目前是完整源文本的字符跨度，不是自动证明语义蕴含。支持比例不是校准置信度；同一助手的审查不能称为独立外部人员验证。

本地导出命令（输出文件必须不存在，避免覆盖历史结果）：

```bash
.venv/bin/python -m evaluation.read_l05_review scaffold \
  examples/read-contracts/health.yaml --output /tmp/read-l05-proposal.json
.venv/bin/python -m evaluation.read_l05_review review-input \
  /tmp/read-l05-proposal.json --output /tmp/read-l05-review-input.json
# 审查者填写 ReadL05Review 格式；不得自动将所有声明标为 supported。
.venv/bin/python -m evaluation.read_l05_review assess \
  /tmp/read-l05-proposal.json /path/to/review.json --output /tmp/read-l05-report.json
```

`scaffold` 是从手工 L0 反向生成的编辑模板，会自动留下审查待办；它不等于从 L1 自动生成 L0.5，不能用于计算正向转译成功率。用户也可按 `ReadL05Proposal` Schema 独立构造提案。

本轮当前开发助手对合成样例做了显式 AI 模拟审查：12 项中 10 项声明获支持、2 项证据不足，另保留 1 个脚手架待办。定位结果为 `/operation/tool`→`/spec/tool`（Skill 未明确绑定工具名）、`/operation/effect`→`/spec/effect`（无实际只读实现证据）。最终 **blocked，0 个晋级合同，0 次工具执行**。这是缺口定位演示，不是 83.3% 转译准确率。见[诊断摘要](benchmarks/read-l05-b2b-summary.json)。

接下来仍需真实或可审查的本地适配器实现证据、独立来源的正向 L1→L0.5 提案、请求语义审查与授权后受限读取。缺证据保持 unresolved，不为获得通过而补造源文。完成异质小批后再考虑扩大评测；生产工程和大规模 Runtime A/B 仍暂停。

详见[纠偏计划](TRANSLATION-CORRECTION-PLAN.md)与[项目进展](PROJECT-STATUS.md)。

### 新的正向 9B + 本地库存实验

[新 L1 样例](../examples/read-local/SKILL.md)明确指定 `read_inventory_device(device_id)`；[本地适配器](../network_provider/local_inventory.py)读取宿主固定的 [inventory.json](../examples/read-local/inventory.json)，不是返回硬编码结果。它限制文件大小、拒绝符号链接/特殊文件/重复 JSON 键，只输出声明字段，不能用请求参数指定路径。数据内容是新建的实验库存，不是实时设备状态或未知测试集。

模型仅提出用途、工具选择、读写分类和未解决问题。输入/输出 Schema 来自工具声明，L0.5 由正向提案和固定源合同组装，不从已有 L0 反向生成。模型输入、原始回答及审查报告保存在独立目录。

`execute_host_read` 复用 `ObservationPolicy`，要求宿主绑定精确合同摘要、能力、Schema、敏感级别和权限范围。身份必须由宿主显式提供，禁止隐式系统身份或 wildcard/system 绕过；再检查角色、数据级别、能力范围及 `device_id:<id>` 对象范围。合同依然不进入全局 active 注册表。`HostReadBinding` 是受信任宿主的库接口，**不是可以交给 LLM/外部请求自行构造的授权令牌**，也不是生产身份认证或沙箱。

本地命令：

```bash
# 新目录；真实调用 qwen3.5:9b，不自动重试、不自动审核。
.venv/bin/python -m evaluation.read_local_demo author artifacts/my-read-forward-run
# 按导出的 review-input.json 人工或显式 AI 模拟审查，另存 review.json。
# 只有审查获支持且 unresolvedQuestions 为空才继续；旗标代表宿主本次本地授权。
.venv/bin/python -m evaluation.read_local_demo run \
  artifacts/my-read-forward-run /path/to/review.json \
  'read_inventory_device device_id=campus-sw1' --allow-local-read
```

请求只接受上述字面命令语法，不能据此宣称任意自然语言意图已安全收敛。宿主样例固定只授予 `campus-sw1` 的读取范围；`idc-sw1`、额外参数及否定/复杂文字请求被拒绝。没有 DSH Agent 循环或浏览器交互接入。

本轮观测：前两次请求 HTTP 400，其中诊断响应明确为本地 grammar 解析失败；第一目录仅保存输入，未保存错误正文。随后只移除解码器 Schema 的字符串长度提示，完整 Pydantic 输出长度校验仍保留。第三次请求成功生成（10.80 秒，547 输入 / 115 输出 token），正确选择工具并描述本地快照，但仍留下 `status` 格式和设备不存在时处理这两个问题。当前助手对 21 项声明的模拟审查获支持，**未解决问题仍使报告 blocked，模型提案实际执行 0 次**；没有删改原回答来放行。

该次实验的单独接线回归读取了本地库存文件，验证权限拒绝、字段过滤和内容不变，但不是该次模型提案执行成功率。其后续问题解析和重新审查已在下节完成；原始提案和失败记录保留。不能靠重复请求直到模型不再提问来隐去成本。见[原实验摘要](benchmarks/read-local-forward-summary.json)。

### 问题解析与修订轨迹（本轮已完成）

`ReadQuestionResolution` 是单独文件，不覆盖原提案。它记录原提案摘要、作者、每个问题的原文/序号、答案及精确源引文。代码要求全部问题逐项对应、原文位置一致；它只提出一个清除已解释问题的候选子版本，工具、参数、输出和权限均保持不变。

新的审查包包含 **全部 21 项基础声明 + 2 项答案充分性判断**。审查者必须再次核对基础声明，并明确引用答案所依据的具体证据片段；仅引用同类文档不足以通过。旧审查摘要不可复用。任何不支持项仍会阻断，全部支持也仅产生未激活合同；执行另需宿主授权。

本次答案及依据：

| 原问题 | 证据与答案 | 修改位置 |
|---|---|---|
| status 是否有特定格式 | 工具 Schema 仅声明 string，无 enum/pattern；Skill 要求按快照字段报告，不推断实时健康 | `/unresolvedQuestions/0`；不改变输出 Schema |
| 设备不存在如何处理 | 适配器 `rows.get` 后检查 dict，不存在即抛出 `ValueError("inventory device not found")`；Skill 同样要求错误而非假成功 | `/unresolvedQuestions/1`；不改变异常代码 |

当前助手完成非独立 AI 模拟解疑/重审后，本地实际读取返回 `{"device_id":"campus-sw1","site":"campus","status":"planned-lab"}`。库存内容摘要与读取时相同；缺少宿主授权、复用旧审查、越权 `idc-sw1`、否定命令四种检查均拒绝且 Provider 调用数为 0。没有新增 9B 调用，前两次 HTTP 失败和原始两个问题仍保留在历史里。

使用新的输出文件保存解析与重审：

```bash
.venv/bin/python -m evaluation.read_question_resolution packet \
  /path/to/original-l05.json /path/to/resolution.json --output /tmp/resolution-review-input.json
.venv/bin/python -m evaluation.read_question_resolution assess \
  /path/to/original-l05.json /path/to/resolution.json --review /path/to/new-review.json \
  --output /tmp/resolution-report.json
.venv/bin/python -m evaluation.read_local_demo run \
  /path/to/original-run /path/to/new-review.json \
  'read_inventory_device device_id=campus-sw1' --resolution /path/to/resolution.json \
  --allow-local-read --output /tmp/resolved-read-receipt.json
```

输出已存在时拒绝覆盖。读取入口从原始模型回答重建父提案，再重新计算解析/审查绑定；不信任外部直接提供的“已清空问题”的子文件。实际制品位于 `artifacts/translator-v2/read-local-resolution-20260907/`；[摘要](benchmarks/read-local-resolution-summary.json)保留父/子/解析/审查/执行摘要。

限制：本解析器只解释问题，不修改合同。若答案要求改变步骤、工具、参数或权限，必须另行构造并审查新合同；不能套用清除问题的流程。引用完整性不证明语义必然正确，本轮答案作者和审查者仍是同一助手。下一步进行异质已知开发样例闭环，不能以这 1 个 Skill 的成功直接启动未知集合或大规模 Runtime 成绩对比。

## English

September 7, 2026 — B2 prototype. A single-operation assisted chain is verified: original 9B proposal → evidence-backed question resolution → renewed current-assistant review → explicit host-authorized local read. The original proposal remains blocked and unchanged. This is a newly authored development example, not autonomous translation or generalization evidence.

`AtomicRead` / `CompiledAtomicRead` extend the existing L0 v2 models/compiler/catalog/CLI. Effect contracts retain their target, preflight, approval and independent-verification requirements. No fabricated effect fields or weakened write gates are used.

Each inactive read contract pins Skill, tool and adapter source text/digests. Compilation checks source JSON integrity, tool identity, input/output schema equality, adapter capability/effect and access-declaration agreement. A contradictory `readOnlyHint` is rejected; a positive hint is not proof. Supported schemas are closed objects with up to 64 scalar fields, including zero/optional inputs; richer schemas fail explicitly.

`instantiate_read` rechecks contract integrity, validates typed arguments, rejects missing/unknown/invalid/non-finite/unresolved values and returns an unauthorized draft. It does not extract arbitrary natural-language intent. Optional omissions remain absent. Contracts are reusable and do not embed per-request fixture values.

`validate_read_result_shape` checks only object shape and produces a digest, not evidence of business correctness, freshness or authentic origin. Required scopes/data classification are declarations, not user authorization or data-release enforcement. Consistent source statements do not prove semantic alignment or provider behavior; hashes are not signatures or authorization tokens. All outputs remain inactive and non-authoritative.

The linked YAML and commands are an explicitly synthetic, hand-authored offline demonstration, not an LLM run or public-Skill translation result. They do not modify the prior sealed B1 sample. Existing effect lookup, Saga projection and effect promotion do not admit read contracts. Read derivation/composition and runtime loading remain unsupported.

B2b now adds `ReadL05Proposal` and a research review entry point. Purpose maps to the compiled description; operation fields map to the existing read spec. A deterministic exhaustive checklist covers purpose, operation/adapter/effect mapping, full schemas, each field's existence/type/requiredness and access/data-classification declarations. L0.5/L0 pointers and exact source spans identify what to revise. Non-executable scope has no fabricated L0 pointer. More than 256 claims is rejected rather than truncated.

The existing source-assessment protocol rejects missing/duplicate claims, unknown/wrong-kind citations and stale digests. Non-supported findings require actionable revisions and block promotion, as do unresolved questions. An all-supported result yields only an inactive candidate, never authority or Gold. Full-source character spans bind citations, not semantic entailment; reviewer support fractions are not calibrated confidence.

The commands above export a reverse scaffold, review input and assessment without overwriting prior output. The scaffold deliberately retains a review question. It is an editing template built from a hand-authored L0, not a forward L1 translation experiment.

The current development assistant's visible-context AI simulation reviewed 12 claims: 10 supported declarations, 2 insufficient-evidence findings, plus the scaffold's unresolved question. The Skill does not explicitly name the tool, and no implementation establishes read-only behavior. The outcome is blocked, zero promoted contracts and zero tool executions. This is not independent review or 83.3% translation accuracy; see the [diagnostic summary](benchmarks/read-l05-b2b-summary.json).

Genuine or auditable local adapter evidence, independently sourced forward L1→L0.5 proposals, request semantics and authorization remain necessary before a constrained local read. Missing evidence stays unresolved. Large Runtime A/B and production expansion remain paused; see the [plan](TRANSLATION-CORRECTION-PLAN.md) and [status](PROJECT-STATUS.md).

The new [local Skill](../examples/read-local/SKILL.md) and [inventory provider](../network_provider/local_inventory.py) supply an explicit tool mapping and auditable implementation. The provider really reads a host-fixed file, restricts size/type, rejects duplicate JSON keys and exports only declared fields. Contents are experimental inventory, not live telemetry or unseen evaluation data. The forward author asks 9B for purpose/tool/effect/questions while the source tool supplies schemas; it does not reverse an existing L0.

The subsequent resolution run retains the original model answer and uses a separate parent-digest-bound sidecar containing each exact question, answer and source quote/offset. It changes only the candidate child's unresolved-question list, not operation/schema/access semantics. A fresh review covers all 21 base claims plus both answer-entailment/sufficiency claims; exact answer citations are required and the old review cannot be reused. The host runner reconstructs the parent from the original model output and recomputes the revision/review binding before execution.

The current assistant's non-independent resolution/review supported both answers: status is an opaque string snapshot field without an invented enum, and a missing device raises the implemented error. The authorized local read returned campus-sw1/campus/planned-lab; inventory content remained unchanged. Missing host approval, stale-review reuse, another device's scope and a negated command were all denied with zero provider calls. No new 9B requests were made and prior failures/questions remain intact. See [resolution evidence](benchmarks/read-local-resolution-summary.json).

The sidecar cannot change contract semantics. If an answer requires a different tool, parameter, step or permission, construct and review a new contract instead. This is assisted single-Skill development evidence; the resolver and reviewer are the same visible-context assistant, not independent humans. Next is a heterogeneous known-development pilot, not an immediate large Runtime benchmark.

Host execution reuses `ObservationPolicy` and binds an exact contract/schema/capability/sensitivity/scope tuple. Explicit host identity, role, clearance and capability/object scopes are checked; implicit-system and wildcard shortcuts are rejected. A host binding is a trusted in-process configuration, not a client-issued token, production identity verifier or sandbox. No global activation occurs. The example accepts only a literal command and allows `campus-sw1`, not arbitrary natural-language intent or a DSH loop.

In the original experiment two requests returned HTTP 400; the retained diagnostic reports grammar initialization failure. The first attempt retained inputs but not the error body. Removing string-length decoder hints while retaining full output validation allowed generation in 10.80 s (547 input / 115 output tokens). Correct tool/snapshot selection retained two questions; the original 21-claim review therefore remains blocked with zero model-proposal executions. Separate wiring tests were not model-success evidence. The later evidence-backed revision described above closes one assisted read without rewriting that history. See the [original summary](benchmarks/read-local-forward-summary.json). Phase C has now completed [heterogeneous intake](TRANSLATION-BOUNDARY-PILOT.md), not translation or execution of the public Skills.
