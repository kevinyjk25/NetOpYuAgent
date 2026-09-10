# 结构化流程接线 / Structured Flow Wiring

## 中文

2026-09-10 增量：结构化 v2 Flow 的分支右侧现在也支持显式 `reference` / `array_length`，与左侧共同经过原绑定器的类型、依赖和时效检查。见[v65 修复及真实 9B 结果](STAGE-2-REPRESENTATION-REPAIR.md)。下文主演示及验收数字是当时冻结结果，不因这个新增表示而改分。

2026-09-09，C3n。**嵌套参数与读取结果现在可以进入现有流程执行器**，不再只停留在离线绑定草案。新增的是版本化表示与编译接线，不是第二套执行引擎。默认 DSH 路由、合同激活与唯一 Effect 准入路径不变。

这是一份人工构造的本地机制演示：真实执行 Python 只读回调，返回惰性 fixture；**没有 LLM、实际设备、网络请求或写操作**。它证明接线和约束行为，不证明公开 Skill 的语义转译准确率。

### 完整处理过程

| 层 / 制品 | 输入、处理与可查看的信息 |
|---|---|
| L1 式源文 | [中文在前的示例](../examples/translation-intake/structured-flow-source.md)：按设备读取接口，依据接口状态与计数分支，最终只生成候选。源文是开发 fixture，不是正式运维建议 |
| 待审 L0.5 式 Tree | [tree.json](../artifacts/translator-v2/structured-flow-20260909/demo/tree.json)：每个语句带文件、字符偏移和原句；参数是显式引用/对象构建，没有自动猜测 |
| 未激活 L0 式图 | [compilation.json](../artifacts/translator-v2/structured-flow-20260909/demo/compilation.json)：Tree 别名降为节点引用，检查支配关系，逐节点编译 Schema/JSON Pointer 绑定；`origins` 关联原句与节点 |
| 宿主准入 | 宿主代码提供确切的合同、回调、身份与范围；`HostFlowConsent` 绑定图和本次参数的摘要。模型 JSON 不能创建可信宿主对象 |
| 受控读取 | [execution.json](../artifacts/translator-v2/structured-flow-20260909/demo/execution.json)：现有 `run_read_flow` → 现有 `execute_host_read`；每次校验参数、权限与返回形状，轨迹记录绑定和读取凭据 |
| 终态 | 本例两次读取后停在 `awaiting_effect_admission`，返回 `enabled: true` 的候选，**不执行写入**；[调用记录](../artifacts/translator-v2/structured-flow-20260909/demo/calls.json)中的 writes 为空 |

此处 L0.5/L0 是研究表示层级，不表示已经走完产品 Promotion、语义审核或合同激活。源偏移正确只证明引文存在，不能证明它支持参数含义、分支极性或全部源义务。`authoring_digest` 将 Tree（包括引用）纳入图摘要，修改后需重新确认。

### 如何在本地复现

在项目根目录执行，输出目录必须尚不存在：

```bash
.venv/bin/python -m evaluation.structured_flow_demo --output artifacts/my-structured-flow
```

可先看 `tree.json` 的自然语言引用，再看 `compilation.json` 的 `origins`、`qualification.argumentBindings`、`qualification.controlSources`，最后看 `execution.json` 的 `trace`。`argumentBinding.arguments` 是每步实际使用的参数，`receipt.payload` 是形状验证后的返回值，`candidate` 只是提交给后续 Effect 准入的材料。

示例显式取 `/interfaces/0`，不是智能选口或集合循环；空数组必须停止。错误计数触发启用候选只是为覆盖分支/参数接线，不能当作生产故障处置策略。

### 保留的安全边界

- 有界 DAG：最多 64 节点、Tree 深度 16；拒绝循环、不可达步骤、未来/自身输出引用以及分支局部别名越界。对象、数组不是任意代码或循环执行权限。
- Schema 与数据：源声明和合同逐次核对；嵌套输入、输出及每次最终参数验证，不做强制转型、补默认值或丢弃约束。支持范围继承[数据绑定 profile](STRUCTURED-DATA-BINDING.md)。
- 读取权限：明确身份、角色、数据等级、能力 scope 和资源 scope。嵌套路径映射到已有权限检查，原始嵌套参数仍传给工具；拒绝隐式身份和 system / `*` 捷径。
- 时效：检查引用值，也重新检查影响当前节点的控制条件来源。限值取图与来源能力的较严格者；**只度量本地读取回执完成后的年龄，不认证设备采样时间**。源缓存、并发状态改变仍需后续独立证据验证。
- 隔离与诊断：复制嵌套参数/结果，避免回调修改已验证证据；内部数据校验提供错误类别与路径，底层异常不透传敏感详情。
- 写边界：只产生候选；编译、confidence、本地测试或读取成功均不授权写入。后续必须由原 Effect 门禁重新绑定/验证分支证据、权限、审批和状态，不能复用过期读取直接写。

`runtimeAuthorityGranted: false` / `contractActivated: false` 表示编译制品不自授执行权、合同未全局激活；本地演示的读权限单独来自宿主显式绑定。它们不表示从未调用回调。

### 已观察的结果与限制

主路径实际 **2 次读取、0 次写入、0 次模型调用**。额外[分支与拒绝记录](../artifacts/translator-v2/structured-flow-20260909/checks/local-variants.json)显示：接口已启用时 1 次读取结束；无错误时 2 次读取结束；空数组、错误接口权限在首读后停止；未认证身份在首读前停止。主演示离线重放与原制品逐字节相同。

与 Git 基线 `14baa0a` 的[兼容性核对](../artifacts/translator-v2/structured-flow-20260909/checks/legacy-compatibility.json)中，原七个开发图的 v1 资格包和摘要保持一致。这不是穷尽兼容证明，更不是七个公开 Skill 成绩。源码覆盖包、失败与回归结果见[证据摘要](benchmarks/structured-flow-summary.json)。

最终验收：**227 项定向测试；静止工作区全量 1949 passed + 81 subtests passed（226.55 秒）**。29 个变更 Python 文件 Ruff、418 个变更文档本地链接及 diff 检查通过。从 Git 基线 + 源码覆盖包重建的隔离目录中，9 份演示制品再次逐字节一致；C3m 的 21 份与 C3l 的 40 份绑定证据未变。初始两个测试因元组修改方式错误失败，修正测试构造后通过；另修复底层抛出内部同类异常时的诊断泄露边界。测试数与总耗时不作为 Skill 语义指标或 Runtime p50/p95。未提交 Git。

### 接下来

接线 smoke 已闭合；**公开源文 → 任务/真实宿主 → 模型 Tree 候选 → 完整语义审核**尚未闭合。本轮不接通默认 DSH、不新增网络设备写能力、不运行规模 Runtime 性能测试。下一步完成四份固定开发源文的引用闭合、任务/宿主声明及源义务映射，记录表示范围外项；再冻结小批 9B 的首次构造和独立评分输入。不得用本合成宿主替代公开 Skill 的真实合同，也不得将机械测试通过数计入 Skill 准确率。

## English

2026-09-10 extension: structured v2 Flow branches also accept an explicit reference/array_length RHS through the same binder, with type, dependency and freshness checks for both operands. See [v65 repairs and 9B evidence](STAGE-2-REPRESENTATION-REPAIR.md). The original demo and acceptance numbers below remain historical frozen results.

C3n, September 9, 2026: nested parameters and read results now run through the **existing shared flow and read gateways**, using versioned data/graph contracts and a source-anchored Tree compiler. No second execution engine, default DSH route or new Effect authorization path is introduced.

The linked fixture is explicitly developer-authored. The local CLI invokes real Python callbacks returning inert fixtures: two reads, zero writes and zero model calls. There are no network requests, actual devices, model-authored proposals or public-Skill generalization claims.

The table above links source text, the review-only Tree, the inactive compiled graph, execution trace and calls. Every Tree statement carries exact character offsets and a source quote; aliases lower into node references and per-node data bindings. `origins`, `qualification.argumentBindings`, `qualification.controlSources`, `trace`, validated receipt payloads and the final candidate expose the chain. Tree content is included in the consent-bound graph digest. Exact quotation is not semantic entailment or complete source coverage; these research representations do not imply completed Promotion or activation.

Use the command above with a fresh output directory. The explicit `/interfaces/0` is a fixed index, not intelligent selection; an empty array blocks. Proposing interface enablement from an error counter is only a wiring fixture, not an operational diagnosis or production runbook.

The shared qualifier enforces a bounded acyclic graph, dominance, lexical scope and typed explicit bindings. Every call retains exact host-contract/schema identity, explicit least-privilege identity and nested resource scopes through the existing observation policy. Source and target data are validated without coercion or discarded constraints. Controlling branch evidence is rechecked after intervening reads; the stricter flow/capability age bound applies. **Age is measured from local receipt completion, not authenticated device sampling time.** Provider mutation is isolated; internal validation diagnostics retain paths while provider exception details are redacted.

Effect leaves remain candidates only. Subsequent Effect admission must independently bind and revalidate evidence, state, authorization and approval. False authority/activation flags mean the artifact cannot authorize itself; the local read callback is separately authorized by trusted host code.

Additional recorded variants stop correctly for empty arrays, denied interface scope and unauthenticated identity, and follow both read-only completion paths. Seven original developer graph fixtures retain identical v1 qualification packets/digests against baseline `14baa0a`; this is bounded compatibility evidence, not exhaustive proof. The positive demo replays byte-identically. See the [versioned evidence summary](benchmarks/structured-flow-summary.json) for code bindings, failures and regression results.

Final validation: **227 targeted tests; 1949 full-suite tests and 81 subtests pass in 226.55 seconds** with a stationary worktree. Lint on 29 changed Python files, 418 local links in changed Markdown and diff checks pass. An isolated baseline-plus-overlay reconstruction reproduces all nine demo artifacts byte-identically; the 21 C3m and 40 C3l bound evidence files remain unchanged. Two initial test-fixture tuple-mutation errors were corrected; a provider exception/structured-diagnostic leakage boundary was also fixed. Counts and elapsed time are not Skill semantic scores or Runtime latency percentiles. Git remains uncommitted.

Next complete the fixed public sources' references, concrete tasks, actual host declarations and source obligations before a frozen small 9B construction/review probe. Whole-Skill model authoring, semantic acceptance, arbitrary loops/complex predicates, live-device writes and large Runtime performance evaluation remain outside this smoke. Mechanical passes are not Skill accuracy or production success probabilities.
