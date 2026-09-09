# C3d：层级流程到现有图 / Hierarchical flow compiler

## 中文

### 当前完成范围

已实现**离线 authoring 编译器**，把顺序、条件和终态表达机械编译到现有 FlowProposal。模型无需填写节点编号、`next` 或分支汇合边。没有新增执行器，也没有把该入口注册为 DSH Tool 或自动激活 L0。

本阶段新模型调用为 0。验证的是“层级表达的含义是否被编译保留”，不是“LLM 是否正确理解自然语言”。下一步才用冻结小批验证 9B 正向生成该表达；C3b/C3c 原始结果不改写，公开 Skill 泛化及大规模 Runtime A/B 仍未解锁。

### 人或模型需要写什么

完整可运行编译示例见 [flow-tree.json](../examples/read-flow/flow-tree.json)，原文是 [flow-source.md](../examples/read-flow/flow-source.md)。这是手工示例，不计为 LLM 正向结果。

| 作者提供 | 编译器负责 | 不替作者判断 |
|---|---|---|
| `read` 的工具、参数、结果别名 `bind`、原文编号 | 别名解析到准确的已有读节点 | 该读取是否属于源业务步骤 |
| `if_equal` 的字段、比较值、`when_equal` / `otherwise` 子流程及各自源编号 | 节点 ID、真假边、公共继续步骤、必经前序节点 | 真假分支业务含义是否写反 |
| 显式 `end` 或已存在的 `effect_candidate` | 终态接线、不制造成功、不调用写工具 | 业务是否确实完成、是否具备审批事实 |

例如，示例先将库存读取结果绑定为 `inventory`；若 `inventory.site` 等于 `campus`，以该结果的设备 ID 再读并结束，否则以 `needs_l1` 结束。源文使用的 `inventory` 是作者选的别名，不是权限或可信身份。

`steps` 按顺序执行。条件两侧写子列表，列表之后的步骤是公共继续部分。空子列表表示继续公共步骤，不表示成功；顶层每条路径必须最终到达显式终态。`effect_candidate` 仍只是现有宿主目标的单个终态候选，不执行，也不允许在其后接其他操作。

### 编译与安全边界

- 按前序遍历分配节点 ID，连接尚未完成的顺序出口，真假子流程各自落到公共继续节点。公共步骤只生成一次；边只能指向后面的节点。
- 读参数只能引用 `input` 或当前作用域中已完成的读取别名；不能引用自己、后序步骤、条件节点或兄弟分支的结果。全树别名唯一，禁止覆盖 `input`。
- **分支内部别名不向外导出**，即使另一分支会终止也一样。暂不支持自动合并两条路径的不同结果；需要时应设计明确的合并合同，而不是猜测来源。这是保守限制，可能拒绝本来可表达的流程。
- 从图机械计算必经前序步骤，再复用既有 source selection、cited review、`lower` 和 `qualify_flow`。参数类型、真实宿主合同、原文引用、DAG、权限和读取结果门禁不放松。
- 最多 64 个图节点、16 层嵌套；死代码、未结束路径和超限输入拒绝，不静默丢掉步骤。不支持循环、并行、多写或自动补偿编排扩展。
- 审查另有 256 项声明上限，源要求较多时可能先触及此上限；超限拒绝，不截断，也不保证所有 64 节点图都可在一个审查包内完成。
- `origins` 保留树位置→图节点映射。审查输入绑定整棵树、原文和映射；即使只改别名而生成图不变，旧审查也不能复用。
- 引用存在不证明语义正确；层级结构也不能把库存 `status` 变成审批事实。错误但类型合法的业务条件会原样编译，仍须源语义审查，绝不宣称自动纠正。

### 本地使用

在仓库根目录执行，输出文件必须尚不存在：

```bash
.venv/bin/python -m evaluation.flow_tree schema --output /tmp/flow-tree-schema-new.json
.venv/bin/python -m evaluation.flow_tree example --output /tmp/flow-tree-example-new.json
.venv/bin/python -m evaluation.flow_tree compile --sources path/to/FlowSources.json --tree path/to/tree.json --output /tmp/flow-tree-compiled-new.json
.venv/bin/python -m evaluation.flow_tree assess --sources path/to/FlowSources.json --tree path/to/tree.json --review path/to/review.json --output /tmp/flow-tree-reviewed-new.json
```

`example` 使用仓库内手工示例和现有只读合同，只编译，不读取设备或调用模型。`compile` 返回 SelectedDraft、现有 L0 图、树/源摘要、位置映射和双向审查输入。`assess` 只返回未激活审查状态，不授予运行权限；有未解决事项时继续阻断。

[示例摘要](benchmarks/flow-tree-example-summary.json)包含五个生成节点和源位置映射；完整审查包由 `compile` 按需生成，不把重复的合同源码全文堆进文档摘要。

### 验证口径

新增离线回归覆盖：顺序、输出引用、反向条件、嵌套条件、分支汇合、提前终止、空分支继续、词法作用域、隐式成功拒绝、单 Effect 终态、权限错误与审查绑定。

测试中有 7 类手工流程和 20 个固定随机种子的有界流程，分别对园区/IDC 两种数据验证。独立的**测试专用树解释器**计算预期读取顺序及终态；实际侧把编译图交给原 `run_read_flow`，经显式本地读权限读取临时库存文件，比较结果与调用顺序并确认文件未改变。解释器只在 tests 内，产品路径无法调用它。

这属于已知机械语义回归，不是新增 54 个 Skill、源语义 Gold 或生产安全概率。权限被拒绝时实际 Provider 调用为 0、结果 blocked，不会转成 false 分支。写候选只编译不发送；全阶段没有实际写操作。

最终 **60 项新增回归**通过，全量 **1043 tests + 81 subtests** 通过（86.60 秒）；定向 Ruff、git diff --check 与旧 C3b/C3c / 新编译示例重放通过。没有重跑旧模型评测或改写旧结果。

### 下一步

后续**不包含手工树答案**的 9B 正向小批已完成：4 个已知流程均未通过完整资格检查，不能用本页离线回归替代转译成功。详见[正向结果与失败定位](FLOW-TREE-FORWARD-PILOT.md)。下一步先分离 Schema/解码兼容性诊断与模型语义生成，再按宿主合同收窄可生成能力；大规模 Runtime A/B 不恢复。

## English

C3d adds an offline authoring compiler from a bounded sequence/if/otherwise tree to the existing FlowProposal graph. Authors supply operations, business conditions, lexical read aliases, source IDs and explicit terminals; the compiler owns node IDs, edges, joins and dominating predecessors. It does not reinterpret polarity, invent facts or activate L0.

Branch-local aliases cannot escape, names are globally unique, and every root path needs an explicit terminal. Empty branches fall through; code after fully terminal paths is rejected. Limits are 64 nodes and 16 nesting levels. Loops, parallel execution, merged branch values and multiple writes remain unsupported. Effect candidates are terminal, inactive proposals only. Existing source/type/contract/DAG qualification and Runtime access rules are reused.

Origins map tree paths to generated nodes. Reviews bind the entire tree, so an alias rename invalidates old review even if the graph is unchanged. Correct citations and graph structure still do not establish business entailment or turn inventory state into approval evidence.

Use the `schema`, `example`, `compile` and `assess` commands above. The [hand-authored tree](../examples/read-flow/flow-tree.json) and [summary](benchmarks/flow-tree-example-summary.json) are compiler examples, not model-generated evidence. The example command does not call models or execute providers.

Offline tests compare seven hand-authored patterns and twenty seeded bounded trees on two sites against a test-only independent interpreter. Actual compiled execution uses the existing read Runtime, explicit local permissions and temporary inventory files; outcomes/read order match and files remain unchanged. Access errors block before provider calls. These are mechanical regressions, not distinct Skills, semantic Gold or production guarantees.

The offline compiler milestone made no new model calls or operational writes. The subsequent [forward 9B pilot](FLOW-TREE-FORWARD-PILOT.md) is now complete, with zero of four known flows fully qualified. Next, isolate Schema/decoder compatibility and narrow generation by host capabilities; offline compiler correctness is not semantic accuracy. Public-Skill generalization and large Runtime A/B remain gated.

Final checks: 60 new regressions; 1043 tests and 81 subtests passed in 86.60 seconds, plus targeted Ruff/diff and old/new evidence replay. Review packets retain a separate 256-claim limit; large source/graph combinations are rejected rather than truncated.
