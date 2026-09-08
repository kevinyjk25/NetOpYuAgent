# 结构化参数与输出绑定 / Structured Data Binding

## 中文

后续 C3p 新增[有界动态列投影](NETDATA-ISOLATED-VALIDATION.md)，接入同一绑定编译器；下文的四种表达式与成绩为 C3m 历史范围，不回填为新能力成绩。

后续状态：C3n 已将版本化绑定接入共享读/流程执行器，见[接线演示与限制](STRUCTURED-FLOW-WIRING.md)。下文保留 C3m 当轮范围与成绩；不回填历史结果，完整模型 authoring/语义审核仍未完成。

2026-09-09，C3m。**已实现结构化数据绑定原语和离线使用入口；尚未接入旧 FlowSources / FlowTree 的模型生成与宿主执行链。** 它解决旧读取合同只能表达扁平标量的问题，不替代完整 L0 的 Evidence、权限、事务和语义审查。

### 解决了什么

此前公开源审查发现，真实 API 的参数可能包含嵌套 `selections`、数组、大小写字段和约束。如果先删掉约束或把对象转成字符串来迁就旧 Schema，模型看到的就不再是原接口。

新增两个独立版本的组件，旧 `ReadObjectSchema`、旧合同摘要和激活路径均不修改：

`jsonschema` 的现有版本约束从 dev 移到 core 依赖声明，使安装核心包后也能导入该纯数据原语；没有新增服务、执行器或本轮安装操作。

- [structured_schema.py](../network_runtime/l0/structured_schema.py)：`netopyu.io/structured-data/v1`，原 Schema 保留与有界验证。
- [structured_bindings.py](../network_runtime/l0/structured_bindings.py)：`netopyu.io/l0-data-binding/v1`，引用/常量/对象/数组 → 可复核绑定计划 → 每次校验后的参数草案；不调用工具。

| 能力 | 新入口行为 |
|---|---|
| 嵌套对象、数组、null、类型集合 | 按原类型验证，不强制转字符串、不插入默认值 |
| camelCase、大写、点号、斜线、波浪号、中文键名 | 原样保留；使用 JSON Pointer，`~1` 表示斜线，`~0` 表示波浪号 |
| required、enum、const、数值界限、长度、数量、唯一性 | 保留并验证；参数类型能匹配不代表这些约束自动成立 |
| 局部 `$ref` / `$defs` | 支持有界、无环的本地 Schema 引用；不联网、不读取本地外部文件 |
| 结构化输出引用 | 先校验完整声明输出，再提取明确路径；可传整个数组或显式固定索引 |
| 源与目标的约束不同 | 不宣称子类型包含证明；每次分别校验源值和最终目标参数 |
| 可选字段、空数组、越界、null | 不能猜默认值或自动选其他项；定位到具体路径并阻断 |
| 宿主 catalog 变化 | 使用时重新核对完整声明和绑定；变化后旧草案拒绝使用 |

目前不支持的 Schema 关键字会明确拒绝，**不会丢掉后继续**。例如 `pattern`、`format`、`anyOf/oneOf/allOf`、条件 Schema、递归/远程引用、引用旁的断言约束。无类型的开放对象可以整值校验，但不能猜测未声明字段的类型。`default` 等注释保留但不执行；`readOnly` 等提示不构成授权。

限制是 profile 的显式组成，不代表 JSON Schema 本身的上限：每份 JSON 最大 1 MiB、深度 32、节点 16,384、单集合 1,024 项；Schema 最多 512 节点、展开深度 16；绑定最多 32 个源、512 表达式节点、深度 16。超限拒绝，不静默截断。

### 一个绑定怎么看

以下只说明数据对应关系，不是 L1 自然语言的语义证明：

```json
{
  "kind": "object",
  "fields": {
    "deviceId": {"kind": "reference", "source": "input", "pointer": "/device/id"},
    "selections": {
      "kind": "object",
      "fields": {
        "states": {"kind": "reference", "source": "input", "pointer": "/states"}
      }
    }
  }
}
```

输入 `{"device":{"id":"lab-router-1"},"states":["down"]}` 生成 `{"deviceId":"lab-router-1","selections":{"states":["down"]}}`。`mappings` 记录目标路径、表达式位置、来源路径、来源是否保证存在，以及每次值校验要求。`sourceValueDigests` 与 `argumentDigest` 绑定本次数据，不证明数据来自可信设备。

四种表达式只有 `literal`、`reference`、`object`、`array`。没有字符串插值、eval、Shell、自动遍历或工具发现。显式引用 `/interfaces/0` 不等于“智能地挑选正确接口”，其业务含义仍需源文/任务审查。

### 本地使用与证据

在项目根目录，无需 LLM、设备或网络：

```bash
.venv/bin/python -m evaluation.structured_binding_probe demo \
  --output artifacts/my-structured-binding-demo
```

每次使用新输出目录。演示沿用上一轮[合成接口声明](../examples/translation-intake/mcp-catalog.json)，没有替公开 Skill 虚构闭合宿主。

| 演示观察 | 结果及含义 |
|---|---|
| 同一输入/输出 Schema 的类型可表达性 | 旧扁平入口 0/2，新结构化入口 2/2；只说明这两份 Schema 的结构兼容，不是 Skill 转译率 |
| 嵌套输入 → 参数草案 | 字段、数组和大小写完整保留，无工具调用 |
| 结构化 fixture → 明确输出路径 | 提取 `interfaces[0]` 成功；fixture 不是实际设备返回 |
| 非法枚举 | 在 `/target/selections/states/0` 阻断 |
| 空数组 | 在 `/sources/status/interfaces/0` 阻断，不猜其他数据 |
| catalog 漂移 | `host_binding_drift`，旧草案不能继续使用 |

真实生成制品：[绑定计划](../artifacts/translator-v2/structured-binding-20260909/demo/host-binding.json)、[参数结果](../artifacts/translator-v2/structured-binding-20260909/demo/arguments.json)、[输出映射](../artifacts/translator-v2/structured-binding-20260909/demo/output-projection.json)、[正负路径报告](../artifacts/translator-v2/structured-binding-20260909/demo/report.json)。报告明确记录 **LLM 调用 0、Provider 调用 0、执行权 false、语义指标 null**。

自己的接入包提供四个字段：`catalog`、`tool`、`sourceSchemas`、`expression`。原始 MCP/API 声明放在 catalog；不是把自然语言 description 当作实际的权限证据。

```bash
.venv/bin/python -m evaluation.structured_binding_probe compile packet.json --output artifacts/my-binding
.venv/bin/python -m evaluation.structured_binding_probe materialize \
  artifacts/my-binding/host-binding.json captured-catalog.json source-values.json \
  --output artifacts/my-arguments
```

`source-values.json` 必须恰好提供所有被引用的来源，每个来源先全量验证。成功只生成 `StructuredArgumentsDraft`，不能直接调用写工具；失败给出类别和路径，不回显被拒绝的数据内容。`sourceBundleDigest` 的编程接口可记录源文关联，但绑定模块并不重新审阅源文或认证它。

### 后续边界

1. 将版本化数据绑定接入完整的 source/task/host authoring 与 FlowTree，并保留控制流支配关系、Evidence 时效和 Effect 门禁；不得借新数据类型绕过旧权限模型。
2. 集合循环、条件 Schema、复杂谓词、跨页语义汇总分别推进。参数能装进 Schema 不代表已支持任意流程。
3. 四份公开开发材料的宿主/用户任务与必要引用仍需闭合，再冻结小批 9B 候选与独立评分输入。不得用本合成演示替代公开 Skill 的语义泛化评测。

默认 DSH/Runtime 行为不变；旧制品和成绩未改写。正式泛化门禁、生产工程冻结边界不变。

可提交的[实现与证据摘要](benchmarks/structured-binding-summary.json)绑定源码覆盖包、依赖声明和正负演示结果；离线回放逐字节一致，上轮 C3l 的 40 份证据摘要仍一致。首次全量回归在同时修改文档时触发 `sourceState` 漂移保护，该负结果保留，停止修改后单独复验通过，不修改冻结校验器来规避。

最终验收：94 项定向测试通过；静止工作区全量 **1908 passed + 81 subtests passed，213.70 秒**。23 个变更 Python 文件 Ruff、390 个文档本地链接、21 份本轮证据与 diff 检查通过。测试数与总耗时属于机械回归，不是 Skill 样本量、语义准确率或 Runtime 时延；未提交 Git。

## English

C3p subsequently adds [bounded dynamic column projection](NETDATA-ISOLATED-VALIDATION.md) to the same binding compiler. The four-operation scope and results below remain historical C3m evidence.

Follow-up: C3n connects these bindings to shared read/flow execution; see [wiring and limits](STRUCTURED-FLOW-WIRING.md). The C3m scope/results below remain historical and unchanged. Full model authoring and semantic review remain open.

C3m implements **versioned structured-data binding primitives and an offline CLI**, not integration into the legacy FlowSources/FlowTree model-authoring or host-execution path. Historical ReadObjectSchema, contract hashes and activation remain unchanged.

The existing `jsonschema` version range moves from dev to core dependency declarations so a core installation can import these pure-data primitives. This adds no service, executor or installation operation in this run.

The [schema profile](../network_runtime/l0/structured_schema.py) preserves nested objects, arrays, null/type sets, original field names, required/enum/const, bounds, lengths, counts, uniqueness and bounded acyclic local schema references. The [binding module](../network_runtime/l0/structured_bindings.py) supports only literals, explicit JSON Pointers, object construction and array construction. There is no coercion, default insertion, interpolation, code execution, implicit selection or provider invocation.

Unsupported constraints fail explicitly rather than disappearing: regex patterns, formats, combinators, conditional schemas, recursive/remote references and assertion siblings of `$ref` are not in this profile. Untyped open-object fields cannot be guessed. Budgets are 1 MiB/32 levels/16,384 JSON nodes/1,024 entries per collection; 512 schema nodes/16 expanded levels; and 32 sources/512 binding nodes/16 expression levels. These are implementation limits, not limitations of JSON Schema itself.

Binding qualification checks paths and overlapping types, **not universal schema subtyping or semantic correctness**. Each materialization validates every supplied source and the complete target; missing optional paths, empty arrays and target constraints can still block. Exact source sets are required. Full tool declarations and catalogs are rebound before use; changed declarations reject old drafts. Read-only hints, content hashes and optional source-bundle links are not permissions or authentication.

Run the `demo`, `compile` and `materialize` commands above in fresh directories. The demo uses the previous explicitly [synthetic interface](../examples/translation-intake/mcp-catalog.json), not an invented public-Skill host. Its two declared schemas move from 0/2 compatibility with the old flat reader to 2/2 in the new profile. Nested arguments and fixed-index output projection succeed on inert fixtures; invalid enums, empty arrays and catalog drift are blocked with precise diagnostics. This is a type/binding result, not a Skill translation success rate.

Inspect the [binding](../artifacts/translator-v2/structured-binding-20260909/demo/host-binding.json), [arguments](../artifacts/translator-v2/structured-binding-20260909/demo/arguments.json), [output projection](../artifacts/translator-v2/structured-binding-20260909/demo/output-projection.json) and [positive/negative report](../artifacts/translator-v2/structured-binding-20260909/demo/report.json). Model calls and provider calls are zero; execution authority is false; semantic and Runtime latency metrics are null. A declared `/interfaces/0` is an explicit index, not proof that the first interface is the business-correct selection.

Next connect structured bindings to complete source/task/host authoring and existing graph dominance, freshness and Effect admission. Collection loops, complex predicates and cross-page semantics remain separate work. Finish concrete public-source alignment before a frozen small 9B probe. This synthetic mechanical demo neither replaces public-Skill semantic evaluation nor unlocks large Runtime experiments.

The [versioned evidence summary](benchmarks/structured-binding-summary.json) binds the source overlay, dependency declarations and positive/negative demo. Offline replay is byte-identical and the forty prior C3l evidence files remain unchanged. The first full suite detected `sourceState` drift during concurrent documentation edits; that result is retained, and the isolated test passed once edits stopped. The freeze validator was not weakened.

Final validation: 94 targeted tests; **1908 full-suite tests plus 81 subtests passed in 213.70 seconds** with a stationary worktree. Twenty-three changed Python files pass lint; 390 local document links, twenty-one bound evidence files and diff checks pass. Counts and elapsed time describe mechanical regression, not Skill sample size, semantic accuracy or Runtime latency. Git remains uncommitted.
