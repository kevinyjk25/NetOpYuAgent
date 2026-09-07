# 合同优先的任务构造 / Contract-first task authoring

## 中文

2026-09-07，B1 开发接口。用于先修复评测构造，不是生产权限接口，也不是完整 Skill→L0 编译器。

### 为什么拆开

旧作者一次生成参数 Schema、任务类型、业务文本和期望处置；即使识别出无参接口，也可能生成缺参任务。现在分为：

```text
固定 Skill 文本 + 固定工具 Schema 文本
  → 源摘要校验与逐声明审查
  → 代码读取 required，推导任务槽位与 N/A
  → 9B 只输出 {user_prompt}
  → 参数/文本协议检查
  → 待任务语义审查（不是 Gold，也不进入 Runtime）
```

`inputSchema` 从工具源文解析，任务模型不能新增必填项或改 Schema。工具来源可以是保存的 API/MCP 合同快照；`origin` 是来源说明，摘要只证明文本未变，**不证明来源真实或接口行为正确**。写流程缺少验证/补偿时不生成虚构工具。

### 输入输出

- `ContractTaskRequest`：Skill 源文本/摘要、单工具源文本/摘要、待审读写性质、固定测试值。
- `ContractReview`：审查者身份说明、`ai_role_simulation` 或 `test_fixture`、绑定审查输入摘要的逐声明判断。必须覆盖工具映射、读写性质和输入 Schema；证据不足或相互矛盾就停止。
- `plan.json`：代码派生的槽位、缺失参数、N/A、私有意图假设和摘要；假设不是 Gold。
- 实际模型输入：工具接口、指定输入字面量及需省略的输入；不包含审查答案、评分标签、槽位编号或 N/A 决定权。这是作者输入，不能作为后续 Translator 输入。
- 模型输出：只允许 `user_prompt`。模型补全被要求省略的参数、改值、输出测试标签或增添协议字段时拒绝。
- `needs_task_semantic_review`：仅文字协议和参数条件匹配；例如请求动词是否忠实于 Skill、否定/指代是否改变语义，仍需审查，不得直接送 Gold。

适用性规则：

| 合同 | 代码派生任务 |
|---|---|
| 无参 / 全部可选 | 1 个正常请求；缺参族记录 `no_required_parameters` |
| k 个必填参数 | 1 个正常请求 + k 个单独省略必填参数的请求 |
| 读写性质未知 / 源审查未支持 | 0 个请求，不调用任务模型 |
| 当前不支持的 Schema | 显式 blocked，不静默简化或伪造默认值 |

B1 只支持封闭对象下最多 12 个标量参数；对象/数组、enum、条件必填、引用等复杂 Schema 会明确报告不支持。通用对抗任务不再硬塞给每个 Skill，后续应按业务边界设计独立测试族。该限制必须计入适配覆盖统计。

字面量边界复查：双引号字符串按 JSON 解码，保留引号、反斜杠、换行及 Unicode 的实际值；非法转义和解码后的未解析占位符拒绝。非有限数值（NaN/Infinity/数值溢出）不能作为参数或源合同 JSON 值。该检查只保证支持语法内的字面量完整性，不证明请求语义正确。

### 本地体验

以下样例为显式标记的**合成合同和测试审查夹具**。9B 真实生成文本，但夹具不是公开 Skill、外部人员审查或泛化证据。无需执行任何工具或脚本：

```bash
# 在项目根目录；使用现有本地 qwen3.5:9b
.venv/bin/python -m evaluation.translation_contract_task_runner run \
  examples/contract-first/request.json \
  examples/contract-first/review.fixture.json \
  --output-root artifacts/contract-first-demo

.venv/bin/python -m evaluation.translation_contract_task_runner inspect \
  artifacts/contract-first-demo
```

要接入另一份源合同，构造同格式 request，先导出新的审查输入，再由隔离审查角色填写 `ContractReview`，不能复制样例里的支持结论：

```bash
.venv/bin/python -m evaluation.translation_contract_task_runner review-input \
  /absolute/path/request.json --output /absolute/path/new-review-input.json
```

源文本变化后必须更新其真实 SHA-256，并重新审查；不应为了让旧审查继续通过而只改摘要。已完成运行重入只读；每次已完成的模型调用独立保存在 `checkpoints/`。中断目录不会自动覆盖或重复运行，须保留证据后另行处理；本版尚未实现部分运行自动恢复。

### 还差什么

1. 获得真实业务接口/适配器的非循环源证据，完成任务语义审查及隔离参考答案。
2. B2：L0.5→现有可复用 L0 编译器→本次请求实例化；本模块没有替代它。
3. 扩展 Schema、步骤/分支支持，再按原门禁冻结和采集未知集合。

权威进度见 [PROJECT-STATUS](PROJECT-STATUS.md)，总体安排见[纠偏计划](TRANSLATION-CORRECTION-PLAN.md)。

首轮真实 9B 的输入、模型和输出摘要见[B1 验证记录](benchmarks/contract-first-b1-summary.json)：槽位适用性正确收敛，但文本只有工具名，仍需修订和语义审查。不能与旧 doc-ingest 试验计算速度提升，因为输入、任务和协议都不同。

## English

2026-09-07, B1 development interface. This repairs evaluation construction; it is neither a production authorization interface nor a whole-Skill compiler.

Pinned Skill and tool-Schema text → digest-bound source review → deterministic required-input slots/N/A → 9B outputs only `user_prompt` → protocol/parameter checks → **task semantic review still required**.

The Schema is parsed from the source document, not authored by the task model. A source URI/digest does not authenticate a provider or prove behavior. Review kinds remain AI simulation or explicit test fixture, never independent human evidence. Unsupported/unknown contracts generate zero tasks and no model calls. Zero/all-optional inputs yield one nominal task; k required inputs yield one nominal plus k one-input-omitted tasks. The model cannot alter the contract, slot applicability or reference answers.

The commands above run a real local 9B author on an explicitly synthetic contract/review fixture. Outputs retain exact inputs, raw responses, per-call checkpoints and a seal. Completed reentry is read-only; partial runs are preserved and are not automatically replayed. For another contract, export `review-input` and obtain a new review instead of copying fixture support labels.

B1 supports closed object schemas with up to 12 scalar parameters. Complex schemas are reported unsupported, not flattened. Passing text/parameter checks does not establish semantic fidelity or Gold. Real source-backed interface review, reusable L0 compilation/request instantiation, richer schemas/flows and post-freeze unknown cohorts remain open. See [status](PROJECT-STATUS.md) and the [correction plan](TRANSLATION-CORRECTION-PLAN.md).

Literal checks decode double-quoted strings as JSON without changing escaped quotes, backslashes, newlines or Unicode. Invalid escapes and decoded unresolved placeholders are rejected. Non-finite values, including exponent overflow inside source JSON, are rejected. These checks establish bounded literal integrity, not natural-language semantic correctness.

The [B1 diagnostic record](benchmarks/contract-first-b1-summary.json) binds the real 9B probe. Slot applicability is deterministic, but the model emitted only a tool name, so task prose still needs revision/review. Different inputs/protocols preclude an acceleration comparison with the previous doc-ingest probe.
