# C3e：生成协议诊断与宿主能力收口 / Protocol Canaries and Host Vocabulary

## 中文

### 要区分的三个问题

C3d 的 4 个正向开发提案全部未合格，原始回答保留在[前轮报告](FLOW-TREE-FORWARD-PILOT.md)。不能仅凭终态被写成 Effect 就断定“9B 不理解业务”，也不能直接断定解码器不支持终态。C3e 分开检查：输出 Schema 是否可表达、模型是否按格式生成、业务源文是否被忠实转译。

本轮 canary 是**明确给答案的复制探针**，不是隐藏答案的 Skill 评测。给定目标对象，测试普通终态、unsupported＋缺能力事项、条件两分支、空分支＋共同终态四类构造。每类三组各一次，共 12 次，均用 `qwen3.5:9b`、不启用 thinking、temperature 0、seed 20260907、context 12,288、输出上限 900。

| 分组 | 结构化接口 | 模型消息正文 |
| --- | --- | --- |
| format-only | 完整递归 FlowTree Schema | 明确目标对象与复制要求 |
| format-visible | 同一完整递归 Schema | 同一目标/要求，另附 Schema |
| json-visible | 普通 JSON 格式约束 | 与 format-visible 相同 |

每项的目标树在冻结前用 JSON Schema 和旧树编译器验证。首次输出必须完整解析、通过旧编译器且与目标 JSON 严格一致才记为 exactCopy；修改节点、类型或空分支都不能算通过。固定顺序、每组单次、不盲、答案可见；这些结果不能证明语义准确率、泛化、速度因果差异或 Runtime 安全性。

### 首次结果：基本构造可表达，不能据此宣称转译改善

| 探针 | format-only | format-visible | json-visible |
| --- | --- | --- | --- |
| 普通终态 | 精确复制 | 精确复制 | 抄入整个封套及 Schema，900 token 截断 |
| unsupported＋事项 | 精确复制 | 精确复制 | 精确复制 |
| 条件两分支 | 精确复制 | 精确复制 | 多出 `expectedObject` 外层，拒绝解析为 FlowTree |
| 空分支＋共同终态 | 精确复制 | 精确复制 | 精确复制 |

两组 Schema 约束各 **4/4** 精确复制，普通 JSON 组 **2/4**；全部首次输出保留，未去掉封套或补齐截断来刷结果。对应 [digest-bound 摘要](benchmarks/flow-tree-canary-summary.json)，本地完整目录为 `artifacts/translator-v2/flow-tree-canary-12-20260907/`。

这排除了本环境下“递归 Schema 完全不能生成这些基本构造”的解释，但不代表所有深度/组合都兼容。format-only 同样通过，因而**正文未附 Schema 不是本轮证据已确认的唯一根因**。正文显式 Schema 是可检查性改进候选，不是已证明的准确率提升。普通 JSON 对照的封套抄写失败也不能推论所有不受 Schema 约束的生成都会失败。

12 项 POST 共 **268.97 秒，14,438 / 2,735 输入/输出 token**，包括两个失败回答的消耗；计时含等待，不含预检和回归。全量回归同时运行，消息长度和格式各异，不比较因果时延提升。重放摘要的失败行保留错误及文件摘要；完整时延/token 从对应原始 `response.json` 统计，不能只统计通过行。

冻结 manifest 为 `sha256:976521c6a4b2bbb5f016aff78e45765c474cf029f73aea73ae9afc6844ae1a5a`。模型调用 12 次，业务工具/Runtime/脚本/写执行均为 0；公开 Skill 数、转译评测成功数均未增加。

新增 **23 项回归**通过；全量 **1079 tests + 81 subtests** 通过（147.50 秒），随后补充宿主合法 Effect 候选断言并再次通过 23 项定向测试。Ruff/diff、新摘要和旧 C3c/C3d 证据重放一致。合法候选仅编译，仍不授予执行权。

### 能力收口已实现，不替模型修改答案

[`evaluation/flow_tree_capabilities.py`](../evaluation/flow_tree_capabilities.py)由实际 `FlowSources` 生成 authoring Schema：

- 没有宿主读工具时，移除读构造器；有读工具时，`tool` 只允许宿主清单中的名字。
- 没有 Effect 目标时，在根/嵌套分支及 discriminator 映射中移除 Effect；有目标时，`binding_id` 只允许精确宿主 ID。读别名不被自动升级为目标。
- 新请求将同一份收窄 Schema 放到模型正文和格式接口，明确终态与 Effect 的区别；不包含用例名规则、目标答案或手工修图。
- 生成后再次验证宿主 Schema，再走原有源引用、类型、参数、词法作用域、DAG 和合同资格检查。解码器忽略约束时也不能绕过这些检查。
- 编译返回的仍是 `compiled_pending_source_review_not_executable`；未知事实、前置依赖、真假业务语义还需要源审查。支持写目标的 Schema 不是写授权。

这只是限制**合法表达空间**。不把非法 Effect 自动改成成功终态，不删除未解决事项，不替模型插审批，不放宽旧门禁；旧 C3c/C3d 生成协议和失败证据均未改动。

### 本地使用

安装开发依赖 `requirements-dev.txt`（新增 `jsonschema` 为离线评测依赖，不加入 Runtime 核心安装）。以下命令在项目根目录运行，输出路径须不存在：

```sh
# 新建独立 canary 批；不会运行业务工具或脚本。
.venv/bin/python -m evaluation.flow_tree_canary freeze /tmp/tree-canary-new
.venv/bin/python -m evaluation.flow_tree_canary run /tmp/tree-canary-new
.venv/bin/python -m evaluation.flow_tree_canary report /tmp/tree-canary-new --output /tmp/tree-canary-report-new.json

# 生成新的宿主收口 Schema / 模型请求；这里只导出，不发起 LLM 调用。
.venv/bin/python -m evaluation.flow_tree_capabilities schema path/to/sources.json --output /tmp/host-tree-schema-new.json
.venv/bin/python -m evaluation.flow_tree_capabilities request path/to/sources.json --output /tmp/host-tree-request-new.json
.venv/bin/python -m evaluation.flow_tree_capabilities validate path/to/sources.json --tree path/to/tree.json --output /tmp/host-tree-validation-new.json
```

`freeze` 保存模型、源码和 12 个完整请求/目标，之后 `run` 不改输入。完整 checkpoint 重入不重复调用；不完整或改动的 checkpoint 拒绝自动恢复。`report` 只重放原始响应及比较摘要，不调用模型。原始 `artifacts/` 被 Git 忽略，仓库报告只是摘要，不是完整可移植原始证据包。

### 后续边界

后续已完成另冻的[不给答案正向批 C3f](FLOW-BOUNDED-FORWARD.md)：4/4 结构合格，但完整源审查仍 blocked。复制探针不计为这些转译成绩。只有跨工具、跨 Skill 的开发验证稳定后，才进入未见集；12 个公开 Skill 整流程及规模化 Runtime A/B 继续保持门禁，不扩展生产工程。

## English

Results: **4/4 exact copies** with format-only, **4/4** with format-plus-visible-Schema, and **2/4** with JSON-plus-visible-Schema. The JSON end probe copied the entire envelope and truncated at 900 tokens; the conditional probe retained an extra `expectedObject` wrapper. All first responses remain unchanged. See the [replay summary](benchmarks/flow-tree-canary-summary.json).

The structured decoder can express these basic constructors in this environment; that does not establish compatibility for every depth/combination. Since format-only also passed, missing visible Schema is **not an established sole root cause** of C3d. Exposing Schema is a diagnosability candidate, not proven accuracy improvement. No answer-free forward run of the new host-bounded request has occurred yet.

All 12 POSTs, including failures, consumed **268.97 seconds and 14,438/2,735 tokens**. Concurrent regression, differing message lengths and fixed ordering preclude causal latency claims. Failed summary rows retain errors/file digests; total costs include their raw response envelopes. Twenty-three new tests passed; full regression passed **1079 tests + 81 subtests** in 147.50 seconds, followed by 23 focused tests after an additional valid-Effect assertion. Ruff/diff and old/new replay passed. Zero business/Runtime/script/write executions, no additional public Skills or translation successes.

C3e separates Schema expressibility, model protocol adherence and actual source fidelity. The previous C3d forward failures remain intact. These are **explicit-answer copying canaries**, not Skill translation: four constructor fixtures, each attempted once under format-only, format-plus-visible-Schema and JSON-plus-visible-Schema conditions. All use 9B, no thinking, temperature zero, seed 20260907, context 12,288 and a 900-token output limit. Targets are validated before freezing; exact copying requires valid compilation and identical JSON, not just parsing. Fixed order and visible answers cannot establish generalization or causal performance.

The new host-vocabulary module removes unavailable read/Effect constructors recursively, enumerates actual tool and Effect IDs, exposes the same Schema to messages and the decoder, then validates output independently before the existing compiler. It never rewrites unsupported targets into successful endings, invents approval or grants authority. Business entailment still needs source review; a permitted Effect constructor is not permission to write.

Use the commands above to freeze/run/replay canaries or export a host-bounded Schema/request and validate a tree. Install the development requirements for offline JSON Schema validation; Runtime core requirements remain unchanged. Completed checkpoints are not retried; modified or partial checkpoints fail closed. Local raw artifacts are separate from Git summaries.

The subsequent [C3f answer-free pilot](FLOW-BOUNDED-FORWARD.md) now qualifies four known flows structurally, but complete source review still blocks all four. Constructor probes contribute zero Skills and no source-translation successes. Public whole-Skill and large Runtime evaluation remain gated; no production expansion.
