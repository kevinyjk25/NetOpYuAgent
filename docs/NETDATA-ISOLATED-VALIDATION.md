# Netdata 隔离宿主与动态列解码 / Isolated Netdata Host and Column Decoding

## 中文

2026-09-09，C3p。**已补齐一项通用表示缺口：按响应中的列元数据解码全部有界行，而不是猜固定索引。** 同时建立可实际调用的本地进程内宿主，经原有读取网关演示 `info → query → 解码 → 本页计数`。这是开发者接线、合成数据的机制验证，不是运行 Netdata 服务、原厂 SDK 互操作或 9B 自动转译成绩。

### 本次实际跑了什么

| 步骤 | 处理 / 可查看制品 | 边界 |
|---|---|---|
| 读取能力声明 | [host-catalog.json](../artifacts/translator-v2/netdata-isolated-20260909/demo/host-catalog.json)：一个 `fixture_netdata_function(node, function, body)` | 它是明确标注的评测 adapter，不冒充原厂 MCP 或来源 wrapper |
| 宿主身份与范围 | 显式本地 netops 身份、confidential 等级、logs scope、node/function scope；通过 `execute_host_read` | 不使用隐式 system / `*`，不是企业认证 |
| 发现参数 | 单独 `{"info": true}`，核对版本、accepted_params 和 listener options | 未知必选 widget、缺参数或无来源时停止；不猜默认值 |
| 单次查询 | 秒制 `after=-86400, before=0`、有界 last、backward、明确 selections | 不接受任意 Function、URL、shell、anchor 或未过滤的查询；不重试、不翻页 |
| 纯数据解码 | [projection.json](../artifacts/translator-v2/netdata-isolated-20260909/demo/projection.json)：`column_rows` 每次读取 `columns.<字段>.index` | 不过滤、不汇总、不改名、不执行 jq；逐行校验，任何一行异常都不能跳过 |
| 领域结果检查 | 返回的 job/IP/category/report type 必须满足请求；微秒时间戳必须落在秒制窗口 | 依赖显式的合成宿主时钟，不认证设备时间 |
| 本页发布 | [summary.json](../artifacts/translator-v2/netdata-isolated-20260909/demo/summary.json)：3 行，crit 1、warning 2 | 只发布枚举计数，不含原始消息、IP、hostname、时间戳；不是生产隐私证明 |

[trace.json](../artifacts/translator-v2/netdata-isolated-20260909/demo/trace.json) 可看到两次经过权限检查的回执摘要和解码步骤。宿主只返回原始形状的合成页，不把循环、分析藏进工具。原始回执只在本示例内存中使用，输出文件不保存原始行。

**终态是 `page_summary_ready_window_coverage_unproven`，不是任务成功。** Log Function 返回一页不代表整个窗口；`partial=false`、少于 last 条、没有 anchor、甚至空页，都不能证明完整的 24 小时总量。`wholeWindowCount=null`、`taskCompleted=false`、`wholeSkillTranslationProven=false` 明确保留。partial=true 或 partial 未知则停止，不生成摘要。分页、保留期、超时完整性仍需后续业务语义和宿主能力支撑。

### 通用算子如何使用

```json
{
  "kind": "column_rows",
  "source": "query",
  "pointer": "",
  "fields": ["timestamp", "TRAP_SEVERITY"],
  "max_rows": 200,
  "max_columns": 64
}
```

`source/pointer` 指向具有 columns/data 的已校验对象。目标 Schema 必须是数组，items 是恰好包含所选原字段名的对象，并声明全部 required、禁止额外字段。所选值的实际类型/枚举由目标 Schema 逐项验证。动态索引只来自当前响应，不能从 metadata 的显示顺序推测。

[column_rows.py](../network_runtime/l0/column_rows.py) 是通用原语，已接入 [structured_bindings.py](../network_runtime/l0/structured_bindings.py)、Tree 别名降级和共享 Flow 引用检查，保持来源支配关系、读取回执时效与仅候选 Effect 边界。旧表达式与既有执行器保留。上限为 256 行、128 列、32 个显式字段，同时受原有 JSON 字节/节点/深度预算约束；超限拒绝，不截断。

重复索引（包括未选字段）、布尔/字符串/负索引、缺列、行宽不符及目标值不合法会给出类别和路径。它没有一般循环、过滤谓词、自动分页或脚本执行能力，也不自动检查数据真实性、隐私和业务完成度。

### 源文如何处理，哪些仍未完成

[source-map.json](../artifacts/translator-v2/netdata-isolated-20260909/demo/source-map.json) 绑定原始公开文件、偏移、原句和明确的适配决策，固定使用 C3o Netdata 源包，不能换版本后沿用行号。

- 原始 token-safe wrapper 被明确替换为无凭据的隔离 adapter；没有执行或转译其认证/缓存脚本。
- 依照显式评测任务，不执行 recipe 的原始日志落盘；发布限于枚举计数，不声称完整复现 recipe 的自由文本输出。
- 源协议中不一致的列索引示例没有用作 fixture Gold；测试另行构造数据，并置换列及行位置检查映射不变性。
- Netdata 的 query 构造、info 检查和计数发布目前仍是 [开发者演示代码](../evaluation/netdata_task_demo.py)，没有假装是模型生成的 L0。通用解码表达式已可进入共享流程，但完整 Netdata 工作流的模型 authoring、源义务审查和准入仍未闭合。

### 本地复现

在项目根目录运行，输出目录须不存在。Git 代码不包含所有第三方源快照；以下命令需要此前已保留的固定源包，缺失时不会联网补取或偷换版本。

```bash
.venv/bin/python -m evaluation.netdata_task_demo \
  --bundle artifacts/translator-v2/task-alignment-20260909/netdata-source/bundle.json \
  --output artifacts/my-netdata-isolated-demo

.venv/bin/python -m pytest -q tests/test_column_rows.py tests/test_netdata_fixture.py
```

验证包括通用字段/嵌套值的 24 种列置换、整页/空页、拒绝路径、宿主权限、秒/微秒、部分返回、错误脱敏和共享 Flow 的别名/支配关系/时效。它们是机制测试，不是几十个新增 Skill 的泛化成绩。最终回归和可复现源码绑定见[机器摘要](benchmarks/netdata-isolated-summary.json)。

最终验证：**190 项定向测试；全量 2041 passed + 81 subtests passed（161.07 秒）**。36 个变更 Python 文件 Ruff、468 个变更文档本地链接及 diff 检查通过。基线加有序源码覆盖包在隔离目录中复现 6 个新版制品，逐字节一致；旧接线示例 9 个制品不变。历史 C3n 的 76 份证据通过对应源码快照校验；不覆盖历史报告。首次新测试检查为 65 passed，随后补充现有测试中的时效/支配断言及 1 项拒绝覆盖检查；仅修正测试格式 lint，没有修改旧 Oracle。Git 未提交。

下一步不扩大 Runtime 对照测试。先把完整任务拆成明确的可编译部分、L1 职责及未支持项，将当前开发者检查映射成待审 L0.5 表示并保留源解释；接入小批 9B 首次构造后，独立评分语义保留、参数和正确停止。不能用这个接线脚本充当模型答案，或将本页演示改称整个公开 Skill 通过。

## English

C3p, September 9, 2026, adds a reusable bounded `column_rows` binding operator and an **explicit synthetic in-process Netdata-shaped host**. The existing read gateway performs discovery and one query; pure data decoding and domain checks produce page counts. This does not run a Netdata server, test vendor interoperability or measure 9B translation.

The linked catalog has one Function-call primitive, not an aggregate tool. Explicit local identity, role, confidential clearance and node/function scopes are enforced before callbacks. Discovery must advertise the required query keys and listener. Query construction accepts a bounded seconds-based relative window and structured selections; arbitrary functions, URLs, scripts, unfiltered queries, retry and pagination are not supported by this isolated profile.

The generic operator preserves original field names, follows current column indices and validates every selected cell against the target Schema. Duplicate/invalid indices, missing columns, malformed rows and exceeded budgets fail without guessing or truncation. It integrates with existing structured bindings, Tree alias lowering and shared Flow dominance/freshness checks, without changing the Effect candidate boundary. Limits: 256 rows, 128 columns and 32 selected fields, also subject to existing JSON budgets. It is not a general loop, filter, sanitizer or completeness checker.

The task demo checks each returned row against source/device/category/type selections and a microsecond timestamp window, using an explicitly synthetic seconds clock. Its three-row page contains one crit and two warning entries. Only enum counts and non-payload trace are exported, not messages, hostnames, source IPs or timestamps. Counts are not a production privacy guarantee.

The terminal is **`page_summary_ready_window_coverage_unproven`**, with null full-window count and false task/whole-Skill completion. A non-partial response, short page, absent cursor or empty page does not prove complete time-window coverage. Partial or unknown scan status stops summary generation. Retention, pagination and completeness remain open.

The source map binds the previously reviewed C3o public snapshot and discloses adaptations: no original authentication/cache scripts, no recipe raw-file writes, enum-count publication rather than free-form output, and no reuse of the inconsistent upstream illustrative indices as Gold. Query/discovery/publication logic is still developer-wired demonstration code, not model-generated L0. The generic operator is wired into Flow; the whole Netdata model-authoring and semantic-admission chain is not.

Use the commands above with retained source snapshots and a fresh output directory. Tests cover twenty-four generic column permutations, empty and malformed pages, access, units, partial scans and shared Flow boundaries. See the [bound validation summary](benchmarks/netdata-isolated-summary.json). These are mechanism tests, not new Skill-accuracy scores.

Next explicitly represent compilable regions, L1 duties and unsupported semantics, bind the developer checks to reviewable L0.5/source decisions, then run a frozen small 9B first-construction and separate semantic review. Large Runtime comparisons remain locked; neither this script nor page-level counts are whole-Skill translation evidence.

Final validation: **190 targeted tests; 2041 full-suite tests and 81 subtests in 161.07 seconds**. Lint on 36 changed Python files, 468 local documentation links and diff checks pass. An isolated baseline-plus-ordered-overlays reconstruction reproduces six new artifacts byte-for-byte; nine legacy wiring artifacts remain unchanged. All 76 C3n evidence bindings verify against the corresponding archived source. Initial new tests passed 65 cases; later checks add freshness/dominance assertions and one overwrite-refusal test. Initial test-format lint was corrected, without changing historical Oracles. Changes remain uncommitted.
