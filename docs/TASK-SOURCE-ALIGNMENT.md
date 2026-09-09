# 任务、源义务与宿主对齐 / Task, Source Obligation and Host Alignment

## 中文

后续 C3p 已补齐[本地隔离宿主与有界列解码](NETDATA-ISOLATED-VALIDATION.md)。下文保留 C3o 的诊断和当轮证据；源冲突、完整窗口及模型转译语义仍未闭合。

2026-09-09，C3o。**本轮把四份固定公开材料细化为可复核的任务档案，并定位了源材料与表达机制的真实缺口。** 这不是新增转译成功率：四个请求均由开发助手依据源文构造，仍缺准确宿主合同和完整语义审查，没有调用 9B 或执行第三方脚本。

### 四份材料现在分别在验证什么

沿用 C3k 每个固定采集批的第一个候选，不按容易通过换样；它们已经是开发材料，不是新未见集。任务范围由显式评测请求约束，原 Skill 和所有已保存附件仍完整保留。

| Skill / 任务规格 | 此次具体任务 | 已记录义务 / 问题 | 仍缺什么 |
|---|---|---:|---|
| OpenMontage [改名候选](../data/task-alignment/agent-name-update.json) | 读取指定 agent，仅提出修改 name 的候选，不登录、安装、创建、删除或外呼 | 7 / 2 | SDK/宿主版本、完整返回 Schema、凭据上下文、更新验证与补偿声明 |
| Netdata [单设备安全 trap](../data/task-alignment/snmp-single-device-security.json) | 单节点、指定 listener/IP、24 小时窗口，只返回脱敏摘要 | 12 / 5 | 节点 info/适配器合同、动态列解码、明确隐私与结果保留策略 |
| DeerFlow [项目 README](../data/task-alignment/project-readme-generation.json) | 依据隔离项目源码生成 README，保留现有结构并呈现文件 | 7 / 2 | 项目源码、文件/呈现工具；内容理解和文档生成继续属于 L1 |
| DevTools [隔离浏览器诊断](../data/task-alignment/isolated-browser-diagnosis.json) | 独立 profile 中观察 localhost 页面，提出诊断/修复候选 | 9 / 2 | 固定 tools/list、隔离页面、会话与 URL 权限；任意 JS 不能自动当安全只读 |

合计 **35 项开发者声明的源义务、11 个待处理问题、12 条未绑定宿主需求**：其中 9 条工具、2 条上下文、1 条策略。不能为凑齐“工具”而虚构认证或脱敏 API；35 也不是穷尽所有语义义务的证明。四份根 SKILL.md 已阅读，附件只对明确记录的区间作本次任务审查，未读区间保持可见。

### 新发现为何不应归因于 9B

1. **源中的隐私要求与 recipe 有张力。** Netdata 根文 50–54 行禁止把敏感原始日志写入持久制品；单设备 recipe 47–54 行却将原始响应写入 `.local/audits/...json`。其 95–102 行的所谓脱敏摘要仍输出 hostname 与自由 `MESSAGE`。是否包含敏感信息不能只看字段名；本任务明确禁止原始落盘，须显式解决取舍，不能直接执行来源脚本。
2. **原始协议示例也可能不一致。** 新补取的 `FUNCTION_UI_REFERENCE.md` 中，416–430 行给 `level.index=1`、`message.index=3`，但 442–450 行的示例数组相应位置是 rowOptions 和 `nginx`，并非注释中的 level/message。这是固定版本的示例问题，不是对实际 Netdata 实现的断言。不能拿这个示例反构造 Gold 或自动改成猜测的固定索引。
3. **嵌套数据支持不等于动态解码支持。** `columns.<key>.index` 决定每行字段位置；当前固定 JSON Pointer 不能表达数据驱动索引、逐行映射及分页完整性。不能通过将所有行藏进假汇总工具或固定取第一行来声称整任务完成。
4. **同是 integer，含义也不同。** `after/before` 使用秒，`anchor`/行时间戳使用微秒。类型通过不能证明量纲和参数来源正确。实际节点的 `info=true` 才提供当前有效参数，静态文档不能替代它。

这些判断均在 Netdata [档案](../artifacts/translator-v2/task-alignment-20260909/dossiers-v2/snmp-single-device-security/report.json)中绑定原文件、摘要、偏移、原句和行号。它们是开发助手审阅，不是独立真人 Gold 或已校准置信度。

### 实现与可解释性

[task_alignment.py](../evaluation/task_alignment.py) 复用现有惰性输入包、源跨度和结构化 Schema profile；没有新增执行器或产品控制面。

- 每份已保存文件都必须列出已读区间，未读附件不能静默消失；引用必须逐字匹配并落在声明的已读区间内。代码验证这些声明的绑定关系，不证明助手真的理解了原文。
- 源义务明确分为候选读取、候选 Effect、L1 推理、约束、宿主前提、条件义务、任务外与未决。不把“本任务不需要”改写成“整个 Skill 已覆盖”。
- 问题保留源冲突、上下文缺失、表达范围外、引用缺口及下一项修改建议；源张力至少关联两处不同原句。没有自动解决冲突或放行的路径。
- 工具需求必须引用明确工具名和完整原始 Schema 才能检查形状；根对象/不支持约束/缺少 outputSchema 均显式报告。上下文/策略单列，仍待宿主审查，不能以 `tools/list` 填补。**形状兼容不证明源 API 对齐、身份真实性或执行权限**。
- `model-input.json` 只含评测请求、全量原始分页与可选宿主声明，不加入本轮义务判定或问题结论。`report.json` 单独保存审阅档案；不能把它拼进转译器输入后再声称答案隔离。
- 模块不读取依赖 target 指向的路径、不执行引用脚本、不访问 API/设备、不生成已激活合同。最终状态恒为 `review_dossier_bound_not_translation_qualified`，不能拿档案绑定当语义准入。

当前原始输入文件数为 25。Netdata 本轮从**同一 commit**额外补取一份明确引用的协议文件，源包增至 **17 文件、202,930 字符、28 页**；旧 16 文件包和失败链保留。`<repo>/...` 普通文本引用此前没有进入词法链接候选，本轮通过显式依赖记录补上；没有宣称补取后传递引用已全部闭合。

### 本地查看与复现

先看表格中的任务规格；每项 `citations` 都有原句。再看输出的 `report.json`：`documentReviews` 定位未读区间，`obligations` 定位流程职责，`findings` 定位需要修订的源解释，`hostRequirements` 定位实际缺少的工具合同。

```bash
.venv/bin/python -m evaluation.task_alignment \
  artifacts/translator-v2/task-alignment-20260909/netdata-source/bundle.json \
  data/task-alignment/snmp-single-device-security.json \
  --output artifacts/my-snmp-task-dossier
```

输出目录必须不存在；可以附加 `--host-catalog captured-tools.json`。这只是提供待审声明，不是授予执行许可。命令依赖本地已保存的源包，只有 Git 中的任务规格不足以还原第三方快照；不可缺文件后偷偷换成另一版本。制品和源码摘要见[本轮机器报告](benchmarks/task-alignment-summary.json)。

### 本轮验证与证据

最终定向 **122 passed**，静止工作区全量 **1975 passed + 81 subtests passed（161.82 秒）**；31 个变更 Python 文件 Ruff、439 个变更文档本地链接和 diff 检查通过。工具/上下文/策略分类修正存为 `dossiers-v2`，原版档案和源码快照保留；四例的源包、任务/模型输入及宿主声明与原版逐字节不变。基于 Git 基线及两层源码快照的隔离重建，20 个新版输出文件逐字节一致。旧版 54 份绑定证据经原源码快照核验；C3n 的 76 份证据未变。

上述均为机械完整性、兼容性和可复现性检查，不是 1975 个 Skill、语义准确率或真实宿主验证。代码仍在 dev，未提交 Git。

### 调整后的下一步

**优先推进网络样本，不继续单纯扩测试条目。** 先为 Netdata 建立实际存在的本地隔离宿主：明确版本的单次 Function 调用、info 与查询响应 fixture、权限及输出策略。允许评测 adapter，但必须明示它是隔离宿主，保留原操作/字段语义，不能冒充原厂接口或把循环/分析藏成一个汇总调用。

随后增加有界、纯数据的按列元数据解码能力，验证重复/越界索引、缺列、空结果、错误响应和不完整结果；不开放任意脚本或通用循环。任务完整性仍由后续源审查判断，不能因为所有机械检查通过就激活合同。宿主与源义务闭合后，再冻结小批 9B 候选和独立评分输入；正式泛化门禁、唯一写路径及生产工程冻结保持不变。

## English

C3p subsequently implements an [isolated host and bounded column decoding](NETDATA-ISOLATED-VALIDATION.md). This document preserves C3o findings; source tensions, complete-window coverage and model translation remain open.

C3o, September 9, 2026, turns the same four fixed public entries into concrete **task/source/host review dossiers**, not successful translations. The evaluation requests are developer-authored. No 9B calls, third-party script execution or real-system operations occur.

The table links four specifications: an agent-name update proposal, single-device security-trap query, project README generation and isolated-browser diagnosis. They contain 35 declared obligations, 11 unresolved findings and 12 unbound host requirements: nine tools, two contexts and one policy. These are not exhaustively discovered obligations or twelve implemented tools; do not invent authentication/sanitization APIs to fill non-tool requirements. Root Skills were read; companion-file review ranges and unread material remain explicit. These are known development cases, not unseen independent Gold.

Source review finds concrete non-model issues. The Netdata recipe writes raw responses to disk despite privacy restrictions and emits hostname/free-form message fields under a sanitized label. The pinned protocol example's level/message indices do not match its annotated row; this is an example inconsistency, not a claim about the actual implementation. Row decoding requires `columns.<key>.index`, which fixed pointers cannot express generically. Seconds-based query bounds and microsecond cursors also share an integer type but not meaning. Current node metadata—not static documentation—is needed to fix the effective parameter contract.

The offline module reuses inert intake, exact spans and structured schema checks. Every retained document has declared review ranges; citations must match those ranges. Obligations separate candidate reads/effects, L1 reasoning, constraints, host preconditions, conditional duties, task exclusions and unknowns. Findings preserve both sides of source tensions and actionable next steps. Named host declarations can be checked structurally, without certifying source mapping, identity, effect safety or authorization. Dependency strings are never opened or executed.

Future author inputs contain the request, complete original source pages and an optional host catalog, **not the developer review decisions**. Review reports are separate. The status remains `review_dossier_bound_not_translation_qualified`; no digest or full inventory automatically becomes semantic acceptance.

One explicitly referenced Netdata protocol file was recovered from the same commit, reaching seventeen files, 202,930 characters and twenty-eight pages. Prior bundles remain intact. This plain-text `<repo>/...` dependency was missed by lexical link discovery and is now explicit; transitive reference closure is still unproven. The four bundles contain 25 files in total. Use the command above with a fresh directory and retained source snapshots; the Git specifications alone do not contain those snapshots. See the [bound summary](benchmarks/task-alignment-summary.json).

Final checks: **122 targeted tests; 1975 full-suite tests plus 81 subtests in 161.82 seconds**. Lint on 31 changed Python files, 439 local links and diff checks pass. Kind separation is retained as `dossiers-v2`, preserving v1 dossiers and source snapshots; all four source bundles, task/model inputs and catalogs remain byte-identical to v1. An isolated baseline-plus-two-overlay reconstruction reproduces twenty v2 files byte-for-byte. The original 54 evidence bindings verify through their archived source overlay; 76 C3n bindings remain unchanged. These are mechanical checks, not Skill counts, semantic accuracy or actual-host verification. Changes remain uncommitted on dev.

Next prioritize an actually instantiated, explicitly isolated Netdata host with a single Function-call primitive, versioned info/query fixtures, permissions and output policy. An evaluation adapter is acceptable if disclosed and source-preserving; an invented aggregate tool hiding loops or reasoning is not. Then add bounded pure-data column decoding, not arbitrary script execution or general loops. Freeze small 9B construction/review only after task/host/source gaps close. Large Runtime comparisons, default routing, the unique Effect path and frozen production-engineering scope remain unchanged.
