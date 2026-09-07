# 整流程正向转译开发实验 / Forward flow translation experiment

## 中文

2026-09-07。已接通 C3 的**一个已知本地流程辅助闭环**：9B 读取源说明和宿主工具合同→生成步骤/条件/引用→确定性结构检查→双向语义审查→有源文本修订→重新审查→宿主授权读取。未完成公开 Skill 小批泛化、无辅助可靠转译或 DSH 整 Agent 循环。

### 这次确实让模型生成了什么

输入是[本地流程源说明](../examples/read-flow/flow-source.md)、宿主输入 Schema、工具输入/输出和运行规则。**没有提供手工参考流程图、答案或执行结果**。现有只读工具合同为宿主提供，不声称本轮连 Provider 合同也由模型生成。

9B 决定业务用途、步骤顺序、条件、真假去向、参数/步骤结果引用和终态；代码绑定真实工具合同、宿主权限/时间预算，不补造步骤。第三版协议中，模型使用从 0 开始的节点序号，编译器只生成对应 ID。它不能替模型修改入口、去向、字面量、参数或删掉未知步骤；错误序号、循环和不可用引用仍被拒绝。

### 三次真实调用：失败没有被抹去

| 开发尝试 | 调整 | 模型调用耗时 | 原始结果 |
|---|---|---:|---|
| 命名节点 v1 | 首次完整流程 Schema | 69.61 s | 入口不是节点、引用源错误、把 needs_l1 当 Effect、虚构条件/自循环；结构阻断 |
| 协议说明 v2 | 补充通用节点类型和宿主规则 | 45.54 s | 节点类型改善，但入口和分支仍指向不存在节点，并有未解决问题；结构阻断 |
| 序号协议 v1 | 编译器拥有 ID，模型只选择序号 | 33.66 s | 结构通过；语义审查仍阻断 |

三次都是 `qwen3.5:9b`、`think=false`。共 **148.80 秒模型响应时间，2,535 输入 / 1,348 输出 token**。这是同一已知源说明上的三个不同开发协议，不是独立重复测试；不能据此声称时延改善比例、转译成功率或 p50/p95。时间不包含人工审查、修订与测试耗时。

第三次保留了正确的五节点流程及数据引用，但有两个解释性根因：

1. `/purpose` 写成“把文档翻译为 JSON”，不是库存条件读取的业务目的，也缺少计划库存/非实时健康的限制。
2. `/nodes/3/explanation` 把 `needs_l1` 描述为 blocked，混淆推理交接与安全拒绝。

32 项逐项审查中，24 项 supported、5 项 contradicted、3 项 insufficient_evidence；8 项发现有交叉覆盖，**不是 8 个独立根因，也不是 75% 转译准确率**。三个原始回答都未获执行准入，原始结果保持 blocked。

### 修订轨迹和实际运行

另外保存 `text-revision.json`：父提案/源摘要、作者、两个旧值/新值及源引文。当前只允许 `/purpose` 和终态 `/explanation` 的文本编辑；不允许改变图、参数、分支或消除问题。源引文必须存在且唯一；原文件不变。语义变更若涉及步骤/参数，应建立新提案，而不是扩大此文字修补入口。

新审查摘要绑定父提案和修订，旧审查不可复用。当前助手逐项重新审查后，32 项 supported；**修订者和审查者是同一个可见项目上下文的助手**，不是独立人工，也不是自动语义证明。该比例不能变成置信度或生产成功概率。

宿主显式授权后，修订版本得到：

- campus-sw1：读取→site=campus→引用第一次返回的设备 ID 再读→read_path_completed。
- idc-sw1：读取→site≠campus→needs_l1，没有调用模型继续推理。
- 共 **3 次实际本地文件读取，0 次写入**。模型原始图与参数未改变，只修订两个说明字段；不是手工重新构建整个 L0。

结果见[可重算摘要](benchmarks/flow-forward-9b-summary.json)，其中保留三次请求/响应、原始/修订审查和运行记录的摘要。完整原始材料在本地 `artifacts/translator-v2/flow-forward-9b-20260907*`，不保证仅克隆源码即可取得。

### 接口与复现

[转译与审查代码](../evaluation/flow_translation.py)接受 `FlowSources`，其中源文本和宿主合同独立于模型提案。CLI 的 author 命令当前使用本地样例；其他源可通过 Python API 提供，但未完成公共工具环境自动接入。

```sh
# 新目录才会调用模型，不覆盖旧响应；每个 author 命令至多一次生成请求
.venv/bin/python -m evaluation.flow_translation author /tmp/new-flow-9b-run

# 查看结构失败定位，不修补提案
.venv/bin/python -m evaluation.flow_translation diagnose /tmp/new-flow-9b-run \
  --output /tmp/flow-diagnostics.json

# 对可结构化提案执行人工/AI 提交的审查；命令不自动生成 supported 判断
.venv/bin/python -m evaluation.flow_translation assess /tmp/new-flow-9b-run \
  /path/to/review.json --output /tmp/flow-review-report.json
```

`revision-packet ROOT REVISION --output PATH` 产生全量重新审查包；`assess` 和 `run` 可传 `--revision PATH`。`run` 还必须传 `--device-id campus-sw1|idc-sw1 --allow-local-read --output PATH`，它从原模型响应重建并核对提案/审查，拒绝未授权、审查阻断、源环境变化和范围外设备。所有输出路径必须不存在。

### 下一步与证据边界

- C3 原型的源→图→审查→辅助运行链已具备，但**9B 无辅助稳定性尚未证明**；不要因辅助结果可运行而恢复大规模 Runtime A/B。
- 下一步先冻结这一版通用协议，扩展已知开发流程的顺序、条件、只读/写候选及 unsupported 样本。分别统计首次结果、结构错误、语义错误、辅助成本和 Skill/步骤覆盖。
- 原 12 个公开 Skill 的完整转译仍未完成；它们缺的宿主合同/依赖不能从此次单工具例子推断为已补齐。
- 本轮未用真实网络设备、没有自动 L1 回入、未验证模型生成写流程与 C2 的全链联动。源数据实时性、TOCTOU、并行/循环/多写边界继续见[最小业务流程](L0-BUSINESS-FLOW.md)。

## English

September 7, 2026: one known local flow now traverses 9B forward generation, structural qualification, bidirectional semantic review, source-backed text revision, fresh review and authorized local reads. This is an **assisted development chain**, not public-Skill generalization, reliable unaided translation or a DSH agent loop.

The model received source prose, host schemas/tools and general runtime rules, not the hand-authored answer graph. It chose steps, conditions, links, argument/output references and terminals. The indexed protocol only alpha-renames array positions to compiler-owned IDs; it does not reconnect edges, change values or repair semantics. Host read contracts remain supplied dependencies, not model-generated provider implementations.

Three qwen3.5:9b/no-think development calls took 69.61, 45.54 and 33.66 seconds (148.80 total; 2,535 input and 1,348 output tokens). Named-node v1 confused identities and terminal roles and generated a cycle; clearer protocol v2 still produced dangling identities/questions. Indexed v1 passed structural checks but failed semantic review. These are different protocols on one known source, not independent trials or comparable p50/p95 performance evidence.

The third answer's purpose described the translator's assignment rather than the inventory business purpose/limitations. Its needs_l1 explanation incorrectly called the path blocked. Review produced 24 supported, five contradicted and three insufficient-evidence claims across 32 checks: eight findings overlap two explanation-level root causes, not eight independent errors or 75% accuracy. All raw attempts remained unadmitted and unexecuted.

A parent/source-bound sidecar changed only purpose and one terminal explanation using exact source quotes. Graph, parameters and questions cannot change through this text-revision interface. Fresh full review was required; the current visible-context assistant was both editor and reviewer, not independent Gold. The assisted child received 32 supported judgments, then explicit host authorization ran three actual local file reads: campus completed the two-read path, IDC stopped at needs_l1. Zero writes or subsequent LLM calls occurred.

The [summary](benchmarks/flow-forward-9b-summary.json) binds original requests/responses, reviews, revision and execution artifacts. Source-only clones may not contain the local raw experiment directories. The [API/CLI](../evaluation/flow_translation.py) accepts independent sources/host contracts; its default author command uses the local fixture. No arbitrary provider auto-registration is claimed. Commands above create new outputs, never overwrite prior evidence; assess requires a supplied review, not auto-approved judgments. The local run path reconstructs the original response, applies an optional bound revision, rechecks review and host scope, and authorizes neither writes nor an agent loop.

Next: freeze the generic protocol for a heterogeneous known-development batch, keep raw and assisted outcomes/costs separate, and preserve unsupported/environment-missing cases. The twelve public Skills remain untested for whole-flow translation. Do not reopen large Runtime A/B or infer autonomous/general/production reliability from this assisted single-flow result.
