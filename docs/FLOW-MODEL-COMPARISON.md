# 转译强模型对照与决策 / Translation Model Comparison and Decision

## 中文

> 2026-09-08 用户后续选择：**继续使用 9B，GPT 暂缓，不再等待 API Key。** 下面保留原两臂实验设计与接入说明，但当前只运行已冻结的 Ollama 臂；不能声称得到跨模型对照。两份父报告仍会显示 GPT `not_run`，这不是当前 9B 工作的阻塞。实际结果及后续计划以 [PROJECT-STATUS](PROJECT-STATUS.md) 最新条目为准。

### 决策：先建立强模型基线，再决定如何重构

2026-09-08。当前问题是转译语义质量，不是生产工程欠缺。此前只围绕 9B 反复修订协议，无法分开模型能力、提示任务负担、Schema 表达和宿主缺能力。因此将以下顺序置于此前“立即重构原文义务提取”的计划之前：

| 方案 | 收益与风险 | 决定 |
|---|---|---|
| 继续只优化 9B | 成本可控，但容易围绕已知四例调协议，仍无法识别模型因素 | 暂缓 |
| 立即重构为先原文义务、后宿主绑定 | 有明确诊断依据，但会同时改变设计与模型输入，缺少强模型参照 | 保留为后续候选，不立即改冻结协议 |
| 冻结协议，建立 GPT/9B 对照，再按失败类型修正 | 先回答较强模型在当前设计下是否可用；避免无证据地扩充类型 | **采用** |

参考模型选 `gpt-5.5-2026-04-23`、`reasoning=high`，不是宣称它最强或已经验证适用。选择可固定的专业推理模型快照，避免滚动别名成为实验变量；账号是否具备权限仍需实际接入确认。9B 保留为效率对照，不再作为唯一质量参照。转译器与运行时编排模型可以不同；换模型不授予 Runtime 权限。

这一顺序参考官方“先满足准确性，再优化时延/成本”的[模型选择原则](https://developers.openai.com/api/docs/guides/model-selection)；具体模型快照和推理设置见[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5)。官方通用能力不是本项目准确率。

### 一个必须显式处理的控制变量

原流程 Schema 存在可选属性和开放参数对象，不等价于 OpenAI 严格结构化输出要求的全部属性 required、封闭对象子集。不能只为 GPT 改 Schema、补缺省或删约束，也不能把接口拒绝当成语义失败。参见[结构化输出边界](https://developers.openai.com/api/docs/guides/structured-outputs)。

本次采用新协议 `flow-model-common-json/v1`：

- 两个模型收到相同的原任务消息、来源、宿主信息，以及完整原始输出 Schema；不输入旧答案、失败诊断或参考映射。
- 两边统一使用 JSON 模式，不在解码器中施加各自不同的严格 Schema 子集；返回后仍由**未修改的原 Schema、编译器与源审查门禁**验证。
- GPT 使用 Responses API、高推理；9B 使用本地 Ollama、`think=false`、温度 0、记录 seed，16K 上下文。两边最大输出预算均 8192，但 tokenizer 和推理 token 语义不同，不声称算力/采样完全相同。
- 因此这是**模型与服务配置对照**，不是仅模型权重不同的因果实验。解码模式也不同于历史 9B 批次，必须重新运行本轮 9B，旧成绩不能充当配对 Control。
- API 无工具定义、无会话历史、`store=false`，固定官方 HTTPS 端点，禁用代理环境继承与重定向；不读取 Codex 登录凭据、不执行源脚本、不调用业务工具。

### 两个实验回答不同问题

1. **mapping-only：固定树后的映射诊断。** 两边接收相同历史 9B 候选树，父原始检查点先重放校验，原始摘要与树均绑定到新实验。它能定位第二阶段，不证明第一阶段正确，也不是端到端 GPT 转译成绩。
2. **end-to-end：从源文重新生成。** 两边各自从相同源文生成树，再根据自己的真实节点生成映射；不传入历史树。第一步失败就不运行第二步，不修图或补答案。

首轮只用当前 **4 个已知开发流程 / 1 个工具 / 0 公开 Skill / 每例 1 次**，用于诊断方向。两种实验分别每臂最多 4、8 次生成调用，合计每臂 12 次。先保留首次失败，再决定是否另冻三次重复批，不能择优挑一次。固定树与完整链成绩不合并成一个分母。

### 评判方法与下一步分岔

必须分别检查格式合规、编译合规、原文完整保真、真实缺能力、错误接受、错误反驳、可用范围与成本。完整源审阅需要明确对比动作、顺序、参数、分支条件/极性、前置、否定、解释和权限边界。正确表示缺审批或缺脚本并停止，可能语义忠实，但不是业务执行成功。

现有报告保留源审阅入口，并额外绑定实验、模型臂、重复编号和案例；审阅声明仍受原 `ReadL05Review` 的完整源/引文门禁约束。没有独立源义务标签和适用范围分母前，`semanticAccuracy`、`generalizationAccuracy`、`unsafeAcceptRate` 保持 null；不是 0%，更不是编译通过率。当前助手已经看过开发样本，其审阅只能标为开发/AI 辅助，不能充当盲测 Gold。编译失败项也不能用局部事后观察冒充完整审阅。

- GPT 明显减少关键语义错误：将强模型作为转译参考，先完成已知小批质量闭环，再在未参与调试的 Skill 上验证；9B 的压缩/降本后置。
- 两边重复出现同类错误：优先检查任务组织、原文义务/宿主规则分离、元数据干扰和判定定义；另冻新版协议，不改本批答案。
- 两边都能正确表示流程、但因真实工具缺失阻断：明确支持边界，不能因拒绝多就推断模型差，也不补造工具。
- 格式好、语义仍差：继续语义提取和验证；不扩展 Runtime 或靠增加 Schema 类型宣称解决。

四例只能排查机制，不能据此确认泛化或切换产品默认模型。后续仍按异质开发批→未参与调试的新集合→原泛化门禁推进。C4–C6 和大规模 Runtime 评测不因本入口完成而解锁。

### 本地使用

实现：[对照入口](../evaluation/flow_model_comparison.py)、[工具隔离传输](../evaluation/flow_model_transport.py)、[离线回归](../tests/test_flow_model_comparison.py)。无需 Key 即可冻结、查看报告和运行离线测试；只有 `run --arm openai` 才需要运行进程中的 `OPENAI_API_KEY`。不要把 Key 写入 Git 或发到聊天中。

项目根目录执行；输出目录必须不存在，报告拒绝覆盖：

```bash
# 1. 两个独立批次；freeze 不调用任何模型
.venv/bin/python -m evaluation.flow_model_comparison freeze artifacts/translator-v2/flow-node-evidence-4-20260908/manifest.json --mode mapping-only --parent-run artifacts/translator-v2/flow-node-evidence-4-20260908 --output artifacts/translator-v2/model-mapping-4-20260908
.venv/bin/python -m evaluation.flow_model_comparison freeze artifacts/translator-v2/flow-node-evidence-4-20260908/manifest.json --mode end-to-end --output artifacts/translator-v2/model-end-to-end-4-20260908

# 2. 在已安全配置 OPENAI_API_KEY 的进程中先运行强模型诊断
.venv/bin/python -m evaluation.flow_model_comparison run artifacts/translator-v2/model-mapping-4-20260908 --arm openai --max-calls 4
.venv/bin/python -m evaluation.flow_model_comparison run artifacts/translator-v2/model-end-to-end-4-20260908 --arm openai --max-calls 8

# 3. 本轮 9B 对照（保持服务配置，不换模型或混用历史结果）
.venv/bin/python -m evaluation.flow_model_comparison run artifacts/translator-v2/model-mapping-4-20260908 --arm ollama --max-calls 4
.venv/bin/python -m evaluation.flow_model_comparison run artifacts/translator-v2/model-end-to-end-4-20260908 --arm ollama --max-calls 8

# 4. 可在任意进度生成新文件；--reviews 可选，缺失审阅不伪造结论
.venv/bin/python -m evaluation.flow_model_comparison report artifacts/translator-v2/model-end-to-end-4-20260908 --output artifacts/translator-v2/model-end-to-end-4-20260908-report.json
```

每阶段保存请求、原始响应、派生结果和收据。完整检查点只重放，已保存响应但派生文件不完整可在离线重建后继续下一阶段；**请求已记录但响应丢失意味着远端可能已经收费，禁止自动重发**。401/429、模型身份不符和传输错误会停止当前运行，不自动换模型或重试。单次运行最多 24 个新生成请求；请求数量限制不是金额承诺。GPT 返回版本与固定快照不符明确阻断，Ollama 制品摘要变化也阻断。

审阅目录格式为 `<reviews>/<arm>/<repeat>/<case>.json`，内容为 `{"context":{"manifestDigest":"…","arm":"openai","repeat":0,"case":"…"},"review": <ReadL05Review>}`。`review` 必须绑定该输出的 `review-input.json`，不能从其他模型输出挪用审阅。

### 初始准备状态（历史，已由后续 9B 结果更新）

入口、无自动执行边界、检查点与离线回归已在初始准备阶段实现；当时未配置 `OPENAI_API_KEY`，尚无新生成调用。随后用户决定只继续 9B，两个 Ollama 臂已完成，见[实际结果与后续方案](FLOW-9B-COMMON-JSON.md)。GPT 仍未运行；此前真实 9B 结果继续保存在[节点证据报告](FLOW-NODE-EVIDENCE.md)，不改写。上文强模型优先顺序是历史设计，不覆盖最新用户选择。

## English

> Subsequent user decision, 2026-09-08: **continue with 9B and defer GPT; API credentials are no longer a prerequisite.** The original two-arm design below is retained, but only the frozen Ollama arm is active. Parent reports retain GPT as not_run; this is not a blocker for the current 9B work and no cross-model result can be claimed. See the latest [Project Status](PROJECT-STATUS.md).

### Decision and experimental boundary

Establish a stronger-model reference **before** another 9B-specific protocol redesign. Continued small-model tuning cannot distinguish capacity limits from task/schema defects. Source-first duty extraction remains a candidate correction, but the frozen translator, compiler, evidence gates and historical results stay unchanged.

The reference is `gpt-5.5-2026-04-23`, high reasoning, versus local `qwen3.5:9b`, no thinking, temperature zero, recorded seeds and 16K context. Availability is not assumed. The dated snapshot and accuracy-first decision follow the official sources linked above; no public benchmark establishes translation accuracy here. Translation and runtime-orchestration models need not match.

The original schemas contain optional/open objects that cannot be submitted unchanged to the strict OpenAI schema subset. New protocol `flow-model-common-json/v1` gives both arms identical messages and the complete original schema in the prompt, uses JSON mode in both transports, then applies unchanged local schema/compiler checks. No schema weakening, default filling or answer repair. Provider, decoder, reasoning and tokenizer differences remain: this compares model/service configurations, not isolated weights. Historical constrained-decoder 9B scores are not the new control arm. Both have an 8192 output-token ceiling, with different reasoning-token semantics.

### Two separately scored probes

- **mapping-only:** the same mechanically qualified, unreviewed historical tree goes to both arms. Parent checkpoints are replayed and bound. This measures conditional mapping, not full translation.
- **end-to-end:** each arm freshly generates its tree and mapping from the same source; no historical tree is supplied. A failed first phase skips mapping.

Initial scope: four known development flows, one tool, zero public Skills, one attempt per phase, one repetition. Mapping-only allows four calls per arm, end-to-end eight; twelve per arm across both probes. Do not pool their denominators or select the best retry. Repetitions, heterogeneous development sources and unseen cohorts follow after diagnostic review; generalization and large Runtime gates remain unchanged.

Report structural qualification, source fidelity, real capability gaps, false acceptance/rejection, coverage and cost separately. A faithful missing-capability stop is not business completion. Source review requires the unchanged complete-source evidence contract plus experiment/arm/repetition/case binding. Reviews are not automatically independent Gold; this development assistant knows the examples. Semantic/generalization accuracy and unsafe-accept rates remain null without independent obligation labels and valid denominators. No model winner is inferred automatically from compilation counts.

If GPT reduces semantic defects, establish quality with the stronger translator before reducing cost. If both repeat the same defects, revise source/host separation, metadata or task organization in a new frozen version. If source fidelity is good but host capabilities are missing, report the boundary rather than inventing tools. Four examples cannot establish broad generalization or authorize a product default-model switch.

### Operation and status

The commands above freeze two independent batches without network calls, explicitly run one arm under a call cap, and create non-overwriting reports. Only live OpenAI generation needs `OPENAI_API_KEY` in that process. No Codex authentication is reused, credentials are not stored, OpenAI requests use a fixed HTTPS endpoint with environment proxies and redirects disabled, `store=false`, no tools and no conversation history. Raw sources/scripts remain inert.

Completed checkpoints replay without calls; saved responses permit rebuilding missing derived files. An attempted request without a saved response is **uncertain**, possibly billed, and is never automatically retried. Authentication/rate-limit/transport/model-identity failures stop the run. Raw requests/responses, costs, model identity, derived files and receipts remain inspectable. Review envelopes bind the manifest, arm, repeat and case as described above.

At initial preparation, implementation and offline tests were complete but no new live generation had occurred. The user subsequently chose 9B-only work: both Ollama arms are now complete; see [actual results and next design](FLOW-9B-COMMON-JSON.md). GPT remains unrun, not a prerequisite. The stronger-model-first ordering above is historical and does not override that choice. Prior measured failures remain intact; no cross-model accuracy is available.
