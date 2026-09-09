# 渐进源文读取与结构化候选 / Progressive Structured Authoring

## 中文

更新：2026-09-09，C3q。**9B 已接入当前结构化 Tree 的构造入口，但首个公开源开发请求停在上下文预算，没有产生候选。** 不能据此计算转译准确率，更不是整 Skill 编译成功。

上一阶段 C3i–C3p 已本地提交到 dev：`f0499ec`，未推送、未合入 master。本阶段是在其上新增的开发工作。研究主线仍是源语义与转译泛化，生产工程和大规模 Runtime A/B 继续冻结。

### 做了什么

新增 [structured_authoring](../evaluation/structured_authoring.py)，复用原检查点、模型传输、源包和结构化编译器，不新增执行器：

```text
完整惰性源包 + 明确任务 + 原宿主 catalog / read contract
  → 冻结输入、源码、环境、9B 模型摘要与预算
  → 根文 + 全量分页索引 → 9B 请求引用页（只读已保存文本）
  → 9B 提出带原句的 Tree + remaining / unresolved
  → 校验原句、定位偏移、校验 Schema / 支配关系 / 合同
  → 未激活编译片段 + 待审材料，或者保留首次失败
```

构造输入没有任务档案中的义务列表、问题判定、示例答案、query builder 或 Oracle。读取合同是显式宿主声明，不是模型推断的权限；本次仅有一个进程内合成 Function 工具，构造阶段不调用它。9B 请求任意包内页时，脚本文本也只作为惰性材料；不能执行脚本、联网取引用或注册工具。

这不是自动激活流程。精确引文只证明原句存在，不证明它支持模型结论。`remaining` 是模型自报，不是穷尽审查；遗漏的义务仍可能存在。`needs_l1`、`unsupported` 或纯停止图即使结构合法，也不能计为可执行任务成功。语义评审/准入仍待完成，默认 DSH 路由未改变。

### 首次真实 9B 结果

固定已有 Netdata 源快照（17 文件、202,930 字符、28 页），使用原单设备 24 小时 security 类普通 trap 开发请求。空对象 `inputSchema` 表示固定请求提案，不是可重用的参数化整 Skill。任务里的 Cloud 路径与隔离宿主之间的适配差异保留，不能声称原厂 Cloud/wrapper 互操作。

| 次序 | 模型/系统实际行为 | 结果 |
|---|---|---|
| 1 | 初始提交根文 `p011` 和完整索引；9B 请求 `p016` 单设备查询指南 | 29.764 秒，6,188 输入 / 198 输出 token |
| 2 | 提交根文和完整指南；9B 请求 `p006` Cloud Function 协议 | 14.662 秒，7,682 输入 / 61 输出 token |
| 3，未调用模型 | 累计原页后的请求需 45,057 字节，超过冻结的 36,000 字节上限 | `context_budget_stopped_no_truncation`；未静默截断，也未把页视为已读 |

实际 **2 次模型调用、合计 44.426 秒、13,870 输入 / 259 输出 token**。28 页中只有 2 页被实际提交；不是语义覆盖率，也没有服务器 tokenizer 级“完整阅读”证明。候选为 0；语义准确率为空；Runtime、Provider 和源脚本执行均为 0。两次样本的请求耗时不代表模型 p95 或 Runtime 性能。

**字节上限是实验资源策略，不是模型上下文容量。** 下一轮待发消息正文共 37,130 UTF-8 字节，序列化 wire 连同格式声明等共 45,057 字节；实际 token 数未知。不能拿它与 49,152-token 的配置直接比较，也没有证明模型物理容量不足。

第一轮请求理由把 `security` 类别称为 severity，原文明确区分 `TRAP_CATEGORY` 与 `TRAP_SEVERITY`。这是检索理由的术语错误；尚无查询参数，不能宣称参数已错或已正确。第二轮主动追踪协议依赖是有用的行为，但“不断累计整页”的输入策略先触及预算，因而当前没有证据将停止归因于 9B 能力不足或 L0 Schema 表达失败。

结果定位：

- [正式首次报告](../artifacts/translator-v2/progressive-authoring-20260909/report/report.json)；[第一轮原始选择](../artifacts/translator-v2/progressive-authoring-20260909/netdata-9b-first/round-000/choice.json)；[第二轮原始选择](../artifacts/translator-v2/progressive-authoring-20260909/netdata-9b-first/round-001/choice.json)。
- [开发诊断](../artifacts/translator-v2/progressive-authoring-20260909/diagnostics-v2/diagnostic.json)绑定 14 份输入/检查点/报告/源码快照。由同一开发助手审阅，不是独立 Gold。诊断初版误用同名摘要字段，修正版分开 source/self digest；两版保留，原模型记录未变。
- 原模型报告摘要：`sha256:06bbd909e2d084c2bcf76ebb70cdf6941456cf90013952be1b10c5b6725489c6`。模型为 `qwen3.5:9b`，artifact `sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。

这些本地 `artifacts` 不随 Git 分发；摘要和复现方法在此保留。不能把缺少本地证据链接误认为远端也包含源快照。

### 使用与复现

输入 JSON 严格只接受 `bundle`、`task`、`taskOrigin`、`inputSchema`、`catalog`、`reads` 六项。`taskOrigin` 当前固定为 `developer_authored_evaluation_request`；每个 catalog 工具必须有原声明完全一致的结构化 read contract。本研究入口不接受 Effect targets。

```bash
# 使用者自己的新输入；ROOT 必须不存在。freeze 只检查本地模型，不做生成。
.venv/bin/python -m evaluation.structured_authoring freeze ROOT --inputs INPUT.json
# 明确授权最多四次新调用；每次来源请求也计费。不要在旧目录调参重跑。
.venv/bin/python -m evaluation.structured_authoring run ROOT --max-new-calls 4 --report-dir NEW_REPORT
# 默认预算为零：已封存轮次离线回放，无新调用。
.venv/bin/python -m evaluation.structured_authoring run ROOT --max-new-calls 0
```

当前协议固定无思考输出、温度 0、seed 20260909、`num_ctx=49152`、`num_predict=4096`；序列化请求另受 36,000 字节硬上限。四次是调用预算，不是四次修复机会。来源请求可以推进下一轮；首次候选无效、传输错误或无进展请求均停止。残缺检查点不自动重试，已有文件不覆盖，源码/模型/输入漂移拒绝继续。

历史源码回放：在新临时目录导出 Git `f0499ec`，叠加本阶段 [source-snapshot.tar.gz](../artifacts/translator-v2/progressive-authoring-20260909/source-snapshot.tar.gz)，使用同一依赖环境，对现有检查点执行 `--max-new-calls 0`。已实际完成隔离重建，报告逐字节一致。

验证：新增 29 项机械测试，跨源包/绑定/Tree/网关/检查点 **211 项定向测试通过**；全量 **2070 passed + 81 subtests passed（161.34 秒）**。两份新增 Python 文件 Ruff、279 个文档本地链接、14 份绑定文件、隔离回放和 diff 检查通过。详见[项目进展](PROJECT-STATUS.md)。这些测试不是 211 个公开 Skill，也不是语义准确率。

### 下一步改进输入组织

先校准请求字节、实际输入 token、输出预留和模型上下文的计量关系，基于资源约束冻结新协议，不按本例“能否通过”择优改阈值。首轮保留，不放宽预算后覆盖。对于完整长源包，下一版再验证**预算内的源义务提取 + 显式依赖账本 + 候选构造时回填原句**：

1. 分文件/页提取与任务相关的义务、前置条件和引用依赖；保留未读、未解、矛盾和模型提取失败。
2. 依赖账本记录精确来源和处理状态；模型摘要只是导航，不能代替原句或自动认证语义。
3. 对明确的局部候选，将必需原句/上下文重新送入模型；跨页依赖未闭合时，只能保留未决职责，不能自动激活。
4. 用不止 Netdata 的长源文/多引用开发请求验证该通用机制，之后再冻结新小批；同源修订单列，不当未见泛化。

这一步解决的是长 Skill 的输入与证据组织，不扩大执行权限，也不把未支持的循环、动态成员检查、聚合、分页或隐私判定伪装成已实现。

## English

C3q connects local `qwen3.5:9b` to the current structured Tree authoring path. The first known-public-source development request stopped at the frozen context budget **before producing a candidate**. Semantic accuracy, whole-Skill translation and Runtime performance remain unmeasured. C3i–C3p were locally committed on dev as `f0499ec`; no push or master merge was performed.

The new module freezes original inert sources, the explicit task/schema, exact host/read declarations, implementation/environment/model digests and call limits. The model sees the root plus a complete page index and may request saved text pages. Candidate statements cite exact unique quotes, deterministically lowered to original offsets before the existing compiler checks types, scope and contracts. No source code, provider or candidate is executed. Reviewer answers and query builders are not constructor inputs. Citation validity and compilation do not establish entailment, complete obligation coverage or activation authority.

The retained Netdata bundle contains 17 files/202,930 characters/28 pages. Call one requested the linked device recipe (29.764 seconds; 6,188 input/198 output tokens); call two requested the Cloud Function guide (14.662 seconds; 7,682/61 tokens). Adding that page would require 45,057 serialized bytes, exceeding the frozen 36,000-byte limit. It was not submitted or silently truncated. Total: two calls, 44.426 seconds, 13,870 input/259 output tokens; two pages submitted, zero candidates/provider/script executions. These are request timings from two samples, not latency percentiles or Runtime measurements.

The first retrieval rationale confused the `security` category with severity; no executable query was generated, so parameter correctness remains unscored. The request's Cloud semantics and the synthetic in-process host remain distinct. This first outcome identifies the experiment's accumulated-page byte budget as the immediate blocker, not a measured model or L0 schema translation failure. The pending message text would contain 37,130 UTF-8 bytes; neither that nor serialized wire size is a measured token count or proof of exhausting the configured 49,152-token context. The fixed task uses an empty object input schema, not a reusable whole-Skill contract.

See the [first report](../artifacts/translator-v2/progressive-authoring-20260909/report/report.json) and [development diagnosis](../artifacts/translator-v2/progressive-authoring-20260909/diagnostics-v2/diagnostic.json). The latter binds fourteen files and is same-assistant review, not independent Gold. Its initial source/self-digest naming mistake was corrected in a separate retained version without changing model checkpoints. Local artifacts are not included in Git distribution.

Use the CLI above with fresh output directories and an explicit zero-to-four new-call budget. Defaults mean offline replay. Failed, partial, drifted or non-progressing checkpoints never become automatic retries. A source-only reconstruction from `f0499ec` plus the archived overlay produced a byte-identical zero-call replay. Twenty-nine new mechanical tests and 211 targeted cross-component tests pass; the full suite passes **2070 tests plus 81 subtests in 161.34 seconds**. Both added Python files pass Ruff; 279 local links, fourteen evidence files and diff checks pass. See [project status](PROJECT-STATUS.md); these are not translated-Skill scores.

First calibrate byte/token/output/context budgets independently of whether this example passes. Then validate bounded source-obligation extraction, explicit unresolved dependencies and original-context rehydration across multiple long/reference-heavy Skills. Summaries remain navigation, not authoritative evidence. Retain this first failure and label same-source revisions separately. Large Runtime evaluation, automatic activation and production engineering remain gated.
