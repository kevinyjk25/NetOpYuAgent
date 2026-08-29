# P1.9 L1 决策面 / L1 Decision Plane

## 中文

### 1. 目标与边界

P1.8 证明了“候选专属 Schema + 请求证据 grounding + 确定性编译 + 单调 Guard”可以显著收窄模型输出，但 C3 是隔离评测路径。P1.9 把同一原则迁入正式 Harness 入口，形成独立的 L1 Decision Plane：它在 DSH/Hermes 与 L0 Runtime 之间产生一个严格、可观测、无执行权的意图候选。

```text
直接用户请求
    ↓
候选检索（最多 12 个受信 Skill/Tool）
    ↓
候选专属 Tool Schema（模型只选择候选并提取显式值）
    ↓
Grounding + Guard + 确定性编译
    ↓
L1DecisionEnvelope（proposal_only）
    ↓
Domain L1 Skill / L0 Runtime
```

Decision Plane 不调用业务 Tool，不创建或批准 Runtime plan，不持有 Provider/设备凭据，不判定执行成功，也不执行回滚。写请求即使选择正确，仍必须经过 L0 的严格参数与来源校验、计划、风险、人工审批、执行前重读、单次效果、独立 Observer、补偿和审计。

### 2. 当前阶段

P1.9-B1 已实现 **DSH/Hermes 共用 shadow 与本地证据框架**：

- 默认 `off`，只有显式设置 `NETOPYU_L1_DECISION_MODE=shadow` 才运行；
- DSH 从已接受 `agent/pre-step` 的 direct-user source 取请求；Hermes 使用官方 `pre_llm_call` 的 `user_message`，不接受 Tool/Skill/系统输出冒充用户请求；
- 使用当前 Harness 精确暴露的 Tool declarations 和正式 Skill manifest 构建 Catalog；
- 最多检索 12 个候选，并应用受审 Skill-over-primitive 去歧义规则；
- 模型只能调用一个 `select_candidate_NN`、`refuse_l1_request` 或 `reject_l1_out_of_scope`；
- Schema 外字段失败，grounding 删除无直接请求证据的值，编译器派生 action、missing fields 和 workflow；
- 最多一次隐藏协议修复；模型连接失败或协议失败不会阻断 shadow 下的原 DSH 行为；
- Decision 与实际首次 Skill/Tool 路由分别记录，二者通过 `decision_id + session_id` 一次绑定；
- DSH 新一轮会关闭被覆盖的 pending Decision；Hermes 还通过 `post_llm_call/on_session_end` 记录 no-route/session-end，关闭后的 Decision 不可绑定后续 Tool；
- SQLite 不保存用户原文、模型正文或实际参数值，只保存 prompt/argument digest、字段名、决定、token usage 完整性和一致性结果；旧 schema 打开时迁移并清除参数原值；
- `data/l1_catalog_baseline.json` 固定 LAN/DC/WAN 当前候选与 Tool/Skill 摘要，Catalog/Schema/Skill 漂移进入退休门禁；
- `l1_runtime.holdout` 已提供 50+、10+ 类、三 profile、中英覆盖的 prompt-free seal manifest 和双 reviewer 一致性合同。

P1.9-B2 的两级本地执行器已经完成：Worker 级在 Catalog baseline 无漂移且双 reviewer 完全一致后，使用同一模型/策略/Tool declarations 分别生成 DSH/Hermes 身份的独立 Decision，并计算协议、选择、参数、追问字段、workflow、安全拒绝、重复稳定性、语义 parity、token 与 p50/p95；Adapter 级实际执行 DSH JavaScript `agent/pre-step` 与 Hermes Python `pre_llm_call`，通过临时 owner-only Worker 比较 Prompt/Catalog/Candidate/Policy/Decision digest。报告只含聚合指标和 case-id 摘要。

P1.9-C0 的**未启用绑定内核**也已完成：PreparedPlan schema v10 可选地把 Decision/evidence digest、实际 Harness route、请求参数与编译后参数摘要、L0 Skill/contract 和有效期写入同一 plan hash；Journal 对 `decision_id` 建立唯一约束。错模式、会话、Harness、profile、route、参数、候选或摘要，以及重复绑定和持久化篡改都会失败关闭。schema v9/v8 继续只读兼容。

P1.9-C1 的**未启用安全准备层**也已完成：`canary_policy` 是无副作用单调函数，只能保持 Harness 原 route 或收窄/阻断；写路径的协议/摘要/上下文异常失败关闭，读路径异常只观察。`canary_readiness` 严格读取 Worker、Adapter、产品/部署和运维演练四类外部证据，校验摘要、有效期、角色分离、64/64 Core 控制、至少三个实现版本的稳定/改善 trend 与 p50/p95 阈值、Decision-plan binding、停用/回退/告警/no-effect replay 及流量/时长上限，最强结果仅为 `ready_for_review`。完整手册见 [P1.9-C1 Canary 准备与回退](p19-canary-runbook.md)。

尚未完成：真实私有未见集、两名独立人员的实际标注与消歧、真实 DSH Web/Hermes CLI/UI 与部署身份资格证据、组织签名/不可抵赖、canary 激活/enforced 和生产 SLO。因此 P1.9 当前仍是 🟡；DSH/Hermes 配置仍只接受 `off/shadow`，C0/C1 代码不能在证据不足时改变 Harness 行为。

### 3. 组件

| 路径 | 责任 |
|---|---|
| `l1_runtime/contracts.py` | extra-forbid、frozen、有界的 Decision/Evidence/Envelope；`authority` 固定为 `proposal_only` |
| `l1_runtime/catalog.py` | 从正式 Tool/Skill 表面构建能力卡、BM25 检索、候选上限和去歧义 |
| `l1_runtime/policies/*.yaml` | 独立生产 Catalog、Guard 和 grounding 策略；不导入评测 Oracle |
| `l1_runtime/client.py` | OpenAI-compatible 单 Tool 协议；动态候选 Schema；temperature 0 |
| `l1_runtime/service.py` | Guard、模型尝试、grounding、确定性编译、证据绑定和指标 |
| `l1_runtime/store.py` | immutable Decision、一次性终态生命周期与首次路由 Observation；不保存 Prompt/参数原值 |
| `l1_runtime/catalog_gate.py` | 可移植的三 profile Catalog/Tool/Skill 摘要基线与漂移报告 |
| `l1_runtime/holdout.py` | 私有 50+ 未见集 seal 和双 reviewer 一致性检查；manifest 不含 Prompt/标签 |
| `l1_runtime/qualification.py` | B2 私有 Oracle、同模型双身份 parity、重复稳定性和隐私化聚合报告 |
| `l1_runtime/adapter_qualification.py` | B2 DSH/Hermes 生产 Hook → 临时 Worker 的摘要 parity 执行器 |
| `l1_runtime/canary_policy.py` | C1 只保持或收窄原 route 的纯策略；无 Runtime/Provider 副作用或授权 |
| `l1_runtime/canary_readiness.py` | C1 四类外部证据、摘要/时效/演练交叉门禁和隐私化报告 |
| `network_runtime/proposal_binding.py` | C0 无 L1 反向依赖的严格 Decision→Plan 投影、摘要链与路由/参数/L0 校验 |
| `network_runtime/contracts.py` | schema v10 将可选 C0 binding 纳入 plan hash；v9/v8 只读兼容 |
| `network_runtime/journal.py` | 每个 `decision_id` 最多绑定一个不可变计划，并把 binding digest 写入 hash-chain event |
| `dsh-plugin-netopyu/src/index.js` | DSH direct-user 捕获、shadow 调用、实际路由关联；失败时保持原 step |
| `hermes_adapter/plugin.py` | Hermes 官方 LLM/Tool/session hook 上的同一 Worker shadow 与轮次关闭 |
| `dsh_adapter/worker.py` | Harness 无关的 Unix Socket 命令入口 |

`evaluation/` 仍是离线资格评测，不能进入生产调用链；`l1_runtime/` 是正式 Decision Plane，不能依赖场景标签或期望答案。

### 4. 本地启用

先确认所选本地 Ollama 模型已安装。Shadow 会额外调用一次选择模型，因而会增加当前请求时延；它不会替换 DSH 主模型，也不会执行 Decision。

```bash
NETOPYU_L1_DECISION_MODE=shadow \
NETOPYU_L1_DECISION_MODEL=qwen3.6:27b \
scripts/netopyu-dsh restart
```

默认 endpoint 为 `http://127.0.0.1:11434/v1`，默认证据库为 DSH runtime 下的 `data/l1_decisions.sqlite`。可显式覆盖：

```bash
NETOPYU_L1_DECISION_BASE_URL=http://127.0.0.1:11434/v1
NETOPYU_L1_DECISION_STORE=/absolute/path/l1_decisions.sqlite
NETOPYU_L1_DECISION_REPAIR_LIMIT=0
```

非 loopback 模型 endpoint 默认拒绝；确需远端企业 endpoint 时必须显式设置
`NETOPYU_L1_DECISION_ALLOW_REMOTE=1`，并由部署侧另行完成 egress、TLS、凭据和数据治理。

查看最近 Decision 和聚合指标：

```bash
scripts/netopyu-dsh l1-decisions 20
scripts/netopyu-dsh l1-metrics 500
```

回到无额外模型调用的默认行为：

```bash
NETOPYU_L1_DECISION_MODE=off scripts/netopyu-dsh restart
```

环境变量必须在启动 DSH 与 Worker 的同一部署环境中设置。Shadow 模式不会改变 Tool allowlist、审批或 Runtime 权限。

Hermes 使用相同的 `NETOPYU_L1_DECISION_*` 环境变量；插件启动时会注册官方
`pre_llm_call/pre_tool_call/post_llm_call/on_session_end` hook。验证当前候选没有未审漂移：

```bash
scripts/netopyu-dsh l1-catalog-check
```

私有未见集不进入仓库。先在仓库外准备 case JSONL 和两个独立 reviewer 的 label JSONL：

```json
{"apiVersion":"netopyu.io/l1-holdout-case/v1","case_id":"private-001","profile":"lan","category":"access-diagnosis","language":"zh","prompt":"<真实但未进入开发集的请求>"}
{"apiVersion":"netopyu.io/l1-holdout-label/v1","case_id":"private-001","reviewer_id":"reviewer-a","action":"clarify","target":"lan-user-access-diagnose","arguments":{},"missing_fields":["user_id","app"],"workflow":[]}
```

每个文件一行一个 JSON object；两名 reviewer 各自完成全量 label 文件，不得共用 reviewer id。`action`
只能是 `select_skill/select_tool/clarify/refuse/out_of_scope`；selection 必须有 target 且无 missing，
clarify 必须有 target 和 missing fields，终止 action 不得携带 target/arguments/workflow。示例占位文本不能
作为真实资格样本。

```bash
scripts/netopyu-dsh l1-holdout-seal \
  /private/cases.jsonl netops-holdout v1 > /private/manifest.json
scripts/netopyu-dsh l1-holdout-adjudicate \
  /private/cases.jsonl /private/manifest.json \
  /private/reviewer-a.jsonl /private/reviewer-b.jsonl
scripts/netopyu-dsh l1-holdout-qualify \
  /private/cases.jsonl /private/manifest.json \
  /private/reviewer-a.jsonl /private/reviewer-b.jsonl \
  qwen3.6:27b /private/qualification-report.json
scripts/netopyu-dsh l1-holdout-adapter-parity \
  /private/cases.jsonl /private/manifest.json \
  /private/reviewer-a.jsonl /private/reviewer-b.jsonl \
  qwen3.6:27b /private/adapter-parity-report.json
```

Seal 只输出覆盖统计与摘要；两份标注有任一分歧就不能进入 holdout run。Qualification 要求五类 action 覆盖、至少两次重复、Catalog clean、不可变模型 artifact digest、输入合同 parity、双 Harness 语义 parity、协议/完整 Oracle/安全/候选召回全部 100% 才标记 `qualified`。本地 Ollama digest 自动发现；其他 endpoint 必须显式提供 `NETOPYU_L1_DECISION_MODEL_ARTIFACT_DIGEST`。报告不输出 Prompt、逐条标签或参数值。

Worker runner 的 `scope.level=shared_worker_decision_contract`；Adapter runner 的 `scope.level=adapter_hook_to_worker`，实际执行生产 DSH/Hermes Hook 并验证原请求摘要与 Worker 收到的一致。模型随机性会显式反映在 Decision digest parity 中。二者仍不启动完整 DSH Web 或 Hermes CLI，也不验证 UI composition、Harness 发行包和部署身份，因此这些现场证据不能由本地 runner 冒充。

### 5. 指标解释

| 指标 | 含义 | 不能代表什么 |
|---|---|---|
| `protocol_success_rate` | 模型输出经 Schema、grounding 和严格合同后能否形成 Decision | 业务意图一定正确 |
| `routing_agreement_rate` | Shadow Decision 与 DSH 首次实际 Skill/Tool 路由是否一致 | 任一方等于人工真值 |
| `direct_tool_argument_exact_rate` | 直接 Tool 的候选参数与 DSH 实际参数是否相同 | Skill 内后续所有参数都正确 |
| `safety_escape_count` | clarify/refuse/out-of-scope/protocol-failure 后，DSH 是否仍首次路由到领域 Skill/Tool | 未观测到的后续行为绝对安全 |
| `repair_rate` / `average_model_attempts` | 协议稳定性和额外模型成本 | 推理质量 |
| `reported_tokens` / `usage_complete_rate` | endpoint 已报告 token 与计量完整性 | 未报告 token 的真实成本 |
| `lifecycle_status_counts` / `unobserved_decisions` | observed、pending、closed 和无路由样本 | 未调用领域 Tool 就等于意图正确 |
| `decision_latency_ms` | 完整 shadow Decision 时延，包括模型 | Runtime 自身时延或生产 SLO |

这些指标的 scope 明确为 `local_shadow_observations`。DSH agreement 是一致性信号，不是真值；P1.9-B 必须增加独立人工/封存 Oracle，才能测 selection、参数、clarification 和 workflow 正确率。

### 6. 升级门禁

Shadow 进入 canary 前至少需要：

1. 独立封存的未见集协议成功率 100%，Schema 越界接受为 0，grounding 后无来源参数接受为 0；
2. 固定安全集最终 escape 为 0，并单独报告模型原始 escape；
3. 人工真值 selection、参数 F1、追问 precision/recall 均达到受审门槛，开发集与 holdout 差距不超过 5 个百分点；
4. DSH/Hermes 对同一请求、Catalog 和模型产生相同 Decision digest；
5. Decision → L0 plan binding 为 100%，无未授权或重复执行；
6. Core-72 保持 64/64，Runtime p50/p95 不越过抗噪退化门槛；
7. 隐私检查确认 Prompt、凭据和参数值不进入 Decision store/trajectory；
8. 明确 canary fail-closed、停用开关、回退版本和审计 runbook。

Canary 只允许 Decision 收窄 Tool/Skill 可见面或要求追问，不能直接授予写权限。Enforced 仍必须把所有效果交给 L0 Runtime。

C0 已实现第 5 项的计划合同；C1 已实现单调策略、第 8 项 runbook 和四类证据的机器一致性门禁，但它们不是第 1–4、6–7 项或组织身份/签名的替代品。只有真实外部证据得到 `ready_for_review`、组织评审通过且另一个独立 Adapter 发布变更验收后，才允许考虑把配置从拒绝 `canary` 改为受控启用；readiness 退出码不得直接驱动部署。

---

## English

### 1. Purpose and boundary

P1.9 moves the candidate-specific Schema, grounding, deterministic compiler, and monotonic Guard principles from the isolated P1.8 evaluation path into a production-facing L1 Decision Plane. The plane narrows direct user language to a strict proposal between the Harness and L0 Runtime.

It cannot call a business Tool, create or approve a plan, hold Provider credentials, declare success, or roll back an effect. Every write still passes the complete L0 plan, approval, revalidation, one-shot effect, independent observation, compensation, and audit path.

### 2. Current scope

P1.9-B1 implements a shared DSH/Hermes shadow and local evidence framework. DSH reads direct-user source from an accepted step; Hermes uses its official `pre_llm_call`, `pre_tool_call`, `post_llm_call`, and `on_session_end` hooks. Both bind the current exposed Tool declarations and Skill manifest, retrieve at most twelve candidates, force one candidate-specific function call, ground values, deterministically derive action/missing fields/workflow, and persist privacy-minimized evidence.

Pending Decisions are observed once or closed as superseded/no-route/session-end, and a closed Decision cannot bind a later route. The store removes raw prompts, model prose, and argument values while retaining digests, keys, reported token completeness, and lifecycle. A portable three-profile Catalog drift gate and private 50+/two-reviewer holdout contracts are present.

The P1.9-B2 local tooling now has two levels. The shared-Worker runner aggregates protocol, selection, argument, clarification, workflow, refusal, repeatability, semantic-parity, token, and latency metrics. The adapter runner executes the production DSH JavaScript `agent/pre-step` and Hermes Python `pre_llm_call` hooks through a temporary owner-only Worker and compares prompt/catalog/candidate/policy/full-Decision digests. Neither emits prompts, per-case labels, or argument values.

P1.9-C0 also provides a disabled binding kernel. PreparedPlan schema v10 can bind Decision/evidence digests, the observed Harness route, request and compiled argument digests, the exact L0 Skill/contract, and lifetime into the plan hash. The Journal enforces one plan per Decision id, while mode/session/Harness/profile/route/argument/candidate/digest drift and persisted tampering fail closed. Schema v9/v8 remains read-compatible. Actual external cases/human adjudication and complete DSH Web/Hermes CLI/UI/deployment certification remain evidence items. Both adapters still reject `canary`, so canary/enforced behavior and production SLOs remain open and P1.9 stays 🟡.

P1.9-C1 adds a disabled safety-readiness layer. Its pure policy can only preserve or block/narrow the original Harness route; invalid write material fails closed and invalid read material is observation-only. Its evidence gate cross-binds expiring Worker, Adapter, real product/deployment, and operations-drill documents, including reviewer/owner separation, 64/64 Core controls, a stable/improved three-implementation Runtime trend within p50/p95 thresholds, complete Decision-plan binding, exercised stop/rollback/alerts/no-effect replay, and bounded traffic/duration. The strongest result is `ready_for_review`, never activation. See the [C1 canary runbook](p19-canary-runbook.md).

### 3. Usage

```bash
NETOPYU_L1_DECISION_MODE=shadow \
NETOPYU_L1_DECISION_MODEL=qwen3.6:27b \
scripts/netopyu-dsh restart

scripts/netopyu-dsh l1-decisions 20
scripts/netopyu-dsh l1-metrics 500
scripts/netopyu-dsh l1-catalog-check
```

The default endpoint is `http://127.0.0.1:11434/v1`; the default store lives under the DSH runtime data directory. Return to the zero-extra-call default with `NETOPYU_L1_DECISION_MODE=off` and restart.

Private cases and both full reviewer label files stay outside the repository. They use one JSON object per
line under `netopyu.io/l1-holdout-case/v1` and `netopyu.io/l1-holdout-label/v1`. Actions are limited to
`select_skill`, `select_tool`, `clarify`, `refuse`, and `out_of_scope`; the strict Decision-shape invariants
apply to every label. Placeholder examples are not qualification evidence.

Non-loopback model endpoints are rejected unless `NETOPYU_L1_DECISION_ALLOW_REMOTE=1` is explicit. That opt-in does not replace deployment egress, TLS, credential, or data-governance controls.

### 4. Metric boundary

Protocol success measures contract formation. Routing agreement measures parity with the Harness route, not correctness. Direct-tool argument exactness excludes later Skill-internal calls. Safety escape counts a first domain route after a non-executable Decision. Lifecycle counts distinguish observed, pending, and explicitly closed turns. Reported token completeness prevents partial endpoint accounting from being presented as full cost. Decision latency includes the selector model and is not Runtime latency or a production SLO.

Promotion to canary requires a sealed unseen set, adjudicated labels, zero accepted Schema/grounding escape, zero final safety escape on the fixed safety suite, DSH/Hermes digest parity, complete Decision-to-plan binding, unchanged Core-72 controls, privacy gates, and an explicit fail-closed rollback runbook.

C0 implements the plan contract and one-shot binding. C1 implements monotonic handling, machine-checkable evidence consistency, and the stop/rollback runbook. Neither waives real human/product evidence, organization identity/signatures, or an independently approved release change. Adapter configuration remains restricted to `off/shadow`, and the readiness exit code must never directly trigger deployment.
