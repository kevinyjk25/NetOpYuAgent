# NetOpYuAgent

EnsuredSkill 是一个网络优先的可靠执行 Runtime 原型。DSH、LLM 和 L1 Skill 负责理解、诊断与提出 Candidate Plan；Runtime 依据 Contract、Evidence、Guard、Risk 和事务状态决定哪些操作真正允许作用于网络。

> 文档状态：2026-09-03。Runtime 机械原型与历史小样本实验已完成，但当前主阶段已前移到 **L1→L0 泛化证明**；该门禁通过前，不再开展或宣称大规模 L0→Runtime 效果证明。阶段事实以[项目进展](docs/PROJECT-STATUS.md)为准。

> 当前是本地参考实现和仿真验证环境，不是生产网络认证。固定测试集的 100% 仅表示对应 Oracle 全部通过，不是生产成功概率。
>
> **当前权威范围已经重置为研究原型。** 企业身份、Provider 供应链、多人治理、Hermes/A2A、HA/DR、WORM 和生产 SLO 均为冻结的未来工程，不是当前架构主线或完成条件。详见 [EnsuredSkill 原型权威准则](docs/ENSUREDSKILL-PROTOTYPE.md)。

## 中文

### 项目设计

项目遵循三个原则：**概率推理与确定执行分离；No evidence, no action；LLM 决定尝试什么，Runtime 决定允许发生什么。**

```mermaid
flowchart TB
    RP[Reasoning Plane<br/>DSH・LLM・L1 Skill] -->|Candidate Plan| RR[Reliability Runtime<br/>Contract・Evidence・Guard・Risk・Transaction]
    RR -->|Validated Operation| IP[Infrastructure Plane<br/>Network/Service Provider・Containerlab]
```

| 层 | 权威职责 | 明确边界 |
|---|---|---|
| Reasoning Plane | 会话、开放式理解、诊断、追问、计划和 L1 编排 | 只能提出候选；产品路径没有直接写权限 |
| Reliability Runtime | Contract、journal-backed Typed Graph、跨步骤 Evidence、Guard、Risk、事务、验证和补偿 | 不做开放式语言推理，不把模型 confidence 当权限 |
| Infrastructure Plane | 通过 MCP/API/CLI/NETCONF 提供事实和效果 | 不判断上层业务意图，不自报成功终态 |

### 当前研究门禁

项目现在严格执行：`L1→L0 泛化证明 → L0 确定性校验 → Runtime 评测`。如果转译只适配少量自建 Skill，后续 Runtime 高分只说明它能稳定执行这组人工合同，不能证明通用价值。

当前静态开发库包含 100 个公开 Skill、72 个仓库、9 个领域：53 个 Runtime 包门禁通过，18 个是标准格式但引用上下文不完整的“仅转译”样本，29 个非标准格式只用于鲁棒性测试。主要转译语料为 71 个 Skill/7 个开发批次；它们已经可见，所以不冒充 unseen 证明。只有冻结 Translator 后，在至少 3 个互不重叠的未知 cohort 上累计达到 ≥50 Skill、≥15 仓库、≥8 领域、≥600 case，并同时通过安全、召回、macro-F1、参数和证据门槛，才允许大规模 Runtime A/B。完整口径、命令和指标见 [L1→L0 泛化门禁](docs/TRANSLATION-GENERALIZATION-GATE.md)。

新的[转译用例构造链](docs/TRANSLATION-CASE-AUTHORING.md)不再把任意公开 Skill 硬配给通用 `resource.apply`：9B 只提出非 Gold 候选，确定性代码校验精确原文锚点、参数证据、风险和事务角色，再输出隐藏作者答案的 Skill–Task–Tool 审查包。首个 12-Skill/36-task 开发批次为 12/12 协议有效、6/12 通过结构门禁并生成 18 个盲审包；6 次修复没有挽回案例。该 50% 只是候选可审率，不是 Translator 准确率。通用转译 Tool Catalog 固定不可执行，只有另行存在 Fixture MCP/真实 Provider adapter 时才可能进入 Runtime。

### Skill 与系统怎样交互

L1、L0.5、L0 和 Runtime Plan 不是同一种 Skill 的不同文件格式，而是不同权威等级：

| 对象 | 作用 | 能否执行写操作 |
|---|---|---:|
| L1 `SKILL.md` | 给 LLM 提供领域知识、追问和编排方法 | 否 |
| L0.5 | 把 L1 拆成可审查的参数、条件、风险、验证与补偿语义 | 否 |
| compiled L0 | 人工审查并激活的精确执行合同 | 只能被 Runtime 使用 |
| Candidate Plan | LLM 针对当前请求提出的 Tool/参数候选 | 否 |
| PreparedPlan | Runtime 绑定 Evidence、审批、Provider、TTL 和摘要后的不可变计划 | 是；且只有 Runtime 持有效果能力 |

```text
用户 → DSH/LLM → L1 选择、诊断、追问 → Candidate Plan
     → 唯一 active L0 → 参数/Evidence/Guard/Risk → plan-bound 审批
     → 写前重校验 → 单次 Effect → 独立 Verify
     → verified_success / rollback_verified / manual_intervention_required
```

合法只读请求可以沿 L1 原生读取路径执行，但仍经过 Observation Policy 和参数校验。写候选若没有唯一 active L0、精确参数或足够 Evidence，只能追问、生成 proposal、请求人工或拒绝，不能回退为原生 Agent 写入。完整的离线 authoring、在线执行、示例和证据定位见 [Skill 与系统交互全景](docs/SKILL-SYSTEM-INTERACTION.md)。

### 两项核心能力

1. **Contract-Governed Skill**：L1→L0.5→L0 是 authoring compilation；最终 L0 固定输入、证据、Guard、资源、风险、后置条件和补偿。模型只能生成待审 proposal。
2. **Evidence-Gated Transaction Runtime**：把 L0 编译为不可变计划和 Typed Execution Graph；写前 Snapshot/Precheck/Revalidate，写后独立 Verify；失败时 Reconcile/Compensate/Verify Recovery。

当前结论是：**Runtime 机制原型与接线闭环完成，但项目核心假设尚未通过跨 Skill 泛化门禁。** 历史六场景、消融和 DSH 配对只作为假设形成与机械证据；不表示 L1→L0 已高泛化，更不表示生产成功概率或真实厂商设备认证。

### 已实现能力

| 能力域 | 当前范围 |
|---|---|
| Harness | DSH 主路径；模型/L1 只输出 Candidate Plan，经窄 Worker bridge 进入 Runtime |
| L1 | LAN/DC/WAN Skill，缺参追问，多步 workflow，领域外和高风险拒绝 |
| L0 | 21 个激活合同；原子、约束、扩展、组合 Saga；21/21 三阶段可解释轨迹 |
| Promotion | L1/L0.5/L0 并排审查、语义映射、低置信告警、合同图、严格安全门禁 |
| Runtime | ReliabilityContract、journal-backed Typed Graph gate、跨步骤 Evidence provenance、Guard、Risk、写前重校验、独立验证、补偿、恢复和分阶段时延 |
| Provider | 协议无关 Observation/Effect Capability；Network Observer、Actor 和本地 Adapter |
| 网络仿真 | Containerlab + FRR：OSPF、eBGP、VLAN、EVPN/VXLAN L2VPN、故障切换和真实容器转发 |
| 评测 | 六类 ES-P0 事务场景、独立安全 scorer、五机制消融矩阵和真实 DSH 配对协议 |

Hermes/A2A、跨域业务 Lab、企业身份与审批、Provider 供应链、治理工作台、HA/DR、远端不可变审计和生产 SLO 的已有代码统一冻结，不计入当前能力或完成度。真实厂商设备、EVPN L3VPN、MPLS L2VPN/L3VPN 也不在当前原型覆盖内。

### 历史小样本结果与当前证据边界

以下实验只改变一个变量：Control 为 `DSH + 同一模型/L1 Skill + 原生工具编排`；Treatment 加入 L0 资格门禁与 Runtime。它们证明机制值得继续研究，但样本参与了早期设计，不能证明转译泛化，也不能解锁新的大规模 Runtime 结论。

| 模型与指标 | DSH + L1 原生 | DSH + L0 auto Runtime |
|---|---:|---:|
| 9B Task Completion | 50.00% | **86.67%** |
| 9B Unsafe / False Commit / Invalid Action | 20.00% / 13.33% / 33.33% | **0 / 0 / 0** |
| 9B Execution Precision / Autonomous Coverage | 59.09% / 43.33% | **100% / 76.67%** |
| 9B p50 / p95 | 90.693 / 158.397 秒 | **44.405 / 72.173 秒** |
| 7B Task Completion | 20.00% | **36.67%** |
| 7B Unsafe / False Commit / Invalid Action | 10.00% / 3.33% / 20.00% | **0 / 0 / 0** |
| 7B Execution Precision / Autonomous Coverage | 50.00% / 20.00% | **100% / 36.67%** |

9B 转译严格通过 58/60，7B 为 38/60，两者误接受均为 0。7B 两臂各有 18/30 次 DSH `EMPTY_RESPONSE`，因此只支持安全边界跨模型稳定，不代表 7B 可用性合格。完整实验矩阵、消融贡献、进程失败、复现命令和摘要见 [ES-P0 本地证据报告](docs/ES-P0-EVIDENCE.md)。固定集结果不是生产概率。

仓库外自动数据通道已封存 240 条模型合成用例，覆盖 6 类 Anthropic Skill、10 类事务/故障、6 个 MCP 域和 3 类语言。qwen3.5:9b 转译协议有效 240/240，可信 Oracle 合格 235/240、fallback 5、false accept 0；10 场景×3 次真实 DSH 配对中，Treatment 将 Task Completion 从 76.67% 提升至 93.33%，unsafe 从 4/30 降至 0/30，p50 从 103.6 秒降至 64.3 秒。它用于在正式人工 ES-P1 前发现价值和边界，**不冒充独立人工 holdout 或生产概率**。方法、完整指标和命令见[仓库外合成 Holdout](docs/SYNTHETIC-HOLDOUT.md)。

公开 Skill 技术链路已完成角色隔离模拟：15 个实际公开 Skill、45 个案例、3 次重复和 270 次真实本地 DSH 实验臂执行中，Gold-blind 9B 转译路由一致 43/45、unsafe Runtime 误接纳为 0；Treatment 将 Task Completion 从 82.22% 提升到 97.78%，L0 路由从 21/42 提升到 42/42，p95 从 109.3 秒降到 56.1 秒。原生只读和 safe-stop 两臂保持相同，唯一残余是 1 个 L1 只读案例的三次失败。该结果是虚拟 Case/Gold 角色和声明式 fixture 的 `ES-P1-Wild-Sim`，**不是真人独立 holdout、真实系统或生产概率**；完整方法、分层结果和边界见[角色隔离模拟报告](docs/ES-P1-WILD-SIMULATED-RESULTS.md)，测试列表见[Skill 索引](docs/benchmarks/es-p1-wild-skill-index.json)，工作流见[公开 Skill 市场语料](docs/ES-P1-PUBLIC-SKILL-CORPUS.md)。

Runtime 结果不能只看一项 `success`：

| 终态 | 应怎样解释 |
|---|---|
| `verified_success` | 唯一正向成功；独立回读证明批准的后置条件成立 |
| `rollback_verified` | 任务失败，但 Runtime 证明已恢复基线；不能计为任务成功 |
| `precondition_changed` / `rejected` / `expired` | 写前安全停止，Effect 未被允许继续 |
| `manual_intervention_required` | 目标或恢复状态无法证明，需要人工检查，禁止自动重试或宣称成功 |

用 `scripts/netopyu-dsh runtime PLAN_ID` 可查看不可变计划、图节点、分阶段时延和 Evidence provenance；用 `runtime-audit` 验证事件摘要链。

### 支持的典型场景

- 新员工应用访问开通：身份、应用、审批、权限 MCP 与网络 L0 Saga 联动；
- LAN 用户接入诊断、授权、撤销和回滚；
- DC 应用访问、Fabric 配置、EVPN/VXLAN L2 路径诊断；
- OSPF/eBGP 路径查询、链路故障切换和恢复；
- 设备配置 edit/push、部署 rollback、节点 drain、服务 restart/rollback；
- L1 Skill 转 L0.5/L0 的 proposal、语义审查和离线资格工作流；
- REST/MCP/SSH/NETCONF/Controller Provider 接入前的合同检查。

### 快速开始

依赖：Python 3.11/3.12、Node.js 22.19+ 或 24+、pnpm、Ollama。Containerlab 场景另需 Linux/Docker/Containerlab。

```bash
git clone https://github.com/kevinyjk25/NetOpYuAgent.git
cd NetOpYuAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

ollama pull qwen3.5:9b
scripts/netopyu-dsh install
scripts/netopyu-dsh settings-sync
scripts/netopyu-dsh model qwen3.5:9b
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
```

打开 <http://127.0.0.1:3080/>。若该端口已被占用，使用 `NETOPYU_DSH_PORT=3081 scripts/netopyu-dsh restart`，并始终以 `scripts/netopyu-dsh status` 输出为准。

### 如何使用

先看三条 Golden Path 和本机就绪状态：

```bash
scripts/netopyu doctor
scripts/netopyu journeys
scripts/netopyu agent-usecases
scripts/netopyu evaluate
```

`agent-usecases` 会给出三个可直接粘贴到 DSH 页面、真正包含 LLM Tool loop 的用例：Runtime L1→L0、L1→L0.5→L0 authoring、四类 MCP 外部系统集成。详细 Prompt 与预期证据见[真实 LLM Agent 用例](docs/AGENTIZED-USE-CASES.md)。

验证自己的外部系统接入包，不连接或激活目标系统：

```bash
scripts/netopyu integration-check \
  --pack examples/integration-rest-mcp/pack.yaml
```

开发和审查 L0：

```bash
scripts/netopyu-l0 validate
scripts/netopyu-l0 list
scripts/netopyu-l0 explain network.lan.user-access.grant
scripts/netopyu-l0 runtime-trajectories-validate
scripts/netopyu-l0 workbench-export \
  --proposal /path/to/proposal --output /tmp/semantic-review.html
```

创建仓库外、角色隔离的正向资格工作区：

```bash
scripts/netopyu-l0 forward-eval-study-kit --output-root /private/forward-study
scripts/netopyu-l0 forward-eval-study-doctor --root /private/forward-study
```

没有独立业务团队时，可先自动生成明确标记为 synthetic、且不能通过正式 ES-P1 门禁的封存用例：

```bash
scripts/netopyu-synthetic-study export /private/synthetic-study --cases 240
cd /private/synthetic-study
env -u PYTHONPATH python3 generate.py --model qwen3.5:9b --resume
```

检查通用 Anthropic Skill 包及其安全路由：

```bash
scripts/netopyu-effect inspect-package --skill /path/to/my-skill
python -m evaluation.progressive_skill_suite
python -m evaluation.general_effect_model \
  --dataset-root /private/synthetic-study \
  --model qwen3.5:9b --output-root artifacts/es-p0-9b-translation
scripts/netopyu-harness-ab \
  --dataset-root /private/synthetic-study \
  --model qwen3.5:9b \
  --translation-report artifacts/es-p0-9b-translation/model-translation.json \
  --output-root artifacts/es-p0-dsh-9b \
  --stratified-patterns --repetitions 3
scripts/netopyu-synthetic-study report /private/synthetic-study \
  --translation-report artifacts/es-p0-9b-translation/model-translation.json \
  --dsh-report artifacts/es-p0-dsh-9b/real-harness-ab.json \
  --output-root artifacts/es-p0-synthetic-evidence
```

Containerlab 实验、审批卡、回滚证据和 Provider 接入的完整操作见[使用与系统接入](docs/getting-started-integration.md)。

### 文档入口

- [文档导航与权威边界](docs/README.md)
- [Skill 与系统交互全景](docs/SKILL-SYSTEM-INTERACTION.md)
- [项目进展与路线图](docs/PROJECT-STATUS.md)
- [架构与 ADR](ARCHITECTURE.md)
- [高层设计](HLD.md)、[低层设计](LLD.md)、[安全设计](SSD.md)
- [ES-P0 本地证据报告](docs/ES-P0-EVIDENCE.md)
- [ES-P1-Wild 角色隔离模拟结果](docs/ES-P1-WILD-SIMULATED-RESULTS.md)
- [仓库外合成 Holdout](docs/SYNTHETIC-HOLDOUT.md)
- [通用渐进式确定化与跨域验证](docs/progressive-determinization.md)
- [真实 Harness 自动 Runtime A/B](docs/general-effect-ab.md)
- [L1 → L0 Promotion](docs/l1-to-l0-promotion.md)
- [正向资格协议](docs/promotion-forward-qualification.md)
- [Runtime A/B 基线](docs/benchmarks/runtime-ab-baseline.md)

---

## English

EnsuredSkill is a network-first Reliability Runtime research prototype. DSH, the LLM, and L1 Skills produce hypotheses and Candidate Plans; Contract, Evidence, Guard, Risk, and transactional state determine what is allowed to reach the network.

The [authoritative prototype charter](docs/ENSUREDSKILL-PROTOTYPE.md) supersedes conflicting production-engineering plans. Enterprise identity, provider supply chain, multi-team governance, Hermes/A2A productization, HA/DR, WORM audit, and production SLOs are frozen future work rather than current architecture or exit criteria.

### Design

The design has three rules: separate probabilistic reasoning from deterministic execution; no evidence means no action; the LLM decides what to attempt while the Runtime decides what is allowed to happen.

| Layer | Authority | Boundary |
|---|---|---|
| Reasoning Plane | DSH, LLM, L1 understanding, diagnosis, clarification and planning | proposes only; the product path has no direct write authority |
| Reliability Runtime | Contract, typed graph, Evidence, Guard, Risk, transaction, verification and compensation | performs no open-ended reasoning; confidence grants no authority |
| Infrastructure Plane | network/service Providers over MCP/API/CLI/NETCONF and labs | owns facts and effects, but cannot self-declare a verified terminal outcome |

L1, L0.5, L0, and a Runtime plan have different authority. L1 is natural-language semantic guidance; L0.5 is a review-only structured proposal; only an explicitly reviewed and activated compiled L0 can govern an effect. A per-request Candidate Plan remains untrusted until the Runtime resolves the exact L0, grounds parameters, validates evidence/guards/risk, and creates a plan-bound approval. An unqualified read may remain read-only; an unqualified write stops safely and never regains native-Agent write authority. See the [complete Skill-to-system interaction guide](docs/SKILL-SYSTEM-INTERACTION.md).

The two core capabilities are:

1. **Contract-Governed Skill authoring.** L1→L0.5→L0 preserves readable intent while producing an executable contract. It creates review proposals only and is distinct from trace-based Experience Compilation.
2. **Evidence-Gated transactional execution.** An active L0 becomes an immutable plan and typed graph. Runtime snapshots, prechecks, revalidates, executes, verifies, commits, reconciles uncertainty, compensates, verifies recovery, and audits terminal evidence.

The Runtime mechanism and wiring prototype is complete, but the core project hypothesis has not yet passed cross-Skill translation generalization. Historical provenance, ablation, and paired-Harness results are retained as hypothesis-forming evidence; they do not establish broad L1-to-L0 validity, real-device qualification, or production readiness.

### Active research gate

The enforced evidence order is now `L1-to-L0 generalization → deterministic L0 validation → Runtime evaluation`. The known development inventory contains 100 public Skills from 72 repositories and nine domains: 53 pass the strict Runtime package gate, 18 are conformant translation-only partial-context inputs, and 29 format variants are robustness-only. The 71 primary Skills form seven development batches and are not unseen evidence. Large Runtime A/B work remains locked until one frozen Translator passes at least three disjoint post-freeze cohorts totaling at least 50 Skills, 15 repositories, eight domains, and 600 cases, including strict safety, recall, macro-F1, exact-parameter, evidence, and construct-alignment gates. See the [L1-to-L0 generalization gate](docs/TRANSLATION-GENERALIZATION-GATE.md).

### Capabilities and evidence

The active prototype includes 21 reviewed L0 contracts and readable three-stage trajectories, LAN/DC/WAN L1 guidance, the DSH path, a journal-backed Typed Graph gate, cross-step evidence provenance and stage latency, network Observation/Effect providers, Containerlab/FRR labs, and the ES-P0 evaluation protocol. Hermes/A2A, enterprise controls, supply-chain admission, governance, and extra domains are frozen experimental code rather than active capability claims.

The local ES-P0 evidence loop is complete. In 30 real paired DSH sessions with `qwen3.5:9b`, treatment improved task completion from 50.00% to 86.67%, execution precision from 59.09% to 100%, and autonomous coverage from 43.33% to 76.67%; unsafe execution, false commits, and invalid actions fell from 20.00%/13.33%/33.33% to zero. With `qwen2.5:7b`, the same safety metrics also fell to zero, but 18/30 sessions in both arms failed at the DSH/model availability layer, so 7B is not availability-qualified. Translation false accepts were zero for both models. See the [ES-P0 local evidence report](docs/ES-P0-EVIDENCE.md).

These are transparent local development results, not production probability, hidden-set generalization, or vendor-device certification. Historical one-shot and evaluator-only native-write experiments remain component/exploratory evidence only and are not product fallback paths.

A repository-external synthetic path has sealed 240 model-authored cases across six Anthropic Skill feature families, ten transaction/fault patterns, six MCP domains, and three language groups. qwen3.5:9b produced 240/240 schema-valid proposals; 235 passed every trusted Oracle, five remained fallback-only, and no rejected proposal received Runtime authority. Across ten stratified scenarios and three real-DSH repetitions, Treatment improved task completion from 76.67% to 93.33%, reduced unsafe executions from 4/30 to 0/30, and reduced p50 latency from 103.6 to 64.3 seconds. This remains model-authored pre-ES-P1 evidence, not independent human qualification or a production probability. See the [synthetic holdout guide](docs/SYNTHETIC-HOLDOUT.md).

The public-Skill path now has a complete role-separated simulation over 15 real public Skills, 45 cases, three repetitions, and 270 real local DSH arm executions. Gold-blind 9B translation matched 43/45 simulated Gold routes with zero unsafe Runtime accepts. Treatment improved task completion from 82.22% to 97.78%, lifted the L0 route from 21/42 to 42/42, and reduced p95 from 109.3 to 56.1 seconds; native-read and safe-stop outcomes were unchanged. This is `ES-P1-Wild-Sim` over virtual Case/Gold roles and declarative fixtures, **not independent-human holdout, real-system evidence, or production probability**. See the [role-separated simulation report](docs/ES-P1-WILD-SIMULATED-RESULTS.md), [tested-Skill index](docs/benchmarks/es-p1-wild-skill-index.json), and [public Skill-market workflow](docs/ES-P1-PUBLIC-SKILL-CORPUS.md).

Execution outcomes are deliberately not a generic success Boolean. `verified_success` is the only positive success and requires independent postcondition evidence. `rollback_verified` proves restoration after a failed task, while `precondition_changed`, `rejected`, and `expired` are safe pre-effect stops. `manual_intervention_required` means the Runtime cannot prove the target or recovery state. Inspect a plan with `scripts/netopyu-dsh runtime PLAN_ID` and verify its event chain with `runtime-audit`.

Production qualification remains open for vendor devices, enterprise identity/change systems, independently owned signing roots, distributed HA/DR, remote immutable audit, and production SLOs. EVPN L3VPN and MPLS L2/L3 VPN are outside the current lab coverage.

### Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
ollama pull qwen3.5:9b
scripts/netopyu-dsh install
scripts/netopyu-dsh settings-sync
scripts/netopyu-dsh model qwen3.5:9b
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
```

Open <http://127.0.0.1:3080/>. Use `scripts/netopyu-dsh status` as the authoritative URL/process check.

```bash
scripts/netopyu doctor
scripts/netopyu journeys
scripts/netopyu agent-usecases
scripts/netopyu evaluate
scripts/netopyu integration-check --pack examples/integration-rest-mcp/pack.yaml
```

The Agent use-case command prints three paste-ready real-LLM DSH journeys: L1-to-L0 Runtime execution, proposal-only L1-to-L0.5-to-L0 authoring, and four-system MCP integration. See the [Agent use cases](docs/AGENTIZED-USE-CASES.md) and [integration guide](docs/getting-started-integration.md).

For L0 development and external qualification:

```bash
scripts/netopyu-l0 validate
scripts/netopyu-l0 runtime-trajectories-validate
scripts/netopyu-l0 forward-eval-study-kit --output-root /private/forward-study
scripts/netopyu-l0 forward-eval-study-doctor --root /private/forward-study
scripts/netopyu-effect inspect-package --skill /path/to/my-skill
```

### Documentation

Start with the [documentation map](docs/README.md), [Skill-to-system interaction guide](docs/SKILL-SYSTEM-INTERACTION.md), [project status](docs/PROJECT-STATUS.md), [architecture](ARCHITECTURE.md), [HLD](HLD.md), [LLD](LLD.md), [SSD](SSD.md), the [ES-P0 local evidence report](docs/ES-P0-EVIDENCE.md), and the [repository-external synthetic holdout](docs/SYNTHETIC-HOLDOUT.md).
