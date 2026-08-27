# NetOpYuAgent

## 中文

### 项目定位

NetOpYuAgent 是运行在 [DeepSeek Harness（DSH）](https://github.com/deepseek-ai/deepseek-harness) 之上的网络领域插件与确定性执行运行时。

DSH 负责通用智能体能力：会话、模型调用、工具调用、Web UI、Skill 生命周期、审批交互和子代理框架。NetOpYuAgent 不再实现第二套通用 Agent Harness，而是提供网络领域必须保留的可靠能力：

- Domain L1 Skills：诊断、追问、跨域协作和业务流程编排；
- Network L0 Skills：参数校验、风险计算、预检、审批绑定、单次执行、结果验证、补偿/回滚和审计；
- LAN、DC、WAN 与 pragmatic 网络工具；
- DSH 插件、Python Worker、A2A provider、作用域记忆、能力检索和离线评测。

> 当前状态：P0 迁移完成；P0.5 本地模拟原型闭环完成。它证明了架构与安全执行链路，但不等于真实生产网络中的“绝对 100% 正确”。真实设备、企业身份、变更窗口、HA、备份恢复与生产 SLO 属于 P1。

### 分层术语

为避免 L0/L1 歧义，本项目统一使用以下名称：

| 名称 | 职责 |
|---|---|
| DSH Platform Layer | 通用 Agent Harness；管理模型、会话、UI、工具与审批交互 |
| NetOpYu Domain Layer | DSH 之上的网络领域能力总层 |
| Domain L1 Skill | 允许模型参与的泛化业务 Skill；负责理解、诊断、追问和编排 |
| Network L0 Skill | 不依赖模型推理的版本化执行合同；负责确定性网络效果 |
| Network Runtime | 编译并执行 Network L0 Skill 的安全运行时 |

### P0.5 完成范围

本地 mock 范围已经具备：

- DSH-only Web UI 和 Agent runtime；
- 版本化 Network L0 Skill 注册表；
- 严格参数类型、必填字段、目标存在性与参数来源校验；
- 不可变 `IntentSpec`、`plan_hash`、`intent_hash` 与 L0 合同哈希；
- DSH `allowed-once` 审批与一次性 Tool Guard grant；
- 执行前状态重校验，阻止 TOCTOU 状态漂移；
- 独立 typed postcondition 验证；
- 合同化补偿/回滚与人工介入终态；
- SQLite 状态机及防篡改事件哈希链；
- 持久化 Python Worker 和故障恢复；
- A2A 发现、委派、深度/循环保护和持久化 continuation；
- 本地 loopback-only DC peer，用于真实 A2A/SSE 协议模拟；
- 作用域记忆、大结果分页、能力检索和隐私最小化轨迹；
- 完整 retirement 门禁：132 个测试、32 个子测试和 7/7 本地端到端检查。

### 快速开始

依赖：

- macOS 或 Linux；
- Python 3.11 或 3.12；
- Node.js 22.19+ 或 24+；
- pnpm；
- Ollama 和本地模型。

```bash
cd /Users/steven/NetOpYuAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

ollama pull qwen3.5:27b
ollama pull qwen2.5:7b

scripts/netopyu-dsh install
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
```

打开 <http://127.0.0.1:3080/>。

完整运行时依赖按用途拆分：

| 文件 | 用途 |
|---|---|
| `requirements-core.txt` | DSH bridge 核心依赖 |
| `requirements-pragmatic.txt` | Netmiko/NAPALM/Nornir 等真实网络适配器 |
| `requirements-observability.txt` | OpenTelemetry 可观测性 |
| `requirements-dev.txt` | 本地/CI 测试门禁 |
| `requirements.txt` | 生产能力全集 |

### 本地 L1 + L0 + A2A 演练

以下流程只改变本地模拟状态，不连接真实设备：

```bash
scripts/netopyu-dsh stop
scripts/netopyu-dsh dc-peer-start

NETOPYU_PROFILE=lan \
NETOPYU_DSH_BACKEND=mock \
NETOPYU_DSH_ENABLE_DESTRUCTIVE=1 \
NETOPYU_DSH_A2A_PEERS=http://127.0.0.1:8765 \
scripts/netopyu-dsh start
```

在 DSH 新会话中使用 `qwen3.5:27b`，输入：

```text
这是本地 mock 网络演练。请调用 lan-new-employee-onboarding-access Skill，
为新员工 erin 开通 CRM 的端到端访问。严格实际调用工具；所有写入都让我在
DSH 的 Network L0 Skill 计划审批卡中审批，不要使用通用提问代替审批。
```

审批卡必须显示精确参数、目标、风险、plan hash、intent hash、L0 Skill 版本与哈希、验证合同和回滚合同。执行后检查：

```bash
scripts/netopyu-dsh runtime-list 5
scripts/netopyu-dsh runtime PLAN_ID
scripts/netopyu-dsh runtime-audit PLAN_ID
```

### 模型使用策略

- 默认使用 `qwen3.5:27b` 进行网络工具会话。
- `qwen2.5:7b` 的本地资格测试未通过自主变更要求：它会跳过指定 Skill、错误选择子代理、在成功后重复发起相同破坏性调用，并错误解释审批拒绝。
- 7B 可以用于只读查询、分类或候选意图生成，但不能独立驱动可变更网络流程。
- 模型只提出候选意图；Network Runtime 决定该意图能否安全进入效果层。

Network L0 Skill 能保证“已校验并经审批的具体计划”按合同执行和验证，不能保证模型提出的计划天然等于用户真实业务意图。

### 安全默认值

- 默认 backend 为 `mock`；pragmatic 模式缺少真实来源时 fail closed。
- 默认只暴露只读工具。
- 写工具必须显式设置 `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1`。
- 环境变量不能绕过单次 DSH 审批。
- 批准绑定到完整计划哈希；参数、目标或合同变化都会使批准失效。
- 授权 grant 最多消费一次，重放和并发重复调用会失败。
- 成功只能由新的独立 postcondition evidence 判定，不能由模型文本或写工具返回值直接判定。
- 真实凭据只允许通过环境或部署密钥系统注入，不写入仓库、计划摘要或轨迹。

### 常用命令

```bash
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
scripts/netopyu-dsh stop
scripts/netopyu-dsh status
scripts/netopyu-dsh logs
scripts/netopyu-dsh models
scripts/netopyu-dsh model qwen3.5:27b
scripts/netopyu-dsh backend
scripts/netopyu-dsh worker-status
scripts/netopyu-dsh dc-peer-start
scripts/netopyu-dsh dc-peer-status
scripts/netopyu-dsh peers
scripts/netopyu-dsh l0-skills
scripts/netopyu-dsh demo-l1-l0
scripts/netopyu-dsh parity
scripts/netopyu-dsh reliability
scripts/netopyu-dsh retirement
```

可变运行时数据默认位于 `~/Library/Application Support/NetOpYuAgent/dsh-runtime`，可用 `NETOPYU_DSH_RUNTIME` 覆盖。运行时 SQLite、日志、虚拟环境和 IDE 文件不会提交到 Git。

### 验证

```bash
scripts/netopyu-dsh retirement
```

该命令是本项目的权威本地门禁，覆盖 Python、Node 插件语法、DSH-only 架构、HITL、A2A、Network Runtime、Skill 投影、检索质量、Worker 并发/恢复和破坏性操作策略。

### 文档

- [ARCHITECTURE.md](ARCHITECTURE.md)：权威架构边界、依赖规则与架构决策；
- [HLD.md](HLD.md)：高层设计、组件、部署和端到端数据流；
- [LLD.md](LLD.md)：模块、接口、状态机、合同与异常处理；
- [SSD.md](SSD.md)：系统规格、安全设计、威胁模型和验收标准。

---

## English

### Project scope

NetOpYuAgent is a network-domain plugin and deterministic execution runtime built on [DeepSeek Harness (DSH)](https://github.com/deepseek-ai/deepseek-harness).

DSH owns the general agent platform: sessions, model calls, tools, Web UI, Skills, approval interaction, and subagents. NetOpYuAgent no longer implements a second general-purpose harness. It contributes only the network-domain capabilities that must remain reliable:

- Domain L1 Skills for diagnosis, clarification, cross-domain collaboration, and workflow orchestration;
- Network L0 Skills for validation, risk assessment, preflight, approval binding, one-shot execution, verification, compensation/rollback, and audit;
- LAN, DC, WAN, and pragmatic network tools;
- the DSH plugin, Python Worker, A2A provider, scoped memory, capability retrieval, and offline evaluation.

> Status: the P0 migration is complete and the P0.5 local-simulation prototype is complete. This validates the architecture and safety pipeline; it does not claim absolute correctness in a real production network. Real devices, enterprise identity, change windows, HA, backup/restore, and production SLOs belong to P1.

### Layer terminology

| Name | Responsibility |
|---|---|
| DSH Platform Layer | General agent harness for models, sessions, UI, tools, and approval interaction |
| NetOpYu Domain Layer | All network-domain capabilities above DSH |
| Domain L1 Skill | Generalized, model-assisted business Skill for reasoning and orchestration |
| Network L0 Skill | Versioned, model-independent effect contract |
| Network Runtime | Safety runtime that compiles and executes Network L0 Skills |

### P0.5 completion scope

The local mock scope includes a DSH-only runtime and UI, versioned Network L0 Skills, strict parameters and provenance, immutable intent/plan/contract hashes, allowed-once approval, one-shot Tool Guard grants, execution-time revalidation, typed independent postconditions, contractual compensation, a tamper-evident SQLite journal, persistent Worker recovery, A2A discovery and continuations, a loopback-only DC peer, scoped memory, large-result paging, capability retrieval, and a complete retirement gate with 132 tests, 32 subtests, and 7/7 end-to-end checks.

### Quick start

Requirements: Python 3.11/3.12, Node.js 22.19+ or 24+, pnpm, Ollama, and a local model.

```bash
cd /Users/steven/NetOpYuAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

ollama pull qwen3.5:27b
ollama pull qwen2.5:7b

scripts/netopyu-dsh install
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
```

Open <http://127.0.0.1:3080/>.

### Local L1 + L0 + A2A exercise

This changes local simulator state only:

```bash
scripts/netopyu-dsh stop
scripts/netopyu-dsh dc-peer-start

NETOPYU_PROFILE=lan \
NETOPYU_DSH_BACKEND=mock \
NETOPYU_DSH_ENABLE_DESTRUCTIVE=1 \
NETOPYU_DSH_A2A_PEERS=http://127.0.0.1:8765 \
scripts/netopyu-dsh start
```

Use `qwen3.5:27b` in a new DSH session and ask it to run the `lan-new-employee-onboarding-access` Skill for user `erin` and application `crm`. Approve writes only through the Network L0 plan card, then inspect and audit the plan:

```bash
scripts/netopyu-dsh runtime PLAN_ID
scripts/netopyu-dsh runtime-audit PLAN_ID
```

### Model policy

`qwen3.5:27b` is the default for network tool sessions. Local qualification rejected `qwen2.5:7b` for autonomous mutations because it skipped the required Skill, selected an unrelated subagent path, retried a destructive action after success, and misinterpreted a rejected approval. It may be used for read-only classification or candidate intent generation behind the Runtime boundary.

A Network L0 Skill guarantees the execution properties of a specific validated and approved plan. It does not guarantee that the model selected the correct business plan.

### Safety defaults

- `mock` is the default backend; an incomplete pragmatic backend fails closed.
- Only read-only tools are exposed by default.
- Mutations require `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1` and a fresh DSH allowed-once decision.
- Approval is bound to the full plan hash and cannot be reused after any parameter, target, or contract change.
- Grants are consumable once and resist replay/concurrent duplication.
- Success requires fresh independent postcondition evidence.
- Credentials must come from environment or deployment secret systems and must not enter plans, trajectories, or Git.

### Verification

```bash
scripts/netopyu-dsh retirement
```

This is the authoritative local gate for Python tests, Node/plugin syntax, DSH-only architecture, HITL, A2A, Network Runtime, Skill projection, retrieval quality, Worker concurrency/recovery, and mutation policy.

### Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): authoritative boundaries, dependency rules, and decisions;
- [HLD.md](HLD.md): high-level components, deployment, and end-to-end flows;
- [LLD.md](LLD.md): modules, interfaces, state machines, contracts, and failure handling;
- [SSD.md](SSD.md): system specification, security design, threat model, and acceptance criteria.
