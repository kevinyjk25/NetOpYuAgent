# NetOpYuAgent

## 中文

### 项目定位

NetOpYuAgent 是可接入多个通用 Agent Harness 的网络领域插件与确定性执行运行时。当前主平台是 [DeepSeek Harness（DSH）](https://github.com/deepseek-ai/deepseek-harness)，并提供基于公开插件 API 的 [Hermes Agent](https://github.com/NousResearch/Hermes-Agent) Adapter。

DSH 或 Hermes 负责会话、模型调用、工具调用、UI/CLI、Skill 生命周期和通用编排。NetOpYuAgent 不实现第二套通用 Agent Harness，而是提供网络领域必须保留的可靠能力：

- Domain L1 Skills：诊断、追问、跨域协作和业务流程编排；
- Network L0 Skills：参数校验、风险计算、预检、审批绑定、单次执行、结果验证、补偿/回滚和审计；
- LAN、DC、WAN 与 pragmatic 网络工具；
- DSH/Hermes Harness Adapter、共享 Python Worker、A2A provider、作用域记忆、能力检索和离线评测。

> 当前状态：P0 迁移完成；P0.5 mock 原型闭环完成；P0.75-A FRR/Containerlab 实验后端已实现。P0.75-A 需要 Linux Containerlab 执行环境，本机依赖未就绪时仍会 fail closed。它证明架构、安全执行、控制面与容器转发链路，不等于真实生产网络中的“绝对 100% 正确”。真实设备、企业身份、变更窗口、HA、备份恢复与生产 SLO 属于 P1。

### 分层术语

为避免 L0/L1 歧义，本项目统一使用以下名称：

| 名称 | 职责 |
|---|---|
| Harness Platform Layer | DSH 或 Hermes；管理模型、会话、UI/CLI、工具与交互 |
| Harness Adapter | 将同一领域能力投影到 DSH 或 Hermes，不拥有网络效果语义 |
| NetOpYu Domain Layer | Harness 之下共享的网络领域能力总层 |
| Domain L1 Skill | 允许模型参与的泛化业务 Skill；负责理解、诊断、追问和编排 |
| Network L0 Skill | 不依赖模型推理的版本化执行合同；负责确定性网络效果 |
| Network Runtime | 编译并执行 Network L0 Skill 的安全运行时 |

### P0.5 完成范围

本地 mock 范围已经具备：

- DSH Web UI 主路径和 Hermes CLI/Gateway 插件路径；
- 版本化 Network L0 Skill 注册表；
- 严格参数类型、必填字段、目标存在性与参数来源校验；
- 不可变 `IntentSpec`、`plan_hash`、`intent_hash` 与 L0 合同哈希；
- DSH `allowed-once` 卡片审批，或 Hermes 用户专属 slash command 审批；
- 执行前状态重校验，阻止 TOCTOU 状态漂移；
- 独立 typed postcondition 验证；
- 合同化补偿/回滚与人工介入终态；
- SQLite 状态机及防篡改事件哈希链；
- 持久化 Python Worker 和故障恢复；
- A2A 发现、委派、深度/循环保护和持久化 continuation；
- 本地 loopback-only DC peer，用于真实 A2A/SSE 协议模拟；
- 作用域记忆、大结果分页、能力检索和隐私最小化轨迹；
- DSH/Hermes 使用同一 Worker、L0 注册表、Network Runtime、验证器和审计；
- 完整 retirement 门禁：148 个测试、32 个子测试和 7/7 本地端到端检查。

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

### Hermes Adapter 本地运行

Hermes 不取代 Network Runtime，只增加一个 Harness 入口。先按 Hermes 官方说明安装 `hermes` CLI，然后运行：

```bash
scripts/netopyu-hermes install
hermes plugins enable netopyu
scripts/netopyu-hermes worker-start
scripts/netopyu-hermes doctor

NETOPYU_HERMES_PROFILE=lan \
NETOPYU_HERMES_ENABLE_DESTRUCTIVE=1 \
NETOPYU_HERMES_OPERATOR_ID=local:steven \
scripts/netopyu-hermes run
```

写工具只返回 `approval_required`，其中不包含 execution nonce。模型必须停止；操作员核对完整计划后，亲自输入返回的精确命令：

```text
/netopyu-approve PLAN_ID FULL_PLAN_HASH
```

拒绝使用 `/netopyu-deny PLAN_ID FULL_PLAN_HASH`。Hermes 重启会丢弃待审批 nonce，旧计划因此无法执行，需要重新 prepare。可在没有 Hermes CLI 的情况下验证 Adapter 与 Runtime 一致性：

Hermes 插件同时提供 `netopyu_skill_catalog`、`netopyu_skill_view`、`netopyu_capability_search` 和显式作用域 Memory。复杂业务先调用 `netopyu_skill_view`，Adapter 会启动 reviewed workflow；后续只读工具结果被记录为 L0 写入的前置 evidence。

```bash
scripts/netopyu-hermes compare
scripts/netopyu-hermes test
```

当前机器尚未安装 Hermes 时，`doctor` 会明确报告该项；这不会影响纯本地 Adapter 合同测试。

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
scripts/netopyu-dsh worker-stop
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

### P0.75-A FRR + Containerlab 实验室

P0.75-A 使用两个 FRR 路由器、主备 OSPF WAN 链路和两个 Alpine 终端，把 L1/L0
从进程内 mock 推进到真实 CLI、真实协议收敛与真实容器转发。版本化
`labs/p075-a-frr/lab.yaml` 固定设备、探测和故障接口；模型不能扩展目标范围。

在 Linux/Containerlab 环境中运行：

```bash
python scripts/netopyu_lab.py preflight
python scripts/netopyu_lab.py deploy --approve-local-lab
python scripts/netopyu_lab.py verify
python scripts/netopyu_lab.py exercise-failover --approve-local-lab
```

Apple Silicon 推荐使用仓库内 `.devcontainer/devcontainer.json`。先启动 Docker
Desktop，再进入 Dev Container，并在容器内执行：

```bash
sudo containerlab deploy --topo labs/p075-a-frr/topology.clab.yml
```

若 macOS 没有原生 `containerlab`，`scripts/netopyu_lab.py` 会在本地已有固定版本
Dood 镜像时自动通过 Docker Desktop 运行同一命令，不会回退 mock。

部署完成后，在 macOS 上用实验配置启动 DSH：

```bash
scripts/netopyu-dsh stop
scripts/netopyu-dsh worker-stop
NETOPYU_CONFIG_PATH=config.lab.yaml \
NETOPYU_DSH_BACKEND=pragmatic \
NETOPYU_DSH_ENABLE_DESTRUCTIVE=1 \
scripts/netopyu-dsh start
```

页面测试请求示例：

```text
这是 netopyu-p075a 本地实验，不是真实设备。请使用
lab-ospf-path-remediation Skill 检查 branch-r1 到 DC 的主备 OSPF 路径，
将 branch-r1 的 eth2 OSPF cost 从 20 调整为 10。
必须先读取配置、确认两个 Full 邻居并执行 branch-to-dc 基线探测；
写入时 verification_probe_id 必须为 branch-to-dc，并等待我审批 L0 计划。
```

园区 + IDC 扩展位于 `labs/p075-a-campus-idc/`：5 台 FRR、Erin/Bob 两个终端、
CRM/Wiki 两个 HTTP 应用。它把新员工用例落到真实容器数据面；接口状态表示 NAC
enforcement，应用服务器的精确源 `/32` 策略表示 RBAC enforcement，最终验证必须得到
员工终端实际 HTTP 响应。部署 topology 后运行：

```bash
NETOPYU_CONFIG_PATH=config.campus-idc-lab.yaml \
NETOPYU_DSH_BACKEND=pragmatic \
python scripts/campus_idc_e2e.py
python scripts/campus_idc_e2e.py --exercise-rollback
python scripts/campus_idc_e2e.py --reset-only
```

常驻 DSH 还需启动 pragmatic DC peer，并把 peer URL 提供给 DSH。脚本默认在测试后恢复
Erin 的初始拒绝状态；`--leave-provisioned` 才保留授权。该机制验证真实 Linux 路由和
HTTP 数据面，但不冒充真实 RADIUS、802.1X、IAM 或厂商设备。

```bash
export NETOPYU_CONFIG_PATH="$PWD/config.campus-idc-lab.yaml"
export NETOPYU_DSH_BACKEND=pragmatic
export NETOPYU_DSH_ENABLE_DESTRUCTIVE=1
export NETOPYU_DSH_LOCAL_DC_PORT=8766
scripts/netopyu-dsh dc-peer-start
scripts/netopyu-dsh start
```

### P0.75-B 典型小型现网

`labs/p075-b-small-production/` 是当前推荐的完整本地实验：10 台 FRR 网络设备与
10 个终端/服务，覆盖有线办公、企业无线、访客无线、双园区核心、双安全出口、双 ISP、
IDC、运维基础设施、DMZ 和模拟 Internet。企业内部运行 OSPF，出口运行 eBGP；11 条
清单探测同时验证允许路径与拒绝路径，HTTP 探测验证应用层结果。

```bash
python scripts/netopyu_lab.py \
  --manifest labs/p075-b-small-production/lab.yaml \
  deploy --approve-local-lab
python scripts/small_production_lab.py reset --approve-local-lab
python scripts/small_production_lab.py verify
python scripts/small_production_lab.py exercise-failover --approve-local-lab
python scripts/campus_idc_e2e.py \
  --manifest labs/p075-b-small-production/lab.yaml \
  --config config.small-production-lab.yaml \
  --exercise-rollback
```

DSH UI 使用 `config.small-production-lab.yaml`。该实验实际验证协议收敛、容器转发、
HTTP、L0 审批、独立后置验证与回滚；FRR 的 `secure-wan-edge` 只表示安全域路由角色，
不冒充真实状态防火墙、NAT、IPS、无线 RF、802.1X、ASIC 或厂商 CLI。

拓扑和路径查询不再从设备配置推断。`lab.yaml` 额外声明 7 个安全域、20 个节点的角色、
26 条双端链路及 52 个接口地址，并在加载时与 Containerlab wiring 做集合精确匹配。
DSH 中的 `lab-deterministic-path-query` Skill 只允许使用以下只读工具：

- `lab_get_topology_graph`：返回审核后的节点、接口、地址、区域、链路和模拟真实性边界；
- `lab_get_endpoint`：区分 endpoint 与 device，并返回唯一接入关系；
- `lab_trace_path`：执行 manifest-bound traceroute，逐跳解析节点、入接口和链路；
- `lab_get_enforcement_path`：合并用户准入、应用策略和实际路径证据。

任一跳点地址未知、相邻关系无法由清单证明或目标未到达时，路径返回 `fail_closed=true`，
模型不得补全。Erin→CRM 的当前主路径应为
`erin-client → access-wired-1 → campus-core-1 → idc-leaf-1 → crm-server`；
内部路径不经过 `security-edge-*`。

L0 成功要求 fresh running-config 和预声明流量探测同时通过。失败时 provider 使用
执行会话快照差异恢复，并由 Runtime 独立重读配置证明恢复；否则进入
`manual_intervention_required`。厂商 CLI、ASIC、性能与无线 RF 不属于本阶段。

### 模型使用策略

- 默认使用 `qwen3.5:27b` 进行网络工具会话。
- `qwen2.5:7b` 的本地资格测试未通过自主变更要求：它会跳过指定 Skill、错误选择子代理、在成功后重复发起相同破坏性调用，并错误解释审批拒绝。
- 7B 可以用于只读查询、分类或候选意图生成，但不能独立驱动可变更网络流程。
- 模型只提出候选意图；Network Runtime 决定该意图能否安全进入效果层。

Network L0 Skill 能保证“已校验并经审批的具体计划”按合同执行和验证，不能保证模型提出的计划天然等于用户真实业务意图。

### 安全默认值

- 默认 backend 为 `mock`；pragmatic 模式缺少真实来源时 fail closed。
- 默认只暴露只读工具。
- DSH 写工具必须显式设置 `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1`；Hermes 使用 `NETOPYU_HERMES_ENABLE_DESTRUCTIVE=1`。
- 环境变量不能绕过 DSH 的单次卡片审批或 Hermes 的用户 slash command。
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
scripts/netopyu-hermes doctor
scripts/netopyu-hermes compare
scripts/netopyu-hermes test
python scripts/netopyu_lab.py preflight
python scripts/netopyu_lab.py verify
```

可变运行时数据默认位于 `~/Library/Application Support/NetOpYuAgent/dsh-runtime`，可用 `NETOPYU_DSH_RUNTIME` 覆盖。运行时 SQLite、日志、虚拟环境和 IDE 文件不会提交到 Git。

### 验证

```bash
scripts/netopyu-dsh retirement
```

该命令是项目主门禁，覆盖 Python、Node 插件语法、Harness 边界、HITL、A2A、Network Runtime、Skill 投影、检索质量、Worker 并发/恢复和破坏性操作策略。`scripts/netopyu-hermes test` 额外覆盖 Hermes 官方 PluginContext 表面、slash 审批、nonce 隐藏、A2A continuation 与 DSH/Hermes Runtime 不变量一致性。

### 文档

- [ARCHITECTURE.md](ARCHITECTURE.md)：权威架构边界、依赖规则与架构决策；
- [HLD.md](HLD.md)：高层设计、组件、部署和端到端数据流；
- [LLD.md](LLD.md)：模块、接口、状态机、合同与异常处理；
- [SSD.md](SSD.md)：系统规格、安全设计、威胁模型和验收标准。

---

## English

### Project scope

NetOpYuAgent is a harness-adaptable network-domain plugin and deterministic execution runtime. [DeepSeek Harness (DSH)](https://github.com/deepseek-ai/deepseek-harness) remains the primary platform, with an additional public-API adapter for [Hermes Agent](https://github.com/NousResearch/Hermes-Agent).

DSH or Hermes owns sessions, model calls, tools, UI/CLI, Skills, and general orchestration. NetOpYuAgent does not implement another general-purpose harness. It contributes the network capabilities that must remain reliable:

- Domain L1 Skills for diagnosis, clarification, cross-domain collaboration, and workflow orchestration;
- Network L0 Skills for validation, risk assessment, preflight, approval binding, one-shot execution, verification, compensation/rollback, and audit;
- LAN, DC, WAN, and pragmatic network tools;
- DSH/Hermes adapters, a shared Python Worker, A2A provider, scoped memory, capability retrieval, and offline evaluation.

> Status: P0 migration and the P0.5 mock prototype are complete. The P0.75-A FRR/Containerlab backend is implemented and fails closed when its Linux lab dependencies are unavailable. It validates architecture, safe execution, control-plane behavior, and container forwarding, not absolute correctness on production hardware. Real devices, enterprise identity, change windows, HA, backup/restore, and production SLOs belong to P1.

### Layer terminology

| Name | Responsibility |
|---|---|
| Harness Platform Layer | DSH or Hermes for models, sessions, UI/CLI, tools, and interaction |
| Harness Adapter | Projects domain capabilities without owning network effect semantics |
| NetOpYu Domain Layer | Shared network-domain capabilities below either harness |
| Domain L1 Skill | Generalized, model-assisted business Skill for reasoning and orchestration |
| Network L0 Skill | Versioned, model-independent effect contract |
| Network Runtime | Safety runtime that compiles and executes Network L0 Skills |

### P0.5 completion scope

The local mock scope includes DSH and Hermes harness adapters, versioned Network L0 Skills, strict parameters and provenance, immutable intent/plan/contract hashes, harness-specific user approval bound to one Runtime nonce, execution-time revalidation, typed independent postconditions, contractual compensation, a tamper-evident SQLite journal, persistent Worker recovery, A2A discovery and continuations, a loopback-only DC peer, scoped memory, large-result paging, capability retrieval, and a complete retirement gate with 148 tests, 32 subtests, and 7/7 end-to-end checks.

### Hermes adapter

Install Hermes separately, then link and run the repository plugin:

```bash
scripts/netopyu-hermes install
hermes plugins enable netopyu
scripts/netopyu-hermes worker-start
NETOPYU_HERMES_PROFILE=lan \
NETOPYU_HERMES_ENABLE_DESTRUCTIVE=1 \
NETOPYU_HERMES_OPERATOR_ID=local:steven \
scripts/netopyu-hermes run
```

A mutating tool only prepares a plan. The model never receives the execution nonce. The operator must type the exact `/netopyu-approve PLAN_ID FULL_PLAN_HASH` slash command. `netopyu_skill_catalog` and `netopyu_skill_view` expose canonical Skills and start reviewed workflows; read results become prerequisite evidence for guarded writes. Adapter parity can be tested without a Hermes installation through `scripts/netopyu-hermes compare` and `scripts/netopyu-hermes test`.

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
scripts/netopyu-dsh worker-stop
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

### P0.75-A FRR + Containerlab lab

The reviewed `labs/p075-a-frr/lab.yaml` declares two FRR routers, redundant OSPF
links, endpoints, probes, and fault targets. Lab writes use the same L0 plan and
approval path as pragmatic devices, but add a strict FRR command allowlist,
fresh configuration reads, a predeclared traffic probe, provider snapshot
compensation, and exact normalized restoration evidence.

```bash
python scripts/netopyu_lab.py preflight
python scripts/netopyu_lab.py deploy --approve-local-lab
python scripts/netopyu_lab.py verify
python scripts/netopyu_lab.py exercise-failover --approve-local-lab
```

On Apple Silicon, deploy from the included Containerlab Dood devcontainer. The
macOS DSH process can then control the sibling lab containers through its
bounded Docker provider. Start DSH with `NETOPYU_CONFIG_PATH=config.lab.yaml`,
`NETOPYU_DSH_BACKEND=pragmatic`, and the normal destructive-tool switch.

```bash
scripts/netopyu-dsh stop
scripts/netopyu-dsh worker-stop
NETOPYU_CONFIG_PATH=config.lab.yaml \
NETOPYU_DSH_BACKEND=pragmatic \
NETOPYU_DSH_ENABLE_DESTRUCTIVE=1 \
scripts/netopyu-dsh start
```

When native `containerlab` is absent on macOS, `scripts/netopyu_lab.py` can use
the pinned local Dood image through Docker Desktop. It never falls back to mock.

The campus + IDC extension in `labs/p075-a-campus-idc/` contains five FRR
routers, two employee endpoints, and two HTTP applications. Endpoint link state
represents NAC enforcement and an exact source `/32` route on the application
server represents RBAC enforcement. A successful workflow must receive the
actual HTTP response from the employee container.

```bash
NETOPYU_CONFIG_PATH=config.campus-idc-lab.yaml \
NETOPYU_DSH_BACKEND=pragmatic \
python scripts/campus_idc_e2e.py
python scripts/campus_idc_e2e.py --exercise-rollback
python scripts/campus_idc_e2e.py --reset-only
```

The test restores Erin's initial denied state unless `--leave-provisioned` is
specified. This validates real Linux routing and HTTP forwarding, not real
RADIUS, 802.1X, IAM, vendor CLI, or hardware behavior.

For the persistent DSH UI, export `NETOPYU_CONFIG_PATH` pointing to
`config.campus-idc-lab.yaml`, set the pragmatic backend and destructive-tool
projection, start the local DC peer on port 8766, and then start DSH. Every
write remains approval-gated despite the projection switch.

### P0.75-B typical small production network

The recommended complete local lab is `labs/p075-b-small-production/`: ten FRR
devices plus ten endpoints/services across wired, corporate wireless, guest,
dual campus core, dual secure edge, dual ISP, IDC, operations, DMZ, and a
simulated Internet. OSPF and eBGP expectations, eleven positive/negative ICMP
paths, real HTTP results, ISP failover/recovery, L1/L0 execution, and verified
rollback are executable gates. Use `config.small-production-lab.yaml` for DSH.
The secure-edge role models routing and zone boundaries; it is not stateful
firewall, NAT, IPS, RF, 802.1X, ASIC, or vendor-CLI emulation.

Topology and path answers no longer infer wiring from device configuration.
The manifest declares seven zones, typed node roles, 26 exact two-ended links,
and 52 interface addresses; loading proves an exact set match with Containerlab
wiring. `lab-deterministic-path-query` is limited to four read tools:
`lab_get_topology_graph`, `lab_get_endpoint`, `lab_trace_path`, and
`lab_get_enforcement_path`. Traceroute hops are resolved to the manifest node,
ingress interface, and link. An unknown hop, unproved adjacency, or unverified
destination returns `fail_closed=true` and must never be completed by model
inference. The current Erin→CRM primary path is
`erin-client → access-wired-1 → campus-core-1 → idc-leaf-1 → crm-server` and
does not traverse either security-edge node.

### Safety defaults

- `mock` is the default backend; an incomplete pragmatic backend fails closed.
- Only read-only tools are exposed by default.
- DSH mutations require `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1`; Hermes mutations require `NETOPYU_HERMES_ENABLE_DESTRUCTIVE=1`. Both require a fresh, exact-plan user decision.
- Approval is bound to the full plan hash and cannot be reused after any parameter, target, or contract change.
- Grants are consumable once and resist replay/concurrent duplication.
- Success requires fresh independent postcondition evidence.
- Credentials must come from environment or deployment secret systems and must not enter plans, trajectories, or Git.

### Verification

```bash
scripts/netopyu-dsh retirement
```

This is the primary local gate for Python tests, Node/plugin syntax, harness boundaries, HITL, A2A, Network Runtime, Skill projection, retrieval quality, Worker concurrency/recovery, and mutation policy. `scripts/netopyu-hermes test` covers the Hermes PluginContext surface, slash approval, hidden nonces, remote continuations, and DSH/Hermes Runtime invariant parity.

### Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): authoritative boundaries, dependency rules, and decisions;
- [HLD.md](HLD.md): high-level components, deployment, and end-to-end flows;
- [LLD.md](LLD.md): modules, interfaces, state machines, contracts, and failure handling;
- [SSD.md](SSD.md): system specification, security design, threat model, and acceptance criteria.
