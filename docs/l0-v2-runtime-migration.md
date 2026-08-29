# L0 v2 生产运行迁移 / L0 v2 Production Runtime Migration

## 中文

### 1. 迁移结论

现有 21 个受审写能力已全部迁移为编译后的 L0 v2 Contract，覆盖 Network、Service、LAN、DC、WAN、Fabric 和跨层 enforcement。生产语义的唯一权威是 `netopyu.io/l0-effect-compiled/v2`；不再允许新增裸 v1 L0 注册。

`ToolContract`、verifier 和 compensator 没有被重复重写。它们作为已经通过故障测试的执行 Adapter，通过精确 `(l0_id, version)` 绑定到 v2 Contract。Adapter 不能定义或扩大 v2 语义，任何参数、desired state、preflight、profile、验证器或补偿关系不一致都会在 Effect 前失败关闭。

### 2. 分层和权威关系

```text
DSH / Hermes + Domain L1 Skill
                 │ 候选意图与参数
                 ▼
Compiled L0 v2 Contract                 语义权威
  id/version/hash/schema/risk/effect/
  preflight/desired-state/verification
                 │ 精确 RuntimeBinding
                 ▼
Qualified execution adapters            实现边界
  ToolContract + verifier + compensator
                 │ Capability SPI
                 ▼
Provider Gateway                         协议边界
  MCP / API / CLI / Containerlab
```

Harness 和模型可以选择 L1、提出参数并请求执行，但不能修改 v2 Contract。Runtime 拥有计划、审批、重校验、结果判定、补偿和审计；Provider Gateway 只负责把已批准 Effect 适配到外部系统。

### 3. 确定性执行流程

1. Runtime 通过精确 L0 id/version 取得编译 Contract，禁止仅凭 tool 名猜测。
2. 参数先经过既有实体/来源校验，再通过 v2 参数 Schema 校验。
3. v2 模板生成 target、desired state 和 preflight 投影，形成不可变 intent/plan/contract hash。
4. Runtime 校验编译 Contract 与 ToolContract、verifier、compensator、profile 的一致性。
5. 操作员审批完整计划；审批绑定一次性 nonce 和上述 hash。
6. 执行前重新读取状态，并再次校验 v2/Adapter parity 和已批准参数。
7. 非图灵完备表达式引擎只允许从 `arguments`、`preflight`、`plan`、`intent` 和 `verification` 白名单根读取值；禁止函数、运算符和代码执行。
8. Provider 只接收按 v2 effect request 模板渲染的参数，模型原始参数不会被直接透传。
9. 独立 verifier 判定结果；失败按精确 compensator 恢复，或进入人工介入终态。

Resolver Registry 只执行显式注册的纯转换。未知 Resolver、未知表达式路径、缺失必需值或审批后投影漂移都会失败关闭。

### 4. 代码位置

| 文件 | 作用 |
|---|---|
| `network_runtime/l0/production.py` | 21 个生产 v2 定义和精确 RuntimeBinding |
| `network_runtime/l0/runtime_loader.py` | 编译 Contract/Adapter parity、参数和 Effect 门禁 |
| `network_runtime/l0/expressions.py` | 受限模板表达式和 Effect 参数渲染 |
| `network_runtime/l0_skills.py` | v2 Contract 到 Runtime Registry 的兼容投影 |
| `network_runtime/engine.py` | prepare、重校验、Provider dispatch、验证和补偿 |
| `tests/test_l0_production_v2.py` | 21/21 覆盖、CLI、表达式和 fail-closed 回归 |
| `network_runtime/l0/production_trajectories/` | 21 份 L1/L0.5/L0 可读档案和 hash 轨迹 |
| `network_runtime/l0/production_trajectory.py` | 档案生成、Promotion parity、round-trip 与防篡改校验 |

### 5. 操作与验收

```bash
scripts/netopyu-l0 runtime-validate
scripts/netopyu-l0 runtime-list
scripts/netopyu-l0 runtime-export --output artifacts/l0-v2/runtime-catalog.json
scripts/netopyu-l0 runtime-trajectories-validate
python -m pytest -q
scripts/netopyu-dsh compare-runtime --iterations 10
```

当前本地门禁结果是 21 个 Contract、21 个 Binding、21 个受审写工具、21 个可读轨迹、21 个 Promotion-ready 语义投影和 21 个精确 round trip；Python 为 239 tests + 81 subtests；Core-72 为 DSH-only 5/64、DSH + Runtime 64/64。它证明本地确定性控制覆盖，不代表生产环境绝对 100% 正确。

### 6. 存量 L0 可读轨迹

每个生产 L0 在 `production_trajectories/<skill-id>/` 保存：Capability Catalog、L1 自然语言、L0.5 结构化自然语言、L0 authoring、L0 compiled、`trajectory.json` 和 `report.json`。验证器重新运行 L1/L0.5 Promotion 约束，去除仅用于 proposal 的来源 labels 后要求语义 hash 与生产 Contract 一致；同时从 authoring 重新编译并要求完整 contract hash 与生产 Catalog 完全相等。

存量 L1/L0.5 是从已受审 L0 反向 bootstrap 的解释基线，而不是历史人工 L1 或模型独立推导结果。这样既不丢失可读性，也不夸大“自然语言自动转换”的证据。新 Skill 仍按真正的正向 `L1 → L0.5 → L0 candidate` 流程开发。

### 7. 明确保留的边界

- `network_runtime/l0/examples/` 的 URL1 REST Capability 仍是教学示例，没有真实 Provider，不属于生产 Catalog。
- L1 → L0 Promotion 的 approve 只产生不可变审查记录，不自动发布或激活。
- 外部 L0 候选仍需 Provider 身份/Schema 认证、故障注入、回滚认证和显式发布。
- 企业审批身份、双人策略、厂商设备、跨主机 HA、远端不可变审计和生产 SLO 仍需 P1 资格认证。

## English

### 1. Migration result

All 21 existing reviewed mutation capabilities now use compiled L0 v2 contracts across Network, Service, LAN, DC, WAN, Fabric, and cross-layer enforcement. `netopyu.io/l0-effect-compiled/v2` is the sole production semantic authority; new raw v1 L0 registrations are not permitted.

Previously qualified ToolContracts, verifiers, and compensators are retained as implementation adapters through exact `(l0_id, version)` bindings. They cannot define or widen v2 semantics. Any drift in parameters, desired state, preflight, profile, verifier, or compensation fails closed before the effect.

### 2. Authority model

DSH/Hermes and Domain L1 produce intent candidates. The compiled L0 v2 contract owns identity, version, hash, schema, risk, effect, preflight, desired state, and verification semantics. An exact RuntimeBinding connects that authority to qualified execution adapters. The Capability SPI and Provider Gateway then adapt the approved effect to MCP, API, CLI, or Containerlab.

The harness and model cannot mutate the contract. Runtime owns planning, approval, revalidation, outcome classification, compensation, and audit; the Provider Gateway owns protocol adaptation only.

### 3. Deterministic execution

Runtime resolves an exact L0 id/version, validates domain provenance and the v2 parameter schema, renders target/desired-state/preflight projections, and binds immutable intent/plan/contract hashes. It checks adapter parity both during prepare and immediately before execution. A non-Turing-complete renderer reads only approved whitelist roots and sends only the v2 effect-request fields to the Provider. Independent verification determines success; contractual compensation restores state or the operation enters manual intervention.

Unknown resolvers, invalid expression paths, missing required values, or post-approval projection drift fail closed.

### 4. Qualification and boundaries

Use `scripts/netopyu-l0 runtime-validate`, `runtime-trajectories-validate`, the full Python suite, and Core-72 as the local gates. The current result is 21 contracts/bindings/reviewed writes, 21 readable trajectories, 21 Promotion-ready projections, 21 exact compiler round trips, 239 tests plus 81 subtests, and 64/64 Runtime control oracles versus 5/64 for DSH only.

Each production L0 archive contains the Capability Catalog, natural-language L1, structured-natural-language L0.5, L0 authoring and compiled artifacts, `trajectory.json`, and `report.json`. Validation reruns Promotion parity after removing proposal-only provenance labels, recompiles authoring, requires the exact production contract hash, and verifies every file and predecessor link.

The existing L1/L0.5 artifacts are reverse-bootstrapped explanation baselines from reviewed L0—not historical human-authored L1 or evidence that a model independently inferred the contract. New Skills continue to use the genuine forward `L1 → L0.5 → L0 candidate` workflow.

The URL1 REST examples still have no real Provider and are not production registrations. Offline promotion approval does not publish or activate a contract. External candidates require Provider/schema qualification, fault injection, rollback certification, and explicit publication. Enterprise approval identity, vendor-device qualification, distributed HA, remote immutable audit, and production SLOs remain P1 work.
