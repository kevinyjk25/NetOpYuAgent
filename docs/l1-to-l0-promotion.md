# L1 → L0 Promotion Pipeline / L1 → L0 下沉流水线

> 中文在前，English follows. Promotion 是辅助开发流程，不是线上授权路径。

## 中文

### 1. 为什么不是“一键自动下沉并执行”

Anthropic Agent Skill 的标准入口是带 YAML frontmatter 的 `SKILL.md`，正文可以包含自然语言流程、脚本和参考资料。自然语言适合表达业务经验，但 API 是否支持独立读取、写是否幂等、什么状态代表成功、失败后如何恢复，不能仅靠语言模型可靠推断。

因此 NetOpYu 把转换拆成三个可审计阶段：

```text
L1 SKILL.md + 受信 Capability Catalog
                  │
                  ▼
 L0.5 StructuredNaturalLanguageSkill
   参数/范围/步骤/风险/Capability 选项
                  │ human-readable, hash-bound
                  ▼
       Agent/人工生成 L0 v2 候选
                  │ untrusted
                  │
                  ▼
       Promotion deterministic checks
                  │
                  ▼
 immutable trajectory + human review
                  │ no activation
                  ▼
       后续 Provider/故障认证与发布
```

L1 保留原始自然语言业务经验，并用一个显式标记的小型 `semantic-intents/v1` YAML 块固定不可猜测的 effect capability、intent kind、targetFields 与 desiredState；L0.5 v3 把该锚点连同参数、约束、流程阶段、风险、停止条件、成功/失败/回滚语义和受信 Capability 选项整理为人可读 YAML，并把 Observation 明确限定到事务 phase；L0 才是机器可执行的严格合同。Agent 加速语义抽取和初稿编写，编译器决定候选是否结构完整，人工与故障测试决定它是否有资格发布。任何单次线上请求都不能生成、批准并立即执行一个新 L0。

### 2. 输入与三阶段产物

1. **L1 Agent Skill**：标准 `SKILL.md`，保留自然语言业务流程，明确参数、工具、风险和停止条件。
2. **Capability Catalog v3**：由 Provider/API 所有者维护，声明 capability id、observation/effect/compensation 角色、tool、profile、输入和输出 Schema；每个 Observation 还必须声明 `observationPhases` 及各阶段不可弱化的最低 `phasePredicates`。
3. **L0.5 Structured Skill v3**：可由工具从 L1 + Catalog 生成，也可由人补充；保持自然语言可读，同时用严格 Schema 固定范围。`semanticIntents` 必须逐字段复制 L1 锚点并按 effect capability 绑定；`preflightObservations`、`successVerificationObservations` 和 `compensationVerificationObservations` 分别限定三个确定阶段。缺少锚点、多个 Effect 无法唯一选择、缺少必要 Observer 或 Compensation 时写入 `unresolvedQuestions` 并阻断 L0。
4. **L0 candidate**：由人或 Agent 依据 L0.5 生成的 Atomic、Derived 或 Composite v2 YAML；在验证前一律不可信。

示例：

- `network_runtime/l0/promotion_examples/url1-network-access/SKILL.md`
- `network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml`
- `network_runtime/l0/promotion_examples/url1-network-access/L0.5.yaml`
- `network_runtime/l0/examples/s1-network-access-grant.yaml`

### 3. 使用

检查 L1：

```bash
scripts/netopyu-l0 promote-inspect \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md
```

生成并先审查 L0.5：

```bash
scripts/netopyu-l0 promote-l05 \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml \
  --output artifacts/l0-promotion/url1-L0.5.yaml
```

生成供任意 Agent/LLM 使用的有界 Prompt Packet：

```bash
scripts/netopyu-l0 promote-prompt \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --l05 artifacts/l0-promotion/url1-L0.5.yaml \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml \
  --output artifacts/l0-promotion/url1-prompt.json
```

Agent 必须返回一个 L0 YAML 候选，或在信息不足时返回 `NEEDS_CLARIFICATION`；不得猜测 Observer、回滚、字段或默认值。对候选执行交叉检查：

```bash
scripts/netopyu-l0 promote-assess \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --l05 artifacts/l0-promotion/url1-L0.5.yaml \
  --candidate network_runtime/l0/examples/s1-network-access-grant.yaml \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml
```

只有 `ready_for_review` 才能封装：

```bash
scripts/netopyu-l0 promote-package \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --l05 artifacts/l0-promotion/url1-L0.5.yaml \
  --candidate network_runtime/l0/examples/s1-network-access-grant.yaml \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml \
  --output artifacts/l0-promotion/url1-proposal

scripts/netopyu-l0 promote-review \
  --proposal artifacts/l0-promotion/url1-proposal \
  --reviewer network-reviewer \
  --decision approve \
  --reason "API schema, observer and rollback semantics reviewed"
```

Proposal 固定保存：

```text
00-capability-catalog.yaml
01-L1-SKILL.md
02-L0.5.yaml
03-L0-authoring.yaml
04-L0-compiled.json
trajectory.json
report.json
```

`trajectory.json` 按顺序记录每阶段格式、文件 hash 和 `previousSha256`，并计算整体 `trajectoryHash`。任一阶段被修改、阶段顺序变化或前后 hash 断链都无法 review。`approve` 只写入一次性 review 记录，不注册 Catalog、不产生审批 nonce，也不授予执行权限。

### 4. 自动检查

- 标准 Skill 名称与目录一致、参数和允许工具可解析；
- L0.5 精确绑定 L1 与 Capability Catalog hash，且阶段链连续；
- L1、L0.5 与 compiled L0 的 effect capability、intent kind、targetFields、desiredState 必须逐字段相等；缺失、增加或漂移均失败关闭；
- L0.5 不得删除/增加 L1 参数、扩大 profile、降低风险或移除审批；
- L0 必须使用 L0.5 允许的 Effect/Observation/Compensation，且不得留下未解决问题；
- 所有 L0 required 参数都在 L1 中有说明；
- primary effect tool 在 L1 的 `allowed-tools` 或 `tool_deps` 中；
- Capability id、provider version、role 和 profile 精确匹配；
- API 必填/未知输入、直接参数类型和范围匹配；
- Preflight/Verification/Compensation Verification 只能使用 Observation，且必须分别声明 `preflight`、`success_verification`、`compensation_verification` phase；
- Snapshot 和 Predicate 字段必须在 API output Schema 声明；
- Compensation 必须使用 compensation 角色；
- L0 风险不得低于 L1；
- L0 v2 严格编译、安全单调、版本/hash/DAG 规则全部通过；
- L1、Capability Catalog、候选、编译结果和 proposal 全部绑定 SHA-256；
- Proposal 文件被修改后不能通过 review。

#### 4.1 需求级语义覆盖门禁

字段和 hash 自洽不等于自然语言语义完整。Promotion 现在把 L1 的参数、
profile、风险、审批、前置条件、参数不得推断、preflight、单次 Effect、独立
验证、未知结果处理和补偿要求拆成带稳定 `requirementId` 的原子要求，并在
`report.json.semanticCoverage` 中保存三级证据：

```text
L1 source text/path → L0.5 path/value → compiled L0 path/predicate/capability
```

每条要求的判定为 `preserved`、`strengthened`、`weakened`、`missing`、
`ambiguous` 或 `non_machine_verifiable`。安全关键要求出现 `missing`、
`weakened` 或 `ambiguous` 会阻断 package/review；L0 出现 L1 未声明的额外
Effect 同样阻断。每条失败都携带 `fix.file`、`fix.path` 和 `fix.hint`，因此
使用者能直接判断应修改 L1、L0.5、L0 authoring 还是 Capability Catalog。

查看摘要和待处理项：

```bash
scripts/netopyu-l0 promote-assess ... | jq '.semanticCoverage.summary'
scripts/netopyu-l0 promote-assess ... | \
  jq '.semanticCoverage.requirements[] | select(.verdict != "preserved" and .verdict != "strengthened")'
```

`semanticCoverage.gate=passed` 只表示确定性提取到的要求通过，不是任意自然
语言等价证明。无法机器执行的业务规则必须保留为
`non_machine_verifiable` 并由独立 reviewer 认证；LLM 只能建议拆分和映射，
不能自行关闭安全门禁。

#### 4.2 Phase-typed Observation 门禁

`role: observation` 只说明 Capability 是读取能力，不能说明它适合前置检查、
成功验证还是补偿验证。Catalog v3 将阶段和最低证明谓词都作为 Provider owner
审查的显式合同：

```yaml
apiVersion: netopyu.io/capability-catalog/v3
capabilities:
  - id: network.resource.get
    role: observation
    observationPhases: [preflight]
    phasePredicates:
      preflight: [{field: facts, operator: exists}]
  - id: network.resource.verify
    role: observation
    observationPhases: [success_verification]
    phasePredicates:
      success_verification: [{field: passed, operator: equals, expected: true}]
  - id: network.resource.rollback-verify
    role: observation
    observationPhases: [compensation_verification]
    phasePredicates:
      compensation_verification: [{field: restored, operator: equals, expected: true}]
```

模型收到的可信 Catalog Packet 包含同一 phase 元数据；候选物化边界先拒绝
错误选择，Promotion 再按 L0 实际位置独立复核。候选可以在 Catalog 已声明
输出字段上增加更强谓词，但必须包含全部 `phasePredicates`；仅检查 `passed`
字段存在不能替代 `passed == true`。未声明 phase 返回
`CAPABILITY_PHASE_UNDECLARED`，选错 phase 返回
`CAPABILITY_PHASE_MISMATCH`，遗漏/改写最低证明返回
`CAPABILITY_PHASE_PREDICATE_MISMATCH`；L0.5 允许列表或 workflow 与 L0 不一致也会以
`L05_*_PHASE_*` finding 失败关闭。旧 Catalog v1 与 L0.5 v2 仍可被解析以便
诊断；Catalog v2 虽有 phase 类型但没有受信证明，因此同样不能通过当前
Promotion。应重新运行 `promote-l05` 并由 Provider owner 审查三个 phase 及其
最低证明，而不是由模型猜测。

对已保存的 210 条 9B Observation 做当前 Runtime 重放时没有调用模型：
203/203 条历史全语义 exact-ready 候选保持可审，唯一一条选错 preflight
Capability 的历史 false-ready 被新增阻断，exact-ready 回归为 0。该重放只
证明这次确定性门禁修复已知缺口，不是新的模型准确率证据。

最终 v7 又完整运行同一公开 210 条矩阵：208 个成功返回的 proposal 全部达到
全语义 exact 和当前 Runtime-ready，0 个 changed/blocked；另两条是本机 Ollama
`model_transport` 超时，没有产生语义候选。Runner 现会逐例 checkpoint 这类
transport 故障并继续，而不会误报为协议语义失败或退出整批。总体 99.05% 和
p50/p95 31.528/79.384 秒必须保留两个超时，且仍只是反向公开单次回归，不是
独立正向资格或生产成功概率。

P2.5-D 不删除任何 L1/Catalog 语义，只把重复的缩进表示改为版本化紧凑 JSON
packet，并将 packet/序列化身份纳入 authoring protocol digest。完整 210 条的
Prompt 字节相对 v7 等价格式下降 18.98%。同一 21-family direct-en smoke 仍为
21/21 全语义 exact/Runtime-ready，输入 token 下降 19.81%，p50/p95 下降
18.10%/14.45%；双能力族 20 条全包装 smoke 也保持 20/20 exact/ready，输入
token 下降 16.32%，p50/p95 下降 11.81%/13.84%。Runner 同时记录有限声明的模型注册表 preflight，并在连续两次
transport 故障后先保存证据再暂停；恢复跳过旧失败而不会静默改写历史。

### 5. 仍需人工/实验认证

转换器不能证明 Provider 的真实行为。发布前仍需认证 API 身份、Observer 独立性和 freshness、职责分离审批、timeout/写结果不确定、错误 postcondition、精确 rollback，以及 DSH/Hermes 的语义入口和模型选择准确率。将来可以在 Runtime UI 中调用同一 Promotion API，但仍只能生成 proposal，不能在线激活。

### 6. 存量 L0 的反向可读基线

新 Skill 使用真正的正向流程：人工领域经验先写 L1，经审查形成 L0.5，再生成 L0 candidate。存量 21 个合同的历史起点已经是受审 L0，因此项目从权威 L0 **反向 bootstrap** L1/L0.5，只用于补齐可读性和可解释性。

每份存量档案仍运行完整 `assess_promotion()`：要求 L1/L0.5 边界不被最终 L0 扩大；去除 proposal-only 来源 labels 后，Promotion 编译结果必须与生产 Contract 语义 hash 一致；原始 authoring 重新编译后的完整 contract hash 也必须精确一致。该结果验证 Promotion 的结构闭环和现有合同的可读投影，不等于模型对任意自然语言的转换准确率。索引见 [生产 L0 轨迹](../network_runtime/l0/production_trajectories/INDEX.md)。

## English

The Promotion Pipeline preserves three explicit stages: the original natural-language `L1 SKILL.md`, a schema-valid but human-readable `L0.5 StructuredNaturalLanguageSkill` v3, and strict L0 authoring/compiled contracts. L1 keeps prose natural but carries one visibly marked, compact `semantic-intents/v1` YAML anchor for semantics that must never be guessed. L0.5 records an exact capability-scoped copy alongside parameters, constraints, workflow phases, risk, stop conditions, outcomes, and trusted capability options. Capability Catalog v3 assigns each Observation to `preflight`, `success_verification`, and/or `compensation_verification` and binds minimum `phasePredicates` for each phase. Candidates may add stronger predicates over declared outputs but cannot omit or alter the minimum proof. Catalog v1/v2 and L0.5 v2 remain readable for migration diagnostics but cannot pass current Promotion. Missing or drifted intent, ambiguous effects, or missing observation/compensation semantics remain unresolved and block promotion.

An immutable package stores the capability catalog, `01-L1-SKILL.md`, `02-L0.5.yaml`, `03-L0-authoring.yaml`, `04-L0-compiled.json`, and a `trajectory.json` hash chain. Deterministic checks prevent L0.5 from drifting from L1 or L0 from widening L0.5. Any file or link tampering blocks review. The Agent accelerates semantic extraction but remains an untrusted candidate producer.

The pipeline deliberately cannot activate Runtime or grant execution authority. A human approval records one immutable review decision only. Provider identity, independent observation, approval separation of duty, indeterminate outcomes, rollback, Harness projection, and fault injection still require separate qualification. This keeps natural-language generalization in L1 while preserving deterministic execution semantics in L0.

Promotion also emits a requirement-level `semanticCoverage` matrix. Stable
requirement IDs link the original L1 clause to L0.5 evidence and concrete
compiled-L0 paths. Missing or weakened requirements block promotion; ambiguous
safety-critical mappings and undeclared extra effects block it as well. Every
row includes an exact fix file/path/hint. `non_machine_verifiable` rules remain
visible for independent review. A passing gate covers deterministically
extracted requirements only; it is not a proof of equivalence for arbitrary
natural language, and an LLM cannot close the gate by assertion.

New Skills use the genuine forward path: domain authors write L1, review L0.5, and then produce an L0 candidate. The 21 existing contracts already began as reviewed L0, so their readable L1/L0.5 baselines are explicitly reverse-bootstrapped for explainability. Each archive still reruns `assess_promotion()`, requires semantic parity after proposal-only labels are removed, and requires an exact full contract hash after recompiling authoring. This validates structural conversion closure and readable projection of existing contracts; it is not a measurement of model accuracy on arbitrary prose. See the [production L0 trajectory index](../network_runtime/l0/production_trajectories/INDEX.md).

P2.5-D preserves all L1 and trusted-Catalog semantics while replacing repeated pretty-printed transport with a versioned compact JSON packet. Packet and serialization identity are bound into the authoring-protocol digest. Across the 210 public prompts this reduces representation bytes by 18.98%; a same-artifact 21-family direct-English smoke retained 21/21 exact/Runtime-ready outcomes while reducing input tokens by 19.81% and p50/p95 by 18.10%/14.45%. A two-family 20-wrapper smoke also retained 20/20 exact/ready while reducing input tokens by 16.32% and p50/p95 by 11.81%/13.84%. Registry preflight evidence has an explicitly narrow claim, and the runner checkpoints then pauses after two consecutive transport failures; resume never rewrites the failed observations.
