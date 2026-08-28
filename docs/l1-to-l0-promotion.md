# L1 → L0 Promotion Pipeline / L1 → L0 下沉流水线

> 中文在前，English follows. Promotion 是辅助开发流程，不是线上授权路径。

## 中文

### 1. 为什么不是“一键自动下沉并执行”

Anthropic Agent Skill 的标准入口是带 YAML frontmatter 的 `SKILL.md`，正文可以包含自然语言流程、脚本和参考资料。自然语言适合表达业务经验，但 API 是否支持独立读取、写是否幂等、什么状态代表成功、失败后如何恢复，不能仅靠语言模型可靠推断。

因此 NetOpYu 把转换拆成两个边界：

```text
L1 SKILL.md + 受信 Capability Catalog
                  │
                  ▼
       Agent 生成候选 / 澄清问题
                  │ untrusted
                  ▼
       Promotion deterministic checks
                  │
                  ▼
      immutable proposal + human review
                  │ no activation
                  ▼
       后续 Provider/故障认证与发布
```

Agent 加速语义抽取和初稿编写；编译器决定候选是否结构完整；人工与故障测试决定它是否有资格发布。任何单次线上请求都不能生成、批准并立即执行一个新 L0。

### 2. 三个输入

1. **L1 Agent Skill**：标准 `SKILL.md`，保留自然语言业务流程，明确参数、工具、风险和停止条件。
2. **Capability Catalog**：由 Provider/API 所有者维护，声明 capability id、observation/effect/compensation 角色、tool、profile、输入和输出 Schema。
3. **L0 candidate**：由人或 Agent 生成的 Atomic、Derived 或 Composite v2 YAML；在验证前一律不可信。

示例：

- `network_runtime/l0/promotion_examples/url1-network-access/SKILL.md`
- `network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml`
- `network_runtime/l0/examples/s1-network-access-grant.yaml`

### 3. 使用

检查 L1：

```bash
scripts/netopyu-l0 promote-inspect \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md
```

生成供任意 Agent/LLM 使用的有界 Prompt Packet：

```bash
scripts/netopyu-l0 promote-prompt \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml \
  --output artifacts/l0-promotion/url1-prompt.json
```

Agent 必须返回一个 L0 YAML 候选，或在信息不足时返回 `NEEDS_CLARIFICATION`；不得猜测 Observer、回滚、字段或默认值。对候选执行交叉检查：

```bash
scripts/netopyu-l0 promote-assess \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --candidate network_runtime/l0/examples/s1-network-access-grant.yaml \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml
```

只有 `ready_for_review` 才能封装：

```bash
scripts/netopyu-l0 promote-package \
  --skill network_runtime/l0/promotion_examples/url1-network-access/SKILL.md \
  --candidate network_runtime/l0/examples/s1-network-access-grant.yaml \
  --capabilities network_runtime/l0/promotion_examples/url1-network-access/capabilities.yaml \
  --output artifacts/l0-promotion/url1-proposal

scripts/netopyu-l0 promote-review \
  --proposal artifacts/l0-promotion/url1-proposal \
  --reviewer network-reviewer \
  --decision approve \
  --reason "API schema, observer and rollback semantics reviewed"
```

`approve` 只写入一次性 review 记录，不注册 Catalog、不产生审批 nonce，也不授予执行权限。

### 4. 自动检查

- 标准 Skill 名称与目录一致、参数和允许工具可解析；
- 所有 L0 required 参数都在 L1 中有说明；
- primary effect tool 在 L1 的 `allowed-tools` 或 `tool_deps` 中；
- Capability id、provider version、role 和 profile 精确匹配；
- API 必填/未知输入、直接参数类型和范围匹配；
- Preflight/Verification/Compensation Verification 只能使用 Observation；
- Snapshot 和 Predicate 字段必须在 API output Schema 声明；
- Compensation 必须使用 compensation 角色；
- L0 风险不得低于 L1；
- L0 v2 严格编译、安全单调、版本/hash/DAG 规则全部通过；
- L1、Capability Catalog、候选、编译结果和 proposal 全部绑定 SHA-256；
- Proposal 文件被修改后不能通过 review。

### 5. 仍需人工/实验认证

转换器不能证明 Provider 的真实行为。发布前仍需认证 API 身份、Observer 独立性和 freshness、职责分离审批、timeout/写结果不确定、错误 postcondition、精确 rollback，以及 DSH/Hermes 的语义入口和模型选择准确率。将来可以在 Runtime UI 中调用同一 Promotion API，但仍只能生成 proposal，不能在线激活。

## English

The Promotion Pipeline turns an Anthropic-standard `SKILL.md`, a trusted API Capability Catalog, and a human/Agent-authored L0 candidate into an immutable review proposal. The Agent accelerates semantic extraction but remains an untrusted candidate producer. Deterministic checks bind L1 parameter/tool coverage, API roles and schemas, L0 compilation, source hashes, and proposal integrity.

The pipeline deliberately cannot activate Runtime or grant execution authority. A human approval records one immutable review decision only. Provider identity, independent observation, approval separation of duty, indeterminate outcomes, rollback, Harness projection, and fault injection still require separate qualification. This keeps natural-language generalization in L1 while preserving deterministic execution semantics in L0.
