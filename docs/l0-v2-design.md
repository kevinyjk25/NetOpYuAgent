# L0 v2 Effect Contract SDK / L0 v2 确定性作用契约

> 中文在前，English follows. 本文描述 L0 v2 authoring/compiler 和生产 Runtime 集成。全部 21 个内置受审写能力已使用编译后的 v2 Contract 作为语义权威；旧 ToolContract、verifier、compensator 仅保留为精确绑定的执行 Adapter。

## 中文

### 1. 目标

L0 v2 同时支持四种表达形式：

| 形式 | 含义 | 运行时行为 |
|---|---|---|
| 原子式 S1 | 一个明确 Effect、独立验证和可选补偿 | 一个不可变 L0 Plan |
| 约束式 S11 | 固定值、缩小范围、增加审批强度 | 编译期展开为独立原子 Contract |
| 扩展式 S11 | 在 S1 上增加参数、Read Preflight、成功谓词和审批强度 | 编译期展开为独立原子 Contract |
| 组合式 S1+S2+… | 多个独立 Effect 组成跨域事务 | 绑定子版本和哈希的 Saga |

继承只存在于 authoring/compiler 阶段。审批和执行时不存在动态父子查找，Runtime 只接收完全展开的 Contract 及其哈希。

### 2. 代码结构

```text
network_runtime/l0/
├── models.py       严格 Pydantic Schema 和编译产物类型
├── compiler.py     继承展开、安全单调检查、子契约哈希绑定
├── catalog.py      多版本/多语义索引、explain/diff/graph、Saga 投影
├── expressions.py  非图灵完备的参数/状态模板渲染器
├── production.py   21 个生产 L0 v2 定义及精确执行 Adapter 绑定
├── runtime_loader.py  Catalog/Adapter 一致性、Resolver 和执行参数门禁
├── cli.py          开发、Promotion 与生产 Catalog 命令
└── examples/
    ├── s1-network-access-grant.yaml
    ├── s11-guest-access-constraint.yaml
    ├── s11-privileged-access-extension.yaml
    ├── s2-application-access-grant.yaml
    └── s111-network-and-application-composite.yaml
```

开发 CLI：

```bash
scripts/netopyu-l0 validate
scripts/netopyu-l0 list
scripts/netopyu-l0 explain network.privileged-access.grant
scripts/netopyu-l0 diff network.access.grant network.guest-access.grant
scripts/netopyu-l0 graph employee.application-access.provision
scripts/netopyu-l0 compile --output artifacts/l0-v2/catalog.json
scripts/netopyu-l0 schema --kind all
scripts/netopyu-l0 runtime-validate
scripts/netopyu-l0 runtime-list
scripts/netopyu-l0 runtime-export --output artifacts/l0-v2/runtime-catalog.json
```

### 3. S1：原子 REST Effect

示例 S1 `network.access.grant@1.0.0` 将 REST URL1 投影为稳定 Capability：

```text
rest.url1.network-access.grant   Actor/写
rest.network-access.get         Observer/独立读
rest.network-access.restore     Actor/恢复
```

S1 声明参数、目标、期望状态、Preflight、Verification、Compensation、审批和失败策略。原子写 Contract 如果没有独立 Preflight/Verification 会被编译器拒绝。URL、认证和 HTTP 细节属于 Provider Adapter，不进入 Skill。

### 4. 约束式 S11

`network.guest-access.grant@1.0.0` 继承 S1，并只允许：

- 固定 `vlan_id=300`；
- 将最长访问时间从 10080 分钟缩短到 480 分钟；
- 为原因增加工单格式；
- 将风险从 medium 提高到 high。

约束派生只能收紧：不能新增 Effect、删除参数、扩大 enum/最大值、降低最小值、取消敏感标记、降低审批或覆盖父级期望状态。

### 5. 扩展式 S11

`network.privileged-access.grant@1.0.0` 仍使用 S1 的同一个 REST Effect，但增加：

- `change_id` 与 `sponsor_id`；
- 工单状态和 Sponsor 身份 Read Preflight；
- 有效期验证谓词；
- `privileged=true` 期望事实；
- critical + dual approval。

扩展只能增加参数、读取和验证，不能替换父 Effect、补偿或失败策略。最终产物是独立、完全展开的 Contract，不依赖运行时继承。

### 6. 组合式 S1+S2

`employee.application-access.provision@1.0.0` 组合：

```text
network-access (S1)
        │ verified
        ▼
application-access (S2)
        │ verified
        ▼
end-to-end checkpoint
```

编译产物为每个步骤绑定：

- 精确 `skill_id@version`；
- 子 Contract hash；
- Effect Capability；
- Compensation Capability；
- 依赖关系和参数映射。

组合 Contract 可直接投影为现有 `SagaDefinition`。当前 Saga 仍要求每个子 L0 Plan 使用自己的真实审批；组合层不能制造或绕过审批。

### 7. 安全规则

编译器当前强制：

1. 严格 Schema，未知字段失败；
2. Skill 使用精确语义版本；
3. 同一个 Capability 可以对应多个语义 Contract；
4. 约束/扩展不能降低父审批强度；
5. 参数 enum、范围和长度只能收紧；
6. 固定值必须满足父类型和范围；
7. 父期望状态不能被改写；
8. Preflight ID、步骤 ID 和 Checkpoint ID 唯一；
9. Composite 参数必须满足子 Contract；
10. Composite 审批不能弱于任一子 Contract；
11. 不允许嵌套 Composite，防止不可审计的递归事务；
12. 编译产物以 canonical JSON 计算 SHA-256；
13. Saga 步骤绑定子 Contract hash，父 Skill 更新不会改变已编译组合语义。
14. 模板中的 argument/preflight/input 引用必须在编译期可解析；
15. Composite 的直接输入必须在类型、范围、enum、pattern 和 resolver 上满足子参数合同。

### 8. 多版本与多语义 Registry

Runtime Registry 使用：

```text
(skill_id, version) -> Contract
tool/capability      -> [S1, S11, ...]
```

相同 Skill 可并存多个版本；未指定版本时选最高语义版本。一个 Tool 存在多个 Skill 语义时，禁止仅依据 Tool 名猜测，必须提供精确 `l0_skill_id`。这避免通用 URL1 在 S1/S11 之间发生静默误选。

### 9. 生产运行集成

已完成：

- 严格 authoring 类型；
- 原子、约束、扩展、组合编译；
- 多版本 Catalog；
- 同 Capability 多语义索引；
- 安全单调检查；
- 子版本/哈希绑定；
- SagaDefinition 投影；
- CLI explain/diff/graph/compile；
- 正向和负向单元测试；
- Registry 多版本/歧义拒绝兼容改造；
- 21/21 个现有生产写工具的编译 v2 Contract；
- 精确 `RuntimeBinding`：每个 v2 id/version 绑定一个 ToolContract、verifier、可选 compensator 和 profile；
- prepare 与执行前重校验阶段的双重 Contract/Adapter parity gate；
- 非图灵完备表达式渲染和 fail-closed Resolver Registry；
- Provider Effect 只接收由已批准参数按 v2 模板渲染的字段；
- 21/21 个生产 L0 的 L1/L0.5/L0 可读存档、Promotion semantic parity 和精确 round trip；
- Core-72 Runtime 64/64 控制 Oracle 和完整 Python 回归。

旧 ToolContract、verifier 和 compensator 仍被复用，但它们只是经过验证的实现 Adapter。Contract id/version/hash、参数 Schema、desired state、preflight 字段、risk/profile、effect capability 和补偿要求以编译 v2 产物为准；任一投影漂移都会在写前失败关闭。新生产 L0 不得再以裸 v1 注册形式加入。

存量合同的解释索引位于 [`production_trajectories/`](../network_runtime/l0/production_trajectories/INDEX.md)。档案中的 L1/L0.5 从受审 L0 反向 bootstrap，并明确保存来源声明；主门禁重新执行 Promotion 和编译，不把可读档案当作运行权威，也不把 bootstrap 当作模型独立转换准确率。

### 10. 仍然保留的安全边界

`examples/` 中 URL1 REST Capability 是 authoring/Promotion 教学示例，尚未接入真实 Provider，因此不属于上述 21 个生产合同，也不会自动出现在 DSH。离线 Promotion 的人工 approve 同样不会自动发布合同；外部候选仍需 Provider 认证、故障注入和显式 Catalog 发布。企业单/双人审批身份、厂商设备和生产 HA 也仍是 P1 资格认证范围。

L1 自然语言辅助构建入口见 [L1 → L0 Promotion Pipeline](l1-to-l0-promotion.md)。它可以生成和审查候选，但不改变上述执行认证边界。

## English

L0 v2 provides a strict authoring/compiler layer for atomic effects, constraint-only derivations, additive derivations, and composite Sagas. Inheritance is compile-time only. Every derived effect is flattened and hashed before approval; every composite step binds an exact child id, version and contract hash.

The SDK uses strict Pydantic schemas, monotonic-security checks, a multi-version/multi-semantic catalog, human-readable explain/graph commands, and projection into the existing durable `SagaDefinition`. A shared REST capability may back S1 and several S11 semantic contracts, but tool name alone can never choose between them.

All 21 built-in reviewed mutation tools now use compiled `netopyu.io/l0-effect-compiled/v2` artifacts as their semantic authority. Exact Runtime bindings retain the previously qualified ToolContracts, verifiers, and compensators only as implementation adapters. Prepare and execution-time revalidation enforce schema, desired-state, preflight, profile, verifier, and compensation parity; Provider effect arguments are rendered from approved values through a non-Turing-complete expression engine. Unknown resolvers and any projection drift fail closed. Core-72 remains 64/64 for the Runtime path.

The [production trajectory index](../network_runtime/l0/production_trajectories/INDEX.md) preserves readable L1/L0.5/L0 artifacts for 21/21 existing contracts. These L1/L0.5 baselines are explicitly reverse-bootstrapped from reviewed L0. The primary gate reruns Promotion semantic parity and exact compiler round trips without treating the readable archive as execution authority or as evidence of independent model inference accuracy.

The bundled URL1 REST manifests remain authoring/Promotion examples without a real Provider and are not auto-registered. An approved offline promotion is also non-executable until separate Provider qualification, fault injection, and explicit Catalog publication. Enterprise approval identity, vendor-device qualification, and production HA remain P1 boundaries. See the [production migration guide](l0-v2-runtime-migration.md) for the exact authority and compatibility model.

See the [L1 → L0 Promotion Pipeline](l1-to-l0-promotion.md) for the natural-language-assisted candidate workflow. It generates review proposals but does not weaken execution qualification.
