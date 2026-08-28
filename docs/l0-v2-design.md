# L0 v2 Effect Contract SDK / L0 v2 确定性作用契约

> 中文在前，English follows. 本文描述可运行的 L0 v2 authoring/compiler 原型；现有 v1 Runtime 执行内核继续作为兼容生产路径，直到 v2 Catalog 完成 Provider 联调和同等故障认证。

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
├── cli.py          validate/list/show/explain/diff/graph/compile/schema
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

兼容 v1 Registry 已改为：

```text
(skill_id, version) -> Contract
tool/capability      -> [S1, S11, ...]
```

相同 Skill 可并存多个版本；未指定版本时选最高语义版本。一个 Tool 存在多个 Skill 语义时，禁止仅依据 Tool 名猜测，必须提供精确 `l0_skill_id`。这避免通用 URL1 在 S1/S11 之间发生静默误选。

### 9. 当前完成边界

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
- v1 Registry 多版本/歧义拒绝兼容改造。

尚未把示例 REST Capability 接入真实 Provider，因此示例 Catalog 当前是 authoring/compiler 原型，不会自动出现在 DSH 页面。进入实际运行前还需要：

1. OpenAPI/MCP Capability Provider；
2. 表达式解析器和参数 Resolver 插件；
3. 编译 Catalog 到 Runtime Plan 的 Loader；
4. DSH/Hermes 为每个语义 Contract 投影独立逻辑入口；
5. 单/双人审批身份实现；
6. 对 S1/S11/Saga 运行与 Core-72 同等级别的故障认证。

L1 自然语言辅助构建入口见 [L1 → L0 Promotion Pipeline](l1-to-l0-promotion.md)。它可以生成和审查候选，但不改变上述执行认证边界。

## English

L0 v2 introduces a strict authoring/compiler layer for atomic effects, constraint-only derivations, additive derivations, and composite Sagas. Inheritance is compile-time only. Every derived effect is flattened and hashed before approval; every composite step binds an exact child id, version and contract hash.

The SDK uses strict Pydantic schemas, monotonic-security checks, a multi-version/multi-semantic catalog, human-readable explain/graph commands, and projection into the existing durable `SagaDefinition`. A shared REST capability may back S1 and several S11 semantic contracts, but tool name alone can never choose between them.

The existing v1 Runtime remains the compatibility execution path while the v2 catalog is qualified. The bundled REST manifests demonstrate authoring and compilation; they are not exposed in DSH until a real Capability Provider, expression/resolver layer, Runtime catalog loader, semantic harness entrypoints, enterprise approval identity, and fault certification are connected.

See the [L1 → L0 Promotion Pipeline](l1-to-l0-promotion.md) for the natural-language-assisted candidate workflow. It generates review proposals but does not weaken execution qualification.
