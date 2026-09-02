# 使用与系统接入 / Usage and Integration

## 中文

### 1. 先选一条 Golden Path

项目只推荐三条首次使用路径。运行 `scripts/netopyu journeys` 可获得机器可读的同一清单。

| 路径 | 适合谁 | 第一个命令 | 会产生什么效果 |
|---|---|---|---|
| Understand | 评估者、架构师 | `scripts/netopyu evaluate` | 只在 `artifacts/convergence/` 生成 JSON/HTML；不执行网络或业务写入 |
| Local demo | 开发者、运维工程师 | `scripts/netopyu demo --scenario l1-l0 --approve-local-simulation` | 在临时 mock 状态执行两次写入，验证后恢复内存并删除临时库 |
| Integrate | 平台/系统团队 | `scripts/netopyu integration-check --pack examples/integration-rest-mcp/pack.yaml` | 只校验提案文件；不连接、注册、发布、激活或执行 |

首次使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

scripts/netopyu doctor
scripts/netopyu journeys
scripts/netopyu evaluate
open artifacts/convergence/cockpit.html       # macOS
# xdg-open artifacts/convergence/cockpit.html # Linux
```

`doctor` 是只读检查。它分别报告评测、接入审查、DSH 源码、模型和 Containerlab 的就绪状态；发现命令不等于模型已资格化或实验拓扑已部署。

### 2. 如何把自己的系统接进来

下层系统统一暴露两类能力：

- `read`：提供 fresh observation，必须经过主体、用途、scope、敏感度、Provider 身份/版本和证据完整性校验；
- `write`：表达单个 effect，必须绑定幂等字段、独立 `read` verifier、风险、审批和补偿语义。

传输形式不是 Runtime 语义。一个能力可以来自 MCP、REST、NETCONF、SSH 或 Controller API；模型看不到凭据，配置文件只接受凭据环境变量名。

接入分四步：

1. **接口提案**：复制 `examples/integration-rest-mcp/pack.yaml`，声明 Provider 和 read/write 能力。运行 `integration-check`。此时状态最多是 `ready_for_l0_authoring`。
2. **领域合同**：为 write 写 L1 自然语言 Skill，转换并审查 L0.5，再编译 L0。把确切 L0 id/version/hash 回填到 Pack。转换不会自动激活。
3. **Provider 资格**：验证身份、版本、输入/输出 Schema、幂等、超时、响应丢失、错误后置条件和重启行为，再进入 Capability Catalog/发布流程。
4. **Harness 投影**：DSH/Hermes 只看到经过 Profile/Catalog 筛选的 L1 候选。写请求仍需生成 immutable plan、人工批准、执行前重校验、独立验证和必要时补偿。

严格验收条件：

- 每个 write 都有独立 read verifier；可逆 write 有明确 compensation；
- 凭据不进入 Prompt、Skill、计划摘要、评测、日志或仓库；
- L0 contract、Provider release/deployment、input/output Schema 和审批绑定同一计划；
- 成功只能来自 fresh postcondition，不能来自模型文本或写接口自报；
- Integration Pack、Promotion Workbench、Capability Catalog 和评测驾驶舱均没有执行权威。

### 3. 从 URL1 到 S1/S11

假设已有 `URL1(param1, …, paramK)`：

1. 在 Pack 中先把查询接口建成 `read`，把变化接口建成 `write`；
2. S1 作为原子 L0：固定参数 Schema、来源、目标、effect、preflight、verifier、compensation 和失败策略；
3. 约束式 S11 继承 S1 并收窄参数/风险；扩展式 S11 增加 preflight/verifier 但不能放宽父合同；组合式 S11 以多个 S1/S2 构成 Saga，并定义逆序补偿；
4. 保存 `01-L1.md → 02-L0.5.yaml → 03-L0-authoring.yaml → 04-L0-compiled.json` 轨迹，人工审查后再进入独立发布过程。

详细 authoring 规则见 [L0 v2 设计](l0-v2-design.md)，转换和 S1/S11 示例见 [L1 → L0 Promotion](l1-to-l0-promotion.md)，Provider 上线边界见 [Provider 供应链](provider-supply-chain.md)。

### 4. 在 DSH 页面测试

先按 README 启动 DSH。默认 `mock` 且只读。只有显式设置 `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1` 才投影写工具，但这个开关不等于审批；每个计划仍需在页面审批卡中确认。

页面请求建议包含：业务目标、对象、环境、原因和“缺参先追问”。不要在提示中放 token、密码、私钥或设备 enable secret。

### 5. 当前边界

本仓库已经证明固定本地 Oracle 上的接口/事务控制和模型资格过程，不证明真实厂商设备行为、企业权限、未见分布生产成功率、HA/DR 或生产 SLO。下一道门是仓库外独立人工 ES-P1 Private Holdout；通过后再进入 ES-P2 小范围真实网络资格。完整交互和终态解释见 [Skill 与系统交互全景](SKILL-SYSTEM-INTERACTION.md)。

---

## English

### 1. Choose one Golden Path

Run `scripts/netopyu journeys` for the machine-readable list:

- **Understand:** `scripts/netopyu evaluate` generates a read-only JSON/HTML cockpit under `artifacts/convergence/`.
- **Local demo:** `scripts/netopyu demo --scenario l1-l0 --approve-local-simulation` performs two temporary mock effects, verifies them, restores memory, and removes temporary stores.
- **Integrate:** `scripts/netopyu integration-check --pack examples/integration-rest-mcp/pack.yaml` validates a proposal without contacting, registering, publishing, activating, or executing anything.

### 2. Integrate an external system

Model every external surface as either a protected `read` observation or a single `write` effect. Transport may be MCP, REST, NETCONF, SSH, or a controller API; the Runtime consumes capability semantics, not transport names.

1. Describe Providers and capabilities in an Integration Pack.
2. Author L1 prose, review the generated L0.5, compile an immutable L0 contract, and bind its exact id/version/hash.
3. Qualify Provider identity/version/schema, idempotency, timeouts, lost responses, incorrect postconditions, and restart behavior; then use the separate Catalog/release workflow.
4. Project only profile-scoped candidates through DSH/Hermes. Every write still requires an immutable plan, human approval, pre-write revalidation, independent verification, and compensation where applicable.

Every write needs an independent read verifier; reversible writes need compensation. Credentials remain deployment-owned and model-hidden. The Integration Pack, Promotion Workbench, Catalog, and cockpit never have execution authority.

### 3. URL1 to S1/S11

For `URL1(param1, …, paramK)`, first separate read and write operations. S1 fixes parameter schema/provenance, target, effect, preflight, verifier, compensation, and failure policy. A constrained S11 narrows S1, an extended S11 adds controls without weakening the parent, and a composed S11 coordinates reviewed S1/S2 effects as a Saga with reverse-order compensation. Preserve the complete L1 → L0.5 → authoring → compiled trajectory and publish only after independent review.

See [L0 v2 design](l0-v2-design.md), [L1 → L0 promotion](l1-to-l0-promotion.md), and [Provider supply chain](provider-supply-chain.md).

### 4. Boundaries

The repository proves local fixed-oracle controls and model-qualification mechanics. It does not qualify real vendor devices, enterprise authority, unseen production success probability, HA/DR, or production SLOs. The next gate is a repository-external independent-human ES-P1 private holdout, followed by narrow ES-P2 real-network qualification. See the [Skill-to-system interaction guide](SKILL-SYSTEM-INTERACTION.md) for lifecycle and terminal-state semantics.
