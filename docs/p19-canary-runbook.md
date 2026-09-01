# P1.9-C1 Canary 准备与回退手册 / Canary Readiness and Rollback Runbook

## 中文

> **状态：`frozen_future_engineering`。** 历史 P1.9 canary 产品化路线已暂停；当前只保留 proposal-only 原则和离线回归，不进入 ES-P0 执行主线。

### 1. 当前状态和安全边界

P1.9-C1 已提供 **canary 安全准备层**，但没有启用 canary。DSH 和 Hermes 的生产配置仍只接受 `off|shadow`；`canary` 会在 Adapter 启动边界被拒绝。仓库当前没有真实仓库外未见集、两名独立人员形成的最终真值、真实 DSH Web/Hermes CLI 产品执行证据和组织级部署身份，因此不能把本地 synthetic fixture 当成上线证据。

C1 只增加两项离线能力：

1. `l1_runtime/canary_policy.py`：把一份严格的 `proposal_only` Decision 与 Harness 已选择的原始 route 比较，结果只能是“原 route 原样继续”或“收窄/阻断”。它不能改 target、改参数、创建计划、批准计划、生成 nonce、调用 Runtime 或 Provider；
2. `l1_runtime/canary_readiness.py`：读取四份外部证据并生成隐私最小化的 `not_ready|ready_for_review` 报告。即使得到 `ready_for_review`，也不授权激活、不修改配置、不改变流量。

### 2. 单调收窄策略

| 输入 | 读 route | 写 route |
|---|---|---|
| L1 selection 与原 Harness route、profile、session、Harness、候选摘要完全一致 | 原 route 原样继续，仍需 Runtime admission | 原 route 原样继续，仍需 C0 Decision→Plan binding 和全部 L0 控制 |
| `clarify` | 阻断并要求上层追问 | 阻断并要求上层追问 |
| `refuse/out_of_scope` | 阻断 | 阻断 |
| selection 不一致 | 阻断，不允许 L1 重路由 | 阻断，不允许 L1 重路由 |
| 协议、摘要或上下文无效 | 原 route 不变，仅记录观察结果 | 失败关闭 |

“继续”不代表授权。身份、RBAC、审批、参数闭包、preflight、plan hash、执行 nonce、Provider admission、结果验证和补偿/回滚仍全部由 Network Runtime 决定。

### 3. 四类必需证据

所有文件必须是普通 JSON 文件、小于 4 MB、不可为符号链接，并通过自身摘要、有效期和交叉绑定。

1. **Worker Oracle 报告**：`netopyu.io/l1-holdout-qualification/v1`，真实仓库外 sealed set、双 reviewer 消歧、同一不可变模型，全部正式 requirements 为 true；
2. **Adapter Hook 报告**：`netopyu.io/l1-adapter-qualification/v1`，同一 set/model/catalog 下 DSH JavaScript 与 Hermes Python Hook 经 owner-only Worker 的 parity 全部通过；
3. **产品/部署证据**：`netopyu.io/l1-canary-product-evidence/v1`，真实 DSH Web UI 和 Hermes CLI 进程、交互 SLO、Decision receipt、发行包和部署身份摘要均被两名不同 reviewer 验证，有效期最长 30 天；
4. **运维证据**：`netopyu.io/l1-canary-ops-evidence/v1`，kill switch、回 shadow、告警送达和 no-effect replay 均以不同 receipt 完成演练；Core-72 风险控制保持 64/64，最近至少三个不同实现版本的 Runtime trend 为 `stable|improved` 且 p50/p95 未越过阈值；Decision→Plan binding 全通过且 replay/authority escape 为 0；初始上限不超过 5% 流量和 120 分钟，自动审批及 Runtime/Provider bypass 必须关闭，有效期最长 7 天。产品 reviewer 与运维 owner 还必须角色分离。

四份证据必须绑定相同的 model artifact、sealed manifest、consensus labels 和 catalog snapshot；产品证据还要绑定两份 B2 报告，运维证据再绑定产品证据。当前本地测试只使用 synthetic fixture 来验证门禁代码，不能作为上述证据。

### 4. 无副作用检查

```bash
scripts/netopyu-dsh l1-canary-readiness \
  /secure/worker-qualification.json \
  /secure/adapter-qualification.json \
  /secure/product-evidence.json \
  /secure/ops-evidence.json \
  /secure/canary-readiness.json
```

退出码 `0` 只表示 `ready_for_review`；退出码 `1` 表示 `not_ready`。报告不输出 Prompt、标签、参数值、reviewer 或 owner 标识。该命令没有激活功能，也不读取或写入 DSH/Hermes 配置。

### 5. 未来激活前的人工门禁

本节是未来流程，当前代码不存在激活命令。真正接入流量前必须由不同角色完成：

1. 证据 reviewer 验证四份文件来自受控环境，并通过组织身份、签名/不可抵赖和 WORM 审计；
2. 产品 owner 审核适用场景、用户体验和只收窄语义；
3. 网络/安全 owner 审核变更窗口、设备范围、告警阈值和 L0 控制不变；
4. incident owner 验证 shadow/off kill switch、回退版本和联系路径；
5. 独立发布系统把受审 evidence digest、版本和最大流量/时长固化到发布单；
6. 只有另一个明确实现、单独验收的 Adapter change 才可接受 `canary`。不得把 readiness CLI 的退出码直接连接到自动部署。

### 6. 停用、回退和事件处置

触发条件包括：任何 unsafe escape、Decision/Plan 绑定失败、重复 Decision、参数/route 不一致、协议错误突增、告警未送达、Runtime Core-72 退化、p95 超出受审 SLO、审计链异常、证据过期或操作员主动停用。

处置顺序：

1. 使用独立 kill switch 把 Decision mode 退回 `shadow`；若 Decision Plane 本身异常则退到 `off`；
2. 停止接纳新的 canary Decision，不重放旧 Decision、approval proof 或 execution nonce；
3. 已进入 Runtime 的计划按其 journal 状态独立处理：未执行计划到期/拒绝，正在执行的计划由 Runtime 验证并按合同补偿。不要通过回退 Adapter 配置伪造网络回滚；
4. 保存 readiness、Decision、plan、事件链、告警和发布摘要，运行 `runtime-audit` 和离线 retirement；
5. 对所有可能受影响的设备/服务做独立状态核验；`outcome_indeterminate` 或补偿失败必须进入人工处置；
6. 修复后重新生成四份证据并重新评审。旧 readiness 报告不得复用。

### 7. 本阶段验收声明

C1 本地验收证明策略结果是单调的、证据缺失会失败关闭、报告隐私最小化、CLI 无配置副作用，而且 DSH/Hermes 仍拒绝 canary。它不证明生产成功率，不替代真实设备、组织身份、安全审批、外部审计或 SLO。

---

## English

> **Status: `frozen_future_engineering`.** The historical P1.9 canary productization path is paused. Only proposal-only semantics and offline regressions remain as reference outside ES-P0 execution.

### 1. Current state and boundary

P1.9-C1 now provides a **canary safety-readiness layer**, not an active canary. Production DSH and Hermes configurations still accept only `off|shadow`; adapters reject `canary` at startup. The repository has no real repository-external unseen set, final truth from two independent human reviewers, real DSH Web/Hermes CLI product evidence, or organization-grade deployment identity. Synthetic test fixtures are not deployment evidence.

C1 adds only two offline capabilities:

1. `l1_runtime/canary_policy.py` compares a strict `proposal_only` Decision with the route already selected by the Harness. It may preserve that route unchanged or block/narrow it. It cannot rewrite the target or arguments, create or approve a plan, issue a nonce, or call the Runtime/Provider;
2. `l1_runtime/canary_readiness.py` reads four external evidence documents and emits a privacy-minimized `not_ready|ready_for_review` report. `ready_for_review` does not activate, authorize, configure, or route traffic.

### 2. Monotonic narrowing

An exact selection match preserves the original Harness route and still requires every Runtime control. Clarification, refusal, out-of-scope, or selection mismatch blocks the route and can never redirect it. Invalid protocol, digest, or context is fail-closed for writes; reads remain unchanged and only produce an observation so the canary adds no read-side availability dependency.

Continuation is not authority. Identity, RBAC, approval, parameter closure, preflight, plan hash, execution nonce, Provider admission, verification, and compensation remain Runtime responsibilities.

### 3. Required evidence

The gate requires: a qualified `netopyu.io/l1-holdout-qualification/v1` Worker report; a passed `netopyu.io/l1-adapter-qualification/v1` adapter-hook report; a `netopyu.io/l1-canary-product-evidence/v1` attestation covering real DSH Web UI and Hermes CLI processes, interaction SLOs, receipts, distributions, and deployment identities; and a `netopyu.io/l1-canary-ops-evidence/v1` attestation covering distinct exercised kill-switch/shadow-rollback/alert/no-effect-replay receipts, 64/64 Core-72 fault controls, a stable or improved three-version Runtime trend within p50/p95 thresholds, perfect Decision-plan binding with zero replay/authority escape, role separation, and bounded traffic/duration without automatic approval or bypasses.

All four documents must bind the same immutable model artifact, sealed manifest, consensus labels, and catalog snapshot. Product evidence binds both B2 reports, and operations evidence binds product evidence. Product validity is capped at 30 days and operations validity at 7 days.

### 4. Side-effect-free check

```bash
scripts/netopyu-dsh l1-canary-readiness WORKER ADAPTER PRODUCT OPS [OUTPUT]
```

Exit `0` means only `ready_for_review`; exit `1` means `not_ready`. The report emits no prompts, labels, argument values, reviewer ids, or owner ids. The command has no activation path and never reads or writes adapter configuration.

### 5. Future activation review

There is no activation command today. A future change requires organization-authenticated/signature-backed evidence, product/network/security/incident-owner review, an independently controlled release record binding the approved digests and limits, and a separately implemented and qualified Adapter change. Never connect the readiness exit code directly to deployment automation.

### 6. Stop, rollback, and incident response

Any escape, binding/replay failure, route or argument disagreement, protocol spike, alert-delivery failure, Core-72 regression, SLO breach, audit anomaly, expiry, or operator stop must return the Decision mode to `shadow` (or `off` if the plane itself is suspect). Stop accepting new canary Decisions and never replay an old Decision, approval proof, or nonce. Existing Runtime plans are resolved by their own journal states and compensators; reverting an Adapter is not a network rollback. Preserve digest-only evidence, audit every affected plan, independently verify device/service state, escalate indeterminate outcomes, and issue new evidence before another review.

### 7. Claim boundary

Local C1 acceptance proves monotonic policy behavior, fail-closed evidence handling, privacy minimization, a side-effect-free CLI, and continued Adapter rejection of canary. It does not establish a production success probability or replace real devices, organization identity, security approval, external audit, or SLO evidence.
