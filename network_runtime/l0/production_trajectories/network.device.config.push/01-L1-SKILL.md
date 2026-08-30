---
name: network-device-config-push
description: Readable bootstrap projection of production L0 network.device.config.push;
  collect exact intent, require approval, and trust only independent verification.
allowed-tools: push_config
metadata:
  skill_id: network_device_config_push
  display_name: network.device.config.push
  purpose: 以精确参数安全执行 network.device.config.push，并通过独立证据验证结果。 / Safely execute network.device.config.push
    with exact inputs and independent evidence.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan
  tags: production,l0-v2,readability,bootstrap
  tool_deps: push_config
  returns: 独立验证的目标状态，或明确的人工介入终态。 / Independently verified desired state or an explicit
    manual-intervention terminal state.
  origin: bootstrap-from-reviewed-production-l0-v2
---

# network.device.config.push

> 本文件是从已受审生产 L0 反向生成的可读基线，用于解释与 round-trip 验证；它不是新的执行授权。  
> This file is a readable baseline reverse-bootstrapped from a reviewed production L0 for explanation and round-trip validation; it grants no execution authority.

## 目标 / Purpose

以精确参数安全执行 network.device.config.push，并通过独立证据验证结果。 / Safely execute network.device.config.push with exact inputs and independent evidence.

## 精确语义意图 / Exact Semantic Intent

以下小型结构块是 L1、L0.5 与 L0 之间的可审计语义锚点；Runtime 必须逐字段保真，
不得从周边自然语言猜测或补全。 / This small structured block is the auditable
semantic anchor across L1, L0.5 and L0; Runtime must preserve every field and may
not guess or complete it from surrounding prose.

<!-- netopyu:semantic-intents/v1 -->
```yaml
- effectCapability: netopyu.tool.push_config
  kind: configure_device
  targetFields:
  - device_id
  desiredState:
    requested_configuration_digest: ${intent.configuration_digest}
```

## Parameters

- `device_id`: device_id；必填 / Required string；最长长度 / maximum length: 4096。
- `config_text`: config_text；必填 / Required string；最长长度 / maximum length: 65536。
- `dry_run`: dry_run；可选 / Optional boolean。

## Steps

1. 收集全部必填参数且不得推断关键值。 / Collect every required input without inferring critical values.
2. 通过合同 Observation 读取并保存审批前状态。 / Read and preserve pre-approval state through contractual observation.
3. 展示并绑定不可变计划，等待明确的一次性人工审批。 / Bind the immutable plan and wait for explicit one-shot human approval.
4. 执行前重读状态；漂移时停止，不发送 Effect。 / Re-read before execution and stop on drift without sending the effect.
5. 仅通过 `push_config` 发送一次受审 Effect。 / Send the reviewed effect exactly once through `push_config`.
6. 使用独立 verifier 判断结果；写响应本身不是成功。 / Use the independent verifier; the write response alone is not success.
7. 按合同补偿或进入人工介入终态。 / Compensate contractually or enter manual intervention.

## Constraints

- 人工审批强制且只绑定当前合同、参数、目标和前态。 / Human approval is mandatory and binds only this contract, inputs, target, and preflight.
- 只允许 profile：lan。 / Allowed profiles only: lan.
- Provider 返回不能替代独立验证。 / A Provider response cannot replace independent verification.
- 写结果不确定时先只读对账，禁止盲目重试。 / Reconcile read-only after an indeterminate result; never retry blindly.
- Non-compensable justification: 该受审合同没有安全自动逆操作；验证失败必须进入人工介入，禁止盲目重试。 / The reviewed contract has no safe automatic inverse; verification failure requires manual intervention and never a blind retry.
