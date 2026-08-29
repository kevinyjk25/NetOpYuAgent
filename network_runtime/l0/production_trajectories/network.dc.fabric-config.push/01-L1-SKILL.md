---
name: network-dc-fabric-config-push
description: Readable bootstrap projection of production L0 network.dc.fabric-config.push;
  collect exact intent, require approval, and trust only independent verification.
allowed-tools: dc_config_push
metadata:
  skill_id: network_dc_fabric_config_push
  display_name: network.dc.fabric-config.push
  purpose: 以精确参数安全执行 network.dc.fabric-config.push，并通过独立证据验证结果。 / Safely execute network.dc.fabric-config.push
    with exact inputs and independent evidence.
  risk_level: high
  requires_hitl: 'true'
  profiles: dc
  tags: production,l0-v2,readability,bootstrap
  tool_deps: dc_config_push
  returns: 独立验证的目标状态，或明确的人工介入终态。 / Independently verified desired state or an explicit
    manual-intervention terminal state.
  origin: bootstrap-from-reviewed-production-l0-v2
---

# network.dc.fabric-config.push

> 本文件是从已受审生产 L0 反向生成的可读基线，用于解释与 round-trip 验证；它不是新的执行授权。  
> This file is a readable baseline reverse-bootstrapped from a reviewed production L0 for explanation and round-trip validation; it grants no execution authority.

## 目标 / Purpose

以精确参数安全执行 network.dc.fabric-config.push，并通过独立证据验证结果。 / Safely execute network.dc.fabric-config.push with exact inputs and independent evidence.

## Parameters

- `node`: node；必填 / Required string；最长长度 / maximum length: 4096。
- `config_lines`: config_lines；必填 / Required array；最短长度 / minimum length: 1；最长长度 / maximum length: 500。
- `reason`: reason；必填 / Required string；最短长度 / minimum length: 1；最长长度 / maximum length: 4096。

## Steps

1. 收集全部必填参数且不得推断关键值。 / Collect every required input without inferring critical values.
2. 通过合同 Observation 读取并保存审批前状态。 / Read and preserve pre-approval state through contractual observation.
3. 展示并绑定不可变计划，等待明确的一次性人工审批。 / Bind the immutable plan and wait for explicit one-shot human approval.
4. 执行前重读状态；漂移时停止，不发送 Effect。 / Re-read before execution and stop on drift without sending the effect.
5. 仅通过 `dc_config_push` 发送一次受审 Effect。 / Send the reviewed effect exactly once through `dc_config_push`.
6. 使用独立 verifier 判断结果；写响应本身不是成功。 / Use the independent verifier; the write response alone is not success.
7. 按合同补偿或进入人工介入终态。 / Compensate contractually or enter manual intervention.

## Constraints

- 人工审批强制且只绑定当前合同、参数、目标和前态。 / Human approval is mandatory and binds only this contract, inputs, target, and preflight.
- 只允许 profile：dc。 / Allowed profiles only: dc.
- Provider 返回不能替代独立验证。 / A Provider response cannot replace independent verification.
- 写结果不确定时先只读对账，禁止盲目重试。 / Reconcile read-only after an indeterminate result; never retry blindly.
- Non-compensable justification: 该受审合同没有安全自动逆操作；验证失败必须进入人工介入，禁止盲目重试。 / The reviewed contract has no safe automatic inverse; verification failure requires manual intervention and never a blind retry.
