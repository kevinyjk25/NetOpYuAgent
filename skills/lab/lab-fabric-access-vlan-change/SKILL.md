---
name: lab-fabric-access-vlan-change
description: 在真实 Containerlab EVPN/VXLAN 网络中，通过固定 L0 合约安全变更一个接入口 VLAN，并验证或精确回滚。
allowed-tools: lab_get_access_vlan, lab_probe, fabric_set_access_vlan
metadata:
  skill_id: lab_fabric_access_vlan_change
  display_name: Local Fabric Access VLAN Change
  purpose: Safely move one manifest-declared access port between declared VLANs.
  risk_level: high
  requires_hitl: 'true'
  profiles: dc
  lab_capability: fabric
  tags: lab,vlan,l1-skill,l0-skill,approval,verification,rollback
  tool_deps: lab_get_access_vlan,lab_probe,fabric_set_access_vlan
  returns: 审批计划、Linux bridge/PVID 证据、流量证据和精确回滚状态
---

# 本地 Fabric 接入口 VLAN 变更

该 L1 Skill 负责诊断和收集明确意图；唯一写操作必须进入
`network.fabric.access-vlan.set` L0 Skill。模型不得生成或执行 shell 命令。

## 确定性流程

1. 追问并取得唯一的 `device_id`、`interface`、目标 `vlan_id`、变更 `reason`，以及用户
   需要保护的可选 `verification_probe_id`。不得猜测缺失参数。
2. 调用 `lab_get_access_vlan(device_id, interface)`。必须看到 `ok=true`、`mode=access`、
   唯一 PVID、untagged 标志和 bridge；否则停止，不允许写入。
3. 如提供流量探测，审批前先调用同一 `lab_probe(verification_probe_id)` 建立业务基线；
   基线失败则停止。Network Runtime 写后仍会独立重跑该探测。
4. 调用 `fabric_set_access_vlan`，参数只能包含上面的精确值。Runtime 会严格验证 VLAN
   数值、设备清单、接入口白名单以及目标 VLAN 是否已在 manifest 声明。
5. 展示不可变计划、风险、目标、审批前证据和后置条件，等待人类明确审批。不得代批。
6. 审批后，L0 Runtime 再读一次端口；与审批前状态不完全一致则不写入。
7. Runtime 使用固定 argv 修改 Linux bridge/PVID，随后 fresh read 验证目标 VLAN、bridge
   和 untagged 状态，并按需运行预声明流量探测。
8. 任一后置条件失败时，自动恢复执行会话保存的精确 bridge/PVID 快照并重新读取。只有
   fresh read 与审批前 typed snapshot 完全相等才标记 `rollback_verified`；否则标记
   `manual_intervention_required`。

## 安全边界

- 只允许 manifest 中的 VTEP、access interface 和 VLAN；trunk 不可通过该工具变更。
- 不接受任意 shell、原始 CLI、任意目标 IP、批量端口或未声明 VLAN。
- 该能力只适用于本地 Containerlab，不可作为真实设备变更授权。

# English

Collect exact intent and baseline evidence, then route the only write through
`network.fabric.access-vlan.set`. The runtime revalidates the approved snapshot,
uses fixed argv, performs a fresh bridge/PVID read and optional manifest-bound traffic
probe, and restores the exact execution-session snapshot when verification fails.
Only an independently proven exact restoration is reported as `rollback_verified`.
