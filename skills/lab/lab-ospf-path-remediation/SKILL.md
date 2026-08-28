---
name: lab-ospf-path-remediation
description: 在P0.75-A本地FRR实验室中诊断分支到DC的OSPF路径，并通过Network L0 Skill执行有界配置修复和端到端流量验证。
allowed-tools: get_device_config, get_ospf_neighbors, lab_probe, edit_device_config
metadata:
  skill_id: lab_ospf_path_remediation
  display_name: Local Lab OSPF Path Remediation
  purpose: Diagnose and safely remediate the reviewed P0.75-A OSPF path
  risk_level: high
  requires_hitl: 'true'
  tags: lab,ospf,l1-skill,l0-skill,verification,rollback
  tool_deps: get_device_config,get_ospf_neighbors,lab_probe,edit_device_config
  returns: 诊断、审批计划、L0执行状态、配置证据、端到端探测证据和回滚状态
---

# 本地实验室 OSPF 路径修复

此 L1 Skill 只适用于 `netopyu-p075a` manifest 中声明的设备和探测，不得把步骤或
命令迁移到真实设备。所有写操作必须进入 `network.device.config.edit` L0 Skill。

## 确定性流程

1. 对目标 `device_id` 调用 `get_device_config`，保存审批前配置证据。
2. 对同一 `device_id` 调用 `get_ospf_neighbors`。必须恰好观察到两个 `Full` 邻居；
   否则停止并报告控制面未收敛，不允许写入。
3. 调用 `lab_probe`，且 `probe_id` 必须为 `branch-to-dc`。基线探测必须成功；否则
   停止并报告当前已有业务中断，不允许通过配置写入掩盖未知故障。
4. 根据用户明确目标构造最小 FRR 命令。仅允许 Network Lab 白名单中的命令，不得使用
   shell、`do`、`copy`、`write`、`reload` 或未声明接口。
5. 调用 `edit_device_config`，必须传入：
   - 精确 `device_id`；
   - 有序 `config_lines`；
   - 明确 `reason`；
   - `verification_probe_id=branch-to-dc`。
6. 向用户展示不可变计划并等待风险审批。不得代替用户批准。
7. 审批后由 L0 Runtime 执行一次；通过 fresh running-config 和预声明流量探测双重验证。
8. 任一后置条件失败时，由 provider 快照补偿并再次读取配置证明精确恢复；无法证明恢复
   时标记 `manual_intervention_required`，不得宣称成功。

## 边界

- 该流程证明控制面配置与容器转发行为，不证明 ASIC、吞吐量、硬件时序或无线 RF。
- 目标、探测和故障接口只能来自 `labs/p075-a-frr/lab.yaml`。
- 模型负责选择该 L1 Skill 和提出参数；步骤图、校验、审批、执行次数、验证和补偿由
  Network Runtime 决定。
