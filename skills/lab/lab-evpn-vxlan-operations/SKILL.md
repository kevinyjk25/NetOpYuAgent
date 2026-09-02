---
name: lab-evpn-vxlan-operations
description: 从真实 Containerlab Linux bridge、VXLAN 与 FRR MP-BGP EVPN 状态诊断本地数据中心网络，并严格报告仿真边界。
allowed-tools: lab_get_fabric_state, lab_get_access_vlan, lab_get_vxlan_state, lab_get_bgp_evpn_summary, lab_get_evpn_routes, lab_probe
metadata:
  skill_id: lab_evpn_vxlan_operations
  display_name: Local EVPN VXLAN Fabric Operations
  purpose: Diagnose the declared local EVPN/VXLAN fabric from observed control-plane and data-plane evidence.
  risk_level: low
  requires_hitl: 'false'
  profiles: dc
  lab_capability: fabric
  tags: lab,vlan,vxlan,bgp,evpn,l2vpn,verification
  tool_deps: lab_get_fabric_state,lab_get_access_vlan,lab_get_vxlan_state,lab_get_bgp_evpn_summary,lab_get_evpn_routes,lab_probe
  returns: 实际 VLAN/VXLAN/EVPN 状态、流量证据以及明确的能力边界
---

# 本地 EVPN/VXLAN 网络运维

该 Skill 只从当前 manifest 声明的 Containerlab 网络和实时观测回答问题，不根据模型知识
补全设备、链路、VNI、邻居或路由。

## 必须遵循的诊断流程

1. 全网健康检查先调用 `lab_get_fabric_state`，以其 `contract` 和 `truth_boundary` 为准。
2. 接入口 VLAN 查询必须提供精确 `device_id` 和 `interface`，调用
   `lab_get_access_vlan`，不得把 endpoint ID 当成 device ID。
3. VXLAN 查询调用 `lab_get_vxlan_state`，只有 Linux VXLAN ID、FRR VNI 和远端 VTEP
   三者同时符合声明时才报告正常。
4. BGP EVPN 邻居调用 `lab_get_bgp_evpn_summary`；路由查询调用
   `lab_get_evpn_routes`，`route_type` 只允许 2、3 或 5。
5. 可达性结论必须调用 manifest 中预声明的 `lab_probe`；不得接受任意地址或 shell 命令。
6. 任一结果 `ok=false`、字段缺失或状态不一致时，报告未验证，不得推测成功。

## 仿真边界

- 该实验室真实运行 802.1Q、Linux bridge、VXLAN 数据面和 FRR MP-BGP EVPN 控制面。
- 当前模式为 EVPN L2VPN；由于本机 Docker Linux 内核未启用 NET_VRF，不宣称 EVPN
  L3VPN、MPLS L2VPN/L3VPN、厂商 CLI、ASIC、无线 RF 或硬件性能。

# English

Use only manifest-bound, live Containerlab observations. Diagnose in this order:
fabric state, exact access-port VLAN, VTEP VXLAN/VNI state, BGP EVPN peers/routes,
then a predeclared traffic probe. Treat any `ok=false`, missing fact, or mismatch as
unverified. This lab proves real Linux 802.1Q/bridge/VXLAN and FRR EVPN L2VPN behavior;
it does not prove EVPN L3VPN, MPLS VPNs, vendor CLI/ASIC behavior, wireless RF, or
hardware performance.
