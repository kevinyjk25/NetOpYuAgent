# 典型小型现网实验 / Typical Small Production Network Lab

## 中文

该实验把 NetOpYu 的本地验证环境从“园区 + IDC 用例骨架”扩展为一套可运行的典型小型企业现网。它包含 10 台 FRR 网络设备和 10 个 Alpine 终端/服务，企业内部使用 OSPF，双运营商出口使用 eBGP；允许和拒绝路径均通过真实容器数据面验证。

```text
                         198.51.100.0/24
                         internet-client
                                |
                    +-----------+-----------+
                    |                       |
                 ISP-1 ===== eBGP ======= ISP-2
                    |                       |
              security-edge-1        security-edge-2
                    | \                   / |
                    |  \                 /  |
                    |   campus-core-1---campus-core-2
                    |      /   |  \       /   |
                    |     /    |   \     /    |
              wired access  wireless   IDC   DMZ
              Erin/Bob/Ops  Carol/Guest |     |
                                     CRM/Wiki/Infra
                                              public-web
```

“安全出口”节点承担双 ISP 边界和安全域路由角色。由于 FRR 不是状态防火墙，本实验用服务器命名空间中的精确源路由黑洞模拟已审核的微隔离/RBAC 策略；它不宣称仿真会话表、NAT、IPS 或厂商防火墙语义。

### 区域与基线

| 区域 | 网段/协议 | 基线行为 |
|---|---|---|
| 有线办公 | VLAN 20/30/60；`10.10.20/30/60.0/24` | Bob 可访问 CRM/Wiki；Erin 初始未准入；Ops 可访问监控 |
| 企业无线 | VLAN 50；`10.10.50.0/24` | Carol 可访问 Wiki |
| 访客无线 | VLAN 40；`10.10.40.0/24` | 可访问 Internet 与 DMZ Portal，拒绝 CRM/Wiki/监控 |
| IDC | `10.20.10/20/30.0/24` | CRM、Wiki、监控分区 |
| DMZ | `10.30.10.0/24` | 内外部均可访问 Portal |
| Internet | eBGP：AS65000 ↔ AS64501/64502 | 双出口、运营商互联、企业 `10.0.0.0/8` 聚合回程 |
| OOB 管理 | Docker `172.20.20.0/24` | 与业务路由分离，仅供本地容器管理 |

### 本地运行

```bash
python scripts/netopyu_lab.py \
  --manifest labs/p075-b-small-production/lab.yaml preflight
python scripts/netopyu_lab.py \
  --manifest labs/p075-b-small-production/lab.yaml \
  deploy --approve-local-lab
python scripts/small_production_lab.py reset --approve-local-lab
python scripts/small_production_lab.py verify
python scripts/small_production_lab.py \
  exercise-failover --approve-local-lab
```

L1 + L0 端到端执行及验证失败回滚：

```bash
python scripts/campus_idc_e2e.py \
  --manifest labs/p075-b-small-production/lab.yaml \
  --config config.small-production-lab.yaml
python scripts/campus_idc_e2e.py \
  --manifest labs/p075-b-small-production/lab.yaml \
  --config config.small-production-lab.yaml \
  --exercise-rollback
```

DSH UI 使用 `config.small-production-lab.yaml`。所有写工具仍需显式开启 destructive projection，并逐次审批 Network L0 计划；模型不能创建清单之外的设备、接口、用户、应用、探测或故障目标。

## English

This lab expands the local campus/IDC use-case skeleton into a runnable small-enterprise production model. Ten FRR devices and ten Alpine endpoints/services provide real OSPF, eBGP, ICMP, and HTTP evidence across wired, wireless, guest, IDC, DMZ, operations, and simulated Internet zones.

The secure-edge nodes model dual-ISP routing and security-zone boundaries. FRR is not a stateful firewall, so reviewed source-specific blackhole routes represent micro-segmentation/RBAC enforcement. The lab does not claim to emulate firewall sessions, NAT, IPS, wireless RF, 802.1X, hardware forwarding, or vendor-specific behavior.

Run the commands in the Chinese section to deploy, restore the reviewed baseline, verify all allow/deny paths, exercise ISP failover/recovery, and execute the same DSH L1 + Network L0 onboarding and rollback flows against this topology.
