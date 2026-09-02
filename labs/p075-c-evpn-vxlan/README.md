# P0.75-C EVPN/VXLAN 本地 Fabric / Local Fabric

## 中文

该实验在 Containerlab 中真实运行 FRR 和 Linux 网络能力，不是 Network Runtime mock。

```text
                         MP-BGP EVPN RR / OSPF underlay
                  +-------------+       +-------------+
                  |   spine-1   |       |   spine-2   |
                  +------+------+       +------+------+
                         |  \                 /  |
                         |   \               /   |
                         |    \             /    |
                  +------+------+       +------+------+
                  |    leaf-1   |       |    leaf-2   |
                  | VTEP .1.1   |.......| VTEP .1.2   |
                  +--+---+---+---+       +---+---+---+--+
                     |   |   |               |   |   |
                    a1  b1 trunk-1          a2  b2 trunk-2

VLAN 10 / L2VNI 10010: host-a1 ↔ host-a2，trunk eth1.10
VLAN 20 / L2VNI 10020: host-b1 ↔ host-b2，trunk eth1.20

点线只表示经 Spine underlay 承载的逻辑 VXLAN overlay，不是 Leaf 间物理链路。
```

- 四条 leaf-spine 三层链路运行 OSPF；Loopback 建立双 RR iBGP MP-BGP EVPN。
- Leaf 使用 Linux bridge、802.1Q 子接口和 VXLAN netdevice；FRR 分发 type-2 MAC 与
  type-3 IMET 路由。
- `host-a*`/`host-b*` 是 untagged access；`trunk-*` 同时承载 VLAN 10/20。
- VLAN 10 与 VLAN 20 是两个隔离 L2 bridge domain，没有跨租户 L3 gateway。

运行与验证：

```bash
python scripts/netopyu_lab.py \
  --manifest labs/p075-c-evpn-vxlan/lab.yaml \
  deploy --approve-local-lab --reconfigure
python scripts/netopyu_lab.py \
  --manifest labs/p075-c-evpn-vxlan/lab.yaml verify
python scripts/netopyu_lab.py \
  --manifest labs/p075-c-evpn-vxlan/lab.yaml \
  exercise-fabric-failover --approve-local-lab
python scripts/evpn_vxlan_runtime_e2e.py --approve-local-lab
```

### 真实性边界

已实际运行：802.1Q、Linux bridge、VXLAN、OSPF、MP-BGP EVPN、L2VPN 转发与单链路故障。

未实现：EVPN L3VPN、MPLS L2VPN/L3VPN、厂商 CLI/ASIC、状态防火墙、AP/RF。当前 Docker
Desktop Linux 内核没有 `CONFIG_NET_VRF`，因此 manifest 固定为 `evpn-vxlan-l2`。

## English

This is a live FRR/Linux Containerlab fabric, not a Network Runtime mock. Four
leaf-spine links run OSPF; loopbacks establish dual-RR MP-BGP EVPN. Each leaf
uses Linux bridges, 802.1Q interfaces, and VXLAN netdevices. VLANs 10 and 20 map
to L2VNIs 10010 and 10020, with access and tagged endpoints on both VTEPs.

The lab proves real L2 EVPN control-plane and forwarding behavior plus
single-link resiliency. It does not implement EVPN L3VPN, MPLS VPNs, vendor
CLI/ASIC behavior, stateful firewalling, APs, or RF. The Docker Desktop kernel
lacks `CONFIG_NET_VRF`, so the manifest deliberately claims L2 EVPN only.
