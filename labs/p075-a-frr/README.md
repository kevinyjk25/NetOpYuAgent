# P0.75-A FRR 本地实验室 / Local FRR Lab

该拓扑使用两个 FRR 路由器、两条具有不同 OSPF cost 的 WAN 链路以及两个 Alpine
终端，覆盖配置变更、控制面收敛、端到端探测和链路/NetEm 故障注入。所有可操作设备、
探测目标和故障接口都固定在 `lab.yaml`，模型不能在运行时发明目标。

This topology contains two FRR routers, redundant OSPF WAN links and two Alpine
endpoints. The reviewed manifest fixes every device, probe and fault target so
runtime input cannot expand the lab's blast radius.

运行入口 / Entry point:

```bash
python scripts/netopyu_lab.py preflight
python scripts/netopyu_lab.py deploy --approve-local-lab
python scripts/netopyu_lab.py verify
python scripts/netopyu_lab.py exercise-failover --approve-local-lab
python scripts/netopyu_lab.py destroy --approve-local-lab
```

Apple Silicon 上实验本身必须运行在 Linux VM/devcontainer/远端 Linux lab host 中；
项目主进程可以继续运行在 macOS。厂商 VM 镜像不属于 P0.75-A。

当前仓库的 `.devcontainer/devcontainer.json` 使用 Containerlab 官方 Dood 镜像。
在 macOS 启动 Docker Desktop 后，用 VS Code 执行 **Dev Containers: Reopen in
Container**，然后在容器终端执行：

```bash
sudo containerlab deploy --topo labs/p075-a-frr/topology.clab.yml
```

该命令通过宿主 Docker daemon 创建实验节点。部署完成后，macOS 上运行的 DSH
可以直接通过受限 `docker exec` provider 操作这些节点。

On Apple Silicon the lab must run inside a Linux VM/devcontainer or a Linux lab
host. Vendor VM images are intentionally outside P0.75-A.
