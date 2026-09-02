# P0.75-A FRR 本地实验室 / Local FRR Lab

## 中文

该拓扑使用两个 FRR 路由器、两条具有不同 OSPF cost 的 WAN 链路和两个 Alpine 终端，覆盖配置变更、控制面收敛、端到端探测以及链路/NetEm 故障注入。所有可操作设备、探测目标和故障接口都固定在 `lab.yaml`，模型不能在运行时扩张目标范围。

运行入口：

```bash
python scripts/netopyu_lab.py preflight
python scripts/netopyu_lab.py deploy --approve-local-lab
python scripts/netopyu_lab.py verify
python scripts/netopyu_lab.py exercise-failover --approve-local-lab
python scripts/netopyu_lab.py destroy --approve-local-lab
```

Apple Silicon 上的实验必须运行在 Linux VM、devcontainer 或远端 Linux lab host 中，项目主进程可继续运行在 macOS。厂商 VM 镜像不属于 P0.75-A。

仓库的 `.devcontainer/devcontainer.json` 使用 Containerlab 官方 Dood 镜像。在 macOS 启动 Docker Desktop 后，用 VS Code 执行 **Dev Containers: Reopen in Container**，然后在容器终端运行：

```bash
sudo containerlab deploy --topo labs/p075-a-frr/topology.clab.yml
```

该命令通过宿主 Docker daemon 创建实验节点。部署完成后，macOS 上的 DSH 可通过受限 `docker exec` Provider 操作这些节点。

## English

This topology contains two FRR routers, two WAN links with different OSPF costs, and two Alpine endpoints. It covers configuration change, control-plane convergence, end-to-end probes, and link/NetEm fault injection. The reviewed `lab.yaml` fixes every device, probe, and fault target so model input cannot expand the blast radius.

Use the commands above for preflight, deployment, verification, failover, and destruction. Mutating lab commands require explicit `--approve-local-lab`.

On Apple Silicon, the lab must run inside a Linux VM/devcontainer or on a remote Linux lab host while the main project may remain on macOS. Vendor VM images are outside P0.75-A. The repository devcontainer uses the official Containerlab Dood image; after Docker Desktop starts, reopen the project in the devcontainer and run the shown `containerlab deploy` command. DSH on macOS can then operate the nodes through the bounded `docker exec` Provider.
