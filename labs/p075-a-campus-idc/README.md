# 企业园区 + IDC Containerlab / Campus + IDC Containerlab

该实验室把原 mock 用例“为新员工 Erin 开通园区网络准入并授予 IDC CRM 访问，最后验证端到端可达”落到真实容器网络。五台 FRR 路由器形成双园区核心和 IDC 边界/叶节点，两个员工终端与两个 HTTP 应用提供真实数据面。

初始状态固定为：Bob 可访问 CRM/Wiki；Erin 的接入口关闭且 CRM 服务器存在针对 Erin `/32` 的黑洞策略。Network Runtime 只能操作 `lab.yaml` 中声明的用户、应用、接口和地址。

本实验室用 Linux 接口状态模拟 NAC enforcement，用应用服务器的精确源地址路由策略模拟 RBAC enforcement；实际路由、策略命中和 HTTP 请求都在容器中发生。它不是 RADIUS、802.1X、无线射频或真实 IAM 产品仿真。

This lab materializes the former mock onboarding case on actual container namespaces and traffic. Linux link state represents NAC enforcement and an exact per-source route on the application server represents RBAC enforcement. Routing and HTTP evidence are real; RADIUS, 802.1X, RF and vendor IAM behavior are not emulated.

```bash
export NETOPYU_DSH_BACKEND=pragmatic
export NETOPYU_CONFIG_PATH="$PWD/config.campus-idc-lab.yaml"
python scripts/netopyu_lab.py --manifest labs/p075-a-campus-idc/lab.yaml status
python scripts/campus_idc_e2e.py
python scripts/campus_idc_e2e.py --exercise-rollback
python scripts/campus_idc_e2e.py --reset-only
```

启动 loopback DC peer 后，可增加 `--peer-url http://127.0.0.1:8766`，使同一用例
通过真实 HTTP A2A transport 执行。脚本使用 `NETOPYU_DSH_NETWORK_RUNTIME_STORE`
与 peer 共享审计库。

After the loopback DC peer is running, add
`--peer-url http://127.0.0.1:8766` to exercise the real HTTP A2A transport.
The script uses `NETOPYU_DSH_NETWORK_RUNTIME_STORE` so both sides share the
same audit journal.
