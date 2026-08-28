# Production L0 trajectories / 生产 L0 轨迹

## 中文

本目录保存全部 21 个存量生产 L0 的可读、可审计基线。每个目录包含 Capability Catalog、L1 自然语言 Skill、L0.5 结构化自然语言 Skill、L0 authoring、L0 compiled、逐级 hash trajectory 和验证报告。

这些 L1/L0.5 是从已经受审的生产 L0 **反向 bootstrap**，用于建立解释基线并验证 Promotion 结构约束和编译 round trip；它不证明模型可以从任意自然语言独立恢复相同合同。后续人工修改必须重新通过 Promotion、Provider/故障认证和显式发布。

| L0 | Tool | Contract hash | 目录 |
|---|---|---|---|
| `network.device.config.edit` | `edit_device_config` | `sha256:af2b0ae18ab4863e2f00df16ed899e7fbc567d9eb2c9482a317bd410502a533b` | [network.device.config.edit](network.device.config.edit/) |
| `network.device.config.push` | `push_config` | `sha256:85c6dffde2d71ddd15a4a2970a350099d63fa13ce97e3333cc4910bcf84e181d` | [network.device.config.push](network.device.config.push/) |
| `network.service.restart` | `restart_service` | `sha256:d31c148ae12efe111b833f42fb189bbf59b49df250c7ac83b225f816efd42579` | [network.service.restart](network.service.restart/) |
| `network.service.rollback` | `rollback_service` | `sha256:3038d914744201716ab8a66885bd23f00b035fa72ee12510ca9b8f5f74469449` | [network.service.rollback](network.service.rollback/) |
| `network.deploy.rollback` | `rollback_deploy` | `sha256:52be2be9960d2cb06066f84ea04380d621874a7d628caa98dfd998afed0fe6cf` | [network.deploy.rollback](network.deploy.rollback/) |
| `network.node.drain` | `drain_node` | `sha256:84cef55dd4ebc4e0fcd172f13145fd91b5e895c656870e739ab04043e442081d` | [network.node.drain](network.node.drain/) |
| `network.resource.failover` | `failover` | `sha256:bf8a83e0587cdb9f3d965f1523cd356eca24df7db9cc5df98802455a6bbf3ab5` | [network.resource.failover](network.resource.failover/) |
| `network.resource.delete` | `delete_resource` | `sha256:583cca1ac7fc2793bd3ce2611764f60152f99d91ffb13d7aef0e046c5d189489` | [network.resource.delete](network.resource.delete/) |
| `network.lan.user-access.grant` | `grant_user_access` | `sha256:98206a9defa6e464800205dbf375abd5faf0bce9718e24cea392156a607efa68` | [network.lan.user-access.grant](network.lan.user-access.grant/) |
| `network.lan.user-access.revoke` | `revoke_user_access` | `sha256:b15a708451cf486e32b4e3b6aca8f7aad6a0df326c7aa59a66527f73b3043909` | [network.lan.user-access.revoke](network.lan.user-access.revoke/) |
| `network.dc.fabric-config.push` | `dc_config_push` | `sha256:50aace0b4d246383b5fe2bbe00d0604916e48892cc8ec9eef45230212386b608` | [network.dc.fabric-config.push](network.dc.fabric-config.push/) |
| `network.dc.app-access.grant` | `dc_grant_app_access` | `sha256:c16600415f3e23496e8dcdf17f9a2b0c9b8171da2b2dd79d2fd65d24b8e80e08` | [network.dc.app-access.grant](network.dc.app-access.grant/) |
| `network.dc.app-access.revoke` | `dc_revoke_app_access` | `sha256:72fd1cd9e44e4a5daf3217c8c94d51a14e1bc69d5da76fb7de13edbcb4d848c0` | [network.dc.app-access.revoke](network.dc.app-access.revoke/) |
| `network.wan.path.failover` | `wan_failover_path` | `sha256:561aa3fa37ce3fdefc9aa43d256c0435719262ab09d8388aed85d9f7f451e2b4` | [network.wan.path.failover](network.wan.path.failover/) |
| `network.fabric.access-vlan.set` | `fabric_set_access_vlan` | `sha256:b6240eca17d6d8cb2b894a2be46b0440f20a521e301b502bd5bb951da04ec434` | [network.fabric.access-vlan.set](network.fabric.access-vlan.set/) |
| `service.access.entitlement.grant` | `access_policy_grant_entitlement` | `sha256:dea8e846f21fb01b00eb71d342f8ce979c88f3bbf26a63199fd67d826f67a896` | [service.access.entitlement.grant](service.access.entitlement.grant/) |
| `service.access.entitlement.revoke` | `access_policy_revoke_entitlement` | `sha256:4c2334d0577c59c2a891523a375583ab30daa9eca1ad0ad7a42a0e906440c293` | [service.access.entitlement.revoke](service.access.entitlement.revoke/) |
| `service.platform.restart` | `platform_restart_service` | `sha256:307126faced2875a89ede969db79f20fb57a2b280a54e27d13b521ddf4e074dc` | [service.platform.restart](service.platform.restart/) |
| `service.platform.rollback` | `platform_rollback_service` | `sha256:634c51f85d7ed7c472d41c87bc67e916f4cf2a6d19ad2cb5e5430cb6a0accb4e` | [service.platform.rollback](service.platform.rollback/) |
| `network.application.enforcement.apply` | `network_apply_app_enforcement` | `sha256:c923493869d9a34d8d012bc70cab3fce34da5d4e69e8512cbec1bb5940c2a712` | [network.application.enforcement.apply](network.application.enforcement.apply/) |
| `network.application.enforcement.revoke` | `network_revoke_app_enforcement` | `sha256:a87f65e520314a51410a1097aef90e1a45cdb1bc6348c5bcef5237cf26d37550` | [network.application.enforcement.revoke](network.application.enforcement.revoke/) |

## English

This directory preserves readable, auditable baselines for all 21 existing production L0 contracts. Each directory contains the Capability Catalog, natural-language L1 Skill, structured-natural-language L0.5 Skill, L0 authoring and compiled artifacts, a predecessor-linked trajectory, and a validation report.

The L1/L0.5 files are reverse-bootstrapped from already reviewed production L0 contracts. They establish an explanation baseline and validate Promotion structure plus exact compiler round trips; they do not prove that a model can independently recover the same contract from arbitrary prose. Any later human change still requires Promotion, Provider/fault qualification, and explicit publication.
