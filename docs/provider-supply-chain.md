# Provider 发布与资格认证供应链 / Provider Release and Qualification Supply Chain

## 中文

### 1. 状态与边界

P1.4-B-ready 已提供本地可运行的 Provider release reference：严格 Manifest、独立 Publisher/Qualifier/Deployer Ed25519 角色、仓库外 JSONL 进程资格协议、固定 9 项故障套件、真实进程重启恢复、OCI image/SBOM/provenance 必需 digest、部署控制器证明、SQLite 发布状态机、兼容性门禁、显式 promote/rollback/deprecate、hash-chain lifecycle audit，以及 Runtime admission。PreparedPlan schema v9 会同时绑定 Provider identity、release、manifest、qualification、deployment、schema、Capability 和 L0 contract。

它不等于生产供应链认证。测试会把 fixture 复制到仓库外临时目录并以独立进程运行，但 fixture 源码仍随本仓库维护；当前密钥临时生成，artifact 仅验证 exact digest 映射，尚未验证真实 OCI registry、SBOM 内容或 SLSA provenance，也没有组织独立仓库/CI/实验室、HSM/企业签名根和外部 WORM 审计。P1.4-B 必须在真实组织边界完成这些现场资格步骤。

### 2. 信任关系

```text
Provider publisher key ──签名──> Provider Manifest
                                      │ manifest digest
Independent qualifier key ─签名─> Qualification Report (固定 9/9)
                                      │
                                      ▼
                              Provider Release Bundle
                                      │ verify/stage/publish/deploy
                                      ▼
Deployment controller ──签名──> exact release/artifact/environment attestation
                                      │ active release + deployment
                                      ▼
MCP/OpenAPI discovery ───────> Provider Admission Gate ─────> Runtime plan v9
```

- Publisher 证明“谁发布了什么版本、artifact 和 Capability 合同”。
- Qualifier 独立证明该 exact Manifest 通过固定故障语义套件。
- Deployer 证明哪个 exact release 及其全部 artifact digest 已部署到哪个环境；证明最长 31 天且必须续期。
- Trust Store 按 `provider_id` 限定 key scope、角色和有效期；同一密钥材料不能跨 publisher、qualifier、deployer 角色复用。
- MCP 的 `release_provider_id` 来自部署配置，而非 Provider 自己的 tool metadata。
- Admission 要求运行时发现的 provider identity、Capability、schema digest、result contract 与激活 Manifest 完全一致；Effect 还必须绑定 Manifest 允许的 L0 contract hash。
- 签名元数据本身也在 Ed25519 payload 内，不能通过修改 expiry、role 或 key id 延长/转换签名。

### 3. Manifest 与资格合同

Manifest 使用 `netopyu.io/provider-manifest/v1`，至少包含：

- provider id/version、部署可验证 identity、Runtime API version；
- 不可变 OCI/image/package/SBOM 等 artifact digest；
- 每个 Capability 的 id/version/kind/action/effect semantics；
- provider role/kind、input/output schema digest、sensitivity/scope/freshness；
- result contract；
- 每个 Effect 允许的 reviewed L0 contract hash；
- compatible/breaking 声明与 breaking release 的 exact `supersedes` digest。

资格报告使用 `netopyu.io/provider-qualification/v1`，固定要求：

1. identity/schema binding；
2. send 前超时不产生效果；
3. operation id 重复调用幂等；
4. out-of-order 拒绝；
5. partial success 可只读 reconcile；
6. unknown terminal 不盲重试；
7. compensation 精确恢复 baseline；
8. compensation failure 进入人工介入；
9. restart 后保留并恢复 operation state。

`network_runtime.provider_qualification.run_provider_qualification()` 消费严格资格目标。`network_runtime.provider_external` 提供仓库外 JSONL 进程 Adapter：绝对 executable/cwd、无 shell、最小环境、请求 UUID/schema 绑定、超时与响应大小上限、transport/operation 错误分离，并真实终止/重启进程验证持久状态。它不会从不受信 bundle 动态导入或执行 Python。生产中外部 Provider 应在隔离 CI/实验环境实现同一协议，由独立 Qualifier 审查并签名证据。

### 4. 发布流程

先查看机器可读 JSON Schema：

```bash
scripts/netopyu-provider schema --kind manifest
scripts/netopyu-provider schema --kind qualification
scripts/netopyu-provider schema --kind bundle
scripts/netopyu-provider schema --kind trust
scripts/netopyu-provider schema --kind external-target
scripts/netopyu-provider schema --kind deployment-attestation
scripts/netopyu-provider schema --kind signed-deployment
```

先对仓库外 Provider 运行固定资格套件：

```bash
scripts/netopyu-provider qualify-external \
  --config external-target.json \
  --manifest provider-manifest.json \
  --tool-name vendor_set_access_vlan \
  --arguments qualification-arguments.json \
  --environment isolated-qualification \
  --output provider-qualification.json
```

Publisher 和独立 Qualifier 分别签名。private key 必须是 owner-only Ed25519 PEM；命令拒绝覆盖已有输出文件：

```bash
scripts/netopyu-provider sign-manifest \
  --manifest provider-manifest.json \
  --private-key /run/secrets/provider-publisher.key \
  --key-id publisher-2026 \
  --output provider-manifest.sig.json

scripts/netopyu-provider sign-qualification \
  --qualification provider-qualification.json \
  --private-key /run/secrets/provider-qualifier.key \
  --key-id qualifier-2026 \
  --output provider-qualification.sig.json

scripts/netopyu-provider bundle \
  --manifest provider-manifest.json \
  --manifest-signature provider-manifest.sig.json \
  --qualification provider-qualification.json \
  --qualification-signature provider-qualification.sig.json \
  --output provider-release.json

scripts/netopyu-provider verify \
  --bundle provider-release.json \
  --trust-store /etc/netopyu/provider-trust.json
```

部署控制器观察运行环境中的 exact artifact digest 后生成 attestation；独立 Deployer 签名并组合：

```bash
scripts/netopyu-provider sign-deployment \
  --deployment-attestation deployment-attestation.json \
  --private-key /run/secrets/provider-deployer.key \
  --key-id deployer-2026 \
  --ttl-seconds 518400 \
  --output deployment-attestation.sig.json

scripts/netopyu-provider deployment-bundle \
  --deployment-attestation deployment-attestation.json \
  --deployment-signature deployment-attestation.sig.json \
  --output signed-deployment.json

scripts/netopyu-provider verify-deployment \
  --bundle provider-release.json \
  --deployment signed-deployment.json \
  --environment staging \
  --trust-store /etc/netopyu/provider-trust.json
```

严格策略在 Trust Store 设置 `required_artifacts: [oci-image, sbom, provenance]` 和 `require_deployment_attestation: true`。

验证后显式 stage、publish、promote：

```bash
export NETOPYU_PROVIDER_RELEASE_DB=/var/lib/netopyu/provider-releases.sqlite

scripts/netopyu-provider stage \
  --bundle provider-release.json \
  --trust-store /etc/netopyu/provider-trust.json

scripts/netopyu-provider publish \
  --release-digest sha256:... \
  --trust-store /etc/netopyu/provider-trust.json

scripts/netopyu-provider promote \
  --release-digest sha256:... \
  --environment staging \
  --deployment signed-deployment.json \
  --trust-store /etc/netopyu/provider-trust.json
```

同一 Provider 的兼容版本必须递增版本且不能删除或改变既有 Capability 的 identity/schema/result/L0/security fields。breaking release 必须在 Manifest 中声明 `compatibility=breaking`、精确绑定当前 Manifest 的 `supersedes`，promote 时还必须提供部署审批引用：

```bash
scripts/netopyu-provider compatibility \
  --previous provider-v1.json --candidate provider-v2.json

scripts/netopyu-provider promote \
  --release-digest sha256:... --environment production \
  --approval-reference CHG-12345 \
  --deployment signed-deployment.json \
  --trust-store /etc/netopyu/provider-trust.json
```

回滚只切换到同一 Provider 已签名、已验证、已发布或已弃用的 exact release，并记录审批引用：

```bash
scripts/netopyu-provider rollback \
  --provider-id vendor.fabric --environment production \
  --approval-reference CHG-ROLLBACK-12345 \
  --deployment signed-rollback-deployment.json \
  --trust-store /etc/netopyu/provider-trust.json

scripts/netopyu-provider audit \
  --trust-store /etc/netopyu/provider-trust.json
```

### 5. Runtime admission

默认 `disabled` 保持本地实验兼容；生产显式启用：

```bash
export NETOPYU_PROVIDER_ADMISSION=enforced
export NETOPYU_PROVIDER_TRUST_STORE=/etc/netopyu/provider-trust.json
export NETOPYU_PROVIDER_RELEASE_DB=/var/lib/netopyu/provider-releases.sqlite
export NETOPYU_PROVIDER_ENVIRONMENT=production
```

每个 MCP server 的部署配置增加 `release_provider_id`。OpenAPI 还必须配置部署拥有的 `provider_identity`，不能使用默认 unpinned identity。Admission 缺配置、无 active release、缺少/过期/不匹配 deployment、签名过期/撤销、资格过期、artifact/identity/schema/result/L0 不同，都会在 Provider 调用前失败关闭。schema-v9 plan 固定 deployment digest；审批后即使 release 不变但部署证明改变，也会在 Provider 调用前以 `precondition_changed` 零写入终止。

SQLite hash chain 只提供本地篡改检测；攻击者若能重写数据库和链头仍可伪造。P1.7 必须把 release event/head 复制到外部 append-only/WORM 审计系统。

---

## English

### 1. Status and boundary

P1.4-B-ready provides a runnable local release reference with strict Manifests, independent Publisher/Qualifier/Deployer Ed25519 roles, a repository-external JSONL process protocol, the fixed nine-case suite with real process restart, required OCI-image/SBOM/provenance digests, deployment-controller attestations, durable lifecycle state, compatibility gates, hash-chained events, and Runtime admission. PreparedPlan schema v9 additionally binds the exact deployment digest.

This is not production certification. Tests copy a fixture outside the repository and execute it as a separate process, but its source remains repository-owned. Keys are ephemeral and artifacts are digest fixtures rather than OCI registry, SBOM-content, or SLSA verification. P1.4-B still requires an independently owned Provider repository, organizational CI/lab and signing/HSM roots, real artifact services, and external WORM audit.

### 2. Security model

The Publisher signs exact artifacts and Capability contracts. An independent Qualifier signs a nine-of-nine report. A separate Deployer signs a short-lived observation of the exact release, artifact map, and environment. Trust keys are role-, provider-, time-, and revocation-scoped, and key material cannot cross the three roles. Deployment configuration—not MCP tool self-assertion—selects `release_provider_id`.

Admission requires discovery and the active deployment proof to match the release exactly. Schema-v9 plans bind the deployment digest, so even same-release redeployment drift fails before Provider invocation.

### 3. Operation

Use the commands in the Chinese section to qualify an external process, emit JSON Schemas, sign three independent evidence classes, bundle, verify, stage, publish, attest deployment, promote, compare, rollback with fresh target-release evidence, inspect status, and audit the local chain. Compatible releases may add capabilities but may not silently change existing security or semantic fields.

Enable Runtime admission with the four environment variables shown above and configure each MCP/OpenAPI deployment with its release provider id and pinned identity. Local SQLite audit is not a remote trust anchor; external append-only/WORM evidence remains P1.7 work.
