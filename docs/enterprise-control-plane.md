# 企业身份与审批控制面 / Enterprise Identity and Approval Control Plane

## 中文

### 1. 状态与边界

P1.3-B1 已提供可运行的 OIDC/JWKS、Gateway sender attestation、HTTP PDP 和 HTTP Change Authority 参考 Adapter。B2-ready 接入包进一步加入按 Harness session 动态签发 Gateway attestation、显式 CA/mTLS transport、无泄密配置 Doctor 和无网络效果的 live contract check。它们已用真实 RS256/JWKS/HTTP 和本地证书完成资格测试，但没有连接具体企业系统，因此仍不是 B2 生产认证。

默认 `local-simulation` 不需要本文件中的配置。启用 `NETOPYU_IDENTITY_MODE=enforced` 后，OIDC、Gateway、PDP 和 Change Authority 必须全部配置；任一缺失或不可用都会关闭受保护 read/write。

### 2. 进程配置

以下值必须由部署系统或 secret manager 注入，不得写入 Git、prompt、Skill、plan 或日志：

```bash
export NETOPYU_IDENTITY_MODE=enforced

export NETOPYU_OIDC_ISSUER=https://idp.example.com
export NETOPYU_OIDC_AUDIENCE=netopyu-runtime
export NETOPYU_OIDC_JWKS_URL=https://idp.example.com/.well-known/jwks.json
export NETOPYU_OIDC_ALGORITHMS=RS256
export NETOPYU_OIDC_MIN_AAL=2

export NETOPYU_GATEWAY_ISSUER=https://agent-gateway.example.com
export NETOPYU_GATEWAY_AUDIENCE=netopyu-gateway
export NETOPYU_GATEWAY_JWKS_URL=https://agent-gateway.example.com/.well-known/jwks.json
export NETOPYU_GATEWAY_MINT_URL=https://agent-gateway.example.com/v1/attest

export NETOPYU_PDP_URL=https://pdp.example.com/v1/decide
export NETOPYU_CHANGE_AUTHORITY_URL=https://change.example.com/v1/qualify
export NETOPYU_CONTROL_PLANE_BEARER_TOKEN='<secret-manager-reference>'

# 生产推荐：部署信任根与双向 TLS。private key 必须为 0600。
export NETOPYU_CONTROL_PLANE_CA_BUNDLE=/run/secrets/enterprise-ca.pem
export NETOPYU_CONTROL_PLANE_CLIENT_CERT=/run/secrets/netopyu-client.pem
export NETOPYU_CONTROL_PLANE_CLIENT_KEY=/run/secrets/netopyu-client.key
export NETOPYU_CONTROL_PLANE_TRUST_ENV=0

export NETOPYU_OIDC_TOKEN='<requester-access-token>'
export NETOPYU_APPROVER_OIDC_TOKEN='<approver-access-token>'
export NETOPYU_CHANGE_TICKET=CHG-12345
```

配置 `NETOPYU_GATEWAY_MINT_URL` 后，DSH/Hermes 只需把当前人的 access token 从进程 secret 投影给 Runtime；Runtime 以 access token、Harness、session 和 purpose 向 Gateway 请求短时 attestation，再完成 `act_sub + subject_jti + harness + session` 校验。若没有 mint endpoint，调用者必须提供 `NETOPYU_GATEWAY_TOKEN` 和 `NETOPYU_APPROVER_GATEWAY_TOKEN`。可用独立 `NETOPYU_GATEWAY_MINT_BEARER_TOKEN`，未配置时回退到公共 control-plane bearer。

所有 JWKS、mint、PDP 和 Change 请求使用同一显式 transport。默认 `trust_env=false`，不会隐式采信进程代理/CA 环境；生产可配置私有 CA 和 mTLS。`NETOPYU_ENTERPRISE_ALLOW_LOOPBACK_HTTP=1` 只允许 `localhost` 或 loopback IP 的本地资格实验；非 loopback HTTP 始终拒绝。

### 3. JWT 合同

人的 access token 必须具有标准 `iss/aud/sub/exp/iat/nbf/jti/sid`，以及：

- `roles`: string array；
- `scope`: 空格分隔字符串或 string array；
- `clearance`: `public|internal|confidential|restricted`；
- `aal`: 1–4 数字，必须不低于配置阈值；
- `token_use`: `access` 或 `access_token`；
- 可选 `purpose` 和 `azp`。

Gateway attestation 是另一 issuer 签发的短时 JWT，必须具有：

- 标准 `iss/aud/sub/exp/iat/nbf/jti/sid`；
- `token_use=gateway_attestation`；
- `harness=dsh|hermes`；
- `client_id`；
- `act_sub` 等于 access token 的 `sub`；
- `subject_jti` 等于 access token 的 `jti`。

OIDC `sid` 表示 IdP 登录会话；Gateway `sid` 表示当前 Harness session。Runtime 不混淆这两个概念。最终 plan/journal 只保存规范化主体、Gateway 公共 identity 和 credential digest，不保存 JWT。

### 4. PDP 合同

Runtime POST `netopyu.pdp-request/v1`，`action` 为：

- `observation.read`；
- `effect.prepare`；
- `effect.approve`。

响应必须为 `netopyu.pdp-decision/v1`，包含 boolean `allow`、`decision_id`、`policy_id`、`policy_version`、`evaluated_at` 和 object `obligations`。支持的审批 obligation 是 `required_approvers`、`separation_of_duties` 和 `require_change_ticket`；它们只能收紧内置 L0 policy。deny、超时、非 2xx、redirect、超大或错误 schema 响应均失败关闭。

### 5. Change Authority 合同

Runtime POST `netopyu.change-query/v1`。`netopyu.change-record/v1` 响应必须匹配 ticket id，并提供 approved status、record id、revision、approved-by、带时区的活动 window、allowed profiles/capabilities/targets 和 risk ceiling。Runtime 只将最小公开 change evidence 加入签名 proof。

### 6. Doctor 与现场合同资格检查

Doctor 不访问网络，只检查必填项、HTTPS/loopback 策略、JWT algorithm/AAL、证书组合和私钥权限。输出只含 endpoint/certificate digest 和 credential-present boolean：

```bash
scripts/netopyu-enterprise doctor
```

live contract check 会实际验证 requester/approver JWT 与 JWKS，按需动态 mint 两个 Gateway attestation，并调用 `observation.read`、`effect.prepare`、`effect.approve` PDP 与 Change Authority；它不会创建计划或执行任何网络/业务写操作：

```bash
export NETOPYU_CONFORMANCE_CHANGE_TICKET=CHG-12345
scripts/netopyu-enterprise contract-test \
  --session-id qualification-20260828 \
  --profile lan \
  --capability-id grant_user_access \
  --target qualification-target \
  --risk-level low
```

输出不包含 token、bearer、原始 endpoint、client key 或主体明文；失败返回非零 exit code。该检查只证明接口合同和当次可达性，不证明高可用、吊销传播或生产 SLO。

### 7. 本地回归测试

```bash
.venv/bin/python -m pytest -q tests/test_enterprise_control_plane.py
scripts/netopyu-dsh retirement
```

测试覆盖成功 read/write 链、动态 session attestation mint、显式 CA/mTLS、无泄密 Doctor/live contract、主体替换、access-token 替换、raw role 注入、unknown kid、PDP deny、ticket deny 和 ticket scope mismatch。真实企业 B2 仍必须使用现场 issuer、组织 RBAC/ABAC 和变更系统验证 key rotation/revocation、吊销传播、Gateway/PDP/审批可用性、证书轮换和不可抵赖审计。

---

## English

### 1. Status and boundary

P1.3-B1 provides runnable OIDC/JWKS, Gateway-attestation, HTTP PDP, and HTTP Change Authority reference adapters. The B2-ready package adds per-Harness-session Gateway minting, explicit CA/mTLS transport, a secret-safe configuration Doctor, and a no-effect live contract check. Real RS256/JWKS/HTTP and local certificate paths are qualified, but no specific enterprise system has been connected; this is not B2 production certification.

The default `local-simulation` mode needs none of these settings. With `NETOPYU_IDENTITY_MODE=enforced`, all four authorities are mandatory and any missing or unavailable dependency closes protected reads and writes.

### 2. Process configuration

Use the environment template in the Chinese section through a deployment system or secret manager. Never place tokens or bearer secrets in Git, prompts, Skills, plans, or logs. With a mint URL, the Runtime obtains a short-lived Gateway JWT bound to the current Harness session; otherwise the two Gateway attestations are mandatory caller credentials. All authority clients share explicit CA/mTLS configuration and disable environment trust by default. Plain HTTP remains loopback-qualification-only.

### 3. Credential contract

The human access token carries standard registered claims plus roles, scope, clearance, assurance level, and access-token use. A separately issued Gateway JWT binds the Harness, Harness session, and client. `act_sub` must equal the human `sub`, while `subject_jti` must equal the access-token `jti`. The IdP login `sid` and Harness-session `sid` remain distinct. Only normalized public identity and credential digests enter the plan or journal.

### 4. Policy and change contracts

The PDP receives `netopyu.pdp-request/v1` for `observation.read`, `effect.prepare`, or `effect.approve` and returns a complete `netopyu.pdp-decision/v1`. Approval obligations can only tighten built-in L0 policy. The Change Authority receives `netopyu.change-query/v1`; its `netopyu.change-record/v1` must prove ticket identity, approval status, revision, active window, resource scope, and risk ceiling. Transport or schema uncertainty fails closed.

### 5. Doctor and qualification

Run the Doctor and contract-test commands in the Chinese section. Doctor is offline and secret-safe. Contract-test exercises requester/approver JWKS verification, dynamic Gateway minting, read/prepare/approve PDP decisions, and change qualification without creating or executing effects. Local regression covers the protocol and negative fault set. B2 still requires the user's real issuer, organizational RBAC/ABAC, key and certificate rotation, revocation propagation, authority availability, and non-repudiation evidence.
