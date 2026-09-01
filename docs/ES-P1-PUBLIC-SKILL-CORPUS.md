# ES-P1 公开 Skill 市场语料 / ES-P1 Public Skill-Market Corpus

## 中文

### 定位

SkillsMP、GitHub 和厂商公开仓库中的 Skill 可以降低用例构造成本，并增加跨作者、跨领域、真实格式和长尾写法的覆盖。它们应作为 `ES-P1-Wild` 公开外部语料层，而不是替代正式的 `ES-P1-Private` 独立私有资格门。

| 证据层 | 回答的问题 | 可支持的主张 |
|---|---|---|
| Synthetic sealed | 自动化流水线和大规模组合是否可运行 | 合成泛化信号 |
| ES-P1-Wild | 对公开市场中真实 Skill 的格式和语义是否兼容 | 公开生态兼容性、外部有效性 |
| ES-P1-Private | 冻结系统能否处理开发者不可见的独立人工用例 | 正式独立泛化资格 |
| ES-P1-Sec | 面对恶意、畸形或越权 Skill 包是否仍保持零执行和 fail-closed | Skill 供应链安全边界 |

公开 Skill 可能已进入模型训练数据，来源之间也可能复制、批量生成或质量不一；Skill 本身通常没有任务输入、Gold 结果和事务 Oracle。因此，公开市场结果不能单独证明隐藏集泛化，也不能直接算生产成功概率。

### 采样和冻结

建议选择 50–100 个 Skill，每个构造 3–5 个任务，形成 200–500 个 paired cases。采样计划在下载前预注册，并按来源仓库、作者、领域、语言和结构特征分层；同一仓库或近重复 Skill 只能进入同一数据 split，防止泄漏。

每个样本必须保存：原始 URL、仓库、commit SHA、检索时间、许可证、Skill/附件摘要、特征标签和排除原因。SkillsMP 只用于发现与元数据检索；评测输入必须回到源 Git 仓库的固定 commit，不能依赖可变搜索结果或“热门度”作为正确性标签。

### 硬安全边界：默认零执行

下载的整个 Skill 包始终是不可信数据。采集和转译阶段必须满足：

1. 下载到仓库外隔离区，不安装到 Agent 的 Skill 搜索路径；
2. 拒绝符号链接、硬链接、路径穿越、特殊文件、嵌套归档和超过限额的文件/目录；
3. 不执行脚本，不执行安装器，不解析 shell substitution，不加载未知插件/MCP，不提供密钥；
4. `scripts/`、hooks、二进制和宏只做摘要、类型识别与静态能力分析；LLM 读取内容也不能授予执行权；
5. 默认无网络、只读输入、临时输出、非 root、CPU/内存/时间/进程数限制；
6. 只有单独的 `ES-P1-Sec` 动态安全实验可以运行样本，而且必须使用一次性强隔离沙箱、空凭证和完整系统调用/网络审计；其结果不得与功能资格混算。

若一个公开 Skill 依赖脚本才能产生语义结果，主功能评测应使用声明式脚本接口或确定性仿真 Provider 替代；无法建立可信 Oracle 时，只评估解析、语义保留和安全停机，不评估任务成功。

### 评测单元与指标

公开 Skill 不是完整用例。每个资格单元应为：

`固定 Skill 包 + 用户任务 + 输入 fixture + Tool/MCP 能力清单 + Gold 意图/参数/风险/Effect 预算 + 预期结果/失败处置`

继续采用同一控制变量：

- Control：`DSH + 原始 L1 Skill → LLM/Tool`；
- Treatment：`DSH + 同一 L1 Skill → 转译门禁 → 合格 L0 Runtime / 不合格 safe-stop`。

至少报告：市场 Skill 接纳率、原始协议有效率、语义覆盖、关键遗漏、虚构 Effect、风险/审批弱化、false accept、safe-stop、任务完成率、执行精度、不安全执行、错误提交、Effect 预算违反、p50/p95，以及按来源/领域/结构/含脚本与否分层的置信区间。不得用下载量、star 或模型自评分替代 Gold Oracle。

### 实施顺序

1. 实现只读 `market-corpus discover/snapshot/inspect`，先支持 SkillsMP 元数据和 GitHub commit 封存；
2. 实现 archive/package quarantine gate 与 `executionPolicy=static_only`；
3. 用 20 个无脚本 Skill 做 importer pilot，再扩到包含 references、scripts、分支、审批和组合的 50–100 Skill；
4. 由独立人员补任务、Gold 和 Oracle，完成 ES-P1-Wild paired run；
5. 保留全新、私有、独立人工 ES-P1-Private 作为正式资格门；
6. 将恶意脚本、提示注入和供应链攻击放入独立 ES-P1-Sec，不在普通 Agent 环境试运行。

### 首轮静态 Pilot（2026-09-01）

第一阶段已实现 `scripts/netopyu-market-corpus`：

- `discover`：从 SkillsMP 只读发现并摘要封存候选；
- `snapshot`：回到 GitHub 源仓库，以 API 或无 checkout 的 bare Git 后端固定 commit；
- `inspect`：校验 Manifest、记录、实际文件集合、逐文件摘要和零执行边界；
- `report`：生成双语静态兼容性报告。

真实 pilot 从 7 类查询发现 100 个候选，处理前 60 个后得到 20 个满足“许可证可识别且无可执行文件表面”的包，覆盖 13 个源仓库。35 个因未声明许可证被拒绝，1 个因存在可执行表面被拒绝，4 个因源包/快照错误被拒绝。整个过程中第三方代码执行次数和可执行文件物化数均为 0。

现有严格 Runtime 包门禁对 20 个包的结果为 15 passed、5 blocked。Blocked 暴露三类真实生态差异：3 个非标准 frontmatter、跨包/缺失资源引用，以及父目录引用。它们不会被静默修复或获得 Runtime 权限。文本静态扫描还标出 destructive command 2、external download 2、privilege escalation 1；这些只是不可执行文本信号。

```bash
scripts/netopyu-market-corpus discover \
  --query 'network automation' --query 'approval workflow' \
  --query 'cloud operations' --query 'incident response' \
  --limit 100 --max-per-repo 5 --output /ABS/PATH/discovery.json

scripts/netopyu-market-corpus snapshot /ABS/PATH/discovery.json \
  --output-root /ABS/PATH/static-pilot --limit 20 \
  --script-policy exclude --license-policy known --source-backend git

scripts/netopyu-market-corpus inspect /ABS/PATH/static-pilot
scripts/netopyu-market-corpus report /ABS/PATH/static-pilot \
  --discovery /ABS/PATH/discovery.json \
  --output-root artifacts/es-p1-wild-pilot
```

完整本地报告见 `artifacts/es-p1-wild-pilot-20/public-skill-pilot-report.{json,md}`；可提交摘要见 [pilot benchmark summary](benchmarks/es-p1-wild-pilot-summary.json)。当前状态是 `static_import_and_model_draft_assist_complete_independent_eval_not_started`：已有非权威模型草案，但还没有独立人员接受的任务、Gold/Oracle 或 DSH paired run。

在此基础上，`author-kit` 已从 15 个通过严格包门禁的 Skill 生成仓库外独立标注工作区，共 45 个任务槽位（每 Skill 3 个：nominal、ambiguous/missing、failure/adversarial）。工作区包含固定包、来源 commit、Task/Gold/Tool Catalog JSON Schema、任务分配和摘要链，但明确不含 Runtime、Evaluator、模型输出或自动 Gold：

```bash
scripts/netopyu-market-corpus author-kit /ABS/PATH/static-pilot \
  --output-root /ABS/PATH/independent-author-kit --tasks-per-skill 3
scripts/netopyu-market-corpus author-kit-inspect /ABS/PATH/independent-author-kit
```

当前本地工作区为 `/Users/steven/Documents/Codex/2026-08-26/wo/ensuredskill-market-author-kit-15`，workspace digest 为 `sha256:d476e417f9218b138433c7cf9f4ddec3674c090d41c6f3ef9389820945ea3012`。它只是角色隔离的空白研究输入，不是完成的用例或资格证据。

### 9B 草案辅助（2026-09-01）

为降低独立作者从空白开始的工作量，新增了显式降级的 `draft-author` 通道。它只向 qwen3.5:9b 披露隔离包中的 `SKILL.md` 和 `references/` 有界文本，不读取或执行脚本、hooks、安装器、插件或 MCP。输出只包含候选 Task 草案；不生成 Gold、Oracle、fixture、签名或 Runtime 执行权。

```bash
scripts/netopyu-market-corpus draft-author /ABS/PATH/independent-author-kit \
  --output-root /ABS/PATH/model-drafts --model qwen3.5:9b
scripts/netopyu-market-corpus draft-inspect \
  /ABS/PATH/model-drafts /ABS/PATH/independent-author-kit
```

本轮对 15 个公开 Skill、45 个预分配槽位运行后的真实结果如下：

| 指标 | 结果 | 解释边界 |
|---|---:|---|
| Assignment 结构/安全形状通过 | 14/15（93.33%） | 只表示 schema、槽位和安全形状合规 |
| Task 槽位草案覆盖 | 42/45（93.33%） | 未通过的 3 个槽位保持空白，交给人工作者 |
| 需要模型修复的 assignment | 12/15 | 说明小模型首轮协议输出并不稳定 |
| 额外修复调用 | 12 | 总模型调用 27 次 |
| 时延 | p50 59.2 秒；p95/max 97.2 秒 | 本机 qwen3.5:9b、15 个 assignment |
| 持续失败 | 1 | nominal 被写成 `proposal`，但 Effect budget 为 0；严格拒绝且未静默改写 |

最终产物位于 `/Users/steven/Documents/Codex/2026-08-26/wo/ensuredskill-market-model-drafts-15-9b-v2`，report digest 为 `sha256:da09d171a1c9f303dc179a029bad3dcffe95a1708b6fcefa9f0536674d2059ca`，独立检查器返回 `verified=true`。这些数字衡量的是草案生成协议与安全形状，不是语义准确率；只有独立人员依据原始 Skill、fixture 和 Tool Catalog 审阅/重写并补充 Gold 与 Oracle 后，相关 case 才可进入 ES-P1-Wild paired evaluation。

### 测试 Skill 索引库

`library` 命令把本轮测试 Skill 生成成自包含、离线、只读的浏览页。左侧可搜索并按草案状态筛选 15 个 Skill；右侧展示固定仓库/commit、许可证、package digest、`SKILL.md`、references/agents 等附件，以及三个任务槽位和 9B 草案。第三方内容只通过 DOM `textContent` 显示，不解析 Markdown/HTML；页面 CSP 禁止网络、对象、frame、表单和外部资源，也没有安装、审批、注册、Tool/MCP 或 Runtime 执行入口。

```bash
scripts/netopyu-market-corpus library /ABS/PATH/independent-author-kit \
  --draft-root /ABS/PATH/model-drafts \
  --snapshot-root /ABS/PATH/static-pilot \
  --output-root artifacts/es-p1-wild-skill-library
scripts/netopyu-market-corpus library-inspect \
  artifacts/es-p1-wild-skill-library
open artifacts/es-p1-wild-skill-library/skill-library.html  # macOS
```

本轮索引包含 15 个 Skill、22 个可显示文件、4 个 reference 文件、45 个任务槽位和 42 个草案任务；`executionSurfaceFileCount=0`、`thirdPartyExecutionAttempted=false`，检查结果为 `verified=true`。完整第三方内容只保留在默认不提交的 `artifacts/`；可提交的 [测试 Skill 元数据索引](benchmarks/es-p1-wild-skill-index.json) 不嵌入第三方文件原文，只保存固定来源链接和评测绑定信息。

## English

Public Skills from SkillsMP, GitHub, and vendor repositories should form an `ES-P1-Wild` ecological-validity track. They reduce authoring cost and expose the system to real package structures and cross-author variation, but they do not replace `ES-P1-Private`: public content may be present in model training data, duplicated or generated, and normally lacks task inputs and trusted outcome Oracles.

The evidence program therefore has four distinct strata: sealed synthetic evidence, public in-the-wild compatibility, independently authored private qualification, and an adversarial package-security track. A recommended public corpus samples 50–100 Skills and derives three to five tasks per Skill, with source-repository grouping, near-duplicate control, commit pinning, license/provenance records, and a preregistered sampling rule.

Every downloaded package is untrusted data. It must remain outside Agent discovery paths; archives, links, special files, installers, hooks, scripts, binaries, macros, plugins, and MCP definitions must never execute during collection or translation. Script-bearing Skills are evaluated statically or against declared deterministic substitutes. Dynamic malicious-package tests belong only in a disposable no-secret, no-network, non-root sandbox with resource limits and system-call auditing.

The evaluation unit is a pinned Skill package plus a user task, fixtures, Tool/MCP catalog, Gold intent/parameters/risk/effect budget, and expected outcome or failure disposition. The paired comparison remains native DSH plus the original L1 Skill versus the same DSH/L1 input routed through the translation gate to qualified L0 Runtime or safe-stop. Public-market results support ecosystem compatibility; only the independent private track may satisfy the formal ES-P1 generalization gate.

The first static pilot is complete. It discovered 100 Skills across seven query families and processed 60 candidates to accept 20 script-free, license-identified packages from 13 repositories. Thirty-five candidates lacked a recognized license, one exposed executable content, and four had source/snapshot failures. No third-party code was executed and no executable file was materialized. The existing strict Runtime package gate passed 15 packages and blocked five for non-standard frontmatter or unresolved/out-of-bound package references. This is useful in-the-wild compatibility evidence, but no independently accepted tasks, Gold/Oracles, or paired DSH evaluation exist yet.

An independent annotation kit has now been exported from the 15 packages that passed the strict gate. It contains 45 blank task slots, pinned package/source evidence, and Task/Gold/Tool-Catalog schemas, but no Runtime, evaluator, model output, generated Gold, credentials, or execution authority. Its role is to make independent authoring reproducible without manufacturing independence inside the project.

A separate, explicitly degraded qwen3.5:9b draft-assistance lane now reduces blank-page authoring work. It discloses only bounded `SKILL.md` and `references/` text from the quarantined packages and never executes package content. In the first 15-assignment run, 14 assignments and 42/45 task slots passed protocol and safety-shape validation (93.33%); 12 assignments needed a repair call, latency was 59.2 seconds p50 and 97.2 seconds p95, and one deterministic proposal/effect-budget inconsistency remained rejected. The sealed report verified successfully. These are draft-protocol metrics, not semantic Gold accuracy: independent humans must review or rewrite every draft and author fixtures, Tool Catalogs, Gold semantics, and Oracles before any paired ES-P1-Wild evaluation.

The `library` command now exports a self-contained offline browser for the 15 tested Skills. Users can search the list, filter draft status, inspect pinned provenance and license metadata, and click through `SKILL.md`, bundled text files, task slots, and model drafts. Untrusted content is inserted only as plain text under a network-free CSP, and the page exposes no install, approval, Tool/MCP, registration, or execution action. The current library contains 22 displayable files, four references, and 42 draft tasks; its digest-bound inspection passes with zero third-party execution. A versioned [metadata-only tested-Skill index](benchmarks/es-p1-wild-skill-index.json) links to exact upstream commits without redistributing package bodies in Git.
