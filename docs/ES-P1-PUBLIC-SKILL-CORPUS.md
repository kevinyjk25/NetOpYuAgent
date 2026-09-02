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

> 2026-09-03 更新：本页后续 20/15-Skill 内容是首轮历史 pilot。当前库已扩展为 100 个静态接纳 Skill、72 个仓库、9 个领域；其中 71 个进入主要转译开发语料、53 个同时 Runtime-package ready、18 个仅转译/引用上下文不完整、29 个只作格式鲁棒性测试。新的证据顺序、硬门禁和命令以 [L1→L0 泛化门禁](TRANSLATION-GENERALIZATION-GATE.md) 为准。大规模 Runtime paired run 在门禁通过前已被代码锁定。

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

后续 ES-P1 新模型运行统一使用 `qwen3.5:9b`。已有 7B 结果只作为已冻结的历史对照保留，不再扩充，也不据此声明当前可用性。

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

### Case Author 独立审阅门

`review-kit` 将模型草案进一步降权：只提取 42 个用户问题候选，主动扣除模型生成的 intended outcome、disposition、能力、参数、风险、审批、Effect budget、验证和恢复字段。每个槽位必须由 Case Author 明确选择 `accept_prompt`、`edit_prompt`、`author_from_scratch` 或 `reject_slot`，填写理由、作者身份、独立性声明和完整 Task；这里禁止出现 Gold/Oracle 字段。

```bash
scripts/netopyu-market-corpus review-kit \
  /ABS/PATH/independent-author-kit /ABS/PATH/model-drafts \
  --output-root /ABS/PATH/case-author-review-kit
scripts/netopyu-market-corpus review-kit-inspect \
  /ABS/PATH/case-author-review-kit
open /ABS/PATH/case-author-review-kit/review-queue.html  # macOS
```

v2 首先加入 Tool Catalog 与 fixture 引用封存；v3 在此基础上增加可执行的声明式 Tool Catalog v2 和 fixture-state Schema。材料只能放在 `materials/catalogs/` 和 `materials/fixtures/`，采用允许的文本/JSON/YAML/CSV 类型，限制单文件和总大小，禁止符号链接、二进制、脚本和未封存文件。当前真实工作区位于 `/Users/steven/Documents/Codex/2026-08-26/wo/ensuredskill-market-assisted-review-kit-15-v3`，workspace digest 为 `sha256:072a5bb83eb95f1c049c6c5c99ae518ff7e21b3d889a20ef1f6f0dec3fdc48ec`。状态仍为 45 pending、42 个问题候选、3 个从零编写槽位、0 个材料文件、`goldAuthorKitExportEligible=false`；v2 工作区仅保留为历史制品。

### 声明式 Tool Catalog v2 与 fixture MCP

为避免每个公开 Skill 都编写一套任意 Python/脚本 Provider，同时确保两个实验臂使用同源环境，新增了通用确定性适配器。Tool Catalog v2 只能从六种受审操作中选择：`static`、`read_record`、`validate_record`、`upsert_record`、`restore_record`、`delete_record`。它只声明工具名、Capability、封闭输入 Schema、读写类型和字段映射；不接受代码、shell、模板执行、动态 import 或任意 callable。

每个 Task 绑定一个 fixture-state v1，声明初始集合、审批决定、故障种子、静态结果和验证不一致补丁。SQLite 保存每个实验臂独立但同源的状态和调用审计。支持 `provider_error_before_send`、`after_send_unknown`、`verification_mismatch` 和 `compensation_failure`。`l1_native` 直接暴露声明工具；`safe_stop` 禁止 mutation；`l0_runtime` 的 effect 必须由 BackendSession 注入内部 `effect_phase`，模型直接调用会得到 `runtime_transaction_required`。

可复制样例：

- [Tool Catalog v2](../evaluation/fixtures/public-skill-fixture/catalog-v2.json)
- [fixture-state v1](../evaluation/fixtures/public-skill-fixture/case-state-v1.json)

官方 MCP stdio 本地启动方式：

```bash
.venv/bin/python -m evaluation.public_skill_fixture_mcp \
  --catalog /ABS/PATH/catalog-v2.json \
  --fixture /ABS/PATH/case-state-v1.json \
  --store /ABS/PATH/fixture-state.sqlite \
  --mode l1_native
```

本适配器只是公开用例的确定性 Infrastructure 仿真，不是生产 Provider，也不能替代 L1→L0 转译报告、Runtime Contract 或 Gold Oracle。

盲态 Gold Author Kit 已实现，但只有在全部 Case Author 决定完成且独立性声明有效后才允许导出。它只复制人工接受的 Task、封存材料、Case Author 来源摘要和空白 Gold/Oracle 模板，不包含模型语义候选；Gold Author 必须声明未看过模型语义答案。即使 Gold 编写完成，检查器也固定返回 `officialEsP1QualificationEligible=false`，因为后续仍需角色隔离的复核、paired run 与正式资格流程。

```bash
scripts/netopyu-market-corpus gold-kit /ABS/PATH/case-author-review-kit \
  --output-root /ABS/PATH/blind-gold-author-kit
scripts/netopyu-market-corpus gold-kit-inspect \
  /ABS/PATH/blind-gold-author-kit
```

对上述真实 v2 工作区执行 `gold-kit` 会明确失败且不创建输出目录，这是当前正确的 fail-closed 结果，不是待绕过的错误。

### Paired Study 输入封存

人工 Gold 完成后，`paired-kit` 才能把 Gold Kit 与原始 Author Kit 绑定成实验输入。输出采用物理分区：`agent/` 只包含固定 Skill 包、人工 Task、Tool Catalog 和 fixture；`scoring/` 只包含 Gold/Oracle；`evidence/` 保存角色与摘要来源。任何一个 DSH 实验臂都不得读取 `scoring/`。Study Plan 固定使用 `qwen3.5:9b`、三个重复、相同 Task/Skill/Tool/fixture/审批与故障输入，并固定 Treatment 不合格时禁止恢复原生写。

```bash
scripts/netopyu-market-corpus paired-kit \
  /ABS/PATH/blind-gold-author-kit /ABS/PATH/independent-author-kit \
  --output-root /ABS/PATH/paired-study-kit
scripts/netopyu-market-corpus paired-kit-inspect \
  /ABS/PATH/paired-study-kit
```

该命令只封存 paired study 输入，不运行模型、不注册 Tool/MCP、不执行第三方内容，也不产生 paired 结果或正式资格。它分别报告 `fixtureMcpInputEligible` 与 `translationReportAttached`：只有前者为真且后续绑定冻结的 9B L1→L0 转译报告，paired runner 输入才可能就绪；Study Kit 自身固定保持 `pairedExecutionInputEligible=false`。检查器会拒绝 Gold 串入 Agent、Skill/材料漂移、能力目录缺失、未封存文件、研究计划漂移和资格标志伪造。当前真实 Review Kit 尚未通过人工门，因此 Gold 与 paired workspace 均不会生成。

### 9B 转译绑定与真实 DSH paired runner

人工 Paired Study 就绪后，后续链路分成三个显式步骤。`translate` 只读取 `agent/`：模型提出语义候选，确定性校验器再检查 Capability、唯一 Effect、参数、审批、预检、验证、补偿和脚本禁用。模型 confidence 只是证据；省略的参数只能从已封存 Tool Catalog 的闭合 input schema 补全，模型填写的参数若与 Catalog 不完全一致则拒绝。纯读目录保留原生 L1 fallback；不合格写固定 safe-stop，不能恢复原生写。

```bash
scripts/netopyu-market-corpus translate /ABS/PATH/paired-study-kit \
  --output-root /ABS/PATH/translation --model qwen3.5:9b
scripts/netopyu-market-corpus paired-bind \
  /ABS/PATH/paired-study-kit /ABS/PATH/translation \
  --output-root /ABS/PATH/bound-study
scripts/netopyu-market-corpus paired-run /ABS/PATH/bound-study \
  --output-root /ABS/PATH/paired-result --model qwen3.5:9b
```

`paired-bind` 独立复核 Study 与转译摘要后封存自包含输入。`paired-run` 给 Control/Treatment 暴露逐字节相同的 L1 Skill、Tool Schema、Task、fixture、审批和故障；唯一变量是合格 Effect 是否由 L0 Runtime 接管。Runtime 对写操作固定执行预检、审批、单次 Effect、独立验证、只读对账和按需补偿。运行进程在两个 Agent 臂结束后才解析 `scoring/`，报告区分 smoke 与完整三重复协议。

2026-09-01 的本地技术 smoke 使用固定 qwen3.5:9b 制品 `sha256:6488…3ea7`：一个合法只读用例在 Control/Treatment 均通过（1/1，原生 L1 fallback）；一个名义角色变更在两臂也均通过（1/1），Treatment 的 L0 路径完成一次 Effect 和独立验证。名义用例不能证明 Runtime 优于原生 Agent，单次时延也不能比较；它只证明转译、Catalog 参数物化、真实 DSH Tool loop、L0 事务和事后 Gold 评分已经接通。验证失败补偿路径另有确定性测试覆盖。真实 15-Skill 工作区仍为 45/45 pending，因此这些受控 fixture 不能计作 ES-P1-Wild 成绩或生产成功概率。

### 角色隔离模拟结果

为在真人独立工作开始前完整验证实验机械链路，另建了明确降级的 `ES-P1-Wild-Sim` 工作区。虚拟 `simulated.case-author-a` 和 `simulated.gold-author-b` 使用物理分离的信息面生成 45 个案例及 Gold/Oracle；provenance 固定 `humanIndependent=false`、`officialEsP1QualificationEligible=false`。Translator 运行时不可见 Gold，两臂全部结束后才评分。

固定 qwen3.5:9b、15 Skill、45 case、3 repetitions、2 个隔离 worker 的 135 组成对观察（270 次实验臂）已完整结束：Control 111/135（82.22%），Treatment 132/135（97.78%）；Treatment 胜 21、Control 胜 0，unsafe 和 false commit 两臂均为 0。差值全部来自 `l0_runtime` 路由（21/42→42/42）和 failure/adversarial 类型（24/45→45/45）；原生只读 15/18、safe-stop 75/75 两臂相同。Treatment p50/p95 为 27.8/56.1 秒，Control 为 32.9/109.3 秒。三轮结果分别固定为 37/45 对 44/45。

9B 转译协议有效 45/45，后验路由一致 43/45（95.56%），unsafe Runtime 误接纳 0；两处偏差都是只读任务的保守降级。唯一 Treatment 残余是 `fivem-debugging` 的一个原生只读 case 三次失败，证明 Runtime 没有掩盖 L1/Tool 选择缺口。完整声明、失败分类、命令和摘要链见 [ES-P1-Wild 角色隔离模拟结果](ES-P1-WILD-SIMULATED-RESULTS.md)。该结果完成本地模拟协议，不改变真人 45-slot 工作区仍 pending 的事实，也不替代 ES-P1-Private。

## English

Public Skills from SkillsMP, GitHub, and vendor repositories should form an `ES-P1-Wild` ecological-validity track. They reduce authoring cost and expose the system to real package structures and cross-author variation, but they do not replace `ES-P1-Private`: public content may be present in model training data, duplicated or generated, and normally lacks task inputs and trusted outcome Oracles.

As of 2026-09-03, the 20/15-Skill material below is retained as the historical first pilot. The current static inventory contains 100 accepted Skills from 72 repositories and nine domains: 71 are primary translation-development inputs, including 53 Runtime-package-ready and 18 conformant partial-context packages; another 29 format variants are robustness-only. The authoritative sequencing and admission rules are in the [L1-to-L0 generalization gate](TRANSLATION-GENERALIZATION-GATE.md). Scaled Runtime paired evaluation is now code-blocked until that gate passes.

The evidence program therefore has four distinct strata: sealed synthetic evidence, public in-the-wild compatibility, independently authored private qualification, and an adversarial package-security track. A recommended public corpus samples 50–100 Skills and derives three to five tasks per Skill, with source-repository grouping, near-duplicate control, commit pinning, license/provenance records, and a preregistered sampling rule.

Every downloaded package is untrusted data. It must remain outside Agent discovery paths; archives, links, special files, installers, hooks, scripts, binaries, macros, plugins, and MCP definitions must never execute during collection or translation. Script-bearing Skills are evaluated statically or against declared deterministic substitutes. Dynamic malicious-package tests belong only in a disposable no-secret, no-network, non-root sandbox with resource limits and system-call auditing.

The evaluation unit is a pinned Skill package plus a user task, fixtures, Tool/MCP catalog, Gold intent/parameters/risk/effect budget, and expected outcome or failure disposition. The paired comparison remains native DSH plus the original L1 Skill versus the same DSH/L1 input routed through the translation gate to qualified L0 Runtime or safe-stop. Public-market results support ecosystem compatibility; only the independent private track may satisfy the formal ES-P1 generalization gate.

The first static pilot is complete. It discovered 100 Skills across seven query families and processed 60 candidates to accept 20 script-free, license-identified packages from 13 repositories. Thirty-five candidates lacked a recognized license, one exposed executable content, and four had source/snapshot failures. No third-party code was executed and no executable file was materialized. The existing strict Runtime package gate passed 15 packages and blocked five for non-standard frontmatter or unresolved/out-of-bound package references. This is useful in-the-wild compatibility evidence, but no independently accepted tasks, Gold/Oracles, or paired DSH evaluation exist yet.

An independent annotation kit has now been exported from the 15 packages that passed the strict gate. It contains 45 blank task slots, pinned package/source evidence, and Task/Gold/Tool-Catalog schemas, but no Runtime, evaluator, model output, generated Gold, credentials, or execution authority. Its role is to make independent authoring reproducible without manufacturing independence inside the project.

A separate, explicitly degraded qwen3.5:9b draft-assistance lane now reduces blank-page authoring work. It discloses only bounded `SKILL.md` and `references/` text from the quarantined packages and never executes package content. In the first 15-assignment run, 14 assignments and 42/45 task slots passed protocol and safety-shape validation (93.33%); 12 assignments needed a repair call, latency was 59.2 seconds p50 and 97.2 seconds p95, and one deterministic proposal/effect-budget inconsistency remained rejected. The sealed report verified successfully. These are draft-protocol metrics, not semantic Gold accuracy: independent humans must review or rewrite every draft and author fixtures, Tool Catalogs, Gold semantics, and Oracles before any paired ES-P1-Wild evaluation.

All new ES-P1 model runs use `qwen3.5:9b`. Existing 7B results remain frozen historical comparisons and will not be extended or treated as current availability evidence.

The `library` command now exports a self-contained offline browser for the 15 tested Skills. Users can search the list, filter draft status, inspect pinned provenance and license metadata, and click through `SKILL.md`, bundled text files, task slots, and model drafts. Untrusted content is inserted only as plain text under a network-free CSP, and the page exposes no install, approval, Tool/MCP, registration, or execution action. The current library contains 22 displayable files, four references, and 42 draft tasks; its digest-bound inspection passes with zero third-party execution. A versioned [metadata-only tested-Skill index](benchmarks/es-p1-wild-skill-index.json) links to exact upstream commits without redistributing package bodies in Git.

The assisted Case Author Review Kit then narrows the model output further: it exposes only 42 candidate user prompts and withholds every model-proposed semantic label, parameter, risk, approval, effect budget, verifier, and recovery field. Each of 45 slots requires an explicit human accept/edit/from-scratch/reject decision, rationale, attribution, independence disclosure, and complete Task. Gold/Oracle fields are structurally forbidden. Review Kit v3 retains the bounded material gate and adds executable declarative Tool Catalog v2 plus fixture-state validation. The current v3 workspace digest is `sha256:072a5bb83eb95f1c049c6c5c99ae518ff7e21b3d889a20ef1f6f0dec3fdc48ec`; it still has 45 pending decisions, zero material files, and is correctly ineligible for Gold-author export. v2 remains historical only.

The generic fixture MCP adapter replaces case-specific executable Provider code with six reviewed declarative operations: static, record read, validation, upsert, restore, and delete. A fixture binds initial state, approval, deterministic fault, static results, and verification-mismatch patches. Separate SQLite stores give both arms identical initial inputs without shared mutations, while an append-only call table records argument/result digests and phases. Native mode exposes declared calls, safe-stop denies mutations, and Runtime mode accepts an effect only when the BackendSession injects the internal execution phase. The official MCP stdio path has been exercised locally. This remains a deterministic study simulator, not a production Provider or Runtime qualification.

A blind Gold Author Kit exporter is now available, but it refuses to create output until that independent Case Author gate is complete. A valid export contains only accepted human Tasks, sealed materials, Case Author provenance digests, and blank Gold/Oracle templates; it excludes model semantic candidates and requires the Gold Author to attest that those candidates were not seen. Completion makes a workspace eligible only for paired-evaluation authoring. The inspector deliberately keeps formal ES-P1 qualification false until the remaining independent review and experiment protocol is completed.

Once human Gold is complete, `paired-kit` binds it to the original Author Kit and creates a sealed study input with a physical information split. `agent/` contains only pinned Skills, human Tasks, Tool Catalogs, and fixtures; `scoring/` contains Gold/Oracles and is unavailable to either DSH arm; `evidence/` binds role provenance and digests. The preregistered plan fixes qwen3.5:9b, three repetitions, identical task/package/catalog/fixture/approval/fault inputs, and no native-write fallback for an unqualified Treatment. Fixture-MCP readiness is reported separately from translation-report attachment; the Study Kit alone is never marked ready for paired execution. This command does not run a model, register tools, execute package content, or produce qualification evidence. The real 45-pending workspace cannot yet produce either Gold or paired inputs.

After a human Paired Study exists, `translate`, `paired-bind`, and `paired-run` complete the technical path. Translation reads only `agent/`; model confidence is evidence rather than authority, omitted parameters can only be materialized from the sealed closed Tool schema, and any model-supplied parameter list must match that schema exactly. Read-only catalogs retain native L1 fallback. A qualified mutation requires a unique Effect plus closed preflight, verification, approval, and compensation bindings; every other mutation safe-stops with no native-write fallback.

The bound runner exposes byte-identical L1 Skills, Tool schemas, tasks, fixtures, approvals, and faults to both real DSH arms. Only the qualified Effect backend changes. Gold/Oracle is parsed after both Agent arms finish. A 2026-09-01 qwen3.5:9b technical smoke exercised one read fallback and one nominal Runtime write. Both Control and Treatment passed each one-case nominal Oracle; the Treatment write issued one Effect and reached independently verified success. This is wiring evidence, not evidence that Runtime outperforms native DSH: a nominal one-shot case and its latency cannot support that claim. Deterministic tests separately cover verification failure and compensation. The real 15-Skill workspace remains 45/45 pending, so no controlled fixture may be reported as an ES-P1-Wild result or production probability.

An explicitly downgraded `ES-P1-Wild-Sim` workspace has now completed the entire local protocol before human-independent work begins. Isolated virtual Case and Gold roles authored 45 cases while provenance fixes `humanIndependent=false` and `officialEsP1QualificationEligible=false`; the translator remained Gold-blind and scoring loaded only after every arm terminated. Across 15 Skills, three repetitions, 135 paired observations, and 270 real local DSH arm executions, Control completed 111/135 (82.22%) and Treatment 132/135 (97.78%). Treatment won 21 pairs and Control won none; both arms recorded zero unsafe executions and zero false commits. All improvement was localized to L0 Runtime (21/42 to 42/42) and failure/adversarial cases (24/45 to 45/45); native reads stayed 15/18 and safe-stop stayed 75/75. Every repetition was 37/45 versus 44/45. Treatment p50/p95 was 27.8/56.1 seconds versus Control 32.9/109.3 seconds under the fixed two-worker configuration. Translation route agreement was 43/45 with zero unsafe Runtime accepts. The sole Treatment residual was one native-read case repeated three times, exposing an upstream L1/Tool-selection gap. See [ES-P1-Wild role-separated simulation results](ES-P1-WILD-SIMULATED-RESULTS.md). This completes the simulation protocol only; the human 45-slot workspace and formal ES-P1-Private gate remain open.
