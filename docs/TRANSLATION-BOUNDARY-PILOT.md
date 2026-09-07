# 异质 Skill 边界小批 / Heterogeneous Skill boundary pilot

## 中文

2026-09-07，C 阶段的选样和环境盘点完成，**完整转译小批尚未完成**。本轮只读取封存文本，未运行 9B、Runtime、第三方脚本或 CLI。它不是新的转译成绩。

### 规模与可检查结果

从现有 100-Skill 已知开发库选取 **12 Skill / 11 仓库**，覆盖无参、可选参数、引用、条件、多步骤、脚本依赖、审批、自由 CLI、读写混合、开放推理 **10 类结构**。源索引包含 8 个领域标签；这是沿用的分类数量，未独立重审领域分类，不能据此宣称八领域泛化。

- [选样与解释](../data/translation_boundary_pilot.json)：结构标签、所需环境和单读链路缺口。
- [机器报告](benchmarks/translation-boundary-pilot.json)：每项固定仓库 commit、包摘要、原文引句、字符位置和文件摘要。
- [生成器](../evaluation/translation_boundary_pilot.py)：校验封存库和精确引文，拒绝重复项、来源变化和非主要转译样本；不会授予执行权。

标注者是可见项目上下文的当前助手，不是独立人员。代码核对引文存在性和摘要，**不自动证明解释正确**。这些样本是刻意覆盖结构的开发选样，不是市场随机样本或 unseen cohort。

| Skill | 需要保留的流程 | 本次需要补齐/核实的宿主环境 |
|---|---|---|
| doc-ingest-analyze | 无参准备→上传→索引→记录；可含写入 | MCP 输入/输出、HTTP 上传效果、vault/absence 接口；无参不代表只读 |
| customer-crm-lookup | CRM 查询、M365 交叉引用、可选同步 | CRM/M365/vault 工具合同，分开读取与同步权限 |
| elasticsearch-query | 服务解析、别名、受限查询 | 查询真实签名、输出 Schema、数据范围 |
| object-storage | put/sign_get/delete、授权、依赖 Skill | 具体 Provider、租户授权、auth/persistence 依赖；授权不等于人工审批 |
| prod-db-readonly | SELECT 查询与账户初始化脚本分离 | SQL 方言/语法约束、只读凭据、输出与数据权限；不运行建用户脚本 |
| network-architecture-audit | 网络捕获、脱敏、推理、认证捕获审批 | 浏览器/HAR 环境、脱敏规则、审批事实 |
| project-new-task | 可选输入、tracker 分支、分支/worktree 写入 | MCP/CLI 工具 Schema、项目状态、写入权限；连接配置不是工具合同 |
| jq | 自由表达式、flags、可变输出 | 宿主 gojq 语义、输入输出约束；不执行组合 shell 示例 |
| release | 多步检查、引用、批准后发布 | 仓库/构建工具、审批系统、发布效果与验证 |
| data-quality-investigation | 假设—证据循环、条件工具调用、修改审批 | 数据源与权限、成本/修改审批；保留开放推理 |
| find-docs | Context7 CLI 解析 ID、限次查询 | CLI 参数/返回合同；原文明确不用 MCP，不能擅自替换为 MCP |
| fivem-onesync | 设计、实现、多客户端验证 | 项目代码、运行环境、依赖文档和测试设施 |

### 为什么没有立即跑 12 次 9B

本次输入只有 Skill 包，没有提交与各 Skill 对应的完整宿主工具合同及授权绑定。某些原文有 API 例子/方法名，但不等于已提供完整的输入、输出、效果和可信实现。既有“包 Runtime-ready”只表示包格式门禁，不代表这些运行环境已就绪。

因此 12 项的 `translationOutcome=not_run`，不是 0% 准确率，也不是宣判全部不可转译。应分别保留：

1. **环境缺失**：缺少工具/依赖/输出/权限事实；不由模型补造。
2. **范围不支持**：即使补齐环境，单读编译路径仍不能表示多步骤、条件、脚本或开放推理；不能把单个子操作算作整个 Skill。
3. **转译错误**：只有给定足够源信息、明确受支持范围并真正尝试后，才能评判参数/步骤/效果偏移。

### 下一步的设计收口

推荐先做**宿主工具上下文绑定 + 混合流程提案**，而不是为这 12 个 Skill 分别开发专用适配器，也不自动执行任意 CLI：

- 输入分为不可改写的 Skill 包与宿主显式提供的工具环境；API/MCP/受约束 CLI 共享合同模型，但保留真实调用形式。端点和示例不自动成为工具合同。
- 9B 提出整个流程的步骤、条件、依赖和未知项；每步区分 L0 候选、L1 推理、缺环境或不支持。输出先是无执行权提案，不新增并行 Runtime。
- 复用现有只读/效果合同编译与审查。可编译子步骤不等于全 Skill 可执行；写入继续安全停止，不能用 L1 fallback 绕过审批。
- 先以确定性回归验证上下文/范围报告，再做少量 9B 正向开发实验，分别记录首次生成与辅助修订、全 Skill/步骤覆盖及成本。真实本地执行仍仅限明确宿主授权的能力。

此范围已经用户确认，并补充要求：整阶段必须包括最小业务流程 Runtime 闭环，不能只有提案。当前已实现[只读条件流程 C1](L0-BUSINESS-FLOW.md)；写前分支证据绑定/重校验和整流程正向转译仍待完成。生产凭据系统、通用 shell 沙箱和规模化 Runtime 性能评测不在本阶段。

### 复现

在包含原封存开发库的项目根目录运行，输出路径必须不存在：

```sh
.venv/bin/python -m evaluation.translation_boundary_pilot \
  artifacts/translator-v2/development-corpus-100 \
  data/translation_boundary_pilot.json \
  --output /tmp/translation-boundary-pilot-review.json
```

仓库中的 JSON 报告可直接审阅；仅克隆源码、不具有该摘要对应封存库时，不能重算源绑定检查。不要换一个新库而沿用旧摘要。

## English

September 7, 2026: phase C intake is complete, **not the translation pilot**. Twelve known-development Skills from eleven repositories cover ten annotated structures. Eight domains are inherited index labels, not independently validated taxonomy or generalization evidence. Selection is purposive, not random or unseen.

The [selection](../data/translation_boundary_pilot.json) records environment requirements and single-read limitations. The [report](benchmarks/translation-boundary-pilot.json) binds repository commits, package digests and exact source quotes/offsets. The [generator](../evaluation/translation_boundary_pilot.py) validates corpus integrity and citation existence, not semantic truth. The reviewer is the current visible-context assistant, not an independent human.

No tool-environment bundle was supplied to this intake. Method examples, endpoint configuration and package-format readiness do not establish complete tool contracts or authorized execution. All twelve translation outcomes are `not_run`, not zero accuracy or intrinsic untranslatability. There were zero model calls, Runtime runs and third-party script/CLI executions.

Keep three diagnoses separate: missing environment facts, unsupported flow semantics, and actual translation errors after an adequately specified attempt. Upload/index/sync, SQL provisioning, authenticated capture, branch creation and publishing must not be silently reduced to one read. Authorization is not necessarily human approval. Open-ended analysis remains reasoning, not deterministic code simply because a Skill describes it.

The recommended next scope is **host tool-context binding plus mixed-flow proposals**: immutable Skill sources and explicit host contracts; 9B-proposed steps, conditions and dependencies; per-step L0-candidate/L1-reasoning/missing/unsupported status; existing compilers and review gates retained. No new executor or arbitrary shell permission is implied. Typed APIs, MCP and constrained CLI can share contract semantics without replacing a source-mandated transport. Partial compilation is not whole-Skill success, and unsupported writes cannot fall back to native agent execution.

The user approved this scope with minimal business-flow execution required, not proposals alone. [C1 read-flow wiring](L0-BUSINESS-FLOW.md) is now implemented; write-time branch binding/revalidation and whole-flow forward translation remain open. Production identity, a general shell sandbox and large Runtime benchmarks remain out of scope. First-pass versus assisted outcomes and whole-Skill versus step coverage remain separate. Reproduction requires the original sealed corpus and a new output path; source-only clones can inspect the report but cannot replay its source checks without that corpus.
