# 阶段 1 验收 / Stage 1 Exit Criteria

## 中文

2026-09-09，用户要求持续修复至能够进入阶段 2。阶段 1 完成的是**可进入小批公开 Skill 验证的开发版本**，不是泛化门禁或生产成功率。C3q–C3x 记录仍作为历史诊断，不继续增加大阶段编号。

在本轮真实模型验证开始前固定以下退出条件：

1. 修复已知表示缺口：确定性数组长度判断、条件之后的结果判断、精确路径上的 L1 剩余职责；未来宿主授权不得变成离线构造时的缺事实。
2. 同一个冻结版本和 9B，至少三种不同的已知开发流程，每种连续两次首轮构造得到可审查、可编译、参数来源正确的候选。必须包括嵌套条件、引用/多步与精确交接；不能全部退化为一个读取后停止。已知公开源允许明确的局部边界，但不能把缺少前置条件的后续调用包含进去。
3. 逐份对照原文做开发者 AI 语义审阅，保留全部失败与修订；模型自评、Schema 合法或源码摘要不作为语义证明。样本仍是开发材料，不冒充独立 Gold 或未见泛化集。
4. 受审的开发候选通过原共享读取引擎的合成本地路径检查：正常、分支、空值/错误类型、越权/错误授权；无源脚本或写操作。仅验证生成图的行为，不重启大规模 Runtime A/B。
5. 全量回归、定向检查、Ruff、文档和历史证据校验通过；留下冻结版本、失败边界与阶段 2 抽样计划。

任一条件不满足就继续阶段 1；不得靠删除源限制、修改旧结果、为某 Skill 写特例或扩大拒绝率宣称完成。阶段 2 才开始小批公开 Skill 扩展；正式 ≥3 cohort / ≥50 Skill / ≥15 仓库 / ≥8 领域 / ≥600 case 门禁保持不变。

### 阶段 2 的交接范围（此阶段不执行）

阶段 1 通过后先停在冻结点，保留当前实现和模型配置。下一阶段建议从已知公开开发库预先选定 **10 个 Skill，至少 6 个仓库、3 类业务领域**，不能只按“当前 Schema 能表达”筛选：应包含普通读取、原工具模式选择、引用、多步/条件、带脚本但禁止执行，以及必须部分交接的流程。

每个 Skill 在运行模型前写明原始来源/版本、合理任务、真实或明确披露的本地 adapter 合同、预期业务义务和不支持边界。任务不可附带期待的 L0 节点、参数答案或路由标签。先做源–任务–工具对齐审查，再使用冻结版本逐份首次构造；开发者 AI 审阅不冒充独立人工。

同时报告 Skill 分母、首次结构通过、语义可用、有效只读片段覆盖、正确/过度停止、参数精确匹配、未决职责丢失和成本。结构通过不能取代语义通过；部分成功不能计整 Skill 成功。保留每个失败和不适配 Skill，诊断通用问题后另建版本及新验证轮次。此小批仍是开发验证，不启动正式未知 cohort 或 DSH/Runtime 大评测。

## English

Fixed before this stage's new model validation: Stage 1 means a development version ready for small public-Skill validation, not generalization or production success. Repair typed array-length decisions, post-read decisions, path-bound remaining duties and authoring-versus-execution separation. Under one frozen implementation and 9B, obtain two consecutive first-construction candidates for each of at least three different known development procedures, including nested decisions, multi-step/reference behavior and exact handoffs. Not all may degrade to a single read. Public-source partial regions must stop before unsupported prerequisites.

Review every candidate against its original source using explicitly developer-AI review, not self-certification or independent Gold. Preserve failures and revisions. Reviewed generated graphs must pass local synthetic branch/invalid-data/access-denial checks through the original shared engine, with no source-script or effect execution. Full regression, lint, documentation and historical evidence checks must pass. Only then enter Stage 2; the separate cross-cohort generalization/large-Runtime gate remains closed.

Stage 2 handoff, not executed here: preselect ten known public development Skills from at least six repositories and three domains, including references, conditional/multi-step reads, host modes, inert scripts and unsupported partial handoffs. Do not select only representable procedures. Freeze source versions, aligned tasks, disclosed original/local-adapter contracts and developer review expectations before model calls; do not include expected L0 nodes or answers in author input. Report Skill-level denominators, first structural/semantic outcomes, useful-region coverage, over-stops, exact parameters, lost duties and cost, keeping partial versus whole-Skill acceptance distinct. Retain every failure; version generic fixes rather than rewriting results. This is still development validation, not unseen proof or large Runtime A/B.
