# 阶段 2：公开 Skill 与双驱动验证 / Public Skills and dual-drive validation

## 中文

状态：2026-09-10，混合版本真实模型调用前固定本轮验收。保留阶段 2 的原始 10 Skill / 10 仓库 / 9 开发领域选择，不按新版是否能表达筛除。v63、v65 失败及费用不得改写。此阶段是小批已知公开开发验证；不是独立 Gold、未见泛化门禁或生产工程。

### 完成条件

1. 同一个冻结版本完成 10 个 Skill 的有界首次构造、失败定位和逐份源–任务–宿主语义审阅。保留失败、输出截断、资源不足、正确部分边界；纯语言 Skill 不计 L0 成功。原始业务任务、工具合同和源包与 v65 相同；第 10 个沿用原纯语言任务。v4 起精确分离旧编译包装，原字符串和变更摘要留存在 task-provenance 中，因此不再声称 author-input 字节相同；没有改变业务请求或降低本验收条件。
2. 至少三个不同公开 Skill 形成经开发者 AI 审阅的有用严格读取＋开放推理流程，用原 L0 读取引擎、披露的本地模拟宿主和真实 qwen3.5:9b 实际执行；逐份检查候选输出，而不以图完成替代业务正确。若未达到，另建版本修复，不重写当前冻结结果。
3. 验证确定性图约束与 LLM 内容质量是两条证据线。负向机制测试覆盖权限/参数/模型身份、候选伪装事实、独立准入、过期证据、并行必需分支失败、超时迟到等；模型调用不得提供工具或写权限。公开候选也必须运行实际参数/权限错误路径检查。
4. 报告全部 Skill 分母、结构通过、语义接受的严格片段/混合任务、纯 L1、正确与过度停止、已检查参数、未完成职责、构造/执行费用和含模型时延。没有判定依据的指标填未评估，不编造零或准确率。
5. 更新使用入口、架构边界、进展和可审查结果；定向与全量回归、变更代码 Ruff、历史证据和差异检查通过。

该退出条件只表示阶段 2 开发验证闭环完成，**不要求也不声称 10 个任意 Skill 都能完整转为确定性 L0**。正式 ≥3 不相交 cohort / ≥50 Skill / ≥15 仓库 / ≥8 领域 / ≥600 case 门禁不变，规模 DSH/Runtime A/B 不解锁。

### 当前实现边界

- 混合图使用原 L0 严格片段，新增受控推理、独立候选准入、固定依赖与有界并行/all-success 汇合。模型结果始终为候选，不是事实、审批或已执行结果。
- 新的紧凑转译入口区分原 Skill、当前用户任务、宿主合同、模型解释四种来源。原文引用读取是离线 authoring，不是访问业务文件。
- 初版 author surface 只生成原子只读片段与有界推理任务；原 Flow 内的确定性分支仍可执行，但新 author surface 尚不表达跨混合节点条件/循环或动态扩图。缺少严格前置条件时必须停在之前并明确说明，不能将其藏进 LLM。
- v6 增加更窄的只读前段入口：原始业务任务不再由模型转述，直接保留在末端受控 L1 节点；未完成边界同时传入。只读片段、开放任务、纯 L1 与部分结果仍分别统计，不以原文留存证明模型执行准确。
- 下游严格操作使用模型候选参数时，必须经过宿主独立准入且限定目标片段；此入口不自动安装准入策略、不自动启用写事务。旧 Effect 审批/验证/补偿路径不被替代。
- 语法、来源锚点、模型自评都不证明语义正确。此轮审阅是同一开发者 AI，不是独立人工。第三方脚本只作为惰性材料，绝不执行。

## English

Fixed before real mixed-model calls: retain the original ten public development Skills, ten repositories and nine developer domains, including failures and tool-free work. Preserve v63/v65 evidence. Complete bounded first construction and source/task/host review for all ten under a frozen revision. Demonstrate at least three reviewed, useful public read-plus-reason flows through the original L0 engine, disclosed synthetic hosts and real qwen3.5:9b; inspect the actual drafts independently of graph completion. Otherwise repair in a new version, never rewrite this run.

Test authorization, parameters, model identity, candidate/fact separation, independent admission, freshness, required parallel failure and late results. Report all denominators, partial versus whole-task acceptance, pure L1, stops, checked arguments, unresolved duties and construction/execution cost. Unknown metrics remain unassessed. Update docs and run targeted/full regression, changed-code lint and evidence checks.

This closes a small known-public development stage, not an arbitrary-Skill deterministic conversion claim. The separate three-cohort / 50-Skill / 15-repository / 8-domain / 600-case gate stays closed. The compact author surface initially covers primitive read regions and bounded reasoning; the underlying original Flow still supports deterministic branches. Unsupported mixed conditions/loops require explicit boundaries before affected calls. Candidate admission is host-installed and region-bound; no new writer, automatic activation or source script execution is introduced. Developer-AI review is not independent Gold.

Since v4, exact legacy authoring-wrapper separation is retained in task-provenance. Business tasks, sources and contracts stay fixed; author-input bytes no longer match v65. v6's narrower read-prefix adapter retains the original task directly in a final bounded L1 node instead of asking the model to paraphrase it. This does not lower acceptance criteria, remove strict duties, or establish whole-Skill correctness.
