# 转译研究代码导航 / Translation Research Code Map

## 中文

阶段 1 基线：Git `43a2b76` / v62。阶段 2 双驱动 v7 小批开发验证已完成，尚未提交，模型仍为 9B。先看[最终结果与实际使用](../docs/STAGE-2-HYBRID-RESULTS.md)；[首批负结果](../docs/STAGE-2-PUBLIC-TRANSFER.md)和[v65 严格表示修复](../docs/STAGE-2-REPRESENTATION-REPAIR.md)保留。不改变默认 DSH 路由，不将混合/局部结果称为完整 L0 转换。

### 当前主路径

| 环节 | 文件 | 权威边界 |
|---|---|---|
| 完整源包 | [translation_intake](translation_intake.py)、[task_alignment](task_alignment.py) | 保留原文/引用/脚本惰性文本，冻结具体任务及原始/披露的本地宿主合同 |
| 冻结、窗口与检查点 | [source_ledger](source_ledger.py) | 显式选择 plan_first + semantic-plan；原始回复、预算、指纹和失败保留 |
| 语义规划 | [source_plan](source_plan.py)、[source_closed_program](source_closed_program.py) | 源义务、闭合控制树与观察命名空间；结构不证明语义 |
| 来源与参数 | [source_inline_program](source_inline_program.py)、[source_argument_slots](source_argument_slots.py) | 原文位置、完整 Schema 参数槽及数据来源；类型兼容不等于业务正确 |
| 惰性表示与编译 | [source_program](source_program.py)、[structured_flow_tree](structured_flow_tree.py) | 白名单解析，无 eval/exec；复用原 Runtime，不执行源脚本 |
| 证据与路径检查 | [stage1_validation](stage1_validation.py)、[stage1_evidence](stage1_evidence.py) | 开发者审阅和合成执行单列，不自动授予权限或称整 Skill 成功 |
| 双驱动前段构造 | [hybrid_authoring](hybrid_authoring.py)、[hybrid_prefix](hybrid_prefix.py)、[hybrid_parameters](hybrid_parameters.py) | 原始任务保留为受控 L1；只读前段编译到原 L0，边界注释不授语义权威 |
| 双驱动冻结与实际执行 | [hybrid_transfer](hybrid_transfer.py)、[hybrid_live_demo](hybrid_live_demo.py)、[hybrid_reasoning_transport](hybrid_reasoning_transport.py) | 新版本/检查点不可覆盖；真实 9B 与合成只读宿主，源脚本不执行 |
| 分开计量与审阅 | [hybrid_evidence](hybrid_evidence.py)、[hybrid_execution_evidence](hybrid_execution_evidence.py)、[hybrid_public_checks](hybrid_public_checks.py) | 结构、局部/完整结果、真实模型成本与模型替身机制测试分列 |

### 最小使用

输入文件必须包含原始 bundle、task、taskOrigin、inputSchema、catalog 和 reads，不能放审阅答案。详见[语义前端](../docs/SEMANTIC-PLAN.md)。

```bash
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN_DIR --inputs inputs.json --profile plan_first --semantic-plan
.venv/bin/python -m evaluation.source_ledger run NEW_RUN_DIR --max-new-calls 1 --report-dir NEW_REPORT_DIR
# 零新调用回放；运行目录和报告目录不能覆盖。
.venv/bin/python -m evaluation.source_ledger run NEW_RUN_DIR --max-new-calls 0
```

max-new-calls 是本次允许的调用数，不是许可自动重试失败。一个流程可能需要源窗口请求、一次规划和逐读取填参。已有未回执请求保持未知；不能删检查点再跑。完整首次构造通过不代表可以跳过源文审核或激活。

### 生命周期分类

- **当前可选路径**：严格表达使用上述 source_ledger plan_first/semantic-plan；混合任务使用 hybrid 构造/审阅/本地执行入口。Runtime 权限边界不变。
- **依赖/兼容保留**：source_catalog、source_modes、source_program_lines、source_program_anchors、source_duty_accounting 等仍被编译、测试或历史回放引用，不因名称旧而删除。
- **历史研究路径**：flow_contract_authoring、flow_read_region、flow_condition_expression、flow_semantic_probe 等见[历史导航](HISTORY-README-20260910.md)及[实验索引](../docs/FLOW-EXPERIMENTS.md)。保留 CLI/API 和原始结果，不自动与当前结果混算。
- **阶段 2 双驱动路径**：公开验证已完成最小开发闭环，不代表泛化通过；`hybrid_live_demo` 仅对明确受审候选执行原严格读取＋真实 9B。`hybrid_public_checks` 是合成接线检查，不是实际 LLM 成功率。见[双驱动边界](../docs/GOVERNED-HYBRID-FLOWS.md)。

清理本身未改执行引擎；v65 功能修订在同一个数据绑定/流程引擎中扩充双侧比较检查，不另建执行器。所有版本单独冻结，不改写 v62/v63 记录。语法可表达不等于语义可信。

## English

Stage 1 baseline: 43a2b76/v62. Stage 2's mixed v7 development loop is complete and uncommitted; see [results and actual use](../docs/STAGE-2-HYBRID-RESULTS.md). Preserve the [original negative batch](../docs/STAGE-2-PUBLIC-TRANSFER.md) and [v65 representation repair](../docs/STAGE-2-REPRESENTATION-REPAIR.md). source_ledger remains the opt-in strict authoring path; hybrid_authoring/prefix/parameters retain the original L1 task after a grounded read prefix. hybrid_transfer freezes all source/task/host inputs; hybrid_live_demo and role-separated reasoning transport invoke actual local 9B only after explicit review. Evidence collectors separate structural, partial/full semantic, actual-model and synthetic-mechanism results. Zero-call replay is distinct from retry; inputs never contain reviewer answers.

Keep dependent helpers and historical authoring paths for imports, regression and original-version replay. Their commands and results remain in the [historical navigation](HISTORY-README-20260910.md); do not mix them into current scores. New hybrid_authoring/parameters/transfer/live_demo modules implement an opt-in, reviewed read/reason path; synthetic public checks are not live LLM accuracy. See the [hybrid boundary](../docs/GOVERNED-HYBRID-FLOWS.md). Functional revisions retain separate freezes and negative evidence; Stage 1 development evidence, Stage 2 transfer and formal generalization remain distinct.
