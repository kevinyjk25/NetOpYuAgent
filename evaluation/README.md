# 评测代码导航 / Evaluation Code Map

## 中文

当前研究协议唯一入口是[考核重置与有限收敛设计](../docs/EVALUATION-RESET-20260916.md)。**R0 最终状态和逐项证据以 [R0 验收报告](../docs/R0-COMPLETION.md)为准；未进入 R1，没有真实模型 run 命令。** 旧研究代码仍可用于追溯和回归，但不再是当前试跑计划。清理与测试分类见[清理记录](../docs/CLEANUP-20260916.md)。

### 当前 R0 测量链

| 模块 | 职责与边界 |
|---|---|
| [bounded_pilot](bounded_pilot.py) | check、prepare、inspect、score；固定协议、预算和判据，不执行真实模型 |
| [bounded_scoring](bounded_scoring.py) / [bounded_probe](bounded_probe.py) | 分开任务／转译／安全／成本，合成正反例检查不能冒充 Agent 成绩 |
| [bounded_budget](bounded_budget.py) / [bounded_execution](bounded_execution.py) | 持久预算总账、未知调用留存；现阶段回调只做 fixture 计量 |
| [bounded_transport](bounded_transport.py) / [chat_codec](chat_codec.py) | 角色绑定的 scripted-only HTTP broker；协议编码不充当 token／事实可信来源 |
| [bounded_provider](bounded_provider.py) | 每 arm 独立 SQLite 模拟 Provider 与真实读回执；不获得产品写权限 |
| [bounded_dsh_probe](bounded_dsh_probe.py) / [bounded_dsh_tools](bounded_dsh_tools.mjs) | 实际 DSH 的原生、编译后 Runtime 读取和拒绝后 fallback 接线；无真实 LLM |
| [dsh_support](dsh_support.py) / [local_read_fixture](local_read_fixture.py) | 独立 DSH 配置／路径助手和只读 fixture，不依赖旧 ledger/reviewer 实验链 |
| [bounded_preflight](bounded_preflight.py) / [离线 helpers](preflight/renderer.md) | 用户已授权的共享 renderer、vocab-only tokenizer 与不可变 prepared request；仅离线预备，不开放 generation 或 live dispatch |
| [bounded_runner](bounded_runner.py) / [官方解析器 helper](runner_codec/README.md) | 精确 token 数组请求序列化与固定上游 Qwen35 解析；不启动真实 runner |
| [bounded_prepared_transport](bounded_prepared_transport.py) | 准备→预留→实际 HTTP 请求→原始解析绑定；仅可连接本模块创建的零推理替身，不能传入外部 URL |
| [bounded_controller](bounded_controller.py) | 固定 12 对机械 DSH 回合；B 在工具回合内编译最多两个只读前段，余下读取显式 fallback；不是整 Skill 编译 |
| [bounded_freeze](bounded_freeze.py) / [bounded_acceptance](bounded_acceptance.py) | 冻结声明执行依赖及独立审阅材料；实际发送前后校验，固定工程总账，不可换目录刷过 |
| [bounded_reacceptance](bounded_reacceptance.py) | 用户显式批准的一次工程重验；同一工程账本关联原失败与固定子记录，原结果不重置、不增加研究预算；不是自动重试入口 |
| [bounded_cases](bounded_cases.py) / [bounded_network_policy](bounded_network_policy.py) | 依赖本地原始来源档案的只读草稿构造器；草稿不自动获得已审阅身份 |
| [bounded_material](bounded_material.py) / [独立材料包](../data/bounded-pilot/r0-development-20260917/README.md) | 完整来源与双份标注、逐案裁决、精确摘要绑定和一次显式最终审阅；离开本机原始档案也可校验 |

零真实模型检查（输出目录必须全新，不能覆盖已有证据）：

```bash
.venv/bin/python -m evaluation.bounded_pilot --help
.venv/bin/python -m evaluation.bounded_probe /tmp/ensuredskill-r0-new-output
.venv/bin/python -m evaluation.bounded_dsh_probe --help
```

历史三路径结果见[接线摘要](../docs/benchmarks/bounded-pilot-r0-integration-summary.json)，当时输入/输出 token 都是 fixture。新 prepared 路径的输入由真实离线 tokenizer 计数，输出仍是明确的替身 fixture；generation 禁止，不替换现用服务。完整控制器验收、6 Skill／12 Task 材料封存与剩余限制见 [R0 验收报告](../docs/R0-COMPLETION.md)。这些不是 9B 语义准确率、真实性能或 36 项 Runtime 门槛。旧[Token 预检](../docs/R0-TOKEN-PREFLIGHT.md)保留原始阶段范围。

### 产品实现与研究代码分开

产品编译与执行适配位于 [skill_authoring](../skill_authoring/)，Runtime 位于 [network_runtime/l0](../network_runtime/l0/)，宿主位于 [dsh_adapter](../dsh_adapter/)。`evaluation.hybrid_*` 中的兼容入口不构成另一套产品编译器。当前可选会话及双驱动边界见[受控会话](../docs/GOVERNED-SESSION.md)、[双驱动流程](../docs/GOVERNED-HYBRID-FLOWS.md)。混合图不等于整 Skill 都已确定性转译；模型候选不能授予权限。

### 测试选择

```bash
.venv/bin/python -m pytest -q                         # current，默认回归
.venv/bin/python -m pytest -q --test-suite=historical # 历史探索性回归
.venv/bin/python -m pytest -q --test-suite=all        # 提交／发布完整回归
```

分类由 [suite_policy.json](../tests/suite_policy.json) 的显式文件名单控制，不按 `flow_*` 等前缀批量排除。Runtime 安全、现用编译器、正式门禁、权限、未知效果与恢复检查仍属 current；新增测试默认 current。CI／retirement 使用 all。测试数变化只说明分类或代码变化，不说明语义能力提高。

### 历史与保留依赖

- `source_ledger`、`source_plan`、`source_program`、flow/duty/reviewer 试验属于此前研究路径；不因已有 CLI 自动重启模型调用。
- 部分旧模块仍被兼容入口、历史重放或当前安全检查引用，不能按文件年龄删除。原 API 保留；新 R0 不再导入旧 semantic-closure、source-ledger、DSH-shadow 或 reviewer 链来取得通用助手。
- `task_delivery_ablation` 单快照双臂诊断及其专用测试已退役。旧结果和冻结源码保留，可从清理前 Git 或本地备份恢复；不再提供该旧 CLI。
- 历史阶段 1/2 的正负结果、Oracle、源码摘要和原始响应均保留。开发中已暴露 Skill 不会因归档变成 unseen。

历史入口：[旧代码导航](HISTORY-README-20260910.md)、[实验索引](../docs/FLOW-EXPERIMENTS.md)、[阶段 1 结果](../docs/STAGE-1-RESULTS.md)、[阶段 2 结果](../docs/STAGE-2-HYBRID-RESULTS.md)。其中“当前／下一步”是当时快照，不替代新协议。

## English

The [bounded evaluation reset](../docs/EVALUATION-RESET-20260916.md) is the current research protocol. **See [R0 acceptance](../docs/R0-COMPLETION.md) for its final status and checklist; R1 has not started and no real-model run command exists.** Historical experiments are retained for traceability and regression, not as the active run plan. See the [cleanup record](../docs/CLEANUP-20260916.md).

The active measurement chain is `bounded_pilot/scoring/budget/execution`, the scripted-only `bounded_transport`, isolated SQLite `bounded_provider`, and the installed-DSH `bounded_dsh_probe`. The user-authorized `bounded_preflight` adds an offline shared renderer, vocab-only tokenizer and immutable prepared request, without enabling generation or live dispatch. Shared DSH helpers, protocol codecs and read fixtures now live independently in `dsh_support`, `chat_codec` and `local_read_fixture`. They do not require old semantic-closure, source-ledger, DSH-shadow or reviewer experiment chains. Historical callers retain compatible exports.

The new `bounded_runner/prepared_transport` binds actual token-array HTTP payloads and official parsing to reservations, using an exclusively local no-inference receiver. `bounded_controller` runs twelve mechanical pairs, with up to two in-session read-prefix compilations per B arm and explicit fallback, not whole-Skill translation. `bounded_freeze/acceptance` pins declared dependencies and source/reference materials before and after the once-only engineering batch. The explicitly authorized `bounded_reacceptance` links one fixed child to the preserved failed engineering study in the same ledger; it is not an automatic retry or a research-budget reset. Input tokens are real offline counts; output usage is synthetic. These are not 9B semantic/performance evidence or the 36-probe gate. [R0 acceptance](../docs/R0-COMPLETION.md) states the full material and control boundaries. [Original integration evidence](../docs/benchmarks/bounded-pilot-r0-integration-summary.json) retains its original fingerprint. Standalone accepted material must not rely on ignored original-acquisition archives.

Product authoring lives in `skill_authoring`, execution in `network_runtime/l0`, and host integration in `dsh_adapter`. Compatibility exports under evaluation do not create another compiler. A governed mixed workflow is not a fully deterministic Skill; model proposals never grant authority.

Pytest defaults to `current`; `--test-suite=historical` runs the explicit research subset and `--test-suite=all` runs both. CI and retirement use all. The exact [manifest](../tests/suite_policy.json) preserves core/formal-gate checks in current; future tests default to current. Counts are engineering inventory, not semantic evidence.

Retain historical API dependencies, source snapshots, failures, labels and reports. The no-consumer `task_delivery_ablation` CLI and its dedicated tests are retired and recoverable from the pre-cleanup commit/local backup. Exposed development Skills remain exposed after archiving. [Historical code map](HISTORY-README-20260910.md) and [experiment index](../docs/FLOW-EXPERIMENTS.md) remain accessible; their old next-step instructions do not authorize another model run.
