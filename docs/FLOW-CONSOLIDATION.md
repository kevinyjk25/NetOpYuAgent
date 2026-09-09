# 转译原型收敛与复验 / Translation Prototype Consolidation

## 中文

2026-09-08。本轮目标是清理收敛并提交，**合入前停止推进**，不是再做一次模型质量优化。清理前代码/测试/可提交摘要快照：`c2ebd78`。

### 清理范围

- 将 Guard 与反事实作者的重复一次调用代码，以及作者/行为探针/合同探针的重复回放校验，收敛到 `evaluation/flow_checkpoint.py`。保留显式调用预算、原始响应、精确派生复验、失败不重试和禁止覆盖的语义。
- 当前入口不再通过 `flow_source_duty_pilot → flow_node_evidence_pilot → ...` 导入历史实验以获取环境/指纹。公共指纹显式列出转译依赖，保守覆盖整个 `network_runtime` Python 源码；任何变化生成新版本，旧记录不自动迁移。
- README 只保留项目设计、能力、证据、场景与使用。当前进展/计划不再堆叠相互冲突的“下一步”；原文归入[阶段历史](PROJECT-HISTORY.md)和[纠偏历史](TRANSLATION-CORRECTION-HISTORY.md)。[代码导航](../evaluation/README.md)与[实验索引](FLOW-EXPERIMENTS.md)区分推荐路线、共享基础和历史比较。
- 没有删除历史实验、失败测试或原始响应。旧代码仍有历史引用/冻结回放用途，不能仅因实验失败就当无效代码；本轮删除的是重复实现和当前入口中的陈旧叙述，历史内容可恢复。

没有改动：源样例、Oracle、9B 提示词/Schema/模型选择、参数和 Guard 推导规则、默认 DSH/Runtime、审批/权限/激活门禁及任何 `data/` 基线。本轮新模型调用、设备调用、第三方脚本执行均为 0；正常回归仅使用既有本地隔离测试夹具。

### 等价性检查

| 检查 | 结果 |
|---|---|
| 6 份合同请求 + 1 份反事实请求 | 与原始发送内容逐对象相同 |
| 6 份规范化结果 | 全部字段相同，包括来源和编译结果 |
| 旧严格推导与新必要条件推导 | 各自完整输出保持相同；不是让两种算法相互相同 |
| 6 份行为结果 | 调用顺序、参数、停止结果与原保存对象相同；仍 33/33 |
| 原始首次生成成绩 | 仍 4/6、29/33，不因清理重算 |
| 旧实现版本检查 | 新代码拒绝旧冻结 manifest；不添加绕过开关 |
| 历史 Git 版本回放 | 使用 `c2ebd78` 复验，不更新旧 manifest 或 receipt |

清理前本地 `artifacts/translator-v2` 的 **1,973 个文件、36,753,844 字节**已记录路径及内容汇总指纹：`bc819925c3245c3adf7607e2c7dd8857f0c0ad057765da9e896957e8d0828ca6`。它用于检查清理没有动证据，不是新的模型成绩或签名证明。原始制品仍在本地 ignored 目录，Git 中只有代码与可提交摘要；仅克隆 Git 不等于获得了全部原始模型响应。

清理前定向检查：807 项 Flow 测试通过。清理后公共检查点/作者/构造定向检查：64 项通过；文档与检查点复查 17 项通过。最终全量 **1706 tests + 81 subtests 通过（149.72 秒）**。首轮全量曾因新增说明文件尚未落盘而出现 1 个断链失败，补齐后整套重跑通过；没有修改或放宽测试标准。

本轮 Flow 模块/测试 Ruff 和 `git diff --check` 通过；11 份变更文档的 456 个本地链接检查无断链。全仓 Ruff 仍为 **224 个既有问题**，在 `c2ebd78` 快照与清理后完全同数同类别；未混入无关模块批量改写，也不宣称全仓零技术债。1,973 个原始证据文件汇总指纹清理后完全一致，两个历史文档正文逐字保留。

### 历史回放方法

旧命令应在记录的实现版本运行，不要把旧目录重新 freeze 成新评测。以下从项目根目录建立独立临时快照，不切换正在开发的分支：

```bash
replay_repo=$(git rev-parse --show-toplevel)
replay_dir=$(mktemp -d /tmp/ensuredskill-replay.XXXXXX)
git archive c2ebd78 | tar -x -C "$replay_dir"
cd "$replay_dir"
"$replay_repo/.venv/bin/python" -m evaluation.flow_behavior_probe report \
  "$replay_repo/artifacts/translator-v2/behavior-repair-6-20260908" \
  --output "$replay_dir/behavior-replay.json"
```

需要原本地制品及匹配的 Python 依赖；工具仍检查代码、环境、原文、请求和完整 receipt。该命令不需要模型服务。不要删除旧检查点，或在新实现上用 `--max-new-calls` 补跑旧评测。

本轮实际使用临时 Git 快照成功回放基准/合同报告及反事实作者的完整检查点，新增模型调用为 0。基准报告摘要保持 `sha256:eb28bad3c2e014d8e182a3ca216cc22394610c934a233c03b62e367cc8f049ad`，合同报告摘要保持 `sha256:e4aafb0faf30d00179c4c03718305813c710678bd47778fc4e161e8fbf5f3ea8`。后续合入或清理分支时，应保留历史快照的可达引用及其原始制品，不能只保存可读指标。

本轮通过只证明重构未改变已检查行为，不解决过拟合或完整源语义。下一阶段须等用户合入后，再按[当前计划](TRANSLATION-CORRECTION-PLAN.md)推进。

## English

2026-09-08. This is consolidation and commit preparation, **not another model-quality iteration**. Stop before the next phase until merge. Snapshot `c2ebd78` preserves pre-cleanup implementation, tests and versioned summaries.

Shared `flow_checkpoint.py` replaces duplicated one-attempt recording and exact replay checks. Current authoring no longer imports historical source-duty/node-evidence pilots for infrastructure. Explicit translation dependencies and all Runtime Python sources are conservatively fingerprinted; old manifests are never rewritten to match new code. Current documentation is separated from historical decisions through the [code map](../evaluation/README.md), [experiment index](FLOW-EXPERIMENTS.md), [phase history](PROJECT-HISTORY.md) and [plan history](TRANSLATION-CORRECTION-HISTORY.md).

No failed experiment/test, source, Oracle, prompt/schema, model choice, inference rule, default DSH path, authority gate or baseline was deleted or changed. Historical modules retain replay/reference value. New model/device/third-party-script calls are zero; existing isolated test fixtures remain part of normal regression.

Exact comparison preserves seven original requests, six complete normalization objects, strict/necessary derivations including unknowns, and six full behavior records with 33/33 scenarios. First-pass remains 4/6 and 29/33. New code rejects old fingerprints; snapshot-based replay preserves strict checks. This verifies refactoring, not semantic generalization.

The local raw-evidence inventory contains 1,973 files / 36,753,844 bytes with the aggregate digest above. Raw responses remain in ignored local artifacts; cloning Git supplies code and summaries, not all raw evidence. The replay command above uses a temporary Git archive and the existing environment/artifacts without switching branches or contacting a model. Do not relabel old artifacts as a new frozen study.

Pre-cleanup Flow regression: 807 passed. Post-cleanup targeted regression: 64 passed; documentation/checkpoint recheck: 17 passed. Final full suite: **1706 tests + 81 subtests passed in 149.72 seconds**. An initial broken link while the new document was not yet written was repaired, then the entire suite reran without relaxing assertions. Targeted Flow Ruff and diff checks pass; 456 local links across 11 changed documents resolve. Global Ruff still reports the same 224 pre-existing issues as the snapshot; this is scoped consolidation, not a zero-debt repository claim.

The 1,973 raw artifacts retain their exact inventory digest; archived document bodies are unchanged. Baseline/contract reports and the counterfactual author checkpoint replayed under the temporary `c2ebd78` archive with zero new model calls, retaining the report digests above. Keep a reachable historical snapshot and its raw evidence when merging or pruning branches. Merge gates the [next plan](TRANSLATION-CORRECTION-PLAN.md); these checks do not prove full-source semantics or eliminate overfitting risk.
