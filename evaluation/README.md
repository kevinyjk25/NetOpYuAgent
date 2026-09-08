# 转译研究代码导航 / Translation Research Code Map

## 中文

本目录包含研究代码和评测，不是默认 DSH 的生产转译器。**当前推荐研究入口只有“合同构造 + 必要条件推导”这一条；旧探针只用于历史诊断/回放。** 全部候选仍须完整源审查和独立宿主授权。合入前不运行新模型批次。

### 当前模块职责

| 模块 | 输入 → 输出 | 不负责什么 |
|---|---|---|
| [flow_contract_authoring](flow_contract_authoring.py) | 原文 + 实际工具合同 → 受约束请求；模型提案 → FlowTree | 不证明参数语义归属，不执行工具 |
| [flow_guard_binding](flow_guard_binding.py) | 显式重复停止规范化、可用布尔事实槽位、条件绑定 | 不猜未提供的前置，不把 OR 改成 AND |
| [flow_guard_counterfactual](flow_guard_counterfactual.py) | 原文/宿主/已有候选 → 正反事实问题和保存的回答 | 不以模型判断授予权限；旧严格 bind 仅保留作比较 |
| [flow_guard_necessity](flow_guard_necessity.py) | 原始正反回答 → 未激活必要条件候选、引用和未知项 | 不证明充分性、路径可行性或完整源语义 |
| [flow_tree](flow_tree.py) | 层级表示 → 现有 L0 流程及可审查来源 | 不新增执行器、不激活合同 |
| [flow_checkpoint](flow_checkpoint.py) | 版本指纹、一次调用、原始响应、严格回放 | 不含案例/Oracle，不自动重试或放宽语义标准 |
| [flow_behavior](flow_behavior.py) / [flow_behavior_probe](flow_behavior_probe.py) / [flow_contract_probe](flow_contract_probe.py) | 私有有限 Oracle → 已有执行器的惰性内存行为比较 | 不把测试成功或正确停止当整 Skill 泛化 |

### 使用入口

从项目根目录执行；`sources.json` 必须是完整 `FlowSources`（原文、真实宿主输入/读合同/效果目标），不是只有一个 Markdown 路径。`proposal.json` 是本轮真实模型输出，`tree.json` 是编译结果中的 `tree` 对象。

```bash
# 离线生成请求和编译保存的提案；不调用模型或设备。
.venv/bin/python -m evaluation.flow_contract_authoring request sources.json --output request.json
.venv/bin/python -m evaluation.flow_contract_authoring compile sources.json --proposal proposal.json --output compilation.json

# 离线推导保存的原始回答。
.venv/bin/python -m evaluation.flow_guard_necessity bind sources.json tree.json --answers answers.json --output necessity.json
```

`flow_guard_necessity author` 是以后经授权运行一次 9B 的入口，不是本轮命令；`--max-new-calls` 默认为 0，已有完整检查点可零调用回放，缺失/失败/版本漂移不会偷偷重跑。此接口只负责候选的布尔前置推导，不等于一个完整 Skill 自动接纳流水线。

结果先看 `status`、`runtimeAuthorityGranted` 和 `fullSourceReview`，再看 `derivations`、`sourceQuotes`、`retainedUncertainty`。生成 Guard 不是授权，保留的 unknown 不能被解释成通过。

### 历史与版本

旧 mapping / canonical / node-evidence / source-duty / compact / common-JSON 探针不再作为推荐入口，路径为兼容旧导入和冻结指纹而保留；不是已证明可删除的无效代码。路线和负结果统一见[实验索引](../docs/FLOW-EXPERIMENTS.md)。

历史检查点必须使用对应 Git 版本；本次之前的修复快照为 `c2ebd78`。新指纹覆盖公共转译依赖与整个 `network_runtime` Python 源码，保守地拒绝代码漂移，不通过旧实验的 import 链继承。不得手改旧 manifest 或将旧数据重新冻结冒充新评测。参见[收敛与回放](../docs/FLOW-CONSOLIDATION.md)。

## English

This directory contains research authoring/evaluation, not the default DSH production translator. The recommended research path is **contract-grounded constructors + necessary-guard synthesis**; historical probes remain diagnostic/replay references. All candidates stay inactive pending complete source review and host authority. No new model batch runs before merge.

`flow_contract_authoring` prepares constrained requests and lowers saved proposals; `flow_guard_binding` owns explicit normalization/slots/binding; `flow_guard_counterfactual` collects two-sided source judgments; `flow_guard_necessity` derives inactive necessary predicates while retaining uncertainty. `flow_tree` reuses existing L0 semantics. `flow_checkpoint` centralizes fingerprints, one-attempt recording and strict offline replay without cases/oracles. Behavior probes use finite private oracles and inert providers, not whole-Skill acceptance.

Run the commands above from the repository root. `sources.json` is a complete `FlowSources` document with actual host contracts, not merely a Markdown filename. `tree.json` is the compiled `tree` object. These commands do not call models or devices. A future explicitly budgeted `flow_guard_necessity author` call handles Boolean guard synthesis only, not an end-to-end semantic admission pipeline.

Inspect status/authority/full-source-review first, then derivations, quotes and retained uncertainty. Unknown feasibility is not success. Historical variants remain at their old paths for compatible replay, indexed [here](../docs/FLOW-EXPERIMENTS.md). Use snapshot `c2ebd78` for pre-cleanup evidence; never bypass code-hash drift or relabel old outputs as fresh evaluation. See [replay guidance](../docs/FLOW-CONSOLIDATION.md).
