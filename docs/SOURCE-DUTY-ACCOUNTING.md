# 源义务对账与 L1 交接 / Source-Duty Accounting and L1 Handoffs

## 中文

更新：2026-09-09，C3x。C3q–C3w 已提交 dev `cd278c3`，未推送。本阶段新增可选的**逐项义务、前置分支和交接位置检查**；完成的是可定位的诊断机制，不是公开 Skill 的高准确率转译。默认 DSH、Runtime 和 `direct` 路径不变。

### 为什么需要这一步

[上一轮](PLAN-FIRST-AUTHORING.md)的真实 9B 候选能生成条件读取流程，但 `remaining.duties=[]`。写一句“需要 L1”没有说明剩下的判断、适用条件、禁止行为，以及从哪里接手。

新入口复用既有源义务提取和原编译器，不新增执行器：

```text
原始源文 → 9B 独立提取义务（不提供当前任务或预写计划）
                        ↓ 保留原文、分类、提取范围；可能遗漏
源文 + 当前任务 + 宿主 → 9B 规划 → 冻结计划
                        ↓
9B 逐项对账：已表示 / 交回 L1 / 未来宿主门禁 / 范围外 / 未决
                        ↓
代码核对 ID、位置、前置分支与每条路径、交接职责
                        ├─ 不一致：原样保存并停止，不填参
                        └─ 结构一致：分槽填参 → 原编译器 → 未激活候选
```

### 检查的内容和明确边界

| 对象 | 自动检查 | 仍不能由此证明 |
|---|---|---|
| 义务清单 | 每个已提取 ID 恰好有一项处置；源包、任务、catalog、计划摘要一致 | 清单涵盖了所有源义务；分类没有误解 |
| 流程位置 | 只能指向冻结计划中的真实读取、判断或终止；控制流义务不能只拿读取当判断 | 引文真正蕴含该动作，参数的业务角色正确 |
| 前置条件 | 每条抵达声明依赖读取的路径，都必须先通过声明的指定分支 | 条件含义、极性正确，所有依赖关系都已提取 |
| L1 交接 | 每个 `needs_l1/unsupported` 终止有准确源义务、条件说明和路径 | L1 后续一定能完成；交接可绕过写入授权 |
| 未来宿主门禁 | 仅执行阶段授权职责可声明对应既有身份/范围/资源门禁 | 已获得权限，门禁等价于整段 wrapper、证据时效或领域检查 |

`requiredBefore` 本身已经提供准确的分支位置；不再要求在 `nodePointers` 重复声明。条件之后两条分支若重新汇合并继续读取，仅“判断在前”还不够，依然拒绝。无法表示的必要条件不能交回 L1 后却继续包含依赖它的调用。

对账最多枚举 256 条语法路径，超过则显式停止，不截断；不做求解器推理来抹除“看似不可能”的分支。提取沿用原单窗口、最多 12 项的机制，**长源文未审阅页、复合句遗漏和重复提取仍可能存在**。这不是完整源覆盖算法。

所有结果保持 `sourceCoverageProven=false`、`semanticEntailmentProven=false`、`wholeSkillTranslationProven=false`、`runtimeAuthorityGranted=false`。不能把“7/7 行对账”写成“100% 语义准确”。本功能仅研究路径启用，未经语义审阅不会成为自动激活凭证。

### 本轮真实 9B 结果：不是成功率提升

[原始报告](../artifacts/translator-v2/source-accounting-20260909/report-wiring-v22/report.json)保留三次 `qwen3.5:9b` 调用：**169.421 秒累计请求耗时，14,224 输入 / 2,433 输出 token**。原六字段输入与上一轮合成接线完全一致，但新增提取/对账阶段，资源与中间上下文不同，不能视为同资源准确率对照。

- 提取 7 行、5 种不同的义务文字；同一“授权/范围/时效”复合句被重复三次。不是 7 个 Skill。
- 生成 2 个读取、一个 `adminUp == false` 分支；仍无明确空列表分支和计数结果判断。
- 对账暴露“用读取代替空列表检查”、未转换结果判断及没有具体职责的 L1 交接；**在填参前停止：0 参数槽、0 编译候选、0 Provider/脚本调用**。
- 原 v22 有 7 条结构问题；开发者复核发现一条来自“已声明的前置 guard 必须重复写入节点清单”的冗余要求。v23 去掉重复要求后，对**同一记录回复**另做零模型调用诊断，剩 6 条；并准确定位计数结果义务选错了前置分支。没有修改旧报告或把这次重分析算成新模型成功。
- 原版报告本地和源码隔离回放逐字节一致。公开 Netdata 未重跑、未执行，新增公开 Skill 为 0，不扩大 Runtime A/B。

更深层问题仍在：单一分类和整句义务把“条件 + 动作 + 输出限制”混在一起；模型又把宿主未来授权误解成当前离线构造缺少的条件。这不能靠多加测试、删除问题行或让同一个模型自评通过来解决。

### 使用与定位

```bash
.venv/bin/python -m evaluation.source_ledger freeze NEW_RUN \
  --inputs INPUT.json --profile plan_first --account-duties
.venv/bin/python -m evaluation.source_ledger run NEW_RUN \
  --max-new-calls 6 --report-dir NEW_REPORT
# 再次运行只回放已记录结果；失败不会自动重试：
.venv/bin/python -m evaluation.source_ledger run NEW_RUN --max-new-calls 0
```

这是离线研究 CLI，不是 3080/3081 页面默认功能。输入仍是原始 Skill 源包、任务、原宿主 Schema 和读取合同，不预装答案。沿用非推理 9B、49,152 上下文和 4,096 输出预算；新目录、源码/模型指纹及断点保护不变。

按顺序查看：

1. `obligation-review.json`：原文、提取文字、类别和阶段。先确认是否遗漏、重复或混合多个职责。
2. `prepared-plan.json`：实际计划、冻结读取、条件和规范化记录。
3. `source-accounting.json`：`issues[].dutyId` 定位义务，`planPointer` 定位节点；同时保留完整模型解释、分支和源位置。
4. 成功编译时的 `handoffs.json`：每个交接终止的准确路径、职责与源文，绑定 Tree 和对账摘要。`remaining.json` 保留旧计划原始 remaining，并另存对账后的未完成职责；编译时重新核对，不能编辑报告来放行。

例如，本轮 `o001 → /steps/0` 定位“空列表检查”被错误映射到读取；`o003 → /steps/1/when_equal/0` 定位计数结果职责与前置分支混淆。修改点首先是义务提取/映射，不是绕开 Runtime 的拒绝。

验收：[摘要、源审阅与 v23 独立重分析](../artifacts/translator-v2/source-accounting-20260909/evidence-summary/report.json)。新增 **30 项机械测试**；202 项定向通过（13.38 秒），全量 **2322 passed + 81 subtests passed（183.33 秒）**，Ruff、文档链接与空白检查通过。v22 的 2320 项全量结果（225.14 秒）也保留。上一阶段 127 份绑定证据不变。该摘要绑定代码、源码快照、真实失败、隔离回放和测试日志；测试数与诊断数均不是语义准确率。

实现：[source_duty_accounting.py](../evaluation/source_duty_accounting.py)、[集成入口](../evaluation/source_ledger.py)、[机械回归](../tests/test_source_duty_accounting.py)。下一步应先拆开复合义务中的条件、动作、输出和授权角色，再检验模型能否正确映射/交接；不要把已知样本改写成更容易通过的 Skill。跨 Skill 泛化与大规模 Runtime 门禁保持不变。C3x 新改动未提交；开头的 `cd278c3` 只包含先前 C3q–C3w 工作。

## English

C3q–C3w were locally committed on dev as `cd278c3`, without a push. C3x adds opt-in source-duty accounting and precise L1 boundaries; it does **not** demonstrate public-Skill translation accuracy. Default DSH/direct routing and Runtime execution are unchanged.

`--account-duties` reuses the existing, task-isolated source inspector. The model then plans a frozen read region, and a separate response accounts for every extracted duty before argument binding. Exact IDs, original source spans, context/plan digests, node kinds and bounded syntactic paths are checked. A required branch must precede every path reaching its declared dependent read. Merely reading a value or checking a predicate whose arms rejoin is insufficient. Each handoff terminal needs source-bound remaining responsibilities. Future host gates are proposed authorization correspondences only, never satisfied permissions or domain-guard equivalence.

The inventory is partial: the existing single-window/twelve-row extraction can omit duties, duplicate compound sentences and misclassify conditions. All-row accounting is not whole-source coverage. Path enumeration stops explicitly beyond 256 traces; no paths are silently discarded. Semantic entailment, guard meaning/polarity, dependency completeness, whole-Skill acceptance and Runtime authority remain unproven.

The linked real 9B run makes **three calls, 169.421 seconds total, 14,224 input/2,433 output tokens** on the unchanged six-field synthetic wiring input. Seven rows contain five distinct requirement strings. A two-read conditional plan is produced, but the audit confuses an observation with an empty-list decision, omits result-decision handoff duties and mixes future authorization with offline authoring. It stops before arguments: zero compiled candidates, provider calls or script execution. These are not seven Skills, new public sources or a same-resource accuracy comparison.

v22 records seven findings. Review identifies one redundant checker requirement: an explicitly named dependency guard need not be duplicated in another node list. v23 removes that redundancy while preserving exact branch/path checks. A separate zero-model-call reanalysis of the recorded reply yields six findings and a more precise wrong-branch diagnostic. Historical checkpoints/reports remain unchanged and replay byte-identically both locally and under their original isolated source snapshot. No public Netdata attempt or broad Runtime evaluation is added.

Use the CLI above with fresh directories. Inspect the original obligation review, frozen plan and source-accounting issues by duty ID and plan pointer. Successful inactive compilation retains precise `handoffs.json` and separately preserves the original remaining list. The checker is recomputed before attachment; editing a report cannot authorize execution. None of this is a new default UI feature or an independent semantic oracle.

The linked evidence summary binds code/snapshots, all three real failures, original-version replays, review and logs. **30 new mechanical tests; 202 targeted passes (13.38 seconds), 2322 full-suite passes plus 81 subtests (183.33 seconds)**, Ruff, documentation links and whitespace checks pass. The earlier v22 full run (2320 tests, 225.14 seconds) is retained. All 127 prior bound artifacts are unchanged.

Next address compound-duty decomposition and condition/action/output/authorization roles, then test correct mapping and handoff without weakening the source or acceptance gates. Mechanical tests and conservative rejection must not be reported as improved semantic translation success. New C3x changes remain uncommitted; `cd278c3` contains only the preceding C3q–C3w work.
