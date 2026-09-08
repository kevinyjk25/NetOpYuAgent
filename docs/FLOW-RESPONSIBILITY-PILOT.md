# 职责映射真实验证 / Responsibility Mapping Real-Model Pilot

## 中文

### 结论：本轮未带来转译可用性提升

2026-09-07，在代码检查点 `1c5fc7e` / 文档检查点 `e3b1d43` 后，新增独立双阶段入口并冻结新版职责映射批次。**8 次真实 qwen3.5:9b 调用：第一步流程 4/4 结构合格，第二步映射 0/4 合格；没有进入完整源审查，没有被接受的未激活流程。**

这是明确的负结果：不能把“更严格地阻断”称为可用性提升。上一版映射资格是 3/4，本版是 0/4，但门禁/表达协议不同，不能直接解释为模型业务理解准确率从 75% 下降到 0%。本轮暴露了**模型的表示/分类问题，也暴露了协议自己无法表达合法终态的问题**。原始结果保持不变，不修答案或在本批调参重跑。

范围仍为 **4 个已知人工开发流程 / 1 工具 / 0 公开 Skill**，不是未知集、独立 Gold 或生产成功率。每个流程都是新调用生成，其树内容恰好与上一批相同；并未读取旧树作为答案。模型批次 Runtime、业务工具、脚本和写执行均为 0。完整源审查数量是 0；下列定位为同一开发助手的事后诊断，不伪造成完整语义评分。

### 四个用例暴露的问题

| 用例 | 编译器首次阻断 | 原始结果进一步诊断 |
|---|---|---|
| direct-read | `/selections/clause-0002/0`：operation 类型选择了 `flow:/steps/1` | 结束节点本身正确，但标签不兼容；这不是已证实的业务图错误。输入/权限/返回/引用的复合检查仍整句归为 error_propagation |
| inverted-branch | `/selections/clause-0002/0`：branch 类型引用了分支叶节点 | 仍缺条件节点 `/steps/1` 来源。更重要的是目录中合法 `needs_l1` 终态的两个标签都没有任何可兼容类型，是协议表达缺口 |
| missing-approval-write | `/objective` 混入仅声明数据解释职责的 clause-0012 | 16 个子要求全含 unresolved，实际已有的读取和停止节点也没有来源映射；“读取输入”被归为 input_shape，而非操作 |
| unavailable-script-prerequisite | `/objective` 包含被类型表排除的 missing_capability_stop 片段 | 15 个子要求全含 unresolved，实际已有 unsupported 终态也未映射。缺脚本的真实前置问题仍保留，不能把整个业务合理停止本身判成失败 |

最后两案需分开看：缺少审批/写合同、缺少脚本时停止是正确边界；**把已有读/停止能力也笼统标成未解决、遗漏其映射**是额外可用性缺口。脚本案的停止要求是否属于整体业务目的，不能仅凭类型表武断判定；目的选择约束本身也需要审视。

机械统计：52 个源片段生成 53 个子要求，53 个引文均精确唯一匹配，但只有 1 个片段生成多个子要求。审批案唯一多项分解重复了整句，并没有形成清晰的局部引文。**精确引用不等于原子分解，更不等于完整原意覆盖。** 事后扫描发现 7 个声明类型/目标不兼容项；审批和脚本共 31 个子要求带 unresolved。它们是诊断计数，不是独立样本准确率，也不是执行失败率。

### 根因与修正方向

1. **类型约束只写进说明与事后校验，没有完整进入生成 Schema。** 当前 Schema 的 kind 与 targets 是独立枚举，模型仍可生成“合法字段、非法组合”。下一版需用按类型区分的 Schema 将必要兼容关系真正收口，但仍保留源语义审查。
2. **目标目录重复且含不可选项。** 同一节点同时出现 operation:/flow: 标签，部分标签对所有类型都不合法；`needs_l1` 则连节点的两个标签都无法映射。应使用单一节点标识、明确终态/交接语义，消除无语义价值的标签选择，不能让模型猜协议别名。
3. **新增类型没有自动带来语义分解。** 原句复制加一个类型并未解决复合要求。需单独评估分解完整性/错分类，并区分“发现无效输入/返回”与“把失败向上阻断”；不能用引用匹配率代替这些指标。
4. **未解决事实和映射缺口混在一起。** 实际不可执行的业务前置必须保留；可映射的读取/安全终态不应因此一概消失。后续应区分“流程正确表达了停止”与“业务能力尚不可用”，不删除真实问题，也不允许原生 L1 写 fallback。

下一步仍在 **C3h 表达协议纠偏**，不是增加 Runtime 执行功能或进入 C4。先完成上述表示完整性反例，尤其所有已有终态都能被表达；保留本批，另冻新版本再验证。不要继续仅追加提示词，或为提升通过率移除分解/完整源审查。

### 成本：输入减少，输出增加

| 指标 | 上版精简双阶段 | 本版职责双阶段 |
|---|---:|---:|
| 第一阶段合格流程 | 4/4 | 4/4 |
| 第二阶段合格映射 | 3/4 | 0/4 |
| 第二阶段输入 token | 25,172 | 10,544 |
| 第二阶段输出 token | 2,619 | 4,283（增加 63.5%） |
| 两步输入 / 输出 token | 36,188 / 3,600 | 21,560 / 5,264 |
| 两步总 token | 39,788 | 26,824（减少 32.6%） |
| 第二阶段 POST 时间 | 413.99 秒 | 224.21 秒 |
| 两步 POST 时间 | 577.64 秒 | 317.90 秒 |

本版第一步 93.69 秒、11,016 / 981 token。所有失败及等待均计入 POST 成本；预检、审查和回归时间不计。第二步预算由 2,200 提至 4,096，实际均正常 stop，未触顶；Schema/载荷、机器负载及部分并行回归不同，**观察时延不是严格因果加速**。成本减少而资格未通过，不能称为更好的总体效果。

### 实现、原始证据与复现

- [独立双阶段入口](../evaluation/flow_responsibility_pilot.py)：复用已验证的流程生成/职责映射函数，单独保留协议、指纹、检查点，不改旧入口。
- [新增 5 项批次回归](../tests/test_flow_responsibility_pilot.py)：新调用、旧答案排除、失败成本、完整检查点复用和篡改拒绝；与职责映射回归合计 80 项通过。
- [完整摘要及逐项原始选择](benchmarks/flow-responsibility-c3h-summary.json)：以摘要绑定完整报告，包含成本、原始阻断、全部子要求和事后诊断；不是修订后的模型答案。

本地原始批次：`artifacts/translator-v2/flow-responsibility-4-20260907`；完整报告：同目录级 `flow-responsibility-4-20260907-report.json`。

Manifest：`sha256:c3afa474ed7d731f27df57f7770d82f0a0405e6e31f2ea940c51b164d15bfac4`。

报告：`sha256:df9d983f51376834d9812f6a75ebbd082c5b8a782c8211bb8bbbb701e98246ba`。

离线重放（输出文件须不存在；不调用模型）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_responsibility_pilot report \
  artifacts/translator-v2/flow-responsibility-4-20260907 \
  --output /tmp/flow-responsibility-replay.json
```

全量 **1293 tests + 81 subtests 通过（160.90 秒）**，Ruff/diff 通过；新报告重放、完整检查点重入不增加调用、旧精简与完整双阶段含审查报告重放一致。回归耗时不是模型性能。原始制品仍为被 Git 忽略的本地文件，摘要不是完整可移植证据包；本轮新增入口/报告尚未提交或推送。

## English

### Negative result, not improved translation usability

Eight fresh qwen3.5:9b calls on four known hand-authored development flows, one tool, zero public Skills: **4/4 flow qualification, 0/4 mapping qualification, zero complete source reviews and zero review-supported inactive flows**. The prior lean mapping qualified 3/4, but protocols/qualification criteria differ: this is not a measured drop from 75% to 0% semantic accuracy. New trees happened to equal the previous trees, but were regenerated, not substituted. No retries, answer repairs or batch Runtime/provider/script/write execution. Post-hoc same-developer inspection is not full semantic review or independent Gold.

Direct read chose the correct end node through an incompatible flow: alias. Branching again omitted the condition-node citation, while the protocol itself offered no compatible kind for either needs_l1 terminal alias. Approval placed data interpretation into objectives, classified reading as input shape, and marked all sixteen requirements unresolved. Script marked all fifteen unresolved, including the available unsupported stop; the objective rule rejected missing-capability-stop clauses even though safe stopping can be part of the requested business outcome. Preserve genuine missing approval/script dependencies, while distinguishing them from unnecessary loss of available read/stop mappings.

All 53 quotes over 52 clauses match uniquely, but only one clause received multiple requirements, repeating the whole sentence. Exact citation is not atomic decomposition or fidelity. Seven declared type/target incompatibilities and thirty-one unresolved requirements are diagnostic counts, not accuracy or executed failures.

### Root causes and next step

Necessary kind/target compatibility is not encoded as a discriminated generation Schema; independent enums allow invalid combinations. Duplicate operation:/flow: aliases include unusable choices, and needs_l1 has no representation at all. Sentence copying with type labels does not establish complete decomposition. Genuine unavailable capabilities and missing annotations are conflated. Next remains C3h: express necessary compatibility in Schema, use unambiguous node identities, cover every existing terminal/handoff, and distinguish correctly represented safe stops from unavailable business execution. Retain full-source review, original issues and effect authority. Freeze a separate subsequent version; do not tune/rewrite this batch or add only more prompt text. C4–C6 remain gated.

### Costs and verification

Mapping input decreased 25,172→10,544, while output increased 2,619→4,283 (+63.5%). Whole-chain input/output: 21,560/5,264; total tokens decreased 39,788→26,824 (−32.6%). Mapping POST: 224.21 s; whole chain: 317.90 s versus prior 577.64 s. First pass: 93.69 s, 11,016/981 tokens. Failures/waiting included; preflight/review/tests excluded. Different payload/Schema, increased second-pass budget (2,200→4,096), load and partial concurrent regressions prevent causal latency claims. All outputs ended normally without budget truncation. Lower cost with no qualified mappings is not better overall effectiveness.

The linked runner, five new pilot tests and digest-bound summary preserve raw selections, original blockers and separate diagnostic findings. Eighty focused tests and **1293 tests + 81 subtests passed in 160.90 s**, with Ruff/diff, new report replay, completed-checkpoint re-entry without extra calls and unchanged previous lean/full two-pass reports. No full semantic verdicts were fabricated for pre-review failures. The offline command above replays local evidence without model calls and refuses existing outputs. Ignored raw artifacts are not included in tracked summaries. This turn's additions remain uncommitted/unpushed.
