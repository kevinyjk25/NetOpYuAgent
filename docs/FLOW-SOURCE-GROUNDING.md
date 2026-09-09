# C3c：源文约束与流程保真诊断 / Source grounding and flow fidelity

## 中文

### 结论先行

**源文约束与审查能力已补充，但整流程转译质量尚未达标。** 本阶段没有修改 Runtime 的执行逻辑、放宽准入或恢复大规模 Runtime A/B，也没有接入新的公开 Skill。C3b 的源文件、模型回答和报告保持不变，旧批次可原样重放。

本次在同一 4 个已知开发流程上验证两种不同协议，每种每项首次调用一次 9B。两批不是独立样本，不可合并成 8 个 Skill 或用作泛化准确率。用例仍为单读、反向分支、缺审批/写合同和缺脚本前置依赖。

| 协议 | 引用＋结构合格 | 真实调用耗时 | 输入 / 输出 token | 后续源审查 |
|---|---:|---:|---:|---|
| 原文摘录 `source-cited-flow/v1` | 1/4 | 166.51 秒 | 5,100 / 1,603 | 缺脚本案 10 项 supported，但真实缺能力事项仍阻断 |
| 原文编号 `source-selected-flow/v1` | 2/4 | 139.82 秒 | 4,872 / 1,383 | 单读案 11 supported / 2 insufficient；审批案 4 supported / 13 contradicted / 4 insufficient，均阻断 |

这是 **4 个已知流程、8 次不同开发协议调用、同一助手参与用例和审查**。累计计时 306.33 秒，9,972 / 2,986 token；不含预检、审查和测试。没有自动修图、重试同一协议或把低耗时安全停止称为加速。0 个获得审查支持的未激活流程，0 次 Provider / Runtime / 写入 / 脚本执行。

[可重算证据报告](benchmarks/flow-source-grounding-c3c-summary.json)分别保存两批各项状态、请求/响应/提案摘要和源审查发现。声明计数重叠，不是准确率或置信度。缺脚本案正确保留了安全停止，不能因为最终 blocked 就称该语义判断错误；单读案执行骨架正确，阻断来自用途限制覆盖与逐节点引用不足，不应等同业务逻辑错误。

### 改动了什么

1. **宿主上下文保留业务含义。** 除输入/输出 Schema，传入既有合同的 description 和来源 Skill 声明；这些是工具含义，不是目标流程，也不授予权限。没有新造审批工具或根据用例名字特判。
2. **用途与说明不再自由生成。** 第一版由模型选原文摘录；第二版改为编译器给定原文编号、模型只选择编号。目的和终态说明由代码原样投影，不由代码补写业务解释。
3. **真假路径分别引用。** 第二版在分支节点 Schema 中强制 `true_source_id`、`false_source_id`，消除可空字段导致的漏填；但两者引用存在，不代表分支极性正确。
4. **声明前置依赖。** 每步 `requires` 必须指向所有到达路径上的前序节点；自依赖、重复依赖和仅在另一分支出现的依赖会拒绝。未声明或错误解释的自然语言依赖仍须审查，代码不声称自动发现所有遗漏。
5. **未解决事项分类。** 区分源歧义、缺宿主能力、不支持控制流；无事项应为 `[]`。不自动删除模型的 “None” 等文本来获得准入，真实缺能力仍阻断。
6. **审查绑定整个映射。** 源要求→操作→条件→依赖的新增声明全部参与审查摘要；变更映射即使图未变，旧审查也失效。原文编号只负责定位，语义蕴含仍是审查判断。

实现位于 `evaluation/flow_grounded_translation.py`（摘录绑定、投影、审查及第一版诊断入口）和 `evaluation/flow_source_selection.py`（当前编号协议开发入口）。二者只产生**未授权实验提案**，不注册 DSH Tool，不激活 L0，也没有新的执行器；最终图继续复用已有 `lower`/`qualify_flow`。

### 没有解决什么：保留反例而不是刷通过率

- 摘录版会引用宿主工具说明而非目标流程原文，并漏填分支引用；其系统提示还继承了旧问题字段名称。该协议保留为诊断历史，不作为后续默认入口。编号版将目标原文与宿主说明分开，使用独立且字段一致的提示词。
- 编号版反向分支仍把真假方向写反，并把分支节点当作读取结果来源，还存在自依赖。**引用协议不能替代业务控制流理解。**
- 编号版审批案继续把库存 `status` 与 `approved` 比较，产生无源的完成出口；仅有字符串类型和工具说明并未阻止事实含义混用。
- 编号版脚本案仍先读取，再生成指向不存在节点的分支；摘录版对此曾正确产生 unsupported。这是开发协议改变后行为不一致，不是稳定提升。
- 单读编号版虽然图正确，但给结束节点选择了数据限制段，未用能支持“完成”的源段；用途也未包含限制。更严格的审查能暴露映射缺口，却会降低准入，必须区分业务错误和证据不足。

### 如何使用与重放

输入为已冻结的 `FlowSources` JSON（目标源文、宿主 Schema、已编译只读合同、已有 Effect 候选和时间预算），不是任意可执行脚本。输出目录必须不存在：

```bash
.venv/bin/python -m evaluation.flow_source_selection author path/to/sources.json path/to/new-output
.venv/bin/python -m evaluation.flow_source_selection assess path/to/new-output path/to/new-review-report.json --review path/to/review.json
```

输出依次保留原始请求/响应、`selected-proposal.json`、摘录映射 `cited-proposal.json`、投影后的 `draft.json`、双向 `review-input.json`、状态。若校验失败，只保留已完成步骤，不补造后续文件。通过 assess 仍不授予执行权限。

本轮报告可重算到一个新的输出文件，不再请求模型：

```bash
.venv/bin/python -m evaluation.flow_source_report artifacts/translator-v2/flow-cited-4-20260907 artifacts/translator-v2/flow-selected-4-20260907 artifacts/translator-v2/flow-c3c-review-20260907 --output /tmp/c3c-evidence-new.json
```

代码、输入、请求和模型摘要均在每批调用前冻结。首批 manifest 曾在任何模型调用前因 JSON 数值序列化差异被完整性断言拦下，重新绑定实际落盘数据后才开始；没有删除或重跑已完成回答。原始制品在本地忽略的 artifacts 目录，Git 摘要不能代替完整原始实验包。

新增 **22 项回归**，全量 **983 tests + 81 subtests** 通过（125.48 秒）；Ruff、diff 和两批报告重放通过。原 C3b 冻结代码未改，仍可重算。

### 下一步：不是继续堆提示词

建议先做一个有界的 **层级流程表达→既有图编译**：模型只表达顺序、条件的相等/否则子流程和引用来源；节点 ID、next、分支汇合和可达性由编译器机械生成。它不替模型判定哪个分支正确，也不新增 Runtime 执行器。这样可把“理解业务错误”和“手写图接线错误”分开衡量。

同时，字段应基于真实宿主合同绑定到有明确用途的业务事实，缺少审批事实合同就保留缺能力，不能靠 `status` 字符串同名推断。没有来源支持的语义类型不猜测。先做编译等价性、分支极性、依赖、错事实引用的离线回归，再冻结少量新表达验证；原 12 个公开 Skill 完整转译、未知集合泛化和 Runtime A/B 仍未解锁。

## English

C3c adds source-grounded proposal/review machinery, **not demonstrated reliable whole-flow translation**. Runtime execution and admission rules are unchanged; C3b artifacts and code remain replayable.

Two protocols each made one 9B call on four known flows. Free excerpts qualified 1/4; compiler-owned source IDs qualified 2/4. These are different development protocols, not eight distinct Skills, independent samples or a causal improvement estimate. Total request time was 306.33 seconds, with 9,972/2,986 input/output tokens, excluding preflight/review/tests. Zero review-supported inactive flows and zero tool/Runtime/write/script executions.

The excerpt protocol's missing-script stop received ten supported claims but retained a genuine unresolved capability. The ID protocol's direct-read draft received eleven supported and two insufficient-evidence claims; its approval draft received four supported, thirteen contradicted and four insufficient judgments. Same-assistant review is not independent evidence. A faithful unsupported stop is not business completion, and insufficient evidence is not necessarily incorrect business logic.

The new protocol forwards real host semantic declarations, selects business/limitation text from target source spans, requires both branch citations, validates declared predecessor dominance, classifies actual unresolved issues and binds all mappings into review. It does not infer complete prerequisites, prove entailment or activate L0. Existing flow compilation remains authoritative.

Failures persist: reversed branches, references to non-read nodes, inventory status used as an approval fact, skipped script prerequisites and unknown successors. The excerpt protocol also exposed host/target citation confusion and an inherited legacy prompt-field mismatch; it is preserved as diagnostic history, not the default interface. Source IDs fix citation mechanics, not business semantics.

See the [replayable report](benchmarks/flow-source-grounding-c3c-summary.json). Twenty-two new regressions; 983 tests and 81 subtests passed in 125.48 seconds, plus Ruff/diff and evidence replay. Raw artifacts remain local/ignored. A pre-inference manifest serialization mismatch was corrected before any model call; completed outputs were never edited or retried.

Next, evaluate a bounded sequence/if/otherwise representation mechanically compiled into the existing graph, separating business interpretation from manual graph wiring. Bind field meanings only to real host contracts, never invented approval facts. First validate offline equivalence/dependency/type rules, then a small newly frozen pilot. Public whole-Skill generalization and large Runtime A/B remain gated.
