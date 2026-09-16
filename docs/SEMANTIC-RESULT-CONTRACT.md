# 双驱动结果合同 / Governed Hybrid Result Contracts

## 中文

2026-09-10：阶段 2 检查点为 `25b08c0`。本轮推进的是后续“语义闭环修复”的**结果边界子阶段**，不是整个语义闭环、完整双驱动转译或未见泛化完成。

本子阶段已完成本地验收：81 项定向检查；全量 **2664 项＋81 子测试通过**；10 个变更 Python 文件 Ruff、文档链接与 Git diff 检查通过；200 项制品/源码/回执完整性检查通过。首轮全量的 1 个文档链接失败与最终全量结果分别保留。新改动尚未提交；未推送、未变更默认 DSH 路由或固定 A/B 基线。

### 解决什么，尚未解决什么

之前的 `governed_graph_completed` 只回答流程有没有按合同运行。模型仍可能在交接草稿里编造时间，在 README 中给出未观察到的命令，或者把局部查询写成全部完成。

现在可在原 `run_hybrid` 上安装宿主批准的 `ResultContract`。图状态与结果评估分别返回：

```text
L1 / Task 引用 → 宿主审查的职责与观察字段映射
                         ↓ 摘要绑定，不是模型自我批准
原严格读取 → 有界 LLM → values + draft + notes
                         ↓
          实际读取回执 / 精确字段检查 / 开放职责保留
                         ↓
        observedFields | 未验证草稿 | 未满足职责与修复定位
```

这层没有自然语言事实裁判。**它能拒绝声明字段与观察不一致，不能自动发现自由草稿中的全部错误。** 当前字段映射由开发者 AI 审查并交宿主安装，不是转译器自动生成并证明正确。把一整段原始笔记复制正确，不代表完成了交接写作。

### 三类职责

| 合同职责 | 实际检查 | 明确不能推出 |
|---|---|---|
| `read_completed` | 本次调度器确实取得指定严格读取节点的结果 | 整个业务过程完成 |
| `observed_value` | 模型指定字段与宿主指定观察位置的 JSON 值精确一致；缺失不补默认值 | 工单、文档或 Provider 内部的陈述已经独立证实 |
| `open_semantics` | 保留未验证推理、缺观察、缺控制能力或需追问职责 | 模型说“完成”就可以清除该职责 |

`source_ref` 复用原审阅档案的职责/引用定位；演示使用原 Task 的精确 `SourceSpan`。`mapping_digest` 绑定审阅材料，**不证明引用蕴含或职责提取完整性**。不新建与 `task_alignment`、源职责映射平行的自动提取器。

模型输出中：`values` 是待核验字段；`draft` 保留原任务要求的自然语言工作；`notes` 是最多六条限制说明。后两者始终是候选内容。初版只有 values/notes，真实测试发现容易退化成罗列观察；因此保留独立 draft，不能为了检查方便删掉 L1 的开放职责。

### 准入与状态

原 `HostHybridConsent` 新增可选 `result_contract_digest`，同时绑定图、参数、身份上下文及结果合同。模型不能添加/删除合同、改变来源、清除开放职责或生成结果批准状态。合同不匹配在任何回调之前拒绝。结果检查不代替原候选准入、权限、证据时效或写事务检查。

结果评估在图终态生成，不能撤销已经发生的读取，也不能代替中途继续执行前的独立准入。它不是行动授权 Gate。若后续要让 LLM 候选参与下一次工具参数，仍须走原 `CandidateAdmission` 和该操作的严格检查。

| `resultAssessment.status` | 含义 |
|---|---|
| `declared_contract_satisfied` | 声明的全部有限职责满足；仍不证明整任务覆盖或自由草稿正确 |
| `partial` | 部分职责满足，仍有明确缺口 |
| `unverified` | 暂无职责被证实满足 |
| `rejected` | 声明的候选字段与实际观察冲突 |
| `blocked` | 图未完成，不能借已运行前段宣布任务完成 |

`completeAnswerApproved`、`wholeTaskDutyCoverageProven`、`modelDraftVerified`、`runtimeAuthorityGranted` 和 `effectAuthorized` 均不因这些检查变成 true。接口使用者应分别展示：宿主渲染的观察字段、**未验证草稿**、未满足职责；不能把模型草稿直接作为“验证通过”的答案。

每个 finding 包含 `dutyId`、`sourceRef`、`contractPointer`、读取节点/JSON Pointer、候选字段 Pointer、错误分类与 `repairTarget`，可定位是字段漏答、字段不符、必要读取未发生，还是仍需推理/控制能力。运行报告不靠关键词扫描草稿或模型 confidence 判定完成。

### 接入与本地验证

- Runtime 类型与结果评估：`network_runtime/l0/result_contract.py`。
- 调度接入：`network_runtime/l0/hybrid_execution.py` 的可选 `result_contract`。
- 原 Task/Skill 引用审查适配：`evaluation/hybrid_result_review.py`。
- 本地模型入口仍是 `evaluation.hybrid_live_demo`，新增 `--result-contract`、`--result-mapping`；必须另有绑定新摘要的宿主审阅文件。
- 三例示范：`evaluation.hybrid_result_batch`，沿用阶段 2 的 Notion、交接、README 原始 Task、Skill、读取计划和隔离夹具。**手写的结果映射仅证明机制，不算新转译结果。**

以下调用产生 3 次真实本地 `qwen3.5:9b` 推理；用 0 替代 3 则只准备材料。`NEW_OUTPUT_DIR` 必须不存在，且不能位于旧阶段 2 证据目录中。

```bash
.venv/bin/python -m evaluation.hybrid_result_batch \
  artifacts/translator-v2/stage2-20260910 NEW_OUTPUT_DIR --max-model-calls 3
```

查看 `summary/report.json`、各例 `execution/summary/report.json` 和 `inputs/mapping.json`。冻结目录保存输入/实现摘要与源码归档，模型原始请求、响应及回执另存；旧失败不覆盖。完整本地证据不随 Git 提交，仓内摘要见 [结果边界记录](benchmarks/semantic-result-summary.json)。

### 本轮发现的复现缺陷

最终 v4 的真实 9B 三例均图完成，但结果职责分别是 **Notion 2/2、交接 2/3、README 2/4**。后两者保持 partial。开发者 AI 另外审阅草稿：交接仍漏掉 Alice/Chen 的责任人信息；README 仍编造未经读取证实的安装、API 和测试示例，不能发布为可靠答案。这些错误没有被 Schema 自动识别，开放职责没有因此被清除。

两个真实版本各 3 次、合计 **6 次 9B 调用**；另有两个零模型预检失败。v3 和 v4 不混算语义改善率，v4 不是未见集，也没有完成原生 DSH 对照。逐例草稿、成本与摘要见[机器记录](benchmarks/semantic-result-summary.json)。

v3 消耗 15,721 输入 / 631 输出 token；v4 为 15,907 / 1,298。v4 三例图时延分别约 12.70、47.11、36.95 秒。恢复完整草稿增加了输出与耗时，不能把更短的观察列表包装为同等任务的性能优化；样本不支持 SLO 或因果加速结论。

两个零模型预检先后暴露同一类顺序问题：边界/源页被嵌为 JSON 字符串后，键顺序变化影响文本摘要；绑定器遍历对象的顺序又影响诊断映射列表。修复为 authoring v8 的稳定 JSON 文本序列化，以及 hybrid qualification 的对象键规范化；**数组、步骤、原文字符串顺序全部保留**。

旧 v7 制品不改写。示范在新目录保留原编译结果，零模型重新编译，并比较完整 Flow：只允许两个已知 JSON 文本封装的键顺序变化，其余指令、工具、参数、依赖、模型和预算不得改变。新的图与结果合同仍需新的摘要批准，不能跳过漂移检查。

### 下一出口

后续的[草稿双向审查/一次修订](BOUNDED-DRAFT-REVIEW.md)已经接入同一调度器进行已知样例验证；下列职责映射与通用语义出口仍未完成。该后续实验不改写本文件 v3/v4 的原草稿与评分。

1. 将关键职责从原文映射到图，并评估映射遗漏与同类型参数错绑，不能以宿主手写合同替代通用转译。
2. 对开放草稿增加逐项证据/职责审查及有界修订；AI 审阅仍不授予事实或执行权限。
3. 补完整“读取 → 推理 → 独立准入 → 后续读取”自动构造、有限选择与证据刷新；目前结果合同不自动生成这些流程。
4. 冻结后做新 cohort，再按正式门禁开展 DSH 对照。生产工程、自动写接线和大规模 A/B 本轮不推进。

## English

Stage 2 is checkpointed as `25b08c0`. This is the **result-boundary substage** of semantic-loop repair, not completion of the semantic loop, automatic mixed-flow authoring or unseen generalization.

Local acceptance passes 81 targeted checks, 2664 full tests plus 81 subtests, Ruff on ten changed Python files, documentation links, diff checks and 200 artifact/source/receipt integrity checks. The initial full run's one missing-document-link failure and final passing run are both retained. New substage changes are uncommitted; default DSH routing, A/B baseline and remotes are unchanged.

An optional host-installed `ResultContract` on the existing `run_hybrid` distinguishes actual read completion, exact observed-value projection and unresolved open duties. Candidate `values` are checked against host-selected observation locations. `draft` retains the original open L1 task; `notes` retain bounded limitations. Neither free-form field is semantically verified. Source locators reuse existing dossier/review references, and the demo checks exact original Task/Skill spans. Mapping digests bind declarations, not entailment or complete duty coverage.

Host consent additionally binds the result contract. A mismatch or silent removal fails before callbacks. Assessments return `declared_contract_satisfied`, `partial`, `unverified`, `rejected` or `blocked`, independently of graph completion. Findings link duty/source/contract locations, observation node and pointer, candidate pointer and repair target. Correctly copying a note does not establish its truth or fulfill an open drafting task. Contract satisfaction never grants action authority, verifies arbitrary prose or approves a complete answer.

The local three-case probe reuses known Stage 2 Notion/handoff/documentation tasks, read plans and synthetic fixtures with **host-authored** result mappings. No new automatic-translation result is claimed. The initial values/notes-only revision made outputs too observation-oriented; the next revision keeps an explicit unverified draft. Both results and all failures remain separate. The command above uses exactly three actual local 9B calls; zero only prepares inputs. Use a fresh output directory outside the original evidence. Input/implementation digests, source archive and model receipts are retained. See the [portable record](benchmarks/semantic-result-summary.json).

Two zero-call preflights exposed JSON object-order sensitivity in embedded text and derived binding diagnostics. Authoring v8 canonicalizes the two JSON text envelopes; hybrid qualification canonicalizes object keys without changing array/step/text order. Original v7 artifacts remain untouched. New compilation compares complete flows modulo only those two JSON envelopes and requires new digest-bound consent—never a hash bypass.

Next: source-to-duty accuracy and parameter semantics; evidence-aware review/revision of open drafts; bounded read/reason/admit/continue authoring and refresh; then frozen unseen cohorts and paired DSH evaluation. Production engineering, default DSH routing and automatic Effect integration are unchanged.

The subsequent [bounded review/revision probe](BOUNDED-DRAFT-REVIEW.md) exercises the same scheduler on known failures. It preserves the original v3/v4 artifacts and scores; generic duty mapping and semantic acceptance remain open.
