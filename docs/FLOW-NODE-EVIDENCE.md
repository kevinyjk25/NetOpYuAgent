# 节点证据与原文剩余要求 / Node Evidence and Residual Source Duties

## 中文

### 为什么调整方案

[分层诊断](FLOW-DIAGNOSTICS.md)表明，上一版并非单纯缺少 L0 构造器：模型会遗漏已有节点证据、重新猜节点职责，并独立选择与自身分类冲突的目的。因此本轮改变生成任务的组织，不继续增加泛化类别或放宽门禁。

**节点必填不等于必须声称支持。** 如果要求模型给每个节点找正向引文，就可能为错误流程编造理由。新版每个实际节点必须显式选择：

- `evidence_candidate`：有候选来源，仍待完整审查；
- `insufficient_evidence`：证据不足；
- `contradicted`：与原文矛盾。

后两者需要说明原因，允许没有正向引用，并阻断编译。任何分支都不能获得执行权限；停止/交接节点有证据不表示缺失审批、脚本或业务写已经完成。

### 新格式与职责

模型只生成两个区域：

```text
node_evidence
  每个实际 L0.5 节点 → 状态、来源片段、精确引文、是否承载业务目的
residuals
  每个源片段 → 未被节点表达的检查、限制、依赖或未解决要求
```

节点键来自实际树，角色由编译器固定：read/effect_candidate 是 operation，条件是 branch，终态分别为 completion/handoff/missing_capability_stop。模型不能把一个 read 节点重新声明成输入格式检查，也不能增删节点、改参数或极性。

每条正向引用可以标记 `objective=true`；不能在反驳/证据不足引用上标记目的。剩余要求只有 `unresolved` 可以保留尚未表达的业务目的；解释性背景和宿主规则不允许成为目的。编译器从这些条目推导原格式的目的来源列表，不再让模型另外重选 `/objective`。

`residuals` 每个源片段都必须出现。只有其所有语义都已由节点引用表达时才能为空；若没有正向节点引用也没有剩余要求，则机械阻断。**有一个引用不意味着整句所有要求已表达**：复合动作、否定、条件、输入/返回校验、错误传播和权限限制仍须逐项完整源审阅。

剩余处理选项保持受限：输入格式、读取权限、返回格式、错误传播、解释边界、权限边界、未解决要求。其类型/目标由编译器确定，避免模型同时猜两个相互依赖的标签。这里的“固定角色”只约束输出结构，不能证明源文适合该处理。

### 编译与安全边界

新协议 `node-evidence-residual-source/v1` 投影至已有 CanonicalMapping，再调用原编译/源审查链。旧协议和冻结结果不变。投影保留所有候选及原始指针，不自动修图、补造引文、删缺能力事项或截断超限要求。

保留旧约束：每片段最多六项投影要求、每节点最多六个来源、目的最多六片段、完整审查最多 256 声明。超限明确阻断；不是通过删掉要求来提高通过率。

完整 reviewInput 同时绑定原文/宿主、树、新提案、Schema、旧格式投影和逐项路径关系。追加每节点“参数、谓词、分支极性、前置和终态均由原文支持”的审查声明。即使两个提案投影出同一旧目的列表，只要新引用/目的标记变化，旧审阅也不能复用。

原文代码保持惰性；代码片段只能作为 unresolved 剩余内容。代码-only 输入仍有合法的“证据不足”输出形状，不能被迫伪造正向证据。缺能力/未解决要求即使忠实表达也继续阻断准入，产品路径不回退为 L1 直接写。

### 使用与证据纪律

- [协议实现](../evaluation/flow_node_evidence.py)
- [独立冻结双阶段批次](../evaluation/flow_node_evidence_pilot.py)
- [协议反例](../tests/test_flow_node_evidence.py)与[批次检查点回归](../tests/test_flow_node_evidence_pilot.py)

```bash
.venv/bin/python -m evaluation.flow_node_evidence request sources.json tree.json --output request.json
.venv/bin/python -m evaluation.flow_node_evidence diagnose sources.json tree.json --proposal mapping.json --output diagnostics.json
.venv/bin/python -m evaluation.flow_node_evidence compile sources.json tree.json --proposal mapping.json --output compilation.json
```

所有输出都拒绝覆盖。`assess` 另需 `--review review.json`，沿用 ReadL05Review 的摘要绑定审查格式。任何测试夹具或同一助手审查都不是独立 Gold。

正式开发批从原始 source manifest 重新生成流程，再生成节点证据和剩余要求；每阶段一次 9B 调用，无旧树/映射答案输入或修图重试。旧失败保持原成绩。结构通过、正确阻断、忠实表达缺能力、真正可用须分开报告；本轮不涉及 Runtime 执行或生产工程扩展。

### 2026-09-08 真实批次：结构局部改善，语义仍未通过

8 次真实 `qwen3.5:9b` 调用，**4/4 流程结构合格、4/4 映射符合生成 Schema、1/4 映射编译合格**；进入完整审阅的反向分支例仍 blocked，未接受任何流程。10 个实际节点均显式表态：7 个候选支持，3 个否定/证据不足；“节点表填满”不能当成节点语义正确率。

| 用例 | 本轮结果 | 语义诊断 |
|---|---|---|
| 单次读取 | 两个节点都有证据，但所有目的标记为 false，编译阻断 | 两个读取句各被另外分配四项宿主校验，存在无依据的剩余要求；不能只修目的标记 |
| 反向分支 | 5 个节点含条件均有证据，映射编译通过 | 读取仍被额外标成输入格式检查，业务分支被标成解释边界，库存含义被标成权限边界 |
| 缺审批变更 | 首读被判证据不足，正确的 unsupported 停止被判矛盾 | 理由混淆了 s000x 与 clause 编号，以及“候选表达”和“已执行/已授权”；真实缺审批/写能力仍应保留 |
| 缺前置脚本 | 正确停止被判矛盾，7 个源片段未被交代 | 模型将“停止候选”误认成禁止执行的操作；8 项 unresolved 不能弥补另外未覆盖的源片段 |

对唯一编译通过例完成同一开发助手的 **95 项声明审阅：69 supported、10 contradicted、16 insufficient_evidence**，结果 blocked。原文、宿主与所有声明均审查留存；这是重叠声明的开发诊断，不是独立语义义务分母，不能计算泛化准确率。其他三例未进入完整审查，上表是事后局部语义检查，不能冒充四例全部完成语义审阅。

与上一批相比，映射编译从 0/4 到 1/4，但完整可用性仍未证明。两步总 token **25,881 → 25,086（−3.1%）**；总 POST **341.16 → 342.92 秒（+0.5% 观察值）**。本轮第一步 82.75 秒、第二步 260.18 秒；输入/输出合计 19,090 / 5,996。不同载荷、协议与机器负载不能做因果性能推断。所有回答正常 stop，无重试/改答案；审批理由出现句子未写完，不能因此声称整次调用达到输出 token 上限。

**33 项新增回归、全量 1511 tests + 81 subtests 通过（131.41 秒）**。全量测试在模型批次结束后运行；测试耗时不是 Runtime 指标。Ruff/diff 与新旧报告重放检查通过。

源 manifest：`sha256:7d5d60d68f4496e3c213dc128f3fbfda67f1007d8ddb7132175b9b7547be4aec`。

含完整审阅的报告：`sha256:36cf4f488d646c2e204d39157d66bc67a3abe99f68f43241bb72e2110ba408d9`。

[原始提案、诊断、审阅与成本摘要](benchmarks/flow-node-evidence-c3h-summary.json)保留所有新提案和 95 项审阅；摘要不替代原始 HTTP/宿主/环境收据。完整本地批次为 `artifacts/translator-v2/flow-node-evidence-4-20260908`，审阅在同级 `-reviews`，完整报告为同级 `-report.json`。

### 方案进一步纠偏：先原文义务，再绑定宿主

2026-09-08 后续决策：以下设计方向保留，但先完成[GPT/9B 同协议诊断](FLOW-MODEL-COMPARISON.md)，以区分模型能力与协议问题；不同时改模型和语义组织。此处记录当时的候选修订，不覆盖最新实验顺序。

本轮暴露的问题不是继续加类型就能解决，下一批前按以下顺序调整：

1. **分开原文义务与宿主固有保障。** 先从原文提取动作、条件、依赖、限制及不确定事项，不让宿主规则菜单诱导模型给每句“补齐所有检查”。宿主必须执行的安全规则可以独立附加，但不能伪称来自那句 L1 原文。
2. **清理第二阶段的元数据干扰。** 只暴露节点真正的操作、参数、边和终态，不把第一阶段未审阅的 source_id 再作为业务事实。来源追踪仍由编译器完整保存，不能丢原始证据。
3. **目的不再反复做独立分类。** 明确“候选业务范围”和“原文已证实的用户目的”之间的关系，不通过自动把 false 改 true 或任意继承父标题消除失败。
4. **候选/执行/授权三者分开验证。** 正确的读取候选或缺能力停止不等于已读取、已批准或已配置；缺证据的正向接受与无依据的反驳都要接受源审阅。保留三态出口，但不能把更多拒绝当作安全提升。

以上是下一步方案纠偏，**尚未实现或证明消除这些语义失败**。本协议/批次已经冻结，不能直接改提示词或答案重跑刷分。先构建对应离线反例与原文义务接口，再另冻新版验证；仍在 C3h，不解锁 C4–C6 或大规模 Runtime 评测，不进入生产工程。

## English

### Design correction

Previous layered diagnostics found redundant node-role inference, omitted citations and objectives inconsistent with the model's own classification. The new protocol changes task organization rather than adding more semantic labels or relaxing gates.

Mandatory node slots do **not** force positive justification. Each actual node must declare evidence_candidate, insufficient_evidence or contradicted. Positive slots require exact source candidates; negative slots require reasons and block compilation without inventing support. A supported stop/handoff does not implement a missing approval, script or business effect.

### Format and compilation

The model supplies `node_evidence` and `residuals`. Actual node keys and roles are compiler-owned. Business objectives are derived from flags on positive node citations or unresolved residual duties, never independently selected again. Background and host-check residuals cannot declare objectives. Every source clause needs an explicit residual entry; an empty residual array is allowed only where nodes account for the whole clause. A citation alone does not prove complete decomposition.

Residual handling remains bounded to input shape, read access, result shape, error propagation, interpretation/authority limits and unresolved duties. Fixed roles/targets prevent incompatible labels, not semantic misclassification. Full source review must still detect missing conditions, negation, checks, prerequisites and limitations.

The protocol projects to existing CanonicalMapping and preserves its compiler and review gates. Every candidate retains its original and projected pointers. Six requirements per clause, six citations per node, six objective clauses and 256 review claims remain limits; excess is rejected rather than truncated. The review packet binds the original proposal, Schema and projection and adds node-specific source-fidelity claims for parameters, predicates, polarity, prerequisites and outcomes. Raw objective-flag changes invalidate prior review even when the projected clause list is unchanged.

Source code is inert and unresolved, including code-only inputs with an explicit no-evidence output shape. Missing capabilities and unresolved business duties still block admission. No native L1 direct-write fallback or execution authority is introduced.

### Evaluation discipline

The linked standalone pilot freshly generates both flow and evidence mapping using qwen3.5:9b, one attempt per phase, with frozen source/implementation/model/environment and raw checkpoints. No previous answer or tree is substituted, no failed answer is repaired/rescored. Structural qualification, faithful stops, semantic review and usable flows remain separate outcomes. Fixtures and same-assistant reviews are not independent Gold; this milestone does not establish public-Skill generalization or production reliability.

### Fresh result and next correction

Subsequent decision: retain the design corrections below as candidates, but first establish the [GPT/9B reference experiment](FLOW-MODEL-COMPARISON.md). Do not simultaneously change the model and semantic protocol or treat historical next steps as the current experimental order.

Eight real calls produced **4/4 qualified flows, 4/4 Schema-conforming mappings and 1/4 compiled mappings**. All ten node slots are explicit, with seven positive candidates and three negative/insufficient declarations. The compiled branch now includes condition evidence, but its full same-developer review of **95 overlapping claims (69 supported, 10 contradicted, 16 insufficient)** remains blocked. No candidate is accepted. The other three cases are mechanically blocked and have only partial post-hoc semantic observations, not complete reviews.

Direct read loses all objective flags and invents host-check residuals for read sentences. The branch additionally labels reads as shape validation, operational paths as interpretation limits and inventory meaning as authority. Approval/script cases wrongly treat read or unsupported-stop candidates as forbidden executed actions and confuse inherited source IDs with clause identifiers. Genuine missing business capabilities remain unresolved; explicit rejection is not automatically correct or a safety gain.

Total tokens **25,881 → 25,086 (−3.1%)**; POST **341.16 → 342.92 s (+0.5% observed)**, first pass 82.75 s and mapping 260.18 s; input/output 19,090/5,996. Different payloads/protocol/load prevent causal performance conclusions. All calls stop normally without retry or answer repair. **33 new tests and 1511 tests + 81 subtests passed in 131.41 s**, with full tests after model completion, plus Ruff/diff and new/old replay. [Evidence summary](benchmarks/flow-node-evidence-c3h-summary.json) includes raw proposals and the complete review, but not all raw host/HTTP receipts.

Before another batch, separate source-only obligation extraction from host-rule binding, strip unreviewed inherited source IDs from second-pass execution context while retaining provenance separately, remove redundant objective classification without inventing intent, and distinguish candidate representation from execution/authorization. These are revised next steps, not implemented semantic fixes. Preserve this frozen batch and test new counterexamples/interfaces before a separately frozen protocol. C3h and generalization gates remain open; no large Runtime evaluation or production expansion is unlocked.
