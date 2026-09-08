# 统一节点与受约束映射 / Canonical Nodes and Constrained Mapping

## 中文

> 后续完整双阶段已运行：流程 4/4、映射 0/4，类型/目标非法组合消失，但目的选择和节点覆盖仍有缺口。详见[新批次结果](FLOW-CANONICAL-PILOT.md)。下文“完整双阶段待验证”保留本次协议修复/探针里程碑当时的状态。

### 完成的修复与证据边界

2026-09-07，针对[上一轮真实职责映射 0/4 的负结果](FLOW-RESPONSIBILITY-PILOT.md)，新增独立 `canonical-node-responsibility/v1` 协议。**统一节点标识、将类型—目标兼容关系写入生成 Schema、补齐现有三种终态，并把“表示忠实”与“业务可执行”分开报告。** 不修改上一轮冻结代码、回答、结果或 Runtime 执行器。

新增 142 项映射回归和 4 项探针回归，共 **146 项通过**；包含 104 项参数化类型/目标组合检查，不是 146 个 Skill 或独立语义样本。另完成 **3 次真实 qwen3.5:9b 固定父流程协议探针，3/3 结构编译通过**，含此前无法映射的 needs_l1。

这不是完整源→模型流程→模型映射的双阶段评测：父流程是显式提供的手工单终态树，源文也明确要求返回该终态；没有提供映射 JSON 答案，没有重试或补改。**仅证明本地接口能处理本轮 Schema 和这三个表示探针**，不证明所有 Schema 分支的解码行为、复合要求分解或 Skill 泛化改善。上一轮 0/4 的结果不变。新版完整双阶段质量仍待验证。

### 1. 同一节点只有一个标识

旧目录把同一节点同时暴露为 operation:/flow:，但部分别名在职责类型表中没有任何合法用途。现在模型只看到 `node:/steps/0` 这样的统一节点键；编译器才负责转换到旧审查结构，**不让模型再选择内部别名**。

| 真实节点 | 可映射职责 | 含义与边界 |
|---|---|---|
| read / effect_candidate | operation | 读取或效果候选，不是已经执行/提交 |
| if_equal | branch / business_prerequisite | 必须是实际条件节点；谓词、极性、前置完整性和顺序仍需审查 |
| end: read_path_completed | completion | 只读路径结束，不是写事务 commit |
| end: needs_l1 | handoff | 返回控制权；不自动调用上层模型、脚本或工具 |
| end: unsupported | missing_capability_stop | 明确停止，不是业务已完成 |

三个终态都有合法表示。若“遇到缺能力必须停止”本来就是源文请求的一部分，允许其成为目的候选；不再单凭类型否定这种目的。是否确实符合原文仍由完整目的/源要求审查决定，不能把数据背景混入目的。

### 2. 兼容关系进入 Schema，而非只放在提示词里

每类子要求使用一个带 `kind: const` 的 Schema 分支，`targets` 枚举只包含该类型可用的真实节点/规则和 unresolved。联合分支由共享定义引用，代码块只允许 unclassified/unresolved。

例如 branch 只能引用实际 if_equal 节点或 unresolved，不能再把分支内部完成节点当作条件。输入/返回形状、读权限和错误传播也不能互相替代。模型接口收到这份 Schema，编译器仍独立按同一 Schema 校验；**不假设所有调用者都会使用受限解码，也不假设受限解码绝不出错**。

这是必要的结构限制，不是语义证明：把“必须检查输入”错分为 interpretation_limit 后选择 documentation，仍可能通过 Schema。缺子要求、断章取义、错谓词、目的混入限制等反例，继续由完整源审查覆盖。条件节点被整体遗漏属于跨记录覆盖缺口，仍由编译器拒绝，不能靠局部 Schema 检查代替。

### 3. 正确表示停止，不等于可以执行业务

审查结果分别提供：

- `representationReviewSupported`：是否每个现有声明都获得审阅者支持；它不是独立 Gold、完整性证明或 calibrated confidence。
- `admissionBlockers`：完整源审查未支持、父流程真实问题、unresolved 子要求等阻断原因。
- `parentIssues` / `unresolvedRequirementPointers`：保留具体缺能力事项与定位。
- `runtimeAuthorityGranted: false`：包括“表示获支持且无上述阻断”的未激活候选，也不授予运行权限。

因此，“必须审批但宿主没有审批合同，所以正确停止”的表示可以获得审查支持，同时业务资格仍 blocked。**映射到停止节点不会消除缺失的审批/脚本或允许原生 L1 直接写入。** 这个拆分不会自动改变评测指标或放宽现有资格门禁。

### 4. 9B 协议探针实测

| 固定父流程终态 | 生成类型 | 编译结果 | POST 时间 | 输入 / 输出 token |
|---|---|---|---:|---:|
| read_path_completed | completion | 待完整源审查 | 8.60 秒 | 1,365 / 115 |
| needs_l1 | handoff | 待完整源审查 | 7.31 秒 | 1,370 / 116 |
| unsupported | missing_capability_stop | 待完整源审查 | 7.46 秒 | 1,363 / 115 |

合计 **23.38 秒，4,098 / 346 token**。3 次均正常 stop，0 次完整语义审查，0 个公开 Skill，0 Runtime/业务工具/脚本/写执行。原始回答都仅用一个子要求引用整句，**尚未单独表达“不调用模型/工具”等限制**；探针通过不能说明完整分解已解决。

部分全量回归并行；时间含 POST 等待，不含预检、审查和回归，不能与上一轮复杂四流程的耗时比较后宣称加速。探针是 decoder/表示接线检查，不是新的 3/3 Skill 转译准确率。

### 实现与复现

- [协议、Schema、投影、编译和审查](../evaluation/flow_canonical_mapping.py)
- [固定父流程探针及冻结/重放入口](../evaluation/flow_canonical_canary.py)
- [映射回归](../tests/test_flow_canonical_mapping.py)、[探针回归](../tests/test_flow_canonical_canary.py)
- [含逐项模型输出与制品摘要的探针报告](benchmarks/flow-canonical-canary-report.json)

通用离线命令与上一协议相同，但模块名为 `evaluation.flow_canonical_mapping`：`request sources.json tree.json --output request.json`、`compile sources.json tree.json proposal.json --output compilation.json`、`assess sources.json tree.json proposal.json review.json --output assessment.json`。输入是实际对应模型的 JSON，输出文件必须不存在；request 只导出请求，不调用模型。

本地探针根目录：`artifacts/translator-v2/flow-canonical-canary-20260907`。仅离线重放（输出文件须不存在）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_canonical_canary report \
  artifacts/translator-v2/flow-canonical-canary-20260907 \
  --output /tmp/flow-canonical-canary-replay.json
```

Manifest 为 `sha256:b78626d9e6003a9b57bc8bea557aeed4956abed7ad0e104443b46e3da1b3d6e5`；报告为 `sha256:fddc4671795284a267025ff3f54db3f70e815ceadf799ba510e08afe87f8a124`。完整检查点重入不重复调用，部分检查点/漂移拒绝，不覆盖原始证据。原始制品在被 Git 忽略的本地 artifacts；提交报告不等于完整原始证据包。

### 尚未完成

验证收尾：全量 **1439 tests + 81 subtests 通过（111.44 秒）**；Ruff/diff、新探针报告重放和完整检查点重入无新增调用、此前职责/精简/完整双阶段三份冻结报告均重放一致。回归包含已有隔离执行测试，其时长不是 Runtime 性能指标。本轮新增文件及文档尚未提交或推送。

仍在 C3h。下一步另冻新版完整双阶段小批，重点检查：结构拒绝是否减少、复合要求/否定/条件是否保真、是否仍过度 unresolved，以及正确表示停止与实际任务可用性的差别。不能把旧回答机械换标签后计为新模型成功。

现有片段/来源/目标/审查上限保留，超限拒绝不截断；多层流程或业务前置是否能完整表达仍受既有图/合同约束。没有新增浏览器展示、生产功能或 Runtime 执行授权；C4–C6 不解锁。

## English

> Subsequent fresh paired validation yielded 4/4 flows but 0/4 mappings: type/target compatibility improved while objective and node-coverage gaps remained. See [new results](FLOW-CANONICAL-PILOT.md). Pending-paired statements below describe the earlier repair/canary milestone.

### Representation repair, not whole-Skill accuracy

The separate `canonical-node-responsibility/v1` protocol repairs duplicate node aliases, encodes kind/target compatibility in the generation Schema, covers all existing terminal outcomes, and distinguishes reviewed representation from business eligibility. Previous frozen protocols, raw answers and negative results remain unchanged; no executor change.

146 new regressions passed (142 mapping, including 104 parameterized kind/target combinations, plus four canary tests). These are not Skills or independent semantic samples. Three real qwen3.5:9b fixed-parent terminal probes compiled successfully: completion, needs_l1 handoff and unsupported stop. Parents are hand-authored, model-visible single-terminal trees and sources explicitly request those outcomes. Mapping JSON answers were not supplied. This checks local decoder/representation plumbing, not fresh source→flow→mapping translation or generalization. Prior 0/4 mapping evidence remains unchanged.

### Canonical targets, schema constraints and review

Each actual node has one `node:` identity. Read/effect candidates map to operation, conditions to branch/business prerequisite, and terminals to completion/handoff/missing_capability_stop. Legacy aliases are compiler-owned and not model choices. Handoff returns control without invoking a model; completion is not a write commit; unsupported remains non-success. Requested stops/handoffs may be objective candidates, subject to source review.

Disjoint constant-kind Schema branches enumerate only compatible actual targets plus unresolved. Opaque code can only be unclassified/unresolved. The compiler independently validates Schema and cross-record node coverage: a branch leaf cannot substitute for a condition, and omitted conditions are not guessed. Wrong source classification, dropped duties, negation loss and incorrect predicates remain possible even with valid Schema, so full-source, atom and decomposition review stays mandatory.

`representationReviewSupported` only reports exhaustive reviewer support, not independent truth. `admissionBlockers`, `parentIssues` and unresolved pointers retain unavailable capabilities and unsupported evidence. A faithfully represented safe stop can be supported while business eligibility remains blocked. All outcomes still grant no Runtime authority, including supported inactive candidates; no L1 write fallback or erased approval/script requirements.

### Probe costs, reproducibility and next step

The three calls totaled **23.38 seconds and 4,098/346 input/output tokens**, all normal stop, zero complete source reviews/public Skills/Runtime/provider/script/write execution. Each answer still used one requirement quoting the entire sentence, without separately mapping the non-invocation limitation: successful structure does not prove decomposition. Partial concurrent regression and fixed simple parents prevent causal timing comparisons with prior complex flows.

The linked report preserves outputs and file digests; CLI generation/compilation/assessment is offline and refuses existing outputs. The replay command above makes no model calls. Complete-checkpoint re-entry is idempotent; incomplete/drifted evidence fails closed. Ignored raw artifacts remain local, not included merely by committing the report.

Next remains C3h: freeze a separate fresh paired batch, measure whole-source fidelity, structural rejection, excessive unresolved selection, supported stops versus business coverage and failure-inclusive costs. Do not relabel old answers and count them as new success. Existing graph/contracts and source/target/review limits still apply without truncation. No UI claim, production expansion or C4–C6 unlock.

Final verification: **1439 tests + 81 subtests passed in 111.44 seconds**, plus Ruff/diff, canary replay and completed-checkpoint re-entry without calls, and unchanged previous responsibility/lean/full two-pass reports. Existing isolated execution fixtures remain in the regression suite; test duration is not Runtime performance. This turn's additions are uncommitted/unpushed.
