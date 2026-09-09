# 统一节点完整双阶段验证 / Canonical-Node Fresh Paired Pilot

## 中文

### 结论：局部表示改善，完整映射仍未合格

2026-09-07，另冻新版批次，以 `qwen3.5:9b` 从原始源文重新生成流程，再生成统一节点映射，**8 次真实调用，流程 4/4 合格、映射 0/4 合格**。四例均在完整源审查前被阻断，完整语义审查数为 0，不伪造语义准确率或审查支持比例。

本轮局部观察是积极的：四份原始映射均符合生成 Schema，没有发现类型—目标非法组合；合法 `needs_l1` 在反向分支案中已实际映射为 handoff。上一版原始回答曾出现 7 个不兼容组合，新版未观察到。但**目的选择冲突和节点来源遗漏仍使完整资格为 0/4**。不能把局部 Schema 合规当作完整转译成功，也不能把所有阻断都归为 Runtime 不安全。

范围仍是 **4 个已知开发流程 / 1 工具 / 0 公开 Skill**。父流程由新调用生成，恰好与上版树内容相同；并非复用旧答案。固定种子、一次尝试，未重试、改答案或修改冻结协议。模型批次 Runtime/业务工具/脚本/写执行均为 0。下列诊断来自同一开发助手的事后检查，不是独立 Gold 或完整源语义审查。

### 结果分层与定位

| 用例 | 首个编译阻断 | 全量原始字段检查发现的其他问题 |
|---|---|---|
| direct-read | objective 含数据背景 clause-0006 | 漏完成节点 `/steps/1`；整句“读取并完成”只有读取映射 |
| inverted-branch | objective 含 clause-0007 等限制/背景 | handoff 和假分支读取/完成均有引用，但仍漏条件节点 `/steps/1`；7 项 unresolved |
| missing-approval-write | objective 中 clause-0005 被模型自己归为 interpretation_limit | 另含背景 clause-0012；已有停止节点映射恢复，但漏读取节点 `/steps/0`；“读取输入”错分 input_shape；10 项 unresolved |
| unavailable-script-prerequisite | objective 含背景 clause-0012 | 15 项全 unresolved，连已有停止节点 `/steps/0` 也未映射；真实缺脚本问题仍保留 |

事后检查分别计算 Schema 合规、目的类型一致性和所有节点覆盖，不因编译器报了第一个错误就忽略后续问题。四例共 **8 个目的/条目类型冲突、4 个节点来源缺口**。52 个源片段对应 54 项要求，引文均精确唯一匹配，只有反向分支的两个片段生成多项要求，仍使用整句引文；**这些机械计数不能替代分解完整性审查**。

unresolved 共 32 项，上版为 31 项，不能据这些非独立、数量变化的条目计算错误率或语义退步幅度。审批/脚本的真实缺能力与“已有读取/停止未被映射”必须区分；没有任何写操作被执行或错误提交。

### 为什么局部修复没有闭环

本轮说明必要的类型/目标关系可以被 Schema 收口，但仍存在 **跨条目结构设计问题**：

1. **重复生成业务目的。** 第一阶段已有目的来源候选，第二阶段却独立再次选择 objective；它与第二阶段自己的类型判断也可能冲突。父来源还包括标题，不能简单继承为已证实正确的意图。
2. **节点覆盖是事后汇总，而非生成结构中的必填项。** 模型只需给每个源片段一项处理，就可能遗漏完成、条件或读取节点。局部类型合法并不保证整图覆盖。
3. **第二阶段在重新猜已有节点的职责。** 第一阶段已有 read 节点，后续却将“读取输入”分类成输入校验；这是二次推断引入的不一致，不应靠新增更多类别解决。
4. **源要求覆盖仍需真正的语义判断。** 复合动作、约束、否定、前置依赖和 unresolved 的范围，不能通过“引用精确”或“模型填满 JSON”证明。

下一步仍在 C3h，但**不再仅追加类型/提示词**：优先使每个实际节点具有必填来源条目，由编译器固定已知节点角色；将目的关联到已有证据条目，减少独立重选来源编号造成的不一致。第一阶段源链接只作候选，标题、极性、参数、前置和完整原意仍需审查。未知/缺能力步骤必须显式保留，不允许自动补业务节点、删限制或原生 L1 写 fallback。先做离线反例，再另冻新批，不改本批。

### 成本与测试

| 指标 | 上版职责双阶段 | 本版统一节点双阶段 |
|---|---:|---:|
| 映射资格 | 0/4 | 0/4 |
| 第二阶段输入 / 输出 token | 10,544 / 4,283 | 9,457 / 4,427 |
| 两步输入 / 输出 token | 21,560 / 5,264 | 20,473 / 5,408 |
| 两步总 token | 26,824 | 25,881（减少 3.5%） |
| 第二阶段 POST 时间 | 224.21 秒 | 248.10 秒 |
| 两步 POST 时间 | 317.90 秒 | 341.16 秒（观察增加 7.3%） |

第一步 93.06 秒、11,016 / 981 token。所有模型回答正常 stop，未达到输出上限。失败和等待计入 POST，预检/审查/回归不计；本轮部分全量回归并行，协议/载荷及机器负载不同，**时延变化不是因果性能结论**。本轮既没有总体可用性提升，也不能以 token 略减宣称整体性能改善。

新增 5 项批次回归，与统一节点映射回归共 **147 项定向通过**；全量 **1444 tests + 81 subtests 通过（214.93 秒）**。Ruff/diff、新报告重放、完整检查点重入无新调用、旧职责/精简/完整双阶段三份报告与终态探针重放均一致。回归耗时不是 Runtime 指标。

### 证据与复现

- [独立冻结批次入口](../evaluation/flow_canonical_pilot.py)及[回归](../tests/test_flow_canonical_pilot.py)
- [带完整原始选择和诊断的摘要](benchmarks/flow-canonical-c3h-summary.json)
- [统一节点设计和单终态探针](FLOW-CANONICAL-MAPPING.md)

本地根目录：`artifacts/translator-v2/flow-canonical-4-20260907`；完整报告：同级 `flow-canonical-4-20260907-report.json`。

Manifest：`sha256:ee93c6ff83c1d6c332b9d47f49a8ee83dd5d2ecf22ffc089db1a54c7b877dab8`。

报告：`sha256:e856360c9fa44bd7b4bc143dac8175cf991123182a260813df71bdc5ac2d5607`。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evaluation.flow_canonical_pilot report \
  artifacts/translator-v2/flow-canonical-4-20260907 \
  --output /tmp/flow-canonical-paired-replay.json
```

仅离线重放，输出须不存在，不调用模型。原始制品仍被 Git 忽略，摘要不等于完整可移植证据包。本轮新增入口/报告未提交或推送；C4–C6 和规模化 Runtime 评测继续保持门禁。

## English

### Local representation gains, no complete mapping qualification

Eight fresh qwen3.5:9b calls on four known development flows produced **4/4 qualified flows but 0/4 qualified mappings**, with zero complete source reviews and no semantic accuracy claimed. All raw mappings satisfy the generation Schema, no incompatible kind/target pairs were observed, and needs_l1 is now actually mapped as handoff. The preceding responsibility batch contained seven incompatible pairs. This local improvement does not establish end-to-end usability: objectives still conflict with declared roles and nodes still lack evidence.

One tool, zero public Skills, no retries/answer repair or batch Runtime/provider/script/write execution. Newly generated trees happened to equal previous trees; old answers were not substituted. Same-developer post-hoc diagnostics are not independent Gold or complete semantic reviews.

### Findings and next direction

Direct read includes background in its objective and omits completion evidence. Branching maps handoff and false-path read/completion but omits the condition node and includes limitations/background in objectives. Approval maps the available stop but loses the read, reclassifies reading as input-shape validation, and treats its business clause as interpretation. Script marks all fifteen requirements unresolved, including the existing stop, and includes background in its objective. Genuine missing approval/script capabilities remain separate from annotation gaps.

There are eight objective/type conflicts and four missing-node citations. All 54 quotes over 52 clauses match uniquely, but only two clauses contain multiple requirements; this is not decomposition completeness. Thirty-two unresolved entries versus thirty-one previously are not independent samples or a semantic error-rate comparison.

Next remains C3h: reduce repeated intent/role inference rather than adding more labels/prompts. Give each actual node a mandatory evidence slot with compiler-owned role, and associate objectives with existing evidence instead of independently reselecting IDs. Parent source links, including headings, remain untrusted candidates; retain full-source, polarity, parameters, prerequisites and unsupported-step review. No guessed business nodes, removed constraints or native-L1 write fallback. Freeze a separate future batch, never repair/rescore this one as new success.

### Costs and reproducibility

Whole-chain input/output: **20,473/5,408 tokens**, total **25,881**, 3.5% below the preceding 26,824. Mapping input/output: 9,457/4,427. Whole POST **341.16 s** versus 317.90 s (+7.3% observed); mapping 248.10 s versus 224.21 s. First pass 93.06 s, 11,016/981 tokens. All calls stopped normally; failure/wait cost included, preflight/review/tests excluded. Different protocol/payload/load and partial concurrent regression prevent causal timing conclusions. Neither usability nor overall performance improvement is established.

Five new runner regressions, 147 focused tests, **1444 tests + 81 subtests passed in 214.93 s**, plus Ruff/diff, new-report replay/checkpoint re-entry without extra calls and unchanged prior responsibility/lean/full two-pass reports and terminal canary. The linked summary preserves raw choices, original blockers and separate diagnostics. The offline command refuses existing outputs and makes no calls; ignored raw artifacts remain local. This turn's additions are uncommitted/unpushed; C4–C6 stay gated.
