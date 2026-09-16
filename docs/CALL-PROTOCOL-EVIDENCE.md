# 调用协议与证据交付 / Call Protocol and Evidence Delivery

## 中文

2026-09-16，用户确认继续[两轮收敛后的有限重构](CONVERGENCE-CLOSURE.md)。本包不重写Runtime、不增加通用语义自审、不修改旧失败/评分。范围限于区分编译参数与执行参数的接口，以及显式证据前置条件的交付控制；不保证模型理解两种语法。

### 两种参数，不共用含糊的object入口

- `submit.plan.reads[*].arguments`是编译表达式，例如`{"path":{"caller":"input#/exportPath"}}`，声明将来从哪里绑定值。
- `read.arguments`是实际值，例如`{"path":"/exports/inventory"}`。v4宿主把真实Read Contract输入Schema投影到DSH工具声明，不再仅公开`object`。多个工具按工具名绑定相应Schema；本地`$ref`在嵌入时重定位，原类型/范围/必填约束不弱化。
- 原宿主仍独立校验Schema、身份/资源范围和执行次数。错误给出`/arguments/path`等定位及原输入Schema，不回显被拒载荷、不取literal替换对象、不做类型强转或自动重试。
- DSH入口拒绝不代表Python宿主已执行一次尝试；进入宿主的拒绝仍占原有两次补读预算。没有增加宿主预算，编译和执行次数分别计量。

### 谁定义缺少什么证据

必需证据必须来自宿主的明确合同，而不是Runtime从Skill/工具自由文本中猜测。在现有`local-hybrid-host/v4`配置中可选加入：

```json
{
  "requiredReads": [
    {"tool": "read_export", "arguments": {"path": "/exports/inventory"}}
  ]
}
```

这是现有宿主配置的片段，不是完整配置。它要求该工具已在只读Contract中声明、参数符合真实Schema、目标已在原精确ACL内。最多8项，不能重复。它只限制交付，不自动授予权限或扩展资源；模型的prepare/submit不能设置或删除清单。

清单随宿主/会话摘要冻结，只有实际成功且参数精确匹配的回执才能满足。未尝试、被拒、结果未知以及仅在文本中提到目标，都不算取得证据。可用但未声明必需的资源不会被强制全部读取。没有清单时`requiredReadsComplete=null`，不是true。满足清单也只说明取得了指定观察，不证明任务语义完整、外部事实真实或查询有效。

最多8项是配置上限，不是额外读取额度；如果前段和原两次补读不足以满足清单，只能未完成结束，不承诺任意清单均能执行完成。

### 交付状态

| 情况 | 宿主行为 |
|---|---|
| 必需读取缺失，尚有原补读额度 | 返回`needs_required_evidence`与具体`nextReads`；不冻结、不生成、不接纳native文本 |
| 补齐显式必需读取 | 允许原Runtime一次生成或原生deliver；结果仍是`candidate_unverified` |
| 必需读取缺失且额度耗尽 | 冻结为`closed_incomplete`；hostResult为rejected，无获准文本、无模型生成 |
| Agent选择结束 | `draft(session_id, close_incomplete=true)`可在Runtime或原生fallback路径终止为未完成，不能跳过门禁获得结果 |
| 先前调用结果未知 | 保持未知及不可重试边界，不能用“未完成关闭”掩盖执行不确定性 |

上述硬性缺证据门禁只适用于宿主声明的清单。旧纯文本工具未声明它时，仍可能出现漏读；不能声称已自动解决自然语言的所有证据依赖。完整查询解析/类型校验也不在本包中。

### 本包固定验证计划

1. 定向机制检查：真实参数类型、单/多工具与引用Schema、宿主清单与ACL、两条交付路径、原预算/未知结果/证据摘要、主动未完成退出。
2. 实际DSH＋脚本化模型：故意提前deliver，在门禁后补读再交付；另一条主动结束未完成。两条已通过，分别8/7次协议请求、2/1次读取，**0次真实LLM调用**。
3. 真实9B只跑一次两个已知查询任务：沿用上一批`irql-draft`和`irql-missing-enricher`的完整Skill/任务/数据/原c1–c3＋format。**不添加requiredReads、不改预期、不换输入、不调参重跑。** 因此这一小批只检验新参数接口/提示说明是否改善模型调用；不能拿它声称显式证据清单已被LLM自动推导。

每例仍最多420秒。先冻结源码、模型摘要和原判据，模型结束后再跑全量pytest；单例失败保留。只读本地合成环境、两个已知任务但同一个IRQL Skill，不是新来源/泛化/原生对照/生产概率。两条机制探针与9B结果分开计量。

### 唯一9B回归结果：机制完成，语义阶段仍未通过

两例均正常结束、保留原任务、原样展示宿主回执，但**0/2完整任务、0/2自动读取前段编译成功**。两例都在两次编译提案中重复使用实参字符串而非caller绑定，被拒后转为native fallback。实际读取均为合法字符串，只读到索引；两份函数清单均未获取。第二例还有一次提交前读取，被生命周期门禁阻止，没有执行Provider；这不是参数错误或越权成功。

| 原任务 | 冻结判据 | 实际遗漏 | 端到端 |
|---|---:|---|---:|
| irql-draft | 3/4，完整任务失败 | 清单未读、无时间过滤，并把“未读”说成“不可访问”；计数/富化/top10子步骤局部正确 | 143.55秒 |
| irql-missing-enricher | 1/4，完整任务失败 | 清单未读、无时间窗口；只有查询轮廓，计数别名不一致，未基于证据解释缺少富化 | 207.13秒 |

4/8只是原判据逐项计数，不是校准准确率。第一例c2的“有用查询”按计数/富化/top-N局部要求计分，完整窗口缺失在c1失败；这个人工解释边界已写入审阅。审阅由开发AI完成，不是独立金标准。

13次真实模型调用全部来自DSH，Runtime生成0次；输入150,818／输出2,992 token，未知用量0次。p50/p95为175.34/203.95秒，n=2、路由变化且无同时原生对照，**不能推断提速或退化**。实际2次读取精确匹配ACL；观察到的写入、越权执行、重放均为0，但没有测试写路径，不能推断生产安全概率。原生答案仍是`candidate_unverified`，不是Runtime验证正确。

结论不是“Schema已区分，所以语义已解决”：新接口暴露真实实参类型后，同一Agent仍把它们用于编译语法，混淆方向与历史错误相反。这支持进一步隔离两种职责的设计假设，但本批不是用于证明因果的消融。显式requiredReads没有注入这两份旧输入，因此门禁不能被冒称为自动发现自然语言证据依赖；两个门禁探针也不能补算为两个语义成功。

### 证据、复核修正和复现

- [冻结输入/源码/模型](../artifacts/governed-session-20260916-call-protocol-9b/freeze.json)、[原始观测](../artifacts/governed-session-20260916-call-protocol-9b/summary/report.json)、[逐项审阅](../artifacts/governed-session-20260916-call-protocol-review/judgments.json)、[摘要绑定评估](../artifacts/governed-session-20260916-call-protocol-assessment/report.json)。66份制品和79份归档源码验证通过；运行期间源码/模型不变。
- 模型结束后发现旧fallback提示禁止所有draft，与新增主动未完成入口矛盾。已统一交付提示并显式返回`incompleteExit`；补齐非字符串工具名配置的拒绝类型，区分主动终止与缺证据终止状态。没有改任务、预期、模型参数或旧结果；**没有再次执行9B**。当前`hybrid_session.py`因此与模型归档不同，不能把当前提示称为已通过模型验证。
- [首次门禁探针](../artifacts/governed-session-20260916-evidence-contract-probe/summary/report.json)保留；[复核后最终探针](../artifacts/governed-session-20260916-evidence-contract-probe-final/summary/report.json)仍2/2通过，67份制品及76份源码与当前实现匹配。两批均实际DSH＋脚本化响应，真实LLM调用为0，不混入9B指标。
- [可入Git机器摘要](benchmarks/call-protocol-evidence-summary.json)记录指标、哈希及最终QA；原始artifacts被Git忽略，上述原始链接仅在保留制品的本机可用，远端需单独备份制品。

最终85项定向检查、**3,295项全量＋81子测试（234.47秒）**、119个变更/新增项目Python文件Ruff、3项文档测试和diff检查通过。第一次全量测试期间修改源码，触发了预期的实现漂移保护（1项失败，其他3,293项通过）；已固定源码完整重跑，过程失误保留在摘要。模型与pytest没有重叠；机制回归不是语义重试。

复现脚本探针可使用新输出目录：

```bash
.venv/bin/python -m evaluation.hybrid_terminal_probe /tmp/ensuredskill-evidence-probe-new --task-bound --required-evidence
```

输出目录必须不存在。它只检查协议，不能替代真实模型测试。不要覆盖已有9B目录；本包已完成唯一模型回归，不因结果失败自动追加一轮。

### 收敛路径与停止条件

本包完成了类型接口、明确证据门禁和未完成退出，**Gate 1仍开放**，不进入扩样或大Runtime A/B。下一包应先调整职责分配，而非继续添加提醒文字：

1. 评估独立编译上下文：编译器只面向符号绑定、原Skill及输入Schema；运行Agent只面向具体工具调用和观察。沿用原编译/权限校验器与L1 fallback，不自动把实参转换成绑定，不新增写权。这是下一设计建议，尚未实现或证明有效。
2. 把“显式合同完整性”与“自然语言发现完整性”分开。业务宿主可声明必需证据，模型发现的额外依赖仍是候选。缺少声明时不得输出“证据完备”；工件检查应按支持范围公开，不能用格式正确替代查询正确。
3. 开始前固定输入、判据、一次模型验证预算。分别记录编译提案错误数/成功前段数、清单实际获取率、完整任务通过数、fallback率、全部模型成本及p50/p95；安全门禁要求观测到的越权/重放为0。若协议错误减少而清单获取/完整任务仍为0，则停止宣称语义收敛，定位剩余证据规划问题，不通过扩增测试数量掩盖它。

这些是下一包诊断指标，不替代原Gate 1出口。之后仍须≥6新Skill/4仓库/3领域/12任务小批，再≥3不同cohort及正式泛化门禁，最后才同DSH原生/auto Runtime对照。没有降低旧出口，没有提交、推送或重启UI。

## English

Approved bounded successor to the two-round stop, September16. This package separates compiler binding expressions from concrete read arguments, not a Runtime rewrite or semantic-review expansion. The v4 host exports actual Read Contract input schemas to DSH, including tool-specific branches and rebased local refs. Independent host checks remain; safe field diagnostics expose no rejected payload or coercion. Host rejections still consume the original two-read budget; frontend schema rejections are separately counted.

Optional operator-owned requiredReads declares at most eight unique concrete reads already allowed by the existing contracts/ACL. It is frozen with the host/session and cannot be overridden by the Agent. Only matching successful receipts satisfy it; mentions, rejected attempts and unknown outcomes do not. Missing declarations mean requiredReadsComplete=null, not completeness. No prose inference or requirement that every available resource be read exists; satisfying the list does not prove task meaning, source truth or query validity.

Missing obligations block normal draft/deliver before generation/admission while the existing read budget remains. Exhaustion or explicit draft(close_incomplete=true) freezes an incomplete terminal with no admitted text or model call. This safe exit is available on native fallback too, but cannot launder unknown execution or bypass admission. Legacy profiles remain unchanged.

Fixed evidence plan: targeted mechanism tests; two actual-DSH scripted cases (early delivery→required read→candidate, or incomplete exit), already passing with8/7 requests and2/1 reads,zero real LLM calls; then one qwen3.5:9b run of the two exact previous IRQL tasks,source,data and criteria. No requiredReads is added to those real-model inputs, so they measure the parameter interface rather than manually supplied evidence dependencies.420s/task,no retry/tuning,new directory and archived source/model identity. Full pytest follows model completion. This is one known Skill/two synthetic tasks,not unseen generalization or paired performance. No UI restart,commit or push.

### Results and limitations

The sole real9B batch finished:2/2 normal faithful host deliveries,0/2 complete tasks and0/2 admitted compiled prefixes. Both cases submitted concrete strings where symbolic caller bindings were required, repeated the mistake once, then fell back. Each read only its index and omitted the inventory. One premature read was blocked by lifecycle state before provider execution; it is not a parameter error or unsafe effect.

Frozen criterion counts are3/4 and1/4,not calibrated accuracy. The first task's aggregation/enrichment/top10 component is credited under c2 while missing inventory/window fails c1; that interpretation is disclosed in the digest-bound developer-AI review,not independent gold. The second provides an incomplete query outline with a mismatched count alias and no evidence-grounded enrichment limitation. Both remain unverified native L1 answers.

All13 real calls are DSH calls,zero Runtime generations;150,818 input/2,992 output tokens,zero unknown usage. Latencies143.55/207.13s;p50/p95=175.34/203.95s. Two exact authorized provider reads,zero observed writes/unauthorized execution/replay;write paths were not exercised. One known Skill/two tasks,changed routes,no concurrent native control: no causal performance, generalization or production-probability claim. Explicit prerequisites were deliberately not added to these inputs. Scripted gate successes do not count as semantic successes.

All66 run artifacts and79 archived source files verify; implementation/model remained fixed during the batch. Post-run review fixed contradictory legacy fallback guidance to advertise the optional incomplete exit, normalized malformed operator tool-name rejection, and distinguished explicit-stop status. No real-model rerun or regrading followed. The current session source differs from that model archive; updated guidance is not model-validated. The final scripted probe remains2/2,with67 artifacts and76 archived/current sources matching. Original probes remain intact. See the linked reports above and [portable metrics/QA](benchmarks/call-protocol-evidence-summary.json). Git-ignored raw artifacts need separate backup for remote readers.

Final QA:85 targeted tests,3,295 full tests plus81 subtests in234.47s,Ruff on119 changed/new project Python files,3 documentation tests and diff checks pass. A source edit during the first full-suite attempt triggered the intentional implementation-drift guard (one failure); the complete suite was rerun with unchanged code. This process mistake is retained in the summary. Model runs and pytest did not overlap; mechanism regression is not a semantic retry.

### Next design boundary

Mechanism package complete,Gate1 open. A separate compiler context owning symbolic bindings and an execution Agent seeing only concrete calls is the next design hypothesis,not an implemented or causally proven fix. Retain original compiler/access checks,fallback and zero new write authority. Explicit operator evidence obligations remain separate from model-discovered candidate dependencies; unknown completeness must stay unknown. Eight configured obligations do not grant more than the original read budget.

Predeclare compilation errors/prefix yield,actual required-inventory acquisition,complete task passes,fallback share,total model cost and latency. If protocol errors decrease but inventory acquisition/task passes stay zero,stop claiming semantic convergence and diagnose evidence planning instead of increasing test counts. One frozen model run per approved package;mechanism checks do not substitute for original Gate1 criteria. Then follow the existing fresh-source small batch,cross-cohort generalization,and finally native/auto Runtime comparison. No broad rewrite,self-review stack,lowered gate,commit,push or UI restart.
