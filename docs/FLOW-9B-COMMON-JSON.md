# 9B 单模型转译诊断 / 9B-Only Translation Diagnostic

## 中文

### 当前选择与实验边界

2026-09-08 用户决定继续使用 `qwen3.5:9b`，GPT 对照暂缓，不再等待 API Key。执行此前已经冻结的 `flow-model-common-json/v1` 的 Ollama 臂；没有修改提示词、Schema、预算或失败答案。原始双臂报告保留 GPT `not_run`，顶层 `pending_model_calls` 只说明双臂对照不完整；查看 `arms.ollama.completed` 判断本轮 9B 是否完成。

本轮只含 **4 个已知开发流程、1 个工具、0 个公开 Skill**。固定树映射与完整链实验使用相同源文，但两个不同问题、两个独立分母；不是 8 个不同 Skill，也不是跨模型对照或泛化证明。源脚本保持惰性，没有 Runtime、业务工具、审批系统或网络配置执行。

模型制品固定为 `sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。采用 no-think、JSON 模式、16K 上下文、8192 输出预算；完整原 Schema 进入提示，再用未改本地门禁验证。其解码/输入/预算不同于历史受约束生成，不能把观察到的差异归因为单一因素。

### 固定树映射：已完成，未证明质量改善

保留上一轮实际树，只重新生成来源映射；无旧映射、诊断或修复答案输入。

| 用例 | 生成 Schema | 映射编译 | 原始问题与局部诊断 |
|---|---|---|---|
| 单次读取 | 通过 | 阻断 | 目的全部 false；两个读取句各另配六类宿主/解释要求，和节点引用叠加后超出六项上限。不能靠提高上限掩盖无来源分类 |
| 反向分支 | 通过 | 阻断 | 目的全部 false；条件节点有引用，但读取被配成输入检查/权限，业务分支被配成返回格式/解释限制 |
| 缺审批变更 | 不通过 | 阻断 | 停止节点的反驳理由 835 字符，超过 600；内容还把“表达一个未来候选”混同“证明已执行/已授权” |
| 缺前置脚本 | 通过 | 阻断 | 原文要求缺脚本时停止，却把 unsupported 停止判为矛盾；目的全部 false |

**4 次真实调用，3/4 生成 Schema 合格，0/4 映射编译合格。** 两个“候选已执行”错误不是多拒绝即可换来的安全收益。结构失败保留，未进入完整源声明审阅；上表是开发者逐项局部诊断，不是独立 Gold 或完整语义准确率。

总输入/输出 **19,706 / 7,329 token**，总 POST **518.54 秒**，请求 p50/p95 **120.08 / 157.11 秒**。全部正常 stop，没有截断、超时、修图或重试；理由超过字段上限不是整次生成达到 token 上限。数字包含失败成本，不包含模型预检、人工/助手分析或测试，也不是 Runtime 时延。

报告：`artifacts/translator-v2/model-mapping-4-20260908-ollama-report.json`，摘要 `sha256:4c5243e706735389200dd6e12681166f47d6dc3fc3b76c4f80730917ce64b765`。

### 完整链：已完成，四个流程结构合格但映射仍未闭合

从原文重新生成，**8 次真实调用，第一阶段 4/4 结构合格，映射生成 Schema 2/4 合格、映射编译 0/4**。单次读取与反向分支仍因所有目的标记为 false 等问题阻断；缺审批、缺脚本映射的理由分别为 670、672 字符，超过 600 字符，且仍混淆候选与执行。两个实验的所有八份映射都没有 `objective=true`，但不能自动翻转标记或据此认定只剩一个字段问题。

完整链总输入/输出 **36,858 / 8,343 token**，总 POST **619.41 秒**；每案例两步合计 p50/p95 为 **154.06 / 170.86 秒**。父报告里的请求级 p50/p95 为 **43.06 / 142.06 秒**，混合了短流程生成与长映射，不能冒充案例级或 Runtime 指标。第一步合计 122.96 秒，第二步 496.46 秒。两种实验总共 **12 次调用、72,236 token、1137.96 秒 POST**，失败成本全部保留，不合并质量分母。均正常 stop，无重试、超时或截断。

完整链报告：`artifacts/translator-v2/model-end-to-end-4-20260908-ollama-report.json`，摘要 `sha256:f1d27c36a75f20c2d7a9199bcb861a6866ccb32d2505239ea7de3643d9d67840`。

### 第一阶段源审查与诊断边界

对单次读取和反向分支的第一阶段原始输出完成完整清单审查，**38 个重叠声明：32 supported、6 insufficient_evidence**，两例均保持 blocked。动作顺序、分支极性和参数依赖在这两例中有来源支持；数据解释、权限与错误处理限制则没有在阶段一候选中完整交代。它们仍作为原文传给第二阶段，**不是原文件被删除或不可恢复的文本丢失**。阶段一未证明完整约束覆盖，不等于已证明业务图错误；不能要求一个只负责业务骨架的阶段独自证明整条转译链完成。

审阅者为同一开发助手，已看过样本，非独立 Gold。38 不是独立语义义务分母，不能算准确率。另两例第一阶段只有局部检查；全部第二阶段因编译失败未进入完整声明审阅。局部检查还发现审批例的首读 `source_id=s0001` 实际指向标题，而非读取指令，说明具体来源锚定也仍有问题，不能把“结构 4/4”解释成语义完全正确。

源审阅决策与校验输出在 `artifacts/translator-v2/model-end-to-end-4-20260908-flow-review-decisions.json` 和同级 `-flow-reviews/`，报告摘要 `sha256:4d37e43e3941c8d15fd0a735d2a829c9ea9997d316fb610b328e3c08893b8b33`。[可追溯摘要](benchmarks/flow-9b-common-json-summary.json)包含两批原文、树、原始映射、机械错误、成本及完整两例第一阶段审阅，不替代原始 HTTP/宿主环境收据。

### 结论与下一步

**不采用 common-JSON 作为 9B 的默认质量优化。** 这轮未证明质量提升，成本也没有给出采用理由。保留其未来跨模型对照用途，后续 9B 开发继续保留完整 Schema 约束，再改任务分工；不重复刷这两个冻结批。

已完成部分支持继续排查以下通用问题，但不能证明所有失败来自同一个根因：

1. **原文要求与宿主固有保障分开。** “读取”不等于源文又要求了所有六类检查；宿主必需保障可以保留，但不能伪称来自这句业务原文。
2. **候选表示、授权、实际执行结果分开。** 转译器验证未来步骤是否忠实表达源文，不要求候选已经在设备上成功执行；真正执行仍需 Runtime 的独立权限/证据门禁。
3. **减少目的的重复判断。** 第一阶段已有业务来源，第二阶段逐条重复真假分类仍可能丢失全部目的。只能通过明确候选范围与源审阅职责重新设计，不能自动把 false 改 true。
4. **保持可追溯失败。** Schema 放宽、提高上限或补引文不能自动修复语义；下一次修订必须另冻版本并统计首次输出，不改本批结果。

先实现“源义务候选提取（无宿主规则菜单）→业务骨架与工具绑定→完整源义务落实审查”的小探针。明确各阶段责任：原文保留是机械事实，义务提取是否完整、候选是否忠实和宿主是否可执行是三个不同问题。使用其他领域/措辞的开发反例验证，不嵌入这四例答案，不为省 token 丢弃标题中可能存在的真实要求。**该新分工尚未实现或获得新的 9B 成绩**；当前完成的是诊断验证与方案收口，C3h 质量门禁仍未通过。

缺少 GPT 对照意味着模型容量影响仍未知。下一步改进只能宣称在 9B 和指定开发输入上的效果，直到异质和新集合通过门禁；不以这四例解锁大规模 Runtime 测试。

### 本轮验证

模型批次结束后运行全量回归：**1543 tests + 81 subtests 通过，137.78 秒**。本轮未修改执行代码或冻结协议，仅新增评测制品、审阅和双语结果文档；回归数量和耗时不是语义准确率或 Runtime 性能。未提交或推送 Git。

两份新批报告及旧节点证据报告均离线重放一致；追溯摘要的 `summaryDigest` 校验通过。模型对照入口/传输/测试文件的定向 Ruff 与 `git diff --check` 通过。不是全仓 Ruff 无告警：上一阶段发现的 224 项既有告警未在本轮清理。

## English

### User choice and scope

The user chose to continue with `qwen3.5:9b` on 2026-09-08 and defer GPT. Run only the already frozen Ollama arm; no new API credentials are required. Original parent reports retain GPT as not_run and their top-level pending status describes the unfinished two-arm comparison. Check `arms.ollama.completed` for the active scope.

Four known development flows, one tool, zero public Skills. Fixed-tree mapping and fresh end-to-end probes have separate denominators; they are not eight unique Skills or evidence of cross-model/generalization performance. The same pinned Q4_K_M model uses JSON mode, no thinking, 16K context and an 8192 output ceiling with the full original schema in the prompt and unchanged local gates. Decoder/input/budget differ from historical constrained generation, preventing single-factor causal claims. No Runtime/business tools/scripts execute.

### Completed fixed-tree mapping probe

Four real calls: **3/4 generated-schema qualification, 0/4 compiled mappings**. Direct read invents six extra handling categories per read sentence, exceeds the projection budget and loses all objective flags. The branch maps conditions but also confuses operations with checks/interpretation and loses all objective flags. Missing approval produces an 835-character rationale above the 600-character cap and confuses a future candidate with proof of execution/authorization. Missing script wrongly contradicts the mandated unsupported stop. These are partial development observations, not complete source reviews or independent semantic accuracy. No failed candidate was repaired/rescored.

Input/output **19,706 / 7,329 tokens**; total POST **518.54 s**; per-request p50/p95 **120.08 / 157.11 s**. All calls ended normally, without retry, truncation or timeout. A field-length violation is not generation-token exhaustion. Costs include failures, exclude preflight/review/tests, and do not measure Runtime latency. The report path and digest appear above.

### Completed end-to-end probe and source review

Eight fresh calls produce **4/4 structurally qualified trees, 2/4 Schema-qualified mappings and 0/4 compiled mappings**. Direct/branch outputs lose all objective flags; approval/script reasons exceed the 600-character cap at 670/672 characters and still confuse candidates with execution. All eight mapping outputs across both modes have no true objective flag; flipping flags would not repair other semantic defects.

End-to-end input/output **36,858 / 8,343 tokens**, POST **619.41 s**. Per-case two-phase p50/p95 are **154.06 / 170.86 s**; mixed per-request p50/p95 are **43.06 / 142.06 s**, not Runtime performance. Flow/mapping totals are 122.96/496.46 s. Across both modes, twelve calls cost 72,236 tokens and 1137.96 s POST, without pooling quality denominators. All stop normally, no retry/truncation/timeout. The exact report path/digest are above.

Complete first-stage checklists for direct read and branch contain **38 overlapping claims: 32 supported, six insufficient**, both blocked. Their business order, polarity and argument dependencies are supported; complete source limitations are not yet accounted for in the stage-one candidate. Original text is still forwarded, not deleted. Stage-specific acceptance must distinguish skeleton fidelity from complete-chain constraint coverage. Same-developer review is not independent Gold or an accuracy denominator. The other two first-stage candidates and all mechanically blocked mappings have partial observations only; the approval read also cites a heading instead of the actual instruction. [Traceable summary](benchmarks/flow-9b-common-json-summary.json) contains raw proposals and the two complete first-stage reviews, not all HTTP/host receipts.

### Decision and next scope

Do **not** adopt common-JSON as the default 9B quality optimization: neither quality nor cost supports it in this probe. Preserve it for a future cross-model comparison; keep constrained Schema generation for subsequent 9B work. Next investigate source-only obligation extraction without host-rule menus, then business/tool binding and complete source-duty review. Distinguish text preservation, candidate fidelity and executable capabilities; do not demand full-chain proof from a skeleton-only stage. Retain title semantics where applicable and test different development wording/domains rather than embedding these four answers. This redesigned split is **not implemented or newly measured yet**. Do not flip flags, relax budgets or invent citations. GPT remains deferred and capacity attribution unknown; C3h and generalization gates remain open. See [Project Status](PROJECT-STATUS.md).

### Verification

After the model batches ended, the full suite passed **1543 tests + 81 subtests in 137.78 s**. This turn changed evidence, reviews and bilingual documentation, not execution code or frozen protocols. Regression counts/timing do not measure semantic accuracy or Runtime performance. No Git commit or push.

Both new reports and the historical node-evidence report replay identically offline; the tracked summary digest verifies. Targeted Ruff for the model-comparison entry/transport/tests and `git diff --check` pass. This is not a clean repository-wide Ruff claim: the 224 pre-existing warnings found during the previous stage remain outside this change.
