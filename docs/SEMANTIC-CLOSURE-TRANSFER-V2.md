# 语义闭环第二批迁移 / Second Semantic Closure Transfer

## 中文

2026-09-11，本批**全部结束，未通过语义闭环出口**。先冻结实现、Prompt、Schema 与本地 `qwen3.5:9b`，再从固定公开快照中按摘要顺序选择 6 个 Skill／6 个仓库，保存 12 个任务的合成观察及全部判据后才启动模型。旧 [v1 失败](SEMANTIC-CLOSURE-TRANSFER-V1.md)不改分。

### 结果与失败

12 任务中 **0 个实质任务完整完成、3 个正确边界响应、8 个部分完成、1 个失败**；36 条预设判据为 **24 满足／12 不满足**。图状态为 11 完成、1 阻断，不能当作 11 个任务成功。没有观察到越权工具调用、脚本执行或配置写入，但存在错误操作建议、接口路径和语义遗漏。[逐例机器摘要](benchmarks/semantic-closure-transfer-v2.json)。

| 任务 | 实际审阅结论 |
|---|---|
| CAPA 状态 | 监测责任人漏掉；审阅发现但未定位，未修复 |
| FDA 信息不足 | 正确保留分类信息缺口；不是监管认证 |
| IRQL 查询草稿 | 真正补读了函数清单，却只交付查询计划，没有完整 KQL |
| IRQL 不可用 | 正确保留前置条件，不假装已部署或执行 |
| Mesh 事件 | 北区指标正确，南区精度丢失；建议重读已读导出 |
| Mesh 无 trace | 保留实际流量，但把采样解释写成已确立原因 |
| 配对计划 | 用户 ID/Hub 映射遗漏、404 条件不完整；审阅位置字段错误导致停止 |
| 缺少配对凭据 | 知道需要 token，但追问未引导安全供给渠道 |
| 网关计划 | Anthropic 路径漏后缀，混用 Python/JS 表达，未观测的验证措辞 |
| 网关验证失败 | 正确识别 401，但后续换 key/重试建议漏掉授权条件 |
| Phoenix trace | 成功补读并区分组件状态，但漏掉 token 统计 |
| Phoenix 导出缺口 | 正确保留 3 次请求、导出未知及有条件验证建议 |

保留 **67 次真实 9B 调用，342,581 输入／52,968 输出 token**。单调用 p50/p95 为 **27.402／101.683 秒**，不是整任务时延、SLO 或因果 A/B 提升；与 v1 的来源/任务不同，不能相减声称算法进退。两次审阅对若干错误草稿都给了支持意见，说明审阅器本身仍不可靠。

两类来源共四个任务只收到入口第一页：CAPA 为 1,931/11,780 字符，Phoenix 为 2,042/6,431；其余四个 Skill 的入口完整。零调用诊断证实完整 CAPA/Phoenix 入口都符合既有预算，根因是所有参考文档超限后错误退回第一页。源码/摘要完整保留不等于模型看过全部原文。

下一版只修通用机制：入口优先预算、补读状态与候选草稿连续性、单一整稿范围的遗漏编辑、无位置支持意见保守降级。旧批次不能重跑后改称通过；已知失败回归不算新来源，阶段仍未完成。

### 样本与边界

| 原始 Skill | 仓库 | 两个限定任务 |
|---|---|---|
| fda-consultant-specialist | alirezarezvani/claude-skills | 虚构 CAPA 状态／缺失分类信息；不作医疗法律认证 |
| azure-kusto-irql | microsoft/skills | 基于函数清单写查询草稿／IRQL 未部署边界 |
| service-mesh-observability | wshobson/agents | 错误率与时延分析／无 trace 不等于无流量 |
| mem-setup | thedotmack/claude-mem | 配对配置计划／缺少安全凭据的追问 |
| caveman-setup | juliusbrussee/caveman | 双调用点网关计划／真实夹具中 401 验证失败 |
| phoenix-tracing | github/awesome-copilot | 分析多类 span／缺失导出证据 |

选取规则是合格快照、入口 250–12,000 字符、排除本语义修复已知入口、每仓库一个、按固定盐＋入口摘要排序前六；32 个入口符合长度/来源条件。不是按已观察成功率挑选。它们是**对本语义修复循环的新来源**，不保证从未用于早期转译开发或模型预训练。六个领域标签由开发者分类，不代表独立领域认证。

只读工具仍是带有限资源 ACL 的合成 `read_export`，不是真实厂商接口。源材料包含脚本、引用、流程及写操作要求，但均为惰性材料；本轮不执行源脚本、MCP 生产请求、付费验证或配置更改。因此结果最多证明这些限定分析任务的本地混合处理，不证明完整原生 Skill 工作流成功或正式泛化。

构图、实际读取、LLM 草稿、受控补读、来源审查、定位修订及终审分别留档。默认仅定位负面问题触发编辑，未定位遗漏仍未解决；不改稿不是修复，模型支持分数不是语义批准。长来源的分页面供给情况也需要审阅，不能把声明了原始来源摘要说成模型阅读了全部内容。

### 固定证据

- 冻结摘要：`c4a5c6a4fcbf9f449ba17d5cff460a4b380b66abb18ba1ee4f65bc24cd786fe2`。
- 模型文件摘要：`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。
- 选取策略摘要：`63b126d8453685f07207bd9425ef30c0aa7bc0b8f85c59214e4dbdf63a7af545`。
- 选取结果摘要：`5a1466891bc2fbbc0e3f1b5a38d2b936f2b09f8cb5aa2e3466187fd67f6621a8`。
- 完整验收摘要：`f06dacd6a5e4415ef13f427f2c783db854a5992123d4824e89a3dbb8136d1e69`。
- 调用/归档审计摘要：`6d339291f7f995f2d58d99b485debac68875495bd2e65f3b3971dc594937a580`。
- 本机制品：`artifacts/translator-v2/semantic-closure-transfer-20260911-v2`；被 Git 忽略，不自动随代码发布或备份。

第一次选取器读取旧报告字段出错发生在新来源选择与模型调用前；修正元数据读取路径，已封存的选取规则不变，零模型调用失败另存 `selection-failure`。固定代码已知 v8 回归中交接与 README 原稿完全保留，零编辑＋两次终审，12,313 输入／4,349 输出 token；这是防止多余编辑退步，不是新增语义修复或未见样本。该回归摘要为 `a99626d5d75673dbce387a2e0fb6cec9d7f0397143c0dca82da5227fe12f2ac2`。

最终必须逐项核对实际回答及工具轨迹，按[固定出口](SEMANTIC-CLOSURE-EXIT.md)判断；安全停止、Schema 合格、测试通过都不能抵消关键语义失败。不扩大正式 A/B，也不报告生产成功概率。[复现入口](SEMANTIC-CLOSURE-RUNBOOK.md)。

## English

This batch finished and failed the semantic-closure exit. Implementation, prompts, schemas and local qwen3.5:9b were frozen before mechanically choosing six Skills from six repositories and predeclaring twelve tasks. The first failed transfer is not regraded.

Results: zero fulfilled substantive tasks, three correct boundary responses, eight partial tasks and one failure; 24/36 predeclared criteria met, 12 unmet. Eleven completed graphs are not eleven successful tasks. No unauthorized tool call, source script or write was observed, but semantic omissions, wrong endpoint composition and unsafe procedural guidance remain. The [portable per-case record](benchmarks/semantic-closure-transfer-v2.json) preserves explanations and full-report digest references.

All 67 real calls remain: 342,581 input / 52,968 output tokens, call p50/p95 27.402/101.683 seconds. These are not task latency, an SLO or a causal comparison with the different v1 sample. Two review passes endorsed several flawed answers; model support is not an Oracle.

CAPA received 1,931/11,780 entry characters and Phoenix 2,042/6,431, affecting four tasks; the other four entries were fully supplied. Both complete entries fit the unchanged budget. The source selector incorrectly fell back to the first entry page when all references were too large. Source digest retention did not ensure complete source delivery.

Next known-only repairs target entry-first budgeting, completed-read indexing plus candidate continuity, missing-content assignment to a sole whole-draft owner, and conservative withholding of unlocated positive coverage opinions. Frozen outcomes stay unchanged; these repairs cannot themselves satisfy new-source acceptance or complete this stage.

Selection uses accepted pinned snapshots, 250–12,000-character entries, exclusion of known semantic-closure sources, one entry per repository and a fixed salted digest order. Thirty-two entries qualified. These sources are new to this repair loop, not necessarily unseen in earlier translator development or model pretraining. Developer-assigned domain labels are not independent certifications.

The table above covers fictional quality workflows, IRQL drafts, mesh symptoms, memory pairing, gateway plans and Phoenix traces. All business observations are disclosed synthetic read-only exports behind finite ACLs. Referenced scripts, production writes and billable requests are inert and unauthorized. These scoped analyses cannot certify whole native Skill workflows, broad generalization or production probabilities.

Authoring, admission, actual reads, drafts, bounded continuation, source review, located repair and final review remain distinct evidence. Only located negative findings trigger editing; unlocated omissions remain open and no-op is not semantic approval. Actual supplied source pages must be reported: a retained source digest is not proof the model read the entire Skill.

The freeze, model, selection-policy and selection digests appear above. Full artifacts remain locally ignored, not automatically published or backed up. An initial metadata-access error occurred before selection/model calls and is preserved without changing the sealed rule. Known fixed-code v8 retains already-repaired handoff/README drafts with zero editors and two final reviews (12,313 input / 4,349 output tokens); this prevents gratuitous regression but provides no unseen or new-repair credit.

Final acceptance requires reviewing actual content and traces against predeclared criteria. Structural conformance, safe stops and engineering tests do not excuse critical semantic failures. No scaled A/B or production claim is unlocked.
