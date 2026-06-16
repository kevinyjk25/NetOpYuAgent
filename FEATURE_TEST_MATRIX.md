# NetOpYuAgent — 功能点 × 测试覆盖矩阵

> 用途:盘清全部功能点 + 暴露测试盲区(供后续补齐)。
> 生成日期:2026-06 · 基于 enterprise-dev 工作树代码事实(376 passed, 48 测试文件)。
> 更新(2026-06):补齐 6 个场景级盲区(S1/S2/S7/S8/S10/S12)+ webui backend 全栈集成测试。
> 新增 6 个测试文件(scenario_*/config_consistency/backend_integration)。
> backend SSE 包装层从"仅手动"升级为全栈 TestClient 行为测试(真实 executor+loop+fake LLM)。
> 剩余盲区:前端零自动化、若干辅助模块(hitl_core/chunk_queue 看门狗、memory auto-consolidate、journal_consumer 后台消费)。

## 梳理层级说明

采用**三层**(在你提的"场景/模块"两层上加一个测试形态维度,让盲区更精确):

- **L1 场景级(跨模块端到端)** — 用户/运维实际经历的能力,横跨多个模块。
- **L2 模块级(模块内功能)** — 每个模块自身的功能单元。
- **测试形态标注**(关键)— 这个项目有大量"源码 grep 断言"测试,能防回归但**测不出运行时行为**,属灰色覆盖。纯按"有/无 tests"会漏判,所以每个功能点标注:
  - 🟢 **行为** — 真跑代码路径(asyncio/stream/run/TestClient),验证实际行为
  - 🟡 **源码断言** — 仅 grep 源码字符串,防结构回归但不验行为
  - 🔵 **仅手动** — 只有双 agent 实跑验证过,无自动化测试
  - 🔴 **无** — 无任何测试

---

# 第一部分:L1 场景级(跨模块端到端)

> 每个场景标注:参与模块链 · 测试形态 · 对应测试文件 · 盲区备注

## S1 — 单 agent 诊断闭环(查询→检索→工具→HITL→答复)
- **模块链**: webui/backend → runtime/loop → retrieval → integrations/router → hitl_core → memory
- **测试**: 🟢 行为 — `test_scenario_diagnosis_loop.py`(工具→结果回注→合成,贯穿) + `test_run_wrapper.py`、`test_handle_tools_phase.py`
- **盲区**: ✅ 已补贯穿测试(2026-06)。剩:真实 retrieval 后端注入的端到端(当前用 in-proc registry)。

## S2 — 破坏性工具 HITL 审批(拦截→卡片→批准→执行)
- **模块链**: runtime/loop(watch-list gate)→ hitl_core/router → hitl_core/store → webui/routes_hitl
- **测试**: 🟢 行为 — `test_run_wrapper.py::test_watchlisted_tool_returns_stop_hitl_not_executed`、`test_scenario_diagnosis_loop.py`(非HITL续跑链)、`test_handle_tools_phase.py`、`test_production_safety.py`
- **盲区**: 同步审批的 approve→follow-up turn 端到端仍缺(`test_h2_async_*` 覆盖异步路径);非HITL多轮续跑链已补。

## S3 — 多目标破坏性批量 HITL([TOOL_BATCH]→N张卡片→批量批准)
- **模块链**: runtime/loop → profiles/network_batch_resolver → hitl_core/batch
- **测试**: 🟢 行为 — `test_batch_resolver.py`(Path A 去重 / Path B 设备prose / 单目标不批量)
- **盲区**: N张卡片在 webui 的批量渲染+批量批准的前端行为无测试(前端整体无自动化测试)。

## S4 — 跨 agent 委派(LAN→DC,[DELEGATE]→A2A→结果回注)
- **模块链**: runtime/loop → task/delegation → registry → A2A transport → 对端 hitl_core/executor
- **测试**: 🟢 行为 — `test_delegation_e2e.py`、`test_delegation_wiring.py`、`test_delegation_a2a_unwrap.py`、`test_delegation_gate.py`;🟡 源码 — `test_delegation_provenance.py`、`test_delegation_outbound_state.py`、`test_delegation_no_double_count.py`
- **盲区**: ✅ 已补 `test_delegation_dispatcher_behavior.py`(fake _stream_request 驱动真实 dispatch,验三种终态+结果累积+no-double-count)+ provenance 改为直接 import 真函数。源码断言降级为 deletion-guard 层。

## S5 — 跨 agent HITL 透传(模式B:对端审批→回调→originator自动续跑)
- **模块链**: task/delegation → A2A → 对端 hitl_core → task/inter/cross_agent_hitl → webui resume driver → /chat/resumptions
- **测试**: 🟢 行为 — `test_cross_agent_hitl.py`(unwrap翻译/关联链/双重resume守卫/buffer)、`test_inbound_delegation_completion.py`
- **盲区**: 🔵 真实双agent的"approve→DC inbound翻DONE→回调POST→LAN续跑流式答复"端到端只手动验过;前端 resumption poll 的 dedup-by-phase 无自动测试。

## S6 — 委派风暴防护(重复委派抑制 + park + 综合轮)
- **模块链**: runtime/loop(`_delegated_targets_this_request` + 综合轮)+ task/delegation(TaskStore gate)
- **测试**: 🟢 行为 — `test_delegation_gate.py`、`test_delegation_park_on_peer_hitl.py`;🟡 源码 — `test_delegation_repeat_guard.py`、`test_synthesis_no_delegate.py`
- **盲区**: repeat_guard / synthesis_no_delegate 是源码断言(综合轮的实际LLM行为靠🔵实跑)。

## S7 — 能力缺口诚实声明(C协议:[CAPABILITY_GAP]→记录→优雅停止→账本)
- **模块链**: llm_engine(prompt铁律)→ runtime/loop → runtime/directive_parser → runtime/skill_journal → webui/routes_skills(/evolution/gaps)
- **测试**: 🟢 行为 — `test_capability_gap.py`(解析/loop记录+停止+strip/stream事件/journal事件)
- **盲区**: 🟢 长链场景已补 `test_capability_gap.py::TestCapabilityGapLongChain`(做前缀→声明缺口的运行时安全网);结构性预检 B/A(P2-1)未实现。/evolution/gaps 聚合端点🔴仍无测试。

## S8 — 技能进化1:选择困难→学习→渐进自动
- **模块链**: skills/catalog(歧义判定)→ runtime/loop(偏好召回+stage)→ skills/skill_preference → memory → hitl(选择卡)
- **测试**: 🟢 行为 — `test_scenario_skill_ambiguity.py`(歧义→选择卡触发 + non_interactive抑制) + `test_skill_preference.py`(12,含真实后端回归)
- **盲区**: ✅ 选择卡门行为已补(2026-06,确认门逻辑正确,之前"实跑没出卡"是高置信LLM抢先自选,非门坏)。剩:HITL-vs-自选张力的产品决策(P1-6)。

## S9 — 技能进化P0/P1/P3:真实轨迹→固化/合并
- **模块链**: runtime/skill_journal(extract_trajectory)→ skills/capability_index(CSI聚类)→ skills/trajectory_miner(P1)/append_merger(P3)→ skills/evolver → webui(/evolution/sweep)
- **测试**: 🟢 行为 — `test_p1_p3_evolution.py`(6)、`test_capability_index.py`(9)、`test_skill_journal_flush.py`
- **盲区**: 🔵 backend的P1每5任务sweep钩子 + P3 append-marker钩子只在🔵实跑+手动endpoint验过,无backend集成测试;真实async embedder下的CSI质量🔵仅实跑(容器是stub)。

## S10 — 用户中断(Stop按钮→中止流→保留部分答复)
- **模块链**: webui前端(AbortController)→ backend SSE生成器 → runtime/loop(GeneratorExit清理)→ stop_policy(USER_CANCELLED)
- **测试**: 🟢 行为 — `test_scenario_user_interrupt.py`(consumer中止→SESSION_END abort→partial保留 的 loop 级契约);前端abort→backend USER_CANCELLED 整链仍🔵手动。
- **盲区**: webui SSE 包装层的 USER_CANCELLED 整链需全 app,仍手动验。

## S11 — 周期任务调度(schedule_create→tick→执行tool/query)
- **模块链**: scheduler/service ← runtime/loop(tool_invoker/query_runner注入)→ webui/routes_schedule
- **测试**: 🟢 行为 — `test_scheduler.py`(10,两种mode/间隔/取消/护栏/历史环)
- **盲区**: 注入的query_runner真实跑一个LLM周期任务的集成路径🔵未验(单测用fake runner)。

## S12 — 配置驱动启停(profile选择→工具/技能/HITL名单加载)
- **模块链**: config → profiles/base → ToolLoader/SkillLoader → policy_engine → main.py 装配
- **测试**: 🟢 行为 — `test_config_consistency.py`(自动发现全部dataclass子配置,校验yaml键无静默丢弃 + 值映射) + `test_profiles.py`(22)、`test_l0_l1_separation.py`
- **盲区**: ✅ config三处一致性已补(2026-06,通用自发现,新增配置节自动覆盖)。

---

# 第二部分:L2 模块级(模块内功能)

> 格式:模块 → 功能点 [测试形态] (测试文件)

## runtime/ — 核心循环
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| loop.py: 多轮 stream 主循环 | 🟢 行为 | test_run_wrapper, test_handle_tools_phase |
| loop.py: run()=stream()收集器 | 🟢 行为 | test_run_wrapper |
| loop.py: 工具调用+HITL门 | 🟢 行为 | test_handle_tools_phase |
| loop.py: 澄清门 | 🟢 行为 | test_clarification_gate |
| loop.py: DELEGATE处理 | 🟢 行为 | test_delegation_e2e |
| loop.py: CAPABILITY_GAP处理 | 🟢 行为 | test_capability_gap |
| loop_helpers.py: 纯helper(strip/is_complete/format) | 🟡 间接 | (经loop测试间接覆盖) **建议补单测** |
| loop_types.py: 公共类型 | 🟢 行为 | (经各测试使用) |
| loop_context.py: 每轮状态 | 🟡 间接 | (无直接测试) |
| directive_parser.py: TOOL/DELEGATE/SKILL_LOAD/CAPABILITY_GAP解析 | 🟢 行为 | test_delegate_directive, test_capability_gap |
| stop_policy.py: 停止判定+USER_CANCELLED+unresolved | 🟡 部分 | test_sprint3_pre **中止/no-progress停止建议补** |
| policy_engine.py: 工具分类+trust_mode | 🟢 行为 | test_production_safety |
| skill_journal.py: 选择/load/tool_call/completion/capability_gap记录 | 🟢 行为 | test_skill_journal_flush, test_capability_gap |
| context_budget.py / _v2.py: 预算装配(legacy+priority) | 🟢 行为 | test_context_budget_priority |
| metrics.py: prometheus计数 | 🟢 行为 | test_sprint3_pre(C1) |
| tracing.py: OTel启停 | 🟢 行为 | test_sprint3_pre |
| tool_cache.py: 工具结果缓存 | 🟢 行为 | test_production_safety |
| hooks.py: 生命周期钩子 | 🔴 | **🔴 无测试** |

## skills/ — 技能引擎 + 进化
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| catalog.py: 技能选择+歧义判定(strong/weak) | 🟢 行为 | test_capability_index间接 **歧义kind建议补直接测** |
| loader.py: profile感知技能加载 | 🟢 行为 | test_profiles |
| evolver.py: 反馈应用+新技能生成+merge+A/B bench | 🟢 行为 | test_skill_evolve_suggest_only |
| skill_format.py: Anthropic SKILL.md解析 | 🟢 行为 | test_anthropic_skill_standard |
| script_runner.py: script-as-tool(AST校验) | 🟢 行为 | test_anthropic_skill_standard |
| skill_preference.py: 偏好记录/召回/stage/demote | 🟢 行为 | test_skill_preference |
| capability_index.py(CSI): 相似度/路由/聚类/async embedder | 🟢 行为 | test_capability_index |
| trajectory_miner.py(P1): 重复轨迹检测+固化 | 🟢 行为 | test_p1_p3_evolution |
| append_merger.py(P3): 追加归属+merge | 🟢 行为 | test_p1_p3_evolution |
| journal_consumer.py: 后台dormant技能消费 | 🔴 | **🔴 无测试** |

## hitl_core/ — 人在环
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| router.py: 审批路由 | 🟢 行为 | test_production_safety, test_h2_async_hitl |
| store.py: 审批持久化(sqlite) | 🟢 行为 | test_sprint3_pre |
| batch.py: 多目标批量卡片 | 🟢 行为 | test_batch_resolver |
| audit.py: 审计日志 | 🟡 部分 | (经其他测试间接) **建议补** |
| coreference.py: 实体共指(neutral/device) | 🟢 行为 | test_l0_l1_separation |
| chunk_queue.py: 异步队列+空闲看门狗 | 🔴 | **🔴 无测试** |
| triggers.py / pipeline.py / schema.py | 🟡 间接 | (经路由测试间接) |
| H2 异步fire-and-forget | 🟢 行为 | test_h2_async_hitl, test_h2_async_resolution |

## task/ — 委派与跨agent
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| delegation.py: build_delegate_fn + TaskStore门 | 🟢 行为 | test_delegation_gate, test_delegation_wiring |
| delegation.py: outbound状态流转(RUNNING→COMPLETED/FAILED/AWAITING_PEER_HITL) | 🟢 行为 | test_delegation_dispatcher_behavior(驱动真实dispatch) |
| inter/cross_agent_hitl.py: 跨agent HITL桥 | 🟢 行为 | test_cross_agent_hitl |
| schemas.py: TaskDefinition/TaskState | 🟢 行为 | (经委派测试使用) |
| A2A event unwrap | 🟢 行为 | test_delegation_a2a_unwrap |
| 心跳防stall | 🟢 行为 | test_dispatcher_heartbeat, test_delegate_heartbeat_no_truncate |

## retrieval/ — 检索 (本轮全层补齐)
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| bm25.py: BM25排序 + CJK分词(字符+bigram) | 🟢 行为 | test_retrieval_layer |
| keyword.py: 子串匹配检索 | 🟢 行为 | test_retrieval_layer |
| embedding.py: 向量检索(async) | 🟢 行为 | test_retrieval_layer(fake embedder) |
| hybrid.py: 混合检索(weighted_sum + RRF) | 🟢 行为 | test_retrieval_layer |
| cache.py: 检索缓存(hit/miss) | 🟢 行为 | test_retrieval_layer |
| llm_judge.py: LLM重排 + 超时回退 | 🟢 行为 | test_retrieval_layer |
| 共享过滤契约(require/exclude_tags/min_score/top_k) | 🟢 行为 | test_retrieval_layer |
| factory.py: tools/skills→corpus + build_retriever降级 | 🟢 行为 | test_retrieval_layer |
| meta_tool.py: 注册/注销/prompt section | 🟢 行为 | test_retrieval_layer |

## registry/ — agent注册发现
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| registry.py: 注册/健康/round_robin | 🟢 行为 | test_registry_is_available |
| discovery.py: AgentCard发现 | 🟢 行为 | test_registry_is_available |
| router.py / store.py / schemas.py | 🟡 间接 | (经发现测试间接) |
| peer-aware prompt section | 🟢 行为 | test_peers_section |

## memory/ — 记忆
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| adapter.py: 事实记录/召回/consolidate | 🟢 行为 | test_skill_preference(经真实后端) |
| find_similar_facts / search_facts | 🟢 行为 | test_skill_preference |
| LLM事实抽取 | 🔵 手动 | **🔴 无直接测试** |
| auto-consolidate(每30轮) | 🔴 | **🔴 无测试** |

## integrations/ — LLM+工具路由
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| llm_engine.py: Ollama引擎+并发信号量 | 🟢 行为 | test_sprint3_pre(D1) |
| llm_engine.py: 系统prompt装配(含铁律) | 🟡 间接 | (经各测试) **prompt铁律建议源码断言锁定** |
| tool_router: MCP/OpenAPI/local路由 | 🟢 行为 | (经loop工具测试) |
| embedder: async embed + stub fallback | 🟢 行为 | test_capability_index间接 |
| 工具不存在的difflib纠错 | 🔴 | **🔴 无测试** |

## profiles/ — L1业务层
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| base.py: profile加载 | 🟢 行为 | test_profiles |
| network_batch_resolver.py | 🟢 行为 | test_batch_resolver |
| lan/dc 工具+技能定义 | 🟢 行为 | test_profiles, test_access_scenario |
| 跨profile工具隔离 | 🟢 行为 | test_profiles |

## scheduler/ — 调度
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| service.py: tick循环+job+历史环+护栏 | 🟢 行为 | test_scheduler |
| tool/query双模式 | 🟢 行为 | test_scheduler |

## webui/ — 接口层
| 功能点 | 形态 | 测试文件 |
|--------|------|----------|
| backend.py: /chat/stream SSE + P0/P1/P3钩子 | 🟢 行为 | test_backend_integration.py(全栈TestClient) |
| routes_hitl.py: HITL审批端点 | 🟢 行为 | test_production_safety |
| routes_skills.py: /evolution/sweep,space,gaps + journal | 🟡 部分 | gaps已覆盖(test_backend_integration);sweep/space仍手动 |
| routes_schedule.py: schedule端点 | 🟢 行为 | test_scheduler间接 |
| routes_system.py: wiring/peers/status | 🟢 行为 | test_peers_section |
| index.html: 前端(全部) | 🔵 手动 | **🔴 前端零自动化测试**(仅node --check语法) |

---

# 第三部分:测试盲区汇总(供补齐,按建议优先级)

## 高优先级(核心行为无验证 / 已踩坑)
1. ✅ **config三处一致性测试** — 已补 `test_config_consistency.py`(通用自发现,新节自动覆盖)。
2. ✅ **长链能力缺口(k<n)运行时安全网** — 已补 `test_capability_gap.py::TestCapabilityGapLongChain`;结构性预检 B/A 仍待 P2-1。
3. ✅ **webui backend集成测试** — 已补 `test_backend_integration.py`(全栈 create_webui_app + 真实 HitlExecutor + TestClient):happy path SSE 序列、破坏性工具 HITL 中断+不执行、capability_gap 经后端、/evolution/gaps 端点。**最大盲区已消除**。
4. ✅ **委派状态流转** — 已补 `test_delegation_dispatcher_behavior.py`(5,真实dispatch+三终态)+ provenance 直接import(6)。源码断言保留为 deletion-guard。
5. ✅ **用户中断** — 已补 `test_scenario_user_interrupt.py`(loop级契约:abort→SESSION_END→partial);webui SSE 整链仍手动。
6. ✅ **进化1歧义→选择卡触发** — 已补 `test_scenario_skill_ambiguity.py`(门逻辑确认正确)。HITL-vs-自选张力是产品决策(P1-6)。
7. ✅ **单agent诊断闭环贯穿** — 已补 `test_scenario_diagnosis_loop.py`(工具→结果回注→合成)。

## 中优先级(模块功能无直接测试)
7. ✅ **retrieval/ 整层** — 已补 `test_retrieval_layer.py`(24):bm25+CJK分词/keyword/embedding/hybrid双融合/cache/llm_judge超时回退/factory/meta_tool/过滤契约。
8. **🔴 /evolution/* 端点** — sweep/space/gaps 自动测试。
9. **🔴 hitl_core/chunk_queue + audit** — 异步队列看门狗、审计日志。
10. **🔴 工具not-found的difflib纠错** — 容易回归的纠错逻辑。

## 低优先级(辅助/间接已覆盖)
11. 🔴 runtime/hooks(生命周期钩子,无测试)。注:tool_cache + metrics 实际已有测试(已更正)。
12. 🔴 memory auto-consolidate, LLM事实抽取
13. 🔴 skills/journal_consumer 后台消费
14. 🟡 loop_helpers/loop_context 直接单测(目前间接覆盖)

## 结构性建议
- **源码断言测试(🟡)逐步升级为行为测试** — 委派三件套(outbound_state/no_double_count/provenance)已升级:behavior 验工作 + grep 防删除两层并存。剩 repeat_guard/synthesis_no_delegate/hitl_submit_scope/multi_agent_identity 仍是纯结构守卫(可按需配行为测试)。
- **前端零自动化** — index.html 仅 node --check 语法检查。若前端逻辑变重,考虑加 Playwright/最小 DOM 测试。
- **🔵 手动验证项应固化为集成测试** — 双agent实跑验过的(S5/S9 backend钩子)随环境流失,CI无法保障。
