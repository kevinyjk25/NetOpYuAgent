# IT 运维监控智能体平台

> **IT Ops Multi-Agent Platform** — 基于 A2A 协议的多智能体运维编排系统，集成 HITL 人机协同、四层记忆体系、Runtime Loop 执行引擎与 WebUI 控制台。

---

## 目录

1. [项目简介](#1-项目简介)
2. [核心架构](#2-核心架构)
3. [模块说明](#3-模块说明)
4. [快速开始](#4-快速开始)
5. [WebUI 使用指南](#5-webui-使用指南)
6. [P0 工具结果缓存演示](#6-p0-工具结果缓存演示)
7. [P1/P2 特性说明](#7-p1p2-特性说明)
8. [Mock 工具与技能目录](#8-mock-工具与技能目录)
9. [测试说明](#9-测试说明)
10. [环境变量](#10-环境变量)
11. [项目结构](#11-项目结构)
12. [开发路线图](#12-开发路线图)

---

## 1. 项目简介

本平台是一个面向 IT 运维场景的**多智能体协作系统**，核心设计哲学来自两个方向的结合：

- **A2A 协议优先**：所有 Agent 间通信基于 Google A2A Protocol v0.3.0，支持跨语言、跨框架互操作
- **Loop 极简、脚手架极强**：借鉴 Claude Code 设计哲学，把主执行循环做薄（Runtime Loop），把上下文治理、工具结果缓存、停止机制、技能按需加载做厚

### 核心能力

| 能力 | 描述 |
|------|------|
| **双路由执行** | 简单查询走 Runtime Loop（无 LangGraph 开销），复杂/破坏性操作走 HITL 图 + TaskPlanner DAG |
| **HITL 人工审批** | LangGraph interrupt 机制，多渠道通知（Slack/PagerDuty/SSE/WebSocket），完整审计链 |
| **四层记忆体系** | L1 实时 → L2 Redis 短期 → L3 ChromaDB 中期 → L4 PostgreSQL 长期，MMR 去重检索 |
| **上下文预算管理** | 每轮 prompt 按优先级分配 token（确认事实 > 工作集 > 记忆 > 工具结果 > 环境），软上限 3200 tokens |
| **P0 工具结果缓存** | 大型工具输出（日志、时序数据、流量记录）自动外部化存储，prompt 仅携带引用 ID，支持按需分页读取 |
| **技能渐进披露** | Level 1（摘要，每轮注入）+ Level 2（详情，按需加载），降低无关 token 消耗 |
| **分叉委托** | 子 Agent 可继承父 Agent 的确认事实和工作集（forked 模式），避免上下文重复传递 |
| **停止策略** | 显式停止条件：最大回合数、工具调用上限、token 预算、低进展检测、低置信度升级 |
| **动态 Agent 注册** | Registry 模块支持运行时注册/注销 Agent，健康检查 + 负载均衡（轮询/随机/最少连接） |

---

## 2. 核心架构

```
外部调用方（RouterAgent / WebUI / Webhook）
          │ A2A JSON-RPC / REST / WebSocket
          ▼
┌─────────────────────────────────────────────────────────┐
│                       API 网关层                         │
│   /api/v1/a2a/*   /hitl/*   /registry/*   /webui/*      │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         执行路由（v3）            │
          │                                  │
          │  classify(query)                 │
          │    SIMPLE → Runtime Loop        │
          │    COMPLEX → HITL Graph + DAG   │
          └──────────────────────────────────┘
               │                    │
    ┌──────────▼────────┐  ┌────────▼────────────────┐
    │  Runtime Loop     │  │  HITL Graph (LangGraph)  │
    │  context_budget   │  │  TaskPlanner DAG         │
    │  stop_policy      │  │  A2ADispatcher           │
    │  skill_catalog    │  │  HitlTaskBridge          │
    └──────────┬────────┘  └────────────────────────--┘
               │
    ┌──────────▼────────────────────────────────────┐
    │         Memory 模块（四层存储）                │
    │  L1 实时 → L2 Redis → L3 Chroma → L4 Postgres │
    └───────────────────────────────────────────────┘
    
    
    Incoming query
    │
    ├── Phase 0 ─────────────────────────────────────────────────
    │   FTS5 search past sessions → LLM summarize → inject to prompt
    │   UserProfile → format_prompt_section() → inject as hidden context
    │
    ├── Phase 1 ─────────────────────────────────────────────────
    │   classify(query) → SIMPLE or COMPLEX
    │
    ├── Phase 2A (SIMPLE) ───────────────────────────────────────
    │   AgentRuntimeLoop.stream()
    │     ├── SkillCatalog Level 1 summary injected each turn
    │     ├── LLM call (Ollama/OpenAI/Anthropic)
    │     │   └── [TOOL:name] parsed from response
    │     ├── ToolRouter.dispatch()
    │     │   ├── MCPClient → NetOps MCP server
    │     │   ├── OpenAPIClient → NMS REST API
    │     │   └── Local mock tools
    │     ├── ToolResultStore → large results cached, [STORED:ref] in prompt
    │     ├── ContextBudgetManager → assemble within token cap
    │     └── StopPolicy → check 6 conditions, stop/continue/escalate
    │
    ├── Phase 2B (COMPLEX) ──────────────────────────────────────
    │   LangGraph HITL graph
    │     ├── intent_classifier (LLM-backed)
    │     ├── evaluate_triggers → first match interrupts
    │     ├── interrupt() → serialize to MemorySaver
    │     ├── 5-channel notification fanout
    │     ├── operator decision → resume
    │     └── _verify_action_result → SkillEvolver.after_task()
    │
    └── Phase 3 (all paths) ─────────────────────────────────────
        Concurrent post-turn hooks:
        ├── FTS5SessionStore.write_turn() → SQLite + trigger → turns_fts
        ├── MemoryCurator.after_turn() → LLM curate → MemoryRouter ENTITY
        ├── UserModelEngine.after_turn() → trait inference → profile update
        └── MemoryRouter.ingest_turn() → L1/L2/L3/L4 fanout
```

---

## 3. 模块说明

### 3.1 `runtime/` — Agent 运行时

| 文件 | 职责 |
|------|------|
| `loop.py` | `AgentRuntimeLoop`：薄主循环，包含 classify/run/stream，集成 P1/P2 特性 |
| `context_budget.py` | `ContextBudgetManager`：per-turn 上下文预算管理；`ToolResultStore`：P0 大结果外部化 |
| `stop_policy.py` | `StopPolicy`：六维停止条件评估；`LoopState`：跨回合状态追踪 |

### 3.2 `hitl/` — 人机协同

| 文件 | 职责 |
|------|------|
| `a2a_integration.py` | `ITOpsHitlAgentExecutor`：v3 版本，双路由（简单/复杂）+ 后置验证钩子 |
| `graph.py` | LangGraph 状态图：6 节点（分类→风险评估→规划→中断→执行→格式化） |
| `triggers.py` | 四类触发器（破坏性操作/告警级别/置信度/歧义意图），优先级链式评估 |
| `decision.py` | `HitlDecisionRouter`：5 种决策处理（approve/reject/edit/escalate/timeout） |
| `review.py` | 5 个通知渠道并发扇出；WebSocket 通道（外部 Agent 系统集成） |
| `audit.py` | 7 种审计事件类型，支持内存/PostgreSQL 后端 |

### 3.3 `memory/` — 四层记忆

| 层级 | 后端 | 检索方式 | TTL |
|------|------|---------|-----|
| L1 实时层 | 进程内列表 | 全量返回 | 请求周期 |
| L2 短期层 | Redis 有序集合 | 时间倒序 | 24 小时 |
| L3 中期层 | ChromaDB 向量索引 | 余弦相似 + 时间衰减 | 30 天 |
| L4 长期层 | PostgreSQL 全文 | pg_trgm 相似度 | 永久 |

### 3.4 `task/` — 任务编排

- `intra/planner.py`：`TaskPlanner`（目标→DAG）+ `TaskScheduler`（并发=5，依赖解析）+ `TaskExecutor`
- `inter/coordinator.py`：`A2ATaskDispatcher`（SSE 委托）+ `MultiRoundCoordinator`（多轮上下文）
- `inter/hitl_bridge.py`：`HitlTaskBridge`（挂起/恢复钩子）

### 3.5 `registry/` — Agent 注册与发现

- 动态抓取 AgentCard（4 条 well-known 路径）
- 健康检查（60s 间隔）+ AgentCard 刷新（300s 间隔）
- 负载均衡：`round_robin`（默认）/ `random` / `least_loaded`

### 3.6 `skills/` — 技能目录（P1）

- `SkillCatalogService`：Level 1 摘要（每轮注入）+ Level 2 详情（按需加载）
- `DEFAULT_SKILL_DEFINITIONS`：9 个预置技能（含 3 个大结果技能 + 1 个 HITL 技能）

### 3.7 `tools/` — Mock 工具

| 工具 | 输出大小 | 触发缓存 |
|------|---------|--------|
| `syslog_search` | ~6,000+ chars | ✅ 是 |
| `prometheus_query` | ~5,000+ chars | ✅ 是 |
| `netflow_dump` | ~10,000+ chars | ✅ 是 |
| `dns_lookup` | < 300 chars | 否，内联返回 |
| `device_info` | < 500 chars | 否，内联返回 |
| `alert_summary` | < 300 chars | 否，内联返回 |
| `service_health` | < 300 chars | 否，内联返回 |
| `read_stored_result` | 按需分页 | — P0 读取工具 |

### 3.8 `webui/` — 浏览器控制台

- `backend.py`：FastAPI 子应用，挂载于 `/webui`
- `static/index.html`：终端风格单页面，无外部 JS 框架依赖

---

## 4. 快速开始

### 4.1 环境要求

- Python 3.9+（推荐 3.12）
- 可选：Redis、PostgreSQL、ChromaDB（不配置时自动降级为内存 stub）

### 4.2 安装依赖

```bash
cd it-ops-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.3 启动服务

```bash
# 开发模式（无外部依赖，全部降级为内存 stub）
uvicorn main:app --reload --port 8000

# 或直接运行
python main.py
```

启动后访问：
- **WebUI 控制台**：http://localhost:8000/webui/
- **A2A 接口**：http://localhost:8000/api/v1/a2a/
- **HITL 接口**：http://localhost:8000/hitl/
- **注册中心**：http://localhost:8000/registry/
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

### 4.4 配置外部依赖（可选）

```bash
export REDIS_URL=redis://localhost:6379
export POSTGRES_DSN=postgresql://user:pass@localhost:5432/itops
export CHROMA_PATH=./chroma_db
export A2A_BASE_URL=http://localhost:8000/api/v1/a2a
```

---

## 5. WebUI 使用指南

打开 http://localhost:8000/webui/ 后界面分为三栏：

### 左栏 — Skills & Tools

- **Skills 面板**：列出所有注册技能，点击技能名可快速填入查询框
  - 绿点 = low risk，黄点 = medium，红点 = high/critical
  - `HITL` 标签 = 执行前需人工审批
- **Quick Tools 面板**：直接调用 Mock 工具，大结果会自动出现在右栏 Cache 面板

### 中栏 — 对话区

| 控件 | 说明 |
|------|------|
| 查询框 | Enter 发送，Shift+Enter 换行，自动高度 |
| `session` | 留空自动生成会话 ID；填写指定 ID 可延续历史 |
| `mode` | `stream`=SSE 流式返回；`sync`=等待完整响应 |
| `delegation` | `fresh`=独立子 Agent；`forked`=继承父 Agent 上下文（P1） |
| `[STORED:...]` 芯片 | 点击可在右栏加载并分页浏览缓存的大结果（P0 演示） |

### 右栏 — Cache / HITL / Stats

- **Cache 面板**：显示所有已缓存的工具结果，支持逐页翻阅
- **HITL 面板**：实时显示待审批的中断，可直接 Approve / Reject
- **Stats 面板**：系统状态快照（技能数、缓存数、Agent 数、内存状态）

---

## 6. P0 工具结果缓存演示

### 问题背景

运维场景中，工具调用（syslog 查询/NetFlow 导出/Prometheus 时序）可能返回数万字节数据。若直接注入 prompt，单次调用就会耗尽上下文窗口。

### 解决方案

`ToolResultStore` + `ContextBudgetManager` 实现两步处理：

1. 工具输出 > 4,000 chars → 存入 `ToolResultStore`，prompt 仅注入：
   ```
   [STORED:syslog_search:a3f9c12b] Preview: Apr 10 09:12:01 ap-01 hostapd…
   ```
2. 模型需要详情时调用 `read_stored_result` 工具分页读取：
   ```
   [TOOL:read_stored_result] {"ref_id": "a3f9c12b", "offset": 0, "length": 2000}
   ```

### WebUI 操作步骤

```
步骤 1：在查询框输入 "search syslogs for errors on ap-01"
         → Runtime Loop 调用 syslog_search（300 行，~6000 chars）
         → 自动存储，返回 [STORED:syslog_search:abc123]
         → 右栏 Cache 面板出现新条目

步骤 2：点击消息中的 [STORED:syslog_search:abc123] 芯片
         → 右栏 Cache 面板显示第一页（2000 chars）
         → 显示：Total: 6XXX chars | Has more: True | Next offset: 2000

步骤 3：点击 "Next ▶" 翻页
         → 加载 offset=2000 的内容

步骤 4：也可直接 API 调用：
         GET http://localhost:8000/webui/tools/result/abc123?offset=0&length=2000
```

### API 直接测试

```bash
# 1. 触发大结果工具
curl -X POST http://localhost:8000/webui/tools/syslog_search \
  -H "Content-Type: application/json" \
  -d '{"args": {"host": "ap-*", "keyword": "error", "lines": 300}}'

# 响应示例：
# {
#   "tool": "syslog_search",
#   "is_stored": true,
#   "ref_id": "a3f9c12b",
#   "raw_length": 6241,
#   "retrieve_url": "/webui/tools/result/a3f9c12b"
# }

# 2. 读取第一页
curl "http://localhost:8000/webui/tools/result/a3f9c12b?offset=0&length=2000"

# 3. 读取下一页
curl "http://localhost:8000/webui/tools/result/a3f9c12b?offset=2000&length=2000"
```

---

## 7. P1/P2 特性说明

### P1：技能渐进披露

每轮 prompt 只注入技能摘要（Level 1）：
```
[AVAILABLE SKILLS]
  syslog_search     [   low]  Search syslog entries across network devices
  prometheus_query  [   low]  Query Prometheus metrics for network devices
  restart_service   [  high] ⚠ HITL  Trigger a rolling restart of a production service
  ...
```

模型发出 `[SKILL_LOAD:syslog_search]` 时，Runtime Loop 按需注入完整参数说明、示例和约束（Level 2），比全量注入节省约 60% tokens。

### P1：分叉委托

```python
# Fresh 模式（默认）：子 Agent 从空白开始
result = await loop.run(query="...", delegation_mode=DelegationMode.FRESH)

# Forked 模式：继承父 Agent 的确认事实和工作集
result = await loop.run(
    query="...",
    delegation_mode=DelegationMode.FORKED,
    parent_state=parent_loop_state,   # 传入父 Agent 的 LoopState
)
# fork_context_policy 控制继承范围：
#   FACTS_ONLY   → 只继承 confirmed_facts（默认）
#   WORKING_SET  → confirmed_facts + working_set
#   FULL         → 全量继承
```

### P1：前置/后置验证

```python
# 前置验证（execute 前自动触发）：
pre = await loop.pre_verify(query, confirmed_facts, env_context)
if not pre.passed:
    # → 自动返回 STOP_HITL，不进入执行阶段

# 后置验证（工具调用后自动触发）：
post = await loop.post_verify(tool_name, result, confirmed_facts)
if not post.passed:
    state.unresolved_points.append(f"Post-verify: {post.reason}")
```

**内置规则（可替换为真实策略引擎）：**
- 破坏性操作 → 直接失败，强制 HITL
- `change_window: False` + 变更操作 → 失败，不允许执行
- `allow_destructive: False` + 生产环境 → 失败

### P1：Confirmed Facts & Working Set

```python
# Confirmed Facts：结构化已确认事实（注入 prompt 最高优先级）
state.record_new_fact("payments-service 在 prod 环境，当前健康")
state.record_new_fact("DNS 解析正常，排除 DNS 故障")

# Working Set：当前聚焦对象（DeviceRef 列表）
working_set = [
    DeviceRef(id="ap-01", label="AP-01 at Site-A"),
    DeviceRef(id="sw-core-01", label="Core Switch"),
]
```

### P2：模型分层提示

```python
decision = loop.classify(query)
print(decision.model_tier)  # "fast_model" 或 "full_model"

# fast_model：check / status / dns / list 等简单查询
# full_model：复杂分析、P0/P1 事件、破坏性操作
# → 生产环境可据此选择不同模型（如 haiku vs sonnet）
```

---

## 8. Mock 工具与技能目录

### 大结果工具（触发 P0 缓存）

```bash
# 查询 syslog（300 行，~6KB）
POST /webui/tools/syslog_search
{"args": {"host": "ap-01", "keyword": "error", "lines": 300}}

# Prometheus 时序查询（8设备×60分钟，~5KB）
POST /webui/tools/prometheus_query
{"args": {"metric": "up", "job": "network_devices", "range_minutes": 60}}

# NetFlow 流量记录（500条，~10KB）
POST /webui/tools/netflow_dump
{"args": {"site": "site-a", "flows": 500}}
```

### 小结果工具（内联返回）

```bash
POST /webui/tools/dns_lookup      {"args": {"hostname": "payments.internal"}}
POST /webui/tools/device_info     {"args": {"device_id": "ap-01"}}
POST /webui/tools/alert_summary   {"args": {"severity": "P1"}}
POST /webui/tools/service_health  {"args": {"service": "payments-service"}}
```

### 缓存读取工具

```bash
# 读取已缓存结果（offset 分页）
POST /webui/tools/read_stored_result
{"args": {"ref_id": "a3f9c12b", "offset": 0, "length": 2000}}
```

---

## 9. 测试说明

### 测试结构

```
tests/
├── test_a2a.py          # A2A 模块 (~30 用例)
├── test_hitl.py         # HITL 模块 (~25 用例)
├── test_memory_task.py  # Memory + Task 模块 (~45 用例)
├── test_registry.py     # Registry 模块 (~30 用例)
├── test_runtime.py      # Runtime Loop 基础 (~50 用例)
└── test_p0_p1_p2.py     # P0/P1/P2 特性 + WebUI (~60 用例)
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 仅运行 P0/P1/P2 特性测试
pytest tests/test_p0_p1_p2.py -v

# 仅运行 Runtime Loop 测试
pytest tests/test_runtime.py -v

# 快速冒烟测试（跳过慢速集成测试）
pytest tests/ -v -k "not integration"

# 带覆盖率
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

### 关键测试场景

| 场景 | 测试类 | 测试方法 |
|------|------|--------|
| P0：大结果存储并分页读取 | `TestP0ToolResultStore` | `test_ref_id_extraction_and_read` |
| P0：syslog 触发缓存 | `TestP0ToolResultStore` | `test_syslog_tool_produces_large_output` |
| P0：WebUI API 分页 | `TestWebUIBackend` | `test_p0_retrieve_stored_result_first_page` |
| P1：技能摘要注入 | `TestP1SkillCatalog` | `test_format_summary_contains_all_skill_ids` |
| P1：技能详情按需加载 | `TestP1SkillCatalog` | `test_load_detail_returns_string` |
| P1：分叉继承父事实 | `TestP1ForkedDelegation` | `test_forked_run_inherits_facts` |
| P1：前置验证阻断破坏性操作 | `TestP1Verification` | `test_pre_verify_fails_destructive_query` |
| P1：前置验证变更窗口检查 | `TestP1Verification` | `test_pre_verify_fails_closed_change_window` |
| P2：模型分层提示 | `TestP2ModelTiering` | `test_simple_dns_suggests_fast_model` |
| 路由：简单查询不走 HITL | `TestRoutingIntegration` | `test_simple_query_uses_runtime_loop` |
| 路由：复杂查询走 HITL | `TestRoutingIntegration` | `test_complex_query_uses_hitl_graph` |

---

## 10. 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `A2A_BASE_URL` | `http://localhost:8000/api/v1/a2a` | 本 Agent 对外 A2A 基础 URL |
| `REDIS_URL` | 空（内存降级） | Redis 连接串 |
| `POSTGRES_DSN` | 空（跳过持久化） | PostgreSQL 连接串 |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB 本地路径 |
| `AGENT_URLS` | 空 | 逗号分隔的对端 Agent URL |
| `REGISTRY_LB` | `round_robin` | 负载均衡策略 |
| `REGISTRY_HEALTH_INTERVAL` | `60` | 健康检查间隔（秒） |
| `HITL_CONFIDENCE_THRESHOLD` | `0.75` | 触发 HITL 的置信度阈值 |
| `HITL_MAX_AUTO_HOST_COUNT` | `5` | 自动执行的最大主机数 |
| `HITL_SLACK_WEBHOOK_URL` | 空 | Slack Incoming Webhook URL |
| `HITL_PAGERDUTY_ROUTING_KEY` | 空 | PagerDuty Events API 路由键 |
| `HOST` | `0.0.0.0` | 服务监听地址（python main.py 模式） |
| `PORT` | `8000` | 服务监听端口 |
| `RELOAD` | `false` | 是否开启热重载 |

---

## 11. 项目结构

```
it-ops-agent/
├── main.py                    # FastAPI 入口，统一装配
├── requirements.txt           # 全量依赖
├── pytest.ini
│
├── a2a/                       # A2A 协议模块（9 个文件）
│   ├── schemas.py             # A2A Pydantic 数据模型
│   ├── agent_card.py          # AgentCard 构建
│   ├── agent_executor.py      # 执行器基类 + 策略处理器链
│   ├── event_queue.py         # Sealed 异步事件队列
│   ├── request_handler.py     # JSON-RPC 方法路由
│   ├── server.py              # FastAPI 子应用工厂
│   ├── push_notifications.py  # 推送通知（指数退避）
│   └── task_store.py          # 任务状态内存存储
│
├── hitl/                      # HITL 模块（9 个文件）
│   ├── schemas.py             # HitlPayload/Decision/AuditRecord 等
│   ├── triggers.py            # 四类触发器 + HitlConfig
│   ├── graph.py               # LangGraph 状态图（6 节点）
│   ├── review.py              # 5 渠道通知 + WebSocket 管理器
│   ├── decision.py            # HitlDecisionRouter（5 种决策）
│   ├── audit.py               # 审计服务（内存/Postgres 双后端）
│   ├── router.py              # FastAPI 路由（含 /ws WebSocket）
│   └── a2a_integration.py     # v3 执行器（双路由 + 后置验证）
│
├── memory/                    # Memory 模块（7 个文件）
│   ├── schemas.py             # MemoryRecord/RetrievalQuery 等
│   ├── router.py              # MemoryRouter 门面（统一读写入口）
│   ├── consolidation.py       # 整合工作器（摘要 + 实体抽取）
│   ├── stores/backends.py     # 四层 Store 实现
│   └── pipelines/             # ingestion.py + retrieval.py
│
├── task/                      # Task 模块（8 个文件）
│   ├── schemas.py             # TaskDefinition/SessionRecord 等
│   ├── intra/store.py         # TaskStore + RetryManager
│   ├── intra/planner.py       # TaskPlanner + TaskScheduler + TaskExecutor
│   ├── inter/session.py       # SessionManager（Redis，TTL=8h）
│   ├── inter/coordinator.py   # A2ATaskDispatcher + MultiRoundCoordinator
│   └── inter/hitl_bridge.py   # HitlTaskBridge
│
├── registry/                  # Registry 模块（6 个文件）
│   ├── schemas.py             # AgentEntry/AgentSkill/ResolutionResult
│   ├── store.py               # InMemory + Redis 双存储
│   ├── discovery.py           # AgentDiscovery（AgentCard 抓取）
│   ├── registry.py            # AgentRegistry（注册/解析/健康检查）
│   └── router.py              # FastAPI 路由
│
├── runtime/                   # Runtime 模块（4 个文件，新增）
│   ├── context_budget.py      # ContextBudgetManager + ToolResultStore（P0）
│   ├── stop_policy.py         # StopPolicy + LoopState（P0）
│   └── loop.py                # AgentRuntimeLoop v2（P1/P2 集成）
│
├── skills/                    # 技能目录模块（2 个文件，新增）
│   └── catalog.py             # SkillCatalogService + DEFAULT_SKILL_DEFINITIONS（P1）
│
├── tools/                     # Mock 工具（2 个文件，新增）
│   └── mock_tools.py          # 8 个 Mock 工具（含 3 个大结果工具）
│
├── webui/                     # WebUI 模块（3 个文件，新增）
│   ├── backend.py             # FastAPI 子应用（17 个端点）
│   └── static/index.html      # 终端风格单页面控制台
│
└── tests/                     # 测试套件（6 个文件）
    ├── test_a2a.py
    ├── test_hitl.py
    ├── test_memory_task.py
    ├── test_registry.py
    ├── test_runtime.py
    └── test_p0_p1_p2.py       # P0/P1/P2 + WebUI（新增，~60 用例）
```

---

## 12. 开发路线图

### 已实现

- [x] A2A Protocol v0.3.0 完整实现（SSE 流式 + WebSocket HITL）
- [x] HITL 五层架构（触发 → 图中断 → 通知扇出 → 决策路由 → 审计）
- [x] 四层 Memory（L1-L4，MMR 检索，整合工作器）
- [x] Task 模块（DAG 调度 + 跨 Agent 委托 + 多轮会话）
- [x] Registry 动态服务发现（健康检查 + LB）
- [x] Runtime Loop v2（薄主循环 + 双路由 + P1/P2 特性）
- [x] P0：ToolResultStore 大结果外部化 + 分页 API
- [x] P0：ContextBudgetManager per-turn 预算管理
- [x] P0：StopPolicy 六维停止条件
- [x] P1：SkillCatalogService 技能渐进披露
- [x] P1：分叉委托（fresh / forked + fork_context_policy）
- [x] P1：Confirmed Facts & Working Set 一等公民
- [x] P1：前置/后置验证钩子
- [x] P2：模型分层提示（fast_model / full_model）
- [x] WebUI 控制台（终端风格，SSE/同步双模式，P0 可视化）

### 待实现（生产化必须项）

- [ ] **JWT/OIDC 鉴权**：`/hitl/decisions/{id}` 必须限制为审批人员角色（最高优先级）
- [ ] **真实 LLM Chain**：替换 `hitl/graph.py`、`runtime/loop.py` 中的关键词 stub（ChatOpenAI + 结构化输出）
- [ ] **真实工具接入**：替换 `task/intra/planner.py:_execute_task` stub（kubectl / Prometheus HTTP API / OpsGenie）
- [ ] **真实 Embedding**：替换 `memory/pipelines/ingestion.py:_EmbedderStub`（OpenAIEmbeddings）
- [ ] **OpenTelemetry 链路追踪**：所有跨模块调用添加 Span，TraceID 通过 session_id 透传

### 待实现（P2 扩展项）

- [ ] Prompt cache 友好策略（稳定前缀优先，变化内容后置）
- [ ] 轻量验证 Agent（独立小模型执行后置健康检查）
- [ ] Agent 版本管理（AgentCard 含 version 字段，金丝雀发布）
- [ ] MCP 协议集成（Model Context Protocol，Agent 访问外部工具的标准接口）

---

*文档版本：v2.0 | 最后更新：2025-01-01 | 内部文件*