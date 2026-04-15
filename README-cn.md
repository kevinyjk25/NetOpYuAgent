# IT 运维多智能体平台

> **IT Ops Multi-Agent Platform** — 基于 A2A 协议的多智能体运维编排系统，集成 HITL 人机协同、Hermes 学习循环、双路由执行引擎与终端风格 WebUI 控制台。

---

## 目录

1. [项目简介](#1-项目简介)
2. [核心架构](#2-核心架构)
3. [模块说明](#3-模块说明)
4. [快速开始](#4-快速开始)
5. [接入真实 LLM（Ollama）](#5-接入真实-llmollama)
6. [WebUI 控制台使用指南](#6-webui-控制台使用指南)
7. [HITL 人工审批流程](#7-hitl-人工审批流程)
8. [Hermes 学习循环](#8-hermes-学习循环)
9. [P0 工具结果缓存](#9-p0-工具结果缓存)
10. [P1 / P2 特性说明](#10-p1--p2-特性说明)
11. [Mock 工具与技能目录](#11-mock-工具与技能目录)
12. [环境变量](#12-环境变量)
13. [项目结构](#13-项目结构)
14. [开发路线图](#14-开发路线图)

---

## 1. 项目简介

本平台是一个面向 IT 运维场景的多智能体协作系统，核心设计理念来自两个方向：

**A2A 协议优先** — 所有 Agent 间通信基于 Google A2A Protocol（HTTP 传输）。Agent 通过内置注册中心实现服务发现、负载均衡和健康检查。任何语言、任何框架实现的 Agent 都可以加入。

**薄循环、厚脚手架** — 借鉴 Claude Code 设计哲学：主执行循环（`AgentRuntimeLoop`）刻意做薄，所有复杂性放在外围脚手架中——上下文预算管理、工具结果缓存、停止策略、技能加载、FTS5 跨会话召回、以及 Hermes 后置学习流水线。

### 核心能力

| 能力 | 描述 |
|---|---|
| **双路由执行** | 简单查询走 Runtime Loop（无 LangGraph 开销）；复杂/破坏性操作走 HITL LangGraph，需人工审批 |
| **HITL 人工审批** | LangGraph `interrupt()` 暂停执行，浏览器弹出审批卡片，决策后图继续运行 |
| **Hermes 学习循环** | 每轮结束后自动运行：FTS5 记忆写入、MemoryCurator 事实提取、UserModelEngine 用户建模、SkillEvolver 技能自动创建 |
| **FTS5 跨会话召回** | SQLite FTS5 存储所有历史轮次；新查询自动检索相似上下文并注入 prompt |
| **P0 工具结果缓存** | 大型工具输出（syslog、Prometheus、NetFlow）外部存储，prompt 仅携带 `[STORED:id]` 引用 |
| **技能目录** | L1 摘要每轮注入，L2 详情按需加载——避免无关技能消耗 token |
| **上下文预算** | 按优先级分配 token：确认事实 > 工作集 > 记忆 > 工具结果 > 环境，软上限 3 200 tokens |
| **停止策略** | 六维评估：最大回合数、最大工具调用次数、token 预算、低进展、低置信度、显式停止信号 |
| **Agent 注册中心** | 运行时动态注册/注销，健康检查，轮询/随机/最少连接负载均衡 |
| **MCP + OpenAPI** | 可插拔工具后端：MCP 服务器（JSON 配置）和任意 OpenAPI 3.0 规范；两者均支持 Mock 模式 |

---

## 2. 核心架构

```
外部调用方（RouterAgent / WebUI / Webhook / curl）
           │  A2A JSON-RPC over HTTP-SSE / REST
           ▼
┌──────────────────────────────────────────────────────────┐
│                       API 网关层                          │
│   /api/v1/a2a/*   /hitl/*   /registry/*   /webui/*       │
└─────────────────────────┬────────────────────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │            执行路由               │
         │                                   │
         │  classify(query)                  │
         │    SIMPLE  → Runtime Loop         │
         │    COMPLEX → HITL Graph           │
         └───────────────────────────────────┘
              │                    │
   ┌──────────▼─────────┐  ┌──────▼──────────────────────┐
   │   Runtime Loop      │  │  HITL Graph（LangGraph）     │
   │   context_budget    │  │    intent_classifier         │
   │   stop_policy       │  │    risk_assessor             │
   │   skill_catalog     │  │    planner                   │
   │   tool_cache        │  │    hitl_interrupt_node ←─────│─── 运维人员
   │   FTS5 召回         │  │    executor                  │
   └──────────┬──────────┘  │    result_formatter          │
              │             └──────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────────────┐
   │              Hermes 后置学习流水线                    │
   │   FTS5SessionStore → MemoryCurator → UserModelEngine  │
   │                    → SkillEvolver                     │
   └──────────────────────────────────────────────────────┘
              │
   ┌──────────▼───────────────────────────────────────────┐
   │               集成层                                  │
   │   OllamaEngine / OpenAIEngine   MCP 客户端            │
   │   OpenAPI 客户端                ToolRouter            │
   └──────────────────────────────────────────────────────┘
```

### 请求流转 — SIMPLE 路径

`classify=SIMPLE` → FTS5 跨会话召回 → `loop.stream()` 调用真实 LLM → 第 1 轮：LLM 选择工具 → 工具执行（大输出 → `ToolResultStore`）→ 第 2 轮：LLM 综合分析 → 停止 → Hermes 流水线触发（FTS5 写入、事实提取、用户建模、技能演化）

### 请求流转 — COMPLEX 路径

`classify=COMPLEX` → `executor.execute()` → `run_with_hitl()` → LangGraph 图启动 → `hitl_interrupt_node` 调用 `interrupt()` → `register_interrupt()` → `TaskArtifactUpdateEvent` 入队 → SSE 流关闭 → 浏览器显示 HITL 审批卡片 → 运维人员点击 Approve/Reject → `POST /hitl/{id}/approve` → `graph.ainvoke(None, thread_cfg)` 恢复 → executor 节点执行 → 结果流式返回

---

## 3. 模块说明

### 3.1 `runtime/` — 执行引擎（8 个文件）

| 文件 | 职责 |
|---|---|
| `loop.py` | `AgentRuntimeLoop`：薄主循环——classify、stream、前/后置验证、工具分发 |
| `context_budget.py` | `ContextBudgetManager`：per-turn token 优先级分配；`ToolResultStore`：P0 大结果外部化 |
| `stop_policy.py` | `StopPolicy`：六维停止条件评估；`LoopState`：跨回合状态追踪 |
| `skill_catalog.py` | `SkillCatalogService` 集成：L1 摘要注入 + L2 按需加载 |
| `model_tier.py` | `ModelTierClassifier`：查询路由到 fast_model 或 full_model |
| `delegation.py` | 分叉/全新委托模式，上下文继承策略 |
| `tool_cache.py` | 复合技能评分（工具重叠度 × 0.6 + 语义相似度 × 0.4） |

### 3.2 `hitl/` — 人机协同（9 个文件）

| 文件 | 职责 |
|---|---|
| `a2a_integration.py` | `ITOpsHitlAgentExecutor`：双路由 + 后置验证钩子 |
| `graph.py` | LangGraph `StateGraph(ITOpsGraphState)`：6 节点（分类→风险评估→规划→中断→执行→格式化） |
| `triggers.py` | 四类触发器：破坏性操作 / 告警级别 / 置信度 / 歧义意图 |
| `decision.py` | `HitlDecisionRouter`：approve / reject / edit / escalate / timeout |
| `review.py` | 五个通知渠道：Slack、PagerDuty、SSE、WebSocket、邮件（并发扇出） |
| `audit.py` | 七种审计事件类型；支持内存和 PostgreSQL 后端 |
| `router.py` | FastAPI 路由：`/hitl/pending`、`/hitl/{id}/approve`、`/hitl/{id}/reject`、`/hitl/ws` |

**LangGraph 状态说明：** `StateGraph(ITOpsGraphState)` 使用 TypedDict 声明状态，LangGraph 会将每个节点的返回值合并到累积状态中（`{**old, **new}`）。若使用裸 `dict`，每个节点只能收到上一个节点的输出，`intent_type` 等早期字段会在 `planner_node` 执行前被静默丢弃，导致触发器评估失败。

### 3.3 `memory/` — 存储与学习（12 个文件）

| 层级 | 后端 | 检索方式 | TTL |
|---|---|---|---|
| L1 实时层 | Python 列表 | 全量返回 | 请求周期 |
| L2 短期层 | Redis 有序集合 | 时间倒序 | 24 小时 |
| L3 中期层 | ChromaDB 向量索引 | 余弦相似 + 时间衰减 | 30 天 |
| L4 长期层 | PostgreSQL 全文 | pg_trgm 相似度 | 永久 |

**Hermes 模块（本版本已激活）：**

| 模块 | 职责 |
|---|---|
| `fts_store.py` | `FTS5SessionStore`：SQLite FTS5，跨会话轮次检索，FTS5 安全查询消毒器 |
| `curator.py` | `MemoryCurator`：LLM 从已完成轮次中提取结构化事实；system/user prompt 分离防止注入 |
| `user_model.py` | `UserModelEngine`：追踪工具偏好、领域专业度、技术水平、行为特征 |

### 3.4 `skills/` — 技能系统（3 个文件）

| 文件 | 职责 |
|---|---|
| `catalog.py` | `SkillCatalogService`：L1 摘要 + L2 详情加载；复合评分（工具重叠 + 语义相似） |
| `evolver.py` | `SkillEvolver`：复杂任务后自动创建技能；LLM 评估复用潜力；基于反馈自我改进 |

### 3.5 `integrations/` — LLM 与工具后端（5 个文件）

| 文件 | 职责 |
|---|---|
| `llm_engine.py` | `OllamaEngine`、`OpenAIEngine`、`AnthropicEngine`、`MockEngine`；自动剥离推理模型 `<think>` 块 |
| `mcp_client.py` | MCP 服务器客户端；JSON 配置或环境变量驱动；支持 Mock 模式 |
| `openapi_client.py` | OpenAPI 3.0 规范消费者；自动生成工具定义；支持 Mock 模式 |
| `tool_router.py` | `ToolRouter`：将工具调用分发到 MCP / OpenAPI / Mock 注册表 |

### 3.6 `task/` — 任务编排（9 个文件）

- `intra/planner.py`：`TaskPlanner`（目标→DAG）+ `TaskScheduler`（并发=5，依赖解析）+ `TaskExecutor`
- `inter/coordinator.py`：`A2ATaskDispatcher`（SSE 委托）+ `MultiRoundCoordinator`（多轮上下文）
- `inter/hitl_bridge.py`：`HitlTaskBridge`（挂起/恢复钩子）

### 3.7 `registry/` — Agent 注册与发现（6 个文件）

- 动态抓取 AgentCard（4 条 well-known 路径）
- 健康检查（60 秒间隔）+ AgentCard 刷新（300 秒间隔）
- 负载均衡：`round_robin`（默认）/ `random` / `least_loaded`

### 3.8 `webui/` — 浏览器控制台

- `backend.py`：FastAPI 子应用，挂载于 `/webui`；`/chat/stream` SSE 端点；`/hitl/*` 审批端点；`/system/wiring` 实时接线状态
- `static/index.html`：终端风格单页控制台，无外部 JS 框架依赖；四个 Tab：Flow、HITL、Cache、Stats

---

## 4. 快速开始

### 环境要求

- Python 3.9+（推荐 3.12）
- 可选：Redis、PostgreSQL、ChromaDB（不配置时自动降级为内存 stub）

### 安装依赖

```bash
cd it-ops-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 启动服务（Mock LLM，无需外部依赖）

```bash
uvicorn main:app --reload --port 8000
```

启动后访问：
- **WebUI 控制台**：http://localhost:8000/webui/
- **A2A 接口**：http://localhost:8000/api/v1/a2a/
- **HITL 接口**：http://localhost:8000/hitl/
- **注册中心**：http://localhost:8000/registry/
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

### 验证启动接线

启动日志会打印：

```
━━ System wiring ━━
  LLM backend : mock
  FTS5 store  : ./data/state.db
  Curator     : yes (llm=stub)
  User model  : yes (llm=stub)
  SkillEvolver: yes (catalog=real)
  Executor    : wired
  HITL router : wired
```

Stats 面板 → **🔌 System Wiring** 显示各模块实时绿/红状态。`GET /webui/system/wiring` 返回 JSON。

---

## 5. 接入真实 LLM（Ollama）

```bash
# 拉取并启动模型
ollama serve
ollama pull qwen3.5:27b          # 或任意模型

# 设置环境变量
export LLM_BACKEND=ollama
export LLM_MODEL=qwen3.5:27b
export LLM_BASE_URL=http://localhost:11434
export HERMES_DATA_DIR=./data
export MCP_USE_MOCK=true
export OPENAPI_USE_MOCK=true

uvicorn main:app --port 8000 --reload
```

启动日志将显示 `LLM backend: ollama/qwen3.5:27b` 和 `Curator: yes (llm=real)`。

**支持的后端：**

| `LLM_BACKEND` | 说明 |
|---|---|
| `ollama` | 本地 Ollama 服务；需设置 `LLM_BASE_URL` 和 `LLM_MODEL` |
| `openai` | OpenAI API；需设置 `OPENAI_API_KEY` |
| `anthropic` | Anthropic API；需设置 `ANTHROPIC_API_KEY` |
| `mock` | 默认；确定性存根，不发送 LLM 请求 |

推理模型（如 `qwen3-coder`）：引擎自动剥离 `<think>…</think>` 块再解析工具调用，并向 Ollama API 传递 `think=False`。

---

## 6. WebUI 控制台使用指南

三栏布局：

### 左栏 — Skills & Tools

- **Skills 面板**：所有已注册技能；点击技能名填入查询框。绿点=低风险，黄点=中等，红点=高/危急。`HITL` 徽章=执行前需人工审批。
- **Quick Tools 面板**：直接调用 Mock 工具；大结果自动出现在 Cache Tab。

### 中栏 — 对话区

| 控件 | 说明 |
|---|---|
| 查询框 | Enter 发送，Shift+Enter 换行 |
| `session` | 留空自动生成会话 ID；填写 ID 可延续历史会话 |
| `mode` | `stream`=SSE 流式逐 token；`sync`=等待完整响应 |
| `delegation` | `fresh`=独立子 Agent；`forked`=继承父 Agent 的确认事实和工作集 |
| `[STORED:…]` 芯片 | 点击在 Cache Tab 分页浏览已缓存的大结果 |

**Flow Tab**：所有模块事件实时流式显示——分类、FTS5 召回、前置验证、工具调用、后置验证、Hermes 事实提取、用户建模、技能演化。点击任意行展开详情。

### 右栏 — HITL / Cache / Stats

- **HITL Tab**：待审批中断，显示触发类型、风险等级、建议操作、Approve / Reject 按钮。每 3 秒轮询 `/hitl/pending`；中断触发时自动切换到此 Tab。
- **Cache Tab**：所有 `ToolResultStore` 条目，支持 Prev/Next 分页浏览大结果。
- **Stats Tab**：技能数、缓存条目数、活跃 Agent 数、Hermes 模块状态、系统接线检查列表。

---

## 7. HITL 人工审批流程

### 触发语句示例

```
restart the payments-service in production
rollback auth-service to version 3.2.1
drain k8s-worker-03 for maintenance
delete the staging database
force failover payments-db to replica
```

### 完整流转

1. `classify()` 返回 `COMPLEX`——日志记录 `Complexity: complex — Destructive action detected`
2. `executor.execute()` 调用 `run_with_hitl()`——LangGraph 图启动
3. `intent_classifier_node` → `risk_assessor_node` → `planner_node` → `route_after_plan()`
4. `DestructiveActionTrigger` 触发——路由到 `hitl_interrupt_node`
5. `interrupt(payload)` 暂停图；checkpoint 保存状态
6. `_handle_interrupt_chunk()` 调用 `register_interrupt()`——payload 写入 `_payload_store`
7. `HitlA2AEventProcessor` 发出 `TaskArtifactUpdateEvent`——SSE 将 `hitl_interrupt` chunk 推送到浏览器
8. 浏览器：`switchTab('hitl')` + `refreshHitl(5)`（5 次重试 × 500 ms）
9. HITL 卡片显示：触发类型、风险等级、建议操作、SLA 倒计时
10. 运维人员点击 **Approve** → `POST /hitl/{id}/approve` → `graph.ainvoke(None, thread_cfg)` 恢复 → executor 节点执行
11. Flow Tab 记录 `⚠ HITL APPROVE` 及执行结果

### 审批决策类型

| 决策 | 效果 |
|---|---|
| `approve` | 图恢复，executor 节点执行建议操作 |
| `reject` | 图路由到 END，任务标记为已拒绝 |
| `edit` | 运维人员修改操作参数后图恢复 |
| `escalate` | 上报给高级审批人；SLA 计时重置 |
| `timeout` | SLA 到期后自动触发 |

### HITL 调试

关键日志行：

```
hitl.graph:    route_after_plan: intent_type='destructive_op' is_destructive=True action_type='restart_service'
hitl.graph:    HITL interrupt — interrupt_id=… trigger=destructive_op risk=high
hitl.graph:    Graph interrupt detection complete: found=True
hitl.decision: HITL registered: interrupt_id=… status=pending store_size=1
webui.backend: /hitl/pending: store_size=1 … returning 1 pending interrupts
```

若 `found=False`：检查是否使用了 `StateGraph(ITOpsGraphState)`（而非裸 `dict`），以及 `intent_classifier_node` 是否在查询中识别到 `"restart"` / `"rollback"` / `"delete"` / `"drain"`。

---

## 8. Hermes 学习循环

每次查询完成后，后置流水线自动运行：

```
轮次完成
    │
    ├─→ FTS5SessionStore.write_turn()
    │       将用户查询 + 助手回复写入 SQLite FTS5
    │       根据工具调用次数和回复长度计算重要性得分
    │
    ├─→ MemoryCurator.after_turn()
    │       LLM 从本轮中提取结构化事实
    │       类型：incident_lesson、config_fact、tool_preference、
    │             entity_relation、procedure_step、env_fact
    │       高置信度事实存入 MemoryRouter（L2–L4）
    │
    ├─→ UserModelEngine.after_turn()
    │       追踪：工具调用频率、领域计数（auth/network/compute/storage）、
    │             技术水平、行为特征
    │       将用户画像章节注入后续查询的 prompt
    │
    └─→ SkillEvolver.after_task()   （仅 COMPLEX 任务）
            LLM 自问：这个解决方案值得做成可复用技能吗？
            若是：生成 Markdown 技能 recipe（步骤、工具、风险）
            存入 SkillCatalogService 供未来查询使用
```

**FTS5 跨会话召回**在每次查询开始时运行：

```
新查询到来
    → curator.recall_for_session(query, session_id)
    → FTS5 在所有历史会话中检索相似轮次
    → 最佳匹配注入 LLM 调用前的 prompt
    → Flow Tab 显示：🔮 FTS5 Recall — N 字符，来自 M 个会话
```

**Hermes 发出的 WebUI 事件：**

| SSE chunk 类型 | 含义 |
|---|---|
| `hermes_write` | 轮次已写入 FTS5 |
| `hermes_curate` | MemoryCurator 提取了 `memories_count` 条事实 |
| `hermes_umodel` | 用户模型已更新——`technical_level`、`domain_counts`、`trait_count` |
| `hermes_skill` | SkillEvolver 创建或更新了技能（仅 COMPLEX） |
| `recall` | 历史上下文已注入——显示 `chars` 和 `sessions_searched` |

**数据目录：** `HERMES_DATA_DIR=./data`（默认）。包含 `state.db`（FTS5 历史轮次）和 SkillEvolver 写入的技能 recipe 文件。

---

## 9. P0 工具结果缓存

### 问题背景

运维场景中，工具调用可能返回数万字节数据——300 行 syslog、60 分钟 Prometheus 时序、500 条 NetFlow 记录。若直接注入 prompt，第一次工具调用就会耗尽上下文窗口。

### 解决方案

`ToolResultStore` + `ContextBudgetManager`——两步外部化：

1. 工具输出 > 4 000 字符 → 存入 `ToolResultStore`，prompt 仅注入：
   ```
   [STORED:syslog_search:a3f9c12b] Preview: Apr 10 09:12:01 ap-01 hostapd…
   ```

2. LLM 需要详情时通过 `read_stored_result` 按需分页读取：
   ```
   [TOOL:read_stored_result] {"ref_id": "a3f9c12b", "offset": 0, "length": 2000}
   ```

### 操作步骤

```
1. 查询："search syslogs for errors on ap-01"
   → Runtime Loop 调用 syslog_search（~6 000 字符）
   → 自动存储；prompt 中注入 [STORED:…] 引用
   → Cache Tab 出现新条目

2. 点击聊天中的 [STORED:syslog_search:abc123] 芯片
   → Cache Tab 加载第一页（2 000 字符）
   → 显示：Total: 6XXX chars | Has more: True | Next offset: 2000

3. 点击 "Next ▶" 继续翻页
```

### API 测试

```bash
# 触发大结果工具
curl -X POST http://localhost:8000/webui/tools/syslog_search \
  -H "Content-Type: application/json" \
  -d '{"args": {"host": "ap-01", "keyword": "error", "lines": 300}}'

# 分页读取缓存结果
curl "http://localhost:8000/webui/tools/result/{ref_id}?offset=0&length=2000"
```

---

## 10. P1 / P2 特性说明

### P1：前置/后置验证

```python
# 前置验证：执行前运行，阻断破坏性操作
pre = await loop.pre_verify(query, confirmed_facts, env_context)
if not pre.passed:
    return STOP_HITL  # 升级到 HITL 图

# 后置验证：每次工具调用后自动运行
post = await loop.post_verify(tool_name, result, confirmed_facts)
if not post.passed:
    state.unresolved_points.append(f"Post-verify: {post.reason}")
```

内置规则：破坏性操作一律失败并升级到 HITL；变更窗口关闭时拒绝变更操作；`allow_destructive=False` + 生产环境时拒绝。

### P1：确认事实与工作集

```python
# 确认事实——以最高优先级注入每轮 prompt
state.record_new_fact("payments-service 在 prod 环境，当前健康")
state.record_new_fact("DNS 解析正常，已排除 DNS 故障")

# 工作集——当前聚焦设备列表
working_set = [
    DeviceRef(id="ap-01", label="AP-01 at Site-A"),
    DeviceRef(id="sw-core-01", label="Core Switch"),
]
```

### P1：分叉委托

```python
# fresh: 独立子 Agent，从零开始
# forked: 继承父 Agent 的确认事实和工作集
delegation = "forked"
```

### P2：模型分层

```python
decision = loop.classify(query)
print(decision.model_tier)   # "fast_model" 或 "full_model"

# fast_model: check / status / dns / list 等简单查询
# full_model: 复杂分析、P0/P1 事件、破坏性操作
# 生产环境可据此路由到不同 Ollama 模型，或 haiku vs sonnet
```

---

## 11. Mock 工具与技能目录

### 大结果工具（触发 P0 缓存）

```bash
POST /webui/tools/syslog_search
{"args": {"host": "ap-01", "keyword": "error", "lines": 300}}    # ~6 KB

POST /webui/tools/prometheus_query
{"args": {"metric": "up", "job": "network_devices", "range_minutes": 60}}  # ~5 KB

POST /webui/tools/netflow_dump
{"args": {"site": "site-a", "flows": 500}}   # ~10 KB
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
POST /webui/tools/read_stored_result
{"args": {"ref_id": "a3f9c12b", "offset": 0, "length": 2000}}
```

### 技能目录（9 个预置技能）

| 技能 ID | 风险 | 需 HITL |
|---|---|---|
| `radius_auth_diagnosis` | low | 否 |
| `bgp_neighbor_check` | low | 否 |
| `dns_resolution_debug` | low | 否 |
| `k8s_pod_restart` | high | **是** |
| `db_failover` | critical | **是** |
| `syslog_bulk_analysis` | low | 否 |
| `network_traffic_analysis` | low | 否 |
| `prometheus_alert_triage` | medium | 否 |
| `change_window_check` | low | 否 |

---

## 12. 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BACKEND` | `mock` | `ollama` \| `openai` \| `anthropic` \| `mock` |
| `LLM_MODEL` | `mistral` | 传给后端的模型名称 |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama 或兼容 OpenAI 接口的基础 URL |
| `HERMES_DATA_DIR` | `./data` | FTS5 数据库和技能文件目录 |
| `MCP_USE_MOCK` | `true` | 使用 Mock MCP 工具（不连真实服务器） |
| `MCP_CONFIG_JSON` | — | MCP 服务器配置（JSON 字符串或 JSON 文件路径） |
| `OPENAPI_USE_MOCK` | `true` | 使用 Mock OpenAPI 工具（不加载真实规范） |
| `OPENAPI_SPEC_URL` | — | OpenAPI 3.0 规范 URL |
| `OPENAPI_BASE_URL` | — | OpenAPI 调用基础 URL |
| `OPENAPI_AUTH_TYPE` | `bearer` | `bearer` \| `api_key` \| `none` |
| `OPENAPI_TOKEN_ENV` | `NETOPS_API_TOKEN` | 持有 API Token 的环境变量名 |
| `A2A_BASE_URL` | `http://localhost:8000/api/v1/a2a` | 本 Agent 对外 A2A 基础 URL |
| `REDIS_URL` | — | Redis 连接串（缺失时自动降级为内存） |
| `POSTGRES_DSN` | — | PostgreSQL 连接串（缺失时跳过持久化） |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB 本地路径 |
| `AGENT_URLS` | — | 逗号分隔的对端 Agent URL（注册中心预填充） |
| `REGISTRY_LB` | `round_robin` | `round_robin` \| `random` \| `least_loaded` |
| `REGISTRY_HEALTH_INTERVAL` | `60` | 健康检查间隔（秒） |
| `HITL_CONFIDENCE_THRESHOLD` | `0.75` | 低于此置信度触发 HITL 歧义检查 |
| `HITL_MAX_AUTO_HOST_COUNT` | `5` | 影响主机数超过此值触发 HITL |
| `HITL_SLACK_WEBHOOK_URL` | — | Slack Incoming Webhook URL |
| `HITL_PAGERDUTY_ROUTING_KEY` | — | PagerDuty Events API 路由键 |
| `HOST` | `0.0.0.0` | 服务监听地址（python main.py 模式） |
| `PORT` | `8000` | 监听端口 |
| `RELOAD` | `false` | 是否开启热重载 |

---

## 13. 项目结构

```
it-ops-agent/
├── main.py                        # FastAPI 入口——逐步装配各服务模块
├── requirements.txt
├── pytest.ini
│
├── a2a/                           # A2A 协议（9 个文件）
│   ├── schemas.py                 # Pydantic 数据模型
│   ├── agent_card.py              # AgentCard 构建
│   ├── agent_executor.py          # 执行器基类 + 处理器链
│   ├── event_queue.py             # Sealed 异步事件队列
│   ├── request_handler.py         # JSON-RPC 方法路由
│   ├── server.py                  # FastAPI 子应用工厂
│   ├── push_notifications.py      # 推送通知（指数退避）
│   └── task_store.py              # 任务状态内存存储
│
├── hitl/                          # 人机协同（9 个文件）
│   ├── schemas.py                 # HitlPayload、HitlDecision、AuditRecord …
│   ├── triggers.py                # 四类触发器 + HitlConfig
│   ├── graph.py                   # LangGraph StateGraph(ITOpsGraphState)，6 节点
│   ├── review.py                  # 五渠道通知 + WebSocket 管理器
│   ├── decision.py                # HitlDecisionRouter——五种决策类型
│   ├── audit.py                   # 审计服务（内存 + PostgreSQL 双后端）
│   ├── router.py                  # FastAPI 路由 + /ws WebSocket 端点
│   └── a2a_integration.py         # ITOpsHitlAgentExecutor——双路由 + 后置验证
│
├── memory/                        # 存储与学习（12 个文件）
│   ├── schemas.py                 # MemoryRecord、RetrievalQuery …
│   ├── router.py                  # MemoryRouter 门面
│   ├── fts_store.py               # FTS5SessionStore——SQLite FTS5 跨会话召回
│   ├── curator.py                 # MemoryCurator——LLM 驱动的事实提取
│   ├── user_model.py              # UserModelEngine——用户专业度与特征追踪
│   ├── consolidation.py           # 整合工作器（摘要 + 实体抽取）
│   ├── stores/backends.py         # L1–L4 存储实现
│   └── pipelines/                 # ingestion.py + retrieval.py
│
├── skills/                        # 技能系统（3 个文件）
│   ├── catalog.py                 # SkillCatalogService + 9 个预置技能
│   └── evolver.py                 # SkillEvolver——自主技能创建与自我改进
│
├── integrations/                  # LLM 与工具后端（5 个文件）
│   ├── llm_engine.py              # Ollama / OpenAI / Anthropic / Mock 引擎
│   ├── mcp_client.py              # MCP 服务器客户端
│   ├── openapi_client.py          # OpenAPI 3.0 消费者
│   └── tool_router.py             # ToolRouter——分发到 MCP / OpenAPI / Mock
│
├── runtime/                       # 执行引擎（8 个文件）
│   ├── loop.py                    # AgentRuntimeLoop——薄主循环
│   ├── context_budget.py          # ContextBudgetManager + ToolResultStore（P0）
│   ├── stop_policy.py             # StopPolicy + LoopState
│   ├── skill_catalog.py           # SkillCatalogService 集成
│   ├── model_tier.py              # ModelTierClassifier（P2）
│   ├── delegation.py              # 分叉/全新委托
│   └── tool_cache.py              # 复合技能评分
│
├── task/                          # 任务编排（9 个文件）
│   ├── schemas.py                 # TaskDefinition、SessionRecord …
│   ├── intra/planner.py           # TaskPlanner + TaskScheduler + TaskExecutor
│   ├── intra/store.py             # TaskStore + RetryManager
│   ├── inter/coordinator.py       # A2ATaskDispatcher + MultiRoundCoordinator
│   ├── inter/session.py           # SessionManager（Redis，TTL=8 小时）
│   └── inter/hitl_bridge.py       # HitlTaskBridge——挂起/恢复钩子
│
├── registry/                      # Agent 注册中心（6 个文件）
│   ├── schemas.py                 # AgentEntry、AgentSkill、ResolutionResult
│   ├── store.py                   # 内存 + Redis 双存储
│   ├── discovery.py               # AgentDiscovery——AgentCard 抓取
│   ├── registry.py                # AgentRegistry——注册/解析/健康检查
│   └── router.py                  # FastAPI 路由
│
├── tools/                         # Mock 工具（2 个文件）
│   └── mock_tools.py              # 8 个 Mock 工具（3 大结果 + 4 内联 + 1 缓存读取）
│
├── webui/                         # 浏览器控制台
│   ├── backend.py                 # FastAPI 子应用——chat/stream、HITL、系统接线
│   └── static/index.html          # 终端风格单页控制台
│
└── tests/                         # 测试套件
    ├── test_a2a.py                # A2A 模块（~30 用例）
    ├── test_hitl.py               # HITL 模块（~25 用例）
    ├── test_memory_task.py        # Memory + Task（~45 用例）
    ├── test_registry.py           # Registry（~30 用例）
    ├── test_runtime.py            # Runtime Loop（~50 用例）
    ├── test_hermes_features.py    # Hermes 学习循环（~43 用例）
    └── test_p0_p1_p2.py           # P0/P1/P2 + WebUI（~60 用例）
```

---

## 14. 开发路线图

### 已实现

- [x] A2A Protocol v0.3.0——完整 SSE 流式 + WebSocket HITL
- [x] HITL 五层架构（触发 → 图中断 → 通知扇出 → 决策路由 → 审计）
- [x] `StateGraph(ITOpsGraphState)`——TypedDict 状态防止节点间键值静默丢失
- [x] 四层记忆（L1–L4，MMR 检索，整合工作器）
- [x] Hermes 后置学习流水线（FTS5 召回、MemoryCurator、UserModelEngine、SkillEvolver）
- [x] 集成层——Ollama / OpenAI / Anthropic / MCP / OpenAPI 工具后端
- [x] Task 模块（DAG 调度 + 跨 Agent 委托 + 多轮会话）
- [x] Registry——动态服务发现、健康检查、负载均衡
- [x] Runtime Loop——薄主循环 + 双路由 + P1/P2 特性
- [x] P0：ToolResultStore 大结果外部化 + 分页读取 API
- [x] P0：ContextBudgetManager per-turn token 优先级分配
- [x] P0：StopPolicy 六维停止条件评估
- [x] P1：SkillCatalogService——L1/L2 渐进披露 + 复合评分
- [x] P1：SkillEvolver——自主技能创建与基于反馈的自我改进
- [x] P1：分叉委托——fresh / forked + 上下文继承
- [x] P1：确认事实 & 工作集作为 prompt 一等公民
- [x] P1：前置/后置验证钩子
- [x] P2：模型分层——fast_model / full_model 路由
- [x] WebUI 控制台——终端风格、SSE/同步双模式、Flow Tab、HITL 审批、P0 可视化

### 待实现（生产化必须项）

- [ ] **JWT / OIDC 鉴权**：`/hitl/{id}/approve` 和 `/hitl/{id}/reject` 必须限制为审批人员角色——最高优先级
- [ ] **真实 LLM Chain**：替换 `intent_classifier_node`、`risk_assessor_node`、`planner_node` 中的关键词存根（结构化输出 LLM 调用）
- [ ] **真实工具接入**：替换 `task/intra/planner.py` 的 `_execute_task` 存根（kubectl / Prometheus HTTP API / OpsGenie）
- [ ] **真实 Embedding**：替换 `memory/pipelines/ingestion.py` 中的 `_EmbedderStub`
- [ ] **OpenTelemetry 链路追踪**：所有跨模块调用添加 Span，TraceID 通过 `session_id` 透传
- [ ] **PostgreSQL checkpoint**：替换 `build_hitl_graph()` 中的 `MemorySaver`，保障生产环境持久性

### 扩展项

- [ ] Prompt cache 友好策略（稳定前缀优先，变化内容后置）
- [ ] 轻量验证 Agent（独立小模型执行后置健康检查）
- [ ] Agent 版本管理（AgentCard `version` 字段，金丝雀发布）
- [ ] MCP 协议集成（Agent 访问外部工具的标准接口）

---

*文档版本：v3.0 | 最后更新：2026-04-15*