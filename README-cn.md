# NetOpYuAgent — IT 运维多智能体平台

> 可插拔、配置驱动的 IT/网络运维多智能体框架。
> 围绕 **模块独立**、**HITL 安全闸**、**可量化质量** 三件事构建。

**语言:** [English](README.md) · [中文](README-cn.md)

---

## 这是什么

NetOpYuAgent 是一个面向 IT 运维的生产级 AI Agent 平台。本地跑在 Ollama 上(或任何 OpenAI 兼容的 LLM),把自主任务循环和 HITL 人工审批闸结合起来,用内置的 golden-set 评测框架持续度量自己。

它**不是聊天机器人**。它是一个运行时,做这些事:

- 每个 query 通过 BM25 + Embedding 混合检索,只把**相关的 tool 和 skill** 塞进 prompt(不是"把所有 tool 都丢给 LLM")
- 碰到破坏性操作(`edit_device_config` / `rollback` / `restart_service` 等)**暂停**,等操作员批准/拒绝/修改,然后继续
- 维护 **5 层记忆**(短期 / 中期 facts / 长期知识 / 用户画像 / skill journal),支持语义召回
- **每个 turn 都在学习**:抽取 facts、更新用户模型、演化可复用 skill、检测 fact 冲突
- 内置 **6 个 CI audit** + retrieval bench + tool-compliance bench,质量是**测出来的**,不是凭感觉

---

## 文档分层

本 README 是浅入口。深度按需查:

| 层级 | 文档 | 何时读 |
|---|---|---|
| **L0 — 上手** | `README.md` / `README-cn.md`(你在这) | 第一次接触项目,想跑起来 |
| **L1 — 架构** | `ARCHITECTURE.md` | 跨模块改动,要看依赖图 + 模块表 + 跨模块约定 |
| **L2 — 模块深入** | `<模块>/DESIGN.md` | 单模块改动,要看数据流 + 设计决策史 |

每个功能模块都有一份 `DESIGN.md`,统一 6 节(职责 / 公开接口 / 数据流 / 设计决策 / 跨模块依赖 / 修改 checklist)。当前 7 份:
`agent_memory/`、`hitl_core/`、`integrations/`、`retrieval/`、`runtime/`、`skills/`、`tools/`。

---

## 快速开始

### 环境要求

- Python 3.11+
- 本地运行的 [Ollama](https://ollama.com) (用真 LLM) — 或者 `LLM_BACKEND=mock` 跑桩
- ~16 GB 内存跑 `qwen3.5:27b`,~6 GB 跑 `qwen2.5:7b`

### 安装

```bash
git clone <repo>
cd NetOpYuAgent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Mock LLM 跑(不需要 Ollama)

```bash
LLM_BACKEND=mock uvicorn main:app --port 8000
```

打开 http://localhost:8000 — WebUI 控制台。

### Ollama 跑(推荐)

```bash
# 先拉模型(一次):
ollama pull qwen2.5:7b
ollama pull nomic-embed-text     # 用作 embedding

# 启动 agent:
LLM_BACKEND=ollama LLM_MODEL=qwen2.5:7b uvicorn main:app --port 8000
```

大模型版:
```bash
LLM_MODEL=qwen3.5:27b uvicorn main:app --port 8000
```

### 验证启动

```bash
# 1. 健康检查
curl http://localhost:8000/health
# → {"status":"ok"}

# 2. WebUI 试一下 — http://localhost:8000
#    "列出所有设备" → 返回 mock 设备清单,不走 HITL
#    "重启 web-01 上的 nginx" → 触发 HITL 审批卡片
```

---

## 核心能力

| 能力 | 实现位置 | 价值 |
|---|---|---|
| **双路径路由** | `runtime/policy_engine.py` | 只读 query 绕过 LLM 评估(8000× 提速);破坏性 query 走 HITL |
| **HITL 审批闸** | `hitl_core/` + `integrations/adapters/hitl_executor.py` | LangGraph 风格 `interrupt()` — 浏览器显示卡片,operator 决定后 agent 恢复 |
| **5 层记忆** | `agent_memory/` | 短/中/长期 + 用户画像 + skill journal;FTS5 + embedding 混合召回 |
| **记忆压缩** | `MemoryAdapter.set_consolidator` | 每 session 每 30 轮,旧 chunk 自动 LLM 摘要;长 session 检索不慢 |
| **Fact 冲突检测** | `integrations/adapters/fact_conflict_detector.py` | 写 fact 前查相似;LLM 判定等价/精化/矛盾/无关并 reconcile |
| **Skill catalog + evolver** | `skills/` | L1 摘要永驻 prompt,L2 详情按需载入;dormant skill 自动改写 prompt |
| **原生工具调用** | `integrations/clients/llm_engine.py` + `schema/ollama_export.py` | (可选)Ollama OpenAI 风格 `tools` API — 结构性消灭"参数填错" |
| **Context 预算** | `runtime/context_budget.py` | 优先级分配:confirmed_facts > working_set > memory > tool_outputs > env |
| **Stop policy** | `runtime/stop_policy.py` | 六维(最大 turn / tool 调用数 / token 预算 / 进展 / 置信度 / 显式停止信号) |
| **MCP + OpenAPI 工具** | `integrations/router/tool_router.py` | 插上 MCP server 和任何 OpenAPI 3.0 spec |

---

## 架构(一张图)

```
外部调用方 (RouterAgent / WebUI / webhook / curl)
       │  A2A JSON-RPC over HTTP-SSE / REST
       ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI (main:app, uvicorn)                 │
│  /api/v1/a2a/*   /hitl/*   /registry/*   /webui/*       │
└───────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────▼──────────────┐
            │  runtime/policy_engine        │
            │   classify_query_intent       │
            │     read_only   → SIMPLE      │
            │     destructive → COMPLEX     │
            │     ambiguous   → LLM eval    │
            └───────────────┬──────────────┘
                            │
              ┌─────────────┴────────────┐
              ▼                          ▼
   ┌──────────────────┐        ┌─────────────────────┐
   │  Runtime Loop     │        │  HITL Pipeline      │
   │   context_budget  │        │   interrupt → 等待   │
   │   stop_policy     │        │   operator decision │
   │   skill_catalog   │        │   → resume          │
   │   tool_cache      │        └─────────┬───────────┘
   │   memory recall   │                  │
   └─────────┬─────────┘                  │
             │                            │
             ▼                            ▼
   ┌────────────────────────────────────────────────────┐
   │  integrations/clients/llm_engine — OllamaEngine     │
   │   [text 协议]    模型输出 [TOOL:name] {json}        │
   │   [native tools] 模型输出结构化 tool_call           │
   │                   → 合成回 [TOOL:] 行供下游解析    │
   └─────────────────────────────┬──────────────────────┘
                                 │
                                 ▼
   ┌────────────────────────────────────────────────────┐
   │  integrations/router/tool_router — ToolRouter       │
   │   分发到:本地 callable / MCP / OpenAPI            │
   └────────────────────────────────────────────────────┘
```

完整依赖图 + 模块边界:见 `ARCHITECTURE.md` §2。

---

## WebUI 控制台

打开 http://localhost:8000 — 三栏布局:

| 栏位 | 内容 |
|---|---|
| **左栏** | Skills 清单(可点击查看)+ Tool registry(mock + pragmatic) |
| **中栏** | Chat(SSE 流式)+ Flow tab(每个 turn 的 tool 调用、停止原因) |
| **右栏** | HITL 审批卡片 + tool-result cache + memory recall |

**右栏 tab**:
- **HITL** — 待审批卡片(Approve / Reject / Edit args / Skip with note)
- **Cache** — 大型 tool 输出(syslog、Prometheus)外置存储,引用为 `[STORED:id]`
- **Memory** — 颜色分类的召回卡:facts(绿)/profile(蓝)/recent turns(灰)
- **Journal** — 每个 session 的 skill 加载 + tool 调用事件,供离线分析

---

## HITL 审批流程

触发条件:query 涉及破坏性 tool(`edit_device_config` / `restart_service` / `rollback_config` 等),或 skill 匹配模糊,或 `cfg.hitl.tool_names` 里显式列出的 tool。

```
LLM 输出 [TOOL:edit_device_config] {device_id:"ap-01", ...}
              │
              ▼  HitlExecutor 在 tool 执行前拦截
        ChunkQueue.push(hitl_card)
              │
              ▼  WebUI 右栏弹卡片
              │   ┌─────────────────────────────────┐
              │   │ Approve | Reject | Edit | Skip  │
              │   └─────────────────────────────────┘
              │
              ▼  operator 点 Approve
        POST /hitl/{id}/approve
              │
              ▼  HitlPipeline.resume(decision)
        Tool 真正执行;结果回到 loop
```

**批量 HITL**:`[TOOL_BATCH:edit_device_config] [{...}, {...}, ...]` 为每个子目标开一张卡片,operator 独立审批每个。SkillEvolver 在所有子任务完成后一次性 fire,用成功 child 的并集喂数据。

详见 `hitl_core/DESIGN.md`。

---

## 记忆与学习循环

每个 turn 之后自动做 6 件事:

1. **FTS5 写入** — 这个 turn(query + LLM 回复 + tool calls)被索引,支持跨 session 召回
2. **Fact 抽取** — LLM 抽出结构化 fact("设备 ap-01 IOS 版本 15.4")写入中期 store
3. **冲突 reconcile** — 新 fact 经过 `FactConflictDetector`:等价 / 精化 / 矛盾 / 无关
4. **用户模型更新** — expertise + traits(例:"该 operator 偏好简洁回复,用 CLI 不用 GUI")
5. **Skill 演化** — `SkillJournalConsumer` 盯着 dormant skill,触发 `SkillEvolver.apply_feedback` 改写 prompt
6. **自动压缩** — 每个 session 每 30 轮,旧 chunk 合并成 LLM 摘要 rollup,长 session 不会越来越慢

`config.yaml` 调阈值:

```yaml
memory:
  auto_consolidate_turns: 30           # 0 关闭
  consolidation_template: "structured" # 或 "legacy"

cross_module:
  journal_to_facts:
    enabled: true                       # 把 journal 观察提升为中期 fact
  fact_conflict_detection:
    enabled: true
    llm_reconcile_enabled: false        # 默认只做廉价启发式
```

完整记忆架构:`agent_memory/DESIGN.md`。

---

## CI 与质量门禁

每个 PR 都被 `scripts/precheck.sh` gate 住:

```bash
./scripts/precheck.sh            # 全部(audit + eval)
./scripts/precheck.sh --audits   # 仅静态 audit(~30s)
./scripts/precheck.sh --eval     # 仅 retrieval eval
```

**6 个静态 audit**(任何 FAIL → PR 不可合):

| Audit | 抓什么 |
|---|---|
| `syntax_sweep` | 任何 `.py` 文件有 parse error |
| `audit_module_independence` | 跨模块 import 违规(例:`evaluation/` import `runtime/`) |
| `audit_imports` | 解析不到任何模块的 import 路径 |
| `audit_prompt_templates` | 未转义的 `{...}` 在 f-string-like prompt 里,format 时会崩 |
| `audit_directive_parsing` | `[TOOL:` 解析绕路 — 工具调用提取单一入口 |
| `audit_wiring` | "幽灵服务" — 被 register 到 `services[...]` 但没外部读者 |

**Retrieval eval gate**:
- CI: BM25 backend,`data/golden_set.jsonl`(25 cases),阈值 `recall@3 ≥ 0.40, MRR ≥ 0.30`
- 本地: hybrid backend,阈值 `recall@3 ≥ 0.65, MRR ≥ 0.55`

**Tool-compliance bench**(`data/tool_compliance_set.jsonl`,18 cases) — **不在 CI**(需要跑着的 Ollama),本地或夜间跑:

```bash
# Baseline:text 协议
python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b

# Native tools(Ollama ≥ 0.4 + 支持 tools 的模型)
python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b --native
```

每个 case 三个独立指标:
- `parse_ok` — 模型语法合法,能解析出 `[TOOL:...]`
- `name_ok` — 模型选对了 tool(或合法替代)
- `args_ok` — 必填 arg 在,pin 死的 value 对,无 forbidden arg

**pre-commit hook**(推荐):`pip install pre-commit && pre-commit install` — commit 前本地跑同一套 `--audits`。

**Branch protection**(GitHub):仓库 Settings → Branches → 把 `Static audits` / `Production safety tests` / `Retrieval eval (BM25)` 设为 required checks。

---

## 原生工具(可选,Tier 1-C)

如果你跑 Ollama ≥ 0.4 + 支持 tools 的模型(qwen2.5+、qwen3、llama3.1+、mistral-nemo、deepseek-v3...),`config.yaml` 一行打开:

```yaml
llm:
  capabilities:
    supports_native_tools: true     # 默认 false
```

重启即可。引擎给 Ollama 喂 OpenAI 风格 `tools` 数组,拿回**结构化 `tool_calls`**,而不是文本 `[TOOL:name] {...}`。引擎再把结构化结果**合成回** `[TOOL:name] {json}` 行,所以 runtime loop + directive parser + HITL 流程**完全不变** — 这是运行时增强,不是架构改动。

**它解决什么**:"模型把 device_id 填错字段" / "模型 JSON 少了个引号" / "模型幻觉出额外参数" — 这些**结构性消失**,因为模型不再手打 args 字符串,API 协议要求 args 是真的 dict。

**怎么量化收益**:跑上面的 compliance bench,带 `--native` 和不带,对比 `args_ok`。

**回滚**:改一行 config + 重启,无代码改动。

---

## 运维 cheatsheet

```bash
# 切模型
LLM_MODEL=qwen3.5:14b uvicorn main:app --port 8000

# 强制某 tool 必走 HITL
HITL_TOOL_NAMES=netflow_dump,db_failover uvicorn main:app

# 开跨模块学习(journal → facts)
# config.yaml:
#   cross_module:
#     journal_to_facts:
#       enabled: true

# 详细 LLM 日志
LLM_LOG_DETAIL=compact LOG_MODE=llm uvicorn main:app --port 8000

# 不 commit 跑 audit
./scripts/precheck.sh --audits

# 跑 retrieval bench
python -m evaluation.cli --golden data/golden_set.jsonl --backend hybrid --top-k 5

# 跑 tool-compliance bench(需要 Ollama)
python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b --verbose

# 检查某 session 的内存
sqlite3 data/memory/midterm.db "SELECT fact, fact_type, confidence FROM facts ORDER BY created_at DESC LIMIT 20"
```

---

## 项目目录

```
NetOpYuAgent/
├── ARCHITECTURE.md            ← 跨模块参考(任何多模块改动,先读)
├── README.md / README-cn.md   ← 上手入口(你在这)
├── main.py                    ← FastAPI app + lifespan;build_services() 装配一切
├── config.py / config.yaml    ← 所有模块激活开关
│
├── runtime/                   ← agent loop / stop policy / directive parser / context budget
│   └── DESIGN.md
├── agent_memory/              ← 5 层记忆 + 压缩 + FTS5 + embedding hybrid
│   └── DESIGN.md
├── hitl_core/                 ← interrupt / decision / batch / audit pipeline
│   └── DESIGN.md
├── retrieval/                 ← BM25 / Embedding / Hybrid / Cache backends + meta tools
│   └── DESIGN.md
├── skills/                    ← catalog + journal + evolver + journal_consumer + loader
│   └── DESIGN.md
├── tools/                     ← mock + pragmatic tool 实现 + metadata
│   └── DESIGN.md
├── integrations/              ← 跨模块胶水(LLM engine / MCP / OpenAPI / adapters)
│   └── DESIGN.md
│
├── schema/                    ← ArgSchema + JSON-Schema / Ollama tools 导出器
├── evaluation/                ← retrieval bench + tool-compliance bench + CLI
├── memory/                    ← 给 runtime 用的 thin MemoryAdapter 门面
├── webui/                     ← FastAPI 子 app + SPA dashboard
├── a2a/                       ← agent-to-agent 协议原语
├── task/                      ← task graph 原语
├── registry/                  ← agent 身份 + 能力注册表
│
├── data/
│   ├── golden_set.jsonl              ← retrieval bench(25 cases)
│   └── tool_compliance_set.jsonl     ← compliance bench(18 cases)
│
├── scripts/
│   ├── precheck.sh                   ← audit + eval 单一入口(CI 和 pre-commit 共用)
│   ├── audit_module_independence.py
│   ├── audit_imports.py
│   ├── audit_wiring.py
│   ├── audit_prompt_templates.py
│   ├── audit_directive_parsing.py
│   └── _audit_common.py
│
├── .github/workflows/ci.yml          ← 每个 PR 跑 3 个并行 job
└── .pre-commit-config.yaml           ← commit 前本地 audit gate
```

某模块的逐文件清单:读它的 `DESIGN.md`。

---

## 贡献指南

### 提 PR 之前

```bash
# 一次性装本地 quality gate:
pip install pre-commit
pre-commit install

# push 之前跑一遍:
./scripts/precheck.sh
```

### 改一个模块

**只读那个模块的** `DESIGN.md`。跑测试;让 CI 抓剩下的。

### 改两个模块

读 `ARCHITECTURE.md` §4("跨模块约定")。绝大多数跨模块连线放在 `cfg.cross_module.*` — 看 `config.yaml`。

### 新加一个 service(跨模块协作者)

1. 在 `main.py:build_services()` 里构造
2. 注册:`services["my_service"] = obj`
3. **必须有至少一个外部文件通过 `services.get("my_service")` 或 `services["my_service"]` 读它** — 否则 `audit_wiring.py` 会把它标成 ghost service,CI 失败
4. 如果它只是 introspection-only(无运行时调用方 — 例如通过 `/system/wiring` 暴露状态),在 `audit_wiring.py` 的 `KEY_WHITELIST` 里加上,附理由

### 加一个 tool-compliance case

往 `data/tool_compliance_set.jsonl` 加一行:

```jsonl
{"query": "你的 query", "expected_tool": "tool_name", "expected_args": {"k": "v"}, "tags": ["destructive"]}
```

CI 验证结构;bench 测真实模型表现。

### 加一个 retrieval golden case

往 `data/golden_set.jsonl` 加(参考已有 case 的 schema)。再跑一次 `./scripts/precheck.sh --eval` 确认阈值还稳。

### 文档规则

- 单模块改动 → 改那个模块的 `DESIGN.md`
- 跨模块改动 → 触及依赖图就改 `ARCHITECTURE.md`
- 新模块 → 必须带 `DESIGN.md`(6 节标准模板)— PR 强制要求
- README 只做 onboarding;深度内容放 `DESIGN.md` / `ARCHITECTURE.md`

---

## Roadmap

### 已完成

- ✅ A2A Protocol v0.3.0 — 完整 SSE 流 + WebSocket HITL
- ✅ HITL pipeline(interrupt + 4 种决策 + 批量 + 审计)— `hitl_core/`
- ✅ 5 层记忆(短/中/长/profile/journal),FTS5 + embedding 混合召回
- ✅ Fact 冲突检测接入 `MemoryAdapter.add_fact`(语义去重 + LLM reconcile)
- ✅ 每 N 轮自动 session 压缩(后台,非阻塞)
- ✅ Hermes 后置 pipeline(fact 抽取 + 用户模型 + skill evolver)
- ✅ MCP + OpenAPI tool 后端;统一 `ToolRouter`
- ✅ Skill catalog(L1/L2 渐进披露 + 组合评分)
- ✅ SkillEvolver — 自主创建 + dormant skill 改写
- ✅ Tool result 外置存储(`[STORED:id]`)+ 分页读 API
- ✅ Context budget 优先级分配(`cfg.context_budget.strategy=priority`)
- ✅ Stop policy 六维评估
- ✅ PolicyEngine 意图分类 fast-path(只读 query 绕开 LLM 评估)
- ✅ Trust-mode 渐进 HITL(cautious / standard / trusted)
- ✅ 生命周期 hooks(`runtime/hooks.py`)用作低成本扩展
- ✅ JWT / API-key auth(`auth_core.py` / `auth.py`)
- ✅ 日志脱敏(secret / API key / SNMP community)
- ✅ 原生 tool API(Ollama OpenAI 风格,通过 `supports_native_tools` 开)
- ✅ Tool-compliance bench(`evaluation/compliance_cli.py`)做 A/B 模型评估
- ✅ Retrieval bench(`evaluation/cli.py`)+ CI gate
- ✅ 6 个静态 audit + CI + pre-commit + branch protection

### 已预留 / 未实现

- ⏳ **Skill 作为 sub-agent** — 高复杂度 skill(≥5 步)做成 LangGraph subgraph,独立 prompt + 预算。`cfg.skill_orchestration.subagent.*` 命名空间已预留。
- ⏳ **MemGPT 风格 LLM 自管理记忆** — Agent 自己提升/降级 tier。等 fact 体系稳定干净再做。
- ⏳ **OpenTelemetry tracing** — 全跨模块调用打 span;`session_id` 作为 TraceID。
- ⏳ **Postgres checkpointer** 给 HITL graph state(替换内存版 `MemorySaver`,生产持久化)。
- ⏳ **分布式多 agent A2A** — 原语已有(`a2a/`),编排层待做。
- ⏳ **每 tool 熔断器** — `ToolMeta` 里有部分实现,UI 集成待做。

---

## 词汇表

| 术语 | 定义 |
|---|---|
| **功能模块**(Functional module) | 与其他功能模块零 import 的代码单元(`memory/`, `hitl_core/` 等)。`audit_module_independence` 静态校验。 |
| **Adapter** | 连接两个功能模块的可选桥。住在 `integrations/adapters/`。 |
| **Skill** | LLM 参考的 markdown 流程(不是可执行代码)。 |
| **Tool** | 原子可调用 — 本地函数 / MCP server method / OpenAPI op。 |
| **Meta-tool** | LLM 用来发现其他 tool 的 tool(`list_tools` / `tool_details`)。 |
| **Journal** | 每 session 的 skill 加载 + tool 调用日志;被 `SkillEvolver` 消费。 |
| **Fact** | 从对话抽取的结构化陈述,存中期记忆。 |
| **HITL gate** | 暂停 agent 直到人类决定的机制。 |
| **Ghost service** | `services[...]` 里有但无外部读者的对象 — `audit_wiring` 抓这种。 |
| **Native tools** | Ollama 的 OpenAI 兼容 `tools` API — 结构化 tool_calls 而不是文本协议。 |
| **Trust mode** | `cautious` / `standard` / `trusted` — 操作员设置的 HITL 触发敏感度。 |

---

## 环境变量

常用开关(完整列表逐字段见 `config.py`):

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` / `openai` / `anthropic` / `mock` |
| `LLM_MODEL` | `qwen3.5:27b` | 任何 Ollama tag 或 OpenAI/Anthropic 模型名 |
| `LLM_BASE_URL` | `http://localhost:11434` | 远端 Ollama 用 |
| `LLM_LOG_DETAIL` | `summary` | `summary` / `compact` / `full` |
| `LLM_SUPPORTS_NATIVE_TOOLS` | `false` | 开 Ollama 原生 tool API(Tier 1-C) |
| `HITL_BACKEND` | `core` | `core`(推荐)或 `langgraph`(老版) |
| `HITL_TOOL_NAMES` | (config.yaml) | 逗号分隔的 tool 名,这些总走 HITL |
| `HITL_SKILL_AMBIGUITY` | `false` | skill 匹配置信度低时也 gate |
| `MEMORY_AUTO_CONSOLIDATE_TURNS` | `30` | 0 关闭每 session 自动压缩 |
| `MEMORY_CONSOLIDATION_TEMPLATE` | `structured` | `structured`(Hermes)或 `legacy` |
| `DTM_COMPACTION_TURNS` | `5` | N 轮后 flush daily `.md`(生产值调高) |
| `NETOPYU_JWT_SECRET` | (auth 开启时必填) | HS256 签名 key |
| `LOG_MODE` | `default` | `llm` / `verbose` / `default` / `quiet` |

---

## License

(见仓库根目录 `LICENSE`)

---

*版本: v4.0 — 2026 年 5 月(Tier 1-C / 2-E 完成后)*
*最后比对: `ARCHITECTURE.md` rev 2026-05*