# Agent Memory Module v5 — 设计、实现与集成文档

**版本**: v5.0 | **测试**: 311 个用例，全部通过

---

## 目录

1. [模块定位与设计哲学](#1-模块定位与设计哲学)
2. [整体架构](#2-整体架构)
3. [七层记忆能力详解](#3-七层记忆能力详解)
4. [执行流程](#4-执行流程)
5. [关键设计决策](#5-关键设计决策)
6. [与 Generic Agent 的能力对比](#6-与-generic-agent-的能力对比)
7. [生产就绪性说明](#7-生产就绪性说明)
8. [集成指南](#8-集成指南)
9. [API 速查](#9-api-速查)
10. [配置参数说明](#10-配置参数说明)
11. [升级路径](#11-升级路径)

---

## 1. 模块定位与设计哲学

### 核心问题

LLM 本身无状态。每次调用是一张白纸。Agent 需要记忆系统来：

- **跨轮次**保留上下文（用户说了什么、Agent 做了什么、工具返回了什么）
- **跨会话**召回历史事实（用户偏好、环境配置、过去的结论）
- **当前轮次**追踪已验证的事实和正在处理的实体（热轨道）
- **积累技能**：把成功的执行路径固化为可复用的 Skill（程序性记忆）
- **从错误学习**：任务失败后写入反思，下次避免重蹈覆辙
- **控制 token 消耗**：长会话压缩历史，保持 context 信息密度

### 两个核心原则

**信息密度最大化**（来自 Generic Agent）：用尽可能少的 token 表达尽可能多的有效信息。通过上下文压缩（Consolidation）、优先级 Budget 管理（ContextBudgetManager）、MMR 多样性去重，让 3200 token 的 context 窗口承载最高价值的记忆。

**生产工程优先**（我们的设计）：多用户隔离、并发安全、TTL 管理、数据持久化、可观测性。面向多用户生产服务，不为追求极简牺牲稳定性。

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      MemoryManager (统一门面)                         │
│  唯一集成入口，协调所有层，管理资源生命周期，支持上下文管理器          │
└────┬──────────┬──────────┬──────────┬──────────┬────────────────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────────┐
│Session  │ │Context │ │  User  │ │Consoli-│ │  Reflection        │
│State    │ │Budget  │ │ Model  │ │dation  │ │  Engine  ← v5 NEW  │
│(热轨道) │ │Manager │ │Engine  │ │← v5NEW │ │                    │
│         │ │        │ │        │ │        │ │ reflect()          │
│confirm_ │ │P1 P2   │ │6维行为 │ │长会话  │ │ → Long+Mid+Skill   │
│facts    │ │P3 P4   │ │建模    │ │摘要压缩│ │                    │
│working_ │ │P5 P6   │ │stated/ │ │        │ └────────────────────┘
│set      │ │        │ │revealed│ └────────┘
│recent_  │ └────────┘ └────────┘
│tools    │
└─────────┘
         │ (冷轨道 — SQLite 持久化)
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ShortTerm     │ │MidTerm       │ │LongTerm      │ │SkillStore │ │
│  │(P0 Tool Cache│ │(Hermes facts)│ │(Claw chunks) │ │← v5 NEW   │ │
│  │ 文件+SQLite  │ │ FTS5+TF-IDF  │ │ FTS5+TF-IDF  │ │ FTS5+     │ │
│  │ 字节offset   │ │ 去重+TTL     │ │ 时效+重要度  │ │ TF-IDF    │ │
│  │ 原子eviction │ │ 置信度衰减   │ │ batch优化    │ │ 成功率EMA │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │          EmbeddingIndex  ← v5 NEW                            │  │
│  │  TFIDFBackend (default) | SentenceTransformer | OpenAI |    │  │
│  │  CallableBackend (any fn) — 统一接口，热插拔                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│                    _db._Pool (共享连接池)                            │
│           WAL | 线程本地连接 | BEGIN IMMEDIATE 写锁                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 文件结构（20个 Python 文件，311 个测试）

```
agent_memory/
├── memory_manager.py          # 统一门面 v5（唯一集成入口）
├── user_model.py              # UserModelEngine（行为建模）
├── session_state.py           # SessionState + MMR + Registry
├── context_budget.py          # ContextBudgetManager + Priority
├── consolidation.py           # MemoryConsolidator + ReflectionEngine ← v5
├── schemas.py                 # 数据模型 + 输入验证
├── stores/
│   ├── _db.py                 # WAL SQLite 连接池
│   ├── long_term_store.py     # 长期 chunks（时效+重要度+batch）
│   ├── mid_term_store.py      # 中期 facts（去重+TTL+衰减）
│   ├── short_term_store.py    # 短期工具缓存（字节offset）
│   └── skill_store.py         # 程序性记忆 ← v5 NEW
├── retrieval/
│   ├── vector_store.py        # TFIDFIndex（LRU，线程安全）
│   ├── fact_extractor.py      # LLM/规则事实提炼
│   └── embedding_store.py     # 语义向量检索（可插拔后端）← v5 NEW
├── tests/
│   ├── test_memory.py         # 106 个核心测试
│   ├── test_user_model.py     # 39 个行为建模测试
│   ├── test_v4_features.py    # 61 个 v4 功能测试
│   ├── test_stores_v4.py      # 42 个存储层专项测试
│   └── test_v5_features.py    # 63 个 v5 新功能测试 ← NEW
└── examples/
    ├── e2e_scenario.py        # 端到端演示（4轮对话，含所有能力）
    └── langchain_integration.py # LangChain 集成
```

---

## 3. 七层记忆能力详解

### 3.1 短期记忆 — ShortTermStore（P0 Tool Cache）

**目的**：工具调用返回的大型输出（日志、API 响应、>4KB）不能放进 prompt，需要外置存储，LLM 按需分段读取。

**分页读取（字节 offset）**：

```
LLM 请求 → read(user_id, ref_id, offset=0, length=2000)
              │
              ├─ 验证 ref_id 存在于 DB（防路径穿越）
              ├─ 检查 expires_at 未过期
              ├─ 以二进制模式打开文件，seek(offset_bytes)
              ├─ 读取 length 字节
              ├─ trim 到合法 UTF-8 边界（处理 CJK/emoji 中断）
              └─ 返回 {content, offset, length, total_bytes, has_more, next_offset}
```

**v4 修复**：字节 offset（非字符 offset）、原子 eviction（SELECT+DELETE 同一事务）、`list_by_tool()` 过滤 API、`garbage_collect()` 孤儿文件清理。

---

### 3.2 中期记忆 — MidTermStore（Hermes Facts）

**目的**：从对话中提炼结构化事实，session 内高优先级检索，跨 session 历史查询。

**去重**：`UNIQUE INDEX ON (user_id, session_id, fact_hash)` + `INSERT OR IGNORE`，`fact_hash = md5(user+session+fact.lower())`。相同文本无论插入多少次只存一条。

**TTL**：`expires_at` 字段，默认 30 天，支持 per-fact 自定义。`evict_expired_facts()` 原子删除（SELECT+DELETE 同一事务）。

**置信度衰减**：

```python
update_fact(fact_id, decay=True)
# confidence *= 0.7, decay_count += 1, 最低 0.1
# 用于：工具验证结果与原有 fact 矛盾时调用
```

---

### 3.3 长期记忆 — LongTermStore（Claw Chunks）

**目的**：存储完整对话轮次的原始文本，支持跨 session 语义召回，带重要度和时效性评分。

**检索评分**：

```
total_score = tfidf(chunk, query)           # TF-IDF 语义
            + 0.5 × fts5_hit               # FTS5 关键词奖励
            + 0.15 × recency_decay()       # 时效：exp(-age/half_life_14d)
            + 0.10 × (importance + source_boost)  # 重要度
```

**source 权重**：`tool_output +0.15` > `document +0.10` > `conversation ±0` > `llm_response -0.05`

**批量写入优化**：禁用 FTS5 trigger → executemany 批量 INSERT → 批量 FTS5 rebuild → 恢复 trigger。2000条 ~80ms（原触发器方式 400ms）。

---

### 3.4 程序性记忆 — SkillStore ← v5 NEW

**目的**：存储 Agent 通过实际执行固化的可复用任务技能（类 Generic Agent 的 Skill Tree）。

**Skill 结构**：

```python
Skill(
    skill_id, user_id, skill_name,
    description,        # 可被 FTS5+TF-IDF 检索
    steps: List[Any],   # 执行步骤（文本 or 工具调用 dict）
    tags: List[str],    # 检索标签
    version: int,       # 同名 skill 自动版本迭代
    success_rate: float,# EMA 成功率，低于 0.3 → deprecated
    use_count, success_count,
    status: "active" | "deprecated" | "archived",
)
```

**检索**：FTS5 关键词 + TF-IDF（description + tags），按 `success_rate × tfidf_score` 排序。

**成功率更新**（EMA，`alpha=0.2`）：

```
new_rate = 0.2 × (success_count/use_count) + 0.8 × old_rate
rate < 0.3 → status = "deprecated"（不再出现在默认检索结果）
```

**Context 注入**：注入 P2 位置（与 `confirmed_facts` 同级），格式：

```
## 可复用技能 (Skill Library)

[Skill: bgp_fault_diagnosis] (v2, 成功率=92%, 用过 15 次)
描述: BGP邻居故障排查：ping验证、syslog分析、联系NOC
标签: bgp, network, diagnosis
执行步骤:
  1. bgp_check(router) — 查看邻居状态，识别 Idle 邻居
  2. syslog_search(peer_ip, hours=24) — 分析连接断开历史
  3. ping(peer_ip) — 验证 L3 可达性
  4. interface_check(router) — 确认本地接口正常
  5. 若 ping 失败且接口正常 → 联系 AS NOC
```

---

### 3.5 语义向量检索 — EmbeddingIndex ← v5 NEW

**目的**：弥补 TF-IDF 无法理解语义同义词的缺陷。"宕机" vs "down" vs "故障"语义相同但 TF-IDF 检索不到。

**可插拔后端**：

```python
# 默认（零依赖，退化为 TF-IDF）
idx = EmbeddingIndex()

# 本地语义（推荐生产）
idx = EmbeddingIndex.from_sentence_transformer("BAAI/bge-small-zh-v1.5")

# OpenAI/兼容 API
idx = EmbeddingIndex.from_openai("sk-...", model="text-embedding-3-small")

# 任意自定义函数
idx = EmbeddingIndex.from_callable(my_embed_fn, dim=1024)
```

**混合检索**（`search_chunks(use_embedding=True)`）：

```
Score = 0.5 × cosine_similarity(q_vec, doc_vec)   # 语义相关性
      + 0.5 × (1 / rank_in_fts_result)             # FTS5 倒数排名

→ 语义 + 关键词双重保障，彼此补充
```

**MMR（语义版）**：`query_mmr()` 用余弦距离计算 MMR，比 Jaccard 版精度更高：

```
mmr_score = λ × rel(d,q) − (1−λ) × max_cosine(d, already_selected)
```

---

### 3.6 历史压缩 — MemoryConsolidator ← v5 NEW

**目的**：长会话（30+ 轮）后 chunks 线性堆积，context window 被历史占满。压缩将旧 chunks 合并为摘要，保持信息密度（类 Generic Agent 的 <30K 窗口策略）。

**触发机制**：

```python
# 自动检测
if mem.should_consolidate(user_id, session_id):
    result = mem.consolidate_session(user_id, session_id)

# 参数控制
MemoryManager(consolidate_after_n_turns=30,  # 超过 N 条触发
              consolidation_keep_recent=10)   # 保留最近 N 条原文
```

**压缩过程**：

```
session chunks (N 条)
  │
  ├─ 最近 keep_recent_n 条 → 保留原文（当前轮上下文，不压缩）
  └─ 更旧的 (N - keep_recent_n) 条 → 按 group_size=8 分组
       │
       ├─ LLM summarize(group) → 1条 summary chunk
       │  (无 LLM → fallback: 拼接截断，仍然有效)
       ├─ summary chunk: source="summary", importance=0.85
       └─ 删除原始 group chunks
```

**效果**：`chunks_before=30 → chunks_after=12`（10条原文 + 2条摘要），节省 60% context token。

---

### 3.7 反思写入 — ReflectionEngine ← v5 NEW

**目的**：任务完成（或失败）后，让 Agent 记录经验教训，避免重蹈覆辙，也使成功路径得以固化。

**三层同步写入**：

```python
result = mem.reflect(
    user_id, session_id,
    task="AS65002 BGP 故障排查",
    outcome="success",    # "success" | "failure" | "partial"
    summary="通过 ping 和 syslog 定位问题，联系 NOC 后恢复",
    skill_id=sk.skill_id,  # 可选：关联 Skill
)

# 自动写入：
# 1. LongTermStore:  source="reflection", importance=0.9
# 2. MidTermStore:   fact_type="lesson", 提炼可操作教训
# 3. SkillStore:     record_outcome(success=True) → 更新 EMA 成功率
```

**教训提炼**（有 LLM 时）：

```
LESSON_EXTRACT_PROMPT → JSON: [{"lesson": "先验证L2再排查BGP", "confidence": 0.85}]
无 LLM → 规则提取含关键词（"建议"/"避免"/"关键"等）的句子
```

**成功 vs 失败的差异**：

```
outcome="success" → lesson confidence: 0.65-0.85
outcome="failure" → lesson confidence: 0.70-0.90（失败教训权重更高）
                  → skill.success_rate 下降（EMA）
outcome="failure" 3次以上 → skill.status = "deprecated"
```

---

### 3.8 用户行为建模 — UserModelEngine（Hermes Honcho）

6维行为模型（v3 引入，v5 保留）：

| 维度 | 说明 |
|---|---|
| `technical_level` | NOVICE / INTERMEDIATE / EXPERT |
| `communication_style` | TERSE / BALANCED / VERBOSE |
| `domain_counts` | 领域频率 `{"network": 15, "auth": 8}` |
| `tool_usage` | 工具使用频率 |
| `hourly_activity` | 活跃时间分布 |
| `traits` | 自由推断特征（EMA 置信度） |

**辩证推理**：`stated_preferences`（用户说的）vs `revealed_preferences`（行为观察的），矛盾时 `trait.contradicted=True`，不注入 prompt。

---

### 3.9 双轨道 + ContextBudgetManager

**热轨道**（内存，零 DB 延迟）：`confirmed_facts`、`working_set`、`recent_tool_results`

**冷轨道**（DB 查询，毫秒级）：`mid_term_facts`、`long_term_chunks`、`user_profile`

**优先级（v5）**：

| 优先级 | Section | 来源 | 驱逐规则 |
|---|---|---|---|
| P1 | user_profile | UserModelEngine | 永不驱逐 |
| P2a | confirmed_facts | 热轨道 | 永不驱逐 |
| P2b | **skills** ← v5 | SkillStore | 永不驱逐（与 P2a 共享） |
| P3 | working_set | 热轨道 | 驱逐优先级 3 |
| P4 | mid_term_facts | DB FTS5+TF-IDF+MMR | 驱逐优先级 4 |
| P5 | long_term_chunks | DB 混合检索+MMR | 驱逐优先级 5 |
| P6 | environment | 外部传入 | 最先驱逐 |

---

## 4. 执行流程

### 4.1 标准每轮调用流程（v5）

```
用户发送 query
       │
       ▼
  1. state = mem.get_session(user_id, session_id)
     → 获取热轨道 SessionState（内存，零 DB 访问）

       │
       ▼
  2. ctx, report = mem.build_context_budgeted(state, query, budget=3200)
     执行顺序：
       P1  UserModelEngine.get_prompt_section()
       P2a state.confirmed_facts (热)
       P2b skill_store.build_skills_context(query, top_k=3)  ← v5
       P3  state.working_set (热)
       P4  mid_term.search() → MMR rerank
       P5  hybrid_search: EmbeddingIndex + FTS5+TF-IDF → MMR rerank  ← v5
       P6  environment (if provided)
     → ContextBudgetManager 按优先级分配 token

       │
       ▼
  3. llm_response = call_llm(f"You are an assistant.\n\n{ctx}", user_query)

       │
       ▼
  4. 工具调用（如有）:
     if len(output) > 4000:
         entry = mem.cache_tool_result(user_id, session_id, tool_name, output)
         state.add_tool_result(tool_name, "", ref_id=entry.ref_id)
     else:
         state.add_tool_result(tool_name, output)

       │
       ▼
  5. 工具验证后写入热轨道:
     state.confirm_fact("R1 ping OK", source="tool:ping")
     state.add_to_working_set("R1", "Router R1", "device", {"ip": "10.0.0.1"})

       │
       ▼
  6. 持久化写入冷轨道:
     chunk = mem.remember(user_id, session_id, text)          # 长期
     mem.distill(user_id, session_id, text)                   # 中期
     mem.update_user_profile(user_id, session_id, u, a, tools) # 画像

       │
       ▼
  7. 可选操作:
     if mem.should_consolidate(user_id, session_id):
         mem.consolidate_session(user_id, session_id)          # 历史压缩

       │
       ▼
  8. state.increment_turn()
     返回 llm_response 给用户
```

### 4.2 任务完成后的反思流程（v5 新增）

```
任务成功/失败
       │
       ▼
  1. mem.reflect(user_id, session_id,
                 task="BGP故障排查",
                 outcome="success",
                 summary="通过ping和syslog定位问题",
                 skill_id=sk.skill_id)

       │ 内部并行写入三层：
       ├─▶ LongTermStore: chunk(source="reflection", importance=0.9)
       ├─▶ MidTermStore:  facts(fact_type="lesson")
       └─▶ SkillStore:    record_outcome(success=True)

       │
       ▼
  2. (可选) 下次遇到类似任务时，build_context_budgeted() 自动:
     - 在 P2 注入 Skill（成功率高的优先）
     - 在 P4 召回 lesson facts
     - 在 P5 召回 reflection chunks（高 importance=0.9，排名靠前）
```

### 4.3 Skill 生命周期

```
初次遇到新任务类型
       │
       ▼
  Agent 自主探索（调用工具、调试、验证）
       │ 成功后
       ▼
  mem.save_skill(skill_name, description, steps, tags)
       │
       ▼
  下次遇到相同类型任务
       │
       ▼
  build_context_budgeted() → P2 自动注入 Skill 步骤
       │
       ▼
  Agent 按 Skill 步骤执行（快速、准确、少探索）
       │
       ▼
  mem.reflect(outcome, skill_id) → record_outcome()
  → success_rate 更新（EMA）
  → rate < 0.3 → deprecated（不再注入，等待修正）
```

---

## 5. 关键设计决策

### 5.1 Skills 为什么放在 P2（与 confirmed_facts 并列）

Skills 是经过实际验证的执行路径（`success_rate > 0.3`），可信度等同于工具已确认的事实。相比 mid_term_facts（可能是 LLM 推断的），Skills 有明确的成功/失败历史数据支撑。

将 Skills 放 P3 或更低会导致：在 token 紧张时 Skill 被驱逐，而 Agent 丢失了最有价值的可复用知识——这违反了"有了 Skill 就不用重新探索"的初衷。

### 5.2 语义 Embedding 为什么可插拔

各场景对 embedding 的需求差异极大：

- 纯中文内网场景 → `BAAI/bge-small-zh-v1.5`（本地，快速，免费）
- 多语言场景 → `text-embedding-3-small`（OpenAI，精度高）
- 隔离网络/私有化 → Ollama 兼容 API
- 开发测试/零依赖 → TF-IDF fallback

`EmbeddingIndex` 提供统一接口，后端切换无需修改任何检索逻辑。混合检索（embedding + FTS5）在两种极端都能工作：纯 TF-IDF 时退化为原有行为，有 embedding 时精度提升。

### 5.3 Consolidation 为什么不在每轮自动触发

压缩需要 LLM 调用（或退化为截断），有成本和延迟。在每轮触发会：
- 增加每轮响应延迟
- 消耗额外 LLM token
- 使 session 的 chunk 数量不稳定（影响调试）

推荐策略：每 N 轮检查一次 `should_consolidate()`，在后台异步执行，或在 session 结束时（`end_session` 前）批量压缩。

### 5.4 Reflect 为什么同时写三层

- **长期 chunk（importance=0.9）**：确保反思内容在跨 session 召回时排名靠前，而不是被普通对话淹没。
- **中期 lesson fact**：lesson 的生命周期短于 chunk，可设置 TTL（如 30 天），让过时的教训自动过期。
- **SkillStore（EMA）**：成功率更新是渐进的，单次成功不会让 Skill 从 deprecated 直接变为 active，需要多次验证才能恢复——避免过拟合单次结果。

---

## 6. 与 Generic Agent 的能力对比

| 维度 | agent_memory v5 | Generic Agent (GA) |
|---|---|---|
| **核心哲学** | 生产工程 + 信息密度最大化 | 信息密度最大化（第一性原理） |
| **程序性记忆** | ✅ SkillStore（v5）完整实现 | ✅ Skill Tree（GA 核心特色） |
| **上下文压缩** | ✅ MemoryConsolidator（v5） | ✅ <30K 窗口是 GA 核心竞争力 |
| **反思写入** | ✅ ReflectionEngine（v5） | ✅ Reflect 模式 |
| **语义检索** | ✅ EmbeddingIndex 可插拔（v5） | ➖ 文件系统 + Skill 匹配 |
| **陈述性记忆** | ✅ 完整四层 + 时效 + 重要度 | ➖ 基础存在 |
| **用户行为建模** | ✅ 6维 UserModelEngine（我们独有） | ❌ 无 |
| **Token 预算管理** | ✅ ContextBudgetManager（我们独有） | 隐式（靠压缩控制） |
| **多用户隔离** | ✅ user_id 完整命名空间 | ❌ 个人使用为主 |
| **生产工程** | ✅ WAL、并发安全、TTL、batch | ➖ 极简 3300 行，工程健壮性较弱 |
| **零依赖** | ✅ TF-IDF 默认，可选外部 | ✅ 极简无依赖 |
| **还缺什么** | 知识图谱；多模态感知 | 多用户；工程健壮性 |

**结论**：v5 在记忆维度上已基本达到 GA 的能力水平，在工程健壮性和多用户支持上有明显优势。剩余差距主要在知识图谱（推理能力）和多模态感知。

---

## 7. 生产就绪性说明

### 7.1 支持的场景

| 场景 | 支持 | 说明 |
|---|---|---|
| 单进程多线程 Agent | ✅ | WAL + 写锁，完整支持 |
| 用户规模 < 10,000 | ✅ | TF-IDF LRU 上限可调 |
| 每用户数据 < 50,000 条 | ✅ | SQLite 性能充足 |
| 中文/日文/emoji 内容 | ✅ | 字节 offset 分页 |
| 进程重启数据恢复 | ✅ | 全部 SQLite 持久化 |
| 工具输出 100MB | ✅ | 文件缓存 + 分页读取 |
| 长会话（100+轮） | ✅ | 历史压缩防 token 爆炸 |
| Skill 积累（越用越聪明） | ✅ | 跨 session 技能复用 |

### 7.2 当前限制

| 限制 | 说明 | 升级路径 |
|---|---|---|
| 单机部署 | SQLite 不支持多机共享 | 替换 `_db.py` 为 PostgreSQL |
| TF-IDF 语义 | 不理解同义词 | 配置 `embedding_fn` 即可升级 |
| 无知识图谱 | 实体推理能力弱 | 未来版本 |
| 无多模态 | 纯文本 | 需要多模态 embedding 支持 |

### 7.3 生产部署建议

```python
mem = MemoryManager(
    data_dir="/var/lib/agent_memory",
    llm_fn=my_llm,
    user_model_llm_fn=my_llm_system,
    embedding_fn=my_embed_fn,          # 语义检索
    default_token_budget=3_200,
    consolidate_after_n_turns=30,
    consolidation_keep_recent=10,
    session_ttl=86_400,
    max_index_docs=50_000,
    skill_deprecate_threshold=0.3,
)

# 定期维护（建议每日 cron）
def daily_maintenance(mem, active_users):
    mem.evict_expired_cache()
    mem.mid_term.evict_expired_facts()
    for user_id in active_users:
        mem.long_term.apply_retention(user_id, max_age_days=90)
    mem.short_term.garbage_collect()
    mem.checkpoint()
```

---

## 8. 集成指南

### 8.1 最简集成（3步）

```python
from agent_memory import MemoryManager

mem = MemoryManager(data_dir="./memory_data")

# 每轮对话
mem.remember(user_id, session_id, text)
ctx = mem.build_context(user_id, query, session_id=session_id)
system_prompt = f"You are an assistant.\n\n{ctx}"
```

### 8.2 完整 v5 集成

```python
with MemoryManager(
    data_dir="./memory_data",
    llm_fn=my_llm,
    embedding_fn=my_embed,
) as mem:

    # 任务开始：检索匹配技能
    skills = mem.find_skills(user_id, user_query, top_k=3)
    if skills:
        print(f"复用已有技能: {skills[0].skill_name}")

    # 每轮对话
    state = mem.get_session(user_id, session_id)
    ctx, report = mem.build_context_budgeted(state, user_query)
    response = call_llm(ctx, user_query)

    # 工具调用
    entry = mem.cache_tool_result(user_id, session_id, "tool", output)
    state.add_tool_result("tool", "", ref_id=entry.ref_id)
    state.confirm_fact("验证结论", source="tool:name")

    # 持久化
    chunk = mem.remember(user_id, session_id, text)
    mem.update_user_profile(user_id, session_id, user_text, response)

    # 任务结束：反思 + 保存技能
    mem.reflect(user_id, session_id, task, outcome="success",
                summary=execution_summary, skill_id=skill_id)
    mem.save_skill(user_id, skill_name, description, steps, tags)

    # 可选：压缩历史
    if mem.should_consolidate(user_id, session_id):
        mem.consolidate_session(user_id, session_id)
```

### 8.3 接入真实语义 Embedding

```python
# sentence-transformers（推荐中文场景）
from agent_memory import EmbeddingIndex
idx = EmbeddingIndex.from_sentence_transformer("BAAI/bge-small-zh-v1.5")
mem = MemoryManager(data_dir="./data", embedding_backend=idx._backend)

# OpenAI embedding
mem = MemoryManager(
    data_dir="./data",
    embedding_fn=lambda t: openai.embeddings.create(
        input=[t], model="text-embedding-3-small"
    ).data[0].embedding,
    embedding_dim=1536,
)

# Ollama 本地部署
mem = MemoryManager(
    data_dir="./data",
    embedding_fn=lambda t: requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": t}
    ).json()["embedding"],
    embedding_dim=768,
)
```

---

## 9. API 速查

### 写入

```python
# 基础记忆
chunk  = mem.remember(user_id, session_id, text, source, importance)
chunks = mem.remember_batch(user_id, session_id, texts)
facts  = mem.distill(user_id, session_id, text)
fact   = mem.add_fact(user_id, session_id, fact, fact_type, confidence, ttl_days)
entry  = mem.cache_tool_result(user_id, session_id, tool_name, content)

# 程序性记忆（v5）
skill  = mem.save_skill(user_id, skill_name, description, steps, tags)
rate   = mem.record_skill_outcome(user_id, skill_id, success=True)

# 反思写入（v5）
result = mem.reflect(user_id, session_id, task, outcome, summary,
                     reason, skill_id)

# 用户画像
profile = mem.update_user_profile(user_id, session_id,
                                   user_text, assistant_text, tool_calls)
```

### 热轨道

```python
state = mem.get_session(user_id, session_id)
state.confirm_fact(text, source)        # P2 永不驱逐
state.retract_fact(text)
state.add_to_working_set(id, label, type, metadata)  # P3
state.add_tool_result(tool_name, content, ref_id)     # P5
state.increment_turn()
mem.end_session(user_id, session_id)
```

### 读取

```python
# 上下文构建
ctx, report = mem.build_context_budgeted(state, query, budget=3200,
                                          include_skills=True,
                                          environment="...")
ctx = mem.build_context(user_id, query, session_id)  # 兼容接口

# 技能检索（v5）
skills = mem.find_skills(user_id, query, top_k=3)

# 缓存分页（字节 offset）
page = mem.read_cached(user_id, ref_id, offset=0, length=2000)
token = mem.get_cache_preview(user_id, ref_id)

# 搜索
r = mem.search_chunks(user_id, query, use_embedding=True, use_mmr=True)
r = mem.search_facts(user_id, query, fact_type, min_confidence)
results = mem.search(user_id, query, layers=["long_term", "mid_term"])
```

### 维护

```python
mem.consolidate_session(user_id, session_id)     # 历史压缩（v5）
mem.should_consolidate(user_id, session_id)       # 检查是否需要压缩

mem.evict_expired_cache()                          # 短期 TTL 清理
mem.mid_term.evict_expired_facts()                 # 中期 TTL 清理
mem.long_term.apply_retention(user_id, max_age_days=90)
mem.short_term.garbage_collect()                   # 孤儿文件清理
mem.checkpoint()                                   # WAL 归零
mem.close() / with MemoryManager(...) as mem:
```

### 统计

```python
stats = mem.stats(user_id, session_id)
# 返回: long_term_chunks, sessions, skill_count,
#       embedding_backend, embedding_index_size,
#       mid_term_facts, short_term_entries,
#       active_hot_sessions, hot_track, user_profile
```

---

## 10. 配置参数说明

```python
MemoryManager(
    data_dir       = "./agent_memory_data",

    # LLM 接入
    llm_fn             = None,   # 事实提炼: (prompt: str) -> str
    user_model_llm_fn  = None,   # 特征推断: (system: str, user: str) -> str

    # 语义 Embedding（v5）
    embedding_fn       = None,   # (text: str) -> List[float]
    embedding_dim      = 0,      # embedding 维度（0=自动检测）
    embedding_backend  = None,   # EmbeddingBackend 实例（更高级）

    # Context budget
    tiktoken_fn            = None,   # 精确 token 计数（可选）
    default_token_budget   = 3_200,
    mmr_lambda             = 0.6,    # 1.0=纯相关性, 0.0=纯多样性

    # 短期缓存
    inline_threshold = 4_000,   # 超过此 chars 缓存到文件
    session_ttl      = 86_400,  # 短期缓存 TTL（秒，24h）

    # 中期 facts
    min_fact_confidence = 0.5,  # 低于此丢弃

    # 长期/中期索引
    max_index_docs      = 50_000,
    max_session_indexes = 512,

    # 用户建模
    contradiction_check_interval = 10,
    enable_user_model = True,

    # 历史压缩（v5）
    consolidate_after_n_turns = 30,  # 触发阈值
    consolidation_keep_recent = 10,  # 保留最近 N 条

    # Skill（v5）
    skill_deprecate_threshold = 0.3,  # 低于此成功率 → deprecated
)
```

---

## 11. 升级路径

### 11.1 升级到语义向量检索（推荐首先做）

```bash
pip install sentence-transformers
```

```python
mem = MemoryManager(
    data_dir="./data",
    embedding_fn=lambda t: __import__('sentence_transformers')
        .SentenceTransformer("BAAI/bge-small-zh-v1.5")
        .encode(t, normalize_embeddings=True).tolist()
)
# 或更优雅：
from agent_memory import EmbeddingIndex
backend = EmbeddingIndex.from_sentence_transformer("BAAI/bge-small-zh-v1.5")._backend
mem = MemoryManager(data_dir="./data", embedding_backend=backend)
```

### 11.2 迁移到 PostgreSQL（多实例部署）

只需修改 `stores/_db.py` 中的 `_Pool._open()`，上层代码零改动：

```python
import psycopg2
def _open(self):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    conn.row_factory = sqlite3.Row  # 需适配 psycopg2 DictCursor
    return conn
# SQL 变更: FTS5 → tsvector+GIN, REAL → DOUBLE PRECISION
# INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
```

### 11.3 添加知识图谱（高级推理）

```python
# 未来版本：在 MidTermStore 之上添加图层
class KnowledgeGraphStore:
    """实体-关系图谱，支持推理查询"""
    def add_triple(self, user_id, subject, predicate, object_)
    def query(self, user_id, sparql_like_pattern)
    # Backend: networkx (in-memory) 或 neo4j (production)
```
