# agent_memory — 设计与实现说明书

> Agent 的**长短期记忆**。每个 store 解决一类问题,统一在 `MemoryManager` 后面。
> **铁律**:这个模块**零外部依赖**(SQLite 是 Python 内置,TF-IDF 自实现)。可以独立 pip install,跟 LangChain / OpenAI 不绑定。embedding 是**可选**插件。

---

## 1. 职责

| 文件 | 职责 | 存储 |
|------|------|------|
| `stores/short_term_store.py` | tool result cache(byte-offset 分页)| In-memory dict |
| `stores/mid_term_store.py` | facts — 结构化、TTL、confidence 衰减 | SQLite + FTS5 |
| `stores/long_term_store.py` | chunks — 对话原文、按 recency + importance 召回 | SQLite + FTS5 |
| `stores/skill_store.py` | 复用型 skills(SkillEvolver 生成的)| SQLite + FTS5 |
| `consolidation.py` | 长会话压缩:旧 chunks → LLM rollup 摘要 | 读写 long_term |
| `user_model.py` | 用户技术水平 / 偏好 / 域熟悉度 推断 | 单 JSON / session-tracked |
| `session_state.py` | 单 session 的 confirmed_facts + working_set | In-memory + 持久化 hook |
| `context_budget.py` | priority-budget 装填(`runtime/context_budget.py` 的 lib 化版本)| 无 |
| `retrieval/fact_extractor.py` | LLM 从 turn 文本抽 fact triples | 不存,产 facts 给 mid_term |
| `retrieval/embedding_store.py` | embedding index(TF-IDF / Sentence-Transformer / OpenAI 后端切换) | dense vectors |
| `retrieval/recall_orchestrator.py` | 多 store 召回融合 | 无 |

---

## 2. 公开接口

### 2.1 主入口 `MemoryManager`

```python
from agent_memory import MemoryManager

mm = MemoryManager(
    data_dir="data/",
    embedding_backend=TFIDFBackend(),     # 或 SentenceTransformerBackend / OpenAIBackend
    llm_fn=async_llm_fn,                  # 可选,用于 fact extraction & consolidation
)

# 写入
mm.add_fact(session_id, user_id, fact_text, confidence=0.8)
mm.add_chunk(session_id, user_id, role="user", text="...")

# 召回
chunks = mm.recall_chunks(query, user_id, top_k=5)        # cross-session
facts  = mm.recall_facts(query, session_id, top_k=10)

# 维护
mm.consolidate_session(user_id, session_id)               # rollup 旧 chunks
mm.expire_old_facts()                                     # TTL gc
```

### 2.2 Schemas(数据类型)

```python
from agent_memory.schemas import MemoryFact, MemoryChunk, ToolResultEntry, RetrievalResult

# MemoryFact: id, fact_text, fact_type, confidence (0-1), session_id, user_id, ttl_days, ...
# MemoryChunk: id, role, text, session_id, user_id, importance (0-1), created_at, ...
# RetrievalResult: items[], scores[], backend_used, latency_ms
```

### 2.3 嵌入后端切换

```python
from agent_memory import TFIDFBackend, SentenceTransformerBackend, OpenAIBackend, CallableBackend

backend = OpenAIBackend(api_key="...", model="text-embedding-3-small")
# 或:CallableBackend(lambda texts: embed_via_ollama(texts))
mm = MemoryManager(embedding_backend=backend)
```

### 2.4 User model

```python
from agent_memory.user_model import UserModelEngine

ume = UserModelEngine(data_dir="data/", llm_fn=...)
profile = ume.get_or_create(user_id="alice")
# profile.technical_level ∈ {"novice","intermediate","expert"}
# profile.domains[topic] = familiarity (0-1)
# profile.traits = ["prefers_brevity", "wants_step_by_step", ...]
```

---

## 3. 核心数据流

### 3.1 写入路径(turn 结束后)

```
runtime.loop after-turn hook
   │
   ▼ memory_adapter.after_turn(session_id, user_id, turn_n, ...)
   │
   ├──► 1. add_chunk(role="user", text=user_query)
   ├──► 2. add_chunk(role="assistant", text=synthesis)
   │       └ chunks 写到 long_term + 索引到 embedding store
   │
   ├──► 3. fact_extractor(...turn_text) → LLM → [fact1, fact2, ...]
   │       └ 每个 fact:
   │          - 走 conflict_detector(若 wired)→ equivalent/refinement/contradiction/unrelated
   │          - 直 insert(若未 wired)
   │
   ├──► 4. user_model.observe(user_id, turn_features...)
   │       └ 增量更新 technical_level / domain familiarity
   │
   └──► 5. consolidate_check(turn_n)
           └ turn_n - last_consolidate ≥ threshold → 起 background task
                   └ MemoryManager.consolidate_session(...)
                          └ 旧 chunks 选 N → LLM rollup → 1 个 rollup chunk + 旧的标记 collapsed
```

### 3.2 召回路径(turn 开始前)

```
runtime.loop pre-turn prompt assembly
   │
   ▼ memory_adapter.recall_for_turn(query, user_id, session_id)
   │
   ▼ recall_orchestrator.recall(query, ...)
   │
   ├──► A. 当前 session facts(mid_term where session_id == s)
   ├──► B. cross-session chunks(long_term + embedding rerank,top_k)
   ├──► C. relevant skills(skill_store FTS5 match)
   └──► D. user_model.summarize(user_id)  → 1 行 profile
   │
   ▼ 合并、按 score 排序、过滤(confidence < 0.3 丢)
   │
   ▼ RetrievalResult → ContextBudgetManager 装填 prompt
```

### 3.3 Embedding store 行为

```
add_chunk(text)
   │
   ▼ embedding_backend.embed([text]) → vec
   │
   ▼ EmbeddingIndex.upsert(chunk_id, vec)
       │
       └ TFIDFBackend: 直接写 in-process matrix
         SentenceTransformer: subprocess / API call,batch 化
         OpenAIBackend: HTTP API,带 retry

recall(query, top_k)
   │
   ▼ embedding_backend.embed([query]) → q_vec
   ▼ cosine_similarity(q_vec, all_chunk_vecs)
   ▼ top_k 索引,从 SQLite fetch full chunks
```

### 3.4 Consolidation(长会话压缩)

```
turn_n=120, last_consolidate=90, threshold=30  → 触发
                                                  │
                                                  ▼
                                   选 session_id 的 oldest N chunks(default 50)
                                                  │
                                                  ▼
                                   LLM prompt:
                                     - "structured" (default, Sprint 2):
                                         "请按 5 节输出 — Goal / Progress /
                                          Decisions / Devices / NextSteps"
                                     - "legacy":
                                         "请压缩为简洁摘要(<200字)"
                                                  │
                                                  ▼
                                   add_chunk(role="rollup", text=summary, importance=0.9)
                                                  │
                                                  ▼
                                   旧 chunks 标记 collapsed=True (不再召回但保留 audit)
                                                  │
                                                  ▼
                                   embedding store 索引 rollup,移除 collapsed
```

### 3.4.1 Hermes-style structured rollup(Sprint 2,2026-05)

LLM prompt 默认产 **5 节固定格式** 而非自由 prose:

```
Goal:      <用户的总体目标,1 句话>
Progress:  <已完成的关键步骤,≤ 3 个 bullet>
Decisions: <做出的重要决定(审计相关),≤ 3 个 bullet>
Devices:   <涉及到的设备/服务 ID,逗号分隔>
NextSteps: <尚未完成或后续可做的事,≤ 2 个 bullet>
```

**为什么改**(对照 Hermes `ContextCompressor`):
- **token budget 可预测**:每节有 cap,整体长度稳定
- **审计 grep-friendly**:reviewer 可以直接 `grep "Devices:"` 找设备列表
- **迭代 re-consolidation 基础**:下次 compact 可以**更新**已有 section(下个 Sprint 工作),而不是从零重总结
- **operator 一致体验**:跨 session 同 shape

**migration plan**:
- Sprint 2:default `structured`,`legacy` 通过 `MEMORY_CONSOLIDATION_TEMPLATE=legacy` env 或 `config.yaml:memory.consolidation_template` 回滚
- Sprint 3+:观察 1-2 周生产数据,如 structured 稳定则**删除 legacy template**(简化 codebase)
- 改 prompt 时:**保持 5 节名称稳定**,变 section 内容 cap 即可

**降级**:LLM 不可用时(`llm_fn=None`)直接走 `_fallback_summary`(纯 truncation),template 字段 no-op。不影响 contract。

---

## 4. 关键设计决策

### 4.1 五个 store 为什么不合并?

不同 store **召回模式不同**:
- ShortTerm:byte-offset 分页读(精确 offset)
- MidTerm:FTS5 keyword + confidence 排序
- LongTerm:FTS5 + embedding + recency × importance fusion
- Skill:FTS5 + tags
- UserModel:整对象 by user_id

混在一张 table 里需要 N 套 query 逻辑,各自 schema 冲突。物理分开后每个 store 独立优化 + 测试。

### 4.2 SQLite + FTS5 而非 Postgres / Elasticsearch?

- **零运维**:文件即库,部署一份代码就行
- **够快**:本地 agent 单用户,SQLite 处理 100M 行无压力
- **FTS5 是 SQLite 自带**:不用额外服务
- **测试隔离**:每个测试 `tempfile.mkdtemp()` 一个独立库,没有共享 state

未来如果上 multi-tenant SaaS,把 `_db.py` 替换成 PG dialect 即可。所有 store 用同一套 `_db.get_pool()` 接口。

### 4.3 Embedding 后端 plugin 化

最小生产部署不能依赖 OpenAI(隐私 / 离线 / 成本)。`TFIDFBackend` 是默认,纯 numpy 自实现,2K 个 chunk 召回延迟 < 5ms。

`SentenceTransformerBackend` 是中端(本地模型,质量更好)。`OpenAIBackend` / `CallableBackend(custom_fn)` 给云端。三者实现 `EmbeddingBackend` 协议,store 不知道在跑哪个。

### 4.4 `FactExtractor` 为什么不直接归 MemoryManager?

抽 fact 需要 LLM,但 MemoryManager 应该可以在**没 LLM** 的情况下工作(测试、降级)。所以 `FactExtractor` 作为可选 wired-in 组件:LLM 不可用时跳过抽 fact,只存 raw chunks。

### 4.5 Auto-consolidate 为什么是 deferred wiring?

`MemoryAdapter.set_consolidator(threshold_turns=30)` 是 main.py 启动后调的 setter,不是构造器参数。原因:
- 不同部署模式(dev / prod / pragmatic)阈值不同,config-driven
- consolidate 是 background task,需要 event loop 存在 → 构造时还不一定有

### 4.6 `MemoryFact.confidence` 衰减?

存的时候每个 fact 有 confidence(LLM 抽的时候判定)。两种衰减:
- **时间**:每天 -0.01,30 天后到 0.7 阈值会被 `expire_old_facts()` 标记
- **冲突**:被新 contradiction 推翻时,confidence 直接降到 0.1

不真删,标记 `invalidated_at`,审计可回溯。

---

## 5. 跨模块依赖

```
agent_memory
   │
   ├── (nothing — zero external dep)
   │
   └── 可选 embedding_backend / llm_fn 通过构造注入

外部依赖 agent_memory 的:
   - integrations/adapters/fact_conflict_detector.py
   - integrations/adapters/memory_facts_adapter.py
   - memory/adapter.py            (高层 wrapper,给 runtime 用)
   - retrieval/                   (HybridRetriever 可读 chunks 做 rerank,但有 audit_module_independence 例外)
   - skills/journal_consumer.py   (读 skill_store)
```

### 5.1 扩展点

| 任务 | 改哪里 |
|------|--------|
| 加新 embedding 后端 | `retrieval/embedding_store.py` 新 class 继承 `EmbeddingBackend` |
| 改 consolidation 策略 | `consolidation.py:MemoryConsolidator.consolidate_session` |
| 加新 fact_type | `schemas.py:MemoryFact` 字段已是 str,扩 type 就是约定字符串。recall 端按需过滤 |
| 改 user_model 维度 | `user_model.py:UserProfile` + `observe()` |
| 换底层数据库 | `stores/_db.py:get_pool` —— 改 connection factory,query 用 SQL 标准子集 |

---

## 6. 修改指南

### 6.1 改之前必须知道

- **SQLite 连接池 `_db.py:_Pool`**:全局共享,关心 thread-safety。每个 store 拿 connection 后立刻还,不要持有。
- **FTS5 query 转义**:每个 store 有 `_fts_safe(q)` 函数,把特殊字符(`"`, `*`, `:`)转义。新加 query 必须先过 `_fts_safe`,不然 SQL injection 风险 + FTS5 syntax error。
- **embedding 索引可能 stale**:`add_chunk` 同步写 SQL,异步索引 vector。短窗口内召回看不到刚加的。如果你的逻辑依赖立刻见到,显式 `mm.flush_embeddings()`。
- **per-operator isolation**:每条 chunk/fact 都带 `user_id`。召回必须 filter by user_id —— `tests/test_production_safety.TestMemoryAdapter:test_per_operator_isolation` 强制验证。

### 6.2 改完必须跑

```bash
# Unit tests(快,无 LLM)
python -m unittest agent_memory.tests -v

# Safety isolation
python -m unittest tests.test_production_safety.TestMemoryAdapter -v
python -m unittest tests.test_production_safety.TestAgentMemoryIntegration -v

# Audit:确认没让 agent_memory 依赖外部模块
./scripts/precheck.sh --audits
```

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| Bob 看到 Alice 的 fact | 召回 query 没 filter `user_id` —— grep `recall_facts`/`recall_chunks` caller |
| Embedding 召回返回空 | TFIDFBackend 没 indexed —— 看 `add_chunk` 路径,可能跳过了 embed step |
| FTS5 报 syntax error | query 含 `"` 或 `*` 没过 `_fts_safe` |
| 长会话越来越慢 | consolidate 没触发?`MemoryAdapter._last_consolidate_turn` 看历史。turn_n - last 是否 ≥ threshold |
| Fact confidence 衰减太快 | `mid_term_store._decay_rate` 默认每天 -0.01,改大改小 |

### 6.4 测试结构

```
agent_memory/tests/
   test_memory.py         — MemoryManager e2e 集成
   test_stores_v4.py      — 单 store 单元(short/mid/long)
   test_user_model.py     — user_model 演化
   test_v4_features.py    — TF-IDF / embedding plugin
   test_v5_features.py    — consolidation / reflection / skill store
```

每个测试用 `tempfile.mkdtemp()` 独立 DB,跑完 `shutil.rmtree`。**没有共享 fixture state**。

---

## 7. 已知限制 & TODO

- **无 vector ANN**:TFIDFBackend 是 brute-force cosine。10K+ chunks 后召回 > 50ms。需要时上 hnswlib / faiss(在 EmbeddingBackend 后面加 ANN 索引层)。
- **embedding 没增量重建**:换 backend 时全 rebuild。10K chunks × Sentence-Transformer 要几分钟。
- **consolidation rollup 不可逆**:旧 chunks 标 collapsed 后召回不到。如果 LLM rollup 错过重要信息,无法找回(只能从 raw 表 query)。**生产建议保留 90 天 raw,之后才允许 vacuum**。
- **没有 multi-process 锁**:两个 worker 同时 add_fact 同一 user_id 同一 session 可能产生重复 fact(SQLite 不会 abort,但 fact_hash dedup 失效)。单 worker 模式下 ok。
- **FactConflictDetector 是异步路径**,如果 LLM 慢(8s)写入路径会延后。可以接受(异步 background)但写完 → 立刻召回看不到该 fact。
