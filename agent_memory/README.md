# Agent Memory Module v3

生产就绪、零外部依赖的 AI Agent 记忆模块。
整合自两个设计来源：**NetOpYuAgent/memory**（Claw + Hermes 记忆架构）和
**NetOpYuAgent/memory/user_model.py**（Hermes Honcho 用户行为建模）。

```
pip install -e .    # 无外部依赖，仅 Python 3.9+ 标准库
```

---

## 架构一览

```
agent_memory/
├── memory_manager.py       # 统一门面（唯一集成入口）
├── user_model.py           # UserModelEngine（v3新增，行为建模）
├── schemas.py              # 数据模型 + 输入验证
├── stores/
│   ├── _db.py              # SQLite连接池（WAL+线程本地+写锁）
│   ├── long_term_store.py  # 长期：FTS5+TF-IDF混合检索
│   ├── mid_term_store.py   # 中期：FTS5+TF-IDF，per-session
│   └── short_term_store.py # 短期：文件缓存+字节offset分页
├── retrieval/
│   ├── vector_store.py     # TF-IDF引擎（LRU，线程安全）
│   └── fact_extractor.py   # 事实提炼（LLM驱动/规则回退）
├── tests/
│   ├── test_memory.py      # 106个测试（存储、并发、安全、性能）
│   └── test_user_model.py  # 39个测试（行为建模）
└── examples/
    └── langchain_integration.py  # 集成示例 + 独立demo
```

### 五层能力对照

| 层 | 类型 | Scope | 实现 | 说明 |
|---|---|---|---|---|
| 短期 | Tool Result Cache | session内 | 文件+SQLite | 字节offset分页，Unicode安全 |
| 中期 | 提炼事实 | session内 | FTS5+TF-IDF | stated/revealed分离 |
| 长期 | 原始Chunks | user跨session | FTS5+TF-IDF | 混合检索，FTS5保留词安全 |
| 跨session | 历史召回 | user全历史 | 同长期 | 无session过滤 |
| **用户建模** | **行为画像** | **user持久** | **SQLite+推理** | **v3新增** |

---

## 快速开始

```python
from agent_memory import MemoryManager

# 上下文管理器确保连接正确关闭、WAL自动checkpoint
with MemoryManager(data_dir="./memory_data") as mem:

    # ── 每轮对话后 ──────────────────────────────────────────────
    chunk = mem.remember("alice", "s1", "BGP session dropped on R1 at 02:15")
    mem.distill("alice", "s1", "Alice prefers Cisco IOS. R1 at 10.0.0.1.")
    mem.update_user_profile(          # NEW: 行为建模
        "alice", "s1",
        user_text="Check ECMP iBGP MPLS paths, I prefer CLI",
        assistant_text="show ip bgp ...",
        tool_calls=[{"tool": "bgp_check"}],
    )

    # ── 缓存大型工具输出 ──────────────────────────────────────────
    entry = mem.cache_tool_result("alice", "s1", "syslog_search", big_output)
    # LLM prompt中注入 reference token：
    token = mem.get_cache_preview("alice", entry.ref_id)
    # → "[STORED:abc123:syslog_search] (12000 chars / 12000 bytes) Preview: ..."
    # LLM按需分页读取（字节offset，中文/emoji安全）：
    page = mem.read_cached("alice", entry.ref_id, offset=0, length=2000)

    # ── 构建LLM system prompt（三层上下文自动融合）────────────────
    ctx = mem.build_context("alice", query="BGP neighbor", session_id="s1")
    # ctx 包含：
    #   [USER PROFILE — inferred from behavior]  ← 行为画像（15%）
    #     Technical level: expert
    #     Preferred tools: bgp_check, syslog_search
    #   ## Distilled Facts (mid-term)            ← 提炼事实（50%）
    #   ## Relevant Memory (long-term)           ← 历史chunks（35%）
    system_prompt = f"You are an IT ops assistant.\n\n{ctx}"
```

---

## UserModelEngine 详解（v3新增）

### 设计来源
来自 `NetOpYuAgent/memory/user_model.py`，实现 Hermes 论文的 §03 用户建模层
（Honcho 系统的"辩证推理"）。整合时修复了原版的4个生产级缺陷。

### 6维行为建模

```python
profile = mem.get_user_profile("alice")

profile.technical_level        # NOVICE / INTERMEDIATE / EXPERT / UNKNOWN
profile.communication_style    # TERSE / BALANCED / VERBOSE / UNKNOWN
profile.domain_counts          # {"network": 15, "auth": 8, "kubernetes": 3}
profile.tool_usage             # {"syslog_search": 12, "bgp_check": 7}
profile.hourly_activity        # {9: 5, 14: 8, 22: 1}  活跃时间分布
profile.traits                 # {"http_library": InferredTrait(value="httpx", conf=0.87)}
```

### Stated vs Revealed（辩证推理）

```python
# 用户说："我总是先写测试"（stated）
# 用户实际：从未调用测试工具（revealed）
# → 矛盾检测：该stated preference被标记为 contradicted=True，不注入prompt

profile.stated_preferences    # 用户明确说出的偏好
profile.revealed_preferences  # 从行为中归纳的偏好
profile.contradictions        # 言行不一的条目列表
```

### 接入LLM做深度特征推断

```python
# 签名：(system_prompt: str, user_content: str) -> str
def my_llm(system: str, user: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}]
    ).choices[0].message.content

mem = MemoryManager(
    data_dir="./data",
    user_model_llm_fn=my_llm,   # 用于特征推断
    llm_fn=my_simple_llm,       # 用于事实提炼（单字符串签名）
)
```

不传 `user_model_llm_fn` 时自动回退到启发式规则（无LLM也能用）。

---

## 生产配置

```python
mem = MemoryManager(
    data_dir="./memory_data",
    # 事实提炼LLM（签名: str -> str）
    llm_fn=None,
    # 用户行为推断LLM（签名: (str, str) -> str）
    user_model_llm_fn=None,
    # 超过此字符数的工具输出缓存到文件
    inline_threshold=4_000,
    # 短期缓存TTL（秒）
    session_ttl=86_400,
    # 低于此置信度的facts丢弃
    min_fact_confidence=0.5,
    # 长期TF-IDF索引上限（LRU eviction）
    max_index_docs=50_000,
    # 中期(user,session)索引对上限（LRU eviction）
    max_session_indexes=512,
    # 每N轮检测一次stated/revealed矛盾
    contradiction_check_interval=10,
    # False则完全禁用UserModelEngine
    enable_user_model=True,
)
```

---

## LangChain 集成

```python
from agent_memory import MemoryManager
from agent_memory.examples.langchain_integration import (
    AgentMemoryCallback, MemoryAugmentedPrompt
)

mem = MemoryManager(data_dir="./memory_data", llm_fn=my_llm)

# 方式1：回调钩子（自动处理所有事件）
callback = AgentMemoryCallback(mem, user_id="alice", session_id="s1")
agent = initialize_agent(tools, llm, callbacks=[callback])

# 构建带记忆的system prompt（在每次call前调用）
system = callback.build_system_prompt("You are an IT ops assistant.", user_query)

# 方式2：手动构建（更灵活）
augmenter = MemoryAugmentedPrompt(mem, user_id="alice", session_id="s1")
system = augmenter.build("You are an IT ops assistant.", user_query)

# 轮换session
callback.new_session("session-002")
```

---

## API 速查

### 写入

| 方法 | 说明 |
|---|---|
| `remember(user_id, session_id, text)` | 存储chunk到长期记忆 |
| `remember_batch(user_id, session_id, texts)` | 批量存储（单事务） |
| `distill(user_id, session_id, text)` | 提炼facts到中期记忆 |
| `add_fact(user_id, session_id, fact, fact_type, confidence)` | 手动添加fact |
| `cache_tool_result(user_id, session_id, tool_name, content)` | 缓存工具输出 |
| `update_user_profile(user_id, session_id, user_text, assistant_text)` | **更新行为画像** |

### 读取

| 方法 | 说明 |
|---|---|
| `read_cached(user_id, ref_id, offset, length)` | 字节offset分页读取 |
| `get_cache_preview(user_id, ref_id)` | 获取[STORED:...]引用token |
| `search(user_id, query, session_id, layers, top_k)` | 多层统一搜索 |
| `search_facts(user_id, query, ...)` | 搜中期facts |
| `search_chunks(user_id, query, ...)` | 搜长期chunks |
| `build_context(user_id, query, session_id, max_chars)` | **构建融合上下文（含画像）** |
| `get_user_profile(user_id)` | **获取UserProfile对象** |
| `get_user_profile_section(user_id)` | **获取画像prompt字符串** |

### 管理

| 方法 | 说明 |
|---|---|
| `list_sessions(user_id)` | 列出用户所有历史session |
| `stats(user_id, session_id)` | 各层统计（含用户画像摘要） |
| `clear_session_cache(user_id, session_id)` | 清理session短期缓存 |
| `evict_expired_cache()` | 原子清理过期缓存（并发安全） |
| `close()` / `with ... as mem:` | 释放连接，WAL checkpoint |
| `checkpoint()` | 手动触发WAL checkpoint |

---

## 运行测试

```bash
# 全量测试（145个）
python -m unittest agent_memory.tests.test_memory agent_memory.tests.test_user_model -v

# 仅核心存储测试（106个）
python -m unittest agent_memory.tests.test_memory -v

# 仅用户建模测试（39个）
python -m unittest agent_memory.tests.test_user_model -v

# 运行集成demo
python -c "
import sys; sys.path.insert(0, '.')
from agent_memory.examples.langchain_integration import demo
demo()
"
```

---

## v3 整合说明

### 两模块对比

| 维度 | 我们的v2 | user_model.py (上传) | v3整合方案 |
|---|---|---|---|
| **信息存取** | ✅ 四层存储+检索 | ❌ 无 | 保留v2全部能力 |
| **用户建模** | ❌ 无 | ✅ 6维行为画像 | 整合user_model设计 |
| **持久化** | ✅ SQLite WAL | ❌ 纯内存dict | 用户画像也持久化到SQLite |
| **用户隔离** | ✅ user_id命名空间 | ❌ session_id索引（碰撞风险） | 改为user_id索引 |
| **线程安全** | ✅ 连接池+写锁 | ❌ 无锁 | 加RLock保护profile |
| **接口类型** | ✅ 同步 | ❌ 纯async | 改为同步，async可选 |
| **LLM集成** | ✅ 单参数fn | ✅ (system,user)双参数fn | 两种签名各司其职 |

### user_model.py 修复的4个CRITICAL问题

1. **无持久化** → profiles 存入 SQLite `user_profiles` 表，进程重启后自动恢复
2. **无用户隔离** → 改按 `user_id`（非 `session_id`）索引，彻底消除跨用户碰撞
3. **非线程安全** → `threading.RLock` 保护所有 profile 读写操作
4. **纯async接口** → 改为同步 `update_profile()`，兼容同步Agent调用链

---

## 依赖

**零外部依赖**，仅 Python 3.9+ 标准库：
`sqlite3`, `pathlib`, `threading`, `hashlib`, `json`, `re`, `math`, `logging`, `dataclasses`

可选升级路径：
- `retrieval/vector_store.py`：TF-IDF → sentence-transformers 或 OpenAI Embeddings
- `stores/_db.py`：SQLite → PostgreSQL（修改连接逻辑）
- `user_model_llm_fn`：传入任意LLM做深度特征推断
