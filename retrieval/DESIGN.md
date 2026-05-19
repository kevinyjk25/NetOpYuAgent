# retrieval — 设计与实现说明书

> 通用**检索框架**。把"按 query 召回 top-K items" 抽象成插件化 retriever。给 tool retrieval / skill retrieval / 任何"语义+关键词混合"的内部需要复用。
> **铁律**:这个模块**不知道** tool / skill / memory 是什么。它只看 `corpus = [{id, text, ...}, ...]`,返回 top-K matches。语义 by `EmbeddingBackend`(从 `agent_memory` 借用接口),关键词 by 自实现 BM25。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `base.py` | 协议:`Retriever`, `Match`, `RetrievalResult` |
| `bm25.py` (185) | 纯 BM25 lexical(支持中英文混合 tokenize)|
| `embedding.py` (231) | 纯向量 cosine retrieval |
| `hybrid.py` (254) | BM25 + Embedding fusion(reciprocal rank fusion 或加权)|
| `keyword.py` | legacy 简单 word-overlap(fallback,几乎不用)|
| `cache.py` (257) | `CachedRetriever` 装饰器,LRU cache query→result |
| `llm_judge.py` (334) | LLM 二次评分排序(slow,可选) |
| `meta_tool.py` (361) | "始终注入"的元工具(list_tools / list_skills / tool_details)|
| `factory.py` (325) | 从 config 构造 retriever,corpus 装配,异步预 index |

---

## 2. 公开接口

### 2.1 协议

```python
from retrieval import Retriever, Match, RetrievalResult

# Retriever 是 abc.ABC:
class Retriever(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, top_k: int = 5) -> RetrievalResult: ...
    
    async def search_async(self, query: str, top_k: int = 5) -> RetrievalResult:
        # default impl: run search() in thread
        ...

# Match: id, text, score, metadata
# RetrievalResult: matches=[Match], backend, latency_ms
```

### 2.2 工厂(主入口)

```python
from retrieval.factory import (
    build_tool_retriever, build_skill_retriever,
    build_tool_retriever_async, build_skill_retriever_async,
    tools_to_corpus, skills_to_corpus,
)

# 同步:
tool_retriever = build_tool_retriever(
    tool_metadata=loader.build_metadata(),     # {name: {description, args, ...}}
    backend="hybrid+cache",                     # "bm25" / "embedding" / "hybrid" / "keyword" / "llm_judge"
    embedding_backend=tfidf_backend,
)

# 异步(并发 embed):
tool_retriever = await build_tool_retriever_async(
    tool_metadata=...,
    backend="hybrid+cache",
    embedding_backend=ollama_embedder,
    concurrency=8,
)

result = tool_retriever.search("show me bgp peers", top_k=5)
for m in result.matches:
    print(m.id, m.score, m.metadata)
```

### 2.3 Meta tools

```python
from retrieval.meta_tool import (
    get_meta_tool_registry,
    make_list_tools_meta_tool, make_list_skills_meta_tool, make_tool_details_meta_tool,
)

reg = get_meta_tool_registry()
reg.register(make_list_tools_meta_tool(tool_loader), always_inject=True)
reg.register(make_list_skills_meta_tool(skill_catalog), always_inject=True)
reg.register(make_tool_details_meta_tool(tool_loader), always_inject=True)

# 在 prompt assembly 时:
for mt in reg.always_inject_iter():
    add_to_system_prompt(mt.signature_block())
# LLM 可以调 list_tools / list_skills / tool_details(name)
```

### 2.4 LLM engine 集成

```python
# main.py:
from integrations.clients.llm_engine import OllamaEngine
engine.attach_retrieval(
    tool_retriever=tool_retriever,
    skill_retriever=skill_retriever,
    meta_tool_registry=reg,
)
# 之后 engine.generate() 每次都会先 retrieve top-K tools+skills,注入 system prompt
```

---

## 3. 核心数据流

### 3.1 Hybrid retrieval

```
query = "如何处理 bgp flap"
                  │
                  ▼
HybridRetriever.search(query, top_k=5)
                  │
                  ├─► BM25Retriever.search(query, top_k=20)
                  │       ↓ 内部 tokenize(中英分流:CJK 单字 + 英文 word)
                  │       ↓ score: idf * tf 归一
                  │       Result: [(id, score), ...] 共 20
                  │
                  ├─► EmbeddingRetriever.search(query, top_k=20)
                  │       ↓ embedding_backend.embed([query]) → q_vec
                  │       ↓ cosine(q_vec, all_corpus_vecs)
                  │       Result: [(id, score), ...] 共 20
                  │
                  ▼ Fusion 合并:
                  │
                  ├─ reciprocal_rank_fusion (默认):
                  │     score(id) = Σ 1/(60 + rank_in_each_retriever)
                  │
                  └─ weighted(可配 bm25_weight=0.5):
                        score = w_bm25 * bm25_norm + (1-w) * embed_norm
                  │
                  ▼ top_k 取前 5
                  ▼
RetrievalResult(matches=[Match(id, score, text, metadata)], backend="hybrid", latency_ms=12.3)
```

### 3.2 Cache layer

```
CachedRetriever(inner=hybrid, max_size=128, ttl_s=300)
                  │
   search(q) ───► hash(q + top_k) lookup
                  │
                  ├─[hit]─► return cached RetrievalResult
                  │
                  └─[miss]► inner.search(q) → store → return
                  
LRU eviction:超过 max_size 删 oldest
TTL gc:每次 lookup 顺便清过期(lazy)
```

### 3.3 LLM judge(可选高质量)

```
LLMJudgeRetriever(inner=hybrid, llm_fn=...)
                  │
   search(q, top_k=5) ───► inner.search(q, top_k=20)  ← 取 20 候选
                  │
                  ▼ LLM prompt:
                  │   "Given query and 20 candidates, rerank by relevance.
                  │    Return JSON [id1, id2, ...]"
                  │
                  ▼ parse JSON
                  ▼ 重排候选,取前 5
                  ▼
RetrievalResult(backend="hybrid+llm_judge", latency_ms=2800 ⚠)
```

慢 10-100x,用在 critical path 之外(比如 nightly batch rerank 优化 corpus weights)。

### 3.4 Corpus 装配

```
tool_metadata = {
    "netflow_dump": {
        "description": "Dump netflow records for a site",
        "args_schema": {...},
        "tags": ["network", "diagnostic"],
    },
    ...
}
                  │
                  ▼
tools_to_corpus(tool_metadata) → [
    {
        "id": "netflow_dump",
        "text": "netflow_dump\nDump netflow records for a site\ntags: network, diagnostic",
        "metadata": {<original entry>}
    },
    ...
]
                  │
                  ▼
BM25Retriever(corpus).search(...) / EmbeddingRetriever(corpus).search(...)
```

`text` 字段是 fuse 多个原数据字段成一段。BM25 / embedding 都 index 这一段。设计:

- **`text` 故意冗余**:tool 名重复在 description 和 tags 中,让"netflow"作为 query 时高 idf
- **`metadata` 留原始**:caller 拿到 match 后能找回 args_schema / etc.

---

## 4. 关键设计决策

### 4.1 为什么自实现 BM25?

- **零依赖**:不引入 `rank_bm25` package
- **中英混合**:`tokenize` 函数对 CJK 字符**逐字切分**(中文无空格分词),英文 word 切。这种混合 tokenizer 现成库都不直接支持。
- **轻量**:185 行,可读,内嵌测试。10K 文档 build index < 100ms。

不做 nltk / jieba —— 单字切分对短 query(tool 名)+ 短 corpus(description)效果足够,jieba 训练 + 加载成本不划算。

### 4.2 Hybrid fusion 默认 RRF

Reciprocal Rank Fusion(`1/(60+rank)`):
- **无超参**(60 是经验常数,不敏感)
- **score 量纲不需对齐**:BM25 是 idf-scaled(0~∞),embedding 是 cosine(-1~1),直接加权要 normalize 麻烦
- **对极端 score 鲁棒**:BM25 偶尔给 50,RRF 把它压成 1/61,不会 dominate

加权融合作为备选(`weighted_fusion=True` config),给确定相对重要性的场景。

### 4.3 `CachedRetriever` 用装饰器模式

```python
hybrid → CachedRetriever(hybrid)  # 同样是 Retriever
```

不动 hybrid 内部,加 cache 是 opt-in。`build_retriever(backend="hybrid+cache")` 自动 wrap。Cache 命中**1us**(dict lookup),miss 走 hybrid(几 ms)。

### 4.4 Meta tools 为什么单独一个文件?

Meta tool 跟普通 tool 不一样:
- **始终注入**(不需要 retrieve)
- **签名** 嵌入 system prompt(让 LLM 知道存在)
- **callable** 实现可以 reach into runtime state(`list_skills` 要看 catalog 实时内容)

放在 retrieval 因为它跟 "tool retriever" 的位置一致 —— 都是 prompt assembly 时给 LLM 的"工具发现"机制。Meta = "总能用的",retrieved = "按需挑出来的"。

### 4.5 异步 build 的必要性

`build_tool_retriever_async` 把 embedding 预 index 并发化:

```
22 tools 顺序 embed:  22 × 200ms = 4.4s
22 tools 并发 8 路:    ceil(22/8) × 200ms ≈ 0.6s
```

启动时间 4.4s → 0.6s。OllamaEmbedder 是 HTTP,完全 IO bound,并发收益线性。

---

## 5. 跨模块依赖

```
retrieval
   │
   ├── agent_memory.retrieval.embedding_store  (EmbeddingBackend 协议,允许复用)
   │     ↑ audit_module_independence 例外:retrieval ↔ agent_memory.retrieval 互不依赖业务,共享 backend lib
   │
   └── (nothing else)

外部依赖 retrieval 的:
   - integrations/clients/llm_engine.py    (retrieval-aware prompt)
   - main.py                               (build_*_retriever_async)
   - skills/journal_consumer.py            (用 BM25 找相似 skills)
```

### 5.1 扩展点

| 任务 | 改哪里 |
|------|--------|
| 加新 retriever backend(ColBERT / cross-encoder) | 新建 `retrieval/<x>.py` 继承 `Retriever`,`factory.build_retriever` 加 case |
| 改 fusion 策略 | `hybrid.py` —— 加 fusion_mode 参数 |
| 中文分词改成 jieba | `bm25.py:tokenize` —— 加 backend 切换 flag,默认保留 CJK 单字 |
| 加 meta tool | `meta_tool.py:make_<name>_meta_tool(...)` 工厂函数,main.py register |
| 改 corpus 字段拼装 | `factory.py:tools_to_corpus` / `skills_to_corpus` |
| Cache 改成 Redis | `cache.py:CachedRetriever` 加 backend 抽象,目前 in-mem dict |

### 5.2 不该在这里加什么

- ❌ Tool/skill 业务逻辑 → `tools/` / `skills/`
- ❌ LLM 调用具体 backend → `integrations/clients/llm_engine.py`
- ❌ Embedding 算法实现 → `agent_memory/retrieval/embedding_store.py`(retrieval 复用)
- ❌ Memory 召回逻辑 → `agent_memory/retrieval/recall_orchestrator.py`

---

## 6. 修改指南

### 6.1 改之前必须知道

- **`tokenize` 测试很微妙**:中英混合 query,断点是空格 + CJK 切换。改之前跑 `pytest -k tokenize`(虽然现在没有)— 至少手测 5 个 mixed query。
- **BM25 idf 平均阈值**:新加 corpus 时,看 idf 分布。如果所有 term 都常见,排序变得无意义 —— corpus 太单一(比如所有 tool description 都含 "netflow")。
- **EmbeddingRetriever 增量索引**:目前 `index(corpus)` 是全量重建。`add(item)` 是 append。如果 corpus 动态变,看 `_vecs` 是 list 还是 ndarray,append 性能不同。
- **Eval 阈值** `recall@3 ≥ 0.40, MRR ≥ 0.30` 在 CI 强制。改 retrieval 必跑 `evaluation/cli.py`,新分降必须解释。

### 6.2 改完必须跑

```bash
# 编译 + audit
./scripts/precheck.sh --audits

# 召回评测(无 LLM,快)
python -m evaluation.cli --golden data/golden_set.jsonl --backend bm25 --top-k 5
python -m evaluation.cli --golden data/golden_set.jsonl --backend hybrid --top-k 5

# 在 v3 build 上现在:
#   bm25:   recall@3=0.90, MRR=0.77
#   hybrid: 略高(语义加成)
# 改后不应低于 -5%
```

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| 中文 query 召回差 | `bm25.tokenize` 是否对该 query 切对了?跑 `tokenize("...")` 看输出 |
| Hybrid 不如 BM25 | embedding backend 是否 init?embed 全 0?TF-IDF 词表小? |
| 启动慢 | `build_tool_retriever_async` 没用 async?embed concurrency 多少? |
| Cache 不生效 | `CachedRetriever` 包了没?key hash(query + top_k)。改 top_k 跳过 cache 正常 |
| Meta tool LLM 不调用 | system prompt 里有 signature?LLM 看到没? |

### 6.4 测试

- **golden_set.jsonl** 是真实评测集(25 cases,中英 / paraphrase / technical)。
- **没有 retrieval 模块单元测试**。建议加 `retrieval/tests/`:
  - `test_bm25.py` —— tokenize edge cases + score monotonicity
  - `test_hybrid.py` —— RRF correctness with mock retrievers
  - `test_cache.py` —— LRU eviction + TTL

---

## 7. 已知限制 & TODO

- **没有 ANN**:embedding brute-force cosine。10K+ corpus 召回 > 50ms。生产规模需要 hnswlib / faiss 接入 `EmbeddingRetriever`。
- **BM25 不支持短语 query**:`"bgp peer down"` 被 tokenize 成 3 个 token,顺序丢失。phrase boost 需要在 `BM25Retriever` 加 windowed 评分。
- **没有 query expansion**:同义词("flap"="flapping"="oscillation")靠 embedding 兜底。如果 embedding 也差,需要 LLM-generated query rewrites(慢)。
- **CJK 单字切分**:对短 query 准,对长描述(中文段落)recall 不高(找不到上下文相关的 5-grams)。需要时加 jieba backend。
- **LLM Judge 没真上线**:成本高,延迟大。当前未在主路径用,只作为离线 corpus 优化工具。
