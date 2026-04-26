"""
agent_memory/retrieval/embedding_store.py
==========================================
语义向量检索引擎 — 支持多种 Embedding 后端

设计原则：
  同一接口（EmbeddingIndex）支持三种后端：
  1. TFIDFBackend   — 默认，零依赖，兼容现有行为（退化为 TF-IDF）
  2. SentenceTransformerBackend — 本地语义 embedding，需安装 sentence-transformers
  3. OpenAIBackend  — OpenAI/compatible API embedding，需 API key

集成方式：
  # 零依赖默认（与原 TFIDFIndex 完全等价）
  idx = EmbeddingIndex()

  # 本地语义 embedding（安装 sentence-transformers 后）
  idx = EmbeddingIndex.from_sentence_transformer("BAAI/bge-small-zh-v1.5")

  # OpenAI embedding
  idx = EmbeddingIndex.from_openai(api_key="sk-...", model="text-embedding-3-small")

  # 自定义 callable (str) -> List[float]
  idx = EmbeddingIndex.from_callable(my_embed_fn, dim=1024)

  # 与现有 LongTermStore/MidTermStore 集成
  # 在 MemoryManager 初始化时通过 embedding_fn 参数注入
  mem = MemoryManager(data_dir="./data", embedding_fn=my_embed_fn)

语义相似度：
  使用余弦相似度（cosine similarity），范围 [-1, 1]，越高越相似。
  TF-IDF 后端使用 Jaccard + TF-IDF 混合分数，量纲不同但排名行为一致。

MMR（Maximal Marginal Relevance）：
  EmbeddingIndex.query_mmr() 使用 embedding 余弦相似度做多样性去重，
  比 session_state.mmr_rerank() 的 Jaccard 方法语义精度更高。
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_memory.retrieval.vector_store import TFIDFIndex

logger = logging.getLogger(__name__)


# ── Cosine similarity utils ───────────────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, _dot(a, b) / (na * nb)))


# ── Abstract backend ──────────────────────────────────────────────────────────

class EmbeddingBackend:
    """Base class for embedding backends."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class TFIDFBackend(EmbeddingBackend):
    """
    Fallback backend that wraps TFIDFIndex.
    Returns a fake 1-D 'embedding' (the TF-IDF score against a fixed query).
    Used internally by EmbeddingIndex to stay compatible without external deps.
    """
    def embed(self, text: str) -> List[float]:
        # TF-IDF backend doesn't actually embed — handled specially in EmbeddingIndex
        return []

    @property
    def dim(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "tfidf"


class CallableBackend(EmbeddingBackend):
    """Wraps any (str) -> List[float] function as a backend."""

    def __init__(self, fn: Callable[[str], List[float]], dim: int = 0) -> None:
        self._fn = fn
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        return self._fn(text)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"callable(dim={self._dim})"


class SentenceTransformerBackend(EmbeddingBackend):
    """
    sentence-transformers backend.
    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name)
            self._dim   = self._model.get_sentence_embedding_dimension()
            logger.info("SentenceTransformer backend loaded: %s (dim=%d)", model_name, self._dim)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"sentence-transformers"


class OpenAIBackend(EmbeddingBackend):
    """
    OpenAI-compatible embedding backend.
    Works with OpenAI, Azure OpenAI, and any compatible API (e.g., Ollama with nomic-embed).
    Install: pip install openai
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
            self._model  = model
            self._dim    = 0   # resolved on first call
            logger.info("OpenAI embedding backend: model=%s", model)
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        vec  = resp.data[0].embedding
        if self._dim == 0:
            self._dim = len(vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        vecs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        if self._dim == 0 and vecs:
            self._dim = len(vecs[0])
        return vecs

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai({self._model})"


# ── EmbeddingIndex ────────────────────────────────────────────────────────────

class EmbeddingIndex:
    """
    向量检索索引，支持多种 embedding 后端。

    当 backend 为 TFIDFBackend（默认）时，内部使用 TFIDFIndex，
    行为与原有系统完全兼容，无任何额外依赖。

    当 backend 为语义 embedding 时，存储向量并用余弦相似度检索，
    支持 MMR 多样性去重。

    用法示例：
        # 默认 TF-IDF（零依赖）
        idx = EmbeddingIndex()

        # sentence-transformers 语义检索
        idx = EmbeddingIndex.from_sentence_transformer("BAAI/bge-small-zh-v1.5")

        # 自定义 embedding function
        idx = EmbeddingIndex.from_callable(my_fn, dim=1024)

        idx.add("doc1", "BGP session dropped on R1")
        results = idx.query("BGP neighbor went down", top_k=5)
        # → [(doc_id, score), ...]
    """

    def __init__(self, backend: Optional[EmbeddingBackend] = None) -> None:
        self._backend = backend or TFIDFBackend()
        self._lock    = threading.Lock()

        if isinstance(self._backend, TFIDFBackend):
            # Use TFIDFIndex natively for compatibility
            self._tfidf = TFIDFIndex()
            self._vectors: Dict[str, List[float]] = {}
        else:
            self._tfidf   = None
            self._vectors = {}   # doc_id → embedding vector

        logger.debug("EmbeddingIndex initialized with backend: %s", self._backend.name)

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_sentence_transformer(cls, model_name: str = "BAAI/bge-small-zh-v1.5") -> "EmbeddingIndex":
        return cls(SentenceTransformerBackend(model_name))

    @classmethod
    def from_openai(cls, api_key: str, model: str = "text-embedding-3-small",
                    base_url: Optional[str] = None) -> "EmbeddingIndex":
        return cls(OpenAIBackend(api_key=api_key, model=model, base_url=base_url))

    @classmethod
    def from_callable(cls, fn: Callable[[str], List[float]], dim: int = 0) -> "EmbeddingIndex":
        return cls(CallableBackend(fn, dim=dim))

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def size(self) -> int:
        with self._lock:
            if self._tfidf:
                return self._tfidf.size
            return len(self._vectors)

    def add(self, doc_id: str, text: str) -> None:
        with self._lock:
            if self._tfidf:
                self._tfidf.add(doc_id, text)
            else:
                try:
                    self._vectors[doc_id] = self._backend.embed(text)
                except Exception as e:
                    logger.warning("Embedding failed for doc %s: %s — skipping", doc_id[:8], e)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            if self._tfidf:
                self._tfidf.remove(doc_id)
            else:
                self._vectors.pop(doc_id, None)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return [(doc_id, score)] sorted by score desc."""
        with self._lock:
            if self._tfidf:
                return self._tfidf.query(text, top_k=top_k)
            if not self._vectors:
                return []
            try:
                q_vec = self._backend.embed(text)
            except Exception as e:
                logger.warning("Embedding query failed: %s", e)
                return []
            scored = [
                (doc_id, cosine_similarity(q_vec, vec))
                for doc_id, vec in self._vectors.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def query_mmr(
        self,
        text:    str,
        top_k:   int   = 5,
        lambda_: float = 0.6,
    ) -> List[Tuple[str, float]]:
        """
        Maximal Marginal Relevance 检索（semantic embedding 版）。
        比 Jaccard MMR 精度更高，因为相似度计算基于 embedding 余弦距离。
        TF-IDF 后端 fallback 到普通 query（无 MMR 增益）。
        """
        if self._tfidf:
            return self._tfidf.query(text, top_k=top_k)

        with self._lock:
            if not self._vectors:
                return []
            try:
                q_vec = self._backend.embed(text)
            except Exception as e:
                logger.warning("MMR embedding failed: %s", e)
                return []

            # Initial relevance scores
            scored = [
                (doc_id, cosine_similarity(q_vec, vec))
                for doc_id, vec in self._vectors.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            if lambda_ >= 1.0 or len(scored) <= 1:
                return scored[:top_k]

            selected: List[Tuple[str, float]] = []
            remaining = list(scored)

            while remaining and len(selected) < top_k:
                best_score = -float("inf")
                best_item  = remaining[0]
                for doc_id, rel in remaining:
                    vec = self._vectors[doc_id]
                    max_sim = max(
                        (cosine_similarity(vec, self._vectors[s_id])
                         for s_id, _ in selected),
                        default=0.0,
                    )
                    mmr = lambda_ * rel - (1 - lambda_) * max_sim
                    if mmr > best_score:
                        best_score = mmr
                        best_item  = (doc_id, mmr)
                selected.append(best_item)
                remaining = [(d, r) for d, r in remaining if d != best_item[0]]

            return selected

    def clear(self) -> None:
        with self._lock:
            if self._tfidf:
                self._tfidf.clear()
            self._vectors.clear()
