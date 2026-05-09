"""
retrieval/embedding.py
----------------------
Cosine-similarity retriever over an injected Embedder.

Indexes items once (sync via run-async-in-thread or async batch); queries
each call the embedder once and dot-product against cached vectors.

The Embedder protocol matches integrations/embedder.py — duck-typed:
  embedder.embed(text) -> list[float]   (async)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional, Sequence

from .base import Match, RetrievalResult, Retriever

logger = logging.getLogger(__name__)


class EmbeddingRetriever(Retriever):
    """Pure embedding cosine retriever.

    Construction:
        EmbeddingRetriever(embedder, dim=768)

    Indexing strategy:
        - index(items) is sync; embeddings are computed via a background
          event loop helper to avoid clashing with the runtime loop
        - Once indexed, retrieve() is fast (just vector ops)
        - Reindex by calling index() again with new items

    Caveats:
        - Embeddings of short texts (e.g. tool descriptions) are noisier
          than long documents. Hybrid (BM25+embed) usually beats pure embed.
    """

    name = "embedding"

    def __init__(self, embedder: Any, dim: int = 768):
        self._embedder = embedder
        self._dim      = dim
        self._items:   list[dict[str, Any]]   = []
        self._vectors: list[list[float]]      = []
        self._index_lock = threading.Lock()

    # ── Indexing ──────────────────────────────────────────────────────

    def index(self, items: Sequence[dict[str, Any]]) -> None:
        """Build the vector index. Embeds each item synchronously by
        running the async embedder on a thread-owned loop."""
        with self._index_lock:
            self._items = list(items)
            self._vectors = [self._embed_blocking(it.get("text", "")) for it in self._items]
            logger.info(
                "EmbeddingRetriever: indexed %d items (dim=%d)",
                len(self._items), self._dim,
            )

    async def index_async(self, items: Sequence[dict[str, Any]]) -> None:
        """Async batch indexing for use during startup inside an event loop."""
        with self._index_lock:
            self._items = list(items)
            tasks = [self._embedder.embed(it.get("text", "")) for it in self._items]
            self._vectors = await asyncio.gather(*tasks)
            logger.info(
                "EmbeddingRetriever: async-indexed %d items (dim=%d)",
                len(self._items), self._dim,
            )

    def _embed_blocking(self, text: str) -> list[float]:
        """Run an async embedder.embed(text) from sync code without breaking
        the caller's event loop. Uses a one-shot thread + new loop."""
        result_holder: dict[str, Any] = {}
        def _go():
            loop = asyncio.new_event_loop()
            try:
                result_holder["v"] = loop.run_until_complete(self._embedder.embed(text))
            except Exception as exc:
                result_holder["err"] = exc
            finally:
                loop.close()
        t = threading.Thread(target=_go, daemon=True)
        t.start()
        t.join(timeout=30.0)
        if "err" in result_holder:
            logger.warning("EmbeddingRetriever: embed failed (%s) — using zero vector", result_holder["err"])
            return [0.0] * self._dim
        v = result_holder.get("v") or [0.0] * self._dim
        # Normalise (defence in depth — most embedders already do)
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    # ── Retrieval ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        require_tags:  Optional[Sequence[str]] = None,
        exclude_tags:  Optional[Sequence[str]] = None,
        min_score:     float = 0.0,
    ) -> RetrievalResult:
        t0 = time.monotonic()
        if not self._items:
            return RetrievalResult(
                matches=[], total_pool=0, query=query,
                elapsed_ms=0.0, backend=self.name,
            )
        q_vec = self._embed_blocking(query)
        return self._rank(q_vec, query, top_k, require_tags, exclude_tags, min_score, t0)

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        *,
        require_tags:  Optional[Sequence[str]] = None,
        exclude_tags:  Optional[Sequence[str]] = None,
        min_score:     float = 0.0,
    ) -> RetrievalResult:
        t0 = time.monotonic()
        if not self._items:
            return RetrievalResult(
                matches=[], total_pool=0, query=query,
                elapsed_ms=0.0, backend=self.name,
            )
        q_vec = await self._embedder.embed(query)
        norm  = sum(x * x for x in q_vec) ** 0.5 or 1.0
        q_vec = [x / norm for x in q_vec]
        return self._rank(q_vec, query, top_k, require_tags, exclude_tags, min_score, t0)

    def _rank(
        self,
        q_vec: list[float],
        query: str,
        top_k: int,
        require_tags: Optional[Sequence[str]],
        exclude_tags: Optional[Sequence[str]],
        min_score: float,
        t0: float,
    ) -> RetrievalResult:
        scored: list[tuple[float, int]] = []
        for i, item in enumerate(self._items):
            if not self._passes_facets(item, require_tags, exclude_tags):
                continue
            v = self._vectors[i] if i < len(self._vectors) else None
            if not v:
                continue
            # Cosine similarity = dot product of normalised vectors
            sim = sum(a * b for a, b in zip(q_vec, v))
            # Map [-1, 1] → [0, 1]
            scored.append(((sim + 1.0) / 2.0, i))

        scored.sort(reverse=True)
        matches: list[Match] = []
        for sim, idx in scored[:top_k]:
            if sim < min_score:
                continue
            matches.append(Match(
                id=self._items[idx]["id"],
                score=sim,
                item=self._items[idx],
                breakdown={"cosine": sim},
            ))

        return RetrievalResult(
            matches=matches,
            total_pool=len(self._items),
            query=query,
            elapsed_ms=(time.monotonic() - t0) * 1000,
            backend=self.name,
        )
