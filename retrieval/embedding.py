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

    async def index_async(
        self,
        items:        Sequence[dict[str, Any]],
        *,
        concurrency:  int   = 8,
        log_every:    int   = 25,
    ) -> None:
        """Async batch indexing for use during startup inside an event loop.

        Args:
            concurrency: max in-flight embed() calls. Prevents connection-pool
                         exhaustion against remote embedders (Ollama/OpenAI).
                         Tune via cfg.retrieval.embed_index_concurrency.
            log_every:   emit a progress log every N items (0 = silent).

        Failures on individual items log a warning and store a zero-vector
        placeholder so the rest of the index still works. This avoids one bad
        text crashing the entire startup.
        """
        items = list(items)
        n = len(items)
        if n == 0:
            with self._index_lock:
                self._items, self._vectors = [], []
            return

        sem = asyncio.Semaphore(max(1, concurrency))
        results: list[list[float]] = [None] * n  # type: ignore[list-item]
        failed   = 0

        async def _one(i: int):
            nonlocal failed
            text = items[i].get("text", "") or ""
            async with sem:
                try:
                    v = await self._embedder.embed(text)
                    if not v:
                        raise ValueError("empty embedding")
                    norm = sum(x * x for x in v) ** 0.5 or 1.0
                    results[i] = [x / norm for x in v]
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "EmbeddingRetriever: embed failed for id=%r (%s) — using zero vector",
                        items[i].get("id", "<?>"), exc,
                    )
                    results[i] = [0.0] * self._dim
            if log_every > 0 and (i + 1) % log_every == 0:
                logger.info(
                    "EmbeddingRetriever: indexed %d/%d items", i + 1, n
                )

        tasks = [asyncio.create_task(_one(i)) for i in range(n)]
        await asyncio.gather(*tasks)

        with self._index_lock:
            self._items   = items
            self._vectors = results

        logger.info(
            "EmbeddingRetriever: async-indexed %d items (dim=%d, concurrency=%d, failed=%d)",
            n, self._dim, concurrency, failed,
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
