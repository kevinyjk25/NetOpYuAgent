"""
retrieval/cache.py
------------------
LRU + TTL caching wrapper for any Retriever.

Why
---
The runtime loop calls retrieve(query, ...) on EVERY turn even though
`query` is invariant within a single user turn (across LLM iterations).
Even with PERF-1 caching at the loop level, the same query reappears
constantly in normal usage:
  - Multi-turn conversations in the same session
  - Identical follow-up questions ("show me again")
  - Common ops queries reused across operators

A query→RetrievalResult cache turns the second-and-onward call into a
~10μs dict lookup instead of an ~5ms BM25 / ~50ms embedding round-trip.

Design
------
Composition over inheritance: CachedRetriever WRAPS a base retriever
and forwards index() / retrieve() / retrieve_async(). The cache key
includes facets (require/exclude tags) so different filter contexts
don't collide.

Invalidation:
  - index() bumps a generation counter, instantly invalidating all
    cached entries (no need to walk and clear).
  - TTL evicts stale entries lazily on get/put.
  - max_entries cap evicts oldest on overflow (LRU via OrderedDict).

Thread/async safety:
  - All mutation under threading.RLock (sync) — safe for both threads
    and concurrent asyncio tasks since asyncio is single-threaded per
    event loop and the lock is reentrant.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Optional, Sequence

from .base import Match, RetrievalResult, Retriever

logger = logging.getLogger(__name__)


class CachedRetriever(Retriever):
    """LRU + TTL cache wrapper around another Retriever.

    Args:
        inner:        the Retriever to wrap (BM25 / Hybrid / Embedding / …)
        max_entries:  max cached query→result pairs before LRU eviction
        ttl_seconds:  drop entries older than this (None = no time bound)
        case_sensitive: if False (default), queries are normalised
                        (lowercase + whitespace-collapse) before keying

    The wrapped retriever's `name` is preserved with a "+cache" suffix
    so logs / observability still show the underlying engine.
    """

    def __init__(
        self,
        inner:          Retriever,
        max_entries:    int   = 1024,
        ttl_seconds:    Optional[float] = 600.0,
        case_sensitive: bool  = False,
    ):
        self._inner          = inner
        self._max            = int(max_entries)
        self._ttl            = ttl_seconds
        self._case_sensitive = case_sensitive
        # OrderedDict: most-recently-used at the END
        self._cache: OrderedDict[str, tuple[float, RetrievalResult]] = OrderedDict()
        self._lock           = threading.RLock()
        self._generation     = 0   # bumped on index() to invalidate
        # Diagnostics
        self._hits            = 0
        self._misses          = 0
        self._evicted_lru     = 0
        self._evicted_ttl     = 0
        self._evicted_gen     = 0

    @property
    def name(self) -> str:
        return f"{self._inner.name}+cache"

    @property
    def corpus(self) -> "list[dict[str, Any]]":
        """Delegate to inner so prompt-builders can resolve safety-net
        items even when the retriever is cache-wrapped."""
        return self._inner.corpus

    # ── Indexing — invalidates the cache ─────────────────────────────

    def index(self, items: Sequence[dict[str, Any]]) -> None:
        with self._lock:
            self._inner.index(items)
            self._generation += 1
            self._cache.clear()
            logger.info(
                "CachedRetriever: re-indexed %d items, cache cleared (gen=%d)",
                len(items), self._generation,
            )

    async def index_async(self, items: Sequence[dict[str, Any]]) -> None:
        """Async variant — used when the inner retriever (e.g. Embedding)
        has an async indexing path. Falls back to sync index() otherwise."""
        if hasattr(self._inner, "index_async"):
            with self._lock:
                self._generation += 1
                self._cache.clear()
            await self._inner.index_async(items)   # async — don't hold lock
            logger.info(
                "CachedRetriever: async re-indexed %d items, cache cleared (gen=%d)",
                len(items), self._generation,
            )
        else:
            self.index(items)

    # ── Retrieval — sync ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        require_tags:  Optional[Sequence[str]] = None,
        exclude_tags:  Optional[Sequence[str]] = None,
        min_score:     float = 0.0,
    ) -> RetrievalResult:
        key, cached = self._lookup(query, top_k, require_tags, exclude_tags, min_score)
        if cached is not None:
            return cached

        result = self._inner.retrieve(
            query, top_k,
            require_tags=require_tags,
            exclude_tags=exclude_tags,
            min_score=min_score,
        )
        self._store(key, result)
        # Mark as not a cache hit — preserve original backend name + timing
        return replace(result, cache_hit=False)

    # ── Retrieval — async ────────────────────────────────────────────

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        *,
        require_tags:  Optional[Sequence[str]] = None,
        exclude_tags:  Optional[Sequence[str]] = None,
        min_score:     float = 0.0,
    ) -> RetrievalResult:
        key, cached = self._lookup(query, top_k, require_tags, exclude_tags, min_score)
        if cached is not None:
            return cached

        # Prefer async path on the inner retriever
        if hasattr(self._inner, "retrieve_async"):
            result = await self._inner.retrieve_async(
                query, top_k,
                require_tags=require_tags,
                exclude_tags=exclude_tags,
                min_score=min_score,
            )
        else:
            result = self._inner.retrieve(
                query, top_k,
                require_tags=require_tags,
                exclude_tags=exclude_tags,
                min_score=min_score,
            )
        self._store(key, result)
        return replace(result, cache_hit=False)

    # ── Diagnostics ──────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss/eviction counters and current size."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size":        len(self._cache),
                "max_entries": self._max,
                "ttl_s":       self._ttl,
                "generation":  self._generation,
                "hits":        self._hits,
                "misses":      self._misses,
                "hit_rate":    round(hit_rate, 3),
                "evicted_lru": self._evicted_lru,
                "evicted_ttl": self._evicted_ttl,
                "evicted_gen": self._evicted_gen,
                "backend":     self.name,
            }

    def clear(self) -> None:
        """Drop all cached entries without re-indexing (for tests / hot reload)."""
        with self._lock:
            self._cache.clear()

    # ── Internal: keying + lookup ────────────────────────────────────

    def _make_key(
        self,
        query: str,
        top_k: int,
        require_tags: Optional[Sequence[str]],
        exclude_tags: Optional[Sequence[str]],
        min_score: float,
    ) -> str:
        q = query if self._case_sensitive else " ".join(query.lower().split())
        rt = ",".join(sorted(require_tags or []))
        et = ",".join(sorted(exclude_tags or []))
        # Generation included so post-reindex lookups always miss
        return f"g={self._generation}|k={top_k}|q={q}|rt={rt}|et={et}|ms={min_score}"

    def _lookup(
        self,
        query: str,
        top_k: int,
        require_tags: Optional[Sequence[str]],
        exclude_tags: Optional[Sequence[str]],
        min_score: float,
    ) -> tuple[str, Optional[RetrievalResult]]:
        key = self._make_key(query, top_k, require_tags, exclude_tags, min_score)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return key, None
            ts, result = entry
            # TTL check
            if self._ttl is not None and (time.monotonic() - ts) > self._ttl:
                del self._cache[key]
                self._evicted_ttl += 1
                self._misses += 1
                return key, None
            # Hit — promote to MRU
            self._cache.move_to_end(key)
            self._hits += 1
            return key, replace(result, cache_hit=True)

    def _store(self, key: str, result: RetrievalResult) -> None:
        with self._lock:
            self._cache[key] = (time.monotonic(), result)
            self._cache.move_to_end(key)
            # Evict oldest while over cap
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
                self._evicted_lru += 1
