"""
retrieval/base.py
-----------------
Retriever interface — pluggable engine for selecting top-K items
(tools, skills, memory chunks, …) given a query.

Design principles:
  - Pure interface, no transport assumptions
  - Items are opaque dicts with two reserved keys: id, text
  - Optional facet filters (e.g. require tag=destructive) for safety nets
  - Both sync and async retrieval — implementations override one or both
  - Factory pattern lets cfg.retrieval.backend pick at runtime

Why a separate package:
  - Tool/skill retrieval improvements are now isolated from runtime/loop.py
  - Algorithms can be A/B tested without touching prompt assembly
  - New retrievers (learning-to-rank, LLM-judge) plug in via the same interface
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class Match:
    """A single retrieved item with its score and provenance."""
    id:       str
    score:    float                   # final composite score, [0..1]
    item:     dict[str, Any]          # the original opaque item dict
    breakdown: dict[str, float] = field(default_factory=dict)
    """Per-signal scores (e.g. {'bm25': 0.7, 'embed': 0.55, 'tag_boost': 0.1})
    so callers can debug ranking without re-running retrieval."""


@dataclass
class RetrievalResult:
    """Top-K matches plus diagnostic metadata."""
    matches:    list[Match]
    total_pool: int                          # how many items were searched
    query:      str
    elapsed_ms: float = 0.0
    backend:    str   = ""                   # which Retriever produced this
    cache_hit:  bool  = False


class Retriever(abc.ABC):
    """Abstract retriever — implementations index a corpus and rank against queries.

    The corpus is supplied at construction OR via index() so retrievers can
    be re-built without restarting the agent.
    """

    name: str = "abstract"

    # ── Indexing ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def index(self, items: Sequence[dict[str, Any]]) -> None:
        """Build / replace the index from items.

        Each item must have at least:
          - id   : stable identifier
          - text : the searchable text representation
        Additional fields (tags, hitl, category, …) are kept and exposed
        via Match.item, and may be used by the implementation for facets.
        """

    # ── Retrieval ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        require_tags:  Optional[Sequence[str]] = None,
        exclude_tags:  Optional[Sequence[str]] = None,
        min_score:     float = 0.0,
    ) -> RetrievalResult:
        """Return up to top_k highest-scoring items matching the query.

        Args:
          require_tags: only return items whose 'tags' field contains ALL of these
          exclude_tags: skip items whose 'tags' contains ANY of these
          min_score:    drop items below this score; 0.0 = no filter
        """

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        *,
        require_tags:  Optional[Sequence[str]] = None,
        exclude_tags:  Optional[Sequence[str]] = None,
        min_score:     float = 0.0,
    ) -> RetrievalResult:
        """Default async wrapper — most retrievers are CPU-bound and fine sync.
        Embedding-based retrievers should override to await network calls.
        """
        return self.retrieve(
            query, top_k,
            require_tags=require_tags,
            exclude_tags=exclude_tags,
            min_score=min_score,
        )

    # ── Inspection ────────────────────────────────────────────────────

    @property
    def corpus(self) -> "list[dict[str, Any]]":
        """Return the indexed item list. Default: looks for self._items
        (which all built-in retrievers populate). Subclasses may override
        when they store the corpus elsewhere (e.g. wrappers like
        CachedRetriever delegate to the inner retriever)."""
        return list(getattr(self, "_items", None) or [])

    # ── Helpers (shared) ──────────────────────────────────────────────

    @staticmethod
    def _passes_facets(
        item:         dict[str, Any],
        require_tags: Optional[Sequence[str]],
        exclude_tags: Optional[Sequence[str]],
    ) -> bool:
        item_tags = set(item.get("tags", []) or [])
        if require_tags and not all(t in item_tags for t in require_tags):
            return False
        if exclude_tags and any(t in item_tags for t in exclude_tags):
            return False
        return True
