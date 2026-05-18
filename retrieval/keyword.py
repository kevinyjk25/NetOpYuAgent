"""
retrieval/keyword.py
--------------------
Legacy keyword-overlap retriever — preserves the original SkillCatalog
scoring as a last-resort fallback when no embedder/BM25 is available.

Use this as a safety net, not as the primary retrieval engine.
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional, Sequence

from .base import Match, RetrievalResult, Retriever


_WORD = re.compile(r"\b\w{2,}\b")


class KeywordRetriever(Retriever):
    """Simple Jaccard-like overlap on word sets.

    Score = |query_words ∩ item_words| / max(|query_words|, 1)
    Tag boost: items whose 'tags' contain any query word get +0.15.

    Works well for short English queries; degrades on CJK because
    \\w doesn't split on character boundaries. Use BM25 instead for production.
    """

    name = "keyword"

    def __init__(self, tag_boost: float = 0.15):
        self.tag_boost = tag_boost
        self._items: list[dict[str, Any]] = []
        self._words_per_item: list[set[str]] = []

    def index(self, items: Sequence[dict[str, Any]]) -> None:
        self._items = list(items)
        self._words_per_item = [
            set(_WORD.findall((it.get("text", "") or "").lower()))
            for it in self._items
        ]

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
        q_lower = query.lower()
        q_words = set(_WORD.findall(q_lower))
        if not q_words or not self._items:
            return RetrievalResult(
                matches=[], total_pool=len(self._items),
                query=query, elapsed_ms=0.0, backend=self.name,
            )

        scored: list[tuple[float, int, dict]] = []
        for i, item in enumerate(self._items):
            if not self._passes_facets(item, require_tags, exclude_tags):
                continue
            iw = self._words_per_item[i]
            overlap = len(q_words & iw) / max(len(q_words), 1)
            tag_match = sum(1 for t in (item.get("tags") or []) if t.lower() in q_lower)
            tag_bonus = self.tag_boost if tag_match > 0 else 0.0
            score = round(min(1.0, overlap + tag_bonus), 4)
            scored.append((score, i, {"overlap": overlap, "tag_bonus": tag_bonus}))

        scored.sort(reverse=True, key=lambda x: x[0])
        matches: list[Match] = []
        for score, idx, breakdown in scored[:top_k]:
            if score < min_score:
                continue
            matches.append(Match(
                id=self._items[idx]["id"],
                score=score,
                item=self._items[idx],
                breakdown=breakdown,
            ))

        return RetrievalResult(
            matches=matches,
            total_pool=len(self._items),
            query=query,
            elapsed_ms=(time.monotonic() - t0) * 1000,
            backend=self.name,
        )
