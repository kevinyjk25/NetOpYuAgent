"""
retrieval/bm25.py
-----------------
BM25-Okapi retriever — strong baseline for short-text matching like
tool descriptions and skill purposes. Outperforms pure embedding for
many short-query scenarios.

CJK handling:
  Standard BM25 tokenisers are word-based and don't split Chinese.
  We use a hybrid tokenisation: ASCII via \\b\\w{2,}\\b, CJK via
  per-character + bigram. This is good enough for the network-ops
  domain without bringing jieba as a dependency.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any, Optional, Sequence

from .base import Match, RetrievalResult, Retriever


_CJK_RANGE = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FFF),  # CJK Unified
    (0xAC00, 0xD7AF),  # Hangul
)
_ASCII_TOKEN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]{1,}\b")


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGE)


def tokenize(text: str) -> list[str]:
    """Hybrid tokenisation:
      - ASCII words via word-boundary regex (lowercased)
      - CJK via single-character tokens + adjacent bigrams (covers term-level
        and 2-char compound matches without an external dep)
    """
    if not text:
        return []
    tokens: list[str] = []
    text = text.lower()

    # 1) ASCII tokens
    tokens.extend(_ASCII_TOKEN.findall(text))

    # 2) CJK tokens (per-char + bigrams)
    cjk_chars: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            cjk_chars.append(ch)
            tokens.append(ch)
        else:
            # boundary — flush any accumulated bigrams
            for i in range(len(cjk_chars) - 1):
                tokens.append(cjk_chars[i] + cjk_chars[i + 1])
            cjk_chars.clear()
    # Final flush
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])

    return tokens


class BM25Retriever(Retriever):
    """Okapi BM25 with hybrid CJK tokenisation.

    Default params tuned for short documents (tool descriptions ~50-200 chars):
      k1 = 1.2  — moderate term-saturation
      b  = 0.5  — partial length normalisation (tool descs vary <5x)
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.5):
        self.k1 = k1
        self.b  = b
        self._items:    list[dict[str, Any]] = []
        self._docs:     list[list[str]]      = []   # tokenised
        self._df:       Counter              = Counter()
        self._avg_len:  float                = 0.0
        self._idf:      dict[str, float]     = {}

    # ── Indexing ──────────────────────────────────────────────────────

    def index(self, items: Sequence[dict[str, Any]]) -> None:
        self._items = list(items)
        self._docs  = [tokenize(it.get("text", "")) for it in self._items]

        if not self._docs:
            self._df, self._avg_len, self._idf = Counter(), 0.0, {}
            return

        self._avg_len = sum(len(d) for d in self._docs) / len(self._docs)
        self._df = Counter()
        for doc in self._docs:
            for term in set(doc):
                self._df[term] += 1

        N = len(self._docs)
        self._idf = {
            term: math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            for term, n in self._df.items()
        }

    # ── Scoring ───────────────────────────────────────────────────────

    def _score_one(self, query_terms: list[str], doc_idx: int) -> float:
        doc = self._docs[doc_idx]
        if not doc:
            return 0.0
        doc_len = len(doc)
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            f   = tf.get(term, 0)
            if f == 0:
                continue
            # Okapi BM25 formula
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_len, 1.0))
            score += idf * num / den
        return score

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
        q_terms = tokenize(query)
        if not q_terms or not self._items:
            return RetrievalResult(
                matches=[], total_pool=len(self._items), query=query,
                elapsed_ms=0.0, backend=self.name,
            )

        scored: list[tuple[float, int]] = []
        max_score = 0.0
        for i in range(len(self._items)):
            if not self._passes_facets(self._items[i], require_tags, exclude_tags):
                continue
            s = self._score_one(q_terms, i)
            if s > max_score:
                max_score = s
            scored.append((s, i))

        # Normalise to [0, 1] for downstream fusion
        norm = max_score if max_score > 0 else 1.0
        scored.sort(reverse=True)

        matches: list[Match] = []
        for raw_score, idx in scored[:top_k]:
            normalised = raw_score / norm
            if normalised < min_score:
                continue
            matches.append(Match(
                id=self._items[idx]["id"],
                score=normalised,
                item=self._items[idx],
                breakdown={"bm25_raw": raw_score, "bm25_norm": normalised},
            ))

        return RetrievalResult(
            matches=matches,
            total_pool=len(self._items),
            query=query,
            elapsed_ms=(time.monotonic() - t0) * 1000,
            backend=self.name,
        )
