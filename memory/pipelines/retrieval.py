"""
memory/pipelines/retrieval.py
------------------------------
Retrieval pipeline: MMR dedup → recency×relevance blend → token budget trim.

MMR (Maximal Marginal Relevance) removes near-duplicate results from
different tiers that cover the same ground, preventing the LLM context
from being flooded with repetitive information.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from ..schemas import MemoryConfig, MemoryRecord, MemoryTier, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)

_MMR_LAMBDA = 0.6          # relevance vs diversity trade-off (0=diversity, 1=relevance)
_SIMILARITY_THRESHOLD = 0.85  # cosine similarity above which two records are "duplicates"


class RetrievalPipeline:
    """
    Post-processes raw (record, score) pairs from all tiers into a
    ranked, deduplicated, token-trimmed list of RetrievalResult objects.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._cfg = config

    async def process(
        self,
        raw: list[tuple[MemoryRecord, float]],
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        if not raw:
            return []

        # 1. Blend recency + relevance scores
        blended = self._blend_scores(raw, query)

        # 2. MMR deduplication
        deduped = self._mmr_dedup(blended, query.top_k)

        # 3. Token budget trim
        trimmed = self._token_trim(deduped, query.max_tokens)

        # 4. Wrap in RetrievalResult
        return [
            RetrievalResult(record=rec, score=score, tier=rec.tier)
            for rec, score in trimmed
        ]

    # ------------------------------------------------------------------

    def _blend_scores(
        self,
        raw: list[tuple[MemoryRecord, float]],
        query: RetrievalQuery,
    ) -> list[tuple[MemoryRecord, float]]:
        """Combine the store's relevance score with a recency bonus."""
        now = datetime.now(timezone.utc).timestamp()
        out = []
        for rec, relevance in raw:
            try:
                created = datetime.fromisoformat(rec.created_at).timestamp()
            except (ValueError, TypeError):
                created = now

            age_hours  = max(0.0, (now - created) / 3_600)
            recency    = math.exp(-0.1 * age_hours)   # half-life ~7 h

            score = (
                query.relevance_weight * relevance
                + query.recency_weight  * recency
            )
            out.append((rec, round(score, 4)))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _mmr_dedup(
        self,
        ranked: list[tuple[MemoryRecord, float]],
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Greedy MMR selection. Keeps records that are relevant but
        not too similar to already-selected ones.
        """
        if not ranked:
            return []

        selected: list[tuple[MemoryRecord, float]] = []
        remaining = list(ranked)

        while remaining and len(selected) < top_k:
            if not selected:
                # First pick: highest relevance
                best = remaining.pop(0)
                selected.append(best)
                continue

            # MMR score = λ·relevance - (1-λ)·max_sim_to_selected
            best_mmr  = -float("inf")
            best_idx  = 0
            for i, (rec, rel) in enumerate(remaining):
                max_sim = max(
                    self._text_similarity(rec.content, s.content)
                    for s, _ in selected
                )
                mmr = _MMR_LAMBDA * rel - (1 - _MMR_LAMBDA) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _token_trim(
        self,
        records: list[tuple[MemoryRecord, float]],
        max_tokens: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """Drop lowest-scoring records until within token budget."""
        total  = 0
        result = []
        for rec, score in records:
            tokens = max(1, len(rec.content) // 4)
            if total + tokens > max_tokens:
                logger.debug(
                    "Token trim: dropping record_id=%s to stay within %d tokens",
                    rec.record_id, max_tokens,
                )
                continue
            result.append((rec, score))
            total += tokens
        return result

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """
        Lightweight Jaccard similarity on word trigrams.
        Replace with embedding cosine similarity in production.
        """
        def trigrams(text: str) -> set[str]:
            words = text.lower().split()
            return {" ".join(words[i:i+3]) for i in range(len(words) - 2)}

        t_a = trigrams(a)
        t_b = trigrams(b)
        if not t_a or not t_b:
            return 0.0
        intersection = len(t_a & t_b)
        union = len(t_a | t_b)
        return intersection / union if union else 0.0
