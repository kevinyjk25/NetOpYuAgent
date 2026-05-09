"""
retrieval/hybrid.py
-------------------
Hybrid retriever — fuses BM25 and embedding scores with configurable weights.

Why hybrid:
  - BM25 catches exact lexical matches (tool names, IDs, command verbs)
  - Embedding catches paraphrases ("net flow" ≈ "netflow", "查询" ≈ "查看")
  - Weighted sum of the two outperforms either alone on short-text retrieval

Default weights (alpha=0.5) tuned for tool/skill descriptions.
Operators can tune via cfg.retrieval.* without code changes.

Reciprocal Rank Fusion (RRF) is also supported as an alternative fusion
method that doesn't require score normalisation — robust when component
scores are on incomparable scales.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

from .base       import Match, RetrievalResult, Retriever
from .bm25       import BM25Retriever
from .embedding  import EmbeddingRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Late-fusion of BM25 + embedding scores.

    Args:
        embedder:   any object with `.embed(text) -> list[float]` async method.
                    If None, hybrid degrades gracefully to BM25-only.
        bm25_weight: weight on normalised BM25 score in [0, 1]
        embed_weight: weight on cosine-similarity score in [0, 1]
        fusion:     "weighted_sum" (default) or "rrf"  (Reciprocal Rank Fusion)
        rrf_k:      RRF constant; 60 is the literature default
        oversample: each component fetches top_k * oversample candidates
                    before fusion (4x default — improves recall)
    """

    name = "hybrid"

    def __init__(
        self,
        embedder:    Optional[Any]   = None,
        bm25_weight: float           = 0.5,
        embed_weight: float          = 0.5,
        *,
        fusion:      str             = "weighted_sum",
        rrf_k:       int             = 60,
        oversample:  int             = 4,
        bm25_kwargs: Optional[dict]  = None,
        embed_dim:   int             = 768,
    ):
        # Sanity
        if fusion not in ("weighted_sum", "rrf"):
            raise ValueError(f"Unknown fusion mode: {fusion}")
        total = bm25_weight + embed_weight
        if total <= 0:
            raise ValueError("bm25_weight + embed_weight must be > 0")
        # Normalise weights so they sum to 1.0
        self._bw = bm25_weight  / total
        self._ew = embed_weight / total
        self._fusion = fusion
        self._rrf_k  = rrf_k
        self._oversample = max(1, oversample)

        self._bm25 = BM25Retriever(**(bm25_kwargs or {}))
        self._embed: Optional[EmbeddingRetriever] = (
            EmbeddingRetriever(embedder, dim=embed_dim) if embedder else None
        )
        if self._embed is None:
            logger.info(
                "HybridRetriever: no embedder supplied — degrading to BM25-only"
            )
            # When no embedder, embed weight goes to BM25
            self._bw, self._ew = 1.0, 0.0

    # ── Indexing ──────────────────────────────────────────────────────

    def index(self, items: Sequence[dict[str, Any]]) -> None:
        items = list(items)
        self._bm25.index(items)
        if self._embed:
            self._embed.index(items)

    async def index_async(self, items: Sequence[dict[str, Any]]) -> None:
        items = list(items)
        self._bm25.index(items)   # bm25 indexing is sync and fast
        if self._embed:
            await self._embed.index_async(items)

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
        n_candidate = top_k * self._oversample

        bm25_res = self._bm25.retrieve(
            query, n_candidate,
            require_tags=require_tags, exclude_tags=exclude_tags,
        )
        embed_res = (
            self._embed.retrieve(
                query, n_candidate,
                require_tags=require_tags, exclude_tags=exclude_tags,
            ) if self._embed else None
        )

        return self._fuse(query, top_k, bm25_res, embed_res, min_score, t0)

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
        n_candidate = top_k * self._oversample

        bm25_res = self._bm25.retrieve(
            query, n_candidate,
            require_tags=require_tags, exclude_tags=exclude_tags,
        )
        embed_res = None
        if self._embed:
            embed_res = await self._embed.retrieve_async(
                query, n_candidate,
                require_tags=require_tags, exclude_tags=exclude_tags,
            )

        return self._fuse(query, top_k, bm25_res, embed_res, min_score, t0)

    # ── Fusion ────────────────────────────────────────────────────────

    def _fuse(
        self,
        query:       str,
        top_k:       int,
        bm25_res:    RetrievalResult,
        embed_res:   Optional[RetrievalResult],
        min_score:   float,
        t0:          float,
    ) -> RetrievalResult:
        if self._fusion == "rrf":
            fused = self._rrf(bm25_res, embed_res)
        else:
            fused = self._weighted_sum(bm25_res, embed_res)

        fused.sort(key=lambda m: m.score, reverse=True)
        out = [m for m in fused[:top_k] if m.score >= min_score]

        # Pool size = unique IDs across the two component results
        pool = {m.id for m in bm25_res.matches}
        if embed_res:
            pool |= {m.id for m in embed_res.matches}
        # Conservative estimate: at least the larger of the two total_pools
        total_pool = max(
            bm25_res.total_pool,
            embed_res.total_pool if embed_res else 0,
        )

        return RetrievalResult(
            matches=out,
            total_pool=total_pool,
            query=query,
            elapsed_ms=(time.monotonic() - t0) * 1000,
            backend=f"{self.name}({self._fusion})",
        )

    def _weighted_sum(
        self,
        bm25_res: RetrievalResult,
        embed_res: Optional[RetrievalResult],
    ) -> list[Match]:
        """Final score = bw * bm25_norm + ew * embed_norm."""
        # Build score maps
        bm25_map: dict[str, Match] = {m.id: m for m in bm25_res.matches}
        emb_map:  dict[str, Match] = {m.id: m for m in embed_res.matches} if embed_res else {}

        all_ids = set(bm25_map) | set(emb_map)
        out: list[Match] = []
        for tid in all_ids:
            bm25_score = bm25_map[tid].score if tid in bm25_map else 0.0
            emb_score  = emb_map[tid].score  if tid in emb_map  else 0.0
            final      = self._bw * bm25_score + self._ew * emb_score
            item       = (bm25_map.get(tid) or emb_map[tid]).item
            out.append(Match(
                id=tid,
                score=round(final, 4),
                item=item,
                breakdown={
                    "bm25":   round(bm25_score, 4),
                    "embed":  round(emb_score, 4),
                    "weight_bm25":  self._bw,
                    "weight_embed": self._ew,
                },
            ))
        return out

    def _rrf(
        self,
        bm25_res: RetrievalResult,
        embed_res: Optional[RetrievalResult],
    ) -> list[Match]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank_in_each_list))."""
        scores: dict[str, float]    = {}
        items:  dict[str, dict]     = {}
        ranks:  dict[str, dict]     = {}

        def _accumulate(res: RetrievalResult, source: str):
            for rank, m in enumerate(res.matches, start=1):
                scores[m.id] = scores.get(m.id, 0.0) + 1.0 / (self._rrf_k + rank)
                items[m.id] = m.item
                ranks.setdefault(m.id, {})[source] = rank

        _accumulate(bm25_res, "bm25")
        if embed_res:
            _accumulate(embed_res, "embed")

        # Normalise to [0, 1] for downstream consumers (max possible score)
        max_score = max(scores.values()) if scores else 1.0
        return [
            Match(
                id=tid,
                score=round(scores[tid] / max_score, 4),
                item=items[tid],
                breakdown={"rrf_raw": scores[tid], **{f"rank_{k}": v for k, v in ranks[tid].items()}},
            )
            for tid in scores
        ]
