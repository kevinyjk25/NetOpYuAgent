"""
retrieval/llm_judge.py
----------------------
LLM-as-judge retriever — uses a small/fast LLM to score the relevance
of each candidate item to the query.

Why this exists
---------------
For some workloads (especially cross-lingual or paraphrase-heavy queries
that defeat both BM25 and embedding cosine), an LLM "is X relevant to Y?"
prompt is more accurate than vector similarity. The downside: 1 LLM call
per candidate item is expensive.

Design
------
LLMJudgeRetriever does NOT score every item in the corpus directly. It
sits ON TOP of a fast first-stage retriever (BM25 or Hybrid) and only
re-ranks the top-N candidates. This is the standard "retrieve-then-rerank"
two-stage pattern from production search.

Signature: LLMJudgeRetriever(first_stage=Hybrid(...), llm_fn=async_callable, ...)

  - first_stage.retrieve(query, K*oversample) → candidates
  - llm_fn(prompt) → score per candidate (batched or single)
  - top-K returned by combined first_stage + judge score

The llm_fn protocol is intentionally minimal:
  async def llm_fn(system: str, user: str) -> str
which matches the injected asynchronous judge-callable contract.
This means LLMJudgeRetriever can be wired without any new infrastructure.

NOT auto-registered. To enable in production:
    cfg.retrieval.backend = "llm_judge"
and provide a `judge_llm_fn` to build_retriever() via factory extension.

When NOT to use this
--------------------
- Real-time chat (adds 200-2000ms per retrieve call)
- High-throughput agent loops (LLM rate limits become bottleneck)
- When BM25/Hybrid + cache already gives good recall

Best fit:
- Long-tail queries where exact-keyword retrieval fails
- Multilingual systems where embedding alone is unreliable
- Offline analysis / batch tasks
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional, Sequence

from .base import Match, RetrievalResult, Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default judge prompt
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_SYSTEM = """You are a relevance judge for a tool/skill retrieval system.
Given a USER QUERY and a list of CANDIDATES (id + description), output a JSON
array of {id, score} where score is a float in [0, 1]:
  1.0 = perfectly relevant — operator clearly needs this capability
  0.5 = related but not the primary fit
  0.0 = irrelevant
Output ONLY the JSON array, no other text. No explanation."""


_DEFAULT_JUDGE_USER_TEMPLATE = """USER QUERY: {query}

CANDIDATES:
{candidate_block}

Output JSON array of {{"id": "...", "score": 0.0-1.0}} for ALL candidates above."""


# ---------------------------------------------------------------------------
# LLMJudgeRetriever
# ---------------------------------------------------------------------------

class LLMJudgeRetriever(Retriever):
    """Two-stage retrieve-then-rerank using an LLM as the second-stage judge.

    Args:
        first_stage:   any Retriever (BM25, Hybrid, …) that produces candidates
        llm_fn:        async callable matching:
                          async def llm_fn(system: str, user: str) -> str
                       Should return a JSON array of {id, score}.
        first_stage_top_k:  number of candidates to forward to the judge
                            (default 15 — enough for diversity, fast enough)
        timeout_seconds:    judge call timeout; on timeout falls back to
                            first_stage scores
        fusion_alpha:       0.0 = pure judge score, 1.0 = pure first-stage,
                            in-between blends both. Default 0.3 leans on
                            judge but uses first-stage as tiebreaker.
        judge_system:       override the default system prompt
    """

    name = "llm_judge"

    def __init__(
        self,
        first_stage:        Retriever,
        llm_fn:             Callable[[str, str], Awaitable[str]],
        *,
        first_stage_top_k:  int   = 15,
        timeout_seconds:    float = 10.0,
        fusion_alpha:       float = 0.3,
        judge_system:       Optional[str] = None,
        max_text_chars:     int   = 200,
    ):
        self._first  = first_stage
        self._llm_fn = llm_fn
        self._fs_k   = int(first_stage_top_k)
        self._to     = float(timeout_seconds)
        self._alpha  = max(0.0, min(1.0, float(fusion_alpha)))
        self._sys    = judge_system or _DEFAULT_JUDGE_SYSTEM
        self._max_chars = int(max_text_chars)

    @property
    def corpus(self):
        """Expose the first-stage retriever's corpus so prompt-builders
        can resolve safety-net items through this wrapper."""
        return getattr(self._first, "corpus", None) or []

    # ── Indexing — delegate to the first stage ───────────────────────

    def index(self, items: Sequence[dict[str, Any]]) -> None:
        self._first.index(items)

    async def index_async(self, items: Sequence[dict[str, Any]], **kwargs) -> None:
        if hasattr(self._first, "index_async"):
            await self._first.index_async(items, **kwargs)
        else:
            self._first.index(items)

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
        """Sync retrieve — falls back to first_stage only because we can't
        block on an async LLM call without a running loop. For correct
        re-ranking, use retrieve_async()."""
        logger.debug(
            "LLMJudgeRetriever.retrieve called sync — re-ranking skipped, "
            "returning first_stage result. Use retrieve_async() for full path."
        )
        res = self._first.retrieve(
            query, top_k,
            require_tags=require_tags,
            exclude_tags=exclude_tags,
            min_score=min_score,
        )
        return RetrievalResult(
            matches=res.matches,
            total_pool=res.total_pool,
            query=query,
            elapsed_ms=res.elapsed_ms,
            backend=f"{self.name}(no-rerank)",
        )

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

        # Stage 1: get an oversampled candidate list from first_stage
        if hasattr(self._first, "retrieve_async"):
            candidates_res = await self._first.retrieve_async(
                query, self._fs_k,
                require_tags=require_tags,
                exclude_tags=exclude_tags,
            )
        else:
            candidates_res = self._first.retrieve(
                query, self._fs_k,
                require_tags=require_tags,
                exclude_tags=exclude_tags,
            )

        if not candidates_res.matches:
            return RetrievalResult(
                matches=[],
                total_pool=candidates_res.total_pool,
                query=query,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                backend=self.name,
            )

        # Stage 2: ask the judge
        judge_scores = await self._call_judge(query, candidates_res.matches)

        # Fusion: alpha * first_stage_score + (1 - alpha) * judge_score
        fused: list[Match] = []
        for m in candidates_res.matches:
            j = judge_scores.get(m.id, 0.0)
            final = self._alpha * m.score + (1.0 - self._alpha) * j
            fused.append(Match(
                id=m.id,
                score=round(final, 4),
                item=m.item,
                breakdown={
                    "first_stage": m.score,
                    "judge":       j,
                    "alpha":       self._alpha,
                },
            ))

        fused.sort(key=lambda x: x.score, reverse=True)
        out = [m for m in fused[:top_k] if m.score >= min_score]

        return RetrievalResult(
            matches=out,
            total_pool=candidates_res.total_pool,
            query=query,
            elapsed_ms=(time.monotonic() - t0) * 1000,
            backend=self.name,
        )

    # ── Judge invocation ──────────────────────────────────────────────

    async def _call_judge(
        self,
        query:      str,
        candidates: list[Match],
    ) -> dict[str, float]:
        """Send a single batched LLM call to score all candidates.

        Returns {id: score}. On any error or timeout, returns empty dict
        and the caller falls back to first-stage-only ranking."""
        # Build the candidate block — id + truncated description
        lines = []
        for m in candidates:
            desc = (m.item.get("description") or "")[: self._max_chars]
            lines.append(f"  - id={m.id!r}  desc={desc!r}")
        candidate_block = "\n".join(lines)

        user = _DEFAULT_JUDGE_USER_TEMPLATE.format(
            query=query, candidate_block=candidate_block,
        )

        try:
            raw = await asyncio.wait_for(
                self._llm_fn(self._sys, user), timeout=self._to,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LLMJudgeRetriever: judge timed out after %.1fs — "
                "using first-stage ranking only", self._to,
            )
            return {}
        except Exception as exc:
            logger.warning("LLMJudgeRetriever: judge call failed (%s)", exc)
            return {}

        return self._parse_judge_response(raw, candidates)

    @staticmethod
    def _parse_judge_response(
        raw: str,
        candidates: list[Match],
    ) -> dict[str, float]:
        """Tolerant JSON parser — strips fences, trims, validates."""
        if not raw:
            return {}
        text = raw.strip()
        # Strip ```json ... ``` fences if present
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLMJudgeRetriever: judge response not JSON (%s) — preview=%r",
                exc, text[:200],
            )
            return {}

        valid_ids = {m.id for m in candidates}
        scores: dict[str, float] = {}

        # Format 1 — list of {id, score} objects (the preferred schema):
        #   [{"id": "tool_a", "score": 0.92}, ...]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                tid = item.get("id")
                sc  = item.get("score")
                if tid not in valid_ids:
                    continue
                try:
                    scores[tid] = max(0.0, min(1.0, float(sc)))
                except (TypeError, ValueError):
                    continue
            return scores

        # Format 2 — compact dict {id: score}, common when LLMs paraphrase:
        #   {"tool_a": 0.92, "tool_b": 0.4, ...}
        if isinstance(data, dict):
            for tid, sc in data.items():
                if tid not in valid_ids:
                    continue
                try:
                    scores[str(tid)] = max(0.0, min(1.0, float(sc)))
                except (TypeError, ValueError):
                    continue
            return scores

        logger.warning(
            "LLMJudgeRetriever: judge response is %s, expected list or dict",
            type(data).__name__,
        )
        return {}
