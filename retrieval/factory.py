"""
retrieval/factory.py
--------------------
Factory + corpus-adapters for production wiring.

build_retriever(cfg, embedder=None) → Retriever
    Returns a Retriever instance based on cfg.retrieval.backend.

build_tool_retriever(cfg, embedder, tool_metadata) → Retriever
    Convenience: build retriever AND index it with the tool metadata
    (output of ToolLoader.build_metadata()).

build_skill_retriever(cfg, embedder, skill_definitions) → Retriever
    Same for skill catalog data.

Corpus adapter helpers convert framework-specific data into the {id, text,
tags, ...} dicts that Retriever.index() expects. Keeping the conversion
in one place means swapping retrievers does not ripple through call-sites.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .base       import Retriever
from .bm25       import BM25Retriever
from .keyword    import KeywordRetriever
from .embedding  import EmbeddingRetriever
from .hybrid     import HybridRetriever
from .cache      import CachedRetriever
from .llm_judge  import LLMJudgeRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retriever(
    cfg: Any,
    embedder: Optional[Any] = None,
    *,
    judge_llm_fn: Optional[Any] = None,
) -> Retriever:
    """Construct a Retriever based on cfg.retrieval.backend.

    Falls back to KeywordRetriever if the requested backend is unavailable.
    """
    backend  = getattr(cfg.retrieval, "backend", "hybrid").lower()
    hyb_cfg  = getattr(cfg.retrieval, "hybrid", None)
    embed_dim = int(getattr(getattr(cfg, "embeddings", None), "dim", 768))

    if backend == "hybrid":
        if embedder is None:
            logger.warning(
                "retrieval.backend=hybrid but no embedder supplied — degrading to BM25"
            )
            return BM25Retriever()
        return HybridRetriever(
            embedder    = embedder,
            bm25_weight = float(getattr(hyb_cfg, "bm25_weight",  0.5)),
            embed_weight= float(getattr(hyb_cfg, "embed_weight", 0.5)),
            fusion      = str  (getattr(hyb_cfg, "fusion",       "weighted_sum")),
            rrf_k       = int  (getattr(hyb_cfg, "rrf_k",        60)),
            oversample  = int  (getattr(hyb_cfg, "oversample",    4)),
            embed_dim   = embed_dim,
        )

    if backend == "bm25":
        return BM25Retriever()

    if backend == "embedding":
        if embedder is None:
            logger.warning(
                "retrieval.backend=embedding but no embedder supplied — degrading to BM25"
            )
            return BM25Retriever()
        return EmbeddingRetriever(embedder, dim=embed_dim)

    if backend == "llm_judge":
        # Two-stage: first-stage Hybrid (or BM25 if no embedder) → LLM judge re-rank.
        # Requires judge_llm_fn (async (system, user) -> str). Falls back to
        # the first stage alone if judge_llm_fn is None.
        first_stage = (
            HybridRetriever(
                embedder    = embedder,
                bm25_weight = float(getattr(hyb_cfg, "bm25_weight",  0.5)),
                embed_weight= float(getattr(hyb_cfg, "embed_weight", 0.5)),
                fusion      = str  (getattr(hyb_cfg, "fusion",       "weighted_sum")),
                rrf_k       = int  (getattr(hyb_cfg, "rrf_k",        60)),
                oversample  = int  (getattr(hyb_cfg, "oversample",    4)),
                embed_dim   = embed_dim,
            ) if embedder else BM25Retriever()
        )
        if judge_llm_fn is None:
            logger.warning(
                "retrieval.backend=llm_judge but no judge_llm_fn supplied — "
                "degrading to first-stage only"
            )
            return first_stage
        # Judge knobs from cfg.retrieval.llm_judge.*  (optional)
        lj_cfg = getattr(cfg.retrieval, "llm_judge", None)
        return LLMJudgeRetriever(
            first_stage,
            llm_fn            = judge_llm_fn,
            first_stage_top_k = int  (getattr(lj_cfg, "first_stage_top_k", 15)) if lj_cfg else 15,
            timeout_seconds   = float(getattr(lj_cfg, "timeout_seconds",  10.0)) if lj_cfg else 10.0,
            fusion_alpha      = float(getattr(lj_cfg, "fusion_alpha",      0.3)) if lj_cfg else 0.3,
            max_text_chars    = int  (getattr(lj_cfg, "max_text_chars",   200)) if lj_cfg else 200,
        )

    if backend == "keyword":
        return KeywordRetriever()

    logger.warning(
        "Unknown retrieval.backend=%r — falling back to BM25", backend
    )
    return BM25Retriever()




def _maybe_wrap_cache(retriever: Retriever, cfg: Any) -> Retriever:
    """Wrap with CachedRetriever when cfg.retrieval.cache.enabled is True.

    Reads:
        cfg.retrieval.cache.enabled       (bool)
        cfg.retrieval.cache.max_entries   (int, default 1024)
        cfg.retrieval.cache.ttl_seconds   (float, default 600.0)
    """
    cache_cfg = getattr(getattr(cfg, "retrieval", None), "cache", None)
    if cache_cfg is None or not getattr(cache_cfg, "enabled", False):
        return retriever
    return CachedRetriever(
        retriever,
        max_entries = int(getattr(cache_cfg, "max_entries", 1024)),
        ttl_seconds = float(getattr(cache_cfg, "ttl_seconds", 600.0)),
    )

# ---------------------------------------------------------------------------
# Corpus adapters
# ---------------------------------------------------------------------------

def tools_to_corpus(tool_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ToolLoader.build_metadata() output into Retriever-ready items.

    Searchable text combines the tool name (lexical anchor), description
    (semantic content), parameter names + descriptions (long-tail terms),
    and tags. This gives BM25 + embedding both lexical and semantic surface
    to match against.
    """
    corpus = []
    for name, info in tool_metadata.items():
        params = info.get("parameters") or {}
        param_text = " ".join(
            f"{p} {d}" for p, d in params.items()
        )
        tags = info.get("tags") or []
        text = " ".join([
            name.replace("_", " "),
            name,                                          # also as-is
            info.get("description", "") or "",
            param_text,
            " ".join(tags),
        ]).strip()
        corpus.append({
            "id":          name,
            "text":        text,
            "description": info.get("description", ""),
            "parameters":  params,
            "tags":        tags,
            "hitl":        bool(info.get("hitl")),
            "returns":     info.get("returns", ""),
        })
    return corpus


def skills_to_corpus(skill_definitions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert SkillCatalog.skill_definitions() into Retriever items.

    Skills have richer structured fields (purpose, tags, parameters,
    description); pulling them all into the searchable text gives both
    lexical and semantic surfaces.
    """
    corpus = []
    for skill_id, defn in skill_definitions.items():
        summary = defn.get("summary", {}) or {}
        detail  = defn.get("detail", {}) or {}
        tags    = list(summary.get("tags") or detail.get("tags") or [])
        text    = " ".join([
            skill_id.replace("_", " "),
            skill_id,
            summary.get("name", "") or "",
            summary.get("purpose", "") or "",
            detail.get("description", "") or "",
            " ".join(tags),
            " ".join((detail.get("parameters") or {}).keys()),
        ]).strip()
        corpus.append({
            "id":          skill_id,
            "text":        text,
            "description": summary.get("purpose") or detail.get("description", ""),
            "tags":        tags,
            "hitl":        bool(summary.get("requires_hitl") or detail.get("requires_hitl")),
            "risk_level":  summary.get("risk_level", "low"),
        })
    return corpus


# ---------------------------------------------------------------------------
# One-shot builders (build + index)
# ---------------------------------------------------------------------------

def build_tool_retriever(
    cfg: Any,
    embedder: Optional[Any],
    tool_metadata: dict[str, dict[str, Any]],
    *,
    judge_llm_fn: Optional[Any] = None,
) -> Retriever:
    """Build a Retriever and index it with tool metadata in one call.
    Wraps with CachedRetriever when cfg.retrieval.cache.enabled.

    judge_llm_fn: required only when cfg.retrieval.backend = 'llm_judge'."""
    r = build_retriever(cfg, embedder, judge_llm_fn=judge_llm_fn)
    r.index(tools_to_corpus(tool_metadata))
    r = _maybe_wrap_cache(r, cfg)
    logger.info(
        "build_tool_retriever: backend=%s indexed %d tools",
        r.name, len(tool_metadata),
    )
    return r


async def build_tool_retriever_async(
    cfg: Any,
    embedder: Optional[Any],
    tool_metadata: dict[str, dict[str, Any]],
    *,
    concurrency: Optional[int] = None,
    judge_llm_fn: Optional[Any] = None,
) -> Retriever:
    """Async batch-indexed tool retriever.

    Use this from FastAPI startup or any async context — it embeds in batches
    with bounded concurrency, avoiding the thread-per-item cost of the sync
    path. Drops indexing time on a 30-tool corpus from ~5-10s to ~1-2s.

    Args:
        concurrency: max in-flight embed() calls. Defaults to
                     cfg.retrieval.embed_index_concurrency (default 8).
    """
    if concurrency is None:
        concurrency = int(getattr(getattr(cfg, "retrieval", None),
                                    "embed_index_concurrency", 8))
    r = build_retriever(cfg, embedder, judge_llm_fn=judge_llm_fn)
    if hasattr(r, "index_async"):
        await r.index_async(tools_to_corpus(tool_metadata), concurrency=concurrency)
    else:
        r.index(tools_to_corpus(tool_metadata))
    r = _maybe_wrap_cache(r, cfg)
    logger.info(
        "build_tool_retriever_async: backend=%s indexed %d tools (concurrency=%d)",
        r.name, len(tool_metadata), concurrency,
    )
    return r


def build_skill_retriever(
    cfg: Any,
    embedder: Optional[Any],
    skill_definitions: dict[str, dict[str, Any]],
    *,
    judge_llm_fn: Optional[Any] = None,
) -> Retriever:
    """Build a Retriever and index it with skill definitions in one call.
    Wraps with CachedRetriever when cfg.retrieval.cache.enabled."""
    r = build_retriever(cfg, embedder, judge_llm_fn=judge_llm_fn)
    r.index(skills_to_corpus(skill_definitions))
    r = _maybe_wrap_cache(r, cfg)
    logger.info(
        "build_skill_retriever: backend=%s indexed %d skills",
        r.name, len(skill_definitions),
    )
    return r


async def build_skill_retriever_async(
    cfg: Any,
    embedder: Optional[Any],
    skill_definitions: dict[str, dict[str, Any]],
    *,
    concurrency:  Optional[int] = None,
    judge_llm_fn: Optional[Any] = None,
) -> Retriever:
    """Async batch-indexed skill retriever — see build_tool_retriever_async."""
    if concurrency is None:
        concurrency = int(getattr(getattr(cfg, "retrieval", None),
                                    "embed_index_concurrency", 8))
    r = build_retriever(cfg, embedder, judge_llm_fn=judge_llm_fn)
    if hasattr(r, "index_async"):
        await r.index_async(skills_to_corpus(skill_definitions), concurrency=concurrency)
    else:
        r.index(skills_to_corpus(skill_definitions))
    r = _maybe_wrap_cache(r, cfg)
    logger.info(
        "build_skill_retriever_async: backend=%s indexed %d skills (concurrency=%d)",
        r.name, len(skill_definitions), concurrency,
    )
    return r
