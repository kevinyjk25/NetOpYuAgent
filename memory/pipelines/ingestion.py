"""
memory/pipelines/ingestion.py
------------------------------
Ingestion pipeline: tokenise → score importance → embed → build MemoryRecord.

Importance scoring uses a fast heuristic by default.
Swap ``_score_importance_llm()`` in for production if you want LLM-based scoring.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from ..schemas import MemoryConfig, MemoryRecord, MemoryRecordType, MemoryTier

logger = logging.getLogger(__name__)

# Keywords that boost importance for IT ops domain
_HIGH_IMPORTANCE_PATTERNS = re.compile(
    r"\b(P0|P1|critical|outage|incident|restart|rollback|alert|pagerduty|"
    r"production|prod|sev\d|severity|down|failure|anomaly)\b",
    re.IGNORECASE,
)
_LOW_IMPORTANCE_PATTERNS = re.compile(
    r"\b(hello|thanks|ok|sure|yes|no|bye|ping|test)\b",
    re.IGNORECASE,
)


class IngestionPipeline:
    """
    Converts raw text into a fully-populated MemoryRecord.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._cfg = config
        self._embedder = _EmbedderStub(config.embedding_model)

    async def process(
        self,
        content: str,
        session_id: str,
        record_type: MemoryRecordType,
        metadata: Optional[dict] = None,
        importance_override: Optional[float] = None,
    ) -> MemoryRecord:
        importance = (
            importance_override
            if importance_override is not None
            else self._score_importance_heuristic(content, record_type)
        )

        # Embed only if it might reach mid-term or above
        embedding = None
        if importance >= 0.4:
            embedding = await self.embed(content)

        record = MemoryRecord(
            session_id=session_id,
            record_type=record_type,
            tier=MemoryTier.SHORT_TERM,   # default; router overrides per store
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            importance=importance,
        )
        logger.debug(
            "Ingestion: type=%s importance=%.2f session=%s",
            record_type.value, importance, session_id,
        )
        return record

    async def embed(self, text: str) -> list[float]:
        """Embed text. Stub returns a deterministic pseudo-vector."""
        return await self._embedder.embed(text)

    # ------------------------------------------------------------------

    @staticmethod
    def _score_importance_heuristic(
        content: str, record_type: MemoryRecordType
    ) -> float:
        score = 0.5

        # Record type baseline
        type_boosts = {
            MemoryRecordType.ENTITY:      0.25,
            MemoryRecordType.INCIDENT:    0.30,
            MemoryRecordType.USER_PREF:   0.25,
            MemoryRecordType.TASK_RESULT: 0.15,
            MemoryRecordType.SUMMARY:     0.10,
        }
        score += type_boosts.get(record_type, 0.0)

        # Keyword signals
        high_matches = len(_HIGH_IMPORTANCE_PATTERNS.findall(content))
        low_matches  = len(_LOW_IMPORTANCE_PATTERNS.findall(content))
        score += min(0.20, high_matches * 0.05)
        score -= min(0.20, low_matches  * 0.05)

        # Length signal: very short = likely low value
        if len(content) < 30:
            score -= 0.10

        return round(min(1.0, max(0.0, score)), 3)


# ---------------------------------------------------------------------------
# Embedder stub (replace with real OpenAI / local model call)
# ---------------------------------------------------------------------------

class _EmbedderStub:
    """
    Deterministic pseudo-embedder for dev/test.
    Replace with:

        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector = await embeddings.aembed_query(text)
    """
    DIM = 1536

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        base   = [b / 255.0 for b in digest]
        # Tile to DIM
        vec = (base * (self.DIM // len(base) + 1))[: self.DIM]
        # Normalise
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
