"""
agent_memory/retrieval/fact_extractor.py

Hermes-style fact extractor.
Converts raw conversation text into structured MemoryFact objects.

Two modes:
1. LLM-driven: pass an llm_fn callable (str) -> str. Works with any backend.
2. Rule-based fallback: heuristics when no LLM is configured.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, List, Optional

# Relative import — schemas lives in the parent package
from agent_memory.schemas import MemoryFact, MemoryChunk

logger = logging.getLogger(__name__)

_MAX_PROMPT_TEXT_LEN = 6_000   # chars sent to LLM (safety truncation)

_EXTRACT_PROMPT = """\
You are a memory curator. Extract important, reusable facts from the conversation below.

Output a JSON array. Each element MUST have:
- "fact": string (concise, self-contained, ≤ 60 words)
- "fact_type": one of ["preference","entity","procedure","lesson","config","env","general"]
- "confidence": float 0.0–1.0

Rules:
- Only extract facts useful in future conversations.
- Ignore greetings, filler, and transient context.
- Return ONLY the JSON array. No markdown fences, no preamble.

Conversation:
{text}

JSON array:"""

_PREFERENCE_PATTERNS = [
    re.compile(r"(?:i |the user )(?:prefer|like|want|need|use|always|never)\s+(.{5,100})", re.I),
    re.compile(r"(?:my|the user'?s?) (?:preference|setting|config) (?:is|are)\s+(.{5,100})", re.I),
]
_ENTITY_PATTERNS = [
    re.compile(r"(\w[\w\s\-\.]{2,30}) (?:is|are) (?:located|hosted|running|at|on)\s+(.{3,80})", re.I),
    re.compile(r"(\w[\w\s\-\.]{2,20}) (?:version|v)\s*([\d\.]{1,10})", re.I),
]


def _rule_based_extract(text: str, user_id: str, session_id: str) -> List[MemoryFact]:
    facts: List[MemoryFact] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < 12:
            continue
        matched = False
        for pat in _PREFERENCE_PATTERNS:
            if pat.search(stripped):
                key = stripped[:120]
                if key not in seen:
                    seen.add(key)
                    facts.append(MemoryFact(
                        user_id=user_id, session_id=session_id,
                        fact=stripped[:200], fact_type="preference", confidence=0.55,
                    ))
                matched = True
                break
        if not matched:
            for pat in _ENTITY_PATTERNS:
                if pat.search(stripped):
                    key = stripped[:120]
                    if key not in seen:
                        seen.add(key)
                        facts.append(MemoryFact(
                            user_id=user_id, session_id=session_id,
                            fact=stripped[:200], fact_type="entity", confidence=0.60,
                        ))
                    break
    return facts


class FactExtractor:
    """
    Extract structured facts from conversation text.

    Usage (LLM-driven):
        def my_llm(prompt: str) -> str:
            return openai_client.chat(prompt)

        extractor = FactExtractor(llm_fn=my_llm)

    Usage (rule-based):
        extractor = FactExtractor()

    The llm_fn signature is simply (prompt_str) -> response_str.
    Any LLM backend works: OpenAI, Anthropic, Ollama, etc.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        min_confidence: float = 0.5,
        max_prompt_chars: int = _MAX_PROMPT_TEXT_LEN,
    ) -> None:
        self._llm_fn = llm_fn
        self._min_confidence = min_confidence
        self._max_prompt_chars = max_prompt_chars

    def extract(
        self,
        text: str,
        user_id: str,
        session_id: str,
        source_chunk_ids: Optional[List[str]] = None,
    ) -> List[MemoryFact]:
        if not text or not text.strip():
            return []
        if self._llm_fn:
            return self._llm_extract(text, user_id, session_id, source_chunk_ids or [])
        return _rule_based_extract(text, user_id, session_id)

    def extract_from_chunks(
        self,
        chunks: List[MemoryChunk],
        user_id: str,
        session_id: str,
    ) -> List[MemoryFact]:
        if not chunks:
            return []
        combined = "\n\n".join(c.text for c in chunks)
        chunk_ids = [c.chunk_id for c in chunks]
        return self.extract(combined, user_id, session_id, chunk_ids)

    # ── private ──────────────────────────────────────────────────────────────

    def _llm_extract(
        self,
        text: str,
        user_id: str,
        session_id: str,
        source_chunk_ids: List[str],
    ) -> List[MemoryFact]:
        # Truncate to avoid overflowing LLM context
        safe_text = text[: self._max_prompt_chars]
        prompt = _EXTRACT_PROMPT.format(text=safe_text)
        try:
            raw = self._llm_fn(prompt)
            # Strip markdown fences defensively
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("LLM response is not a JSON array")
            facts: List[MemoryFact] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                fact_text = str(item.get("fact", "")).strip()
                fact_type = str(item.get("fact_type", "general")).strip()
                try:
                    confidence = float(item.get("confidence", 0.7))
                except (TypeError, ValueError):
                    confidence = 0.7
                if not fact_text or confidence < self._min_confidence:
                    continue
                facts.append(MemoryFact(
                    user_id=user_id, session_id=session_id,
                    fact=fact_text, fact_type=fact_type, confidence=confidence,
                    source_chunk_ids=source_chunk_ids,
                ))
            return facts
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("LLM fact extraction failed (%s), falling back to rules", exc)
            return _rule_based_extract(text, user_id, session_id)
        except Exception as exc:
            logger.error("Unexpected error in LLM fact extraction: %s", exc, exc_info=True)
            return _rule_based_extract(text, user_id, session_id)
