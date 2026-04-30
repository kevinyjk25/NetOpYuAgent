"""
agent_memory/retrieval/fact_extractor.py

Hermes-style fact extractor.
Converts raw conversation text into structured MemoryFact objects.

Two modes:
1. LLM-driven: pass an llm_fn callable. Two signatures supported:
     llm_fn(prompt: str) -> str                  # legacy single-prompt
     llm_fn(system: str, user: str) -> str       # split prompt (preferred)
   The system+user form lets the LLM treat the curation rules as a
   stable system instruction and the conversation data as user content,
   which gives much better instruction-following on small models.
2. Rule-based fallback: heuristics when no LLM is configured.
"""
from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Callable, List, Optional

# Relative import — schemas lives in the parent package
from agent_memory.schemas import MemoryFact, MemoryChunk

logger = logging.getLogger(__name__)

_MAX_PROMPT_TEXT_LEN = 6_000   # chars sent to LLM (safety truncation)


# Stable system instruction — kept verbatim across calls so the LLM provider
# can cache it. The dynamic conversation goes in the user message.
_EXTRACT_SYSTEM = """\
You are a memory curator for an IT-operations AI assistant.
Your job: extract concise, reusable facts from a conversation turn so they can be
recalled in future sessions.

Output format
-------------
A JSON array. Each element MUST have:
- "fact":       string, concise self-contained sentence ≤ 60 words, in the SAME
                language as the conversation.
- "fact_type":  one of [preference, entity, procedure, lesson, config, env, general]
                · preference — operator habits revealed by repeated choices
                · entity     — named devices, services, sites, people
                · procedure  — tool patterns / runbook steps that worked
                · lesson     — what worked or failed in a specific incident
                · config     — device/service configuration values
                · env        — site/network topology, environment facts
                · general    — anything else worth keeping
- "confidence": float 0.0–1.0
- "tags":       list of short string tags (≤ 4 items, lowercase, e.g.
                ["radius","ap-01"]) — used as retrieval shortcuts.

Rules
-----
- Only extract facts useful in FUTURE conversations. Skip greetings, fillers,
  acknowledgements, and transient status updates ("checking now…").
- When tool calls are shown, use them as evidence for entity / config / procedure
  facts (the tool output IS the ground truth for what the device looks like).
- Prefer specific over generic ("ap-01 RADIUS timeout 4s" not "device has timeout").
- Return ONLY the JSON array — no markdown fences, no preamble, no explanation.
- If there are no useful facts, return [].

Example 1 (English)
Conversation:
User: Check switch sw-3 config
Tool calls: [{"tool":"get_device_config","args":{"device_id":"sw-3"}}]
Assistant: sw-3 is a Cisco 9300 in dc-east. SNMP polling is enabled. NTP server is 10.0.1.5. RADIUS timeout is 4s (recommend ≤3s).

JSON:
[
  {"fact":"sw-3 is a Cisco 9300 deployed in dc-east","fact_type":"entity","confidence":0.9,"tags":["sw-3","dc-east"]},
  {"fact":"sw-3 NTP server is 10.0.1.5","fact_type":"config","confidence":0.85,"tags":["sw-3","ntp"]},
  {"fact":"sw-3 RADIUS timeout is 4s; recommended ≤3s","fact_type":"config","confidence":0.85,"tags":["sw-3","radius"]}
]

Example 2 (Chinese)
Conversation:
User: 帮我检查 ap-01 配置
Tool calls: [{"tool":"get_device_config","args":{"device_id":"ap-01"}}]
Assistant: ap-01 是 Cisco Catalyst 9115AXI，位于 site-a。SSID corp-wifi VLAN 20 WPA2。RADIUS timeout=4s 建议 ≤3s。

JSON:
[
  {"fact":"ap-01 是 Cisco Catalyst 9115AXI，部署于 site-a","fact_type":"entity","confidence":0.9,"tags":["ap-01","site-a"]},
  {"fact":"ap-01 SSID corp-wifi 使用 VLAN 20 + WPA2 认证","fact_type":"config","confidence":0.85,"tags":["ap-01","ssid"]},
  {"fact":"ap-01 RADIUS timeout 配置为 4s，建议调整为 ≤3s","fact_type":"config","confidence":0.85,"tags":["ap-01","radius"]}
]
"""


# Deep-nudge system prompt — used by periodic review to find facts the
# per-turn distiller missed AND to spot contradictions with existing facts.
# Returns an OBJECT with two arrays so the caller can route corrections to
# HITL review without auto-applying them.
_DEEP_NUDGE_SYSTEM = """\
You are doing a deep review of an IT operator's recent session activity.

You will be given:
  - Recent conversation turns (multiple turns concatenated)
  - Existing curated facts (do NOT duplicate these)

Find:
  1. NEW facts the per-turn distiller missed — especially CROSS-TURN patterns
     (e.g. "operator always checks NTP first before RADIUS", "this site
     consistently has authentication delays after 9pm"). These would not be
     visible from any single turn.
  2. CONTRADICTIONS between existing facts and recent behaviour. Example:
     existing fact says "operator prefers tcpdump", but in the last 5 turns
     they exclusively used wireshark. Flag it for review (don't auto-apply).

Respond with ONLY a JSON object (no fences, no preamble):
{
  "new_facts":     [ {"fact": "...", "fact_type": "...", "confidence": 0.0-1.0, "tags": [...]}, ... ],
  "contradictions":[ {"old_fact": "...", "new_observation": "...", "reason": "..."}, ... ]
}

If nothing new and no contradictions: {"new_facts": [], "contradictions": []}
"""


def _strip_thinking_blocks(raw: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models
    (qwen3-thinking, deepseek-r1, etc.). Even when the LLM engine is supposed
    to suppress them, residual tags can leak through and break JSON parsing."""
    if not raw:
        return raw
    # Multi-line, case-insensitive, also tolerate <Think> / </THINK> variants
    cleaned = re.sub(
        r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE
    )
    return cleaned.strip()


def _salvage_json_array(raw: str) -> Optional[list]:
    """Try multiple strategies to recover a JSON array from a small-model
    response. Small open-source LLMs frequently wrap the array in markdown
    fences, prefix it with 'Sure! Here is...', or trail with explanations.
    Returns None if no valid array can be salvaged."""
    if not raw:
        return None
    # Always strip <think> blocks first — thinking models leak them through.
    raw = _strip_thinking_blocks(raw)
    # Strategy 1: strip common fences and try direct parse.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Strategy 2: locate the first '[' and the matching ']' (greedy).
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # Strategy 3: line-by-line reconstruct (every {...} line is one fact).
    fragments = re.findall(r"\{[^{}]+\}", cleaned, flags=re.DOTALL)
    if fragments:
        items = []
        for frag in fragments:
            try:
                obj = json.loads(frag)
                if isinstance(obj, dict) and "fact" in obj:
                    items.append(obj)
            except json.JSONDecodeError:
                continue
        if items:
            return items
    return None


def _salvage_json_object(raw: str) -> Optional[dict]:
    """Like _salvage_json_array but for object-shaped responses (deep nudge)."""
    if not raw:
        return None
    raw = _strip_thinking_blocks(raw)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


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

    Usage (LLM-driven, single-prompt — legacy):
        def my_llm(prompt: str) -> str: ...
        extractor = FactExtractor(llm_fn=my_llm)

    Usage (LLM-driven, system+user — preferred for small models):
        def my_llm(system: str, user: str) -> str: ...
        extractor = FactExtractor(llm_fn=my_llm)

    The class auto-detects the signature via inspect; both work.

    Usage (rule-based fallback only):
        extractor = FactExtractor()

    Any LLM backend works: OpenAI, Anthropic, Ollama, etc.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[..., str]] = None,
        min_confidence: float = 0.5,
        max_prompt_chars: int = _MAX_PROMPT_TEXT_LEN,
    ) -> None:
        self._llm_fn = llm_fn
        self._min_confidence = min_confidence
        self._max_prompt_chars = max_prompt_chars
        # Detect llm_fn signature once. Two-arg form is preferred (cleaner
        # system/user separation); single-arg form is the legacy interface.
        self._llm_takes_system = self._detect_two_arg(llm_fn)

    @staticmethod
    def _detect_two_arg(fn: Optional[Callable]) -> bool:
        if fn is None:
            return False
        try:
            sig = inspect.signature(fn)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            return len(params) >= 2
        except (TypeError, ValueError):
            return False

    def _call_llm(self, system: str, user: str) -> str:
        """Invoke llm_fn with the right signature; always return a string."""
        if self._llm_fn is None:
            return ""
        try:
            if self._llm_takes_system:
                return self._llm_fn(system, user) or ""
            # Legacy single-prompt: concatenate cleanly so the system block
            # still leads.
            combined = f"{system}\n\n=== Now extract from this turn ===\n{user}"
            return self._llm_fn(combined) or ""
        except Exception as exc:
            logger.warning("FactExtractor: llm_fn raised %s", exc)
            return ""

    def extract(
        self,
        text: str,
        user_id: str,
        session_id: str,
        source_chunk_ids: Optional[List[str]] = None,
        tool_calls: Optional[List[dict]] = None,
    ) -> List[MemoryFact]:
        """Extract facts from a conversation turn.

        `tool_calls` (new): list of {"tool": str, "args": dict, "result": str}
        dicts. When provided, the tool invocations and short result previews
        are included in the LLM context so it can ground entity/config facts
        on the actual tool output (e.g. extract "ap-01 timeout=4s" using the
        `get_device_config` result as evidence)."""
        if not text or not text.strip():
            return []
        if self._llm_fn:
            return self._llm_extract(
                text, user_id, session_id,
                source_chunk_ids or [], tool_calls or [],
            )
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

    # ── deep nudge (periodic cross-turn review with contradiction detection) ──

    def deep_review(
        self,
        recent_text: str,
        existing_facts: List[str],
        user_id: str,
        session_id: str,
    ) -> tuple[List[MemoryFact], List[dict]]:
        """Periodic deep review — finds cross-turn patterns the per-turn
        distiller missed AND flags contradictions with existing facts.

        Returns (new_facts, contradictions). Contradictions are NOT applied
        automatically; the caller routes them to HITL review.
        """
        if not self._llm_fn:
            return [], []
        existing_block = "\n".join(f"- {f}" for f in existing_facts[:20])
        user_content = (
            f"=== Recent turns ===\n{recent_text[:self._max_prompt_chars]}\n\n"
            f"=== Existing facts (do NOT duplicate) ===\n{existing_block}"
        )
        raw = self._call_llm(_DEEP_NUDGE_SYSTEM, user_content)
        data = _salvage_json_object(raw)
        if not data:
            logger.warning(
                "FactExtractor.deep_review: unparseable response, raw[%d]=%r",
                len(raw), raw[:200],
            )
            return [], []
        new_facts = self._items_to_facts(
            data.get("new_facts", []), user_id, session_id, []
        )
        contradictions = data.get("contradictions", []) or []
        if contradictions and not isinstance(contradictions, list):
            contradictions = []
        logger.info(
            "FactExtractor.deep_review: %d new facts, %d contradictions",
            len(new_facts), len(contradictions),
        )
        return new_facts, contradictions

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_tool_calls(tool_calls: List[dict]) -> str:
        """Format tool calls as compact evidence for the prompt. Truncates
        result previews to 200 chars to avoid swamping the LLM with noise."""
        if not tool_calls:
            return "(none)"
        lines = []
        for tc in tool_calls[:5]:   # cap to 5 most relevant
            tool = tc.get("tool", "?")
            args = tc.get("args", {})
            result = tc.get("result", "") or tc.get("output", "")
            args_str = json.dumps(args, ensure_ascii=False)[:160]
            result_str = str(result)[:200].replace("\n", " ")
            lines.append(f"- {tool}({args_str}) → {result_str}")
        return "\n".join(lines)

    def _items_to_facts(
        self,
        items: list,
        user_id: str,
        session_id: str,
        source_chunk_ids: List[str],
    ) -> List[MemoryFact]:
        """Shared dict-list → MemoryFact conversion with confidence/type
        validation. Used by both per-turn extract and deep review."""
        facts: List[MemoryFact] = []
        for item in items:
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
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t)[:40] for t in tags[:6] if t]
            metadata = {"tags": tags} if tags else {}
            facts.append(MemoryFact(
                user_id=user_id, session_id=session_id,
                fact=fact_text, fact_type=fact_type, confidence=confidence,
                source_chunk_ids=source_chunk_ids,
                metadata=metadata,
            ))
        return facts

    def _llm_extract(
        self,
        text: str,
        user_id: str,
        session_id: str,
        source_chunk_ids: List[str],
        tool_calls: List[dict],
    ) -> List[MemoryFact]:
        # Truncate to avoid overflowing LLM context. Reserve some budget
        # for the tool-call block so we don't truncate mid-evidence.
        text_budget = self._max_prompt_chars - 800   # leave room for tool block
        safe_text = text[: max(text_budget, 1000)]
        tool_block = self._format_tool_calls(tool_calls)
        user_content = (
            f"=== Conversation turn ===\n{safe_text}\n\n"
            f"=== Tool calls (use as evidence) ===\n{tool_block}"
        )
        raw = ""
        try:
            raw = self._call_llm(_EXTRACT_SYSTEM, user_content)
            data = _salvage_json_array(raw)
            if data is None:
                logger.warning(
                    "FactExtractor: LLM response was not parseable as a JSON "
                    "array — falling back to rules. raw[%d chars] preview=%r",
                    len(raw), raw[:240],
                )
                return _rule_based_extract(text, user_id, session_id)
            facts = self._items_to_facts(data, user_id, session_id, source_chunk_ids)
            logger.info(
                "FactExtractor: LLM extracted %d facts from %d chars text + %d tool calls "
                "(raw response %d chars, %d items pre-filter)",
                len(facts), len(safe_text), len(tool_calls), len(raw), len(data),
            )
            return facts
        except Exception as exc:
            logger.error(
                "Unexpected error in LLM fact extraction: %s — raw[%d]=%r",
                exc, len(raw), raw[:240], exc_info=True,
            )
            return _rule_based_extract(text, user_id, session_id)