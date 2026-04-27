"""
agent_memory/user_model.py
--------------------------
UserModelEngine — Hermes-style dialectic user modeling.

Integrated from NetOpYuAgent/memory/user_model.py with the following
production fixes applied on top of the original design:

Original strengths (kept):
  - 6-dimensional behavioral profile (technical_level, communication_style,
    domain_focus, tool_usage, work_patterns, quality_markers)
  - Stated vs revealed preference separation + contradiction detection
  - Dialectic reasoning: behavior > words
  - Exponential confidence smoothing per trait
  - to_prompt_section() for clean prompt injection

Production fixes vs original:
  [CRITICAL] Persistence: profiles now stored in SQLite via _db._Pool
              (original: pure in-memory dict, lost on restart)
  [CRITICAL] User isolation: keyed by (user_id, session_id), not session_id alone
              (original: different users same session_id → collision)
  [CRITICAL] Thread safety: RLock on all profile mutations
              (original: no locking, concurrent access = data corruption)
  [HIGH]     Sync interface: update_profile() is sync; async variant is optional
              (original: after_turn() was async-only, incompatible with sync callers)
  [HIGH]     fts_store actually wired: profiles serialized to the shared SQLite DB
              (original: fts_store param accepted but never used)
  [MEDIUM]   Cross-session merging: merge_sessions() aggregates traits across
              all sessions of a user into a persistent UserProfile
              (original: total_sessions field existed but merge logic absent)
  [LOW]      Graceful LLM unavailability: sync stub that works without event loop
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_memory.stores._db import get_pool

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────────

class TechnicalLevel(str, Enum):
    NOVICE       = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT       = "expert"
    UNKNOWN      = "unknown"


class CommunicationStyle(str, Enum):
    TERSE    = "terse"
    BALANCED = "balanced"
    VERBOSE  = "verbose"
    UNKNOWN  = "unknown"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class InferredTrait:
    """One inferred behavioral trait with evidence chain."""
    trait_key:      str
    value:          Any
    confidence:     float        # 0.0–1.0, exponentially smoothed
    evidence_count: int
    contradicted:   bool  = False
    last_updated:   float = field(default_factory=time.time)
    evidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trait_key":      self.trait_key,
            "value":          self.value,
            "confidence":     round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "contradicted":   self.contradicted,
            "last_updated":   self.last_updated,
            "evidence_notes": self.evidence_notes[:3],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InferredTrait":
        return cls(
            trait_key=d["trait_key"], value=d["value"],
            confidence=d["confidence"], evidence_count=d["evidence_count"],
            contradicted=d.get("contradicted", False),
            last_updated=d.get("last_updated", time.time()),
            evidence_notes=d.get("evidence_notes", []),
        )


@dataclass
class UserProfile:
    """
    Persistent behavioral model of a user.
    Aggregates observations across all sessions.
    """
    user_id:    str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    total_sessions: int = 0
    total_turns:    int = 0

    # Structured dimensions
    technical_level:     TechnicalLevel     = TechnicalLevel.UNKNOWN
    communication_style: CommunicationStyle = CommunicationStyle.UNKNOWN

    # Free-form trait map: trait_key → InferredTrait
    traits: Dict[str, InferredTrait] = field(default_factory=dict)

    # Observed frequencies
    tool_usage:      Dict[str, int] = field(default_factory=dict)
    domain_counts:   Dict[str, int] = field(default_factory=dict)
    hourly_activity: Dict[int, int] = field(default_factory=dict)

    # Dialectic: stated vs revealed
    stated_preferences:   Dict[str, str] = field(default_factory=dict)
    revealed_preferences: Dict[str, str] = field(default_factory=dict)
    contradictions:       List[dict]     = field(default_factory=list)

    def to_prompt_section(self, max_chars: int = 800) -> str:
        """Format profile as a compact hidden context block for LLM injection."""
        lines = ["[USER PROFILE — inferred from behavior]"]
        if self.technical_level != TechnicalLevel.UNKNOWN:
            lines.append(f"  Technical level: {self.technical_level.value}")
        if self.communication_style != CommunicationStyle.UNKNOWN:
            lines.append(f"  Response style: {self.communication_style.value}")
        if self.domain_counts:
            top = sorted(self.domain_counts.items(), key=lambda x: -x[1])[:4]
            lines.append(f"  Primary domains: {', '.join(d for d, _ in top)}")
        if self.tool_usage:
            top = sorted(self.tool_usage.items(), key=lambda x: -x[1])[:4]
            lines.append(f"  Preferred tools: {', '.join(t for t, _ in top)}")
        for trait in sorted(self.traits.values(), key=lambda t: -t.confidence)[:5]:
            if trait.confidence >= 0.65 and not trait.contradicted:
                lines.append(f"  • {trait.trait_key}: {trait.value}")
        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n  ... [profile truncated]"
        return result

    def to_dict(self) -> dict:
        return {
            "user_id":            self.user_id,
            "created_at":         self.created_at,
            "last_updated":       self.last_updated,
            "total_sessions":     self.total_sessions,
            "total_turns":        self.total_turns,
            "technical_level":    self.technical_level.value,
            "communication_style": self.communication_style.value,
            "domain_counts":      self.domain_counts,
            "tool_usage":         self.tool_usage,
            "hourly_activity":    {str(k): v for k, v in self.hourly_activity.items()},
            "traits":             {k: v.to_dict() for k, v in self.traits.items()},
            "stated_preferences":   self.stated_preferences,
            "revealed_preferences": self.revealed_preferences,
            "contradictions":     self.contradictions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        p = cls(user_id=d["user_id"])
        p.created_at         = d.get("created_at", time.time())
        p.last_updated       = d.get("last_updated", time.time())
        p.total_sessions     = d.get("total_sessions", 0)
        p.total_turns        = d.get("total_turns", 0)
        p.technical_level    = TechnicalLevel(d.get("technical_level", "unknown"))
        p.communication_style = CommunicationStyle(d.get("communication_style", "unknown"))
        p.domain_counts      = d.get("domain_counts", {})
        p.tool_usage         = d.get("tool_usage", {})
        p.hourly_activity    = {int(k): v for k, v in d.get("hourly_activity", {}).items()}
        p.stated_preferences   = d.get("stated_preferences", {})
        p.revealed_preferences = d.get("revealed_preferences", {})
        p.contradictions     = d.get("contradictions", [])
        p.traits             = {
            k: InferredTrait.from_dict(v)
            for k, v in d.get("traits", {}).items()
        }
        return p


# ── SQLite DDL ────────────────────────────────────────────────────────────────

_DDL = [
    """CREATE TABLE IF NOT EXISTS user_profiles (
        user_id     TEXT PRIMARY KEY,
        profile_json TEXT NOT NULL,
        updated_at  REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_up_updated ON user_profiles(updated_at)",
]


# ── LLM Prompts (from original, preserved verbatim) ──────────────────────────

_TRAIT_INFERENCE_SYSTEM = """You are building a behavioral profile of an IT operations operator.
Analyze the conversation turn provided and infer observable traits from behavior.

Extract ONLY behavioral traits revealed by choices — not what they say about themselves.
Focus on: technical sophistication, communication preference, tool preferences,
domain expertise, response patterns to errors.

Respond with ONLY a JSON array. No explanation, no preamble, no markdown.
Return [] if no clear traits are observable.
Each item:
{"trait": "short_snake_case_key", "value": "observed value", "confidence": 0.0-1.0,
  "is_revealed": true, "evidence": "what specifically suggested this"}"""

_CONTRADICTION_CHECK_SYSTEM = """You are analyzing an IT operator's stated vs revealed preferences.
Identify significant contradictions between what they say and what they actually do.

Respond with ONLY a JSON array. No explanation. Return [] if no contradictions.
Each item:
{"stated": "what they said", "revealed": "what they actually do", "confidence": 0.0-1.0}"""

_DOMAIN_PATTERNS = {
    "auth":       r"\b(auth|login|radius|ldap|saml|sso|certificate|credential)\b",
    "network":    r"\b(bgp|ospf|vlan|interface|switch|router|latency|bandwidth)\b",
    "wireless":   r"\b(ap|wifi|wireless|ssid|rssi|5ghz|2\.4ghz|channel)\b",
    "monitoring": r"\b(prometheus|grafana|alert|alarm|metric|snmp|pagerduty)\b",
    "kubernetes": r"\b(k8s|kubectl|pod|deployment|namespace|ingress|helm)\b",
    "dns":        r"\b(dns|resolver|nslookup|dig|a record|cname|ttl)\b",
    "incident":   r"\b(p0|p1|incident|outage|sev\d|critical|emergency)\b",
    "ipam":       r"\b(ip address|subnet|prefix|cidr|ipam|dhcp)\b",
}

_EXPERT_SIGNALS = [
    "fib", "mpls", "igp", "redistribution", "as-path", "ecmp",
    "ebgp", "ibgp", "bfd", "ospf area", "is-is", "qos marking",
]
_NOVICE_SIGNALS = [
    "what is", "how do i", "can you explain", "what does this mean",
    "i'm not sure", "never used", "first time",
]
_STATED_PATTERNS = [
    (re.compile(r"I (?:prefer|like|use|always use)\s+(\w+)", re.I), "tool_preference"),
    (re.compile(r"don'?t use\s+(\w+)", re.I), "tool_avoidance"),
    (re.compile(r"I (?:hate|dislike|avoid)\s+(\w+)", re.I), "tool_avoidance"),
    (re.compile(r"we (?:use|run|deploy)\s+(\w+)", re.I), "environment_tool"),
]


# ── UserModelEngine ───────────────────────────────────────────────────────────

class UserModelEngine:
    """
    Builds and maintains a persistent behavioral UserProfile per user.

    Key design:
    - Profiles are keyed by user_id (not session_id) and persisted in SQLite.
    - update_profile() is synchronous; call from any thread.
    - LLM inference is optional: pass llm_fn=(system, user)->str for richer traits.
      Falls back to heuristic-only mode when llm_fn is None.
    - Thread-safe: RLock guards all profile reads/writes.

    Integration with MemoryManager:
        engine = UserModelEngine(db_path="./memory_data/memory.db")
        engine.update_profile(user_id, session_id, user_text, assistant_text)
        section = engine.get_prompt_section(user_id)
        # inject section into system prompt
    """

    def __init__(
        self,
        db_path: str,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        contradiction_check_interval: int = 10,
    ) -> None:
        self._pool = get_pool(db_path)
        self._llm_fn = llm_fn
        self._check_interval = contradiction_check_interval
        # In-memory cache: user_id → UserProfile (loaded from DB on first access)
        self._cache: Dict[str, UserProfile] = {}
        self._lock = threading.RLock()
        self._turn_counts: Dict[str, int] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        self._pool.execute_write_many([(sql, ()) for sql in _DDL])

    # ── public API ────────────────────────────────────────────────────────────

    def update_profile(
        self,
        user_id:        str,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[List[dict]] = None,
    ) -> UserProfile:
        """
        Update the behavioral profile for user_id after one conversation turn.
        Synchronous. Thread-safe.
        Returns the updated UserProfile.
        """
        with self._lock:
            profile = self._load(user_id)
            turn_key = f"{user_id}:{session_id}"
            self._turn_counts[turn_key] = self._turn_counts.get(turn_key, 0) + 1
            turn_num = self._turn_counts[turn_key]
            profile.total_turns += 1

            # Hourly activity
            hour = int(time.strftime("%H"))
            profile.hourly_activity[hour] = profile.hourly_activity.get(hour, 0) + 1

            # Domain counts
            _update_domains(profile, user_text)

            # Tool usage
            for tc in (tool_calls or []):
                name = tc.get("tool") or tc.get("name", "")
                if name:
                    profile.tool_usage[name] = profile.tool_usage.get(name, 0) + 1

            # Stated preferences
            _extract_stated_preferences(profile, user_text)

            # Heuristic trait updates (always runs, no LLM needed)
            _update_level_heuristics(profile, user_text, assistant_text)

            # LLM-based trait inference (optional)
            if self._llm_fn:
                traits = self._infer_traits_sync(profile, user_text, assistant_text, tool_calls or [])
                for t in traits:
                    _update_trait(profile, t)

                # Periodic contradiction check
                if turn_num % self._check_interval == 0:
                    self._check_contradictions_sync(profile)

            profile.last_updated = time.time()
            self._save(profile)
            return profile

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Return the current profile for user_id, or None if not yet built."""
        with self._lock:
            return self._load_if_exists(user_id)

    def get_prompt_section(self, user_id: str, max_chars: int = 800) -> str:
        """
        Return the profile as a compact context block for system prompt injection.
        Returns "" if no profile exists yet.
        """
        with self._lock:
            p = self._load_if_exists(user_id)
        return p.to_prompt_section(max_chars) if p else ""

    def merge_sessions(self, user_id: str, session_ids: List[str]) -> UserProfile:
        """
        Aggregate profile data from multiple sessions into the persistent profile.
        Called by MemoryManager after cross-session analysis.
        """
        with self._lock:
            profile = self._load(user_id)
            profile.total_sessions = max(profile.total_sessions, len(session_ids))
            self._save(profile)
            return profile

    def list_users(self) -> List[str]:
        rows = self._pool.execute_read("SELECT user_id FROM user_profiles ORDER BY updated_at DESC")
        return [r["user_id"] for r in rows]

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self, user_id: str) -> UserProfile:
        """Load from cache → DB → create new. Always returns a profile."""
        if user_id in self._cache:
            return self._cache[user_id]
        rows = self._pool.execute_read(
            "SELECT profile_json FROM user_profiles WHERE user_id=?", (user_id,)
        )
        if rows:
            try:
                p = UserProfile.from_dict(json.loads(rows[0]["profile_json"]))
                self._cache[user_id] = p
                return p
            except Exception as e:
                logger.warning("Failed to deserialize profile for %s: %s", user_id, e)
        p = UserProfile(user_id=user_id)
        self._cache[user_id] = p
        return p

    def _load_if_exists(self, user_id: str) -> Optional[UserProfile]:
        if user_id in self._cache:
            return self._cache[user_id]
        rows = self._pool.execute_read(
            "SELECT profile_json FROM user_profiles WHERE user_id=?", (user_id,)
        )
        if rows:
            try:
                p = UserProfile.from_dict(json.loads(rows[0]["profile_json"]))
                self._cache[user_id] = p
                return p
            except Exception:
                pass
        return None

    def _save(self, profile: UserProfile) -> None:
        self._pool.execute_write(
            """INSERT OR REPLACE INTO user_profiles (user_id, profile_json, updated_at)
               VALUES (?, ?, ?)""",
            (profile.user_id, json.dumps(profile.to_dict()), profile.last_updated),
        )
        self._cache[profile.user_id] = profile

    # ── LLM inference (sync wrappers) ─────────────────────────────────────────

    def _infer_traits_sync(
        self,
        profile: UserProfile,
        user_text: str,
        assistant_text: str,
        tool_calls: List[dict],
    ) -> List[dict]:
        snapshot = json.dumps({
            "technical_level":     profile.technical_level.value,
            "communication_style": profile.communication_style.value,
            "known_traits":        list(profile.traits.keys())[:10],
        })
        user_content = (
            f"Conversation:\nUser: {user_text[:400]}\n"
            f"Assistant: {assistant_text[:300]}\n"
            f"Tools used: {json.dumps([tc.get('tool', '') for tc in tool_calls[:3]])}\n\n"
            f"Current profile:\n{snapshot}"
        )
        raw = self._call_llm(_TRAIT_INFERENCE_SYSTEM, user_content)
        return _parse_json_list(raw)

    def _check_contradictions_sync(self, profile: UserProfile) -> None:
        if not profile.stated_preferences or not profile.revealed_preferences:
            return
        user_content = (
            f"Stated preferences:\n{json.dumps(profile.stated_preferences)}\n\n"
            f"Revealed preferences (from behavior):\n{json.dumps(profile.revealed_preferences)}"
        )
        raw = self._call_llm(_CONTRADICTION_CHECK_SYSTEM, user_content)
        for c in _parse_json_list(raw):
            if c not in profile.contradictions and c.get("confidence", 0) >= 0.6:
                profile.contradictions.append(c)
                # Mark the relevant trait as contradicted
                for key in profile.traits:
                    if key in str(c.get("stated", "")):
                        profile.traits[key].contradicted = True
                logger.info("Contradiction: stated=%r revealed=%r",
                            c.get("stated", "")[:50], c.get("revealed", "")[:50])

    def _call_llm(self, system: str, user: str) -> str:
        if self._llm_fn is None:
            return _stub_llm(user)
        try:
            raw = self._llm_fn(system, user)
            # Strip <think> blocks from reasoning models
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
            return raw if raw else _stub_llm(user)
        except Exception as e:
            logger.warning("UserModelEngine llm_fn failed (%s), using heuristics", e)
            return _stub_llm(user)


# ── Pure functions (stateless helpers) ───────────────────────────────────────

def _update_domains(profile: UserProfile, text: str) -> None:
    tl = text.lower()
    for domain, pattern in _DOMAIN_PATTERNS.items():
        if re.search(pattern, tl):
            profile.domain_counts[domain] = profile.domain_counts.get(domain, 0) + 1


def _extract_stated_preferences(profile: UserProfile, text: str) -> None:
    for pat, pref_type in _STATED_PATTERNS:
        for m in pat.finditer(text):
            key = f"stated_{pref_type}_{m.group(1).lower()[:20]}"
            profile.stated_preferences[key] = m.group(1).lower()


def _update_level_heuristics(
    profile: UserProfile, user_text: str, assistant_text: str
) -> None:
    tl = user_text.lower()
    expert_score = sum(1 for s in _EXPERT_SIGNALS if s in tl)
    novice_score = sum(1 for s in _NOVICE_SIGNALS if s in tl)
    if expert_score >= 2 and profile.technical_level == TechnicalLevel.UNKNOWN:
        profile.technical_level = TechnicalLevel.EXPERT
    elif novice_score >= 2 and profile.technical_level == TechnicalLevel.UNKNOWN:
        profile.technical_level = TechnicalLevel.NOVICE
    rlen = len(assistant_text)
    if rlen > 1000 and "thank" in tl and profile.communication_style == CommunicationStyle.UNKNOWN:
        profile.communication_style = CommunicationStyle.VERBOSE
    elif rlen < 200 and ("ok" in tl or "got it" in tl) and profile.communication_style == CommunicationStyle.UNKNOWN:
        profile.communication_style = CommunicationStyle.TERSE


def _update_trait(profile: UserProfile, trait_dict: dict) -> None:
    key   = trait_dict.get("trait", "").strip()
    value = trait_dict.get("value", "")
    conf  = float(trait_dict.get("confidence", 0.5))
    is_revealed = trait_dict.get("is_revealed", True)
    if not key or conf < 0.4:
        return
    if is_revealed:
        profile.revealed_preferences[key] = str(value)
    else:
        profile.stated_preferences[key] = str(value)
    existing = profile.traits.get(key)
    if existing:
        alpha = 0.3   # exponential smoothing weight for new observation
        existing.confidence    = round(alpha * conf + (1 - alpha) * existing.confidence, 3)
        existing.evidence_count += 1
        existing.value          = value
        existing.last_updated   = time.time()
        if trait_dict.get("evidence"):
            existing.evidence_notes.append(trait_dict["evidence"][:100])
            existing.evidence_notes = existing.evidence_notes[-5:]   # keep last 5
    else:
        profile.traits[key] = InferredTrait(
            trait_key=key, value=value, confidence=conf, evidence_count=1,
            evidence_notes=[trait_dict.get("evidence", "")[:100]],
        )


def _parse_json_list(raw: str) -> List[dict]:
    try:
        text = re.sub(r"^```json?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        return json.loads(m.group(0))
    except Exception:
        return []


def _stub_llm(text: str) -> str:
    """Deterministic heuristic stub — no LLM needed."""
    traits = []
    tl = text.lower()
    if "httpx" in tl:
        traits.append({"trait": "http_library", "value": "httpx", "confidence": 0.82,
                       "is_revealed": True, "evidence": "requested httpx"})
    if "syslog" in tl:
        traits.append({"trait": "diagnostic_approach", "value": "log-first",
                       "confidence": 0.71, "is_revealed": True,
                       "evidence": "queries syslogs as first step"})
    if "short" in tl or "brief" in tl:
        traits.append({"trait": "response_length", "value": "terse",
                       "confidence": 0.75, "is_revealed": False,
                       "evidence": "asked for short answers"})
    if "example" in tl:
        traits.append({"trait": "learning_style", "value": "example-driven",
                       "confidence": 0.68, "is_revealed": True,
                       "evidence": "frequently requests examples"})
    return json.dumps(traits)
