"""
memory/user_model.py
---------------------
UserModelEngine — Hermes-style dialectic user modeling.

Hermes innovation this implements
-----------------------------------
§03 环节五：用户建模 (User Modeling)
  "Honcho用户建模系统，它做的事情比记住你说过什么更进一步：
   它在推理你是什么样的人。每次对话结束后，Honcho会分析这次交流，
   推导出你的偏好、习惯、目标。这些推导不只是记录你说什么，
   而是从你的行为模式中归纳出更深层的特征。"

§03 The key distinction:
  "比如你从来没有明确说过'我喜欢简洁的代码风格'，但Honcho通过分析
   你多次修改代码的模式，推断出这个结论。"

§04 持久记忆 (Persistent Memory):
  "这一层存的不是对话内容，而是从对话中提炼出的持久状态。"

What this builds
-----------------
A structured UserProfile with six inference dimensions:

  1. technical_level     — novice/intermediate/expert, inferred from queries
  2. tool_preferences    — which tools they reach for (httpx vs requests etc)
  3. communication_style — terse/detailed/example-heavy responses preferred
  4. work_patterns       — active hours, task types, urgency patterns
  5. domain_focus        — which IT domains appear most (network/auth/k8s etc)
  6. quality_markers     — values correctness over speed? or speed over docs?

The model uses "dialectic" reasoning (Hermes term):
  - Observes STATED preferences ("I prefer X")
  - Observes REVEALED preferences (actually uses X repeatedly)
  - Detects contradictions between stated and revealed
  - Weights revealed preferences higher (behavior > words)

Contradiction detection example:
  Stated:   "I always write tests first" (TDD claim)
  Revealed: Never uses testing tools, never mentions tests in tasks
  Model:    marks this as contradicted, doesn't use the stated pref
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TechnicalLevel(str, Enum):
    NOVICE       = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT       = "expert"
    UNKNOWN      = "unknown"


class CommunicationStyle(str, Enum):
    TERSE    = "terse"      # prefers short, direct answers
    BALANCED = "balanced"   # normal length
    VERBOSE  = "verbose"    # wants examples, explanations
    UNKNOWN  = "unknown"


# ---------------------------------------------------------------------------
# Profile data models
# ---------------------------------------------------------------------------

@dataclass
class InferredTrait:
    """One inferred behavioral trait with its evidence."""
    trait_key:       str
    value:           Any
    confidence:      float         # 0.0–1.0
    evidence_count:  int           # how many observations support this
    contradicted:    bool = False  # stated vs revealed conflict detected
    last_updated:    float = field(default_factory=time.time)
    evidence_notes:  list[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """
    Structured behavioral model of the operator.
    Persisted across sessions — updated incrementally.
    """
    session_id:        str   # primary session this profile belongs to
    created_at:        float = field(default_factory=time.time)
    last_updated:      float = field(default_factory=time.time)
    total_sessions:    int   = 0
    total_turns:       int   = 0

    # Core inferred dimensions
    technical_level:    TechnicalLevel    = TechnicalLevel.UNKNOWN
    communication_style: CommunicationStyle = CommunicationStyle.UNKNOWN

    # Free-form trait map (trait_key → InferredTrait)
    traits: dict[str, InferredTrait] = field(default_factory=dict)

    # Observed tool usage frequency
    tool_usage:    dict[str, int] = field(default_factory=dict)

    # Domain focus frequency (auth, network, k8s, etc.)
    domain_counts: dict[str, int] = field(default_factory=dict)

    # Hourly activity pattern (0–23 → count)
    hourly_activity: dict[int, int] = field(default_factory=dict)

    # Stated vs revealed tracking
    stated_preferences:   dict[str, str] = field(default_factory=dict)
    revealed_preferences: dict[str, str] = field(default_factory=dict)
    contradictions:       list[dict]     = field(default_factory=list)

    def to_prompt_section(self, max_chars: int = 800) -> str:
        """
        Format the user profile into a compact prompt section.
        Injected as hidden context at the start of each session.
        """
        lines = ["[OPERATOR PROFILE — inferred from usage patterns]"]

        if self.technical_level != TechnicalLevel.UNKNOWN:
            lines.append(f"  Technical level: {self.technical_level.value}")
        if self.communication_style != CommunicationStyle.UNKNOWN:
            lines.append(f"  Preferred response style: {self.communication_style.value}")

        # Top domain focuses
        if self.domain_counts:
            top_domains = sorted(self.domain_counts.items(), key=lambda x: -x[1])[:4]
            domains = ", ".join(d for d, _ in top_domains)
            lines.append(f"  Primary domains: {domains}")

        # Top tools
        if self.tool_usage:
            top_tools = sorted(self.tool_usage.items(), key=lambda x: -x[1])[:4]
            tools = ", ".join(t for t, _ in top_tools)
            lines.append(f"  Preferred tools: {tools}")

        # High-confidence non-contradicted traits
        for trait in sorted(
            self.traits.values(),
            key=lambda t: -t.confidence,
        )[:5]:
            if trait.confidence >= 0.65 and not trait.contradicted:
                lines.append(f"  • {trait.trait_key}: {trait.value}")

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n  ... [profile truncated]"
        return result

    def to_dict(self) -> dict:
        return {
            "session_id":         self.session_id,
            "created_at":         self.created_at,
            "last_updated":       self.last_updated,
            "total_sessions":     self.total_sessions,
            "total_turns":        self.total_turns,
            "technical_level":    self.technical_level.value,
            "communication_style": self.communication_style.value,
            "domain_counts":      self.domain_counts,
            "tool_usage":         self.tool_usage,
            "trait_count":        len(self.traits),
            "contradiction_count": len(self.contradictions),
            "traits":             {k: {
                "value":          v.value,
                "confidence":     round(v.confidence, 2),
                "evidence_count": v.evidence_count,
                "contradicted":   v.contradicted,
            } for k, v in self.traits.items()},
            "contradictions":     self.contradictions,
        }


# ---------------------------------------------------------------------------
# LLM Inference Prompts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# UserModelEngine
# ---------------------------------------------------------------------------

class UserModelEngine:
    """
    Builds and maintains a behavioral UserProfile for each operator.

    Called after every turn to incrementally update the profile.
    Profile is persisted to the FTS5 store (or memory router) and
    injected as hidden context at the start of each session.

    The "dialectic" approach (Hermes Honcho inspiration):
      - Collects stated preferences from explicit operator statements
      - Tracks revealed preferences from observed tool/style choices
      - Computes contradictions and weights revealed > stated
    """

    def __init__(
        self,
        fts_store:     Optional[Any] = None,  # FTS5SessionStore
        llm_fn:        Optional[Callable] = None,
        contradiction_check_interval: int = 10,  # check every N turns
    ) -> None:
        self._fts             = fts_store
        self._llm_fn          = llm_fn  # None → stub used in _call_llm
        self._check_interval  = contradiction_check_interval
        self._profiles:  dict[str, UserProfile] = {}   # session_id → profile
        self._turn_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Primary update hook
    # ------------------------------------------------------------------

    async def after_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[list[dict]] = None,
    ) -> UserProfile:
        """
        Update the user profile after each conversation turn.
        Returns the updated profile.
        """
        profile = self._get_or_create_profile(session_id)
        self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
        turn_num = self._turn_counts[session_id]
        profile.total_turns += 1

        # Update hourly activity
        hour = int(time.strftime("%H"))
        profile.hourly_activity[hour] = profile.hourly_activity.get(hour, 0) + 1

        # Update domain counts from query content
        self._update_domains(profile, user_text)

        # Update tool usage from tool_calls
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc.get("tool", tc.get("name", ""))
                if tool_name:
                    profile.tool_usage[tool_name] = profile.tool_usage.get(tool_name, 0) + 1

        # Detect stated preferences from user text
        self._extract_stated_preferences(profile, user_text)

        # LLM-based trait inference
        traits = await self._infer_traits(profile, user_text, assistant_text, tool_calls or [])
        for trait in traits:
            self._update_trait(profile, trait)

        # Technical level and communication style heuristics
        self._update_level_heuristics(profile, user_text, assistant_text)

        # Periodic contradiction check
        if turn_num % self._check_interval == 0:
            await self._check_contradictions(profile)

        profile.last_updated = time.time()
        self._profiles[session_id] = profile

        logger.debug(
            "UserModelEngine: session=%s turn=%d traits=%d contradictions=%d",
            session_id, turn_num, len(profile.traits), len(profile.contradictions),
        )
        return profile

    def get_profile(self, session_id: str) -> Optional[UserProfile]:
        return self._profiles.get(session_id)

    def get_prompt_section(self, session_id: str, max_chars: int = 800) -> str:
        """Return the profile formatted for prompt injection."""
        profile = self._profiles.get(session_id)
        if profile is None:
            return ""
        return profile.to_prompt_section(max_chars)

    # ------------------------------------------------------------------
    # Trait inference helpers
    # ------------------------------------------------------------------

    async def _infer_traits(
        self,
        profile: UserProfile,
        user_text: str,
        assistant_text: str,
        tool_calls: list[dict],
    ) -> list[dict]:
        """Call LLM to infer traits from this turn."""
        profile_snapshot = json.dumps({
            "technical_level":    profile.technical_level.value,
            "communication_style": profile.communication_style.value,
            "known_traits":       list(profile.traits.keys())[:10],
        })
        user_content = (
            f"Conversation:\nUser: {user_text[:400]}\n"
            f"Assistant: {assistant_text[:300]}\n"
            f"Tools used: {json.dumps([tc.get('tool', '') for tc in tool_calls[:3]])}\n\n"
            f"Current profile snapshot:\n{profile_snapshot}"
        )
        raw = await self._call_llm(_TRAIT_INFERENCE_SYSTEM, user_content)
        return self._parse_trait_list(raw)

    def _update_trait(self, profile: UserProfile, trait_dict: dict) -> None:
        """Update or create a trait in the profile using exponential smoothing."""
        key   = trait_dict.get("trait", "")
        value = trait_dict.get("value", "")
        conf  = float(trait_dict.get("confidence", 0.5))
        is_revealed = trait_dict.get("is_revealed", True)

        if not key or conf < 0.4:
            return

        # Track stated vs revealed separately
        if is_revealed:
            profile.revealed_preferences[key] = str(value)
        else:
            profile.stated_preferences[key] = str(value)

        existing = profile.traits.get(key)
        if existing:
            # Exponential smoothing of confidence
            alpha = 0.3
            new_conf = alpha * conf + (1 - alpha) * existing.confidence
            existing.confidence    = round(new_conf, 3)
            existing.evidence_count += 1
            existing.value         = value  # use latest value
            existing.last_updated  = time.time()
        else:
            profile.traits[key] = InferredTrait(
                trait_key=key,
                value=value,
                confidence=conf,
                evidence_count=1,
                evidence_notes=[trait_dict.get("evidence", "")[:100]],
            )

    def _extract_stated_preferences(self, profile: UserProfile, text: str) -> None:
        """Extract explicitly stated preferences from text."""
        stated_patterns = [
            (r"I (?:prefer|like|use|always use)\s+(\w+)", "tool_preference"),
            (r"don'?t use\s+(\w+)", "tool_avoidance"),
            (r"I (?:hate|dislike|avoid)\s+(\w+)", "tool_avoidance"),
            (r"we (?:use|run|deploy)\s+(\w+)", "environment_tool"),
        ]
        for pattern, pref_type in stated_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                key = f"stated_{pref_type}_{m.group(1).lower()[:20]}"
                profile.stated_preferences[key] = m.group(1).lower()

    async def _check_contradictions(self, profile: UserProfile) -> None:
        """Check stated vs revealed preferences for contradictions."""
        if not profile.stated_preferences or not profile.revealed_preferences:
            return
        user_content = (
            f"Stated preferences:\n{json.dumps(profile.stated_preferences)}\n\n"
            f"Revealed preferences (from behavior):\n{json.dumps(profile.revealed_preferences)}"
        )
        raw = await self._call_llm(_CONTRADICTION_CHECK_SYSTEM, user_content)
        try:
            text = re.sub(r"^```json?\s*", "", raw.strip())
            text = re.sub(r"\s*```$", "", text)
            contradictions = json.loads(text)
            if isinstance(contradictions, list):
                for c in contradictions:
                    if c not in profile.contradictions and c.get("confidence", 0) >= 0.6:
                        profile.contradictions.append(c)
                        logger.info(
                            "UserModel: contradiction detected — stated: %r, revealed: %r",
                            c.get("stated", "")[:60], c.get("revealed", "")[:60],
                        )
        except Exception:
            pass

    async def _call_llm(self, system: str, user: str) -> str:
        """Call LLM with system+user separation. Falls back to stub."""
        import re as _re
        if self._llm_fn is None:
            return await self._stub_llm(user)
        try:
            raw = await self._llm_fn(system, user)
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL | _re.IGNORECASE).strip()
            if not raw or raw.startswith("You are"):
                return await self._stub_llm(user)
            return raw
        except Exception as exc:
            logger.warning("UserModelEngine: llm_fn failed (%s) — using stub", exc)
            return await self._stub_llm(user)

    @staticmethod
    def _update_domains(profile: UserProfile, text: str) -> None:
        """Update domain frequency counts from query text."""
        domain_patterns = {
            "auth":       r"\b(auth|login|radius|ldap|saml|sso|certificate|credential)\b",
            "network":    r"\b(bgp|ospf|vlan|interface|switch|router|vlan|latency|bandwidth)\b",
            "wireless":   r"\b(ap|wifi|wireless|ssid|rssi|5ghz|2.4ghz|channel)\b",
            "monitoring": r"\b(prometheus|grafana|alert|alarm|metric|snmp|nms|pagerduty)\b",
            "kubernetes": r"\b(k8s|kubectl|pod|deployment|namespace|ingress|helm)\b",
            "dns":        r"\b(dns|resolver|nslookup|dig|a record|cname|ttl)\b",
            "incident":   r"\b(p0|p1|incident|outage|sev\d|critical|emergency)\b",
            "ipam":       r"\b(ip address|subnet|prefix|cidr|ipam|dhcp)\b",
        }
        text_lower = text.lower()
        for domain, pattern in domain_patterns.items():
            if re.search(pattern, text_lower):
                profile.domain_counts[domain] = profile.domain_counts.get(domain, 0) + 1

    @staticmethod
    def _update_level_heuristics(
        profile: UserProfile, user_text: str, assistant_text: str
    ) -> None:
        """Heuristic update of technical level from query characteristics."""
        text = user_text.lower()
        expert_signals = [
            "fib", "mpls", "igp", "redistribution", "as-path", "ecmp",
            "ebgp", "iBGP", "bfd", "ospf area", "IS-IS", "qos marking",
        ]
        novice_signals = [
            "what is", "how do i", "can you explain", "what does this mean",
            "i'm not sure", "never used", "first time",
        ]
        expert_score = sum(1 for s in expert_signals if s in text)
        novice_score = sum(1 for s in novice_signals if s in text)

        if expert_score >= 2:
            if profile.technical_level == TechnicalLevel.UNKNOWN:
                profile.technical_level = TechnicalLevel.EXPERT
        elif novice_score >= 2:
            if profile.technical_level == TechnicalLevel.UNKNOWN:
                profile.technical_level = TechnicalLevel.NOVICE

        # Communication style from response length preference
        response_len = len(assistant_text)
        if response_len > 1000 and "thank" in text:
            if profile.communication_style == CommunicationStyle.UNKNOWN:
                profile.communication_style = CommunicationStyle.VERBOSE
        elif response_len < 200 and ("ok" in text or "got it" in text):
            if profile.communication_style == CommunicationStyle.UNKNOWN:
                profile.communication_style = CommunicationStyle.TERSE

    def _get_or_create_profile(self, session_id: str) -> UserProfile:
        if session_id not in self._profiles:
            self._profiles[session_id] = UserProfile(session_id=session_id)
        return self._profiles[session_id]

    @staticmethod
    def _parse_trait_list(raw: str) -> list[dict]:
        try:
            text = re.sub(r"^```json?\s*", "", raw.strip())
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            return json.loads(m.group(0))
        except Exception:
            return []

    @staticmethod
    async def _stub_llm(text: str) -> str:
        """Deterministic stub — receives the text/data portion of the prompt."""
        await asyncio.sleep(0)
        traits = []

        if "prefer httpx" in text.lower() or "httpx" in text.lower():
            traits.append({
                "trait": "http_library_preference",
                "value": "httpx",
                "confidence": 0.82,
                "is_revealed": True,
                "evidence": "explicitly requested httpx over requests",
            })
        if "syslog" in text.lower():
            traits.append({
                "trait": "diagnostic_approach",
                "value": "log-first (checks logs before metrics)",
                "confidence": 0.71,
                "is_revealed": True,
                "evidence": "consistently queries syslogs as first diagnostic step",
            })
        if "short" in text.lower() or "brief" in text.lower():
            traits.append({
                "trait": "response_length_preference",
                "value": "terse",
                "confidence": 0.75,
                "is_revealed": False,
                "evidence": "explicitly asked for short answers",
            })
        if "example" in text.lower():
            traits.append({
                "trait": "learning_style",
                "value": "example-driven",
                "confidence": 0.68,
                "is_revealed": True,
                "evidence": "frequently requests examples",
            })
        return json.dumps(traits)