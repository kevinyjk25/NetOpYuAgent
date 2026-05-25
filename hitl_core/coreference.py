"""
hitl_core.coreference — Pluggable focus-entity inference.

When a user query doesn't name an entity but the recent context clearly
shows a single in-focus entity (e.g. "fix the config" right after
discussing ap-01), this helper recovers the entity so downstream
pipeline steps don't have to ask the user to repeat themselves.

Design:

  * Domain-neutral. The host injects entity patterns + tool-call
    signatures; the resolver matches them against context strings.
    No hardcoded "ap-01" / "sw-core-01" knowledge.

  * Three-tier evidence (order = trust):
      1. Already-named entity in current query - no resolution needed
      2. Most-recent structured tool-call mention (TOOL_EXEC: foo|{...})
      3. Most-recent free-text entity mention in context

  * Returns either a single entity id (most common case) or None.
    Callers decide whether to default-pick the most-recent or escalate
    to a clarification when context is ambiguous.

Typical wiring:

    coref = Coreferencer(
        entity_patterns=[
            r"(?<![a-z0-9])(ap|sw|router|switch)[-_]?[a-z0-9]*[-_]?\\d+(?![a-z0-9])",
            r"\\bsvc[-_]\\w+\\b",
        ],
        tool_call_signatures=[
            "get_device_config", "edit_device_config", "validate_device_config",
            "restart_service", "drain_node",
        ],
    )

    focus = coref.infer(
        query="fix the config",
        context_strings=[
            'TOOL_EXEC: get_device_config|{"device_id": "ap-01"} ...',
            "PREV_ANALYSIS: ## ap-01 config ...",
        ],
    )
    # -> "ap-01"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Pronouns / elliptic references that signal "the entity I just discussed".
# Without one of these, a query like "查询站点a 流量" sharing no entity with
# recall is a NEW TOPIC, not a coreference; treating it as one led to
# operators reporting "I asked about site-A but the agent kept analysing
# sw-core-01 because the previous turn happened to mention it".
#
# Match either:
#   - Chinese pronoun / discourse marker: 它/这/那/此/上面/上一/刚才/继续
#   - English pronoun / elliptical: it/this/that, "the device", "the same",
#     "continue", "go on" (the user's natural continuation cue)
#
# Conservative by design: false positives here force re-resolution on a
# query that didn't need it (mild verbosity in the LLM prompt). False
# negatives — failing to bind a true coreference — fall through to LLM
# inference from recall, which is its baseline behaviour anyway.
_PRONOUN_RE = re.compile(
    r"("
    r"它|这台|那台|这个|那个|此|上面|上一|刚才|刚刚|继续|接着|然后呢|"
    r"\bit\b|\bthis\b|\bthat\b|\bthe (?:device|same|one)\b|"
    r"\bcontinue\b|\bgo on\b|\bproceed\b"
    r")",
    re.IGNORECASE,
)


def _strip_hitl_templates(query: str) -> str:
    """Strip HITL augmentation markers from a query so they don't leak
    pronouns/keywords into coreference detection.

    When the agent_loop_resumer re-runs after a HITL choose/clarification,
    it augments the original query with marker blocks like
    `[OPERATOR DISAMBIGUATION] ... proceed using ...` or
    `[RESOLVED FROM CONTEXT] target_device = ap-02`. These template texts
    contain pronoun-like words ("proceed", "this", "it") and entity
    mentions that trick the heuristic into firing on a fresh query.

    Strip everything from the first `[OPERATOR ...]` / `[RESOLVED ...]`
    onward so only the operator's original wording is examined.
    """
    if not query:
        return ""
    # Match the start of any HITL augmentation block.
    # Greedy from first \n\n[ATTRIBUTE_IN_CAPS] onward.
    m = re.search(r"\n\n\[(?:OPERATOR|RESOLVED) [A-Z\- ]+\]", query)
    return query[:m.start()] if m else query


def _query_looks_like_coreference(query: str) -> bool:
    """Heuristic: should we attempt to resolve a focus entity for this
    query? Only when the query CONTAINS a pronoun or elliptical marker,
    OR is an extremely terse continuation cue (≤2 chars / 1 short word).

    Tuned to:
      - Catch '继续' / '更多' (2 chars), 'next' / 'more' / 'go' (1 word)
      - NOT catch '查询流量' (4 chars but a complete verb-object phrase
        — a new request, not a continuation).

    Conservative by design: a missed coreference falls through to whatever
    the LLM can infer from recall, which is its baseline. A false-positive
    binding can corrupt a fresh user query (the bug we're fixing here)
    so we err on the side of NOT binding when ambiguous.

    HITL-template markers ([OPERATOR DISAMBIGUATION], [RESOLVED FROM
    CONTEXT], etc) are stripped before the check so their boilerplate
    text ("proceed using context ...") doesn't trigger detection.
    """
    q = _strip_hitl_templates(query or "").strip()
    if not q:
        return False
    if _PRONOUN_RE.search(q):
        return True
    # Extremely terse continuations only — 2 chars OR a single short word.
    # 4-char phrases like '查询流量' (verb+object) are NEW requests; we
    # let the operator name the entity if they want it bound.
    if len(q) <= 2:
        return True
    if " " not in q and len(q) <= 5 and not any(c in q for c in "?。?,,"):
        # Single token, English-ish "next" / "more" / "redo"
        if q.isascii() and q.isalpha():
            return True
    return False


@dataclass
class CoreferenceResult:
    """Diagnostics + the inferred entity. UIs can show "we picked X
    because of Y" without re-running the heuristic."""
    entity: Optional[str]
    source: str = ""           # "query" / "tool_call" / "free_mention" / ""
    evidence: str = ""         # the matched string snippet, for debugging


class Coreferencer:
    """Pluggable focus-entity resolver.

    Construction:
      entity_patterns: list of regex strings. Each must define a
        match group capturing the entity id. The first capture group
        across patterns is taken as the id.

      tool_call_signatures: tool names that carry an entity id in their
        args. The resolver scans for "TOOL_EXEC: {tool}|{...{key}...}"
        lines and pulls the id by name. Default key is "device_id";
        configurable via tool_arg_key.

    Thread-safe (no mutable state). Cheap (regex-only); safe to call
    on every turn.
    """

    def __init__(
        self,
        *,
        entity_patterns: Iterable[str],
        tool_call_signatures: Iterable[str] = (),
        tool_arg_key: str = "device_id",
    ):
        self._entity_patterns = [re.compile(p, re.IGNORECASE) for p in entity_patterns]
        # Pre-compile a single regex covering all tool signatures + the
        # arg key. Format expected: "TOOL_EXEC: foo|{...\"device_id\": \"X\"...}"
        sig_re = "|".join(re.escape(s) for s in tool_call_signatures)
        self._tool_call_re = (
            re.compile(
                rf'TOOL_EXEC:\s*({sig_re})\s*\|\s*\{{[^}}]*'
                rf'"{re.escape(tool_arg_key)}"\s*:\s*"([^"]+)"',
                re.IGNORECASE,
            )
            if sig_re else None
        )

    def infer(
        self,
        *,
        query: str,
        context_strings: Iterable[str] = (),
    ) -> CoreferenceResult:
        """Resolve focus entity given a query + a sequence of context
        strings (typically: recall context, confirmed_facts list,
        recent assistant turns).

        Returns CoreferenceResult.entity = None when:
          • Query already names an entity (no resolution needed)
          • No usable evidence found in context
        """
        # Step 1: query already names an entity → no need to resolve
        for pat in self._entity_patterns:
            if pat.search(query or ""):
                return CoreferenceResult(entity=None, source="query")

        # Coalesce all context into one searchable string. Order
        # matters: later strings are more recent (caller's responsibility
        # to pass them in chronological order).
        ctx = "\n".join(s for s in context_strings if s)
        if not ctx.strip():
            return CoreferenceResult(entity=None, source="")

        # Step 2: structured tool-call mention (most reliable)
        if self._tool_call_re:
            matches = list(self._tool_call_re.finditer(ctx))
            if matches:
                last = matches[-1]
                return CoreferenceResult(
                    entity=last.group(2),
                    source="tool_call",
                    evidence=last.group(0)[:120],
                )

        # Step 3: free-text mention — pick the LAST entity mentioned
        # anywhere in the context (proxy for "most recently discussed").
        # GATED: only fall back to free_mention when the query syntactically
        # looks like a coreference (contains a pronoun, elliptical marker,
        # or is a very short continuation cue). A query like "站点a 异常流量"
        # is a NEW topic and should NOT inherit the prior turn's entity
        # just because the prior turn happened to mention one. Previously
        # this gating was absent and operators saw their fresh queries
        # silently bound to stale devices.
        if not _query_looks_like_coreference(query or ""):
            return CoreferenceResult(entity=None, source="")

        # Collect ALL distinct entities mentioned anywhere in context.
        # Last-mention wins as before, but we also need the SET to detect
        # ambiguity: when the recall surface mentions multiple devices
        # (e.g. ap-01 AND ap-02 from a prior batch) AND the query is a
        # bare continuation cue ('请继续') that doesn't name one, we
        # can't be sure which the operator means → don't bind. Returning
        # entity=None lets the LLM pick from full recall context, which
        # is the safer behaviour than committing to a guess.
        all_mentions: list[str] = []
        last_mention: Optional[str] = None
        last_evidence = ""
        for pat in self._entity_patterns:
            for m in pat.finditer(ctx):
                all_mentions.append(m.group(0))
                last_mention = m.group(0)
                last_evidence = ctx[max(0, m.start()-20):m.end()+20]

        if last_mention:
            distinct = {e.lower() for e in all_mentions}
            # Continuation cue (≤ 6 chars, like '请继续' / '继续' / '继续呢')
            # with multiple candidate entities → ambiguous, don't bind.
            stripped_q = _strip_hitl_templates(query or "").strip()
            is_bare_continuation = len(stripped_q) <= 6 and len(distinct) >= 2
            if is_bare_continuation:
                logger.debug(
                    "coref: bare continuation %r matches %d entities %s — "
                    "not binding (ambiguous)", stripped_q, len(distinct),
                    sorted(distinct),
                )
                return CoreferenceResult(entity=None, source="ambiguous")
            return CoreferenceResult(
                entity=last_mention,
                source="free_mention",
                evidence=last_evidence,
            )

        return CoreferenceResult(entity=None, source="")

    # Convenience — returns just the entity id, drops diagnostics
    def infer_entity(
        self, *, query: str, context_strings: Iterable[str] = (),
    ) -> Optional[str]:
        return self.infer(query=query, context_strings=context_strings).entity


# ---------------------------------------------------------------------------
# IT-ops convenience preset (opt-in)
# ---------------------------------------------------------------------------
# Hosts targeting the original NetOpYuAgent IT-ops domain can just
# import and use; other domains build their own Coreferencer.

DEFAULT_DEVICE_PATTERN = (
    r"(?<![a-z0-9])"
    r"(?:ap|sw|router|switch)[-_]?[a-z0-9]*[-_]?\d+"
    r"(?![a-z0-9])"
)

DEFAULT_DEVICE_TOOL_SIGNATURES = (
    "get_device_config", "validate_device_config", "edit_device_config",
    "push_config", "restart_service", "drain_node",
    "failover", "delete_resource", "syslog_search",
)

def build_neutral_coreferencer() -> Coreferencer:
    """Domain-free Coreferencer: no entity patterns, no tool signatures, so
    it always resolves to "no entity". This is the L0 default — a non-network
    agent gets no spurious device coreference. The active business profile
    (L1) injects a domain-specific coreferencer (e.g.
    build_default_device_coreferencer for network) when it wants one.
    (L0/L1 Stage B, 2026-05.)
    """
    return Coreferencer(entity_patterns=[], tool_call_signatures=())


def build_default_device_coreferencer() -> Coreferencer:
    """Pre-configured Coreferencer for IT-ops device queries.

    Hosts in other domains build their own. This is just a "if you're
    using hitl_core for the original use-case, here's the wiring you'd
    write anyway" shortcut.
    """
    return Coreferencer(
        entity_patterns=[DEFAULT_DEVICE_PATTERN],
        tool_call_signatures=DEFAULT_DEVICE_TOOL_SIGNATURES,
        tool_arg_key="device_id",
    )