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
        # anywhere in the context (proxy for "most recently discussed")
        last_mention: Optional[str] = None
        last_evidence = ""
        for pat in self._entity_patterns:
            for m in pat.finditer(ctx):
                last_mention = m.group(0)
                last_evidence = ctx[max(0, m.start()-20):m.end()+20]
        if last_mention:
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