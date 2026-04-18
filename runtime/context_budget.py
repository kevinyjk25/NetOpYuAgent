"""
runtime/context_budget.py
--------------------------
ContextBudgetManager  — governs exactly what enters the LLM prompt each turn.
ToolResultStore       — externalises large tool outputs so only references enter
                        the prompt, not the full payload.

Why this exists
---------------
The existing MemoryRouter.retrieve() returns candidates, but nothing previously
decided *how much* of that goes into the prompt, nor what happens when a tool
returns 50,000 bytes of log data.  This module fills that gap.

Two responsibilities
--------------------
1. ContextBudgetManager
   • Accepts memory results, tool results, confirmed_facts, working_set
   • Enforces per-slot token budgets
   • Collapses oversized tool outputs to a summary + reference ID
   • Returns a single formatted string ready for prompt injection

2. ToolResultStore
   • Stores large tool outputs keyed by a reference ID
   • A follow-up "read_result" tool can retrieve a slice by ID + offset
   • Uses an in-memory dict (dev); replace backing with Redis/S3 in prod

Budget slots (default token limits)
------------------------------------
  memory_context   1 500   retrieved memory records
  tool_results       800   per-tool result (truncated to this)
  confirmed_facts    400   structured session facts
  working_set        300   recent hotlist (last 3 devices / logs)
  system_env         200   site / permission / change-window metadata
  ─────────────────────
  Total             3 200   per-turn prompt budget (soft cap)

Usage
-----
    budget_mgr = ContextBudgetManager()

    prompt_section = budget_mgr.assemble(
        memory_results=results,          # list[RetrievalResult]
        tool_outputs={"prometheus": big_json},
        confirmed_facts=["payments-service is in prod", "DNS confirmed OK"],
        working_set=[DeviceRef(id="ap-01", label="AP-01 at Site-A")],
        env_context={"site": "Site-A", "change_window": False},
    )
    # → compact string ready for {context} slot in the system prompt
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ToolResultStore
# ---------------------------------------------------------------------------

class ToolResultStore:
    """
    Stores large tool outputs externally so the prompt only carries a reference.

    In production swap _store for Redis or object storage.
    """

    MAX_INLINE_CHARS = 4_000   # ~1 000 tokens: store externally above this

    def __init__(self) -> None:
        self._store: dict[str, str] = {}   # ref_id → full text

    def store(self, tool_name: str, raw_output: str) -> str:
        """
        Store *raw_output* if it exceeds the inline threshold.

        Returns a short reference string for prompt injection:
            '[STORED:tool_name:ref_id — use read_result(ref_id) to access]'
        Or the original text if small enough.
        """
        if len(raw_output) <= self.MAX_INLINE_CHARS:
            return raw_output

        ref_id = str(uuid.uuid4())[:8]
        self._store[ref_id] = raw_output
        preview = raw_output[:300].replace("\n", " ").strip()
        label = f"[STORED:{tool_name}:{ref_id}] Preview: {preview}..."
        logger.debug(
            "ToolResultStore: stored %d chars for tool=%s ref=%s",
            len(raw_output), tool_name, ref_id,
        )
        return label

    def read(self, ref_id: str, offset: int = 0, length: int = 2_000) -> Optional[str]:
        """Retrieve a slice of a stored result (for a 'read_result' tool call)."""
        full = self._store.get(ref_id)
        if full is None:
            return None
        return full[offset : offset + length]

    def clear_session(self, session_id: str) -> None:
        """No-op in the base impl; override when keys are namespaced by session."""
        pass

    @property
    def stored_count(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Budget slots
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Token limits for each prompt section."""
    memory_tokens:          int = 1_500
    tool_result_tokens:     int = 800     # per individual tool
    confirmed_facts_tokens: int = 400
    working_set_tokens:     int = 300
    env_context_tokens:     int = 200
    total_cap_tokens:       int = 3_200   # soft overall cap


@dataclass
class DeviceRef:
    """Lightweight reference to a network device or resource in the working set."""
    id:    str
    label: str
    meta:  dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.label} ({self.id})"


# ---------------------------------------------------------------------------
# ContextBudgetManager
# ---------------------------------------------------------------------------

class ContextBudgetManager:
    """
    Assembles the per-turn context section that is injected into the LLM prompt.

    Call assemble() once per turn.  It returns a compact, token-budgeted string
    that covers memory, tool results, confirmed facts, working set, and
    environment metadata — in that priority order.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        tool_store: Optional[ToolResultStore] = None,
    ) -> None:
        self._cfg   = config or BudgetConfig()
        self._store = tool_store or ToolResultStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        memory_results: Optional[list[Any]] = None,         # list[RetrievalResult]
        tool_outputs:   Optional[dict[str, str]] = None,    # tool_name → raw output
        confirmed_facts: Optional[list[str]] = None,
        working_set:    Optional[list[DeviceRef]] = None,
        env_context:    Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Build the full context string for this turn.

        Sections are assembled in priority order:
            1. Confirmed facts   (highest — structured knowledge)
            2. Working set       (current focus objects)
            3. Memory context    (retrieved history)
            4. Tool outputs      (current-turn tool results, possibly truncated)
            5. Environment       (site / permissions / change window)
        """
        sections: list[str] = []
        total_tokens = 0

        # 1. Confirmed facts
        if confirmed_facts:
            text = self._format_confirmed_facts(confirmed_facts)
            tok  = self._estimate_tokens(text)
            if total_tokens + tok <= self._cfg.total_cap_tokens:
                if tok <= self._cfg.confirmed_facts_tokens:
                    sections.append(text)
                    total_tokens += tok

        # 2. Working set
        if working_set:
            text = self._format_working_set(working_set)
            tok  = self._estimate_tokens(text)
            budget_left = self._cfg.working_set_tokens
            if tok > budget_left:
                text = self._trim_to_tokens(text, budget_left)
                tok  = budget_left
            if total_tokens + tok <= self._cfg.total_cap_tokens:
                sections.append(text)
                total_tokens += tok

        # 3. Memory context
        if memory_results:
            text = self._format_memory(memory_results)
            tok  = self._estimate_tokens(text)
            budget_left = self._cfg.memory_tokens
            if tok > budget_left:
                text = self._trim_to_tokens(text, budget_left)
                tok  = budget_left
            remaining = self._cfg.total_cap_tokens - total_tokens
            if remaining > 50:
                actual = min(tok, remaining)
                if actual < tok:
                    text = self._trim_to_tokens(text, actual)
                sections.append(text)
                total_tokens += min(tok, remaining)

        # 4. Tool outputs
        if tool_outputs:
            tool_section = self._format_tool_outputs(tool_outputs)
            tok = self._estimate_tokens(tool_section)
            remaining = self._cfg.total_cap_tokens - total_tokens
            if remaining > 50:
                if tok > remaining:
                    tool_section = self._trim_to_tokens(tool_section, remaining)
                    tok = remaining
                sections.append(tool_section)
                total_tokens += tok

        # 5. Environment context
        if env_context:
            text = self._format_env(env_context)
            tok  = self._estimate_tokens(text)
            remaining = self._cfg.total_cap_tokens - total_tokens
            if remaining > 20 and tok <= self._cfg.env_context_tokens:
                sections.append(text)
                total_tokens += tok

        logger.debug(
            "ContextBudgetManager: assembled %d sections, ~%d tokens",
            len(sections), total_tokens,
        )
        return "\n\n".join(sections) if sections else ""

    def store_tool_result(self, tool_name: str, raw_output: str) -> str:
        """
        Store a tool result and return either the raw text (if small)
        or a reference label (if large).  Call this before assemble().
        """
        return self._store.store(tool_name, raw_output)

    def read_stored_result(self, ref_id: str, offset: int = 0) -> Optional[str]:
        """Retrieve a page of a previously stored tool result."""
        return self._store.read(ref_id, offset)

    @property
    def tool_store(self) -> ToolResultStore:
        return self._store

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_confirmed_facts(facts: list[str]) -> str:
        items = "\n".join(f"  • {f}" for f in facts[:20])
        return f"[CONFIRMED FACTS]\n{items}"

    @staticmethod
    def _format_working_set(ws: list[DeviceRef]) -> str:
        items = "\n".join(f"  • {ref}" for ref in ws[:10])
        return f"[WORKING SET — current focus objects]\n{items}"

    @staticmethod
    def _format_memory(results: list[Any]) -> str:
        lines = []
        for r in results:
            tier = getattr(r.tier, "value", str(r.tier)).upper()
            content = getattr(r, "record", r)
            text = getattr(content, "content", str(content))
            lines.append(f"[{tier}] {text}")
        header = (
            "[MEMORY CONTEXT]\n"
            "# NOTE: this is RECALLED HISTORY from prior sessions, not current tool results.\n"
            "# You may still call tools to get fresh data for this request."
        )
        return header + "\n\n" + "\n\n".join(lines)

    @staticmethod
    def _format_tool_outputs(outputs: dict[str, str]) -> str:
        """
        Format accumulated tool results for the LLM context.

        Keys are _call_key fingerprints (e.g. "validate_device_config|{"device_id": "ap-01"}").
        We strip the args fingerprint for display, keeping just the tool name label.
        All results from the current session accumulate here — never overwritten.
        """
        parts = ["Tool outputs:"]
        for key, output in outputs.items():
            # Key format: "tool_name|{args_json}" or plain "tool_name"
            label = key.split("|")[0] if "|" in key else key
            parts.append(f"[TOOL: {label}]\n{output}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_env(env: dict[str, Any]) -> str:
        lines = [f"  {k}: {v}" for k, v in env.items()]
        return "[ENVIRONMENT]\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimate: 4 chars per token."""
        return max(1, len(text) // 4)

    @staticmethod
    def _trim_to_tokens(text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 20] + "\n... [truncated]"