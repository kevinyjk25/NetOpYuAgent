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


class _SqliteStoreProxy:
    """Dict-like proxy over the SQLite results table for backward-compat access."""
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM results").fetchone()
        return row[0] if row else 0

    def get(self, ref_id: str, default: str = "") -> str:
        ref_id = ref_id.strip("[]")
        if ":" in ref_id:
            ref_id = ref_id.rsplit(":", 1)[-1].strip()
        row = self._conn.execute(
            "SELECT content FROM results WHERE ref_id=?", (ref_id,)
        ).fetchone()
        return row[0] if row else default


class ToolResultStore:
    """
    Stores large tool outputs externally so the prompt only carries a reference.

    Backed by SQLite (default) for persistence across restarts.
    Pass db_path=":memory:" for in-memory (tests only).
    Pass db_path=None to auto-locate at data/tool_results.db.
    """

    MAX_INLINE_CHARS = 4_000   # ~1 000 tokens: store externally above this
    TTL_SECONDS      = 86_400  # prune entries older than 24 h

    def __init__(self, db_path: Optional[str] = None) -> None:
        import sqlite3, pathlib, time as _time, os
        if db_path is None:
            _data_dir = pathlib.Path("data")
            _data_dir.mkdir(exist_ok=True)
            db_path = str(_data_dir / "tool_results.db")
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS results "
            "(ref_id TEXT PRIMARY KEY, content TEXT, created_at REAL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON results(created_at)")
        self._conn.commit()
        # Prune stale entries from prior runs
        self._prune()
        # Compatibility shim: expose dict-like _store interface for code that
        # reads self._store.get() directly (HTTP endpoint total_chars calc)
        self._store = _SqliteStoreProxy(self._conn)

    def _prune(self) -> None:
        import time as _time
        cutoff = _time.time() - self.TTL_SECONDS
        self._conn.execute("DELETE FROM results WHERE created_at < ?", (cutoff,))
        self._conn.commit()

    def store(self, tool_name: str, raw_output: str) -> str:
        """
        Store *raw_output* if it exceeds the inline threshold.
        Persisted to SQLite — survives restarts.
        Returns '[STORED:tool_name:ref_id] Preview: <first 80 chars>'
        or the original text if small enough to inline.
        """
        import time as _time
        if len(raw_output) <= self.MAX_INLINE_CHARS:
            return raw_output
        import uuid
        ref_id = uuid.uuid4().hex[:8]
        self._conn.execute(
            "INSERT OR REPLACE INTO results(ref_id, content, created_at) VALUES(?,?,?)",
            (ref_id, raw_output, _time.time()),
        )
        self._conn.commit()
        preview = raw_output[:80].replace("\n", " ")
        return f"[STORED:{tool_name}:{ref_id}] Preview: {preview}"


    def read(self, ref_id: str, offset: int = 0, length: int = 2_000) -> Optional[str]:
        """Retrieve a slice of a stored result. Accepts full label or plain ref_id."""
        # Normalise: strip brackets and tool_name: prefix
        ref_id = ref_id.strip("[]")
        if ":" in ref_id:
            ref_id = ref_id.rsplit(":", 1)[-1].strip()
        row = self._conn.execute(
            "SELECT content FROM results WHERE ref_id=?", (ref_id,)
        ).fetchone()
        full = row[0] if row else None
        if full is None:
            full = None
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
        """
        Format memory recall results. Accepts:
        - Plain strings (from MemoryAdapter.recall_for_session)
        - Records with .tier and .record attributes (legacy format)
        """
        lines = []
        for r in results:
            if isinstance(r, str):
                # MemoryAdapter returns a single recalled-context string per call
                if r.strip():
                    lines.append(r.strip())
                continue
            # Legacy record format with .tier
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

        Per-entry compression rules to prevent context overload:
        - [STORED:] entries: show only the preview line (full data is in the store)
        - read_stored_result entries: show in full (this IS the paged data)
        - Other entries: show first 600 chars — enough to act on, not enough to overload
        - Most recent entry: shown in full up to 1200 chars

        This prevents the LLM from receiving multi-KB raw data from previous turns
        while still getting the current turn's full result.
        """
        _STORED_LABEL_LIMIT = 1   # [STORED:] label is one line, show as-is
        _PAGED_RESULT_LIMIT = 1200  # read_stored_result: show up to 1200 chars
        _NORMAL_LIMIT       = 600   # other tools: 600 chars is enough context
        _LATEST_BONUS       = 600   # extra chars for the most recent result

        parts = ["Tool outputs:"]
        items = list(outputs.items())
        for i, (key, output) in enumerate(items):
            label = key.split("|")[0] if "|" in key else key
            is_latest = (i == len(items) - 1)

            if "[STORED:" in output:
                # Only show the reference label + preview — full data is in the store
                # Extract just the [STORED:...] line and the preview
                lines = output.splitlines()
                stored_lines = [l for l in lines if l.startswith("[STORED:") or l.startswith("Preview:")]
                display = "\n".join(stored_lines[:3]) if stored_lines else output[:200]
            elif label == "read_stored_result":
                # Paged data — show in full but capped (LLM needs this to answer)
                cap = _PAGED_RESULT_LIMIT + (_LATEST_BONUS if is_latest else 0)
                display = output[:cap] + (f"\n… [{len(output)-cap} more chars — call read_stored_result with next offset]" if len(output) > cap else "")
            else:
                # Normal result — show enough to act on
                cap = _NORMAL_LIMIT + (_LATEST_BONUS if is_latest else 0)
                display = output[:cap] + (f"\n… [truncated — {len(output)} total chars]" if len(output) > cap else "")

            parts.append(f"[TOOL: {label}]\n{display}")
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

def compress_paged_outputs(outputs: dict) -> dict:
    """
    Keep only the most recent read_stored_result page per ref_id in tool_outputs.
    Older pages are replaced by a compact summary note.

    This prevents context overflow when paging through large stored results across
    many turns. The LLM's own response text (written while reading each page) is
    saved to FTS5 by Hermes, so findings from prior pages survive via memory recall.
    """
    import json as _j, re as _re

    paged: dict = {}   # ref_id → [(offset, key, val)]
    result: dict = {}

    for key, val in outputs.items():
        tool = key.split("|")[0] if "|" in key else key
        if tool == "read_stored_result" and "_summary" not in key:
            try:
                args   = _j.loads(key.split("|", 1)[1]) if "|" in key else {}
                ref    = args.get("ref_id", "").strip("[]")
                ref    = ref.rsplit(":", 1)[-1].strip() if ":" in ref else ref
                offset = int(args.get("offset", 0))
                paged.setdefault(ref, []).append((offset, key, val))
            except Exception:
                result[key] = val
        else:
            result[key] = val

    for ref_id, pages in paged.items():
        pages = sorted(pages, key=lambda x: x[0])
        if len(pages) == 1:
            result[pages[0][1]] = pages[0][2]
        else:
            last_val = pages[-1][2]
            has_more = "Has more: True" in last_val
            total_m  = _re.search(r"Total size:\s*([\d,]+)", last_val)
            total_sz = total_m.group(1) if total_m else "?"
            covered  = pages[-2][0] + 2000
            note = (
                f"[PAGED-SUMMARY ref_id={ref_id} pages_read={len(pages)} "
                f"bytes_covered=0-{covered} total={total_sz} has_more={has_more}]\n"
                f"Prior page findings written to response — see memory context for analysis so far."
            )
            summary_key = _j.dumps({"read_stored_result": ref_id, "_summary": True})
            result[summary_key] = note
            result[pages[-1][1]] = pages[-1][2]

    return result
