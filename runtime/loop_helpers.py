"""
runtime/loop_helpers.py
-----------------------
Pure, stateless helpers extracted from AgentRuntimeLoop (Item 4 refactor,
2026-05). These were @staticmethod on the loop class; they never touched
``self``, so moving them to module level reduces the size of loop.py without
any behaviour change.

AgentRuntimeLoop keeps same-named thin @staticmethod wrappers that forward
here, so every existing call site (`self._strip_thinking(...)`,
`self._is_complete(...)`, etc.) is unchanged.

Module-independence: this file imports ONLY `re` and `runtime.directive_parser`
(a sibling runtime module). It must never import task/ or registry/.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # type-only; no runtime import cycle
    from .stop_policy import StopDecision


def query_mentions_concrete_target(q: str) -> bool:
    """Heuristic: does the query name a specific device/service?
    Looks for tokens like ap-NN, sw-core-NN, router-NN, IPs, hostnames.

    IMPORTANT: do NOT use \\b here — Python regex \\b only treats
    ASCII letter<->non-letter as a word boundary, so "ap-01" tucked
    between Chinese characters (e.g. "修复ap-01设备") fails the
    \\b match. Use ASCII-explicit lookbehind/lookahead instead so
    Chinese-glued device IDs are correctly recognised.
    """
    q_lower = q.lower()
    # Common device-id patterns: ap-01, sw-core-01, router-02, radius-01.
    # The (?<![a-z0-9]) / (?![a-z0-9]) anchors mean "not preceded /
    # followed by another ASCII alphanumeric" — Chinese characters
    # satisfy this constraint, so embedded device IDs are matched.
    if re.search(
        r"(?<![a-z0-9])[a-z]{2,}[-_]\w{2,}(?![a-z0-9])", q_lower
    ):
        return True
    # Also accept the structured device pattern used elsewhere in
    # this file (ap/sw/router/switch + digits) to stay consistent.
    if re.search(
        r"(?<![a-z0-9])(ap|sw|router|switch)[-_]?[a-z0-9]*[-_]?\d+(?![a-z0-9])",
        q_lower,
    ):
        return True
    # IPv4 (no character-class boundary issue for digits + dots)
    if re.search(r"(?<!\d)\d+\.\d+\.\d+\.\d+(?!\d)", q):
        return True
    return False


def strip_thinking(response: str) -> str:
    """
    Remove <think>...</think> blocks emitted by thinking models
    (qwen3, deepseek-r1, etc.) before tool parsing or display.
    Preserves everything outside the think block.
    """
    # Remove <think>...</think> blocks (case-insensitive, multiline, non-greedy)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def skill_loads_in(response: str) -> set:
    from runtime.directive_parser import find_skill_load_names
    return set(find_skill_load_names(response))


def is_complete(response: str, tool_calls: list) -> bool:
    # If the LLM emitted a SKILL_LOAD directive, it needs one more turn
    # to read the loaded detail and then call the actual tools.
    # A pure SKILL_LOAD response (no prose, no tool calls) means the model
    # asked for skill detail and is waiting for it — keep the loop running
    # so next turn can read the loaded detail. Only mark complete if the
    # model produced real prose + tool calls alongside the SKILL_LOAD.
    from runtime.directive_parser import find_skill_load_names, strip_skill_load_directives
    skill_loads = find_skill_load_names(response)
    if skill_loads:
        stripped = strip_skill_load_directives(response).strip()
        if len(stripped) == 0 and len(tool_calls) == 0:
            # Pure SKILL_LOAD — keep looping so next turn sees the detail
            return False
        # SKILL_LOAD plus other content — completion follows the tool-call rule
        return len(tool_calls) == 0
    return len(tool_calls) == 0


def format_final(chunks: list[str], decision: "StopDecision") -> str:
    base = "\n".join(chunks) if chunks else ""
    stop = decision.summary or decision.reason
    if stop:
        return f"{base}\n\n---\n{stop}".strip()
    return base.strip()


# ---------------------------------------------------------------------------
# Tool-ledger helpers (extracted Item 4, 2026-05)
# ---------------------------------------------------------------------------

def page_default_size_for_ledger() -> int:
    """Page size used by build_tool_ledger for read_stored_result coverage
    estimates. Loaded from cfg.context_budget_display.page_default_size;
    defaults to 2000."""
    try:
        from config import cfg as _app_cfg
        return int(getattr(getattr(_app_cfg, "context_budget_display", None), "page_default_size", 2000))
    except Exception:
        return 2000


def call_key(tool_name: str, tool_args: dict) -> str:
    """
    Deduplicate tool calls by name+args fingerprint, not just name.
    This allows calling the same tool with different args (e.g. different
    device_ids) in one session without the second being blocked as a duplicate.
    Only blocks genuinely identical calls (same tool, same arguments).
    """
    import json as _json
    try:
        args_sig = _json.dumps(tool_args, sort_keys=True)
    except Exception:
        args_sig = str(tool_args)
    return f"{tool_name}|{args_sig}"


def build_tool_ledger(
    tool_outputs: dict,
    tool_reg: dict,
    raw_outputs: dict,
) -> list[str]:
    """
    Build confirmed_facts ledger entries from the current session's tool_outputs.
    Written at end of stream() so next HTTP request can seed called_tools and
    surface existing ref_ids without re-fetching.

    Collapsing rules:
    - Multiple read_stored_result pages for the same ref_id → single summary entry
    - [STORED:] labels annotated with total_size from their read results
    - Inline results recorded with size
    """
    import json as _j

    # Pass 1: collect total_size and page count per ref_id from read_stored_result results
    ref_info: dict = {}   # ref_id → {total_size, pages, last_offset}
    for key, val in raw_outputs.items():
        tool = key.split("|")[0] if "|" in key else key
        if tool != "read_stored_result" or "_summary" in key:
            continue
        try:
            args   = _j.loads(key.split("|", 1)[1]) if "|" in key else {}
            ref    = args.get("ref_id", "").strip("[]")
            ref    = ref.rsplit(":", 1)[-1].strip() if ":" in ref else ref
            offset = int(args.get("offset", 0))
            if not ref:
                continue
            total_m = re.search(r"Total size:\s*([\d,]+)", val)
            total   = int(total_m.group(1).replace(",", "")) if total_m else 0
            if ref not in ref_info:
                ref_info[ref] = {"total_size": total, "last_offset": offset, "pages": 0}
            else:
                if offset > ref_info[ref]["last_offset"]:
                    ref_info[ref]["last_offset"] = offset
                    ref_info[ref]["total_size"]   = total
            ref_info[ref]["pages"] += 1
        except Exception:
            pass

    ledger: list[str] = []
    seen:   set[str]  = set()
    seen_read_refs:  set[str] = set()   # track which refs already have a read entry

    for key, stored in tool_outputs.items():
        if key in seen or "_summary" in key:
            continue
        seen.add(key)
        tool_name = key.split("|")[0] if "|" in key else key

        # Collapse all read_stored_result pages for same ref_id into ONE entry
        if tool_name == "read_stored_result":
            try:
                args   = _j.loads(key.split("|", 1)[1]) if "|" in key else {}
                ref    = args.get("ref_id", "").strip("[]")
                ref    = ref.rsplit(":", 1)[-1].strip() if ":" in ref else ref
            except Exception:
                ref = ""
            if ref and ref in seen_read_refs:
                continue   # already emitted a summary for this ref_id
            if ref:
                seen_read_refs.add(ref)
                info    = ref_info.get(ref, {})
                pages   = info.get("pages", 1)
                total   = info.get("total_size", 0)
                covered = info.get("last_offset", 0) + page_default_size_for_ledger()
                ledger.append(
                    f"TOOL_EXEC: read_stored_result|ref={ref} pages_read={pages} "
                    f"bytes_covered=0-{min(covered, total)} total={total}"
                )
            continue

        raw  = raw_outputs.get(key, stored)
        ref_m = re.search(r"\[STORED:\w+:(\w+)\]", stored)
        if ref_m:
            ref_id = ref_m.group(1)
            total  = ref_info.get(ref_id, {}).get("total_size", len(raw))
            ledger.append(
                f"TOOL_EXEC: {key} → ref={ref_id} total_size={total} "
                f"[reuse: read_stored_result ref_id={ref_id}]"
            )
        else:
            ledger.append(f"TOOL_EXEC: {key} → inline size={len(raw)}")

    return ledger
