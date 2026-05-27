"""
profiles/network_batch_resolver.py
==================================

L1 business logic — network-ops multi-target batch HITL resolver.

Extracted from `runtime/loop.py:_handle_tools` in the L0/L1 separation (Stage B,
2026-05). The L0 loop knows only "ask the injected resolver whether this single
destructive tool call should expand into a batch". The decision — and the
device-id parsing it needs — is network-specific and lives here, injected by the
lan/dc profiles. A non-network agent injects nothing and gets single-target HITL.

Contract (matches AgentRuntimeLoop's batch_resolver_fn):
    resolve_network_batch(
        *, tool_name, tool_args, llm_response,
        hitl_tool_names, confirmed_facts, all_parsed,
    ) -> Optional[list[tuple[str, dict]]]

Returns a list of (tool_name, args) pairs (current + siblings) when a batch is
detected, else None (→ single-target HITL). Pure function, no I/O.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Device-id pattern (sw-core-01, ap-02, router-1, …). ASCII-explicit anchors so
# Chinese-glued ids (e.g. "为sw-core-01下发") still match.
_DEV_RE = re.compile(
    r"(?<![a-z0-9])"
    r"(?:sw|ap|router|switch|core)[-_]?[a-z0-9]*[-_]?\d+"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_DEVICE_ARG_KEYS = ("device_id", "device", "target")


def _call_key(tool_name: str, tool_args: dict) -> str:
    import json
    try:
        return f"{tool_name}|{json.dumps(tool_args, sort_keys=True)}"
    except Exception:
        return f"{tool_name}|{tool_args}"


def resolve_network_batch(
    *,
    tool_name: str,
    tool_args: dict,
    llm_response: str,
    hitl_tool_names,
    confirmed_facts: list[str],
    all_parsed: list[tuple[str, dict]],
) -> Optional[list[tuple[str, dict]]]:
    """See module docstring. Returns batch list or None."""
    # ── Path A: LLM already emitted multiple [TOOL:] of the SAME destructive
    # name this turn → dedup into one batch.
    siblings = [
        (n, a) for (n, a) in all_parsed
        if n == tool_name and a != tool_args and n in hitl_tool_names
    ]
    if siblings:
        seen = {_call_key(tool_name, tool_args)}
        batch = [(tool_name, tool_args)]
        for (n, a) in siblings:
            k = _call_key(n, a)
            if k not in seen:
                seen.add(k)
                batch.append((n, a))
        logger.info(
            "network_batch: path A — %d %s calls in one turn",
            len(batch), tool_name,
        )
        return batch

    # ── Path B: ONE [TOOL:] but prose names multiple devices → fabricate
    # N-1 sibling calls by copying args + swapping the device-id field.
    current_device = str(
        tool_args.get("device_id") or tool_args.get("device")
        or tool_args.get("target") or ""
    ).strip()
    device_key = next((k for k in _DEVICE_ARG_KEYS if k in tool_args), None)
    if not (current_device and device_key):
        return None

    # Strip [TOOL:]/[TOOL_BATCH:]/[ALIAS:] so we only scan narrative prose.
    try:
        from runtime.directive_parser import (
            strip_tool_directives as _stt,
            strip_tool_batch_directives as _stb,
        )
        prose = _stb(_stt(llm_response))
    except Exception:
        prose = llm_response
    prose = re.sub(r"\[ALIAS\s*:\s*[^=\]]+?\s*=\s*[^\]]+?\s*\]", "", prose)

    mentioned: list[str] = []
    seen_dev: set[str] = set()
    for m in _DEV_RE.finditer(prose):
        did = m.group(0)
        if did.lower() not in seen_dev:
            seen_dev.add(did.lower())
            mentioned.append(did)

    # Drop alias user-terms already recorded as ENTITY_ALIAS facts.
    alias_terms = set()
    for f in (confirmed_facts or []):
        if f.startswith("ENTITY_ALIAS: "):
            mm = re.match(
                r"ENTITY_ALIAS:\s*'([^']+)'\s*actually refers to\s*'([^']+)'", f
            )
            if mm:
                alias_terms.add(mm.group(1).lower())
    if alias_terms:
        mentioned = [d for d in mentioned if d.lower() not in alias_terms]

    other_devices = [d for d in mentioned if d.lower() != current_device.lower()]
    if not (
        current_device.lower() in {d.lower() for d in mentioned}
        and 1 <= len(other_devices) <= 4   # 2-5 total
    ):
        return None

    fabricated = []
    for other in other_devices:
        new_args = copy.deepcopy(tool_args)
        new_args[device_key] = other
        orig_reason = str(new_args.get("reason", "")).strip()
        new_args["reason"] = (
            f"[auto-derived from {current_device} — verify before approving] "
            f"{orig_reason}"
        ).strip()
        fabricated.append((tool_name, new_args))

    if not fabricated:
        return None
    logger.info(
        "network_batch: path B — prose mentions %d devices %r, fabricating "
        "%d sibling %s call(s)",
        len(mentioned), mentioned, len(fabricated), tool_name,
    )
    return [(tool_name, tool_args)] + fabricated
