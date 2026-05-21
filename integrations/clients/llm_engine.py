"""
integrations/llm_engine.py
---------------------------
Real LLM engine that replaces the keyword-matching _call_llm() stub
in runtime/loop.py and the intent_classifier_node stub in hitl/graph.py.

Supported backends
-------------------
  ollama   — local Ollama server (mistral, llama3.2, qwen2.5, etc.)
  openai   — OpenAI API (gpt-4o, gpt-4o-mini)
  anthropic — Claude via API (claude-sonnet-4-6, claude-haiku-4-5)
  mock     — deterministic mock for testing (no LLM needed)

Tool call format
-----------------
The LLM is prompted to emit tool calls using this structured JSON format,
which the existing _parse_tool_calls() regex in loop.py already handles:

    [TOOL:tool_name] {"arg1": "value1", "arg2": "value2"}

For Ollama with function-calling support (mistral-nemo, qwen2.5-coder),
native tool_call messages are parsed automatically.

Integration
-----------
This module provides two integration points:

1. Replace AgentRuntimeLoop._call_llm():
    loop = AgentRuntimeLoop(...)
    loop._call_llm = llm_engine.call  # monkey-patch OR subclass

2. Replace hitl/graph.py intent_classifier_node via:
    from integrations.clients.llm_engine import LLMEngine
    engine = LLMEngine.from_config(cfg)
    # Use engine.classify_intent(query) in intent_classifier_node

Usage
-----
    engine = LLMEngine.from_config({
        "backend": "ollama",
        "model":   "mistral",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
    })

    # As _call_llm replacement in AgentRuntimeLoop:
    response = await engine.call(query, context, loop_state)

    # Classify intent (for HITL graph):
    intent = await engine.classify_intent(query)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent classification result
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent_type:   str    # destructive_op | alert_analysis | trend_prediction | general_query
    confidence:    float  # 0.0 – 1.0
    intent_summary: str
    candidates:    list[dict]


# ---------------------------------------------------------------------------
# Base LLM engine
# ---------------------------------------------------------------------------


def _strip_stored_result_section(prompt: str, *signals: str) -> str:
    """Remove read_stored_result instruction blocks when there's no [STORED:]
    reference in any of the provided signal strings.

    Activation rule:
      - Scan every signal string for the literal `[STORED:` marker.
      - If absent everywhere, strip the instruction blocks so the LLM can't
        misuse the tool with a hallucinated ref_id.

    Idempotent and safe when the blocks aren't present.
    """
    import re as _re
    haystack = "\n".join(s for s in signals if isinstance(s, str))
    if "[STORED:" in haystack:
        return prompt   # blocks stay relevant

    # SLIM-style "STRICT" block (multi-line, ends at blank line or next ALL-CAPS heading)
    prompt = _re.sub(
        r"\n*read_stored_result usage \(STRICT\):.*?(?=\n\n[A-Z]|\n\n\{|\Z)",
        "\n",
        prompt, count=1, flags=_re.DOTALL,
    )

    # PAGINATED READING block
    prompt = _re.sub(
        r"\n*PAGINATED READING[\s\S]*?aggregating ALL findings[^\n]*\.",
        "",
        prompt, count=1,
    )

    # Full-template numbered line that mentions it
    prompt = _re.sub(
        r"\n6\. read_stored_result usage[^\n]*",
        "",
        prompt, count=1,
    )

    # Collapse triple blank lines that may result
    prompt = _re.sub(r"\n\n\n+", "\n\n", prompt)
    return prompt

class LLMEngine:
    """
    Unified LLM interface for the Agent Runtime Loop and HITL graph.

    Subclass or configure via LLMEngine.from_config(cfg).
    """

    # Slim system prompt — used from turn 2+ when shorten_tool_system_after_turn fires.
    # The LLM has already learned the tool-call format from turn 1's full prompt;
    # repeating the rules every turn just wastes tokens.
    TOOL_CALL_SYSTEM_SLIM = """You are an expert IT network operations assistant.

TOOL CALL FORMAT: [TOOL:name] {{"arg": "value"}}
Rules: one tool per response, never repeat a call, end with analysis (no [TOOL:] line) when you have enough info.
MULTI-TARGET DESTRUCTIVE BATCH: when the SAME destructive tool needs to run on MULTIPLE TARGETS (e.g. "下发到 ap-01 和 ap-02"), use `[TOOL_BATCH:tool_name] [<args_dict_1>, <args_dict_2>, ...]` — a JSON array of args dicts, one per target. The system expands to N HITL cards under one batch_id. Example:
  [TOOL_BATCH:edit_device_config] [{{"device_id": "ap-01", "config_lines": [...], "reason": "..."}}, {{"device_id": "ap-02", "config_lines": [...], "reason": "..."}}]
Destructive tools (⚠HITL) — propose with concrete args; the operator will review before execution.
ENTITY ALIAS: if the user used the wrong entity name and you found the real one (via list_devices etc), emit `[ALIAS: user_term = real_term]` so the correction survives to subsequent turns. Then USE THE REAL NAME in all tool calls.

read_stored_result usage (STRICT):
- ONLY call [TOOL:read_stored_result] when a previous tool output literally contains a `[STORED:name:ref_id]` label.
- ref_id is the id INSIDE that label (e.g. `6ac5ade7` or `netflow_dump:6ac5ade7`) — NEVER a device name, hostname, or query string.
- If a tool result is already shown inline (no [STORED:] label), DO NOT call read_stored_result on it.

PAGINATED READING — only relevant after a [STORED:] label appears:
- Use length=4000 (or higher) on every call; default reads of 100 chars waste turns.
- The page response includes the line "Next offset: N" (or "EOF") — use that EXACT N as the offset of your NEXT call. NEVER restart from offset=0 once you have already read a page.
- After EACH page, write 2-3 sentences of key findings BEFORE calling the next page.
- A summary line `[PAGED-SUMMARY ref_id=... pages_read=N bytes_covered=0-M has_more=True/False]` in the context tells you what you have already read. Trust it; do not re-read pages.
- Older pages are dropped from context to save tokens; only your written findings survive across pages.
- When Has more: False, write the complete analysis aggregating ALL findings you wrote earlier.
- CRITICAL: If a page says "Has more: True", you MUST continue paging (using Next offset) until "Has more: False". Do NOT pivot to other tools, SKILL_LOAD, or final analysis while data remains unread.
- Example first call:  [TOOL:read_stored_result] {{"ref_id": "abc123", "offset": 0, "length": 4000}}
- Example next call:   [TOOL:read_stored_result] {{"ref_id": "abc123", "offset": 4000, "length": 4000}}

{extra_tools_section}

{skill_summary}

{confirmed_facts_section}
"""

    TOOL_CALL_SYSTEM = """You are an expert IT network operations assistant.

TOOL CALLING FORMAT — use EXACTLY this syntax on its own line:
[TOOL:tool_name] {{"arg1": "value1", "arg2": "value2"}}

STRICT RULES — follow exactly:
1. Call AT MOST ONE [TOOL:name] per response — never list multiple [TOOL:] lines. For multi-target destructive operations, use ONE [TOOL_BATCH:name] directive instead (see "DESTRUCTIVE OPERATIONS — multi-target batches" below).
2. NEVER repeat a tool call you have already made this session
3. When tool results appear in the context below, DO NOT call that tool again
4. When you have enough information to answer, write your analysis WITHOUT any [TOOL:...] line
5. Keep responses concise — this is a production operations environment
6. read_stored_result usage (STRICT): call ONLY when a prior tool output contains a literal `[STORED:name:ref_id]` label. The ref_id is that label's id (e.g. `6ac5ade7`), NEVER a device name or query string. If a tool result is shown inline, do NOT call read_stored_result on it.

DESTRUCTIVE OPERATIONS — for tools marked ⚠ HITL (edit_device_config, push_config, restart_service, rollback_deploy, drain_node, failover, delete_resource):
- DO NOT ask the user "are you sure?" or "do you approve?" in plain text
- DO NOT wait for the user to confirm before emitting the tool call
- INSTEAD: emit the [TOOL:name] line directly with concrete parameters (device_id, config_lines, changes, reason, etc.) inferred from the gathered context
- The system AUTOMATICALLY intercepts every destructive tool call before execution and shows an HITL approval card to the operator
- The operator reviews YOUR proposed parameters in that card; they can approve, reject, or edit the parameters before the tool actually runs
- Your job is to PROPOSE THE COMPLETE FIX, not to ask permission. If you don't propose a concrete fix, the operator has nothing to review.

DESTRUCTIVE OPERATIONS — multi-target batches (use [TOOL_BATCH:] directive):
- When the SAME destructive tool needs to run on MULTIPLE TARGETS as one logical step (e.g. user said "fix both ap-01 and ap-02", "下发到这两个设备", "push this ACL to all 3 access switches"), use the [TOOL_BATCH:name] directive instead of emitting one [TOOL:] per target.
- Syntax: `[TOOL_BATCH:tool_name] <JSON array of args dicts>` on its own line.
- Each element in the array is one args dict, just like a normal [TOOL:] would carry. The system expands the batch into N independent HITL approval cards under ONE batch_id, so the operator reviews all N targets together and can approve them as a group.
- This is the ONLY way to surface a multi-target destructive operation in one HITL round. If you emit only ONE [TOOL:] and describe the other targets in prose, the second target will NEVER appear — the system will not infer it from text.
- DO NOT mix [TOOL_BATCH:] with [TOOL:] of the same name in the same response. Pick ONE: either single-target [TOOL:] or multi-target [TOOL_BATCH:].
- Single-target destructive operations still use plain [TOOL:name] as before — [TOOL_BATCH:] is only for multi-target.

WORKED EXAMPLE — operator says "fix both ap-01 and ap-02":
[TOOL_BATCH:edit_device_config] [
  {{"device_id": "ap-01", "config_lines": ["no radius-server timeout 4", "radius-server timeout 3", "interface GigabitEthernet0", "ip access-group MGMT in"], "reason": "Fix RADIUS timeout and apply MGMT ACL on ap-01"}},
  {{"device_id": "ap-02", "config_lines": ["no radius-server timeout 4", "radius-server timeout 3", "interface GigabitEthernet0", "ip access-group MGMT in"], "reason": "Fix RADIUS timeout and apply MGMT ACL on ap-02"}}
]

TOOLS vs SKILLS — critical distinction:
- TOOLS (callable with [TOOL:name]): executable functions. Call them directly.
- SKILLS listed in "Available skills" without a matching TOOL: procedural guides only — use the directive `[SKILL_LOAD:skill_id]` exactly. IMPORTANT: SKILL_LOAD is its OWN top-level directive — NEVER prefix it with TOOL: (write `[SKILL_LOAD:netflow_analysis]`, NOT `[TOOL:SKILL_LOAD:netflow_analysis]`).
- If a name appears in BOTH the tool list AND the skills list (e.g. get_device_config, validate_device_config, list_devices), it IS a real callable tool — use [TOOL:name] directly. SKILL_LOAD is NOT needed.
- Only use [SKILL_LOAD:skill_id] for skills that have no corresponding [TOOL:] entry in the AVAILABLE TOOLS list above.

ENTITY ALIAS CORRECTIONS — when the user's term doesn't match a real entity:
- If the user mentions an entity name that doesn't exist in the system but you find the REAL matching name (e.g. user said "core-01" but list_devices returned "sw-acc-01" / "sw-acc-02" instead), EMIT an `[ALIAS:...]` directive so the correction sticks across turns.
- Syntax: `[ALIAS: user_term = real_term]` on its own line (e.g. `[ALIAS: core-01 = sw-acc-01]`).
- One directive per correction. Multiple aliases are fine — emit multiple lines.
- These are folded into the session's confirmed_facts and surfaced in the next turn's prompt under "ENTITY ALIASES". Once recorded, USE THE REAL NAME in subsequent tool calls — do NOT keep re-resolving the same correction every turn.

DELEGATION TO PEER AGENTS — when a subtask is OUTSIDE your domain (use [DELEGATE:] directive):
- This agent is specialized for its own domain (see your AVAILABLE TOOLS). If the user's request needs a capability you do NOT have a tool for — but a PEER agent specializes in it (e.g. you are a LAN agent and the task needs data-center fabric / BGP-EVPN / VXLAN work that a DC agent handles) — delegate that subtask.
- Syntax: `[DELEGATE:agent_id] <subtask description>` on its own line (e.g. `[DELEGATE:dc-agent] check BGP EVPN neighbor status on spine-1`).
- By capability instead of a fixed id: `[DELEGATE:*capability] <subtask>` (e.g. `[DELEGATE:*dc_fabric_diagnose] trace the fabric path to leaf-3`) — the system picks a healthy peer advertising that capability.
- Share your gathered facts with the peer (only when relevant): add `#forked` → `[DELEGATE:dc-agent#forked] <subtask>`. Default (no modifier) sends ONLY the subtask description (the peer starts fresh).
- STRICT: `[DELEGATE:]` is MUTUALLY EXCLUSIVE with `[TOOL:]` in one response — pick ONE per turn. If you emit both, the tool runs and the delegation is ignored. Do the local tool calls first (in earlier turns), then delegate in a turn by itself.
- After you delegate, the peer's result is injected back into your context next turn — synthesize it into your final answer for the user. Do NOT delegate the same subtask twice.
- Only delegate what you genuinely cannot do locally. If you have a tool for it, use the tool.

- Only emit [ALIAS:...] when you're CONFIDENT about the mapping (e.g. you queried list_devices and there's no ambiguity). For unclear cases, ask the user instead.

INVENTORY QUERIES — when asked what devices exist:
- Use [TOOL:list_devices] {{}} to get ALL devices in one call
- Use [TOOL:list_devices] {{"type": "switch"}} for wired switches only
- Use [TOOL:list_devices] {{"type": "wireless_ap"}} for wireless APs only

CONFIGURATION QUERIES — when asked about device config:
- Use [TOOL:get_device_config] {{"device_id": "<id>"}} for full config
- Use [TOOL:get_device_config] {{"device_id": "<id>", "section": "radius"}} for one section
- Use [TOOL:validate_device_config] to check for errors
- Use [TOOL:edit_device_config] to apply fixes — HITL approval required
- Use the diff tool (if available in AVAILABLE TOOLS) to see uncommitted config changes

SERVICE OPERATIONS — for service restarts and rollbacks (HITL required, approval card appears):
- Use the appropriate HITL-flagged tool from the AVAILABLE TOOLS list above.
- These always trigger an approval card before execution.

TOOL RESULT HANDLING — when a tool returns an error or empty result:
- If a tool returns "[Error]" or "not found" or "No devices found": report this fact clearly to the user. Do NOT invent or hallucinate data. Say what the tool returned.
- If list_devices returns an empty list: tell the user "No devices are currently registered in the system."
- If get_syslog or a device query returns "Device not found": tell the user the device is not reachable or not in inventory.
- NEVER synthesise or fabricate log entries, device data, or metrics that were not returned by a tool.
- An empty or error result IS a valid answer — report it honestly, then suggest what the user could do next.

STOP CONDITION: Once you have gathered enough information to fully answer the user's question, write your final analysis WITHOUT any [TOOL:] line.
- For single-device queries: one tool call is usually enough — summarise after that result.
- For multi-device queries (e.g. "check all devices at <site>"): call the tool once per device, then summarise ALL results together. Do NOT stop after the first device.
- NEVER call the same tool with the same arguments twice.

LARGE DATA STRATEGY — when reading a stored result page by page:
- After EACH page, write 2-3 sentences of key findings in your response BEFORE calling the next page.
- Example: "Page 1 findings: 3 flows to port 3389 (RDP) from internal IPs — potential lateral movement."
- These findings are saved to memory and recalled when you write the final analysis.
- When all pages are read (Has more: False), write your complete analysis using all recalled findings.
- Do NOT try to hold all data in memory — write findings incrementally.

VISUAL OUTPUT — use diagrams when they make answers clearer:
- For network topology / 组网 questions, draw an ASCII tree or box-and-line diagram showing how devices connect (core → access → AP, with router at the edge).
- For relationships, hierarchies, sequences, or flows, an ASCII diagram is often clearer than a bullet list.
- For richer renderable diagrams, you may emit a fenced ```mermaid block (graph TD, sequenceDiagram, flowchart, etc.) — the UI renders it.
- Keep diagrams concise. Use real device IDs and IPs from tool results, never invent them.
- Pair the diagram with a short prose summary so the user gets both views.

RESPONSE STRUCTURE — write answers the UI can render cleanly:
- The frontend renders your reply as Markdown. Use it. Do NOT cram everything into one paragraph.
- Use `## Heading` for top-level sections (e.g. ## 概览, ## 详细分布, ## 拓扑, ## 总结).
- Use bullet lists (`- item`) for enumerations like device lists, findings, recommendations.
- Use `**bold**` for key terms (device IDs, status flags, totals) — but sparingly.
- Use a Markdown table when comparing 3+ items across the same fields.
- For diagrams, ALWAYS put them inside a ```mermaid or plain ``` fenced code block on their own lines — never inline.
- Keep each paragraph focused on one idea. Break long answers into clear sections.
- End with a one-line summary or a call-to-action ("如需查看…，请告诉我设备 ID。").

ASCII topology example (generic — replace placeholder names with real device IDs from tool results):
```
                    [<router>]   ← edge
                         │
              ┌──────────┴──────────┐
         [<core-1>]              [<core-2>]   ← redundant pair
              │                      │
     ┌────────┼────────┐       ┌─────┴──────┐
[<acc-1>] [<acc-2>] [<acc-3>]            …
     │           │           │
   <ap-a,b>  <ap-c,d>      …                ← Wi-Fi APs
```

Mermaid example (generic — replace placeholder names with real device IDs):
```mermaid
graph TD
  R[<router>] --> C1[<core-1>]
  R --> C2[<core-2>]
  C1 --> A1[<acc-1>]
  C1 --> A2[<acc-2>]
  C2 --> A3[<acc-3>]
  A1 --> AP1[<ap-a>]
  A1 --> AP2[<ap-b>]
```

{extra_tools_section}

{skill_summary}

{confirmed_facts_section}
"""

    INTENT_SYSTEM = """Classify the IT operations query into exactly one intent type.
Return ONLY valid JSON, no other text.

Intent types:
- destructive_op: involves restarting, reloading, rollback, delete, drain, failover, wipe, shutdown
- alert_analysis: involves analysing alerts, incidents, P0/P1/P2 events, outages
- trend_prediction: involves predicting, forecasting, trending, capacity planning
- general_query: any other diagnostic, status check, or information request

Return format:
{"intent_type": "...", "confidence": 0.0-1.0, "intent_summary": "one sentence description"}"""

    def __init__(self, model: str, temperature: float = 0.1,
                 max_tokens: int = 2048,
                 capabilities: "LLMCapabilitiesProto | None" = None) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        # Per-model behaviour (thinking_tag, format_compliance, etc.).
        # If None, callers get default qwen3.5-compatible behaviour via
        # the legacy substring detection paths.
        self.capabilities = capabilities
        # Retrieval framework attachments (optional, set via attach_retrieval()).
        # When None, _build_system_prompt falls back to legacy full-catalog dump.
        self._tool_retriever:     Any = None
        self._skill_retriever:    Any = None
        self._meta_tool_registry: Any = None
        # D1 (Sprint 3, 2026-05): concurrency cap. The semaphore is created
        # lazily on first use (inside the running loop) because asyncio
        # primitives must bind to the active event loop, and __init__ runs
        # before uvicorn's loop exists. 0 = unlimited (legacy). Configured
        # via set_max_concurrent_calls() from main.py after construction.
        self._max_concurrent_calls: int = 0
        self._llm_semaphore: Any = None
        self._llm_sem_loop: Any = None   # loop the semaphore was bound to

    def set_max_concurrent_calls(self, n: int) -> None:
        """Set the in-flight LLM call cap. 0 disables limiting.

        Called from main.py with cfg.llm.max_concurrent_calls. The actual
        Semaphore is created lazily in _acquire_slot() so it binds to the
        running event loop, not whatever loop (if any) existed at startup.
        """
        self._max_concurrent_calls = max(0, int(n))
        # Reset any previously-created semaphore so the new limit applies.
        self._llm_semaphore = None
        self._llm_sem_loop = None
        logger.info(
            "LLMEngine: max_concurrent_calls=%s",
            self._max_concurrent_calls or "unlimited",
        )

    def _get_semaphore(self) -> Any:
        """Lazily create + return the semaphore bound to the current loop.

        Returns None when limiting is disabled (max_concurrent_calls=0).
        Re-creates the semaphore if the event loop changed (e.g. tests that
        spin up fresh loops) to avoid 'bound to a different loop' errors.
        """
        if self._max_concurrent_calls <= 0:
            return None
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None   # no running loop → can't gate; run unbounded
        if self._llm_semaphore is None or self._llm_sem_loop is not loop:
            self._llm_semaphore = asyncio.Semaphore(self._max_concurrent_calls)
            self._llm_sem_loop = loop
        return self._llm_semaphore

    def attach_retrieval(
        self,
        *,
        tool_retriever:     Any = None,
        skill_retriever:    Any = None,
        meta_tool_registry: Any = None,
    ) -> None:
        """Attach the prompt-time retrieval framework. Idempotent — call any
        time after construction. Pass None to leave a slot unchanged."""
        if tool_retriever     is not None: self._tool_retriever     = tool_retriever
        if skill_retriever    is not None: self._skill_retriever    = skill_retriever
        if meta_tool_registry is not None: self._meta_tool_registry = meta_tool_registry
        logger.info(
            "LLMEngine: retrieval attached  tool=%s skill=%s meta=%s",
            getattr(tool_retriever, "name", None),
            getattr(skill_retriever, "name", None),
            "yes" if meta_tool_registry else "no",
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMEngine":
        backend = cfg.get("backend", "mock").lower()
        model   = cfg.get("model", "mistral")
        temp    = cfg.get("temperature", 0.1)
        max_tok = cfg.get("max_tokens", 2048)

        if backend == "ollama":
            return OllamaEngine(
                model=model, temperature=temp, max_tokens=max_tok,
                base_url=cfg.get("base_url", "http://localhost:11434"),
                capabilities=cfg.get("capabilities"),
            )
        if backend == "openai":
            return OpenAIEngine(
                model=model, temperature=temp, max_tokens=max_tok,
                api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                base_url=cfg.get("base_url"),
                capabilities=cfg.get("capabilities"),
            )
        if backend == "anthropic":
            return AnthropicEngine(
                model=model, temperature=temp, max_tokens=max_tok,
                api_key_env=cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
                capabilities=cfg.get("capabilities"),
            )
        # Default: mock
        return MockEngine(model=model, temperature=temp, max_tokens=max_tok)

    async def call(
        self,
        query:   str,
        context: str,
        state:   Any = None,
        skill_catalog: Any = None,
    ) -> str:
        """
        Main entry point replacing AgentRuntimeLoop._call_llm().
        Returns the LLM's response text (may include [TOOL:...] directives).
        """
        raise NotImplementedError

    async def classify_intent(self, query: str) -> IntentResult:
        """
        Classify query intent for HITL graph intent_classifier_node.
        Returns IntentResult with type, confidence, and summary.
        """
        raise NotImplementedError


    # ─────────────────────────────────────────────────────────────────
    # Section builders (retriever-aware)
    # ─────────────────────────────────────────────────────────────────

    def _build_tools_section(
        self,
        *,
        query:              Optional[str],
        tool_retriever:     Any,
        meta_tool_registry: Any,
        tool_registry:      dict | None,
    ) -> str:
        """Build the AVAILABLE TOOLS prompt section.

        Three layers, concatenated in order:
          1. META TOOLS (always injected, from MetaToolRegistry)
          2. SAFETY-NET TOOLS (always injected — HITL gate would not fire
             unless the LLM emits the call, so destructive tools must be
             visible regardless of query)
          3. RETRIEVED TOOLS (top-K from query, when retriever + query present)

        Falls back to the full ToolLoader dump when no retriever / no query.
        """
        try:
            from config import cfg as _app_cfg
            tool_top_k = int(getattr(_app_cfg.retrieval, "tool_top_k", 5))
            extra_always = list(
                getattr(_app_cfg.retrieval, "always_inject_extra_tools", []) or []
            )
            hitl_names = list(getattr(_app_cfg.tools, "hitl_tool_names", []) or [])
        except Exception:
            tool_top_k, extra_always, hitl_names = 5, [], []

        lines: list[str] = []

        # 1) Meta tools
        if meta_tool_registry is not None:
            try:
                meta_block = meta_tool_registry.build_prompt_section()
                if meta_block:
                    lines.append(meta_block)
            except Exception as exc:
                logger.warning("MetaToolRegistry section failed: %s", exc)

        # 2/3) Retrieval-driven listing
        if tool_retriever is not None and query:
            try:
                # Always-inject tools (HITL safety net + extras)
                always_inject_set = set(hitl_names) | set(extra_always)

                # Retrieve top-K matching the query
                res = tool_retriever.retrieve(query, top_k=tool_top_k)
                retrieved_ids = [m.id for m in res.matches]

                # Compute the union: always-injected + retrieved (no dups)
                final_ids: list[str] = []
                seen: set[str] = set()
                for tid in list(always_inject_set) + retrieved_ids:
                    if tid in seen:
                        continue
                    seen.add(tid)
                    final_ids.append(tid)

                # Look up full metadata for each (the retriever item already
                # contains description/parameters/tags from the corpus adapter)
                # but tags/HITL etc are stored on the corpus item — pull them.
                # We stored full metadata in retriever items, so look them up.
                items_by_id = {m.id: m.item for m in res.matches}
                # For always-inject IDs not in retrieved set, try to load from corpus.
                # Use the retriever's internal item list if present.
                _all_corpus = getattr(tool_retriever, "corpus", None) or []  # works through wrappers
                for it in _all_corpus:
                    if it["id"] in always_inject_set and it["id"] not in items_by_id:
                        items_by_id[it["id"]] = it

                tool_lines = ["AVAILABLE TOOLS (top-K matched + safety-net):"]
                for tid in final_ids:
                    info = items_by_id.get(tid)
                    if info is None:
                        # Tool not in retriever index (shouldn't happen) — skip
                        continue
                    hitl = " ⚠HITL" if info.get("hitl") else ""
                    tool_lines.append(
                        f"  [TOOL:{tid}]{hitl} — {info.get('description','')[:140]}"
                    )
                    params = info.get("parameters") or {}
                    if params:
                        tool_lines.append(
                            "    Args: " + ", ".join(list(params.keys())[:6])
                        )
                tool_lines.append(
                    "  (use [TOOL:list_tools] to discover other tools by description)"
                )
                lines.append("\n".join(tool_lines))
                return "\n\n".join(p for p in lines if p)
            except Exception as exc:
                logger.warning(
                    "Retriever-driven tool section failed (%s) — falling back to full dump",
                    exc,
                )

        # Fallback: full dump (legacy behaviour)
        try:
            from tools.loader import ToolLoader as _TL
            import config as _cfg
            _tl = _TL(mode=_cfg.cfg.mode, profile=_cfg.cfg.agent.profile)
            full = _tl.tool_section_for_prompt()
            if tool_registry:
                _mode_names = set(_tl.build_metadata().keys())
                _extra = {n for n in tool_registry if n not in _mode_names}
                if _extra:
                    extra_block = ["\nUPLOADED/REGISTERED TOOLS:"]
                    for n in sorted(_extra):
                        extra_block.append(f'  [TOOL:{n}] {{"<arg>": "<value>"}}')
                    full = full + "\n" + "\n".join(extra_block)
            lines.append(full)
        except Exception:
            if tool_registry:
                _ll = ["AVAILABLE TOOLS (use [TOOL:name] format):"]
                for n in sorted(tool_registry):
                    _ll.append(f"  [TOOL:{n}]")
                lines.append("\n".join(_ll))
        return "\n\n".join(p for p in lines if p)

    def _build_skills_section(
        self,
        *,
        query:           Optional[str],
        skill_retriever: Any,
        skill_catalog:   Any,
    ) -> str:
        """Build the Available skills prompt section.

        Retriever-driven top-K when available; falls back to
        skill_catalog.format_summary() (legacy full dump).
        """
        try:
            from config import cfg as _app_cfg
            skill_top_k = int(getattr(_app_cfg.retrieval, "skill_top_k", 3))
        except Exception:
            skill_top_k = 3

        # Retrieval path
        if skill_retriever is not None and query:
            try:
                res = skill_retriever.retrieve(query, top_k=skill_top_k)
                if not res.matches:
                    return "Available skills: (none matched — use [TOOL:list_skills] to search)"
                lines = [f"Available skills (top {len(res.matches)} for query):"]
                for m in res.matches:
                    info = m.item
                    hitl = " ⚠HITL" if info.get("hitl") else ""
                    lines.append(
                        f"  [{m.id}]{hitl} (score={m.score:.2f}) — "
                        f"{info.get('description','')[:120]}"
                    )
                lines.append(
                    "  (use [TOOL:list_skills] for more, [SKILL_LOAD:id] to read full guide)"
                )
                return "\n".join(lines)
            except Exception as exc:
                logger.warning(
                    "Retriever-driven skill section failed (%s) — falling back",
                    exc,
                )

        # Legacy fallback
        if skill_catalog:
            try:
                return "Available skills:\n" + skill_catalog.format_summary()
            except Exception:
                pass
        return ""

    def _build_system_prompt(
        self, context: str, skill_catalog: Any = None,
        confirmed_facts: list[str] | None = None,
        tool_registry: dict | None = None,
        *,
        query:           Optional[str] = None,
        turn_no:         int           = 1,
        tool_retriever:  Any           = None,
        skill_retriever: Any           = None,
        meta_tool_registry: Any        = None,
    ) -> str:
        # ── Tools section — retriever-driven when available ────────────
        # NEW: cfg.retrieval drives top-K tool selection so the prompt only
        # contains tools relevant to the current query, plus always-injected
        # meta tools (list_tools, list_skills, ...) and HITL safety tools.
        # FALLBACK: if no retriever / no query / retrieval disabled, the full
        # ToolLoader catalog is dumped (legacy behaviour preserved).
        extra_tools_section = self._build_tools_section(
            query=query,
            tool_retriever=tool_retriever,
            meta_tool_registry=meta_tool_registry,
            tool_registry=tool_registry,
        )

        # ── Skill summary — retriever-driven when available ──────────
        skill_summary = self._build_skills_section(
            query=query,
            skill_retriever=skill_retriever,
            skill_catalog=skill_catalog,
        )

        # ── Confirmed facts (+ tool ledger + prior analysis) ───────────────────────
        facts_section = ""
        if confirmed_facts:
            tool_exec_lines, prev_analysis_lines = [], []
            alias_lines, semantic_facts = [], []
            for _f in confirmed_facts:
                if _f.startswith("TOOL_EXEC: "):
                    tool_exec_lines.append(_f[len("TOOL_EXEC: "):])
                elif _f.startswith("PREV_ANALYSIS: "):
                    prev_analysis_lines.append(_f[len("PREV_ANALYSIS: "):])
                elif _f.startswith("ENTITY_ALIAS: "):
                    # User said one term, system has a different real name.
                    # Carry this prominently to every turn so the LLM stops
                    # re-rediscovering the same correction.
                    alias_lines.append(_f[len("ENTITY_ALIAS: "):])
                else:
                    semantic_facts.append(_f)
            _parts = []
            # Aliases first — strongest priority since they fix entity
            # resolution at the foundation. If the LLM keeps tool-call'ing
            # the user's wrong term, every downstream observation will
            # also be wrong, so this must be unmissable.
            if alias_lines:
                _parts.append(
                    "ENTITY ALIASES (the user's terms map to these real "
                    "system names — USE THE REAL NAMES in all tool calls):\n"
                    + "\n".join(f"  ⚠ {l}" for l in alias_lines[-10:])
                )
            if tool_exec_lines:
                # Filter: show stored-data entries (with ref=) and skip per-page reads
                _filtered = [
                    l for l in tool_exec_lines[-20:]
                    if not (l.startswith("read_stored_result|") and "pages_read" not in l
                            and "inline" not in l)
                ]
                if _filtered:
                    _parts.append(
                        "DATA ALREADY FETCHED (do NOT re-fetch — reuse existing ref_ids):\n"
                        + "\n".join(f"  ✓ {l}" for l in _filtered[-12:])
                    )
            if prev_analysis_lines:
                _parts.append(
                    "PREVIOUS ANALYSIS RESULTS (context for follow-up questions):\n"
                    + "\n".join(f"  → {l[:300]}" for l in prev_analysis_lines[-3:])
                )
            if semantic_facts:
                _parts.append(
                    "Confirmed facts from this session:\n"
                    + "\n".join(f"  • {f}" for f in semantic_facts[-8:])
                )
            facts_section = "\n\n".join(_parts)

        # ── Pick full vs slim template based on turn ───────────────
        # Saves ~3000 chars/turn after the LLM has learnt the format.
        try:
            from config import cfg as _app_cfg
            _shorten_after = int(getattr(_app_cfg.retrieval, "shorten_tool_system_after_turn", 1))
        except Exception:
            _shorten_after = 1
        _template = (
            self.TOOL_CALL_SYSTEM_SLIM
            if turn_no > _shorten_after
            else self.TOOL_CALL_SYSTEM
        )
        system = _template.format(
            skill_summary=skill_summary,
            confirmed_facts_section=facts_section,
            extra_tools_section=extra_tools_section,
        )
        if context:
            system += f"\n\nContext:\n{context}"

        # Suppress read_stored_result instruction block when no [STORED:] is in
        # any context the LLM can see (prevents misuse with hallucinated ref_id).
        try:
            system = _strip_stored_result_section(
                system, context or "", facts_section or "", extra_tools_section or "",
            )
        except Exception as _ssrs_exc:
            logger.debug("_strip_stored_result_section failed: %s", _ssrs_exc)

        return system

    @staticmethod
    def _parse_intent_json(text: str) -> IntentResult:
        """Parse the intent classification JSON response."""
        text = text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```json?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return IntentResult(
                intent_type=data.get("intent_type", "general_query"),
                confidence=float(data.get("confidence", 0.65)),
                intent_summary=data.get("intent_summary", ""),
                candidates=[{
                    "intent":     data.get("intent_type", "general_query"),
                    "confidence": float(data.get("confidence", 0.65)),
                }],
            )
        except Exception:
            logger.warning("Failed to parse intent JSON: %r", text[:200])
            # Fallback: keyword-based
            q = text.lower()
            if any(k in q for k in ("restart","rollback","delete","drain","failover")):
                return IntentResult("destructive_op", 0.90, "Destructive operation detected",
                                    [{"intent":"destructive_op","confidence":0.90}])
            return IntentResult("general_query", 0.60, "General IT ops query",
                                [{"intent":"general_query","confidence":0.60}])


# ---------------------------------------------------------------------------
# Ollama engine
# ---------------------------------------------------------------------------

class OllamaEngine(LLMEngine):
    """
    Ollama local LLM engine.

    Supports standard models (mistral, llama3.2, qwen2.5) and thinking
    models (qwen3.5:27b, qwen3.5:35b, deepseek-r1). Thinking models emit
    <think>...</think> blocks which are stripped before tool parsing and
    before the response reaches the loop — preventing the tool parser from
    finding [TOOL:] directives inside reasoning text, and preventing the
    "thinking block" from being shown to the user as part of the answer.
    """

    # Model name substrings that identify thinking models
    THINKING_MODELS = {"qwen3", "qwen3.5", "deepseek-r1", "deepseek-r2", "qwq", "marco-o1"}

    def __init__(self, model: str, temperature: float, max_tokens: int,
                 base_url: str = "http://localhost:11434",
                 think: bool = False,
                 capabilities: "LLMCapabilitiesProto | None" = None) -> None:
        super().__init__(model, temperature, max_tokens, capabilities=capabilities)
        self._base_url = base_url.rstrip("/")
        self._think    = think   # passed as think= to Ollama API for thinking models

    @property
    def _is_thinking_model(self) -> bool:
        # Capability-driven (config) takes precedence.
        cap = self.capabilities
        if cap is not None:
            tag = (getattr(cap, "thinking_tag", "") or "").strip().lower()
            # Explicit "off"/"none"/"" means non-thinking model — regardless
            # of what the model NAME suggests.
            if tag in ("", "none", "off", "false", "no"):
                return False
            return True
        # Legacy fallback: substring match against the historical set.
        # Kept so callers that instantiate OllamaEngine without going
        # through from_config (e.g. tests) keep working.
        return any(k in self.model.lower() for k in self.THINKING_MODELS)

    def _strip_think(self, text: str) -> str:
        """Strip the configured reasoning tag block from model output.

        Tag defaults to "think" (qwen3.x, deepseek-r1). New models may use
        "reasoning" or something else — set llm.capabilities.thinking_tag
        in config.yaml. If the tag is set to "none"/"off"/empty, no
        stripping happens (non-thinking models).
        """
        import re
        cap = self.capabilities
        tag = (getattr(cap, "thinking_tag", "think") if cap else "think") or "think"
        tag = tag.strip().lower()
        if tag in ("", "none", "off", "false", "no"):
            return text.strip()
        # Escape the tag in case it contains regex metachars (defensive)
        pat = rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
        cleaned = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        turns           = getattr(state, "turns", 1) if state else 1

        # On Turn 2+: if THIS TURN's tool_outputs are non-empty, add a stop
        # instruction so the model synthesizes rather than calling another tool.
        # We use state._current_tool_outputs_count (set by loop.py) to distinguish
        # On turn 2+: if we already have multiple device results, nudge the LLM
        # to synthesise rather than keep calling more tools. But allow sequential
        # multi-device calls (e.g. validate each of 7 devices one at a time).
        # Build "already checked" section from accumulated tool_outputs keys
        # This tells the LLM exactly which device/args combos it has already run.
        _tool_output_keys = getattr(state, "_tool_output_keys", []) if state else []
        _cur_tool_count   = getattr(state, "_current_tool_outputs_count", 0) if state else 0
        _max_tool_calls   = getattr(self, "_max_tool_calls", 20)

        stop_note = ""
        if _tool_output_keys:
            # Show LLM what has already been run this session
            checked_lines = []
            import json as _json
            for k in _tool_output_keys:
                if "|" in k:
                    tname, args_str = k.split("|", 1)
                    try:
                        args = _json.loads(args_str)
                        if tname == "read_stored_result":
                            # Normalise ref_id for display and show paging status
                            rid = args.get("ref_id", "?").strip("[]")
                            if ":" in rid:
                                rid = rid.rsplit(":", 1)[-1].strip()
                            off = args.get("offset", 0)
                            # Check if this read had more data available
                            raw_val = (getattr(state, "_tool_outputs_raw", {}) or {}).get(k, "")
                            has_more = "Has more: True" in raw_val
                            if has_more:
                                next_off = off + 2000
                                next_call = '{' + f'"ref_id": "{rid}", "offset": {next_off}' + '}'
                                checked_lines.append(
                                    f"  - {tname}(ref_id={rid}, offset={off}) done "
                                    f"— MORE DATA: [TOOL:read_stored_result] {next_call} (write key findings in your response first)"
                                )
                            else:
                                checked_lines.append(
                                    f"  - {tname}(ref_id={rid}, offset={off}) done — all pages read"
                                )
                        else:
                            dev = args.get("device_id") or args.get("site") or args_str[:40]
                            checked_lines.append(f"  - {tname}({dev}) ✓ done")
                    except Exception:
                        checked_lines.append(f"  - {k} ✓ done")
                else:
                    checked_lines.append(f"  - {k} ✓ done")
            stop_note = "\n\nALREADY COMPLETED THIS SESSION:\n" + "\n".join(checked_lines)
            stop_note += "\nDo NOT repeat any of the above calls. Move to the next unchecked device."
            # Synthesis prompt: fire only when data gathering is truly complete.
            # Do NOT fire if:
            # - A read_stored_result call has "Has more: True" (more pages to read)
            # - The only results are [STORED:] labels (LLM hasn't read the data yet)
            _has_more_pages = any(
                "Has more: True" in v
                for v in (state._tool_outputs_raw.values() if hasattr(state, "_tool_outputs_raw") else [])
            )
            _all_stored = all(
                "[STORED:" in v
                for v in (state._tool_outputs_raw.values() if hasattr(state, "_tool_outputs_raw") else ["x"])
            )
            # Identify large-data tools by their registry tags (traffic/metrics)
            # so no tool names are hardcoded here
            try:
                import config as _cfg2
                from tools.loader import ToolLoader as _TL2
                _large_data_tools = {
                    n for n, info in _TL2(mode=_cfg2.cfg.mode, profile=_cfg2.cfg.agent.profile).build_metadata().items()
                    if any(t in info.get("tags", []) for t in ("traffic", "metrics"))
                }
            except Exception:
                _large_data_tools = set()
            _n_real_results = sum(
                1 for k in _tool_output_keys
                if k.split("|")[0] not in _large_data_tools or "|" not in k
            )
            # Use real-results count (excluding big-data tools that legitimately
            # generate many calls for paging) so the synthesis nudge fires only
            # when the model has actually gathered enough small-tool data.
            if _n_real_results >= 3 and not _has_more_pages and not _all_stored:
                stop_note += (
                    "\n\nYou have gathered sufficient tool results. "
                    "Provide your complete analysis and recommendations now."
                )

        if turns > 1 and _cur_tool_count >= _max_tool_calls:
            stop_note += (
                "\n\nNOTE: You have gathered enough results. "
                "Please now provide your complete analysis and recommendations. "
                "Do NOT emit any further [TOOL:...] lines."
            )

        # Pass the live tool_registry so uploaded tools appear in the system prompt
        _tool_reg = getattr(state, "_tool_registry", None) if state else None
        system = self._build_system_prompt(
            context, skill_catalog, confirmed_facts, _tool_reg,
            query=query,
            turn_no=turns,
            tool_retriever=self._tool_retriever,
            skill_retriever=self._skill_retriever,
            meta_tool_registry=self._meta_tool_registry,
        )
        if stop_note:
            system += stop_note

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ]

        # ── Conversation logging ─────────────────────────────────────────────
        # Controlled by LLM_LOG_DETAIL env var (no restart needed — read per call):
        #   off     → only char/token counts shown (current default)
        #   compact → first 400 chars of system prompt + full user query + response
        #   full    → complete system prompt, full user query, full response
        #
        # Set before starting:  export LLM_LOG_DETAIL=compact
        # Or switch live:       export LLM_LOG_DETAIL=full  (takes effect next call)
        import os as _os
        _detail = _os.getenv("LLM_LOG_DETAIL", "compact").lower()
        _sep    = "─" * 72

        if _detail in ("compact", "full"):
            _sys_log = system if _detail == "full" else (system[:400] + (" …" if len(system) > 400 else ""))
            logger.info(
                "LLM▶ TURN %d  model=%s  system=%d chars  user=%d chars\n"
                "%s\n[SYSTEM]\n%s\n%s\n[USER]\n%s\n%s",
                turns, self.model, len(system), len(query),
                _sep, _sys_log, _sep, query, _sep,
            )
        else:
            logger.info(
                "LLM▶ turn=%d model=%s system_chars=%d user_chars=%d",
                turns, self.model, len(system), len(query),
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM REQUEST (full) turn=%d\n%s\n[SYSTEM]\n%s\n%s\n[USER]\n%s\n%s",
                turns, _sep, system, _sep, query, _sep,
            )

        # ── Native tools branch ──────────────────────────────────────
        # When the model + Ollama version support OpenAI-style `tools`,
        # ship the JSON-schema tool specs alongside the messages and get
        # back STRUCTURED tool_calls instead of free-text. We then convert
        # those tool_calls into [TOOL:name] {...} lines appended to the
        # visible content, so runtime/loop.py keeps using its existing
        # directive parser — zero changes downstream.
        #
        # See integrations/DESIGN.md §3 for the bridge rationale: this
        # keeps HITL, batch executor, chunk queue, skill evolver, and
        # MemoryAdapter wiring all unchanged — native tools is a pure
        # runtime upgrade, not an architectural change. Config gated via
        # `llm.capabilities.supports_native_tools` (default False).
        cap = self.capabilities
        _use_native = bool(getattr(cap, "supports_native_tools", False))
        if _use_native:
            tools_payload = self._build_native_tools_payload(state, skill_catalog)
        else:
            tools_payload = None

        raw = await self._chat(messages, tools=tools_payload)

        # Normalize to a single `result` string for downstream parsing.
        # In native mode `raw` is {"content": str, "tool_calls": [...]} and we
        # synthesize directive lines from tool_calls so the loop's existing
        # _parse_tool_calls path picks them up unchanged.
        if isinstance(raw, dict):
            content = raw.get("content", "") or ""
            tool_calls = raw.get("tool_calls", []) or []
            if tool_calls:
                # Append synthesized directives. We use the SAME wire format
                # the loop's parser already accepts — that's the bridge.
                import json as _json
                synth_lines: list[str] = []
                for tc in tool_calls:
                    name = tc.get("name") or ""
                    args = tc.get("arguments") or {}
                    if not name:
                        continue
                    try:
                        args_str = _json.dumps(args, ensure_ascii=False, separators=(",", ":"))
                    except (TypeError, ValueError):
                        args_str = "{}"
                    synth_lines.append(f"[TOOL:{name}] {args_str}")
                # Place directives at the end, separated by blank line so
                # the user-facing prose (if any) and the directive don't
                # share a line — matches how the prompt instructs the
                # text-mode model to lay them out.
                result = (content + ("\n\n" if content.strip() else "") +
                          "\n".join(synth_lines))
                logger.info(
                    "OllamaEngine: native tools → %d tool_call(s) synthesized as "
                    "[TOOL:] directives — names=%s",
                    len(tool_calls), [tc.get("name") for tc in tool_calls],
                )
            else:
                result = content
        else:
            result = self._strip_think(raw)

        # Consistent detection: use the directive parser's tolerance so
        # trace and log entries match what the parser will actually see.
        # A substring check misses variants like `[TOOL: name]` (space
        # after colon) that the parser correctly recognizes.
        from runtime.directive_parser import has_any_tool_directive as _has_tool_directive
        _has_tool = _has_tool_directive(result)

        if _detail in ("compact", "full"):
            _resp_log = result if _detail == "full" else (result[:400] + (" …" if len(result) > 400 else ""))
            logger.info(
                "LLM◀ TURN %d  chars=%d  tool_call=%s\n%s\n[RESPONSE]\n%s\n%s",
                turns, len(result), _has_tool, _sep, _resp_log, _sep,
            )
        else:
            logger.info(
                "LLM◀ turn=%d response_chars=%d has_tool_call=%s",
                turns, len(result), _has_tool,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM RESPONSE (full) turn=%d\n%s\n%s\n%s",
                turns, _sep, result, _sep,
            )

        # CAP 6: attach trace to state so stream() can yield it as an SSE event
        if state is not None:
            if not hasattr(state, "_llm_traces"):
                state._llm_traces = []
            state._llm_traces.append({
                "turn":           turns,
                "model":          self.model,
                "system_chars":   len(system),
                "context_chars":  len(context),
                "user_chars":     len(query),
                "response_chars": len(result),
                "has_tool_call":  _has_tool,
                "system_preview": system[:300],
                "response_preview": result[:300],
            })

        return result

    async def classify_intent(self, query: str) -> IntentResult:
        messages = [
            {"role": "system", "content": self.INTENT_SYSTEM},
            {"role": "user",   "content": f"Query: {query}"},
        ]
        raw  = await self._chat(messages)
        text = self._strip_think(raw)
        return self._parse_intent_json(text)

    def _build_native_tools_payload(
        self, state: Any, skill_catalog: Any,
    ) -> list[dict] | None:
        """Build the Ollama `tools` array for this turn.

        Strategy:
          1. Use the existing tool retriever to pick top-K names matching
             the query — same shortlist the text-protocol prompt uses, so
             native and text modes see equivalent tool surfaces.
          2. Add the HITL safety-net names (always-inject) so the model
             can never lose visibility on destructive ops.
          3. Look each name up in the schema registry; convert to Ollama
             tool spec via schema/ollama_export.

        Returns None if no schemas are registered (caller falls back to
        text protocol — see call() for the branch logic). Returns an
        empty list only if the registry is non-empty but no names matched,
        which would be a misconfiguration worth logging.
        """
        try:
            from schema.registry import get_schema_registry
            from schema.ollama_export import export_for_ollama
        except ImportError:
            logger.warning(
                "OllamaEngine native tools requested but schema/ollama_export "
                "is missing — falling back to text protocol"
            )
            return None

        reg = get_schema_registry()
        if len(reg) == 0:
            # Schema registry not populated yet (e.g. very early startup
            # before tool registration). Skip native mode this turn —
            # text protocol still works (the model knows tools from the
            # system prompt either way).
            return None

        # Determine which tools to surface. The retriever returns a query-
        # ranked shortlist; we add HITL names so destructive ops are never
        # accidentally absent.
        names: set[str] = set()

        # Always-inject set: HITL tool names + meta-tools that should be
        # available every turn.
        always: set[str] = set()
        if self._tool_retriever is not None:
            try:
                # Mirror the text-protocol path: top-K by query similarity.
                # _max_tools_in_prompt is set by the loop based on context budget.
                top_k = getattr(state, "_max_tools_in_prompt", 0) if state else 0
                top_k = top_k or 12   # 12 is a reasonable default for native mode
                # `query` lives in state for retriever consumers
                q = getattr(state, "_current_query", "") if state else ""
                if q:
                    res = self._tool_retriever.retrieve(q, top_k=top_k)
                    for m in res.matches:
                        names.add(m.id)
            except Exception as exc:
                # Retrieval failures must not kill the turn — fall back to
                # exporting every registered schema (token cost is the
                # only downside).
                logger.debug(
                    "native tools shortlist retrieval failed (%s); "
                    "exporting full registry", exc,
                )
                names = {s.tool_name for s in reg.list_all()}

        # HITL safety-net: union in the destructive tool names so the
        # model can always reach them.
        for hitl_name in getattr(self, "_hitl_tool_names", []) or []:
            if reg.has(hitl_name):
                always.add(hitl_name)
        names |= always

        # If retrieval was disabled / empty, export everything we have.
        if not names:
            names = {s.tool_name for s in reg.list_all()}

        tools = export_for_ollama(reg.list_all(), allowed_names=names)
        if not tools:
            logger.warning(
                "OllamaEngine: native tools enabled but 0 tools matched "
                "(names_requested=%d registry_size=%d) — text protocol fallback",
                len(names), len(reg),
            )
            return None
        return tools

    async def _chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ) -> str | dict:
        """Send a chat request to Ollama.

        Returns:
          - `str` (legacy / text protocol): just the assistant content.
          - `dict` (native-tools mode): {"content": str, "tool_calls": [...]}.
            tool_calls is empty when the model chose to reply with prose
            instead of a tool call this turn.

        The dual return is so callers that don't pass `tools=` keep their
        single-string interface (and all their existing parsing). Callers
        that DO pass tools get the structured form back.

        See `schema/ollama_export.py` for the tool-spec converter and
        `OllamaEngine.call()` for the bridge that synthesizes [TOOL:]
        directives from structured tool_calls so downstream parsers
        (runtime/loop.py + directive_parser) don't need to change.
        """
        # Sprint-3-pre: optional tracing. start_span degrades to no-op
        # when OTel is disabled / not installed, so this is zero-cost.
        from runtime.tracing import start_span
        _msg_chars = sum(len(m.get("content", "") or "") for m in (messages or []))
        with start_span(
            "llm.call",
            **{
                "llm.model":            getattr(self, "_model", "unknown"),
                "llm.message.count":    len(messages or []),
                "llm.message.chars":    _msg_chars,
                "llm.native_tools":     bool(tools),
                "llm.tools.count":      len(tools or []),
            },
        ) as _span:
            # D1: gate through the concurrency semaphore so a single query's
            # fan-out (20+ internal calls) can't saturate Ollama. When the
            # cap is disabled (0) or there's no running loop, _get_semaphore
            # returns None and we run unbounded (legacy behaviour).
            # C1: record LLM call count + duration + in-flight gauge.
            from runtime import metrics as _metrics
            _model_label = getattr(self, "model", None) or "unknown"
            _sem = self._get_semaphore()
            if _sem is None:
                with _metrics.track_active_llm(), _metrics.time_llm_call(_model_label):
                    return await self._chat_impl(messages, tools=tools, _span=_span)
            async with _sem:
                with _metrics.track_active_llm(), _metrics.time_llm_call(_model_label):
                    return await self._chat_impl(messages, tools=tools, _span=_span)

    async def _chat_impl(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        _span: Any = None,
    ) -> str | dict:
        """Internal _chat body — wrapped by _chat() for tracing."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("pip install httpx to use OllamaEngine")

        payload: dict = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        # Native tools: pass through to Ollama. Ollama (≥ 0.4) supports the
        # OpenAI-style `tools` field; the model receives them as a structured
        # spec and returns `message.tool_calls` instead of text args.
        # Avoid sending an empty `tools=[]` — Ollama would still emit the
        # tools-mode handshake, costing tokens and confusing smaller models.
        if tools:
            payload["tools"] = tools
        # Ollama ≥ 0.6 supports think= parameter for thinking models
        if self._is_thinking_model:
            payload["think"] = self._think  # False = suppress think blocks in API response

        # Use separate connect / read timeouts.
        # connect: fail fast if Ollama not running.
        # read: generous — large contexts on qwen3.5:27b can take 3-5 min.
        _timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)

        for _attempt in range(2):   # one retry on ReadTimeout
            try:
                async with httpx.AsyncClient(timeout=_timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/chat", json=payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # CAP 6: log token usage from Ollama response
                    usage = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
                    if any(usage):
                        logger.info(
                            "LLM tokens: prompt=%d completion=%d total=%d model=%s",
                            usage[0], usage[1], sum(usage), self.model,
                        )
                    msg = data.get("message") or {}
                    content = self._strip_think(msg.get("content") or "")
                    # Native-tools mode: return structured dict so caller can
                    # render args directly without parsing free-text. Ollama
                    # returns tool_calls as a list of:
                    #   {"function": {"name": "X", "arguments": {<dict>} or "json-str"}}
                    # We normalise arguments to dict here so downstream code
                    # never has to think about it.
                    if tools is not None:
                        raw_calls = msg.get("tool_calls") or []
                        norm_calls = []
                        for rc in raw_calls:
                            fn = (rc or {}).get("function") or {}
                            name = fn.get("name") or ""
                            args = fn.get("arguments") or {}
                            if isinstance(args, str):
                                # Some Ollama versions return arguments as a
                                # JSON-encoded string; some return a dict.
                                # Coerce to dict so consumers don't branch.
                                import json as _json
                                try:
                                    args = _json.loads(args) if args.strip() else {}
                                except _json.JSONDecodeError:
                                    logger.warning(
                                        "OllamaEngine: tool_calls[%r].arguments was a "
                                        "non-JSON string — passing through as {'_raw': ...}",
                                        name,
                                    )
                                    args = {"_raw": args}
                            if name:
                                norm_calls.append({"name": name, "arguments": args})
                        return {"content": content, "tool_calls": norm_calls}
                    # Even with think=False in the API, strip any residual <think> tags
                    return content
            except httpx.ReadTimeout:
                if _attempt == 0:
                    logger.warning(
                        "OllamaEngine: ReadTimeout on attempt 1 — retrying (context=%d chars)...",
                        len(str(payload.get("messages", "")))
                    )
                    await asyncio.sleep(2)
                    continue
                logger.error("OllamaEngine: ReadTimeout after retry — context may be too large")
                raise RuntimeError(
                    f"Ollama timed out after 300s (context too large or model overloaded). "
                    f"Try: reduce context, use a faster model, or increase Ollama resources."
                )
            except Exception as exc:
                logger.error("OllamaEngine error: %s", exc)
                raise RuntimeError(
                    f"Ollama call failed: {exc}. "
                    f"Is Ollama running at {self._base_url}? "
                    f"Run: ollama serve && ollama pull {self.model}"
                )

    async def stream_call(self, query: str, context: str,
                          state: Any = None, skill_catalog: Any = None):
        """Streaming version — yields text chunks."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("pip install httpx to use OllamaEngine")

        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        system = self._build_system_prompt(
            context, skill_catalog, confirmed_facts,
            query=query,
            turn_no=turns,
            tool_retriever=self._tool_retriever,
            skill_retriever=self._skill_retriever,
            meta_tool_registry=self._meta_tool_registry,
        )
        payload = {
            "model":    self.model,
            "messages": [{"role":"system","content":system},{"role":"user","content":query}],
            "stream":   True,
            "options":  {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            text  = chunk.get("message", {}).get("content", "")
                            if text:
                                yield text
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# OpenAI engine
# ---------------------------------------------------------------------------

class OpenAIEngine(LLMEngine):
    """OpenAI API engine (GPT-4o, GPT-4o-mini)."""

    def __init__(self, model: str, temperature: float, max_tokens: int,
                 api_key_env: str = "OPENAI_API_KEY",
                 base_url: Optional[str] = None,
                 capabilities: "LLMCapabilitiesProto | None" = None) -> None:
        super().__init__(model, temperature, max_tokens, capabilities=capabilities)
        self._api_key  = os.getenv(api_key_env, "")
        self._base_url = base_url

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        system = self._build_system_prompt(
            context, skill_catalog, confirmed_facts,
            query=query,
            turn_no=turns,
            tool_retriever=self._tool_retriever,
            skill_retriever=self._skill_retriever,
            meta_tool_registry=self._meta_tool_registry,
        )
        try:
            from openai import AsyncOpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            client = AsyncOpenAI(**kwargs)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},{"role":"user","content":query}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            raise RuntimeError("pip install openai to use OpenAIEngine")
        except Exception as exc:
            logger.error("OpenAIEngine error: %s", exc)
            raise

    async def classify_intent(self, query: str) -> IntentResult:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self._api_key)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role":"system","content":self.INTENT_SYSTEM},
                    {"role":"user","content":f"Query: {query}"},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return self._parse_intent_json(resp.choices[0].message.content or "{}")
        except Exception as exc:
            logger.error("OpenAIEngine.classify_intent error: %s", exc)
            return IntentResult("general_query", 0.5, str(exc), [])


# ---------------------------------------------------------------------------
# Anthropic engine
# ---------------------------------------------------------------------------

class AnthropicEngine(LLMEngine):
    """Claude API engine (claude-sonnet-4-6, claude-haiku-4-5)."""

    def __init__(self, model: str, temperature: float, max_tokens: int,
                 api_key_env: str = "ANTHROPIC_API_KEY",
                 capabilities: "LLMCapabilitiesProto | None" = None) -> None:
        super().__init__(model, temperature, max_tokens, capabilities=capabilities)
        self._api_key = os.getenv(api_key_env, "")

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        system = self._build_system_prompt(
            context, skill_catalog, confirmed_facts,
            query=query,
            turn_no=turns,
            tool_retriever=self._tool_retriever,
            skill_retriever=self._skill_retriever,
            meta_tool_registry=self._meta_tool_registry,
        )
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self._api_key)
            resp = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role":"user","content":query}],
            )
            return resp.content[0].text if resp.content else ""
        except ImportError:
            raise RuntimeError("pip install anthropic to use AnthropicEngine")
        except Exception as exc:
            logger.error("AnthropicEngine error: %s", exc)
            raise

    async def classify_intent(self, query: str) -> IntentResult:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self._api_key)
            resp = await client.messages.create(
                model=self.model,
                max_tokens=256,
                system=self.INTENT_SYSTEM,
                messages=[{"role":"user","content":f"Query: {query}"}],
            )
            return self._parse_intent_json(resp.content[0].text if resp.content else "{}")
        except Exception as exc:
            logger.error("AnthropicEngine.classify_intent error: %s", exc)
            return IntentResult("general_query", 0.5, str(exc), [])


# ---------------------------------------------------------------------------
# Mock engine (for testing and CI)
# ---------------------------------------------------------------------------

class MockEngine(LLMEngine):
    """
    Deterministic mock engine.  No LLM required.
    Gives structured, realistic-looking responses based on keyword matching.
    Production-safe for tests and demos.
    """

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        await asyncio.sleep(0)
        q = query.lower()

        if "syslog" in q or "log" in q:
            return (
                f"I need to check syslogs for: {query}\n"
                "[TOOL:get_syslog] {\"host\": \"ap-*\", \"severity\": \"error\", \"lines\": 100}\n"
                "Checking for error patterns..."
            )
        if "device" in q or "status" in q:
            dev = re.search(r"(ap-\d+|sw-\w+|router-\w+)", q)
            device_id = dev.group(1) if dev else "sw-core-01"
            return (
                f"Checking device status for: {query}\n"
                f"[TOOL:get_device_status] {{\"device_id\": \"{device_id}\"}}\n"
                "Fetching current device metrics..."
            )
        if "interface" in q or "metric" in q or "utilisa" in q or "bandwidth" in q:
            return (
                f"Querying interface metrics for: {query}\n"
                "[TOOL:query_interface_metrics] {\"host\": \"sw-core-01\", \"interface\": \"GigE0/0\", \"duration\": \"1h\"}\n"
                "Fetching utilisation data..."
            )
        if "bgp" in q or "routing" in q or "prefix" in q:
            return (
                f"Checking BGP state for: {query}\n"
                "[TOOL:get_bgp_summary] {\"router\": \"router-01\"}\n"
                "Retrieving BGP neighbour table..."
            )
        if "ip" in q or "address" in q or "ipam" in q:
            return (
                f"Looking up IP information for: {query}\n"
                "[TOOL:search_ip_addresses] {\"prefix\": \"10.0.0.0/8\"}\n"
                "Searching IPAM database..."
            )
        if "incident" in q or "ticket" in q or "open" in q:
            return (
                f"Checking incidents for: {query}\n"
                "[TOOL:list_incidents] {\"severity\": \"P1\", \"status\": \"open\"}\n"
                "Fetching open incidents from incident management system..."
            )
        if "config" in q or "change" in q or "diff" in q:
            dev = re.search(r"(ap-\d+|sw-\w+)", q)
            device_id = dev.group(1) if dev else "sw-core-01"
            return (
                f"Checking configuration changes for: {query}\n"
                f"[TOOL:get_config_diff] {{\"device_id\": \"{device_id}\"}}\n"
                "Comparing current config with last backup..."
            )
        return (
            f"Analysing: {query}\n"
            "Based on available context, this appears to be a general network query. "
            "No specific tool call is needed at this stage. "
            "Please provide more specific details such as device name, time range, or affected service."
        )

    async def classify_intent(self, query: str) -> IntentResult:
        await asyncio.sleep(0)
        q = query.lower()
        if any(k in q for k in ("restart","rollback","delete","drain","failover","flush","shutdown")):
            return IntentResult("destructive_op", 0.95, f"Destructive operation: {query[:60]}",
                                [{"intent":"destructive_op","confidence":0.95}])
        if any(k in q for k in ("alert","alarm","p0","p1","outage","incident","down")):
            return IntentResult("alert_analysis", 0.87, f"Alert/incident analysis: {query[:60]}",
                                [{"intent":"alert_analysis","confidence":0.87}])
        if any(k in q for k in ("predict","forecast","trend","capacity","growth")):
            return IntentResult("trend_prediction", 0.82, f"Trend/prediction: {query[:60]}",
                                [{"intent":"trend_prediction","confidence":0.82}])
        return IntentResult("general_query", 0.72, f"General query: {query[:60]}",
                            [{"intent":"general_query","confidence":0.72},
                             {"intent":"alert_analysis","confidence":0.40}])


# ---------------------------------------------------------------------------
# Patching helpers — wire LLMEngine into AgentRuntimeLoop
# ---------------------------------------------------------------------------

def patch_runtime_loop(loop: Any, engine: LLMEngine) -> None:
    """
    Monkey-patch an existing AgentRuntimeLoop instance to use a real LLM engine.

    Usage:
        from integrations.clients.llm_engine import LLMEngine, patch_runtime_loop
        engine = LLMEngine.from_config({"backend": "ollama", "model": "mistral"})
        patch_runtime_loop(services["runtime_loop"], engine)
    """
    import types

    async def real_call_llm(self_loop, query: str, context: str, state: Any) -> str:
        return await engine.call(
            query=query, context=context, state=state,
            skill_catalog=self_loop._skill_catalog,
        )

    loop._call_llm = types.MethodType(real_call_llm, loop)
    logger.info(
        "patch_runtime_loop: AgentRuntimeLoop._call_llm patched → %s(%s)",
        engine.__class__.__name__, engine.model,
    )


_RISK_SYSTEM = """You are an IT security risk evaluator for network operations.
Analyse the query and classify risk. Return ONLY valid JSON, no other text.

Return format:
{"is_destructive": true|false, "risk_level": "low"|"medium"|"high"|"critical", "risk_reasons": ["reason1"]}

Risk levels:
  critical — irreversible data loss, production outage, security breach
  high     — service disruption, destructive config change, many hosts affected
  medium   — single-host change, recoverable, non-critical service
  low      — read-only diagnostic, no state change
"""

_PLANNER_SYSTEM = """You are an IT network operations planner.
Given the query and its classified intent, produce a concrete action plan.
Return ONLY valid JSON, no other text, no markdown fences.

Return format:
{"action_type": "string", "target": "string", "parameters": {}, "estimated_impact": "string", "reversible": true|false, "plan_steps": ["step1", "step2"]}

Keep plan_steps to 3-6 steps. Be specific. action_type should be a snake_case verb.
"""


def patch_hitl_graph(engine: LLMEngine, tool_registry: dict | None = None) -> None:
    """
    Monkey-patch hitl/graph.py nodes to use the real LLM engine.
    Patches: intent_classifier_node, risk_assessor_node, planner_node.
    Optionally injects tool_registry into executor_node.

    LEGACY-ONLY: This function targets the LangGraph-based legacy HITL
    backend. The core HITL backend (HITL_BACKEND=core) does not need it —
    main.py only calls patch_hitl_graph from the legacy branch.

    If the legacy package is not installed (the default in this build),
    raise a clear error instead of an ImportError on `hitl.graph`. Callers
    in core mode should not invoke this; if they do (e.g. defensive
    re-patching after a tool registry refresh), the call site already
    wraps it in try/except.
    """
    try:
        import hitl.graph as _graph    # noqa: F401  (legacy package, may be absent)
    except ImportError as e:
        raise NotImplementedError(
            "patch_hitl_graph requires the legacy `hitl.graph` package, "
            "which is not part of this build. Use HITL_BACKEND=core, which "
            "does not need this patch. (underlying ImportError: %s)" % e
        ) from e
    import re as _re
    import json as _json
    from hitl_core.schema import RiskLevel

    # ── intent classifier ─────────────────────────────────────────────
    async def _intent(state: dict) -> dict:
        query  = state.get("query", "")
        result = await engine.classify_intent(query)
        logger.info("intent_classifier(LLM): %s conf=%.2f", result.intent_type, result.confidence)
        return {
            "intent_type":       result.intent_type,
            "intent_confidence": result.confidence,
            "intent_candidates": result.candidates,
            "intent_summary":    result.intent_summary,
        }

    # ── risk assessor ─────────────────────────────────────────────────
    async def _risk(state: dict) -> dict:
        query       = state.get("query", "")
        intent_type = state.get("intent_type", "general_query")
        intent_sum  = state.get("intent_summary", "")
        prompt = (f"Query: {query}\nIntent: {intent_type} — {intent_sum}\n"
                  "Assess the risk of executing this network operation.")
        try:
            if hasattr(engine, "_chat"):
                raw = await engine._chat([
                    {"role": "system", "content": _RISK_SYSTEM},
                    {"role": "user",   "content": prompt},
                ])
            else:
                raw = await engine.call(prompt, "", None)
            text = engine._strip_think(raw) if hasattr(engine, "_strip_think") else raw
            text = _re.sub(r"^```json?\s*", "", text.strip())
            text = _re.sub(r"\s*```$", "", text)
            data = _json.loads(text)
            rl   = getattr(RiskLevel, data.get("risk_level", "medium").upper(), RiskLevel.MEDIUM)
            logger.info("risk_assessor(LLM): risk=%s is_destructive=%s", rl, data.get("is_destructive"))
            return {"is_destructive": bool(data.get("is_destructive", False)),
                    "risk_level": rl, "risk_reasons": data.get("risk_reasons", [])}
        except Exception as exc:
            logger.warning("risk_assessor(LLM) parse failed (%s) — heuristic fallback", exc)
            is_destr = intent_type == "destructive_op"
            return {"is_destructive": is_destr,
                    "risk_level": RiskLevel.HIGH if is_destr else RiskLevel.LOW,
                    "risk_reasons": ["LLM parse failed; heuristic used"]}

    # ── planner ───────────────────────────────────────────────────────
    async def _planner(state: dict) -> dict:
        query       = state.get("query", "")
        intent_type = state.get("intent_type", "general_query")
        intent_sum  = state.get("intent_summary", "")
        risk_level  = getattr(state.get("risk_level"), "value", "medium")
        risk_rsns   = "; ".join(state.get("risk_reasons", []))
        prompt = (f"Query: {query}\nIntent: {intent_type} — {intent_sum}\n"
                  f"Risk: {risk_level} ({risk_rsns})\n"
                  "Produce a concrete network operations action plan.")
        try:
            if hasattr(engine, "_chat"):
                raw = await engine._chat([
                    {"role": "system", "content": _PLANNER_SYSTEM},
                    {"role": "user",   "content": prompt},
                ])
            else:
                raw = await engine.call(prompt, "", None)
            text = engine._strip_think(raw) if hasattr(engine, "_strip_think") else raw
            text = _re.sub(r"^```json?\s*", "", text.strip())
            text = _re.sub(r"\s*```$", "", text)
            data = _json.loads(text)
            action = {
                "action_type":      data.get("action_type", "llm_answer"),
                "target":           data.get("target", "network"),
                "parameters":       data.get("parameters", {}),
                "estimated_impact": data.get("estimated_impact", ""),
                "reversible":       data.get("reversible", True),
            }
            plan_steps = data.get("plan_steps", ["Analyse", "Execute", "Verify"])
            logger.info("planner(LLM): action_type=%s steps=%d", action["action_type"], len(plan_steps))
            return {"proposed_action": action, "plan_steps": plan_steps}
        except Exception as exc:
            logger.warning("planner(LLM) parse failed (%s) — fallback plan", exc)
            return {"proposed_action": {"action_type": "llm_answer", "target": "network",
                                        "parameters": {}, "reversible": True},
                    "plan_steps": ["Analyse query", "Execute best action", "Verify result"]}

    # ── executor (with optional real tool dispatch) ────────────────────
    if tool_registry is not None:
        _orig_executor = _graph.executor_node
        async def _executor_with_tools(state: dict) -> dict:
            state_copy = dict(state)
            state_copy["_tool_registry"] = tool_registry
            return await _orig_executor(state_copy)
        _graph.executor_node = _executor_with_tools

    _graph.intent_classifier_node = _intent
    _graph.risk_assessor_node     = _risk
    _graph.planner_node           = _planner

    logger.info(
        "patch_hitl_graph: intent+risk+planner patched → %s(%s)",
        engine.__class__.__name__, getattr(engine, "model", "?"),
    )