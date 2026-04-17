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
    from integrations.llm_engine import LLMEngine
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

class LLMEngine:
    """
    Unified LLM interface for the Agent Runtime Loop and HITL graph.

    Subclass or configure via LLMEngine.from_config(cfg).
    """

    TOOL_CALL_SYSTEM = """You are an expert IT network operations assistant.

TOOL CALLING FORMAT — use EXACTLY this syntax on its own line:
[TOOL:tool_name] {{"arg1": "value1", "arg2": "value2"}}

Examples:
[TOOL:syslog_search] {{"host": "ap-01", "severity": "error", "lines": 50}}
[TOOL:list_devices] {{}}
[TOOL:list_devices] {{"type": "switch"}}
[TOOL:list_interfaces] {{"device_id": "sw-core-01"}}
[TOOL:get_device_config] {{"device_id": "sw-core-01"}}
[TOOL:get_device_config] {{"device_id": "ap-01", "section": "radius"}}
[TOOL:edit_device_config] {{"device_id": "sw-core-01", "section": "ntp", "changes": {{"servers": ["10.0.1.1", "10.0.1.2"]}}, "reason": "add NTP redundancy"}}
[TOOL:validate_device_config] {{"device_id": "sw-core-01"}}
[TOOL:diff_device_config] {{"device_id": "sw-core-01"}}
[TOOL:device_info] {{"device_id": "sw-core-01"}}
[TOOL:dns_lookup] {{"hostname": "payments.internal"}}

STRICT RULES — follow exactly:
1. Call AT MOST ONE tool per response — never list multiple [TOOL:] lines
2. NEVER repeat a tool call you have already made this session
3. When tool results appear in the context below, DO NOT call that tool again
4. When you have enough information to answer, write your analysis WITHOUT any [TOOL:...] line
5. Keep responses concise — this is a production operations environment
6. Large results are shown as [STORED:tool:ref_id] — use [TOOL:read_stored_result] {{"ref_id": "..."}} to read pages

TOOLS vs SKILLS — critical distinction:
- TOOLS (use [TOOL:name]): executable functions that query/modify real systems
- SKILLS (shown as "Available skills"): procedural guides telling you WHICH tools to call in sequence
- NEVER call [TOOL:skill_name] — skills are NOT callable tools
- When a skill is relevant, READ its steps (via [SKILL_LOAD:skill_id]) then call the TOOLS it describes

INVENTORY QUERIES — when asked what devices exist:
- Use [TOOL:list_devices] {{}} to get ALL devices in one call
- Use [TOOL:list_devices] {{"type": "switch"}} for wired switches only
- Use [TOOL:list_devices] {{"type": "wireless_ap"}} for wireless APs only

CONFIGURATION QUERIES — when asked about device config:
- Use [TOOL:get_device_config] {{"device_id": "<id>"}} for full config
- Use [TOOL:get_device_config] {{"device_id": "<id>", "section": "radius"}} for one section
- Use [TOOL:validate_device_config] to check for errors
- Use [TOOL:edit_device_config] to apply fixes

STOP CONDITION: If the context section shows tool results — either a "Tool outputs:" header
or any "[TOOL: tool_name]" block — provide your final analysis NOW. Do NOT call any more tools.
The results are already there; summarise and answer the user directly.

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
                 max_tokens: int = 2048) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

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
            )
        if backend == "openai":
            return OpenAIEngine(
                model=model, temperature=temp, max_tokens=max_tok,
                api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                base_url=cfg.get("base_url"),
            )
        if backend == "anthropic":
            return AnthropicEngine(
                model=model, temperature=temp, max_tokens=max_tok,
                api_key_env=cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
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

    def _build_system_prompt(
        self, context: str, skill_catalog: Any = None,
        confirmed_facts: list[str] | None = None,
        tool_registry: dict | None = None,
    ) -> str:
        # ── Uploaded / extra tools section ───────────────────────────
        # Base tools are listed in TOOL_CALL_SYSTEM examples.
        # Any tools registered AFTER startup (via upload) are injected here
        # so the LLM knows they exist and can call them.
        _BASE_TOOLS = {
            # Built-in mock tools (always present)
            "syslog_search", "prometheus_query", "netflow_dump", "dns_lookup",
            "device_info", "alert_summary", "service_health", "restart_service",
            "read_stored_result", "process_stored_chunks", "list_devices", "list_interfaces",
            # Config tools (examples in system prompt — don't re-list in UPLOADED section)
            "get_device_config", "edit_device_config",
            "validate_device_config", "diff_device_config",
        }
        extra_tools_section = ""
        if tool_registry:
            extra = {n: fn for n, fn in tool_registry.items() if n not in _BASE_TOOLS}
            if extra:
                lines = ["UPLOADED TOOLS — also available, use the same [TOOL:name] format:"]
                for name in sorted(extra.keys()):
                    lines.append("  [TOOL:" + name + '] {"<arg>": "<value>"}')
                extra_tools_section = "\n".join(lines)

        # ── Skill summary ─────────────────────────────────────────────
        skill_summary = ""
        if skill_catalog:
            try:
                skill_summary = "Available skills:\n" + skill_catalog.format_summary()
            except Exception:
                pass

        # ── Confirmed facts ───────────────────────────────────────────
        facts_section = ""
        if confirmed_facts:
            facts_section = "Confirmed facts from this session:\n" + \
                            "\n".join(f"  • {f}" for f in confirmed_facts[-10:])

        system = self.TOOL_CALL_SYSTEM.format(
            skill_summary=skill_summary,
            confirmed_facts_section=facts_section,
            extra_tools_section=extra_tools_section,
        )
        if context:
            system += f"\n\nContext:\n{context}"
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
                 think: bool = False) -> None:
        super().__init__(model, temperature, max_tokens)
        self._base_url = base_url.rstrip("/")
        self._think    = think   # passed as think= to Ollama API for thinking models

    @property
    def _is_thinking_model(self) -> bool:
        return any(k in self.model.lower() for k in self.THINKING_MODELS)

    def _strip_think(self, text: str) -> str:
        """Strip <think>...</think> reasoning blocks from model output."""
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        turns           = getattr(state, "turns", 1) if state else 1

        # On Turn 2+: if tool results are already in context, add an explicit
        # stop instruction so the model doesn't repeat the same tool call
        stop_note = ""
        if turns > 1 and context and (
            "Tool outputs:" in context or
            "[TOOL: " in context or          # _format_tool_outputs produces this
            "[STORED:" in context or
            "Result for tool" in context
        ):
            stop_note = (
                "\n\nCRITICAL: Tool results are already shown in the context above. "
                "DO NOT emit any [TOOL:...] line. Provide your final analysis now."
            )

        # Pass the live tool_registry so uploaded tools appear in the system prompt
        _tool_reg = getattr(state, "_tool_registry", None) if state else None
        system = self._build_system_prompt(context, skill_catalog, confirmed_facts, _tool_reg)
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
        _detail = _os.getenv("LLM_LOG_DETAIL", "full").lower()
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

        raw = await self._chat(messages)
        result = self._strip_think(raw)

        if _detail in ("compact", "full"):
            _resp_log = result if _detail == "full" else (result[:400] + (" …" if len(result) > 400 else ""))
            _has_tool = "[TOOL:" in result
            logger.info(
                "LLM◀ TURN %d  chars=%d  tool_call=%s\n%s\n[RESPONSE]\n%s\n%s",
                turns, len(result), _has_tool, _sep, _resp_log, _sep,
            )
        else:
            logger.info(
                "LLM◀ turn=%d response_chars=%d has_tool_call=%s",
                turns, len(result), "[TOOL:" in result,
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
                "has_tool_call":  "[TOOL:" in result,
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

    async def _chat(self, messages: list[dict]) -> str:
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
        # Ollama ≥ 0.6 supports think= parameter for thinking models
        if self._is_thinking_model:
            payload["think"] = self._think  # False = suppress think blocks in API response

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
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
                # Even with think=False in the API, strip any residual <think> tags
                return self._strip_think(data["message"]["content"])
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
        system = self._build_system_prompt(context, skill_catalog, confirmed_facts)
        payload = {
            "model":    self.model,
            "messages": [{"role":"system","content":system},{"role":"user","content":query}],
            "stream":   True,
            "options":  {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
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
                 base_url: Optional[str] = None) -> None:
        super().__init__(model, temperature, max_tokens)
        self._api_key  = os.getenv(api_key_env, "")
        self._base_url = base_url

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        system = self._build_system_prompt(context, skill_catalog, confirmed_facts)
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
                 api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        super().__init__(model, temperature, max_tokens)
        self._api_key = os.getenv(api_key_env, "")

    async def call(self, query: str, context: str,
                   state: Any = None, skill_catalog: Any = None) -> str:
        confirmed_facts = getattr(state, "confirmed_facts", None) if state else None
        system = self._build_system_prompt(context, skill_catalog, confirmed_facts)
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
        from integrations.llm_engine import LLMEngine, patch_runtime_loop
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


def patch_hitl_graph(engine: LLMEngine) -> None:
    """
    Monkey-patch hitl/graph.py's intent_classifier_node to use a real LLM engine.

    Call this BEFORE building the HITL graph:
        patch_hitl_graph(engine)
        graph = build_hitl_graph(hitl_config)
    """
    import hitl.graph as _graph

    async def real_intent_classifier(state: dict) -> dict:
        query  = state.get("query", "")
        result = await engine.classify_intent(query)
        return {
            "intent_type":       result.intent_type,
            "intent_confidence": result.confidence,
            "intent_candidates": result.candidates,
            "intent_summary":    result.intent_summary,
        }

    _graph.intent_classifier_node = real_intent_classifier
    logger.info(
        "patch_hitl_graph: intent_classifier_node patched → %s(%s)",
        engine.__class__.__name__, engine.model,
    )