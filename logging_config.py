"""
logging_config.py
-----------------
Logging configuration for the IT Ops Agent.

Import this at the top of main.py (or any entry point) to get structured,
levelled output.  Three preset modes:

    normal   — INFO for everything (what you see now)
    llm      — DEBUG for LLM/tool interactions; INFO for everything else
    llm_only — ONLY LLM requests/responses and tool calls; all other output suppressed
    verbose  — DEBUG for everything

Usage
-----
    # In main.py or at the CLI:
    import logging_config
    logging_config.configure(mode="llm")

    # Or via env var (no code change needed):
    LOG_MODE=llm      uvicorn main:app --port 8001   # LLM debug + normal app logs
    LOG_MODE=llm_only uvicorn main:app --port 8001   # ONLY LLM/tool lines, nothing else

    # Or enable just one logger at runtime:
    logging_config.set_llm_debug(True)

What each mode shows
--------------------
normal
    2026-04-15 10:30:10 INFO  integrations.llm_engine: LLM▶ turn=1 model=qwen3.5:27b system_chars=3463 user_chars=34
    2026-04-15 10:30:10 INFO  integrations.llm_engine: LLM tokens: prompt=361 completion=78 total=439
    2026-04-15 10:30:10 INFO  integrations.llm_engine: LLM◀ turn=1 response_chars=142 has_tool_call=True
    2026-04-15 10:30:10 INFO  runtime.loop:             TOOL▶ syslog_search args={'host': 'radius-*', 'lines': 300}
    2026-04-15 10:30:10 INFO  runtime.loop:             TOOL◀ syslog_search result_chars=6284 stored=True

llm  (LOG_MODE=llm)
    ... all of the above, plus:
    2026-04-15 10:30:10 DEBUG integrations.llm_engine: LLM REQUEST turn=1
    ────────────────────────────────────────────────────────────────────────
    [SYSTEM]
    You are an expert IT operations assistant...
    <full system prompt>
    ────────────────────────────────────────────────────────────────────────
    [USER]
    why is RADIUS authentication failing?
    ────────────────────────────────────────────────────────────────────────

    2026-04-15 10:30:10 DEBUG integrations.llm_engine: LLM RESPONSE turn=1
    ────────────────────────────────────────────────────────────────────────
    Analysing RADIUS auth failures on ap-01.
    [TOOL:syslog_search] {"host": "radius-*", "severity": "error", "lines": 300}
    ────────────────────────────────────────────────────────────────────────

    2026-04-15 10:30:10 DEBUG runtime.loop:             TOOL ARGS
    ────────────────────────────────────────────────────────────────────────
    {
      "host": "radius-*",
      "severity": "error",
      "lines": 300
    }
    ────────────────────────────────────────────────────────────────────────

    2026-04-15 10:30:10 DEBUG runtime.loop:             TOOL RESULT syslog_search
    ────────────────────────────────────────────────────────────────────────
    # syslog_search host=radius-* keyword=error results=300 matched=75 query_time=0.05s
    Apr 15 10:28:01 radius-01 radiusd[1234]: [ERROR] RADIUS Access-Reject for user alice@corp.com
    ...
    ────────────────────────────────────────────────────────────────────────

verbose  (LOG_MODE=verbose)
    DEBUG for all loggers including httpx, fastapi, langgraph internals.
"""

import logging
import os
import sys


# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------

_FMT_NORMAL  = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_FMT_VERBOSE = "%(asctime)s %(levelname)-8s %(name)-40s %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# Loggers that produce the detailed LLM/tool trace lines
_LLM_LOGGERS = [
    "integrations.llm_engine",
    "runtime.loop",
]

# "llm_only" mode: these loggers are shown; everything else is silenced
_LLM_ONLY_LOGGERS = [
    "integrations.llm_engine",   # LLM requests, responses, token counts
    "runtime.loop",              # TOOL▶ / TOOL◀ calls and results
]

# Loggers kept at WARNING even in verbose mode (too noisy)
_QUIET_LOGGERS = [
    "httpcore",
    "httpx",
    "asyncio",
    "urllib3",
    "multipart",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure(mode: str = "") -> None:
    """
    Configure logging for the whole application.

    mode: "normal" | "llm" | "llm_only" | "verbose" | "" (reads LOG_MODE env var)
          llm_only — silences all loggers except integrations.llm_engine and runtime.loop,
                     then sets those to DEBUG. Shows only LLM requests/responses and tool calls.
                     Usage: LOG_MODE=llm_only uvicorn main:app --port 8001
    """
    if not mode:
        mode = os.getenv("LOG_MODE", "normal").lower()

    fmt = _FMT_VERBOSE if mode == "verbose" else _FMT_NORMAL

    # Root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)

    if mode == "verbose":
        root.setLevel(logging.DEBUG)
    elif mode == "llm_only":
        # Show ONLY LLM and tool call lines — silence all other loggers.
        # Useful for debugging LLM behaviour without FastAPI / memory noise.
        root.setLevel(logging.WARNING)
        for name in _LLM_ONLY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    # Quiet noisy third-party loggers regardless of mode
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    if mode == "llm":
        # LLM/tool loggers get DEBUG; everything else stays at INFO
        for name in _LLM_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

    _log_active_mode(mode)


def set_llm_debug(enabled: bool = True) -> None:
    """
    Toggle DEBUG logging for LLM interactions at runtime without restarting.

        import logging_config
        logging_config.set_llm_debug(True)   # turn on
        logging_config.set_llm_debug(False)  # turn off

    Can also be toggled via the /webui/system/log-level endpoint.
    """
    level = logging.DEBUG if enabled else logging.INFO
    for name in _LLM_LOGGERS:
        logging.getLogger(name).setLevel(level)
    logging.getLogger(__name__).info(
        "LLM debug logging %s", "ENABLED" if enabled else "DISABLED"
    )


def _log_active_mode(mode: str) -> None:
    logging.getLogger(__name__).info(
        "Logging configured: mode=%r  (change with LOG_MODE env var or logging_config.set_llm_debug())",
        mode,
    )