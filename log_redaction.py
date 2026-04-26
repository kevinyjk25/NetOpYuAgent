"""
log_redaction.py
────────────────
Logging filter that redacts secrets from log records and tool outputs
before they reach disk, the LLM, or the chat history.

Pattern set
-----------
Network-device specific:
    password \\S+              → password ***REDACTED***
    secret \\S+                → secret ***REDACTED***
    community \\S+             → community ***REDACTED***
    key \\S+                   → key ***REDACTED***
    encrypted-password \\S+    → encrypted-password ***REDACTED***
    pre-shared-key \\S+        → pre-shared-key ***REDACTED***

Generic:
    Bearer <token>             → Bearer ***REDACTED***
    Authorization: <anything>  → Authorization: ***REDACTED***
    api[_-]?key=<val>          → api_key=***REDACTED***

The filter applies to:
    - logger output (LogRedactionFilter on root + module loggers)
    - tool result strings before they go into ToolResultStore or LLM context
      (redact_text() called by tool wrappers)
    - exception messages bubbled up to chat (redact_text())

Usage
-----
    from log_redaction import install_log_filter, redact_text

    install_log_filter()                        # call once at startup
    safe = redact_text(tool_output_or_exception)
"""
from __future__ import annotations

import logging
import re

# ── Patterns (case-insensitive, whole-word boundaries) ────────────────────────

_REDACTION_PATTERNS = [
    # IOS / NX-OS / EOS configuration secrets
    # Match optional cipher-type digit (0/5/7) before the secret value
    (re.compile(r"\b(password)\s+(?:[0-9]\s+)?(\S+)",            re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(secret)\s+(?:[0-9]\s+)?(\S+)",              re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(community)\s+(\S+)",                        re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(encrypted-password)\s+(\S+)",               re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(pre-shared-key)\s+(?:[0-9]\s+)?(\S+)",      re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(passphrase)\s+(\S+)",                       re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(authentication-key)\s+(\S+)",               re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(key-string)\s+(?:[0-9]\s+)?(\S+)",          re.IGNORECASE), r"\1 ***REDACTED***"),

    # HTTP authorization headers
    (re.compile(r"\b(Bearer)\s+([\w\-\._~+/=]+)",   re.IGNORECASE), r"\1 ***REDACTED***"),
    (re.compile(r"\b(Authorization):\s*\S+",        re.IGNORECASE), r"\1: ***REDACTED***"),

    # Common cloud/API key forms
    (re.compile(r"\b(api[_-]?key)\s*[:=]\s*\S+",    re.IGNORECASE), r"\1=***REDACTED***"),
    (re.compile(r"\b(token)\s*[:=]\s*[A-Za-z0-9\-\._~+/=]{16,}", re.IGNORECASE), r"\1=***REDACTED***"),

    # SNMP community strings (alternate form)
    (re.compile(r"\bsnmp-server\s+community\s+\S+", re.IGNORECASE), r"snmp-server community ***REDACTED***"),
]


def redact_text(text: str) -> str:
    """Apply all redaction patterns to a string. Safe to call on any text;
    returns the input unchanged if no patterns match.

    Use on:
      - tool output strings before storing or passing to LLM
      - exception messages caught from device drivers (Netmiko/NAPALM)
      - any user-facing error message in the chat
    """
    if not isinstance(text, str) or not text:
        return text
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class LogRedactionFilter(logging.Filter):
    """Logging filter that scrubs known secret patterns from formatted
    log records before the handler emits them. Attached to the root logger
    so every handler (console, file, syslog) sees redacted output."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the formatted message (post-arg-substitution)
        try:
            msg = record.getMessage()
            redacted = redact_text(msg)
            if redacted != msg:
                record.msg  = redacted
                record.args = ()    # already substituted into msg
        except Exception:
            pass    # never block logging due to filter error
        return True


def install_log_filter() -> None:
    """Attach the redaction filter to every existing logger and to the root.
    Call once at startup, before any tool output is logged."""
    redactor = LogRedactionFilter()
    logging.getLogger().addFilter(redactor)

    # Also attach to existing module loggers — some of them may already exist
    for name in list(logging.Logger.manager.loggerDict.keys()):
        try:
            logging.getLogger(name).addFilter(redactor)
        except Exception:
            pass
