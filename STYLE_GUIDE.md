# NetOpYuAgent — Code Style & Naming Guide

Minimal, opinionated rules. Most important first.

## 1. Module/package naming — singular for ports, plural for collections

The codebase mixes singular and plural at the package level. The rule going
forward:

| Pattern | Use | Examples |
|---------|------|---------|
| Singular | When the module is **the** thing (interface, runtime, single concept) | `runtime/`, `memory/`, `task/`, `auth/`, `hitl_core/` (the HITL pipeline) |
| Plural | When the module is **a collection** of similar items | `tools/`, `skills/`, `integrations/`, `mock_file/`, `routes_*/` |
| Either is fine | When both readings work | `retrieval/` (singular by convention), `task/intra/`, `task/inter/` |

### Current state of the codebase

```
✓ singular: runtime/  memory/  task/  hitl_core/  registry/  retrieval/
✓ plural:   tools/    skills/  integrations/    mock_file/    agent_memory/stores/
            agent_memory/retrieval/
~ outliers: hitl_core/schema.py  (singular file in singular package — OK because
            it's a single schema definitions file, not a collection)
            schemas.py would also be acceptable; don't churn it
```

### Sub-module naming

Sub-files inside a package follow the **plural rule for collections of items**,
**singular for single concept files**:

| File | Why singular/plural |
|------|---------------------|
| `hitl_core/router.py`   | one router class |
| `hitl_core/schema.py`   | one schema definitions file |
| `hitl_core/triggers.py` | several trigger functions |
| `hitl_core/batch.py`    | one batch coordinator + helpers |
| `runtime/directive_parser.py` | one parser |
| `agent_memory/stores/`  | collection of store implementations |

### Don't churn

Renaming an established public module is **never worth it for style alone**.
The guide is for new code. Existing names are grandfathered.

## 2. Function naming

| Prefix | Use |
|--------|-----|
| `_foo`  | Module-private. Outside callers must not import. |
| `foo`   | Public to the module's package. |
| `_foo_` (e.g. `_call_key`) | Local helper inside a function/method body. |
| `Foo` (PascalCase) | Class. |
| `FOO`  | Module-level constant. |

### Async naming

Async functions are NOT prefixed with `a` — match Python convention:

```python
# good
async def load_pending(self) -> list[Entry]: ...

# bad — needless 'a' prefix
async def aload_pending(self) -> list[Entry]: ...
```

Exception: when a sync **and** async pair coexist on the same class with
the same name, suffix the async one (this matches the wider ecosystem):

```python
def get(self, key) -> str | None: ...
async def get_async(self, key) -> str | None: ...
```

## 3. Public API boundaries

Every package's `__init__.py` MUST declare `__all__`. Things not in
`__all__` are **private**, even if Python lets you import them.

```python
# good — explicit surface
from .core import Engine, Result, Error
__all__ = ["Engine", "Result", "Error"]

# bad — implicit surface, callers can reach internals
from .core import *
```

Cross-package imports MUST come from the public `__all__`:

```python
# good
from memory import MemoryAdapter, RecallResult

# bad — reaches into the engine layer that memory/ encapsulates
from agent_memory.memory_manager import MemoryManager
```

(See `memory/ARCHITECTURE.md` for the port-vs-engine pattern this enforces.)

## 4. Directive parsing — must go through `runtime/directive_parser.py`

Any code that parses LLM output for `[TOOL:...]`, `[TOOL_BATCH:...]`,
`[SKILL_LOAD:...]`, or similar directives MUST use functions from
`runtime/directive_parser.py`. Inline regexes for these patterns are
rejected by `scripts/audit_directive_parsing.py`.

This rule exists because the parsing rules have **gradually accumulated
tolerance** (whitespace, code fences, `<think>` blocks, `[TOOL: name]`
with space-after-colon, `[TOOL:SKILL_LOAD:X]` mistake recovery, etc.).
Centralising means every caller benefits from new tolerance fixes without
having to remember which 5 regexes to update.

## 5. Logging

```python
logger = logging.getLogger(__name__)
```

Use lazy `%`-formatting in logger calls — defers string interpolation
until the level filter has been checked:

```python
# good
logger.info("Tool HITL raised: id=%s tool=%s target=%s", iid, name, target)

# bad — string built even when log level is WARNING
logger.info(f"Tool HITL raised: id={iid} tool={name} target={target}")
```

Exception: f-strings are fine when the log fires **once** at startup or
in an exception handler (where the cost is irrelevant). Don't religiously
convert them.

## 6. Type hints

Public function signatures **must** be typed. Private helpers (`_foo`)
should be typed where it aids reading; optional otherwise.

```python
# good
async def load_pending(
    self, *, limit: int = 100,
) -> list[CheckpointEntry]: ...

# acceptable for private
def _normalize(x): ...
```

Use `|` (PEP 604) over `Union`:

```python
# good
def parse(text: str) -> int | None: ...

# old style — don't add new instances
def parse(text: str) -> Optional[int]: ...
```

`Optional[X]` already in the codebase is fine; don't rewrite it for style.

## 7. Comments

- Comments explain **why**, not **what**. The what is in the code.
- A 3-line comment is often better than a 30-line comment.
- Code that needs a long comment to be understood often wants refactoring
  instead.
- The exception: **subtle invariants** and **historical context that
  prevents regression** (e.g. "Path B nudge was tried and broke X — see
  audit round 6"). These deserve full paragraphs.

## 8. Tests

- Engine internals → `<package>/tests/`
- Cross-module behaviour → `tests/` at repo root
- One-off scripts that prove a fix → `scripts/audit_*.py` (CI-style)
- New parser variants → add a regression case to the directive_parser
  doctest set, not a new test file

## 9. Imports

Order:
1. `from __future__ import ...`
2. Standard library
3. Third-party (fastapi, pydantic, etc.)
4. First-party — top-level packages, then sub-packages

```python
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from memory import MemoryAdapter
from runtime.directive_parser import find_tool_directives
from .stop_policy import LoopState
```

Group with blank lines. Within a group, alphabetical.

## 10. When in doubt

- Match the surrounding style.
- If the surrounding style is bad and would mislead someone reading your
  new code, fix the surrounding style in the same PR — but only the
  immediate surroundings, not the whole file.
- Big style sweeps deserve their own PR with no behaviour changes.
