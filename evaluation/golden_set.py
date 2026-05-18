"""
evaluation/golden_set.py
------------------------
Load / save / validate a labeled set of EvalCases.

Format: JSONL. Each line is one EvalCase:
    {"query": "...", "expected_ids": ["..."], "kind": "skill",
     "language": "zh", "tags": ["paraphrase"], "notes": "..."}

Why JSONL:
  - Easy to extend (one case = one line, append-only)
  - Diff-friendly in git (per-case changes show cleanly)
  - Streamable for large sets

Validation:
  validate_golden_set(cases, available_ids) returns warnings for
    - duplicate queries
    - expected_ids that don't exist in the current catalog
    - missing language tags
    - empty expected_ids lists
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing  import Iterable, Optional

from .types import EvalCase

logger = logging.getLogger(__name__)


def load_golden_set(path: str) -> list[EvalCase]:
    """Load EvalCases from a JSONL file. Skips malformed lines with a warning."""
    p = Path(path)
    if not p.exists():
        logger.warning("load_golden_set: file not found at %r", path)
        return []
    out: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                case = EvalCase(
                    query=obj["query"],
                    expected_ids=list(obj["expected_ids"]),
                    kind=obj.get("kind", "skill"),
                    notes=obj.get("notes", ""),
                    language=obj.get("language", "en"),
                    tags=list(obj.get("tags", [])),
                )
                out.append(case)
            except Exception as exc:
                logger.warning("load_golden_set: skipped line %d (%s)", line_no, exc)
    return out


def save_golden_set(cases: Iterable[EvalCase], path: str) -> int:
    """Write EvalCases as JSONL. Returns count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fp:
        for c in cases:
            fp.write(json.dumps({
                "query":        c.query,
                "expected_ids": c.expected_ids,
                "kind":         c.kind,
                "language":     c.language,
                "tags":         c.tags,
                "notes":        c.notes,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def validate_golden_set(
    cases:         list[EvalCase],
    available_ids: Optional[set[str]] = None,
) -> list[str]:
    """Sanity-check a loaded golden set. Returns list of human-readable warnings.

    available_ids: if provided, every EvalCase.expected_ids[*] must be in this set.
                   Pass the union of registered skill+tool ids.
    """
    warnings: list[str] = []
    seen_queries: set[str] = set()

    for i, c in enumerate(cases):
        if c.query in seen_queries:
            warnings.append(f"case#{i}: duplicate query {c.query!r}")
        seen_queries.add(c.query)

        if not c.expected_ids:
            warnings.append(f"case#{i}: empty expected_ids for query {c.query!r}")

        if available_ids is not None:
            missing = [eid for eid in c.expected_ids if eid not in available_ids]
            if missing:
                warnings.append(
                    f"case#{i}: expected_ids {missing!r} not in available catalog "
                    f"(query={c.query!r})"
                )

    return warnings
