"""Offline, review-gated workflow mining for privacy-minimized DSH trajectories."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_id(sequence: tuple[str, ...]) -> str:
    digest = hashlib.sha256(json.dumps(sequence).encode()).hexdigest()[:16]
    return f"workflow-{digest}"


def _read_sequences(source_database: str) -> dict[str, tuple[str, ...]]:
    source = Path(source_database).expanduser().resolve()
    if not source.is_file():
        return {}
    sessions: dict[str, list[str]] = defaultdict(list)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trajectory_events'"
        ).fetchone()
        if exists is None:
            return {}
        rows = connection.execute(
            "SELECT session_id, event_type, payload_json FROM trajectory_events ORDER BY id"
        )
        for session_id, event_type, payload_json in rows:
            if event_type != "tool:start":
                continue
            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError):
                continue
            tool_name = str(payload.get("tool_name") or "").strip()
            if tool_name and not tool_name.startswith("netopyu_trajectory_"):
                sessions[str(session_id)].append(tool_name)
    return {session: tuple(tools) for session, tools in sessions.items() if tools}


def mine_candidates(
    *, source_database: str, review_database: str, min_occurrences: int = 2,
    apply_changes: bool = False,
) -> dict[str, Any]:
    """Mine repeated exact tool sequences without reading prompts or argument values."""
    if not 2 <= min_occurrences <= 1000:
        raise ValueError("min_occurrences must be between 2 and 1000")
    sequences = _read_sequences(source_database)
    counts = Counter(sequence for sequence in sequences.values() if len(sequence) >= 2)
    candidates = [{
        "candidate_id": _candidate_id(sequence),
        "tool_sequence": list(sequence),
        "occurrences": occurrences,
        "session_count": occurrences,
        "privacy": "tool names only; no prompts, argument values, or results",
    } for sequence, occurrences in sorted(counts.items()) if occurrences >= min_occurrences]

    stored = 0
    destination = Path(review_database).expanduser().resolve()
    if apply_changes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS learning_candidates (
                  candidate_id TEXT PRIMARY KEY,
                  candidate_json TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL,
                  reviewed_at TEXT,
                  reviewer TEXT,
                  decision_reason TEXT,
                  proposal_path TEXT
                )
            """)
            for candidate in candidates:
                result = connection.execute("""
                    INSERT OR IGNORE INTO learning_candidates
                      (candidate_id, candidate_json, status, created_at)
                    VALUES (?, ?, 'pending', ?)
                """, (candidate["candidate_id"], json.dumps(candidate), _now()))
                stored += result.rowcount
    return {
        "ok": True,
        "mode": "apply" if apply_changes else "dry-run",
        "sessions_scanned": len(sequences),
        "candidates": candidates,
        "new_candidates_stored": stored,
        "automatic_skill_changes": False,
        "review_database": str(destination),
    }


def review_candidate(
    *, review_database: str, candidate_id: str, decision: str, reviewer: str,
    reason: str = "", proposal_directory: str | None = None,
) -> dict[str, Any]:
    """Atomically approve/reject a pending candidate; approval creates a proposal only."""
    decision = decision.strip().lower()
    reviewer = reviewer.strip()
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if not reviewer:
        raise ValueError("reviewer is required")
    database = Path(review_database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"learning database does not exist: {database}")
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT candidate_json, status FROM learning_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown learning candidate {candidate_id}")
        if row[1] != "pending":
            raise RuntimeError(f"candidate {candidate_id} is already {row[1]}")
        candidate = json.loads(row[0])
        proposal_path: str | None = None
        if decision == "approve":
            if not proposal_directory:
                raise ValueError("proposal_directory is required for approval")
            proposal_root = Path(proposal_directory).expanduser().resolve()
            proposal = proposal_root / re.sub(r"[^a-zA-Z0-9_-]", "-", candidate_id)
            proposal.mkdir(parents=True, exist_ok=False)
            tools = candidate["tool_sequence"]
            steps = "\n".join(f"{index}. Call `{tool}` and validate its result." for index, tool in enumerate(tools, 1))
            (proposal / "SKILL.md").write_text(
                "---\nname: " + candidate_id + "\ndescription: Reviewed DSH trajectory proposal; not auto-installed.\n---\n\n"
                "# Reviewed workflow proposal\n\n" + steps + "\n",
                encoding="utf-8",
            )
            (proposal / "review.json").write_text(json.dumps({
                "candidate": candidate, "reviewer": reviewer, "reason": reason,
                "reviewed_at": _now(), "auto_installed": False,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            proposal_path = str(proposal)
        status = "approved" if decision == "approve" else "rejected"
        changed = connection.execute("""
            UPDATE learning_candidates
               SET status = ?, reviewed_at = ?, reviewer = ?, decision_reason = ?, proposal_path = ?
             WHERE candidate_id = ? AND status = 'pending'
        """, (status, _now(), reviewer, reason, proposal_path, candidate_id)).rowcount
        if changed != 1:
            raise RuntimeError(f"candidate {candidate_id} was concurrently reviewed")
    return {
        "ok": True, "candidate_id": candidate_id, "status": status,
        "reviewer": reviewer, "proposal_path": proposal_path,
        "auto_installed": False,
    }
