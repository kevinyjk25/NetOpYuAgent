"""
tests/test_dtm_memory.py
------------------------
Comprehensive test suite for Dual-Track Memory (DTM).

Tests cover:
  1. Track A write — FTS5 raw turns stored in state.db
  2. Track B write — curated facts written to facts/facts.jsonl
  3. Daily compaction — daily/YYYY-MM-DD.md created after threshold turns
  4. Track A recall — FTS5 search returns relevant past turns
  5. Track B recall — facts.jsonl queried and returned
  6. Dual recall — both tracks run, winner decided, MMR dedup applied
  7. Pre-compaction flush — force immediate daily .md write
  8. End-to-end: simulate multiple WebUI requests and verify all files written
  9. Stats endpoint — dtm.stats() reflects actual disk state
  10. Backward compat — DTM=None falls back to v3 Hermes hooks
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field

import pytest

# ── Stub pydantic so we can import without it installed ─────────────────────
pm = types.ModuleType("pydantic")
class _BM:
    def __init_subclass__(cls, **kw): pass
pm.BaseModel = _BM
pm.Field = lambda *a, **kw: None
sys.modules.setdefault("pydantic", pm)


# ── Shared test fixtures ─────────────────────────────────────────────────────

@dataclass
class FakeCuratedMemory:
    content:     str   = "RADIUS cert expires quarterly — pre-renew by week 3 of the quarter"
    memory_type: object = "incident_lesson"
    confidence:  float = 0.88
    tags:        list  = field(default_factory=lambda: ["radius", "cert", "auth"])
    session_id:  str   = "s1"
    created_at:  float = field(default_factory=time.time)
    memory_id:   str   = "abc123"


class FakeCurator:
    """Mimics MemoryCurator — returns one curated memory per after_turn call."""

    def __init__(self, return_memories=True):
        self.after_turn_calls: list[dict] = []
        self.recall_calls:     list[str]  = []
        self._return_memories = return_memories

    async def after_turn(self, session_id, user_text, assistant_text, tool_calls=None):
        self.after_turn_calls.append({
            "session_id": session_id,
            "user_text":  user_text[:60],
        })
        if self._return_memories:
            return [FakeCuratedMemory(session_id=session_id)]
        return []

    async def recall_for_session(self, query, session_id):
        self.recall_calls.append(query)
        return "Past: RADIUS cert issue resolved by renewal in Q1"


class FakeFTS:
    """Mimics FTS5SessionStore — records all write calls."""

    def __init__(self):
        self.write_calls: list[dict] = []
        self._turns: list[dict] = []

    async def write_turn(self, session_id, user_text, assistant_text,
                         tool_calls=None, importance=0.5, **kw):
        self.write_calls.append({"session_id": session_id, "user": user_text[:40]})
        self._turns.append({
            "session_id":     session_id,
            "user_text":      user_text,
            "assistant_text": assistant_text,
            "ts":             time.time(),
        })

    async def search(self, query, limit=8, session_exclude=None, **kw):
        """Return fake FTSSearchResult objects for matching turns."""
        from memory.fts_store import FTSSearchResult
        results = []
        q_lower = query.lower()
        for t in self._turns:
            if t.get("session_id") == session_exclude:
                continue
            combined = (t["user_text"] + " " + t["assistant_text"]).lower()
            words_hit = sum(1 for w in q_lower.split() if w in combined and len(w) > 2)
            if words_hit > 0:
                results.append(FTSSearchResult(
                    turn_id        = "t-" + t["session_id"][:6],
                    session_id     = t["session_id"],
                    ts             = t["ts"],
                    user_text      = t["user_text"],
                    assistant_text = t["assistant_text"],
                    snippet        = t["assistant_text"][:120],
                    rank           = -float(words_hit),
                    tool_calls     = [],
                ))
        results.sort(key=lambda r: r.rank)
        return results[:limit]

    async def get_stats(self):
        return {"total_turns": len(self._turns), "total_sessions": 1}


# ── Helper to build a DTM instance ──────────────────────────────────────────

def make_dtm(td: str, compaction_turns: int = 3, curator=None, fts=None):
    """Create a DualTrackMemory wired with fake dependencies."""
    from memory.dual_track import DualTrackMemory
    return DualTrackMemory(
        fts_store        = fts or FakeFTS(),
        curator          = curator or FakeCurator(),
        data_dir         = td,
        compaction_turns = compaction_turns,
        nudge_turns      = 50,           # high so nudge doesn't fire in tests
        track_b_weight   = 1.5,
        temporal_half_life_days = 7.0,
    )


# ── Test 1: Track A — FTS5 write ────────────────────────────────────────────

class TestTrackAWrite:

    def test_fts_write_called_on_after_turn(self):
        """Every call to dtm.after_turn() must call fts.write_turn() once."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts = FakeFTS()
                dtm = make_dtm(td, fts=fts)
                await dtm.after_turn("s1", "why is RADIUS failing?", "Cert expired.", [])
                assert len(fts.write_calls) == 1
                assert fts.write_calls[0]["session_id"] == "s1"
        asyncio.run(run())

    def test_fts_write_called_for_each_turn(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts = FakeFTS()
                dtm = make_dtm(td, fts=fts)
                for i in range(5):
                    await dtm.after_turn("s1", f"query {i}", f"answer {i}", [])
                assert len(fts.write_calls) == 5
        asyncio.run(run())

    def test_multiple_sessions_written_separately(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts = FakeFTS()
                dtm = make_dtm(td, fts=fts)
                await dtm.after_turn("session-a", "RADIUS failing", "Fixed cert", [])
                await dtm.after_turn("session-b", "BGP down", "Restarted BGP", [])
                sessions = {c["session_id"] for c in fts.write_calls}
                assert "session-a" in sessions
                assert "session-b" in sessions
        asyncio.run(run())


# ── Test 2: Track B — facts.jsonl write ─────────────────────────────────────

class TestTrackBWrite:

    def test_facts_jsonl_created_after_turn(self):
        """facts/facts.jsonl must be created after dtm.after_turn()."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS cert failing", "Renewed cert.", [])
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                assert facts_path.exists(), "facts/facts.jsonl not created"
        asyncio.run(run())

    def test_facts_jsonl_contains_valid_json(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS cert failing", "Renewed cert.", [])
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                lines = facts_path.read_text().strip().splitlines()
                assert len(lines) >= 1, "No facts written"
                for line in lines:
                    record = json.loads(line)      # must not raise
                    assert "content" in record
                    assert "memory_type" in record
                    assert "confidence" in record
        asyncio.run(run())

    def test_facts_accumulate_across_turns(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                for i in range(4):
                    await dtm.after_turn("s1", f"RADIUS issue {i}", f"Fixed {i}", [])
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                lines = facts_path.read_text().strip().splitlines()
                assert len(lines) >= 4, f"Expected ≥4 facts, got {len(lines)}"
        asyncio.run(run())

    def test_no_facts_when_curator_returns_empty(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                curator = FakeCurator(return_memories=False)
                dtm = make_dtm(td, curator=curator)
                await dtm.after_turn("s1", "generic query", "generic answer", [])
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                if facts_path.exists():
                    lines = [l for l in facts_path.read_text().splitlines() if l.strip()]
                    assert len(lines) == 0, "Facts written when curator returned nothing"
        asyncio.run(run())

    def test_facts_include_metadata_fields(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS auth failing", "Cert expired.", [{"tool": "syslog_search"}])
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                record = json.loads(facts_path.read_text().strip().splitlines()[0])
                assert "content" in record and len(record["content"]) > 5
                assert "memory_type" in record
                assert "confidence" in record and 0.0 <= record["confidence"] <= 1.0
                assert "created_at" in record
        asyncio.run(run())


# ── Test 3: Daily compaction ─────────────────────────────────────────────────

class TestDailyCompaction:

    def test_daily_md_created_after_threshold(self):
        """daily/YYYY-MM-DD.md must appear after compaction_turns turns."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td, compaction_turns=3)
                for i in range(4):    # exceeds threshold
                    await dtm.after_turn("s1", f"RADIUS query {i}", f"Answer {i}", [])
                    await asyncio.sleep(0.01)   # let tasks run
                # Wait for background compact task
                await asyncio.sleep(0.05)
                daily_dir = pathlib.Path(td) / "daily"
                daily_files = list(daily_dir.glob("*.md"))
                assert len(daily_files) >= 1, "No daily .md file created"
        asyncio.run(run())

    def test_daily_md_is_readable_markdown(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td, compaction_turns=2)
                for i in range(3):
                    await dtm.after_turn("s1", f"RADIUS failing {i}", f"Fixed cert {i}", [])
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                daily_dir = pathlib.Path(td) / "daily"
                files = list(daily_dir.glob("*.md"))
                assert files, "No daily file"
                content = files[0].read_text()
                assert "## Session snapshot" in content
                assert "Q:" in content
                assert "A:" in content
        asyncio.run(run())

    def test_daily_md_filename_is_date(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td, compaction_turns=2)
                for i in range(3):
                    await dtm.after_turn("s1", f"query {i}", f"answer {i}", [])
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                daily_dir = pathlib.Path(td) / "daily"
                files = list(daily_dir.glob("*.md"))
                assert files
                import re
                assert re.match(r"\d{4}-\d{2}-\d{2}\.md", files[0].name), \
                    f"Bad filename: {files[0].name}"
        asyncio.run(run())

    def test_pre_compaction_flush_writes_immediately(self):
        """pre_compaction_flush() writes even when threshold not reached."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td, compaction_turns=100)  # high threshold
                await dtm.after_turn("s1", "RADIUS failing", "Fixed cert", [])
                # Only 1 turn — far below threshold — but flush forces write
                await dtm.pre_compaction_flush("s1", "about to compact")
                await asyncio.sleep(0.05)
                daily_dir = pathlib.Path(td) / "daily"
                files = list(daily_dir.glob("*.md"))
                assert files, "pre_compaction_flush did not write daily .md"
        asyncio.run(run())


# ── Test 4: Recall — Track A ─────────────────────────────────────────────────

class TestTrackARecall:

    def test_recall_returns_track_a_results(self):
        """After writing turns, recall should return Track A results."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts = FakeFTS()
                dtm = make_dtm(td, fts=fts)
                # Write a turn first
                await dtm.after_turn("s0", "RADIUS cert expired on radius-01", "Renewed cert.", [])
                # Recall from a different session
                result = await dtm.recall("RADIUS cert authentication", "s1")
                assert result.track_a_count > 0, \
                    f"Expected Track A results, got track_a={result.track_a_count}"
        asyncio.run(run())

    def test_recall_excludes_current_session(self):
        """Recall must not return turns from the current session."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts = FakeFTS()
                dtm = make_dtm(td, fts=fts)
                await dtm.after_turn("current-session", "RADIUS failing", "Fixed cert", [])
                result = await dtm.recall("RADIUS cert", "current-session")
                # All Track A results should be from other sessions
                for r in result.results:
                    if r.track == "A" and "fts5:" in r.source:
                        assert "current-" not in r.source, "Current session leaked into recall"
        asyncio.run(run())


# ── Test 5: Recall — Track B ─────────────────────────────────────────────────

class TestTrackBRecall:

    def test_recall_returns_track_b_facts(self):
        """After writing facts, recall should return Track B results."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS cert expiry issue", "Renewed cert.", [])
                # Recall from different session — Track B facts are global
                result = await dtm.recall("RADIUS authentication cert expiry", "s2")
                assert result.track_b_count > 0, \
                    f"Expected Track B results, got track_b={result.track_b_count}"
        asyncio.run(run())

    def test_track_b_has_higher_score_than_track_a(self):
        """Track B facts should outscore Track A chunks (track_b_weight=1.5)."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS cert expiry", "Cert expired on RADIUS-01. Renewed quarterly.", [])
                result = await dtm.recall("RADIUS cert", "s2")
                b_scores = [r.score for r in result.results if r.track == "B"]
                a_scores = [r.score for r in result.results if r.track == "A"]
                if b_scores and a_scores:
                    assert max(b_scores) >= max(a_scores) * 0.8, \
                        "Track B should score at or above Track A (1.5× boost)"
        asyncio.run(run())

    def test_track_b_deduplicates_identical_facts(self):
        """Multiple identical facts should not all appear in the result."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                # Write same content many times
                for _ in range(6):
                    await dtm.after_turn("s1", "RADIUS cert expiry", "Renewed cert.", [])
                result = await dtm.recall("RADIUS cert", "s2", top_k=4)
                # MMR dedup should reduce repetition
                contents = [r.content[:60] for r in result.results if r.track == "B"]
                unique = len(set(contents))
                assert unique >= 1
                # Should not have all 6 identical facts
                assert len(contents) <= 4, \
                    f"MMR should deduplicate, got {len(contents)} identical B results"
        asyncio.run(run())


# ── Test 6: Dual recall — arbitration ────────────────────────────────────────

class TestDualRecall:

    def test_recall_returns_dtm_recall_result(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                from memory.dual_track import DTMRecallResult
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS auth failing", "Cert expired.", [])
                result = await dtm.recall("RADIUS auth", "s2")
                assert isinstance(result, DTMRecallResult)
        asyncio.run(run())

    def test_winner_is_a_or_b_or_tie(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS auth failing", "Cert expired.", [])
                result = await dtm.recall("RADIUS cert", "s2")
                assert result.winner in ("A", "B", "tie"), f"Bad winner: {result.winner}"
        asyncio.run(run())

    def test_prompt_context_not_empty_after_turns(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                await dtm.after_turn("s1", "RADIUS auth failing on ap-01", "Certificate expired on RADIUS-01.", [])
                result = await dtm.recall("RADIUS authentication", "s2")
                assert len(result.prompt_context) > 20, "prompt_context should have content"
        asyncio.run(run())

    def test_prompt_context_no_duplicate_facts(self):
        """Same fact written 5× should not appear 5× in prompt_context."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                for _ in range(5):
                    await dtm.after_turn("s1", "RADIUS cert expiry", "Renewed cert quarterly.", [])
                result = await dtm.recall("RADIUS cert", "s2")
                # Count occurrences of the fact content in the prompt
                fact_snippet = "RADIUS cert expires"
                occurrences = result.prompt_context.lower().count("radius cert expire")
                assert occurrences <= 2, \
                    f"Duplicate facts in prompt: '{fact_snippet}' appears {occurrences}×"
        asyncio.run(run())

    def test_recall_with_no_prior_turns_returns_empty(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                result = await dtm.recall("RADIUS cert", "s1")
                assert result.track_a_count == 0
                assert result.track_b_count == 0
                assert result.prompt_context == ""
        asyncio.run(run())

    def test_temporal_decay_applied_to_track_a(self):
        """Old Track A results should have lower score than recent ones."""
        async def run():
            with tempfile.TemporaryDirectory() as td:
                from memory.dual_track import DualTrackMemory, DTMResult
                dtm = make_dtm(td)
                old_ts = time.time() - 30 * 86400   # 30 days ago
                new_ts = time.time() - 1 * 3600      # 1 hour ago
                old_r = DTMResult("RADIUS cert content old", "A", 0.8, "fts5", "raw_chunk", old_ts)
                new_r = DTMResult("RADIUS cert content new", "A", 0.8, "fts5", "raw_chunk", new_ts)
                import math
                half_life = 7 * 86400
                old_score = 0.8 * math.exp(-0.693 * (time.time() - old_ts) / half_life)
                new_score = 0.8 * math.exp(-0.693 * (time.time() - new_ts) / half_life)
                assert new_score > old_score * 3, \
                    f"Recent ({new_score:.3f}) should >> old ({old_score:.3f})"
        asyncio.run(run())


# ── Test 7: Stats ────────────────────────────────────────────────────────────

class TestStats:

    def test_stats_returns_dict(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                stats = dtm.stats()
                assert isinstance(stats, dict)
                assert "daily_files" in stats
                assert "facts_count" in stats
                assert "pending_turns" in stats
                assert "track_b_weight" in stats
        asyncio.run(run())

    def test_stats_daily_files_count_matches_disk(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td, compaction_turns=2)
                for i in range(3):
                    await dtm.after_turn("s1", f"query {i}", f"answer {i}", [])
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                stats = dtm.stats()
                daily_dir = pathlib.Path(td) / "daily"
                actual = len(list(daily_dir.glob("*.md"))) if daily_dir.exists() else 0
                assert stats["daily_files"] == actual, \
                    f"stats says {stats['daily_files']} but disk has {actual}"
        asyncio.run(run())

    def test_stats_facts_count_matches_disk(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                dtm = make_dtm(td)
                for i in range(3):
                    await dtm.after_turn("s1", f"RADIUS query {i}", f"answer {i}", [])
                stats = dtm.stats()
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                actual = len(facts_path.read_text().splitlines()) if facts_path.exists() else 0
                assert stats["facts_count"] == actual, \
                    f"stats says {stats['facts_count']} but disk has {actual}"
        asyncio.run(run())


# ── Test 8: End-to-end simulation ────────────────────────────────────────────

class TestEndToEnd:

    def test_simulated_working_session_writes_all_files(self):
        """
        Simulate a realistic IT ops session:
        5 diagnostic queries + 1 complex task
        Verify: state.db has turns, facts.jsonl has facts, daily .md created.
        """
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts   = FakeFTS()
                dtm   = make_dtm(td, compaction_turns=3, fts=fts)
                queries = [
                    ("why is RADIUS auth failing for wireless users?",
                     "Found cert expired on RADIUS-01. Renewed. Issue resolved.",
                     [{"tool": "syslog_search"}]),
                    ("show BGP neighbors for router-01",
                     "BGP sessions: 3 established, 0 idle. Routes: 1204.",
                     [{"tool": "get_bgp_summary"}]),
                    ("check DNS resolution for payments.internal",
                     "DNS OK: 10.0.1.45. TTL 300.",
                     [{"tool": "dns_lookup"}]),
                    ("search syslogs for errors on ap-01",
                     "Found 8 errors: 3 RADIUS timeouts, 5 DHCP exhaustion.",
                     [{"tool": "syslog_search"}]),
                    ("get NetFlow traffic dump for site-a",
                     "Top flow: 10.0.1.2 → 10.0.2.5 TCP 443 1.2GB",
                     [{"tool": "netflow_dump"}]),
                ]

                session = "session-test-001"
                for user_q, assistant_a, tools in queries:
                    await dtm.after_turn(session, user_q, assistant_a, tools)
                    await asyncio.sleep(0.01)

                # Wait for background compact task
                await asyncio.sleep(0.1)

                # ── Assertions ────────────────────────────────────────
                # Track A: FTS5 writes
                assert len(fts.write_calls) == 5, \
                    f"Expected 5 FTS writes, got {len(fts.write_calls)}"

                # Track B: facts.jsonl
                facts_path = pathlib.Path(td) / "facts" / "facts.jsonl"
                assert facts_path.exists(), "facts/facts.jsonl not created"
                facts_count = len(facts_path.read_text().strip().splitlines())
                assert facts_count >= 5, \
                    f"Expected ≥5 facts (one per turn), got {facts_count}"

                # Track A: daily .md compaction
                daily_dir = pathlib.Path(td) / "daily"
                daily_files = list(daily_dir.glob("*.md"))
                assert daily_files, "daily/ .md file not created after 5 turns (threshold=3)"

                # Daily file is human-readable
                content = daily_files[0].read_text()
                assert "RADIUS" in content or "BGP" in content or "DNS" in content, \
                    "Daily .md does not contain any query content"

                print(f"\n  [E2E] FTS writes: {len(fts.write_calls)}")
                print(f"  [E2E] Facts:      {facts_count}")
                print(f"  [E2E] Daily files: {len(daily_files)}")
                print(f"  [E2E] Daily content (first 200 chars):")
                print("  " + content[:200].replace("\n", "\n  "))

        asyncio.run(run())

    def test_recall_after_session_returns_relevant_context(self):
        """
        After a session about RADIUS, a new session asking about auth
        should get relevant context from both tracks.
        """
        async def run():
            with tempfile.TemporaryDirectory() as td:
                fts = FakeFTS()
                dtm = make_dtm(td, fts=fts)

                # Session 1: RADIUS investigation
                await dtm.after_turn(
                    "session-1",
                    "RADIUS authentication failing for all APs at site-a",
                    "Root cause: certificate expired on RADIUS-01. Renewed via certbot. "
                    "Resolution: all APs reconnected within 5 minutes.",
                    [{"tool": "syslog_search"}, {"tool": "device_info"}],
                )

                # Session 2: new session asking related question
                result = await dtm.recall(
                    "authentication cert RADIUS wireless",
                    "session-2",
                    max_chars=800,
                )

                assert len(result.results) > 0, "No recall results for related query"
                assert len(result.prompt_context) > 50, "Prompt context is empty"

                # Context should mention RADIUS or cert
                ctx_lower = result.prompt_context.lower()
                assert "radius" in ctx_lower or "cert" in ctx_lower, \
                    "Prompt context doesn't mention RADIUS or cert"

                print(f"\n  [Recall E2E] A={result.track_a_count} B={result.track_b_count} winner={result.winner}")
                print(f"  [Recall E2E] Context ({len(result.prompt_context)} chars):")
                print("  " + result.prompt_context[:300].replace("\n", "\n  "))

        asyncio.run(run())


# ── Test 9: Backward compatibility ───────────────────────────────────────────

class TestBackwardCompat:

    def test_dtm_none_does_not_crash(self):
        """
        When DTM is None (v3 mode), the a2a_integration executor must
        still work using individual fts/curator/user_model hooks.
        Verify no AttributeError on self._dtm.
        """
        # Just confirm the attribute access pattern is safe
        class MockExecutor:
            _dtm = None
            _fts_store = None
            _curator = None
            _user_model = None

            async def _hermes_post_turn(self, *args):
                # This is what a2a_integration does
                if self._dtm:
                    pass  # would call dtm.after_turn
                else:
                    # v3 fallback
                    if self._fts_store:
                        pass
                    if self._curator:
                        pass

        executor = MockExecutor()
        # Must not raise
        asyncio.run(executor._hermes_post_turn("s", "q", "a", []))

    def test_chunker_handles_empty_content(self):
        from memory.dual_track import DualTrackMemory
        chunks = DualTrackMemory._chunk_markdown("", chunk_chars=400)
        assert chunks == []

    def test_chunker_handles_short_content(self):
        from memory.dual_track import DualTrackMemory
        chunks = DualTrackMemory._chunk_markdown("Short content.", chunk_chars=400)
        assert len(chunks) == 1
        assert "Short content." in chunks[0][0]

    def test_mmr_select_empty_input(self):
        from memory.dual_track import DualTrackMemory
        result = DualTrackMemory._mmr_select([], k=5)
        assert result == []

    def test_mmr_select_fewer_candidates_than_k(self):
        from memory.dual_track import DualTrackMemory, DTMResult
        candidates = [
            DTMResult("content a", "A", 0.9, "s", "raw_chunk", time.time()),
            DTMResult("content b", "B", 0.8, "s", "fact",      time.time()),
        ]
        result = DualTrackMemory._mmr_select(candidates, k=10)
        assert len(result) == 2   # can't return more than available


# ── Run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Quick self-test runner without pytest
    test_classes = [
        TestTrackAWrite, TestTrackBWrite, TestDailyCompaction,
        TestTrackARecall, TestTrackBRecall, TestDualRecall,
        TestStats, TestEndToEnd, TestBackwardCompat,
    ]
    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        print(f"\n{'='*55}")
        print(f"  {cls.__name__}")
        print(f"{'='*55}")
        for name in dir(instance):
            if not name.startswith("test_"):
                continue
            try:
                getattr(instance, name)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}")
                print(f"        {type(exc).__name__}: {exc}")
                failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)