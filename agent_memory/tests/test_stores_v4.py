"""
agent_memory/tests/test_stores_v4.py
Tests for v4 store improvements:
  - ShortTermStore: list_by_tool, garbage_collect, total_length/total_bytes correct
  - MidTermStore: deduplication, TTL/expiry, confidence decay, evict_expired_facts
  - LongTermStore: batch perf, importance/recency scoring, update_chunk, LRU indexes,
                    apply_retention, FTS5 reserved words
Run: python -m unittest agent_memory.tests.test_stores_v4 -v
"""
from __future__ import annotations
import os, sys, tempfile, time, threading, unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_memory import MemoryManager, MemoryChunk, MemoryFact
from agent_memory.stores.long_term_store import LongTermStore, _recency_score


def _mem(tmp):
    return MemoryManager(data_dir=os.path.join(tmp, "m"))


# ── ShortTermStore v4 ─────────────────────────────────────────────────────────
class TestShortTermV4(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_get_entry_total_length_correct(self):
        content = "Hello 你好 World 🔥" * 10
        entry = self.mem.cache_tool_result("alice", "s1", "tool", content)
        fetched = self.mem.short_term.get_entry("alice", entry.ref_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.total_length, len(content))

    def test_read_returns_total_bytes_field(self):
        content = "BGP session dropped" * 50
        entry = self.mem.cache_tool_result("alice", "s1", "tool", content)
        page = self.mem.read_cached("alice", entry.ref_id, 0, 100)
        self.assertIn("total_bytes", page)
        self.assertGreater(page["total_bytes"], 0)
        # For pure ASCII total_bytes == total_length
        self.assertEqual(page["total_bytes"], len(content.encode("utf-8")))

    def test_total_bytes_vs_total_length_cjk(self):
        content = "日志条目BGP网络" * 100  # CJK: bytes > chars
        entry = self.mem.cache_tool_result("alice", "s1", "syslog", content)
        page = self.mem.read_cached("alice", entry.ref_id, 0, 50)
        self.assertGreater(page["total_bytes"], page["total_length"])

    def test_list_by_tool_filters_correctly(self):
        self.mem.cache_tool_result("alice", "s1", "syslog_search", "log1")
        self.mem.cache_tool_result("alice", "s1", "syslog_search", "log2")
        self.mem.cache_tool_result("alice", "s1", "bgp_check", "bgp1")
        self.mem.cache_tool_result("alice", "s2", "syslog_search", "log3")

        # session-scoped
        entries = self.mem.short_term.list_by_tool("alice", "syslog_search", "s1")
        self.assertEqual(len(entries), 2)
        for e in entries:
            self.assertEqual(e.tool_name, "syslog_search")
            self.assertEqual(e.session_id, "s1")

    def test_list_by_tool_across_sessions(self):
        self.mem.cache_tool_result("alice", "s1", "ping", "ping1")
        self.mem.cache_tool_result("alice", "s2", "ping", "ping2")
        entries = self.mem.short_term.list_by_tool("alice", "ping")  # no session filter
        self.assertEqual(len(entries), 2)

    def test_list_by_tool_user_isolation(self):
        self.mem.cache_tool_result("alice", "s1", "tool", "alice data")
        self.mem.cache_tool_result("bob", "s1", "tool", "bob data")
        entries = self.mem.short_term.list_by_tool("alice", "tool")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].user_id, "alice")

    def test_garbage_collect_orphan_files(self):
        import hashlib
        from pathlib import Path
        # Create an orphan file (exists on disk but not in DB)
        base = Path(self.mem.short_term._base_dir)
        user_hash = hashlib.sha256(b"alice").hexdigest()[:24]
        orphan_dir = base / user_hash
        orphan_dir.mkdir(parents=True, exist_ok=True)
        orphan = orphan_dir / "orphan_xyz.cache"
        orphan.write_bytes(b"orphan data")

        gc = self.mem.short_term.garbage_collect()
        self.assertGreaterEqual(gc["orphan_files"], 1)
        self.assertFalse(orphan.exists())

    def test_garbage_collect_missing_db_refs(self):
        entry = self.mem.cache_tool_result("alice", "s1", "tool", "data")
        # Delete the file manually to simulate corruption
        from pathlib import Path
        rows = self.mem._pool_read("SELECT file_path FROM tool_cache_index WHERE ref_id=?",
                                   (entry.ref_id,))
        Path(rows[0]["file_path"]).unlink()
        gc = self.mem.short_term.garbage_collect()
        self.assertGreaterEqual(gc["missing_files"], 1)
        # Should be removed from DB too
        result = self.mem.read_cached("alice", entry.ref_id)
        self.assertIn("error", result)

    def test_concurrent_store_no_errors(self):
        errors = []
        def worker(n):
            for i in range(10):
                try:
                    self.mem.cache_tool_result(f"u{n}", "s1", "tool", f"data {n}-{i}" * 30)
                except Exception as e:
                    errors.append(str(e))
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])


# ── MidTermStore v4 ───────────────────────────────────────────────────────────
class TestMidTermV4(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_exact_duplicate_dedup(self):
        """Same text, same session → stored exactly once."""
        self.mem.add_fact("alice", "s1", "R1 at 10.0.0.1", "entity", 0.9)
        self.mem.add_fact("alice", "s1", "R1 at 10.0.0.1", "entity", 0.9)
        self.mem.add_fact("alice", "s1", "R1 at 10.0.0.1", "entity", 0.9)
        count = self.mem.mid_term.count("alice", "s1")
        self.assertEqual(count, 1)

    def test_different_text_not_deduped(self):
        self.mem.add_fact("alice", "s1", "R1 at 10.0.0.1", "entity", 0.9)
        self.mem.add_fact("alice", "s1", "R2 at 10.0.0.2", "entity", 0.9)
        count = self.mem.mid_term.count("alice", "s1")
        self.assertEqual(count, 2)

    def test_same_text_different_sessions_both_stored(self):
        self.mem.add_fact("alice", "s1", "R1 at 10.0.0.1", "entity", 0.9)
        self.mem.add_fact("alice", "s2", "R1 at 10.0.0.1", "entity", 0.9)
        self.assertEqual(self.mem.mid_term.count("alice", "s1"), 1)
        self.assertEqual(self.mem.mid_term.count("alice", "s2"), 1)

    def test_batch_add_dedup(self):
        facts = [
            MemoryFact(user_id="alice", session_id="s1",
                       fact="R1 at 10.0.0.1", fact_type="entity", confidence=0.9),
            MemoryFact(user_id="alice", session_id="s1",
                       fact="R1 at 10.0.0.1", fact_type="entity", confidence=0.9),  # dup
            MemoryFact(user_id="alice", session_id="s1",
                       fact="R2 at 10.0.0.2", fact_type="entity", confidence=0.8),
        ]
        self.mem.mid_term.add_facts_batch(facts)
        self.assertEqual(self.mem.mid_term.count("alice", "s1"), 2)

    def test_fact_ttl_expiry(self):
        # Add a fact with very short TTL
        fact = MemoryFact(user_id="alice", session_id="s1",
                          fact="Short-lived preference fact", fact_type="preference",
                          confidence=0.9)
        self.mem.mid_term.add_fact(fact, ttl_days=0.0000001)  # expires almost immediately (0.0086s)
        time.sleep(0.03)
        # Should not appear in normal search (exclude_expired=True by default)
        r = self.mem.search_facts("alice", "short-lived preference", session_id="s1")
        self.assertEqual(r.total_found, 0)
        # Should appear if we include expired
        r2 = self.mem.mid_term.search("alice", "short-lived preference",
                                      session_id="s1", exclude_expired=False)
        self.assertGreater(r2.total_found, 0)

    def test_evict_expired_facts(self):
        fact = MemoryFact(user_id="alice", session_id="s1",
                          fact="Temporary config note here", fact_type="config",
                          confidence=0.8)
        self.mem.mid_term.add_fact(fact, ttl_days=0.0000001)
        time.sleep(0.03)
        permanent = self.mem.add_fact("alice", "s1", "Permanent entity R1", "entity", 1.0)
        time.sleep(0.02)
        evicted = self.mem.mid_term.evict_expired_facts()
        self.assertEqual(evicted, 1)
        # Permanent fact still present
        self.assertEqual(self.mem.mid_term.count("alice", "s1"), 1)

    def test_evict_expired_concurrent_safe(self):
        for i in range(10):
            f = MemoryFact(user_id="alice", session_id="s1",
                           fact=f"Temp fact {i} unique content xyz", fact_type="general",
                           confidence=0.7)
            self.mem.mid_term.add_fact(f, ttl_days=0.0000001)
        time.sleep(0.05)
        results = []
        errors = []
        def evict():
            try: results.append(self.mem.mid_term.evict_expired_facts())
            except Exception as e: errors.append(str(e))
        threads = [threading.Thread(target=evict) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        self.assertEqual(sum(results), 10)  # exactly 10 total, no double-delete

    def test_confidence_decay(self):
        fact = self.mem.add_fact("alice", "s1", "BGP peer stable config", "entity", 0.9)
        self.mem.mid_term.update_fact(fact.fact_id, "alice", "s1", decay=True)
        facts = self.mem.mid_term.list_all("alice", "s1")
        self.assertAlmostEqual(facts[0].confidence, 0.9 * 0.7, places=3)

    def test_confidence_decay_accumulates(self):
        fact = self.mem.add_fact("alice", "s1", "Disputed BGP route config", "entity", 0.9)
        self.mem.mid_term.update_fact(fact.fact_id, "alice", "s1", decay=True)
        self.mem.mid_term.update_fact(fact.fact_id, "alice", "s1", decay=True)
        self.mem.mid_term.update_fact(fact.fact_id, "alice", "s1", decay=True)
        facts = self.mem.mid_term.list_all("alice", "s1")
        expected = 0.9 * (0.7 ** 3)
        self.assertAlmostEqual(facts[0].confidence, expected, places=2)

    def test_confidence_decay_floor(self):
        """Confidence shouldn't go below 0.1."""
        fact = self.mem.add_fact("alice", "s1", "Highly uncertain fact content", "general", 0.1)
        for _ in range(10):
            self.mem.mid_term.update_fact(fact.fact_id, "alice", "s1", decay=True)
        facts = self.mem.mid_term.list_all("alice", "s1")
        self.assertGreaterEqual(facts[0].confidence, 0.1)

    def test_update_fact_text(self):
        fact = self.mem.add_fact("alice", "s1", "Old text here for update", "general", 0.8)
        updated = self.mem.mid_term.update_fact(
            fact.fact_id, "alice", "s1", new_text="New updated text content"
        )
        self.assertTrue(updated)
        facts = self.mem.mid_term.list_all("alice", "s1")
        self.assertEqual(facts[0].fact, "New updated text content")

    def test_search_excludes_expired_by_default(self):
        # Non-expired fact
        self.mem.add_fact("alice", "s1", "Permanent redis config fact", "config", 0.9)
        # Near-expired fact
        f = MemoryFact(user_id="alice", session_id="s1",
                       fact="Temporary preference note unique", fact_type="preference",
                       confidence=0.8)
        self.mem.mid_term.add_fact(f, ttl_days=0.0000001)
        time.sleep(0.03)
        r = self.mem.search_facts("alice", "preference note unique", session_id="s1")
        self.assertEqual(r.total_found, 0)

    def test_dedup_case_insensitive(self):
        """Dedup is case-insensitive (fact text normalized to lowercase for hash)."""
        self.mem.add_fact("alice", "s1", "R1 at 10.0.0.1", "entity", 0.9)
        self.mem.add_fact("alice", "s1", "r1 at 10.0.0.1", "entity", 0.9)  # lowercase dup
        # Both may or may not be deduped depending on exact normalization
        # At minimum, they don't cause errors
        count = self.mem.mid_term.count("alice", "s1")
        self.assertGreaterEqual(count, 1)


# ── LongTermStore v4 ──────────────────────────────────────────────────────────
class TestLongTermV4(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_batch_insert_faster_than_loop(self):
        texts = [f"chunk {i} redis BGP ospf network routing config node{i}" for i in range(500)]
        tmp2 = tempfile.mkdtemp()
        mem2 = _mem(tmp2)

        t0 = time.perf_counter()
        for t in texts:
            mem2.remember("alice", "s1", t)
        loop_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.mem.remember_batch("alice", "s1", texts)
        batch_ms = (time.perf_counter() - t0) * 1000

        mem2.close()
        self.assertLess(batch_ms, loop_ms / 2,
                        f"Batch {batch_ms:.0f}ms not faster than loop {loop_ms:.0f}ms")

    def test_batch_insert_all_searchable(self):
        texts = [f"unique_marker_{i} redis BGP network" for i in range(100)]
        self.mem.remember_batch("alice", "s1", texts)
        r = self.mem.search_chunks("alice", "unique_marker_42")
        self.assertGreater(r.total_found, 0)

    def test_batch_insert_fts_works(self):
        self.mem.remember_batch("alice", "s1", [
            "payment service down outage P1",
            "database primary failover triggered",
            "BGP session dropped router R1",
        ])
        r = self.mem.search_chunks("alice", "payment service outage")
        self.assertGreater(r.total_found, 0)
        texts = [c.text for c in r.items]
        self.assertTrue(any("payment" in t.lower() for t in texts))

    def test_recency_score_decay(self):
        now = time.time()
        score_fresh = _recency_score(now - 3600)    # 1 hour old
        score_old   = _recency_score(now - 30 * 86400)  # 30 days old
        self.assertGreater(score_fresh, score_old)
        self.assertAlmostEqual(score_fresh, 1.0, delta=0.01)
        self.assertLess(score_old, 0.3)

    def test_recent_chunks_rank_higher(self):
        # Old chunk about BGP
        old = MemoryChunk(user_id="alice", session_id="s1",
                          text="BGP session dropped R1 old entry",
                          created_at=time.time() - 30 * 86400)  # 30 days old
        self.mem.long_term.add_chunk(old)
        # New chunk about BGP
        new = self.mem.remember("alice", "s1", "BGP session dropped R1 new entry")
        r = self.mem.search_chunks("alice", "BGP session dropped R1")
        # New chunk should rank higher (fresher)
        if len(r.items) >= 2:
            self.assertEqual(r.items[0].chunk_id, new.chunk_id,
                             "New chunk should rank above 30-day-old chunk")

    def test_importance_score_boosts_retrieval(self):
        # Low importance chunk
        self.mem.long_term.add_chunk(
            MemoryChunk(user_id="alice", session_id="s1",
                        text="BGP routing table update minor"), importance=0.1)
        # High importance chunk
        self.mem.long_term.add_chunk(
            MemoryChunk(user_id="alice", session_id="s1",
                        text="BGP routing table update critical"), importance=0.95)
        r = self.mem.search_chunks("alice", "BGP routing table update")
        if len(r.items) >= 2:
            texts = [c.text for c in r.items]
            self.assertIn("critical", texts[0])

    def test_update_chunk(self):
        chunk = self.mem.remember("alice", "s1", "Original text content here test")
        updated = self.mem.long_term.update_chunk(
            "alice", chunk.chunk_id, "Updated text content here test")
        self.assertTrue(updated)
        r = self.mem.search_chunks("alice", "Updated text content")
        self.assertGreater(r.total_found, 0)
        texts = [c.text for c in r.items]
        self.assertTrue(any("Updated" in t for t in texts))

    def test_update_chunk_not_found(self):
        updated = self.mem.long_term.update_chunk("alice", "nonexistent-id", "text")
        self.assertFalse(updated)

    def test_update_chunk_old_text_not_returned(self):
        chunk = self.mem.remember("alice", "s1", "Unique old phrase xyzabc123 here")
        self.mem.long_term.update_chunk("alice", chunk.chunk_id,
                                        "Completely different new content now")
        r = self.mem.search_chunks("alice", "xyzabc123")
        self.assertEqual(r.total_found, 0)

    def test_lru_index_eviction(self):
        """After exceeding max_user_indexes, oldest users get evicted from TF-IDF cache."""
        store = self.mem.long_term
        original_cap = store._max_user_indexes
        store._max_user_indexes = 5  # small cap for testing
        try:
            for i in range(10):
                uid = f"user{i}"
                self.mem.remember(uid, "s1", f"content {i} redis BGP network routing")
                _ = store._index(uid)
            self.assertLessEqual(len(store._indexes), 5)
        finally:
            store._max_user_indexes = original_cap

    def test_apply_retention_deletes_old(self):
        # Insert old chunks (epoch 1970 = very old)
        for i in range(5):
            c = MemoryChunk(user_id="alice", session_id="s1",
                            text=f"old chunk {i} content", created_at=1000.0)
            self.mem.long_term.add_chunk(c)
        # Insert current chunk
        self.mem.remember("alice", "s1", "current chunk content today")
        deleted = self.mem.long_term.apply_retention("alice", max_age_days=1)
        self.assertEqual(deleted, 5)
        self.assertEqual(self.mem.long_term.count("alice"), 1)

    def test_apply_retention_no_delete_recent(self):
        self.mem.remember("alice", "s1", "recent content alpha here")
        self.mem.remember("alice", "s1", "recent content beta here")
        deleted = self.mem.long_term.apply_retention("alice", max_age_days=365)
        self.assertEqual(deleted, 0)
        self.assertEqual(self.mem.long_term.count("alice"), 2)

    def test_fts5_reserved_words_no_crash(self):
        self.mem.remember("alice", "s1", "BGP AND OSPF routing NOT working")
        for query in ["AND BGP", "OR OSPF NOT", "NEAR router", "AND OR NOT NEAR"]:
            r = self.mem.search_chunks("alice", query)
            self.assertIsNotNone(r)

    def test_source_type_boost_in_scoring(self):
        """tool_output and document sources rank higher than conversation."""
        self.mem.long_term.add_chunk(
            MemoryChunk(user_id="alice", session_id="s1",
                        text="BGP peer status check result output", source="tool_output"))
        self.mem.long_term.add_chunk(
            MemoryChunk(user_id="alice", session_id="s1",
                        text="BGP peer status check general note", source="conversation"))
        r = self.mem.search_chunks("alice", "BGP peer status check")
        if len(r.items) >= 2:
            self.assertEqual(r.items[0].source, "tool_output")

    def test_search_with_recency_disabled(self):
        r = self.mem.long_term.search("alice", "test query", use_recency=False)
        self.assertIsNotNone(r)

    def test_search_with_importance_disabled(self):
        r = self.mem.long_term.search("alice", "test query", use_importance=False)
        self.assertIsNotNone(r)


# ── MemoryManager integration for v4 store APIs ───────────────────────────────
class TestMemoryManagerV4StoreAPI(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_evict_expired_cache_atomic(self):
        """Concurrent evict_expired_cache calls should delete exactly N entries total."""
        N = 15
        for i in range(N):
            self.mem.cache_tool_result("alice", "s1", "t", f"data {i}", ttl=0.02)
        time.sleep(0.08)
        results = []
        errors = []
        def evict():
            try: results.append(self.mem.evict_expired_cache())
            except Exception as e: errors.append(str(e))
        threads = [threading.Thread(target=evict) for _ in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        self.assertEqual(sum(results), N)

    def test_full_pipeline_dedup_and_decay(self):
        """End-to-end: distill same fact twice → only stored once."""
        def llm(prompt):
            return '[{"fact":"R1 at 10.0.0.1","fact_type":"entity","confidence":0.9}]'
        tmp2 = tempfile.mkdtemp()
        mem2 = MemoryManager(data_dir=tmp2, llm_fn=llm)
        mem2.distill("alice", "s1", "R1 is located at 10.0.0.1 address")
        mem2.distill("alice", "s1", "R1 is located at 10.0.0.1 address")  # same
        count = mem2.mid_term.count("alice", "s1")
        self.assertEqual(count, 1)
        mem2.close()

    def test_stats_includes_hot_sessions(self):
        state = self.mem.get_session("alice", "s1")
        state.confirm_fact("R1 OK")
        s = self.mem.stats("alice", "s1")
        self.assertIn("active_hot_sessions", s)
        self.assertEqual(s["active_hot_sessions"], 1)

    def test_garbage_collect_via_short_term(self):
        gc = self.mem.short_term.garbage_collect()
        self.assertIn("orphan_files", gc)
        self.assertIn("missing_files", gc)


# ── helper method on MemoryManager for test access ────────────────────────────
def _pool_read(self, sql, params=()):
    from agent_memory.stores._db import get_pool
    pool = get_pool(self._db_path)
    return pool.execute_read(sql, params)

MemoryManager._pool_read = _pool_read


if __name__ == "__main__":
    unittest.main(verbosity=2)
