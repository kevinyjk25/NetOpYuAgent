"""
agent_memory/tests/test_memory.py — Production test suite v2
Run: python -m unittest agent_memory.tests.test_memory -v
  or: python agent_memory/tests/test_memory.py
"""
from __future__ import annotations
import os, sys, time, tempfile, threading, unittest, sqlite3

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_memory import MemoryManager, MemoryChunk, MemoryFact, FactExtractor, TFIDFIndex
from agent_memory.schemas import _validate_str


def _mem(tmp): return MemoryManager(data_dir=os.path.join(tmp, "m"))
def _mem_llm(tmp):
    return MemoryManager(
        data_dir=os.path.join(tmp, "ml"),
        llm_fn=lambda p: '[{"fact":"User prefers Python 3.12","fact_type":"preference","confidence":0.9}]'
    )


# ─── Schemas / Validation ─────────────────────────────────────────────────────
class TestSchemaValidation(unittest.TestCase):

    def test_chunk_empty_text_raises(self):
        with self.assertRaises(ValueError):
            MemoryChunk(user_id="alice", session_id="s1", text="")

    def test_chunk_none_user_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            MemoryChunk(user_id=None, session_id="s1", text="hello")

    def test_chunk_whitespace_text_raises(self):
        with self.assertRaises(ValueError):
            MemoryChunk(user_id="alice", session_id="s1", text="   ")

    def test_chunk_long_text_truncated(self):
        big = "x" * 2_000_000
        chunk = MemoryChunk(user_id="alice", session_id="s1", text=big)
        self.assertLessEqual(len(chunk.text), 1_000_000)

    def test_fact_confidence_clamped(self):
        f = MemoryFact(user_id="u", session_id="s", fact="test fact here", confidence=1.5)
        self.assertEqual(f.confidence, 1.0)
        f2 = MemoryFact(user_id="u", session_id="s", fact="test fact here", confidence=-0.5)
        self.assertEqual(f2.confidence, 0.0)

    def test_fact_invalid_type_defaults(self):
        f = MemoryFact(user_id="u", session_id="s", fact="test fact here", fact_type="bad_type")
        self.assertEqual(f.fact_type, "general")

    def test_validate_str_none_raises(self):
        with self.assertRaises(ValueError):
            _validate_str(None, "field", 100)

    def test_validate_str_empty_raises(self):
        with self.assertRaises(ValueError):
            _validate_str("  ", "field", 100, allow_empty=False)


# ─── TF-IDF ──────────────────────────────────────────────────────────────────
class TestTFIDFIndex(unittest.TestCase):

    def test_add_and_query(self):
        idx = TFIDFIndex()
        idx.add("d1", "redis cache ttl expiry config")
        idx.add("d2", "python language programming basics")
        idx.add("d3", "redis cluster replication setup")
        ids = [r[0] for r in idx.query("redis config")]
        self.assertTrue("d1" in ids or "d3" in ids)

    def test_remove(self):
        idx = TFIDFIndex()
        idx.add("d1", "remove this document content text")
        idx.remove("d1")
        self.assertEqual(idx.query("remove document"), [])

    def test_empty_query(self):
        self.assertEqual(TFIDFIndex().query("anything"), [])

    def test_size_after_ops(self):
        idx = TFIDFIndex()
        idx.add("a", "hello world text")
        idx.add("b", "foo bar baz qux")
        self.assertEqual(idx.size, 2)
        idx.remove("a")
        self.assertEqual(idx.size, 1)

    def test_lru_eviction(self):
        idx = TFIDFIndex(max_docs=3)
        idx.add("d1", "first document text content")
        idx.add("d2", "second document text content")
        idx.add("d3", "third document text content")
        idx.add("d4", "fourth document text content")   # evicts d1
        self.assertEqual(idx.size, 3)
        ids = [r[0] for r in idx.query("first document")]
        self.assertNotIn("d1", ids)

    def test_thread_safe_concurrent_add(self):
        idx = TFIDFIndex(max_docs=10_000)
        errors = []
        def worker(n):
            try:
                for i in range(100):
                    idx.add(f"d_{n}_{i}", f"redis bgp network config item {n} {i}")
            except Exception as e:
                errors.append(str(e))
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        self.assertGreater(idx.size, 0)

    def test_top_k_respected(self):
        idx = TFIDFIndex()
        for i in range(20):
            idx.add(f"d{i}", f"redis cluster node config item{i}")
        self.assertLessEqual(len(idx.query("redis", top_k=3)), 3)

    def test_update_doc(self):
        idx = TFIDFIndex()
        idx.add("d1", "redis cache config settings")
        idx.add("d1", "postgres database connection pool")
        self.assertEqual(idx.size, 1)
        results = idx.query("postgres")
        self.assertTrue(any(r[0] == "d1" for r in results))


# ─── Long-Term ────────────────────────────────────────────────────────────────
class TestLongTerm(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem(self.tmp)
    def tearDown(self): self.mem.close() if hasattr(self.mem, 'close') else None

    def test_store_and_retrieve(self):
        self.mem.remember("alice", "s1", "BGP neighbor state went down on R1 router")
        r = self.mem.search_chunks("alice", "BGP routing network")
        self.assertGreater(r.total_found, 0)
        self.assertTrue(any("BGP" in c.text for c in r.items))

    def test_user_isolation(self):
        self.mem.remember("alice", "s1", "Alice project X confidential data here")
        self.mem.remember("bob", "s1", "Bob project Y confidential data here")
        r = self.mem.search_chunks("alice", "confidential project data")
        for c in r.items:
            self.assertEqual(c.user_id, "alice")

    def test_cross_session(self):
        self.mem.remember("alice", "old", "Redis cluster 6 nodes configured setup")
        self.mem.remember("alice", "new", "Prometheus disk metrics alert query")
        r = self.mem.search_chunks("alice", "Redis cluster nodes")
        self.assertIn("old", {c.session_id for c in r.items})

    def test_list_sessions(self):
        self.mem.remember("alice", "s1", "content session one text")
        self.mem.remember("alice", "s2", "content session two text different")
        sessions = self.mem.list_sessions("alice")
        self.assertIn("s1", sessions)
        self.assertIn("s2", sessions)

    def test_fts_special_chars_no_crash(self):
        self.mem.remember("alice", "s1", 'Error: "timeout" on port 8080 config')
        r = self.mem.search_chunks("alice", 'timeout "error" port')
        self.assertIsNotNone(r)

    def test_count(self):
        self.mem.remember("alice", "s1", "chunk one text content here")
        self.mem.remember("alice", "s1", "chunk two text content here")
        self.mem.remember("bob", "s1", "bob user chunk content text")
        self.assertEqual(self.mem.long_term.count("alice"), 2)
        self.assertEqual(self.mem.long_term.count("bob"), 1)

    def test_metadata_preserved(self):
        self.mem.remember("alice", "s1", "chunk with metadata content here", metadata={"turn": 5})
        r = self.mem.search_chunks("alice", "chunk metadata content")
        self.assertGreater(r.total_found, 0)
        self.assertEqual(r.items[0].metadata.get("turn"), 5)

    def test_source_filter(self):
        self.mem.remember("alice", "s1", "conversation turn text content", source="conversation")
        self.mem.remember("alice", "s1", "tool output data text content", source="tool_output")
        r = self.mem.search_chunks("alice", "text content", source_filter="conversation")
        for c in r.items:
            self.assertEqual(c.source, "conversation")

    def test_batch_write(self):
        texts = [f"batch chunk number {i} about redis BGP network" for i in range(20)]
        chunks = self.mem.remember_batch("alice", "s1", texts)
        self.assertEqual(len(chunks), 20)
        self.assertEqual(self.mem.long_term.count("alice"), 20)

    def test_elapsed_ms_populated(self):
        self.mem.remember("alice", "s1", "content for elapsed time test")
        r = self.mem.search_chunks("alice", "content elapsed time")
        self.assertGreaterEqual(r.elapsed_ms, 0)

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            self.mem.remember("alice", "s1", "")

    def test_none_user_rejected(self):
        with self.assertRaises((ValueError, TypeError, Exception)):
            self.mem.remember(None, "s1", "some text here content")

    def test_delete_chunk(self):
        chunk = self.mem.remember("alice", "s1", "chunk to be deleted later")
        result = self.mem.long_term.delete_chunk("alice", chunk.chunk_id)
        self.assertTrue(result)
        self.assertEqual(self.mem.long_term.count("alice"), 0)


# ─── Mid-Term ─────────────────────────────────────────────────────────────────
class TestMidTerm(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem(self.tmp)
    def tearDown(self): self.mem.close() if hasattr(self.mem, 'close') else None

    def test_add_and_search(self):
        self.mem.add_fact("alice", "s1", "Alice prefers dark mode UI always", "preference", 0.95)
        r = self.mem.search_facts("alice", "UI preference dark mode", session_id="s1")
        self.assertGreater(r.total_found, 0)

    def test_type_filter(self):
        self.mem.add_fact("alice", "s1", "Use Python 3.12 for all scripts", "preference", 0.9)
        self.mem.add_fact("alice", "s1", "R1 router is at 10.0.0.1 address", "entity", 0.8)
        r = self.mem.search_facts("alice", "config scripts", session_id="s1", fact_type="preference")
        for f in r.items:
            self.assertEqual(f.fact_type, "preference")

    def test_confidence_filter(self):
        self.mem.add_fact("alice", "s1", "Low confidence guess unverified", "general", 0.3)
        self.mem.add_fact("alice", "s1", "High confidence verified known fact", "general", 0.9)
        r = self.mem.search_facts("alice", "fact verified", session_id="s1", min_confidence=0.7)
        for f in r.items:
            self.assertGreaterEqual(f.confidence, 0.7)

    def test_session_scoping(self):
        self.mem.add_fact("alice", "s1", "Session1 Redis cache node fact", "general", 1.0)
        self.mem.add_fact("alice", "s2", "Session2 Postgres database fact", "general", 1.0)
        r = self.mem.search_facts("alice", "Redis Postgres session", session_id="s1")
        self.assertTrue({f.session_id for f in r.items} <= {"s1"})

    def test_cross_session_no_filter(self):
        self.mem.add_fact("alice", "s1", "Old session Redis cache preference", "preference", 0.9)
        self.mem.add_fact("alice", "s2", "New session Mongo database preference", "preference", 0.8)
        r = self.mem.search_facts("alice", "database cache preference")
        self.assertGreater(r.total_found, 0)

    def test_update_and_delete(self):
        f = self.mem.add_fact("alice", "s1", "Original text fact here now", "general", 0.7)
        self.mem.mid_term.update_fact(f.fact_id, "alice", "s1", "Updated text fact here", 0.95)
        all_f = self.mem.mid_term.list_all("alice", "s1")
        self.assertEqual(all_f[0].fact, "Updated text fact here")
        self.assertAlmostEqual(all_f[0].confidence, 0.95)
        self.mem.mid_term.delete_fact("alice", "s1", f.fact_id)
        self.assertEqual(self.mem.mid_term.count("alice", "s1"), 0)

    def test_batch_add(self):
        facts = [
            MemoryFact(user_id="alice", session_id="s1",
                       fact=f"Fact number {i} about redis cluster node", fact_type="general")
            for i in range(10)
        ]
        self.mem.mid_term.add_facts_batch(facts)
        self.assertEqual(self.mem.mid_term.count("alice", "s1"), 10)

    def test_source_chunk_ids_preserved(self):
        chunk = self.mem.remember("alice", "s1", "source chunk text content here")
        f = self.mem.add_fact("alice", "s1", "derived fact from source chunk",
                              source_chunk_ids=[chunk.chunk_id])
        all_f = self.mem.mid_term.list_all("alice", "s1")
        self.assertIn(chunk.chunk_id, all_f[0].source_chunk_ids)

    def test_empty_fact_rejected(self):
        with self.assertRaises(ValueError):
            self.mem.add_fact("alice", "s1", "", "general", 0.9)


# ─── Short-Term ───────────────────────────────────────────────────────────────
class TestShortTerm(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem(self.tmp)
    def tearDown(self): self.mem.close() if hasattr(self.mem, 'close') else None

    def test_store_and_read(self):
        entry = self.mem.cache_tool_result("alice", "s1", "ping", "A" * 500)
        r = self.mem.read_cached("alice", entry.ref_id, 0, 100)
        self.assertEqual(r["content"], "A" * 100)
        self.assertEqual(r["total_length"], 500)
        self.assertTrue(r["has_more"])
        self.assertEqual(r["next_offset"], 100)

    def test_paging_continuity(self):
        content = "".join(str(i % 10) for i in range(900))
        entry = self.mem.cache_tool_result("alice", "s1", "syslog", content)
        out = ""
        for offset in range(0, 900, 300):
            out += self.mem.read_cached("alice", entry.ref_id, offset, 300)["content"]
        self.assertEqual(out, content)

    def test_full_reconstruction(self):
        content = "Z" * 5000
        entry = self.mem.cache_tool_result("alice", "s1", "big_tool", content)
        out, offset = "", 0
        while True:
            page = self.mem.read_cached("alice", entry.ref_id, offset, 1000)
            out += page["content"]
            if not page["has_more"]: break
            offset = page["next_offset"]
        self.assertEqual(out, content)

    def test_user_isolation(self):
        entry = self.mem.cache_tool_result("alice", "s1", "tool", "alice private data")
        r = self.mem.read_cached("bob", entry.ref_id, 0, 100)
        self.assertIn("error", r)

    def test_preview_token_format(self):
        entry = self.mem.cache_tool_result("alice", "s1", "syslog_search", "log entry " * 200)
        preview = self.mem.get_cache_preview("alice", entry.ref_id)
        self.assertIn("[STORED:", preview)
        self.assertIn("syslog_search", preview)
        self.assertIn("chars", preview)

    def test_last_page_no_more(self):
        entry = self.mem.cache_tool_result("alice", "s1", "tool", "Short content text")
        r = self.mem.read_cached("alice", entry.ref_id, 0, 1000)
        self.assertFalse(r["has_more"])
        self.assertIsNone(r["next_offset"])

    def test_expiry(self):
        tmp2 = tempfile.mkdtemp()
        m2 = MemoryManager(data_dir=tmp2, session_ttl=0.01)
        entry = m2.cache_tool_result("alice", "s1", "t", "data content here", ttl=0.01)
        time.sleep(0.08)
        r = m2.read_cached("alice", entry.ref_id, 0, 100)
        self.assertIn("error", r)
        self.assertGreaterEqual(m2.evict_expired_cache(), 1)

    def test_clear_session(self):
        self.mem.cache_tool_result("alice", "s1", "t1", "data one content")
        self.mem.cache_tool_result("alice", "s1", "t2", "data two content")
        self.mem.cache_tool_result("alice", "s2", "t3", "data three other session")
        self.assertEqual(self.mem.clear_session_cache("alice", "s1"), 2)
        self.assertEqual(len(self.mem.short_term.list_session("alice", "s2")), 1)

    def test_invalid_offset_returns_error(self):
        entry = self.mem.cache_tool_result("alice", "s1", "tool", "content data here")
        r = self.mem.read_cached("alice", entry.ref_id, offset=-1, length=100)
        self.assertIn("error", r)

    def test_path_traversal_rejected(self):
        r = self.mem.read_cached("alice", "../../../etc/passwd", 0, 100)
        self.assertIn("error", r)

    def test_should_cache_threshold(self):
        self.assertFalse(self.mem.should_cache("short text"))
        self.assertTrue(self.mem.should_cache("X" * 5000))

    def test_auto_remember(self):
        content = "Important syslog output data " * 50
        self.mem.cache_tool_result("alice", "s1", "big_tool", content, auto_remember=True)
        r = self.mem.search_chunks("alice", "big_tool syslog Important output")
        self.assertGreaterEqual(r.total_found, 1)


# ─── FactExtractor ────────────────────────────────────────────────────────────
class TestFactExtractor(unittest.TestCase):

    def test_rule_based(self):
        ext = FactExtractor()
        facts = ext.extract("I prefer Python 3.12 for all automation scripts.", "alice", "s1")
        self.assertIsInstance(facts, list)

    def test_empty_text_returns_empty(self):
        ext = FactExtractor()
        self.assertEqual(ext.extract("", "alice", "s1"), [])
        self.assertEqual(ext.extract("   ", "alice", "s1"), [])

    def test_llm_driven(self):
        ext = FactExtractor(
            llm_fn=lambda p: '[{"fact":"Uses Python 3.12","fact_type":"preference","confidence":0.9}]'
        )
        facts = ext.extract("some input text here", "alice", "s1")
        self.assertEqual(len(facts), 1)
        self.assertAlmostEqual(facts[0].confidence, 0.9)

    def test_llm_bad_json_fallback(self):
        ext = FactExtractor(llm_fn=lambda p: "NOT JSON AT ALL !!!")
        facts = ext.extract("I prefer dark mode interface", "alice", "s1")
        self.assertIsInstance(facts, list)   # fallback, no exception raised

    def test_min_confidence_filter(self):
        ext = FactExtractor(
            llm_fn=lambda p: '[{"fact":"low conf","fact_type":"general","confidence":0.2}]',
            min_confidence=0.5,
        )
        self.assertEqual(ext.extract("some text here", "alice", "s1"), [])

    def test_long_text_truncated_to_llm(self):
        received_lens = []
        def capture_llm(prompt):
            received_lens.append(len(prompt))
            return '[]'
        ext = FactExtractor(llm_fn=capture_llm)
        ext.extract("X" * 100_000, "alice", "s1")
        # Prompt must be shorter than raw input due to _MAX_PROMPT_TEXT_LEN truncation
        self.assertLess(received_lens[0], 50_000)

    def test_source_chunk_ids_passed(self):
        ext = FactExtractor(
            llm_fn=lambda p: '[{"fact":"a fact","fact_type":"general","confidence":0.8}]'
        )
        facts = ext.extract("text here", "alice", "s1", source_chunk_ids=["c1", "c2"])
        self.assertEqual(facts[0].source_chunk_ids, ["c1", "c2"])


# ─── Concurrency ─────────────────────────────────────────────────────────────
class TestConcurrency(unittest.TestCase):

    def test_concurrent_writes_no_crash(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        errors = []
        def worker(uid):
            for i in range(20):
                try:
                    mem.remember(uid, "s1", f"chunk {i} redis BGP network routing config node{i}")
                    mem.add_fact(uid, "s1", f"entity fact {i} host config router", "entity", 0.8)
                except Exception as e:
                    errors.append(f"{uid}: {e}")
        threads = [threading.Thread(target=worker, args=(f"u{n}",)) for n in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [], f"Concurrent write errors: {errors[:3]}")

    def test_concurrent_read_write(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        for i in range(30):
            mem.remember("alice", "s1", f"preloaded chunk {i} redis BGP ospf routing config")
        errors = []
        def writer():
            for i in range(20):
                try:
                    mem.remember("alice", "s1", f"write chunk {i} redis network routing")
                except Exception as e:
                    errors.append(f"W: {e}")
        def reader():
            for _ in range(30):
                try:
                    mem.search_chunks("alice", "redis BGP network routing")
                except Exception as e:
                    errors.append(f"R: {e}")
        threads = (
            [threading.Thread(target=writer) for _ in range(2)] +
            [threading.Thread(target=reader) for _ in range(4)]
        )
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [], f"Read/write errors: {errors[:3]}")

    def test_concurrent_cache_ops(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        errors = []
        saved: list[tuple[str, str]] = []
        lock = threading.Lock()
        def cache_worker(n):
            for i in range(5):
                try:
                    e = mem.cache_tool_result(f"u{n}", "s1", "tool",
                                             f"data content item {n} row {i} " * 20)
                    with lock:
                        saved.append((f"u{n}", e.ref_id))
                except Exception as ex:
                    errors.append(str(ex))
        threads = [threading.Thread(target=cache_worker, args=(n,)) for n in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        for uid, rid in saved[:5]:
            r = mem.read_cached(uid, rid, 0, 100)
            self.assertNotIn("error", r, f"Read failed: {r}")


# ─── Integration ──────────────────────────────────────────────────────────────
class TestIntegration(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem_llm(self.tmp)
    def tearDown(self): self.mem.close() if hasattr(self.mem, 'close') else None

    def test_remember_distill_pipeline(self):
        chunk = self.mem.remember("alice", "s1", "Alice uses Python 3.12 for automation scripts")
        facts = self.mem.distill("alice", "s1", "Alice prefers Python 3.12",
                                 source_chunk_ids=[chunk.chunk_id])
        self.assertGreater(len(facts), 0)

    def test_build_context_within_budget(self):
        for i in range(20):
            self.mem.remember("alice", "s1", f"BGP session AS{i} dropped at 02:{i:02d} UTC on R1")
            self.mem.add_fact("alice", "s1", f"AS{i} peering managed by alice team", "entity", 0.9)
        ctx = self.mem.build_context("alice", "BGP neighbor", session_id="s1", max_chars=3000)
        self.assertLessEqual(len(ctx), 3200)   # small tolerance for headers

    def test_build_context_empty_when_no_data(self):
        self.assertEqual(self.mem.build_context("nobody_user", "anything here"), "")

    def test_unified_search_all_layers(self):
        self.mem.remember("alice", "s1", "Prometheus disk alert 95 percent on node01")
        self.mem.add_fact("alice", "s1", "node01 is in rack B datacenter east", "entity", 0.9)
        r = self.mem.search("alice", "disk Prometheus node alert", session_id="s1")
        self.assertIn("long_term", r)
        self.assertIn("mid_term", r)

    def test_layer_selection(self):
        self.mem.remember("alice", "s1", "Redis cluster 6 nodes production environment")
        r = self.mem.search("alice", "Redis cluster", layers=["long_term"])
        self.assertIn("long_term", r)
        self.assertNotIn("mid_term", r)

    def test_cross_session_alias(self):
        self.mem.remember("alice", "s1", "Historical BGP config from old session data")
        r = self.mem.search("alice", "BGP historical config", layers=["cross_session"])
        self.assertIn("cross_session", r)
        self.assertGreater(r["cross_session"].total_found, 0)

    def test_stats_accurate(self):
        self.mem.remember("alice", "s1", "chunk content text here now for test")
        self.mem.add_fact("alice", "s1", "a fact detail content here", "general", 1.0)
        self.mem.cache_tool_result("alice", "s1", "tool", "cached data content here")
        s = self.mem.stats("alice", "s1")
        self.assertEqual(s["long_term_chunks"], 1)
        self.assertEqual(s["mid_term_facts"], 1)
        self.assertEqual(s["short_term_entries"], 1)

    def test_cross_user_isolation(self):
        self.mem.remember("alice", "s1", "Alice confidential project alpha data secret")
        self.mem.remember("bob", "s1", "Bob confidential project beta data secret")
        for c in self.mem.search_chunks("alice", "confidential project secret").items:
            self.assertEqual(c.user_id, "alice")
        for c in self.mem.search_chunks("bob", "confidential project secret").items:
            self.assertEqual(c.user_id, "bob")

    def test_distill_empty_text_returns_empty(self):
        # With the fake LLM that always returns a fact, empty text guard in
        # memory_manager.distill() should short-circuit before calling extractor
        facts = self.mem.distill("alice", "s1", "")
        self.assertEqual(facts, [])


# ─── Cross-Session ────────────────────────────────────────────────────────────
class TestCrossSession(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem(self.tmp)
    def tearDown(self): self.mem.close() if hasattr(self.mem, 'close') else None

    def test_old_session_recalled(self):
        self.mem.remember("alice", "week-ago", "Redis sentinel 3 nodes configured setup")
        self.mem.remember("alice", "today", "Asking about Redis sentinel nodes again today")
        r = self.mem.search_chunks("alice", "Redis sentinel nodes")
        self.assertIn("week-ago", {c.session_id for c in r.items})

    def test_cross_session_facts(self):
        self.mem.add_fact("alice", "s1", "Alice prefers CLI over GUI always", "preference", 0.9)
        self.mem.add_fact("alice", "s2", "Alice uses vim editor exclusively here", "preference", 0.85)
        r = self.mem.search_facts("alice", "Alice editor preference tool")
        self.assertGreaterEqual(r.total_found, 1)

    def test_new_session_sees_old_chunks(self):
        self.mem.remember("alice", "old-sess", "Payments DB primary host on 192.168.1.100")
        r = self.mem.search_chunks("alice", "payments database host IP address")
        self.assertGreater(r.total_found, 0)
        self.assertEqual(r.items[0].session_id, "old-sess")

    def test_restart_persistence(self):
        self.mem.remember("alice", "s1", "Critical config payments on 192.168.1.100 host")
        self.mem.add_fact("alice", "s1", "Alice manages payments service primary", "entity", 0.9)
        mem2 = MemoryManager(data_dir=os.path.join(self.tmp, "m"))
        r = mem2.search_chunks("alice", "payments critical config host")
        self.assertGreater(r.total_found, 0)
        r2 = mem2.search_facts("alice", "payments alice manages service")
        self.assertGreater(r2.total_found, 0)


# ─── Performance ──────────────────────────────────────────────────────────────
class TestPerformance(unittest.TestCase):

    def test_batch_insert_faster_than_loop(self):
        tmp1, tmp2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        m1, m2 = MemoryManager(data_dir=tmp1), MemoryManager(data_dir=tmp2)
        texts = [f"content chunk {i} redis BGP network routing config node{i}" for i in range(100)]

        t0 = time.perf_counter()
        for t in texts:
            m1.remember("alice", "s1", t)
        loop_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        m2.remember_batch("alice", "s1", texts)
        batch_time = time.perf_counter() - t0

        self.assertLess(batch_time, loop_time / 3,
                        f"Batch {batch_time:.3f}s not 3x faster than loop {loop_time:.3f}s")

    def test_search_100_chunks_under_50ms(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        texts = [f"chunk {i} about redis BGP network routing config node{i}" for i in range(100)]
        mem.remember_batch("alice", "s1", texts)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            mem.search_chunks("alice", "redis network BGP routing")
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = sum(times) / len(times)
        self.assertLess(avg_ms, 50, f"Average search {avg_ms:.1f}ms exceeds 50ms SLA")

    def test_sqlite_wal_mode_enabled(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        mem.remember("alice", "s1", "WAL mode test content chunk here")
        db_path = os.path.join(tmp, "memory.db")
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode, "wal", "SQLite WAL mode must be enabled for production")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ─── v3 新增：覆盖本次所有修复项 ────────────────────────────────────────────

class TestClose(unittest.TestCase):
    """close() / context manager / WAL checkpoint"""

    def test_close_method_exists(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        mem.remember("alice", "s1", "close test content here")
        mem.close()                            # must not raise

    def test_context_manager(self):
        tmp = tempfile.mkdtemp()
        with MemoryManager(data_dir=tmp) as mem:
            mem.remember("alice", "s1", "context manager test content")
        # after __exit__ pool is closed; re-opening should work fine
        with MemoryManager(data_dir=tmp) as mem2:
            r = mem2.search_chunks("alice", "context manager test")
            self.assertGreater(r.total_found, 0)

    def test_double_close_safe(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        mem.close()
        mem.close()                            # second close must not raise

    def test_wal_checkpoint_on_close(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "memory.db")
        mem = MemoryManager(data_dir=tmp)
        for i in range(200):
            mem.remember("alice", "s1", f"checkpoint test chunk {i} content redis BGP")
        wal_before = os.path.getsize(db_path + "-wal") if os.path.exists(db_path + "-wal") else 0
        mem.close()
        wal_after = os.path.getsize(db_path + "-wal") if os.path.exists(db_path + "-wal") else 0
        # WAL should shrink (TRUNCATE checkpoint) or be gone after close
        self.assertLessEqual(wal_after, wal_before,
                             f"WAL not checkpointed: before={wal_before} after={wal_after}")

    def test_manual_checkpoint(self):
        tmp = tempfile.mkdtemp()
        with MemoryManager(data_dir=tmp) as mem:
            for i in range(50):
                mem.remember("alice", "s1", f"checkpoint manual chunk {i} content")
            mem.checkpoint()                   # must not raise


class TestUnicodePaging(unittest.TestCase):
    """Binary-mode byte-offset paging — correct for all Unicode."""

    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem(self.tmp)
    def tearDown(self): self.mem.close()

    def test_chinese_full_reconstruction(self):
        content = "日志条目：系统警告 redis BGP网络路由配置 " * 80
        entry = self.mem.cache_tool_result("alice", "s1", "syslog", content)
        out, offset = "", 0
        while True:
            page = self.mem.read_cached("alice", entry.ref_id, offset, 200)
            self.assertNotIn("error", page, f"Read error at offset {offset}: {page}")
            out += page["content"]
            if not page["has_more"]:
                break
            offset = page["next_offset"]
        self.assertEqual(out, content)

    def test_emoji_full_reconstruction(self):
        content = "🔥 Alert: disk 95% on node01 🚨 " * 60
        entry = self.mem.cache_tool_result("alice", "s1", "alert", content)
        out, offset = "", 0
        while True:
            page = self.mem.read_cached("alice", entry.ref_id, offset, 150)
            self.assertNotIn("error", page)
            out += page["content"]
            if not page["has_more"]:
                break
            offset = page["next_offset"]
        self.assertEqual(out, content)

    def test_mixed_ascii_cjk(self):
        content = "BGP session: " + "路由器R1 " * 50 + " dropped at 02:15 UTC"
        entry = self.mem.cache_tool_result("alice", "s1", "tool", content)
        page = self.mem.read_cached("alice", entry.ref_id, 0, 100)
        self.assertNotIn("error", page)
        self.assertTrue(len(page["content"]) > 0)

    def test_byte_offsets_reported(self):
        content = "Hello 你好 World"
        entry = self.mem.cache_tool_result("alice", "s1", "tool", content)
        page = self.mem.read_cached("alice", entry.ref_id, 0, 6)
        # next_offset must be a valid byte boundary (not char boundary)
        if page.get("has_more"):
            page2 = self.mem.read_cached("alice", entry.ref_id,
                                         page["next_offset"], 100)
            self.assertNotIn("error", page2)

    def test_total_bytes_field_present(self):
        content = "data " * 100
        entry = self.mem.cache_tool_result("alice", "s1", "tool", content)
        page = self.mem.read_cached("alice", entry.ref_id, 0, 50)
        self.assertIn("total_bytes", page)
        self.assertGreater(page["total_bytes"], 0)


class TestAtomicEvict(unittest.TestCase):
    """evict_expired() must delete each entry exactly once under concurrency."""

    def test_concurrent_evict_no_double_delete(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp, session_ttl=0.02)
        N = 20
        for i in range(N):
            mem.cache_tool_result("alice", "s1", "tool", f"data item {i}", ttl=0.02)
        time.sleep(0.08)

        results = []
        errors = []

        def evict():
            try:
                results.append(mem.evict_expired_cache())
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=evict) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [], f"Evict errors: {errors}")
        total = sum(results)
        self.assertEqual(total, N,
                         f"Expected {N} total evictions, got {total} across {results}")

    def test_evict_only_expired(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        # Two entries: one expires soon, one lives long
        mem.cache_tool_result("alice", "s1", "tool_short", "expires soon", ttl=0.02)
        mem.cache_tool_result("alice", "s1", "tool_long", "lives long", ttl=9999)
        time.sleep(0.08)
        n = mem.evict_expired_cache()
        self.assertEqual(n, 1)
        remaining = mem.short_term.list_session("alice", "s1")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].tool_name, "tool_long")
        mem.close()


class TestFTS5SafeQuery(unittest.TestCase):
    """FTS5 reserved words must not crash search."""

    def setUp(self): self.tmp = tempfile.mkdtemp(); self.mem = _mem(self.tmp)
    def tearDown(self): self.mem.close()

    def _seed(self):
        self.mem.remember("alice", "s1", "BGP neighbor state dropped on R1 router")
        self.mem.add_fact("alice", "s1", "R1 is at 10.0.0.1", "entity", 0.9)

    def test_AND_in_query(self):
        self._seed()
        r = self.mem.search_chunks("alice", "AND BGP")
        self.assertIsNotNone(r)

    def test_OR_in_query(self):
        self._seed()
        r = self.mem.search_chunks("alice", "BGP OR OSPF")
        self.assertIsNotNone(r)

    def test_NOT_in_query(self):
        self._seed()
        r = self.mem.search_chunks("alice", "NOT BGP")
        self.assertIsNotNone(r)

    def test_NEAR_in_query(self):
        self._seed()
        r = self.mem.search_chunks("alice", "NEAR BGP router")
        self.assertIsNotNone(r)

    def test_all_reserved_words(self):
        self._seed()
        r = self.mem.search_chunks("alice", "AND OR NOT NEAR COLUMN ROW MATCH")
        self.assertIsNotNone(r)   # must not raise

    def test_special_chars_in_query(self):
        self._seed()
        r = self.mem.search_chunks("alice", 'BGP "neighbor" (state) *dropped*')
        self.assertIsNotNone(r)

    def test_empty_query_no_crash(self):
        self._seed()
        r = self.mem.search_chunks("alice", "")
        self.assertIsNotNone(r)

    def test_fts_reserved_in_facts(self):
        self._seed()
        r = self.mem.search_facts("alice", "AND OR NOT entity")
        self.assertIsNotNone(r)


class TestSnapshotReads(unittest.TestCase):
    """execute_read uses BEGIN DEFERRED for snapshot consistency."""

    def test_read_sees_committed_write(self):
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        mem.remember("alice", "s1", "snapshot test write content here")
        # read immediately after write must see the data
        r = mem.search_chunks("alice", "snapshot test write")
        self.assertGreater(r.total_found, 0)
        mem.close()

    def test_concurrent_reads_consistent(self):
        """Multiple threads reading simultaneously all see consistent state."""
        tmp = tempfile.mkdtemp()
        mem = MemoryManager(data_dir=tmp)
        for i in range(30):
            mem.remember("alice", "s1", f"consistency chunk {i} redis BGP network routing")
        errors = []
        counts = []

        def reader():
            try:
                r = mem.search_chunks("alice", "redis BGP network")
                counts.append(r.total_found)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        # All reads should see the same count
        self.assertEqual(len(set(counts)), 1, f"Inconsistent counts: {counts}")
        mem.close()


class TestAgentMemoryCallback(unittest.TestCase):
    """AgentMemoryCallback correctness."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = MemoryManager(data_dir=self.tmp)
        self.cb = AgentMemoryCallback(self.mem, "alice", "s1")

    def tearDown(self):
        self.mem.close()

    def test_on_llm_end_stores_chunk(self):
        class FakeGen:
            text = "BGP session dropped on R1 at 02:15 UTC today"
        class FakeResult:
            generations = [[FakeGen()]]
        self.cb.on_llm_end(FakeResult())
        r = self.mem.search_chunks("alice", "BGP session dropped")
        self.assertGreater(r.total_found, 0)

    def test_pending_buffer_populated(self):
        class FakeGen:
            text = "Redis cluster is healthy all nodes green"
        class FakeResult:
            generations = [[FakeGen()]]
        self.cb.on_llm_end(FakeResult())
        self.assertEqual(len(self.cb._pending), 1)
        cid, text = self.cb._pending[0]
        self.assertIn("Redis", text)

    def test_on_tool_end_small_inline(self):
        self.cb.on_tool_end("ping ok 10.0.0.1", name="ping")
        r = self.mem.search_chunks("alice", "ping ok")
        self.assertGreater(r.total_found, 0)
        self.assertIsNone(self.cb.last_tool_token)

    def test_on_tool_end_large_cached(self):
        big_output = "syslog line data\n" * 500
        self.cb.on_tool_end(big_output, name="syslog_search")
        self.assertIsNotNone(self.cb.last_tool_token)
        self.assertIn("[STORED:", self.cb.last_tool_token)

    def test_distill_pending_uses_buffered_text(self):
        class FakeGen:
            text = "I prefer Python 3.12 for all automation scripts here"
        class FakeResult:
            generations = [[FakeGen()], [FakeGen()]]
        self.cb.on_llm_end(FakeResult())
        self.cb._distill_pending()
        # pending should be cleared
        self.assertEqual(len(self.cb._pending), 0)

    def test_on_agent_finish_flushes(self):
        class FakeGen:
            text = "Final answer: BGP config is correct and verified"
        class FakeResult:
            generations = [[FakeGen()]]
        self.cb.on_llm_end(FakeResult())
        self.assertGreater(len(self.cb._pending), 0)
        self.cb.on_agent_finish(None)
        self.assertEqual(len(self.cb._pending), 0)

    def test_new_session_rotates(self):
        class FakeGen:
            text = "Session 1 content about Redis cluster nodes"
        class FakeResult:
            generations = [[FakeGen()]]
        self.cb.on_llm_end(FakeResult())
        self.cb.new_session("s2")
        self.assertEqual(self.cb.session_id, "s2")
        self.assertEqual(len(self.cb._pending), 0)
        self.assertEqual(self.cb._turn_count, 0)

    def test_distill_every_n_turns(self):
        """Facts are distilled automatically every N turns."""
        cb = AgentMemoryCallback(self.mem, "alice", "s-auto",
                                 distill_every_n_turns=2)
        class FakeGen:
            text = "I prefer Python 3.12 always for all scripts"
        class FakeResult:
            generations = [[FakeGen()]]
        cb.on_llm_end(FakeResult())   # turn 1 — no distill yet
        self.assertEqual(len(cb._pending), 1)
        cb.on_llm_end(FakeResult())   # turn 2 — distill fires
        self.assertEqual(len(cb._pending), 0)


# Import AgentMemoryCallback for the new tests
from agent_memory.examples.langchain_integration import AgentMemoryCallback
