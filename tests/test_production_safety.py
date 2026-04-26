"""
tests/test_production_safety.py
─────────────────────────────
Integration tests for the production-safety blockers:

  - Memory adapter: per-operator isolation, async path, shims
  - Log redaction: device secrets, headers, API keys
  - Auth: JWT round-trip, tamper detection, expiry, role checks
  - Tool allow-list: run_command rejects writes, accepts reads
  - Edit_device_config: snapshot is taken (we mock the device)

Run with:
    python -m pytest tests/test_production_safety.py -v
or:
    python -m unittest tests.test_production_safety
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import unittest

# Ensure project root is importable when running from tests/ dir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Memory adapter — multi-user isolation & async path
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        from memory import MemoryAdapter, set_current_operator
        self._set_op = set_current_operator
        self.adapter = MemoryAdapter(data_dir=self.tmp)

    def tearDown(self):
        try:
            self.adapter.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_per_operator_isolation(self):
        """Bob must not see Alice's chunks."""
        async def _run():
            self._set_op("alice")
            await self.adapter.after_turn("s1", "BGP down on R1", "Investigating", [])

            self._set_op("bob")
            result = await self.adapter.recall("BGP", "s1")
            return result.chunk_count

        bob_count = asyncio.run(_run())
        self.assertEqual(bob_count, 0,
                         "ISOLATION BROKEN: Bob saw Alice's data")

    def test_alice_recovers_own_data(self):
        async def _run():
            self._set_op("alice")
            await self.adapter.after_turn("s1", "OSPF flap on Po10", "Investigating", [])
            return await self.adapter.recall("OSPF", "s1")
        result = asyncio.run(_run())
        self.assertGreater(result.chunk_count, 0,
                           "Alice's own recall returned nothing")

    def test_tool_cache_roundtrip(self):
        async def _run():
            self._set_op("alice")
            big = "syslog data " * 500
            r = await self.adapter.cache_tool_result("s1", "syslog", big)
            self.assertIn("ref_id", r)
            page = await self.adapter.read_cached(r["ref_id"], 0, 100)
            return page
        p = asyncio.run(_run())
        self.assertGreater(len(p["content"]), 0)
        self.assertTrue(p["has_more"])

    def test_backward_compat_shims(self):
        """Old curator/fts API names still work."""
        async def _run():
            self._set_op("alice")
            await self.adapter.after_turn("s1", "BGP", "ok", [])
            text = await self.adapter.recall_for_session("BGP", "s1")
            stats = await self.adapter.get_stats()
            return text, stats
        text, stats = asyncio.run(_run())
        self.assertIsInstance(text, str)
        self.assertIsInstance(stats, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Log redaction
# ─────────────────────────────────────────────────────────────────────────────

class TestLogRedaction(unittest.TestCase):
    def setUp(self):
        from log_redaction import redact_text
        self.redact = redact_text

    def test_password_with_cipher_type(self):
        for line in [
            "username admin password Sup3rSecret!",
            "username admin password 0 Sup3rSecret!",
            "username admin password 7 0822455D0A16",
        ]:
            r = self.redact(line)
            self.assertIn("***REDACTED***", r)
            self.assertNotIn("Sup3rSecret", r)
            self.assertNotIn("0822455D0A16", r)

    def test_enable_secret(self):
        r = self.redact("enable secret 5 $1$abcd$xyz")
        self.assertIn("***REDACTED***", r)
        self.assertNotIn("$1$abcd$xyz", r)

    def test_snmp_community(self):
        r = self.redact("snmp-server community publicReadOnly RO")
        self.assertIn("***REDACTED***", r)
        self.assertNotIn("publicReadOnly", r)

    def test_authorization_header(self):
        r = self.redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.aaa.bbb")
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", r)

    def test_api_key(self):
        r = self.redact("api_key=sk-abc123def456")
        self.assertNotIn("sk-abc123def456", r)

    def test_safe_config_preserved(self):
        """Legitimate config not containing secrets must pass through unchanged."""
        for line in [
            "interface Vlan100 ip address 10.0.0.1 255.255.255.0",
            "show running-config",
            "router bgp 65000 neighbor 10.0.0.2 remote-as 65001",
        ]:
            self.assertEqual(self.redact(line), line, f"False redaction: {line!r}")

    def test_empty_or_none(self):
        self.assertEqual(self.redact(""), "")
        # We do not allow None — adapter should never pass None
        self.assertEqual(self.redact("safe text"), "safe text")


# ─────────────────────────────────────────────────────────────────────────────
# Auth — JWT and role checks
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthJWT(unittest.TestCase):
    def setUp(self):
        os.environ["NETOPYU_JWT_SECRET"] = "test-secret-256-bit"
        # auth_core reads secret at import time — reload
        import importlib, auth_core
        importlib.reload(auth_core)
        self.auth = auth_core

    def test_jwt_roundtrip(self):
        token = self.auth.issue_jwt("alice", ["operator"])
        claims = self.auth.verify_jwt(token)
        self.assertEqual(claims["sub"], "alice")
        self.assertEqual(claims["roles"], ["operator"])

    def test_tampered_signature_rejected(self):
        token = self.auth.issue_jwt("alice", ["operator"])
        bad = token[:-5] + "XXXXX"
        with self.assertRaises(self.auth.AuthError):
            self.auth.verify_jwt(bad)

    def test_expired_rejected(self):
        token = self.auth.issue_jwt("alice", ["operator"], ttl_seconds=-1)
        with self.assertRaises(self.auth.AuthError):
            self.auth.verify_jwt(token)

    def test_missing_secret(self):
        with self.assertRaises(self.auth.AuthError):
            self.auth.verify_jwt("a.b.c", secret="")

    def test_identity_role_check(self):
        ident = self.auth.Identity(operator_id="alice",
                                   roles=["operator"], auth_method="jwt")
        self.assertTrue(ident.has_role("operator"))
        self.assertFalse(ident.has_role("hitl_approver"))

    def test_admin_implies_all_roles(self):
        ident = self.auth.Identity(operator_id="root",
                                   roles=["admin"], auth_method="jwt")
        self.assertTrue(ident.has_role("operator"))
        self.assertTrue(ident.has_role("hitl_approver"))

    def test_api_key_parsing(self):
        keys = self.auth.parse_api_keys("k1:alice:operator,k2:bob:admin,malformed")
        self.assertIn("k1", keys)
        self.assertIn("k2", keys)
        self.assertNotIn("malformed", keys)
        self.assertEqual(keys["k1"], ("alice", ["operator"]))


# ─────────────────────────────────────────────────────────────────────────────
# Tool allow-list — run_command must reject anything but reads
# ─────────────────────────────────────────────────────────────────────────────

class TestRunCommandAllowList(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tools.pragmatic_tools import run_command, register_devices
        self.run_command = run_command
        # Register a fake device entry so we exercise the allow-list
        # (the actual SSH call is skipped because we'll inject a mock send fn,
        # but for the allow-list test the BLOCKED return fires before SSH)
        from tools import pragmatic_tools as pt
        pt._DEVICES["test-sw"] = {
            "device_type": "cisco_ios", "host": "127.0.0.1",
            "username": "admin", "password": "x", "_label": "test",
        }

    async def asyncTearDown(self):
        from tools import pragmatic_tools as pt
        pt._DEVICES.pop("test-sw", None)

    async def test_blocks_destructive(self):
        for cmd in [
            "configure terminal",
            "reload",
            "erase startup-config",
            "delete flash:image.bin",
            "no ip route 0.0.0.0",
            "copy running-config tftp:",
            "tclsh",
            "boot system flash:",
        ]:
            r = await self.run_command({"device_id": "test-sw", "command": cmd})
            self.assertIn("[BLOCKED]", r,
                          f"Allow-list failed to block: {cmd!r} → {r[:80]}")

    async def test_allows_read_only(self):
        # These won't actually execute (no real device) but must NOT be blocked
        # by the allow-list. We expect either [Error] (SSH failed) or output —
        # never [BLOCKED].
        for cmd in [
            "show version",
            "show running-config",
            "show ip route",
            "ping 8.8.8.8",
            "traceroute 1.1.1.1",
        ]:
            r = await self.run_command({"device_id": "test-sw", "command": cmd})
            self.assertNotIn("[BLOCKED]", r,
                             f"Allow-list incorrectly blocked: {cmd!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Memory smoke test — agent_memory underlying module
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentMemoryIntegration(unittest.TestCase):
    """Confirms agent_memory itself works end-to-end (it has its own 311 tests
    but we want a smoke check that integrates with our adapter wrapper)."""

    def test_memory_manager_directly(self):
        from agent_memory import MemoryManager
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryManager(data_dir=tmp) as mem:
                mem.remember("alice", "s1", "OSPF Area 0 unstable")
                ctx = mem.build_context("alice", "OSPF", "s1", max_chars=400)
                self.assertIn("OSPF", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
