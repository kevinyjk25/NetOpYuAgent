"""
tests/test_h2_async_resolution.py
=================================

Regression tests for the H2 async-HITL resolution machinery (2026-05 fixes):

  Bug 1 — double-fire race: the operator-decision path (router.deliver) and
          the SLA-timeout watchdog must never BOTH invoke on_resolved for the
          same interrupt. Ownership is claimed atomically via
          claim_async_pending(); whoever wins resolves, the loser is a no-op.

  Bug 2 — leaked entry: a producer that registers an async pending via
          register_async_pending() always gets an SLA watchdog, so a
          no-response interrupt eventually fires on_resolved(None) and the
          registry entry is reclaimed (no permanent leak).

  Plus: the happy path (operator decides → on_resolved(decision)) and a
        multi-"agent" isolation check (two interrupt_ids resolve independently
        and only deliver their own facts — the in-process analogue of two
        agent processes, since each agent process has its own module globals).

These run without httpx / fastapi (pure hitl_core + asyncio), so they execute
in the sandbox and in CI.
"""
from __future__ import annotations

import asyncio
import unittest


def _schema():
    from hitl_core.schema import (
        HitlPayload, ProposedAction, HitlDecision,
        TriggerKind, RiskLevel, InterruptMode, DecisionKind,
        CheckpointEntry, ResumeHandle, InterruptState,
    )
    return locals()


def _make_payload(interrupt_id: str, sla_seconds: int = 600):
    s = _schema()
    return s["HitlPayload"](
        interrupt_id   = interrupt_id,
        thread_id      = "sess-1",
        context_id     = "sess-1",
        title          = "test async hitl",
        description    = "unit test",
        proposed_action = s["ProposedAction"](
            action_type = "tool_call:test",
            target      = "user-x",
            parameters  = {},
            risk_level  = s["RiskLevel"].LOW,
            reversible  = True,
        ),
        trigger_kind   = s["TriggerKind"].EXTERNAL_DELEGATION,
        risk_level     = s["RiskLevel"].LOW,
        interrupt_mode = s["InterruptMode"].ASYNC_NONBLOCKING,
        sla_seconds    = sla_seconds,
    )


async def _build_router_with_entry(interrupt_id: str, sla_seconds: int = 600):
    """Create an InMemory store + router, persist a PENDING async entry."""
    from hitl_core.store import InMemoryCheckpointStore
    from hitl_core.router import HitlRouter
    s = _schema()
    store = InMemoryCheckpointStore()
    router = HitlRouter(store=store)
    payload = _make_payload(interrupt_id, sla_seconds)
    entry = s["CheckpointEntry"](
        interrupt_id  = interrupt_id,
        payload       = payload,
        resume_handle = s["ResumeHandle"](resumer_name="async_hitl", state={}),
    )
    await store.save(entry)
    return store, router, payload


class TestAsyncHappyPath(unittest.TestCase):
    def test_operator_decision_fires_on_resolved_once(self):
        async def run():
            from hitl_core.router import register_async_pending
            from hitl_core.pipeline import AsyncPendingHitl
            s = _schema()
            iid = "iid-happy"
            store, router, payload = await _build_router_with_entry(iid)

            calls = []
            async def on_resolved(i, decision, default, diverged):
                calls.append((i, decision, default, diverged))

            register_async_pending(
                AsyncPendingHitl(
                    interrupt_id = iid, payload = payload,
                    default_value = "permission_ok", on_resolved = on_resolved,
                    sla_seconds = 600, session_id = "sess-1",
                ),
                store=store,
            )
            decision = s["HitlDecision"](
                interrupt_id = iid,
                decision     = s["DecisionKind"].APPROVE,
                operator_id  = "op-1",
            )
            out = await router.deliver(decision)
            # async_resolved may be top-level or nested under "result"
            # depending on dispatch wrapping — assert on the callback instead,
            # which is the real contract.
            self.assertEqual(len(calls), 1)
            # APPROVE → not diverged
            self.assertFalse(calls[0][3])
            self.assertIsNotNone(out)
        asyncio.run(run())


class TestSlaTimeout(unittest.TestCase):
    def test_timeout_fires_on_resolved_none_and_reclaims(self):
        async def run():
            from hitl_core.router import register_async_pending, _async_registry
            from hitl_core.pipeline import AsyncPendingHitl
            iid = "iid-timeout"
            store, router, payload = await _build_router_with_entry(iid, sla_seconds=1)

            calls = []
            audits = []
            async def on_resolved(i, decision, default, diverged):
                calls.append((i, decision, default, diverged))
            async def on_audit(kind, iid_, detail):
                audits.append((kind, iid_))

            register_async_pending(
                AsyncPendingHitl(
                    interrupt_id = iid, payload = payload,
                    default_value = "permission_ok", on_resolved = on_resolved,
                    sla_seconds = 1, session_id = "sess-1",
                ),
                store=store, on_audit=on_audit,
            )
            self.assertIn(iid, _async_registry)
            await asyncio.sleep(1.4)   # let the watchdog fire
            # on_resolved fired once with decision=None (timeout)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0][1])
            self.assertTrue(calls[0][3])   # diverged=True on timeout
            # Registry reclaimed
            self.assertNotIn(iid, _async_registry)
            # ASYNC_TIMEOUT audited
            self.assertEqual(len(audits), 1)
        asyncio.run(run())


class TestDoubleFireRace(unittest.TestCase):
    def test_decision_and_timeout_resolve_only_once(self):
        """The race fixed by claim_async_pending: even if the SLA fires at
        ~the same time as an operator decision, on_resolved runs ONCE."""
        async def run():
            from hitl_core.router import register_async_pending, _async_registry
            from hitl_core.pipeline import AsyncPendingHitl
            s = _schema()
            iid = "iid-race"
            store, router, payload = await _build_router_with_entry(iid, sla_seconds=1)

            calls = []
            async def on_resolved(i, decision, default, diverged):
                # simulate an await point inside the callback (like SSE/audit)
                await asyncio.sleep(0)
                calls.append(decision)

            register_async_pending(
                AsyncPendingHitl(
                    interrupt_id = iid, payload = payload,
                    default_value = "permission_ok", on_resolved = on_resolved,
                    sla_seconds = 1, session_id = "sess-1",
                ),
                store=store,
            )
            decision = s["HitlDecision"](
                interrupt_id = iid,
                decision     = s["DecisionKind"].APPROVE,
                operator_id  = "op-1",
            )
            # Fire the operator decision right as the SLA window elapses, then
            # wait past the SLA so the watchdog also runs.
            await asyncio.sleep(0.9)
            await router.deliver(decision)
            await asyncio.sleep(0.6)
            # Exactly one resolution total (operator won; watchdog no-op).
            self.assertEqual(len(calls), 1)
            self.assertNotIn(iid, _async_registry)
        asyncio.run(run())


class TestMultiInterruptIsolation(unittest.TestCase):
    def test_two_interrupts_resolve_independently(self):
        """In-process analogue of two agents: distinct interrupt_ids resolve
        without cross-contaminating each other's on_resolved."""
        async def run():
            from hitl_core.router import register_async_pending
            from hitl_core.pipeline import AsyncPendingHitl
            s = _schema()
            store_a, router_a, p_a = await _build_router_with_entry("iid-A")
            store_b, router_b, p_b = await _build_router_with_entry("iid-B")

            seen = {"A": [], "B": []}
            async def mk(tag):
                async def cb(i, decision, default, diverged):
                    seen[tag].append(i)
                return cb

            register_async_pending(
                AsyncPendingHitl(interrupt_id="iid-A", payload=p_a,
                                 default_value="ok", on_resolved=await mk("A"),
                                 sla_seconds=600, session_id="sess-A"),
                store=store_a,
            )
            register_async_pending(
                AsyncPendingHitl(interrupt_id="iid-B", payload=p_b,
                                 default_value="ok", on_resolved=await mk("B"),
                                 sla_seconds=600, session_id="sess-B"),
                store=store_b,
            )
            await router_a.deliver(s["HitlDecision"](
                interrupt_id="iid-A", decision=s["DecisionKind"].APPROVE,
                operator_id="op",
            ))
            # Only A resolved; B untouched.
            self.assertEqual(seen["A"], ["iid-A"])
            self.assertEqual(seen["B"], [])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
