"""Complex multi-round delegation scenarios — boundary probing.

The delegation model (2026-06) is single-hop, one-delegation-per-target-per-
request, first-directive-wins-per-turn. Multi-hop / parallel fan-out / auto-
delegate are explicitly NOT implemented (TODO 2B future). These tests drive
the REAL loop with a scripted delegate_fn to pin what the system does at the
edges — so we can tell a genuine bug from a known limitation.

delegate_fn contract (the network seam):
    async def delegate_fn(directive, session_id, shared_facts, *, original_query="")
        -> async generator of peer chunks ({token}/{error}/{hitl_interrupt})
The loop accumulates peer tokens and injects a result observation for the
next turn; the LLM then synthesizes or delegates again.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig
from runtime.stop_policy import StopOutcome


def _peer_stream(chunks):
    """Build an async-gen delegate_fn that yields the given chunks, keyed by
    target so a single fn can serve multiple peers with different scripts."""
    async def _fn(directive, session_id, shared_facts, *, original_query=""):
        for ch in chunks.get(directive.target, [{"token": f"(no script for {directive.target})"}]):
            yield ch
    return _fn


def _loop(llm_fn, delegate_fn):
    return AgentRuntimeLoop(memory_router=None, config=RuntimeConfig(),
                            llm_fn=llm_fn, delegate_fn=delegate_fn)


class TestChainedDelegationTwoPeers(unittest.TestCase):
    """A real cross-domain workflow: LAN asks DC (app layer), then asks a
    SECOND peer (e.g. security), then synthesizes. Two DISTINCT targets are
    allowed — the per-request set only blocks REPEATS."""
    def test_delegate_to_two_different_peers_then_synthesize(self):
        async def t():
            peer_scripts = {
                "dc-agent":  [{"token": "CRM app ACL: alice DENIED"}],
                "sec-agent": [{"token": "alice account: not in crm-users group"}],
            }
            delegate = _peer_stream(peer_scripts)

            turns = iter([
                # turn 1: delegate to dc
                "先查应用层权限。\n[DELEGATE:dc-agent] 查 alice 的 CRM 权限",
                # turn 2: delegate to a DIFFERENT peer
                "再查安全组归属。\n[DELEGATE:sec-agent] 查 alice 的安全组",
                # turn 3: synthesize from both results
                ("综合两个 peer 的结果:alice 被 CRM ACL 拒绝,且不在 crm-users 安全组。"
                 "根因是安全组缺失导致 ACL 拒绝。建议将 alice 加入 crm-users 组。"),
            ])
            async def llm(query, context, state):
                return next(turns, "综合完成,已给出根因与建议。")

            loop = _loop(llm, delegate)
            res = await loop.run(query="诊断 alice 访问 CRM 失败(跨域)",
                                 session_id="chain-2peer", tool_registry={})
            # both peers' results made it into the final synthesis
            self.assertIn("crm-users", res.final_response)
            self.assertIn("安全组", res.final_response)
            self.assertEqual(res.outcome, StopOutcome.STOP_GRACEFUL)
        asyncio.run(t())


class TestRepeatSameTargetSuppressed(unittest.TestCase):
    """Re-delegating the SAME target in one request is suppressed + a
    synthesis turn is forced (the case-1 storm guard)."""
    def test_second_delegate_same_target_suppressed(self):
        async def t():
            delegate_calls = []

            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                delegate_calls.append(directive.target)
                yield {"token": "dc partial result — inconclusive"}

            turns = iter([
                "[DELEGATE:dc-agent] 查 alice 权限",
                # LLM wrongly tries to re-delegate the same peer
                "结果不够,再查一次。\n[DELEGATE:dc-agent] 再查 alice 权限",
                # forced synthesis turn after suppression
                ("基于 dc-agent 已返回的结果(不确定),如实说明:当前无法完全确认 alice 权限,"
                 "dc-agent 返回结果不足,建议人工核查 CRM ACL 配置。"),
            ])
            async def llm(query, context, state):
                return next(turns, "已如实总结缺口,未重复委派。")

            loop = _loop(llm, delegate)
            res = await loop.run(query="诊断 alice", session_id="repeat-same",
                                 tool_registry={})
            # dc-agent was delegated to exactly ONCE despite two attempts
            self.assertEqual(delegate_calls.count("dc-agent"), 1,
                             f"repeat to same target must be suppressed: {delegate_calls}")
            # the final answer honestly reports the shortfall (not a fake
            # re-delegation) — accept any of the honest-gap phrasings.
            haystack = res.final_response + " " + " ".join(res.unresolved)
            self.assertTrue(
                any(w in haystack for w in ("缺口", "不足", "无法", "人工", "核查")),
                f"suppressed repeat must yield an honest gap report: {haystack}")
        asyncio.run(t())


class TestMultipleDirectivesOneTurn(unittest.TestCase):
    """If the LLM emits two [DELEGATE:] in one turn, only the FIRST is
    honored (the execution model is one hand-off per turn)."""
    def test_only_first_directive_honored(self):
        async def t():
            called = []

            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                called.append(directive.target)
                yield {"token": f"result from {directive.target}"}

            turns = iter([
                # both in one turn — only dc-agent (first) should fire
                "并行查两个。\n[DELEGATE:dc-agent] 查A\n[DELEGATE:sec-agent] 查B",
                "综合 dc-agent 的结果给出答复,内容足够完成用户请求,无需其他信息。",
            ])
            async def llm(query, context, state):
                return next(turns, "完成。")

            loop = _loop(llm, delegate)
            await loop.run(query="q", session_id="multi-directive", tool_registry={})
            self.assertEqual(called, ["dc-agent"],
                             f"only the first directive should be honored: {called}")
        asyncio.run(t())


class TestPeerErrorPropagation(unittest.TestCase):
    """A peer returning an error must be surfaced honestly to the next turn,
    not silently dropped or treated as success."""
    def test_peer_error_reaches_synthesis(self):
        async def t():
            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                yield {"token": "starting"}
                yield {"error": "dc-agent: BGP EVPN lookup timeout"}

            turns = iter([
                "[DELEGATE:dc-agent] 查 VXLAN 路径",
                # the injected [委派失败] observation should let the LLM report it
                ("委派 dc-agent 失败(BGP EVPN 查询超时)。基于本地能力,无法完成 VXLAN 路径追踪,"
                 "已如实告知用户 dc-agent 暂时不可用,建议稍后重试或人工介入。"),
            ])
            async def llm(query, context, state):
                return next(turns, "已如实报告 peer 错误。")

            loop = _loop(llm, delegate)
            res = await loop.run(query="追踪 VXLAN 路径", session_id="peer-err",
                                 tool_registry={})
            self.assertIn("dc-agent", res.final_response)
            # honest failure report, not a fabricated success
            self.assertTrue(
                any(w in res.final_response for w in ("失败", "不可用", "超时", "重试")),
                f"peer error must be honestly reported: {res.final_response}")
        asyncio.run(t())


class TestPeerHitlPark(unittest.TestCase):
    """When a delegated peer raises an operator-approval HITL, the request
    PARKS (outcome=STOP_HITL): this stream ends, the final answer arrives
    later via the async result callback."""
    def test_peer_hitl_parks_the_request(self):
        async def t():
            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                yield {"token": "checking app acl"}
                yield {"type": "hitl_interrupt", "hitl_interrupt": True,
                       "interrupt_id": "peer-int-1"}

            async def llm(query, context, state):
                return "[DELEGATE:dc-agent] 授予 alice CRM 权限(破坏性,需审批)"

            loop = _loop(llm, delegate)
            res = await loop.run(query="给 alice 授权 CRM", session_id="peer-hitl",
                                 tool_registry={})
            # request parks — not a normal graceful completion
            self.assertEqual(res.outcome, StopOutcome.STOP_HITL,
                             "peer HITL must park the request as STOP_HITL")
        asyncio.run(t())


class TestForkVsFreshFactSharing(unittest.TestCase):
    """The fork/fresh modifier controls what the peer sees. A leak here is a
    real privacy/context boundary bug: fresh must NOT hand the peer the
    parent's confirmed facts."""
    def test_forked_carries_parent_facts_and_query(self):
        async def t():
            cap = {}
            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                cap["forked"] = directive.forked
                cap["shared"] = list(shared_facts)
                cap["orig_q"] = original_query
                yield {"token": "ok"}
            turns = iter(["[DELEGATE:dc-agent#forked] 基于已知继续",
                          "综合完成,基于共享事实给出足够详细的最终答复内容。"])
            async def llm(q, c, s): return next(turns, "done")
            loop = _loop(llm, delegate)
            await loop.run(query="诊断 alice 权限", session_id="fork-carry",
                           confirmed_facts=["alice 在 VLAN 20"], tool_registry={})
            self.assertTrue(cap["forked"])
            self.assertIn("alice 在 VLAN 20", cap["shared"])
            self.assertEqual(cap["orig_q"], "诊断 alice 权限")
        asyncio.run(t())

    def test_fresh_isolates_parent_facts(self):
        async def t():
            cap = {}
            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                cap["forked"] = directive.forked
                cap["shared"] = list(shared_facts)
                yield {"token": "ok"}
            turns = iter(["[DELEGATE:dc-agent] fresh 委派",
                          "综合完成,给出足够详细的最终答复内容满足长度要求。"])
            async def llm(q, c, s): return next(turns, "done")
            loop = _loop(llm, delegate)
            await loop.run(query="诊断", session_id="fresh-isolate",
                           confirmed_facts=["敏感事实 X", "敏感事实 Y"],
                           tool_registry={})
            self.assertFalse(cap["forked"])
            self.assertEqual(cap["shared"], [],
                             "fresh delegation must NOT leak parent facts to the peer")
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
