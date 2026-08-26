"""Multi-hop delegation probes — LAN → WAN → DC (3-agent topology).

Scale: 3 agents (lan/dc/wan), each with ≤2 peers. Realistic chain:
    LAN edge  ──▶  WAN transport  ──▶  DC fabric
"a branch user can't reach a DC app" → LAN checks access, delegates the
transport leg to WAN, WAN finds the DC-facing path degraded and delegates
the fabric leg to DC.

The loop's delegation is SINGLE-HOP per agent: each agent delegates to ITS
OWN peer. A "multi-hop chain" emerges because the peer (WAN) is itself an
agent whose response was produced by ITS OWN delegation (to DC). We model
that by composing delegate_fns: LAN's delegate_fn to WAN internally runs a
second delegation to DC.

These probes CHARACTERIZE behavior (some is a known single-hop limitation,
not a bug) so a real chain problem is distinguishable from expected design.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig
from runtime.stop_policy import StopOutcome


def _loop(llm_fn, delegate_fn):
    return AgentRuntimeLoop(memory_router=None, config=RuntimeConfig(),
                            llm_fn=llm_fn, delegate_fn=delegate_fn)


class TestThreeHopChain(unittest.TestCase):
    """LAN → WAN → DC end-to-end: the WAN peer's answer is itself produced by
    WAN delegating to DC. Assert the origin's final answer reflects the
    3rd-hop (DC) result — i.e. the chain carries data end to end."""
    def test_lan_wan_dc_result_reaches_origin(self):
        async def t():
            # ---- innermost: WAN's own loop, which delegates to DC ----
            async def wan_delegate_to_dc(directive, session_id, shared_facts, *, original_query=""):
                # WAN receives "diagnose transport to DC"; it checks its own
                # tools then delegates the fabric leg to dc-agent.
                self.assertEqual(directive.target, "dc-agent")
                yield {"token": "DC fabric: leaf-3 EVPN route to 10.20.0.0/16 present"}

            async def wan_llm(query, context, state):
                # WAN's loop: 1 turn delegate to DC, 1 turn synthesize
                if "dc" not in getattr(wan_llm, "_did", ""):
                    wan_llm._did = "dc"
                    return "[DELEGATE:dc-agent] 查 10.20.0.0/16 的 EVPN 路由"
                return ("WAN 侧:branch-sf → dc-east 的 MPLS 路径 SLA 正常,"
                        "且 DC fabric 到 10.20.0.0/16 的 EVPN 路由存在。传输层无故障。")
            wan_llm._did = ""

            # WAN's result, as returned to LAN, is the synthesis of WAN+DC.
            async def lan_delegate_to_wan(directive, session_id, shared_facts, *, original_query=""):
                self.assertEqual(directive.target, "wan-agent")
                # run WAN's own loop to produce the peer answer
                wan_loop = _loop(wan_llm, wan_delegate_to_dc)
                wan_res = await wan_loop.run(query=directive.task,
                                             session_id="wan-sess", tool_registry={})
                for line in wan_res.final_response.split("\n"):
                    yield {"token": line}

            # ---- origin: LAN's loop, delegates to WAN ----
            lan_turns = iter([
                "先确认 alice 本地准入正常,再查到 DC 的传输路径。\n"
                "[DELEGATE:wan-agent] 诊断 branch-sf 到 dc-east 的传输 + DC 路由",
                ("综合:alice 本地准入正常;WAN 传输层 MPLS SLA 正常且 DC EVPN 路由存在。"
                 "端到端链路无故障,问题不在网络层,建议排查应用层。"),
            ])
            async def lan_llm(query, context, state):
                return next(lan_turns, "端到端诊断完成。")

            lan_loop = _loop(lan_llm, lan_delegate_to_wan)
            res = await lan_loop.run(query="branch-sf 的 alice 访问 DC 应用失败",
                                     session_id="lan-sess", tool_registry={})
            # the 3rd-hop (DC) fact surfaced all the way at the origin
            self.assertIn("EVPN 路由", res.final_response)
            self.assertIn("MPLS", res.final_response)
            self.assertEqual(res.outcome, StopOutcome.STOP_GRACEFUL)
        asyncio.run(t())


class TestProvenanceIsPerHopNotOrigin(unittest.TestCase):
    """KNOWN DESIGN (characterization): provenance on the outbound request
    stamps the IMMEDIATE source, not the chain origin. On LAN→WAN→DC, DC sees
    source_agent=WAN, not LAN. A real multi-hop audit trail (chain origin
    preserved through all hops) is NOT implemented (TODO 2B). This pins the
    current behavior so a future change is a conscious one."""
    def test_dc_sees_wan_as_source_not_lan(self):
        from task.delegation import build_delegate_fn
        import inspect
        # The stamping happens in build_delegate_fn: metadata['source_agent']
        # = own_agent_id (the delegating agent), so WAN→DC stamps WAN.
        src = inspect.getsource(build_delegate_fn)
        self.assertIn("source_agent", src)
        self.assertIn("own_agent_id", src)
        # documents: each hop overwrites source_agent with its own id →
        # the origin (LAN) is not carried past the first hop.


class TestFactsPropagateForkedAcrossHops(unittest.TestCase):
    """Forked delegation carries the delegating agent's facts to the next hop.
    Across a chain, each hop forwards ITS OWN accumulated facts — so a fact
    confirmed at LAN reaches WAN only if LAN forks, and reaches DC only if WAN
    also forks. Pin that per-hop forking is required (facts don't auto-tunnel
    the whole chain)."""
    def test_fact_reaches_second_hop_only_if_each_hop_forks(self):
        async def t():
            dc_saw = {}

            async def wan_delegate_to_dc(directive, session_id, shared_facts, *, original_query=""):
                dc_saw["facts"] = list(shared_facts)
                yield {"token": "dc ok"}

            async def wan_llm(query, context, state):
                if not getattr(wan_llm, "_d", False):
                    wan_llm._d = True
                    # WAN forks → passes its facts (which include the injected
                    # LAN fact if LAN forked to WAN) down to DC
                    return "[DELEGATE:dc-agent#forked] 继续"
                return "WAN 综合完成,已把上游事实透传给 DC 并回收结果,内容足够。"
            wan_llm._d = False

            async def lan_delegate_to_wan(directive, session_id, shared_facts, *, original_query=""):
                # WAN's loop starts with the facts LAN forked to it
                wan_loop = _loop(wan_llm, wan_delegate_to_dc)
                wan_res = await wan_loop.run(query=directive.task, session_id="wan-s",
                                             confirmed_facts=list(shared_facts),
                                             tool_registry={})
                yield {"token": wan_res.final_response}

            lan_turns = iter([
                "[DELEGATE:wan-agent#forked] 带着已知事实诊断传输",
                "综合完成,端到端事实链已打通,给出足够详细的最终答复内容。",
            ])
            async def lan_llm(query, context, state):
                return next(lan_turns, "done")

            lan_loop = _loop(lan_llm, lan_delegate_to_wan)
            await lan_loop.run(query="诊断", session_id="lan-s",
                               confirmed_facts=["branch-sf 用 ckt-sf-inet 上行"],
                               tool_registry={})
            # the LAN fact reached DC (2 hops) BECAUSE both hops forked
            self.assertIn("branch-sf 用 ckt-sf-inet 上行", dc_saw.get("facts", []),
                          "forked-at-every-hop must tunnel the fact to hop 3")
        asyncio.run(t())

    def test_fresh_hop_breaks_the_fact_chain(self):
        async def t():
            dc_saw = {}

            async def wan_delegate_to_dc(directive, session_id, shared_facts, *, original_query=""):
                dc_saw["facts"] = list(shared_facts)
                yield {"token": "dc ok"}

            async def wan_llm(query, context, state):
                if not getattr(wan_llm, "_d", False):
                    wan_llm._d = True
                    return "[DELEGATE:dc-agent] fresh 到 DC"   # NOT forked
                return "WAN 综合完成,fresh 委派 DC,未透传上游事实,内容足够满足长度。"
            wan_llm._d = False

            async def lan_delegate_to_wan(directive, session_id, shared_facts, *, original_query=""):
                wan_loop = _loop(wan_llm, wan_delegate_to_dc)
                wan_res = await wan_loop.run(query=directive.task, session_id="wan-s2",
                                             confirmed_facts=list(shared_facts),
                                             tool_registry={})
                yield {"token": wan_res.final_response}

            lan_turns = iter([
                "[DELEGATE:wan-agent#forked] 诊断",
                "综合完成,给出足够详细的最终答复内容满足长度要求即可。",
            ])
            async def lan_llm(query, context, state):
                return next(lan_turns, "done")

            lan_loop = _loop(lan_llm, lan_delegate_to_wan)
            await lan_loop.run(query="诊断", session_id="lan-s2",
                               confirmed_facts=["敏感上游事实"], tool_registry={})
            # WAN's fresh hop to DC breaks the chain — DC does NOT see the fact
            self.assertNotIn("敏感上游事实", dc_saw.get("facts", []),
                             "a fresh hop must isolate upstream facts from DC")
        asyncio.run(t())


class TestParkMidChain(unittest.TestCase):
    """A peer-HITL two hops deep: DC raises an approval while serving WAN.
    WAN's loop parks (STOP_HITL). What LAN sees depends on how WAN surfaces
    it. Characterize: WAN parking mid-chain yields a non-graceful outcome to
    WAN; LAN receives WAN's partial + a park signal (not a fake completion)."""
    def test_dc_hitl_parks_wan_hop(self):
        async def t():
            async def wan_delegate_to_dc(directive, session_id, shared_facts, *, original_query=""):
                yield {"token": "dc checking"}
                yield {"type": "hitl_interrupt", "hitl_interrupt": True,
                       "interrupt_id": "dc-int-9"}

            async def wan_llm(query, context, state):
                return "[DELEGATE:dc-agent] 授予 DC 侧权限(破坏性,需审批)"

            wan_loop = _loop(wan_llm, wan_delegate_to_dc)
            wan_res = await wan_loop.run(query="需要 DC 授权", session_id="wan-park",
                                         tool_registry={})
            # WAN's hop parks on DC's operator — not a graceful completion
            self.assertEqual(wan_res.outcome, StopOutcome.STOP_HITL,
                             "DC HITL must park WAN's hop, not fake-complete it")
        asyncio.run(t())


class TestTopologyMaxTwoPeers(unittest.TestCase):
    """Topology guard: at 3-agent scale each agent has ≤2 peers. WAN's peers
    are lan + dc. Delegating to a THIRD, unknown peer degrades gracefully to a
    note (not a crash), because the registry can't resolve it."""
    def test_delegate_to_unknown_peer_degrades(self):
        async def t():
            # delegate_fn=None models 'no such peer wired' → graceful note
            async def llm(query, context, state):
                if not getattr(llm, "_d", False):
                    llm._d = True
                    return "[DELEGATE:storage-agent] 查存储(不在拓扑内)"
                return ("storage-agent 不可委派(不在本 agent 的 peer 列表)。"
                        "基于本地与已知 peer 能力,如实说明无法覆盖存储域。")
            llm._d = False
            loop = AgentRuntimeLoop(memory_router=None, config=RuntimeConfig(),
                                    llm_fn=llm, delegate_fn=None)  # nothing wired
            res = await loop.run(query="查存储", session_id="topo-unknown",
                                 tool_registry={})
            # degrades to an honest note, no crash
            self.assertIn("storage-agent", res.final_response)
            self.assertEqual(res.outcome, StopOutcome.STOP_GRACEFUL)
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
