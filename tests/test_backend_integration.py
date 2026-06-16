"""Backend integration tests — the biggest remaining blind spot.

Until now every scenario was covered at the loop level; the webui backend
SSE wrapper (`/chat/stream`: classify → route → executor → on_chunk → SSE,
plus the post-turn evolution hooks) was only verified by hand on the live
two-agent setup. This drives the REAL stack — `create_webui_app` + a real
`HitlExecutor` (in-memory stores via build_default_executor) + a real
`AgentRuntimeLoop` — through FastAPI's TestClient with a fake LLM.

No Ollama, no pydantic-incompatible deps: the only stub is llm_fn.
"""
import json
import unittest

try:
    from fastapi.testclient import TestClient
    _HAVE_FASTAPI = True
except Exception:
    _HAVE_FASTAPI = False

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig
from integrations.adapters.hitl_executor import build_default_executor
from webui.backend import create_webui_app


def _sse_events(resp_text: str) -> list[dict]:
    out = []
    for line in resp_text.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:]))
            except Exception:
                pass
    return out


def _app(llm_fn, *, hitl_names=frozenset(), tool_registry=None, extra_services=None):
    tool_registry = tool_registry or {}
    cfg = RuntimeConfig(hitl_tool_names=hitl_names)
    loop = AgentRuntimeLoop(config=cfg, llm_fn=llm_fn)
    execu, router = build_default_executor(
        runtime_loop=loop, llm_engine=None, tool_registry=tool_registry)
    services = {"runtime_loop": loop, "executor": execu,
                "tool_registry": tool_registry}
    if extra_services:
        services.update(extra_services)
    return create_webui_app(services), services


@unittest.skipUnless(_HAVE_FASTAPI, "fastapi/httpx not installed")
class TestChatStreamHappyPath(unittest.TestCase):
    def test_read_only_query_streams_full_sequence(self):
        async def llm(query, context, state):
            return ("设备 sw-01 状态正常,所有端口 up,这是一段足够长的诊断"
                    "答复用于验证后端流式链路完整。")
        app, _ = _app(llm)
        c = TestClient(app)
        r = c.post("/chat/stream",
                   json={"query": "检查 sw-01 状态", "session_id": "it-happy"})
        self.assertEqual(r.status_code, 200)
        ev = _sse_events(r.text)
        kinds = [e.get("type") for e in ev if e.get("type")]
        # the backend's real SSE lifecycle
        self.assertIn("classify", kinds)
        self.assertIn("done", kinds)
        # answer tokens streamed
        answer = "".join(e.get("token", "") for e in ev)
        self.assertIn("sw-01", answer)

    def test_missing_executor_errors_cleanly(self):
        async def llm(query, context, state):
            return "x" * 40
        loop = AgentRuntimeLoop(config=RuntimeConfig(), llm_fn=llm)
        app = create_webui_app({"runtime_loop": loop, "executor": None})
        c = TestClient(app)
        r = c.post("/chat/stream", json={"query": "q", "session_id": "it-noexec"})
        self.assertEqual(r.status_code, 200)   # streams an error event, not a 500
        ev = _sse_events(r.text)
        self.assertTrue(any(e.get("type") == "error" for e in ev))


@unittest.skipUnless(_HAVE_FASTAPI, "fastapi/httpx not installed")
class TestChatStreamHitlInterrupt(unittest.TestCase):
    def test_destructive_tool_interrupts_and_does_not_execute(self):
        """THE backend-level safety contract: a watch-listed tool proposed by
        the LLM produces an HITL interrupt in the SSE stream and the tool does
        NOT run before approval."""
        executed = []

        async def restart_service(args):
            executed.append(args)
            return "restarted"

        async def llm(query, context, state):
            return '执行重启。\n[TOOL:restart_service] {"service": "crm"}'

        app, _ = _app(llm, hitl_names={"restart_service"},
                      tool_registry={"restart_service": restart_service})
        c = TestClient(app)
        r = c.post("/chat/stream",
                   json={"query": "重启 crm 服务", "session_id": "it-hitl"})
        self.assertEqual(r.status_code, 200)
        ev = _sse_events(r.text)
        kinds = [e.get("type") for e in ev if e.get("type")]
        self.assertIn("hitl_interrupt", kinds,
                      f"expected hitl_interrupt in {kinds}")
        self.assertEqual(executed, [],
                         "destructive tool must NOT execute before approval")


@unittest.skipUnless(_HAVE_FASTAPI, "fastapi/httpx not installed")
class TestChatStreamCapabilityGap(unittest.TestCase):
    def test_capability_gap_surfaces_through_backend(self):
        async def llm(query, context, state):
            return ("我已确认 LAN 准入正常,但重置 AD 域控密码本 agent 无对应工具"
                    "也无法委派。\n[CAPABILITY_GAP: 重置 AD 域控密码 — 缺少 AD 管理工具]")
        app, _ = _app(llm)
        c = TestClient(app)
        r = c.post("/chat/stream",
                   json={"query": "诊断并重置 alice 的 AD 密码", "session_id": "it-gap"})
        self.assertEqual(r.status_code, 200)
        ev = _sse_events(r.text)
        kinds = [e.get("type") for e in ev if e.get("type")]
        self.assertIn("capability_gap", kinds,
                      f"capability_gap must reach the SSE stream; got {kinds}")
        # marker stripped from streamed prose
        answer = "".join(e.get("token", "") for e in ev)
        self.assertNotIn("CAPABILITY_GAP", answer)


@unittest.skipUnless(_HAVE_FASTAPI, "fastapi/httpx not installed")
class TestEvolutionGapsEndpoint(unittest.TestCase):
    def test_gaps_endpoint_aggregates(self):
        from runtime.skill_journal import get_journal_store, SkillJournal
        store = get_journal_store()
        j = SkillJournal(session_id="it-gapep", query="扫描WiFi热力图")
        j.record_capability_gap(turn=1, detail="缺少无线热力图扫描工具",
                                query="扫描WiFi热力图")
        j.record_completion(outcome="completed", total_turns=1)
        f = j.to_dict(); f["_complete"] = True; store.append(f)

        async def llm(query, context, state):
            return "x" * 40
        app, _ = _app(llm)
        c = TestClient(app)
        r = c.get("/evolution/gaps")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertGreaterEqual(body["total_gap_events"], 1)
        details = [g["detail"] for g in body["gaps"]]
        self.assertTrue(any("热力图" in d for d in details))


if __name__ == "__main__":
    unittest.main(verbosity=2)
